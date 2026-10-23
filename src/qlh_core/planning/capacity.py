"""Capacity admission for metadata-first PyTorch pipeline loading.

The solver consumes only a pipeline descriptor and explicit free-memory budgets.
It never imports torch or opens model weight files.  A successful result always
covers every decoder layer exactly once and assigns the input/output components
to the first/last execution node respectively.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from typing import Any

from qlh_core.contracts.load import RuntimeLoadSpec
from qlh_core.planning.placement import placement_weight, stage_placement_cost


CAPACITY_PLAN_SCHEMA_VERSION = 1


class PipelineCapacityError(ValueError):
    """The descriptor or node capacity input is not safe to solve."""


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or (isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer())):
        raise PipelineCapacityError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineCapacityError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise PipelineCapacityError(f"{field} must not be negative")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineCapacityError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise PipelineCapacityError(f"{field} must be positive")
    return parsed


def _normalize_layer_ranges(
    value: Any,
    field: str,
    *,
    total_layers: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Validate a worker's advertised half-open layer intervals."""
    if not isinstance(value, (list, tuple)):
        raise PipelineCapacityError(f"{field} must be a list of [start, end) ranges")
    normalized: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise PipelineCapacityError(
                f"{field}[{index}] must be a [start, end) range"
            )
        start, end = item
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise PipelineCapacityError(
                f"{field}[{index}] bounds must be integers"
            )
        if start < 0 or end <= start:
            raise PipelineCapacityError(
                f"{field}[{index}] must satisfy 0 <= start < end"
            )
        if total_layers is not None and end > total_layers:
            raise PipelineCapacityError(
                f"{field}[{index}] exceeds total_layers={total_layers}"
            )
        normalized.append((start, end))
    return tuple(sorted(set(normalized)))


def _normalize_range_weight_bytes(value, field, *, total_layers):
    """Exact device-peak costs including the quoted range's boundary units."""
    if not isinstance(value, (list, tuple)):
        raise PipelineCapacityError(f"{field} must be a list of exact range quotes")
    normalized, seen = [], set()
    for index, quote in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(quote, dict):
            raise PipelineCapacityError(f"{item_field} must be an object")
        span = _normalize_layer_ranges([quote.get("layer_range")], item_field,
                                      total_layers=total_layers)[0]
        if span in seen:
            raise PipelineCapacityError(f"{field} contains duplicate layer_range")
        seen.add(span)
        weight = quote.get("weight_bytes")
        if type(weight) is not int or weight <= 0:
            raise PipelineCapacityError(f"{item_field}.weight_bytes must be a positive integer")
        for flag in ("has_embedding", "has_lm_head"):
            if type(quote.get(flag)) is not bool:
                raise PipelineCapacityError(f"{item_field}.{flag} must be a boolean")
        normalized.append({"layer_range": span, "weight_bytes": weight,
                           "has_embedding": quote["has_embedding"],
                           "has_lm_head": quote["has_lm_head"]})
    return tuple(sorted(normalized, key=lambda item: item["layer_range"]))


def _normalize_layer_artifacts(
    value: Any,
    field: str,
    *,
    total_layers: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate exact ready-artifact ranges and their boundary semantics."""
    if not isinstance(value, (list, tuple)) or not value:
        raise PipelineCapacityError(f"{field} must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise PipelineCapacityError(f"{item_field} must be an object")
        allowed = {
            "layer_range", "segment_mode", "model_id", "artifact_sha256",
            "source_model_sha256", "source_model_id", "hidden_size",
            "tokenizer_sha256",
            # ★ 2026-10-11（方案 A：worker 上报本段逐层字节）：
            #   接力不加载整模 ⇒ 主节点只持有**本段**的逐层字节，而求解器需要**整模**的
            #   逐层字节（`_descriptor_costs` 的 `every decoder layer must have a positive
            #   byte count` 与各节点容量账都用它）。因此让每个 worker 在 `layer_artifacts`
            #   里申报**自己那一段**的逐层字节，主节点按 `layer_range` 平移拼出整模视图。
            "layer_weight_bytes",
        }
        required = {
            "layer_range", "segment_mode", "model_id", "artifact_sha256",
        }
        if set(item) - allowed or not required.issubset(item):
            raise PipelineCapacityError(
                f"{item_field} has invalid or missing fields"
            )
        ranges = _normalize_layer_ranges(
            [item.get("layer_range")],
            f"{item_field}.layer_range",
            total_layers=total_layers,
        )
        layer_range = ranges[0]
        if layer_range in seen_ranges:
            raise PipelineCapacityError(
                f"{field} must not contain duplicate layer ranges"
            )
        seen_ranges.add(layer_range)
        mode = str(item.get("segment_mode", "") or "").strip().lower()
        if mode not in {"head", "middle", "tail"}:
            raise PipelineCapacityError(
                f"{item_field}.segment_mode must be head, middle, or tail"
            )
        model_id = str(item.get("model_id", "") or "").strip()
        artifact_sha256 = str(item.get("artifact_sha256", "") or "").strip().lower()
        source_sha256 = str(
            item.get("source_model_sha256", "") or ""
        ).strip().lower()
        contract_fields = {"source_model_id", "hidden_size", "tokenizer_sha256"}
        present_contract_fields = contract_fields.intersection(item)
        if present_contract_fields and present_contract_fields != contract_fields:
            raise PipelineCapacityError(
                f"{item_field} model preflight fields must be declared together"
            )
        if not model_id:
            raise PipelineCapacityError(f"{item_field}.model_id must not be empty")
        if len(artifact_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in artifact_sha256
        ):
            raise PipelineCapacityError(
                f"{item_field}.artifact_sha256 must be a SHA-256 digest"
            )
        if source_sha256 and (
            len(source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_sha256)
        ):
            raise PipelineCapacityError(
                f"{item_field}.source_model_sha256 must be a SHA-256 digest"
            )
        normalized_item = {
            "layer_range": layer_range,
            "segment_mode": mode,
            "model_id": model_id,
            "artifact_sha256": artifact_sha256,
            "source_model_sha256": source_sha256,
        }
        # ★ 2026-10-11（方案 A）：worker 申报的**本段局部**逐层字节。主节点据此按
        #   `layer_range` 平移到源模型坐标系，补齐自己那份 descriptor 的缺口
        #   （接力不加载整模 ⇒ 主节点只有本段字节，而求解器要求每层正字节）。
        #   长度必须等于本段跨度；任一层非正即拒绝（不让残缺容量账进入求解器）。
        segment_bytes = item.get("layer_weight_bytes")
        if segment_bytes is not None:
            if not isinstance(segment_bytes, list) or not segment_bytes:
                raise PipelineCapacityError(
                    f"{item_field}.layer_weight_bytes must be a non-empty list"
                )
            if len(segment_bytes) != layer_range[1] - layer_range[0]:
                raise PipelineCapacityError(
                    f"{item_field}.layer_weight_bytes length must equal the segment span"
                )
            sizes = [
                _non_negative_int(
                    value, f"{item_field}.layer_weight_bytes[{offset}]",
                )
                for offset, value in enumerate(segment_bytes)
            ]
            if any(size <= 0 for size in sizes):
                raise PipelineCapacityError(
                    f"{item_field}.layer_weight_bytes must be all positive"
                )
            normalized_item["layer_weight_bytes"] = sizes
        if present_contract_fields:
            source_model_id = str(item.get("source_model_id", "") or "").strip()
            tokenizer_sha256 = str(
                item.get("tokenizer_sha256", "") or ""
            ).strip().lower()
            hidden_size = _non_negative_int(
                item.get("hidden_size"), f"{item_field}.hidden_size",
            )
            if not source_model_id:
                raise PipelineCapacityError(
                    f"{item_field}.source_model_id must not be empty"
                )
            if hidden_size <= 0:
                raise PipelineCapacityError(
                    f"{item_field}.hidden_size must be positive"
                )
            if len(tokenizer_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in tokenizer_sha256
            ):
                raise PipelineCapacityError(
                    f"{item_field}.tokenizer_sha256 must be a SHA-256 digest"
                )
            normalized_item.update({
                "source_model_id": source_model_id,
                "hidden_size": hidden_size,
                "tokenizer_sha256": tokenizer_sha256,
            })
        normalized.append(normalized_item)
    return tuple(sorted(normalized, key=lambda item: item["layer_range"]))


def _normalize_layer_budget(value: Any, field: str) -> dict[str, Any]:
    """Validate a worker's self-declared forward-layer budget.

    与 `layer_ranges` 的分工：`layer_ranges` 是该节点**当前已就绪、马上能跑**的区间；
    `layer_budget.max_layers` 是它在本地裁层之后**能承载的层数上限**。有了后者，
    调度可以把任意连续区间分配给具备本地裁层条件的节点，而不是只能迁就它手上
    那份预先切好的工件。
    """
    if not isinstance(value, dict):
        raise PipelineCapacityError(f"{field} must be an object")
    for key in ("available_bytes", "per_layer_bytes", "max_layers"):
        if key not in value:
            raise PipelineCapacityError(f"{field}.{key} is required")
    available_bytes = _non_negative_int(
        value["available_bytes"], f"{field}.available_bytes"
    )
    per_layer_bytes = _non_negative_int(
        value["per_layer_bytes"], f"{field}.per_layer_bytes"
    )
    if per_layer_bytes <= 0:
        raise PipelineCapacityError(f"{field}.per_layer_bytes must be positive")
    max_layers = _non_negative_int(value["max_layers"], f"{field}.max_layers")
    if max_layers <= 0:
        raise PipelineCapacityError(f"{field}.max_layers must be positive")
    local_cut = value.get("local_cut", False)
    if not isinstance(local_cut, bool):
        raise PipelineCapacityError(f"{field}.local_cut must be a boolean")
    return {
        "available_bytes": available_bytes,
        "per_layer_bytes": per_layer_bytes,
        "max_layers": max_layers,
        "local_cut": local_cut,
    }


def _descriptor_costs(
    descriptor: dict[str, Any],
) -> tuple[list[int], int, int, int]:
    total_layers = _non_negative_int(descriptor.get("total_layers", 0), "total_layers")
    raw_layers = descriptor.get("layer_weight_bytes")
    if total_layers <= 0 or not isinstance(raw_layers, list):
        raise PipelineCapacityError("descriptor is missing exact layer_weight_bytes")
    if len(raw_layers) != total_layers:
        raise PipelineCapacityError("layer_weight_bytes does not match total_layers")
    layer_bytes = [
        _non_negative_int(value, f"layer_weight_bytes[{index}]")
        for index, value in enumerate(raw_layers)
    ]
    if any(value <= 0 for value in layer_bytes):
        raise PipelineCapacityError("every decoder layer must have a positive byte count")

    components = descriptor.get("component_weight_bytes")
    if not isinstance(components, dict):
        raise PipelineCapacityError("descriptor is missing component_weight_bytes")
    embedding_bytes = _non_negative_int(
        components.get("embedding", 0), "component_weight_bytes.embedding"
    )
    per_node_bytes = _non_negative_int(
        components.get("final_norm", 0), "component_weight_bytes.final_norm"
    )
    output_bytes = sum(
        _non_negative_int(components.get(name, 0), f"component_weight_bytes.{name}")
        for name in ("lm_head", "other")
    )
    # Tied Qwen3 checkpoints store the output projection only once under the
    # embedding key.  A separate last-stage worker still needs that tensor,
    # so charge it as output capacity when no explicit LM Head tensor exists.
    if bool(descriptor.get("tie_word_embeddings", False)) and output_bytes == 0:
        output_bytes = embedding_bytes
    unsupported = {
        name: _non_negative_int(
            components.get(name, 0), f"component_weight_bytes.{name}"
        )
        for name in ("visual", "mtp", "multimodal")
    }
    # ★ #31 M3：架构可以**显式声明**某些分量"进了权重索引、但**不参与层执行**"
    #   （hybrid 的 visual / mtp —— 依据是运行时结构里根本没有这两类子模块，见
    #   `已知问题记录.md` #31 §31.6.1 的实测）。这类分量**不计入容量账**（没有任何节点会加载它们）；
    #   **其余分量一律照旧 fail-closed**，且标记里的未知名字本身也要报错（防手抖）。
    ignored = descriptor.get("runtime_ignored_components") or []
    if not isinstance(ignored, (list, tuple)):
        raise PipelineCapacityError("descriptor runtime_ignored_components must be a list")
    unknown = [str(name) for name in ignored if name not in unsupported]
    if unknown:
        raise PipelineCapacityError(
            "descriptor runtime_ignored_components has unknown entries: " + ", ".join(unknown)
        )
    ignored = set(ignored)
    active_unsupported = [
        name for name, size in unsupported.items()
        if size > 0 and name not in ignored
    ]
    if active_unsupported:
        raise PipelineCapacityError(
            "descriptor has separately placeable components without a runtime plan: "
            + ", ".join(active_unsupported)
        )
    return layer_bytes, embedding_bytes, per_node_bytes, output_bytes


def _normalize_nodes(
    nodes: list[dict[str, Any]],
    *,
    total_layers: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(nodes, list):
        raise PipelineCapacityError("nodes must be a list")
    usable: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in nodes:
        if not isinstance(raw, dict):
            raise PipelineCapacityError("each node capacity record must be an object")
        node_id = str(raw.get("node_id", "") or "").strip()
        if not node_id or node_id in seen:
            raise PipelineCapacityError("node_id must be non-empty and unique")
        seen.add(node_id)
        capacity_bytes = _non_negative_int(raw.get("capacity_bytes", 0), "capacity_bytes")
        reserve_bytes = _non_negative_int(raw.get("reserve_bytes", 0), "reserve_bytes")
        capacity_source = str(raw.get("capacity_source", "explicit") or "explicit")
        # Relay-exempt nodes are transport participants, not local model
        # placement candidates. Keep them normalized so their zero-layer
        # assignment is counted even without a model-memory budget.
        if capacity_bytes <= reserve_bytes and capacity_source != "relay_exempt":
            excluded.append({
                "node_id": node_id,
                "role": str(raw.get("role", "client") or "client"),
                "reason_code": "node_capacity_unavailable",
                "capacity_bytes": capacity_bytes,
                "reserve_bytes": reserve_bytes,
                "capacity_source": capacity_source,
                **capacity_budget(0, capacity_bytes=capacity_bytes, reserve_bytes=reserve_bytes),
            })
            continue
        try:
            score = float(raw.get("score", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PipelineCapacityError("score must be finite") from exc
        if not math.isfinite(score):
            raise PipelineCapacityError("score must be finite")
        normalized = {
            "node_id": node_id,
            "role": str(raw.get("role", "client") or "client"),
            "capacity_bytes": capacity_bytes,
            "reserve_bytes": reserve_bytes,
            "runtime_multiplier": _positive_float(
                raw.get("runtime_multiplier", 1.0), "runtime_multiplier"
            ),
            "score": score,
            "capacity_source": capacity_source,
            "execution_device": str(raw.get("execution_device", "unknown") or "unknown"),
        }
        if "configured_reserve_bytes" in raw or "retained_component_reserve_bytes" in raw:
            for field in ("configured_reserve_bytes", "retained_component_reserve_bytes"):
                if type(raw.get(field)) is not int or raw[field] < 0:
                    raise PipelineCapacityError(f"node[{node_id}].{field} must be a nonnegative integer")
                normalized[field] = raw[field]
            if normalized["configured_reserve_bytes"] + normalized["retained_component_reserve_bytes"] != reserve_bytes:
                raise PipelineCapacityError(f"node[{node_id}].reserve_bytes breakdown mismatch")
        if "retained_component_bytes" in raw:
            if type(raw["retained_component_bytes"]) is not int:
                raise PipelineCapacityError("retained_component_bytes must be an integer")
            normalized["retained_component_bytes"] = _non_negative_int(
                raw["retained_component_bytes"], "retained_component_bytes")
        if "load_spec" in raw:
            try:
                spec = (raw["load_spec"] if isinstance(raw["load_spec"], RuntimeLoadSpec)
                        else RuntimeLoadSpec.from_dict(raw["load_spec"]))
            except (TypeError, ValueError) as exc:
                raise PipelineCapacityError(f"node[{node_id}].load_spec is invalid") from exc
            supplied_digest = raw.get("load_spec_digest", spec.digest)
            if supplied_digest != spec.digest:
                raise PipelineCapacityError(f"node[{node_id}].load_spec_digest mismatch")
            if normalized["execution_device"] not in ("unknown", spec.device):
                raise PipelineCapacityError(f"node[{node_id}].load_spec device mismatch")
            engine = raw.get("engine")
            materializations = {
                "pytorch": {"safetensors_key_filtered", "safetensors_layer_offload"},
                "llama_cpp": {"gguf_segment"},
            }.get(engine)
            if materializations is not None and spec.materialization not in materializations:
                raise PipelineCapacityError(f"node[{node_id}].load_spec engine mismatch")
            if spec.materialization == "safetensors_layer_offload" and (
                    not spec.device.startswith("cuda:") or "range_weight_bytes" not in raw):
                raise PipelineCapacityError(
                    f"node[{node_id}].load_spec layer offload requires CUDA and exact range_weight_bytes")
            if reserve_bytes < spec.kv_cache_bytes:
                raise PipelineCapacityError(f"node[{node_id}].reserve_bytes does not cover KV")
            normalized.update(load_spec=spec.to_dict(), load_spec_digest=spec.digest,
                              execution_device=spec.device)
        elif "load_spec_digest" in raw:
            raise PipelineCapacityError(f"node[{node_id}].load_spec_digest has no spec")
        host_fields = ("host_required_bytes", "host_capacity_bytes")
        offload = normalized.get("load_spec", {}).get("materialization") == "safetensors_layer_offload"
        if offload or "host_required_bytes" in raw:
            for field in host_fields:
                if type(raw.get(field)) is not int or raw[field] <= 0:
                    raise PipelineCapacityError(f"node[{node_id}].{field} must be a positive integer")
                normalized[field] = raw[field]
            if normalized["host_required_bytes"] > normalized["host_capacity_bytes"]:
                raise PipelineCapacityError(f"node[{node_id}].host_required_bytes exceeds host_capacity_bytes")
        elif "host_capacity_bytes" in raw:
            # Resident CUDA inventory also samples host memory. This observation
            # alone introduces no host requirement and cannot block GPU admission.
            normalized["host_capacity_bytes"] = _non_negative_int(
                raw["host_capacity_bytes"], f"node[{node_id}].host_capacity_bytes")
        if "layer_weight_bytes" in raw:
            sizes = raw["layer_weight_bytes"]
            if (not isinstance(sizes, (list, tuple))
                    or total_layers is None or len(sizes) != total_layers
                    or any(type(size) is not int or size <= 0 for size in sizes)):
                raise PipelineCapacityError(
                    f"node[{node_id}].layer_weight_bytes must quote every decoder layer"
                )
            normalized["layer_weight_bytes"] = tuple(sizes)
        if "range_weight_bytes" in raw:
            normalized["range_weight_bytes"] = _normalize_range_weight_bytes(
                raw["range_weight_bytes"], f"node[{node_id}].range_weight_bytes",
                total_layers=total_layers,
            )
        # Candidate generation constraints are independent of ready artifacts.
        # In particular, local_cut must never relax a host/source range budget.
        if "candidate_layer_ranges" in raw:
            normalized["candidate_layer_ranges"] = _normalize_layer_ranges(
                raw["candidate_layer_ranges"], f"node[{node_id}].candidate_layer_ranges",
                total_layers=total_layers,
            )
        if "layer_ranges" in raw:
            normalized["layer_ranges"] = _normalize_layer_ranges(
                raw.get("layer_ranges"),
                f"node[{node_id}].layer_ranges",
                total_layers=total_layers,
            )
        if "layer_artifacts" in raw:
            normalized["layer_artifacts"] = _normalize_layer_artifacts(
                raw.get("layer_artifacts"),
                f"node[{node_id}].layer_artifacts",
                total_layers=total_layers,
            )
            if "layer_ranges" in normalized:
                advertised_ranges = set(normalized["layer_ranges"])
                artifact_ranges = {
                    item["layer_range"] for item in normalized["layer_artifacts"]
                }
                if artifact_ranges != advertised_ranges:
                    raise PipelineCapacityError(
                        f"node[{node_id}].layer_artifacts and layer_ranges must match"
                    )
        # ★ 2026-10-03：设备自荐的层容量（本地裁层后可承载的层数上限）。
        if "layer_budget" in raw:
            normalized["layer_budget"] = _normalize_layer_budget(
                raw.get("layer_budget"), f"node[{node_id}].layer_budget"
            )
        # ★ 2026-10-05（DIST-3 三机实测）：工件段类型（`head`/`middle`/`tail`）。
        #   只在取值为已知三种之一时透传；其它值（含缺失）视为未声明 ⇒ 不参与
        #   求解器的段类型约束，保持旧行为。
        if "segment_mode" in raw:
            raw_segment_mode = raw.get("segment_mode")
            if not isinstance(raw_segment_mode, str) or raw_segment_mode.lower() not in (
                "head", "middle", "tail",
            ):
                raise PipelineCapacityError(
                    f"node[{node_id}].segment_mode must be head, middle, or tail"
                )
            normalized["segment_mode"] = raw_segment_mode.lower()
        usable.append(normalized)
    # The execution contract puts a participating master at the input stage.
    # This order chooses stage positions; the objective below chooses amounts.
    usable.sort(
        key=lambda node: (
            node["role"] != "master",
            -node["score"],
            -(node["capacity_bytes"] - node["reserve_bytes"]),
            node["node_id"],
        )
    )
    excluded.sort(key=lambda node: node["node_id"])
    return usable, excluded


def capacity_budget(
    raw_bytes: int, *, capacity_bytes: int, reserve_bytes: int = 0,
    runtime_multiplier: float = 1.0, safety_margin: float = 1.2,
) -> dict[str, int]:
    """The single byte calculation used by admission and its diagnostics.

    available_bytes is the supplied free-memory sample before reservations;
    required_bytes includes the reservation. Legacy capacity_bytes and
    reserve_bytes retain those meanings. KV bytes in a load spec must already
    be covered by reserve_bytes, so they are never charged twice.
    """
    raw = _non_negative_int(raw_bytes, "raw_bytes")
    available = _non_negative_int(capacity_bytes, "capacity_bytes")
    reserved = _non_negative_int(reserve_bytes, "reserve_bytes")
    multiplier = _positive_float(runtime_multiplier, "runtime_multiplier")
    margin = _positive_float(safety_margin, "safety_margin")
    if margin < 1.0:
        raise PipelineCapacityError("safety_margin must be at least 1.0")
    try:
        scaled = raw * multiplier * margin
    except OverflowError as exc:
        raise PipelineCapacityError("runtime weight budget must be finite") from exc
    if not math.isfinite(scaled):
        raise PipelineCapacityError("runtime weight budget must be finite")
    required = reserved + math.ceil(scaled)
    return {
        "required_bytes": required, "available_bytes": available,
        "reserved_bytes": reserved, "shortfall_bytes": max(0, required - available),
        "allocatable_bytes": max(0, available - reserved),
        "headroom_bytes": available - required,
    }


def _node_budget(raw_bytes: int, node: dict[str, Any], safety_margin: float) -> dict[str, int]:
    return capacity_budget(
        raw_bytes, capacity_bytes=node.get("capacity_bytes", 0),
        reserve_bytes=node["reserve_bytes"],
        runtime_multiplier=node.get("runtime_multiplier", 1.0),
        safety_margin=safety_margin,
    )


def _required_bytes(raw_bytes: int, node: dict[str, Any], safety_margin: float) -> int:
    """Compatibility for manual plans and existing budget channels."""
    return _node_budget(raw_bytes, node, safety_margin)["required_bytes"]


def _relay_zero_layer_assignments(
    relay_nodes: list[dict[str, Any]], total_layers: int,
    claims: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """★ A1 / X 档（Y 档第二条缺口 6/2b）：把 relay 段节点作为**零层**条目并入 assignments。

    主节点据此给它下发 `engine="relay_middle"` 配置（`scheduler_pipeline.py:298-302` 按
    `_relay_segment_for_worker()` 决定），worker 侧于是**不加载任何层**、只转发
    （`peer.py` 的 relay 分支）。

    ★ Y 档第二条：条目带上该段**认领的层区间**（`claims[node_id]`，形如 `(8, 24)`）——
    下游的层覆盖校验（`pipeline_node_contract.validate_pipeline_nodes`）必须知道
    "`[8,24)` 由远端段执行"，否则会误判为"层没人覆盖"而拒掉合法拓扑。

    `plan_identity` / `plan_id` **不含**它们（那里只用上面的 `assignments`）⇒
    零层条目不改变 plan 的身份摘要。
    """
    claims = claims or {}
    entries = []
    for node in relay_nodes:
        start, end = claims.get(node["node_id"], (total_layers, total_layers))
        entries.append({
            "node_id": node["node_id"],
            "role": node["role"],
            "start_layer": int(start),
            "end_layer": int(end),
            "layers_count": 0,
            "has_embedding": False,
            "has_lm_head": False,
            "claimed_layers": [int(start), int(end)],
            "raw_weight_bytes": 0,
            "required_bytes": 0,
            "capacity_bytes": node["capacity_bytes"],
            "headroom_bytes": node["capacity_bytes"],
            "reserve_bytes": 0,
            "runtime_multiplier": 1.0,
            "execution_device": node["execution_device"],
            "capacity_source": "relay_exempt",
            "score": node["score"],
            **capacity_budget(0, capacity_bytes=node["capacity_bytes"], safety_margin=1.0),
        })
    return entries


def _uncovered_layer_ranges(ranges, total_layers: int) -> list[tuple[int, int]]:
    """★ 2026-10-08（真机复测）：`[0, total_layers)` 里**没有任何节点区间覆盖**的段。

    真机上 `pipeline_layer_range_coverage_insufficient` 被反复误读成"工件区间配置错"，
    而实际原因是 **worker 进程被系统回收/重连、掉出候选**（实测 Y700 每 4–16 分钟重建一次，
    落在断开窗口里的请求就报这个码）。把缺口写进 `reason`，现场一眼可分"区间不连续"与
    "有节点不在"。
    """
    covered = sorted(
        (int(start), int(stop))
        for start, stop in (ranges or ())
        if int(stop) > int(start)
    )
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, stop in covered:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < int(total_layers or 0):
        gaps.append((cursor, int(total_layers)))
    return gaps


def _merge_source_layer_bytes(
    descriptor: dict[str, Any], nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """用各 worker 申报的本段逐层字节补齐 descriptor 的**整模**视图。

    ★ 2026-10-11（方案 A）：接力不加载整模（用户裁定：这是接力的初衷）⇒ 主节点只持有
    **本段**逐层字节，而 `_descriptor_costs` 要求每一层都有正字节数。各 worker 在
    `layer_artifacts[].layer_weight_bytes` 里申报**本段局部**口径的逐层字节
    （下标 0 = 段内第一层），这里按 `layer_range` 平移到源模型坐标系后填入缺口。

    补齐值仅用于形成求解器的候选层视图；实际计费必须来自最终选中的工件。
    只接受已规范化的可用节点与同源工件，对重叠报价取上界；未补齐则保持拒绝。
    """
    if not isinstance(descriptor, dict):
        return descriptor
    raw = descriptor.get("layer_weight_bytes")
    if not isinstance(raw, list) or not raw:
        return descriptor
    try:
        total = int(descriptor.get("total_layers", 0) or 0)
    except (TypeError, ValueError):
        return descriptor
    if total <= 0 or len(raw) != total:
        return descriptor
    merged = [
        _non_negative_int(value, f"layer_weight_bytes[{index}]")
        for index, value in enumerate(raw)
    ]
    if all(value > 0 for value in merged):
        return descriptor
    expected_source = str(descriptor.get("source_model_sha256", "") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_source) is None:
        # Fail-closed on purpose: with part of `layer_weight_bytes` unknown we cannot
        # account for those layers at all. 跨格式主节点（PyTorch 描述器 + GGUF 层段）
        # 由调用点补齐该凭据（见 `scheduler` 的容量求解入口），这里不放宽。
        raise PipelineCapacityError(
            "partial layer_weight_bytes requires source_model_sha256"
        )
    filled = 0
    for node in nodes or []:
        for artifact in node.get("layer_artifacts", ()):
            segment_bytes = artifact.get("layer_weight_bytes")
            start, end = artifact["layer_range"]
            if artifact["source_model_sha256"] != expected_source or not segment_bytes:
                continue
            for offset, size in enumerate(segment_bytes):
                index = start + offset
                if int(raw[index]) > 0:
                    continue
                if merged[index] <= 0:
                    filled += 1
                merged[index] = max(merged[index], size)
    if filled <= 0 or any(value <= 0 for value in merged):
        return descriptor
    merged_descriptor = dict(descriptor)
    merged_descriptor["layer_weight_bytes"] = merged
    return merged_descriptor


def solve_pipeline_capacity(
    descriptor: dict[str, Any],
    nodes: list[dict[str, Any]],
    *,
    safety_margin: float = 1.2,
    require_distributed: bool = False,
    local_layer_budget: int | None = None,
    relay_claims: dict[str, tuple[int, int]] | None = None,
    prefer_all_workers: bool = False,
    required_node_id: str | None = None,
) -> dict[str, Any]:
    """Return an all-or-nothing contiguous layer placement.

    ``capacity_bytes`` must describe currently free memory, not installed
    physical memory. ``reserve_bytes`` is charged once to every participating
    node for allocator, activation, KV-cache and loading workspace headroom.
    Optional ``range_weight_bytes`` quotes replace additive decoder/component
    bytes with an exact device peak for a range and its input/output flags.
    Missing quotes reject that candidate. Host budgets remain independently
    projected into candidate ranges; free host memory never augments VRAM here.
    When ``require_distributed`` is true, a single-node full-model placement is
    deliberately rejected; the returned plan must contain at least two
    participating nodes. This is used by an explicitly distributed request,
    while ordinary capacity inspection remains single-node efficient.
    ``required_node_id`` additionally requires that node to execute a nonempty
    segment; supply uses it for the master instead of rejecting an otherwise
    feasible result after the solver has optimized the master away.

    ``prefer_all_workers`` flips the optimization between feasible placements
    from "fewest nodes" (default: least network hops) to "most nodes" — the
    latter selects the longest feasible chain. Within the selected node count,
    minimize the sum of squared decoder-work fractions divided by device
    capability. Stronger nodes therefore take more work when capacity permits.
    Master-first order preserves the execution contract; it grants no extra
    decoder layers. Source/artifact ranges and byte budgets remain hard gates.
    """

    if required_node_id is not None and (
            not isinstance(required_node_id, str) or not required_node_id
            or required_node_id.strip() != required_node_id):
        raise PipelineCapacityError("required_node_id must be a nonempty node identity")
    safety_margin = _positive_float(safety_margin, "safety_margin")
    if safety_margin < 1.0:
        raise PipelineCapacityError("safety_margin must be at least 1.0")
    original_layer_bytes = descriptor.get("layer_weight_bytes")
    total_layers = _non_negative_int(descriptor.get("total_layers", 0), "total_layers")
    usable, excluded = _normalize_nodes(nodes, total_layers=total_layers)
    quoted_artifacts: dict[tuple[str, tuple[int, int]], tuple[str, tuple[int, ...]]] = {}
    for node in usable:
        for artifact in node.get("layer_artifacts", ()):
            sizes = artifact.get("layer_weight_bytes")
            if sizes is None:
                continue
            identity = (artifact["artifact_sha256"], artifact["layer_range"])
            quote = (artifact["source_model_sha256"], tuple(sizes))
            if identity in quoted_artifacts and quoted_artifacts[identity] != quote:
                raise PipelineCapacityError("conflicting layer_weight_bytes for artifact_sha256")
            quoted_artifacts[identity] = quote
    descriptor = _merge_source_layer_bytes(descriptor, usable)
    layer_bytes, embedding_bytes, per_node_bytes, output_bytes = _descriptor_costs(
        descriptor
    )
    expected_source = str(descriptor.get("source_model_sha256", "") or "").strip().lower()
    missing_prefix = [0]
    for value in original_layer_bytes:
        missing_prefix.append(missing_prefix[-1] + (int(value) == 0))
    # ★ A1 / X 档（Y 档第二条缺口 6）：relay 段节点**不占层、不计容量**，但**算参与节点**。
    #   `scheduler.py` 的 `_get_pipeline_capacity_nodes` 给它打 `capacity_source="relay_exempt"`
    #   （段工件在远端 relay_mid_service，本节点只转发 ⇒ 不需要本地容量预算）。
    #   若把它留在 `usable`：求解器会要求它真装下若干层 ⇒ 必然失败（实测 plan 被拒）。
    #   若整个剔除：`len(usable) < 2` 又会把"master + relay"这种**合法**拓扑拒掉。
    #   ⇒ 单列：不参与分层搜索，但计入"分布式可用节点数"，并以**零层条目**进 assignments。
    relay_only = [n for n in usable if n.get("capacity_source") == "relay_exempt"]
    usable = [n for n in usable if n.get("capacity_source") != "relay_exempt"]
    total_layers = len(layer_bytes)
    # ★ Y 档第二条：relay 段认领的层由**远端段服务**执行 ⇒ 本机层节点只需放下 `[0, k)`。
    #   `local_layer_budget` 由调度层算出（`Scheduler._relay_claimed_layer_prefix`，那里带
    #   重叠 / 角色 / 连续性校验）；缺省 = 全部层（对既有路径零影响）。
    #   ⚠️ 末位本机节点**仍拿 `lm_head`**：主节点收到末节点回的 hidden 后要自己跑
    #   Norm + LM Head（见 `scheduler_pipeline.py:4594-4595` 的推荐拓扑）⇒ 这正是不选
    #   `tail` 段而选 `middle[8,24)` 的原因（tail 只回 token，拿不到 hidden）。
    layer_budget = total_layers if local_layer_budget is None else int(local_layer_budget)
    if not (0 <= layer_budget <= total_layers):
        raise PipelineCapacityError(
            f"local_layer_budget 越界: {layer_budget} / total={total_layers}"
        )
    raw_model_bytes = (
        sum(layer_bytes) + embedding_bytes + per_node_bytes + output_bytes
    )
    prefix = [0]
    for value in layer_bytes:
        prefix.append(prefix[-1] + value)

    checks: dict[tuple[str, int, int], dict[str, Any]] = {}
    capacity_checks: list[dict[str, Any]] = []
    exclusions: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    candidate_exclusions: list[dict[str, Any]] = []

    def exclude(node, start, end, reason_code, budget=None):
        exclusions[(node["node_id"], start, end, reason_code)] = {
            "node_id": node["node_id"], "layer_range": [start, end],
            "reason_code": reason_code, **(budget or {}),
        }

    base = {
        "schema_version": CAPACITY_PLAN_SCHEMA_VERSION,
        "model_id": str(descriptor.get("model_id", "") or ""),
        "model_type": str(descriptor.get("model_type", "") or ""),
        "model_sha256": str(descriptor.get("model_sha256", "") or ""),
        "total_layers": total_layers,
        "raw_model_bytes": raw_model_bytes,
        "safety_margin": safety_margin,
        "candidate_node_count": len(usable),
        "excluded_nodes": excluded,
        "capacity_checks": capacity_checks,
        "candidate_exclusions": candidate_exclusions,
    }
    required_node_index = next((index for index, node in enumerate(usable)
                                if node["node_id"] == required_node_id), None)
    if required_node_id is not None and required_node_index is None:
        return {
            **base, "status": "rejected", "admitted": False,
            "reason_code": "pipeline_required_node_unavailable",
            "reason": f"required executor {required_node_id} has no usable model capacity",
            "required_node_id": required_node_id,
            "assignments": [], "control_only_nodes": [node["node_id"] for node in usable],
        }
    if not usable and not relay_only:
        return {
            **base,
            "status": "rejected",
            "admitted": False,
            "reason_code": (
                "pipeline_distributed_workers_unavailable"
                if require_distributed
                else "pipeline_capacity_nodes_unavailable"
            ),
            "assignments": [],
            "control_only_nodes": [],
        }
    if require_distributed and len(usable) + len(relay_only) < 2:
        return {
            **base,
            "status": "rejected",
            "admitted": False,
            "reason_code": "pipeline_distributed_workers_unavailable",
            "reason": "distributed placement requires at least two usable PC nodes",
            "assignments": [],
            "control_only_nodes": [node["node_id"] for node in usable],
        }

    work_total = prefix[layer_budget] or 1
    range_quotes = {
        node["node_id"]: {quote["layer_range"]: quote for quote in node["range_weight_bytes"]}
        for node in usable if "range_weight_bytes" in node
    }
    if range_quotes:
        base["capacity_cost_mode"] = "exact_range"

    @lru_cache(maxsize=None)
    def placement_objective(candidate):
        costs = [stage_placement_cost(
            (prefix[value[2]] - prefix[value[1]]) / work_total,
            usable[value[0]]["score"], value[4], usable[value[0]]["capacity_bytes"],
        ) for value in candidate]
        return (
            -len(candidate) if prefer_all_workers else len(candidate),
            sum(cost[0] for cost in costs),
            sum(cost[1] for cost in costs),
            -sum(usable[value[0]]["score"] for value in candidate),
            tuple((usable[value[0]]["node_id"], value[1], value[2]) for value in candidate),
        )

    @lru_cache(maxsize=None)
    def search(
        node_index: int,
        cursor: int,
        started: bool,
        used_count: int,
        source_model_sha256: str,
    ):
        if cursor == layer_budget:
            # A mandatory executor cannot be skipped by completing the model
            # on earlier nodes. Past its index, the skip branch below proves use.
            if required_node_index is not None and node_index <= required_node_index:
                return None
            # ★ Y 档第二条：relay 段**算参与节点**（它承载远端段工件），只是不占本机容量
            #   ⇒ "分布式"的判据是 `本机层节点数 + relay 段数 >= 2`，而不是只看前者。
            #   否则 `master(0-8) + relay 段(8-24)` 这种**合法**拓扑会被误拒。
            participating = used_count + len(relay_only)
            if require_distributed and participating < 2:
                return None
            return ()
        if node_index >= len(usable):
            return None
        node = usable[node_index]
        best = None if node["node_id"] == required_node_id else search(
            node_index + 1, cursor, started, used_count, source_model_sha256,
        )
        remaining = layer_budget - cursor
        for count in range(remaining, 0, -1):
            end = cursor + count
            candidates = node.get("candidate_layer_ranges")
            if candidates is not None and (cursor, end) not in candidates:
                exclude(node, cursor, end, "node_candidate_layer_range_unavailable")
                continue
            # ★ 2026-10-03：`layer_budget.max_layers` 是承载上界，始终生效。
            advertised_budget = node.get("layer_budget")
            if advertised_budget is not None and count > advertised_budget["max_layers"]:
                exclude(node, cursor, end, "node_layer_budget_exceeded")
                continue
            # 只有当设备声明"能按分配在本地裁层"（`local_cut=true`）时，`layer_ranges`
            # 才降级为"当前已就绪"的信息、不再限制分配；否则它仍是硬约束 ——
            # 派给设备一个它手上没有工件的区间，请求必然失败。
            if advertised_budget is None or not advertised_budget.get("local_cut"):
                allowed_ranges = node.get("layer_ranges")
                if allowed_ranges is not None and not any(
                    start == cursor and end == allowed_end
                    for start, allowed_end in allowed_ranges
                ):
                    exclude(node, cursor, end, "node_layer_range_not_advertised")
                    continue
            # ★ 2026-10-05（DIST-3 三机实测）：**段类型约束**。
            #
            #   区间包含判据不够：`layer_ranges` 只说"本节点覆盖哪些层"，不区分工件
            #   是首段 / 中间段 / 末段。Y700 广告 `[8,24]`（实为 `mid8-24`，
            #   `mode=middle`）⇒ `[20,24)` 落在该区间内、被照分，而中间段工件
            #   **没有 lm_head / final_norm** ⇒ 必然执行失败（实测报
            #   `remote worker reported a Stage error`）。
            #
            #   段类型来自设备声明的 `segment_mode`（取工件 manifest 的 `mode`）；
            #   未声明的设备不参与本约束，保持旧行为。
            matching_artifact = None
            if advertised_budget is None or not advertised_budget.get("local_cut"):
                artifacts = node.get("layer_artifacts")
                if artifacts is not None:
                    matching_artifact = next((
                        artifact for artifact in artifacts
                        if artifact["layer_range"] == (cursor, end)
                    ), None)
                    if matching_artifact is None:
                        # The worker supplied per-artifact metadata, so ranges
                        # without it are legacy-only and cannot be scheduled
                        # safely as an executable segment.
                        exclude(node, cursor, end, "node_layer_artifact_unavailable")
                        continue
            segment_mode = (
                matching_artifact.get("segment_mode")
                if matching_artifact is not None
                else None if advertised_budget is not None
                and advertised_budget.get("local_cut")
                else node.get("segment_mode")
            )
            if segment_mode == "tail" and end != layer_budget:
                # 末段工件只含末尾层，接不了中间段。
                exclude(node, cursor, end, "node_segment_mode_mismatch")
                continue
            if segment_mode == "middle" and (cursor == 0 or end == layer_budget):
                # 中间段工件既无 embedding 也无 lm_head。
                exclude(node, cursor, end, "node_segment_mode_mismatch")
                continue
            if segment_mode == "head" and cursor != 0:
                # 首段工件只含开头层，接不了后续段。
                exclude(node, cursor, end, "node_segment_mode_mismatch")
                continue
            artifact_source_sha256 = (
                str(matching_artifact.get("source_model_sha256", "") or "")
                if matching_artifact is not None else ""
            )
            if (
                matching_artifact is not None and expected_source
                and artifact_source_sha256 != expected_source
            ):
                exclude(node, cursor, end, "node_source_identity_mismatch")
                continue
            if (
                source_model_sha256
                and artifact_source_sha256
                and source_model_sha256 != artifact_source_sha256
            ):
                exclude(node, cursor, end, "node_source_identity_mismatch")
                continue
            next_source_sha256 = source_model_sha256 or artifact_source_sha256
            segment_bytes = (
                matching_artifact.get("layer_weight_bytes")
                if matching_artifact is not None else None
            )
            if (
                matching_artifact is not None and segment_bytes is None
                and matching_artifact["artifact_sha256"] != str(
                    descriptor.get("model_sha256", "") or ""
                ).strip().lower()
            ):
                exclude(node, cursor, end, "node_artifact_byte_quote_missing")
                continue
            if segment_bytes is None and "layer_weight_bytes" in node:
                segment_bytes = node["layer_weight_bytes"][cursor:end]
            if segment_bytes is None and missing_prefix[end] != missing_prefix[cursor]:
                exclude(node, cursor, end, "node_artifact_byte_quote_missing")
                continue
            raw_bytes = (
                sum(segment_bytes) if segment_bytes is not None
                else prefix[end] - prefix[cursor]
            ) + per_node_bytes
            has_embedding = not started
            has_lm_head = end == layer_budget
            if "load_spec" in node:
                components = node["load_spec"]["components"]
                if (has_embedding != ("embedding" in components)
                        or has_lm_head != ("lm_head" in components)):
                    exclude(node, cursor, end, "node_load_spec_components_mismatch")
                    continue
            range_quote = None
            if node["node_id"] in range_quotes:
                range_quote = range_quotes[node["node_id"]].get((cursor, end))
                if range_quote is None:
                    exclude(node, cursor, end, "node_range_weight_quote_missing")
                    continue
                if (range_quote["has_embedding"] != has_embedding
                        or range_quote["has_lm_head"] != has_lm_head):
                    exclude(node, cursor, end, "node_range_weight_components_mismatch")
                    continue
                # Nonadditive peaks already include decoder and boundary units.
                # The common budget function still applies reserve/margin once.
                raw_bytes = range_quote["weight_bytes"]
            elif "retained_component_bytes" in node:
                raw_bytes += node["retained_component_bytes"] - per_node_bytes
            else:
                if has_embedding:
                    raw_bytes += embedding_bytes
                if has_lm_head:
                    raw_bytes += output_bytes
            budget = _node_budget(raw_bytes, node, safety_margin)
            required = budget["required_bytes"]
            checks[(node["node_id"], cursor, end)] = {
                "node_id": node["node_id"], "layer_range": [cursor, end],
                "raw_weight_bytes": raw_bytes, **budget,
                "reason_code": "node_capacity_insufficient" if budget["shortfall_bytes"] else "",
            }
            if budget["shortfall_bytes"]:
                exclude(node, cursor, end, "node_capacity_insufficient", budget)
                continue
            tail = search(
                node_index + 1, end, True, used_count + 1, next_source_sha256,
            )
            if tail is None:
                continue
            item = (
                node_index,
                cursor,
                end,
                raw_bytes,
                required,
                has_embedding,
                has_lm_head,
                node["layer_artifacts"].index(matching_artifact)
                if matching_artifact is not None else None,
            )
            candidate = (item,) + tail
            if best is None:
                best = candidate
                continue
            if placement_objective(candidate) < placement_objective(best):
                best = candidate
        return best

    solved = search(0, 0, False, 0, "")
    capacity_checks.extend(checks[key] for key in sorted(checks))
    candidate_exclusions.extend(exclusions[key] for key in sorted(exclusions))
    if solved is None:
        allocatable_bytes = sum(
            _node_budget(0, node, safety_margin)["allocatable_bytes"]
            for node in usable
        )
        if any("candidate_layer_ranges" in node for node in usable):
            without_candidates = [
                {key: value for key, value in node.items() if key != "candidate_layer_ranges"}
                for node in usable
            ]
            relaxed = solve_pipeline_capacity(
                descriptor, without_candidates + relay_only, safety_margin=safety_margin,
                require_distributed=require_distributed, local_layer_budget=local_layer_budget,
                relay_claims=relay_claims, required_node_id=required_node_id,
                prefer_all_workers=prefer_all_workers,
            )
            if relaxed.get("admitted"):
                return {
                    **base, "status": "rejected", "admitted": False,
                    "reason_code": "pipeline_candidate_layer_range_unsatisfied",
                    "reason": "source coverage or staging ranges cannot form a capacity-safe contiguous split",
                    "assignments": [], "allocatable_bytes": allocatable_bytes,
                    "control_only_nodes": [node["node_id"] for node in usable],
                }
        segment_constrained = any(
            node.get("segment_mode") is not None
            or node.get("layer_artifacts") is not None
            for node in usable
        )
        if segment_constrained:
            segmentless_nodes = [
                {
                    key: value for key, value in node.items()
                    if key not in {"segment_mode", "layer_artifacts"}
                }
                for node in usable
            ]
            segmentless = solve_pipeline_capacity(
                descriptor,
                segmentless_nodes + relay_only,
                safety_margin=safety_margin,
                require_distributed=require_distributed,
                local_layer_budget=local_layer_budget,
                relay_claims=relay_claims,
                required_node_id=required_node_id,
                prefer_all_workers=prefer_all_workers,
            )
            if segmentless.get("admitted"):
                return {
                    **base,
                    "status": "rejected",
                    "admitted": False,
                    "reason_code": "pipeline_segment_contract_unsatisfied",
                    "reason": (
                        "advertised layer artifact modes or source identities "
                        "cannot form the requested pipeline"
                    ),
                    "allocatable_bytes": allocatable_bytes,
                    **({} if range_quotes else {"raw_capacity_deficit_bytes": max(
                        0, raw_model_bytes - allocatable_bytes
                    )}),
                    "assignments": [],
                    "control_only_nodes": [node["node_id"] for node in usable],
                }
        # Distinguish a topology contract failure from a plain memory
        # shortage. Re-run the same admission with the range contract
        # removed; only a plan that becomes admissible proves the advertised
        # ranges are the cause of the rejection.
        range_constrained = any(node.get("layer_ranges") is not None for node in usable)
        unconstrained_admission = False
        if range_constrained:
            unconstrained_nodes = [
                {
                    key: value for key, value in node.items()
                    if key not in {
                        "layer_ranges", "layer_artifacts", "segment_mode",
                    }
                }
                for node in usable
            ]
            unconstrained = solve_pipeline_capacity(
                descriptor,
                unconstrained_nodes + relay_only,
                safety_margin=safety_margin,
                require_distributed=require_distributed,
                local_layer_budget=local_layer_budget,
                relay_claims=relay_claims,
                required_node_id=required_node_id,
                prefer_all_workers=prefer_all_workers,
            )
            unconstrained_admission = bool(unconstrained.get("admitted"))
        if range_constrained and unconstrained_admission:
            # ★ 2026-10-08：把「缺口区间」写进 reason —— 现场多次把本码误读成"工件区间配置错"，
            #   真因往往是 worker 掉线/未准入（见 `_uncovered_layer_ranges` 的说明）。
            # ★ 2026-10-09（真机实测 BUG）：覆盖判定**必须把 master 自己的本地段算进去**。
            #   原先只收集 worker 的 advertised `layer_ranges`，于是只要有任意 worker 声明了区间
            #   （`range_constrained=True`），master 承担的前缀 `[0, local_layer_budget)` 就被判成
            #   "未覆盖" ⇒ `master[0,20) + android[20,24)` 这种**正确拓扑恒被拒**
            #   （实测：`reason=pipeline_layer_range_coverage_insufficient`、`uncovered=[0,20)`，
            #   且求解器本身 admitted=True ⇒ 前后自相矛盾）。
            #   worker 的 `layer_ranges` 是**执行契约**（不能给它分配区间外的层），
            #   但它不该反过来否定 master 的本地段。
            covered_ranges = [
                item
                for node in usable
                for item in (node.get("layer_ranges") or [])
            ]
            if local_layer_budget is not None and layer_budget > 0:
                # ★ 只在**显式**给出 `local_layer_budget` 时才算 master 的段：
                #   budget is None 表示"无本地段约束"（整模 / 未声明），**不是**"master 跑全部"，
                #   此时仍按 worker 的 advertised ranges 判覆盖（既有语义，见
                #   `test_advertised_layer_ranges_reject_uncovered_cursor`）。
                covered_ranges.append([0, int(layer_budget)])
            gaps = _uncovered_layer_ranges(
                covered_ranges,
                int(descriptor.get("total_layers") or 0),
            )
            if not gaps:
                # ★ 2026-10-09（真机实测）：**缺口为空却仍被拒** ⇒ 不是"区间不连续"，而是
                #   "每个节点必须装下自己那一段"这条硬约束下**某节点的空闲内存不够**
                #   （总量 `raw_capacity_deficit_bytes=0`，但切分后单节点不够 —— 实测本机
                #   `free=1.7GB` 却要跑 fp32 模型的 20 层，就是此例）。
                #   原先一律报 `pipeline_layer_range_coverage_insufficient`，现场把
                #   "内存不够"误读成"工件区间配置错"，浪费大量排查时间（见 #78 缺口 2）。
                return {
                    **base,
                    "status": "rejected",
                    "admitted": False,
                    "reason_code": "pipeline_capacity_single_node_insufficient",
                    "reason": (
                        "advertised layer_ranges constrain the contiguous split so that no "
                        "single node can host its own segment within its free memory "
                        "(aggregate capacity is sufficient; check per-node free memory)"
                    ),
                    "allocatable_bytes": allocatable_bytes,
                    **({} if range_quotes else {"raw_capacity_deficit_bytes": max(0, raw_model_bytes - allocatable_bytes)}),
                    "assignments": [],
                    "uncovered_layer_ranges": [],
                    "control_only_nodes": [node["node_id"] for node in usable],
                }
            reason = (
                "advertised layer_ranges cannot cover the requested contiguous layer interval"
            )
            if gaps:
                reason = (
                    f"{reason}（未覆盖区间: "
                    f"{', '.join(f'[{start},{stop})' for start, stop in gaps)}；"
                    f"若这些区间本应有节点承担，检查该节点是否掉线/未准入）"
                )
            return {
                **base,
                "status": "rejected",
                "admitted": False,
                "reason_code": "pipeline_layer_range_coverage_insufficient",
                "reason": reason,
                "allocatable_bytes": allocatable_bytes,
                **({} if range_quotes else {"raw_capacity_deficit_bytes": max(0, raw_model_bytes - allocatable_bytes)}),
                "assignments": [],
                "uncovered_layer_ranges": [list(gap) for gap in gaps],
                "control_only_nodes": [node["node_id"] for node in usable],
            }
        return {
            **base,
            "status": "rejected",
            "admitted": False,
            "reason_code": (
                "pipeline_distributed_capacity_insufficient"
                if require_distributed
                else "pipeline_cluster_capacity_insufficient"
            ),
            "allocatable_bytes": allocatable_bytes,
            **({} if range_quotes else {"raw_capacity_deficit_bytes": max(0, raw_model_bytes - allocatable_bytes)}),
            "assignments": [],
            "control_only_nodes": [node["node_id"] for node in usable],
        }

    assignments = []
    used_ids: set[str] = set()
    for (
        node_index, start, end, raw_bytes, required,
        has_embedding, has_lm_head, artifact_index,
    ) in solved:
        node = usable[node_index]
        artifact = (
            node["layer_artifacts"][artifact_index]
            if artifact_index is not None else None
        )
        used_ids.add(node["node_id"])
        assignment = {
            "node_id": node["node_id"],
            "role": node["role"],
            "start_layer": start,
            "end_layer": end,
            "layers_count": end - start,
            "has_embedding": has_embedding,
            "has_lm_head": has_lm_head,
            "raw_weight_bytes": raw_bytes,
            "required_bytes": required,
            "capacity_bytes": node["capacity_bytes"],
            "headroom_bytes": node["capacity_bytes"] - required,
            "reserve_bytes": node["reserve_bytes"],
            "runtime_multiplier": node["runtime_multiplier"],
            "execution_device": node["execution_device"],
            "capacity_source": node["capacity_source"],
            "score": node["score"],
            "placement_weight": placement_weight(node["score"]),
            "placement_work_fraction": (prefix[end] - prefix[start]) / work_total,
            **_node_budget(raw_bytes, node, safety_margin),
        }
        for field in ("configured_reserve_bytes", "retained_component_reserve_bytes",
                      "host_required_bytes", "host_capacity_bytes"):
            if field in node:
                assignment[field] = node[field]
        if "retained_component_bytes" in node:
            assignment["retained_component_bytes"] = node["retained_component_bytes"]
        if "load_spec" in node:
            assignment.update(load_spec=RuntimeLoadSpec.from_dict(node["load_spec"]).to_dict(),
                              load_spec_digest=node["load_spec_digest"])
        if "layer_ranges" in node:
            assignment["layer_ranges"] = [
                [range_start, range_end]
                for range_start, range_end in node["layer_ranges"]
            ]
        if artifact is not None:
            assignment["layer_artifact"] = {
                **artifact,
                "layer_range": list(artifact["layer_range"]),
            }
        if node["node_id"] in range_quotes:
            quote = range_quotes[node["node_id"]][(start, end)]
            assignment["range_weight_quote"] = {**quote, "layer_range": [start, end]}
        assignments.append(assignment)

    selected_layer_bytes = 0
    for index, start, end, raw, _, embedding, lm_head, artifact_index in solved:
        node = usable[index]
        if node["node_id"] in range_quotes:
            artifact = node["layer_artifacts"][artifact_index] if artifact_index is not None else {}
            sizes = artifact.get("layer_weight_bytes")
            if sizes is None:
                sizes = node.get("layer_weight_bytes", layer_bytes)[start:end]
            # Model size remains the complete decoder bytes, not its device peak.
            selected_layer_bytes += sum(sizes)
        else:
            selected_layer_bytes += raw - (
                node["retained_component_bytes"] if "retained_component_bytes" in node else
                per_node_bytes + (embedding_bytes if embedding else 0) + (output_bytes if lm_head else 0)
            )
    base["raw_model_bytes"] = (
        selected_layer_bytes + sum(layer_bytes[layer_budget:])
        + embedding_bytes + per_node_bytes + output_bytes
    )

    plan_identity = {
        "descriptor_sha256": str(descriptor.get("model_sha256", "") or ""),
        "model_id": base["model_id"],
        "safety_margin": safety_margin,
        "assignments": [],
    }
    for item in assignments:
        identity_item = {
            key: item[key]
            for key in (
                "node_id", "start_layer", "end_layer", "has_embedding",
                "has_lm_head", "required_bytes", "capacity_bytes",
            )
        }
        if "retained_component_bytes" in item:
            identity_item["retained_component_bytes"] = item["retained_component_bytes"]
        for field in ("host_required_bytes", "host_capacity_bytes"):
            if field in item:
                identity_item[field] = item[field]
        if "range_weight_quote" in item:
            identity_item["range_weight_quote"] = item["range_weight_quote"]
        if "load_spec" in item:
            identity_item["load_spec_digest"] = item["load_spec_digest"]
        if "layer_ranges" in item:
            identity_item["layer_ranges"] = item["layer_ranges"]
        if "layer_artifact" in item:
            identity_item["layer_artifact"] = item["layer_artifact"]
        plan_identity["assignments"].append(identity_item)
    plan_id = hashlib.sha256(
        json.dumps(plan_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    full_model_fits = []
    for node in usable:
        candidates = node.get("candidate_layer_ranges")
        if candidates is not None and (0, total_layers) not in candidates:
            continue
        if "load_spec" in node and not {"embedding", "lm_head"}.issubset(node["load_spec"]["components"]):
            continue
        allowed_ranges = node.get("layer_ranges")
        if allowed_ranges is not None and not any(
            start == 0 and total_layers == end
            for start, end in allowed_ranges
        ):
            continue
        full_bytes = max(raw_model_bytes, base["raw_model_bytes"])
        if "layer_weight_bytes" in node:
            full_bytes = sum(node["layer_weight_bytes"]) + node.get(
                "retained_component_bytes", embedding_bytes + per_node_bytes + output_bytes,
            )
        if node["node_id"] in range_quotes:
            full_quote = range_quotes[node["node_id"]].get((0, total_layers))
            if (full_quote is None or not full_quote["has_embedding"]
                    or not full_quote["has_lm_head"]):
                continue
            full_bytes = full_quote["weight_bytes"]
        required = _required_bytes(full_bytes, node, safety_margin)
        if required <= node["capacity_bytes"]:
            full_model_fits.append(node["node_id"])

    objective = placement_objective(solved)
    base["placement_policy"] = {
        "prefer_all_workers": bool(prefer_all_workers),
        "work_basis": "descriptor_decoder_weight_bytes",
        "objective": "capability_weighted_work_square",
    }
    base["placement_objective"] = {
        "node_count": len(assignments),
        "weighted_work_cost": objective[1],
        "memory_pressure_cost": objective[2],
        "score_total": -objective[3],
    }
    participating_count = len(assignments) + len(relay_only)
    return {
        **base,
        "status": "admitted",
        "admitted": True,
        "reason_code": "distributed_forced" if require_distributed else "",
        "plan_id": plan_id,
        "assignments": assignments + _relay_zero_layer_assignments(
            relay_only, total_layers, relay_claims
        ),
        "control_only_nodes": [
            node["node_id"] for node in usable if node["node_id"] not in used_ids
        ],
        "participating_node_count": participating_count,
        "single_node_full_model_candidates": full_model_fits,
        "aggregate_only": participating_count > 1,
        "require_distributed": bool(require_distributed),
    }
