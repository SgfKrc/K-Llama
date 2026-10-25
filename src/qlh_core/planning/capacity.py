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
    # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**可切区间**（可选，`min_layer`/`max_layer`）。
    #
    #   `local_cut` 只解锁**切分粒度**（能在工件覆盖的区间内任意切），**不解锁工件边界**。
    #   持有**整模**、真能自裁的 worker 不需要这两项（给了也不影响：没声明就不检查，
    #   行为与改动前逐字节一致）。而只持有**裁好的部分工件**的执行器 —— 例如本机 CPU 段的
    #   `q35-9b-local-mid8-24.gguf`（只含 `[8,24)`）—— **必须**声明它们；否则求解器会把
    #   工件覆盖不到的层派给它，而那种失败发生在**执行期**（层号越界/权重缺失），
    #   不是规划期报错，属于"部署显示成功、第一次推理才炸"的最坏形态（真机实测踩到过：
    #   `assignments=[('master','cpu',0,8), ...]`，而工件是 `mid8-24`）。
    bounds: dict[str, int] = {}
    low = value.get("min_layer")
    high = value.get("max_layer")
    if (low is None) != (high is None):
        raise PipelineCapacityError(
            f"{field}.min_layer and {field}.max_layer must be given together"
        )
    if low is not None and high is not None:
        low_value = _non_negative_int(low, f"{field}.min_layer")
        high_value = _non_negative_int(high, f"{field}.max_layer")
        if high_value <= low_value:
            raise PipelineCapacityError(
                f"{field}.max_layer must be greater than {field}.min_layer"
            )
        bounds = {"min_layer": low_value, "max_layer": high_value}
    return {
        "available_bytes": available_bytes,
        "per_layer_bytes": per_layer_bytes,
        "max_layers": max_layers,
        "local_cut": local_cut,
        **bounds,
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
    seen: set[tuple[str, str]] = set()
    for raw in nodes:
        if not isinstance(raw, dict):
            raise PipelineCapacityError("each node capacity record must be an object")
        node_id = str(raw.get("node_id", "") or "").strip()
        # ★ 2026-10-25（LOCAL-EXECUTORS-01）：唯一性判据由 `node_id` 放宽到
        #   `(node_id, executor_id)` —— 同一物理节点上的 GPU 段与 CPU 段**必须**能共存
        #   （本票的目标形态）。这与阶段 2 对 `ExecutionPlan` / `validate_pipeline_nodes`
        #   的改动**同源**；那两处改了、这里漏了，实测环境门就卡在
        #   `pipeline_capacity_descriptor_invalid: node_id must be non-empty and unique`。
        #   `executor_id` 缺省 "" ⇒ 键退化为 `(node_id, "")` ⇒ 与改动前逐位等价。
        executor_id = str(raw.get("executor_id") or "").strip()
        if not node_id or (node_id, executor_id) in seen:
            raise PipelineCapacityError("node_id must be non-empty and unique")
        seen.add((node_id, executor_id))
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
        # ★ 2026-10-25（#117 收尾）：**物化模式与主机内存额度必须随节点透传到求解器**。
        #   `_node_budget` 要据此把主机内存计入可分配容量 —— `safetensors_layer_offload`
        #   下权重驻留主机内存，只按 `capacity_bytes`（对 pytorch+cuda 是**显存**，实测
        #   7451181056 ≈ 7.45 GB）参与求解，会把 master 从能扛 24 层压到只有 8 层，
        #   主机那 ~32 GB 全程闲置（这正是"容量不准入（只算了 VRAM）"的根因）。
        #   上面是白名单式字段挑拣，二者此前不在其中 ⇒ `_node_budget` 永远读不到，
        #   求解器侧的改动会形同虚设。仅在实际出现时附加：未标注物化的节点
        #   （既有拓扑与既有测试）行为完全不变。
        if raw.get("materialization"):
            normalized["materialization"] = str(raw["materialization"])
        if "host_capacity_bytes" in raw:
            normalized["host_capacity_bytes"] = _non_negative_int(
                raw["host_capacity_bytes"], "host_capacity_bytes")
        # ★ 2026-10-25：**每层推理成本标定值**（毫秒，相对量）。求解器原先按解码器
        #   **字节**份额均衡，把最慢的设备当成和最快的设备一样贵 —— 实测三机每层成本
        #   差 2.3 倍（master 12.6 / rog-5060 8.7 / android 20.1 ms），典型后果是 rog
        #   扛 24 层成为瓶颈而最慢的 android 只扛 8 层。标定值由**调用方**（读
        #   `QLH_LAYER_COST_MS`）填入：本模块是纯求解器，不读环境（见
        #   tests/test_refactor_capacity_load_spec.py 的 ambient-state 契约）。
        #   缺省不填 ⇒ 求解器按 1.0，与旧行为逐字节等价。
        if "layer_cost_ms" in raw:
            cost = raw["layer_cost_ms"]
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0:
                raise PipelineCapacityError(
                    f"node[{node_id}].layer_cost_ms must be a positive number")
            normalized["layer_cost_ms"] = float(cost)
        # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**执行器身份与共享内存域**。
        #   `executor_id` 缺省 ""（= 该节点的默认/主执行器 = 今天的行为）；
        #   `capacity_domain_id` 缺省 = `node_id`（= 每节点一个独立域 = 今天的行为）。
        #   只有显式给出 `capacity_domain_id` **且** 同域有多个执行器时，容量才按
        #   **物理机总账**校验（见 `validate_capacity_domains`）—— 否则同一台物理机上的
        #   两个执行器会各自按整机可用内存记账，静默给出物理上装不下的计划。
        if "executor_id" in raw:
            normalized["executor_id"] = str(raw["executor_id"] or "")
        if raw.get("capacity_domain_id"):
            normalized["capacity_domain_id"] = str(raw["capacity_domain_id"])
        if "domain_total_bytes" in raw:
            normalized["domain_total_bytes"] = _non_negative_int(
                raw["domain_total_bytes"], "domain_total_bytes")
        # ★ 2026-10-25：**域内申报额**（专用通道，与 `host_capacity_bytes` 分开）。
        #   调用方用它来表达"这个执行器在该共享域里占多少"，且**每次重新推导** ⇒ 幂等。
        #   不能复用 `host_capacity_bytes`：那个字段还承载 `safetensors_layer_offload`
        #   的 host 预算（见 `_node_budget`），扣减它会连带削掉 offload 额度。
        if "domain_allocatable_bytes" in raw:
            normalized["domain_allocatable_bytes"] = _non_negative_int(
                raw["domain_allocatable_bytes"], "domain_allocatable_bytes")
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
                # ★ 2026-10-23（#117）：`layer_ranges` 是"**能承担**哪些层段"，
                #   `layer_artifacts` 是"**现成**有哪些段工件" —— 二者不必相等。
                #   自裁节点声明整域 `[0, total)`、手上可能只留着一个旧段，要求
                #   集合相等会直接判描述器非法（实测 `node[rog-5060].layer_artifacts
                #   and layer_ranges must match`，供给恒失败）。正确约束是**区间
                #   覆盖**：每个现成工件都要落在声明的可承担区间内。
                advertised_ranges = [
                    tuple(item) for item in normalized["layer_ranges"]
                ]
                if not all(
                    any(
                        low <= item["layer_range"][0] and item["layer_range"][1] <= high
                        for low, high in advertised_ranges
                    )
                    for item in normalized["layer_artifacts"]
                ):
                    raise PipelineCapacityError(
                        f"node[{node_id}].layer_artifacts must be covered by layer_ranges"
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
    #
    # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**段类型必须参与排序**。
    #   排序决定 DFS 的遍历次序，而 DFS 是**单向**的（`cursor` 只前进、不回头）
    #   ⇒ 段类型必须与层顺序一致。实测（`scripts/probe_local_executor_placement.py`）：
    #   `fastest`（`prefer_all_workers` 为真、目标第一项就是 `-len(candidate)`，3 段必
    #   优于 2 段）下求解器仍只输出 2 段 —— 因为本机 CPU 段**继承了主段的
    #   `role="master"`**，与主段同组后按容量降序排到**第一位**（11.2 GB > 主段 6.4 GB），
    #   而它是 `middle`，在 `cursor=0` 被 `cursor != 0` 拒掉后再不被回看 ⇒ 结构性跳过。
    #
    #   次序与理由（**不是**"最灵活就排最后"）：
    #     head (0)      —— 强制首段（必须含 embedding），必须最先落位；
    #     未声明 (0.5)  —— 最灵活，可作任意段 ⇒ **越早越好**：它选择多，先安排不会
    #                      挤掉别人。放到最后反而会被有约束的节点占满位置 ⇒ 无解
    #                      （实测 `test_per_artifact_modes_select_the_matching_range_and_identity`：
    #                      未声明 mode 的 `edge` 排到 `tail` 之后 ⇒ tail 先占末段 ⇒
    #                      `edge` 无处安放 ⇒ `admitted=False`）；
    #     middle (1)    —— 不能是首段也不能是末段；
    #     tail (2)      —— 强制末段，约束最强，兜底。
    mode_order = {"head": 0, "": 0.5, "middle": 1, "tail": 2}
    usable.sort(
        key=lambda node: (
            node["role"] != "master",
            mode_order.get(str(node.get("segment_mode") or "").lower(), 0.5),
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
    # ★ 2026-10-25（#117 收尾）：**offload 节点的可分配额度应以主机内存为准**。
    #   `safetensors_layer_offload` 的语义是权重**驻留主机内存**、显存只放下当前执行的
    #   那一个单元（见 `_LayerOffloadRuntime`：逐执行单元 stage → 用完释放）。但求解器
    #   此前只认 `capacity_bytes`，而对 `pytorch`+`cuda` 的 master 那是**显存**
    #   （实测 7451181056 ≈ 7.45 GB）⇒ master 只能拿到显存装得下的 8 层，主机那 ~32 GB
    #   全程闲置（实测 master 恒为 `[0,8)`）。这正是"容量不准入（只算了 VRAM）"的根因。
    #   上游 `project_supply_capacity_nodes` 已按 `host_capacity_bytes` 把**装不下的候选
    #   区间筛掉**（`host_weight_peak_bytes × margin + reserve > host_available` 即跳过），
    #   因此这里把额度换成主机额度是**自洽**的：候选合法性由 host 峰值决定，额度也由
    #   host 决定，不再拿显存去衡量放在主机里的权重。
    #   仅在该物化被显式标注时生效；未标注的节点（既有拓扑与既有测试）行为完全不变。
    capacity = node.get("capacity_bytes", 0)
    if node.get("materialization") == "safetensors_layer_offload":
        host_capacity = node.get("host_capacity_bytes")
        if host_capacity:
            capacity = host_capacity
    return capacity_budget(
        raw_bytes, capacity_bytes=capacity,
        reserve_bytes=node["reserve_bytes"],
        runtime_multiplier=node.get("runtime_multiplier", 1.0),
        safety_margin=safety_margin,
    )


def _required_bytes(raw_bytes: int, node: dict[str, Any], safety_margin: float) -> int:
    """Compatibility for manual plans and existing budget channels."""
    return _node_budget(raw_bytes, node, safety_margin)["required_bytes"]


def capacity_domains(usable: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One entry per **shared physical budget**.

    ★ 2026-10-25（LOCAL-EXECUTORS-01）：一台物理机可以承载多个执行器（例如
    master 的 GPU 段 + 同一台机器的 CPU 段）。它们共享同一份 RAM，但容量是按
    `node_id` 记账的 ⇒ 两个执行器会各自按整机可用内存申报，求解器则把它们当成
    两台独立物理机，于是给出**物理上装不下**的计划却 `admitted=True`（实测：
    两节点各 200 容量、合计需求 242，仍被准入，且每段都显示 headroom=79）。

    `capacity_domain_id` 就是这个共享域的键；**缺省 = `node_id`**，即每个节点自成一域
    ——这正是今天的行为，所以既有调用方的记账**逐字节不变**。
    """
    domains: dict[str, dict[str, Any]] = {}
    for node in usable:
        domain_id = str(node.get("capacity_domain_id") or node["node_id"])
        entry = domains.get(domain_id)
        if entry is None:
            entry = domains[domain_id] = {"domain_total_bytes": None, "executors": []}
        total = node.get("domain_total_bytes")
        if total is not None:
            total = int(total)
            if entry["domain_total_bytes"] is None:
                entry["domain_total_bytes"] = total
            elif int(entry["domain_total_bytes"]) != total:
                raise PipelineCapacityError(
                    f"capacity domain {domain_id!r} declares conflicting "
                    f"domain_total_bytes ({entry['domain_total_bytes']} vs {total})")
        # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**域账必须用同一类资源**。
        #   `capacity_bytes` 对 CUDA 节点是 **VRAM**（`capacity_source="gpu.vram_free_gb"`），
        #   对纯 CPU/GGUF 执行器是 **RAM** —— 直接把两者相加会把两种资源混成一份总账。
        #   实测：主段 VRAM 6.0 GB + 本机 CPU 段 RAM 10.2 GB = 16.2 GB > 总账 14.6 GB
        #   ⇒ 报 `pipeline_capacity_domain_exceeded` 超额 1.56 GB，而物理上并不冲突
        #   （VRAM 与 RAM 是两份独立资源）。
        #   共享**内存**域的本质是"同一份 RAM"，所以按主机内存记账，优先级：
        #     1. `domain_allocatable_bytes` —— **调用方显式声明的域内额度**。这是唯一
        #        幂等的通道：份额由声明值推导（例如按 `QLH_LOCAL_CPU_EXECUTOR_HOST_FRACTION`
        #        从 `domain_total_bytes` 重新计算），而不是"在上一次结果上做减法"。
        #        实测教训：先前直接对 `host_capacity_bytes` 原地扣减，而
        #        `_get_pipeline_capacity_nodes` **会被多次调用、每次重建 records**
        #        ⇒ 扣除只对那一次返回值生效，供给拿到的仍是未扣的快照 ⇒ 超额照旧。
        #        也不能直接改 `host_capacity_bytes`：它另有用途（`safetensors_layer_offload`
        #        的 host 预算，见 `_node_budget`），扣它会连带削掉 offload 额度。
        #     2. `host_capacity_bytes` —— 主机内存额度。
        #     3. `capacity_bytes` —— 兜底（对纯 CPU/GGUF 节点与前两者同义）。
        explicit_budget = node.get("domain_allocatable_bytes")
        if explicit_budget is not None:
            host_budget = explicit_budget
        elif node.get("host_capacity_bytes") is not None:
            host_budget = node["host_capacity_bytes"]
        else:
            host_budget = node["capacity_bytes"]
        entry["executors"].append({
            "node_id": node["node_id"],
            "executor_id": str(node.get("executor_id") or ""),
            "allocatable_bytes": max(0, int(host_budget) - int(node["reserve_bytes"])),
        })
    return domains


def validate_capacity_domains(
    usable: list[dict[str, Any]],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Fail closed when one domain's executors claim more than its total.

    只对**显式声明了总账**的域校验（`domain_total_bytes is not None`）：单执行器域与
    未声明总账的域一律视为"无需跨执行器校验"，因此既有拓扑与既有测试零影响。
    方向是**宁可多拒**（安全方向）：过度拒绝只是少用一台机器，静默超配会让部署在
    真实内存压力下崩掉。
    """
    domains = capacity_domains(usable)
    report: dict[str, dict[str, Any]] = {}
    admitted = True
    for domain_id, entry in domains.items():
        total = entry["domain_total_bytes"]
        claimed = sum(item["allocatable_bytes"] for item in entry["executors"])
        excess = 0 if total is None else max(0, claimed - int(total))
        if excess:
            admitted = False
        report[domain_id] = {
            "domain_total_bytes": total,
            "claimed_allocatable_bytes": claimed,
            "excess_bytes": excess,
            "executors": entry["executors"],
        }
    return admitted, report


def _domain_report_is_informative(report: dict[str, dict[str, Any]]) -> bool:
    """有无**跨执行器**域需要报告（单执行器域不占篇幅、也不改变既有输出形态）。"""
    return any(entry["domain_total_bytes"] is not None or len(entry["executors"]) > 1
               for entry in report.values())


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


def _self_cut_segment_bytes(node, start, end) -> list[int] | None:
    """Per-layer byte quote for a self-cutting node, derived from its own artifacts.

    Such a node holds no artifact matching the *requested* range, but every artifact
    reports per-layer bytes for its own range — and per-layer size is a property of
    the model, so the average is a sound quote for any range. Without this the solver
    falls back to whole-model accounting and rejects every assignment.
    """
    total_layers = 0
    total_bytes = 0
    for artifact in node.get("layer_artifacts") or ():
        quote = artifact.get("layer_weight_bytes")
        if not isinstance(quote, (list, tuple)) or not quote:
            continue
        positive = []
        for value in quote:
            try:
                size = int(value)
            except (TypeError, ValueError):
                continue
            if size > 0:
                positive.append(size)
        if not positive:
            continue
        total_layers += len(positive)
        total_bytes += sum(positive)
    if total_layers <= 0 or total_bytes <= 0:
        return None
    per_layer = total_bytes // total_layers
    if per_layer <= 0:
        return None
    return [per_layer] * max(0, int(end) - int(start))


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
    # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**同机容量去重**。
    #   一台物理机可以承载多个执行器（master 的 GPU 段 + 同机 CPU 段），它们共享同一份
    #   RAM，但容量按 `node_id` 记账 ⇒ 两个执行器各自按整机可用内存申报，求解器把它们
    #   当成两台独立物理机，于是给出**物理上装不下**的计划却 `admitted=True`（实测：
    #   两节点各 200 容量、合计需求 242，仍被准入，且每段都显示 headroom=79）。
    #   位置：relay 已剔除（relay 段不占层、不计容量，不该参与域账），且 `base` 已构造
    #   （拒绝分支与既有分支保持同形）。
    #   生效条件：只对**显式声明总账**的域校验（`domain_total_bytes`）。未声明 ⇒ 每节点
    #   自成一域、一律不校验 ⇒ 既有拓扑与既有测试零影响；域报告也只在"有跨执行器域"
    #   时才写进结果，避免改动既有输出形态。
    domains_admitted, domain_report = validate_capacity_domains(usable)
    if _domain_report_is_informative(domain_report):
        base["capacity_domains"] = domain_report
    if not domains_admitted:
        return {
            **base,
            "status": "rejected",
            "admitted": False,
            "reason_code": "pipeline_capacity_domain_exceeded",
            "reason": ("同一物理内存域内的执行器申报额度之和超过该域总账；"
                       "按节点分别记账会让同机多执行器各自按整机可用内存申报。"),
            "assignments": [],
            "control_only_nodes": [node["node_id"] for node in usable],
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
    # ★ 2026-10-25：切点求解的**成本加权**。`placement_work_fraction` 原先只按解码器
    #   **字节**算份额，于是求解器把"每层 20.1 ms 的 android"和"每层 8.7 ms 的 rog"
    #   当成同等昂贵，均衡结果与真实端到端步时脱节（实测：rog 扛 24 层成瓶颈，最慢的
    #   android 却只扛 8 层）。这里把每个执行器的字节份额乘以它的**每层成本倍数**，
    #   `stage_placement_cost` 本身不改 —— 它的均衡条件 `w_i/w_j = weight_i/weight_j`
    #   于是作用于"成本"而非"字节"，这正是我们要的。
    #
    #   归一化与缺省：
    #     * 全部未标定（`costs` 为空）⇒ 所有倍数恒为 1.0 ⇒ **与旧行为逐字节等价**
    #       （可作回归门）。
    #     * 有标定时以**最快**者为基准 1.0；未标定的执行器取"已标定中最慢"的倍数，
    #       避免它因为没有数据反而被当成最快而被**过度分配**（安全方向）。
    #     * `work_total` 保持**未加权**：它是同一候选空间里的公共常数，公共缩放不改变
    #       argmin（目标第一项整体乘同一个因子），而让它随分配变化会破坏
    #       `placement.py` 的可加性论证（suffix optimum 依赖目标可加）。
    _calibrated = [float(node["layer_cost_ms"]) for node in usable
                   if node.get("layer_cost_ms")]
    _cost_reference = min(_calibrated) if _calibrated else 1.0

    def layer_cost_multiplier(node):
        cost = node.get("layer_cost_ms")
        if not cost:
            return (max(_calibrated) / _cost_reference) if _calibrated else 1.0
        return float(cost) / _cost_reference

    range_quotes = {
        node["node_id"]: {quote["layer_range"]: quote for quote in node["range_weight_bytes"]}
        for node in usable if "range_weight_bytes" in node
    }
    if range_quotes:
        base["capacity_cost_mode"] = "exact_range"

    @lru_cache(maxsize=None)
    def placement_objective(candidate):
        costs = [stage_placement_cost(
            ((prefix[value[2]] - prefix[value[1]]) / work_total)
            * layer_cost_multiplier(usable[value[0]]),
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
            # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**`local_cut` 只解锁切分粒度，不解锁工件边界**。
            #
            #   `local_cut=true` 的原意是"本设备手上有整模，能按分配在本地裁层"⇒ 任意区间
            #   都行，故 `layer_ranges` 降级为信息、不再约束。但本机 CPU 段拿的是**裁好的
            #   部分工件**（`q35-9b-local-mid8-24.gguf` 只含 `[8,24)`），它**不能**自裁到
            #   任意区间。此前直接复用 `local_cut` ⇒ 边界失效 ⇒ 真机实测把 `[0,8)` 派给
            #   了它（2026-10-26 01:04 `assignments=[('master','cpu',0,8), ...]`），而执行期
            #   要拿 `mid8-24.gguf` 去算 `[0,8)` ⇒ 层号越界/权重缺失，正是"部署显示成功、
            #   第一次推理才炸"的最坏形态。
            #
            #   修法：`layer_budget` 可携带 `min_layer`/`max_layer` 声明**可切区间**。
            #   给了边界就必须落在边界内；**没给边界则保持原行为（不检查）** —— 远端
            #   worker 的 budget 来自 hello capabilities、不含这两项，逐字节不变。
            budget_bounds: tuple[int, int] | None = None
            if advertised_budget is not None:
                low = advertised_budget.get("min_layer")
                high = advertised_budget.get("max_layer")
                if low is not None and high is not None:
                    budget_bounds = (int(low), int(high))
            if advertised_budget is None or not advertised_budget.get("local_cut"):
                allowed_ranges = node.get("layer_ranges")
                if allowed_ranges is not None and not any(
                    start == cursor and end == allowed_end
                    for start, allowed_end in allowed_ranges
                ):
                    exclude(node, cursor, end, "node_layer_range_not_advertised")
                    continue
            elif budget_bounds is not None and not (
                budget_bounds[0] <= cursor and end <= budget_bounds[1]
            ):
                exclude(node, cursor, end, "node_layer_budget_out_of_bounds")
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
            # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**`local_cut` 让段类型失效，前提是它真有整模**。
            #
            #   原判据：只要 `local_cut` 为真，`segment_mode` 就取 `None` ⇒ head/middle/tail
            #   约束**全部跳过**。这对"手上有整模、能按分配裁出任意段"的 worker 是对的。
            #   但**只持有裁好的部分工件**的执行器（本机 CPU 段的 `mid8-24.gguf`）不能当
            #   首段/末段 —— 它没有 embedding、也没有 `lm_head`/`final_norm`。若因为
            #   `local_cut` 而跳过段类型检查，求解器可能把它摆到 tail 位置，而那种失败
            #   发生在**执行期**（缺 `lm_head` ⇒ `Stage error`），不是规划期报错。
            #   实测（2026-10-26，`scripts/probe_local_executor_placement.py` 场景 E/F）：
            #   本机 CPU 段正是靠 `local_cut` 才被允许拿非 middle 位置。
            #
            #   判据：`layer_budget` 带了**可切边界**（`min_layer`/`max_layer`）⇒ 说明它
            #   只有部分工件 ⇒ 段类型**仍然生效**；没带边界（整模）⇒ 保持原行为跳过。
            has_partial_artifact = bool(
                advertised_budget is not None
                and advertised_budget.get("min_layer") is not None
                and advertised_budget.get("max_layer") is not None
            )
            segment_mode = (
                matching_artifact.get("segment_mode")
                if matching_artifact is not None
                else node.get("segment_mode")
                if has_partial_artifact
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
                # ★ 2026-10-23（#117）：自裁节点（`local_cut`）没有匹配请求区间的
                #   工件，此前唯一的字节来源是 master 描述符的逐层前缀和；只要区间
                #   跨过零字节层（embedding / norm 等）就被判"报价缺失"并排除 ⇒
                #   求解器退回按**整模**计费（实测 1 层要 4.96GB、三机只凑得出 18 层
                #   < 32），而"去掉段约束就能解"正是这个失真造成的假象。改为用本节点
                #   任意工件的逐层字节求每层平均，再按区间长度估算。
                segment_bytes = _self_cut_segment_bytes(node, cursor, end)
                if segment_bytes is None:
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
            # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**检查结果按执行器身份区分**。
            #   同机 CPU 段与主段共用 `node_id="master"`，此前 key 只用
            #   `(node_id, cursor, end)` ⇒ 两者的检查结果**互相覆盖**，对外只剩后写的
            #   那一条 —— 实测把**主段**的显存额度（4.47 GB）安到了 **CPU 段**的区间上，
            #   于是一条"CPU 段被要求 28 层、只有 3.96 GB"的假记录，掩盖了真实原因。
            executor_id = str(node.get("executor_id") or "")
            checks[(node["node_id"], executor_id, cursor, end)] = {
                "node_id": node["node_id"], "executor_id": executor_id,
                "layer_range": [cursor, end],
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
    # ★ 2026-10-26（LOCAL-EXECUTORS-01）：这里**不**放宽 `candidate_layer_ranges`。
    #
    #   一度想让"带候选 / 去掉候选"两种都解、按目标取优，理由是真机上 `android [[8,20]]`
    #   + `TABLET [[20,32]]` 把 `[8,32)` 占满、本机 CPU 段挤不进去。但那是**方向错了**：
    #   候选区间表达的是"**我手上确实有这个段**"这个事实，它**就该是硬约束**
    #   （`test_candidate_ranges_and_layer_limits_remain_hard_with_local_cut` 这个名字
    #   就是设计意图的直述），放宽它等于允许把一个设备的算力规划到它并没有的工件上。
    #
    #   真正的问题在**写入侧**：`model_supply_capacity` 把**上一轮**的切点当成这一轮的
    #   候选，于是每轮沿用上轮切点、布局被锁死。要修的是"谁写了这个候选、凭什么"。
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
            # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**执行器身份与共享内存域必须随
            #   assignment 一起流到部署侧**。否则下游（`scheduler_pipeline` 的部署构造）
            #   无法把"同一物理节点上的第二个执行器"认出来，本机 CPU 段永远不会被规划
            #   出来 —— 执行通路虽然接好了，却没有任何东西会去用它。
            #   仅**非空**时输出 ⇒ 既有计划的 assignment 形态与逐位内容不变（回归门）。
            **({"executor_id": str(node.get("executor_id") or "")}
               if node.get("executor_id") else {}),
            **({"capacity_domain_id": str(node["capacity_domain_id"])}
               if node.get("capacity_domain_id") else {}),
            "placement_weight": placement_weight(node["score"]),
            # 必须与 `placement_objective` 用**同一个**成本倍数，否则报告与实际选择
            # 脱节（这个字段被下游当"该段占了多大份额"读）。
            "placement_work_fraction": ((prefix[end] - prefix[start]) / work_total)
            * layer_cost_multiplier(node),
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
    # ★ 2026-10-25：**makespan 报告**（**不参与选择**，只让"切点好不好"可测）。
    #   求解目标 `capability_weighted_work_square` 是**均衡近似**，不是端到端步时
    #   （= 最长段）。实测证明两者会背离：`fastest` 档下按**成本**均衡把 rog 推成
    #   新瓶颈 174 ms，而按**字节**均衡的最长段只有 161 ms（端到端反而慢 7%）——
    #   见 docs/本机跨框架层段成本与干扰实测-2026-10-25.md。
    #   所以"选了哪条切点"与"这条切点实际多快"必须能分别看到，否则无法判断是否值得
    #   为 makespan 另开一条求解路径（那需要重新论证最优性，因为现有 suffix optimum
    #   依赖目标**可加**，而 max 不可加）。
    #   数据来自调用方填的 `layer_cost_ms`（本模块是纯求解器、不读环境）。只要有任意
    #   一个**参与**节点带标定就给出报告；未标定段的 `estimated_ms` 留 None 而不猜，
    #   并单列在 `uncalibrated_segments` 里 —— 半真半假的步时估计比没有更危险。
    if any(usable[value[0]].get("layer_cost_ms") for value in solved):
        segments_ms = []
        for value in solved:
            node = usable[value[0]]
            cost = node.get("layer_cost_ms")
            layers = int(value[2]) - int(value[1])
            segments_ms.append({
                "node_id": node["node_id"],
                "layer_range": [int(value[1]), int(value[2])],
                "layers_count": layers,
                "layer_cost_ms": float(cost) if cost else None,
                "estimated_ms": (layers * float(cost)) if cost else None,
            })
        measured = [item for item in segments_ms if item["estimated_ms"] is not None]
        if measured:
            bottleneck = max(measured, key=lambda item: item["estimated_ms"])
            base["step_time_estimate"] = {
                "basis": "layers_count_x_calibrated_layer_ms",
                "estimated_step_ms": bottleneck["estimated_ms"],
                "bottleneck_node_id": bottleneck["node_id"],
                "bottleneck_layer_range": bottleneck["layer_range"],
                "segments": segments_ms,
                "uncalibrated_segments": [item["node_id"] for item in segments_ms
                                          if item["estimated_ms"] is None],
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
