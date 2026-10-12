"""Bounded metadata-only preferences for existing, verified segment assets.

Admission remains owned by the injected supply planner, including fresh runtime
budgets, alignment, proper segments and distributed/all-worker policy. Advertised
ranges describe registered assets; they never establish a loaded-ready ACK.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_FORMATS = {"llama_cpp": "gguf", "pytorch": "safetensors"}


@dataclass(frozen=True)
class _Candidate:
    node_id: str
    start: int
    end: int
    weight_bytes: int
    model_id: str
    artifact_sha256: str


def _sequence(value: Any) -> list | tuple:
    return value if isinstance(value, (list, tuple)) else ()


def _span(value: Any, total: int, alignment: int) -> tuple[int, int] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(bound) is not int for bound in value)):
        return None
    start, end = value
    if not (0 <= start < end <= total and end - start < total
            and start % alignment == end % alignment == 0):
        return None
    return start, end


def _model_matches(models: Any, artifact: Mapping, engine: str) -> bool:
    matches = [model for model in _sequence(models)
               if isinstance(model, Mapping)
               and model.get("model_id") == artifact.get("model_id")]
    return (len(matches) == 1 and matches[0].get("sha256") == artifact.get("artifact_sha256")
            and matches[0].get("engine") == engine
            and matches[0].get("format") == _FORMATS.get(engine))


def _candidates(descriptors, capacity_records, nodes, *, source_model_id,
                source_model_sha256, tokenizer_sha256, engine_by_node,
                capabilities_by_node, alignment):
    reference = descriptors.get("llama_cpp") or descriptors.get(engine_by_node.get("master")) or {}
    total = reference.get("total_layers")
    if (type(total) is not int or total < 2 or type(alignment) is not int
            or alignment < 1 or total % alignment
            or not isinstance(source_model_id, str) or not _ID.fullmatch(source_model_id)
            or any(not isinstance(value, str) or not _SHA.fullmatch(value)
                   for value in (source_model_sha256, tokenizer_sha256))):
        return [], total
    active = {node.get("node_id"): node for node in nodes if isinstance(node, Mapping)}
    candidates = []
    seen = set()
    for record in capacity_records:
        if not isinstance(record, Mapping):
            continue
        node_id = record.get("node_id")
        if not isinstance(node_id, str) or node_id == "master" or node_id not in active:
            continue
        engine = engine_by_node.get(node_id)
        source = descriptors.get(engine) or {}
        if (engine not in _FORMATS or active[node_id].get("engine") != engine
                or source.get("total_layers") != total
                or source.get("hidden_size") != reference.get("hidden_size")):
            continue
        caps = record if capabilities_by_node is None else capabilities_by_node.get(node_id, {})
        if not isinstance(caps, Mapping):
            continue
        artifacts = _sequence(record.get("layer_artifacts"))
        if len(artifacts) > 64:
            continue
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            if any(artifact.get(key) != expected for key, expected in (
                    ("source_model_id", source_model_id),
                    ("source_model_sha256", source_model_sha256),
                    ("tokenizer_sha256", tokenizer_sha256))):
                continue
            interval = _span(artifact.get("layer_range"), total, alignment)
            if interval is None:
                continue
            start, end = interval
            mode = "head" if start == 0 else "tail" if end == total else "middle"
            if (artifact.get("segment_mode") != mode
                    or type(artifact.get("hidden_size")) is not int
                    or artifact["hidden_size"] <= 0
                    or artifact["hidden_size"] != source.get("hidden_size")):
                continue
            if not any(_span(value, total, alignment) == interval
                       for value in _sequence(record.get("layer_ranges"))):
                continue
            artifact_id, digest = artifact.get("model_id"), artifact.get("artifact_sha256")
            if (not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id)
                    or not isinstance(digest, str) or not _SHA.fullmatch(digest)
                    or not _model_matches(caps.get("models"), artifact, engine)):
                continue
            if capabilities_by_node is not None:
                if (engine not in _sequence(caps.get("engines"))
                        or not any(item == artifact for item in _sequence(caps.get("layer_artifacts")))
                        or not any(_span(value, total, alignment) == interval
                                   for value in _sequence(caps.get("layer_ranges")))
                        or ("models" in record and not _model_matches(record["models"], artifact, engine))):
                    continue
            sizes = artifact.get("layer_weight_bytes")
            if (not isinstance(sizes, (list, tuple)) or len(sizes) != end - start
                    or any(type(size) is not int or size <= 0 for size in sizes)):
                continue
            identity = (node_id, start, end, artifact_id, digest)
            if identity not in seen:
                seen.add(identity)
                candidates.append(_Candidate(node_id, start, end, sum(sizes), artifact_id, digest))
    candidates.sort(key=lambda item: (-item.weight_bytes, item.node_id, item.start,
                                      item.end, item.model_id, item.artifact_sha256))
    return candidates, total


def _preserves(plan, preferred, total, alignment, engine_by_node):
    if not plan.get("admitted"):
        return False
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or len(assignments) < 2:
        return False
    cursor, assigned = 0, {}
    for item in assignments:
        if not isinstance(item, Mapping):
            return False
        node_id = item.get("node_id")
        interval = _span([item.get("start_layer"), item.get("end_layer")], total, alignment)
        if (not isinstance(node_id, str) or node_id in assigned or interval is None
                or interval[0] != cursor or item.get("engine") != engine_by_node.get(node_id)):
            return False
        assigned[node_id] = list(interval)
        cursor = interval[1]
    return (assignments[0].get("node_id") == "master" and cursor == total
            and all(assigned.get(node_id) in ranges for node_id, ranges in preferred.items()))


def _annotate(plan, preferred, strategy):
    result = copy.deepcopy(plan)
    result.update(reuse_preferred_node_ids=sorted(preferred), target_preference=strategy,
                  loaded_ready=False)
    return result


def plan_segment_supply_reuse(
    descriptors: dict[str, dict], capacity_records: list[dict], nodes: list[dict], *,
    planner: Callable[..., dict], source_model_id: str, source_model_sha256: str,
    tokenizer_sha256: str, engine_by_node: Mapping[str, str],
    capabilities_by_node: Mapping[str, dict] | None = None, alignment: int = 1,
) -> dict:
    """Select preferences without solving capacity or modifying advertisements.

    ``nodes`` are the new target engines/runtime budgets. Bind them, descriptors,
    runtime projections and safety margin into ``planner``; it must call existing
    ``plan_segment_supply`` with distributed/all-worker admission unchanged.
    The only callback argument is ``preferred_ranges`` in absolute source layer
    coordinates; ``{}`` requests unrestricted planning. Exceptions propagate.

    ``capabilities_by_node`` contains worker_snapshot(node_id)['capabilities'].
    Pass an empty body after a failed snapshot: no engine proof means no reuse.
    If the map is omitted, models inherited into capacity records provide proof.
    A model's physical id/digest and supported engine/format must match its
    artifact. Installed engines, node.engine and guessed formats are insufficient.

    First try all verified worker ranges jointly. On rejection, add individual
    assets by descending advertised decoder weight bytes (never runtime-scaled
    estimates); keep an addition only if admission retains every fixed worker.
    One worker can execute one segment. Calls are bounded by candidate_count + 2.
    Metadata distinguishes joint_fixed, greedy_fixed and unrestricted selection;
    it describes target preference, never execution or loaded readiness.
    """
    candidates, total = _candidates(
        descriptors, capacity_records, nodes, source_model_id=source_model_id,
        source_model_sha256=source_model_sha256, tokenizer_sha256=tokenizer_sha256,
        engine_by_node=engine_by_node, capabilities_by_node=capabilities_by_node,
        alignment=alignment,
    )
    if candidates:
        joint = {}
        for item in candidates:
            joint.setdefault(item.node_id, []).append([item.start, item.end])
        joint = {node_id: sorted({tuple(span) for span in spans})
                 for node_id, spans in sorted(joint.items())}
        joint = {node_id: [list(span) for span in spans] for node_id, spans in joint.items()}
        plan = planner(preferred_ranges=copy.deepcopy(joint))
        if _preserves(plan, joint, total, alignment, engine_by_node):
            return _annotate(plan, joint, "joint_fixed")
        fixed, selected = {}, None
        for item in candidates:
            if item.node_id in fixed:
                continue
            trial = {**fixed, item.node_id: [[item.start, item.end]]}
            plan = planner(preferred_ranges=copy.deepcopy(trial))
            if _preserves(plan, trial, total, alignment, engine_by_node):
                fixed, selected = trial, plan
        if selected is not None:
            return _annotate(selected, fixed, "greedy_fixed")
    return _annotate(planner(preferred_ranges={}), {}, "unrestricted")
