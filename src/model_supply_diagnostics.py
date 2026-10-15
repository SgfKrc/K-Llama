"""Read-only explanations of rejected model-supply capacity checks.

Candidate budgets are examples, never admission decisions or executable plans.
All runtime placement decisions remain with ``solve_pipeline_capacity``. Inputs
are already projected into aligned groups, with boundary charges in reserve and
conservative engine ratios in runtime_multiplier; do not charge either twice.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from model_supply_host_capacity import (
    HostCapacityProjectionError,
    allowed_host_layer_ranges,
)
from pipeline_capacity import (
    PipelineCapacityError,
    _descriptor_costs,
    _normalize_nodes,
    _required_bytes,
    solve_pipeline_capacity,
)


_REASONS = {
    "pipeline_distributed_workers_unavailable": "当前有可用容量的节点不足以形成分布式层段接力",
    "pipeline_capacity_nodes_unavailable": "当前没有可用于模型层段的容量节点",
    "pipeline_distributed_capacity_insufficient": "当前各节点容量无法满足分布式层段预算",
    "pipeline_cluster_capacity_insufficient": "当前各节点容量无法满足层段预算",
    "pipeline_capacity_single_node_insufficient": "当前层段约束下，存在节点无法容纳自己的层段",
    "pipeline_layer_range_coverage_insufficient": "当前节点的可用层段范围无法组成连续接力",
    "pipeline_segment_contract_unsatisfied": "当前工件的层段类型或来源约束无法组成接力",
}
_RESOURCE_NAMES = {"ram": "内存", "vram": "显存", "host_ram": "源张量暂存内存"}
_CANDIDATE_ACTION = (
    "按候选缺口检查对应节点的内存或显存占用，或加入资源充足的节点、改选较小模型后重新规划。"
    "这是候选切分的预算示例，不代表最小缺口；释放所列字节不保证整体准入。"
)


def _node_summary(node: Mapping[str, Any]) -> dict[str, Any]:
    device = str(node.get("execution_device") or "unknown")
    return {
        "node_id": str(node.get("node_id") or ""),
        "engine": str(node.get("engine") or "unknown"),
        "execution_device": device,
        "resource": "vram" if device.lower().partition(":")[0] == "cuda" else "ram",
        "available_bytes": int(node.get("capacity_bytes", 0)),
        "reserve_bytes": int(node.get("reserve_bytes", 0)),
    }


def _details(nodes: list[dict[str, Any]]) -> str:
    details = []
    for node in nodes:
        detail = (
            f"{node['node_id']}（{node['engine']} / {node['execution_device']}，"
            f"{_RESOURCE_NAMES[node['resource']]}）：可用 {node['available_bytes']} 字节"
        )
        if "required_bytes" in node:
            start, end = node["layer_range"]
            detail += (
                f"，层段 [{start},{end}) 预算 {node['required_bytes']} 字节"
                f"，缺口 {node['deficit_bytes']} 字节"
            )
        detail += f"（含预留 {node['reserve_bytes']} 字节）"
        details.append(detail)
    return "；".join(details)


def _capacity_candidate(
    descriptor: dict[str, Any], nodes: list[dict[str, Any]], safety_margin: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Solve one generously budgeted copy without changing relative node order."""
    layers, embedding, norm, output = _descriptor_costs(descriptor)
    # Include exact artifact quotes if present. Normal supply projections remove
    # them, but never relax a caller's additional range/artifact contracts here.
    raw_upper = sum(layers)
    for node in nodes:
        for artifact in node.get("layer_artifacts", ()):
            sizes = artifact.get("layer_weight_bytes")
            if sizes is not None:
                raw_upper = max(raw_upper, sum(sizes))
    raw_upper += embedding + norm + output
    increment = 1
    for node in nodes:
        capacity = int(node.get("capacity_bytes", 0))
        budget = {
            "reserve_bytes": int(node.get("reserve_bytes", 0)),
            "runtime_multiplier": float(node.get("runtime_multiplier", 1.0)),
        }
        if (capacity < 0 or budget["reserve_bytes"] < 0
                or not math.isfinite(budget["runtime_multiplier"])
                or budget["runtime_multiplier"] <= 0):
            raise PipelineCapacityError("invalid diagnostic node budget")
        increment = max(increment, _required_bytes(raw_upper, budget, safety_margin))
    boosted = deepcopy(nodes)
    for node in boosted:
        # A common additive increment preserves capacity - reserve differences.
        # Setting every capacity to one large constant would reorder equal-score
        # nodes by reserve/id and could invent an unavailable execution topology.
        node["capacity_bytes"] = int(node.get("capacity_bytes", 0)) + increment
    candidate = solve_pipeline_capacity(
        deepcopy(descriptor), boosted, safety_margin=safety_margin,
        require_distributed=True, prefer_all_workers=True, required_node_id="master",
    )
    if not candidate.get("admitted"):
        return None
    assignments = candidate["assignments"]
    total = len(layers)
    # Supply requires an executing master and at least two proper segments.
    # Relay-only participation must never turn an entire model into an example.
    if (len(assignments) < 2 or assignments[0]["node_id"] != "master"
            or assignments[0]["start_layer"] != 0
            or assignments[-1]["end_layer"] != total
            or any(not 0 < item["layers_count"] < total for item in assignments)
            or any(left["end_layer"] != right["start_layer"]
                   for left, right in zip(assignments, assignments[1:]))):
        return None
    ordered, _ = _normalize_nodes(boosted, total_layers=total)
    return assignments, ordered


def explain_supply_capacity(
    plan: dict, *, grouped_descriptor: dict, projected_nodes: list[dict],
    alignment: int, total_layers: int, safety_margin: float,
) -> dict:
    """Explain a rejected supply plan using one capacity-only diagnostic solve.

    ``projected_nodes`` and ``grouped_descriptor`` must be the exact inputs to
    the rejected solve, before converting group indices back to decoder layers.
    Unknown requirements are omitted if no proper candidate can be formed.
    The result deliberately exposes neither assignments nor loaded-ready state.
    """
    if (type(alignment) is not int or alignment <= 0
            or type(total_layers) is not int or total_layers <= 0
            or total_layers % alignment
            or grouped_descriptor.get("total_layers") != total_layers // alignment):
        raise PipelineCapacityError("diagnostic alignment does not match grouped descriptor")
    if (type(safety_margin) not in (int, float)
            or not math.isfinite(safety_margin) or safety_margin < 1):
        raise PipelineCapacityError("safety_margin must be finite and at least 1")
    reason_code = plan.get("reason_code", "SUPPLY_CAPACITY_INSUFFICIENT")
    original_reason = str(plan.get("reason") or "")
    reason = original_reason or _REASONS.get(reason_code, "当前容量与层段约束无法形成分布式接力")
    result = {
        "reason_code": reason_code,
        "reason": reason,
        "nodes": [_node_summary(node) for node in projected_nodes],
        "suggested_action": (
            "检查节点的当前可用容量、连接及允许层段范围，补齐相应资源或调整模型后重新规划。"
            "当前约束下未得到合法候选，无法给出可靠的逐节点需求或缺口。"
        ),
        "scope": "candidate_plan",
    }
    try:
        candidate = _capacity_candidate(grouped_descriptor, projected_nodes, safety_margin)
    except (PipelineCapacityError, TypeError, ValueError, OverflowError, KeyError):
        # Diagnostics must not replace the authoritative rejection with a
        # second error, guessed capacity number, or unconstrained topology.
        candidate = None
    if candidate is None:
        return result
    assignments, ordered = candidate
    originals = {node["node_id"]: node for node in projected_nodes}
    by_id = {item["node_id"]: item for item in assignments}
    summaries = []
    for node in ordered:
        row = _node_summary(originals[node["node_id"]])
        item = by_id.get(node["node_id"])
        if item is not None:
            required = item["required_bytes"]
            row.update(
                required_bytes=required,
                deficit_bytes=max(0, required - row["available_bytes"]),
                layer_range=[item["start_layer"] * alignment, item["end_layer"] * alignment],
            )
        summaries.append(row)
    prefix = _REASONS.get(reason_code, "当前容量与层段约束未通过准入")
    if original_reason:
        prefix += f"（原拒绝原因：{original_reason}）"
    result.update(
        nodes=summaries,
        reason=f"{prefix}。保持当前节点顺序与层段约束的候选预算：{_details(summaries)}。",
        suggested_action=_CANDIDATE_ACTION,
    )
    return result


def explain_host_capacity(
    projection: Mapping[str, Any], node_id: str, available_bytes: int,
    reserve_bytes: int, safety_margin: float, alignment: int,
    layer_range: list[int] | tuple[int, int] | None = None,
) -> dict:
    """Explain minimum proper source staging or an explicitly selected segment.

    Host budget validation is shared with ``allowed_host_layer_ranges``. Positive
    layer costs and a constant boundary charge make a one-group span cheapest;
    the master is constrained to [0, alignment), remote ties use earliest start.
    Source bytes never fall back to target/runtime estimates or full-model cost.
    """
    allowed_host_layer_ranges(
        projection, node_id, available_bytes, reserve_bytes, safety_margin, alignment,
    )
    total = projection["total_layers"]
    scope = "minimum_segment" if layer_range is None else "candidate_plan"
    row = {
        "node_id": node_id,
        "engine": str(projection.get("engine") or "pytorch"),
        "execution_device": str(projection.get("execution_device") or "cuda"),
        "resource": "host_ram",
        "available_bytes": available_bytes,
        "reserve_bytes": reserve_bytes,
    }
    result = {
        "reason_code": "SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT",
        "reason": "当前对齐粒度下没有合法的非整模源层段，无法计算源张量暂存需求",
        "nodes": [row],
        "suggested_action": "检查模型层数与切分对齐约束，选择支持非整模层段接力的模型后重新规划。",
        "scope": scope,
    }
    layers = projection["source_layer_weight_bytes"]
    if layer_range is None:
        if alignment == total:
            return result
        start = 0 if node_id == "master" else min(
            range(0, total, alignment),
            key=lambda index: (sum(layers[index:index + alignment]), index),
        )
        end = start + alignment
    else:
        if (not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2
                or any(type(value) is not int for value in layer_range)):
            raise HostCapacityProjectionError("layer_range must be an integer [start, end) pair")
        start, end = layer_range
        if (not 0 <= start < end <= total or end - start == total
                or start % alignment or end % alignment
                or (node_id == "master" and start != 0)):
            raise HostCapacityProjectionError("layer_range must be an aligned proper source segment")
    components = projection["source_component_weight_bytes"]
    keys = (("embedding", "final_norm", "other") if node_id == "master"
            else ("embedding", "final_norm", "lm_head", "other"))
    source_bytes = sum(layers[start:end]) + sum(components[key] for key in keys)
    required = math.ceil(source_bytes * safety_margin) + reserve_bytes
    row.update(
        required_bytes=required, deficit_bytes=max(0, required - available_bytes),
        layer_range=[start, end],
    )
    kind = "最小合法源层段" if scope == "minimum_segment" else "指定源层段"
    state = "容量不足" if row["deficit_bytes"] else "预算可满足，该项不能替代完整准入"
    result.update(
        reason=f"{kind}暂存{state}：{_details([row])}。",
        suggested_action=(
            "检查该节点的主机内存占用；显存充足不能替代源张量暂存内存。"
            "可释放主机内存、使用资源充足的节点或改选较小模型，再重新进行完整规划；"
            "所列缺口不保证其他层段与资源约束也能通过。"
        ),
    )
    return result
