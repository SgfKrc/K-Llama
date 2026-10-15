"""Bounded offline placement planning for a measured PyTorch operator DAG.

The planner is research-only: it imports no torch runtime and never dispatches
inference. It accepts matched, uninstrumented costs, applies the operator
registry's capability/correctness gates, and compares with the existing
continuous-layer relay objective.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from src.relay_cut_objective import SegmentProfile, plan_relay_cut_n_segments
from src.torch_operator_registry import (
    DEFAULT_TORCH_OPERATOR_REGISTRY,
    NumericalEvidence,
    OperatorContext,
    OperatorRegistry,
    SelectionPolicy,
)


INPUT_SCHEMA = "qlh.torch_hetero_plan_input.v1"
REPORT_SCHEMA = "qlh.torch_hetero_plan_report.v1"
OPERATOR_PROFILE_SCHEMA = "qlh.torch_operator_profile.v1"
LAYER_PROFILE_SCHEMA = "qlh.torch_layer_profile.v1"
SCHEDULE_MODEL = "topological_device_queue_and_link_fifo_v1"
MIB = 1024 ** 2
GIB = 1024 ** 3


@dataclass(frozen=True)
class OperatorProfileEvidence:
    """Validated real operator-profile coverage for offline planner inputs.

    The profile script records profiler-attributed operator times. Those values
    are useful for coverage and shape inspection, but are intentionally not
    accepted as planner costs because profiler instrumentation changes timing
    semantics. Independent wall-clock costs remain a separate input contract.
    """

    profile_schema: str
    phase: str
    device_type: str
    model_fingerprint: str
    workload_fingerprint: str
    profile_ref: str
    operator_count: int
    dispatch_call_count: int
    unmatched_dispatch_calls: int
    logical_operator_counts: Mapping[str, int]
    unmapped_profile_operators: tuple[str, ...]
    wall_samples_ms: tuple[float, ...]
    warmup: Mapping[str, Any]
    profiler_costs_usable: bool = False

    def __post_init__(self) -> None:
        if self.profile_schema != OPERATOR_PROFILE_SCHEMA:
            raise ValueError("unsupported operator profile schema")
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("operator profile phase must be prefill or decode")
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("operator profile device must be cpu or cuda")
        if not self.model_fingerprint or not self.workload_fingerprint or not self.profile_ref:
            raise ValueError("operator profile identity and source are required")
        _positive_int(self.operator_count, "operator profile operator_count")
        _positive_int(self.dispatch_call_count, "operator profile dispatch_call_count")
        _positive_int(self.unmatched_dispatch_calls, "operator profile unmatched calls", allow_zero=True)
        if len(self.wall_samples_ms) < 3:
            raise ValueError("operator profile needs at least three wall samples")
        for sample in self.wall_samples_ms:
            _finite(sample, "operator profile wall sample", minimum=0.000001)
        if not isinstance(self.warmup, Mapping):
            raise ValueError("operator profile warmup must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "logical_operator_counts": dict(self.logical_operator_counts),
            "unmapped_profile_operators": list(self.unmapped_profile_operators),
            "wall_samples_ms": list(self.wall_samples_ms),
            "profiler_costs_usable": False,
        }


def _profile_operator_mapping(registry: OperatorRegistry) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for logical in registry.catalog()["logical_operators"]:
        logical_id = logical["operator_id"]
        for observed in logical.get("observed_profile_operators", ()):
            previous = mapping.setdefault(observed, logical_id)
            if previous != logical_id:
                raise ValueError(f"profile operator maps to multiple logical operators: {observed}")
    return mapping


def load_operator_profile_evidence(
    profile: Mapping[str, Any],
    *,
    phase: str,
    device_id: str,
    profile_ref: str,
    model_fingerprint: str,
    workload_fingerprint: str,
    registry: OperatorRegistry = DEFAULT_TORCH_OPERATOR_REGISTRY,
) -> OperatorProfileEvidence:
    """Validate a real ``TORCH-OP-PROFILE-01`` phase without loading torch.

    This is deliberately an evidence adapter, not a timing shortcut. The
    profiler rows establish observed operator coverage and dispatch counts;
    their attributed times remain ineligible for ``MeasuredOperatorCost``.
    """
    profile = _mapping(profile, "operator profile")
    if profile.get("schema_version") != OPERATOR_PROFILE_SCHEMA:
        raise ValueError(f"profile schema_version must be {OPERATOR_PROFILE_SCHEMA}")
    phases = _mapping(profile.get("phases"), "operator profile.phases")
    phase_data = _mapping(phases.get(phase), f"operator profile.phases.{phase}")
    rows = phase_data.get("operators")
    if not isinstance(rows, list) or not rows:
        raise ValueError("operator profile phase must contain non-empty operators")
    runtime = _mapping(profile.get("runtime"), "operator profile.runtime")
    runtime_device = _mapping(runtime.get("device"), "operator profile.runtime.device")
    runtime_type = str(runtime_device.get("type", ""))
    if runtime_type != device_id.split(":", 1)[0]:
        raise ValueError("operator profile device does not match requested device")
    wall = _mapping(phase_data.get("uninstrumented_wall"), "operator profile wall")
    samples = wall.get("samples_ms")
    if not isinstance(samples, list):
        raise ValueError("operator profile wall samples must be a list")
    mapping = _profile_operator_mapping(registry)
    logical_counts: dict[str, int] = {}
    unmapped: set[str] = set()
    dispatch_count = 0
    for row in rows:
        row = _mapping(row, "operator profile row")
        operator = row.get("operator")
        calls = row.get("calls")
        if not isinstance(operator, str) or not operator.startswith("aten::"):
            raise ValueError("operator profile row has invalid operator name")
        _positive_int(calls, "operator profile row calls")
        if row.get("phase") != phase:
            raise ValueError("operator profile row phase mismatch")
        dispatch_count += calls
        logical_id = mapping.get(operator)
        if logical_id is None:
            unmapped.add(operator)
        else:
            logical_counts[logical_id] = logical_counts.get(logical_id, 0) + calls
    declared_dispatch = phase_data.get("dispatch_call_count")
    if declared_dispatch != dispatch_count:
        raise ValueError("operator profile dispatch count does not match rows")
    unmatched = phase_data.get("unmatched_dispatch_calls")
    _positive_int(unmatched, "operator profile unmatched calls", allow_zero=True)
    if unmatched:
        raise ValueError("operator profile contains unmatched dispatch calls")
    return OperatorProfileEvidence(
        profile_schema=profile["schema_version"],
        phase=phase,
        device_type=runtime_type,
        model_fingerprint=model_fingerprint,
        workload_fingerprint=workload_fingerprint,
        profile_ref=profile_ref,
        operator_count=len(rows),
        dispatch_call_count=dispatch_count,
        unmatched_dispatch_calls=unmatched,
        logical_operator_counts=logical_counts,
        unmapped_profile_operators=tuple(sorted(unmapped)),
        wall_samples_ms=tuple(float(value) for value in samples),
        warmup=_mapping(phase_data.get("warmup", {}), "operator profile warmup"),
    )


def _finite(value: float, name: str, *, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class DeviceResource:
    device_id: str
    device_type: str
    available_memory_bytes: int
    capabilities: frozenset[str] = frozenset()
    enabled_feature_gates: frozenset[str] = frozenset()
    memory_safety_margin: float = 1.2
    profile_ref: str = ""

    def __post_init__(self) -> None:
        if not self.device_id.strip() or self.device_type not in {"cpu", "cuda"}:
            raise ValueError("device_id and a cpu/cuda device_type are required")
        _positive_int(self.available_memory_bytes, "available_memory_bytes")
        _finite(self.memory_safety_margin, "memory_safety_margin", minimum=1.0)
        if not self.profile_ref:
            raise ValueError("device profile_ref is required")

    @classmethod
    def from_device_profile(
        cls,
        device_id: str,
        device_type: str,
        profile: Mapping[str, Any],
        *,
        capabilities: frozenset[str] = frozenset(),
        enabled_feature_gates: frozenset[str] = frozenset(),
        memory_safety_margin: float = 1.2,
        profile_ref: str,
    ) -> "DeviceResource":
        """Extract currently available RAM/VRAM from DeviceProfiler.to_dict()."""
        profile = _mapping(profile, "device profile")
        if device_type == "cpu":
            ram = _mapping(profile.get("ram"), "device profile.ram")
            available_gb = ram.get("available_gb")
        elif device_type == "cuda":
            gpus = profile.get("gpus") or []
            selected = profile.get("selected_gpu_index", 0)
            gpu = next((item for item in gpus if item.get("index") == selected), None)
            if gpu is None:
                gpu = profile.get("gpu")
            gpu = _mapping(gpu, "device profile.gpu")
            if gpu.get("cuda_available") is not True:
                raise ValueError("selected GPU profile does not declare CUDA availability")
            available_gb = gpu.get("vram_free_gb")
        else:
            raise ValueError("device_type must be cpu or cuda")
        _finite(available_gb, "available memory in GiB", minimum=0.000001)
        return cls(
            device_id=device_id,
            device_type=device_type,
            available_memory_bytes=int(float(available_gb) * GIB),
            capabilities=capabilities,
            enabled_feature_gates=enabled_feature_gates,
            memory_safety_margin=memory_safety_margin,
            profile_ref=profile_ref,
        )


@dataclass(frozen=True)
class LinkProfile:
    source_device_id: str
    destination_device_id: str
    bandwidth_bytes_per_second: float
    latency_ms: float
    samples: int
    warmup_consistent: bool
    profile_ref: str

    def __post_init__(self) -> None:
        if not self.source_device_id or not self.destination_device_id:
            raise ValueError("link endpoints are required")
        if self.source_device_id == self.destination_device_id:
            raise ValueError("same-device links are implicit")
        _finite(self.bandwidth_bytes_per_second, "link bandwidth", minimum=1.0)
        _finite(self.latency_ms, "link latency")
        _positive_int(self.samples, "link samples")
        if self.samples < 3 or self.warmup_consistent is not True or not self.profile_ref:
            raise ValueError("link profile needs >=3 stable samples and an artifact reference")


@dataclass(frozen=True)
class OperatorNode:
    node_id: str
    operator_id: str
    layer_index: int
    dtype: str
    shape_fingerprint: str
    resident_bytes: int
    workspace_bytes: int

    def __post_init__(self) -> None:
        if not self.node_id or not self.operator_id or not self.dtype or not self.shape_fingerprint:
            raise ValueError("operator node identity, dtype, and shape fingerprint are required")
        _positive_int(self.layer_index, "layer_index", allow_zero=True)
        _positive_int(self.resident_bytes, "resident_bytes", allow_zero=True)
        _positive_int(self.workspace_bytes, "workspace_bytes", allow_zero=True)


@dataclass(frozen=True)
class TensorEdge:
    source_node_id: str
    destination_node_id: str
    tensor_bytes: int

    def __post_init__(self) -> None:
        if not self.source_node_id or not self.destination_node_id:
            raise ValueError("tensor edge endpoints are required")
        if self.source_node_id == self.destination_node_id:
            raise ValueError("self edges are invalid")
        _positive_int(self.tensor_bytes, "tensor_bytes")


@dataclass(frozen=True)
class MeasuredOperatorCost:
    node_id: str
    device_id: str
    operator_id: str
    implementation_id: str
    phase: str
    dtype: str
    shape_fingerprint: str
    model_fingerprint: str
    workload_fingerprint: str
    samples_ms: tuple[float, ...]
    warmup_consistent: bool
    instrumented: bool
    source_ref: str

    def __post_init__(self) -> None:
        for value in (
            self.node_id, self.device_id, self.operator_id, self.implementation_id,
            self.phase, self.dtype, self.shape_fingerprint, self.model_fingerprint,
            self.workload_fingerprint, self.source_ref,
        ):
            if not value:
                raise ValueError("measured operator cost has a missing identity field")
        if len(self.samples_ms) < 3:
            raise ValueError("operator costs require at least three samples")
        for sample in self.samples_ms:
            _finite(sample, "operator timing sample", minimum=0.000001)
        if not isinstance(self.warmup_consistent, bool) or not isinstance(self.instrumented, bool):
            raise ValueError("cost warmup/instrumentation flags must be booleans")

    @property
    def median_ms(self) -> float:
        ordered = sorted(self.samples_ms)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2


def load_layer_profile_parts(
    profile: Mapping[str, Any],
    *,
    phase: str,
    device_id: str,
    profile_ref: str,
) -> tuple[tuple[OperatorNode, ...], tuple[TensorEdge, ...], tuple[MeasuredOperatorCost, ...]]:
    """Convert an isolated layer-profile phase into planner input parts.

    The source report must carry one real node and >=3 independent wall samples
    per layer. This function does not relax planner gates: instrumented or
    warmup-inconsistent rows remain represented and are rejected later by the
    planner rather than silently repaired here.
    """
    profile = _mapping(profile, "layer profile")
    if profile.get("schema_version") != LAYER_PROFILE_SCHEMA:
        raise ValueError(f"profile schema_version must be {LAYER_PROFILE_SCHEMA}")
    if phase not in {"prefill", "decode"}:
        raise ValueError("layer profile phase must be prefill or decode")
    if not device_id or device_id not in {"cpu", "cuda"}:
        raise ValueError("layer profile device_id must be cpu or cuda")
    if not profile_ref:
        raise ValueError("layer profile reference is required")
    model_fingerprint = profile.get("model_fingerprint")
    workload_fingerprint = profile.get("workload_fingerprint")
    if not isinstance(model_fingerprint, str) or not isinstance(workload_fingerprint, str):
        raise ValueError("layer profile fingerprints are required")
    runtime = _mapping(profile.get("runtime"), "layer profile.runtime")
    runtime_device = _mapping(runtime.get("device"), "layer profile.runtime.device")
    if runtime_device.get("type") != device_id:
        raise ValueError("layer profile device mismatch")
    phase_data = _mapping(
        _mapping(profile.get("phase_profiles"), "layer profile.phase_profiles").get(phase),
        f"layer profile.phase_profiles.{phase}",
    )
    raw_nodes = phase_data.get("nodes")
    raw_edges = phase_data.get("edges")
    raw_costs = phase_data.get("costs")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("layer profile phase must contain nodes")
    if not isinstance(raw_edges, list) or not isinstance(raw_costs, list):
        raise ValueError("layer profile phase must contain edges and costs")
    nodes = tuple(OperatorNode(**_mapping(item, "layer profile node")) for item in raw_nodes)
    edges = tuple(TensorEdge(**_mapping(item, "layer profile edge")) for item in raw_edges)
    node_ids = {node.node_id for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("layer profile contains duplicate node ids")
    nodes_by_id = {node.node_id: node for node in nodes}
    costs: list[MeasuredOperatorCost] = []
    cost_node_ids: set[str] = set()
    for item in raw_costs:
        item = dict(_mapping(item, "layer profile cost"))
        cost_node_id = item.get("node_id")
        if cost_node_id in cost_node_ids:
            raise ValueError("layer profile contains duplicate cost node ids")
        cost_node_ids.add(cost_node_id)
        node = nodes_by_id.get(cost_node_id)
        if node is None:
            raise ValueError("layer profile cost references an unknown node")
        for field in ("operator_id", "dtype", "shape_fingerprint"):
            if item.get(field) != getattr(node, field):
                raise ValueError(f"layer profile cost {field} does not match node")
        raw_device = str(item.get("device_id", ""))
        if raw_device.split(":", 1)[0] != device_id:
            raise ValueError("layer profile cost device mismatch")
        if item.get("phase") != phase:
            raise ValueError("layer profile cost phase mismatch")
        if item.get("model_fingerprint") != model_fingerprint:
            raise ValueError("layer profile cost model fingerprint mismatch")
        if item.get("workload_fingerprint") != workload_fingerprint:
            raise ValueError("layer profile cost workload fingerprint mismatch")
        samples = item.get("samples_ms")
        if not isinstance(samples, list):
            raise ValueError("layer profile cost samples must be a list")
        item["device_id"] = device_id
        item["samples_ms"] = tuple(samples)
        item["source_ref"] = f"{profile_ref}#{item.get('source_ref', 'cost')}"
        costs.append(MeasuredOperatorCost(**item))
    if cost_node_ids != node_ids:
        raise ValueError("layer profile nodes and costs do not cover the same layers")
    return nodes, edges, tuple(costs)


@dataclass(frozen=True)
class LayerFitEvidence:
    device_id: str
    model_fingerprint: str
    workload_fingerprint: str
    phase: str
    fixed_ms: float
    ms_per_layer: float
    layer_bytes: tuple[int, ...]
    samples: int
    r2: float
    artifact_ref: str

    def __post_init__(self) -> None:
        for value in (
            self.device_id, self.model_fingerprint, self.workload_fingerprint,
            self.phase, self.artifact_ref,
        ):
            if not value:
                raise ValueError("layer fit evidence identity fields are required")
        _finite(self.fixed_ms, "layer fixed_ms")
        _finite(self.ms_per_layer, "layer ms_per_layer", minimum=0.000001)
        _positive_int(self.samples, "layer-fit samples")
        _finite(self.r2, "layer-fit r2")
        if self.r2 > 1.0:
            raise ValueError("layer-fit r2 must not exceed 1")
        if not self.layer_bytes or any(isinstance(x, bool) or not isinstance(x, int) or x < 1
                                       for x in self.layer_bytes):
            raise ValueError("layer_bytes must contain positive integers")


@dataclass(frozen=True)
class ContinuousLayerBaseline:
    device_order: tuple[str, ...]
    layer_fits: tuple[LayerFitEvidence, ...]
    total_layers: int
    hidden_size: int
    hidden_dtype: str
    prefill_tokens: int
    decode_tokens: int
    non_split_bytes: int = 0
    #: ⚠️ 2026-10-14（22c 缺口②）：这里的默认 1 只适用于**合成输入**的实验场景 ——
    #: 本类是 `torch_hetero_plan` 的合成基线，**没有源 GGUF 路径**，无法自动读
    #: `<arch>.full_attention_interval`。**真实入口**（`scripts/relay_cut_plan.py:116-127`）
    #: 已在缺省时自动从 `--src-gguf` 读（`cut_multiple_from_gguf`，见 `#67`）。
    #: ⇒ 对 hybrid（如 Qwen3.5，`full_attention_interval=4`），**用本类时必须显式传
    #:   `cut_multiple`**，否则切点整体错位、llama.cpp 报 `missing tensor`。
    cut_multiple: int = 1
    minimum_layers_per_segment: int = 1
    minimum_fit_r2: float = 0.9

    def __post_init__(self) -> None:
        if len(self.device_order) < 2 or len(set(self.device_order)) != len(self.device_order):
            raise ValueError("continuous baseline needs an ordered set of >=2 unique devices")
        _positive_int(self.total_layers, "total_layers")
        _positive_int(self.hidden_size, "hidden_size")
        _positive_int(self.prefill_tokens, "prefill_tokens")
        _positive_int(self.decode_tokens, "decode_tokens")
        _positive_int(self.non_split_bytes, "non_split_bytes", allow_zero=True)
        _positive_int(self.cut_multiple, "cut_multiple")
        _positive_int(self.minimum_layers_per_segment, "minimum_layers_per_segment")
        _finite(self.minimum_fit_r2, "minimum_fit_r2", minimum=0.0)
        if self.minimum_fit_r2 > 1.0:
            raise ValueError("minimum_fit_r2 must not exceed 1")
        if not self.hidden_dtype:
            raise ValueError("hidden_dtype is required")


@dataclass(frozen=True)
class HeterogeneousPlanInput:
    model_fingerprint: str
    workload_fingerprint: str
    model_parameter_count: int
    phase: str
    nodes: tuple[OperatorNode, ...]
    edges: tuple[TensorEdge, ...]
    devices: tuple[DeviceResource, ...]
    links: tuple[LinkProfile, ...]
    costs: tuple[MeasuredOperatorCost, ...]
    numerical_evidence: Mapping[str, NumericalEvidence]
    continuous_baseline: ContinuousLayerBaseline
    max_search_states: int = 100_000

    def __post_init__(self) -> None:
        if not self.model_fingerprint or not self.workload_fingerprint:
            raise ValueError("model and workload fingerprints are required")
        _positive_int(self.model_parameter_count, "model_parameter_count")
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        if not self.nodes or not self.devices:
            raise ValueError("at least one graph node and device are required")
        _positive_int(self.max_search_states, "max_search_states")


@dataclass(frozen=True)
class NodePlacement:
    node_id: str
    operator_id: str
    layer_index: int
    device_id: str
    implementation_id: str
    measured_median_ms: float
    cost_source_ref: str
    correctness_evidence_ref: str | None


@dataclass(frozen=True)
class HeterogeneousPlan:
    admitted: bool
    reason: str
    model_fingerprint: str
    workload_fingerprint: str
    phase: str
    placements: tuple[NodePlacement, ...] = ()
    compute_work_ms: float | None = None
    transfer_work_ms: float | None = None
    total_ms: float | None = None
    memory_by_device: Mapping[str, Mapping[str, int]] | None = None
    states_explored: int = 0
    baseline_comparison: Mapping[str, Any] | None = None
    rejected_candidates: tuple[str, ...] = ()
    schedule_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA,
            **asdict(self),
            "placements": [asdict(item) for item in self.placements],
            "rejected_candidates": list(self.rejected_candidates),
        }


@dataclass(frozen=True)
class _Choice:
    device: DeviceResource
    cost: MeasuredOperatorCost


class _SearchLimitExceeded(RuntimeError):
    pass


def _device_memory(
    nodes: Sequence[OperatorNode],
    edges: Sequence[TensorEdge],
    assignments: Sequence[_Choice | None],
    devices: Mapping[str, DeviceResource],
) -> tuple[dict[str, dict[str, int]], bool]:
    resident = {device_id: 0 for device_id in devices}
    workspace = {device_id: 0 for device_id in devices}
    boundary_buffers = {device_id: 0 for device_id in devices}
    node_index = {node.node_id: index for index, node in enumerate(nodes)}
    for node, choice in zip(nodes, assignments):
        if choice is None:
            continue
        device_id = choice.device.device_id
        resident[device_id] += node.resident_bytes
        workspace[device_id] = max(workspace[device_id], node.workspace_bytes)
    for edge in edges:
        source_choice = assignments[node_index[edge.source_node_id]]
        destination_choice = assignments[node_index[edge.destination_node_id]]
        if (
            source_choice is not None and destination_choice is not None
            and source_choice.device.device_id != destination_choice.device.device_id
        ):
            boundary_buffers[source_choice.device.device_id] += edge.tensor_bytes
            boundary_buffers[destination_choice.device.device_id] += edge.tensor_bytes
    result: dict[str, dict[str, int]] = {}
    feasible = True
    for device_id, device in devices.items():
        transient = workspace[device_id] + boundary_buffers[device_id]
        raw_required = resident[device_id] + transient
        required = math.ceil(raw_required * device.memory_safety_margin)
        result[device_id] = {
            "resident_bytes": resident[device_id],
            "workspace_peak_bytes": workspace[device_id],
            "boundary_buffer_bytes_upper_bound": boundary_buffers[device_id],
            "required_with_margin_bytes": required,
            "available_bytes": device.available_memory_bytes,
        }
        if raw_required and required > device.available_memory_bytes:
            feasible = False
    return result, feasible


def _baseline_comparison(
    request: HeterogeneousPlanInput,
    hetero_total_ms: float | None,
) -> dict[str, Any]:
    baseline = request.continuous_baseline
    reject = lambda reason: {"admitted": False, "reason": reason}
    if baseline.device_order != tuple(device.device_id for device in request.devices):
        return reject("baseline_device_order_mismatch")
    if len(baseline.layer_fits) != len(baseline.device_order):
        return reject("baseline_fit_count_mismatch")
    if tuple(fit.device_id for fit in baseline.layer_fits) != baseline.device_order:
        return reject("baseline_fit_order_mismatch")
    if any(
        fit.model_fingerprint != request.model_fingerprint
        or fit.workload_fingerprint != request.workload_fingerprint
        or fit.phase != request.phase
        for fit in baseline.layer_fits
    ):
        return reject("baseline_experiment_identity_mismatch")
    if any(fit.samples < 3 or fit.r2 < baseline.minimum_fit_r2 for fit in baseline.layer_fits):
        return reject("baseline_fit_quality_gate_failed")
    if any(len(fit.layer_bytes) != baseline.total_layers for fit in baseline.layer_fits):
        return reject("baseline_layer_bytes_length_mismatch")
    layer_bytes = baseline.layer_fits[0].layer_bytes
    if any(fit.layer_bytes != layer_bytes for fit in baseline.layer_fits[1:]):
        return reject("baseline_layer_bytes_disagree")
    actual_layer_bytes = [0] * baseline.total_layers
    seen_layers: set[int] = set()
    for node in request.nodes:
        if node.layer_index >= baseline.total_layers:
            return reject("operator_layer_index_out_of_range")
        actual_layer_bytes[node.layer_index] += node.resident_bytes
        seen_layers.add(node.layer_index)
    if seen_layers != set(range(baseline.total_layers)):
        return reject("operator_graph_layer_coverage_incomplete")
    if tuple(actual_layer_bytes) != layer_bytes:
        return reject("operator_and_layer_baseline_memory_mismatch")

    device_by_id = {device.device_id: device for device in request.devices}
    node_by_id = {node.node_id: node for node in request.nodes}
    peak_workspace_bytes = max(node.workspace_bytes for node in request.nodes)
    if any(
        node_by_id[edge.source_node_id].layer_index
        > node_by_id[edge.destination_node_id].layer_index
        for edge in request.edges
    ):
        return reject("baseline_layer_order_mismatch")
    peak_layer_boundary_bytes = 0
    for boundary in range(baseline.total_layers - 1):
        crossing_bytes = sum(
            edge.tensor_bytes
            for edge in request.edges
            if node_by_id[edge.source_node_id].layer_index <= boundary
            < node_by_id[edge.destination_node_id].layer_index
        )
        peak_layer_boundary_bytes = max(peak_layer_boundary_bytes, crossing_bytes)
    safety_margin = max(device.memory_safety_margin for device in request.devices)
    transient_bytes = peak_workspace_bytes + peak_layer_boundary_bytes
    effective_capacity_by_device = {
        device_id: max(
            0,
            math.floor(
                resource.available_memory_bytes - transient_bytes * safety_margin,
            ),
        )
        for device_id, resource in device_by_id.items()
    }
    link_by_pair = {
        (link.source_device_id, link.destination_device_id): link
        for link in request.links
    }
    profiles: list[SegmentProfile] = []
    for index, fit in enumerate(baseline.layer_fits):
        incoming = (
            link_by_pair.get((baseline.device_order[index - 1], fit.device_id))
            if index else None
        )
        outgoing = (
            link_by_pair.get((fit.device_id, baseline.device_order[index + 1]))
            if index + 1 < len(baseline.device_order) else None
        )
        if index and incoming is None:
            return reject("baseline_link_profile_missing")
        resource = device_by_id[fit.device_id]
        bandwidth_mbps = (
            outgoing.bandwidth_bytes_per_second * 8 / 1_000_000 if outgoing else 0.0
        )
        common = {
            "node_id": fit.device_id,
            "engine": "pytorch",
            "capacity_bytes": effective_capacity_by_device[fit.device_id],
            "bandwidth_mbps": bandwidth_mbps,
            "rtt_ms": incoming.latency_ms if incoming else 0.0,
            "source": "same_workload_layer_fit",
        }
        if request.phase == "decode":
            profiles.append(SegmentProfile(
                **common,
                ms_per_layer_decode=fit.ms_per_layer,
                ms_fixed_decode=fit.fixed_ms,
            ))
        else:
            profiles.append(SegmentProfile(
                **common,
                ms_per_layer_prefill=fit.ms_per_layer,
                ms_fixed_prefill=fit.fixed_ms,
            ))

    cut_plan = plan_relay_cut_n_segments(
        total_layers=baseline.total_layers,
        layer_bytes=layer_bytes,
        n_embd=baseline.hidden_size,
        segments=profiles,
        non_split_bytes=baseline.non_split_bytes,
        hidden_dtype=baseline.hidden_dtype,
        cut_multiple=baseline.cut_multiple,
        min_layers_per_segment=baseline.minimum_layers_per_segment,
        safety_margin=max(device.memory_safety_margin for device in request.devices),
        weights={"capacity": 0.0, "latency": 1.0, "risk": 0.0},
        prefill_tokens=baseline.prefill_tokens if request.phase == "prefill" else 0,
        decode_tokens=baseline.decode_tokens if request.phase == "decode" else 0,
    )
    result = cut_plan.to_dict()
    if not cut_plan.admitted:
        return {"admitted": False, "reason": cut_plan.reason, "plan": result}
    phase_key = "prefill_ms" if request.phase == "prefill" else "decode_ms"
    baseline_ms = cut_plan.latency_estimate.get(phase_key)
    if not isinstance(baseline_ms, (int, float)) or baseline_ms <= 0:
        return {"admitted": False, "reason": "baseline_latency_unavailable", "plan": result}
    result.update({
        "admitted": True,
        "reason": "compared_same_model_workload_phase",
        "resource_adjustment": {
            "peak_workspace_bytes": peak_workspace_bytes,
            "peak_layer_boundary_bytes": peak_layer_boundary_bytes,
            "effective_capacity_by_device": effective_capacity_by_device,
            "safety_margin": safety_margin,
        },
        "baseline_ms": baseline_ms,
        "heterogeneous_ms": hetero_total_ms,
        "speedup_vs_contiguous": (
            round(baseline_ms / hetero_total_ms, 6)
            if hetero_total_ms and hetero_total_ms > 0 else None
        ),
        "latency_delta_ms": (
            round(hetero_total_ms - baseline_ms, 6)
            if hetero_total_ms is not None else None
        ),
    })
    return result


def plan_operator_placement(
    request: HeterogeneousPlanInput,
    *,
    registry: OperatorRegistry = DEFAULT_TORCH_OPERATOR_REGISTRY,
    policy: SelectionPolicy = SelectionPolicy(),
) -> HeterogeneousPlan:
    """Find the minimum-cost exact DAG placement under correctness and memory gates."""
    fail = lambda reason, rejected=(): HeterogeneousPlan(
        admitted=False,
        reason=reason,
        model_fingerprint=request.model_fingerprint,
        workload_fingerprint=request.workload_fingerprint,
        phase=request.phase,
        rejected_candidates=tuple(rejected),
    )
    node_index = {node.node_id: index for index, node in enumerate(request.nodes)}
    if len(node_index) != len(request.nodes):
        return fail("duplicate_operator_node_id")
    if len({device.device_id for device in request.devices}) != len(request.devices):
        return fail("duplicate_device_id")
    device_by_id = {device.device_id: device for device in request.devices}
    if any(device_id not in device_by_id for device_id in request.continuous_baseline.device_order):
        return fail("baseline_references_unknown_device")
    if any(
        cost.model_fingerprint != request.model_fingerprint
        or cost.workload_fingerprint != request.workload_fingerprint
        or cost.phase != request.phase
        for cost in request.costs
    ):
        return fail("mixed_operator_cost_experiment_identity")

    seen_edges: set[tuple[str, str]] = set()
    for edge in request.edges:
        pair = (edge.source_node_id, edge.destination_node_id)
        if pair in seen_edges:
            return fail("duplicate_tensor_edge")
        seen_edges.add(pair)
        if edge.source_node_id not in node_index or edge.destination_node_id not in node_index:
            return fail("tensor_edge_references_unknown_node")
        if node_index[edge.source_node_id] >= node_index[edge.destination_node_id]:
            return fail("operator_nodes_not_in_topological_order")
    link_by_pair = {
        (link.source_device_id, link.destination_device_id): link
        for link in request.links
    }
    if len(link_by_pair) != len(request.links):
        return fail("duplicate_link_profile")
    if any(
        link.source_device_id not in device_by_id
        or link.destination_device_id not in device_by_id
        for link in request.links
    ):
        return fail("link_references_unknown_device")

    cost_by_key: dict[tuple[str, str, str], MeasuredOperatorCost] = {}
    for cost in request.costs:
        key = (cost.node_id, cost.device_id, cost.implementation_id)
        if key in cost_by_key:
            return fail("duplicate_operator_cost")
        node = request.nodes[node_index[cost.node_id]] if cost.node_id in node_index else None
        if node is None or cost.device_id not in device_by_id:
            return fail("operator_cost_references_unknown_node_or_device")
        if (
            cost.operator_id != node.operator_id
            or cost.dtype != node.dtype
            or cost.shape_fingerprint != node.shape_fingerprint
        ):
            return fail("operator_cost_signature_mismatch")
        cost_by_key[key] = cost

    choices_by_node: list[list[_Choice]] = []
    rejected: list[str] = []
    for node in request.nodes:
        node_choices: list[_Choice] = []
        try:
            for device in request.devices:
                context = OperatorContext(
                    device=device.device_type,
                    dtype=node.dtype,
                    phase=request.phase,
                    capabilities=device.capabilities,
                    enabled_feature_gates=device.enabled_feature_gates,
                    model_fingerprint=request.model_fingerprint,
                    workload_fingerprint=request.workload_fingerprint,
                    model_parameter_count=request.model_parameter_count,
                )
                compatible = registry.compatible_implementations(
                    node.operator_id, context, policy=policy,
                    evidence=request.numerical_evidence,
                )
                for implementation in compatible:
                    cost = cost_by_key.get((node.node_id, device.device_id,
                                            implementation.implementation_id))
                    if cost is None:
                        continue
                    if not cost.warmup_consistent:
                        rejected.append(f"{node.node_id}/{device.device_id}: warmup inconsistent")
                        continue
                    if cost.instrumented:
                        rejected.append(f"{node.node_id}/{device.device_id}: instrumented cost")
                        continue
                    node_choices.append(_Choice(device=device, cost=cost))
        except KeyError:
            return fail(f"unregistered_logical_operator:{node.operator_id}")
        node_choices.sort(key=lambda choice: (
            choice.cost.median_ms, choice.device.device_id, choice.cost.implementation_id,
        ))
        if not node_choices:
            return fail(f"no_matched_eligible_cost:{node.node_id}", rejected)
        choices_by_node.append(node_choices)

    incoming: list[list[tuple[int, TensorEdge]]] = [[] for _ in request.nodes]
    for edge in request.edges:
        incoming[node_index[edge.destination_node_id]].append((node_index[edge.source_node_id], edge))

    assignments: list[_Choice | None] = [None] * len(request.nodes)
    best: tuple[
        float, float, float, tuple[_Choice, ...], dict[str, dict[str, int]],
    ] | None = None
    explored = 0
    search_limit_hit = False

    def search(
        index: int,
        device_ready_ms: Mapping[str, float],
        node_finish_ms: list[float],
        link_ready_ms: Mapping[tuple[str, str], float],
        compute_work_ms: float,
        transfer_work_ms: float,
    ) -> None:
        nonlocal best, explored, search_limit_hit
        if search_limit_hit:
            return
        explored += 1
        if explored > request.max_search_states:
            search_limit_hit = True
            return
        partial_makespan = max(device_ready_ms.values(), default=0.0)
        if best is not None and partial_makespan >= best[0]:
            return
        if index == len(request.nodes):
            memory, feasible = _device_memory(
                request.nodes, request.edges, assignments, device_by_id,
            )
            if feasible:
                best = (
                    partial_makespan,
                    compute_work_ms,
                    transfer_work_ms,
                    tuple(choice for choice in assignments if choice is not None),
                    memory,
                )
            return
        node = request.nodes[index]
        for choice in choices_by_node[index]:
            dependency_ready_ms = 0.0
            edge_transfer_work_ms = 0.0
            missing_link = False
            next_link_ready_ms = dict(link_ready_ms)
            for source_index, edge in incoming[index]:
                source_choice = assignments[source_index]
                if source_choice is None:
                    missing_link = True
                    break
                if source_choice.device.device_id == choice.device.device_id:
                    dependency_ready_ms = max(
                        dependency_ready_ms, node_finish_ms[source_index],
                    )
                    continue
                pair = (source_choice.device.device_id, choice.device.device_id)
                link = link_by_pair.get(pair)
                if link is None:
                    missing_link = True
                    break
                edge_cost_ms = (
                    link.latency_ms
                    + edge.tensor_bytes * 1000 / link.bandwidth_bytes_per_second
                )
                transfer_start_ms = max(
                    node_finish_ms[source_index], next_link_ready_ms.get(pair, 0.0),
                )
                transfer_finish_ms = transfer_start_ms + edge_cost_ms
                next_link_ready_ms[pair] = transfer_finish_ms
                dependency_ready_ms = max(dependency_ready_ms, transfer_finish_ms)
                edge_transfer_work_ms += edge_cost_ms
            if missing_link:
                continue
            start_ms = max(
                device_ready_ms[choice.device.device_id], dependency_ready_ms,
            )
            finish_ms = start_ms + choice.cost.median_ms
            assignments[index] = choice
            _, memory_feasible = _device_memory(
                request.nodes, request.edges, assignments, device_by_id,
            )
            if memory_feasible:
                next_device_ready_ms = dict(device_ready_ms)
                next_device_ready_ms[choice.device.device_id] = finish_ms
                node_finish_ms[index] = finish_ms
                search(
                    index + 1,
                    next_device_ready_ms,
                    node_finish_ms,
                    next_link_ready_ms,
                    compute_work_ms + choice.cost.median_ms,
                    transfer_work_ms + edge_transfer_work_ms,
                )
                node_finish_ms[index] = 0.0
            assignments[index] = None

    search(
        0,
        {device_id: 0.0 for device_id in device_by_id},
        [0.0] * len(request.nodes),
        {},
        0.0,
        0.0,
    )
    if search_limit_hit:
        return HeterogeneousPlan(
            admitted=False,
            reason="exact_search_state_limit_exceeded",
            model_fingerprint=request.model_fingerprint,
            workload_fingerprint=request.workload_fingerprint,
            phase=request.phase,
            states_explored=explored,
            rejected_candidates=tuple(rejected),
        )
    if best is None:
        return HeterogeneousPlan(
            admitted=False,
            reason="no_feasible_assignment",
            model_fingerprint=request.model_fingerprint,
            workload_fingerprint=request.workload_fingerprint,
            phase=request.phase,
            states_explored=explored,
            rejected_candidates=tuple(rejected),
        )

    total_ms, compute_work_ms, transfer_work_ms, chosen, memory = best
    placements = tuple(
        NodePlacement(
            node_id=node.node_id,
            operator_id=node.operator_id,
            layer_index=node.layer_index,
            device_id=choice.device.device_id,
            implementation_id=choice.cost.implementation_id,
            measured_median_ms=round(choice.cost.median_ms, 6),
            cost_source_ref=choice.cost.source_ref,
            correctness_evidence_ref=(
                request.numerical_evidence[choice.cost.implementation_id].artifact_ref
                if choice.cost.implementation_id in request.numerical_evidence else None
            ),
        )
        for node, choice in zip(request.nodes, chosen)
    )
    comparison = _baseline_comparison(request, total_ms)
    return HeterogeneousPlan(
        admitted=True,
        reason="best_exact_feasible_assignment",
        model_fingerprint=request.model_fingerprint,
        workload_fingerprint=request.workload_fingerprint,
        phase=request.phase,
        placements=placements,
        compute_work_ms=round(compute_work_ms, 6),
        transfer_work_ms=round(transfer_work_ms, 6),
        total_ms=round(total_ms, 6),
        memory_by_device=memory,
        states_explored=explored,
        baseline_comparison=comparison,
        rejected_candidates=tuple(rejected),
        schedule_model=SCHEDULE_MODEL,
    )


def _scenario_from_dict(payload: Mapping[str, Any]) -> HeterogeneousPlanInput:
    if payload.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"scenario schema_version must be {INPUT_SCHEMA}")
    request = _mapping(payload.get("request"), "request")
    nodes = tuple(OperatorNode(**item) for item in payload.get("nodes", ()))
    edges = tuple(TensorEdge(**item) for item in payload.get("edges", ()))
    devices = tuple(DeviceResource(
        **{**item,
           "capabilities": frozenset(item.get("capabilities", ())),
           "enabled_feature_gates": frozenset(item.get("enabled_feature_gates", ()))})
        for item in payload.get("devices", ())
    )
    links = tuple(LinkProfile(**item) for item in payload.get("links", ()))
    costs = tuple(MeasuredOperatorCost(
        **{**item, "samples_ms": tuple(item.get("samples_ms", ()))})
        for item in payload.get("costs", ())
    )
    evidence: dict[str, NumericalEvidence] = {}
    for item in payload.get("numerical_evidence", ()):
        implementation_id = item["implementation_id"]
        if implementation_id in evidence:
            raise ValueError(f"duplicate numerical evidence: {implementation_id}")
        evidence[implementation_id] = NumericalEvidence(**item["evidence"])
    baseline_data = _mapping(payload.get("continuous_baseline"), "continuous_baseline")
    baseline = ContinuousLayerBaseline(
        **{
            **baseline_data,
            "device_order": tuple(baseline_data.get("device_order", ())),
            "layer_fits": tuple(
                LayerFitEvidence(**{**fit, "layer_bytes": tuple(fit["layer_bytes"])})
                for fit in baseline_data.get("layer_fits", ())
            ),
        }
    )
    return HeterogeneousPlanInput(
        **request,
        nodes=nodes,
        edges=edges,
        devices=devices,
        links=links,
        costs=costs,
        numerical_evidence=evidence,
        continuous_baseline=baseline,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="matched scenario JSON")
    parser.add_argument("--json-out", required=True, type=Path, help="planner report JSON")
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.json_out.resolve():
            raise ValueError("input and output paths must differ")
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        scenario = _scenario_from_dict(_mapping(payload, "scenario"))
        report = plan_operator_placement(scenario).to_dict()
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: invalid hetero planning input: {exc}", file=sys.stderr)
        return 2
    print(f"[{report['reason']}] {args.json_out}")
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ContinuousLayerBaseline",
    "DeviceResource",
    "HeterogeneousPlan",
    "HeterogeneousPlanInput",
    "INPUT_SCHEMA",
    "LayerFitEvidence",
    "LinkProfile",
    "LAYER_PROFILE_SCHEMA",
    "MeasuredOperatorCost",
    "OPERATOR_PROFILE_SCHEMA",
    "OperatorNode",
    "OperatorProfileEvidence",
    "REPORT_SCHEMA",
    "SCHEDULE_MODEL",
    "TensorEdge",
    "load_operator_profile_evidence",
    "load_layer_profile_parts",
    "main",
    "plan_operator_placement",
]
