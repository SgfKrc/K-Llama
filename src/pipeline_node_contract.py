"""Engine-neutral contracts for contiguous layer-pipeline nodes.

The module is intentionally independent from the scheduler and engine
implementations.  It validates topology and artifact identity before a plan
reaches an execution path, and provides adapters for existing capacity, Relay,
and llama RPC contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

try:
    from .task_provider import canonical_model_engine
except ImportError:  # pragma: no cover - top-level import in runtime bundles
    from task_provider import canonical_model_engine


PIPELINE_NODE_SCHEMA_VERSION = 1
RESOURCE_VIEW_SCHEMA_VERSION = 1

#: 已知的层流水线节点类型。
#:
#: * ``local``            —— 本机 node
#: * ``remote_rpc``       —— llama.cpp RPC 远端设备
#: * ``remote_pipeline``  —— 远端层流水线节点（含 Android worker）
#: * ``cross_framework``  —— 跨框架接力（如 PyTorch 上游 + llama.cpp 下游）
#: * ``android_worker``   —— **★ 2026-09-20**：Android Full Worker。
#:
#: 关于 ``android_worker``：它不是 ``cross_framework`` 的同义词。``cross_framework``
#: 描述的是**接力两侧框架不同**（PyTorch→llama.cpp）；Android worker 两侧都是
#: llama.cpp（裁层 GGUF），只是**跑在手机上**且用 JNI 注入 ``llama_batch.embd``。
#: 单独立一个 kind 是为了让「节点能力探测」能按类型区分 —— 见
#: ``android/.../AndroidWorkerCapabilities.SUPPORTED_STAGE_TYPES``。
#:
#: 注意：本集合目前**仅作声明**（``__all__`` 导出给外部读），``PipelineNode``
#: 的 ``kind`` 校验并不强制取值在此集合内（见 ``PipelineNode.__post_init__``）。
KNOWN_NODE_KINDS = frozenset({
    "local",
    "remote_rpc",
    "remote_pipeline",
    "cross_framework",
    "android_worker",
})


class PipelineNodeContractError(ValueError):
    """A node, artifact, or complete layer layout is not admissible."""


def _canonical_engine(value: Any, field: str) -> str:
    try:
        return canonical_model_engine(value)
    except ValueError as exc:
        raise PipelineNodeContractError(f"{field} is unsupported") from exc


def _non_negative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineNodeContractError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise PipelineNodeContractError(f"{field} must not be negative")
    return parsed


def _finite_float(value: Any, field: str, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineNodeContractError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise PipelineNodeContractError(f"{field} is invalid")
    return parsed


def _layer_range(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PipelineNodeContractError(f"{field} must be a two-item range")
    start = _non_negative_int(value[0], f"{field}.start")
    end = _non_negative_int(value[1], f"{field}.end")
    if end <= start:
        raise PipelineNodeContractError(f"{field} must be a non-empty half-open range")
    return start, end


@dataclass(frozen=True)
class PipelineNodeCapacity:
    """Free-memory admission attached to one execution segment."""

    capacity_bytes: int
    required_bytes: int = 0
    reserve_bytes: int = 0
    execution_device: str = "cpu"
    capacity_source: str = ""
    runtime_multiplier: float = 1.0
    score: float = 0.0

    def __post_init__(self) -> None:
        capacity = _non_negative_int(self.capacity_bytes, "capacity_bytes")
        required = _non_negative_int(self.required_bytes, "required_bytes")
        reserve = _non_negative_int(self.reserve_bytes, "reserve_bytes")
        multiplier = _finite_float(
            self.runtime_multiplier, "runtime_multiplier", minimum=0.000001,
        )
        score = _finite_float(self.score, "score")
        if required > capacity:
            raise PipelineNodeContractError("required_bytes exceeds capacity_bytes")
        if required and reserve > required:
            raise PipelineNodeContractError("reserve_bytes exceeds required_bytes")
        object.__setattr__(self, "capacity_bytes", capacity)
        object.__setattr__(self, "required_bytes", required)
        object.__setattr__(self, "reserve_bytes", reserve)
        object.__setattr__(self, "runtime_multiplier", multiplier)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "execution_device", str(self.execution_device or "cpu"))
        object.__setattr__(self, "capacity_source", str(self.capacity_source or ""))

    @property
    def headroom_bytes(self) -> int:
        return self.capacity_bytes - self.required_bytes

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["headroom_bytes"] = self.headroom_bytes
        return result


@dataclass(frozen=True)
class PipelineArtifactRef:
    """Identity and optional GGUF geometry for one node's artifact."""

    model_sha256: str
    artifact_kind: str
    artifact_sha256: str = ""
    manifest_sha256: str = ""
    architecture: str = ""
    source_layer_range: tuple[int, int] | None = None
    block_count: int | None = None
    nextn_predict_layers: int = 0

    def __post_init__(self) -> None:
        model_sha256 = str(self.model_sha256 or "").strip()
        artifact_kind = str(self.artifact_kind or "").strip()
        if not model_sha256:
            raise PipelineNodeContractError("artifact model_sha256 is required")
        if not artifact_kind:
            raise PipelineNodeContractError("artifact_kind is required")
        source_range = (
            _layer_range(self.source_layer_range, "source_layer_range")
            if self.source_layer_range is not None else None
        )
        block_count = (
            _non_negative_int(self.block_count, "block_count")
            if self.block_count is not None else None
        )
        nextn = _non_negative_int(
            self.nextn_predict_layers, "nextn_predict_layers",
        )
        if block_count is None and nextn:
            raise PipelineNodeContractError(
                "nextn_predict_layers requires block_count metadata",
            )
        if block_count is not None and nextn > block_count:
            raise PipelineNodeContractError(
                "nextn_predict_layers exceeds block_count",
            )
        object.__setattr__(self, "model_sha256", model_sha256)
        object.__setattr__(self, "artifact_kind", artifact_kind)
        object.__setattr__(self, "artifact_sha256", str(self.artifact_sha256 or ""))
        object.__setattr__(self, "manifest_sha256", str(self.manifest_sha256 or ""))
        object.__setattr__(self, "architecture", str(self.architecture or ""))
        object.__setattr__(self, "source_layer_range", source_range)
        object.__setattr__(self, "block_count", block_count)
        object.__setattr__(self, "nextn_predict_layers", nextn)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.source_layer_range is not None:
            result["source_layer_range"] = list(self.source_layer_range)
        return result


def _capacity(value: PipelineNodeCapacity | Mapping[str, Any]) -> PipelineNodeCapacity:
    if isinstance(value, PipelineNodeCapacity):
        return value
    if not isinstance(value, Mapping):
        raise PipelineNodeContractError("capacity must be PipelineNodeCapacity or a mapping")
    fields = {
        key: value[key]
        for key in (
            "capacity_bytes", "required_bytes", "reserve_bytes",
            "execution_device", "capacity_source", "runtime_multiplier", "score",
        )
        if key in value
    }
    return PipelineNodeCapacity(**fields)


def _artifact(value: PipelineArtifactRef | Mapping[str, Any]) -> PipelineArtifactRef:
    if isinstance(value, PipelineArtifactRef):
        return value
    if not isinstance(value, Mapping):
        raise PipelineNodeContractError("artifact must be PipelineArtifactRef or a mapping")
    fields = {
        key: value[key]
        for key in (
            "model_sha256", "artifact_kind", "artifact_sha256", "manifest_sha256",
            "architecture", "source_layer_range", "block_count",
            "nextn_predict_layers",
        )
        if key in value
    }
    return PipelineArtifactRef(**fields)


@dataclass(frozen=True)
class PipelineNode:
    """One contiguous execution segment in an engine-neutral layer pipeline."""

    node_id: str
    kind: str
    layer_range: tuple[int, int]
    engine: str
    location: str
    capacity: PipelineNodeCapacity | Mapping[str, Any]
    artifact: PipelineArtifactRef | Mapping[str, Any]
    has_embedding: bool = False
    has_lm_head: bool = False
    federated: bool = False
    cross_engine: bool = False
    handoff_at: int | None = None

    def __post_init__(self) -> None:
        node_id = str(self.node_id or "").strip()
        kind = str(self.kind or "").strip()
        engine = _canonical_engine(self.engine, "node engine")
        location = str(self.location or "").strip()
        if not node_id:
            raise PipelineNodeContractError("node_id is required")
        if not kind:
            raise PipelineNodeContractError("node kind is required")
        if not location:
            raise PipelineNodeContractError("node location is required")
        handoff_at = (
            _non_negative_int(self.handoff_at, "handoff_at")
            if self.handoff_at is not None else None
        )
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "layer_range", _layer_range(self.layer_range, "layer_range"))
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "capacity", _capacity(self.capacity))
        object.__setattr__(self, "artifact", _artifact(self.artifact))
        object.__setattr__(self, "has_embedding", bool(self.has_embedding))
        object.__setattr__(self, "has_lm_head", bool(self.has_lm_head))
        object.__setattr__(self, "federated", bool(self.federated))
        object.__setattr__(self, "cross_engine", bool(self.cross_engine))
        object.__setattr__(self, "handoff_at", handoff_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "layer_range": list(self.layer_range),
            "engine": self.engine,
            "location": self.location,
            "capacity": self.capacity.to_dict(),
            "artifact": self.artifact.to_dict(),
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            "federated": self.federated,
            "cross_engine": self.cross_engine,
            "handoff_at": self.handoff_at,
        }


@dataclass(frozen=True)
class PipelineLayout:
    """A validated, complete, and identity-consistent layer topology."""

    total_layers: int
    nodes: tuple[PipelineNode, ...]
    model_sha256: str
    architecture: str = ""

    @property
    def is_distributed(self) -> bool:
        return any(node.federated for node in self.nodes)

    @property
    def engines(self) -> tuple[str, ...]:
        return tuple(sorted({node.engine for node in self.nodes}))

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self._payload(), ensure_ascii=True, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _payload(self) -> dict[str, Any]:
        by_device: dict[str, dict[str, int]] = {}
        for node in self.nodes:
            item = by_device.setdefault(node.capacity.execution_device, {
                "node_count": 0,
                "capacity_bytes": 0,
                "required_bytes": 0,
                "headroom_bytes": 0,
            })
            item["node_count"] += 1
            item["capacity_bytes"] += node.capacity.capacity_bytes
            item["required_bytes"] += node.capacity.required_bytes
            item["headroom_bytes"] += node.capacity.headroom_bytes
        return {
            "schema_version": PIPELINE_NODE_SCHEMA_VERSION,
            "total_layers": self.total_layers,
            "model_sha256": self.model_sha256,
            "architecture": self.architecture,
            "is_distributed": self.is_distributed,
            "node_count": len(self.nodes),
            "engines": list(self.engines),
            "aggregate_capacity": {
                "capacity_bytes": sum(node.capacity.capacity_bytes for node in self.nodes),
                "required_bytes": sum(node.capacity.required_bytes for node in self.nodes),
                "headroom_bytes": sum(node.capacity.headroom_bytes for node in self.nodes),
                "by_execution_device": by_device,
            },
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["contract_sha256"] = self.contract_sha256
        return result


def validate_pipeline_nodes(
    nodes: Iterable[PipelineNode], *, total_layers: int,
) -> PipelineLayout:
    """Validate exact layer coverage, ownership, identity, and GGUF geometry."""
    total = _non_negative_int(total_layers, "total_layers")
    if total <= 0:
        raise PipelineNodeContractError("total_layers must be positive")
    try:
        values = tuple(nodes)
    except TypeError as exc:
        raise PipelineNodeContractError("nodes must be iterable") from exc
    if not values:
        raise PipelineNodeContractError("pipeline requires at least one node")
    if any(not isinstance(node, PipelineNode) for node in values):
        raise PipelineNodeContractError("every layout entry must be a PipelineNode")
    if len({node.node_id for node in values}) != len(values):
        raise PipelineNodeContractError("pipeline node_id values must be unique")

    ordered = tuple(sorted(values, key=lambda node: node.layer_range))
    model_sha256 = ordered[0].artifact.model_sha256
    architectures = {
        node.artifact.architecture for node in ordered if node.artifact.architecture
    }
    if len(architectures) > 1:
        raise PipelineNodeContractError("pipeline artifact architecture mismatch")

    layered_indexes = [
        i for i, n in enumerate(ordered)
        if n.layer_range[1] > n.layer_range[0]
        and str(getattr(n.capacity, "capacity_source", "") or "") != "relay_exempt"
    ]
    first_layered_index = layered_indexes[0] if layered_indexes else 0
    last_layered_index = layered_indexes[-1] if layered_indexes else len(ordered) - 1
    cursor = 0
    for index, node in enumerate(ordered):
        start, end = node.layer_range
        if start != cursor or end > total:
            raise PipelineNodeContractError(
                "pipeline layer ranges must cover every layer exactly once",
            )
        if node.artifact.model_sha256 != model_sha256:
            raise PipelineNodeContractError("pipeline model identity mismatch")
        # ★ Y 档第二条：relay 段是**零层占位**（层由远端段执行），可能排在最前/最后
        #   ⇒ embedding / LM Head 的归属判据要落在「**有层的**首/末节点」上，而不是
        #   简单的 index 0 / last。主节点收到末节点回的 hidden 后自己跑 Norm + LM Head
        #   （`scheduler_pipeline.py:4594-4595` 的推荐拓扑）⇒ `master[0,8) + middle[8,24)`
        #   这种形态里 master 既是有层的首节点、也是有层的末节点。
        is_relay_placeholder = (
            str(getattr(node.capacity, "capacity_source", "") or "") == "relay_exempt"
        )
        if not is_relay_placeholder:
            if node.has_embedding != (index == first_layered_index):
                raise PipelineNodeContractError(
                    "only the first layered pipeline node must own embedding",
                )
            if node.has_lm_head != (index == last_layered_index):
                raise PipelineNodeContractError(
                    "only the last layered pipeline node must own LM Head",
                )
        if node.kind == "cross_framework" and node.handoff_at is None:
            raise PipelineNodeContractError(
                "cross_framework node requires an explicit handoff_at",
            )
        if node.handoff_at is not None and node.handoff_at not in {start, end}:
            raise PipelineNodeContractError(
                "handoff_at must be a pipeline node boundary",
            )

        artifact_range = node.artifact.source_layer_range or node.layer_range
        artifact_start, artifact_end = artifact_range
        if artifact_start > start or artifact_end < end or artifact_end > total:
            raise PipelineNodeContractError(
                "artifact source_layer_range does not contain the assigned layers",
            )
        block_count = node.artifact.block_count
        nextn = node.artifact.nextn_predict_layers
        if block_count is not None:
            if block_count - nextn != artifact_end - artifact_start:
                raise PipelineNodeContractError(
                    "artifact block_count and nextn_predict_layers are inconsistent",
                )
            if artifact_end < total and nextn:
                raise PipelineNodeContractError(
                    "artifact without the model tail must set nextn_predict_layers to zero",
                )
        cursor = end
    if cursor != total:
        raise PipelineNodeContractError(
            "pipeline layer ranges must cover every layer exactly once",
        )
    return PipelineLayout(
        total_layers=total,
        nodes=ordered,
        model_sha256=model_sha256,
        architecture=next(iter(architectures), ""),
    )


def pipeline_layout_from_capacity_plan(
    plan: Mapping[str, Any],
    *,
    node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    default_engine: str = "pytorch",
) -> PipelineLayout:
    """Map an admitted ``solve_pipeline_capacity`` result into this contract."""
    if not isinstance(plan, Mapping) or plan.get("admitted") is not True:
        raise PipelineNodeContractError("capacity plan is not admitted")
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise PipelineNodeContractError("capacity plan has no assignments")
    total_layers = _non_negative_int(plan.get("total_layers", 0), "total_layers")
    model_sha256 = str(plan.get("model_sha256", "") or "").strip()
    architecture = str(plan.get("model_type", "") or "")
    metadata = node_metadata or {}
    nodes: list[PipelineNode] = []
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise PipelineNodeContractError("capacity assignment must be an object")
        node_id = str(assignment.get("node_id", "") or "")
        meta = dict(metadata.get(node_id, {}))
        start = _non_negative_int(assignment.get("start_layer", 0), "start_layer")
        end = _non_negative_int(assignment.get("end_layer", 0), "end_layer")
        is_local = bool(meta.get(
            "is_local", assignment.get("role") == "master",
        ))
        selected_artifact = assignment.get("layer_artifact")
        if not isinstance(selected_artifact, Mapping):
            selected_artifact = {}
        artifact_kind = str(
            meta.get("artifact_kind")
            or ("gguf_segment" if selected_artifact else "safetensors_assignment")
        )
        artifact_value = meta.get("artifact") or PipelineArtifactRef(
            # The layout-wide identity remains the logical source model.  A
            # Route-A worker can execute a format-specific GGUF segment whose
            # byte digest is intentionally different from the master's model.
            model_sha256=model_sha256,
            artifact_kind=artifact_kind,
            artifact_sha256=str(
                meta.get("artifact_sha256")
                or selected_artifact.get("artifact_sha256", "")
            ),
            manifest_sha256=str(meta.get("manifest_sha256") or ""),
            architecture=str(meta.get("architecture") or architecture),
            source_layer_range=(start, end),
        )
        nodes.append(PipelineNode(
            node_id=node_id,
            kind=str(meta.get("kind") or ("local" if is_local else "remote_pipeline")),
            layer_range=(start, end),
            # A selected layer_artifact is the exact GGUF segment executed by
            # Route A.  Node-level metadata may describe a co-located PyTorch
            # full model, which must not overwrite this assignment identity.
            engine=(
                "llama_cpp" if artifact_kind == "gguf_segment"
                else str(meta.get("engine") or default_engine)
            ),
            location=str(meta.get("location") or ("local" if is_local else f"node:{node_id}")),
            capacity=PipelineNodeCapacity(
                capacity_bytes=assignment.get("capacity_bytes", 0),
                required_bytes=assignment.get("required_bytes", 0),
                reserve_bytes=assignment.get("reserve_bytes", 0),
                execution_device=assignment.get("execution_device", "cpu"),
                capacity_source=assignment.get("capacity_source", ""),
                runtime_multiplier=assignment.get("runtime_multiplier", 1.0),
                score=assignment.get("score", 0.0),
            ),
            artifact=artifact_value,
            has_embedding=bool(assignment.get("has_embedding", False)),
            has_lm_head=bool(assignment.get("has_lm_head", False)),
            federated=bool(meta.get("federated", not is_local)),
            cross_engine=bool(meta.get("cross_engine", False)),
            handoff_at=meta.get("handoff_at"),
        ))
    annotated: list[PipelineNode] = []
    previous_engine = ""
    for node in sorted(nodes, key=lambda item: item.layer_range):
        crosses_engine = bool(previous_engine and previous_engine != node.engine)
        if crosses_engine:
            node = replace(
                node,
                cross_engine=True,
                handoff_at=node.layer_range[0],
            )
        annotated.append(node)
        previous_engine = node.engine
    return validate_pipeline_nodes(annotated, total_layers=total_layers)


def pipeline_layout_from_relay_handoff(
    handoff: Any,
    *,
    upstream_node_id: str = "relay-upstream",
    downstream_node_id: str = "relay-downstream",
    upstream_location: str = "local",
    downstream_location: str = "local",
    upstream_capacity: PipelineNodeCapacity | Mapping[str, Any] | None = None,
    downstream_capacity: PipelineNodeCapacity | Mapping[str, Any] | None = None,
) -> PipelineLayout:
    """Map an admitted RelayHandoff object without importing relay_contract."""
    upstream = getattr(handoff, "upstream", None)
    downstream = getattr(handoff, "downstream", None)
    cut_layer = getattr(handoff, "cut_layer", None)
    if upstream is None or downstream is None or cut_layer is None:
        raise PipelineNodeContractError("relay handoff is incomplete")
    total_layers = _non_negative_int(getattr(upstream, "n_layer", 0), "n_layer")
    cut = _non_negative_int(cut_layer, "cut_layer")
    default_capacity = PipelineNodeCapacity(capacity_bytes=0)
    upstream_engine = _canonical_engine(upstream.engine, "upstream engine")
    downstream_engine = _canonical_engine(downstream.engine, "downstream engine")
    cross_engine = upstream_engine != downstream_engine
    downstream_federated = downstream_location != "local"
    nodes = [
        PipelineNode(
            node_id=upstream_node_id,
            kind="local" if upstream_location == "local" else "remote_rpc",
            layer_range=(0, cut),
            engine=upstream_engine,
            location=upstream_location,
            capacity=upstream_capacity or default_capacity,
            artifact=PipelineArtifactRef(
                model_sha256=upstream.model_sha256,
                artifact_kind="gguf",
                artifact_sha256=getattr(upstream, "artifact_sha256", ""),
                architecture=getattr(upstream, "architecture", ""),
                source_layer_range=(0, total_layers),
                block_count=getattr(upstream, "block_count", None),
                nextn_predict_layers=getattr(upstream, "nextn_predict_layers", 0),
            ),
            has_embedding=True,
            federated=upstream_location != "local",
        ),
        PipelineNode(
            node_id=downstream_node_id,
            kind=(
                "cross_framework" if cross_engine
                else "remote_rpc" if downstream_federated
                else "local"
            ),
            layer_range=(cut, total_layers),
            engine=downstream_engine,
            location=downstream_location,
            capacity=downstream_capacity or default_capacity,
            artifact=PipelineArtifactRef(
                model_sha256=downstream.model_sha256,
                artifact_kind="gguf",
                artifact_sha256=getattr(downstream, "artifact_sha256", ""),
                architecture=getattr(downstream, "architecture", ""),
                source_layer_range=(cut, total_layers),
                block_count=getattr(downstream, "block_count", None),
                nextn_predict_layers=getattr(downstream, "nextn_predict_layers", 0),
            ),
            has_lm_head=True,
            federated=downstream_federated,
            cross_engine=cross_engine,
            handoff_at=cut,
        ),
    ]
    return validate_pipeline_nodes(nodes, total_layers=total_layers)


def pipeline_node_from_assignment_manifest(
    manifest: Mapping[str, Any],
    *,
    kind: str,
    location: str,
    capacity: PipelineNodeCapacity | Mapping[str, Any],
    engine: str = "pytorch",
    federated: bool = False,
    cross_engine: bool = False,
    handoff_at: int | None = None,
) -> PipelineNode:
    """Map one ``build_assignment_manifest`` result into a pipeline node."""
    if not isinstance(manifest, Mapping):
        raise PipelineNodeContractError("assignment manifest must be an object")
    if manifest.get("manifest_kind") != "pytorch_pipeline_assignment":
        raise PipelineNodeContractError("assignment manifest kind is invalid")
    assigned_range = _layer_range(manifest.get("layer_range"), "manifest.layer_range")
    return PipelineNode(
        node_id=str(manifest.get("node_id", "") or ""),
        kind=kind,
        layer_range=assigned_range,
        engine=engine,
        location=location,
        capacity=capacity,
        artifact=PipelineArtifactRef(
            model_sha256=str(manifest.get("model_sha256", "") or ""),
            artifact_kind="safetensors_assignment",
            manifest_sha256=str(manifest.get("manifest_sha256", "") or ""),
            architecture=str(manifest.get("model_type", "") or ""),
            source_layer_range=assigned_range,
        ),
        has_embedding=bool(manifest.get("has_embedding", False)),
        has_lm_head=bool(manifest.get("has_lm_head", False)),
        federated=federated,
        cross_engine=cross_engine,
        handoff_at=handoff_at,
    )


def pipeline_node_from_rpc_lease(
    lease: Any,
    *,
    capacity: PipelineNodeCapacity | Mapping[str, Any] | None = None,
    artifact: PipelineArtifactRef | Mapping[str, Any] | None = None,
    location: str = "",
) -> PipelineNode:
    """Map one RpcShardLease allocation into a remote pipeline node."""
    allocation = getattr(lease, "allocation", None)
    if not isinstance(allocation, Mapping):
        raise PipelineNodeContractError("RPC lease allocation is invalid")
    raw_range = allocation.get("layer_range")
    if raw_range is None:
        raw_range = (
            allocation.get("start_layer"), allocation.get("end_layer"),
        )
    assigned_range = _layer_range(raw_range, "allocation.layer_range")
    model_sha256 = str(getattr(lease, "model_sha256", "") or "")
    artifact_value = artifact or PipelineArtifactRef(
        model_sha256=model_sha256,
        artifact_kind=str(allocation.get("artifact_kind") or "gguf"),
        artifact_sha256=str(allocation.get("artifact_sha256") or ""),
        architecture=str(allocation.get("architecture") or ""),
        source_layer_range=assigned_range,
        block_count=allocation.get("block_count"),
        nextn_predict_layers=allocation.get("nextn_predict_layers", 0),
    )
    return PipelineNode(
        node_id=str(getattr(lease, "worker_id", "") or ""),
        kind="remote_rpc",
        layer_range=assigned_range,
        engine=_canonical_engine(
            allocation.get("engine") or "llama.cpp", "allocation.engine",
        ),
        location=location or str(allocation.get("location") or "remote"),
        capacity=capacity or PipelineNodeCapacity(capacity_bytes=0),
        artifact=artifact_value,
        has_embedding=bool(allocation.get("has_embedding", False)),
        has_lm_head=bool(allocation.get("has_lm_head", False)),
        federated=True,
        cross_engine=bool(allocation.get("cross_engine", False)),
        handoff_at=allocation.get("handoff_at"),
    )


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0 else 0.0


def _safe_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _project_resources(raw: Mapping[str, Any]) -> dict[str, Any]:
    profile = raw.get("device_info")
    profile = profile if isinstance(profile, Mapping) else {}
    cpu = profile.get("cpu")
    cpu = cpu if isinstance(cpu, Mapping) else {}
    ram = profile.get("ram")
    ram = ram if isinstance(ram, Mapping) else {}
    raw_gpus = profile.get("gpus")
    if isinstance(raw_gpus, list) and raw_gpus:
        gpus = [gpu for gpu in raw_gpus if isinstance(gpu, Mapping)]
    else:
        gpu = profile.get("gpu")
        gpus = [gpu] if isinstance(gpu, Mapping) and gpu else []
    state = str(raw.get("state", "unknown") or "unknown").lower()
    available = bool(raw.get("is_available", state in {"online", "busy"}))
    return {
        "node_id": str(raw.get("node_id", "") or ""),
        "role": str(raw.get("role", "") or ""),
        "node_type": str(raw.get("node_type", "") or ""),
        "state": state,
        "available": available,
        "cpu": {
            "physical_cores": _safe_int(cpu.get("physical_cores", 0)),
            "logical_cores": _safe_int(cpu.get("logical_cores", 0)),
        },
        "ram": {
            "total_gb": _safe_float(ram.get("total_gb", 0)),
            "available_gb": _safe_float(ram.get("available_gb", 0)),
        },
        "gpu": {
            "count": len(gpus),
            "cuda_count": sum(bool(gpu.get("cuda_available", False)) for gpu in gpus),
            "vram_total_gb": sum(_safe_float(gpu.get("vram_total_gb", 0)) for gpu in gpus),
            "vram_free_gb": sum(_safe_float(gpu.get("vram_free_gb", 0)) for gpu in gpus),
        },
    }


def build_aggregate_resource_view(
    nodes: Sequence[Mapping[str, Any]], *, local_node_id: str,
) -> dict[str, Any]:
    """Return a read-only hardware projection without addresses or local paths."""
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        raise PipelineNodeContractError("resource nodes must be a sequence")
    if any(not isinstance(node, Mapping) for node in nodes):
        raise PipelineNodeContractError("every resource node must be an object")
    projected = [_project_resources(node) for node in nodes]
    local_id = str(local_node_id or "")
    local = next((node for node in projected if node["node_id"] == local_id), None)
    remote = [node for node in projected if node["node_id"] != local_id]
    available = [node for node in projected if node["available"]]
    available_remote = [node for node in remote if node["available"]]

    totals = {
        "physical_cores": sum(node["cpu"]["physical_cores"] for node in available),
        "logical_cores": sum(node["cpu"]["logical_cores"] for node in available),
        "ram_total_gb": round(sum(node["ram"]["total_gb"] for node in available), 3),
        "ram_available_gb": round(
            sum(node["ram"]["available_gb"] for node in available), 3,
        ),
        "gpu_count": sum(node["gpu"]["count"] for node in available),
        "cuda_gpu_count": sum(node["gpu"]["cuda_count"] for node in available),
        "vram_total_gb": round(
            sum(node["gpu"]["vram_total_gb"] for node in available), 3,
        ),
        "vram_free_gb": round(
            sum(node["gpu"]["vram_free_gb"] for node in available), 3,
        ),
    }
    local_available = local if local and local["available"] else None
    remote_available_count = len(available_remote)
    return {
        "schema_version": RESOURCE_VIEW_SCHEMA_VERSION,
        "scope": "cluster",
        "is_distributed": bool(local and local["available"] and remote_available_count),
        "node_count": len(projected),
        "available_node_count": len(available),
        "unavailable_node_count": len(projected) - len(available),
        "remote_available_count": remote_available_count,
        "available": {
            "local": local_available,
            "remote": available_remote,
        },
        "totals": totals,
    }


__all__ = [
    "KNOWN_NODE_KINDS",
    "PIPELINE_NODE_SCHEMA_VERSION",
    "RESOURCE_VIEW_SCHEMA_VERSION",
    "PipelineArtifactRef",
    "PipelineLayout",
    "PipelineNode",
    "PipelineNodeCapacity",
    "PipelineNodeContractError",
    "build_aggregate_resource_view",
    "pipeline_layout_from_capacity_plan",
    "pipeline_layout_from_relay_handoff",
    "pipeline_node_from_assignment_manifest",
    "pipeline_node_from_rpc_lease",
    "validate_pipeline_nodes",
]
