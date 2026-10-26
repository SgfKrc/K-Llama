"""Fail-closed recovery planning for a contiguous layer pipeline.

The execution engines own loading and inference.  This module owns only the
control-plane transition: re-solve the surviving topology, prove the required
artifacts are available, and fence the old topology before publishing a new
one.  Keeping that state separate prevents a disconnect from publishing an
unloadable assignment to the data plane.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

try:  # Support both ``src.*`` tests and the scheduler's source-root imports.
    from .llama_rpc_contract import LeaseDecision, RpcShardLease, RpcShardLeaseBook
    from .pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity
    from .pipeline_node_contract import (
        PipelineArtifactRef,
        PipelineLayout,
        PipelineNode,
        PipelineNodeContractError,
        pipeline_layout_from_capacity_plan,
        validate_pipeline_nodes,
    )
except ImportError:  # pragma: no cover - exercised by the application entrypoint.
    from llama_rpc_contract import LeaseDecision, RpcShardLease, RpcShardLeaseBook
    from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity
    from pipeline_node_contract import (
        PipelineArtifactRef,
        PipelineLayout,
        PipelineNode,
        PipelineNodeContractError,
        pipeline_layout_from_capacity_plan,
        validate_pipeline_nodes,
    )


RESHARD_SCHEMA_VERSION = 1


class PipelineReshardError(ValueError):
    """A reshard request cannot be planned or committed safely."""


def _range(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise PipelineReshardError(f"{field} must be a two-item range")
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise PipelineReshardError(f"{field} must contain integers") from exc
    if start < 0 or end <= start:
        raise PipelineReshardError(f"{field} must be a non-empty half-open range")
    return start, end


def _identity(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise PipelineReshardError(f"{field} is required")
    return result


def _node_identity(node: Any) -> tuple[str, str]:
    """返回节点的完整身份 `(node_id, executor_id)`。

    ★ 2026-10-27（A3/A4）：一台物理机现在可以同时承载多个**执行器** —— 主执行器
    `executor_id=""`（GPU 段，PyTorch）与附加执行器 `executor_id="cpu"`（CPU 段，
    llama.cpp keep-head）**共用同一个 `node_id`**。唯一身份因此是二元组
    （`PipelineNode.identity_key` / `Segment.identity_key` 都已经是这个形状）。

    本模块既有调用方可能仍传入**旧形态的 duck-typed 节点**（只有 `node_id`）⇒
    用 `getattr` 兜底，缺省 `""` 与"该节点的主执行器"等价（与
    `pipeline_node_contract.py` 里 `PipelineNode.executor_id` 的约定一致）。
    """
    node_id = str(getattr(node, "node_id", "") or "").strip()
    executor_id = str(getattr(node, "executor_id", "") or "").strip()
    return (node_id, executor_id)


def _capacity_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    """容量记录的身份。`_get_pipeline_capacity_nodes` 只给**附加**执行器带
    `executor_id`（主执行器记录没有该键）⇒ 缺省 `""` 即主执行器。"""
    return (
        str(record.get("node_id", "") or "").strip(),
        str(record.get("executor_id") or "").strip(),
    )


def _failure_targets(
    values: Iterable[Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """把失败输入规范化为「节点级」与「执行器级」两组（语义不同，不能合并）。

    ★ 2026-10-27（A4）：`failed_node_ids` 原本只收 `node_id`，是**节点级**判据 ——
    TCP 断线 = 整机失联，该机器上**所有**执行器一起死（`scheduler.py` 断线路径传的
    正是裸 `node_id`，这一语义必须保持：裸字符串 = 整机）。

    但"**只有 CPU 执行器坏了**"（llama.cpp 段加载失败 / 段崩溃）在节点级无法表达：
    传 `master` 会把健康的 GPU 段一起抹掉。实测（2026-10-27 纯计算复现，布局
    `master[0,8) + master#cpu[8,12) + worker-a[12,20) + worker-b[20,32)`）：

        failed={"master"} → accepted=True，候选节点 = ['worker-a', 'worker-b']
        —— master 上**没坏**的 GPU 段 `[0,8)` 也被连带剔除。

    所以二元组 `(node_id, executor_id)` 是**执行器级**判据；`(node_id, "")` 表示
    "只主执行器坏"（与裸 `node_id` **不等价**，刻意如此）。
    """
    nodes: set[str] = set()
    executors: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, str):
            nodes.add(_identity(value, "failed_node_id"))
            continue
        if isinstance(value, (tuple, list)) and len(value) == 2:
            node_id = _identity(value[0], "failed_node_id")
            executors.add((node_id, str(value[1] or "").strip()))
            continue
        raise PipelineReshardError(
            "failed node identity must be a node_id or a (node_id, executor_id) pair",
        )
    return tuple(sorted(nodes)), tuple(sorted(executors))


def _failed_identity_keys(
    layout_nodes: Iterable[Any],
    node_level: tuple[str, ...],
    executor_level: tuple[tuple[str, str], ...],
) -> set[tuple[str, str]]:
    """把两组失败目标展开成**精确**的 `(node_id, executor_id)` 集合。

    节点级目标按**当前布局**展开为该 `node_id` 的所有执行器键：只有这样才能让
    容量记录过滤、已就绪证明过滤、候选布局校验共用同一套判据（否则每处都要写
    "节点级 or 执行器级"两遍，且容易只改一半 —— A3/A4 本来就是这个坑的形状）。
    展开前调用方已校验每个目标都存在于布局中，因此不会出现"展开成空集"的静默
    漏判。
    """
    keys = set(executor_level)
    wanted = set(node_level)
    keys.update(
        identity
        for identity in (_node_identity(node) for node in layout_nodes)
        if identity[0] in wanted
    )
    return keys


@dataclass(frozen=True)
class PipelineArtifactAvailability:
    """A verified artifact range currently available to one pipeline node."""

    node_id: str
    model_sha256: str
    artifact_kind: str
    layer_range: tuple[int, int]
    artifact_sha256: str = ""
    verified: bool = True
    has_embedding: bool = False
    has_lm_head: bool = False
    # ★ 2026-10-27（A3）：**执行器身份必须进 `covers()` 的判据**。同机主执行器与
    #   CPU 执行器共用 `node_id`，只看 `node_id` 会让主段的证明"覆盖"CPU 段的要求。
    #   必须留在**末尾且带默认值** `""`：`scripts/llama_pc_rpc.py`、
    #   `scheduler._commit_ready_pipeline_reshard` 与既有测试全部用关键字构造，
    #   但一旦有人改用位置参数，多出来的字段会**静默**错位（把 `verified` 当
    #   `executor_id`），所以字段顺序不能再动。缺省 `""` = 该节点主执行器，
    #   与 `PipelineNode.executor_id` 的约定一致。
    executor_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identity(self.node_id, "node_id"))
        object.__setattr__(self, "model_sha256", _identity(
            self.model_sha256, "model_sha256",
        ))
        object.__setattr__(self, "artifact_kind", _identity(
            self.artifact_kind, "artifact_kind",
        ))
        object.__setattr__(self, "layer_range", _range(self.layer_range, "layer_range"))
        object.__setattr__(self, "artifact_sha256", str(self.artifact_sha256 or "").strip())
        object.__setattr__(self, "verified", bool(self.verified))
        object.__setattr__(self, "has_embedding", bool(self.has_embedding))
        object.__setattr__(self, "has_lm_head", bool(self.has_lm_head))
        object.__setattr__(self, "executor_id", str(self.executor_id or "").strip())

    @classmethod
    def from_node(cls, node: PipelineNode) -> "PipelineArtifactAvailability":
        return cls(
            node_id=node.node_id,
            model_sha256=node.artifact.model_sha256,
            artifact_kind=node.artifact.artifact_kind,
            layer_range=node.artifact.source_layer_range or node.layer_range,
            artifact_sha256=node.artifact.artifact_sha256,
            verified=True,
            has_embedding=node.has_embedding,
            has_lm_head=node.has_lm_head,
            # ★ 2026-10-27（A3）：从 `PipelineNode` 直接读执行器身份（该字段自
            #   2026-10-25 起存在）；`getattr` 兜底是为了兼容没有该字段的旧
            #   duck-typed 节点 —— 缺省 `""` 表示主执行器，行为与旧版本一致。
            executor_id=str(getattr(node, "executor_id", "") or "").strip(),
        )

    def covers(self, requirement: "PipelineArtifactRequirement") -> bool:
        start, end = self.layer_range
        required_start, required_end = requirement.layer_range
        if not self.verified:
            return False
        # ★ 2026-10-27（A3）：身份是 `(node_id, executor_id)`，不是 `node_id`。
        #   只看 `node_id` 时，主执行器的证明会"覆盖"同机 CPU 执行器的新切段：
        #   主段的 `artifact.source_layer_range` 常是**整模** `[0,total)`（远端整模
        #   GGUF 与 relay/layout 适配都写 `(0,total)`）⇒ 区间包含判据恒成立 ⇒
        #   `assets_ready=True` **伪就绪**，真实失败拖到执行期才炸。实测
        #   （2026-10-27 纯计算复现）：主段 `[0,8)` 的证明以 `layer_range=(0,32)`
        #   去覆盖 CPU 段 `(8,24)` 的 requirement，`covers()` 返回 **True**。
        #   反过来（kind 不同）则一直等不到证明 ⇒ 永远 `assets_required`。
        return (
            self.node_id == requirement.node_id
            and self.executor_id == requirement.executor_id
            and self.model_sha256.casefold() == requirement.model_sha256.casefold()
            and self.artifact_kind == requirement.artifact_kind
            and start <= required_start
            and end >= required_end
            and (not requirement.has_embedding or self.has_embedding)
            and (not requirement.has_lm_head or self.has_lm_head)
            and (
                not requirement.artifact_sha256
                or (
                    bool(self.artifact_sha256)
                    and self.artifact_sha256.casefold()
                    == requirement.artifact_sha256.casefold()
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "model_sha256": self.model_sha256,
            "artifact_kind": self.artifact_kind,
            "layer_range": list(self.layer_range),
            "artifact_sha256": self.artifact_sha256,
            "verified": self.verified,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            # 显式空串而非省略：读这份投影的人要能区分"主执行器"与"旧版本没有这个维度"。
            "executor_id": self.executor_id,
        }


@dataclass(frozen=True)
class PipelineArtifactRequirement:
    """Artifact coverage a candidate node must prove before activation."""

    node_id: str
    model_sha256: str
    artifact_kind: str
    layer_range: tuple[int, int]
    artifact_sha256: str = ""
    has_embedding: bool = False
    has_lm_head: bool = False
    # ★ 2026-10-27（A3）：与 `PipelineArtifactAvailability.executor_id` 配对使用。
    #   同样**必须留在末尾且带默认值**（理由见上面的 availability：位置参数错位会
    #   静默把 `has_lm_head` 当 `executor_id`）。缺省 `""` = 主执行器。
    executor_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identity(self.node_id, "node_id"))
        object.__setattr__(self, "model_sha256", _identity(
            self.model_sha256, "model_sha256",
        ))
        object.__setattr__(self, "artifact_kind", _identity(
            self.artifact_kind, "artifact_kind",
        ))
        object.__setattr__(self, "layer_range", _range(self.layer_range, "layer_range"))
        object.__setattr__(self, "artifact_sha256", str(self.artifact_sha256 or "").strip())
        object.__setattr__(self, "has_embedding", bool(self.has_embedding))
        object.__setattr__(self, "has_lm_head", bool(self.has_lm_head))
        object.__setattr__(self, "executor_id", str(self.executor_id or "").strip())

    @classmethod
    def from_node(cls, node: PipelineNode) -> "PipelineArtifactRequirement":
        artifact: PipelineArtifactRef = node.artifact
        return cls(
            node_id=node.node_id,
            model_sha256=artifact.model_sha256,
            artifact_kind=artifact.artifact_kind,
            layer_range=node.layer_range,
            artifact_sha256=artifact.artifact_sha256,
            has_embedding=node.has_embedding,
            has_lm_head=node.has_lm_head,
            # ★ 2026-10-27（A3）：候选布局给出的 requirement 必须带上执行器身份，
            #   否则"同机主段证明覆盖 CPU 段要求"的伪就绪无法被 `covers()` 拦住。
            executor_id=str(getattr(node, "executor_id", "") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "model_sha256": self.model_sha256,
            "artifact_kind": self.artifact_kind,
            "layer_range": list(self.layer_range),
            "artifact_sha256": self.artifact_sha256,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
            "executor_id": self.executor_id,
        }


@dataclass(frozen=True)
class PipelineReshardMove:
    """One exact layer interval whose execution owner changes."""

    layer_range: tuple[int, int]
    source_node_id: str
    target_node_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_range", _range(self.layer_range, "layer_range"))
        object.__setattr__(self, "source_node_id", _identity(
            self.source_node_id, "source_node_id",
        ))
        object.__setattr__(self, "target_node_id", _identity(
            self.target_node_id, "target_node_id",
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_range": list(self.layer_range),
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
        }


@dataclass(frozen=True)
class PipelineReshardPlan:
    """A fully validated but not necessarily asset-ready topology change."""

    plan_id: str
    base_epoch: int
    base_contract_sha256: str
    failed_node_ids: tuple[str, ...]
    capacity_plan: Mapping[str, Any]
    candidate_layout: PipelineLayout
    moves: tuple[PipelineReshardMove, ...]
    requirements: tuple[PipelineArtifactRequirement, ...]
    missing_requirements: tuple[PipelineArtifactRequirement, ...]
    # ★ 2026-10-27（A4）：**执行器级**失败目标，如 `(("master", "cpu"),)`。
    #   与 `failed_node_ids`（节点级 = 整机失联）语义不同，必须分开保存，否则
    #   "只坏了 CPU 执行器"会被投影成"整台 master 失败"，前端/运维据此会去重启
    #   整机或放弃健康的 GPU 段。放在**末尾且带默认值**：既有调用方（含
    #   `replace()` 的使用者）不必改。
    failed_executor_ids: tuple[tuple[str, str], ...] = ()

    @property
    def assets_ready(self) -> bool:
        return not self.missing_requirements

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESHARD_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "base_epoch": self.base_epoch,
            "base_contract_sha256": self.base_contract_sha256,
            "failed_node_ids": list(self.failed_node_ids),
            "failed_executor_ids": [list(pair) for pair in self.failed_executor_ids],
            "assets_ready": self.assets_ready,
            "capacity_plan_id": str(self.capacity_plan.get("plan_id", "") or ""),
            "capacity_reason_code": str(self.capacity_plan.get("reason_code", "") or ""),
            "candidate_layout": self.candidate_layout.to_dict(),
            "moves": [move.to_dict() for move in self.moves],
            "requirements": [item.to_dict() for item in self.requirements],
            "missing_requirements": [
                item.to_dict() for item in self.missing_requirements
            ],
        }


@dataclass(frozen=True)
class PipelineReshardDecision:
    accepted: bool
    reason_code: str
    plan: PipelineReshardPlan | None = None
    reason: str = ""

    @property
    def ready_to_commit(self) -> bool:
        return bool(self.accepted and self.plan is not None and self.plan.assets_ready)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESHARD_SCHEMA_VERSION,
            "status": "ready" if self.ready_to_commit else (
                "staged" if self.accepted else "rejected"
            ),
            "accepted": self.accepted,
            "ready_to_commit": self.ready_to_commit,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "plan": self.plan.to_dict() if self.plan else None,
        }


@dataclass(frozen=True)
class PipelineReshardCommitDecision:
    accepted: bool
    reason_code: str
    epoch: int
    layout: PipelineLayout
    lease: RpcShardLease | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESHARD_SCHEMA_VERSION,
            "status": "committed" if self.accepted else "rejected",
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "epoch": self.epoch,
            "layout": self.layout.to_dict(),
            "lease": (
                {
                    "lease_id": self.lease.lease_id,
                    "epoch": self.lease.epoch,
                    "attempt": self.lease.attempt,
                    "status": self.lease.status,
                }
                if self.lease is not None else None
            ),
        }


def _node_for_layer(layout: PipelineLayout, layer: int) -> PipelineNode:
    for node in layout.nodes:
        start, end = node.layer_range
        if start <= layer < end:
            return node
    raise PipelineReshardError(f"layout has no owner for layer {layer}")


def _moves(current: PipelineLayout, candidate: PipelineLayout) -> tuple[PipelineReshardMove, ...]:
    boundaries = sorted({
        0,
        current.total_layers,
        *[boundary for node in current.nodes for boundary in node.layer_range],
        *[boundary for node in candidate.nodes for boundary in node.layer_range],
    })
    values: list[PipelineReshardMove] = []
    for start, end in zip(boundaries, boundaries[1:]):
        before = _node_for_layer(current, start).node_id
        after = _node_for_layer(candidate, start).node_id
        if before == after:
            continue
        if (
            values
            and values[-1].source_node_id == before
            and values[-1].target_node_id == after
            and values[-1].layer_range[1] == start
        ):
            previous = values.pop()
            values.append(PipelineReshardMove(
                (previous.layer_range[0], end), before, after,
            ))
        else:
            values.append(PipelineReshardMove((start, end), before, after))
    return tuple(values)


def _candidate_metadata(
    current: PipelineLayout,
    supplied: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    result = {str(node_id): dict(value) for node_id, value in (supplied or {}).items()}
    for node in current.nodes:
        metadata = result.setdefault(node.node_id, {})
        metadata.setdefault("is_local", not node.federated and node.location == "local")
        metadata.setdefault("location", node.location)
        metadata.setdefault("kind", node.kind)
        metadata.setdefault("federated", node.federated)
        metadata.setdefault("cross_engine", node.cross_engine)
        metadata.setdefault("engine", node.engine)
        metadata.setdefault("artifact_kind", node.artifact.artifact_kind)
        metadata.setdefault("architecture", node.artifact.architecture or current.architecture)
    return result


def _availability_values(
    current: PipelineLayout,
    failed_identities: set[tuple[str, str]],
    supplied: Iterable[PipelineArtifactAvailability],
) -> tuple[PipelineArtifactAvailability, ...]:
    """「已就绪证明」= 当前布局（剔除失败身份）+ 调用方补充的证明。

    ★ 2026-10-27（A4）：过滤判据从 `node_id` 换成 `(node_id, executor_id)`。
    只按 `node_id` 过滤时，"只有 CPU 执行器坏了"会把**同机健康的 GPU 段**的证明
    一起丢掉（主段 `[0,8)` 明明装着、可用），恢复计划于是凭空退化成
    `assets_required`；反向（主段坏、CPU 段健康）同样丢错。
    """
    values = [
        PipelineArtifactAvailability.from_node(node)
        for node in current.nodes
        if _node_identity(node) not in failed_identities
    ]
    for item in supplied:
        if not isinstance(item, PipelineArtifactAvailability):
            raise PipelineReshardError(
                "artifact availability must be PipelineArtifactAvailability",
            )
        if (item.node_id, item.executor_id) not in failed_identities:
            values.append(item)
    return tuple(values)


def _missing_requirements(
    requirements: tuple[PipelineArtifactRequirement, ...],
    availability: Iterable[PipelineArtifactAvailability],
) -> tuple[PipelineArtifactRequirement, ...]:
    available = tuple(availability)
    return tuple(
        requirement for requirement in requirements
        if not any(item.covers(requirement) for item in available)
    )


def plan_pipeline_reshard(
    current_layout: PipelineLayout,
    *,
    base_epoch: int,
    failed_node_ids: Iterable[str | tuple[str, str]],
    descriptor: Mapping[str, Any],
    capacity_nodes: Iterable[Mapping[str, Any]],
    node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_availability: Iterable[PipelineArtifactAvailability] = (),
    safety_margin: float = 1.2,
) -> PipelineReshardDecision:
    """Re-solve a failed topology without publishing it to an execution engine.

    Capacity records must describe the survivor's recovery budget, including
    the temporary residency required while a new layer artifact is loaded.
    A distributed layout stays distributed; a two-node loss is deliberately
    rejected here so the pre-existing local fallback remains a distinct path.

    ``failed_node_ids`` 的元素有两种语义（★ 2026-10-27，A4）：
      * 裸 `node_id` —— **节点级**：整机失联，该机器上所有执行器一起退场；
      * `(node_id, executor_id)` —— **执行器级**：只退场这一个执行器
        （`""` 就是该节点的主执行器）。
    """
    try:
        if not isinstance(current_layout, PipelineLayout):
            raise PipelineReshardError("current_layout must be PipelineLayout")
        validated = validate_pipeline_nodes(
            current_layout.nodes, total_layers=current_layout.total_layers,
        )
        epoch = int(base_epoch)
        if epoch < 1:
            raise PipelineReshardError("base_epoch must be positive")
        # ★ 2026-10-27（A4）：失败目标分两级 —— 裸 `node_id` 仍是**节点级**（整机
        #   失联，`scheduler.py` 断线路径的既有语义），`(node_id, executor_id)`
        #   才是**执行器级**。细节与实测见 `_failure_targets()`。
        failed, failed_executors = _failure_targets(failed_node_ids)
        if not failed and not failed_executors:
            raise PipelineReshardError("at least one failed node is required")
        known_ids = {node.node_id for node in validated.nodes}
        known_identities = {_node_identity(node) for node in validated.nodes}
        unknown = sorted(set(failed) - known_ids)
        unknown_executors = sorted(
            f"{node_id}#{executor_id}"
            for node_id, executor_id in failed_executors
            if (node_id, executor_id) not in known_identities
        )
        if unknown or unknown_executors:
            raise PipelineReshardError(
                "failed node is not in the active layout: "
                + ", ".join(unknown + unknown_executors),
            )
        # 判据统一展开成精确的 `(node_id, executor_id)` 集合：容量记录过滤、已就绪
        # 证明过滤、候选校验三处共用，避免"只改一半"留下旁路。
        failed_identities = _failed_identity_keys(
            validated.nodes, failed, failed_executors,
        )
        if not isinstance(descriptor, Mapping):
            raise PipelineReshardError("descriptor must be a mapping")
        descriptor_sha = str(descriptor.get("model_sha256", "") or "").strip()
        if descriptor_sha.casefold() != validated.model_sha256.casefold():
            return PipelineReshardDecision(
                False,
                "pipeline_reshard_model_identity_mismatch",
                reason="descriptor model_sha256 does not match the active layout",
            )
        survivors = [
            dict(item) for item in capacity_nodes
            if isinstance(item, Mapping)
            and _capacity_identity(item) not in failed_identities
        ]
        # ★ 2026-10-27（:446 阻断级）：这里判的是"存活**容量记录**够不够两条"，
        #   旧写法数的却是**不同 node_id 的个数** ⇒ 同一台机器上的两个执行器
        #   （主执行器 + `executor_id="cpu"`）只计 1 个 ⇒ 明明有两条记录却报
        #   `pipeline_reshard_survivors_unavailable`。实测（2026-10-27 纯计算复现，
        #   布局 master[0,8) + master#cpu[8,24) + worker[24,32)，worker 失败，
        #   容量记录只有 master 主 + master#cpu 两条）：
        #     survivors 记录数 = 2 / 不同 node_id 数 = 1
        #     → accepted=False reason=pipeline_reshard_survivors_unavailable
        #       reason="fewer than two survivor capacity records are available"
        #   而下游 `solve_pipeline_capacity(require_distributed=True)` 的判据本来就是
        #   **记录数**（`capacity.py` 的 `len(usable) < 2`）⇒ 这里改成 `len(survivors)`
        #   才与 reason 文案（"capacity **records**"）和求解器口径一致。
        if len(survivors) < 2:
            return PipelineReshardDecision(
                False,
                "pipeline_reshard_survivors_unavailable",
                reason="fewer than two survivor capacity records are available",
            )
        capacity_plan = solve_pipeline_capacity(
            dict(descriptor), survivors, safety_margin=safety_margin,
            require_distributed=True,
        )
        if not capacity_plan.get("admitted"):
            return PipelineReshardDecision(
                False,
                str(capacity_plan.get(
                    "reason_code", "pipeline_reshard_capacity_insufficient",
                )),
                reason=str(capacity_plan.get("reason", "")),
            )
        candidate = pipeline_layout_from_capacity_plan(
            capacity_plan,
            node_metadata=_candidate_metadata(validated, node_metadata),
        )
        # ★ 2026-10-27（A4）：候选布局的校验同样必须是**执行器级**身份。
        #   只比 `node_id` 时有两个反向错误：失败目标是 `(master, "cpu")` 时，
        #   候选里健康的 `(master, "")` 会与 `"master"` 撞上（若把失败集合统一成
        #   裸 node_id）；而传裸 `master`（整机失联）时又会把没坏的 GPU 段一起判死
        #   —— 实测（2026-10-27 纯计算复现）：`failed={"master"}` 的候选节点是
        #   `['worker-a', 'worker-b']`，本机 GPU 段 `[0,8)` 被整体剔除。
        #   错误文本仍以 "failed node" 起头（既有断言依赖它），但判据已是
        #   "失败**身份**"。
        retained = sorted(
            f"{node_id}#{executor_id}"
            for node_id, executor_id in (
                _node_identity(node) for node in candidate.nodes
            )
            if (node_id, executor_id) in failed_identities
        )
        if retained:
            raise PipelineReshardError(
                "candidate layout retained a failed node: " + ", ".join(retained),
            )
        requirements = tuple(
            PipelineArtifactRequirement.from_node(node) for node in candidate.nodes
        )
        available = _availability_values(
            validated, failed_identities, artifact_availability,
        )
        missing = _missing_requirements(requirements, available)
        moves = _moves(validated, candidate)
        plan_identity = {
            "schema_version": RESHARD_SCHEMA_VERSION,
            "base_epoch": epoch,
            "base_contract_sha256": validated.contract_sha256,
            "failed_node_ids": failed,
            # ★ 2026-10-27（A4）：执行器级失败必须参与 plan_id —— 否则
            #   `{("master", "cpu")}` 与 `{"master"}` 在身份上不可区分（前者只剔
            #   CPU 段、后者剔整机，候选布局可能相同但语义完全不同）。
            "failed_executor_ids": [list(pair) for pair in failed_executors],
            "candidate_contract_sha256": candidate.contract_sha256,
            "requirements": [item.to_dict() for item in requirements],
        }
        plan_id = hashlib.sha256(json.dumps(
            plan_identity, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        plan = PipelineReshardPlan(
            plan_id=plan_id,
            base_epoch=epoch,
            base_contract_sha256=validated.contract_sha256,
            failed_node_ids=failed,
            capacity_plan=capacity_plan,
            candidate_layout=candidate,
            moves=moves,
            requirements=requirements,
            missing_requirements=missing,
            failed_executor_ids=failed_executors,
        )
        return PipelineReshardDecision(
            True,
            "" if plan.assets_ready else "pipeline_reshard_assets_required",
            plan=plan,
        )
    except (PipelineNodeContractError, PipelineCapacityError, TypeError, ValueError) as exc:
        return PipelineReshardDecision(
            False, "pipeline_reshard_input_invalid", reason=str(exc),
        )


class PipelineReshardCoordinator:
    """Stage and atomically activate recovery layouts using RPC epoch fencing."""

    _TOPOLOGY_SHARD_ID = "pipeline-topology"
    _COORDINATOR_ID = "pipeline-reshard-coordinator"

    def __init__(
        self,
        layout: PipelineLayout,
        *,
        lease_book: RpcShardLeaseBook | None = None,
        artifact_availability: Iterable[PipelineArtifactAvailability] = (),
        control_certificate: Mapping[str, Any] | None = None,
    ) -> None:
        self._layout = validate_pipeline_nodes(layout.nodes, total_layers=layout.total_layers)
        self._lock = threading.RLock()
        self._lease_book = lease_book or RpcShardLeaseBook()
        self._control_certificate = control_certificate
        if self._lease_book._control_fence is not None and control_certificate is None:
            control_certificate = self._lease_book._control_fence.current_certificate()
            self._control_certificate = control_certificate
        self._topology_lease = self._lease_book.assign(
            self._TOPOLOGY_SHARD_ID,
            self._COORDINATOR_ID,
            self._layout.model_sha256,
            {"contract_sha256": self._layout.contract_sha256},
            lease_seconds=86400.0,
            certificate=control_certificate,
        )
        self._availability = list(_availability_values(
            self._layout, set(), artifact_availability,
        ))
        self._staged: dict[str, PipelineReshardPlan] = {}

    @property
    def layout(self) -> PipelineLayout:
        with self._lock:
            return self._layout

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._topology_lease.epoch

    @property
    def topology_lease(self) -> RpcShardLease:
        with self._lock:
            return self._topology_lease

    def fence(self, epoch: int) -> LeaseDecision:
        """Check whether a caller still owns the active topology epoch."""
        with self._lock:
            return self._lease_book.check(self._topology_lease.lease_id, int(epoch))

    def stage_failure(
        self,
        failed_node_ids: Iterable[str | tuple[str, str]],
        *,
        descriptor: Mapping[str, Any],
        capacity_nodes: Iterable[Mapping[str, Any]],
        node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        artifact_availability: Iterable[PipelineArtifactAvailability] = (),
        safety_margin: float = 1.2,
    ) -> PipelineReshardDecision:
        """Stage a recovery plan for failed nodes **or** single executors.

        `failed_node_ids` 的元素既可以是裸 `node_id`（整机失联）也可以是
        `(node_id, executor_id)`（只坏了一个执行器）；语义与实测见
        `plan_pipeline_reshard` / `_failure_targets`。
        """
        with self._lock:
            decision = plan_pipeline_reshard(
                self._layout,
                base_epoch=self._topology_lease.epoch,
                failed_node_ids=failed_node_ids,
                descriptor=descriptor,
                capacity_nodes=capacity_nodes,
                node_metadata=node_metadata,
                artifact_availability=tuple(self._availability) + tuple(artifact_availability),
                safety_margin=safety_margin,
            )
            if decision.accepted and decision.plan is not None:
                # Only one recovery transaction may be authoritative for an
                # epoch. A later failure supersedes every earlier candidate.
                self._staged.clear()
                self._staged[decision.plan.plan_id] = decision.plan
            return decision

    def record_artifact(self, availability: PipelineArtifactAvailability) -> None:
        if not isinstance(availability, PipelineArtifactAvailability):
            raise PipelineReshardError(
                "availability must be PipelineArtifactAvailability",
            )
        with self._lock:
            self._availability.append(availability)
            for plan_id, plan in tuple(self._staged.items()):
                self._staged[plan_id] = replace(
                    plan,
                    missing_requirements=_missing_requirements(
                        plan.requirements, self._availability,
                    ),
                )

    def staged_plan(self, plan_id: str) -> PipelineReshardPlan | None:
        """Return one immutable staged plan for execution-plane attestation."""
        with self._lock:
            return self._staged.get(str(plan_id))

    def commit(
        self, plan_id: str, *, expected_epoch: int,
        control_certificate: Mapping[str, Any] | None = None,
    ) -> PipelineReshardCommitDecision:
        with self._lock:
            plan = self._staged.get(str(plan_id))
            current_epoch = self._topology_lease.epoch
            if plan is None:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_unknown_plan", current_epoch, self._layout,
                )
            try:
                requested_epoch = int(expected_epoch)
            except (TypeError, ValueError):
                requested_epoch = -1
            if requested_epoch != current_epoch or plan.base_epoch != current_epoch:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_stale_epoch", current_epoch, self._layout,
                )
            if plan.base_contract_sha256 != self._layout.contract_sha256:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_stale_layout", current_epoch, self._layout,
                )
            if self._lease_book.check(
                self._topology_lease.lease_id, current_epoch,
            ).accepted is False:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_fenced", current_epoch, self._layout,
                )
            missing = _missing_requirements(plan.requirements, self._availability)
            if missing:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_assets_not_ready", current_epoch, self._layout,
                )
            next_lease = self._lease_book.reassign(
                self._TOPOLOGY_SHARD_ID,
                self._COORDINATOR_ID,
                plan.candidate_layout.model_sha256,
                {"contract_sha256": plan.candidate_layout.contract_sha256},
                reason="reshard",
                lease_seconds=86400.0,
                certificate=control_certificate or self._control_certificate,
            )
            self._control_certificate = control_certificate or self._control_certificate
            self._layout = plan.candidate_layout
            self._topology_lease = next_lease
            self._staged.clear()
            return PipelineReshardCommitDecision(
                True, "pipeline_reshard_committed", next_lease.epoch, self._layout,
                next_lease,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            plans = list(self._staged.values())
            staged = [plan.to_dict() for plan in plans]
            return {
                "schema_version": RESHARD_SCHEMA_VERSION,
                "status": (
                    "ready" if plans and all(plan.assets_ready for plan in plans)
                    else "staged" if plans
                    else "active"
                ),
                "epoch": self._topology_lease.epoch,
                "layout": self._layout.to_dict(),
                "staged": staged,
            }
