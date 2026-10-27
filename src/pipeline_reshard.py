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
import logging
import threading
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

try:  # Support both ``src.*`` tests and the scheduler's source-root imports.
    from .executor_coexistence import (
        additional_executor_identities,
        choose_plan,
        has_candidate_locks,
        missing_additional_executors,
    )
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
    from executor_coexistence import (
        additional_executor_identities,
        choose_plan,
        has_candidate_locks,
        missing_additional_executors,
    )
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


logger = logging.getLogger(__name__)

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
    """One exact layer interval whose execution owner *identity* changes."""

    layer_range: tuple[int, int]
    source_node_id: str
    target_node_id: str
    # ★ 2026-10-27（审计 §9.4 #3）：**执行器身份必须进"谁搬走了"的判据**。
    #   同机主执行器与 CPU 执行器共用 `node_id`，只比裸 `node_id` 时"同一台机器上
    #   换了执行器"看起来**没有发生变化** ⇒ 部署侧完全看不到这次换人。
    #   实测对照（2026-10-27 纯计算，布局
    #   `master#""[0,8) + master#cpu[8,12) + worker-b[12,32)`
    #   → `master#""[0,16) + worker-b[16,32)`，即 `failed={("master","cpu")}` 后的
    #   典型重排）：某层 owner 从 `master#cpu` 变 `master#""`，旧 `_moves()` 的
    #   结果里**没有这一条**（只记了 `worker-b→master`），执行器换人被静默吞掉。
    #   必须留在**末尾且带默认值**：既有调用方按位置传前三个字段
    #   （`PipelineReshardMove((start, end), before, after)`），一旦把新字段插到
    #   中间，位置参数会**静默**错位。缺省 `""` = 该节点主执行器。
    source_executor_id: str = ""
    target_executor_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_range", _range(self.layer_range, "layer_range"))
        object.__setattr__(self, "source_node_id", _identity(
            self.source_node_id, "source_node_id",
        ))
        object.__setattr__(self, "target_node_id", _identity(
            self.target_node_id, "target_node_id",
        ))
        object.__setattr__(
            self, "source_executor_id", str(self.source_executor_id or "").strip(),
        )
        object.__setattr__(
            self, "target_executor_id", str(self.target_executor_id or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "layer_range": list(self.layer_range),
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
        }
        # ★ 仅**非空**时输出：缺省即主执行器 ⇒ 既有拓扑的 moves 投影逐字节不变
        #   （与 `PipelineNode.to_dict()` / `plan_id` 的兼容规则同一口径）。
        if self.source_executor_id:
            payload["source_executor_id"] = self.source_executor_id
        if self.target_executor_id:
            payload["target_executor_id"] = self.target_executor_id
        return payload


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
    """列出"执行器身份发生变化"的层区间。

    ★ 2026-10-27（审计 §9.4 #3）：判据从裸 `node_id` 换成 `_node_identity()` ——
    `(node_id, executor_id)`。只比 `node_id` 时，**同一台机器上换执行器**被判成
    "没有变化"。实测对照（2026-10-27 纯计算，同一对布局
    `master#""[0,8) + master#cpu[8,12) + worker-b[12,32)`
    → `master#""[0,16) + worker-b[16,32)`，即 CPU 段退场、健康的 GPU 段接回 `[8,12)`）：
      旧 `_moves()` = [worker-b→master (12,16)]          ← `[8,12)` 的执行器换人**丢失**
      新 `_moves()` = [{master#cpu→master#"" (8,12)}, {worker-b→master (12,16)}]
    部署侧因此看不到"执行器换人"，也就不知道 `[8,12)` 的权重需要由 GPU 段重新装载。
    合并相邻同源同宿区间时同样比四元组，否则 `A#cpu→B` 与 `A#""→B` 会被并成一条。
    """
    boundaries = sorted({
        0,
        current.total_layers,
        *[boundary for node in current.nodes for boundary in node.layer_range],
        *[boundary for node in candidate.nodes for boundary in node.layer_range],
    })
    values: list[PipelineReshardMove] = []
    for start, end in zip(boundaries, boundaries[1:]):
        before = _node_identity(_node_for_layer(current, start))
        after = _node_identity(_node_for_layer(candidate, start))
        if before == after:
            continue
        if (
            values
            and (values[-1].source_node_id, values[-1].source_executor_id) == before
            and (values[-1].target_node_id, values[-1].target_executor_id) == after
            and values[-1].layer_range[1] == start
        ):
            previous = values.pop()
            values.append(PipelineReshardMove(
                (previous.layer_range[0], end),
                before[0], after[0], before[1], after[1],
            ))
        else:
            values.append(PipelineReshardMove(
                (start, end), before[0], after[0], before[1], after[1],
            ))
    return tuple(values)


def _candidate_metadata(
    current: PipelineLayout,
    supplied: Mapping[Any, Mapping[str, Any]] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """候选布局的节点元数据，键是**完整执行器身份** `(node_id, executor_id)`。

    ★ 2026-10-27（审计 §9.4 #4）：此前键是裸 `node_id` 且用 `setdefault` ⇒ 同机的
    第二个执行器（`executor_id="cpu"`）读到的是**第一段**（主执行器）的属性。
    实测对照（2026-10-27 纯计算，布局 `master#""[0,8) + master#cpu[8,12) +
    worker-a[12,20) + worker-b[20,32)`，`supplied` 为旧裸键形态）：
      旧 `_candidate_metadata()` 的键 = `['master', 'worker-a', 'worker-b']`，
        `master#cpu` 命中 `'master'` ⇒ `artifact_kind='safetensors_assignment'`
        （应为 `gguf_segment`）、`engine='pytorch'`（应为 `llama_cpp`）、
        `kind='local'`（应为 `cross_framework`）、`federated=False`，
        且 `handoff_at` 直接丢失（主段那份 meta 里没有这个键）。
      新写法键 = `[('master',''), ('master','cpu'), ...]`，CPU 段取到
        `gguf_segment / llama_cpp / cross_framework / handoff_at=8`。
    这是 A1（`scheduler._attach_pipeline_node_contract` 用裸 `node_id` 建 engines 表
    ⇒ 主段被盖上 CPU 段引擎）在 reshard 路径的**镜像**：同一个键，反方向的串味。

    入参 `supplied` 仍可能是旧形态（裸 `node_id` 为键，来自
    `scheduler._pipeline_node_metadata()`）：裸键按"主执行器"归一成
    `(node_id, "")`，附加执行器**不**从裸键继承（回退规则见
    `pipeline_node_contract._layout_metadata_for()`）。
    """
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in (supplied or {}).items():
        if not isinstance(key, (tuple, list)):
            result[(str(key or "").strip(), "")] = dict(value)
    for key, value in (supplied or {}).items():
        if isinstance(key, (tuple, list)) and len(key) == 2:
            identity = (str(key[0] or "").strip(), str(key[1] or "").strip())
            result[identity] = dict(value)
    for node in current.nodes:
        metadata = result.setdefault(_node_identity(node), {})
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
    node_metadata: Mapping[Any, Mapping[str, Any]] | None = None,
    artifact_availability: Iterable[PipelineArtifactAvailability] = (),
    safety_margin: float = 1.2,
    prefer_all_workers: bool = False,
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

    ``prefer_all_workers`` 直接透传给 `solve_pipeline_capacity`（语义见那里：
    True = 可行解之间优先**用尽**可用执行器，False = 优先**节点数最少**）。
    调用方（`scheduler`）应传 `config.prefer_all_workers_enabled()` —— 与
    `get_pipeline_capacity_plan` 同源，否则恢复计划会与 prepare 阶段的节点数策略
    打架（一边要铺开、一边要收拢），布局来回重切。
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
        # ★ 2026-10-27（审计 §9.4 #2）：**恢复计划也必须走共存取优**。
        #
        #   此前这里用新鲜 `capacity_nodes` 裸解一次：既没把 `prefer_all_workers`
        #   传下去（求解器缺省 False = 可行解里选**节点数最少**的），也没有在解出
        #   "不含本机 CPU 段"的布局后再争一次 ⇒ 重分片仍会把附加执行器**结构性
        #   甩掉**。它不读 `candidate_layer_ranges` 作为"是否需要重解"的判据，所以
        #   不是 B1 的同一**触发形态**；但"可行但已不该沿用"的结果完全一样。
        #
        #   实测（2026-10-27 纯计算复现，布局
        #   `master#""[0,8) + master#cpu[8,12) + worker-a[12,20) + worker-b[20,32)`，
        #   CPU 段工件钉在 `[8,12)`，`failed={"worker-b"}`，存活记录
        #   master 主 + master#cpu + worker-a）：
        #     prefer_all_workers=False → candidate =
        #       [('master','',0,21), ('worker-a','',21,32)]        ← CPU 段被甩掉
        #     prefer_all_workers=True  → candidate =
        #       [('master','',0,8), ('master','cpu',8,12), ('worker-a','',12,32)]
        #   同一布局再把陈旧候选锁占满（master 锁 `[0,21)`、worker-a 锁 `[21,32)`、
        #   CPU 段无锁）时：
        #     False → baseline 原样保留（released 求解同样出不了 CPU 段，取优判平）
        #     True  → `released_better`，取到含 CPU 段的三段方案
        #   —— 与审计 §9.4 #1 一致：`balanced` 档的决定性因素是 `prefer_all_workers`，
        #   释放切点本身不会凭空造出 CPU 段。
        #
        #   修法与 `scheduler.get_pipeline_capacity_plan` 的 B1 分支**同形**（共用
        #   `executor_coexistence`，不复制判定逻辑）：
        #     1. baseline 求解时透传 `prefer_all_workers`；
        #     2. baseline 里缺少**配置了的**附加执行器 ⇒ 放开
        #        `candidate_layer_ranges`（上一轮供给写下的"我手上确实有这个段"）
        #        再解一次；没有候选锁时不重解 —— 那是同一个问题，只多一次 DFS；
        #     3. 两案用 `choose_plan()` 取优（打平保留 baseline，机制才收敛）。
        #
        #   ⚠️ 不放宽 `candidate_layer_ranges` 在**求解器**里的语义
        #   （`capacity.py:1482-1491` 有完整论证）：它表达"手上确实有这个工件"，
        #   是硬事实。这里动的只是**释放时机**与节点数策略。
        capacity_plan = solve_pipeline_capacity(
            dict(descriptor), survivors, safety_margin=safety_margin,
            require_distributed=True, prefer_all_workers=prefer_all_workers,
        )
        expected_additional = additional_executor_identities(survivors)
        missing_additional = (
            missing_additional_executors(capacity_plan, survivors)
            if capacity_plan.get("admitted") else set()
        )
        release_reason = ""
        if not capacity_plan.get("admitted"):
            # 与 `:446` 的既有语义一致：无解时也释放一次（released 可能仍无解，
            # 但它的 `reason_code` 更接近真相）。没有候选锁就没什么可释放的。
            release_reason = "capacity" if has_candidate_locks(survivors) else ""
        elif missing_additional and has_candidate_locks(survivors):
            release_reason = "missing_additional_executors"
        if release_reason:
            released_records = [
                {key: value for key, value in record.items()
                 if key != "candidate_layer_ranges"}
                for record in survivors
            ]
            released_plan = None
            try:
                released_plan = solve_pipeline_capacity(
                    dict(descriptor), released_records,
                    safety_margin=safety_margin,
                    require_distributed=True,
                    prefer_all_workers=prefer_all_workers,
                )
            except PipelineCapacityError as exc:
                # 与 B1 收尾同义：baseline 已可行时，released 求解的异常**不能**
                # 反过来打掉可用布局（为了修"僵化"却让本来能跑的布局失败）。
                # 与 scheduler 的差别：本函数的"无解"是**返回值**（带具体
                # reason_code）而不是异常；若在这里抛出，外层的
                # `pipeline_reshard_input_invalid` 会覆盖既有拒绝原因，运维看到的
                # 会是"输入非法"而不是"容量不足"。故两种情形都保留 baseline。
                logger.warning(
                    "重分片切点释放求解失败，保留 baseline: reason=%s "
                    "baseline_admitted=%s detail=%s",
                    release_reason, capacity_plan.get("admitted"), exc,
                    exc_info=True,
                )
                released_plan = None
            if released_plan is not None and not capacity_plan.get("admitted"):
                # 原行为：无解时采用 released（可能仍无解，但诊断更接近真相）。
                capacity_plan = released_plan
            elif released_plan is not None:
                chosen, decision = choose_plan(
                    capacity_plan, released_plan, expected_additional,
                )
                logger.info(
                    "重分片切点释放取优: decision=%s reason=%s "
                    "prefer_all_workers=%s expected_additional=%s "
                    "missing_additional=%s baseline_admitted=%s released_admitted=%s "
                    "baseline_assignments=%s released_assignments=%s",
                    decision, release_reason, prefer_all_workers,
                    sorted(expected_additional), sorted(missing_additional),
                    capacity_plan.get("admitted"), released_plan.get("admitted"),
                    [(a.get("node_id"), a.get("executor_id", ""),
                      a.get("start_layer"), a.get("end_layer"))
                     for a in capacity_plan.get("assignments", [])],
                    [(a.get("node_id"), a.get("executor_id", ""),
                      a.get("start_layer"), a.get("end_layer"))
                     for a in released_plan.get("assignments", [])],
                )
                if chosen is released_plan:
                    capacity_plan = released_plan
        if not capacity_plan.get("admitted"):
            return PipelineReshardDecision(
                False,
                str(capacity_plan.get(
                    "reason_code", "pipeline_reshard_capacity_insufficient",
                )),
                reason=str(capacity_plan.get("reason", "")),
            )
        engines = {
            _capacity_identity(item): item["engine"]
            for item in survivors if item.get("engine")
        }
        capacity_plan = {
            **capacity_plan,
            "assignments": [
                {**item, "engine": engines[_capacity_identity(item)]}
                if not item.get("engine") and _capacity_identity(item) in engines
                else dict(item)
                for item in capacity_plan["assignments"]
            ],
        }
        candidate = pipeline_layout_from_capacity_plan(
            capacity_plan,
            node_metadata=_candidate_metadata(validated, node_metadata),
        )
        capacity_plan["pipeline_layout"] = candidate.to_dict()
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
        node_metadata: Mapping[Any, Mapping[str, Any]] | None = None,
        artifact_availability: Iterable[PipelineArtifactAvailability] = (),
        safety_margin: float = 1.2,
        prefer_all_workers: bool = False,
    ) -> PipelineReshardDecision:
        """Stage a recovery plan for failed nodes **or** single executors.

        `failed_node_ids` 的元素既可以是裸 `node_id`（整机失联）也可以是
        `(node_id, executor_id)`（只坏了一个执行器）；语义与实测见
        `plan_pipeline_reshard` / `_failure_targets`。

        `prefer_all_workers` 必须由调用方按运行时策略传入
        （`config.prefer_all_workers_enabled()`）—— 本模块刻意不读环境：
        `get_pipeline_capacity_plan` 与恢复计划必须是**同一套**节点数策略，
        否则 prepare 想要铺开、reshard 想要收拢，布局会来回重切。
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
                prefer_all_workers=prefer_all_workers,
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
