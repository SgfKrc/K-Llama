"""Authoritative per-worker assignment identity, contract, receipt and phase.

``WorkerAssignmentRegistry`` is the single source of truth for an active
legacy layer assignment.  Callers derive compatibility views such as
``_layer_config_pushed`` from registry phase instead of maintaining parallel
expected/ACK collections.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional

#: 相位：非终止态按序前进；终止态只能从非终止态进入。
PHASE_STAGED = "staged"
PHASE_PUSHING = "pushing"
PHASE_ACKED = "acked"
PHASE_READY = "ready"
PHASE_RELEASED = "released"
PHASE_ABORTED = "aborted"

ACTIVE_PHASES = (PHASE_STAGED, PHASE_PUSHING, PHASE_ACKED, PHASE_READY)
TERMINAL_PHASES = (PHASE_RELEASED, PHASE_ABORTED)

#: 相位之间的允许前进方向（同相位重复设置视为幂等，不算越级）。
_PHASE_ORDER = {
    PHASE_STAGED: 0,
    PHASE_PUSHING: 1,
    PHASE_ACKED: 2,
    PHASE_READY: 3,
}

#: 本模块自有的 reason code（撤销原因复用 `task_worker_adapter.RELEASE_REASON_*`）。
REASON_CONFIG_PUBLISHED = "config_published"
REASON_CONFIG_ACKED = "config_acked"
REASON_CONFIG_CLEARED = "config_cleared"
REASON_RECONFIGURED = "reconfigured"
#: worker 侧确认释放该层配置（`stage` 级 released ACK）。
REASON_WORKER_RELEASED = "worker_released"


def new_assignment_id() -> str:
    return f"asg_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class WorkerAssignmentState:
    """一个 worker 当前 assignment 的完整身份与相位（不可变快照）。"""

    node_id: str
    assignment_id: str
    config_id: str = ""
    connection_generation: int = 0
    lease_id: str = ""
    lease_epoch: int = 0
    phase: str = PHASE_STAGED
    reason_code: str = ""
    updated_at: float = 0.0
    expected_config: Mapping[str, Any] = field(default_factory=dict, repr=False)
    acknowledgement: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def snapshot(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "assignment_id": self.assignment_id,
            "config_id": self.config_id,
            "connection_generation": int(self.connection_generation),
            "lease_id": self.lease_id,
            "lease_epoch": int(self.lease_epoch),
            "phase": self.phase,
            "reason_code": self.reason_code,
            "updated_at": float(self.updated_at),
            "expected_config": copy.deepcopy(dict(self.expected_config)),
            "acknowledgement": copy.deepcopy(dict(self.acknowledgement)),
        }


class WorkerAssignmentRegistry:
    """per-node assignment 状态的唯一写入点（线程安全由调用方的配置锁保证）。

    ⚠️ **调用方必须在持有 `_layer_config_lock` 时使用**：本类不自己加锁，以免与既有
    锁序（配置锁 → 其它锁）冲突。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] = new_assignment_id,
    ) -> None:
        self._clock = clock
        self._id_factory = id_factory
        self._states: dict[str, WorkerAssignmentState] = {}

    # -- 写入 ---------------------------------------------------------------

    def begin(
        self,
        node_id: str,
        *,
        config_id: str = "",
        connection_generation: int = 0,
        assignment_id: str = "",
        expected_config: Optional[Mapping[str, Any]] = None,
        acknowledgement: Optional[Mapping[str, Any]] = None,
        reason_code: str = REASON_CONFIG_PUBLISHED,
    ) -> WorkerAssignmentState:
        """开始（或换代）一个节点的 assignment —— **每次都换 `assignment_id`**。

        这是「同节点重连/重配必须换代际」的落点：旧 assignment 的身份与 lease 不会被
        新 assignment 复用，因此迟到的 ACK 无法冒充当前配置的就绪。
        """
        key = str(node_id or "")
        previous = self._states.get(key)
        if previous is not None and not previous.terminal:
            reason = REASON_RECONFIGURED
        else:
            reason = reason_code
        expected = copy.deepcopy(dict(expected_config or {}))
        identity = str(
            assignment_id or expected.get("assignment_id", "") or self._id_factory()
        )
        expected["assignment_id"] = identity
        effective_config_id = str(
            config_id or expected.get("config_id", "") or ""
        )
        receipt = copy.deepcopy(dict(acknowledgement or {}))
        if receipt:
            receipt["assignment_id"] = identity
        state = WorkerAssignmentState(
            node_id=key,
            assignment_id=identity,
            config_id=effective_config_id,
            connection_generation=int(connection_generation),
            lease_id="",
            lease_epoch=0,
            phase=PHASE_PUSHING,
            reason_code=reason,
            updated_at=self._clock(),
            expected_config=expected,
            acknowledgement=receipt,
        )
        self._states[key] = state
        return state

    def update_expected(
        self,
        node_id: str,
        expected_config: Mapping[str, Any],
        *,
        reason_code: str = REASON_CONFIG_PUBLISHED,
        reset_phase: bool = True,
    ) -> Optional[WorkerAssignmentState]:
        """Replace the current wire contract without changing its identity.

        Pipeline prepare → commit and retransmission are phases of one
        assignment. They retain one ``assignment_id`` while the expected
        payload advances atomically.
        """
        key = str(node_id or "")
        state = self._states.get(key)
        if state is None or state.terminal:
            return None
        expected = copy.deepcopy(dict(expected_config or {}))
        expected["assignment_id"] = state.assignment_id
        updated = replace(
            state,
            config_id=str(expected.get("config_id", state.config_id) or ""),
            phase=PHASE_PUSHING if reset_phase else state.phase,
            reason_code=str(reason_code or ""),
            expected_config=expected,
            acknowledgement={},
            updated_at=self._clock(),
        )
        self._states[key] = updated
        return updated

    def record_ack(
        self,
        node_id: str,
        acknowledgement: Mapping[str, Any],
    ) -> Optional[WorkerAssignmentState]:
        """Record a receipt only when it belongs to the current assignment."""
        key = str(node_id or "")
        state = self._states.get(key)
        if state is None:
            return None
        receipt = copy.deepcopy(dict(acknowledgement or {}))
        if not self.identity_matches(
            key,
            assignment_id=receipt.get("assignment_id"),
            config_id=receipt.get("config_id"),
            generation=receipt.get("generation"),
        ):
            return None
        if dict(state.acknowledgement) == receipt:
            return state
        updated = replace(
            state,
            acknowledgement=receipt,
            updated_at=self._clock(),
        )
        self._states[key] = updated
        return updated

    def attach_lease(
        self, node_id: str, *, lease_id: str, lease_epoch: int,
    ) -> Optional[WorkerAssignmentState]:
        """把 lease 绑到当前 assignment（终止态不再接受变更）。"""
        state = self._states.get(str(node_id or ""))
        if state is None or state.terminal:
            return None
        updated = replace(
            state,
            lease_id=str(lease_id or ""),
            lease_epoch=int(lease_epoch),
            updated_at=self._clock(),
        )
        self._states[updated.node_id] = updated
        return updated

    def transition(
        self, node_id: str, *, phase: str, reason_code: str = "",
    ) -> Optional[WorkerAssignmentState]:
        """按序前进相位；越级/回退/终止态变更一律**拒绝**（返回 `None`）。

        拒绝而不是抛错：调用方（既有 ACK/重试路径）把它们当作「不是当前 assignment 的
        事件」忽略即可 —— 这正是替代「多集合交叉判定」的地方。
        """
        state = self._states.get(str(node_id or ""))
        if state is None or state.terminal:
            return None
        if phase in TERMINAL_PHASES:
            return None
        current = _PHASE_ORDER.get(state.phase, -1)
        target = _PHASE_ORDER.get(phase, -1)
        if target < 0 or target < current:
            return None
        if target == current and str(reason_code or "") == state.reason_code:
            return state
        updated = replace(
            state,
            phase=phase,
            reason_code=str(reason_code or ""),
            updated_at=self._clock(),
        )
        self._states[updated.node_id] = updated
        return updated

    def release(
        self, node_id: str, *, reason_code: str, aborted: bool = False,
    ) -> Optional[WorkerAssignmentState]:
        """进入终止态并记录**单一** reason code（重复撤销幂等返回当前状态）。"""
        key = str(node_id or "")
        state = self._states.get(key)
        if state is None:
            return None
        if state.terminal:
            return state
        updated = replace(
            state,
            phase=PHASE_ABORTED if aborted else PHASE_RELEASED,
            reason_code=str(reason_code or ""),
            lease_id="",
            lease_epoch=0,
            updated_at=self._clock(),
        )
        self._states[key] = updated
        return updated

    # -- 读取 ---------------------------------------------------------------

    def state(self, node_id: str) -> Optional[WorkerAssignmentState]:
        return self._states.get(str(node_id or ""))

    def assignment_id(self, node_id: str) -> str:
        state = self.state(node_id)
        return "" if state is None else state.assignment_id

    def phase(self, node_id: str) -> str:
        state = self.state(node_id)
        return "" if state is None else state.phase

    def is_ready(self, node_id: str) -> bool:
        state = self.state(node_id)
        return state is not None and state.phase == PHASE_READY

    def identity_matches(
        self,
        node_id: str,
        *,
        assignment_id: object,
        config_id: object,
        generation: object,
    ) -> bool:
        """Compare one wire receipt with the complete current identity."""
        state = self.state(node_id)
        if state is None:
            return False
        if str(assignment_id or "") != state.assignment_id:
            return False
        if str(config_id or "") != state.config_id:
            return False
        expected = state.expected_config
        if "generation" not in expected:
            return generation in (None, "", 0, "0")
        try:
            return int(generation) == int(expected.get("generation"))
        except (TypeError, ValueError):
            return False

    def expected(self, node_id: str) -> dict[str, Any]:
        state = self.state(node_id)
        if state is None or state.terminal:
            return {}
        return copy.deepcopy(dict(state.expected_config))

    def acknowledgement(self, node_id: str) -> dict[str, Any]:
        state = self.state(node_id)
        if state is None or state.terminal:
            return {}
        return copy.deepcopy(dict(state.acknowledgement))

    def expected_configs(self) -> dict[str, dict[str, Any]]:
        return {
            node_id: copy.deepcopy(dict(state.expected_config))
            for node_id, state in self._states.items()
            if not state.terminal and state.expected_config
        }

    def acknowledgements(self) -> dict[str, dict[str, Any]]:
        return {
            node_id: copy.deepcopy(dict(state.acknowledgement))
            for node_id, state in self._states.items()
            if not state.terminal and state.acknowledgement
        }

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            node_id: state.snapshot()
            for node_id, state in sorted(self._states.items())
        }

    def restore(self, snapshot: Mapping[str, Mapping[str, Any]]) -> None:
        """Replace the registry with one previously emitted snapshot."""
        restored: dict[str, WorkerAssignmentState] = {}
        for raw_node_id, raw in (snapshot or {}).items():
            if not isinstance(raw, Mapping):
                continue
            node_id = str(raw.get("node_id", raw_node_id) or raw_node_id)
            assignment_id = str(raw.get("assignment_id", "") or "")
            phase = str(raw.get("phase", "") or "")
            if not node_id or not assignment_id or phase not in {
                *ACTIVE_PHASES, *TERMINAL_PHASES,
            }:
                continue
            restored[node_id] = WorkerAssignmentState(
                node_id=node_id,
                assignment_id=assignment_id,
                config_id=str(raw.get("config_id", "") or ""),
                connection_generation=int(
                    raw.get("connection_generation", 0) or 0
                ),
                lease_id=str(raw.get("lease_id", "") or ""),
                lease_epoch=int(raw.get("lease_epoch", 0) or 0),
                phase=phase,
                reason_code=str(raw.get("reason_code", "") or ""),
                updated_at=float(raw.get("updated_at", self._clock()) or 0.0),
                expected_config=copy.deepcopy(dict(
                    raw.get("expected_config", {})
                    if isinstance(raw.get("expected_config", {}), Mapping)
                    else {}
                )),
                acknowledgement=copy.deepcopy(dict(
                    raw.get("acknowledgement", {})
                    if isinstance(raw.get("acknowledgement", {}), Mapping)
                    else {}
                )),
            )
        self._states = restored

    def drop(self, node_id: str) -> None:
        """移除记录（测试与节点彻底退场时使用）。"""
        self._states.pop(str(node_id or ""), None)

    def clear(self) -> None:
        """清空全部记录（对应用户侧的重启/幽灵节点清理语义）。"""
        self._states.clear()

    def invalidate(
        self, node_id: str, *, reason_code: str,
    ) -> Optional[WorkerAssignmentState]:
        """把当前 assignment 的相位**退回** `pushing`（本轮就绪证据作废）。

        与 [transition] 的「只许前进」不冲突：这不是状态推进，而是「刚收到的 ACK 不成立
        ⇒ 重新等待 ACK」，因此保留 `assignment_id` 与 generation（迟到 ACK 仍无法冒充）。
        终止态不接受作废（已终止的 assignment 不再等 ACK）。
        """
        key = str(node_id or "")
        state = self._states.get(key)
        if state is None or state.terminal:
            return None
        updated = replace(
            state,
            phase=PHASE_PUSHING,
            reason_code=str(reason_code or ""),
            updated_at=self._clock(),
        )
        self._states[key] = updated
        return updated


def assignment_state_summary(
    states: Mapping[str, WorkerAssignmentState] | Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """按相位计数（诊断/日志用）。"""
    counts: dict[str, int] = {}
    for value in states.values():
        phase = value.phase if isinstance(value, WorkerAssignmentState) else str(
            value.get("phase", "")
        )
        counts[phase] = counts.get(phase, 0) + 1
    return counts
