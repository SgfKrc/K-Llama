"""Fail-closed recovery decisions for durable task and handoff state.

Recovery is deliberately a metadata operation.  It never copies an in-memory
snapshot, model state, prompt, output, lease, or runtime handle to another
node.  A retry decision only records that an explicit re-attachment may retry
the work; it is not a second submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


RECOVERY_SCHEMA_VERSION = "qlh.cluster.recovery.v1"
RECOVERY_ACTIONS = frozenset(
    {"ignore", "continue", "retry", "cancel", "handoff_timeout", "fail"}
)
TERMINAL_WORKFLOW_STATES = frozenset({"completed", "failed", "cancelled"})
TERMINAL_ATTEMPT_STATES = frozenset(
    {"completed", "failed", "expired", "cancelled"}
)


class RecoveryError(ValueError):
    """Raised when a recovery candidate is not a valid journal projection."""


@dataclass(frozen=True)
class RecoveryDecision:
    workflow_id: str
    action: str
    reason: str
    previous_state: str
    retry_stage_ids: tuple[str, ...] = ()
    expired_attempts: int = 0
    already_applied: bool = False

    def __post_init__(self) -> None:
        if not self.workflow_id or self.action not in RECOVERY_ACTIONS:
            raise RecoveryError("invalid recovery decision")
        if not self.reason:
            raise RecoveryError("recovery reason must not be empty")
        if self.expired_attempts < 0:
            raise RecoveryError("expired_attempts must not be negative")

    @property
    def terminal(self) -> bool:
        return self.action in {"cancel", "handoff_timeout", "fail"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "workflow_id": self.workflow_id,
            "action": self.action,
            "reason": self.reason,
            "previous_state": self.previous_state,
            "retry_stage_ids": list(self.retry_stage_ids),
            "expired_attempts": self.expired_attempts,
            "already_applied": self.already_applied,
            "terminal": self.terminal,
        }


def _latest_event_type(events: Iterable[Mapping[str, Any]]) -> str:
    latest: Mapping[str, Any] | None = None
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if latest is None or int(event.get("sequence", 0) or 0) >= int(
            latest.get("sequence", 0) or 0
        ):
            latest = event
    return str(latest.get("event_type", "")) if latest else ""


def decide_recovery(
    snapshot: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]] = (),
) -> RecoveryDecision:
    """Classify a journal projection without submitting any work.

    The journal does not contain root inputs or model output bodies.  Therefore
    a retry is only advertised when the persisted stage explicitly declares
    itself retry-safe; the caller must later provide a fresh execution context.
    """

    if not isinstance(snapshot, Mapping):
        raise RecoveryError("recovery snapshot must be an object")
    workflow_id = str(snapshot.get("workflow_id", ""))
    state = str(snapshot.get("state", ""))
    if not workflow_id or not state:
        raise RecoveryError("recovery snapshot requires workflow_id and state")
    try:
        sequence = int(snapshot.get("last_sequence", 0))
    except (TypeError, ValueError) as exc:
        raise RecoveryError("recovery snapshot has invalid last_sequence") from exc
    if sequence <= 0:
        raise RecoveryError("recovery snapshot has invalid last_sequence")

    applied = bool(snapshot.get("recovery_applied", False))
    if applied:
        return RecoveryDecision(
            workflow_id, "ignore", str(snapshot.get("recovery_reason") or "recovery_applied"),
            state, already_applied=True,
        )
    if state in TERMINAL_WORKFLOW_STATES:
        return RecoveryDecision(workflow_id, "ignore", "terminal", state)

    handoff_state = str(
        snapshot.get("handoff_state") or snapshot.get("handoff_status") or ""
    )
    if bool(snapshot.get("handoff_pending")) or handoff_state in {"prepared", "fenced"}:
        return RecoveryDecision(
            workflow_id, "handoff_timeout", "handoff_timeout", state,
        )

    latest_event = _latest_event_type(events)
    if bool(snapshot.get("cancel_requested")) or latest_event == "workflow_cancel_requested":
        return RecoveryDecision(
            workflow_id, "cancel", "recovered_cancel_request", state,
        )

    if bool(snapshot.get("recovery_continuable")):
        return RecoveryDecision(
            workflow_id, "continue", "journal_state_continuable", state,
        )

    if state == "result_ready":
        # The journal stores result metadata only.  Do not replay model work or
        # pretend that an output body survived a process crash.
        return RecoveryDecision(
            workflow_id, "fail", "coordinator_restarted_before_result_commit", state,
        )

    stages = snapshot.get("stages", [])
    if not isinstance(stages, list):
        raise RecoveryError("recovery snapshot stages must be a list")
    retry_stage_ids: list[str] = []
    expired_attempts = 0
    unsafe = False
    for raw_stage in stages:
        if not isinstance(raw_stage, Mapping):
            raise RecoveryError("recovery snapshot stage must be an object")
        stage_state = str(raw_stage.get("state", ""))
        if stage_state in {"completed", "failed", "skipped", "cancelled"}:
            continue
        attempts = raw_stage.get("attempts", [])
        if not isinstance(attempts, list):
            raise RecoveryError("recovery snapshot attempts must be a list")
        expired_attempts += sum(
            1
            for attempt in attempts
            if isinstance(attempt, Mapping)
            and str(attempt.get("state", "")) not in TERMINAL_ATTEMPT_STATES
        )
        retry_safe = bool(raw_stage.get("retry_safe"))
        pure = bool(raw_stage.get("pure"))
        if retry_safe and pure:
            stage_id = str(raw_stage.get("stage_id", ""))
            if stage_id:
                retry_stage_ids.append(stage_id)
        else:
            unsafe = True

    if retry_stage_ids and not unsafe:
        return RecoveryDecision(
            workflow_id, "retry", "coordinator_restarted_during_execution", state,
            tuple(retry_stage_ids), expired_attempts,
        )
    return RecoveryDecision(
        workflow_id, "fail", "coordinator_restarted_during_execution", state,
        expired_attempts=expired_attempts,
    )


__all__ = [
    "RECOVERY_ACTIONS",
    "RECOVERY_SCHEMA_VERSION",
    "RecoveryDecision",
    "RecoveryError",
    "decide_recovery",
]
