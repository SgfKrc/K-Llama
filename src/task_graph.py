"""Task-level DAG execution with explicit workflow and stage state."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, cast

from task_journal import JournalEvent, TaskJournal, TaskJournalError
from cluster_recovery import RecoveryError, decide_recovery
from request_deadline import (
    REQUEST_DEADLINE_EXCEEDED,
    RequestDeadline,
)
from request_outcome import (
    OUTCOME_COMPLETED,
    REASON_GENERATION_CANCELLED,
    TerminalOutcomeLatch,
)
from task_provider import (
    CallbackExecutionProvider,
    DEPENDENCY_FAILURES_KEY,
    ExecutionProvider,
    ModelIdentity,
    PROVIDER_ID_PATTERN,
    ProviderError,
    ProviderExecutionError,
    ProviderRegistry,
    Reservation,
    StageAttempt as ProviderStageAttempt,
    StageRequest,
    StageResult as ProviderStageResult,
    sanitize_result_metadata,
)


TERMINAL_WORKFLOW_STATES = {"completed", "failed", "cancelled"}
TERMINAL_STAGE_STATES = {"completed", "failed", "skipped", "cancelled"}
TERMINAL_ATTEMPT_STATES = {"completed", "failed", "expired", "cancelled"}

WORKFLOW_STATE_TRANSITIONS = {
    "created": {"running", "cancelled"},
    "running": {"result_ready", "failed", "cancelled"},
    "result_ready": {"completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}
STAGE_STATE_TRANSITIONS = {
    "blocked": {"ready", "skipped", "cancelled"},
    "ready": {"running", "failed", "skipped", "cancelled"},
    "running": {"ready", "completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "skipped": set(),
    "cancelled": set(),
}
ATTEMPT_STATE_TRANSITIONS = {
    "running": {"completed", "failed", "expired", "cancelled"},
    "completed": set(),
    "failed": set(),
    "expired": set(),
    "cancelled": set(),
}


class TaskGraphError(RuntimeError):
    """Base error for task-graph validation and execution."""

    def __init__(self, message: str, *, code: str = "task_graph_error"):
        self.code = code
        super().__init__(message)


class TaskGraphUnavailable(TaskGraphError):
    """Raised when the task graph cannot preserve required durable state."""

    def __init__(self, message: str):
        super().__init__(message, code="task_graph_unavailable")


class WorkflowNotFound(TaskGraphError):
    def __init__(self, workflow_id: str):
        super().__init__(workflow_id, code="workflow_not_found")


class WorkflowCancelled(TaskGraphError):
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        super().__init__(
            f"workflow {workflow_id} cancelled", code="workflow_cancelled",
        )


class WorkflowDeadlineExceeded(TaskGraphError):
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        super().__init__(
            f"workflow {workflow_id} request deadline exceeded",
            code=REQUEST_DEADLINE_EXCEEDED,
        )


class WorkflowExecutionError(TaskGraphError):
    def __init__(
        self,
        workflow_id: str,
        stage_id: str,
        message: str,
        *,
        code: str = "task_graph_execution_failed",
    ):
        self.workflow_id = workflow_id
        self.stage_id = stage_id
        super().__init__(
            f"workflow {workflow_id} stage {stage_id} failed: {message}",
            code=code,
        )


class _StageRetryScheduled(TaskGraphError):
    """Internal control flow: a ready Stage moved to its next Provider."""


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    stage_type: str
    depends_on: tuple[str, ...] = ()
    provider: str = "local_full_model"
    fallback_providers: tuple[str, ...] = ()
    pure: bool = False
    accept_timeout_seconds: float = 10.0
    lease_timeout_seconds: float = 300.0
    minimum_successful_dependencies: Optional[int] = None
    max_same_provider_retries: int = 0
    retry_safe: bool = False
    root_input_overrides: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    # Fixed templates may bind one completed dependency field into this Stage's
    # root input. Bound dependencies are not forwarded to the Provider.
    input_bindings: dict[str, tuple[str, str]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    # When omitted, the workflow identity is used for backwards compatibility.
    model_identity: Optional[ModelIdentity] = None
    # Explicit wire fields for specialized stage types such as layer_forward.
    stage_fields: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


@dataclass
class AttemptRecord:
    attempt_id: str
    provider: str
    provider_kind: str = ""
    provider_node_id: str = ""
    reservation_id: str = ""
    lease_id: str = ""
    lease_epoch: int = 0
    lease_expires_at: float = 0.0
    lease_deadline_monotonic: float = field(
        default=0.0, repr=False,
    )
    state: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: str = ""
    result_metadata: dict = field(default_factory=dict)
    result_sha256: str = ""

    def snapshot(self) -> dict:
        duration = 0.0
        if self.finished_at is not None:
            duration = max(0.0, self.finished_at - self.started_at)
        return {
            "attempt_id": self.attempt_id,
            "provider": self.provider,
            "provider_kind": self.provider_kind,
            "provider_node_id": self.provider_node_id,
            "reservation_id": self.reservation_id,
            "reservation_active": (
                bool(self.reservation_id)
                and self.state not in TERMINAL_ATTEMPT_STATES
            ),
            "lease_id": self.lease_id,
            "lease_epoch": self.lease_epoch,
            "lease_expires_at": self.lease_expires_at,
            "lease_enforced": self.lease_expires_at > 0,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(duration, 6),
            "error": self.error,
            "result_metadata": dict(self.result_metadata),
            "result_sha256": self.result_sha256,
        }


@dataclass
class StageRecord:
    spec: StageSpec
    state: str = "blocked"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: str = ""
    attempts: list[AttemptRecord] = field(default_factory=list)
    output: Optional[dict] = None
    provider_index: int = 0
    lease_epoch: int = 0
    winner_attempt_id: str = ""
    output_digest: str = ""
    retry_count: int = 0
    same_provider_retry_count: int = 0
    last_retry_error_code: str = ""
    result_rejection_count: int = 0
    last_result_rejection_reason: str = ""
    last_result_rejected_at: Optional[float] = None

    def provider_candidates(self) -> tuple[str, ...]:
        return (self.spec.provider, *self.spec.fallback_providers)

    def selected_provider(self) -> str:
        candidates = self.provider_candidates()
        return candidates[min(self.provider_index, len(candidates) - 1)]

    def minimum_successful_dependencies(self) -> int:
        configured = self.spec.minimum_successful_dependencies
        return len(self.spec.depends_on) if configured is None else configured

    def snapshot(
        self,
        effective_model_identity: Optional[ModelIdentity] = None,
    ) -> dict:
        output_digest = ""
        output_size = 0
        if self.output is not None:
            encoded = json.dumps(
                self.output, ensure_ascii=False, sort_keys=True, default=str,
            ).encode("utf-8")
            output_digest = hashlib.sha256(encoded).hexdigest()
            output_size = len(encoded)
        duration = 0.0
        if self.started_at is not None and self.finished_at is not None:
            duration = max(0.0, self.finished_at - self.started_at)
        return {
            "stage_id": self.spec.stage_id,
            "stage_type": self.spec.stage_type,
            "model_identity": (
                effective_model_identity.snapshot()
                if effective_model_identity is not None
                else (
                    self.spec.model_identity.snapshot()
                    if self.spec.model_identity is not None else None
                )
            ),
            "depends_on": list(self.spec.depends_on),
            "provider": (
                self.attempts[-1].provider
                if self.attempts else self.selected_provider()
            ),
            "requested_provider": self.spec.provider,
            "fallback_providers": list(self.spec.fallback_providers),
            "selected_provider": self.selected_provider(),
            "pure": self.spec.pure,
            "accept_timeout_seconds": self.spec.accept_timeout_seconds,
            "lease_timeout_seconds": self.spec.lease_timeout_seconds,
            "minimum_successful_dependencies": (
                self.minimum_successful_dependencies()
            ),
            "input_bindings": {
                target: {
                    "dependency_stage_id": source[0],
                    "output_key": source[1],
                }
                for target, source in self.spec.input_bindings.items()
            },
            "stage_fields": dict(self.spec.stage_fields),
            "max_same_provider_retries": self.spec.max_same_provider_retries,
            "retry_safe": self.spec.retry_safe,
            "lease_epoch": self.lease_epoch,
            "winner_attempt_id": self.winner_attempt_id,
            "retry_count": self.retry_count,
            "same_provider_retry_count": self.same_provider_retry_count,
            "last_retry_error_code": self.last_retry_error_code,
            "result_rejection_count": self.result_rejection_count,
            "last_result_rejection_reason": (
                self.last_result_rejection_reason
            ),
            "last_result_rejected_at": self.last_result_rejected_at,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(duration, 6),
            "error": self.error,
            "attempts": [attempt.snapshot() for attempt in self.attempts],
            "output_available": self.output is not None,
            "output_sha256": self.output_digest or output_digest,
            "output_size_bytes": output_size,
        }


@dataclass
class WorkflowRecord:
    workflow_id: str
    request_id: str
    template: str
    final_stage_id: str
    stages: dict[str, StageRecord]
    session_id: str = ""
    model_identity: Optional[ModelIdentity] = None
    runtime_context: dict = field(default_factory=dict, repr=False)
    request_deadline: Optional[RequestDeadline] = field(
        default=None, repr=False,
    )
    terminal_outcome: TerminalOutcomeLatch = field(
        default_factory=TerminalOutcomeLatch, repr=False,
    )
    state: str = "created"
    last_sequence: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    result_ready_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: str = ""
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False,
    )
    cancel_recorded: bool = field(default=False, repr=False)
    lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False,
    )

    def snapshot(self) -> dict:
        with self.lock:
            duration = 0.0
            duration_end = self.finished_at or self.result_ready_at
            if self.started_at is not None and duration_end is not None:
                duration = max(0.0, duration_end - self.started_at)
            stages = [
                stage.snapshot(self.effective_model_identity(stage.spec))
                for stage in self.stages.values()
            ]
            completed = sum(stage["state"] == "completed" for stage in stages)
            failed = sum(stage["state"] == "failed" for stage in stages)
            skipped = sum(stage["state"] == "skipped" for stage in stages)
            cancelled = sum(stage["state"] == "cancelled" for stage in stages)
            attempts = sum(len(stage["attempts"]) for stage in stages)
            retries = sum(int(stage["retry_count"]) for stage in stages)
            same_provider_retries = sum(
                int(stage["same_provider_retry_count"])
                for stage in stages
            )
            rejections = sum(
                int(stage["result_rejection_count"]) for stage in stages
            )
            terminal_reason = self.terminal_outcome.snapshot()
            final_dependency_ids: set[str] = set()
            pending = [self.final_stage_id]
            while pending:
                stage_id = pending.pop()
                if stage_id in final_dependency_ids:
                    continue
                final_dependency_ids.add(stage_id)
                pending.extend(self.stages[stage_id].spec.depends_on)
            final_path_incomplete = any(
                self.stages[stage_id].state in {"failed", "skipped"}
                for stage_id in final_dependency_ids
            )
            return {
                "workflow_id": self.workflow_id,
                "request_id": self.request_id,
                "session_id": self.session_id,
                "model_identity": (
                    self.model_identity.snapshot()
                    if self.model_identity is not None else None
                ),
                "template": self.template,
                "state": self.state,
                "last_sequence": self.last_sequence,
                "final_stage_id": self.final_stage_id,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "result_ready_at": self.result_ready_at,
                "finished_at": self.finished_at,
                "duration_seconds": round(duration, 6),
                "error": self.error,
                "request_deadline_epoch_ms": (
                    self.request_deadline.expires_at_epoch_ms
                    if self.request_deadline is not None else None
                ),
                "terminal_reason": terminal_reason,
                "stage_count": len(stages),
                "completed_stage_count": completed,
                "failed_stage_count": failed,
                "skipped_stage_count": skipped,
                "partial_result": bool(
                    self.state in {"result_ready", "completed"}
                    and final_path_incomplete
                ),
                "cancelled_stage_count": cancelled,
                "attempt_count": attempts,
                "retry_count": retries,
                "same_provider_retry_count": same_provider_retries,
                "result_rejection_count": rejections,
                "cancel_requested": (
                    self.cancel_event.is_set() and self.state != "completed"
                    and terminal_reason["reason_code"] in {
                        "", REASON_GENERATION_CANCELLED,
                    }
                ),
                "stages": stages,
            }

    def effective_model_identity(
        self,
        stage: StageSpec,
    ) -> Optional[ModelIdentity]:
        """Resolve a Stage identity without changing the workflow default."""
        return (
            stage.model_identity
            if stage.model_identity is not None
            else self.model_identity
        )


StageExecutor = Callable[
    [StageSpec, dict[str, dict], dict, threading.Event], dict
]

WORKFLOW_ID_PATTERN = re.compile(r"^wf_[A-Za-z0-9_-]{8,96}$")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
def dual_candidate_template() -> tuple[list[StageSpec], str]:
    """Return the first fixed workflow: two candidates followed by aggregation."""
    return [
        StageSpec("candidate_a", "full_inference"),
        StageSpec("candidate_b", "full_inference"),
        StageSpec(
            "aggregate",
            "aggregate",
            depends_on=("candidate_a", "candidate_b"),
            minimum_successful_dependencies=1,
            max_same_provider_retries=1,
            retry_safe=True,
        ),
    ], "aggregate"


class TaskGraphCoordinator:
    """Run bounded, in-memory workflows without sharing model-internal state."""

    def __init__(
        self,
        max_records: int = 100,
        journal: Optional[TaskJournal] = None,
        provider_registry: Optional[ProviderRegistry] = None,
        max_parallel_stages: int = 4,
        availability_error: str = "",
    ):
        self._max_records = max(1, int(max_records))
        self._journal = journal
        self._provider_registry = provider_registry or ProviderRegistry()
        self._max_parallel_stages = max(1, min(int(max_parallel_stages), 32))
        self._stage_executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.RLock()
        self._active_runs: dict[str, int] = {}
        self._run_condition = threading.Condition(threading.RLock())
        self._closing = False
        self._journal_error = str(availability_error or "")
        self._last_recovery = {
            "recovered_workflows": 0,
            "expired_attempts": 0,
            "failed_stages": 0,
            "skipped_stages": 0,
        }
        self._last_cleanup = {
            "deleted_workflows": 0,
            "deleted_events": 0,
            "deleted_by_age": 0,
            "deleted_by_limit": 0,
            "remaining_terminal": 0,
        }
        self._workflows: dict[str, WorkflowRecord] = {}
        self._pending_cancellations: dict[str, float] = {}
        self._active_provider_attempts: dict[
            str, tuple[ProviderRegistry, str]
        ] = {}
        self._provider_activity_lock = threading.RLock()
        self._lock = threading.RLock()

    @staticmethod
    def _require_transition(
        entity_type: str,
        entity_id: str,
        current_state: str,
        next_state: str,
        transitions: dict[str, set[str]],
    ) -> None:
        if next_state not in transitions.get(current_state, set()):
            raise TaskGraphError(
                f"invalid {entity_type} transition for {entity_id}: "
                f"{current_state} -> {next_state}"
            )

    def _record_event_locked(
        self,
        workflow: WorkflowRecord,
        *,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: dict,
        occurred_at: Optional[float] = None,
    ) -> None:
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        previous_sequence = workflow.last_sequence
        sequence = previous_sequence + 1
        event_time = time.time() if occurred_at is None else occurred_at
        workflow.last_sequence = sequence
        if self._journal is None:
            return
        event = JournalEvent(
            event_id=f"evt_{uuid.uuid4().hex}",
            workflow_id=workflow.workflow_id,
            sequence=sequence,
            entity_type=entity_type,
            entity_id=entity_id,
            event_type=event_type,
            occurred_at=event_time,
            payload=payload,
        )
        try:
            self._journal.append_event(
                event, self._durable_snapshot_locked(workflow),
            )
        except TaskJournalError as exc:
            workflow.last_sequence = previous_sequence
            self._journal_error = f"task journal unavailable: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc

    @staticmethod
    def _durable_snapshot_locked(workflow: WorkflowRecord) -> dict:
        snapshot = workflow.snapshot()
        workflow_error = str(snapshot.pop("error", "") or "")
        snapshot["error"] = ""
        snapshot["error_present"] = bool(workflow_error)
        for stage in snapshot.get("stages", []):
            stage_error = str(stage.pop("error", "") or "")
            stage["error"] = ""
            stage["error_present"] = bool(stage_error)
            for attempt in stage.get("attempts", []):
                attempt_error = str(attempt.pop("error", "") or "")
                attempt["error"] = ""
                attempt["error_present"] = bool(attempt_error)
        return snapshot

    def _transition_workflow_locked(
        self,
        workflow: WorkflowRecord,
        next_state: str,
        *,
        error: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        previous = (
            workflow.state,
            workflow.started_at,
            workflow.result_ready_at,
            workflow.finished_at,
            workflow.error,
        )
        current_state = workflow.state
        self._require_transition(
            "workflow",
            workflow.workflow_id,
            current_state,
            next_state,
            WORKFLOW_STATE_TRANSITIONS,
        )
        changed_at = time.time() if now is None else now
        workflow.state = next_state
        if next_state == "running":
            workflow.started_at = changed_at
        elif next_state == "result_ready":
            workflow.result_ready_at = changed_at
        elif next_state in TERMINAL_WORKFLOW_STATES:
            workflow.finished_at = changed_at
        if error is not None:
            workflow.error = error
        elif next_state == "completed":
            workflow.error = ""
        try:
            self._record_event_locked(
                workflow,
                entity_type="workflow",
                entity_id=workflow.workflow_id,
                event_type="workflow_state_changed",
                occurred_at=changed_at,
                payload={
                    "from_state": current_state,
                    "to_state": next_state,
                    "has_error": bool(error),
                },
            )
        except Exception:
            (
                workflow.state,
                workflow.started_at,
                workflow.result_ready_at,
                workflow.finished_at,
                workflow.error,
            ) = previous
            raise

    def _transition_stage_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        next_state: str,
        *,
        now: Optional[float] = None,
        error: Optional[str] = None,
        output: Optional[dict] = None,
        set_output: bool = False,
    ) -> None:
        previous = (
            stage.state,
            stage.started_at,
            stage.finished_at,
            stage.error,
            stage.output,
        )
        current_state = stage.state
        self._require_transition(
            "stage",
            stage.spec.stage_id,
            current_state,
            next_state,
            STAGE_STATE_TRANSITIONS,
        )
        changed_at = time.time() if now is None else now
        stage.state = next_state
        if next_state == "running" and stage.started_at is None:
            stage.started_at = changed_at
        if next_state in TERMINAL_STAGE_STATES:
            stage.finished_at = changed_at
        if error is not None:
            stage.error = error
        if set_output:
            stage.output = output
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage",
                entity_id=stage.spec.stage_id,
                event_type="stage_state_changed",
                occurred_at=changed_at,
                payload={
                    "from_state": current_state,
                    "to_state": next_state,
                    "stage_type": stage.spec.stage_type,
                    "provider": stage.selected_provider(),
                    "lease_epoch": stage.lease_epoch,
                    "has_error": bool(error),
                },
            )
        except Exception:
            (
                stage.state,
                stage.started_at,
                stage.finished_at,
                stage.error,
                stage.output,
            ) = previous
            raise

    def _transition_attempt_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
        next_state: str,
        *,
        now: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        previous = (attempt.state, attempt.finished_at, attempt.error)
        current_state = attempt.state
        self._require_transition(
            "attempt",
            attempt.attempt_id,
            current_state,
            next_state,
            ATTEMPT_STATE_TRANSITIONS,
        )
        changed_at = time.time() if now is None else now
        attempt.state = next_state
        if next_state in TERMINAL_ATTEMPT_STATES:
            attempt.finished_at = changed_at
        if error is not None:
            attempt.error = error
        try:
            self._record_event_locked(
                workflow,
                entity_type="attempt",
                entity_id=attempt.attempt_id,
                event_type=f"attempt_{next_state}",
                occurred_at=changed_at,
                payload={
                    "from_state": current_state,
                    "to_state": next_state,
                    "stage_id": stage.spec.stage_id,
                    "provider": attempt.provider,
                    "provider_kind": attempt.provider_kind,
                    "provider_node_id": attempt.provider_node_id,
                    "reservation_id": attempt.reservation_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "lease_expires_at": attempt.lease_expires_at,
                    "has_error": bool(error),
                },
            )
        except Exception:
            attempt.state, attempt.finished_at, attempt.error = previous
            raise

    def _append_attempt_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
    ) -> None:
        stage.attempts.append(attempt)
        try:
            self._record_event_locked(
                workflow,
                entity_type="attempt",
                entity_id=attempt.attempt_id,
                event_type="attempt_started",
                occurred_at=attempt.started_at,
                payload={
                    "state": attempt.state,
                    "stage_id": stage.spec.stage_id,
                    "provider": attempt.provider,
                    "provider_kind": attempt.provider_kind,
                    "provider_node_id": attempt.provider_node_id,
                    "reservation_id": attempt.reservation_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "lease_expires_at": attempt.lease_expires_at,
                },
            )
        except Exception:
            stage.attempts.remove(attempt)
            raise

    def _finish_stage_attempt_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
        next_state: str,
        *,
        now: Optional[float] = None,
        error: Optional[str] = None,
        output: Optional[dict] = None,
        set_output: bool = False,
        result_metadata: Optional[dict] = None,
        result_digest: str = "",
        set_winner: bool = False,
    ) -> None:
        previous_stage = (
            stage.state,
            stage.started_at,
            stage.finished_at,
            stage.error,
            stage.output,
            stage.winner_attempt_id,
            stage.output_digest,
        )
        previous_attempt = (
            attempt.state,
            attempt.finished_at,
            attempt.error,
            dict(attempt.result_metadata),
            attempt.result_sha256,
        )
        stage_from_state = stage.state
        attempt_from_state = attempt.state
        self._require_transition(
            "stage",
            stage.spec.stage_id,
            stage_from_state,
            next_state,
            STAGE_STATE_TRANSITIONS,
        )
        self._require_transition(
            "attempt",
            attempt.attempt_id,
            attempt_from_state,
            next_state,
            ATTEMPT_STATE_TRANSITIONS,
        )
        changed_at = time.time() if now is None else now
        stage.state = next_state
        stage.finished_at = changed_at
        attempt.state = next_state
        attempt.finished_at = changed_at
        if error is not None:
            stage.error = error
            attempt.error = error
        if set_output:
            stage.output = output
        if result_metadata is not None:
            attempt.result_metadata = dict(result_metadata)
        if result_digest:
            attempt.result_sha256 = result_digest
        if set_winner:
            stage.winner_attempt_id = attempt.attempt_id
            stage.output_digest = result_digest
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage_attempt",
                entity_id=f"{stage.spec.stage_id}:{attempt.attempt_id}",
                event_type=f"stage_attempt_{next_state}",
                occurred_at=changed_at,
                payload={
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": attempt.attempt_id,
                    "stage_from_state": stage_from_state,
                    "attempt_from_state": attempt_from_state,
                    "to_state": next_state,
                    "provider": attempt.provider,
                    "provider_kind": attempt.provider_kind,
                    "provider_node_id": attempt.provider_node_id,
                    "reservation_id": attempt.reservation_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "result_sha256": result_digest,
                    "winner": set_winner,
                    "has_error": bool(error),
                },
            )
        except Exception:
            (
                stage.state,
                stage.started_at,
                stage.finished_at,
                stage.error,
                stage.output,
                stage.winner_attempt_id,
                stage.output_digest,
            ) = previous_stage
            (
                attempt.state,
                attempt.finished_at,
                attempt.error,
                attempt.result_metadata,
                attempt.result_sha256,
            ) = previous_attempt
            raise

    @staticmethod
    def _output_sha256(output: dict) -> str:
        encoded = json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _reject_stage_result_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        result: ProviderStageResult,
        *,
        reason: str,
        result_digest: str,
        now: float,
    ) -> dict:
        previous = (
            stage.result_rejection_count,
            stage.last_result_rejection_reason,
            stage.last_result_rejected_at,
        )
        stage.result_rejection_count += 1
        stage.last_result_rejection_reason = reason
        stage.last_result_rejected_at = now
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage_result",
                entity_id=(
                    f"{stage.spec.stage_id}:"
                    f"{result.attempt_id or 'unknown'}"
                ),
                event_type="stage_result_rejected",
                occurred_at=now,
                payload={
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": result.attempt_id,
                    "provider": result.provider_id,
                    "received_lease_epoch": result.lease_epoch,
                    "current_lease_epoch": stage.lease_epoch,
                    "winner_attempt_id": stage.winner_attempt_id,
                    "result_sha256": result_digest,
                    "reason": reason,
                },
            )
        except Exception:
            (
                stage.result_rejection_count,
                stage.last_result_rejection_reason,
                stage.last_result_rejected_at,
            ) = previous
            raise
        return {
            "status": "rejected",
            "reason": reason,
            "stage_id": stage.spec.stage_id,
            "attempt_id": result.attempt_id,
            "lease_epoch": result.lease_epoch,
            "winner_attempt_id": stage.winner_attempt_id,
            "output_sha256": stage.output_digest,
        }

    def _submit_stage_result_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        result: ProviderStageResult,
        *,
        now: Optional[float] = None,
    ) -> dict:
        submitted_at = time.time() if now is None else now
        if not isinstance(result.output, dict):
            return self._reject_stage_result_locked(
                workflow,
                stage,
                result,
                reason="invalid_result_schema",
                result_digest="",
                now=submitted_at,
            )
        try:
            result_digest = self._output_sha256(result.output)
        except (TypeError, ValueError):
            return self._reject_stage_result_locked(
                workflow,
                stage,
                result,
                reason="invalid_result_schema",
                result_digest="",
                now=submitted_at,
            )
        if stage.winner_attempt_id:
            if (
                result.attempt_id == stage.winner_attempt_id
                and result_digest == stage.output_digest
            ):
                return {
                    "status": "idempotent",
                    "reason": "already_committed",
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": result.attempt_id,
                    "lease_epoch": result.lease_epoch,
                    "winner_attempt_id": stage.winner_attempt_id,
                    "output_sha256": stage.output_digest,
                }
            reason = (
                "winner_digest_mismatch"
                if result.attempt_id == stage.winner_attempt_id
                else "winner_already_committed"
            )
            return self._reject_stage_result_locked(
                workflow,
                stage,
                result,
                reason=reason,
                result_digest=result_digest,
                now=submitted_at,
            )
        if workflow.state in TERMINAL_WORKFLOW_STATES:
            return self._reject_stage_result_locked(
                workflow,
                stage,
                result,
                reason="workflow_terminal",
                result_digest=result_digest,
                now=submitted_at,
            )
        attempt = next(
            (
                item for item in stage.attempts
                if item.attempt_id == result.attempt_id
            ),
            None,
        )
        if attempt is None:
            reason = "attempt_not_owned_by_stage"
        elif result.provider_id != attempt.provider:
            reason = "provider_identity_mismatch"
        elif result.lease_epoch != attempt.lease_epoch:
            reason = "attempt_epoch_mismatch"
        elif attempt.lease_epoch != stage.lease_epoch:
            reason = "stale_lease_epoch"
        elif attempt.state != "running":
            reason = "attempt_not_running"
        elif stage.state != "running":
            reason = "stage_not_running"
        elif (
            attempt.lease_deadline_monotonic > 0
            and time.monotonic() > attempt.lease_deadline_monotonic
        ) or (
            attempt.lease_deadline_monotonic <= 0
            and attempt.lease_expires_at > 0
            and submitted_at > attempt.lease_expires_at
        ):
            reason = "lease_expired"
        else:
            self._finish_stage_attempt_locked(
                workflow,
                stage,
                attempt,
                "completed",
                now=submitted_at,
                output=result.output,
                set_output=True,
                result_metadata=result.metadata,
                result_digest=result_digest,
                set_winner=True,
            )
            return {
                "status": "committed",
                "reason": "winner_committed",
                "stage_id": stage.spec.stage_id,
                "attempt_id": attempt.attempt_id,
                "lease_epoch": attempt.lease_epoch,
                "winner_attempt_id": attempt.attempt_id,
                "output_sha256": result_digest,
            }
        return self._reject_stage_result_locked(
            workflow,
            stage,
            result,
            reason=reason,
            result_digest=result_digest,
            now=submitted_at,
        )

    @staticmethod
    def _can_retry_stage(stage: StageRecord) -> bool:
        return (
            stage.spec.pure
            and not stage.winner_attempt_id
            and stage.provider_index + 1 < len(stage.provider_candidates())
        )

    @staticmethod
    def _can_retry_same_provider(
        stage: StageRecord,
        attempt: Optional[AttemptRecord] = None,
    ) -> bool:
        return (
            stage.spec.retry_safe
            and not stage.winner_attempt_id
            and stage.same_provider_retry_count
            < stage.spec.max_same_provider_retries
            and (
                attempt is None
                or attempt.provider_kind in {
                    "local_full_model", "callback_compatibility",
                }
            )
        )

    def _advance_ready_stage_provider_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        *,
        error_code: str,
    ) -> None:
        if stage.state != "ready" or not self._can_retry_stage(stage):
            raise TaskGraphError("Stage is not eligible for Provider fallback")
        previous_index = stage.provider_index
        previous_retry_count = stage.retry_count
        previous_retry_error = stage.last_retry_error_code
        previous_provider = stage.selected_provider()
        stage.provider_index += 1
        stage.retry_count += 1
        stage.last_retry_error_code = error_code
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage",
                entity_id=stage.spec.stage_id,
                event_type="stage_reservation_retry_scheduled",
                payload={
                    "stage_id": stage.spec.stage_id,
                    "from_provider": previous_provider,
                    "to_provider": stage.selected_provider(),
                    "error_code": error_code,
                    "pure": stage.spec.pure,
                },
            )
        except Exception:
            stage.provider_index = previous_index
            stage.retry_count = previous_retry_count
            stage.last_retry_error_code = previous_retry_error
            raise

    def _retire_stage_attempt_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
        *,
        attempt_state: str,
        retry: bool,
        retry_same_provider: bool = False,
        error_code: str,
        error: str,
        now: Optional[float] = None,
    ) -> None:
        if (
            retry
            and retry_same_provider
            and not self._can_retry_same_provider(stage, attempt)
        ):
            raise TaskGraphError("Stage is not eligible for same-Provider retry")
        stage_next_state = "ready" if retry else "failed"
        self._require_transition(
            "stage",
            stage.spec.stage_id,
            stage.state,
            stage_next_state,
            STAGE_STATE_TRANSITIONS,
        )
        self._require_transition(
            "attempt",
            attempt.attempt_id,
            attempt.state,
            attempt_state,
            ATTEMPT_STATE_TRANSITIONS,
        )
        previous_stage = (
            stage.state,
            stage.finished_at,
            stage.error,
            stage.provider_index,
            stage.retry_count,
            stage.same_provider_retry_count,
            stage.last_retry_error_code,
        )
        previous_attempt = (
            attempt.state,
            attempt.finished_at,
            attempt.error,
        )
        changed_at = time.time() if now is None else now
        previous_provider = stage.selected_provider()
        stage.state = stage_next_state
        stage.finished_at = None if retry else changed_at
        stage.error = "" if retry else error
        attempt.state = attempt_state
        attempt.finished_at = changed_at
        attempt.error = error
        if retry:
            if retry_same_provider:
                stage.same_provider_retry_count += 1
            else:
                stage.provider_index += 1
            stage.retry_count += 1
            stage.last_retry_error_code = error_code
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage_attempt",
                entity_id=f"{stage.spec.stage_id}:{attempt.attempt_id}",
                event_type=(
                    "stage_attempt_retry_scheduled"
                    if retry else "stage_attempt_failed"
                ),
                occurred_at=changed_at,
                payload={
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": attempt.attempt_id,
                    "attempt_state": attempt_state,
                    "stage_state": stage_next_state,
                    "lease_epoch": attempt.lease_epoch,
                    "from_provider": previous_provider,
                    "to_provider": (
                        stage.selected_provider() if retry else ""
                    ),
                    "same_provider": bool(retry and retry_same_provider),
                    "error_code": error_code,
                    "retry": retry,
                },
            )
        except Exception:
            (
                stage.state,
                stage.finished_at,
                stage.error,
                stage.provider_index,
                stage.retry_count,
                stage.same_provider_retry_count,
                stage.last_retry_error_code,
            ) = previous_stage
            (
                attempt.state,
                attempt.finished_at,
                attempt.error,
            ) = previous_attempt
            raise

    def _journal_snapshot(self, workflow_id: str) -> Optional[dict]:
        if self._journal is None:
            return None
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        try:
            snapshot = self._journal.get_snapshot(workflow_id)
            return self._decorate_persisted_snapshot(snapshot)
        except TaskJournalError as exc:
            self._journal_error = f"task journal unavailable: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc

    def _journal_snapshots(self, limit: int) -> list[dict]:
        if self._journal is None:
            return []
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        try:
            return [
                cast(dict, self._decorate_persisted_snapshot(snapshot))
                for snapshot in self._journal.list_snapshots(limit=limit)
            ]
        except TaskJournalError as exc:
            self._journal_error = f"task journal unavailable: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc

    @staticmethod
    def _decorate_persisted_snapshot(
        snapshot: Optional[dict],
    ) -> Optional[dict]:
        if snapshot is None:
            return None
        decorated = dict(snapshot)
        if decorated.get("state") not in TERMINAL_WORKFLOW_STATES:
            recovery_action = str(decorated.get("recovery_action", ""))
            recovery_applied = bool(decorated.get("recovery_applied", False))
            decorated["recovery_pending"] = bool(
                not recovery_applied or recovery_action == "retry"
            )
            decorated["runtime_status"] = (
                "recovery_retry_pending"
                if recovery_action == "retry" and recovery_applied
                else (
                    "recovery_continued"
                    if recovery_action == "continue" and recovery_applied
                    else "persisted_unrecovered"
                )
            )
        else:
            decorated["recovery_pending"] = False
            decorated["runtime_status"] = "terminal"
        return decorated

    @property
    def journal(self) -> Optional[TaskJournal]:
        """本 coordinator 持有的持久化 journal。

        复制路径（`src/journal_replication_service.py`）用它取快照与事件 —— 只读调用
        ``get_snapshot()`` / ``list_events()``，不写回、不改状态。给它一个**公开**访问器是因为
        `task_journal.py` 有单实例锁（`_acquire_instance_lock`），复制侧不能另开一个 journal。
        """

        return self._journal

    def journal_status(self) -> dict:
        if self._journal is None:
            return {
                "enabled": False,
                "available": not bool(self._journal_error),
                "backend": "memory",
                "error": self._journal_error,
                "last_recovery": dict(self._last_recovery),
                "last_cleanup": dict(self._last_cleanup),
            }
        status = self._journal.health()
        if not status.get("available", False) and not self._journal_error:
            self._journal_error = (
                "task journal unavailable: "
                + str(status.get("error") or "health check failed")
            )
        if self._journal_error:
            status["available"] = False
            status["error"] = self._journal_error
        status["last_recovery"] = dict(self._last_recovery)
        status["last_cleanup"] = dict(self._last_cleanup)
        return status

    def register_provider(self, provider: ExecutionProvider) -> None:
        self._provider_registry.register(provider)

    def replace_provider(self, provider: ExecutionProvider) -> None:
        """Install a refreshed Provider object when no attempt is using it."""
        self._provider_registry.replace(provider)

    def has_provider(self, provider_id: str) -> bool:
        return self._provider_registry.has_provider(provider_id)

    def provider_ids(self) -> list[str]:
        return self._provider_registry.provider_ids()

    def provider_status(self) -> list[dict]:
        return self._provider_registry.inspect()

    @staticmethod
    def _recovery_event_id(
        workflow_id: str,
        last_sequence: int,
        action: str = "restart_recovery_v1",
    ) -> str:
        digest = hashlib.sha256(
            f"{workflow_id}:{last_sequence}:{action}".encode("utf-8")
        ).hexdigest()
        return f"evt_recovery_{digest}"

    @staticmethod
    def _duration_seconds(started_at: Any, finished_at: float) -> float:
        try:
            if started_at is None:
                return 0.0
            return round(max(0.0, finished_at - float(started_at)), 6)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _build_recovered_snapshot(
        cls,
        snapshot: dict,
        recovered_at: float,
        events: Iterable[dict] = (),
    ) -> tuple[dict, dict]:
        workflow_id = str(snapshot.get("workflow_id", ""))
        if not WORKFLOW_ID_PATTERN.fullmatch(workflow_id):
            raise TaskJournalError("invalid workflow_id in recovery snapshot")
        try:
            previous_sequence = int(snapshot["last_sequence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskJournalError(
                f"invalid recovery sequence for {workflow_id}"
            ) from exc
        if previous_sequence <= 0:
            raise TaskJournalError(
                f"invalid recovery sequence for {workflow_id}"
            )
        try:
            decision = decide_recovery(snapshot, events)
        except RecoveryError as exc:
            raise TaskJournalError(
                f"invalid recovery candidate for {workflow_id}: {exc}"
            ) from exc
        if decision.action == "ignore":
            raise TaskJournalError(
                f"recovery candidate already applied: {workflow_id}"
            )
        previous_state = decision.previous_state
        recovery_reason = decision.reason

        recovered = json.loads(json.dumps(snapshot))
        expired_attempts = 0
        failed_stages = 0
        skipped_stages = 0
        cancelled_stages = 0
        stages = recovered.get("stages", [])
        if not isinstance(stages, list):
            raise TaskJournalError(
                f"invalid stage snapshot for {workflow_id}"
            )
        for stage in stages:
            if not isinstance(stage, dict):
                raise TaskJournalError(
                    f"invalid stage snapshot for {workflow_id}"
                )
            attempts = stage.get("attempts", [])
            if not isinstance(attempts, list):
                raise TaskJournalError(
                    f"invalid attempt snapshot for {workflow_id}"
                )
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    raise TaskJournalError(
                        f"invalid attempt snapshot for {workflow_id}"
                    )
                if attempt.get("state") not in TERMINAL_ATTEMPT_STATES:
                    attempt["state"] = "expired"
                    attempt["reservation_active"] = False
                    attempt["finished_at"] = recovered_at
                    attempt["duration_seconds"] = cls._duration_seconds(
                        attempt.get("started_at"), recovered_at,
                    )
                    attempt["error"] = ""
                    attempt["error_present"] = True
                    attempt["recovery_reason"] = recovery_reason
                    attempt["error_code"] = recovery_reason
                    expired_attempts += 1

            stage_state = str(stage.get("state", ""))
            if stage_state in TERMINAL_STAGE_STATES:
                continue
            if decision.action == "retry" and stage.get("stage_id") in decision.retry_stage_ids:
                stage["state"] = "ready"
                stage["finished_at"] = None
                stage["duration_seconds"] = 0.0
                stage["error"] = ""
                stage["error_present"] = False
                stage["recovery_reason"] = recovery_reason
                stage["error_code"] = ""
                continue
            if decision.action == "continue":
                continue
            if decision.action == "cancel":
                stage["state"] = "cancelled"
                cancelled_stages += 1
            elif stage_state in {"blocked", "ready", "created"}:
                stage["state"] = "skipped"
                skipped_stages += 1
            else:
                stage["state"] = "failed"
                failed_stages += 1
            stage["finished_at"] = recovered_at
            stage["duration_seconds"] = cls._duration_seconds(
                stage.get("started_at"), recovered_at,
            )
            stage["error"] = ""
            stage["error_present"] = True
            stage["recovery_reason"] = recovery_reason
            stage["error_code"] = recovery_reason

        if decision.action == "retry" or decision.action == "continue":
            recovered["state"] = previous_state
            recovered["finished_at"] = snapshot.get("finished_at")
        elif decision.action == "cancel":
            recovered["state"] = "cancelled"
            recovered["finished_at"] = recovered_at
        else:
            recovered["state"] = "failed"
            recovered["finished_at"] = recovered_at
        recovered["last_sequence"] = previous_sequence + 1
        recovered["duration_seconds"] = cls._duration_seconds(
            recovered.get("started_at"), recovered_at,
        )
        recovered["error"] = ""
        recovered["error_present"] = True
        recovered["recovery_reason"] = recovery_reason
        recovered["error_code"] = (
            "" if decision.action in {"retry", "continue"}
            else recovery_reason
        )
        recovered["recovered_after_restart"] = True
        recovered["recovered_at"] = recovered_at
        recovered["cancel_requested"] = False
        recovered["recovery_action"] = decision.action
        recovered["recovery_applied"] = True
        recovered["completed_stage_count"] = sum(
            stage.get("state") == "completed" for stage in stages
        )
        recovered["failed_stage_count"] = sum(
            stage.get("state") == "failed" for stage in stages
        )
        recovered["skipped_stage_count"] = sum(
            stage.get("state") == "skipped" for stage in stages
        )
        recovered["partial_result"] = False
        recovered["cancelled_stage_count"] = sum(
            stage.get("state") == "cancelled" for stage in stages
        )
        return recovered, {
            "previous_state": previous_state,
            "recovery_reason": recovery_reason,
            "recovery_action": decision.action,
            "retry_stage_ids": list(decision.retry_stage_ids),
            "expired_attempts": expired_attempts,
            "failed_stages": failed_stages,
            "skipped_stages": skipped_stages,
            "cancelled_stages": cancelled_stages,
        }

    def recover_persisted_workflows(self, batch_size: int = 100) -> dict:
        if self._journal is None:
            return dict(self._last_recovery)
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        safe_batch_size = max(1, min(int(batch_size), 1000))
        summary = {
            "recovered_workflows": 0,
            "expired_attempts": 0,
            "failed_stages": 0,
            "skipped_stages": 0,
        }
        try:
            processed: set[str] = set()
            while True:
                candidates = self._journal.list_nonterminal_snapshots(
                    limit=safe_batch_size,
                )
                candidates = [
                    snapshot for snapshot in candidates
                    if str(snapshot.get("workflow_id", "")) not in processed
                ]
                if not candidates:
                    break
                inserted_count = 0
                for snapshot in candidates:
                    workflow_id = str(snapshot.get("workflow_id", ""))
                    processed.add(workflow_id)
                    if bool(snapshot.get("recovery_applied", False)):
                        continue
                    recovered_at = time.time()
                    events = self._journal.list_events(workflow_id)
                    recovered, details = self._build_recovered_snapshot(
                        snapshot, recovered_at, events,
                    )
                    previous_sequence = int(snapshot["last_sequence"])
                    action = str(details["recovery_action"])
                    event_type = (
                        "workflow_recovered_after_restart"
                        if action == "fail"
                        else f"workflow_recovery_{action}"
                    )
                    event = JournalEvent(
                        event_id=self._recovery_event_id(
                            workflow_id, previous_sequence,
                            "restart_recovery_v1" if action == "fail" else action,
                        ),
                        workflow_id=workflow_id,
                        sequence=previous_sequence + 1,
                        entity_type="workflow",
                        entity_id=workflow_id,
                        event_type=event_type,
                        occurred_at=recovered_at,
                        payload=details,
                    )
                    inserted = self._journal.append_event(event, recovered)
                    if inserted:
                        inserted_count += 1
                        summary["recovered_workflows"] += 1
                        summary["expired_attempts"] += int(
                            details["expired_attempts"]
                        )
                        summary["failed_stages"] += int(
                            details["failed_stages"]
                        )
                        summary["skipped_stages"] += int(
                            details["skipped_stages"]
                        )
                        action_key = {
                            "retry": "retried_workflows",
                            "cancel": "cancelled_workflows",
                            "continue": "continued_workflows",
                            "handoff_timeout": "handoff_timeouts",
                        }.get(action)
                        if action_key:
                            summary.setdefault(action_key, 0)
                            summary[action_key] += 1
                if inserted_count == 0:
                    break
        except TaskJournalError as exc:
            self._journal_error = f"task journal recovery failed: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc
        self._last_recovery = summary
        return dict(summary)

    def cleanup_journal(
        self,
        *,
        max_age_days: int = 0,
        max_records: int = 0,
        now: Optional[float] = None,
    ) -> dict:
        if self._journal is None:
            return dict(self._last_cleanup)
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        try:
            result = self._journal.cleanup_terminal(
                max_age_seconds=max(0, int(max_age_days)) * 86400.0,
                max_records=max(0, int(max_records)),
                now=now,
            )
        except TaskJournalError as exc:
            self._journal_error = f"task journal cleanup failed: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc
        self._last_cleanup = dict(result)
        return dict(result)

    @property
    def availability_error(self) -> str:
        return self._journal_error

    def _finish_active_run(self, workflow_id: str) -> None:
        with self._run_condition:
            self._active_runs.pop(workflow_id, None)
            self._run_condition.notify_all()

    def close(self) -> None:
        with self._lock:
            self._closing = True
            workflows = list(self._workflows.values())
        for workflow in workflows:
            with workflow.lock:
                if workflow.state not in TERMINAL_WORKFLOW_STATES:
                    try:
                        self._request_cancel_locked(
                            workflow,
                            "coordinator closed before result commit",
                        )
                    except Exception:
                        workflow.cancel_event.set()
                        self._cancel_active_provider_attempts_locked(workflow)
        with self._executor_lock:
            executor = self._stage_executor
            self._stage_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with self._run_condition:
            while self._active_runs:
                self._run_condition.wait()
        try:
            self._provider_registry.close()
        finally:
            if self._journal is not None:
                self._journal.close()

    def _cancel_active_provider_attempts_locked(
        self,
        workflow: WorkflowRecord,
    ) -> None:
        with self._provider_activity_lock:
            active_attempts = [
                (
                    attempt.attempt_id,
                    self._active_provider_attempts.get(attempt.attempt_id),
                )
                for stage in workflow.stages.values()
                for attempt in stage.attempts
                if attempt.state not in TERMINAL_ATTEMPT_STATES
            ]
        for attempt_id, active in active_attempts:
            if active is None:
                continue
            registry, provider_id = active
            self._cancel_provider_attempt(
                registry, provider_id, attempt_id,
            )

    @staticmethod
    def _cancel_provider_attempt(
        registry: ProviderRegistry,
        provider_id: str,
        attempt_id: str,
    ) -> None:
        try:
            registry.cancel(provider_id, attempt_id)
        except Exception:
            pass

    def _request_cancel_locked(
        self,
        workflow: WorkflowRecord,
        reason: str = "cancelled",
        *,
        terminal_reason_code: str = REASON_GENERATION_CANCELLED,
        terminal_source: str = "cancel_request",
    ) -> None:
        if workflow.state in TERMINAL_WORKFLOW_STATES:
            return
        terminal_reason = self._observe_workflow_stop(workflow)
        if terminal_reason:
            workflow.terminal_outcome.decide(
                terminal_reason_code,
                source=terminal_source,
            )
        else:
            terminal_reason = workflow.terminal_outcome.decide(
                terminal_reason_code,
                source=terminal_source,
            )
        was_set = workflow.cancel_event.is_set()
        workflow.cancel_event.set()
        try:
            if workflow.state == "result_ready":
                self._transition_workflow_locked(
                    workflow,
                    (
                        "cancelled"
                        if terminal_reason == REASON_GENERATION_CANCELLED
                        else "failed"
                    ),
                    error=(
                        reason
                        if terminal_reason == REASON_GENERATION_CANCELLED
                        else terminal_reason
                    ),
                )
                workflow.cancel_recorded = True
            elif not workflow.cancel_recorded:
                self._record_event_locked(
                    workflow,
                    entity_type="workflow",
                    entity_id=workflow.workflow_id,
                    event_type="workflow_cancel_requested",
                    payload={"state": workflow.state},
                )
                workflow.cancel_recorded = True
        except Exception:
            if not was_set:
                workflow.cancel_event.clear()
            raise
        self._cancel_active_provider_attempts_locked(workflow)

    @staticmethod
    def validate(stages: Iterable[StageSpec], final_stage_id: str) -> list[StageSpec]:
        specs = list(stages)
        if not specs:
            raise TaskGraphError("task graph must contain at least one stage")
        ids = [stage.stage_id for stage in specs]
        if any(not stage_id for stage_id in ids):
            raise TaskGraphError("stage_id must not be empty")
        if DEPENDENCY_FAILURES_KEY in ids:
            raise TaskGraphError("stage_id uses a reserved task-graph key")
        if any(
            not isinstance(provider_id, str)
            or not PROVIDER_ID_PATTERN.fullmatch(provider_id)
            for spec in specs
            for provider_id in (spec.provider, *spec.fallback_providers)
        ):
            raise TaskGraphError("stage provider_id must be non-empty and safe")
        for spec in specs:
            if (
                spec.model_identity is not None
                and not isinstance(spec.model_identity, ModelIdentity)
            ):
                raise TaskGraphError(
                    f"stage {spec.stage_id} model_identity must be a ModelIdentity"
                )
            candidates = (spec.provider, *spec.fallback_providers)
            if len(candidates) != len(set(candidates)):
                raise TaskGraphError(
                    f"stage {spec.stage_id} Provider candidates must be unique"
                )
            if len(spec.fallback_providers) > 4:
                raise TaskGraphError(
                    f"stage {spec.stage_id} has too many fallback Providers"
                )
            if not isinstance(spec.pure, bool):
                raise TaskGraphError(
                    f"stage {spec.stage_id} pure must be a bool"
                )
            if not isinstance(spec.retry_safe, bool):
                raise TaskGraphError(
                    f"stage {spec.stage_id} retry_safe must be a bool"
                )
            if not isinstance(spec.root_input_overrides, dict) or any(
                not isinstance(key, str) or not key
                for key in spec.root_input_overrides
            ):
                raise TaskGraphError(
                    f"stage {spec.stage_id} root input overrides must use non-empty string keys"
                )
            if not isinstance(spec.input_bindings, dict):
                raise TaskGraphError(
                    f"stage {spec.stage_id} input bindings must be an object"
                )
            if not isinstance(spec.stage_fields, dict):
                raise TaskGraphError(
                    f"stage {spec.stage_id} stage fields must be an object"
                )
            if spec.stage_type != "layer_forward" and spec.stage_fields:
                raise TaskGraphError(
                    f"stage {spec.stage_id} stage fields are only valid for layer_forward"
                )
            for target_key, source in spec.input_bindings.items():
                if (
                    not isinstance(target_key, str)
                    or not target_key
                    or not isinstance(source, tuple)
                    or len(source) != 2
                    or any(not isinstance(value, str) or not value for value in source)
                ):
                    raise TaskGraphError(
                        f"stage {spec.stage_id} input bindings are invalid"
                    )
                if source[0] not in spec.depends_on:
                    raise TaskGraphError(
                        f"stage {spec.stage_id} input binding source must be a dependency"
                    )
            dependency_count = len(spec.depends_on)
            minimum_successful = spec.minimum_successful_dependencies
            if minimum_successful is None:
                minimum_successful = dependency_count
            if (
                isinstance(minimum_successful, bool)
                or not isinstance(minimum_successful, int)
                or minimum_successful < 0
                or minimum_successful > dependency_count
                or (dependency_count > 0 and minimum_successful == 0)
            ):
                valid_range = (
                    "0" if dependency_count == 0
                    else f"[1, {dependency_count}]"
                )
                raise TaskGraphError(
                    f"stage {spec.stage_id} minimum successful dependencies "
                    f"must be {valid_range}"
                )
            same_provider_retries = spec.max_same_provider_retries
            if (
                isinstance(same_provider_retries, bool)
                or not isinstance(same_provider_retries, int)
                or same_provider_retries < 0
                or same_provider_retries > 3
            ):
                raise TaskGraphError(
                    f"stage {spec.stage_id} same-Provider retries must be "
                    "an integer in [0, 3]"
                )
            if same_provider_retries > 0 and not spec.retry_safe:
                raise TaskGraphError(
                    f"stage {spec.stage_id} same-Provider retries require "
                    "retry_safe=true"
                )
            try:
                accept_timeout = float(spec.accept_timeout_seconds)
            except (TypeError, ValueError) as exc:
                raise TaskGraphError(
                    f"stage {spec.stage_id} accept timeout must be numeric"
                ) from exc
            if (
                not accept_timeout > 0
                or not accept_timeout < float("inf")
                or accept_timeout > 300.0
            ):
                raise TaskGraphError(
                    f"stage {spec.stage_id} accept timeout must be in (0, 300]"
                )
            try:
                lease_timeout = float(spec.lease_timeout_seconds)
            except (TypeError, ValueError) as exc:
                raise TaskGraphError(
                    f"stage {spec.stage_id} lease timeout must be numeric"
                ) from exc
            if (
                not lease_timeout > 0
                or not lease_timeout < float("inf")
                or lease_timeout > 3600.0
            ):
                raise TaskGraphError(
                    f"stage {spec.stage_id} lease timeout must be in (0, 3600]"
                )
        if len(ids) != len(set(ids)):
            raise TaskGraphError("stage_id must be unique")
        known = set(ids)
        if final_stage_id not in known:
            raise TaskGraphError("final stage does not exist")
        for stage in specs:
            if stage.stage_id in stage.depends_on:
                raise TaskGraphError(f"stage {stage.stage_id} depends on itself")
            missing = set(stage.depends_on) - known
            if missing:
                raise TaskGraphError(
                    f"stage {stage.stage_id} has missing dependencies: "
                    f"{sorted(missing)}"
                )

        indegree = {stage.stage_id: len(stage.depends_on) for stage in specs}
        children: dict[str, list[str]] = {stage_id: [] for stage_id in ids}
        for stage in specs:
            for dependency in stage.depends_on:
                children[dependency].append(stage.stage_id)
        ready = [stage_id for stage_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            stage_id = ready.pop()
            visited += 1
            for child in children[stage_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(specs):
            raise TaskGraphError("task graph must be acyclic")
        return specs

    def _get_stage_executor(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._closing:
                raise TaskGraphUnavailable(
                    "task graph coordinator is closing"
                )
            if self._stage_executor is None:
                self._stage_executor = ThreadPoolExecutor(
                    max_workers=self._max_parallel_stages,
                    thread_name_prefix="task-graph-stage",
                )
            return self._stage_executor

    def _select_ready_batch(
        self,
        ready: list[StageRecord],
        provider_registry: ProviderRegistry,
    ) -> list[StageRecord]:
        statuses = {
            str(item.get("provider_id", "")): item
            for item in provider_registry.inspect()
        }
        remaining = {
            provider_id: max(
                0,
                int(status.get("max_concurrency", 0))
                - int(status.get("active_reservations", 0)),
            )
            for provider_id, status in statuses.items()
        }
        scheduled_by_provider: dict[str, int] = {}
        batch: list[StageRecord] = []
        for stage in ready:
            if len(batch) >= self._max_parallel_stages:
                break
            provider_id = stage.selected_provider()
            status = statuses.get(provider_id)
            invalid_provider = (
                status is None
                or not bool(status.get("healthy", False))
                or stage.spec.stage_type not in status.get(
                    "supported_stage_types", [],
                )
            )
            if invalid_provider:
                batch.append(stage)
                scheduled_by_provider[provider_id] = (
                    scheduled_by_provider.get(provider_id, 0) + 1
                )
                continue
            if remaining.get(provider_id, 0) > 0:
                batch.append(stage)
                remaining[provider_id] -= 1
                scheduled_by_provider[provider_id] = (
                    scheduled_by_provider.get(provider_id, 0) + 1
                )
                continue
            if scheduled_by_provider.get(provider_id, 0) == 0:
                # No in-batch attempt owns the slot. Run one Stage so reserve()
                # can report the external busy/unavailable condition precisely.
                batch.append(stage)
                scheduled_by_provider[provider_id] = 1
        return batch

    def _run_ready_batch(
        self,
        workflow: WorkflowRecord,
        ready: list[StageRecord],
        root_input: dict,
        provider_registry: ProviderRegistry,
    ) -> None:
        prepared: list[tuple[StageRecord, StageRequest, Reservation]] = []
        errors: dict[str, Exception] = {}
        for stage in ready:
            try:
                request, reservation = self._prepare_stage(
                    workflow,
                    stage,
                    root_input,
                    provider_registry,
                )
                prepared.append((stage, request, reservation))
            except _StageRetryScheduled:
                continue
            except Exception as exc:
                errors[stage.spec.stage_id] = exc
                if isinstance(
                    exc,
                    (TaskGraphUnavailable, WorkflowCancelled),
                ):
                    workflow.cancel_event.set()
                    break

        if any(
            isinstance(error, (TaskGraphUnavailable, WorkflowCancelled))
            for error in errors.values()
        ):
            cleanup_errors = []
            for _stage, _request, reservation in prepared:
                try:
                    provider_registry.release(reservation.reservation_id)
                except Exception as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            try:
                self._raise_ready_batch_error(workflow, ready, errors)
            except Exception as original:
                for cleanup_exc in cleanup_errors:
                    original.add_note(
                        f"reservation cleanup also failed: {cleanup_exc}"
                    )
                raise

        if len(prepared) == 1:
            stage, request, reservation = prepared[0]
            try:
                self._run_stage(
                    workflow,
                    stage,
                    provider_registry,
                    request,
                    reservation,
                )
            except Exception as exc:
                errors[stage.spec.stage_id] = exc
            self._raise_ready_batch_error(workflow, ready, errors)
            return

        try:
            executor = self._get_stage_executor() if prepared else None
        except Exception as exc:
            cleanup_errors = []
            for _stage, _request, reservation in prepared:
                try:
                    provider_registry.release(reservation.reservation_id)
                except Exception as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            for cleanup_exc in cleanup_errors:
                exc.add_note(
                    f"reservation cleanup also failed: {cleanup_exc}"
                )
            if workflow.cancel_event.is_set():
                raise WorkflowCancelled(workflow.workflow_id) from exc
            raise
        futures: dict[Future[None], StageRecord] = {}
        submitted_reservations: set[str] = set()
        for stage, request, reservation in prepared:
            try:
                future = cast(ThreadPoolExecutor, executor).submit(
                    self._run_stage,
                    workflow,
                    stage,
                    provider_registry,
                    request,
                    reservation,
                )
                futures[future] = stage
                submitted_reservations.add(reservation.reservation_id)
            except Exception as exc:
                errors[stage.spec.stage_id] = exc
                break

        for prepared_stage, _request, reservation in prepared:
            if reservation.reservation_id not in submitted_reservations:
                try:
                    provider_registry.release(reservation.reservation_id)
                except Exception as cleanup_exc:
                    errors.setdefault(
                        prepared_stage.spec.stage_id, cleanup_exc,
                    )

        for future in as_completed(futures):
            stage = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors[stage.spec.stage_id] = exc
                if isinstance(
                    exc,
                    (TaskGraphUnavailable, WorkflowCancelled),
                ):
                    workflow.cancel_event.set()

        self._raise_ready_batch_error(workflow, ready, errors)

    def _raise_ready_batch_error(
        self,
        workflow: WorkflowRecord,
        ready: list[StageRecord],
        errors: dict[str, Exception],
    ) -> None:
        if not errors:
            return

        unavailable = next(
            (
                error for error in errors.values()
                if isinstance(error, TaskGraphUnavailable)
            ),
            None,
        )
        if unavailable is not None:
            raise unavailable
        cancelled = next(
            (
                error for error in errors.values()
                if isinstance(error, WorkflowCancelled)
            ),
            None,
        )
        if cancelled is not None:
            raise cancelled
        for stage in ready:
            error = errors.get(stage.spec.stage_id)
            if (
                error is not None
                and not self._can_continue_after_stage_failure(
                    workflow, stage.spec.stage_id,
                )
            ):
                raise error

    @staticmethod
    def _can_continue_after_stage_failure(
        workflow: WorkflowRecord,
        failed_stage_id: str,
    ) -> bool:
        """Return true when the failed Stage does not make the final path impossible."""
        with workflow.lock:
            failed_stage = workflow.stages.get(failed_stage_id)
            if failed_stage is None or failed_stage.state != "failed":
                return False

            final_dependencies: set[str] = set()
            pending = [workflow.final_stage_id]
            while pending:
                stage_id = pending.pop()
                if stage_id in final_dependencies:
                    continue
                final_dependencies.add(stage_id)
                pending.extend(workflow.stages[stage_id].spec.depends_on)
            if failed_stage_id not in final_dependencies:
                return True

            possible: dict[str, bool] = {}

            def can_complete(stage_id: str) -> bool:
                cached = possible.get(stage_id)
                if cached is not None:
                    return cached
                candidate = workflow.stages[stage_id]
                if candidate.state == "completed":
                    result = True
                elif candidate.state in TERMINAL_STAGE_STATES:
                    result = False
                else:
                    result = sum(
                        can_complete(dependency_id)
                        for dependency_id in candidate.spec.depends_on
                    ) >= candidate.minimum_successful_dependencies()
                possible[stage_id] = result
                return result

            return can_complete(workflow.final_stage_id)

    def run_template(
        self,
        template: str,
        root_input: dict,
        execute_stage: Optional[StageExecutor] = None,
        request_id: str = "",
        session_id: str = "",
        model_identity: Optional[ModelIdentity] = None,
        runtime_context: Optional[dict] = None,
        workflow_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        request_deadline: Optional[RequestDeadline] = None,
    ) -> tuple[dict, dict]:
        if template != "dual_candidate":
            raise TaskGraphError(f"unsupported task graph template: {template}")
        stages, final_stage_id = dual_candidate_template()
        return self.run(
            stages=stages,
            final_stage_id=final_stage_id,
            root_input=root_input,
            execute_stage=execute_stage,
            request_id=request_id,
            session_id=session_id,
            model_identity=model_identity,
            runtime_context=runtime_context,
            template=template,
            workflow_id=workflow_id,
            cancel_event=cancel_event,
            request_deadline=request_deadline,
        )

    def run(
        self,
        stages: Iterable[StageSpec],
        final_stage_id: str,
        root_input: dict,
        execute_stage: Optional[StageExecutor] = None,
        request_id: str = "",
        session_id: str = "",
        model_identity: Optional[ModelIdentity] = None,
        runtime_context: Optional[dict] = None,
        template: str = "custom",
        workflow_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        request_deadline: Optional[RequestDeadline] = None,
    ) -> tuple[dict, dict]:
        specs = self.validate(stages, final_stage_id)
        resolved_workflow_id = workflow_id or f"wf_{uuid.uuid4().hex}"
        if not WORKFLOW_ID_PATTERN.fullmatch(resolved_workflow_id):
            raise TaskGraphError(
                "workflow_id must start with wf_ and contain 8-96 safe characters"
            )
        resolved_session_id = str(session_id or "")
        if (
            resolved_session_id
            and SESSION_ID_PATTERN.fullmatch(resolved_session_id) is None
        ):
            raise TaskGraphError("session_id contains unsupported characters")
        if model_identity is not None and not isinstance(
            model_identity, ModelIdentity,
        ):
            raise TaskGraphError("model_identity must be a ModelIdentity")
        if request_deadline is not None and not isinstance(
            request_deadline, RequestDeadline,
        ):
            raise TaskGraphError(
                "request_deadline must be a RequestDeadline"
            )
        workflow = WorkflowRecord(
            workflow_id=resolved_workflow_id,
            request_id=request_id,
            template=template,
            final_stage_id=final_stage_id,
            stages={spec.stage_id: StageRecord(spec=spec) for spec in specs},
            session_id=resolved_session_id,
            model_identity=model_identity,
            runtime_context=dict(runtime_context or {}),
            cancel_event=cancel_event or threading.Event(),
            request_deadline=request_deadline,
        )
        with self._lock:
            if self._closing:
                raise TaskGraphUnavailable(
                    "task graph coordinator is closing"
                )
            if self._journal_error:
                raise TaskGraphUnavailable(self._journal_error)
            self._prune_locked()
            if resolved_workflow_id in self._workflows:
                raise TaskGraphError(
                    f"workflow_id already exists: {resolved_workflow_id}"
                )
            if self._journal_snapshot(resolved_workflow_id) is not None:
                raise TaskGraphError(
                    f"workflow_id already exists: {resolved_workflow_id}"
                )
            if self._pending_cancellations.pop(resolved_workflow_id, None) is not None:
                workflow.cancel_event.set()
            self._workflows[resolved_workflow_id] = workflow
            with self._run_condition:
                self._active_runs[resolved_workflow_id] = threading.get_ident()

        try:
            with workflow.lock:
                self._transition_workflow_locked(workflow, "running")
        except TaskGraphUnavailable:
            with self._lock:
                if self._workflows.get(resolved_workflow_id) is workflow:
                    self._workflows.pop(resolved_workflow_id, None)
            self._finish_active_run(resolved_workflow_id)
            raise
        except Exception:
            self._finish_active_run(resolved_workflow_id)
            raise
        execution_registry = self._provider_registry
        owns_execution_registry = False
        if execute_stage is not None:
            execution_registry = ProviderRegistry()
            owns_execution_registry = True
            try:
                specs_by_id = {spec.stage_id: spec for spec in specs}

                def execute_callback(
                    request: StageRequest,
                    provider_cancel_event: threading.Event,
                ) -> dict:
                    return execute_stage(
                        specs_by_id[request.stage_id],
                        request.dependencies,
                        request.root_input,
                        provider_cancel_event,
                    )

                stage_types_by_provider: dict[str, list[str]] = {}
                for spec in specs:
                    for provider_id in (
                        spec.provider, *spec.fallback_providers,
                    ):
                        stage_types_by_provider.setdefault(
                            provider_id, [],
                        ).append(spec.stage_type)
                for provider_id, stage_types in stage_types_by_provider.items():
                    execution_registry.register(CallbackExecutionProvider(
                        provider_id=provider_id,
                        executor=execute_callback,
                        supported_stage_types=tuple(dict.fromkeys(stage_types)),
                    ))
            except Exception as exc:
                try:
                    execution_registry.close()
                except Exception:
                    pass
                try:
                    self._mark_failed(workflow, str(exc))
                finally:
                    self._finish_active_run(resolved_workflow_id)
                raise WorkflowExecutionError(
                    resolved_workflow_id,
                    "provider_setup",
                    str(exc),
                ) from exc
        try:
            while True:
                self._raise_if_cancelled(workflow)
                with workflow.lock:
                    if all(
                        stage.state in TERMINAL_STAGE_STATES
                        for stage in workflow.stages.values()
                    ):
                        break

                    ready: list[StageRecord] = []
                    for stage in workflow.stages.values():
                        if stage.state == "blocked":
                            dependencies = [
                                workflow.stages[dependency]
                                for dependency in stage.spec.depends_on
                            ]
                            dependency_terminal = all(
                                dependency.state in TERMINAL_STAGE_STATES
                                for dependency in dependencies
                            )
                            successful_dependencies = sum(
                                dependency.state == "completed"
                                for dependency in dependencies
                            )
                            required_successes = (
                                stage.minimum_successful_dependencies()
                            )
                            if (
                                dependency_terminal
                                and successful_dependencies < required_successes
                            ):
                                self._transition_stage_locked(
                                    workflow,
                                    stage,
                                    "skipped",
                                    error="insufficient_successful_dependencies",
                                )
                            elif (
                                dependency_terminal
                                and successful_dependencies >= required_successes
                            ):
                                self._transition_stage_locked(
                                    workflow, stage, "ready",
                                )
                        if stage.state == "ready":
                            ready.append(stage)

                if not ready:
                    with workflow.lock:
                        unfinished = [
                            stage for stage in workflow.stages.values()
                            if stage.state not in TERMINAL_STAGE_STATES
                        ]
                    if unfinished:
                        raise TaskGraphError("task graph made no progress")
                    break

                batch = self._select_ready_batch(ready, execution_registry)
                if not batch:
                    raise TaskGraphError(
                        "no ready Stage has schedulable Provider capacity"
                    )
                self._run_ready_batch(
                    workflow,
                    batch,
                    root_input,
                    execution_registry,
                )

            with workflow.lock:
                stop_reason = self._observe_workflow_stop(workflow)
                if stop_reason == REASON_GENERATION_CANCELLED:
                    raise WorkflowCancelled(workflow.workflow_id)
                if stop_reason == REQUEST_DEADLINE_EXCEEDED:
                    raise WorkflowDeadlineExceeded(workflow.workflow_id)
                final_stage = workflow.stages[final_stage_id]
                if final_stage.state != "completed" or final_stage.output is None:
                    raise TaskGraphError("final stage did not complete")
                self._transition_workflow_locked(workflow, "result_ready")
            return final_stage.output, workflow.snapshot()
        except TaskGraphUnavailable:
            raise
        except WorkflowCancelled:
            self._mark_cancelled(workflow)
            raise
        except WorkflowDeadlineExceeded as exc:
            workflow.terminal_outcome.decide(
                REQUEST_DEADLINE_EXCEEDED,
                source="task_graph",
            )
            self._mark_failed(workflow, str(exc))
            raise
        except WorkflowExecutionError as exc:
            workflow.terminal_outcome.decide(
                exc.code,
                source="task_graph",
            )
            self._mark_failed(workflow, str(exc))
            raise
        except Exception as exc:
            stop_reason = self._observe_workflow_stop(workflow)
            if stop_reason == REASON_GENERATION_CANCELLED:
                self._mark_cancelled(workflow)
                raise WorkflowCancelled(workflow.workflow_id) from exc
            if stop_reason == REQUEST_DEADLINE_EXCEEDED:
                self._mark_failed(workflow, str(exc))
                raise WorkflowDeadlineExceeded(
                    workflow.workflow_id,
                ) from exc
            workflow.terminal_outcome.decide(
                getattr(exc, "code", "task_graph_execution_failed"),
                source="task_graph",
            )
            self._mark_failed(workflow, str(exc))
            raise WorkflowExecutionError(
                resolved_workflow_id, "graph", str(exc),
            ) from exc
        finally:
            try:
                if owns_execution_registry:
                    execution_registry.close()
            finally:
                self._finish_active_run(resolved_workflow_id)

    def _prepare_stage(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        root_input: dict,
        provider_registry: ProviderRegistry,
    ) -> tuple[StageRequest, Reservation]:
        self._raise_if_cancelled(workflow)
        with workflow.lock:
            dependencies: dict[str, dict] = {}
            for dependency in stage.spec.depends_on:
                dependency_stage = workflow.stages[dependency]
                if (
                    dependency_stage.state == "completed"
                    and dependency_stage.output is not None
                ):
                    dependencies[dependency] = cast(
                        dict, dependency_stage.output,
                    )
            failed_dependencies = {}
            for dependency in stage.spec.depends_on:
                dependency_stage = workflow.stages[dependency]
                if dependency_stage.state == "completed":
                    continue
                candidate_code = str(
                    dependency_stage.error.partition(":")[0]
                    or dependency_stage.last_retry_error_code
                    or "stage_failed"
                )
                error_code = (
                    candidate_code
                    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", candidate_code)
                    else "stage_failed"
                )
                failed_dependencies[dependency] = {
                    "state": dependency_stage.state,
                    "error_code": error_code,
                }
            if failed_dependencies:
                dependencies[DEPENDENCY_FAILURES_KEY] = failed_dependencies
            selected_provider = stage.selected_provider()
        stage_root_input = dict(root_input)
        replacement = stage.spec.root_input_overrides.get("__replace__")
        if replacement is not None:
            if not isinstance(replacement, dict):
                raise TaskGraphError(
                    f"stage {stage.spec.stage_id} root input replacement must be an object"
                )
            stage_root_input = dict(replacement)
        else:
            stage_root_input.update(stage.spec.root_input_overrides)
        for target_key, (dependency_stage_id, output_key) in (
            stage.spec.input_bindings.items()
        ):
            dependency_output = dependencies.get(dependency_stage_id)
            if not isinstance(dependency_output, dict) or output_key not in dependency_output:
                raise TaskGraphError(
                    f"stage {stage.spec.stage_id} input binding source output is unavailable"
                )
            value = dependency_output[output_key]
            stage_root_input[target_key] = value
            dependencies.pop(dependency_stage_id, None)
        provider_request = StageRequest(
            workflow_id=workflow.workflow_id,
            request_id=workflow.request_id,
            stage_id=stage.spec.stage_id,
            stage_type=stage.spec.stage_type,
            provider_id=selected_provider,
            dependencies=dependencies,
            root_input=stage_root_input,
            model_identity=workflow.effective_model_identity(stage.spec),
            runtime_context=workflow.runtime_context,
            stage_fields=dict(stage.spec.stage_fields),
        )
        accept_timeout = float(stage.spec.accept_timeout_seconds)
        if workflow.request_deadline is not None:
            try:
                accept_timeout = min(
                    accept_timeout,
                    workflow.request_deadline.require_remaining(),
                )
            except Exception as exc:
                raise WorkflowDeadlineExceeded(
                    workflow.workflow_id,
                ) from exc
        try:
            reservation = provider_registry.reserve(
                provider_request,
                timeout_seconds=accept_timeout,
                cancel_event=workflow.cancel_event,
            )
        except ProviderError as exc:
            if (
                workflow.request_deadline is not None
                and workflow.request_deadline.expired()
            ):
                raise WorkflowDeadlineExceeded(
                    workflow.workflow_id,
                ) from exc
            with workflow.lock:
                if workflow.cancel_event.is_set():
                    raise WorkflowCancelled(workflow.workflow_id) from exc
                if exc.retryable and self._can_retry_stage(stage):
                    self._advance_ready_stage_provider_locked(
                        workflow,
                        stage,
                        error_code=exc.code,
                    )
                    raise _StageRetryScheduled(
                        f"{stage.spec.stage_id} retry scheduled"
                    ) from exc
                self._transition_stage_locked(
                    workflow,
                    stage,
                    "failed",
                    error=f"{exc.code}: {exc}",
                )
            raise WorkflowExecutionError(
                workflow.workflow_id,
                stage.spec.stage_id,
                f"{exc.code}: {exc}",
            ) from exc
        with workflow.lock:
            if workflow.cancel_event.is_set():
                try:
                    provider_registry.release(reservation.reservation_id)
                finally:
                    raise WorkflowCancelled(workflow.workflow_id)
        return provider_request, reservation

    def _execute_provider_attempt(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
        provider_attempt: ProviderStageAttempt,
        provider_registry: ProviderRegistry,
        reservation: Reservation,
        lease_timeout_seconds: float,
    ) -> ProviderStageResult:
        if (
            attempt.lease_expires_at <= 0
            and workflow.request_deadline is None
        ):
            return provider_registry.execute(
                provider_attempt, reservation, workflow.cancel_event,
            )

        done = threading.Event()
        outcome: dict = {}

        def execute_provider() -> None:
            try:
                outcome["result"] = provider_registry.execute(
                    provider_attempt, reservation, workflow.cancel_event,
                )
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                outcome["received_at"] = time.time()
                done.set()

        threading.Thread(
            target=execute_provider,
            name=f"provider-execute-{attempt.attempt_id}",
            daemon=True,
        ).start()
        lease_duration = max(
            0.001, min(float(lease_timeout_seconds), 3599.0)
        )
        renew_window = max(0.05, min(30.0, lease_duration / 3.0))
        renewal_supported = True
        while True:
            if done.is_set():
                error = outcome.get("error")
                if isinstance(error, BaseException):
                    raise error
                result = outcome.get("result")
                if not isinstance(result, ProviderStageResult):
                    raise ProviderExecutionError(
                        "provider execution returned no result",
                        code="invalid_provider_result",
                        provider_id=attempt.provider,
                    )
                return result
            stop_reason = self._observe_workflow_stop(workflow)
            if stop_reason == REASON_GENERATION_CANCELLED:
                self._cancel_provider_attempt(
                    provider_registry, attempt.provider, attempt.attempt_id,
                )
                raise WorkflowCancelled(workflow.workflow_id)
            if stop_reason == REQUEST_DEADLINE_EXCEEDED:
                self._cancel_provider_attempt(
                    provider_registry, attempt.provider, attempt.attempt_id,
                )
                raise ProviderExecutionError(
                    "request deadline exceeded during provider execution",
                    code=REQUEST_DEADLINE_EXCEEDED,
                    provider_id=attempt.provider,
                )
            with workflow.lock:
                deadline = attempt.lease_deadline_monotonic
                lease_expires_at = attempt.lease_expires_at
            now = time.monotonic()
            remaining = (
                deadline - now
                if deadline > 0
                else workflow.request_deadline.require_remaining()
            )
            if remaining <= 0:
                self._cancel_provider_attempt(
                    provider_registry, attempt.provider, attempt.attempt_id,
                )
                raise ProviderExecutionError(
                    "provider execution lease expired",
                    code="lease_expired",
                    provider_id=attempt.provider,
                    retryable=True,
                )
            if (
                deadline > 0
                and renewal_supported
                and remaining <= renew_window
            ):
                renewed_deadline = time.time() + lease_duration
                if renewed_deadline <= lease_expires_at + 0.001:
                    renewal_supported = False
                    continue
                try:
                    renewal_supported = provider_registry.renew_lease(
                        attempt.provider,
                        attempt.attempt_id,
                        attempt.lease_id,
                        attempt.lease_epoch,
                        renewed_deadline,
                    )
                    if renewal_supported:
                        with workflow.lock:
                            self._renew_stage_lease_locked(
                                workflow,
                                stage,
                                attempt,
                                attempt.lease_id,
                                attempt.lease_epoch,
                                renewed_deadline,
                            )
                        continue
                except Exception:
                    self._cancel_provider_attempt(
                        provider_registry,
                        attempt.provider,
                        attempt.attempt_id,
                    )
                    raise
            if not done.wait(min(0.05, remaining)):
                continue

    def _run_stage(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        provider_registry: ProviderRegistry,
        provider_request: StageRequest,
        reservation: Reservation,
    ) -> None:
        attempt: Optional[AttemptRecord] = None
        provider_attempt: Optional[ProviderStageAttempt] = None
        result: Optional[ProviderStageResult] = None
        result_retained = False
        attempt_started = False
        try:
            with workflow.lock:
                if workflow.cancel_event.is_set():
                    raise WorkflowCancelled(workflow.workflow_id)
                started_at = time.time()
                previous_epoch = stage.lease_epoch
                lease_epoch = previous_epoch + 1
                attempt = AttemptRecord(
                    attempt_id=f"att_{uuid.uuid4().hex}",
                    provider=reservation.provider_id,
                    provider_kind=reservation.provider_kind,
                    provider_node_id=reservation.provider_node_id,
                    reservation_id=reservation.reservation_id,
                    lease_id=f"lease_{uuid.uuid4().hex}",
                    lease_epoch=lease_epoch,
                    lease_expires_at=0.0,
                    started_at=started_at,
                )
                stage.lease_epoch = lease_epoch
                try:
                    self._transition_stage_locked(
                        workflow, stage, "running", now=started_at,
                    )
                except Exception:
                    stage.lease_epoch = previous_epoch
                    raise
                self._append_attempt_locked(workflow, stage, attempt)
                attempt_started = True
                if reservation.provider_kind not in {
                    "local_full_model", "callback_compatibility",
                }:
                    lease_timeout = float(stage.spec.lease_timeout_seconds)
                    attempt.lease_expires_at = time.time() + lease_timeout
                    attempt.lease_deadline_monotonic = (
                        time.monotonic() + lease_timeout
                    )
                provider_attempt = ProviderStageAttempt(
                    attempt_id=attempt.attempt_id,
                    request=provider_request,
                    provider_id=reservation.provider_id,
                    lease_id=attempt.lease_id,
                    lease_epoch=attempt.lease_epoch,
                    lease_expires_at=attempt.lease_expires_at,
                    accept_timeout_seconds=stage.spec.accept_timeout_seconds,
                    request_deadline=workflow.request_deadline,
                )
            with self._provider_activity_lock:
                self._active_provider_attempts[attempt.attempt_id] = (
                    provider_registry,
                    reservation.provider_id,
                )
            result = self._execute_provider_attempt(
                workflow,
                stage,
                attempt,
                cast(ProviderStageAttempt, provider_attempt),
                provider_registry,
                reservation,
                stage.spec.lease_timeout_seconds,
            )
            self._raise_if_cancelled(workflow)
            with workflow.lock:
                submission = self._submit_stage_result_locked(
                    workflow, stage, result,
                )
                if submission["status"] in {"committed", "idempotent"}:
                    result_retained = True
                    return
                if stage.winner_attempt_id:
                    return
                rejection_reason = str(submission["reason"])
                retry = (
                    rejection_reason == "lease_expired"
                    and self._can_retry_stage(stage)
                )
                self._retire_stage_attempt_locked(
                    workflow,
                    stage,
                    attempt,
                    attempt_state=(
                        "expired"
                        if rejection_reason == "lease_expired"
                        else "failed"
                    ),
                    retry=retry,
                    error_code=rejection_reason,
                    error=f"{rejection_reason}: Provider result rejected",
                )
            if retry:
                return
            raise WorkflowExecutionError(
                workflow.workflow_id,
                stage.spec.stage_id,
                f"{rejection_reason}: Provider result rejected",
            )
        except TaskGraphUnavailable:
            raise
        except WorkflowCancelled:
            if attempt_started and attempt is not None:
                with workflow.lock:
                    if (
                        attempt.state == "running"
                        and stage.state == "running"
                    ):
                        self._finish_stage_attempt_locked(
                            workflow,
                            stage,
                            attempt,
                            "cancelled",
                            now=time.time(),
                        )
            raise
        except WorkflowExecutionError:
            raise
        except Exception as exc:
            stop_reason = self._observe_workflow_stop(workflow)
            if (
                stop_reason == REQUEST_DEADLINE_EXCEEDED
                and not (
                    isinstance(exc, ProviderError)
                    and exc.code == REQUEST_DEADLINE_EXCEEDED
                )
            ):
                exc = ProviderExecutionError(
                    "request deadline exceeded during provider execution",
                    code=REQUEST_DEADLINE_EXCEEDED,
                    provider_id=(attempt.provider if attempt is not None else ""),
                )
            if (
                isinstance(exc, ProviderError)
                and exc.code == REQUEST_DEADLINE_EXCEEDED
            ):
                workflow.terminal_outcome.decide(
                    REQUEST_DEADLINE_EXCEEDED,
                    source="provider_execution",
                )
            stop_reason = self._observe_workflow_stop(workflow)
            if stop_reason == REASON_GENERATION_CANCELLED:
                if attempt_started and attempt is not None:
                    with workflow.lock:
                        if (
                            attempt.state == "running"
                            and stage.state == "running"
                        ):
                            self._finish_stage_attempt_locked(
                                workflow,
                                stage,
                                attempt,
                                "cancelled",
                                now=time.time(),
                            )
                raise WorkflowCancelled(workflow.workflow_id) from exc
            if not attempt_started or attempt is None:
                raise
            with workflow.lock:
                if (
                    attempt.state == "completed"
                    and stage.winner_attempt_id == attempt.attempt_id
                ):
                    return
                if attempt.state != "running" or stage.state != "running":
                    raise
                retry_next_provider = (
                    isinstance(exc, ProviderError)
                    and exc.retryable
                    and self._can_retry_stage(stage)
                )
                retry_same_provider = (
                    not retry_next_provider
                    and isinstance(exc, ProviderError)
                    and exc.same_provider_retryable
                    and self._can_retry_same_provider(stage, attempt)
                )
                retry = retry_next_provider or retry_same_provider
                error_code = (
                    exc.code
                    if isinstance(exc, ProviderError)
                    else "stage_execution_failed"
                )
                error = (
                    f"{error_code}: {exc}"
                    if isinstance(exc, ProviderError)
                    else str(exc)
                )
                if error_code == "lease_expired":
                    self._reject_stage_result_locked(
                        workflow,
                        stage,
                        ProviderStageResult(
                            output={},
                            provider_id=attempt.provider,
                            attempt_id=attempt.attempt_id,
                            lease_epoch=attempt.lease_epoch,
                        ),
                        reason="lease_expired",
                        result_digest="",
                        now=time.time(),
                    )
                self._retire_stage_attempt_locked(
                    workflow,
                    stage,
                    attempt,
                    attempt_state=(
                        "expired"
                        if retry_next_provider
                        or error_code in {
                            "lease_expired", REQUEST_DEADLINE_EXCEEDED,
                        }
                        else "failed"
                    ),
                    retry=retry,
                    retry_same_provider=retry_same_provider,
                    error_code=error_code,
                    error=error,
                )
            if retry:
                return
            raise WorkflowExecutionError(
                workflow.workflow_id, stage.spec.stage_id, error,
                code=error_code,
            ) from exc
        finally:
            if attempt is not None:
                with self._provider_activity_lock:
                    self._active_provider_attempts.pop(
                        attempt.attempt_id, None,
                    )
            active_exception = sys.exc_info()[1]
            cleanup_failure: Optional[WorkflowExecutionError] = None
            if result is not None and not result_retained:
                try:
                    provider_registry.discard_result(
                        attempt.provider,
                        attempt.attempt_id,
                        result.output,
                    )
                except Exception as discard_exc:
                    if active_exception is None:
                        cleanup_failure = WorkflowExecutionError(
                            workflow.workflow_id,
                            stage.spec.stage_id,
                            "uncommitted provider result cleanup failed: "
                            f"{discard_exc}",
                        )
                    else:
                        active_exception.add_note(
                            "uncommitted provider result cleanup also failed: "
                            f"{discard_exc}"
                        )
            try:
                provider_registry.release(reservation.reservation_id)
            except Exception as release_exc:
                cleanup_target = active_exception or cleanup_failure
                if cleanup_target is None:
                    raise WorkflowExecutionError(
                        workflow.workflow_id,
                        stage.spec.stage_id,
                        f"provider reservation cleanup failed: {release_exc}",
                    ) from release_exc
                cleanup_target.add_note(
                    "provider reservation cleanup also failed: "
                    f"{release_exc}"
                )
            if cleanup_failure is not None:
                raise cleanup_failure

    @staticmethod
    def _observe_workflow_stop(workflow: WorkflowRecord) -> str:
        """Return the canonical request terminal reason, if any.

        The latch is authoritative once decided. Later cancel/deadline
        observations are retained as secondary diagnostics and may wake active
        work, but they cannot reverse the workflow's terminal class.
        """

        existing = workflow.terminal_outcome.snapshot()["reason_code"]
        cancel_requested = workflow.cancel_event.is_set()
        deadline_expired = bool(
            workflow.request_deadline is not None
            and workflow.request_deadline.expired()
        )
        if existing:
            if cancel_requested:
                workflow.terminal_outcome.decide(
                    REASON_GENERATION_CANCELLED,
                    source="cancel_event",
                )
            if deadline_expired:
                workflow.terminal_outcome.decide(
                    REQUEST_DEADLINE_EXCEEDED,
                    source="request_deadline",
                )
            return str(existing)
        if cancel_requested:
            return workflow.terminal_outcome.decide(
                REASON_GENERATION_CANCELLED,
                source="cancel_event",
            )
        if deadline_expired:
            return workflow.terminal_outcome.decide(
                REQUEST_DEADLINE_EXCEEDED,
                source="request_deadline",
            )
        return ""

    @staticmethod
    def _raise_if_cancelled(workflow: WorkflowRecord) -> None:
        stop_reason = TaskGraphCoordinator._observe_workflow_stop(workflow)
        if stop_reason == REASON_GENERATION_CANCELLED:
            raise WorkflowCancelled(workflow.workflow_id)
        if stop_reason == REQUEST_DEADLINE_EXCEEDED:
            raise WorkflowDeadlineExceeded(workflow.workflow_id)

    def _mark_cancelled(self, workflow: WorkflowRecord) -> None:
        now = time.time()
        with workflow.lock:
            if workflow.state in TERMINAL_WORKFLOW_STATES:
                return
            terminal_reason = workflow.terminal_outcome.decide(
                REASON_GENERATION_CANCELLED,
                source="workflow_cancelled",
            )
            if terminal_reason != REASON_GENERATION_CANCELLED:
                self._mark_failed(workflow, terminal_reason)
                return
            workflow.cancel_event.set()
            for stage in workflow.stages.values():
                for attempt in stage.attempts:
                    if attempt.state not in TERMINAL_ATTEMPT_STATES:
                        self._transition_attempt_locked(
                            workflow,
                            stage,
                            attempt,
                            "cancelled",
                            now=now,
                        )
                if stage.state not in TERMINAL_STAGE_STATES:
                    self._transition_stage_locked(
                        workflow, stage, "cancelled", now=now,
                    )
            self._transition_workflow_locked(
                workflow, "cancelled", error="cancelled", now=now,
            )

    def _mark_failed(
        self, workflow: WorkflowRecord, error: str = "",
    ) -> None:
        now = time.time()
        with workflow.lock:
            if workflow.state in TERMINAL_WORKFLOW_STATES:
                return
            if not workflow.terminal_outcome.reason_code:
                workflow.terminal_outcome.decide(
                    "task_graph_execution_failed",
                    source="workflow_failed",
                )
            for stage in workflow.stages.values():
                for attempt in stage.attempts:
                    if attempt.state not in TERMINAL_ATTEMPT_STATES:
                        self._transition_attempt_locked(
                            workflow,
                            stage,
                            attempt,
                            "failed",
                            now=now,
                            error=error,
                        )
                if stage.state not in TERMINAL_STAGE_STATES:
                    next_state = "failed" if stage.state == "running" else "skipped"
                    self._transition_stage_locked(
                        workflow,
                        stage,
                        next_state,
                        now=now,
                        error=error if next_state == "failed" else None,
                    )
            self._transition_workflow_locked(
                workflow, "failed", error=error, now=now,
            )

    @staticmethod
    def _ensure_terminal_outcome_locked(workflow: WorkflowRecord) -> str:
        """Make every terminal workflow occupy the first-terminal latch."""

        existing = workflow.terminal_outcome.snapshot()["reason_code"]
        if existing:
            return str(existing)
        if workflow.state == "completed":
            reason_code = OUTCOME_COMPLETED
            source = "result_commit"
        elif workflow.state == "cancelled":
            reason_code = REASON_GENERATION_CANCELLED
            source = "workflow_cancelled"
        else:
            reason_code = "task_graph_execution_failed"
            source = "workflow_failed"
        return workflow.terminal_outcome.decide(
            reason_code, source=source,
        )

    def commit_result(self, workflow_id: str) -> dict:
        """Commit a result-ready workflow without allowing terminal reversal."""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                if workflow.state == "completed":
                    self._ensure_terminal_outcome_locked(workflow)
                    return workflow.snapshot()
                if workflow.state == "cancelled":
                    raise WorkflowCancelled(workflow_id)
                if workflow.state != "result_ready":
                    raise TaskGraphError(
                        f"workflow {workflow_id} is not ready to commit: "
                        f"{workflow.state}"
                    )
                if workflow.cancel_event.is_set():
                    stop_reason = self._observe_workflow_stop(workflow)
                    if stop_reason == REASON_GENERATION_CANCELLED:
                        self._mark_cancelled(workflow)
                        raise WorkflowCancelled(workflow_id)
                    self._mark_failed(workflow, stop_reason)
                    raise WorkflowDeadlineExceeded(workflow_id)
                stop_reason = self._observe_workflow_stop(workflow)
                if stop_reason == REQUEST_DEADLINE_EXCEEDED:
                    self._mark_failed(workflow, stop_reason)
                    raise WorkflowDeadlineExceeded(workflow_id)
                self._transition_workflow_locked(workflow, "completed")
                self._ensure_terminal_outcome_locked(workflow)
                return workflow.snapshot()

    def renew_stage_lease(
        self,
        workflow_id: str,
        stage_id: str,
        attempt_id: str,
        lease_id: str,
        lease_epoch: int,
        lease_expires_at: float,
    ) -> dict:
        """Extend one running attempt lease through the Coordinator state machine."""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                stage = workflow.stages.get(stage_id)
                if stage is None:
                    raise TaskGraphError(f"stage does not exist: {stage_id}")
                attempt = next(
                    (
                        item for item in stage.attempts
                        if item.attempt_id == attempt_id
                    ),
                    None,
                )
                if attempt is None:
                    raise TaskGraphError(
                        f"attempt does not exist: {attempt_id}"
                    )
                return self._renew_stage_lease_locked(
                    workflow,
                    stage,
                    attempt,
                    lease_id,
                    lease_epoch,
                    lease_expires_at,
                )

    def _renew_stage_lease_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
        lease_id: str,
        lease_epoch: int,
        lease_expires_at: float,
    ) -> dict:
        if (
            stage.state != "running"
            or attempt.state != "running"
            or attempt.lease_id != lease_id
            or attempt.lease_epoch != int(lease_epoch)
            or stage.lease_epoch != int(lease_epoch)
        ):
            raise TaskGraphError("lease identity is stale or not running")
        now = time.time()
        deadline = float(lease_expires_at)
        if (
            workflow.request_deadline is not None
            and workflow.request_deadline.expired()
        ):
            raise WorkflowDeadlineExceeded(workflow.workflow_id)
        if (
            not deadline > now
            or not deadline < float("inf")
            or deadline <= attempt.lease_expires_at
            or deadline > now + 3600.0
        ):
            raise TaskGraphError(
                "renewed lease deadline must extend the current lease "
                "within 3600 seconds"
            )
        previous_deadline = attempt.lease_expires_at
        previous_monotonic_deadline = attempt.lease_deadline_monotonic
        attempt.lease_expires_at = deadline
        attempt.lease_deadline_monotonic = time.monotonic() + max(
            0.0, deadline - time.time(),
        )
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage_attempt",
                entity_id=f"{stage.spec.stage_id}:{attempt.attempt_id}",
                event_type="stage_attempt_lease_renewed",
                payload={
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": attempt.attempt_id,
                    "lease_id": lease_id,
                    "lease_epoch": int(lease_epoch),
                    "lease_expires_at": deadline,
                },
            )
        except Exception:
            attempt.lease_expires_at = previous_deadline
            attempt.lease_deadline_monotonic = previous_monotonic_deadline
            raise
        return attempt.snapshot()

    def submit_stage_result(
        self,
        workflow_id: str,
        stage_id: str,
        result: ProviderStageResult,
    ) -> dict:
        """Submit one Provider result through the lease fencing gate."""
        if not isinstance(result, ProviderStageResult):
            raise TaskGraphError("result must be a StageResult")
        normalized_result = ProviderStageResult(
            output=result.output,
            provider_id=str(result.provider_id or ""),
            metadata=sanitize_result_metadata(
                result.metadata if isinstance(result.metadata, dict) else {}
            ),
            attempt_id=str(result.attempt_id or ""),
            lease_epoch=result.lease_epoch,
        )
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                stage = workflow.stages.get(stage_id)
                if stage is None:
                    raise TaskGraphError(
                        f"workflow {workflow_id} has no stage {stage_id}"
                    )
                return self._submit_stage_result_locked(
                    workflow, stage, normalized_result,
                )

    def discard_result(
        self,
        workflow_id: str,
        reason: str = "cancelled before result commit",
        *,
        terminal_reason_code: str = REASON_GENERATION_CANCELLED,
    ) -> dict:
        """Discard an uncommitted final result while preserving terminal states."""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                if workflow.state == "cancelled":
                    return workflow.snapshot()
                if workflow.state != "result_ready":
                    raise TaskGraphError(
                        f"workflow {workflow_id} result cannot be discarded from "
                        f"state {workflow.state}"
                    )
                self._request_cancel_locked(
                    workflow,
                    reason,
                    terminal_reason_code=terminal_reason_code,
                    terminal_source="result_discard",
                )
                return workflow.snapshot()

    def get(self, workflow_id: str) -> dict:
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is not None:
                return workflow.snapshot()
            snapshot = self._journal_snapshot(workflow_id)
            if snapshot is None:
                raise WorkflowNotFound(workflow_id)
            return snapshot

    def list(self, limit: int = 20, session_id: str = "") -> list[dict]:
        safe_limit = max(1, min(int(limit), self._max_records))
        session_filter = str(session_id or "")
        with self._lock:
            by_id = {
                snapshot["workflow_id"]: snapshot
                for snapshot in (
                    []
                    if self._journal_error
                    else self._journal_snapshots(
                        self._max_records if session_filter else safe_limit
                    )
                )
            }
            for workflow in self._workflows.values():
                snapshot = workflow.snapshot()
                by_id[snapshot["workflow_id"]] = snapshot
            workflows = sorted(
                (
                    workflow for workflow in by_id.values()
                    if not session_filter
                    or workflow.get("session_id", "") == session_filter
                ),
                key=lambda workflow: float(workflow.get("created_at", 0.0)),
                reverse=True,
            )
            return workflows[:safe_limit]

    def cancel(self, workflow_id: str) -> dict:
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                if workflow.state not in TERMINAL_WORKFLOW_STATES:
                    self._request_cancel_locked(
                        workflow, "cancelled before result commit",
                    )
                else:
                    self._ensure_terminal_outcome_locked(workflow)
                    workflow.terminal_outcome.decide(
                        REASON_GENERATION_CANCELLED,
                        source="cancel_request",
                    )
            return workflow.snapshot()

    def request_cancel(self, workflow_id: str) -> Optional[dict]:
        """Cancel a workflow or fence a valid ID that has not registered yet."""
        if not WORKFLOW_ID_PATTERN.fullmatch(workflow_id):
            raise TaskGraphError(
                "workflow_id must start with wf_ and contain 8-96 safe characters"
            )
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                snapshot = self._journal_snapshot(workflow_id)
                if snapshot is not None:
                    if snapshot.get("recovery_pending"):
                        raise TaskGraphUnavailable(
                            f"workflow {workflow_id} requires restart recovery"
                        )
                    return snapshot
                self._prune_pending_cancellations_locked()
                if len(self._pending_cancellations) >= self._max_records:
                    oldest = min(
                        self._pending_cancellations,
                        key=lambda item: self._pending_cancellations[item],
                    )
                    self._pending_cancellations.pop(oldest, None)
                self._pending_cancellations[workflow_id] = time.time()
                return None
            with workflow.lock:
                if workflow.state not in TERMINAL_WORKFLOW_STATES:
                    self._request_cancel_locked(
                        workflow, "cancelled before result commit",
                    )
                else:
                    self._ensure_terminal_outcome_locked(workflow)
                    workflow.terminal_outcome.decide(
                        REASON_GENERATION_CANCELLED,
                        source="cancel_request",
                    )
            return workflow.snapshot()

    def _prune_pending_cancellations_locked(self) -> None:
        cutoff = time.time() - 300.0
        expired = [
            workflow_id
            for workflow_id, created_at in self._pending_cancellations.items()
            if created_at < cutoff
        ]
        for workflow_id in expired:
            self._pending_cancellations.pop(workflow_id, None)

    def _prune_locked(self) -> None:
        self._prune_pending_cancellations_locked()
        if len(self._workflows) < self._max_records:
            return
        ordered = sorted(
            self._workflows.values(), key=lambda workflow: workflow.created_at,
        )
        for workflow in ordered:
            if workflow.state in TERMINAL_WORKFLOW_STATES:
                self._workflows.pop(workflow.workflow_id, None)
                if len(self._workflows) < self._max_records:
                    return
        if len(self._workflows) >= self._max_records:
            raise TaskGraphError("task graph registry is full")
