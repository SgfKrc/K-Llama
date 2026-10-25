"""Explicit term/certificate-first leader handoff coordinator.

The coordinator records only handoff metadata.  Journals, model assets,
sessions, live worker leases, and in-flight task snapshots stay on their
owners and are reconciled by the recovery ticket.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from cluster_auto_role import AutoRoleController
from cluster_control_contract import (
    ControlContractError,
    ControlPlaneAuthority,
)
from cluster_fence import ControlFence
from cluster_quorum import QuorumCollector, QuorumOutcome


HANDOFF_SCHEMA_VERSION = "qlh.cluster.handoff.v1"
HANDOFF_STATES = frozenset(
    {"prepared", "awaiting_quorum", "committed", "aborted", "failed"}
)


class HandoffError(ValueError):
    """Stable fail-closed error for handoff contract misuse."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


@dataclass(frozen=True)
class HandoffRecord:
    handoff_id: str
    cluster_id: str
    old_leader_id: str
    new_leader_id: str
    old_term: int
    new_term: int | None
    old_certificate_digest: str
    new_certificate_digest: str
    manifest_digest: str
    reason: str
    operator: str
    state: str
    result: str
    created_at_ms: int
    updated_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "handoff_id": self.handoff_id,
            "cluster_id": self.cluster_id,
            "old_leader_id": self.old_leader_id,
            "new_leader_id": self.new_leader_id,
            "old_term": self.old_term,
            "new_term": self.new_term,
            "old_certificate_digest": self.old_certificate_digest,
            "new_certificate_digest": self.new_certificate_digest,
            "manifest_digest": self.manifest_digest,
            "reason": self.reason,
            "operator": self.operator,
            "state": self.state,
            "result": self.result,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }


def _now_ms(value: int | None) -> int:
    if value is None:
        return int(time.time() * 1000)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HandoffError("handoff_time_invalid", "now_ms must be a non-negative integer")
    return value


def _text(value: Any, *, field: str, max_length: int = 256) -> str:
    result = str(value or "").strip()
    if not result or len(result) > max_length:
        raise HandoffError("handoff_input_invalid", f"{field} is invalid")
    return result


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping) or not manifest:
        raise HandoffError("handoff_manifest_invalid", "handoff manifest must be a non-empty object")
    try:
        encoded = json.dumps(
            dict(manifest), ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HandoffError("handoff_manifest_invalid", "handoff manifest is not canonical JSON") from exc
    if len(encoded) > 64 * 1024:
        raise HandoffError("handoff_manifest_invalid", "handoff manifest exceeds 64 KiB")
    return hashlib.sha256(encoded).hexdigest()


class HandoffCoordinator:
    """Coordinate a planned leader handoff through the existing HA contracts."""

    def __init__(
        self,
        node_id: str,
        *,
        authority: ControlPlaneAuthority,
        collector: QuorumCollector,
        fence: ControlFence | None = None,
        old_role_controller: AutoRoleController | None = None,
        new_role_controller: AutoRoleController | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.node_id = _text(node_id, field="node_id")
        if not isinstance(authority, ControlPlaneAuthority):
            raise HandoffError("handoff_authority_invalid", "authority is invalid")
        if not isinstance(collector, QuorumCollector):
            raise HandoffError("handoff_collector_invalid", "collector is invalid")
        if collector.voter_set.cluster_id != authority.voter_set.cluster_id:
            raise HandoffError("handoff_authority_mismatch", "collector and authority use different clusters")
        if fence is not None and not isinstance(fence, ControlFence):
            raise HandoffError("handoff_fence_invalid", "fence is invalid")
        if fence is not None and fence.authority is not None and fence.authority is not authority:
            raise HandoffError("handoff_authority_mismatch", "fence and authority must share one trust root")
        for role_controller, field in (
            (old_role_controller, "old_role_controller"),
            (new_role_controller, "new_role_controller"),
        ):
            if role_controller is not None and not isinstance(role_controller, AutoRoleController):
                raise HandoffError("handoff_role_controller_invalid", f"{field} is invalid")
            if role_controller is not None and role_controller.authority is not authority:
                raise HandoffError("handoff_authority_mismatch", f"{field} uses a different trust root")
        if event_sink is not None and not callable(event_sink):
            raise HandoffError("handoff_event_sink_invalid", "event_sink must be callable")
        self.authority = authority
        self.collector = collector
        self.fence = fence
        self.old_role_controller = old_role_controller
        self.new_role_controller = new_role_controller
        self.event_sink = event_sink
        self._lock = threading.RLock()
        self._record: HandoffRecord | None = None
        self._events: list[dict[str, Any]] = []

    @property
    def record(self) -> HandoffRecord | None:
        with self._lock:
            return self._record

    def prepare(
        self,
        new_leader_id: str,
        manifest: Mapping[str, Any],
        *,
        reason: str,
        operator: str,
        now_ms: int | None = None,
        handoff_id: str | None = None,
    ) -> HandoffRecord:
        now = _now_ms(now_ms)
        target = _text(new_leader_id, field="new_leader_id")
        reason_text = _text(reason, field="reason")
        operator_text = _text(operator, field="operator")
        if target not in self.authority.voter_set.voter_map:
            raise HandoffError("handoff_target_invalid", "new leader is not a configured voter")
        if target == self.node_id:
            raise HandoffError("handoff_target_invalid", "new leader must differ from old leader")
        manifest_digest = _manifest_digest(manifest)
        certificate = self.authority.current_certificate()
        if certificate is None:
            raise HandoffError("handoff_certificate_missing", "current leader certificate is required")
        if certificate.leader_id != self.node_id:
            raise HandoffError("handoff_not_current_leader", "local node does not hold the current leader certificate")
        try:
            self.authority.admit_control_write(certificate, now_ms=now)
        except ControlContractError as exc:
            raise HandoffError(exc.code, str(exc)) from exc
        if self.old_role_controller is not None and self.old_role_controller.state != "leader":
            raise HandoffError("handoff_not_current_leader", "old role controller is not leader")

        resolved_id = _text(handoff_id or f"handoff-{uuid.uuid4().hex}", field="handoff_id")
        with self._lock:
            existing = self._record
            if existing is not None and existing.state not in {"aborted", "failed"}:
                if (
                    existing.new_leader_id == target
                    and existing.manifest_digest == manifest_digest
                    and existing.reason == reason_text
                    and existing.operator == operator_text
                ):
                    return existing
                raise HandoffError("handoff_conflict", "another handoff is already active")
            record = HandoffRecord(
                handoff_id=resolved_id,
                cluster_id=self.authority.voter_set.cluster_id,
                old_leader_id=self.node_id,
                new_leader_id=target,
                old_term=certificate.term,
                new_term=None,
                old_certificate_digest=certificate.digest(),
                new_certificate_digest="",
                manifest_digest=manifest_digest,
                reason=reason_text,
                operator=operator_text,
                state="prepared",
                result="prepared",
                created_at_ms=now,
                updated_at_ms=now,
            )
            self._record = record
            self._emit_locked("handoff_prepared", record)
            return record

    def commit(
        self,
        *,
        available_voter_ids: Sequence[str],
        now_ms: int | None = None,
        lease_id: str | None = None,
    ) -> HandoffRecord:
        from qlh_core.control.clock import operation_clock
        read_clock = operation_clock(now_ms)
        now = read_clock()
        with self._lock:
            record = self._record
            if record is None:
                raise HandoffError("handoff_not_prepared", "handoff must be prepared first")
            if record.state == "committed":
                return record
            if record.state == "aborted":
                raise HandoffError("handoff_aborted", "handoff has been aborted")
            if record.state == "failed":
                raise HandoffError("handoff_failed", "handoff is terminally failed")
            current_certificate = self.authority.current_certificate()
            current_digest = current_certificate.digest() if current_certificate is not None else ""
            if (
                record.state == "prepared" and current_digest != record.old_certificate_digest
            ) or (
                record.state == "awaiting_quorum"
                and current_digest not in {"", record.old_certificate_digest}
            ):
                raise HandoffError("handoff_certificate_changed", "current certificate changed after handoff preparation")
            if self.old_role_controller is not None:
                self.old_role_controller.on_quorum_loss("handoff_in_progress")
            self._record = self._replace(record, state="awaiting_quorum", result="old_leader_fenced", updated_at_ms=now)
            self._emit_locked("handoff_old_leader_fenced", self._record)

        try:
            outcome = self.collector.acquire(
                record.new_leader_id,
                available_voter_ids=tuple(available_voter_ids),
                now_ms=read_clock(),
                lease_id=lease_id,
            )
        except (ControlContractError, ValueError, RuntimeError) as exc:
            outcome = QuorumOutcome(False, "quorum_unavailable")
            failure_reason = getattr(exc, "code", "quorum_unavailable")
        else:
            failure_reason = outcome.reason

        if not outcome.accepted or outcome.certificate is None:
            with self._lock:
                current = self._record
                if current is None:
                    raise HandoffError("handoff_not_prepared", "handoff disappeared")
                self._record = self._replace(
                    current, state="awaiting_quorum", result=str(failure_reason), updated_at_ms=now,
                )
                self._emit_locked("handoff_quorum_unavailable", self._record)
                return self._record

        certificate = outcome.certificate
        try:
            if self.new_role_controller is not None:
                decision = self.new_role_controller.consume_certificate(certificate, now_ms=read_clock(), outcome=outcome)
                if not decision.accepted or decision.state != "leader":
                    raise HandoffError("handoff_target_rejected", decision.reason)
            elif self.fence is not None:
                self.fence.install_certificate(certificate, now_ms=read_clock())
            else:
                self.authority.install_certificate(certificate, now_ms=read_clock())
        except (HandoffError, ControlContractError, TypeError, ValueError) as exc:
            reason_text = getattr(exc, "code", "handoff_target_rejected")
            with self._lock:
                current = self._record
                self._record = self._replace(
                    current,
                    state="failed",
                    new_term=certificate.term,
                    new_certificate_digest=certificate.digest(),
                    result=str(reason_text),
                    updated_at_ms=now,
                )
                self._emit_locked("handoff_failed", self._record)
                return self._record

        with self._lock:
            current = self._record
            self._record = self._replace(
                current,
                state="committed",
                new_term=certificate.term,
                new_certificate_digest=certificate.digest(),
                result="committed",
                updated_at_ms=now,
            )
            self._emit_locked("handoff_committed", self._record)
            return self._record

    def abort(self, *, now_ms: int | None = None, reason: str = "aborted") -> HandoffRecord:
        now = _now_ms(now_ms)
        reason_text = _text(reason, field="reason")
        with self._lock:
            record = self._record
            if record is None:
                raise HandoffError("handoff_not_prepared", "handoff must be prepared first")
            if record.state == "committed":
                raise HandoffError("handoff_committed", "committed handoff cannot be aborted")
            if record.state == "aborted":
                return record
            if record.state == "awaiting_quorum":
                raise HandoffError("handoff_old_leader_fenced", "fenced handoff requires a new quorum or recovery ticket")
            self._record = self._replace(record, state="aborted", result=reason_text, updated_at_ms=now)
            self._emit_locked("handoff_aborted", self._record)
            return self._record

    def recover_pending(
        self,
        *,
        now_ms: int | None = None,
        timeout_ms: int = 30_000,
    ) -> HandoffRecord | None:
        """Resolve an orphaned fenced handoff without restoring old write access.

        A process restart cannot safely recreate the old role controller from a
        handoff snapshot.  Until a fresh quorum commits, the old role remains
        fenced.  Once the bounded recovery window expires, the operation is
        made auditable as ``handoff_timeout`` and must be started again with a
        new certificate-first handoff.
        """
        now = _now_ms(now_ms)
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise HandoffError("handoff_timeout_invalid", "timeout_ms must be a positive integer")
        with self._lock:
            record = self._record
            if record is None or record.state != "awaiting_quorum":
                return record
            if now < record.updated_at_ms:
                raise HandoffError("handoff_time_invalid", "now_ms cannot move backwards")
            if now - record.updated_at_ms < timeout_ms:
                return record
            self._record = self._replace(
                record,
                state="failed",
                result="handoff_timeout",
                updated_at_ms=now,
            )
            self._emit_locked("handoff_timeout", self._record)
            return self._record

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            record = self._record
            return {
                "schema_version": HANDOFF_SCHEMA_VERSION,
                "state": record.state if record else "idle",
                "record": record.to_dict() if record else None,
                "events": [dict(event) for event in self._events[-64:]],
            }

    def _replace(self, record: HandoffRecord | None, **changes: Any) -> HandoffRecord:
        if record is None:
            raise HandoffError("handoff_not_prepared", "handoff record is missing")
        values = record.to_dict()
        values.pop("schema_version", None)
        values.update(changes)
        return HandoffRecord(**values)

    def _emit_locked(self, event_type: str, record: HandoffRecord) -> None:
        event = {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "event_type": event_type,
            "handoff_id": record.handoff_id,
            "cluster_id": record.cluster_id,
            "old_leader_id": record.old_leader_id,
            "new_leader_id": record.new_leader_id,
            "old_term": record.old_term,
            "new_term": record.new_term,
            "old_certificate_digest": record.old_certificate_digest,
            "new_certificate_digest": record.new_certificate_digest,
            "manifest_digest": record.manifest_digest,
            "reason": record.reason,
            "operator": record.operator,
            "state": record.state,
            "result": record.result,
            "occurred_at_ms": record.updated_at_ms,
        }
        self._events.append(event)
        if self.event_sink is not None:
            self.event_sink(dict(event))


__all__ = [
    "HANDOFF_SCHEMA_VERSION",
    "HANDOFF_STATES",
    "HandoffCoordinator",
    "HandoffError",
    "HandoffRecord",
]
