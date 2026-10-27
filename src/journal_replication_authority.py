"""Fail-closed authority checks for durable replicated journal prefixes."""

from __future__ import annotations

import threading
from typing import Any, Sequence

from journal_replication import DurableJournalReplica
from journal_watermark import CommitWatermark


class JournalReplicationAuthorityError(RuntimeError):
    """Stable error raised when a journal prefix cannot prove authority."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


class JournalPrefixAuthority:
    """Treat a locally durable, self-consistent journal prefix as authoritative.

    ``held`` reports a verified durable prefix. ``enforce`` checks it against
    a persisted commitment, truncating uncommitted tails and refusing recovery
    when the committed prefix cannot be proven.
    """

    def __init__(
        self,
        replica: DurableJournalReplica,
        *,
        workflows: Sequence[str] = (),
        recovery: Any = None,
    ) -> None:
        self.replica = replica
        self.workflows = tuple(sorted({str(item) for item in workflows if str(item)}))
        self.recovery = recovery
        self._lock = threading.RLock()
        self.truncated: dict[str, int] = {}
        self.floor_provider = lambda: ()
        self.model_control = None
        self.requests = None

    def _recover(self, workflow_id: str, *, permit=None) -> None:
        """Try one recovery pass; failure leaves the authority check closed."""
        if self.recovery is None:
            return
        try:
            import inspect
            if permit is not None:
                permit.check()
            if permit is not None and "permit" in inspect.signature(self.recovery).parameters:
                self.recovery(workflow_id, permit=permit)
            elif permit is not None:
                permit.mutate(lambda: self.recovery(workflow_id))
            else:
                self.recovery(workflow_id)
        except Exception:  # noqa: BLE001 - recovery failure cannot grant authority
            return

    def held(self, workflow_id: str) -> CommitWatermark | None:
        """Return the durable, self-consistent prefix, or ``None`` if unverified."""
        from qlh_core.contracts.deployment import MODEL_CONTROL_DOMAIN
        from qlh_core.contracts.execution import REQUEST_DOMAIN

        if workflow_id in {MODEL_CONTROL_DOMAIN, REQUEST_DOMAIN}:
            owner = self.model_control if workflow_id == MODEL_CONTROL_DOMAIN else self.requests
            if owner is None:
                return None
            status = owner.repository.recovery_status()
            return (
                CommitWatermark(
                    workflow_id=workflow_id,
                    last_sequence=status["durable_sequence"],
                    checkpoint_digest=status["checkpoint_digest"],
                )
                if status.get("verified")
                else None
            )
        with self._lock:
            try:
                status = self.replica.recovery_status(str(workflow_id))
            except Exception:  # noqa: BLE001 - an unreadable prefix is not held
                return None
        if not status.get("verified"):
            return None
        return CommitWatermark(
            workflow_id=str(workflow_id),
            last_sequence=int(status["durable_sequence"]),
            checkpoint_digest=str(status["checkpoint_digest"]),
        )

    def propose(self) -> tuple[CommitWatermark, ...]:
        """Advertise only verified workflows and never drop a durable floor."""
        floors = {item.workflow_id: item for item in self.floor_provider()}
        proposed = []
        for workflow_id in sorted(set(self.workflows) | set(floors)):
            item = self.held(workflow_id)
            floor = floors.get(workflow_id)
            if floor is not None and (
                item is None
                or item.last_sequence < floor.last_sequence
                or (
                    item.last_sequence == floor.last_sequence
                    and item.checkpoint_digest != floor.checkpoint_digest
                )
            ):
                raise JournalReplicationAuthorityError(
                    "committed_prefix_behind", "required recovery floor is unavailable",
                )
            if item is not None:
                proposed.append(item)
        return tuple(proposed)

    def enforce(self, commitment: Any, *, permit=None) -> None:
        """Check a persisted commitment, truncating any uncommitted tail."""
        watermarks = tuple(getattr(commitment, "commit_watermark", ()) or ())
        if not watermarks:
            return
        for watermark in watermarks:
            if permit is not None:
                permit.check()
            workflow_id = watermark.workflow_id
            from qlh_core.contracts.deployment import MODEL_CONTROL_DOMAIN
            from qlh_core.contracts.execution import REQUEST_DOMAIN

            if workflow_id in {MODEL_CONTROL_DOMAIN, REQUEST_DOMAIN}:
                held = self.held(workflow_id)
                if held is None or held != watermark:
                    raise JournalReplicationAuthorityError(
                        "committed_prefix_behind", "owner recovery domain is unavailable",
                    )
                continue
            with self._lock:
                prefix = self.replica.durable_prefix(workflow_id)
                if int(prefix["last_sequence"]) > int(watermark.last_sequence):
                    guarded = (
                        {"permission_check": permit.check}
                        if permit is not None and isinstance(self.replica, DurableJournalReplica)
                        else {}
                    )
                    truncate = lambda: self.replica.truncate_to(
                        workflow_id, last_sequence=int(watermark.last_sequence), **guarded,
                    )
                    result = permit.mutate(truncate) if permit is not None else truncate()
                    self.truncated[workflow_id] = int(result.get("removed_events", 0))
                held = self.held(workflow_id)
                if held is None or held.last_sequence < int(watermark.last_sequence):
                    self._recover(workflow_id, permit=permit)
                    held = self.held(workflow_id)
            if held is None:
                raise JournalReplicationAuthorityError(
                    "committed_prefix_behind",
                    f"replica prefix for {workflow_id} is not self-consistent yet",
                )
            if held.last_sequence != watermark.last_sequence:
                raise JournalReplicationAuthorityError(
                    "committed_prefix_behind",
                    f"replica prefix {held.last_sequence} < committed {watermark.last_sequence}",
                )
            if held.checkpoint_digest != watermark.checkpoint_digest:
                raise JournalReplicationAuthorityError(
                    "committed_prefix_forked",
                    f"replica prefix diverges from the committed point {watermark.last_sequence}",
                )


__all__ = ["JournalPrefixAuthority", "JournalReplicationAuthorityError"]
