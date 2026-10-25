"""The sole in-memory owner of accepted authority and recovery admission."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from qlh_core.contracts.authority import (
    AuthorityError, AuthorityProof, RecoveryFloor, RecoveryPermit,
    ValidatedAuthority, require_preserved_floor,
)


class AuthorityService:
    def __init__(self, validate: Callable[[Any, int | None], ValidatedAuthority], *, clock=None) -> None:
        self._validate = validate
        self._clock = clock
        self._lock = threading.RLock()
        self._record: ValidatedAuthority | None = None
        self._pending: ValidatedAuthority | None = None
        self._term = 0
        self._floor: tuple[RecoveryFloor, ...] = ()
        self._revision = 0
        self._recovering = False
        self._inflight_revision = None
        # Once takeover recovery starts, partial durable writes may exist even
        # after failure, cancellation or revocation. Only a completed protected
        # recovery can discharge this obligation.
        self._recovery_required = False

    def _check_candidate(self, record: ValidatedAuthority) -> None:
        proof = record.proof
        if self._inflight_revision is not None or self._recovering:
            raise AuthorityError("control_certificate_not_current", "authority recovery is in progress")
        if proof.identity.term < self._term:
            raise AuthorityError("control_certificate_stale", "certificate term is older than the committed term")
        if proof.identity.term == self._term:
            previous_record = self._record or self._pending
            if previous_record is None:
                raise AuthorityError("control_certificate_stale", "a revoked certificate cannot be installed again")
            previous = previous_record.proof
            if (proof.identity != previous.identity
                    or proof.expires_at_ms < previous.expires_at_ms
                    or (proof.expires_at_ms == previous.expires_at_ms
                        and proof.certificate_digest != previous.certificate_digest)):
                raise AuthorityError("control_certificate_conflict", "renewal must extend the same authority lease")
        require_preserved_floor(self._floor, proof.committed_floor)

    def _reading_clock(self, now_ms):
        if self._clock is not None:
            return self._clock
        from qlh_core.control.clock import operation_clock
        return operation_clock(now_ms)

    def install(self, certificate: Any, *, now_ms: int | None = None,
                recover: Callable[[RecoveryPermit], None] | None = None) -> Any:
        read_clock = self._reading_clock(now_ms)
        # Normalize once. Every later check and recovery consumes this snapshot.
        record = self._validate(certificate, read_clock())
        candidate = record.certificate
        with self._lock:
            record = self._validate(candidate, read_clock())
            self._check_candidate(record)
            same_identity = self._record is not None and self._record.proof.identity == record.proof.identity
            if recover is None and self._recovery_required:
                raise AuthorityError("control_certificate_not_current", "authority requires successful protected recovery")
            if recover is None or same_identity:
                self._publish(record)
                return record.verification
            # A validated takeover closes old admission before any recovery IO.
            self._record = None
            self._pending = record
            self._term = record.proof.identity.term
            self._floor = record.proof.committed_floor
            self._revision += 1
            revision = self._revision
            self._recovering = True
            self._inflight_revision = revision
            self._recovery_required = True

        def check_permission():
            with self._lock:
                if revision != self._revision or not self._recovering:
                    raise AuthorityError("control_certificate_not_current", "recovery permission was cancelled or consumed")
                self._validate(candidate, read_clock())

        def mutate(action):
            # Cancellation and durable mutation share the same short owner lock.
            # Network fetching happens outside this critical section.
            with self._lock:
                check_permission()
                return action()

        permit = RecoveryPermit(record.proof, revision, candidate, check_permission, mutate)
        try:
            check_permission()
            recover(permit)
            with self._lock:
                check_permission()
                record = self._validate(candidate, read_clock())
                self._publish(record)
                self._recovery_required = False
                return record.verification
        except BaseException:
            with self._lock:
                if revision == self._revision:
                    # Preserve the known floor/term, with every write fenced.
                    self._recovering = False
            raise
        finally:
            with self._lock:
                if self._inflight_revision == revision:
                    self._inflight_revision = None

    def _publish(self, record):
        self._record = record
        self._pending = None
        self._term = record.proof.identity.term
        self._floor = record.proof.committed_floor
        self._recovering = False
        self._revision += 1

    def cancel_recovery(self):
        with self._lock:
            if self._pending is None:
                return
            self._pending = None
            self._recovering = False
            self._revision += 1

    def revoke(self, digest: str) -> bool:
        with self._lock:
            record = self._record or self._pending
            if record is None or record.proof.certificate_digest != digest:
                return False
            self._record = None
            self._pending = None
            self._recovering = False
            self._revision += 1
            return True

    def observation(self) -> tuple[int, AuthorityProof | None, bool]:
        with self._lock:
            return self._term, self._record.proof if self._record else None, self._recovering

    def current_certificate(self) -> Any:
        with self._lock:
            return self._record.certificate if self._record else None

    def admit(self, certificate: Any, *, now_ms: int | None = None) -> AuthorityProof:
        with self._lock:
            record = self._validate(certificate, now_ms)
            if self._record is None or self._recovering:
                raise AuthorityError("control_certificate_not_current", "certificate is not active")
            if record.proof.identity.term < self._term:
                raise AuthorityError("control_certificate_stale", "certificate term is older than the committed term")
            if record.proof != self._record.proof:
                raise AuthorityError("control_certificate_not_current", "control write requires the current proof")
            return record.proof

    def admit_bound_identity(self, identity, *, now_ms=None):
        with self._lock:
            if self._record is None or self._recovering or self._record.proof.identity != identity:
                raise AuthorityError("control_certificate_not_current", "bound authority is no longer active")
            self._validate(self._record.certificate, now_ms)
            return self._record.proof
