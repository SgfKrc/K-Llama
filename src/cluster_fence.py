"""Runtime fencing adapter for the P4.5 control plane.

The quorum protocol owns certificate construction.  This module owns the
runtime boundary: extracting a certificate from a transport, admitting a
control write, and emitting a stable audit event.  It deliberately does not
elect a leader or persist certificates.
"""

from __future__ import annotations

import base64
import contextvars
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from cluster_control_contract import (
    CONTROL_CONTRACT_ERROR_CODES,
    CONTROL_SCHEMA_VERSION,
    ControlContractError,
    ControlPlaneAuthority,
    ControlWritePermit,
    QuorumCertificate,
    VoterSet,
)

logger = logging.getLogger(__name__)

FENCE_HEADER = "X-QLH-Control-Certificate"
FENCE_ERROR_CODES = frozenset(
    {
        "control_fence_unavailable",
        "control_certificate_missing",
        "control_certificate_invalid",
        "control_certificate_expired",
        "control_certificate_not_quorum",
        "control_certificate_stale",
        "control_certificate_conflict",
        "control_certificate_not_current",
        "control_certificate_cluster_mismatch",
        "control_certificate_epoch_mismatch",
        "control_certificate_leader_unknown",
        "control_certificate_signature_invalid",
        "control_certificate_duplicate_vote",
        "control_voter_unknown",
        "control_crypto_unavailable",
    }
)

# Middleware and scheduler calls share the permit without passing certificates
# through every legacy method signature.  The context is cleared at the HTTP
# boundary and is copied into FastAPI's worker thread by run_in_threadpool.
_current_permit: contextvars.ContextVar[ControlWritePermit | None] = contextvars.ContextVar(
    "qlh_control_write_permit", default=None
)


class ControlFenceError(ControlContractError):
    """Public, stable failure raised by a runtime control fence."""

    def __init__(self, code: str, message: str, *, action: str = "") -> None:
        if code == "control_fence_unavailable":
            self.code = code
            ValueError.__init__(self, message)
        else:
            super().__init__(code, message)
        self.action = action


def _env_bool(name: str, default: bool = False, environment=None) -> bool:
    value = (os.environ if environment is None else environment).get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _public_error(exc: ControlContractError) -> dict[str, str]:
    return {"code": exc.code, "message": str(exc)}


def _decode_header(value: str) -> dict[str, Any]:
    raw = value.strip()
    if not raw:
        raise ControlFenceError("control_certificate_missing", "control write requires a quorum certificate")
    try:
        decoded = base64.urlsafe_b64decode(raw + ("=" * (-len(raw) % 4)))
        document = json.loads(decoded.decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlFenceError("control_certificate_invalid", "control certificate header is invalid") from exc
    if not isinstance(document, dict):
        raise ControlFenceError("control_certificate_invalid", "control certificate must be an object")
    return document


def encode_certificate_header(certificate: QuorumCertificate | Mapping[str, Any]) -> str:
    """Encode a certificate for ``X-QLH-Control-Certificate``."""
    value = certificate.to_dict() if isinstance(certificate, QuorumCertificate) else dict(certificate)
    return base64.urlsafe_b64encode(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")


def certificate_from_headers(headers: Mapping[str, str]) -> dict[str, Any] | None:
    value = headers.get(FENCE_HEADER) or headers.get(FENCE_HEADER.lower())
    if not value:
        return None
    return _decode_header(value)


class ControlFence:
    """Certificate gate shared by API, scheduler, and TCP control paths."""

    def __init__(
        self,
        authority: ControlPlaneAuthority | None = None,
        *,
        required: bool = False,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.authority = authority
        self.required = bool(required)
        self.audit_sink = audit_sink
        self.role_guard: Callable[[], bool] | None = None
        self.peer_guard: Callable[[], bool] | None = None
        self.peer_certificate_consumer = None

    def set_role_guard(self, guard: Callable[[], bool] | None) -> None:
        if guard is not None and not callable(guard):
            raise TypeError("role guard must be callable")
        self.role_guard = guard

    def set_peer_guard(self, guard, *, consume_certificate=None) -> None:
        if guard is not None and not callable(guard):
            raise TypeError("peer guard must be callable")
        if consume_certificate is not None and not callable(consume_certificate):
            raise TypeError("peer certificate consumer must be callable")
        self.peer_guard = guard
        self.peer_certificate_consumer = consume_certificate

    def _require_local_role(self, action: str, source: str) -> None:
        if self.role_guard is None or source == "auto_role":
            return
        guard = self.peer_guard if source == "tcp_peer" and self.peer_guard is not None else self.role_guard
        if not guard():
            error = ControlFenceError(
                "control_certificate_not_current",
                "local node has no current leader authority",
                action=action,
            )
            self._audit(action, source, "", error=error)
            raise error

    @classmethod
    def from_environment(cls, environment=None) -> "ControlFence":
        """Build only from explicit configuration; never invent voter keys."""
        environment = dict(os.environ if environment is None else environment)
        required = _env_bool("QLH_CONTROL_FENCE_REQUIRED", False, environment) or (
            environment.get("QLH_NODE_ROLE", "").strip().lower() == "auto"
        )
        raw_voters = environment.get("QLH_CONTROL_VOTER_SET", "").strip()
        authority = None
        if raw_voters:
            try:
                value = json.loads(raw_voters)
                voter_set = VoterSet.from_dict(value)
                authority = ControlPlaneAuthority(
                    cluster_id=voter_set.cluster_id,
                    voter_set=voter_set,
                    static_role=environment.get("QLH_NODE_ROLE") if environment.get("QLH_NODE_ROLE") in {"master", "client"} else "unknown",
                )
                # An explicit voter set is an explicit opt-in, even when the
                # boolean switch was omitted.
                required = True
            except (ControlContractError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.error("invalid QLH_CONTROL_VOTER_SET: %s", exc)
                required = True
        return cls(authority, required=required)

    @property
    def enabled(self) -> bool:
        return self.required

    def install_certificate(
        self,
        certificate: QuorumCertificate | Mapping[str, Any],
        *,
        now_ms: int | None = None,
        recovery=None,
    ) -> dict[str, Any]:
        if self.authority is None:
            raise ControlFenceError(
                "control_fence_unavailable",
                "control fence has no configured voter set",
            )
        try:
            validation = self.authority.install_certificate(certificate, now_ms=now_ms, recovery=recovery)
        except ControlContractError as exc:
            error = ControlFenceError(exc.code, str(exc), action="certificate.install")
            self._audit("certificate.install", "runtime", "", error=error)
            raise error from exc
        return {
            "certificate_digest": validation.certificate_digest,
            "term": validation.term,
            "voter_set_epoch": validation.voter_set_epoch,
            "valid_voter_ids": list(validation.valid_voter_ids),
        }

    def current_certificate(self) -> dict[str, Any] | None:
        if self.authority is None:
            return None
        certificate = self.authority.current_certificate()
        if certificate is None:
            return None
        return certificate.to_dict()

    def admit(
        self,
        certificate: QuorumCertificate | Mapping[str, Any] | None,
        *,
        action: str,
        source: str = "runtime",
        request_id: str = "",
        now_ms: int | None = None,
    ) -> ControlWritePermit | None:
        if not self.required:
            return None
        self._require_local_role(action, source)
        if self.authority is None:
            error = ControlFenceError(
                "control_fence_unavailable",
                "control fence is required but no voter set is configured",
                action=action,
            )
            self._audit(action, source, request_id, error=error)
            raise error
        try:
            permit = self.authority.admit_control_write(certificate, now_ms=now_ms)
        except ControlContractError as exc:
            error = ControlFenceError(exc.code, str(exc), action=action)
            self._audit(action, source, request_id, error=error)
            raise error from exc
        self._audit(action, source, request_id, permit=permit)
        return permit

    def admit_peer(self, certificate, *, action: str, observe=False, now_ms=None):
        """Admit a current leader's transport proof without granting local API writes."""
        if self.required and observe and self.peer_certificate_consumer is not None:
            try:
                self.peer_certificate_consumer(certificate, now_ms=now_ms)
            except ValueError as exc:
                code = getattr(exc, "code", "control_certificate_invalid")
                if code not in CONTROL_CONTRACT_ERROR_CODES:
                    code = "control_certificate_not_current"
                raise ControlFenceError(code, "peer authority was not accepted", action=action) from exc
        return self.admit(certificate, action=action, source="tcp_peer", now_ms=now_ms)

    def require_current_permit(self, *, action: str) -> ControlWritePermit | None:
        if not self.required:
            return None
        permit = _current_permit.get()
        if permit is None:
            error = ControlFenceError(
                "control_certificate_missing",
                "control write has no admitted certificate",
                action=action,
            )
            self._audit(action, "scheduler", "", error=error)
            raise error
        self._require_local_role(action, "scheduler")
        certificate = self.authority.current_certificate() if self.authority else None
        try:
            if certificate is None or certificate.digest() != permit.certificate_digest:
                raise ControlFenceError(
                    "control_certificate_not_current",
                    "control write permit is no longer current",
                    action=action,
                )
            self.authority.admit_control_write(certificate)
        except ControlContractError as exc:
            error = ControlFenceError(exc.code, str(exc), action=action)
            self._audit(action, "scheduler", "", error=error)
            raise error from exc
        return permit

    def current_permit(self) -> ControlWritePermit | None:
        """Return the request-scoped permit for a downstream lease operation."""
        return _current_permit.get()

    def admit_http(self, headers: Mapping[str, str], *, action: str, request_id: str = "") -> ControlWritePermit | None:
        certificate = certificate_from_headers(headers)
        permit = self.admit(certificate, action=action, source="http", request_id=request_id)
        _current_permit.set(permit)
        return permit

    def admit_runtime(
        self,
        certificate: QuorumCertificate | Mapping[str, Any] | None,
        *,
        action: str,
        source: str = "runtime",
    ) -> ControlWritePermit | None:
        """Admit a non-HTTP caller and bind the permit to its context."""
        permit = self.admit(certificate, action=action, source=source)
        _current_permit.set(permit)
        return permit

    def control_context_reset(self, token: contextvars.Token) -> None:
        _current_permit.reset(token)

    def clear_context(self) -> None:
        _current_permit.set(None)

    def _audit(
        self,
        action: str,
        source: str,
        request_id: str,
        *,
        permit: ControlWritePermit | None = None,
        error: ControlContractError | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "event": "control_write_admitted" if error is None else "control_write_rejected",
            "schema_version": CONTROL_SCHEMA_VERSION,
            "action": str(action),
            "source": str(source),
            "request_id": str(request_id or ""),
            "timestamp_ms": int(time.time() * 1000),
        }
        if permit is not None:
            event.update({"term": permit.term, "lease_id": permit.lease_id, "certificate_digest": permit.certificate_digest})
        if error is not None:
            event["error_code"] = error.code
        sink = self.audit_sink
        if sink is not None:
            try:
                sink(event)
            except Exception:
                logger.warning("control fence audit sink failed", exc_info=True)
        logger.info("control_fence event=%s action=%s code=%s term=%s", event["event"], action, event.get("error_code", "ok"), event.get("term", "-"))

    def snapshot(self) -> dict[str, Any]:
        if self.authority is None:
            return {"enabled": self.enabled, "available": False, "reason": "control_fence_unavailable"}
        return {"enabled": self.enabled, "available": True, **self.authority.snapshot().to_dict()}


__all__ = [
    "ControlFence",
    "ControlFenceError",
    "FENCE_ERROR_CODES",
    "FENCE_HEADER",
    "certificate_from_headers",
    "encode_certificate_header",
]
