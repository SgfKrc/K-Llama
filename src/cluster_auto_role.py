"""Explicit-opt-in automatic role state machine for P4.5.

The controller consumes quorum-issued certificates and never manufactures a
write authority by itself.  Static ``master``/``client`` deployments remain
unchanged; callers must explicitly construct this controller and provide the
real quorum collector before auto mode can attempt takeover.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from cluster_control_contract import (
    CommitWatermark,
    ControlContractError,
    ControlPlaneAuthority,
    QuorumCertificate,
)
from cluster_fence import ControlFence
from cluster_quorum import QuorumCollector, QuorumOutcome


AUTO_ROLE_SCHEMA_VERSION = "qlh.cluster.auto_role.v1"
AUTO_ROLE_MODES = frozenset({"auto", "master", "client"})
AUTO_ROLE_STATES = frozenset({"starting", "leader", "follower", "read_only", "stopped"})


class AutoRoleError(ValueError):
    """Stable input/state error for the explicit auto-role adapter."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


@dataclass(frozen=True)
class AutoRoleDecision:
    accepted: bool
    state: str
    runtime_role: str
    reason: str
    term: int | None = None
    leader_id: str = ""
    certificate_digest: str = ""
    lease_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "state": self.state,
            "runtime_role": self.runtime_role,
            "reason": self.reason,
            "term": self.term,
            "leader_id": self.leader_id,
            "certificate_digest": self.certificate_digest,
            "lease_id": self.lease_id,
        }


class AutoRoleController:
    """Fail-closed role state machine backed by quorum certificates.

    ``start`` and ``on_reconnect`` may attempt a takeover only when the caller
    explicitly supplies an available voter set.  A missing set, collector, or
    strict majority always yields ``read_only``.  This is deliberately an
    adapter around the existing quorum protocol, not a second election engine.
    """

    def __init__(
        self,
        node_id: str,
        *,
        mode: str = "auto",
        authority: ControlPlaneAuthority | None = None,
        collector: QuorumCollector | None = None,
        fence: ControlFence | None = None,
        commit_watermark_provider: Callable[[], Sequence[CommitWatermark]] | None = None,
        committed_prefix_guard: Callable[[QuorumCertificate], None] | None = None,
    ) -> None:
        resolved_node_id = str(node_id or "").strip()
        if not resolved_node_id:
            raise AutoRoleError("node_id_missing", "auto role node id is required")
        resolved_mode = str(mode or "").strip().lower()
        if resolved_mode not in AUTO_ROLE_MODES:
            raise AutoRoleError("role_mode_invalid", "role mode must be auto, master, or client")
        if fence is not None and not isinstance(fence, ControlFence):
            raise AutoRoleError("fence_invalid", "fence must be a ControlFence")
        if authority is not None and not isinstance(authority, ControlPlaneAuthority):
            raise AutoRoleError("authority_invalid", "authority must be a ControlPlaneAuthority")
        if fence is not None and fence.authority is not None:
            if authority is not None and authority is not fence.authority:
                raise AutoRoleError("authority_mismatch", "fence and authority must share one trust root")
            authority = fence.authority
        if collector is not None and not isinstance(collector, QuorumCollector):
            raise AutoRoleError("collector_invalid", "collector must be a QuorumCollector")
        if commit_watermark_provider is not None and not callable(commit_watermark_provider):
            raise AutoRoleError("watermark_provider_invalid", "commit watermark provider must be callable")
        if committed_prefix_guard is not None and not callable(committed_prefix_guard):
            raise AutoRoleError("watermark_guard_invalid", "committed prefix guard must be callable")
        self.node_id = resolved_node_id
        self.mode = resolved_mode
        self.authority = authority
        self.collector = collector
        self.fence = fence
        #: #93：leader 侧「我认为已提交到哪」的提议来源（None = 不声明，与旧行为一致）。
        self.commit_watermark_provider = commit_watermark_provider
        #: #93：接管侧校验钩子——用自己的复制副本确认已提交前缀（落后需追赶、多出需截断）。
        self.committed_prefix_guard = committed_prefix_guard
        self._lock = threading.RLock()
        self._state = "starting"
        self._reason = "startup_pending"
        self._certificate: QuorumCertificate | None = None
        self._last_outcome: QuorumOutcome | None = None
        self._reconnect_requires_new_quorum = False
        self._generation = 0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def runtime_role(self) -> str:
        if self.mode == "auto" and self.state == "leader":
            self.can_write()
        with self._lock:
            return "master" if self._state == "leader" else "client"

    @property
    def certificate(self) -> QuorumCertificate | None:
        with self._lock:
            return self._certificate

    def start(
        self,
        *,
        available_voter_ids: Sequence[str] | None = None,
        now_ms: int | None = None,
    ) -> AutoRoleDecision:
        """Start static mode or explicitly attempt an auto takeover."""

        if self.mode == "master":
            return self._transition("leader", "static_master", accepted=True)
        if self.mode == "client":
            return self._transition("follower", "static_client", accepted=True)
        if self.state == "stopped":
            self._transition("starting", "startup_pending", accepted=False)
        if available_voter_ids is None:
            return self._transition("read_only", "quorum_unavailable", accepted=False)
        return self.attempt_takeover(available_voter_ids=available_voter_ids, now_ms=now_ms)

    def _proposed_commit_watermark(self) -> tuple[CommitWatermark, ...]:
        """#93：leader 自己声称的「已提交前缀」。没有来源 ⇒ 空元组（= 不声明）。

        提议**失败**时同样返回空元组：宁可退化成"不承诺"，也不能把一次读取失败
        变成"假装已提交"。真正的把关在 voter 侧（它们会拒签自己没耐久持有的前缀）。
        """
        provider = self.commit_watermark_provider
        if provider is None:
            return ()
        try:
            proposed = tuple(provider())
        except Exception:  # noqa: BLE001 - 提议必须 fail-safe 成「不声明」
            return ()
        return tuple(item for item in proposed if isinstance(item, CommitWatermark))

    def attempt_takeover(
        self,
        *,
        available_voter_ids: Sequence[str],
        now_ms: int | None = None,
        lease_id: str | None = None,
    ) -> AutoRoleDecision:
        """Ask the existing quorum collector for a certificate; never self-sign."""

        if self.mode != "auto":
            return self._transition("read_only", "auto_mode_disabled", accepted=False)
        if self.state == "stopped":
            return self._transition("stopped", "stopped", accepted=False)
        collector = self.collector
        if collector is None:
            return self._transition("read_only", "quorum_unavailable", accepted=False)
        with self._lock:
            if self._state == "stopped":
                return self._transition("stopped", "stopped", accepted=False)
            generation = self._generation
        from qlh_core.control.clock import operation_clock
        read_clock = operation_clock(now_ms)
        try:
            proposed = self._proposed_commit_watermark()
            outcome = collector.acquire(
                self.node_id,
                available_voter_ids=tuple(available_voter_ids),
                now_ms=read_clock(),
                lease_id=lease_id,
                commit_watermark=proposed,
            )
        except (ControlContractError, ValueError, RuntimeError) as exc:
            with self._lock:
                if generation != self._generation:
                    return self._transition(self._state, "role_changed", accepted=False)
                return self._transition("read_only", "quorum_unavailable", accepted=False)
        with self._lock:
            if generation != self._generation:
                return self._transition(self._state, "role_changed", accepted=False)
            self._last_outcome = outcome
            if not outcome.accepted or outcome.certificate is None:
                return self._transition("read_only", outcome.reason, accepted=False, outcome=outcome)
        return self.consume_certificate(outcome.certificate, now_ms=read_clock(), outcome=outcome)

    def renew_lease(
        self,
        *,
        available_voter_ids: Sequence[str],
        now_ms: int | None = None,
    ) -> AutoRoleDecision:
        """Renew only the currently installed lease, fencing on quorum loss."""
        with self._lock:
            certificate = self._certificate
            generation = self._generation
            if self._state == "stopped":
                return self._transition("stopped", "stopped", accepted=False)
            if self._state != "leader" or certificate is None or self.collector is None:
                return self._transition("read_only", "quorum_unavailable", accepted=False)
        from qlh_core.control.clock import operation_clock
        read_clock = operation_clock(now_ms)
        try:
            # #93：续期时重新提议水位（提交点会前进）；提议失败 ⇒ 退回证书里的原承诺，
            # 绝不因为"读不到新水位"就把已有承诺降级掉。
            proposed = self._proposed_commit_watermark() or tuple(certificate.commit_watermark)
            outcome = self.collector.renew(
                certificate, available_voter_ids=tuple(available_voter_ids), now_ms=read_clock(),
                commit_watermark=proposed,
            )
        except (ControlContractError, ValueError, RuntimeError):
            outcome = None
        with self._lock:
            if self._generation != generation or self._certificate is not certificate:
                return self._transition(self._state, "role_changed", accepted=False)
            if outcome is None or not outcome.accepted or outcome.certificate is None:
                return self.on_quorum_loss(outcome.reason if outcome is not None else "quorum_unavailable")
        return self.consume_certificate(outcome.certificate, now_ms=read_clock(), outcome=outcome)

    def _enforce_committed_prefix(self, certificate: QuorumCertificate | Mapping[str, Any], *, permit=None) -> None:
        """#93：接管前的「已提交前缀」核对（由部署注入的 ``committed_prefix_guard`` 执行）。

        约定：guard 正常返回 = 本地前缀与证书承诺一致（必要时它已经完成截断/追赶）；
        抛异常 = 不能接管。这里统一转成 `AutoRoleError`，让状态机走既有 fail-closed 路径
        降到 read_only，而不是把底层异常码泄漏成「未知原因」。
        """
        guard = self.committed_prefix_guard
        if guard is None:
            return
        if not isinstance(certificate, QuorumCertificate):
            certificate = QuorumCertificate.from_dict(certificate)
        try:
            import inspect
            if permit is not None:
                permit.check()
            if permit is not None and "permit" in inspect.signature(guard).parameters:
                guard(certificate, permit=permit)
            else:
                guard(certificate)
            if permit is not None:
                permit.check()
        except AutoRoleError:
            raise
        except Exception as exc:  # noqa: BLE001 - 一律 fail-closed
            raise AutoRoleError(
                "committed_prefix_unverified",
                str(exc) or "committed prefix could not be verified",
            ) from exc

    def consume_certificate(
        self,
        certificate: QuorumCertificate | Mapping[str, Any],
        *,
        now_ms: int | None = None,
        outcome: QuorumOutcome | None = None,
    ) -> AutoRoleDecision:
        """Install and consume a quorum certificate, demoting on any failure."""

        with self._lock:
            if self._state == "stopped":
                return self._transition("stopped", "stopped", accepted=False)
            generation = self._generation
        try:
            # The authority issues recovery permission only after all pure
            # certificate checks; no invalid proof can truncate a replica.
            recovery = lambda permit: self._enforce_committed_prefix(permit.certificate, permit=permit)
            if self.fence is not None:
                self.fence.install_certificate(certificate, now_ms=now_ms, recovery=recovery)
                current = self.fence.authority.current_certificate() if self.fence.authority else None
            elif self.authority is not None:
                self.authority.install_certificate(certificate, now_ms=now_ms, recovery=recovery)
                current = self.authority.current_certificate()
            else:
                raise AutoRoleError("authority_unavailable", "auto role has no control authority")
            if current is None:
                raise AutoRoleError("certificate_missing", "certificate was not installed")
        except (AutoRoleError, ControlContractError, TypeError, ValueError) as exc:
            with self._lock:
                if generation != self._generation:
                    return AutoRoleDecision(False, self._state, "client", "role_changed")
                reason = getattr(exc, "code", "control_certificate_invalid")
                return self._transition("read_only", str(reason), accepted=False)
        with self._lock:
            if generation != self._generation or self._state == "stopped":
                if self.authority is not None:
                    self.authority.revoke_certificate(current)
                return AutoRoleDecision(False, self._state, "client", "role_changed")
            return self._accept_certificate(current, outcome)

    def _accept_certificate(self, current, outcome):
        if outcome is not None:
            self._last_outcome = outcome
        self._certificate = current
        self._reconnect_requires_new_quorum = False
        if current.leader_id == self.node_id:
            return self._transition("leader", "quorum_certificate_accepted", accepted=True)
        return self._transition("follower", "valid_certificate_other_leader", accepted=True)

    def on_quorum_loss(self, reason: str = "quorum_unavailable") -> AutoRoleDecision:
        """Drop write authority immediately when voter reachability is lost."""
        with self._lock:
            if self._state == "stopped":
                return self._transition("stopped", "stopped", accepted=False)
            self._reconnect_requires_new_quorum = True
        return self._transition("read_only", str(reason or "quorum_unavailable"), accepted=False)

    def on_disconnect(self) -> AutoRoleDecision:
        """Connection loss is a write fence even if the last certificate is unexpired."""

        return self.on_quorum_loss("quorum_unavailable")

    def on_reconnect(
        self,
        *,
        certificate: QuorumCertificate | Mapping[str, Any] | None = None,
        available_voter_ids: Sequence[str] | None = None,
        now_ms: int | None = None,
    ) -> AutoRoleDecision:
        if self.mode == "auto" and available_voter_ids is not None:
            return self.attempt_takeover(available_voter_ids=available_voter_ids, now_ms=now_ms)
        if certificate is not None:
            try:
                candidate = (
                    certificate
                    if isinstance(certificate, QuorumCertificate)
                    else QuorumCertificate.from_dict(certificate)
                )
            except (ControlContractError, TypeError, ValueError):
                return self._transition("read_only", "control_certificate_invalid", accepted=False)
            with self._lock:
                requires_new_quorum = self._reconnect_requires_new_quorum
                previous = self._certificate
            if requires_new_quorum and previous is not None and candidate.digest() == previous.digest():
                return self._transition("read_only", "quorum_unavailable", accepted=False)
            return self.consume_certificate(certificate, now_ms=now_ms)
        return self._transition("read_only", "quorum_unavailable", accepted=False)

    def stop(self) -> AutoRoleDecision:
        return self._transition("stopped", "stopped", accepted=False)

    def can_write(self, *, now_ms: int | None = None) -> bool:
        with self._lock:
            state = self._state
            certificate = self._certificate
            if self.mode == "master" and self.fence is None:
                return state == "leader"
            if state != "leader" or certificate is None:
                return False
            try:
                # Runtime-role observation is also used by the API log handler.
                # Validate the shared authority without logging a control write:
                # audit -> log handler -> runtime_role would recurse and invert
                # the logging-handler/controller locks across concurrent threads.
                if self.authority is not None:
                    self.authority.admit_control_write(certificate, now_ms=now_ms)
                else:
                    return False
                return True
            except (ControlContractError, ValueError, TypeError):
                self.on_quorum_loss("control_certificate_not_current")
                return False

    def can_follow(self, *, now_ms: int | None = None) -> bool:
        """A follower may consume leader commands while local writes remain fenced."""
        with self._lock:
            if self._state not in {"leader", "follower"} or self._certificate is None or self.authority is None:
                return False
            try:
                self.authority.admit_control_write(self._certificate, now_ms=now_ms)
                return True
            except (ControlContractError, ValueError, TypeError):
                return False

    def observe_peer_certificate(self, certificate, *, now_ms=None):
        # Renewals of an already installed proof are ordinary observations.
        # A replay after disconnect still goes through on_reconnect's new-proof fence.
        from cluster_control_contract import QuorumCertificate
        try:
            candidate = certificate if isinstance(certificate, QuorumCertificate) else QuorumCertificate.from_dict(certificate)
        except (ControlContractError, TypeError, ValueError):
            raise AutoRoleError("control_certificate_invalid", "invalid peer authority") from None
        with self._lock:
            current = self._certificate
            same = current is not None and candidate.digest() == current.digest()
            if same and self._state in {"leader", "follower"}:
                return
        decision = self.on_reconnect(certificate=candidate, now_ms=now_ms)
        if not decision.accepted:
            raise AutoRoleError(decision.reason, "peer authority was not accepted")

    def snapshot(self, *, now_ms: int | None = None) -> dict[str, Any]:
        writable = self.can_write(now_ms=now_ms)
        with self._lock:
            certificate = self._certificate
            outcome = self._last_outcome
            state = self._state
            reason = self._reason
        certificate_view = certificate.to_dict() if certificate is not None else None
        return {
            "schema_version": AUTO_ROLE_SCHEMA_VERSION,
            "mode": self.mode,
            "node_id": self.node_id,
            "state": state,
            "runtime_role": "master" if state == "leader" else "client",
            "writable": writable,
            "reason": reason,
            "certificate": certificate_view,
            "last_outcome": outcome.to_dict() if outcome is not None else None,
        }

    def _transition(
        self,
        state: str,
        reason: str,
        *,
        accepted: bool,
        outcome: QuorumOutcome | None = None,
    ) -> AutoRoleDecision:
        if state not in AUTO_ROLE_STATES:
            raise AutoRoleError("state_invalid", "unsupported auto role state")
        with self._lock:
            if state in {"read_only", "stopped"} and self.authority is not None:
                self.authority.cancel_recovery()
            if (
                state in {"read_only", "stopped"}
                and self.mode == "auto"
                and self.authority is not None
                and self._certificate is not None
            ):
                self.authority.revoke_certificate(self._certificate)
            self._generation += 1
            self._state = state
            self._reason = str(reason)
            if outcome is not None:
                self._last_outcome = outcome
            certificate = self._certificate
        return AutoRoleDecision(
            accepted=accepted,
            state=state,
            runtime_role="master" if state == "leader" else "client",
            reason=str(reason),
            term=certificate.term if certificate is not None else None,
            leader_id=certificate.leader_id if certificate is not None else "",
            certificate_digest=certificate.digest() if certificate is not None else "",
            lease_id=certificate.lease_id if certificate is not None else "",
        )


__all__ = [
    "AUTO_ROLE_MODES",
    "AUTO_ROLE_SCHEMA_VERSION",
    "AUTO_ROLE_STATES",
    "AutoRoleController",
    "AutoRoleDecision",
    "AutoRoleError",
]
