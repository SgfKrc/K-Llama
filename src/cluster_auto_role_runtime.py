"""Runtime wiring for the explicit quorum-backed automatic role.

The election state machine is intentionally transport agnostic.  This module
provides the small composition-root adapter that turns the configured local
voter and HTTP voter peers into the existing :class:`QuorumCollector`.
Automatic mode is fail-closed: incomplete configuration leaves the scheduler
in read-only state and never manufactures a certificate.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import math
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from cluster_auto_role import AutoRoleController
from cluster_control_contract import (
    CommitWatermark,
    ControlContractError,
    ControlPlaneAuthority,
    VoterSet,
)
from cluster_fence import ControlFence
from cluster_quorum import (
    QuorumCollector,
    QuorumError,
    QuorumPolicy,
    QuorumVoter,
    SQLiteVoterLedger,
)
from journal_replication import DurableJournalReplica
from journal_replication_service import JournalReplicationSettings, recover_journal_prefix


AUTO_ROLE_RPC_HEADER = "X-QLH-Quorum-Secret"
AUTO_ROLE_RPC_PATH = "/api/cluster/quorum/voter"
AUTO_ROLE_RPC_TIMEOUT_SECONDS = 8.0
logger = logging.getLogger(__name__)


class AutoRoleRuntimeError(RuntimeError):
    """Stable configuration/transport error for the auto-role adapter."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


def _decode_key(value: str) -> bytes:
    text = str(value or "").strip()
    if not text:
        raise AutoRoleRuntimeError("auto_role_private_key_missing", "voter private key is not configured")
    try:
        decoded = base64.urlsafe_b64decode(text + ("=" * (-len(text) % 4)))
    except (ValueError, TypeError) as exc:
        raise AutoRoleRuntimeError("auto_role_private_key_invalid", "voter private key is not base64") from exc
    if len(decoded) != 32:
        raise AutoRoleRuntimeError("auto_role_private_key_invalid", "voter private key must contain 32 raw bytes")
    return decoded


def _private_key_from_environment() -> Any:
    raw = os.environ.get("QLH_CONTROL_VOTER_PRIVATE_KEY", "")
    path = os.environ.get("QLH_CONTROL_VOTER_PRIVATE_KEY_FILE", "").strip()
    if path:
        try:
            raw = Path(path).expanduser().read_text(encoding="ascii").strip()
        except OSError as exc:
            raise AutoRoleRuntimeError("auto_role_private_key_unreadable", "voter private key file cannot be read") from exc
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:
        raise AutoRoleRuntimeError("auto_role_crypto_unavailable", "cryptography is required for automatic role") from exc
    return Ed25519PrivateKey.from_private_bytes(_decode_key(raw))


def _json_env(name: str, default: Any) -> Any:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AutoRoleRuntimeError("auto_role_config_invalid", f"{name} is not valid JSON") from exc


def _rpc_endpoint(value: str) -> str:
    endpoint = str(value or "").strip().rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise AutoRoleRuntimeError("auto_role_peer_invalid", "voter peer endpoint must use http or https")
    return endpoint + AUTO_ROLE_RPC_PATH


class HTTPRemoteVoter:
    """Duck-typed ``QuorumVoter`` proxy used by ``QuorumCollector``."""

    def __init__(self, voter_id: str, endpoint: str, *, secret: str, timeout: float) -> None:
        self.voter_id = str(voter_id)
        self.endpoint = _rpc_endpoint(endpoint)
        self._secret = str(secret)
        self._timeout = float(timeout)

    def _call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                AUTO_ROLE_RPC_HEADER: self._secret,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise QuorumError("quorum_unavailable", "voter peer is unreachable") from exc
        if not isinstance(value, dict) or not value.get("ok", False):
            code = str(value.get("code", "quorum_unavailable")) if isinstance(value, dict) else "quorum_unavailable"
            if code not in {"quorum_unavailable", "quorum_vote_rejected", "quorum_active_lease", "quorum_term_stale", "quorum_duplicate_vote", "quorum_expiry_regressed"}:
                code = "quorum_unavailable"
            raise QuorumError(code, "voter peer rejected the request")
        return value

    def reserve_term(self, leader_id: str, *, now_ms: int | None = None) -> int:
        return int(self._call({"op": "reserve_term", "leader_id": leader_id, "now_ms": now_ms})["term"])

    def prepare(self, leader_id: str, term: int, *, now_ms: int | None = None) -> None:
        self._call({"op": "prepare", "leader_id": leader_id, "term": int(term), "now_ms": now_ms})

    def sign_vote(self, certificate: Any, *, now_ms: int | None = None) -> Any:
        from cluster_control_contract import VoterSignature

        value = self._call({"op": "sign_vote", "certificate": certificate.to_dict(), "now_ms": now_ms})
        return VoterSignature.from_dict(value["signature"])

    def commit_certificate(self, certificate: Any, *, now_ms: int | None = None) -> None:
        self._call({"op": "commit_certificate", "certificate": certificate.to_dict(), "now_ms": now_ms})

    def snapshot(self) -> dict[str, Any]:
        return dict(self._call({"op": "snapshot"}).get("snapshot", {}))


class JournalPrefixAuthority:
    """#93：把本地 journal 复制副本当成「已提交前缀」的权威。

    两个方向的能力：
    - ``held(workflow_id)``：本节点**耐久持有且自洽**的前缀（供 voter 签名前自检、
      以及 leader 提议水位）；
    - ``enforce(certificate)``：接管前核对证书承诺 —— 本地**多出的未提交尾部先截断**，
      之后仍落后或无法自证 ⇒ 抛错（fail-closed，必须先靠 checkpoint 追赶/重建）。
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
        #: #93：落后时「主动追赶一次」的钩子；默认没有 ⇒ 就只能拒绝接管（fail-closed）。
        self.recovery = recovery
        self._lock = threading.RLock()
        #: 诊断：本次进程内因接管核对而被截断的 workflow → 被丢弃的事件数。
        self.truncated: dict[str, int] = {}

    def _recover(self, workflow_id: str) -> None:
        """落后时主动追赶一次；追赶失败不改结论（仍然是「不够格接管」）。"""
        if self.recovery is None:
            return
        try:
            self.recovery(workflow_id)
        except Exception:  # noqa: BLE001 - 追赶失败不改变判定
            return

    def held(self, workflow_id: str) -> CommitWatermark | None:
        """本地耐久持有且**自洽**的前缀；不完整（例如刚截断过）时为 None。"""
        with self._lock:
            try:
                status = self.replica.recovery_status(str(workflow_id))
            except Exception:  # noqa: BLE001 - 读不出来就当「不持有」，绝不猜
                return None
        if not status.get("verified"):
            return None
        return CommitWatermark(
            workflow_id=str(workflow_id),
            last_sequence=int(status["durable_sequence"]),
            checkpoint_digest=str(status["checkpoint_digest"]),
        )

    def propose(self) -> tuple[CommitWatermark, ...]:
        """leader 提议：只声明**已经自洽**的 workflow（读不到的就不承诺）。"""
        proposed = []
        for workflow_id in self.workflows:
            item = self.held(workflow_id)
            if item is not None:
                proposed.append(item)
        return tuple(proposed)

    def enforce(self, certificate: Any) -> None:
        """接管前核对：先截断未提交尾部，再要求本地前缀 ≥ 且一致于证书承诺。"""
        watermarks = tuple(getattr(certificate, "commit_watermark", ()) or ())
        if not watermarks:
            return
        for watermark in watermarks:
            workflow_id = watermark.workflow_id
            with self._lock:
                prefix = self.replica.durable_prefix(workflow_id)
                if int(prefix["last_sequence"]) > int(watermark.last_sequence):
                    result = self.replica.truncate_to(
                        workflow_id, last_sequence=int(watermark.last_sequence),
                    )
                    self.truncated[workflow_id] = int(result.get("removed_events", 0))
                held = self.held(workflow_id)
                if held is None or held.last_sequence < int(watermark.last_sequence):
                    # 落后（或截断后不自洽）⇒ 先尝试**主动追赶一次**，再判定。
                    self._recover(workflow_id)
                    held = self.held(workflow_id)
            if held is None:
                raise AutoRoleRuntimeError(
                    "committed_prefix_behind",
                    f"replica prefix for {workflow_id} is not self-consistent yet",
                )
            if held.last_sequence < watermark.last_sequence:
                raise AutoRoleRuntimeError(
                    "committed_prefix_behind",
                    f"replica prefix {held.last_sequence} < committed {watermark.last_sequence}",
                )
            if (
                held.last_sequence == watermark.last_sequence
                and held.checkpoint_digest != watermark.checkpoint_digest
            ):
                raise AutoRoleRuntimeError(
                    "committed_prefix_forked",
                    f"replica prefix diverges from the committed point {watermark.last_sequence}",
                )


class AutoRoleRuntime:
    """Configured local voter, collector, and local RPC handler."""

    def __init__(
        self,
        *,
        voter: QuorumVoter,
        collector: QuorumCollector,
        secret: str,
        commit_watermark_provider: Any = None,
        committed_prefix_guard: Any = None,
    ) -> None:
        self.voter = voter
        self.collector = collector
        self.secret = str(secret)
        #: #93：透传给 AutoRoleController 的提交证明来源与接管核对钩子。
        self.commit_watermark_provider = commit_watermark_provider
        self.committed_prefix_guard = committed_prefix_guard
        self._renewal_lock = threading.Lock()
        self._renewal_thread: threading.Thread | None = None
        self._renewal_stop: threading.Event | None = None

    @property
    def available_voter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.collector.voters))

    def authenticate(self, provided: str | None) -> bool:
        return bool(self.secret) and hmac.compare_digest(str(provided or ""), self.secret)

    def start_lease_renewal(self, controller: AutoRoleController) -> None:
        with self._renewal_lock:
            if self._renewal_thread is not None and self._renewal_thread.is_alive():
                return
            stop = threading.Event()
            self._renewal_stop = stop
            self._renewal_thread = threading.Thread(
                target=self._renew_lease_loop,
                args=(controller, stop),
                name="auto-role-lease-renewal",
                daemon=True,
            )
            self._renewal_thread.start()

    def stop_lease_renewal(self) -> None:
        with self._renewal_lock:
            stop = self._renewal_stop
            thread = self._renewal_thread
            self._renewal_stop = None
            self._renewal_thread = None
            if stop is not None:
                stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

    def _renew_lease_loop(self, controller: AutoRoleController, stop: threading.Event) -> None:
        interval = max(0.01, min(self.collector.policy.lease_duration_ms / 3000.0, 5.0))
        while not stop.wait(interval):
            if stop.is_set() or controller.state != "leader":
                continue
            try:
                decision = controller.renew_lease(available_voter_ids=self.available_voter_ids)
            except Exception:
                if not stop.is_set():
                    controller.on_quorum_loss()
                logger.exception("auto-role lease renewal failed closed")
                continue
            if not decision.accepted:
                logger.warning("auto-role lease renewal fenced: %s", decision.reason)

    def handle_rpc(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Handle one authenticated local-voter RPC without exposing exceptions."""
        try:
            operation = str(payload.get("op", ""))
            now_ms = payload.get("now_ms")
            if operation == "reserve_term":
                return {"ok": True, "term": self.voter.reserve_term(str(payload["leader_id"]), now_ms=now_ms)}
            if operation == "prepare":
                self.voter.prepare(str(payload["leader_id"]), int(payload["term"]), now_ms=now_ms)
                return {"ok": True}
            if operation == "sign_vote":
                from cluster_control_contract import QuorumCertificate

                certificate = QuorumCertificate.from_dict(payload["certificate"])
                return {"ok": True, "signature": self.voter.sign_vote(certificate, now_ms=now_ms).to_dict()}
            if operation == "commit_certificate":
                from cluster_control_contract import QuorumCertificate

                self.voter.commit_certificate(QuorumCertificate.from_dict(payload["certificate"]), now_ms=now_ms)
                return {"ok": True}
            if operation == "snapshot":
                return {"ok": True, "snapshot": self.voter.snapshot().to_dict()}
            return {"ok": False, "code": "quorum_unavailable"}
        except (KeyError, TypeError, ValueError, QuorumError, ControlContractError):
            return {"ok": False, "code": "quorum_unavailable"}


def build_auto_role_runtime(*, fence: ControlFence | None = None) -> AutoRoleRuntime:
    """Build a runtime from explicit environment configuration.

    ``QLH_CONTROL_VOTER_SET`` and a private key are intentionally mandatory.
    A deployment may list peer API base URLs in
    ``QLH_CONTROL_VOTER_PEERS`` as ``{"voter-b": "https://host:8000"}``.
    """
    raw_voter_set = _json_env("QLH_CONTROL_VOTER_SET", None)
    if not isinstance(raw_voter_set, Mapping):
        raise AutoRoleRuntimeError("auto_role_voter_set_missing", "QLH_CONTROL_VOTER_SET is required")
    try:
        voter_set = VoterSet.from_dict(raw_voter_set)
    except (ControlContractError, TypeError, ValueError) as exc:
        raise AutoRoleRuntimeError("auto_role_voter_set_invalid", "configured voter set is invalid") from exc
    voter_id = os.environ.get("QLH_CONTROL_VOTER_ID", "").strip()
    if not voter_id or voter_id not in voter_set.voter_map:
        raise AutoRoleRuntimeError("auto_role_voter_id_invalid", "QLH_CONTROL_VOTER_ID is not in voter set")
    if fence is not None and fence.authority is not None and fence.authority.voter_set != voter_set:
        raise AutoRoleRuntimeError("auto_role_authority_mismatch", "control fence voter set differs from auto-role voter set")
    private_key = _private_key_from_environment()
    ledger_path = os.environ.get("QLH_CONTROL_VOTER_LEDGER", "").strip()
    if not ledger_path:
        ledger_path = str(Path(os.environ.get("QLH_STATE_DIR", ".qlh-state")).expanduser() / f"quorum-{voter_id}.sqlite3")
    # #93：若部署给了本地 journal 复制副本，就用它当「已提交前缀」的权威：
    # ① voter 签名前自检是否耐久持有；② leader 提议水位；③ 接管前核对（截断/拒绝）。
    replica_path = os.environ.get("QLH_JOURNAL_REPLICA", "").strip()
    prefix_authority: JournalPrefixAuthority | None = None
    if replica_path:
        workflows = [
            item.strip()
            for item in os.environ.get("QLH_COMMIT_WATERMARK_WORKFLOWS", "").split(",")
            if item.strip()
        ]
        replica = DurableJournalReplica(
            replica_path,
            source_node_id=os.environ.get("QLH_JOURNAL_SOURCE_NODE_ID", "").strip() or voter_id,
            stream_id=os.environ.get("QLH_JOURNAL_STREAM_ID", "").strip() or "journal-1",
        )
        # #93：配置了 quorum peers 就意味着「落后时可以主动追赶」——从对端拉回一个
        # 已自洽的 checkpoint 再重试接管核对；没配就保持「拒绝接管」，绝不凑合上线。
        replication_settings = JournalReplicationSettings.from_env()
        recovery = None
        if replication_settings.quorum_peers:

            def recovery(  # noqa: F811 - 有意覆盖上面的 None
                workflow_id: str,
                _replica: DurableJournalReplica = replica,
                _settings: JournalReplicationSettings = replication_settings,
            ) -> None:
                recover_journal_prefix(_replica, workflow_id=workflow_id, settings=_settings)

        prefix_authority = JournalPrefixAuthority(
            replica, workflows=workflows, recovery=recovery,
        )
    local = QuorumVoter(
        voter_id=voter_id,
        private_key=private_key,
        voter_set=voter_set,
        ledger=SQLiteVoterLedger(ledger_path, voter_id=voter_id, cluster_id=voter_set.cluster_id, voter_set_epoch=voter_set.voter_set_epoch),
        watermark_source=prefix_authority.held if prefix_authority is not None else None,
    )
    secret = os.environ.get("QLH_CONTROL_VOTER_RPC_SECRET", "") or os.environ.get("QLH_CLUSTER_SECRET", "")
    if not secret:
        raise AutoRoleRuntimeError("auto_role_rpc_secret_missing", "voter RPC requires a shared secret")
    peers = _json_env("QLH_CONTROL_VOTER_PEERS", {})
    if not isinstance(peers, Mapping):
        raise AutoRoleRuntimeError("auto_role_peers_invalid", "QLH_CONTROL_VOTER_PEERS must be a JSON object")
    lease_ms = int(os.environ.get("QLH_CONTROL_VOTER_LEASE_MS", "10000"))
    if lease_ms < 1000:
        raise AutoRoleRuntimeError("auto_role_lease_too_short", "auto-role lease must be at least one second")
    timeout = float(os.environ.get("QLH_CONTROL_VOTER_RPC_TIMEOUT", AUTO_ROLE_RPC_TIMEOUT_SECONDS))
    if not math.isfinite(timeout) or timeout <= 0:
        raise AutoRoleRuntimeError("auto_role_rpc_timeout_invalid", "voter RPC timeout must be positive and finite")
    peer_count = max(1, len(peers))
    timeout = min(timeout, lease_ms / (8000.0 * peer_count))
    voters: dict[str, Any] = {voter_id: local}
    for peer_id, endpoint in peers.items():
        peer_id = str(peer_id)
        if peer_id == voter_id:
            continue
        if peer_id not in voter_set.voter_map:
            raise AutoRoleRuntimeError("auto_role_peer_invalid", "voter peer is not in voter set")
        voters[peer_id] = HTTPRemoteVoter(peer_id, str(endpoint), secret=secret, timeout=timeout)
    if len(voters) < voter_set.quorum_size:
        raise AutoRoleRuntimeError("auto_role_peers_incomplete", "configured voter peers cannot form a quorum")
    collector = QuorumCollector(voter_set=voter_set, voters=voters, policy=QuorumPolicy(lease_duration_ms=lease_ms))
    return AutoRoleRuntime(
        voter=local,
        collector=collector,
        secret=secret,
        commit_watermark_provider=prefix_authority.propose if prefix_authority is not None else None,
        committed_prefix_guard=prefix_authority.enforce if prefix_authority is not None else None,
    )


def install_auto_role_controller(scheduler: Any, fence: ControlFence | None = None) -> AutoRoleRuntime | None:
    """Attach an auto controller only when ``QLH_NODE_ROLE=auto`` is explicit."""
    if os.environ.get("QLH_NODE_ROLE", "").strip().lower() != "auto":
        return None
    runtime = build_auto_role_runtime(fence=fence)
    # ``auto`` is a controller mode, not a static role in the control
    # contract.  Keep the authority's compatibility role ``unknown`` while
    # the AutoRoleController owns the runtime role state.
    authority = fence.authority if fence is not None and fence.authority is not None else ControlPlaneAuthority(
        cluster_id=runtime.collector.voter_set.cluster_id,
        voter_set=runtime.collector.voter_set,
        static_role="unknown",
    )
    if fence is None or fence.authority is None or not fence.required:
        # A direct Scheduler composition still needs the same write fence as
        # the API composition root; auto mode must never become an unfenced
        # static master merely because the API module was not imported.
        fence = ControlFence(authority, required=True)
        scheduler.set_control_fence(fence)
    controller = AutoRoleController(
        runtime.voter.voter_id, mode="auto", authority=authority,
        collector=runtime.collector, fence=fence,
        commit_watermark_provider=runtime.commit_watermark_provider,
        committed_prefix_guard=runtime.committed_prefix_guard,
    )
    fence.set_role_guard(controller.can_write)
    scheduler.set_auto_role_controller(controller)
    scheduler._auto_role_runtime = runtime
    return runtime


__all__ = [
    "AUTO_ROLE_RPC_HEADER",
    "AUTO_ROLE_RPC_PATH",
    "AutoRoleRuntime",
    "AutoRoleRuntimeError",
    "HTTPRemoteVoter",
    "JournalPrefixAuthority",
    "build_auto_role_runtime",
    "install_auto_role_controller",
]
