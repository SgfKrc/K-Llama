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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cluster_auto_role import AutoRoleController, AutoRoleDecision
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


def _private_key_from_environment(environment=None) -> Any:
    environment = os.environ if environment is None else environment
    raw = environment.get("QLH_CONTROL_VOTER_PRIVATE_KEY", "")
    path = environment.get("QLH_CONTROL_VOTER_PRIVATE_KEY_FILE", "").strip()
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


def _json_env(name: str, default: Any, environment=None) -> Any:
    environment = os.environ if environment is None else environment
    raw = environment.get(name, "").strip()
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
        # Cluster credentials travel directly to the configured peer, independent
        # of user download proxies or urllib's process-global cached opener.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

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
            with self._opener.open(request, timeout=self._timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise QuorumError("quorum_unavailable", "voter peer is unreachable") from exc
        if not isinstance(value, dict) or not value.get("ok", False):
            code = str(value.get("code", "quorum_unavailable")) if isinstance(value, dict) else "quorum_unavailable"
            # ★ #93（真机演练暴露）：白名单**必须**含水位类错误码，否则「voter 不持有被
            #   提议的已提交前缀」会被降级成通用的 quorum_unavailable，主节点就无法区分
            #   「多数不可用」与「多数不持有」—— 这正是本票要消除的歧义。
            if code not in {
                "quorum_unavailable", "quorum_vote_rejected", "quorum_active_lease",
                "quorum_term_stale", "quorum_duplicate_vote", "quorum_expiry_regressed",
                "quorum_watermark_unheld", "quorum_watermark_unavailable",
            }:
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

    def leader_state(self):
        return self._call({"op": "leader_certificate"})

    def leader_certificate(self):
        return self.leader_state().get("certificate")


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
        self.floor_provider = lambda: ()
        self.model_control = None
        self.requests = None

    def _recover(self, workflow_id: str, *, permit=None) -> None:
        """落后时主动追赶一次；追赶失败不改结论（仍然是「不够格接管」）。"""
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
        except Exception:  # noqa: BLE001 - 追赶失败不改变判定
            return

    def held(self, workflow_id: str) -> CommitWatermark | None:
        """本地耐久持有且**自洽**的前缀；不完整（例如刚截断过）时为 None。"""
        from qlh_core.contracts.deployment import MODEL_CONTROL_DOMAIN
        from qlh_core.contracts.execution import REQUEST_DOMAIN
        if workflow_id in {MODEL_CONTROL_DOMAIN, REQUEST_DOMAIN}:
            owner = self.model_control if workflow_id == MODEL_CONTROL_DOMAIN else self.requests
            if owner is None:
                return None
            status = owner.repository.recovery_status()
            return (CommitWatermark(workflow_id=workflow_id, last_sequence=status["durable_sequence"],
                                    checkpoint_digest=status["checkpoint_digest"])
                    if status.get("verified") else None)
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
        # Durable signed/committed domains must survive configuration edits and
        # read failures. An unavailable floor prevents renewal instead of shrinking it.
        floors = {item.workflow_id: item for item in self.floor_provider()}
        proposed = []
        for workflow_id in sorted(set(self.workflows) | set(floors)):
            item = self.held(workflow_id)
            floor = floors.get(workflow_id)
            if floor is not None and (item is None or item.last_sequence < floor.last_sequence
                    or (item.last_sequence == floor.last_sequence and item.checkpoint_digest != floor.checkpoint_digest)):
                raise AutoRoleRuntimeError("committed_prefix_behind", "required recovery floor is unavailable")
            if item is not None:
                proposed.append(item)
        return tuple(proposed)

    def enforce(self, certificate: Any, *, permit=None) -> None:
        """接管前核对：先截断未提交尾部，再要求本地前缀 ≥ 且一致于证书承诺。"""
        watermarks = tuple(getattr(certificate, "commit_watermark", ()) or ())
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
                    raise AutoRoleRuntimeError("committed_prefix_behind", "owner recovery domain is unavailable")
                continue
            with self._lock:
                prefix = self.replica.durable_prefix(workflow_id)
                if int(prefix["last_sequence"]) > int(watermark.last_sequence):
                    guarded = {"permission_check": permit.check} if permit is not None and isinstance(self.replica, DurableJournalReplica) else {}
                    truncate = lambda: self.replica.truncate_to(workflow_id, last_sequence=int(watermark.last_sequence), **guarded)
                    result = permit.mutate(truncate) if permit is not None else truncate()
                    self.truncated[workflow_id] = int(result.get("removed_events", 0))
                held = self.held(workflow_id)
                if held is None or held.last_sequence < int(watermark.last_sequence):
                    # 落后（或截断后不自洽）⇒ 先尝试**主动追赶一次**，再判定。
                    self._recover(workflow_id, permit=permit)
                    held = self.held(workflow_id)
            if held is None:
                raise AutoRoleRuntimeError(
                    "committed_prefix_behind",
                    f"replica prefix for {workflow_id} is not self-consistent yet",
                )
            if held.last_sequence != watermark.last_sequence:
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
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.voter = voter
        self.collector = collector
        self.secret = str(secret)
        self._clock = clock if clock is not None else lambda: int(time.time() * 1000)
        #: #93：透传给 AutoRoleController 的提交证明来源与接管核对钩子。
        self.commit_watermark_provider = commit_watermark_provider
        self.committed_prefix_guard = committed_prefix_guard
        self._renewal_lock = threading.Lock()
        self._renewal_thread: threading.Thread | None = None
        self._renewal_stop: threading.Event | None = None
        self._controller: AutoRoleController | None = None
        self.tcp_port = 0
        self.leader_ready = lambda: True

    @property
    def available_voter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.collector.voters))

    def authenticate(self, provided: str | None) -> bool:
        return bool(self.secret) and hmac.compare_digest(str(provided or ""), self.secret)

    def start_lease_renewal(self, controller: AutoRoleController, *, on_decision=None) -> None:
        with self._renewal_lock:
            if self._renewal_thread is not None and self._renewal_thread.is_alive():
                return
            self._controller = controller
            stop = threading.Event()
            self._renewal_stop = stop
            self._renewal_thread = threading.Thread(
                target=self._renew_lease_loop,
                args=(controller, stop, on_decision),
                name="auto-role-lease-renewal",
                daemon=True,
            )
            self._renewal_thread.start()

    def stop_lease_renewal(self, timeout: float = 30.0) -> None:
        with self._renewal_lock:
            stop, thread = self._renewal_stop, self._renewal_thread
            if stop is not None:
                stop.set()
        if thread is not None:
            if thread is threading.current_thread():
                raise RuntimeError("renewal thread cannot close its own resources")
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError("authority renewal is still draining")
        with self._renewal_lock:
            if self._renewal_thread is thread:
                self._renewal_stop = None
                self._renewal_thread = None

    def close(self) -> None:
        self.stop_lease_renewal()
        replica = getattr(self, "_replica", None)
        if replica is not None:
            replica.close()
            self._replica = None

    def observe_leader(self, controller: AutoRoleController) -> AutoRoleDecision | None:
        """Consume a configured peer's signed proof through the existing controller."""
        from cluster_control_contract import QuorumCertificate, validate_certificate

        candidates = []
        for voter in tuple(self.collector.voters.values()):
            read = getattr(voter, "leader_certificate", None)
            if not callable(read):
                continue
            try:
                value = read()
                if value is None:
                    continue
                certificate = QuorumCertificate.from_dict(value)
                validate_certificate(certificate, self.collector.voter_set, now_ms=self._clock())
                current = controller.certificate
                if current is not None and certificate.term < current.term:
                    continue
                if (current is not None and certificate.term == current.term
                        and certificate.expires_at_ms < current.expires_at_ms):
                    continue
                if certificate.leader_id != controller.node_id:
                    candidates.append(certificate)
            except Exception:
                # A failed peer read never grants authority or erases a live proof.
                continue
        for certificate in sorted(candidates, key=lambda c: (c.term, c.expires_at_ms), reverse=True):
            current = controller.certificate
            if (controller.state == "follower" and current is not None
                    and current.digest() == certificate.digest()):
                return AutoRoleDecision(True, "follower", "client", "valid_certificate_other_leader",
                                        current.term, current.leader_id, current.digest(), current.lease_id)
            decision = controller.on_reconnect(certificate=certificate, now_ms=self._clock())
            if decision.accepted:
                return decision
        return None

    def leader_endpoint(self, leader_id):
        from cluster_control_contract import QuorumCertificate, validate_certificate
        from urllib.parse import urlsplit

        peer = self.collector.voters.get(leader_id)
        if not isinstance(peer, HTTPRemoteVoter):
            return None
        try:
            value = peer.leader_state()
            certificate = QuorumCertificate.from_dict(value["certificate"])
            validate_certificate(certificate, self.collector.voter_set, now_ms=self._clock())
        except (QuorumError, ControlContractError, KeyError, TypeError, ValueError):
            return None
        if certificate.leader_id != leader_id:
            return None
        port = value.get("tcp_port")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            return None
        return urlsplit(peer.endpoint).hostname, port

    def supervise_once(self, controller: AutoRoleController) -> AutoRoleDecision | None:
        if controller.state == "stopped" or getattr(controller, "mode", "auto") != "auto":
            return None
        if controller.state == "leader":
            return controller.renew_lease(available_voter_ids=self.available_voter_ids)
        observed = self.observe_leader(controller)
        if observed is not None:
            return observed
        certificate = controller.certificate
        if controller.state == "follower" and certificate is not None and certificate.expires_at_ms > self._clock():
            return None
        return controller.on_reconnect(available_voter_ids=self.available_voter_ids)

    def _renew_lease_loop(self, controller: AutoRoleController, stop: threading.Event, on_decision=None) -> None:
        interval = max(0.01, min(self.collector.policy.lease_duration_ms / 3000.0, 5.0))
        while not stop.wait(interval):
            if stop.is_set():
                return
            try:
                decision = self.supervise_once(controller)
                if stop.is_set():
                    return
                if on_decision is not None and decision is not None:
                    on_decision(decision)
            except Exception:
                if not stop.is_set():
                    fenced = controller.on_quorum_loss()
                    if on_decision is not None:
                        try:
                            on_decision(fenced)
                        except Exception:
                            logger.exception("auto-role fenced resource notification failed")
                logger.exception("auto-role supervision failed closed")
                continue
            if decision is not None and not decision.accepted:
                logger.debug("auto-role supervision waiting: %s", decision.reason)

    def handle_rpc(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Handle one authenticated local-voter RPC without exposing exceptions."""
        try:
            operation = str(payload.get("op", ""))
            # The wire timestamp is untrusted. Lease exclusion and expiry use
            # a fresh receiver-local sample for every voter operation.
            if operation == "reserve_term":
                return {"ok": True, "term": self.voter.reserve_term(str(payload["leader_id"]), now_ms=self._clock())}
            if operation == "prepare":
                self.voter.prepare(str(payload["leader_id"]), int(payload["term"]), now_ms=self._clock())
                return {"ok": True}
            if operation == "sign_vote":
                from cluster_control_contract import QuorumCertificate

                certificate = QuorumCertificate.from_dict(payload["certificate"])
                return {"ok": True, "signature": self.voter.sign_vote(certificate, now_ms=self._clock()).to_dict()}
            if operation == "commit_certificate":
                from cluster_control_contract import QuorumCertificate

                self.voter.commit_certificate(QuorumCertificate.from_dict(payload["certificate"]), now_ms=self._clock())
                return {"ok": True}
            if operation == "snapshot":
                return {"ok": True, "snapshot": self.voter.snapshot().to_dict()}
            if operation == "leader_certificate":
                controller = self._controller
                certificate = controller.certificate if controller is not None and controller.can_write() and self.leader_ready() else None
                return {"ok": True, "certificate": certificate.to_dict() if certificate is not None else None, "tcp_port": self.tcp_port}
            return {"ok": False, "code": "quorum_unavailable"}
        except (QuorumError, ControlContractError) as exc:
            # ★ #93（真机演练暴露）：**保留稳定契约码**。一律返回 quorum_unavailable 会让
            #   远程调用方把「voter 不持有被提议的已提交前缀」误当成容量/网络问题，
            #   从而掩盖一致性风险 —— 与 collector 侧的区分逻辑配套。
            return {"ok": False, "code": str(getattr(exc, "code", "") or "quorum_unavailable")}
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "code": "quorum_unavailable"}


def build_auto_role_runtime(*, fence: ControlFence | None = None, environment=None, model_control=None, requests=None) -> AutoRoleRuntime:
    """Build a runtime from explicit environment configuration.

    ``QLH_CONTROL_VOTER_SET`` and a private key are intentionally mandatory.
    A deployment may list peer API base URLs in
    ``QLH_CONTROL_VOTER_PEERS`` as ``{"voter-b": "https://host:8000"}``.
    """
    environment = dict(os.environ if environment is None else environment)
    raw_voter_set = _json_env("QLH_CONTROL_VOTER_SET", None, environment)
    if not isinstance(raw_voter_set, Mapping):
        raise AutoRoleRuntimeError("auto_role_voter_set_missing", "QLH_CONTROL_VOTER_SET is required")
    try:
        voter_set = VoterSet.from_dict(raw_voter_set)
    except (ControlContractError, TypeError, ValueError) as exc:
        raise AutoRoleRuntimeError("auto_role_voter_set_invalid", "configured voter set is invalid") from exc
    voter_id = environment.get("QLH_CONTROL_VOTER_ID", "").strip()
    if not voter_id or voter_id not in voter_set.voter_map:
        raise AutoRoleRuntimeError("auto_role_voter_id_invalid", "QLH_CONTROL_VOTER_ID is not in voter set")
    if fence is not None and fence.authority is not None and fence.authority.voter_set != voter_set:
        raise AutoRoleRuntimeError("auto_role_authority_mismatch", "control fence voter set differs from auto-role voter set")
    private_key = _private_key_from_environment(environment)
    replication_settings = JournalReplicationSettings.from_env(environment)
    secret = environment.get("QLH_CONTROL_VOTER_RPC_SECRET", "") or environment.get("QLH_CLUSTER_SECRET", "")
    if not secret:
        raise AutoRoleRuntimeError("auto_role_rpc_secret_missing", "voter RPC requires a shared secret")
    peers = _json_env("QLH_CONTROL_VOTER_PEERS", {}, environment)
    if not isinstance(peers, Mapping):
        raise AutoRoleRuntimeError("auto_role_peers_invalid", "QLH_CONTROL_VOTER_PEERS must be a JSON object")
    lease_ms = int(environment.get("QLH_CONTROL_VOTER_LEASE_MS", "10000"))
    if lease_ms < 1000:
        raise AutoRoleRuntimeError("auto_role_lease_too_short", "auto-role lease must be at least one second")
    timeout = float(environment.get("QLH_CONTROL_VOTER_RPC_TIMEOUT", AUTO_ROLE_RPC_TIMEOUT_SECONDS))
    if not math.isfinite(timeout) or timeout <= 0:
        raise AutoRoleRuntimeError("auto_role_rpc_timeout_invalid", "voter RPC timeout must be positive and finite")
    peer_count = max(1, len(peers))
    timeout = min(timeout, lease_ms / (8000.0 * peer_count))
    voters: dict[str, Any] = {}
    for peer_id, endpoint in peers.items():
        peer_id = str(peer_id)
        if peer_id == voter_id:
            continue
        if peer_id not in voter_set.voter_map:
            raise AutoRoleRuntimeError("auto_role_peer_invalid", "voter peer is not in voter set")
        voters[peer_id] = HTTPRemoteVoter(peer_id, str(endpoint), secret=secret, timeout=timeout)
    if len(voters) + 1 < voter_set.quorum_size:
        raise AutoRoleRuntimeError("auto_role_peers_incomplete", "configured voter peers cannot form a quorum")
    ledger_path = environment.get("QLH_CONTROL_VOTER_LEDGER", "").strip()
    if not ledger_path:
        ledger_path = str(Path(environment.get("QLH_STATE_DIR", ".qlh-state")).expanduser() / f"quorum-{voter_id}.sqlite3")
    # #93：若部署给了本地 journal 复制副本，就用它当「已提交前缀」的权威：
    # ① voter 签名前自检是否耐久持有；② leader 提议水位；③ 接管前核对（截断/拒绝）。
    replica_path = environment.get("QLH_JOURNAL_REPLICA", "").strip()
    prefix_authority: JournalPrefixAuthority | None = None
    replica = None
    try:
        if replica_path:
            workflows = [
                item.strip()
                for item in environment.get("QLH_COMMIT_WATERMARK_WORKFLOWS", "").split(",")
                if item.strip()
            ]
            replica = DurableJournalReplica(
                replica_path,
                source_node_id=environment.get("QLH_JOURNAL_SOURCE_NODE_ID", "").strip() or voter_id,
                stream_id=environment.get("QLH_JOURNAL_STREAM_ID", "").strip() or "journal-1",
            )
            # #93：配置了 quorum peers 就意味着「落后时可以主动追赶」——从对端拉回一个
            # 已自洽的 checkpoint 再重试接管核对；没配就保持「拒绝接管」，绝不凑合上线。
            recovery = None
            if replication_settings.quorum_peers:

                def recovery(  # noqa: F811 - 有意覆盖上面的 None
                    workflow_id: str,
                    *, permit=None,
                    _replica: DurableJournalReplica = replica,
                    _settings: JournalReplicationSettings = replication_settings,
                ) -> None:
                    recover_journal_prefix(_replica, workflow_id=workflow_id, settings=_settings,
                                           mutation_guard=permit.mutate if permit is not None else None,
                                           permission_check=permit.check if permit is not None else None)

            prefix_authority = JournalPrefixAuthority(
                replica, workflows=workflows, recovery=recovery,
            )
        ledger = SQLiteVoterLedger(ledger_path, voter_id=voter_id, cluster_id=voter_set.cluster_id, voter_set_epoch=voter_set.voter_set_epoch)
        if prefix_authority is None:
            # Even without a graph replica, preserve the complete durable floor.
            class UnavailableReplica:
                def recovery_status(self, domain):
                    return {"verified": False}
                def close(self):
                    pass
            prefix_authority = JournalPrefixAuthority(UnavailableReplica())
        prefix_authority.floor_provider = ledger.recovery_floors
        prefix_authority.model_control = model_control
        prefix_authority.requests = requests
        local = QuorumVoter(
            voter_id=voter_id,
            private_key=private_key,
            voter_set=voter_set,
            ledger=ledger,
            watermark_source=prefix_authority.held if prefix_authority is not None else None,
        )
    except BaseException:
        if replica is not None:
            replica.close()
        raise
    voters[voter_id] = local
    collector = QuorumCollector(voter_set=voter_set, voters=voters, policy=QuorumPolicy(lease_duration_ms=lease_ms))
    result = AutoRoleRuntime(
        voter=local,
        collector=collector,
        secret=secret,
        commit_watermark_provider=prefix_authority.propose if prefix_authority is not None else None,
        committed_prefix_guard=prefix_authority.enforce if prefix_authority is not None else None,
    )
    result._replica = prefix_authority.replica if prefix_authority is not None else None
    return result


def install_auto_role_controller(scheduler: Any, fence: ControlFence | None = None, *, environment=None) -> AutoRoleRuntime | None:
    """Attach an auto controller only when ``QLH_NODE_ROLE=auto`` is explicit."""
    environment = dict(os.environ if environment is None else environment)
    if environment.get("QLH_NODE_ROLE", "").strip().lower() != "auto":
        return None
    runtime = build_auto_role_runtime(fence=fence, environment=environment,
                                      model_control=getattr(scheduler, "_model_control", None),
                                      requests=getattr(getattr(scheduler, "_execution_app", None), "requests", None))
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
    fence.set_role_guard(lambda: controller.can_write() and getattr(scheduler, "_auto_role_binding_ready", True))
    fence.set_peer_guard(lambda: controller.can_follow() and
                         (controller.state != "leader" or getattr(scheduler, "_auto_role_binding_ready", True)),
                         consume_certificate=controller.observe_peer_certificate)
    runtime._controller = controller
    runtime.leader_ready = lambda: getattr(scheduler, "_auto_role_binding_ready", True)
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
