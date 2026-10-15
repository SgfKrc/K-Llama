"""TC-N2.4 task-worker control plane with physical admission pending."""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

from request_deadline import REQUEST_DEADLINE_EXCEEDED

from task_provider import (
    DEPENDENCY_FAILURES_KEY,
    ModelIdentity,
    ProviderBusy,
    ProviderCapabilities,
    ProviderExecutionError,
    ProviderReservationError,
    ProviderUnavailable,
    Reservation,
    StageAttempt,
    StageRequest,
    StageResult,
    canonical_model_engine,
)

from task_worker_chunks import (
    build_stage_chunk,
    plan_stage_payload_chunks,
    stage_chunk_ref,
)
from task_worker_protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    WorkerMessage,
    WorkerProtocolError,
    build_message,
    canonical_message_bytes,
    decode_message,
    hidden_fits_stage_frame,
    negotiate_protocol_version,
    stage_input_sha256,
)


_MESSAGE_CACHE_LIMIT = 1024

#: ★ 2026-10-07（DIST-NEXT-4b）：reservation 撤销的**单一 reason code**。
#: 每次撤销都带其中一个值记一条 `event=task_worker_reservations_released` ——
#: 否则「这个节点的槽位为什么被释放」在日志里说不清（审计 P0-3 要求单一 reason code）。
RELEASE_REASON_DISCONNECTED = "worker_tcp_disconnected"
RELEASE_REASON_HEARTBEAT_STALE = "worker_heartbeat_stale"
RELEASE_REASON_SERVICE_RESTART = "worker_service_restarted"
RELEASE_REASONS = (
    RELEASE_REASON_DISCONNECTED,
    RELEASE_REASON_HEARTBEAT_STALE,
    RELEASE_REASON_SERVICE_RESTART,
)


def remote_provider_id(node_id: str) -> str:
    """Return a stable Provider ID for one authenticated worker node."""
    raw = str(node_id or "")
    safe = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "_.-")
        else "_"
        for character in raw
    ).strip("._") or "worker"
    base = f"remote_{safe}"
    if len(base) <= 64 and safe == raw:
        return base
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"remote_{safe[:46]}_{digest}"[:64]


def _message_id(prefix: str) -> str:
    return f"msg_{prefix}{uuid.uuid4().hex}"


def _message_digest(message: WorkerMessage) -> str:
    return hashlib.sha256(canonical_message_bytes(message)).hexdigest()


def _canonical_capabilities(value: Any) -> str:
    """把 capabilities 折成可比字符串（dict 顺序无关）。

    用于「幂等 hello」：判断 worker 这次上报的能力与上次是否**内容相同**。
    不可 JSON 序列化的值退化为 `repr` —— 宁可判成"变了"（多重推一次层配置），
    也不要漏判（漏了就永远不推）。
    """
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(value)


class TaskWorkerControlPlane:
    """Own hello negotiation and health state without accepting Stage work."""

    def __init__(self, *, health_timeout_seconds: float = 120.0):
        self._health_timeout_seconds = max(1.0, float(health_timeout_seconds))
        self._workers: dict[str, dict[str, Any]] = {}
        self._coordinator: dict[str, Any] = {}
        self._worker_hello_pending = False
        self._worker_hello_owner: tuple | None = None
        # REGISTER ACK may precede the peer's task-worker hello. Fence that
        # connection from legacy layer assignment during this handshake.
        self._coordinator_connections: dict[str, dict[str, Any]] = {}
        self._seen: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self._seen_order: collections.deque[tuple[str, str]] = collections.deque()
        self._rejected_message_count = 0
        self._lock = threading.RLock()

    @staticmethod
    def _reject(code: str, message: str) -> WorkerProtocolError:
        return WorkerProtocolError(message, code=code, field="message_type")

    def _remember(
        self,
        peer_id: str,
        message: WorkerMessage,
        response: Optional[WorkerMessage] = None,
    ) -> None:
        key = (peer_id, message.message_id)
        response_snapshot = response.snapshot() if response is not None else {}
        self._seen[key] = (_message_digest(message), response_snapshot)
        self._seen_order.append(key)
        while len(self._seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._seen_order.popleft()
            self._seen.pop(expired, None)

    def _duplicate_response(
        self, peer_id: str, message: WorkerMessage,
    ) -> Optional[WorkerMessage]:
        cached = self._seen.get((peer_id, message.message_id))
        if cached is None:
            return None
        digest, response = cached
        if digest != _message_digest(message):
            self._rejected_message_count += 1
            raise self._reject(
                "message_id_conflict",
                "message_id was reused with different content",
            )
        if not response:
            return message
        return decode_message(response)

    def begin_worker_hello(
        self,
        *,
        node_id: str,
        capabilities: Mapping[str, Any],
        worker_kind: str = "pc_full_worker",
        sent_at_ms: Optional[int] = None,
        registration_owner: tuple | None = None,
    ) -> Optional[WorkerMessage]:
        """Build one v2 hello; overlapping negotiations are deliberately fenced."""
        now_ms = int(time.time() * 1000) if sent_at_ms is None else int(sent_at_ms)
        with self._lock:
            if self._worker_hello_pending:
                previous = self._worker_hello_owner
                if (registration_owner is None
                        or (previous is not None and previous[:3] == registration_owner)):
                    return None
                # A new authenticated registration supersedes a pending old hello,
                # even when that registration's disconnect callback was delayed.
            message = build_message(
                "hello",
                {
                    "node_id": node_id,
                    "worker_kind": worker_kind,
                    "min_version": PROTOCOL_VERSION,
                    "max_version": PROTOCOL_VERSION,
                    "capabilities": dict(capabilities),
                },
                message_id=_message_id("hello_"),
                sent_at_ms=now_ms,
                version=PROTOCOL_VERSION,
            )
            self._worker_hello_pending = True
            self._worker_hello_owner = (
                (*registration_owner, message.message_id)
                if registration_owner is not None else None
            )
            self._coordinator = {
                "node_id": "",
                "connected": True,
                "healthy": False,
                "accepted": False,
                "selected_version": 0,
                "hello_sent_at": now_ms / 1000.0,
                "last_transport_heartbeat": time.time(),
                "reason_code": "negotiation_pending",
            }
            return message

    def receive_on_coordinator(
        self,
        peer_id: str,
        raw: bytes | str | Mapping[str, Any],
        *,
        coordinator_node_id: str,
        sent_at_ms: Optional[int] = None,
        connection_generation: int | None = None,
    ) -> WorkerMessage:
        """Validate a worker hello and return a deterministic hello_ack."""
        message = decode_message(raw)
        with self._lock:
            connection = self._coordinator_connections.get(str(peer_id))
            if connection_generation is not None and (
                connection is None
                or not connection["connected"]
                or connection["generation"] != connection_generation
            ):
                raise self._reject("worker_connection_changed", "hello registration is no longer current")
            duplicate = self._duplicate_response(peer_id, message)
            if duplicate is not None:
                return duplicate
            if message.message_type != "hello":
                self._rejected_message_count += 1
                raise self._reject(
                    "control_plane_only",
                    "the coordinator hello handler accepts hello messages only",
                )

            payload = message.payload
            accepted = True
            selected_version = 0
            reason_code = ""
            if payload["node_id"] != peer_id:
                accepted = False
                reason_code = "node_identity_mismatch"
            else:
                try:
                    selected_version = negotiate_protocol_version(
                        payload["min_version"],
                        payload["max_version"],
                        local_min_version=PROTOCOL_VERSION,
                        local_max_version=PROTOCOL_VERSION,
                    )
                except WorkerProtocolError:
                    accepted = False
                    selected_version = 0
                    # ⚠️ 该 code 是稳定契约值（**不改**，避免破坏既有消费者）；
                    # 但语义是「双方版本区间无交集」，不是「必须 v2」——见 v3 层段引入后。
                    reason_code = "protocol_v2_required"

            ack_version = selected_version if accepted else message.version
            now_ms = (
                int(time.time() * 1000)
                if sent_at_ms is None else int(sent_at_ms)
            )
            ack = build_message(
                "hello_ack",
                {
                    "coordinator_node_id": coordinator_node_id,
                    "accepted": accepted,
                    "selected_version": selected_version,
                    "reason_code": reason_code,
                },
                message_id=_message_id("helloack_"),
                sent_at_ms=now_ms,
                version=ack_version,
            )
            now = time.time()
            previous = self._workers.get(peer_id)
            # ★ 幂等 hello（#28）：capabilities 的**内容**是否变化。协调方据它决定要不要
            #   重推层配置 —— 此前每次 hello 后都无条件
            #   `push_layer_config_to_clients()`（无内容比较），一旦层段路径也补
            #   `refresh_task_worker_capabilities()`，就形成
            #   `hello → push → load_layer_range → refresh → hello` 自激环：每次 push 都取
            #   新 generation，worker 端永远判成"新配置"。
            #   首次 hello（`previous is None`）等效"变化"，保证仍会推一次。
            connection_rebound = bool(connection and connection["pending"])
            capabilities_changed = connection_rebound or _canonical_capabilities(
                payload.get("capabilities")
            ) != _canonical_capabilities(
                previous.get("capabilities") if isinstance(previous, dict) else None
            )
            self._workers[peer_id] = {
                "node_id": peer_id,
                "worker_kind": payload["worker_kind"],
                "connected": True,
                "accepted": accepted,
                "selected_version": selected_version,
                "capabilities": payload["capabilities"],
                "capabilities_changed": capabilities_changed,
                "connection_rebound": connection_rebound,
                "hello_received_at": now,
                "last_transport_heartbeat": now,
                "reason_code": reason_code,
            }
            if not accepted:
                self._rejected_message_count += 1
            self._remember(peer_id, message, ack)
            return ack

    def receive_on_worker(
        self, raw: bytes | str | Mapping[str, Any],
    ) -> WorkerMessage:
        """Accept the coordinator's v2 hello acknowledgement."""
        message = decode_message(raw)
        with self._lock:
            duplicate = self._duplicate_response("coordinator", message)
            if duplicate is not None:
                return duplicate
            if message.message_type != "hello_ack":
                self._rejected_message_count += 1
                raise self._reject(
                    "control_plane_only",
                    "the worker hello handler accepts hello_ack messages only",
                )
            payload = message.payload
            if not self._worker_hello_pending:
                self._rejected_message_count += 1
                raise self._reject(
                    "unexpected_hello_ack",
                    "hello_ack arrived without a pending hello",
                )
            accepted = bool(payload["accepted"])
            if accepted and payload["selected_version"] != PROTOCOL_VERSION:
                self._rejected_message_count += 1
                raise self._reject(
                    "protocol_v2_required",
                    "the PC Full Worker adapter requires protocol v2",
                )
            now = time.time()
            self._coordinator = {
                "node_id": payload["coordinator_node_id"],
                "connected": True,
                "healthy": accepted,
                "accepted": accepted,
                "selected_version": payload["selected_version"],
                "hello_ack_received_at": now,
                "last_transport_heartbeat": now,
                "reason_code": payload["reason_code"],
            }
            self._worker_hello_pending = False
            if not accepted:
                self._rejected_message_count += 1
            self._remember("coordinator", message)
            return message

    def mark_worker_heartbeat(self, peer_id: str) -> None:
        with self._lock:
            worker = self._workers.get(peer_id)
            if worker is not None and worker.get("connected"):
                worker["last_transport_heartbeat"] = time.time()

    def record_rejection(self) -> None:
        """Account for outer transport or node-admission rejection."""
        with self._lock:
            self._rejected_message_count += 1

    def mark_worker_connection_pending(
        self, peer_id: str, *, connection_generation: int = 0,
    ) -> bool:
        with self._lock:
            peer_id = str(peer_id)
            current = self._coordinator_connections.get(peer_id)
            if current and connection_generation <= current["generation"]:
                # Neither a stale REGISTER nor a repeated post-ACK callback
                # may reset the handshake of a newer/already accepted socket.
                return False
            self._coordinator_connections[peer_id] = {
                "generation": connection_generation, "pending": True, "connected": True,
            }
            worker = self._workers.get(peer_id)
            if worker is not None:
                worker["connected"] = False
            return True

    def resolve_worker_connection_pending(
        self, peer_id: str, *, connection_generation: int | None = None,
    ) -> bool:
        with self._lock:
            peer_id = str(peer_id)
            current = self._coordinator_connections.get(peer_id)
            if connection_generation is not None and (
                current is None or not current["connected"]
                or current["generation"] != connection_generation
            ):
                return False
            if current is not None:
                current["pending"] = False
            return True

    def pending_worker_ids(self) -> set[str]:
        with self._lock:
            return {node for node, state in self._coordinator_connections.items()
                    if state["connected"] and state["pending"]}

    def connected_layer_stage_workers(
        self, peer_ids: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        """Return accepted layer-capable workers without the health overlay.

        Recovery must publish a fresh assignment before readiness is checked.
        The normal status projection is heartbeat-gated, so using it as the
        candidate source can deadlock recovery at ``not_configured``.  This
        still requires an accepted hello and a live task-worker connection;
        execution keeps the normal provider health gate.
        """
        allowed = {str(value) for value in peer_ids} if peer_ids is not None else None
        with self._lock:
            workers = []
            for node_id, worker in self._workers.items():
                if allowed is not None and node_id not in allowed:
                    continue
                if not worker.get("connected") or not worker.get("accepted"):
                    continue
                capabilities = worker.get("capabilities")
                if not isinstance(capabilities, dict):
                    continue
                stage_types = capabilities.get("stage_types", [])
                ranges = capabilities.get("layer_ranges")
                if (
                    worker.get("selected_version", 0) < 2
                    or "layer_forward" not in stage_types
                    or not ranges
                ):
                    continue
                workers.append({
                    "node_id": node_id,
                    "worker_kind": worker.get("worker_kind", ""),
                    "selected_version": worker.get("selected_version", 0),
                    "capabilities": dict(capabilities),
                    "connected": True,
                    "accepted": True,
                })
            return workers

    def mark_coordinator_heartbeat(self) -> None:
        with self._lock:
            if self._coordinator.get("connected"):
                self._coordinator["last_transport_heartbeat"] = time.time()

    def disconnect_worker(self, peer_id: str) -> None:
        with self._lock:
            connection = self._coordinator_connections.get(str(peer_id))
            if connection is not None:
                connection["connected"] = connection["pending"] = False
            worker = self._workers.get(peer_id)
            if worker is not None:
                worker["connected"] = False
                worker["disconnected_at"] = time.time()

    def disconnect_coordinator(self, *, owner: tuple | None = None) -> bool:
        """Retire a coordinator, optionally only for the hello that failed.

        Keep the owner after ACK: a delayed send failure must also distinguish
        successive hellos on the same registration, not just pending ones.
        """
        with self._lock:
            if owner is not None and self._worker_hello_owner != owner:
                return False
            if self._coordinator:
                self._coordinator["connected"] = False
                self._coordinator["healthy"] = False
                self._coordinator["disconnected_at"] = time.time()
            self._worker_hello_pending = False
            self._worker_hello_owner = None
            return True

    def worker_snapshot(self, peer_id: str) -> dict[str, Any]:
        """Return one public worker snapshot for Provider inspection."""
        now = time.time()
        with self._lock:
            worker = self._workers.get(peer_id)
            if worker is None:
                return {}
            snapshot = self._healthy_snapshot(worker, now)
            snapshot["provider_id"] = remote_provider_id(peer_id)
            return snapshot

    def coordinator_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if not self._coordinator:
                return {}
            return self._healthy_snapshot(self._coordinator, now)

    def _healthy_snapshot(self, peer: Mapping[str, Any], now: float) -> dict:
        snapshot = dict(peer)
        capabilities = snapshot.get("capabilities")
        if isinstance(capabilities, dict):
            snapshot["capabilities"] = {
                key: (
                    [dict(item) if isinstance(item, dict) else item for item in value]
                    if isinstance(value, list) else value
                )
                for key, value in capabilities.items()
            }
        last_seen = float(snapshot.get("last_transport_heartbeat", 0.0) or 0.0)
        snapshot["healthy"] = bool(
            snapshot.get("connected")
            and snapshot.get("accepted")
            and last_seen > 0
            and now - last_seen <= self._health_timeout_seconds
        )
        snapshot["task_dispatch_enabled"] = False
        snapshot["manual_stage_dispatch_enabled"] = bool(
            # ★ 2026-09-20：原为硬编码 `selected_version == 2`，协议升到 v3 后
            #   **永远为 False**（实测：能力正常但 provider 一直 unhealthy）。
            #   语义是「v2 及以上的数据面通道可用」，故与当前版本号解耦。
            snapshot["healthy"] and int(snapshot.get("selected_version") or 0) >= 2
        )
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
        stage_types = capabilities.get("stage_types", [])
        snapshot["layer_stage_dispatch_enabled"] = bool(
            snapshot["manual_stage_dispatch_enabled"]
            and "layer_forward" in stage_types
            and capabilities.get("layer_ranges")
        )
        return snapshot

    def status(self, *, role: str) -> dict[str, Any]:
        """Return public N2.4 control state before deployment gate overlay."""
        now = time.time()
        with self._lock:
            workers = []
            for node_id in sorted(self._workers):
                worker = self._healthy_snapshot(self._workers[node_id], now)
                worker["provider_id"] = remote_provider_id(node_id)
                workers.append(worker)
            coordinator = (
                self._healthy_snapshot(self._coordinator, now)
                if self._coordinator else {}
            )
            if role == "master":
                connected = any(item["healthy"] for item in workers)
            else:
                connected = bool(coordinator.get("healthy", False))
            return {
                "transport": "existing_tcp_length_prefixed",
                "transport_max_message_bytes": MAX_MESSAGE_BYTES,
                "phase": "TC-N2.4",
                "control_plane_ready": True,
                "control_plane_connected": connected,
                "adapter_connected": False,
                "task_dispatch_enabled": False,
                "manual_stage_dispatch_enabled": connected,
                "layer_stage_dispatch_enabled": any(
                    item.get("layer_stage_dispatch_enabled") for item in workers
                ) if role == "master" else connected,
                "lease_renew_enabled": connected,
                "stage_cancel_enabled": connected,
                "stage_message_replay_enabled": True,
                "auto_provider_selection_enabled": connected,
                "admission_state": "n2_4_physical_validation_pending",
                "connected_worker_count": sum(
                    bool(item["healthy"]) for item in workers
                ),
                "workers": workers,
                "coordinator": coordinator,
                "rejected_message_count": self._rejected_message_count,
            }


@dataclass
class _PendingRemoteAttempt:
    attempt: StageAttempt
    lease_expires_at: float
    lease_deadline_monotonic: float
    accepted: bool = False
    accept_event: threading.Event = field(default_factory=threading.Event)
    result_event: threading.Event = field(default_factory=threading.Event)
    cancel_ack_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[StageResult] = None
    error: Optional[BaseException] = None
    offer_sent: bool = False
    cancel_requested: bool = False
    cancel_enqueued: bool = False
    cancel_acknowledged: bool = False
    #: ★ 2026-10-07（DIST-NEXT-1）：「已收到取消 ACK」与「执行确实已停止」是两个
    #: 事实。前者由 `cancel_acknowledged` 表示；后者只有对端显式回报
    #: `execution_state == "execution_stopped"` 时才成立（缺省/旧对端 = 未证明）。
    cancel_ack_execution_state: str = ""
    cancel_execution_stopped: bool = False
    #: 对端**主动**取消（本地用户取消 / Service 回收），本端并未请求。
    cancel_remote_initiated: bool = False
    released: bool = False
    released_at: float = 0.0
    terminal_kind: str = ""
    terminal_decided_at_monotonic: float = 0.0
    _terminal_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False,
    )

    def decide_terminal(
        self,
        kind: str,
        *,
        result: Optional[StageResult] = None,
        error: Optional[BaseException] = None,
    ) -> bool:
        """Latch exactly one attempt outcome and wake every result waiter."""
        if (result is None) == (error is None):
            raise ValueError("attempt terminal requires exactly one outcome")
        with self._terminal_lock:
            if self.terminal_kind:
                return False
            self.terminal_kind = str(kind)
            self.terminal_decided_at_monotonic = time.monotonic()
            self.result = result
            self.error = error
            self.accept_event.set()
            self.result_event.set()
            return True

    def terminal_snapshot(
        self,
    ) -> tuple[str, Optional[StageResult], Optional[BaseException]]:
        with self._terminal_lock:
            return self.terminal_kind, self.result, self.error


class RemoteFullWorkerProvider:
    """Remote Provider whose results always enter TaskGraph fencing."""

    provider_kind = "remote_full_worker"

    def __init__(
        self,
        *,
        node_id: str,
        peer_snapshot: Callable[[], Mapping[str, Any]],
        send_message: Callable[[WorkerMessage], None],
        accept_timeout_seconds: float = 10.0,
    ):
        self.node_id = str(node_id)
        self.provider_id = remote_provider_id(self.node_id)
        self._peer_snapshot = peer_snapshot
        self._send_message = send_message
        self._accept_timeout_seconds = max(
            0.1, min(float(accept_timeout_seconds), 60.0)
        )
        self._reservations: dict[str, tuple[Reservation, StageRequest]] = {}
        self._executed_reservations: set[str] = set()
        self._pending: dict[str, _PendingRemoteAttempt] = {}
        self._reservation_attempts: dict[str, str] = {}
        self._seen_messages: dict[str, str] = {}
        self._seen_order: collections.deque[str] = collections.deque()
        #: ★ 2026-10-07（DIST-NEXT-1）：取消合同的分类计数。把「请求已发出」
        #: 「ACK 已收到」「执行确证已停止」「对端主动取消」「迟到结果被吸收」
        #: 分开统计 —— 否则「请求已取消」会被当成「执行已停止」。
        self._cancel_ack_states: collections.Counter[str] = collections.Counter()
        self._cancel_execution_stopped = 0
        self._cancel_remote_initiated = 0
        self._late_stage_responses = 0
        self._closed = False
        self._lock = threading.RLock()
        self._outbound_queue: queue.Queue[
            tuple[WorkerMessage, Callable[[Exception], None]]
        ] = queue.Queue(maxsize=64)
        self._outbound_stop = threading.Event()
        self._outbound_start_lock = threading.Lock()
        self._outbound_thread: Optional[threading.Thread] = None
        #: ★ 2026-10-11（三机实测）：**回程**分片装配器。与输入方向共用同一实现
        #:   （`StageChunkAssembler` 的语义、上限与错误码完全一致）。
        from task_worker_chunks import StageChunkAssembler

        self._result_chunk_assembler = StageChunkAssembler()

    def _absorb_result_chunk_locked(
        self, pending: "_PendingRemoteAttempt", payload: Mapping[str, Any],
    ) -> None:
        """累积一条回程分片（调用方持锁）。不合法即抛错，不落半条。"""
        encoded = payload.get("payload_b64")
        try:
            raw = base64.b64decode(str(encoded or ""), validate=True)
        except Exception as exc:  # noqa: BLE001 - 规范化为协议错误码
            raise WorkerProtocolError(
                "stage_result_chunk payload is not valid base64",
                code="invalid_chunk_payload",
                field="payload.payload_b64",
            ) from exc
        self._result_chunk_assembler.add(
            attempt_id=pending.attempt.attempt_id,
            chunk_index=int(payload.get("chunk_index", -1)),
            chunk_count=int(payload.get("chunk_count", 0)),
            payload=raw,
            payload_sha256=str(payload.get("payload_sha256") or ""),
        )
        # 落盘到达痕迹：之前只有"装配成功"才打日志，导致"分片没到"与"分片到了但没触发
        # 装配"两种情况无法区分（真机实测：ROG 侧已 `stage_result_chunked` + `sent`，
        # 而 master 侧既无装配日志也无错误，只能看到 60s 超时）。
        logger.info(
            "event=task_worker_stage_result_chunk_received attempt_id=%s index=%d count=%d bytes=%d",
            pending.attempt.attempt_id,
            int(payload.get("chunk_index", -1)),
            int(payload.get("chunk_count", 0)),
            len(raw),
        )

    def _expand_result_chunks_locked(
        self, pending: "_PendingRemoteAttempt", output: Any,
    ) -> Any:
        """把 `output.hidden_out_ref` 还原成内联 `hidden_out_f32`（调用方持锁）。

        分片未齐 ⇒ `incomplete_chunks`；摘要不符 ⇒ `chunk_digest_mismatch`（两者都 fail-closed）。
        还原成功后丢弃该 attempt 的分片状态。
        """
        if not isinstance(output, Mapping):
            return output
        ref = output.get("hidden_out_ref")
        if not isinstance(ref, Mapping):
            return output
        attempt_id = pending.attempt.attempt_id
        raw = self._result_chunk_assembler.assemble(attempt_id)
        self._result_chunk_assembler.discard(attempt_id)
        declared = str(ref.get("payload_sha256") or "").strip().lower()
        actual = hashlib.sha256(raw).hexdigest()
        if declared and declared != actual:
            raise WorkerProtocolError(
                "assembled stage result payload digest mismatch",
                code="chunk_digest_mismatch",
                field="payload.output.hidden_out_ref",
            )
        restored = dict(output)
        restored.pop("hidden_out_ref", None)
        restored["hidden_out_f32"] = base64.b64encode(raw).decode("ascii")
        logger.info(
            "event=task_worker_stage_result_chunks_assembled attempt_id=%s bytes=%d",
            attempt_id, len(raw),
        )
        return restored

    def _send_outbound_messages(self) -> None:
        while True:
            try:
                message, on_error = self._outbound_queue.get(timeout=0.2)
            except queue.Empty:
                with self._outbound_start_lock:
                    if self._outbound_queue.empty():
                        self._outbound_thread = None
                        return
                continue
            try:
                self._send_message(message)
                if message.message_type == "stage_cancel":
                    logger.info(
                        "event=task_worker_stage_cancel_sent node_id=%s "
                        "workflow_id=%s stage_id=%s attempt_id=%s",
                        self.node_id,
                        message.payload.get("workflow_id", ""),
                        message.payload.get("stage_id", ""),
                        message.payload.get("attempt_id", ""),
                    )
            except Exception as exc:
                try:
                    on_error(exc)
                except Exception:
                    pass
            finally:
                self._outbound_queue.task_done()

    def _queue_outbound_message(
        self,
        message: WorkerMessage,
        on_error: Callable[[Exception], None],
    ) -> bool:
        with self._outbound_start_lock:
            if self._outbound_stop.is_set():
                on_error(ConnectionError("remote worker Provider is closed"))
                return False
            try:
                self._outbound_queue.put_nowait((message, on_error))
            except queue.Full as exc:
                on_error(exc)
                return False
            if self._outbound_thread is None:
                self._outbound_thread = threading.Thread(
                    target=self._send_outbound_messages,
                    name=f"task-worker-outbound-{self.provider_id}",
                    daemon=True,
                )
                self._outbound_thread.start()
            return True

    def _check_duplicate_locked(self, message: WorkerMessage) -> bool:
        digest = self._seen_messages.get(message.message_id)
        if digest is None:
            return False
        if digest != _message_digest(message):
            raise WorkerProtocolError(
                "message_id was reused with different Stage content",
                code="message_id_conflict",
                field="message_id",
            )
        return True

    def _remember_message_locked(self, message: WorkerMessage) -> None:
        self._seen_messages[message.message_id] = _message_digest(message)
        self._seen_order.append(message.message_id)
        while len(self._seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._seen_order.popleft()
            self._seen_messages.pop(expired, None)

    def _take_cancel_message_locked(
        self, pending: _PendingRemoteAttempt,
    ) -> Optional[WorkerMessage]:
        """Build one cancellation only after its Stage offer is on the wire."""
        if (
            not pending.offer_sent
            or not pending.cancel_requested
            or pending.cancel_enqueued
        ):
            return None
        pending.cancel_enqueued = True
        attempt = pending.attempt
        return build_message(
            "stage_cancel",
            {
                "workflow_id": attempt.request.workflow_id,
                "stage_id": attempt.request.stage_id,
                "attempt_id": attempt.attempt_id,
                "lease_id": attempt.lease_id,
                "lease_epoch": attempt.lease_epoch,
                "reason_code": "coordinator_cancelled",
            },
            message_id=_message_id("cancel_"),
            sent_at_ms=int(time.time() * 1000),
            version=PROTOCOL_VERSION,
        )

    def _queue_cancel_message(
        self, attempt_id: str, message: WorkerMessage,
    ) -> None:
        def on_send_error(_exc: Exception) -> None:
            with self._lock:
                current = self._pending.get(attempt_id)
                if current is not None:
                    current.cancel_acknowledged = True
                    current.cancel_ack_event.set()

        self._queue_outbound_message(message, on_send_error)

    def _prune_pending_locked(self) -> None:
        now = time.monotonic()
        expired = [
            attempt_id
            for attempt_id, pending in self._pending.items()
            if pending.released
            and pending.released_at > 0
            and now - pending.released_at >= 5.0
        ]
        for attempt_id in expired:
            self._pending.pop(attempt_id, None)

    def _absorb_late_stage_response_locked(
        self,
        pending: _PendingRemoteAttempt,
        message: WorkerMessage,
    ) -> WorkerMessage:
        self._remember_message_locked(message)
        self._late_stage_responses += 1
        logger.info(
            "event=task_worker_late_stage_response_ignored node_id=%s "
            "message_type=%s attempt_id=%s terminal_kind=%s",
            self.node_id,
            message.message_type,
            pending.attempt.attempt_id,
            pending.terminal_snapshot()[0],
        )
        return message

    def _request_remote_cancel_locked(
        self,
        pending: _PendingRemoteAttempt,
    ) -> Optional[WorkerMessage]:
        if pending.cancel_requested:
            return None
        terminal_kind = pending.terminal_snapshot()[0]
        if terminal_kind not in {
            "provider_cancelled",
            "remote_accept_timeout",
            "lease_expired",
            REQUEST_DEADLINE_EXCEEDED,
        }:
            return None
        pending.cancel_requested = True
        return self._take_cancel_message_locked(pending)

    def _request_remote_cancel(self, attempt_id: str) -> None:
        message = None
        with self._lock:
            pending = self._pending.get(attempt_id)
            if pending is not None:
                message = self._request_remote_cancel_locked(pending)
        if message is not None:
            self._queue_cancel_message(attempt_id, message)

    def _snapshot(self) -> dict[str, Any]:
        try:
            return dict(self._peer_snapshot() or {})
        except Exception:
            return {}

    @staticmethod
    def _model_matches(
        requested: Optional[ModelIdentity], models: Any,
    ) -> bool:
        if requested is None or not isinstance(models, list):
            return False
        expected = requested.snapshot()
        for model in models:
            if not isinstance(model, dict):
                continue
            try:
                candidate = dict(model)
                candidate["engine"] = canonical_model_engine(model.get("engine"))
            except ValueError:
                continue
            if candidate == expected:
                return True
        return False

    @staticmethod
    def _layer_model_matches(
        requested: Optional[ModelIdentity], models: Any,
    ) -> bool:
        """Match the physical artifact identity used by Route A layer workers.

        The coordinator's logical model id and revision may differ from the
        deterministic layer alias, but the engine, format, and source digest
        must remain exact so a crop from another model cannot be selected.
        """
        if requested is None or not isinstance(models, list):
            return False
        expected = requested.snapshot()
        for model in models:
            if not isinstance(model, dict):
                continue
            try:
                engine = canonical_model_engine(model.get("engine"))
            except ValueError:
                continue
            if (
                engine == expected["engine"]
                and model.get("format") == expected["format"]
                and model.get("sha256") == expected["sha256"]
            ):
                return True
        return False

    def _supports_stage_chunked_input(self) -> bool:
        """对端是否声明能接收 `stage_chunk` + `hidden_ref`（默认否）。"""
        capabilities = (self._snapshot() or {}).get("capabilities")
        if not isinstance(capabilities, dict):
            return False
        return capabilities.get("stage_chunked_input") is True

    def _maybe_send_stage_chunks(self, attempt: StageAttempt) -> dict[str, Any]:
        """★ 2026-10-07（DIST-NEXT-2b）：超预算的层段 hidden 先分片发出，返回改写后的 root_input。

        * 非层段 / 无内联 hidden / 未超预算 / 对端未声明能力 ⇒ **原样返回**（零行为变化）；
        * 声明能力且超预算 ⇒ 同步发出 `chunk_count` 条 `stage_chunk`，再把 `hidden_f32`
          换成 `hidden_ref`（`total_bytes` / `payload_sha256` 供 worker 装配校验）；
        * 发送失败 ⇒ `ProviderExecutionError`（明确失败，不静默退回内联 —— 那会在协议层
          抛 `message_too_large`，把「分片发不出去」的原因丢掉）。
        """
        request = attempt.request
        root_input = request.root_input if isinstance(request.root_input, dict) else {}
        encoded = root_input.get("hidden_f32")
        if not isinstance(encoded, str):
            return root_input
        stage_fields = request.stage_fields or {}
        spec = stage_fields.get("hidden_spec") or {}
        n_tokens = int(spec.get("n_tokens", 0) or 0)
        n_embd = int(spec.get("n_embd", 0) or 0)
        dtype = str(spec.get("dtype", "float32") or "float32")
        if n_tokens < 1 or n_embd < 1:
            return root_input
        if hidden_fits_stage_frame(n_tokens, n_embd, dtype):
            return root_input
        if not self._supports_stage_chunked_input():
            # 对端不支持 ⇒ 维持内联路径（超限由 dispatch 前的预检负责拒绝）。
            return root_input
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ProviderExecutionError(
                "layer stage hidden is not valid base64",
                code="invalid_stage_input",
                provider_id=self.provider_id,
            ) from exc
        try:
            plan = plan_stage_payload_chunks(raw)
        except WorkerProtocolError as exc:
            raise ProviderExecutionError(
                f"layer stage hidden cannot be chunked: {exc}",
                code=getattr(exc, "code", "stage_payload_too_large"),
                provider_id=self.provider_id,
            ) from exc
        sent_at_ms = int(time.time() * 1000)
        for index, chunk in enumerate(plan.chunks):
            chunk_message = build_stage_chunk(
                workflow_id=request.workflow_id,
                stage_id=request.stage_id,
                attempt_id=attempt.attempt_id,
                lease_id=attempt.lease_id,
                lease_epoch=attempt.lease_epoch,
                provider_id=self.provider_id,
                chunk_index=index,
                chunk_count=plan.chunk_count,
                payload=chunk,
                total_bytes=plan.total_bytes,
                message_id=_message_id("chunk_"),
                sent_at_ms=sent_at_ms,
            )
            try:
                self._send_message(chunk_message)
            except Exception as exc:
                raise ProviderExecutionError(
                    "failed to send a layer stage input chunk",
                    code="remote_worker_disconnected",
                    provider_id=self.provider_id,
                    retryable=True,
                ) from exc
        logger.info(
            "event=task_worker_stage_chunks_sent node_id=%s attempt_id=%s "
            "chunks=%d bytes=%d",
            self.node_id, attempt.attempt_id, plan.chunk_count, plan.total_bytes,
        )
        chunked = {
            key: value for key, value in root_input.items() if key != "hidden_f32"
        }
        chunked["hidden_ref"] = stage_chunk_ref(plan)
        return chunked

    def inspect(self) -> ProviderCapabilities:
        snapshot = self._snapshot()
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
        stage_types = capabilities.get("stage_types", [])
        if not isinstance(stage_types, list):
            stage_types = []
        worker_kind = snapshot.get("worker_kind", "pc_full_worker")
        resource_gate = capabilities.get("resource_gate")
        resource_admitted = True
        if worker_kind == "android_full_worker":
            resource_admitted = bool(
                isinstance(resource_gate, dict)
                and resource_gate.get("admitted") is True
                and not resource_gate.get("reason_code")
            )
        # N2.3 still exposes one controlled slot even if hello advertises
        # future multi-Stage capacity.
        max_concurrency = 1
        with self._lock:
            self._prune_pending_locked()
            active = len(self._reservations)
            closed = self._closed
        manual_dispatch_enabled = bool(
            snapshot.get("manual_stage_dispatch_enabled")
        )
        layer_dispatch_enabled = bool(
            snapshot.get("layer_stage_dispatch_enabled")
        )
        healthy = bool(
            not closed
            and snapshot.get("healthy")
            # ★ 2026-09-20：原为硬编码 `== 2`（第二处），协议升 v3 后 provider 恒 unhealthy。
            #   语义是「v2 及以上的数据面通道可用」，与具体版本号解耦。
            and int(snapshot.get("selected_version") or 0) >= 2
            # Provider health is generic.  The reservation path applies the
            # stage-specific gate so a layer-only capability cannot disable
            # ordinary full_inference dispatch on the same worker.
            and (manual_dispatch_enabled or layer_dispatch_enabled)
            and resource_admitted
        )
        return ProviderCapabilities(
            provider_id=self.provider_id,
            provider_kind=self.provider_kind,
            supported_stage_types=tuple(
                value for value in stage_types
                # ★ 2026-09-20：`layer_forward`（v3 层段）也必须能透传，否则
                #   安卓 worker 即使声明了层段能力，也会在这里被静默抹掉。
                if value in {"full_inference", "aggregate", "layer_forward"}
            ),
            max_concurrency=max_concurrency,
            active_reservations=active,
            healthy=healthy,
            available=healthy and active < max_concurrency,
            node_id=self.node_id,
        )

    def cancel_diagnostics(self) -> dict[str, Any]:
        """取消合同的分类计数（DIST-NEXT-1）。

        把「收到 ACK」「执行确证已停止」「对端主动取消」「迟到结果被吸收」
        分开统计。调用方（health / 测试 / 后续 metrics）据此区分
        「请求已取消」与「执行已停止」，不再由单一布尔冒充。
        """
        with self._lock:
            ack_states = dict(self._cancel_ack_states)
            pending_with_cancel = sum(
                1 for pending in self._pending.values()
                if pending.cancel_requested
            )
            still_in_flight = sum(
                1 for pending in self._pending.values()
                if pending.cancel_requested
                and pending.cancel_acknowledged
                and not pending.cancel_execution_stopped
            )
            return {
                "cancel_ack_execution_states": ack_states,
                "cancel_execution_stopped_confirmed": self._cancel_execution_stopped,
                "cancel_remote_initiated": self._cancel_remote_initiated,
                "late_stage_responses_absorbed": self._late_stage_responses,
                "pending_cancel_requests": pending_with_cancel,
                "pending_cancel_ack_in_flight": still_in_flight,
            }

    def supports_model_identity(
        self, model_identity: ModelIdentity, stage_type: str,
    ) -> bool:
        status = self.inspect()
        snapshot = self._snapshot()
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            return False
        matcher = (
            self._layer_model_matches
            if stage_type == "layer_forward"
            else self._model_matches
        )
        return bool(
            status.healthy
            and stage_type in status.supported_stage_types
            and matcher(
                model_identity, capabilities.get("models", []),
            )
        )

    def model_identities(self) -> tuple[ModelIdentity, ...]:
        """Return validated immutable identities advertised by this v2 Worker."""
        capabilities = self._snapshot().get("capabilities", {})
        models = capabilities.get("models", []) if isinstance(capabilities, dict) else []
        identities = []
        for model in models if isinstance(models, list) else []:
            if not isinstance(model, dict):
                continue
            try:
                identities.append(ModelIdentity(**model))
            except (TypeError, ValueError):
                continue
        return tuple(identities)

    def reserve(self, request: StageRequest) -> Reservation:
        snapshot = self._snapshot()
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
        status = self.inspect()
        if request.provider_id != self.provider_id:
            raise ProviderReservationError(
                "stage request targets a different remote provider",
                code="provider_request_mismatch",
                provider_id=self.provider_id,
            )
        if DEPENDENCY_FAILURES_KEY in request.dependencies:
            raise ProviderUnavailable(
                "protocol v2 remote workers do not support partial dependency input",
                code="partial_dependencies_not_supported",
                provider_id=self.provider_id,
                retryable=True,
            )
        if request.stage_type not in status.supported_stage_types:
            raise ProviderUnavailable(
                "remote worker does not support the requested stage type",
                code="unsupported_stage_type",
                provider_id=self.provider_id,
            )
        stage_dispatch_enabled = (
            bool(snapshot.get("layer_stage_dispatch_enabled"))
            if request.stage_type == "layer_forward"
            else bool(snapshot.get("manual_stage_dispatch_enabled"))
        )
        if not stage_dispatch_enabled:
            raise ProviderUnavailable(
                "remote worker is not admitted for the requested Stage type",
                code="stage_dispatch_not_admitted",
                provider_id=self.provider_id,
                retryable=True,
            )
        if request.model_identity is None:
            raise ProviderUnavailable(
                "remote execution requires an exact model identity",
                code="model_identity_required",
                provider_id=self.provider_id,
            )
        matcher = (
            self._layer_model_matches
            if request.stage_type == "layer_forward"
            else self._model_matches
        )
        if not matcher(request.model_identity, capabilities.get("models", [])):
            raise ProviderUnavailable(
                "remote worker does not have the exact requested model",
                code="model_identity_mismatch",
                provider_id=self.provider_id,
                retryable=True,
            )
        with self._lock:
            self._prune_pending_locked()
            if self._closed or not status.healthy:
                raise ProviderUnavailable(
                    "remote worker is not healthy",
                    code="remote_worker_unavailable",
                    provider_id=self.provider_id,
                    retryable=True,
                )
            if len(self._reservations) >= status.max_concurrency:
                raise ProviderBusy(
                    "remote worker has no free manual Stage slot",
                    code="remote_worker_busy",
                    provider_id=self.provider_id,
                    retryable=True,
                )
            reservation = Reservation(
                reservation_id=f"res_{uuid.uuid4().hex}",
                provider_id=self.provider_id,
                workflow_id=request.workflow_id,
                stage_id=request.stage_id,
                created_at=time.time(),
                selection_reason=(
                    "auto_remote_provider"
                    if request.runtime_context.get(
                        "task_graph_remote_policy"
                    ) == "auto"
                    else "explicit_remote_provider"
                ),
                provider_kind=self.provider_kind,
                provider_node_id=self.node_id,
            )
            self._reservations[reservation.reservation_id] = (
                reservation, request,
            )
            return reservation

    @staticmethod
    def _identity_matches(
        payload: Mapping[str, Any], attempt: StageAttempt,
    ) -> bool:
        return all((
            payload.get("workflow_id") == attempt.request.workflow_id,
            payload.get("stage_id") == attempt.request.stage_id,
            payload.get("attempt_id") == attempt.attempt_id,
            payload.get("lease_id") == attempt.lease_id,
            payload.get("lease_epoch") == attempt.lease_epoch,
            payload.get("provider_id") == attempt.provider_id,
        ))

    @staticmethod
    def _wait(
        event: threading.Event,
        pending: _PendingRemoteAttempt,
        cancel_event: threading.Event,
        deadline: float | Callable[[], float],
        *,
        timeout_code: str | Callable[[], str],
        provider_id: str,
    ) -> None:
        while True:
            current_deadline = deadline() if callable(deadline) else deadline
            remaining = current_deadline - time.monotonic()
            if event.wait(max(0.0, min(0.05, remaining))):
                break
            if cancel_event.is_set():
                error = ProviderExecutionError(
                    "remote Stage wait was cancelled locally",
                    code="provider_cancelled",
                    provider_id=provider_id,
                )
                pending.decide_terminal("provider_cancelled", error=error)
                break
            current_deadline = deadline() if callable(deadline) else deadline
            if time.monotonic() >= current_deadline:
                code = timeout_code() if callable(timeout_code) else timeout_code
                error = ProviderExecutionError(
                    "remote Stage response timed out",
                    code=code,
                    provider_id=provider_id,
                    retryable=code != REQUEST_DEADLINE_EXCEEDED,
                )
                pending.decide_terminal(code, error=error)
                break
        _kind, _result, error = pending.terminal_snapshot()
        if error is not None:
            raise error

    def execute(
        self,
        attempt: StageAttempt,
        reservation: Reservation,
        cancel_event: threading.Event,
    ) -> StageResult:
        with self._lock:
            owned = self._reservations.get(reservation.reservation_id)
            if owned is None or owned[0] != reservation:
                raise ProviderReservationError(
                    "remote reservation is unknown",
                    code="invalid_reservation",
                    provider_id=self.provider_id,
                )
            if (
                owned[1] != attempt.request
                or attempt.provider_id != self.provider_id
                or reservation.provider_id != self.provider_id
            ):
                raise ProviderReservationError(
                    "remote attempt does not match its reservation",
                    code="attempt_reservation_mismatch",
                    provider_id=self.provider_id,
                )
            if reservation.reservation_id in self._executed_reservations:
                raise ProviderReservationError(
                    "remote reservation has already been executed",
                    code="reservation_already_executed",
                    provider_id=self.provider_id,
                )
            if attempt.request.model_identity is None:
                raise ProviderExecutionError(
                    "remote attempt has no model identity",
                    code="model_identity_required",
                    provider_id=self.provider_id,
                )
            now_epoch = time.time()
            now_monotonic = time.monotonic()
            pending = _PendingRemoteAttempt(
                attempt=attempt,
                lease_expires_at=attempt.lease_expires_at,
                lease_deadline_monotonic=now_monotonic + max(
                    0.0, attempt.lease_expires_at - now_epoch,
                ),
            )
            self._pending[attempt.attempt_id] = pending
            self._reservation_attempts[reservation.reservation_id] = (
                attempt.attempt_id
            )
            self._executed_reservations.add(reservation.reservation_id)

        lease_expires_at_ms = int(attempt.lease_expires_at * 1000)
        # ★ 2026-10-07（DIST-NEXT-2b）：超帧预算的 hidden 走**有序分片**（仅当对端声明
        #   `stage_chunked_input`）。分片必须在 offer **之前**发出：offer 只带 `hidden_ref`，
        #   worker 收到 offer 时要求分片已齐备。未声明能力/未超预算 ⇒ 原样返回（零行为变化）。
        root_input = self._maybe_send_stage_chunks(attempt)
        # The worker derives a relative lease from this timestamp. Sampling it
        # before a large chunk upload would silently add the whole upload time
        # back to the execution lease/request budget.
        sent_at_ms = int(time.time() * 1000)
        offer_payload = {
                    "workflow_id": attempt.request.workflow_id,
                    "request_id": attempt.request.request_id,
                    "stage_id": attempt.request.stage_id,
                    "stage_type": attempt.request.stage_type,
                    "attempt_id": attempt.attempt_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "lease_expires_at_ms": lease_expires_at_ms,
                    "provider_id": self.provider_id,
                    "root_input": root_input,
                    "dependencies": attempt.request.dependencies,
                    "input_sha256": stage_input_sha256(
                        root_input,
                        attempt.request.dependencies,
                    ),
                    "model_identity": attempt.request.model_identity.snapshot(),
                }
        if attempt.request_deadline is not None:
            offer_payload["request_deadline_ms"] = (
                attempt.request_deadline.expires_at_epoch_ms
            )
        if attempt.request.stage_type == "layer_forward":
            offer_payload.update(attempt.request.stage_fields)
        elif attempt.request.stage_fields:
            raise ProviderExecutionError(
                "non-layer stage contains specialized stage fields",
                code="invalid_stage_fields",
                provider_id=self.provider_id,
            )
        offer = build_message(
                "stage_offer",
                offer_payload,
                message_id=_message_id("offer_"),
                sent_at_ms=sent_at_ms,
                version=PROTOCOL_VERSION,
        )
        try:
            self._send_message(offer)
            # ★ 2026-10-08（诊断，定位后降级）：真机卡点是「18 片已发、offer 似乎从不被
            #   worker 读到」⇒ 先确认 offer 到底有没有写出去（以及它是否已是 hidden_ref 形态）。
            logger.debug(
                "event=task_worker_stage_offer_sent node_id=%s attempt_id=%s stage_id=%s "
                "bytes=%d chunked=%s",
                self.provider_id, attempt.attempt_id, attempt.request.stage_id,
                len(json.dumps(offer, default=str).encode("utf-8")),
                isinstance(root_input.get("hidden_ref"), dict),
            )
        except Exception as exc:
            error = ProviderExecutionError(
                "failed to send Stage offer to the remote worker",
                code="remote_worker_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            )
            # The send failure and local cancellation can cross while the
            # synchronous transport call is blocked. Whichever terminal was
            # latched first remains authoritative.
            with self._lock:
                current = self._pending.get(attempt.attempt_id)
                if current is pending:
                    pending.decide_terminal("offer_send_failure", error=error)
                    if pending.cancel_requested:
                        pending.cancel_acknowledged = True
                        pending.cancel_ack_event.set()
                    terminal_error = pending.terminal_snapshot()[2]
                else:
                    terminal_error = None
            raise terminal_error or error from exc
        cancel_message = None
        with self._lock:
            current = self._pending.get(attempt.attempt_id)
            if current is pending:
                pending.offer_sent = True
                cancel_message = self._take_cancel_message_locked(pending)
        if cancel_message is not None:
            self._queue_cancel_message(attempt.attempt_id, cancel_message)
        accept_timeout_deadline = time.monotonic() + min(
            self._accept_timeout_seconds,
            max(0.001, float(attempt.accept_timeout_seconds)),
        )
        request_deadline_monotonic = (
            attempt.request_deadline.expires_at_monotonic
            if attempt.request_deadline is not None else float("inf")
        )

        def accept_deadline() -> float:
            return min(
                pending.lease_deadline_monotonic,
                accept_timeout_deadline,
                request_deadline_monotonic,
            )

        def accept_timeout_code() -> str:
            if request_deadline_monotonic <= min(
                pending.lease_deadline_monotonic, accept_timeout_deadline,
            ):
                return REQUEST_DEADLINE_EXCEEDED
            if pending.lease_deadline_monotonic <= accept_timeout_deadline:
                return "lease_expired"
            return "remote_accept_timeout"
        # ★ 2026-10-09（接口税量化 · 方向2）：把一次 stage 往返拆成「发出→accept」与
        #   「accept→result」两段。此前只有 stage 总耗时（docs #62 的 `s`），无法分辨
        #   固定开销落在哪一次往返上，也无法验证 WiFi lock 是否真的压低了层段 RTT。
        #   短字段名（`a=`/`r=`）：master.log 的 stdout 重定向会把每行截断在 ~119 字符。
        _t_send = time.perf_counter()
        try:
            self._wait(
                pending.accept_event,
                pending,
                cancel_event,
                accept_deadline,
                timeout_code=accept_timeout_code,
                provider_id=self.provider_id,
            )
        except ProviderExecutionError as exc:
            # ★ 2026-10-05（DIST-3「取消」场景，已知问题 #47）：本地取消必须
            #   **向对端传播** `stage_cancel`。本类 `cancel()` 里早就有构造与发送
            #   逻辑，但此前只在 `remote_accept_timeout` 这一条路径调用过。
            if exc.code in (
                "remote_accept_timeout", "lease_expired",
                "provider_cancelled", REQUEST_DEADLINE_EXCEEDED,
            ):
                self._request_remote_cancel(attempt.attempt_id)
            raise
        if not pending.accepted:
            raise ProviderExecutionError(
                "remote worker did not accept the Stage",
                code="remote_stage_not_accepted",
                provider_id=self.provider_id,
            )
        _t_accept = time.perf_counter()
        try:
            self._wait(
                pending.result_event,
                pending,
                cancel_event,
                lambda: min(
                    pending.lease_deadline_monotonic,
                    request_deadline_monotonic,
                ),
                timeout_code=lambda: (
                    REQUEST_DEADLINE_EXCEEDED
                    if request_deadline_monotonic
                    <= pending.lease_deadline_monotonic
                    else "lease_expired"
                ),
                provider_id=self.provider_id,
            )
        except ProviderExecutionError as exc:
            # ★ 2026-10-05（同上）：等结果阶段被取消（`provider_cancelled`）或租约
            #   过期时，同样要让对端停手 —— 否则 worker 白跑完整个 Stage，回传结果
            #   时本端已无对应 pending（`reason=unknown_attempt`）。
            if exc.code in (
                "provider_cancelled", "lease_expired",
                REQUEST_DEADLINE_EXCEEDED,
            ):
                self._request_remote_cancel(attempt.attempt_id)
            raise
        logger.info(
            "perf2 a=%.0f r=%.0f",
            (_t_accept - _t_send) * 1000.0,
            (time.perf_counter() - _t_accept) * 1000.0,
        )
        _kind, result, _error = pending.terminal_snapshot()
        if result is None:
            raise ProviderExecutionError(
                "remote worker returned no Stage result",
                code="invalid_provider_result",
                provider_id=self.provider_id,
            )
        return result

    def handle_message(
        self, raw: bytes | str | Mapping[str, Any],
    ) -> WorkerMessage:
        message = decode_message(raw)
        if message.message_type not in {
            "stage_accept", "stage_result", "stage_error", "stage_cancelled",
            # ★ 2026-10-11（三机实测）：**回程**分片，与输入方向的 `stage_chunk` 对称。
            "stage_result_chunk",
        }:
            raise WorkerProtocolError(
                "message is not a coordinator-side Stage response",
                code="invalid_message_direction",
                field="message_type",
            )
        payload = message.payload
        attempt_id = str(payload.get("attempt_id", ""))
        with self._lock:
            self._prune_pending_locked()
            if self._check_duplicate_locked(message):
                return message
            pending = self._pending.get(attempt_id)
            if pending is None:
                raise WorkerProtocolError(
                    "Stage response has no pending attempt",
                    code="unknown_attempt",
                    field="payload.attempt_id",
                )
            if not self._identity_matches(payload, pending.attempt):
                # ★ 2026-10-03：这类不匹配此前只报码、看不出是哪个字段对不上，
                #   真机排 Route A 时只能反复试。把两侧值一并打出来。
                logger.warning(
                    "Stage response identity mismatch: reply=%s expected=%s",
                    {
                        key: payload.get(key)
                        for key in (
                            "workflow_id", "stage_id", "attempt_id",
                            "lease_id", "lease_epoch", "provider_id",
                        )
                    },
                    {
                        "workflow_id": pending.attempt.request.workflow_id,
                        "stage_id": pending.attempt.request.stage_id,
                        "attempt_id": pending.attempt.attempt_id,
                        "lease_id": pending.attempt.lease_id,
                        "lease_epoch": pending.attempt.lease_epoch,
                        "provider_id": pending.attempt.provider_id,
                    },
                )
                raise WorkerProtocolError(
                    "Stage response identity does not match the pending attempt",
                    code="attempt_identity_mismatch",
                    field="payload",
                )
            if message.message_type == "stage_result_chunk":
                # ★ 2026-10-11（三机实测）：**回程**分片。只按 attempt 累积、**不推进终态**；
                #   随后的 `stage_result`（其 `output.hidden_out_ref` 引用这些分片）才决定结果。
                #   与输入方向 `stage_chunk` 的语义完全对称。
                self._absorb_result_chunk_locked(pending, payload)
                return message
            terminal_kind = pending.terminal_snapshot()[0]
            if terminal_kind and message.message_type in {
                "stage_accept", "stage_result", "stage_error",
            }:
                return self._absorb_late_stage_response_locked(pending, message)
            if (
                terminal_kind
                and message.message_type == "stage_cancelled"
                and not pending.cancel_requested
                and terminal_kind != "remote_cancel"
            ):
                return self._absorb_late_stage_response_locked(pending, message)
            if message.message_type == "stage_accept":
                if pending.accept_event.is_set():
                    raise WorkerProtocolError(
                        "Stage acceptance was already recorded",
                        code="duplicate_stage_response",
                        field="message_type",
                    )
                if payload["accepted"]:
                    pending.accepted = True
                    pending.accept_event.set()
                else:
                    # ★ 2026-10-03：把 worker 给的 `reason_code` 带进消息。此前只塞进
                    #   `code` 字段，异常在别处被转述成一层笼统的
                    #   `route_a_stage_execution_failed: remote worker rejected the Stage offer`
                    #   ⇒ 跨机层段失败时完全看不出是「身份不符」「租约过期」还是
                    #   「不支持该 stage」，只能靠逐层加日志去猜。
                    error = ProviderReservationError(
                        "remote worker rejected the Stage offer"
                        f" (reason_code={payload['reason_code']}"
                        f", retryable={bool(payload['retryable'])})",
                        code=payload["reason_code"],
                        provider_id=self.provider_id,
                        retryable=bool(payload["retryable"]),
                    )
                    if not pending.decide_terminal(
                        "reservation_rejected", error=error,
                    ):
                        return self._absorb_late_stage_response_locked(
                            pending, message,
                        )
            elif message.message_type == "stage_result":
                if not pending.accepted:
                    if pending.terminal_snapshot()[0]:
                        return self._absorb_late_stage_response_locked(
                            pending, message,
                        )
                    raise WorkerProtocolError(
                        "Stage result arrived before acceptance",
                        code="result_before_accept",
                        field="message_type",
                    )
                result = StageResult(
                    output=self._expand_result_chunks_locked(pending, payload["output"]),
                    provider_id=payload["provider_id"],
                    metadata=payload["metadata"],
                    attempt_id=payload["attempt_id"],
                    lease_epoch=payload["lease_epoch"],
                )
                if not pending.decide_terminal("stage_result", result=result):
                    return self._absorb_late_stage_response_locked(
                        pending, message,
                    )
            elif message.message_type == "stage_error":
                # ★ 2026-10-09（稳定性 #73）：带上 worker 回传的 `reason`（新增字段；
                #   老 worker 不带 ⇒ 退化为原文案，兼容）。否则真因（例如 hidden 维度不匹配
                #   `[2048]` vs `(33, 896)`）会在顶层被吞掉，用户只看到「禁止整模回退」
                #   这类与真因无关的二次错误，换模型也修不好。
                _stage_reason = str(payload.get("reason") or "").strip()
                error = ProviderExecutionError(
                    "remote worker reported a Stage error"
                    + (f": {_stage_reason[:200]}" if _stage_reason else ""),
                    code=payload["error_code"],
                    provider_id=self.provider_id,
                    retryable=bool(payload["retryable"]),
                )
                if not pending.decide_terminal("stage_error", error=error):
                    return self._absorb_late_stage_response_locked(
                        pending, message,
                    )
                # ★ 2026-10-07（DIST-NEXT-1d 真机复测发现）：此前只把泛化消息抛出，
                #   worker 回传的 `error_code` / `retryable` **没有落日志** ⇒ 现场只能看到
                #   「remote worker reported a Stage error」，无法区分是身份不匹配、预算超限
                #   还是执行失败。这里补一条具名事件（与 cancel 路径的事件风格一致）。
                logger.info(
                    "event=task_worker_stage_error node_id=%s stage_id=%s attempt_id=%s "
                    "error_code=%s retryable=%s",
                    self.provider_id,
                    payload.get("stage_id", "-"),
                    payload.get("attempt_id", "-"),
                    payload["error_code"],
                    payload["retryable"],
                )
            else:
                # ★ 2026-10-07（DIST-NEXT-1）：取消合同。
                #   `execution_state` 可选：旧对端不带 ⇒ `unknown`（不冒充已停止）。
                execution_state = str(payload.get("execution_state") or "unknown")
                terminal_kind = pending.terminal_snapshot()[0]
                if not pending.cancel_requested and terminal_kind != "remote_cancel":
                    # 对端**主动**取消（Android 本地用户取消 / Service 回收）。
                    # 此前一律按 `unexpected_stage_cancelled` 拒绝 ⇒ master 只能等到
                    # 租约/步骤超时，且在 coordinator 侧看不出是谁取消的。现在收敛成
                    # 单一 reason：该 attempt 由对端终止，本端以可重试的远端取消结束等待。
                    error = ProviderExecutionError(
                        "remote worker cancelled the Stage",
                        code=str(
                            payload.get("reason_code") or "remote_worker_cancelled"
                        ),
                        provider_id=self.provider_id,
                        retryable=True,
                    )
                    if not pending.decide_terminal("remote_cancel", error=error):
                        return self._absorb_late_stage_response_locked(
                            pending, message,
                        )
                    pending.cancel_remote_initiated = True
                    self._cancel_remote_initiated += 1
                    logger.info(
                        "event=task_worker_stage_cancel_remote_initiated node_id=%s "
                        "workflow_id=%s stage_id=%s attempt_id=%s reason_code=%s "
                        "execution_state=%s",
                        self.node_id,
                        payload.get("workflow_id", ""),
                        payload.get("stage_id", ""),
                        attempt_id,
                        payload.get("reason_code", ""),
                        execution_state,
                    )
                if pending.cancel_acknowledged and not pending.cancel_execution_stopped:
                    # 第二条 ACK 可以把「仍在执行」升级为「已停止」——这是取消合同
                    # 允许的唯一迟到的正向更新（其余迟到响应走上面的吸收分支）。
                    pass
                elif pending.cancel_acknowledged:
                    self._remember_message_locked(message)
                    return message
                # 一条 ACK 只证明「取消请求已送达」；只有对端显式回报
                # `execution_stopped` 才证明「执行已停止」。
                first_ack = not pending.cancel_acknowledged
                previous_state = pending.cancel_ack_execution_state
                pending.cancel_acknowledged = True
                if execution_state != "unknown":
                    pending.cancel_ack_execution_state = execution_state
                if (
                    execution_state == "execution_stopped"
                    and not pending.cancel_execution_stopped
                ):
                    pending.cancel_execution_stopped = True
                    self._cancel_execution_stopped += 1
                if first_ack or execution_state != previous_state:
                    self._cancel_ack_states[execution_state] += 1
                pending.cancel_ack_event.set()
                if first_ack:
                    logger.info(
                        "event=task_worker_stage_cancel_acknowledged node_id=%s "
                        "workflow_id=%s stage_id=%s attempt_id=%s execution_state=%s",
                        self.node_id,
                        payload.get("workflow_id", ""),
                        payload.get("stage_id", ""),
                        attempt_id,
                        execution_state,
                    )
                elif execution_state != previous_state:
                    logger.info(
                        "event=task_worker_stage_cancel_execution_stopped node_id=%s "
                        "workflow_id=%s stage_id=%s attempt_id=%s execution_state=%s",
                        self.node_id,
                        payload.get("workflow_id", ""),
                        payload.get("stage_id", ""),
                        attempt_id,
                        execution_state,
                    )
            self._remember_message_locked(message)
        return message

    def renew_lease(
        self,
        attempt_id: str,
        lease_id: str,
        lease_epoch: int,
        lease_expires_at: float,
    ) -> bool:
        with self._lock:
            pending = self._pending.get(attempt_id)
            if pending is None:
                raise ProviderExecutionError(
                    "remote lease renewal has no pending attempt",
                    code="unknown_attempt",
                    provider_id=self.provider_id,
                )
            attempt = pending.attempt
            deadline = float(lease_expires_at)
            if (
                attempt.lease_id != lease_id
                or attempt.lease_epoch != int(lease_epoch)
                or deadline <= pending.lease_expires_at
            ):
                raise ProviderExecutionError(
                    "remote lease renewal identity is stale",
                    code="stale_lease",
                    provider_id=self.provider_id,
                )
            if (
                attempt.request_deadline is not None
                and deadline > attempt.request_deadline.expires_at_epoch
            ):
                raise ProviderExecutionError(
                    "remote lease renewal exceeds the request deadline",
                    code="lease_exceeds_request_deadline",
                    provider_id=self.provider_id,
                )
            message = build_message(
                "lease_renew",
                {
                    "workflow_id": attempt.request.workflow_id,
                    "stage_id": attempt.request.stage_id,
                    "attempt_id": attempt.attempt_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "lease_expires_at_ms": int(deadline * 1000),
                },
                message_id=_message_id("renew_"),
                sent_at_ms=int(time.time() * 1000),
                version=PROTOCOL_VERSION,
            )
            pending.lease_expires_at = deadline
            pending.lease_deadline_monotonic = time.monotonic() + max(
                0.0, deadline - time.time(),
            )

        def on_send_error(_exc: Exception) -> None:
            error = ProviderExecutionError(
                "failed to renew the remote Stage lease",
                code="remote_worker_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            )
            with self._lock:
                current = self._pending.get(attempt_id)
                if current is not None:
                    current.decide_terminal(
                        "renew_send_failure", error=error,
                    )

        if not self._queue_outbound_message(message, on_send_error):
            raise ProviderExecutionError(
                "remote Stage lease renewal queue is full",
                code="remote_worker_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            )
        return True

    def cancel(self, attempt_id: str) -> None:
        message = None
        with self._lock:
            pending = self._pending.get(attempt_id)
            if pending is None or pending.cancel_requested:
                return
            error = ProviderExecutionError(
                "remote Stage was cancelled locally",
                code="provider_cancelled",
                provider_id=self.provider_id,
            )
            if not pending.decide_terminal("provider_cancelled", error=error):
                return
            message = self._request_remote_cancel_locked(pending)

        if message is not None:
            self._queue_cancel_message(attempt_id, message)

    def notify_disconnect(
        self, *, reason_code: str = RELEASE_REASON_DISCONNECTED,
    ) -> None:
        """对端断连：唤醒所有 pending，并**主动回收本 provider 的全部 reservation**。

        ★ 2026-10-05（DIST-2 要求 2：「节点掉线…必须释放 lease」）：此前这里只把
        pending 置 error 并 set 三个 event，**不释放 reservation** —— 释放完全依赖
        上层 `task_graph._run_stage` 的 `finally`。一旦上层没走到那里（异常路径、
        外层取消、进程卡住），条目就会留在 `_reservations` /
        `_executed_reservations` / `_reservation_attempts` 里，而该 provider 在对端
        重连前不会再有活动 ⇒ 「已预留未执行」的槽位被永久占用。

        `release()` 是**纯本地 dict 操作**（不向 worker 发任何消息），在断连路径上
        调用没有副作用；先唤醒 pending 再回收，顺序保证等待方先拿到
        `remote_worker_disconnected` 错误。

        ★ 2026-10-07（DIST-NEXT-4b）：`reason_code` 由调用方给出（断线 / 心跳过期 /
        服务重建），调用方据此记**单一**撤销事件；错误码本身保持
        `remote_worker_disconnected`（等待方语义不变）。
        """
        with self._lock:
            for pending in self._pending.values():
                error = ProviderExecutionError(
                    "remote worker disconnected",
                    code="remote_worker_disconnected",
                    provider_id=self.provider_id,
                    retryable=True,
                )
                pending.decide_terminal("disconnect", error=error)
                pending.cancel_ack_event.set()
            reservation_ids = list(self._reservations.keys())
        # `release()` 内部自己取 `self._lock` ⇒ 必须在锁外调用（这里是普通 Lock，
        # 锁内再取会自锁）。
        for reservation_id in reservation_ids:
            self.release(reservation_id)
        if reservation_ids:
            logger.info(
                "event=task_worker_reservations_released node_id=%s reason=%s "
                "released=%d",
                self.node_id, reason_code, len(reservation_ids),
            )

    def release_stale_reservations(self, reason_code: str) -> list[str]:
        """★ 2026-10-07（DIST-NEXT-4b）：撤销**没有在跑 attempt** 的 reservation。

        与 [notify_disconnect] 的区别：断线是「对端确定没了」⇒ 全部回收；心跳过期
        时对端可能只是心跳线程失效、stage 仍在跑 ⇒ 只回收**已终结**（无 pending 或
        pending 的 `result_event` 已置）的条目，避免打断 in-flight 执行。返回被撤销的
        reservation id，供调用方按 `reason_code` 记一条单一事件。
        """
        with self._lock:
            freed: list[str] = []
            for reservation_id in list(self._reservations.keys()):
                attempt_id = self._reservation_attempts.get(reservation_id, "")
                pending = self._pending.get(attempt_id) if attempt_id else None
                if pending is not None and not pending.result_event.is_set():
                    continue        # in-flight ⇒ 交给取消/租约路径收敛
                freed.append(reservation_id)
        # `release()` 自带锁 ⇒ 锁外调用。
        for reservation_id in freed:
            self.release(reservation_id)
        return freed

    def release(self, reservation_id: str) -> None:
        with self._lock:
            self._reservations.pop(reservation_id, None)
            self._executed_reservations.discard(reservation_id)
            attempt_id = self._reservation_attempts.pop(
                reservation_id, "",
            )
            pending = self._pending.get(attempt_id)
            if pending is not None:
                pending.released = True
                pending.released_at = time.monotonic()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.notify_disconnect()
        with self._lock:
            self._reservations.clear()
            self._executed_reservations.clear()
            self._reservation_attempts.clear()
            self._pending.clear()
        self._outbound_stop.set()
