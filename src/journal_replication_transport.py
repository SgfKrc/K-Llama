"""Authenticated framed TCP transport for durable journal checkpoints.

The transport is intentionally small and synchronous. Deployments should run
it over an authenticated private network or TLS tunnel; the HMAC binds the
declared source identity and checkpoint to a shared secret, but does not
provide confidentiality or quorum commitment.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
import threading
from typing import Any, Mapping

from journal_replication import (
    DurableJournalReplica,
    JournalCheckpoint,
    JournalReplicationError,
    REPLICATION_SCHEMA_VERSION,
)


TRANSPORT_SCHEMA_VERSION = "qlh.journal.replication.transport.v1"
DEFAULT_MAX_FRAME_BYTES = 32 * 1024 * 1024


class JournalReplicationTransportError(ConnectionError):
    """Transport, authentication, or remote apply failure."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JournalReplicationTransportError("transport payload is not canonical JSON") from exc


def _mac(secret: bytes, value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "mac"}
    return hmac.new(secret, _canonical(unsigned), hashlib.sha256).hexdigest()


def _recv_exact(connection: socket.socket, length: int, max_frame_bytes: int) -> bytes:
    if length < 0 or length > max_frame_bytes:
        raise JournalReplicationTransportError("invalid frame length")
    output = bytearray()
    while len(output) < length:
        chunk = connection.recv(length - len(output))
        if not chunk:
            raise JournalReplicationTransportError("peer closed before frame completed")
        output.extend(chunk)
    return bytes(output)


def _read_frame(connection: socket.socket, max_frame_bytes: int) -> dict[str, Any]:
    (length,) = struct.unpack("!I", _recv_exact(connection, 4, max_frame_bytes))
    if length == 0:
        raise JournalReplicationTransportError("empty frame")
    try:
        value = json.loads(_recv_exact(connection, length, max_frame_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JournalReplicationTransportError("invalid JSON frame") from exc
    if not isinstance(value, dict):
        raise JournalReplicationTransportError("frame must contain an object")
    return value


def _write_frame(connection: socket.socket, value: Mapping[str, Any], max_frame_bytes: int) -> None:
    body = _canonical(value)
    if not body or len(body) > max_frame_bytes:
        raise JournalReplicationTransportError("frame exceeds configured limit")
    connection.sendall(struct.pack("!I", len(body)) + body)


class JournalReplicationTcpServer:
    """Single-checkpoint TCP receiver backed by ``DurableJournalReplica``."""

    def __init__(
        self,
        replica: DurableJournalReplica,
        *,
        bind_host: str = "127.0.0.1",
        port: int = 0,
        secret: bytes,
        timeout: float = 5.0,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        if len(secret) < 32:
            raise JournalReplicationTransportError("replication secret must be at least 32 bytes")
        if max_frame_bytes < 1024:
            raise JournalReplicationTransportError("max_frame_bytes must be at least 1024")
        self.replica = replica
        self.bind_host = bind_host
        self.port = int(port)
        self.secret = bytes(secret)
        self.timeout = max(0.1, float(timeout))
        self.max_frame_bytes = int(max_frame_bytes)
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._ready = threading.Event()
        self.address: tuple[str, int] | None = None

    def start(self) -> tuple[str, int]:
        if self._thread is not None:
            raise JournalReplicationTransportError("server has already been started")
        self._thread = threading.Thread(
            target=self._serve,
            name="journal-replication-tcp",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(self.timeout):
            self.close()
            raise JournalReplicationTransportError("replication receiver startup timed out")
        if self._startup_error is not None:
            raise JournalReplicationTransportError("replication receiver failed to start") from self._startup_error
        assert self.address is not None
        return self.address

    def _serve(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.bind_host, self.port))
            listener.listen(8)
            listener.settimeout(0.2)
            self._listener = listener
            bound = listener.getsockname()
            self.address = (str(bound[0]), int(bound[1]))
            self._ready.set()
            while not self._stop.is_set():
                try:
                    connection, _peer = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                with connection:
                    connection.settimeout(self.timeout)
                    self._handle(connection)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            try:
                listener.close()
            except OSError:
                pass

    def _dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """按 `operation` 分派（#93 起两条：推入 ``apply_checkpoint`` / 拉取 ``get_checkpoint``）。"""

        operation = request.get("operation")
        if operation == "apply_checkpoint":
            raw_checkpoint = request.get("checkpoint")
            if not isinstance(raw_checkpoint, Mapping):
                raise JournalReplicationTransportError("checkpoint_required")
            checkpoint = JournalCheckpoint.from_dict(raw_checkpoint)
            if request.get("source_node_id") != checkpoint.source_node_id:
                raise JournalReplicationTransportError("source_identity_mismatch")
            return self.replica.apply_checkpoint(checkpoint)
        if operation == "get_checkpoint":
            # #93：跨机追赶的**拉取**方向。只把已自洽的前缀交出去 ——
            # `export_checkpoint` 自身会在不自洽时拒绝，这里不做任何降级。
            workflow_id = request.get("workflow_id")
            if not isinstance(workflow_id, str) or not workflow_id:
                raise JournalReplicationTransportError("workflow_id_required")
            if request.get("source_node_id") != self.replica.source_node_id:
                raise JournalReplicationTransportError("source_identity_mismatch")
            return {"checkpoint": self.replica.export_checkpoint(workflow_id).to_dict()}
        raise JournalReplicationTransportError("unsupported_operation")

    def _handle(self, connection: socket.socket) -> None:
        try:
            request = _read_frame(connection, self.max_frame_bytes)
            supplied_mac = request.get("mac")
            if not isinstance(supplied_mac, str) or not hmac.compare_digest(
                supplied_mac, _mac(self.secret, request)
            ):
                raise JournalReplicationTransportError("authentication_failed")
            if request.get("transport_schema") != TRANSPORT_SCHEMA_VERSION:
                raise JournalReplicationTransportError("unsupported_transport_schema")
            result = self._dispatch(request)
            response: dict[str, Any] = {
                "transport_schema": TRANSPORT_SCHEMA_VERSION,
                "ok": True,
                "result": result,
            }
        except (JournalReplicationError, JournalReplicationTransportError, ValueError):
            response = {
                "transport_schema": TRANSPORT_SCHEMA_VERSION,
                "ok": False,
                "error": "replication_rejected",
            }
        except (OSError, socket.timeout):
            return
        try:
            response["mac"] = _mac(self.secret, response)
            _write_frame(connection, response, self.max_frame_bytes)
        except (OSError, JournalReplicationTransportError):
            return

    def close(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.timeout + 1.0)
            if thread.is_alive():
                raise JournalReplicationTransportError("replication receiver did not stop")

    def __enter__(self) -> "JournalReplicationTcpServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def send_checkpoint(
    host: str,
    port: int,
    checkpoint: JournalCheckpoint,
    *,
    secret: bytes,
    timeout: float = 5.0,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> dict[str, Any]:
    """Send one authenticated checkpoint and verify its durable ACK."""

    if len(secret) < 32:
        raise JournalReplicationTransportError("replication secret must be at least 32 bytes")
    request: dict[str, Any] = {
        "transport_schema": TRANSPORT_SCHEMA_VERSION,
        "operation": "apply_checkpoint",
        "source_node_id": checkpoint.source_node_id,
        "checkpoint": checkpoint.to_dict(),
    }
    request["mac"] = _mac(secret, request)
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.1, timeout)) as connection:
            connection.settimeout(max(0.1, timeout))
            _write_frame(connection, request, max_frame_bytes)
            response = _read_frame(connection, max_frame_bytes)
    except (OSError, JournalReplicationTransportError) as exc:
        raise JournalReplicationTransportError("journal replication exchange failed") from exc
    supplied_mac = response.get("mac")
    if not isinstance(supplied_mac, str) or not hmac.compare_digest(
        supplied_mac, _mac(secret, response)
    ):
        raise JournalReplicationTransportError("replication acknowledgement authentication failed")
    if response.get("transport_schema") != TRANSPORT_SCHEMA_VERSION:
        raise JournalReplicationTransportError("unsupported acknowledgement schema")
    if response.get("ok") is not True or not isinstance(response.get("result"), dict):
        raise JournalReplicationTransportError("remote replica rejected checkpoint")
    result = response["result"]
    if (
        result.get("source_node_id") != checkpoint.source_node_id
        or result.get("stream_id") != checkpoint.stream_id
        or result.get("workflow_id") != checkpoint.workflow_id
        or int(result.get("durable_sequence", 0)) != checkpoint.last_sequence
        or result.get("checkpoint_digest") != checkpoint.checkpoint_digest
    ):
        raise JournalReplicationTransportError("replica acknowledgement watermark mismatch")
    return result


def request_checkpoint(
    host: str,
    port: int,
    *,
    source_node_id: str,
    stream_id: str,
    workflow_id: str,
    secret: bytes,
    timeout: float = 5.0,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> JournalCheckpoint:
    """#93：从对端**拉**一个已自洽的 checkpoint（落后副本的追赶方向）。

    与 `send_checkpoint` 严格对偶：同样带 MAC、同样校验身份。对端只会在**自洽**时
    交出前缀（见 `DurableJournalReplica.export_checkpoint`），因此这里拿到的必然是
    「可以当成已提交前缀」的那一点。
    """

    if len(secret) < 32:
        raise JournalReplicationTransportError("replication secret must be at least 32 bytes")
    if not workflow_id or not source_node_id or not stream_id:
        raise JournalReplicationTransportError("checkpoint identity is required")
    request: dict[str, Any] = {
        "transport_schema": TRANSPORT_SCHEMA_VERSION,
        "operation": "get_checkpoint",
        "source_node_id": source_node_id,
        "workflow_id": workflow_id,
    }
    request["mac"] = _mac(secret, request)
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.1, timeout)) as connection:
            connection.settimeout(max(0.1, timeout))
            _write_frame(connection, request, max_frame_bytes)
            response = _read_frame(connection, max_frame_bytes)
    except (OSError, JournalReplicationTransportError) as exc:
        raise JournalReplicationTransportError("journal replication exchange failed") from exc
    supplied_mac = response.get("mac")
    if not isinstance(supplied_mac, str) or not hmac.compare_digest(
        supplied_mac, _mac(secret, response)
    ):
        raise JournalReplicationTransportError("replication acknowledgement authentication failed")
    if response.get("transport_schema") != TRANSPORT_SCHEMA_VERSION:
        raise JournalReplicationTransportError("unsupported acknowledgement schema")
    if response.get("ok") is not True or not isinstance(response.get("result"), dict):
        raise JournalReplicationTransportError("remote replica rejected the request")
    raw = response["result"].get("checkpoint")
    if not isinstance(raw, Mapping):
        raise JournalReplicationTransportError("remote replica did not return a checkpoint")
    checkpoint = JournalCheckpoint.from_dict(raw)
    if (
        checkpoint.source_node_id != source_node_id
        or checkpoint.stream_id != stream_id
        or checkpoint.workflow_id != workflow_id
    ):
        raise JournalReplicationTransportError("returned checkpoint identity mismatch")
    return checkpoint


__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "JournalReplicationTcpServer",
    "JournalReplicationTransportError",
    "TRANSPORT_SCHEMA_VERSION",
    "request_checkpoint",
    "send_checkpoint",
]
