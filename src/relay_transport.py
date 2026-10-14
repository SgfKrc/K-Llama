"""Framed loopback transport for experimental Relay hidden-state handoff.

The transport is deliberately independent from both inference engines. It carries raw
little-endian hidden tensors to the existing stdio runner while enforcing framing, bounds,
ordering, and loopback-only exposure. Passing this transport does not open the Relay
production admission gate.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import BinaryIO, Sequence

try:  # 兼容两种入口：`import relay_transport`（src 在 sys.path）与 `from src.relay_transport import …`
    from relay_hidden_quant import (
        HIDDEN_QUANT_MODES,
        decode_hidden,
        encode_hidden,
        expected_quantized_bytes,
    )
except ImportError:  # pragma: no cover - 包路径导入
    from src.relay_hidden_quant import (
        HIDDEN_QUANT_MODES,
        decode_hidden,
        encode_hidden,
        expected_quantized_bytes,
    )

RELAY_WIRE_MAGIC = b"QLHR"
RELAY_WIRE_VERSION = 1
RELAY_DTYPE = "float32_le"
RELAY_DTYPE_BYTES = 4
#: ★ 2026-09-23（A5）：帧头 `flags` 的**低 3 位** = hidden 压缩档（0 = f32 无压缩）。
#: 旧实现对任何非零 flags 直接判 `unsupported_flags` ⇒ 新档位对旧对端**天然 fail-closed**，
#: 因此**不需要升 version**：v1 客户端发 `flags=0` 时与旧行为逐字节一致。
RELAY_FLAG_QUANT_MASK = 0b111
#: 档位 ↔ flags 低 3 位的映射（顺序即编码；见 `relay_hidden_quant.HIDDEN_QUANT_MODES`）。
RELAY_QUANT_MODES = tuple(HIDDEN_QUANT_MODES)
RELAY_QUANT_CODES = {mode: index for index, mode in enumerate(RELAY_QUANT_MODES)}
RELAY_DEFAULT_MAX_TOKENS = 4096
_UINT32_MAX = (1 << 32) - 1
RELAY_DEFAULT_MAX_PAYLOAD = 256 * 1024 * 1024
#: ★ P3：`HIDDEN_SEQ` 帧的元数据（seq/pos）上限 —— 帧校验按 hidden 字节 + 该上限放宽。
RELAY_SEQ_META_LIMIT = 64 * 1024

_HEADER = struct.Struct("!4sBBHIIQ")
_TOKEN = struct.Struct("<i")
_COUNT = struct.Struct("<i")
#: ★ P3：`HIDDEN_SEQ` payload 头 = `n_tokens` + 元数据字节数（其后是 meta JSON + f32 数据）。
_SEQ_HEADER = struct.Struct("<II")
_RELAY_ERROR_PAYLOAD_LIMIT = 64

logger = logging.getLogger(__name__)

# These are the only values allowed to cross the Relay trust boundary.  The
# existing protocol codes remain stable; implementation failures are grouped
# so exception class names and messages never become wire data.
RELAY_REMOTE_ERROR = "remote_error"
RELAY_PROTOCOL_ERROR = "relay_protocol_error"
RELAY_TRANSPORT_ERROR = "relay_transport_error"
RELAY_INTERNAL_ERROR = "relay_internal_error"
RELAY_RUNNER_ERROR = "runner_failed"
REQUEST_DEADLINE_EXCEEDED = 'request_deadline_exceeded'
_RELAY_ERROR_CODES = frozenset({
    "client_closed",
    "connection_closed_mid_frame",
    "hidden_frame_required",
    "hidden_payload_size_mismatch",
    "hidden_seq_frame_required",
    "hidden_seq_meta_invalid",
    "hidden_seq_meta_missing",
    "hidden_seq_meta_shape_invalid",
    "hidden_seq_meta_too_large",
    "hidden_seq_meta_truncated",
    "hidden_seq_meta_unknown",
    "hidden_seq_multi_membership_unsupported",
    "hidden_seq_payload_too_small",
    "hidden_seq_token_count_mismatch",
    "hidden_seq_unsupported",
    "hidden_sequence_mode_changed",
    #: ★ P4.5：远端不支持"送 token 跑上游段"这种请求（旧 runner）时 fail-loud。
    "token_frame_unsupported",
    #: ★ P4.5：`TOKENS` 帧里声明的 token 数与 payload 实际长度不一致。
    "token_frame_count_mismatch",    "invalid_close_ack",
    "invalid_close_frame",
    "invalid_frame_header",
    "invalid_hidden_shape",
    "invalid_token_response",
    "invalid_transport_limits",
    "non_loopback_bind_rejected",
    "non_loopback_endpoint_rejected",
    "payload_too_large",
    "request_sequence_mismatch",
    "response_sequence_mismatch",
    "runner_closed_mid_response",
    "runner_failed",
    "runner_not_available",
    "runner_pipe_missing",
    "token_count_exceeds_limit",
    "unknown_frame_kind",
    "unsupported_flags",
    #: ★ A5：hidden 压缩档不受支持（旧对端 / 未知档位）—— 跨信任边界的稳定错误码。
    "unsupported_hidden_quant",
    "unsupported_version",
    RELAY_INTERNAL_ERROR,
    RELAY_PROTOCOL_ERROR,
    RELAY_REMOTE_ERROR,
    RELAY_TRANSPORT_ERROR,
    REQUEST_DEADLINE_EXCEEDED,
})


class RelayFrameKind(IntEnum):
    HIDDEN = 1
    TOKEN = 2
    CLOSE = 3
    ERROR = 4
    #: ★ P3：带 `seq_ids` / `positions` 的 hidden 请求（响应仍是纯 `HIDDEN`）——
    #: 多序列数据流跨机时必须逐 token 显式绑定，不能靠远端隐式位置递增。
    HIDDEN_SEQ = 5
    #: ★ P4.5：**上游段请求** —— 客户端送 token id 列表，远端跑自己的 head 段并回 `HIDDEN`。
    #: 用途：无 PC 集群下层接力要能"全设备"运行（head 段也在设备上），
    #: 与 `HIDDEN`（吃 hidden 吐 hidden）/ `TOKEN`（吃 hidden 吐 token）配成三种远端角色。
    TOKENS = 6


class RelayProtocolError(RuntimeError):
    """The peer or runner violated the experimental Relay wire contract."""


@dataclass(frozen=True)
class RelayFrame:
    kind: RelayFrameKind
    sequence: int
    n_tokens: int = 0
    payload: bytes = b""
    #: ★ 2026-09-23（A5）：payload 里 hidden 的压缩档（`none` = f32 原样；见 `RELAY_QUANT_CODES`）。
    quant: str = "none"


@dataclass(frozen=True)
class RelayBridgeResult:
    frames: int
    tokens: int
    payload_bytes: int
    closed_cleanly: bool
    error: str = ""

    def to_dict(self) -> dict[str, int | bool | str]:
        return asdict(self)


def is_loopback_host(host: str) -> bool:
    candidate = str(host).strip().lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def expected_hidden_bytes(n_tokens: int, n_embd: int, quant: str = "none") -> int:
    """该档位下 hidden 的字节数（`quant="none"` ⇒ f32，与旧行为一致）。

    ★ 2026-09-23（A5）：压缩档会改变 payload 长度 ⇒ 长度校验必须**按档位换算**，否则压缩帧要么被
    当成"尺寸不符"拒掉，要么（更糟）被当成合法的 f32 数据喂进 runner。
    """
    tokens = int(n_tokens)
    width = int(n_embd)
    if tokens < 1 or width < 1:
        raise RelayProtocolError("invalid_hidden_shape")
    if quant in (None, "none"):
        return tokens * width * RELAY_DTYPE_BYTES
    if quant not in RELAY_QUANT_CODES:
        raise RelayProtocolError("unsupported_hidden_quant")
    return expected_quantized_bytes(quant, tokens, width)


def quantize_upload(hidden: bytes, n_tokens: int, n_embd: int,
                    quant: str = "none") -> bytes:
    """★ A5：把**法线化后的 f32 hidden** 压成该档位的 wire payload。

    调用方永远给 f32（`expected_hidden_bytes(count, n_embd)` 校验过的长度）⇒ 压缩只发生在 wire 上。
    `none` 档原样返回（与旧行为逐字节一致）；未知档位 fail-loud（不静默退回未压缩）。
    """
    mode = quant or "none"
    if mode == "none":
        return bytes(hidden)
    try:
        return encode_hidden(hidden, mode, int(n_tokens), int(n_embd))
    except ValueError as exc:
        raise RelayProtocolError("unsupported_hidden_quant") from exc


def frame_hidden_bytes(frame: RelayFrame, *, n_embd: int) -> bytes:
    """★ A5：把帧里的 hidden 解回 **f32 bytes**（`none` 档原样返回）—— runner 只认 f32。

    设计意图：**压缩只发生在 wire 上**，段实现（shim / pip 引擎 / Android JNI）完全无感；
    所以旧对端收到 `flags=0` 的帧行为不变，而新对端可以选档位省带宽。
    """
    quant = getattr(frame, "quant", "none") or "none"
    if quant == "none":
        return bytes(frame.payload)
    try:
        return decode_hidden(frame.payload, quant, int(frame.n_tokens), int(n_embd))
    except ValueError as exc:
        raise RelayProtocolError("hidden_payload_size_mismatch") from exc


def _validate_hidden_seq_meta(meta: object, n_tokens: int) -> None:
    """Validate the bounded, per-token metadata carried by ``HIDDEN_SEQ``."""

    if not isinstance(meta, dict):
        raise RelayProtocolError("hidden_seq_meta_invalid")
    allowed = {"n_seq_id", "seq_ids", "positions"}
    unknown = [key for key in meta if key not in allowed]
    if unknown:
        raise RelayProtocolError(f"hidden_seq_meta_unknown:{unknown[0]}")

    # ★ decode 正确性（安全评审 #5）：`seq_ids` 与 `positions` **必须同时给出** ——
    #   只给其一（或给空 meta）会静默退回"远端按 `n_past` 隐式自增"，那正是本协议
    #   **明确禁止**的行为（见 `serve_relay_middle_connection` 处「绝不退回远端按隐式
    #   位置猜」的注释）。宁可 fail-closed，也不产出"位置可能错"的 hidden。
    if "seq_ids" not in meta or "positions" not in meta:
        raise RelayProtocolError("hidden_seq_meta_missing")

    for key in ("n_seq_id", "seq_ids", "positions"):
        if key not in meta:
            continue
        values = meta[key]
        if not isinstance(values, (list, tuple)) or len(values) != n_tokens:
            raise RelayProtocolError("hidden_seq_meta_shape_invalid")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RelayProtocolError("hidden_seq_meta_shape_invalid")
        if key == "n_seq_id" and any(value < 1 for value in values):
            raise RelayProtocolError("hidden_seq_meta_shape_invalid")
        if key == "n_seq_id" and any(value != 1 for value in values):
            raise RelayProtocolError("hidden_seq_multi_membership_unsupported")


def encode_hidden_seq(hidden: bytes, *, n_tokens: int, meta: dict[str, object],
                      quant: str = "none", n_embd: int | None = None) -> bytes:
    """★ P3：把 `hidden` + 元数据打包成 `HIDDEN_SEQ` 的 payload。

    布局：`<n_tokens:u32><n_meta_bytes:u32><meta JSON><hidden（按 quant 档）>`。
    元数据只允许 `n_seq_id` / `seq_ids` / `positions`（其余键一律拒绝，避免协议被当通用通道）。

    ★ A5：`quant != "none"` 时**必须**给 `n_embd`（压缩要按宽度分块）。
    """
    count = int(n_tokens)
    if isinstance(n_tokens, bool) or count < 1 or count > _UINT32_MAX:
        raise RelayProtocolError("invalid_hidden_shape")
    _validate_hidden_seq_meta(meta, count)
    try:
        meta_bytes = json.dumps(meta, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RelayProtocolError("hidden_seq_meta_invalid") from exc
    if len(meta_bytes) > RELAY_SEQ_META_LIMIT:
        raise RelayProtocolError("hidden_seq_meta_too_large")
    mode = quant or "none"
    if mode == "none":
        body = bytes(hidden)
    else:
        width = int(n_embd or 0)
        if width < 1:
            raise RelayProtocolError("invalid_hidden_shape")
        body = quantize_upload(hidden, count, width, mode)
    return _SEQ_HEADER.pack(count, len(meta_bytes)) + meta_bytes + body


def decode_hidden_seq(payload: bytes, *, n_embd: int,
                      quant: str = "none") -> tuple[bytes, int, dict[str, object]]:
    """★ P3：拆 `HIDDEN_SEQ` payload → `(hidden(f32), n_tokens, meta)`，形状不符即拒。

    ★ A5：`quant != "none"` 时按档位换算长度并**解回 f32**（对 runner 透明）。
    """
    if len(payload) < _SEQ_HEADER.size:
        raise RelayProtocolError("hidden_seq_payload_too_small")
    n_tokens, meta_len = _SEQ_HEADER.unpack_from(payload, 0)
    if meta_len > RELAY_SEQ_META_LIMIT:
        raise RelayProtocolError("hidden_seq_meta_too_large")
    start = _SEQ_HEADER.size
    end = start + int(meta_len)
    if end > len(payload):
        raise RelayProtocolError("hidden_seq_meta_truncated")
    try:
        meta = json.loads(payload[start:end].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelayProtocolError("hidden_seq_meta_invalid") from exc
    _validate_hidden_seq_meta(meta, int(n_tokens))
    hidden = payload[end:]
    if len(hidden) != expected_hidden_bytes(int(n_tokens), int(n_embd), quant):
        raise RelayProtocolError("hidden_payload_size_mismatch")
    mode = quant or "none"
    if mode != "none":
        try:
            hidden = decode_hidden(hidden, mode, int(n_tokens), int(n_embd))
        except ValueError as exc:
            raise RelayProtocolError("hidden_payload_size_mismatch") from exc
    return hidden, int(n_tokens), meta


def _io_timeout(
    sock: socket.socket,
    *,
    deadline_monotonic: float | None,
    timeout: float | None,
    deadline_error: str,
) -> bool:
    """Apply the earliest I/O boundary and report whether deadline was limiting."""
    deadline_limited = False
    budget = None if timeout is None else max(0.0, float(timeout))
    if deadline_monotonic is not None:
        remaining = float(deadline_monotonic) - time.monotonic()
        if remaining <= 0:
            raise RelayProtocolError(deadline_error)
        if budget is None or remaining <= budget:
            budget = remaining
            deadline_limited = True
    if budget is not None:
        if budget <= 0:
            raise RelayProtocolError(deadline_error)
        sock.settimeout(max(1e-6, budget))
    return deadline_limited


def _recv_exact(
    sock: socket.socket,
    size: int,
    *,
    deadline_monotonic: float | None = None,
    timeout: float | None = None,
    deadline_error: str = REQUEST_DEADLINE_EXCEEDED,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        deadline_limited = _io_timeout(
            sock,
            deadline_monotonic=deadline_monotonic,
            timeout=timeout,
            deadline_error=deadline_error,
        )
        try:
            chunk = sock.recv(remaining)
        except socket.timeout as exc:
            if deadline_limited or (
                deadline_monotonic is not None
                and time.monotonic() >= float(deadline_monotonic)
            ):
                raise RelayProtocolError(deadline_error) from exc
            raise
        if not chunk:
            raise RelayProtocolError("connection_closed_mid_frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(
    sock: socket.socket,
    frame: RelayFrame,
    *,
    deadline_monotonic: float | None = None,
    timeout: float | None = None,
    deadline_error: str = REQUEST_DEADLINE_EXCEEDED,
) -> None:
    payload = bytes(frame.payload)
    quant = getattr(frame, "quant", "none") or "none"
    code = RELAY_QUANT_CODES.get(quant)
    if code is None:
        raise RelayProtocolError("unsupported_hidden_quant")
    try:
        header = _HEADER.pack(
            RELAY_WIRE_MAGIC,
            RELAY_WIRE_VERSION,
            int(frame.kind),
            code,
            int(frame.sequence),
            int(frame.n_tokens),
            len(payload),
        )
    except (OverflowError, struct.error, ValueError) as exc:
        raise RelayProtocolError("invalid_frame_header") from exc
    deadline_limited = _io_timeout(
        sock,
        deadline_monotonic=deadline_monotonic,
        timeout=timeout,
        deadline_error=deadline_error,
    )
    try:
        sock.sendall(header + payload)
    except socket.timeout as exc:
        if deadline_limited or (
            deadline_monotonic is not None
            and time.monotonic() >= float(deadline_monotonic)
        ):
            raise RelayProtocolError(deadline_error) from exc
        raise


def recv_frame(
    sock: socket.socket,
    *,
    max_payload_bytes: int = RELAY_DEFAULT_MAX_PAYLOAD,
    deadline_monotonic: float | None = None,
    timeout: float | None = None,
    deadline_error: str = REQUEST_DEADLINE_EXCEEDED,
) -> RelayFrame:
    raw = _recv_exact(
        sock,
        _HEADER.size,
        deadline_monotonic=deadline_monotonic,
        timeout=timeout,
        deadline_error=deadline_error,
    )
    magic, version, kind_value, flags, sequence, n_tokens, payload_size = _HEADER.unpack(raw)
    if magic != RELAY_WIRE_MAGIC:
        raise RelayProtocolError("invalid_magic")
    if version != RELAY_WIRE_VERSION:
        raise RelayProtocolError("unsupported_version")
    # ★ A5：低 3 位是 hidden 压缩档；其余位仍然未定义 ⇒ 必须拒（未知 flags 不能当合法帧）。
    if flags & ~RELAY_FLAG_QUANT_MASK:
        raise RelayProtocolError("unsupported_flags")
    quant_code = flags & RELAY_FLAG_QUANT_MASK
    if quant_code >= len(RELAY_QUANT_MODES):
        raise RelayProtocolError("unsupported_hidden_quant")
    quant = RELAY_QUANT_MODES[quant_code]
    try:
        kind = RelayFrameKind(kind_value)
    except ValueError as exc:
        raise RelayProtocolError("unknown_frame_kind") from exc
    if payload_size > int(max_payload_bytes):
        raise RelayProtocolError("payload_too_large")
    payload = (
        _recv_exact(
            sock,
            payload_size,
            deadline_monotonic=deadline_monotonic,
            timeout=timeout,
            deadline_error=deadline_error,
        )
        if payload_size
        else b""
    )
    return RelayFrame(kind=kind, sequence=sequence, n_tokens=n_tokens, payload=payload,
                      quant=quant)


def encode_tokens(tokens: Sequence[int]) -> bytes:
    """★ P4.5：上游段请求的 payload —— 紧凑 i32 数组（token id 列表）。"""
    values = [int(t) for t in tokens]
    if not values:
        raise RelayProtocolError("token_count_exceeds_limit")
    return b"".join(_TOKEN.pack(v) for v in values)


def decode_tokens(payload: bytes, *, limit: int) -> list[int]:
    """解析 `TOKENS` 帧的 payload；长度必须是 4 的倍数且不超过 `limit` 个 token。"""
    size = _TOKEN.size
    if not payload or len(payload) % size != 0:
        raise RelayProtocolError("invalid_token_response")
    count = len(payload) // size
    if count < 1 or count > int(limit):
        raise RelayProtocolError("token_count_exceeds_limit")
    return [int(_TOKEN.unpack_from(payload, i * size)[0]) for i in range(count)]


def _error_code_from_frame(frame: RelayFrame, expected_sequence: int) -> str:
    """从 `ERROR` 帧解出**稳定码**（不合法一律回落 `remote_error`）。

    ★ 2026-09-24：此前只有 token 返回路径（`_decode_token`）解 ERROR 帧；三条 **hidden 返回路径**
    （`request_hidden` / `request_hidden_seq` / `request_hidden_from_tokens`）直接把它当成
    "非 HIDDEN 帧" ⇒ 抛 `hidden_response_required`，于是服务端**明明发来**的 `runner_failed`
    / `hidden_payload_size_mismatch` 等稳定码被丢掉（而 `hidden_response_required` 本身不在
    白名单 ⇒ 调用方最终只看到笼统的 `relay_protocol_error`）。这直接破坏"具名回退"：
    `_fallback_reason` 写不出真实原因，失败会退化成"与模型算错难以区分"。三条路径现在共用本函数。
    """
    if frame.sequence != expected_sequence:
        raise RelayProtocolError("response_sequence_mismatch")
    if len(frame.payload) > _RELAY_ERROR_PAYLOAD_LIMIT:
        return RELAY_REMOTE_ERROR
    try:
        code = frame.payload.decode("ascii")
    except UnicodeDecodeError:
        return RELAY_REMOTE_ERROR
    return code if code in _RELAY_ERROR_CODES else RELAY_REMOTE_ERROR


def _decode_token(frame: RelayFrame, expected_sequence: int) -> int:
    if frame.kind == RelayFrameKind.ERROR:
        raise RelayProtocolError(_error_code_from_frame(frame, expected_sequence))
    if frame.sequence != expected_sequence:
        raise RelayProtocolError("response_sequence_mismatch")
    if frame.kind != RelayFrameKind.TOKEN or frame.n_tokens != 1 or len(frame.payload) != 4:
        raise RelayProtocolError("invalid_token_response")
    return _TOKEN.unpack(frame.payload)[0]


class RelayTcpClient:
    """One ordered Relay session over loopback or a local SSH tunnel endpoint."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        n_embd: int,
        timeout: float = 60.0,
        max_tokens: int = RELAY_DEFAULT_MAX_TOKENS,
        deadline_monotonic: float | None = None,
    ) -> None:
        if not is_loopback_host(host):
            raise RelayProtocolError("non_loopback_endpoint_rejected")
        if int(n_embd) < 1 or int(max_tokens) < 1:
            raise RelayProtocolError("invalid_transport_limits")
        self.host = "127.0.0.1" if str(host).lower() == "localhost" else str(host)
        self.port = int(port)
        self.n_embd = int(n_embd)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self._deadline_monotonic: float | None = None
        self.tighten_deadline(deadline_monotonic)
        self.max_payload_bytes = (expected_hidden_bytes(self.max_tokens, self.n_embd)
                                 + RELAY_SEQ_META_LIMIT + _SEQ_HEADER.size)
        connect_timeout = self.timeout
        deadline_limited = False
        if self._deadline_monotonic is not None:
            remaining = self._deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise RelayProtocolError(REQUEST_DEADLINE_EXCEEDED)
            if remaining <= connect_timeout:
                connect_timeout = remaining
                deadline_limited = True
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=max(1e-6, connect_timeout)
            )
        except socket.timeout as exc:
            if deadline_limited or (
                self._deadline_monotonic is not None
                and time.monotonic() >= self._deadline_monotonic
            ):
                raise RelayProtocolError(REQUEST_DEADLINE_EXCEEDED) from exc
            raise
        self._sock.settimeout(self.timeout)
        self._sequence = 0
        self._closed = False

    @property
    def deadline_monotonic(self) -> float | None:
        return self._deadline_monotonic

    def tighten_deadline(self, deadline_monotonic: float | None) -> None:
        if deadline_monotonic is None:
            return
        candidate = float(deadline_monotonic)
        if self._deadline_monotonic is None or candidate < self._deadline_monotonic:
            self._deadline_monotonic = candidate

    def _exchange(
        self,
        frame: RelayFrame,
        *,
        max_payload_bytes: int,
        deadline_monotonic: float | None = None,
    ) -> RelayFrame:
        if self._closed:
            raise RelayProtocolError("client_closed")
        self.tighten_deadline(deadline_monotonic)
        operation_deadline = time.monotonic() + self.timeout
        deadline_error = RELAY_TRANSPORT_ERROR
        if (
            self._deadline_monotonic is not None
            and self._deadline_monotonic <= operation_deadline
        ):
            operation_deadline = self._deadline_monotonic
            deadline_error = REQUEST_DEADLINE_EXCEEDED
        try:
            send_frame(
                self._sock,
                frame,
                deadline_monotonic=operation_deadline,
                deadline_error=deadline_error,
            )
            return recv_frame(
                self._sock,
                max_payload_bytes=max_payload_bytes,
                deadline_monotonic=operation_deadline,
                deadline_error=deadline_error,
            )
        except (OSError, RelayProtocolError):
            # A timeout, partial frame, or protocol failure leaves stream
            # alignment unknown. Such a connection is never reusable.
            self.abort()
            raise

    def _decode_token_response(self, response: RelayFrame, sequence: int) -> int:
        try:
            token = _decode_token(response, sequence)
            if token < 0:
                raise RelayProtocolError("runner_failed")
            return token
        except RelayProtocolError:
            self.abort()
            raise

    def _decode_hidden_response(
        self, response: RelayFrame, sequence: int, count: int
    ) -> bytes:
        try:
            if response.kind == RelayFrameKind.ERROR:
                raise RelayProtocolError(
                    _error_code_from_frame(response, sequence)
                )
            if response.sequence != sequence:
                raise RelayProtocolError("response_sequence_mismatch")
            if response.kind != RelayFrameKind.HIDDEN:
                raise RelayProtocolError("hidden_response_required")
            if response.n_tokens != count:
                raise RelayProtocolError("hidden_token_count_mismatch")
            if len(response.payload) != expected_hidden_bytes(
                count, self.n_embd, response.quant
            ):
                raise RelayProtocolError("hidden_payload_size_mismatch")
            return frame_hidden_bytes(response, n_embd=self.n_embd)
        except RelayProtocolError:
            self.abort()
            raise

    def request_token(
        self,
        hidden: bytes,
        *,
        n_tokens: int,
        quant: str = "none",
        seq_meta: dict[str, object] | None = None,
        deadline_monotonic: float | None = None,
    ) -> int:
        """Send hidden to a tail and receive its argmax token.

        ``seq_meta`` carries explicit per-token sequence/position bindings for
        decode. The legacy ``HIDDEN`` frame remains the default.
        """
        if self._closed:
            raise RelayProtocolError("client_closed")
        count = int(n_tokens)
        if count > self.max_tokens:
            raise RelayProtocolError("token_count_exceeds_limit")
        if len(hidden) != expected_hidden_bytes(count, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        if seq_meta is None:
            kind = RelayFrameKind.HIDDEN
            payload = quantize_upload(hidden, count, self.n_embd, quant)
        else:
            kind = RelayFrameKind.HIDDEN_SEQ
            payload = encode_hidden_seq(hidden, n_tokens=count, meta=seq_meta,
                                        quant=quant, n_embd=self.n_embd)
        sequence = self._sequence
        response = self._exchange(
            RelayFrame(
                kind,
                sequence,
                n_tokens=count,
                payload=payload,
                quant=quant or "none",
            ),
            max_payload_bytes=self.max_payload_bytes,
            deadline_monotonic=deadline_monotonic,
        )
        token = self._decode_token_response(response, sequence)
        self._sequence += 1
        return token

    def request_hidden_from_tokens(
        self,
        tokens: "Sequence[int]",
        *,
        deadline_monotonic: float | None = None,
    ) -> bytes:
        """★ P4.5 **上游段往返**：送 token id 列表，远端跑它自己的 head 段并回 `HIDDEN`。

        与 `request_hidden`（吃 hidden 吐 hidden）、`request_token`（吃 hidden 吐 token）
        配成三种远端角色 —— 三者齐备后，层接力可以**完全不依赖本机**（无 PC 集群场景）。

        远端语义必须与本机 keep-head 一致（末层输出、`output_norm` 之前）；响应
        必须带相同 `n_tokens`，否则 fail-loud。
        """
        if self._closed:
            raise RelayProtocolError("client_closed")
        values = [int(t) for t in tokens]
        count = len(values)
        if count < 1 or count > self.max_tokens:
            raise RelayProtocolError("token_count_exceeds_limit")
        sequence = self._sequence
        response = self._exchange(
            RelayFrame(RelayFrameKind.TOKENS, sequence, n_tokens=count,
                       payload=encode_tokens(values)),
            max_payload_bytes=self.max_payload_bytes,
            deadline_monotonic=deadline_monotonic,
        )
        hidden = self._decode_hidden_response(response, sequence, count)
        self._sequence += 1
        # ★ A5：下行档位由**服务端**决定 ⇒ 按帧里的档位解回 f32（对调用方永远是 f32）。
        return hidden

    def request_hidden(
        self,
        hidden: bytes,
        *,
        n_tokens: int,
        quant: str = "none",
        deadline_monotonic: float | None = None,
    ) -> bytes:
        """★ 中间段往返：发 HIDDEN，收 HIDDEN（远端段交出它自己的 hidden）。

        与 `request_token`（末段，收 token）配对 —— 这正是「1 个 torch 上游 + n 个
        llama 下游」链式拼接所需的**两种远端角色**：中间段吐 hidden、末段吐 token。

        远端实现的语义必须与本地 keep-head 一致（末层输出，`output_norm` 之前）；
        Android 侧由 `nativeLayerForwardHiddenKeepHead` 提供同一语义。

        ★ A5：`quant` 只作用于**上行**（本函数发出的帧）；响应仍按 f32 校验（下行压缩本轮未启用）。
        """
        if self._closed:
            raise RelayProtocolError("client_closed")
        count = int(n_tokens)
        if count > self.max_tokens:
            raise RelayProtocolError("token_count_exceeds_limit")
        if len(hidden) != expected_hidden_bytes(count, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        payload = quantize_upload(hidden, count, self.n_embd, quant)
        sequence = self._sequence
        response = self._exchange(
            RelayFrame(RelayFrameKind.HIDDEN, sequence, n_tokens=count, payload=payload,
                       quant=quant or "none"),
            max_payload_bytes=self.max_payload_bytes,
            deadline_monotonic=deadline_monotonic,
        )
        output = self._decode_hidden_response(response, sequence, count)
        self._sequence += 1
        # ★ A5：下行档位由**服务端**决定 ⇒ 按帧里的档位解回 f32（对调用方永远是 f32）。
        return output

    def request_hidden_seq(
        self,
        hidden: bytes,
        *,
        n_tokens: int,
        meta: dict[str, object],
        quant: str = "none",
        deadline_monotonic: float | None = None,
    ) -> bytes:
        """★ P3：**多序列**中间段往返 —— 请求帧带 `seq_ids` / `positions`（`HIDDEN_SEQ`）。

        响应仍是纯 `HIDDEN`（远端已按显式 seq/pos 算完）。元数据只允许
        `n_seq_id` / `seq_ids` / `positions`（见 `encode_hidden_seq`）。
        """
        if self._closed:
            raise RelayProtocolError("client_closed")
        count = int(n_tokens)
        if count > self.max_tokens:
            raise RelayProtocolError("token_count_exceeds_limit")
        if len(hidden) != expected_hidden_bytes(count, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        payload = encode_hidden_seq(hidden, n_tokens=count, meta=meta, quant=quant,
                                    n_embd=self.n_embd)
        sequence = self._sequence
        response = self._exchange(
            RelayFrame(RelayFrameKind.HIDDEN_SEQ, sequence, n_tokens=count, payload=payload,
                       quant=quant or "none"),
            max_payload_bytes=self.max_payload_bytes,
            deadline_monotonic=deadline_monotonic,
        )
        output = self._decode_hidden_response(response, sequence, count)
        self._sequence += 1
        # ★ A5：下行档位由**服务端**决定 ⇒ 按帧里的档位解回 f32（对调用方永远是 f32）。
        return output

    def abort(self) -> None:
        """Close immediately without sending CLOSE on a non-reusable stream."""
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def close(self, *, grace_timeout: float | None = None) -> None:
        if self._closed:
            return
        if grace_timeout is not None and float(grace_timeout) <= 0:
            self.abort()
            return
        close_timeout = self.timeout
        if grace_timeout is not None:
            close_timeout = min(close_timeout, float(grace_timeout))
        now = time.monotonic()
        close_deadline = now + close_timeout
        deadline_error = RELAY_TRANSPORT_ERROR
        if (
            self._deadline_monotonic is not None
            and self._deadline_monotonic <= close_deadline
        ):
            close_deadline = self._deadline_monotonic
            deadline_error = REQUEST_DEADLINE_EXCEEDED
        if close_deadline <= now:
            self.abort()
            return
        self._closed = True
        try:
            send_frame(
                self._sock,
                RelayFrame(RelayFrameKind.CLOSE, self._sequence),
                deadline_monotonic=close_deadline,
                deadline_error=deadline_error,
            )
            response = recv_frame(
                self._sock,
                max_payload_bytes=1024,
                deadline_monotonic=close_deadline,
                deadline_error=deadline_error,
            )
            token = _decode_token(response, self._sequence)
            if token != -1:
                raise RelayProtocolError("invalid_close_ack")
        finally:
            self._sock.close()

    def __enter__(self) -> "RelayTcpClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class StdioRelayRunner:
    """Adapter for the current ``int32 count + float32 hidden -> int32 token`` runner."""

    def __init__(self, command: Sequence[str], *, n_embd: int, timeout: float = 60.0) -> None:
        self.n_embd = int(n_embd)
        self.timeout = float(timeout)
        if self.n_embd < 1 or not command:
            raise RelayProtocolError("invalid_runner_configuration")
        self._process = subprocess.Popen(
            [str(part) for part in command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._closed = False

    @staticmethod
    def _read_pipe_exact(pipe: BinaryIO, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = pipe.read(remaining)
            if not chunk:
                raise RelayProtocolError("runner_closed_mid_response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def request_token(self, hidden: bytes, *, n_tokens: int) -> int:
        if self._closed or self._process.poll() is not None:
            raise RelayProtocolError("runner_not_available")
        if len(hidden) != expected_hidden_bytes(n_tokens, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:
            raise RelayProtocolError("runner_pipe_missing")
        stdin.write(_COUNT.pack(int(n_tokens)))
        stdin.write(hidden)
        stdin.flush()
        return _TOKEN.unpack(self._read_pipe_exact(stdout, _TOKEN.size))[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stdin = self._process.stdin
        stdout = self._process.stdout
        if self._process.poll() is None and stdin is not None and stdout is not None:
            try:
                stdin.write(_COUNT.pack(0))
                stdin.flush()
                self._read_pipe_exact(stdout, _TOKEN.size)
            except (BrokenPipeError, RelayProtocolError):
                pass
        try:
            self._process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)


def _safe_error_code(value: BaseException | str, fallback: str) -> str:
    candidate = getattr(value, "code", None) or str(value)
    return candidate if candidate in _RELAY_ERROR_CODES else fallback


def _send_error(sock: socket.socket, sequence: int, code: str) -> None:
    payload = _safe_error_code(code, RELAY_INTERNAL_ERROR).encode("ascii")
    if len(payload) > _RELAY_ERROR_PAYLOAD_LIMIT:
        payload = RELAY_INTERNAL_ERROR.encode("ascii")
    send_frame(sock, RelayFrame(RelayFrameKind.ERROR, sequence, payload=payload))


def serve_relay_connection(
    sock: socket.socket,
    runner,
    *,
    n_embd: int,
    max_tokens: int = RELAY_DEFAULT_MAX_TOKENS,
) -> RelayBridgeResult:
    """Forward one ordered client session to a runner and return bounded evidence stats."""

    width = int(n_embd)
    limit = int(max_tokens)
    max_payload = (expected_hidden_bytes(limit, width)
                   + RELAY_SEQ_META_LIMIT + _SEQ_HEADER.size)
    sequence = 0
    frames = 0
    tokens = 0
    payload_bytes = 0
    hidden_mode: str | None = None
    try:
        while True:
            frame = recv_frame(sock, max_payload_bytes=max_payload)
            if frame.sequence != sequence:
                raise RelayProtocolError("request_sequence_mismatch")
            if frame.kind == RelayFrameKind.CLOSE:
                if frame.n_tokens != 0 or frame.payload:
                    raise RelayProtocolError("invalid_close_frame")
                # ★ 只 `reset()`，**绝不 `close()`** —— 这是服务端引擎的生存期守卫。
                # `CLOSE` 的语义是"结束**本次会话**"，不是"销毁服务端引擎"：runner 属于**服务进程**、
                # 要跨连接复用，`close()` 会把底层 handle 置空 ⇒ 该服务此后**每个连接**都失败
                # （实测：帧完全正确却报 `rc=-5 参数非法`，现象与"模型算错"难以区分，只能靠重启恢复）。
                # 引擎的真正释放发生在进程退出时（见 `run_service` 的 finally）。
                logger.info("relay 段收到 CLOSE 帧 ⇒ runner.reset()（会话结束）")
                try:
                    runner.reset()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Relay runner reset failed: code=%s", RELAY_RUNNER_ERROR)
                    raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
                send_frame(
                    sock,
                    RelayFrame(RelayFrameKind.TOKEN, sequence, n_tokens=1, payload=_TOKEN.pack(-1)),
                )
                return RelayBridgeResult(frames, tokens, payload_bytes, True)
            if frame.kind in {RelayFrameKind.HIDDEN, RelayFrameKind.HIDDEN_SEQ}:
                frame_mode = (
                    "seq" if frame.kind == RelayFrameKind.HIDDEN_SEQ else "plain"
                )
                if hidden_mode is not None and frame_mode != hidden_mode:
                    raise RelayProtocolError("hidden_sequence_mode_changed")
                hidden_mode = frame_mode
            if frame.kind == RelayFrameKind.HIDDEN_SEQ:
                hidden, n_tokens, meta = decode_hidden_seq(
                    frame.payload, n_embd=width, quant=frame.quant)
                if n_tokens != frame.n_tokens:
                    raise RelayProtocolError("hidden_seq_token_count_mismatch")
                if n_tokens < 1 or n_tokens > limit:
                    raise RelayProtocolError("token_count_exceeds_limit")
                if not hasattr(runner, "request_token_seq"):
                    raise RelayProtocolError("hidden_seq_unsupported")
                try:
                    token = int(runner.request_token_seq(
                        hidden, n_tokens=n_tokens, meta=meta))
                except RelayProtocolError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Relay tail runner (seq) failed: code=%s",
                                     RELAY_RUNNER_ERROR)
                    raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
                if token < 0:
                    raise RelayProtocolError("runner_failed")
                send_frame(
                    sock,
                    RelayFrame(RelayFrameKind.TOKEN, sequence, n_tokens=1,
                               payload=_TOKEN.pack(token)),
                )
                frames += 1
                tokens += n_tokens
                payload_bytes += len(frame.payload)
                sequence += 1
                continue
            if frame.kind != RelayFrameKind.HIDDEN:
                raise RelayProtocolError("hidden_frame_required")
            if frame.n_tokens < 1 or frame.n_tokens > limit:
                raise RelayProtocolError("token_count_exceeds_limit")
            if len(frame.payload) != expected_hidden_bytes(frame.n_tokens, width, frame.quant):
                raise RelayProtocolError("hidden_payload_size_mismatch")

            try:
                # ★ A5：按帧里的档位**解回 f32** —— runner 只认 f32（压缩对段实现完全透明）。
                token = int(runner.request_token(frame_hidden_bytes(frame, n_embd=width),
                                                 n_tokens=frame.n_tokens))
            except RelayProtocolError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Relay runner request failed: code=%s", RELAY_RUNNER_ERROR)
                raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
            if token < 0:
                raise RelayProtocolError("runner_failed")
            send_frame(
                sock,
                RelayFrame(RelayFrameKind.TOKEN, sequence, n_tokens=1, payload=_TOKEN.pack(token)),
            )
            frames += 1
            tokens += frame.n_tokens
            payload_bytes += len(frame.payload)
            sequence += 1
    except RelayProtocolError as exc:
        code = _safe_error_code(exc, RELAY_PROTOCOL_ERROR)
        logger.warning("Relay protocol failure: code=%s detail=%s", code, str(exc))
        try:
            _send_error(sock, sequence, code)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, code)
    except OSError as exc:
        logger.warning("Relay transport failure: code=%s detail=%s", RELAY_TRANSPORT_ERROR, exc)
        try:
            _send_error(sock, sequence, RELAY_TRANSPORT_ERROR)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, RELAY_TRANSPORT_ERROR)
    except Exception as exc:  # noqa: BLE001
        # Never put exception class names or messages on the Relay wire.
        logger.exception("Relay internal failure: code=%s", RELAY_INTERNAL_ERROR)
        try:
            _send_error(sock, sequence, RELAY_INTERNAL_ERROR)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, RELAY_INTERNAL_ERROR)


def open_loopback_listener(host: str, port: int, *, backlog: int = 1) -> socket.socket:
    """Create a listener that cannot be exposed directly to a LAN or tailnet."""

    if not is_loopback_host(host):
        raise RelayProtocolError("non_loopback_bind_rejected")
    bind_host = "127.0.0.1" if str(host).lower() == "localhost" else str(host)
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind_host, int(port)))
    listener.listen(max(1, int(backlog)))
    return listener


def serve_relay_middle_connection(
    sock: socket.socket,
    runner,
    *,
    n_embd: int,
    max_tokens: int = RELAY_DEFAULT_MAX_TOKENS,
    hidden_quant: str = "none",
) -> RelayBridgeResult:
    """★ 中间段服务：HIDDEN → `runner.request_hidden()` → HIDDEN（末位 argmax 不传）。

    与 `serve_relay_connection`（末段，回 TOKEN）配对。`runner` 必须提供
    `request_hidden(hidden_bytes, n_tokens=...) -> bytes` 与 `close()`；
    主仓的 `llama_keep_head.KeepHeadUpstream`（经 `forward_hidden_to_hidden`）与
    Android 的 `nativeLayerForwardHiddenKeepHead` 语义一致。

    ★ A5 两个方向：
    * **上行**（客户端 → 本服务）：档位写在请求帧的 `flags` 低 3 位（`frame.quant`）——
      本函数按它解回 f32 再喂 runner；
    * **下行**（本服务 → 客户端）：档位由 `hidden_quant` 单方面决定（客户端只跟随解压）。
      两端可以选不同档位（例如上行 `f16`、下行 `int8_block128`）。
    """

    width = int(n_embd)
    limit = int(max_tokens)
    max_payload = (expected_hidden_bytes(limit, width)
                   + RELAY_SEQ_META_LIMIT + _SEQ_HEADER.size)
    sequence = 0
    frames = 0
    tokens = 0
    payload_bytes = 0
    hidden_mode: str | None = None
    try:
        while True:
            frame = recv_frame(sock, max_payload_bytes=max_payload)
            if frame.sequence != sequence:
                raise RelayProtocolError("request_sequence_mismatch")
            if frame.kind == RelayFrameKind.CLOSE:
                if frame.n_tokens != 0 or frame.payload:
                    raise RelayProtocolError("invalid_close_frame")
                # ★ 同 `serve_relay_connection`：只 reset，不 close（见那里的详细说明）。
                logger.info("relay middle 段收到 CLOSE 帧 ⇒ runner.reset()（会话结束）")
                try:
                    runner.reset()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Relay middle runner reset failed: code=%s",
                                     RELAY_RUNNER_ERROR)
                    raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
                send_frame(
                    sock,
                    RelayFrame(RelayFrameKind.TOKEN, sequence, n_tokens=1,
                               payload=_TOKEN.pack(-1)),
                )
                return RelayBridgeResult(frames, tokens, payload_bytes, True)
            if frame.kind in {RelayFrameKind.HIDDEN, RelayFrameKind.HIDDEN_SEQ}:
                frame_mode = (
                    "seq" if frame.kind == RelayFrameKind.HIDDEN_SEQ else "plain"
                )
                if hidden_mode is not None and frame_mode != hidden_mode:
                    raise RelayProtocolError("hidden_sequence_mode_changed")
                hidden_mode = frame_mode
            if frame.kind == RelayFrameKind.HIDDEN_SEQ:
                # ★ P3 多序列：payload 自带 seq/pos；远端 runner 必须支持显式绑定，
                # 否则 fail-loud（绝不退回"远端按隐式位置猜"——那会静默算错）。
                # ★ A5：按帧里的档位解回 f32（`HIDDEN_SEQ` 的 hidden 段可能被压缩）。
                hidden, n_tokens, meta = decode_hidden_seq(frame.payload, n_embd=width,
                                                           quant=frame.quant)
                if n_tokens != frame.n_tokens:
                    raise RelayProtocolError("hidden_seq_token_count_mismatch")
                if n_tokens < 1 or n_tokens > limit:
                    raise RelayProtocolError("token_count_exceeds_limit")
                if not hasattr(runner, "request_hidden_seq"):
                    raise RelayProtocolError("hidden_seq_unsupported")
                try:
                    produced = bytes(runner.request_hidden_seq(hidden, n_tokens=n_tokens,
                                                               meta=meta))
                except RelayProtocolError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Relay middle runner (seq) failed: code=%s",
                                     RELAY_RUNNER_ERROR)
                    raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
                if len(produced) != expected_hidden_bytes(n_tokens, width):
                    raise RelayProtocolError("hidden_payload_size_mismatch")
                send_frame(
                    sock,
                    RelayFrame(RelayFrameKind.HIDDEN, sequence, n_tokens=n_tokens,
                               payload=quantize_upload(produced, n_tokens, width, hidden_quant),
                               quant=hidden_quant or "none"),
                )
                frames += 1
                tokens += n_tokens
                payload_bytes += len(frame.payload)
                sequence += 1
                continue
            if frame.kind == RelayFrameKind.TOKENS:
                # ★ P4.5 上游段：payload 是 token id 列表；远端 runner 必须支持该角色，
                # 否则 fail-loud（绝不猜测"这是 hidden 还是 token"）。
                incoming = decode_tokens(frame.payload, limit=limit)
                if len(incoming) != frame.n_tokens:
                    raise RelayProtocolError("token_frame_count_mismatch")
                if not hasattr(runner, "request_hidden_from_tokens"):
                    raise RelayProtocolError("token_frame_unsupported")
                try:
                    produced = bytes(runner.request_hidden_from_tokens(incoming))
                except RelayProtocolError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Relay upstream runner failed: code=%s",
                                     RELAY_RUNNER_ERROR)
                    raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
                if len(produced) != expected_hidden_bytes(len(incoming), width):
                    raise RelayProtocolError("hidden_payload_size_mismatch")
                send_frame(
                    sock,
                    RelayFrame(RelayFrameKind.HIDDEN, sequence, n_tokens=len(incoming),
                               payload=quantize_upload(produced, len(incoming), width,
                                                       hidden_quant),
                               quant=hidden_quant or "none"),
                )
                frames += 1
                tokens += len(incoming)
                payload_bytes += len(frame.payload)
                sequence += 1
                continue
            if frame.kind != RelayFrameKind.HIDDEN:
                raise RelayProtocolError("hidden_frame_required")
            if frame.n_tokens < 1 or frame.n_tokens > limit:
                raise RelayProtocolError("token_count_exceeds_limit")
            if len(frame.payload) != expected_hidden_bytes(frame.n_tokens, width, frame.quant):
                raise RelayProtocolError("hidden_payload_size_mismatch")

            try:
                # ★ A5：按档位解回 f32 再喂 runner（压缩对段实现透明）。
                produced = bytes(runner.request_hidden(frame_hidden_bytes(frame, n_embd=width),
                                                       n_tokens=frame.n_tokens))
            except RelayProtocolError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Relay middle runner failed: code=%s", RELAY_RUNNER_ERROR)
                raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
            if len(produced) != expected_hidden_bytes(frame.n_tokens, width):
                raise RelayProtocolError("hidden_payload_size_mismatch")
            send_frame(
                sock,
                RelayFrame(RelayFrameKind.HIDDEN, sequence, n_tokens=frame.n_tokens,
                           payload=quantize_upload(produced, frame.n_tokens, width,
                                                   hidden_quant),
                           quant=hidden_quant or "none"),
            )
            frames += 1
            tokens += frame.n_tokens
            payload_bytes += len(frame.payload)
            sequence += 1
    except RelayProtocolError as exc:
        code = _safe_error_code(exc, RELAY_PROTOCOL_ERROR)
        logger.warning("Relay middle protocol failure: code=%s detail=%s", code, str(exc))
        try:
            _send_error(sock, sequence, code)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, code)
    except OSError as exc:
        logger.warning("Relay middle transport failure: code=%s detail=%s",
                       RELAY_TRANSPORT_ERROR, exc)
        try:
            _send_error(sock, sequence, RELAY_TRANSPORT_ERROR)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, RELAY_TRANSPORT_ERROR)
    except Exception as exc:  # noqa: BLE001
        # Never put exception class names or messages on the Relay wire.
        logger.exception("Relay middle internal failure: code=%s", RELAY_INTERNAL_ERROR)
        try:
            _send_error(sock, sequence, RELAY_INTERNAL_ERROR)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, RELAY_INTERNAL_ERROR)
