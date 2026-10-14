"""Wire and loopback bridge tests for CORE-RELAY-XFRAME-01."""

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from src.relay_transport import (
    RELAY_QUANT_CODES,
    RELAY_TRANSPORT_ERROR,
    RELAY_WIRE_MAGIC,
    RELAY_WIRE_VERSION,
    REQUEST_DEADLINE_EXCEEDED,
    RelayFrame,
    RelayFrameKind,
    RelayProtocolError,
    RelayTcpClient,
    expected_hidden_bytes,
    frame_hidden_bytes,
    open_loopback_listener,
    quantize_upload,
    recv_frame,
    send_frame,
    serve_relay_connection,
)


class _FakeRunner:
    def __init__(self) -> None:
        self.requests: list[tuple[int, bytes]] = []
        self.closed = False
        self.resets = 0

    def request_token(self, hidden: bytes, *, n_tokens: int) -> int:
        self.requests.append((n_tokens, hidden))
        return 1000 + n_tokens

    def request_token_seq(self, hidden: bytes, *, n_tokens: int,
                          meta: dict[str, object]) -> int:
        self.requests.append((n_tokens, hidden))
        return 1000 + n_tokens

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


def test_frame_round_trip_over_partial_socket_reads():
    left, right = socket.socketpair()
    try:
        frame = RelayFrame(RelayFrameKind.HIDDEN, 7, n_tokens=2, payload=b"abcdefgh")
        send_frame(left, frame)
        received = recv_frame(right)
    finally:
        left.close()
        right.close()

    assert received == frame


@pytest.mark.parametrize(
    ("request_deadline", "expected_error"),
    [
        (True, REQUEST_DEADLINE_EXCEEDED),
        (False, RELAY_TRANSPORT_ERROR),
    ],
)
def test_client_uses_one_absolute_deadline_across_partial_reads(
    request_deadline: bool, expected_error: str
):
    listener = open_loopback_listener("127.0.0.1", 0)
    port = int(listener.getsockname()[1])
    server_errors: list[BaseException] = []

    def serve_slow_response() -> None:
        connection, _ = listener.accept()
        try:
            recv_frame(connection)
            response = struct.pack(
                "!4sBBHIIQ",
                RELAY_WIRE_MAGIC,
                RELAY_WIRE_VERSION,
                int(RelayFrameKind.TOKEN),
                0,
                0,
                1,
                4,
            ) + struct.pack("<i", 123)
            for value in response:
                connection.sendall(bytes([value]))
                time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            server_errors.append(exc)
        finally:
            connection.close()

    thread = threading.Thread(target=serve_slow_response, daemon=True)
    thread.start()
    deadline = time.monotonic() + 0.4
    client = RelayTcpClient(
        "127.0.0.1",
        port,
        n_embd=2,
        timeout=1.0 if request_deadline else 0.4,
        deadline_monotonic=deadline if request_deadline else None,
    )
    started = time.monotonic()
    try:
        with pytest.raises(RelayProtocolError, match=expected_error):
            client.request_token(
                b"\x00" * expected_hidden_bytes(1, 2),
                n_tokens=1,
            )
    finally:
        listener.close()
        thread.join(timeout=2)

    assert time.monotonic() - started < 0.8
    assert client._closed is True
    assert server_errors


def test_protocol_failure_marks_transport_session_non_reusable():
    listener = open_loopback_listener("127.0.0.1", 0)
    port = int(listener.getsockname()[1])

    def serve_bad_sequence() -> None:
        connection, _ = listener.accept()
        try:
            recv_frame(connection)
            send_frame(
                connection,
                RelayFrame(
                    RelayFrameKind.TOKEN,
                    99,
                    n_tokens=1,
                    payload=struct.pack("<i", 7),
                ),
            )
        finally:
            connection.close()

    thread = threading.Thread(target=serve_bad_sequence, daemon=True)
    thread.start()
    client = RelayTcpClient("127.0.0.1", port, n_embd=2, timeout=1.0)
    try:
        with pytest.raises(RelayProtocolError, match="response_sequence_mismatch"):
            client.request_token(
                b"\x00" * expected_hidden_bytes(1, 2), n_tokens=1
            )
    finally:
        listener.close()
        thread.join(timeout=2)

    assert client._closed is True
    with pytest.raises(RelayProtocolError, match="client_closed"):
        client.request_token(
            b"\x00" * expected_hidden_bytes(1, 2), n_tokens=1
        )


def test_close_grace_is_an_absolute_short_budget():
    listener = open_loopback_listener("127.0.0.1", 0)
    port = int(listener.getsockname()[1])

    def ignore_close_ack() -> None:
        connection, _ = listener.accept()
        try:
            recv_frame(connection)
            time.sleep(0.5)
        finally:
            connection.close()

    thread = threading.Thread(target=ignore_close_ack, daemon=True)
    thread.start()
    client = RelayTcpClient("127.0.0.1", port, n_embd=2, timeout=5.0)
    started = time.monotonic()
    try:
        with pytest.raises(RelayProtocolError, match=RELAY_TRANSPORT_ERROR):
            client.close(grace_timeout=0.1)
        elapsed = time.monotonic() - started
    finally:
        listener.close()
        thread.join(timeout=2)

    assert elapsed < 0.4
    assert client._closed is True


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (struct.pack("!4sBBHIIQ", b"BAD!", 1, 1, 0, 0, 1, 0), "invalid_magic"),
        (struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, 2, 1, 0, 0, 1, 0), "unsupported_version"),
        (struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, 1, 99, 0, 0, 1, 0), "unknown_frame_kind"),
        (struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, 1, 1, 0, 0, 1, 99), "payload_too_large"),
    ],
)
def test_invalid_headers_fail_before_payload_allocation(header: bytes, reason: str):
    left, right = socket.socketpair()
    try:
        left.sendall(header)
        with pytest.raises(RelayProtocolError, match=reason):
            recv_frame(right, max_payload_bytes=16)
    finally:
        left.close()
        right.close()


def test_truncated_payload_is_rejected():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, 1, 1, 0, 0, 1, 8) + b"four")
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(RelayProtocolError, match="connection_closed_mid_frame"):
            recv_frame(right)
    finally:
        left.close()
        right.close()


def test_loopback_client_bridge_preserves_order_and_resets_runner():
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    runner = _FakeRunner()
    results = []

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            results.append(serve_relay_connection(connection, runner, n_embd=2, max_tokens=4))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with RelayTcpClient("127.0.0.1", port, n_embd=2, max_tokens=4) as client:
            assert client.request_token(b"\x00" * expected_hidden_bytes(1, 2), n_tokens=1) == 1001
            assert client.request_token(b"\x01" * expected_hidden_bytes(3, 2), n_tokens=3) == 1003
    finally:
        thread.join(timeout=2)
        listener.close()

    assert not thread.is_alive()
    assert [request[0] for request in runner.requests] == [1, 3]
    # ★ `CLOSE` 只 reset、**不 close**：引擎属于服务进程，要跨连接复用（见下面的双会话回归测试）
    assert runner.closed is False
    assert runner.resets >= 1
    assert results[0].closed_cleanly is True
    assert results[0].frames == 2
    assert results[0].tokens == 4


def test_tail_session_rejects_switch_between_explicit_and_implicit_positions():
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    runner = _FakeRunner()
    results = []

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            results.append(serve_relay_connection(connection, runner, n_embd=2))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    payload = b"\x00" * expected_hidden_bytes(1, 2)
    client = RelayTcpClient("127.0.0.1", port, n_embd=2)
    try:
        assert client.request_token(
            payload, n_tokens=1,
            seq_meta={"seq_ids": [0], "positions": [7]},
        ) == 1001
        with pytest.raises(RelayProtocolError, match="hidden_sequence_mode_changed"):
            client.request_token(payload, n_tokens=1)
        client._closed = True
        client._sock.close()
        thread.join(timeout=3)
    finally:
        if not client._closed:
            client._closed = True
            client._sock.close()
        listener.close()

    assert not thread.is_alive()
    assert len(runner.requests) == 1
    assert results[0].error == "hidden_sequence_mode_changed"


def test_close_frame_does_not_destroy_runner_across_sessions():
    """★ 回归守卫：`CLOSE` 只 reset、**不 close** —— 否则服务端只能成功服务**一次**。

    真实故障（2026-09-22 实测）：探针每次实验结束都会发 `CLOSE`；旧实现里服务端调
    `runner.close()` 销毁底层 handle，于是**该服务的第二个连接必然失败** ——
    帧完全正确（32 tokens / 114688 字节）却报 `rc=-5 参数非法`，现象与"模型算错"难以区分，
    只能靠重启服务恢复。本测试用**两个连续会话**把该行为钉死。
    """
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    runner = _FakeRunner()
    sessions = []

    def serve() -> None:
        for _ in range(2):  # 连续两次会话（模拟"先跑 A 组，紧接着跑 B 组"）
            connection, _ = listener.accept()
            with connection:
                sessions.append(
                    serve_relay_connection(connection, runner, n_embd=2, max_tokens=4))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        for _ in range(2):
            with RelayTcpClient("127.0.0.1", port, n_embd=2, max_tokens=4) as client:
                assert client.request_token(
                    b"\x00" * expected_hidden_bytes(1, 2), n_tokens=1) == 1001
    finally:
        thread.join(timeout=3)
        listener.close()

    assert not thread.is_alive()
    assert runner.closed is False        # ★ 引擎必须活着（旧实现这里是 True）
    assert runner.resets >= 2            # ★ 两次会话各 reset 一次
    assert [session.closed_cleanly for session in sessions] == [True, True]
    assert [session.frames for session in sessions] == [1, 1]


def test_bridge_rejects_shape_mismatch_and_returns_error_frame():
    left, right = socket.socketpair()
    runner = _FakeRunner()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(serve_relay_connection(right, runner, n_embd=2)),
        daemon=True,
    )
    thread.start()
    try:
        send_frame(left, RelayFrame(RelayFrameKind.HIDDEN, 0, n_tokens=2, payload=b"too-short"))
        response = recv_frame(left)
    finally:
        thread.join(timeout=2)
        left.close()
        right.close()

    assert response.kind == RelayFrameKind.ERROR
    assert response.payload == b"hidden_payload_size_mismatch"
    assert result[0].closed_cleanly is False
    assert runner.requests == []


def test_non_loopback_endpoints_are_rejected_without_opening_a_socket():
    with pytest.raises(RelayProtocolError, match="non_loopback_bind_rejected"):
        open_loopback_listener("100.100.52.106", 0)
    with pytest.raises(RelayProtocolError, match="non_loopback_endpoint_rejected"):
        RelayTcpClient("100.100.52.106", 12345, n_embd=2)


def test_client_rejects_local_shape_and_limit_errors_before_sending():
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    accepted = []
    thread = threading.Thread(target=lambda: accepted.append(listener.accept()), daemon=True)
    thread.start()
    client = RelayTcpClient("127.0.0.1", port, n_embd=2, max_tokens=2)
    try:
        with pytest.raises(RelayProtocolError, match="hidden_payload_size_mismatch"):
            client.request_token(b"bad", n_tokens=1)
        with pytest.raises(RelayProtocolError, match="token_count_exceeds_limit"):
            client.request_token(b"", n_tokens=3)
    finally:
        client._sock.close()
        listener.close()
        thread.join(timeout=2)
        for connection, _ in accepted:
            connection.close()


def test_tokens_frame_roundtrip_helper():
    """★ P4.5 上游段请求：`TOKENS` 帧的 payload 编解码（紧凑 i32 数组）。"""
    from src.relay_transport import decode_tokens, encode_tokens

    payload = encode_tokens([1, 42, 151936])
    assert len(payload) == 12
    assert decode_tokens(payload, limit=8) == [1, 42, 151936]


def test_tokens_frame_rejects_bad_payload_shape():
    """长度不是 4 的倍数 / 空 payload / 超限都必须 fail-loud（不猜语义）。"""
    from src.relay_transport import decode_tokens, encode_tokens

    with pytest.raises(RelayProtocolError, match="invalid_token_response"):
        decode_tokens(b"", limit=8)
    with pytest.raises(RelayProtocolError, match="invalid_token_response"):
        decode_tokens(b"\x01\x02\x03", limit=8)
    with pytest.raises(RelayProtocolError, match="token_count_exceeds_limit"):
        decode_tokens(encode_tokens([1] * 9), limit=8)
    with pytest.raises(RelayProtocolError, match="token_count_exceeds_limit"):
        encode_tokens([])


def test_tokens_frame_kind_is_distinct():
    """新帧类型不得与既有 HIDDEN / TOKEN / HIDDEN_SEQ 碰撞（协议兼容的硬约束）。"""
    kinds = {RelayFrameKind.HIDDEN, RelayFrameKind.TOKEN, RelayFrameKind.CLOSE,
             RelayFrameKind.ERROR, RelayFrameKind.HIDDEN_SEQ, RelayFrameKind.TOKENS}
    assert len(kinds) == 6


# ---------------------------------------------------------------- ★ A5：hidden 压缩档上 wire


def test_quantized_hidden_round_trips_through_flags():
    """★ A5：档位编码进 `flags` 低 3 位；发送端压、接收端解，解出的 f32 长度不变。"""
    import numpy as np

    tokens, width = 2, 256
    values = (np.arange(tokens * width, dtype=np.float32) / 17.0)
    raw = np.ascontiguousarray(values, dtype="<f4").tobytes()

    left, right = socket.socketpair()
    try:
        for quant in ("none", "f16", "int8_block128", "int4_block128"):
            payload = quantize_upload(raw, tokens, width, quant)
            assert len(payload) == expected_hidden_bytes(tokens, width, quant)
            send_frame(left, RelayFrame(RelayFrameKind.HIDDEN, 0, n_tokens=tokens,
                                        payload=payload, quant=quant))
            frame = recv_frame(right, max_payload_bytes=expected_hidden_bytes(tokens, width))
            assert frame.quant == quant
            assert frame.payload == payload
            decoded = frame_hidden_bytes(frame, n_embd=width)
            assert len(decoded) == tokens * width * 4
            if quant == "none":
                assert decoded == raw
        # `none` 档 = flags 0 ⇒ 与旧对端的帧逐字节兼容
        assert RELAY_QUANT_CODES["none"] == 0
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    ("flags", "expected_code"),
    [(0b1000, "unsupported_flags"), (0b0111, "unsupported_hidden_quant")],
)
def test_reserved_flags_and_unknown_quant_codes_fail_closed(flags: int,
                                                            expected_code: str) -> None:
    """保留位（高位）与未知档位码都必须 fail-closed —— 旧对端行为不变、新对端不能瞎猜。"""
    left, right = socket.socketpair()
    try:
        header = struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, RELAY_WIRE_VERSION,
                             int(RelayFrameKind.HIDDEN), flags, 0, 1, 0)
        left.sendall(header)
        with pytest.raises(RelayProtocolError, match=expected_code):
            recv_frame(right)
    finally:
        left.close()
        right.close()


def test_quantized_frame_size_mismatch_is_rejected():
    """声明 f16 档却给 f32 长度的 payload ⇒ 必须拒（不能把未压缩数据当压缩帧解）。"""
    tokens, width = 1, 128
    left, right = socket.socketpair()
    try:
        raw = b"\x00" * (tokens * width * 4)
        send_frame(left, RelayFrame(RelayFrameKind.HIDDEN, 0, n_tokens=tokens,
                                    payload=raw, quant="f16"))
        frame = recv_frame(right, max_payload_bytes=expected_hidden_bytes(tokens, width))
        assert frame.quant == "f16"
        with pytest.raises(RelayProtocolError, match="hidden_payload_size_mismatch"):
            frame_hidden_bytes(frame, n_embd=width)
    finally:
        left.close()
        right.close()
