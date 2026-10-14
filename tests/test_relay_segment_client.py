"""tests/test_relay_segment_client.py — ★ A1 / X 档：Relay 段执行器的往返与「失败必须具名」。

不需要模型：用 `serve_relay_connection` / `serve_relay_middle_connection` + 假 runner，
在真正的 loopback 线协议上跑（范式同 `tests/test_relay_transport.py`、
`tests/test_relay_middle_transport.py`）。

⚠️ 本文件含**「该红必须红」**的元用例（`test_guard_*`）：它们断言的不是某个业务分支的开心路径，
而是「**守卫真的在用**」—— 例如把错误码白名单清空后，原本稳定的码**必然**变值；把异常文本
直接当错误码时，**必然**会泄漏。任何人拆掉这些守卫，本文件立刻变红。
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import relay_segment_client as segment_module  # noqa: E402
from relay_segment_client import (  # noqa: E402
    SEGMENT_ROLES,
    SUPPORTED_QUANT_MODES,
    RelaySegmentClient,
    RelaySegmentError,
    stable_error_code,
)
from relay_transport import (  # noqa: E402
    RELAY_PROTOCOL_ERROR,
    RELAY_TRANSPORT_ERROR,
    REQUEST_DEADLINE_EXCEEDED,
    RelayProtocolError,
    expected_hidden_bytes,
    open_loopback_listener,
    serve_relay_connection,
    serve_relay_middle_connection,
)

N_EMBD = 4
N_TOKENS = 2
MAX_TOKENS = 8


def _hidden(seed: int = 0) -> bytes:
    """`n_tokens × n_embd` 个 f32 的原始字节（X 档线上永远是 f32）。"""
    return bytes((seed + i) % 256 for i in range(expected_hidden_bytes(N_TOKENS, N_EMBD)))


# ---- 假 runner（服务端侧）-------------------------------------------------


class _PlusOneMiddleRunner:
    """逐字节 +1 —— 既能验证往返正确，也能暴露"原样返回"这类假通过。"""

    def __init__(self) -> None:
        self.seen: list[tuple[int, bytes]] = []
        self.meta: dict[str, object] | None = None
        self.resets = 0

    def request_hidden(self, hidden: bytes, *, n_tokens: int) -> bytes:
        self.seen.append((n_tokens, hidden))
        return bytes((value + 1) % 256 for value in hidden)

    def request_hidden_seq(self, hidden: bytes, *, n_tokens: int,
                           meta: dict[str, object]) -> bytes:
        self.seen.append((n_tokens, hidden))
        self.meta = meta
        return bytes((value + 1) % 256 for value in hidden)

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        pass


class _HeadRunner:
    """上游段：吃 token ids 吐 hidden。"""

    def __init__(self) -> None:
        self.seen: list[list[int]] = []
        self.resets = 0

    def request_hidden_from_tokens(self, tokens) -> bytes:
        self.seen.append([int(t) for t in tokens])
        return bytes((index % 251) for index in range(expected_hidden_bytes(N_TOKENS, N_EMBD)))

    def reset(self) -> None:
        self.resets += 1


class _TailRunner:
    """末段：吃 hidden 吐 token。"""

    def __init__(self, token: int = 4242) -> None:
        self.token = int(token)
        self.seen: list[tuple[int, bytes]] = []
        self.seq_meta: dict[str, object] | None = None
        self.resets = 0

    def request_token(self, hidden: bytes, *, n_tokens: int) -> int:
        self.seen.append((n_tokens, hidden))
        return self.token

    def request_token_seq(self, hidden: bytes, *, n_tokens: int,
                          meta: dict[str, object]) -> int:
        self.seen.append((n_tokens, hidden))
        self.seq_meta = meta
        return self.token

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        pass


class _ExplodingRunner:
    """抛一个**带敏感文本**的非协议异常：必须被收敛成稳定码，文本绝不上 wire。"""

    leak = "boom: root cause at /etc/shadow line 42"

    def __init__(self) -> None:
        self.resets = 0

    def request_hidden(self, hidden: bytes, *, n_tokens: int) -> bytes:
        raise RuntimeError(self.leak)

    def reset(self) -> None:
        self.resets += 1


# ---- loopback 服务端 ------------------------------------------------------


def _listen() -> tuple[socket.socket, int]:
    listener = open_loopback_listener("127.0.0.1", 0)
    return listener, int(listener.getsockname()[1])


def _serve_once(listener: socket.socket, runner, *, middle: bool):
    box: dict[str, object] = {}

    def _run() -> None:
        sock, _ = listener.accept()
        try:
            if middle:
                box["bridge"] = serve_relay_middle_connection(
                    sock, runner, n_embd=N_EMBD, max_tokens=MAX_TOKENS)
            else:
                box["bridge"] = serve_relay_connection(
                    sock, runner, n_embd=N_EMBD, max_tokens=MAX_TOKENS)
        finally:
            sock.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, box


# ---- 开心路径 -------------------------------------------------------------


def test_middle_round_trip_is_not_identity():
    runner = _PlusOneMiddleRunner()
    listener, port = _listen()
    try:
        thread, _ = _serve_once(listener, runner, middle=True)
        client = RelaySegmentClient("127.0.0.1", port, n_embd=N_EMBD, role="middle")
        try:
            outcome = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)
        finally:
            client.close()
        thread.join(timeout=5)
    finally:
        listener.close()

    assert outcome.ok, outcome.error
    expected = bytes((value + 1) % 256 for value in _hidden())
    assert outcome.hidden == expected
    assert outcome.hidden != _hidden()          # 排除"原样返回"的假通过
    assert outcome.frames == 1
    assert outcome.payload_bytes == len(expected)
    assert outcome.elapsed_ms > 0
    assert runner.seen and runner.seen[0][0] == N_TOKENS


def test_tail_round_trip_returns_token():
    runner = _TailRunner(token=31337)
    listener, port = _listen()
    try:
        thread, _ = _serve_once(listener, runner, middle=False)
        client = RelaySegmentClient("127.0.0.1", port, n_embd=N_EMBD, role="tail")
        try:
            outcome = client.forward_hidden_to_token(_hidden(), n_tokens=N_TOKENS)
        finally:
            client.close()
        thread.join(timeout=5)
    finally:
        listener.close()

    assert outcome.ok, outcome.error
    assert outcome.token == 31337
    assert outcome.hidden == b""


def test_tail_seq_meta_round_trip_preserves_decode_positions():
    runner = _TailRunner(token=31338)
    listener, port = _listen()
    try:
        thread, _ = _serve_once(listener, runner, middle=False)
        client = RelaySegmentClient("127.0.0.1", port, n_embd=N_EMBD, role="tail")
        try:
            outcome = client.forward_hidden_to_token(
                _hidden(), n_tokens=N_TOKENS, seq_meta={
                    "seq_ids": [0, 0], "positions": [17, 18],
                })
        finally:
            client.close()
        thread.join(timeout=5)
    finally:
        listener.close()

    assert outcome.ok, outcome.error
    assert outcome.token == 31338
    assert runner.seq_meta == {"seq_ids": [0, 0], "positions": [17, 18]}


def test_head_round_trip_returns_hidden():
    runner = _HeadRunner()
    listener, port = _listen()
    try:
        thread, _ = _serve_once(listener, runner, middle=True)
        client = RelaySegmentClient("127.0.0.1", port, n_embd=N_EMBD, role="head")
        try:
            outcome = client.forward_tokens([11, 22])
        finally:
            client.close()
        thread.join(timeout=5)
    finally:
        listener.close()

    assert outcome.ok, outcome.error
    assert len(outcome.hidden) == expected_hidden_bytes(N_TOKENS, N_EMBD)
    assert runner.seen == [[11, 22]]


def test_seq_meta_goes_through_hidden_seq_frame():
    runner = _PlusOneMiddleRunner()
    listener, port = _listen()
    try:
        thread, _ = _serve_once(listener, runner, middle=True)
        client = RelaySegmentClient("127.0.0.1", port, n_embd=N_EMBD, role="middle")
        meta = {"n_seq_id": [1, 1], "seq_ids": [0, 1], "positions": [0, 0]}
        try:
            outcome = client.forward_hidden(_hidden(), n_tokens=N_TOKENS, seq_meta=meta)
        finally:
            client.close()
        thread.join(timeout=5)
    finally:
        listener.close()

    assert outcome.ok, outcome.error
    assert runner.meta is not None
    assert list(runner.meta["seq_ids"]) == [0, 1]


# ---- 失败必须具名（绝不静默）----------------------------------------------


def test_unreachable_segment_is_named_not_silent():
    """段不可达 ⇒ **具名传输错误**，而不是"静默返回空 hidden"。"""
    # 先占一个端口再立刻释放，得到一个"几乎肯定没人监听"的端口号。
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = int(probe.getsockname()[1])
    probe.close()

    client = RelaySegmentClient("127.0.0.1", dead_port, n_embd=N_EMBD, role="middle",
                                timeout=1.0)
    outcome = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)

    assert outcome.ok is False
    assert outcome.error == RELAY_TRANSPORT_ERROR
    assert outcome.hidden == b""
    assert client.frames == 0


def test_expired_absolute_deadline_prevents_connect(monkeypatch):
    created: list[object] = []

    class _UnexpectedTransport:
        def __init__(self, *_args, **_kwargs):
            created.append(self)

    monkeypatch.setattr(segment_module, "RelayTcpClient", _UnexpectedTransport)
    client = RelaySegmentClient(
        "127.0.0.1",
        50183,
        n_embd=N_EMBD,
        role="middle",
        deadline_monotonic=time.monotonic() - 1.0,
    )

    outcome = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)

    assert outcome.ok is False
    assert outcome.error == REQUEST_DEADLINE_EXCEEDED
    assert created == []


def test_failed_round_trip_discards_session_before_next_call(monkeypatch):
    class _BrokenTransport:
        instances: list["_BrokenTransport"] = []

        def __init__(self, *_args, **_kwargs):
            self.aborted = False
            type(self).instances.append(self)

        def tighten_deadline(self, _deadline):
            pass

        def request_hidden(self, *_args, **_kwargs):
            raise RelayProtocolError("connection_closed_mid_frame")

        def abort(self):
            self.aborted = True

    monkeypatch.setattr(segment_module, "RelayTcpClient", _BrokenTransport)
    client = RelaySegmentClient(
        "127.0.0.1", 50183, n_embd=N_EMBD, role="middle"
    )

    first = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)
    second = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)

    assert first.error == "connection_closed_mid_frame"
    assert second.error == "connection_closed_mid_frame"
    assert len(_BrokenTransport.instances) == 2
    assert all(instance.aborted for instance in _BrokenTransport.instances)
    assert client._client is None


def test_runner_exception_is_collapsed_without_leaking_text():
    """runner 抛异常 ⇒ 稳定码 `runner_failed`，且**异常文本绝不出现在错误码里**。"""
    runner = _ExplodingRunner()
    listener, port = _listen()
    try:
        thread, _ = _serve_once(listener, runner, middle=True)
        client = RelaySegmentClient("127.0.0.1", port, n_embd=N_EMBD, role="middle")
        try:
            outcome = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)
        finally:
            client.close()
        thread.join(timeout=5)
    finally:
        listener.close()

    assert outcome.ok is False
    assert outcome.error == "runner_failed"
    assert "boom" not in outcome.error
    assert "/etc/shadow" not in outcome.error
    # ★ 失败路径**不**触发 `runner.reset()`：服务端发完 ERROR 帧即结束本会话，不会再读 CLOSE 帧
    #   （`reset()` 只在**干净** CLOSE 时调用，见 `relay_transport` 的 CLOSE 分支）。
    #   这很关键：runner 是**服务进程**的长期资产，任何失败都不得销毁它（生存期守卫）。
    assert runner.resets == 0


def test_clean_close_resets_but_keeps_engine():
    """干净结束 ⇒ 服务端只 `reset()`（清 KV/位置），**绝不**销毁引擎（生存期守卫的对照）。"""
    runner = _PlusOneMiddleRunner()
    listener, port = _listen()
    try:
        thread, _ = _serve_once(listener, runner, middle=True)
        client = RelaySegmentClient("127.0.0.1", port, n_embd=N_EMBD, role="middle")
        try:
            outcome = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)
            assert outcome.ok, outcome.error
        finally:
            client.close()          # 发 CLOSE ⇒ 服务端应 reset()
        thread.join(timeout=5)
    finally:
        listener.close()

    assert runner.resets == 1


def test_payload_size_mismatch_is_named():
    listener, port = _listen()
    try:
        thread, _ = _serve_once(listener, _PlusOneMiddleRunner(), middle=True)
        client = RelaySegmentClient("127.0.0.1", port, n_embd=N_EMBD, role="middle")
        try:
            outcome = client.forward_hidden(b"too-short", n_tokens=N_TOKENS)
        finally:
            client.close()
        thread.join(timeout=5)
    finally:
        listener.close()

    assert outcome.ok is False
    assert outcome.error == "hidden_payload_size_mismatch"


def test_non_loopback_endpoint_is_rejected():
    """Relay 只能走 loopback / 本地 SSH 隧道端点（跨机由隧道承担）。"""
    client = RelaySegmentClient("10.1.2.3", 50183, n_embd=N_EMBD, role="middle", timeout=0.5)
    outcome = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)

    assert outcome.ok is False
    assert outcome.error == "non_loopback_endpoint_rejected"


# ---- X 档的范围守卫 -------------------------------------------------------


@pytest.mark.parametrize("quant", ["f16", "int8_block128", "int4_block128", "bogus"])
def test_quant_modes_are_out_of_scope_for_x(quant: str):
    """X 档不含量化档 ⇒ **显式拒绝**（静默按 none 走会掩盖配置错误）。"""
    with pytest.raises(ValueError, match="X 档只允许"):
        RelaySegmentClient("127.0.0.1", 50183, n_embd=N_EMBD, role="middle", quant=quant)


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="未知段角色"):
        RelaySegmentClient("127.0.0.1", 50183, n_embd=N_EMBD, role="router")


def test_declared_scope_constants():
    assert SEGMENT_ROLES == ("head", "middle", "tail")
    assert SUPPORTED_QUANT_MODES == ("none",)


def test_metrics_contract_fields():
    """X 档要求 metrics 恰好含这 5 个字段（调度层直接并进既有 metrics）。"""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = int(probe.getsockname()[1])
    probe.close()

    client = RelaySegmentClient("127.0.0.1", dead_port, n_embd=N_EMBD, role="middle", timeout=0.5)
    outcome = client.forward_hidden(_hidden(), n_tokens=N_TOKENS)
    metrics = outcome.to_metrics()

    assert set(metrics) == {"relay_segment", "relay_frames", "relay_tokens",
                            "relay_payload_bytes", "relay_error"}
    assert metrics["relay_segment"] == f"middle@127.0.0.1:{dead_port}"
    assert metrics["relay_error"] == RELAY_TRANSPORT_ERROR   # 失败也**如实**进 metrics


# ---- 「该红必须红」的守卫元用例 -------------------------------------------


def test_guard_whitelist_is_actually_used(monkeypatch: pytest.MonkeyPatch):
    """★「该红必须红」：证明**白名单真的在生效**。

    正常时 `runner_failed` 原样通过；把白名单清空后，它**必然**退化成
    `relay_internal_error`。任何人把 `stable_error_code` 改成"不查白名单、直接返回原字符串"，
    这条立刻红 —— 而那种改动会让任意文本直接上 wire。
    """
    assert stable_error_code("runner_failed") == "runner_failed"
    monkeypatch.setattr(segment_module, "_RELAY_ERROR_CODES", frozenset())
    assert stable_error_code("runner_failed") == "relay_internal_error"


def test_guard_never_leaks_arbitrary_exception_text():
    """★「该红必须红」：任意异常文本**绝不**成为错误码。"""
    leak = "boom: root cause at /etc/shadow line 42"
    code = stable_error_code(RuntimeError(leak))

    assert leak not in code
    assert "/etc/shadow" not in code
    assert code == RELAY_PROTOCOL_ERROR


def test_guard_fallback_itself_must_be_whitelisted():
    """★「该红必须红」：`fallback` 也不能随手传 —— 非白名单的 fallback 会被兜到 internal。"""
    assert stable_error_code(RuntimeError("x"), fallback="not_a_real_code") == "relay_internal_error"
    assert stable_error_code(RuntimeError("x"), fallback="runner_failed") == "runner_failed"


def test_guard_segment_error_never_carries_unknown_code():
    """★「该红必须红」：`:class:`RelaySegmentError`` 的 `code` 永远落在白名单内。"""
    error = RelaySegmentError("totally_made_up_code", role="middle", endpoint="127.0.0.1:50183")

    assert error.code == RELAY_PROTOCOL_ERROR
    assert "totally_made_up_code" not in str(error)
    assert error.role == "middle"
    assert error.endpoint == "127.0.0.1:50183"
