"""tests/test_scheduler_relay_segment.py — ★ A1 / X 档：调度层「Relay 段委托」的接线级用例。

不打起完整调度器：构造 `SchedulerPipelineMixin` 的**最小假实例**（只 stub 三个协作方法），
把 `_normalize_relay_segment` 与 `_handle_layer_forward_via_relay` 当**单元**测 —— 毫秒级覆盖全部分支，
不需要模型、不需要 master/worker 拓扑（真拓扑回归在 X 档后续的 `tests/test_scheduler_relay_segment.py`
扩展里）。线协议部分用真实 loopback 服务 + 假 runner。

⚠️ 含「该红必须红」断言：
1. `PIPELINE_RELAY_ENABLED` **默认必须是 False**（默认打开 = 生产路径被静默改动 ⇒ 立刻红）；
2. 段失败时**必须抛具名异常**、**不得**静默发出空 hidden（把失败吞掉 ⇒ 立刻红）。
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

import config  # noqa: E402
import scheduler_pipeline  # noqa: E402
from relay_segment_client import (  # noqa: E402
    RelaySegmentError,
    RelaySegmentOutcome,
)
from relay_transport import (  # noqa: E402
    RELAY_TRANSPORT_ERROR,
    expected_hidden_bytes,
    open_loopback_listener,
    serve_relay_middle_connection,
)
from scheduler_pipeline import SchedulerPipelineMixin  # noqa: E402
from scheduler_pipeline import RELAY_HIDDEN_WIRE_FORMAT, _decode_relay_hidden, _encode_relay_hidden  # noqa: E402

N_EMBD = 4
N_TOKENS = 2


@pytest.fixture(autouse=True)
def _relay_probe_only_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """本文件整体验证的是 A1 relay 的**行为**（探针/实验语义）。

    2026-10-04 产品裁定（分票规划 DIST-0）把 A1 从生产调度入口剔除后，
    `PIPELINE_RELAY_PROBE_ONLY` 默认为 1 ⇒ `_relay_segment_for_worker` 在生产
    路径直接不供给 relay 段。这些用例直接调用该方法，属探针语义，故在本文件
    统一关掉闸门。**产品路径的剔除行为**由
    `test_relay_segment_for_worker_respects_switch`（关掉 `RELAY_ENABLED`）与
    启动期角色互斥检测（`api_server`）另行覆盖。
    """
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_PROBE_ONLY", False)


class _PlusOneRunner:
    def __init__(self) -> None:
        self.seen: list[tuple[int, bytes]] = []
        self.meta: dict[str, object] | None = None
        self.resets = 0

    def request_hidden(self, hidden: bytes, *, n_tokens: int) -> bytes:
        self.seen.append((n_tokens, hidden))
        return bytes((value + 1) % 256 for value in hidden)

    def request_hidden_seq(self, hidden: bytes, *, n_tokens: int, meta) -> bytes:
        self.seen.append((n_tokens, hidden))
        self.meta = meta
        return bytes((value + 1) % 256 for value in hidden)

    def reset(self) -> None:
        self.resets += 1


def _hidden(seed: int = 0) -> bytes:
    return bytes((seed + i) % 256 for i in range(expected_hidden_bytes(N_TOKENS, N_EMBD)))


class _Harness:
    """最小假实例：只 stub 与 relay 分支协作的三个方法，并记录发回主节点的结果。"""

    def __init__(self) -> None:
        self.obj = object.__new__(SchedulerPipelineMixin)
        self.sent: list[dict[str, object]] = []
        self.began: list[str] = []
        self.obj.get_effective_node_id = lambda: "worker-1"     # type: ignore[method-assign]
        self.obj._begin_local_pipeline_task = self.began.append   # type: ignore[method-assign]
        self.obj._send_layer_result = self._send_layer_result     # type: ignore[method-assign]

    def _send_layer_result(self, client_id: str, task_id: str, **kwargs) -> bool:
        self.sent.append({"client_id": client_id, "task_id": task_id, **kwargs})
        return True


def _listen_middle(runner):
    listener = open_loopback_listener("127.0.0.1", 0)
    port = int(listener.getsockname()[1])
    box: dict[str, object] = {}

    def _run() -> None:
        sock, _ = listener.accept()
        try:
            box["bridge"] = serve_relay_middle_connection(sock, runner, n_embd=N_EMBD,
                                                          max_tokens=64)
        finally:
            sock.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return listener, port, thread


def _spec(port: int, **overrides) -> dict[str, object]:
    spec = {"role": "middle", "host": "127.0.0.1", "port": port, "n_embd": N_EMBD,
            "timeout": 5.0, "layer_start": 8, "layer_end": 16}
    spec.update(overrides)
    return spec


def test_relay_wire_format_is_raw_f32_and_shape_preserving():
    import torch

    value = torch.arange(8, dtype=torch.float16).reshape(1, 2, 4)
    encoded, shape = _encode_relay_hidden(value)
    raw = __import__("base64").b64decode(encoded)
    assert shape == [1, 2, 4]
    assert len(raw) == 1 * 2 * 4 * 4
    assert _decode_relay_hidden(raw, shape).dtype == torch.float32
    assert _decode_relay_hidden(raw, shape).shape == value.shape


def test_relay_wire_format_rejects_shape_mismatch():
    import torch

    encoded, _ = _encode_relay_hidden(torch.zeros((1, 2, N_EMBD)))
    raw = __import__("base64").b64decode(encoded)
    with pytest.raises(ValueError, match="length mismatch"):
        _decode_relay_hidden(raw, [1, 3, N_EMBD])


# ---- _normalize_relay_segment：严格拒绝 ------------------------------------


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        (None, "缺失"),
        ("middle", "不是 dict"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4,
          "layer_start": 0, "layer_end": 1, "extra": 1}, "未知键"),
        ({"role": "router", "host": "127.0.0.1", "port": 1, "n_embd": 4,
          "layer_start": 0, "layer_end": 1}, "未知角色"),
        ({"role": "middle", "host": "10.1.2.3", "port": 1, "n_embd": 4,
          "layer_start": 0, "layer_end": 1}, "非 loopback"),
        ({"role": "middle", "host": "127.0.0.1", "port": 0, "n_embd": 4,
          "layer_start": 0, "layer_end": 1}, "端口越界"),
        ({"role": "middle", "host": "127.0.0.1", "port": 70000, "n_embd": 4,
          "layer_start": 0, "layer_end": 1}, "端口越界"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 0,
          "layer_start": 0, "layer_end": 1}, "n_embd 非法"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4,
          "layer_start": 0, "layer_end": 1, "timeout": 0}, "timeout 非法"),
        ({"role": "middle", "host": "127.0.0.1", "port": "x", "n_embd": 4,
          "layer_start": 0, "layer_end": 1}, "端口不可解析"),
        # ★ Y 档第二条：层区间**必填** —— 缺了就是"半懂规格"，必须整条不认（fail-closed），
        #   否则主节点无从把该段的层从自己范围里扣除（那正是本线修的那个根因）。
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4}, "缺层区间"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4,
          "layer_start": 8}, "只给起点"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4,
          "layer_end": 8}, "只给终点"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4,
          "layer_start": 8, "layer_end": 8}, "空区间"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4,
          "layer_start": -1, "layer_end": 8}, "负起点"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4,
          "layer_start": 16, "layer_end": 8}, "起点不早于终点"),
    ],
)
def test_normalize_rejects_out_of_scope(raw, why: str):
    assert SchedulerPipelineMixin._normalize_relay_segment(raw) is None, why


def test_normalize_accepts_middle_loopback():
    spec = SchedulerPipelineMixin._normalize_relay_segment(
        {"role": "MIDDLE", "host": "localhost", "port": 50183, "n_embd": 896,
         "layer_start": 8, "layer_end": 16, "timeout": 30})
    assert spec == {"role": "middle", "host": "localhost", "port": 50183, "n_embd": 896,
                    "timeout": 30.0, "layer_start": 8, "layer_end": 16}


def test_normalize_defaults_timeout():
    spec = SchedulerPipelineMixin._normalize_relay_segment(
        {"role": "middle", "host": "127.0.0.1", "port": 50183, "n_embd": 896,
         "layer_start": 8, "layer_end": 16})
    assert spec is not None and spec["timeout"] == 60.0


@pytest.mark.parametrize(
    ("role", "start", "end"),
    [("middle", 8, 16), ("tail", 16, 24)],
)
def test_normalize_accepts_hidden_input_roles(role: str, start: int, end: int):
    """当前 scheduler relay 输入是 hidden，只接受 middle/tail。"""
    spec = SchedulerPipelineMixin._normalize_relay_segment(
        {"role": role, "host": "127.0.0.1", "port": 50183, "n_embd": 896,
         "layer_start": start, "layer_end": end})
    assert spec == {"role": role, "host": "127.0.0.1", "port": 50183, "n_embd": 896,
                    "timeout": 60.0, "layer_start": start, "layer_end": end}


def test_normalize_rejects_head_until_token_input_protocol_is_wired():
    spec = SchedulerPipelineMixin._normalize_relay_segment(
        {"role": "head", "host": "127.0.0.1", "port": 50183, "n_embd": 896,
         "layer_start": 0, "layer_end": 8})
    assert spec is None


# ---- _handle_layer_forward_via_relay：成功路径 ----------------------------


def test_via_relay_forwards_hidden_and_sends_result():
    runner = _PlusOneRunner()
    listener, port, thread = _listen_middle(runner)
    harness = _Harness()
    try:
        harness.obj._handle_layer_forward_via_relay(
            _spec(port),
            data={"hidden_states": _hidden(), "task_id": "t1", "step": 0},
            task_id="t1", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        thread.join(timeout=5)
    finally:
        listener.close()

    assert harness.began == ["t1"]                    # begin/finish 平衡的前提
    assert len(harness.sent) == 1
    sent = harness.sent[0]
    assert sent["client_id"] == "master" and sent["task_id"] == "t1"
    result = sent["result_data"]
    expected = bytes((value + 1) % 256 for value in _hidden())
    assert result["hidden_states"] == expected        # 逐字节正确（且不是原样返回）
    assert result["hidden_states"] != _hidden()
    assert result["hidden_shape"] == [N_TOKENS, N_EMBD]
    assert result["chain_path"] == ["worker-1"]
    assert result["metrics"]["relay_executed"] is True
    assert set(result["metrics"]) >= {"relay_segment", "relay_frames", "relay_tokens",
                                      "relay_payload_bytes", "relay_error"}
    assert result["metrics"]["relay_error"] == ""
    assert runner.seen and runner.seen[0][0] == N_TOKENS


def test_via_relay_passes_seq_meta():
    runner = _PlusOneRunner()
    listener, port, thread = _listen_middle(runner)
    harness = _Harness()
    try:
        harness.obj._handle_layer_forward_via_relay(
            _spec(port),
            data={"hidden_states": _hidden(), "task_id": "t2", "step": 0,
                  "seq_ids": [0, 1], "positions": [0, 0]},
            task_id="t2", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        thread.join(timeout=5)
    finally:
        listener.close()

    assert runner.meta is not None
    assert list(runner.meta["seq_ids"]) == [0, 1]
    assert list(runner.meta["n_seq_id"]) == [1, 1]


def test_relay_session_is_reused_until_task_finish():
    runner = _PlusOneRunner()
    listener = open_loopback_listener("127.0.0.1", 0)
    port = int(listener.getsockname()[1])
    box: dict[str, object] = {}

    def _run() -> None:
        sock, _ = listener.accept()
        try:
            box["bridge"] = serve_relay_middle_connection(
                sock, runner, n_embd=N_EMBD, max_tokens=64,
            )
        finally:
            sock.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    harness = _Harness()
    try:
        spec = _spec(port)
        harness.obj._handle_layer_forward_via_relay(
            spec, data={"hidden_states": _hidden(), "task_id": "reuse", "step": 0},
            task_id="reuse", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        harness.obj._handle_layer_forward_via_relay(
            spec, data={"hidden_states": _hidden(1), "task_id": "reuse", "step": 1},
            task_id="reuse", step=1, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        assert len(runner.seen) == 2
        assert "reuse" in harness.obj._relay_segment_clients
        harness.obj._close_relay_segment_client("reuse")
        assert "reuse" not in harness.obj._relay_segment_clients
        thread.join(timeout=5)
    finally:
        listener.close()


def test_relay_session_cache_key_ignores_dynamic_remaining_timeout():
    harness = _Harness()
    first = harness.obj._relay_segment_client_for_task(
        "stable-timeout", _spec(50183, timeout=5.0), n_embd=N_EMBD,
    )
    second = harness.obj._relay_segment_client_for_task(
        "stable-timeout", _spec(50183, timeout=0.25), n_embd=N_EMBD,
    )

    assert second is first
    harness.obj._close_relay_segment_client("stable-timeout")


def test_via_relay_passes_wire_deadline_as_absolute_monotonic(
        monkeypatch: pytest.MonkeyPatch):
    observed: dict[str, float | None] = {}

    class CapturingClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def forward_hidden(
                self, hidden, *, n_tokens, seq_meta=None,
                deadline_monotonic=None):
            del n_tokens, seq_meta
            observed["deadline"] = deadline_monotonic
            return RelaySegmentOutcome(
                hidden=bytes(hidden),
                role="middle",
                endpoint="127.0.0.1:50183",
            )

        def close(self, **_kwargs):
            return None

    monkeypatch.setattr(scheduler_pipeline, "RelaySegmentClient", CapturingClient)
    harness = _Harness()
    deadline_ms = int((time.time() + 1.0) * 1000)
    before = time.monotonic()

    harness.obj._handle_layer_forward_via_relay(
        _spec(50183),
        data={
            "hidden_states": _hidden(),
            "task_id": "deadline-wire",
            "step": 0,
            "request_deadline_ms": deadline_ms,
        },
        task_id="deadline-wire", step=0, config_id="c1",
        model_sha256="sha", model_type="qwen", received_chain_path=[],
    )

    assert observed["deadline"] is not None
    assert before < observed["deadline"] <= before + 1.2


def test_via_relay_expired_wire_deadline_does_not_construct_client(
        monkeypatch: pytest.MonkeyPatch):
    constructed = []

    class UnexpectedClient:
        def __init__(self, *_args, **_kwargs):
            constructed.append(True)

    monkeypatch.setattr(scheduler_pipeline, "RelaySegmentClient", UnexpectedClient)
    harness = _Harness()

    with pytest.raises(RelaySegmentError) as excinfo:
        harness.obj._handle_layer_forward_via_relay(
            _spec(50183),
            data={
                "hidden_states": _hidden(),
                "task_id": "expired-wire",
                "step": 0,
                "request_deadline_ms": int(time.time() * 1000) - 1,
            },
            task_id="expired-wire", step=0, config_id="c1",
            model_sha256="sha", model_type="qwen", received_chain_path=[],
        )

    assert excinfo.value.code == "request_deadline_exceeded"
    assert constructed == []


# ---- 失败/越界必须具名（绝不静默）-----------------------------------------


def test_via_relay_failure_is_named_and_sends_nothing():
    """段不可达 ⇒ **抛具名异常**（供外层回传 `relay_segment_failed:*`），**不发**空 hidden。

    「该红必须红」：任何人把这里的 raise 改成"发一个空 hidden 继续"，本用例立刻红 ——
    而那种改动会让失败退化成"模型算错"（正是我们要消灭的那类症状）。
    """
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = int(probe.getsockname()[1])
    probe.close()

    harness = _Harness()
    with pytest.raises(RelaySegmentError) as excinfo:
        harness.obj._handle_layer_forward_via_relay(
            _spec(dead_port, timeout=1.0),
            data={"hidden_states": _hidden(), "task_id": "t3", "step": 0},
            task_id="t3", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )

    assert excinfo.value.code == RELAY_TRANSPORT_ERROR
    assert harness.sent == []          # 绝不在失败时发出 LAYER_RESULT


def test_via_relay_rejects_chain_next():
    """X 档只做 2 段拓扑：带 `chain_next` 显式拒绝（>2 段属 Y 档）。"""
    harness = _Harness()
    with pytest.raises(RuntimeError, match="链式转发"):
        harness.obj._handle_layer_forward_via_relay(
            _spec(50183),
            data={"hidden_states": _hidden(), "task_id": "t4", "step": 0,
                  "chain_next": {"node_id": "worker-2"}},
            task_id="t4", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
    assert harness.sent == []


def test_via_relay_rejects_wrong_hidden_size():
    harness = _Harness()
    with pytest.raises(RuntimeError, match="长度与 n_embd 不匹配"):
        harness.obj._handle_layer_forward_via_relay(
            _spec(50183),
            data={"hidden_states": b"too-short", "task_id": "t5", "step": 0},
            task_id="t5", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
    assert harness.sent == []


def test_via_relay_rejects_missing_hidden():
    harness = _Harness()
    with pytest.raises(RuntimeError, match="需要 hidden_states"):
        harness.obj._handle_layer_forward_via_relay(
            _spec(50183), data={"task_id": "t6", "step": 0},
            task_id="t6", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
    assert harness.sent == []


# ---- 主节点侧：下发规格与具名回退 -----------------------------------------


def test_parse_relay_segment_map_accepts_valid_and_drops_invalid():
    """配置解析：只认范围内、且**声明了层区间**的条目，其余**整条丢弃**（绝不下发"半懂"规格）。"""
    raw = ("worker-2=middle@127.0.0.1:50183#896#8-16;"           # 合法
           "worker-3=router@127.0.0.1:50184#896#8-16;"          # ★ 未知角色，丢
           "worker-10=tail@127.0.0.1:50190#896#16-24;"          # ★ Y-(b)：tail 合法
           "worker-4=middle@10.0.0.5:50185#896#8-16;"           # 非 loopback，丢
           "broken;worker-5=middle@127.0.0.1:notaport#896#8-16;"  # 非法，丢
           "worker-7=middle@127.0.0.1:50187#896;"               # ★ 缺层区间，丢
           "worker-8=middle@127.0.0.1:50188#896#16-16;"         # ★ 空区间，丢
           "worker-9=middle@127.0.0.1:50189#896#8-16#extra;"    # ★ 多余字段，丢
           "worker-6=middle@127.0.0.1:50186#896#16-24")         # 合法
    parsed = SchedulerPipelineMixin._parse_relay_segment_map(raw)

    assert set(parsed) == {"worker-2", "worker-6", "worker-10"}
    assert parsed["worker-2"] == {"role": "middle", "host": "127.0.0.1", "port": 50183,
                                  "n_embd": 896, "timeout": 60.0,
                                  "layer_start": 8, "layer_end": 16}
    assert parsed["worker-6"]["layer_start"] == 16
    assert parsed["worker-6"]["layer_end"] == 24


def test_parse_relay_segment_map_empty_is_empty():
    assert SchedulerPipelineMixin._parse_relay_segment_map("") == {}
    assert SchedulerPipelineMixin._parse_relay_segment_map("   ;  ") == {}


def test_relay_segment_for_worker_respects_switch(monkeypatch: pytest.MonkeyPatch):
    """★ 开关关闭 ⇒ **永远不下发**（对既有路径零影响）。"""
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", False)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                        "worker-2=middle@127.0.0.1:50183#896#8-16")
    harness = _Harness()

    assert harness.obj._relay_segment_for_worker("worker-2") is None


def test_relay_segment_for_worker_probe_only_excludes_from_production(
    monkeypatch: pytest.MonkeyPatch,
):
    """★ 2026-10-04 产品裁定（分票规划 DIST-0）：A1 已从产品调度入口剔除。

    `QLH_RELAY_PROBE_ONLY` 默认 1 ⇒ **即使 relay 全开也不供给 relay 段**，
    生产请求继续走 A3，不做静默切换。本文件的 autouse fixture 为测 relay 行为
    而关掉了该闸门，所以这里显式打开它。
    """
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_PROBE_ONLY", True)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                        "worker-2=middle@127.0.0.1:50183#896#8-16")
    harness = _Harness()

    # 配置齐全，但闸门开着 ⇒ 生产路径拿不到 relay 段。
    assert harness.obj._relay_segment_for_worker("worker-2") is None
    # 具名诊断只记一次（每进程一份），不刷屏。
    assert getattr(harness.obj, "_relay_probe_only_warned", False) is True

    # 显式关掉闸门（探针/实验语义）⇒ 恢复原有行为。
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_PROBE_ONLY", False)
    assert harness.obj._relay_segment_for_worker("worker-2") is not None


def test_relay_segment_for_worker_returns_spec_and_caches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                        "worker-2=middle@127.0.0.1:50183#896#8-16")
    harness = _Harness()

    spec = harness.obj._relay_segment_for_worker("worker-2")
    assert spec is not None and spec["port"] == 50183
    assert spec["layer_start"] == 8 and spec["layer_end"] == 16
    assert harness.obj._relay_segment_for_worker("worker-9") is None


def test_relay_segment_for_worker_accepts_client_prefix_alias(monkeypatch: pytest.MonkeyPatch):
    """Deployment names may omit the transport client's ``client_`` prefix."""
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(
        scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
        "tablet=middle@127.0.0.1:50183#896#8-16",
    )
    harness = _Harness()

    spec = harness.obj._relay_segment_for_worker("client_tablet")

    assert spec is not None
    assert spec["port"] == 50183


def test_relay_segment_for_worker_exact_name_wins_over_alias(monkeypatch: pytest.MonkeyPatch):
    """An explicit client_ key must remain authoritative when both names exist."""
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(
        scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
        "tablet=middle@127.0.0.1:50183#896#8-16;"
        "client_tablet=tail@127.0.0.1:50184#896#16-24",
    )
    harness = _Harness()

    spec = harness.obj._relay_segment_for_worker("client_tablet")

    assert spec is not None
    assert spec["role"] == "tail"
    assert spec["port"] == 50184

    # 缓存：解析一次后进程内稳定（改配置不影响已解析结果）
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                        "worker-9=middle@127.0.0.1:50199#896#8-16")
    assert harness.obj._relay_segment_for_worker("worker-9") is None


def test_via_relay_failure_message_is_readable_for_fallback_reason():
    """★ 失败消息必须能直接落进 `_fallback_reason`（带 `relay_segment_failed:` 前缀）。

    「该红必须红」：把 `detail` 去掉（只留白名单码）会让这条红 —— 而主节点的
    `_fallback_reason` 就只能写笼统码，正是我们要消灭的"与模型算错难以区分"。
    """
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = int(probe.getsockname()[1])
    probe.close()

    harness = _Harness()
    with pytest.raises(RelaySegmentError) as excinfo:
        harness.obj._handle_layer_forward_via_relay(
            _spec(dead_port, timeout=1.0),
            data={"hidden_states": _hidden(), "task_id": "t7", "step": 0},
            task_id="t7", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )

    message = str(excinfo.value)
    assert message.startswith(f"relay_segment_failed:{RELAY_TRANSPORT_ERROR}")
    assert "middle@127.0.0.1" in message
    assert excinfo.value.code == RELAY_TRANSPORT_ERROR     # code 与可读消息解耦
    assert harness.sent == []


# ---- 接线闭环 -------------------------------------------------------------


def test_end_to_end_config_to_worker_relay_roundtrip(monkeypatch: pytest.MonkeyPatch):
    """★ 接线闭环：主节点配置 ⇒ 下发规格 ⇒ worker 认它 ⇒ 真 middle 服务跑通。

    这是 X 档「本机可闭环」的核心证据：全链路（配置解析 → 两侧判据同源 → 段委托 →
    逐字节往返）都在本进程内完成，**不需要模型、不需要多节点**。
    """
    runner = _PlusOneRunner()
    listener, port, thread = _listen_middle(runner)
    harness = _Harness()
    try:
        monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
        monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                            f"worker-1=middle@127.0.0.1:{port}#{N_EMBD}#8-16")

        # ① 主节点侧：解析配置并取该 worker 的规格（= 会随 LAYER_FORWARD 下发的那个 dict）
        spec = harness.obj._relay_segment_for_worker("worker-1")
        assert spec is not None

        # ② worker 侧：收到的规格必须被 `_normalize_relay_segment` **原样**接受
        #    （两侧判据同源 ⇒ 不会出现"主节点下发、worker 不认"的隐性不对称）
        normalized = SchedulerPipelineMixin._normalize_relay_segment(spec)
        assert normalized == spec

        # ③ 执行：真 loopback 往返 ⇒ 逐字节正确
        harness.obj._handle_layer_forward_via_relay(
            normalized,
            data={"hidden_states": _hidden(), "task_id": "e2e", "step": 0},
            task_id="e2e", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        thread.join(timeout=5)
    finally:
        listener.close()

    result = harness.sent[0]["result_data"]
    assert result["hidden_states"] == bytes((v + 1) % 256 for v in _hidden())
    assert result["hidden_states"] != _hidden()
    assert result["metrics"]["relay_executed"] is True
    assert result["metrics"]["relay_tokens"] == N_TOKENS


# ---- 「该红必须红」：开关默认必须关 ---------------------------------------


def test_guard_relay_switch_defaults_off():
    """★「该红必须红」：`QLH_RELAY_ENABLED` **默认关**。

    默认打开意味着生产路径被静默改动（没有显式 opt-in 的节点也会走 Relay 分支），
    任何人把默认值改成 True，这条立刻红。
    """
    assert config.PIPELINE_RELAY_ENABLED is False
    assert scheduler_pipeline.PIPELINE_RELAY_ENABLED is False


def test_guard_branch_requires_both_switch_and_spec(monkeypatch: pytest.MonkeyPatch):
    """★「该红必须红」：委托需要**开关 + 合法规格**同时成立。

    直接按 `_handle_layer_forward_locked` 里的分支条件断言：任一条件不成立都必须
    退回既有拒绝路径（而不是走 Relay）。
    """
    def _would_delegate(raw: object, enabled: bool) -> bool:
        spec = SchedulerPipelineMixin._normalize_relay_segment(raw)
        return bool(enabled) and spec is not None

    good = {"role": "middle", "host": "127.0.0.1", "port": 50183, "n_embd": 896,
            "layer_start": 8, "layer_end": 16}
    assert _would_delegate(good, True) is True
    assert _would_delegate(good, False) is False        # 开关关 ⇒ 不委托
    assert _would_delegate(None, True) is False         # 无规格 ⇒ 不委托
    assert _would_delegate({"role": "router", "host": "127.0.0.1", "port": 1,
                            "n_embd": 4, "layer_start": 0, "layer_end": 1}, True) is False
    # ★ Y-(b)：`tail` 角色合法（其区间语义由切分校验把关）⇒ **可以**委托
    assert _would_delegate({"role": "tail", "host": "127.0.0.1", "port": 1,
                            "n_embd": 4, "layer_start": 16, "layer_end": 24}, True) is True
    # ★ Y 档第二条：缺层区间 ⇒ 半懂规格不得生效，同样不委托
    assert _would_delegate({"role": "middle", "host": "127.0.0.1", "port": 1,
                            "n_embd": 4}, True) is False


# ---- 「该红必须红」：请求级路由偏好必须能挡住 relay 委派 -------------------


def _enable_relay_with_one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """打开开关 + 给一个 worker 配好 middle 段（两处都走 monkeypatch，不碰环境变量）。"""
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(
        scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
        "worker-2=middle@127.0.0.1:50183#896#8-16",
    )


def test_guard_local_only_never_delegates_to_a_relay_segment(monkeypatch: pytest.MonkeyPatch):
    """★「该红必须红」：`routing_preference == "local_only"` ⇒ **绝不**委派给远端 relay 段。

    这是 §4.7「X 档包含」第 3 项的**请求级那一半**。把 `_relay_segment_for_worker` 里
    `local_only ⇒ None` 那个闸门去掉，本用例立刻红 —— 届时带「只要本地算」语义的请求仍会被
    派给远端段。API 层虽然已经在 `local_only` 时跳过整条流水线路径（`api_server.py`），
    但这里守的是**流水线内部**那条边界：换任何入口进来都不该被绕过。
    """
    _enable_relay_with_one_worker(monkeypatch)
    harness = _Harness()

    # 先证明这条链路本身是通的 —— 否则下面的 None 可能只是"什么都没配上"的假通过。
    assert harness.obj._relay_segment_for_worker("worker-2") is not None
    assert harness.obj._relay_segment_for_worker("worker-2", "auto") is not None

    # ⇒ `local_only` 必须挡住。
    assert harness.obj._relay_segment_for_worker("worker-2", "local_only") is None


def test_default_routing_preference_is_equivalent_to_auto(monkeypatch: pytest.MonkeyPatch):
    """不传 `routing_preference` 必须与显式 `"auto"` **等价** ⇒ 旧调用方零影响。"""
    _enable_relay_with_one_worker(monkeypatch)
    harness = _Harness()

    default = harness.obj._relay_segment_for_worker("worker-2")
    explicit = harness.obj._relay_segment_for_worker("worker-2", "auto")

    assert default == explicit
    assert default is not None


def test_only_local_only_blocks_delegation(monkeypatch: pytest.MonkeyPatch):
    """闸门**只**认 `local_only` —— `auto` / `distributed_required` / 空串 / 未来新值都不挡。"""
    _enable_relay_with_one_worker(monkeypatch)
    harness = _Harness()

    for preference in ("auto", "distributed_required", "", "some-future-value"):
        assert harness.obj._relay_segment_for_worker("worker-2", preference) is not None, preference


# ---- Y 档第一条：主节点展示 relay 指标 -------------------------------------


def test_extract_relay_metrics_picks_only_the_five_keys():
    """只取五个 relay 键（其余 metrics 字段不外带），供主节点 status 展示。"""
    metrics = {
        "time_ms": 12.5, "kv_cache": False, "relay_executed": True,
        "relay_segment": "middle@127.0.0.1:50183",
        "relay_frames": 4, "relay_tokens": 4,
        "relay_payload_bytes": 4096, "relay_error": None,
        "fallback_reason": "",           # 不该被带出来
    }

    picked = SchedulerPipelineMixin._extract_relay_metrics(metrics)

    assert picked == {
        "relay_segment": "middle@127.0.0.1:50183",
        "relay_frames": 4,
        "relay_tokens": 4,
        "relay_payload_bytes": 4096,
        "relay_error": None,
    }


def test_guard_extract_relay_metrics_does_not_misfire_on_plain_path():
    """★「该红必须红」：**普通 pytorch 路径**的 metrics 必须取不出 relay 字段。

    守卫的是"误报"：把 `_extract_relay_metrics` 里 `relay_segment` 非空那个判断去掉，
    任何 metrics 都会被当成 relay 结果，主节点 status 就**凭空显示** relay 指标 ——
    那是最难查的一类假证据。去掉该判断，本用例立刻红。
    """
    plain = {"time_ms": 8.0, "kv_cache": True, "distributed_used": True}

    assert SchedulerPipelineMixin._extract_relay_metrics(plain) == {}
    # 半残 metrics（只有 relay_* 键、没有 relay_segment）同样不算 relay 结果。
    assert SchedulerPipelineMixin._extract_relay_metrics({"relay_frames": 3}) == {}
    # 非 dict 输入不得抛。
    assert SchedulerPipelineMixin._extract_relay_metrics(None) == {}
    assert SchedulerPipelineMixin._extract_relay_metrics("relay_segment") == {}


def test_pipeline_status_relay_is_empty_before_any_relay_run():
    """没走过 relay 时 status 的 `relay` 必须是**空 dict**（不是 None、也不能缺键）。"""
    harness = _status_harness()

    status = harness.obj._get_pipeline_status()

    assert status["relay"] == {}


def test_pipeline_status_surfaces_the_last_relay_metrics():
    """收到过 relay 结果后，status 必须把它读出来（这是 Y 档第一条的可见产物）。"""
    harness = _status_harness()
    harness.obj._last_relay_metrics = {
        "node_id": "worker-2", "task_id": "t1", "step": 0,
        "relay_segment": "middle@127.0.0.1:50183",
        "relay_frames": 4, "relay_tokens": 4,
        "relay_payload_bytes": 4096, "relay_error": None,
    }

    status = harness.obj._get_pipeline_status()

    assert status["relay"]["relay_segment"] == "middle@127.0.0.1:50183"
    assert status["relay"]["relay_tokens"] == 4
    assert status["relay"]["node_id"] == "worker-2"
    # 必须是副本，调用方改它不该污染调度器内部状态。
    status["relay"]["relay_tokens"] = 999
    assert harness.obj._last_relay_metrics["relay_tokens"] == 4


def _status_harness() -> _Harness:
    """`_get_pipeline_status()` 需要的最小实例。

    该方法的依赖比 relay 分支宽（`_nodes_lock` / `_inference_lock` / `nodes` 等），
    这里一并补齐 —— 免得 `AttributeError` 抢在断言的语义之前把用例变成"假红"。
    """
    harness = _Harness()
    harness.obj._host = None
    harness.obj.nodes = {}
    harness.obj._nodes_lock = threading.RLock()
    harness.obj._inference_lock = threading.RLock()
    harness.obj.get_layer_assignments = lambda: {"assignments": []}
    harness.obj.get_distributed_inference_enabled = lambda: False
    harness.obj._effective_role = lambda: "master"
    harness.obj._get_pipeline_readiness = lambda: {
        "ready": False, "reason_code": "no_workers", "reason": "无", "workers": [],
    }
    return harness
