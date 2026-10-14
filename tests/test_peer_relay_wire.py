"""Regression tests for the product PeerClient relay wire contract."""

from __future__ import annotations

import base64
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT), str(ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from inference_service.peer import (  # noqa: E402
    PeerClient,
    RELAY_HIDDEN_WIRE_FORMAT,
)
from scheduler_pipeline import _decode_relay_hidden  # noqa: E402


class _Outcome:
    ok = True
    role = "middle"
    error = ""
    endpoint = "127.0.0.1:50283"

    def __init__(self, hidden: bytes):
        self.hidden = hidden

    def to_metrics(self):
        return {"relay_segment": "middle"}


class _RelayClient:
    received: bytes | None = None

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def forward_hidden(
        self, hidden: bytes, *, n_tokens: int, deadline_monotonic=None
    ):
        assert n_tokens == 2
        type(self).received = hidden
        values = np.frombuffer(hidden, dtype=np.float32).copy() + 1.0
        return _Outcome(values.tobytes())


def _peer(sent: list[dict], *, role: str = "middle") -> PeerClient:
    peer = object.__new__(PeerClient)
    peer._node_id = "surface-worker"
    peer._layer_execution_lock = threading.RLock()
    peer._layer_config_lock = threading.RLock()
    peer._kv_cache_lock = threading.RLock()
    peer._local_pipeline_cancelled = set()
    peer._local_pipeline_steps = {}
    peer._active_pipeline_task_ids = set()
    peer._kv_cache = {}
    peer._relay_sessions = {}
    peer._active_layer_config = {
        "engine": "relay_middle",
        "config_id": "cfg-relay",
        "model_sha256": "relay-sha",
        "model_type": "qwen2",
        "relay_segment": {
            "host": "127.0.0.1",
            "port": 50283,
            "role": role,
            "n_embd": 3,
            "timeout": 2.0,
        }
    }
    peer._send_layer_result = lambda _task_id, payload, error=None: sent.append(
        {**payload, **({"error": error} if error else {})}
    ) or True
    return peer


def test_peer_relay_roundtrip_uses_raw_f32_contract(monkeypatch):
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _RelayClient)
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sent: list[dict] = []
    peer = _peer(sent)

    peer._handle_layer_forward_via_relay(
        {
            "task_id": "t1",
            "step": 0,
            "hidden_states": base64.b64encode(values.tobytes()).decode("ascii"),
            "hidden_shape": list(values.shape),
            "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
        }
    )

    assert _RelayClient.received == values.tobytes()
    assert len(sent) == 1 and "error" not in sent[0]
    assert sent[0]["hidden_wire_format"] == RELAY_HIDDEN_WIRE_FORMAT
    assert sent[0]["hidden_shape"] == [1, 2, 3]
    returned = _decode_relay_hidden(
        sent[0]["hidden_states"], sent[0]["hidden_shape"]
    )
    np.testing.assert_array_equal(returned.numpy(), values + 1.0)


def test_peer_relay_rejects_raw_f32_without_shape(monkeypatch):
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _RelayClient)
    sent: list[dict] = []
    peer = _peer(sent)

    peer._handle_layer_forward_via_relay(
        {
            "task_id": "t2",
            "hidden_states": base64.b64encode(np.zeros(3, dtype=np.float32).tobytes()).decode("ascii"),
            "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
        }
    )

    assert len(sent) == 1
    assert "hidden_shape is required" in sent[0]["error"]


class _SessionRelayClient:
    instances: list["_SessionRelayClient"] = []

    def __init__(self, *_args, **kwargs):
        self.calls = 0
        self.close_calls = 0
        self.close_graces: list[float | None] = []
        self.deadline_monotonic = kwargs.get("deadline_monotonic")
        self.timeout = kwargs.get("timeout")
        type(self).instances.append(self)

    def tighten_deadline(self, deadline):
        if deadline is None:
            return
        if self.deadline_monotonic is None or deadline < self.deadline_monotonic:
            self.deadline_monotonic = deadline

    def forward_hidden(
        self, hidden: bytes, *, n_tokens: int, deadline_monotonic=None
    ):
        self.tighten_deadline(deadline_monotonic)
        self.calls += 1
        return _Outcome((np.frombuffer(hidden, dtype=np.float32) + self.calls).tobytes())

    def close(self, *, grace_timeout=None):
        self.close_calls += 1
        self.close_graces.append(grace_timeout)


class _SeqRelayClient:
    seen: dict[str, object] | None = None

    def __init__(self, *_args, **_kwargs):
        pass

    def forward_hidden(
        self, hidden: bytes, *, n_tokens: int, seq_meta=None,
        deadline_monotonic=None,
    ):
        type(self).seen = seq_meta
        return _Outcome(hidden)

    def close(self):
        pass


def test_peer_relay_passes_explicit_positions_to_middle(monkeypatch):
    _SeqRelayClient.seen = None
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _SeqRelayClient)
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sent: list[dict] = []
    peer = _peer(sent)
    payload = _payload(values, task_id="seq-task")
    payload.update({"seq_ids": [0, 0], "positions": [17, 18]})

    peer._handle_layer_forward_via_relay(payload)

    assert _SeqRelayClient.seen == {
        "n_seq_id": [1, 1], "seq_ids": [0, 0], "positions": [17, 18],
    }


def test_peer_relay_rejects_mismatched_positions_before_connect(monkeypatch):
    _SeqRelayClient.seen = None
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _SeqRelayClient)
    sent: list[dict] = []
    peer = _peer(sent)
    payload = _payload(np.zeros((1, 2, 3), dtype=np.float32), task_id="bad-seq")
    payload.update({"seq_ids": [0], "positions": [17, 18]})

    peer._handle_layer_forward_via_relay(payload)

    assert _SeqRelayClient.seen is None
    assert sent[0]["error"] == "relay seq_ids/positions must match hidden token count"


def test_peer_relay_expired_wire_deadline_never_constructs_client(monkeypatch):
    constructed: list[object] = []

    class _UnexpectedRelayClient:
        def __init__(self, *_args, **_kwargs):
            constructed.append(self)

    monkeypatch.setattr(
        "relay_segment_client.RelaySegmentClient", _UnexpectedRelayClient
    )
    sent: list[dict] = []
    peer = _peer(sent)
    payload = _payload(np.zeros((1, 2, 3), dtype=np.float32), task_id="expired")
    payload["request_deadline_ms"] = int(time.time() * 1000) - 1

    peer._handle_layer_forward_via_relay(payload)

    assert constructed == []
    assert peer._relay_sessions == {}
    assert sent == [{"error": "request_deadline_exceeded"}]


def test_peer_relay_session_key_ignores_dynamic_timeout(monkeypatch):
    _SessionRelayClient.instances.clear()
    monkeypatch.setattr(
        "relay_segment_client.RelaySegmentClient", _SessionRelayClient
    )
    sent: list[dict] = []
    peer = _peer(sent)
    deadline_ms = int((time.time() + 5.0) * 1000)
    values = np.zeros((1, 2, 3), dtype=np.float32)

    first = _payload(values, task_id="stable-session")
    first["request_deadline_ms"] = deadline_ms
    peer._handle_layer_forward_via_relay(first)
    peer._active_layer_config["relay_segment"]["timeout"] = 0.25
    second = _payload(values, task_id="stable-session")
    second.update({"step": 1, "request_deadline_ms": deadline_ms})
    peer._handle_layer_forward_via_relay(second)

    assert len(_SessionRelayClient.instances) == 1
    session = _SessionRelayClient.instances[0]
    assert session.calls == 2
    assert session.timeout == 2.0
    assert session._relay_endpoint_key == ("127.0.0.1", 50283, 3, "middle")
    assert session.deadline_monotonic is not None


def test_peer_relay_failed_outcome_discards_session(monkeypatch):
    class _FailedOutcome:
        ok = False
        error = "relay_transport_error"

    class _FailingRelayClient(_SessionRelayClient):
        def forward_hidden(self, *_args, **_kwargs):
            self.calls += 1
            return _FailedOutcome()

    _FailingRelayClient.instances.clear()
    monkeypatch.setattr(
        "relay_segment_client.RelaySegmentClient", _FailingRelayClient
    )
    sent: list[dict] = []
    peer = _peer(sent)

    peer._handle_layer_forward_via_relay(
        _payload(np.zeros((1, 2, 3), dtype=np.float32), task_id="failed")
    )

    session = _FailingRelayClient.instances[0]
    assert session.close_graces == [0.0]
    assert peer._relay_sessions == {}
    assert sent == [{"error": "relay_segment_failed:relay_transport_error"}]


@pytest.mark.parametrize("terminal_event", ["done", "abort"])
def test_peer_relay_reuses_task_session_until_terminal_event(monkeypatch, terminal_event):
    """Decode steps share one relay connection; terminal cleanup sends one CLOSE."""
    _SessionRelayClient.instances.clear()
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _SessionRelayClient)
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sent: list[dict] = []
    peer = _peer(sent)
    for step in (0, 1):
        payload = _payload(values, task_id="session-task")
        payload["step"] = step
        payload.update({
            "config_id": "cfg-relay",
            "model_sha256": "relay-sha",
            "model_type": "qwen2",
            "use_kv_cache": step > 0,
        })
        peer._handle_layer_forward(payload)

    assert len(_SessionRelayClient.instances) == 1
    session = _SessionRelayClient.instances[0]
    assert session.calls == 2
    assert session.close_calls == 0
    getattr(peer, f"_handle_pipeline_{terminal_event}")({"task_id": "session-task"})
    assert session.close_calls == 1
    assert session.close_graces == ([0.0] if terminal_event == "abort" else [None])
    assert "session-task" not in peer._relay_sessions
    assert "session-task" not in peer._active_pipeline_task_ids


def test_peer_relay_rejects_forward_after_abort_before_connect(monkeypatch):
    _SessionRelayClient.instances.clear()
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _SessionRelayClient)
    sent: list[dict] = []
    peer = _peer(sent)
    peer._local_pipeline_cancelled.add("aborted-task")
    payload = _payload(np.zeros((1, 2, 3), dtype=np.float32), task_id="aborted-task")
    payload.update({
        "config_id": "cfg-relay", "model_sha256": "relay-sha",
        "model_type": "qwen2", "use_kv_cache": False,
    })

    peer._handle_layer_forward(payload)

    assert _SessionRelayClient.instances == []
    assert sent == []


def test_peer_abort_breaks_blocking_relay_before_waiting_execution_lock(monkeypatch):
    class _StoppedOutcome:
        ok = False
        error = "relay_transport_error"

    class _BlockingRelayClient:
        instances: list["_BlockingRelayClient"] = []

        def __init__(self, *_args, **_kwargs):
            self.entered = threading.Event()
            self.released = threading.Event()
            self.close_graces: list[float | None] = []
            type(self).instances.append(self)

        def forward_hidden(self, *_args, **_kwargs):
            self.entered.set()
            assert self.released.wait(timeout=2)
            return _StoppedOutcome()

        def close(self, *, grace_timeout=None):
            self.close_graces.append(grace_timeout)
            self.released.set()

    monkeypatch.setattr(
        "relay_segment_client.RelaySegmentClient", _BlockingRelayClient
    )
    sent: list[dict] = []
    peer = _peer(sent)
    payload = _payload(
        np.zeros((1, 2, 3), dtype=np.float32), task_id="blocking-abort"
    )
    payload.update({
        "config_id": "cfg-relay",
        "model_sha256": "relay-sha",
        "model_type": "qwen2",
        "use_kv_cache": False,
    })
    worker = threading.Thread(target=peer._handle_layer_forward, args=(payload,))
    worker.start()
    wait_until = time.monotonic() + 1.0
    while not _BlockingRelayClient.instances and time.monotonic() < wait_until:
        time.sleep(0.001)
    assert _BlockingRelayClient.instances
    session = _BlockingRelayClient.instances[0]
    assert session.entered.wait(timeout=1)

    started = time.monotonic()
    peer._handle_pipeline_abort({"task_id": "blocking-abort"})
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert time.monotonic() - started < 1.0
    assert session.close_graces == [0.0]
    assert peer._relay_sessions == {}
    assert "blocking-abort" in peer._local_pipeline_cancelled


def test_peer_relay_rejects_out_of_order_step_before_connect(monkeypatch):
    _SessionRelayClient.instances.clear()
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _SessionRelayClient)
    sent: list[dict] = []
    peer = _peer(sent)
    peer._local_pipeline_steps["ordered-task"] = 0
    peer._active_pipeline_task_ids.add("ordered-task")
    payload = _payload(np.zeros((1, 2, 3), dtype=np.float32), task_id="ordered-task")
    payload.update({
        "step": 2,
        "config_id": "cfg-relay", "model_sha256": "relay-sha",
        "model_type": "qwen2", "use_kv_cache": True,
    })

    peer._handle_layer_forward(payload)

    assert _SessionRelayClient.instances == []
    assert len(sent) == 1
    assert "越序" in sent[0]["error"]


def test_peer_relay_success_records_step_and_rejects_duplicate_prefill(monkeypatch):
    _SessionRelayClient.instances.clear()
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _SessionRelayClient)
    sent: list[dict] = []
    peer = _peer(sent)
    payload = _payload(np.zeros((1, 2, 3), dtype=np.float32), task_id="duplicate-task")
    payload.update({
        "config_id": "cfg-relay", "model_sha256": "relay-sha",
        "model_type": "qwen2", "use_kv_cache": False,
    })

    peer._handle_layer_forward(payload)
    peer._handle_layer_forward(payload)

    assert len(_SessionRelayClient.instances) == 1
    assert _SessionRelayClient.instances[0].calls == 1
    assert peer._local_pipeline_steps["duplicate-task"] == 0
    assert len(sent) == 2
    assert "重复 prefill" in sent[1]["error"]


def test_peer_relay_sessions_close_on_release_and_disconnect():
    """Config release and the disconnect cleanup path must close all sessions."""
    peer = _peer([])
    peer._send_layer_config_ack = lambda _payload: True

    first = _SessionRelayClient()
    peer._relay_sessions["release-task"] = first
    peer._handle_layer_config_locked({"release": True, "node_id": "surface-worker"})
    assert first.close_calls == 1
    assert peer._relay_sessions == {}

    second = _SessionRelayClient()
    peer._relay_sessions["disconnect-task"] = second
    peer._close_all_relay_sessions()
    assert second.close_calls == 1
    assert peer._relay_sessions == {}


# ---- ★ Y-(b)：tail 段（hidden → token）与不支持角色的具名拒绝 -------------


class _TailOutcome:
    ok = True
    role = "tail"
    error = ""
    endpoint = "127.0.0.1:50284"
    hidden = b""

    def __init__(self, token: int):
        self.token = token

    def to_metrics(self):
        return {"relay_segment": "tail@127.0.0.1:50284"}


class _TailRelayClient:
    received: bytes | None = None
    called: str = ""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def forward_hidden(
        self, hidden: bytes, *, n_tokens: int, deadline_monotonic=None
    ):  # pragma: no cover
        type(self).called = "forward_hidden"
        raise AssertionError("tail 段不应走 forward_hidden")

    def forward_hidden_to_token(
        self, hidden: bytes, *, n_tokens: int, deadline_monotonic=None
    ):
        type(self).called = "forward_hidden_to_token"
        assert n_tokens == 2
        type(self).received = hidden
        return _TailOutcome(4242)


def _payload(values: np.ndarray, task_id: str = "t-tail") -> dict:
    return {
        "task_id": task_id,
        "step": 0,
        "hidden_states": base64.b64encode(values.tobytes()).decode("ascii"),
        "hidden_shape": list(values.shape),
        "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
    }


def test_peer_relay_tail_returns_token_and_never_calls_middle(monkeypatch):
    """★ Y-(b)：`tail` 段必须走 `forward_hidden_to_token`，回 **token**（不带 hidden）。"""
    _TailRelayClient.called = ""
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _TailRelayClient)
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sent: list[dict] = []
    peer = _peer(sent, role="tail")

    peer._handle_layer_forward_via_relay(_payload(values))

    assert _TailRelayClient.called == "forward_hidden_to_token"
    assert _TailRelayClient.received == values.tobytes()
    assert len(sent) == 1 and "error" not in sent[0]
    assert sent[0]["token"] == 4242
    assert "hidden_states" not in sent[0]
    assert sent[0]["metrics"]["relay_executed"] is True


def test_peer_relay_rejects_role_unsupported_on_the_client_side(monkeypatch):
    """★ `head` 在**从节点这一侧**不支持（head 段要 token 序列，而从节点收到的是 hidden）
    ⇒ 必须**具名拒绝**：绝不"当成 middle 硬算"（那会静默产出错误数值）。"""
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _TailRelayClient)
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sent: list[dict] = []
    peer = _peer(sent, role="head")

    peer._handle_layer_forward_via_relay(_payload(values, task_id="t-head"))

    assert len(sent) == 1
    assert sent[0].get("error") == "relay_role_unsupported:head"
