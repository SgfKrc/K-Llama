"""★ 2026-10-08（真机复测根因）：master 必须回应 task worker 心跳。

Android task worker 的 `SocketTaskWorkerTransport` 设了 `soTimeout = 45_000`
（`android/.../TaskWorkerClient.kt:472`/`:480`），并且每 15 秒发一次 `heartbeat`。
此前 master 在 `heartbeat` 分支里**只**记录（`mark_worker_heartbeat` + 刷新
`last_heartbeat`/rtt）、**从不回包** ⇒ 连接在 45 秒后必然 `SocketTimeoutException` ⇒
断开重连。真机表现正是「Y700 每 ~37 秒客户端主动断开一次」
（`tcp_comm: 客户端 android-21af7c52 已断开`），落在断开窗口里的请求报
`pipeline_layer_range_coverage_insufficient` ⇒ "分布式时好时坏"。

PC 从节点那条路径早在 `tcp_comm.py:1711` 回 `HEARTBEAT_ACK`；本用例把 task worker 这条
补上，并锁定「回包类型 + 回显 t_send」这两件事。
"""
import os
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tcp_comm as tcp_comm_mod  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from tcp_comm import TCPClient  # noqa: E402
from transport_port import MessageType  # noqa: E402


def _scheduler_with_fake_server():
    calls = []

    def _send_to_client(client_id, data, msg_type):
        calls.append((client_id, data, msg_type))

    fake_server = SimpleNamespace(
        send_to_client=_send_to_client,
        get_client_info=lambda client_id: {},
    )
    node = SimpleNamespace(last_heartbeat=0.0, avg_rtt_ms=0.0, last_rtt_ms=0.0)
    host = SimpleNamespace(
        _effective_role=lambda: "master",
        _task_worker_control=SimpleNamespace(mark_worker_heartbeat=lambda cid: None),
        _tcp_server=fake_server,
        nodes={"android-21af7c52": node},
        _nodes_lock=threading.RLock(),
        _tcp_server_default=None,
    )
    return host, calls


def test_task_worker_heartbeat_is_acknowledged_with_echo():
    host, calls = _scheduler_with_fake_server()

    Scheduler._on_tcp_message(
        host,
        "android-21af7c52",
        {"type": "heartbeat", "data": {"client_id": "android-21af7c52", "t_send": 12345}},
    )

    assert calls, "master 未回心跳应答（Android 侧 soTimeout=45s ⇒ 周期性断开重连）"
    client_id, data, msg_type = calls[0]
    assert client_id == "android-21af7c52"
    assert data == {"t_send": 12345}
    assert msg_type is MessageType.HEARTBEAT_ACK
    assert msg_type.value == "heartbeat_ack"


def test_heartbeat_ack_failure_does_not_break_handling():
    """应答发送失败（例如对端刚断开）不得影响主流程 —— 只记 debug 日志。"""
    host, _ = _scheduler_with_fake_server()

    def _boom(*_args, **_kwargs):
        raise OSError("socket closed")

    host._tcp_server = SimpleNamespace(
        send_to_client=_boom,
        get_client_info=lambda client_id: {},
    )

    Scheduler._on_tcp_message(
        host, "android-21af7c52", {"type": "heartbeat", "data": {"t_send": 1}},
    )
    # 没有异常抛出即通过；last_heartbeat 仍应被刷新（走的是同一条分支）。
    assert host.nodes["android-21af7c52"].last_heartbeat > 0


def test_client_missing_heartbeat_acks_disconnects_once_and_fences_late_ack(
    monkeypatch,
):
    """A half-open socket cannot stay registered merely because send succeeds."""
    client = TCPClient(
        server_host="127.0.0.1", server_port=1, client_id="worker-ack-watch",
    )
    original_sock = object()
    replacement_sock = object()
    client.sock = original_sock
    client._running = True
    client._registered = True
    client._connection_generation = 1
    client._heartbeat_quality.begin_generation(1)

    sent = []
    disconnects = []
    reconnects = []
    clock = [10.0]

    def fake_time():
        value = clock[0]
        clock[0] += 1.0
        return value

    def send_data(data, message_type, connection_sock=None):
        assert connection_sock is original_sock
        sent.append((data, message_type))

    def reconnect():
        reconnects.append(client._connection_generation)
        with client._disconnect_callback_lock:
            client._connection_generation = 2
            client.sock = replacement_sock
            client._registered = True
            client._disconnect_notified = False
            client._last_heartbeat_send = 100.0
            client._last_heartbeat_ack_send = 0.0
            client._heartbeat_ack_missed = 1
            client._heartbeat_quality.begin_generation(2)
            client._heartbeat_quality.record_send(2, 100.0)

    client.on_disconnect = lambda: disconnects.append("lost")
    monkeypatch.setattr(client, "send_data", send_data)
    monkeypatch.setattr(client, "_reconnect", reconnect)
    monkeypatch.setattr(tcp_comm_mod.time, "time", fake_time)
    monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _seconds: None)

    client._heartbeat_loop(1, original_sock)

    assert len(sent) == client.MAX_HEARTBEAT_ACK_MISSED
    assert disconnects == ["lost"]
    assert reconnects == [1]
    assert client._connection_generation == 2
    assert client.sock is replacement_sock
    before = client.get_network_quality_snapshot()
    assert before["generation"] == 2
    assert before["pending_heartbeat"] is True
    assert client._heartbeat_ack_missed == 1

    # A delayed ACK from generation 1 is only a stale liveness observation;
    # it cannot clear generation 2's pending heartbeat or revive old metrics.
    clock[0] = 100.1
    client._handle_heartbeat_ack(
        {"data": {"t_send": 10.0}},
        connection_generation=1,
    )

    after = client.get_network_quality_snapshot()
    assert after["generation"] == 2
    assert after["pending_heartbeat"] is True
    assert after["sample_count"] == 0
    assert client._heartbeat_ack_missed == 1
    assert client.avg_rtt_ms == 0.0


def test_stale_watchdog_does_not_reconnect_after_disconnect_callback_replaces_generation(
    monkeypatch,
):
    client = TCPClient(
        server_host="127.0.0.1", server_port=1, client_id="worker-stale-watch",
    )
    original_sock = object()
    replacement_sock = object()
    client.sock = original_sock
    client._running = True
    client._registered = True
    client._connection_generation = 1
    client._heartbeat_quality.begin_generation(1)
    client._last_heartbeat_send = 10.0
    client._last_heartbeat_ack_send = 0.0
    client._heartbeat_ack_missed = client.MAX_HEARTBEAT_ACK_MISSED - 1
    reconnects = []

    def replace_generation():
        with client._disconnect_callback_lock:
            client._connection_generation = 2
            client.sock = replacement_sock
            client._registered = True
            client._disconnect_notified = False
            client._heartbeat_quality.begin_generation(2)

    client.on_disconnect = replace_generation
    monkeypatch.setattr(
        client, "_reconnect", lambda: reconnects.append("unexpected"),
    )
    monkeypatch.setattr(
        client,
        "send_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("timed-out generation must not send another heartbeat")
        ),
    )

    client._heartbeat_loop(1, original_sock)

    assert reconnects == []
    assert client._connection_generation == 2
    assert client.sock is replacement_sock
    assert client.is_registered is True
