"""Ticket 13: real PC TCP registration + hello recover after transport faults.

All endpoints are ephemeral loopback sockets. conftest.py isolates persistence;
model discovery and runtime-config writes are replaced before connecting.
"""
from __future__ import annotations

import logging
import os
import socket
import struct
import threading
import time

import pytest

import config
import scheduler as scheduler_module
import tcp_comm
from scheduler import Scheduler
from task_worker_adapter import TaskWorkerControlPlane
from transport_port import MessageType


def _wait_for(predicate, message):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.005)
    assert predicate(), message


@pytest.mark.parametrize("fault", ["reset", "eof", "missing_heartbeat_ack"])
def test_accepted_pc_worker_recovers_registration_hello_and_heartbeat(monkeypatch, caplog, fault):
    # [reset] 分支在 Windows 上用 caplog.records 检查真实的 ConnectionResetError，
    # 所以这里显式给 tcp_comm 挂捕获级别，不依赖 root handler 的当前状态。
    caplog.set_level(logging.WARNING, logger="tcp_comm")
    monkeypatch.setattr(config, "CLUSTER_SECRET", "ticket13-isolated-loopback-secret")
    monkeypatch.setattr(scheduler_module, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    monkeypatch.setattr(scheduler_module, "_configured_node_id", lambda: "ticket13-pc")
    monkeypatch.setattr(scheduler_module, "_sync_runtime_node_config", lambda **kwargs: None)
    monkeypatch.setattr(tcp_comm, "HEARTBEAT_INTERVAL", 0.03)
    monkeypatch.setattr(tcp_comm, "RECONNECT_DELAY", 0.01)
    monkeypatch.setattr(tcp_comm, "_CLIENT_SOCKET_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(tcp_comm, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm, "detect_lan_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(tcp_comm.TCPClient, "_compute_local_model_sha256", staticmethod(lambda: ""))

    worker = Scheduler()
    worker._role_override = "client"
    monkeypatch.setattr(worker, "get_distributed_inference_enabled", lambda: True)
    monkeypatch.setattr(worker, "_task_worker_capabilities", lambda: {
        "stage_types": ["aggregate"], "engines": ["pytorch"],
        "models": [], "max_concurrency": 1,
    })
    coordinator = TaskWorkerControlPlane()
    server = tcp_comm.TCPServer(host="127.0.0.1", port=0)
    fault_applied = threading.Event()
    dropped_heartbeats = []
    hellos = []
    errors = []
    original_heartbeat = server._handle_heartbeat
    original_server_epoch = [None]
    acknowledged_generations = []
    original_ack = tcp_comm.TCPClient._handle_heartbeat_ack

    def acknowledge(client, msg, connection_generation=None):
        original_ack(client, msg, connection_generation)
        if (connection_generation == client._connection_generation
                and client._last_heartbeat_ack_send == msg["data"].get("t_send")):
            acknowledged_generations.append(connection_generation)

    monkeypatch.setattr(tcp_comm.TCPClient, "_handle_heartbeat_ack", acknowledge)

    def heartbeat(client_id, msg):
        info = server.get_client_info(client_id)
        if (fault == "missing_heartbeat_ack" and fault_applied.is_set()
                and info and info["registration_epoch"] == original_server_epoch[0]):
            dropped_heartbeats.append(msg["data"]["t_send"])
            return
        original_heartbeat(client_id, msg)

    monkeypatch.setattr(server, "_handle_heartbeat", heartbeat)

    def receive(client_id, msg):
        try:
            if msg["type"] == MessageType.REGISTER.value:
                if not server.confirm_registration(client_id):
                    raise AssertionError("registration rejected")
            elif msg["type"] == MessageType.TASK_WORKER.value:
                ack = coordinator.receive_on_coordinator(
                    client_id, msg["data"], coordinator_node_id="master",
                )
                hellos.append(msg["data"])
                server.send_to_client(client_id, ack.snapshot(), MessageType.TASK_WORKER)
            elif msg["type"] == MessageType.STATUS_REQ.value:
                conn = server._get_client(client_id)
                if fault == "reset":
                    linger = struct.pack("hh" if os.name == "nt" else "ii", 1, 0)
                    conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
                else:
                    conn.sock.shutdown(socket.SHUT_RDWR)
                conn.sock.close()
                fault_applied.set()
        except Exception as exc:
            errors.append(exc)

    server.start(on_message=receive)
    port = server.sock.getsockname()[1]
    old_recv = old_heartbeat = None
    client = None
    try:
        assert worker.connect_to_master("127.0.0.1", port)["status"] == "connected"
        client = worker._tcp_client
        _wait_for(lambda: worker._task_worker_control.coordinator_snapshot().get("healthy"),
                  "initial PC hello was not accepted")
        _wait_for(lambda: client._connection_generation in acknowledged_generations,
                  "initial heartbeat was not acknowledged")
        old_sock = client.sock
        old_generation = client._connection_generation
        old_epoch = client.registration_epoch
        old_owner = worker._task_worker_control._worker_hello_owner
        old_recv, old_heartbeat = client._recv_thread, client._heartbeat_thread
        original_server_epoch[0] = server.get_client_info(client.client_id)["registration_epoch"]

        if fault == "missing_heartbeat_ack":
            fault_applied.set()
        else:
            client.send_data({"ticket13_fault": fault}, MessageType.STATUS_REQ)
            assert fault_applied.wait(3), errors

        def recovered():
            snapshot = worker._task_worker_control.coordinator_snapshot()
            owner = worker._task_worker_control._worker_hello_owner
            return (client.registration_epoch > old_epoch and client.is_registered
                    and snapshot.get("healthy") and owner is not None
                    and owner[1] == client.registration_epoch
                    and owner != old_owner and len(hellos) >= 2
                    and client._connection_generation in acknowledged_generations
                    and not client._reconnect_in_progress)

        _wait_for(recovered, f"{fault}: PC process survived but protocol did not recover: {errors}")
        assert worker._tcp_client is client, "recovery must retain the single TCP retry owner"
        assert client.sock is not old_sock
        assert client._connection_generation > old_generation
        assert client.reconnect_state == "ready"
        assert client.reconnect_failure_count == 0
        assert not client._reconnect_retry_scheduled
        assert not errors
        if fault == "reset" and os.name == "nt":
            # Windows：SO_LINGER(1,0)+close 稳定产生真 RST（WinError 10054）。
            # Linux 上同样的注入只在套接字上还有未读数据时才保证 RST，否则对端合法地
            # 看到干净的 FIN/EOF —— 上面的注入与恢复断言已经覆盖行为，这里只额外要求
            # Windows 那条平台特有的异常证据。
            resets = [record.exc_info[1] for record in caplog.records
                      if record.exc_info and isinstance(record.exc_info[1], ConnectionResetError)]
            assert resets, "fault injection must produce a real TCP reset"
            assert any(getattr(exc, "winerror", None) == 10054 for exc in resets)
        if fault == "missing_heartbeat_ack":
            # 注入必须真的发生：服务端丢够 ACK。
            assert len(dropped_heartbeats) >= client.MAX_HEARTBEAT_ACK_MISSED - 1
            # 这里**不能**断言某条具体日志。客户端有多个检测器（ACK 阈值、心跳发送失败、
            # 接收循环异常、disconnect 通知），哪个先发现连接已死会随平台与调度变化 ——
            # CI（Linux）实测 caplog 为空，说明它没打任何 WARNING 就完成了重连。
            # 判据由上面的注入证据与下面的自愈断言（reconnect_state=ready、
            # reconnect_failure_count=0、generation 前进、hello 再次 accepted）承担。
        for thread in (old_recv, old_heartbeat):
            thread.join(2)
            assert not thread.is_alive(), "old transport generation did not exit"
    finally:
        if client is not None:
            client.on_disconnect = None
            client.disconnect()
        worker._worker_prepare.close(wait=True)
        connections = server._client_items_snapshot()
        for _client_id, conn in connections:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        server.stop()
        for thread in (old_recv, old_heartbeat,
                       getattr(client, "_recv_thread", None),
                       getattr(client, "_heartbeat_thread", None)):
            if thread is not None:
                thread.join(2)
                assert not thread.is_alive()


def test_registration_wait_interrupted_by_concurrent_socket_close(monkeypatch, caplog):
    """注册确认等待期间本代连接被别的线程置换/关闭 ⇒ 记为「注册被中断」，不计失败。

    2026-10-16 实测（ROG，master 重启之后）：worker 每分钟都在重连，但每次都在
    「注册确认等待」处报 `WinError 10038`（在一个非套接字上尝试了一个操作），
    并被记成一次注册失败，累加到 `worker_reconnect_exhausted` 后彻底退化成
    每分钟一次的低频重试。

    真因是「等待确认」这一段既没有持有本代 socket 的局部引用，也没有做
    generation 校验：本代 socket 被别的线程（上一代传输收尾，或 disconnect()）
    置换并关闭后，`recv_exact` 就落到已关闭的 fd 上。上游同函数里注册成功后的
    发布路径本来就守这个纪律（显式传 generation + socket），只有等待段漏了。
    """
    monkeypatch.setattr(config, "CLUSTER_SECRET", "ticket13-interrupted-loopback-secret")
    monkeypatch.setattr(scheduler_module, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    monkeypatch.setattr(scheduler_module, "_configured_node_id", lambda: "ticket13-interrupted")
    monkeypatch.setattr(scheduler_module, "_sync_runtime_node_config", lambda **kwargs: None)
    monkeypatch.setattr(tcp_comm, "HEARTBEAT_INTERVAL", 0.03)
    monkeypatch.setattr(tcp_comm, "RECONNECT_DELAY", 0.01)
    # 放宽确认等待窗口，好让测试稳定地落在「正在等注册确认」的时刻。
    monkeypatch.setattr(tcp_comm, "_CLIENT_SOCKET_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(tcp_comm, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm, "detect_lan_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(tcp_comm.TCPClient, "_compute_local_model_sha256", staticmethod(lambda: ""))

    worker = Scheduler()
    worker._role_override = "client"
    monkeypatch.setattr(worker, "get_distributed_inference_enabled", lambda: True)
    monkeypatch.setattr(worker, "_task_worker_capabilities", lambda: {
        "stage_types": ["aggregate"], "engines": ["pytorch"],
        "models": [], "max_concurrency": 1,
    })
    coordinator = TaskWorkerControlPlane()
    server = tcp_comm.TCPServer(host="127.0.0.1", port=0)
    hold_ack = threading.Event()
    register_waiting = threading.Event()
    hellos = []
    errors = []

    def receive(client_id, msg):
        try:
            if msg["type"] == MessageType.REGISTER.value:
                if hold_ack.is_set():
                    register_waiting.set()  # 收到注册但故意不回 ACK
                    return
                if not server.confirm_registration(client_id):
                    raise AssertionError("registration rejected")
            elif msg["type"] == MessageType.TASK_WORKER.value:
                ack = coordinator.receive_on_coordinator(
                    client_id, msg["data"], coordinator_node_id="master",
                )
                hellos.append(msg["data"])
                server.send_to_client(client_id, ack.snapshot(), MessageType.TASK_WORKER)
        except Exception as exc:
            errors.append(exc)

    server.start(on_message=receive)
    port = server.sock.getsockname()[1]
    client = None
    try:
        assert worker.connect_to_master("127.0.0.1", port)["status"] == "connected"
        client = worker._tcp_client
        _wait_for(lambda: worker._task_worker_control.coordinator_snapshot().get("healthy"),
                  "initial PC hello was not accepted")
        assert client.reconnect_failure_count == 0

        # 让下一次注册卡在等待确认：服务端收下 REGISTER 却不回 ACK。
        hold_ack.set()
        _client_id, conn = server._client_items_snapshot()[0]
        conn.sock.shutdown(socket.SHUT_RDWR)
        conn.sock.close()

        assert register_waiting.wait(5), "client never reached the registration wait"

        # 模拟「别的线程关闭了本代连接」：disconnect() 正是递增 generation 再关闭 socket。
        # **不要**把 self.sock 置 None —— 那样 recv_data() 会直接返回 None，客户端把它
        # 当成「未收到注册确认」（真失败）而不是「本代被中断」；留在原地的已关闭描述符
        # 才会让等待中的 recv 抛出 OSError，走到被中断的分类分支。
        with client._disconnect_callback_lock:
            doomed = client.sock
            client._connection_generation += 1
        assert doomed is not None
        try:
            doomed.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        doomed.close()

        # The next registration clears last_register_error on entry; the
        # reconnect outcome persists until successful registration instead.
        _wait_for(lambda: client.last_reconnect_error == "registration_interrupted",
                  "interrupted registration was not classified: "
                  f"{client.last_register_error!r}")
        assert client.last_reconnect_error == "registration_interrupted"
        assert client.reconnect_failure_count == 0, (
            "被主动中断的注册不得计入失败，否则会被放大成 worker_reconnect_exhausted")
        assert "worker_reconnect_exhausted" not in caplog.text

        # 恢复正常的注册确认之后仍要能自愈。
        hold_ack.clear()
        _wait_for(lambda: client.is_registered
                  and worker._task_worker_control.coordinator_snapshot().get("healthy")
                  and len(hellos) >= 2,
                  f"client did not recover after the interrupted registration: {errors}")
        assert client.reconnect_failure_count == 0
        assert not errors
    finally:
        if client is not None:
            client.on_disconnect = None
            client.disconnect()
        worker._worker_prepare.close(wait=True)
        for _client_id, connection in server._client_items_snapshot():
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        server.stop()
