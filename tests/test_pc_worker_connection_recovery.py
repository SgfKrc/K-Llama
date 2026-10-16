"""Ticket 13: real PC TCP registration + hello recover after transport faults.

All endpoints are ephemeral loopback sockets. conftest.py isolates persistence;
model discovery and runtime-config writes are replaced before connecting.
"""
from __future__ import annotations

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
        if fault == "reset":
            resets = [record.exc_info[1] for record in caplog.records
                      if record.exc_info and isinstance(record.exc_info[1], ConnectionResetError)]
            assert resets, "fault injection must produce a real TCP reset"
            if os.name == "nt":
                assert any(getattr(exc, "winerror", None) == 10054 for exc in resets)
        if fault == "missing_heartbeat_ack":
            assert "heartbeat ACK timed out" in caplog.text
            assert len(dropped_heartbeats) >= client.MAX_HEARTBEAT_ACK_MISSED - 1
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
