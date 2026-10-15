"""Event-controlled regressions using normally imported production owners."""
import threading
from types import SimpleNamespace

import pytest

from scheduler_task_worker import SchedulerTaskWorkerMixin
from task_worker_adapter import TaskWorkerControlPlane
from tcp_comm import TCPClient, parse_message


class RecordingSocket:
    def __init__(self):
        self.closed = False
        self.packets = []

    def sendall(self, packet):
        if self.closed:
            raise OSError("socket retired")
        self.packets.append(parse_message(packet[4:])["data"])

    def shutdown(self, *_):
        self.closed = True

    def close(self):
        self.closed = True


def publish_registration(client, sock=None):
    sock = sock or RecordingSocket()
    with client._disconnect_callback_lock:
        client.sock = sock
        client._registration_epoch += 1
        client._connection_generation += 1
        client._registered = client._running = True
        client._disconnect_notified = False
    return sock


class Worker(SchedulerTaskWorkerMixin):
    def __init__(self):
        self._task_worker_control = TaskWorkerControlPlane()
        self._task_worker_refresh_lock = threading.Lock()
        self._task_worker_refresh_generation = 0
        self._task_worker_refresh_requested = False
        self._tcp_client = TCPClient(client_id="hello-owner-worker")
        publish_registration(self._tcp_client)

    def get_distributed_inference_enabled(self):
        return True

    def _effective_role(self):
        return "client"

    def _scheduler_facade_global(self, name):
        assert name == "TASK_WORKER_EXPERIMENTAL_ENABLED"
        return True

    def get_effective_node_id(self):
        return "hello-owner-worker"

    def _task_worker_capabilities(self):
        return {"stage_types": ["aggregate"], "engines": ["pytorch"],
                "models": [], "max_concurrency": 1}


def accept_hello(worker):
    coordinator = TaskWorkerControlPlane()
    hello = worker._tcp_client.sock.packets[-1]
    ack = coordinator.receive_on_coordinator(
        "hello-owner-worker", hello, coordinator_node_id="master",
    )
    worker._task_worker_control.receive_on_worker(ack.snapshot())
    assert worker._task_worker_control.coordinator_snapshot()["healthy"]


def start_sender(worker):
    result = []
    thread = threading.Thread(
        target=lambda: result.append(worker._send_task_worker_hello(worker._tcp_client)),
        name="old-hello", daemon=True,
    )
    thread.start()
    return thread, result


@pytest.mark.parametrize("change", ["epoch", "client", "disconnected"])
def test_change_after_capabilities_before_begin_cannot_publish(monkeypatch, change):
    worker = Worker()
    old_client, old_sock = worker._tcp_client, worker._tcp_client.sock
    entered, release = threading.Event(), threading.Event()

    def node_id():
        if threading.current_thread().name == "old-hello":
            entered.set()
            assert release.wait(5)
        return "hello-owner-worker"

    monkeypatch.setattr(worker, "get_effective_node_id", node_id)
    thread, result = start_sender(worker)
    try:
        assert entered.wait(5)
        old_client.disconnect()  # Must not wait for the capability/node-id work.
        if change == "epoch":
            publish_registration(old_client)
        elif change == "client":
            replacement = TCPClient(client_id="hello-owner-worker")
            publish_registration(replacement)
            with worker._task_worker_refresh_lock:
                worker._tcp_client = replacement
        if change != "disconnected":
            assert worker._send_task_worker_hello()
            accept_hello(worker)
            new_owner = worker._task_worker_control._worker_hello_owner
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert result == [False]
    assert old_sock.packets == []
    if change != "disconnected":
        assert len(worker._tcp_client.sock.packets) == 1
        assert worker._task_worker_control._worker_hello_owner == new_owner
        assert worker._task_worker_control.coordinator_snapshot()["healthy"]
    else:
        assert worker._task_worker_control._worker_hello_pending is False


@pytest.mark.parametrize("replacement", [False, True])
def test_change_after_begin_binds_socket_and_preserves_new_ack(monkeypatch, replacement):
    worker = Worker()
    old_client, old_sock = worker._tcp_client, worker._tcp_client.sock
    entered, release = threading.Event(), threading.Event()
    send = old_client.send_data

    def paused_send(*args, **kwargs):
        if threading.current_thread().name == "old-hello":
            assert kwargs["connection_sock"] is old_sock
            entered.set()
            assert release.wait(5)
        return send(*args, **kwargs)

    monkeypatch.setattr(old_client, "send_data", paused_send)
    thread, result = start_sender(worker)
    try:
        assert entered.wait(5)
        old_client.disconnect()
        current = TCPClient(client_id="hello-owner-worker") if replacement else old_client
        publish_registration(current)
        with worker._task_worker_refresh_lock:
            worker._tcp_client = current
        # No unconditional coordinator reset: the new owner replaces old pending.
        assert worker._send_task_worker_hello()
        accept_hello(worker)
        new_owner = worker._task_worker_control._worker_hello_owner
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert result == [False]
    assert old_sock.packets == []
    assert len(current.sock.packets) == 1
    assert worker._task_worker_control._worker_hello_owner == new_owner
    assert worker._task_worker_control.coordinator_snapshot()["healthy"]


@pytest.mark.parametrize("ack_before_cleanup", [False, True])
def test_blocked_old_send_failure_cannot_clear_successor(monkeypatch, ack_before_cleanup):
    worker = Worker()
    client = worker._tcp_client
    send_entered, send_release = threading.Event(), threading.Event()
    cleanup_entered, cleanup_release = threading.Event(), threading.Event()
    old_sock = client.sock

    def blocked_send(_):
        send_entered.set()
        assert send_release.wait(5)
        raise OSError("old socket closed")

    monkeypatch.setattr(old_sock, "sendall", blocked_send)
    disconnect = worker._task_worker_control.disconnect_coordinator
    cleanups = []

    def paused_cleanup(*, owner=None):
        if threading.current_thread().name == "old-hello":
            cleanup_entered.set()
            assert cleanup_release.wait(5)
        cleared = disconnect(owner=owner)
        cleanups.append(cleared)
        return cleared

    monkeypatch.setattr(worker._task_worker_control, "disconnect_coordinator", paused_cleanup)
    thread, result = start_sender(worker)
    try:
        assert send_entered.wait(5)
        # The sender holds _send_lock, but no registration/reconnect lock.
        client.disconnect()
        publish_registration(client)
        send_release.set()
        assert cleanup_entered.wait(5)
        assert worker._send_task_worker_hello()
        new_owner = worker._task_worker_control._worker_hello_owner
        if ack_before_cleanup:
            accept_hello(worker)
    finally:
        send_release.set()
        cleanup_release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert result == [False]
    assert cleanups == [False]
    assert worker._task_worker_control._worker_hello_owner == new_owner
    if not ack_before_cleanup:
        assert worker._task_worker_control._worker_hello_pending
        accept_hello(worker)
    assert worker._task_worker_control.coordinator_snapshot()["healthy"]
    assert client.is_registered


def test_capability_failure_without_hello_cannot_clear_new_owner(monkeypatch, caplog):
    worker = Worker()
    entered, release = threading.Event(), threading.Event()
    capabilities = worker._task_worker_capabilities

    def inspect():
        if threading.current_thread().name == "old-hello":
            entered.set()
            assert release.wait(5)
            raise OSError("artifact disappeared")
        return capabilities()

    monkeypatch.setattr(worker, "_task_worker_capabilities", inspect)
    thread, result = start_sender(worker)
    try:
        assert entered.wait(5)
        worker._tcp_client.disconnect()
        publish_registration(worker._tcp_client)
        assert worker._send_task_worker_hello()
        accept_hello(worker)
        new_owner = worker._task_worker_control._worker_hello_owner
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert result == [False]
    assert worker._task_worker_control._worker_hello_owner == new_owner
    assert worker._task_worker_control.coordinator_snapshot()["healthy"]
    assert "event=task_worker_hello_capabilities_failed" in caplog.text
    assert "coordinator_cleared=False" in caplog.text


def test_delayed_refresh_keeps_registration_captured_when_scheduled(monkeypatch):
    import scheduler_task_worker as worker_module
    worker = Worker()
    queued = []

    class DeferredThread:
        def __init__(self, *, target, args, **kwargs):
            self.run = lambda: target(*args)

        def start(self):
            queued.append(self.run)

    monkeypatch.setattr(worker_module, "threading", SimpleNamespace(Thread=DeferredThread))
    assert worker.refresh_task_worker_capabilities()
    worker._tcp_client.disconnect()
    publish_registration(worker._tcp_client)
    assert worker._send_task_worker_hello()
    accept_hello(worker)
    owner = worker._task_worker_control._worker_hello_owner
    assert queued[0]() is False
    assert len(worker._tcp_client.sock.packets) == 1
    assert worker._task_worker_control._worker_hello_owner == owner
    assert worker._task_worker_control.coordinator_snapshot()["healthy"]


@pytest.mark.parametrize("accepted", [False, True])
def test_old_hello_id_cleanup_cannot_retire_same_registration_refresh(accepted):
    worker = Worker()
    assert worker._send_task_worker_hello()
    old_owner = worker._task_worker_control._worker_hello_owner
    accept_hello(worker)
    assert worker._send_task_worker_hello()
    owner = worker._task_worker_control._worker_hello_owner
    assert owner[:3] == old_owner[:3]
    assert owner[3] != old_owner[3]
    if accepted:
        accept_hello(worker)
    assert worker._task_worker_control.disconnect_coordinator(owner=old_owner) is False
    assert worker._task_worker_control._worker_hello_owner == owner
    assert worker._task_worker_control._worker_hello_pending is (not accepted)
    if accepted:
        assert worker._task_worker_control.coordinator_snapshot()["healthy"]


def test_current_send_failure_cleans_only_its_owner(monkeypatch, caplog):
    worker = Worker()

    def fail(_):
        raise OSError("current socket failed")

    monkeypatch.setattr(worker._tcp_client.sock, "sendall", fail)
    assert worker._send_task_worker_hello() is False
    assert worker._task_worker_control._worker_hello_owner is None
    assert worker._task_worker_control._worker_hello_pending is False
    assert not worker._task_worker_control.coordinator_snapshot()["connected"]
    assert "event=task_worker_hello_send_failed" in caplog.text
    assert "coordinator_cleared=True" in caplog.text
