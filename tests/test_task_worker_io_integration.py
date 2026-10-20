"""PC worker I/O isolation over real framing and event-controlled registration races."""
from __future__ import annotations

import collections
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from scheduler_task_worker import SchedulerTaskWorkerMixin, _TaskWorkerActiveAttempt
from task_worker_adapter import TaskWorkerControlPlane
from task_worker_protocol import build_message, canonical_sha256
from task_worker_io import WorkerIOSender, WorkerIOError
from tcp_comm import TCPClient, MessageType, parse_message


class RecordingSocket:
    def __init__(self):
        self.packets = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def sendall(self, packet):
        self.entered.set()
        assert self.release.wait(5)
        if self.fail:
            raise OSError("injected write failure")
        self.packets.append(parse_message(packet[4:])["data"])


class Worker(SchedulerTaskWorkerMixin):
    def __init__(self, sock=None):
        self._task_worker_control = TaskWorkerControlPlane()
        self._task_worker_stage_lock = threading.RLock()
        self._task_worker_refresh_lock = threading.RLock()
        self._task_worker_active_attempts = {}
        self._task_worker_completed_attempts = {}
        self._task_worker_seen_messages = {}
        self._task_worker_seen_order = collections.deque()
        self._tcp_client = TCPClient(client_id="io-worker")
        self.publish(sock or RecordingSocket())
        self._worker_prepare = SimpleNamespace(invalidate=lambda *a, **kw: None, close=lambda **kw: None)

    def publish(self, sock):
        client = self._tcp_client
        with client._disconnect_callback_lock:
            client.sock = sock
            client._registered = client._running = True
            client._registration_epoch += 1
            client._connection_generation += 1
        return client.capture_registration_owner()

    def _effective_role(self):
        return "client"

    def close(self):
        sender = getattr(self, "_task_worker_io_sender", None)
        self._cleanup_task_worker_prepare("worker_prepare_stopped", close=True)
        if sender is not None:
            sender.close(wait=True)
            assert not sender.snapshot()["thread_alive"]


def payload():
    return dict(workflow_id="wf_ioborrow0001", stage_id="tail", attempt_id="att_ioborrow0001",
                lease_id="lease_ioborrow0001", lease_epoch=1, provider_id="remote_io-worker")


def response(kind="stage_result", index=0):
    fields = payload()
    if kind == "stage_accept":
        fields.update(accepted=True, reason_code="", retryable=False)
    elif kind == "stage_cancelled":
        fields.update(reason_code="provider_cancelled", execution_state="execution_stopped")
    else:
        output = {"content": str(index)}
        fields.update(output=output, output_sha256=canonical_sha256(output), metadata={})
    return build_message(kind, fields, message_id=f"msg_ioborrow{index:04d}", sent_at_ms=int(time.time()*1000))


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    monkeypatch.setenv("QLH_TASK_WORKER_IO_ISOLATION", "1")


@pytest.mark.parametrize("value,enabled", [("0", False), ("1", True), ("true", True), ("ON", True), ("", False), ("typo", False)])
def test_switch_is_explicit(monkeypatch, value, enabled):
    monkeypatch.setenv("QLH_TASK_WORKER_IO_ISOLATION", value)
    assert Worker._task_worker_io_enabled() is enabled


def test_default_path_stays_synchronous(monkeypatch):
    monkeypatch.delenv("QLH_TASK_WORKER_IO_ISOLATION")
    worker = Worker()
    worker._send_task_worker_response("request", response())
    assert worker._tcp_client.sock.packets[0]["message_type"] == "stage_result"
    assert not hasattr(worker, "_task_worker_io_sender")


def test_accept_waits_for_real_write():
    worker = Worker()
    sock = worker._tcp_client.sock
    sock.release.clear()
    done = threading.Event()
    errors = []
    def accept():
        try:
            worker._send_task_worker_response("request", response("stage_accept"))
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()
    thread = threading.Thread(target=accept)
    thread.start()
    try:
        assert sock.entered.wait(3)
        assert not done.is_set()
        sock.release.set()
        assert done.wait(3)
        assert not errors
        assert sock.packets[0]["message_type"] == "stage_accept"
    finally:
        sock.release.set()
        thread.join(3)
        worker.close()


def test_one_result_batch_keeps_order_and_counts_inflight_budget():
    worker = Worker()
    worker._prepare_task_worker_request(response())
    sock = worker._tcp_client.sock
    sock.release.clear()
    rows = [response(index=i) for i in range(3)]
    try:
        future = worker._send_task_worker_responses(rows[0].message_id, rows, wait=False)
        assert sock.entered.wait(3)
        snapshot = worker._task_worker_io_sender.snapshot()
        assert snapshot["pending_items"] == 1
        assert snapshot["pending_bytes"] >= sum(len(r.payload["output"]["content"]) for r in rows)
        assert not future.done()
        sock.release.set()
        future.result(3)
        assert [r["message_id"] for r in sock.packets] == [r.message_id for r in rows]
        assert worker._task_worker_io_sender.snapshot()["pending_items"] == 0
        assert len(worker._task_worker_seen_messages[rows[0].message_id][1]) == 3
    finally:
        sock.release.set()
        worker.close()


@pytest.mark.parametrize("change", ["socket", "epoch", "client", "registered"])
def test_queued_old_reply_never_writes_successor(change):
    worker = Worker()
    original = worker._tcp_client.sock
    blocker, started = threading.Event(), threading.Event()
    sender = worker._task_worker_io_sender = WorkerIOSender()
    first = sender.submit(lambda: (started.set(), blocker.wait(5)), charge_bytes=1, is_current=lambda: True)
    assert started.wait(3)
    future = worker._send_task_worker_responses("request", [response()], wait=False)
    try:
        if change == "socket":
            worker.publish(RecordingSocket())
        elif change == "epoch":
            worker.publish(original)
        elif change == "client":
            worker._tcp_client = TCPClient(client_id="io-successor")
            worker.publish(RecordingSocket())
        else:
            worker._tcp_client._registered = False
        blocker.set()
        first.result(3)
        with pytest.raises(WorkerIOError) as retired:
            future.result(3)
        assert retired.value.code == "worker_io_owner_retired"
        assert original.packets == []
        assert worker._tcp_client.sock.packets == []
    finally:
        blocker.set()
        worker.close()


def test_queue_full_caches_whole_batch_for_retry_without_partial_send():
    worker = Worker()
    row = response()
    worker._prepare_task_worker_request(row)
    worker._task_worker_io_sender = WorkerIOSender(max_items=1, max_bytes=1)
    try:
        with pytest.raises(WorkerIOError) as full:
            worker._send_task_worker_responses(row.message_id, [row, response(index=1)], wait=False)
        assert full.value.code == "worker_io_full"
        assert len(worker._task_worker_seen_messages[row.message_id][1]) == 2
        assert worker._tcp_client.sock.packets == []
    finally:
        worker.close()


def test_mid_batch_failure_stops_remaining_frames_and_replay_binds_new_owner():
    worker = Worker()
    row = response()
    worker._prepare_task_worker_request(row)
    worker._tcp_client.sock.fail = True
    try:
        future = worker._send_task_worker_responses(row.message_id, [row, response(index=1)], wait=False)
        with pytest.raises(ConnectionError):
            future.result(3)
        assert worker._tcp_client.sock.packets == []
        cached = worker._task_worker_seen_messages[row.message_id][1]
        assert len(cached) == 2
        worker.publish(RecordingSocket())
        worker._replay_task_worker_responses(cached)
        # A synchronous barrier runs after the whole replay batch.
        worker._send_task_worker_responses(None, [response("stage_accept", index=2)], wait=True)
        assert [r["message_id"] for r in worker._tcp_client.sock.packets] == [r.message_id for r in [row, response(index=1), response("stage_accept", index=2)]]
    finally:
        worker.close()


def test_cleanup_retires_waiting_batch_and_prevents_late_stage_entry():
    worker = Worker()
    entered, release = threading.Event(), threading.Event()
    sender = worker._task_worker_io_sender = WorkerIOSender()
    first = sender.submit(lambda: (entered.set(), release.wait(5)), charge_bytes=1, is_current=lambda: True)
    assert entered.wait(3)
    future = worker._send_task_worker_responses("request", [response()], wait=False)
    owner, generation = worker._task_worker_response_owner()
    calls = []
    worker._handle_task_worker_stage_offer = lambda msg: calls.append(msg)
    try:
        worker._cleanup_task_worker_prepare("local_inference_mode")
        worker._run_task_worker_stage_offer(response(), owner, generation)
        assert not calls
        assert future.cancelled() or future.done()
        release.set()
        first.result(3)
        sender.close(wait=True)
        assert worker._tcp_client.sock.packets == []
        # The physically stopped receipt may still go to the same live owner.
        worker._send_task_worker_response("request", response("stage_cancelled"))
        worker._send_task_worker_responses(None, [response("stage_accept")], wait=True)
        assert worker._tcp_client.sock.packets[0]["message_type"] == "stage_cancelled"
    finally:
        release.set()
        sender.close(wait=True)
        worker.close()


def test_real_receive_loop_reads_cancel_and_duplicate_while_writer_is_blocked():
    left, right = socket.socketpair()
    left.settimeout(2)
    right.settimeout(2)
    worker = Worker(left)
    master = TCPClient(client_id="io-master")
    master.sock = right
    received, first_entered, release = threading.Event(), threading.Event(), threading.Event()
    sender = worker._task_worker_io_sender = WorkerIOSender()
    block = sender.submit(lambda: (first_entered.set(), release.wait(5)), charge_bytes=1, is_current=lambda: True)
    assert first_entered.wait(3)
    fields = payload()
    active = _TaskWorkerActiveAttempt(**fields, lease_expires_at_ms=int(time.time()*1000)+5000,
                                     lease_deadline_monotonic=time.monotonic()+5)
    worker._task_worker_active_attempts[fields["attempt_id"]] = active
    calls = []
    def callback(frame):
        worker._handle_task_worker_message("master", frame)
        calls.append(frame["data"]["message_id"])
        if len(calls) == 2:
            received.set()
    worker._tcp_client.on_message = callback
    client = worker._tcp_client
    thread = threading.Thread(target=client._recv_loop, args=(client._connection_generation, left), daemon=True)
    thread.start()
    cancel = build_message("stage_cancel", {**{k:v for k,v in fields.items() if k != "provider_id"}, "reason_code": "provider_cancelled"}, message_id="msg_iocancel0001", sent_at_ms=int(time.time()*1000))
    try:
        master.send_data(cancel.snapshot(), MessageType.TASK_WORKER)
        master.send_data(cancel.snapshot(), MessageType.TASK_WORKER)
        assert received.wait(3), "second receive must complete before any reply is written"
        assert active.cancel_event.is_set()
        assert not release.is_set()
        assert sender.snapshot()["pending_items"] == 3
        assert len(worker._task_worker_seen_messages[cancel.message_id][1]) == 1
    finally:
        release.set()
        block.result(3)
        worker._tcp_client._running = False
        for stream in (left, right):
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            stream.close()
        thread.join(3)
        worker.close()
        assert not thread.is_alive()

@pytest.mark.parametrize("mode", ["0", "1"])
def test_actual_stage_waits_for_accept_write_before_executor(monkeypatch, mode):
    from scripts.comm_merge_bench import Loopback, binary_switch
    monkeypatch.setenv("QLH_TASK_WORKER_IO_ISOLATION", mode)
    entered, release, computed = threading.Event(), threading.Event(), threading.Event()
    with binary_switch(True):
        link = Loopback(binary=True, compute_delay_ms=0, master_accept_delay_ms=0, timeout=5)
        send, compute = link.worker_tcp.send_data, link.compute
        def slow_accept(data, kind, **kwargs):
            if data.get("message_type") == "stage_accept":
                entered.set()
                assert release.wait(4)
            return send(data, kind, **kwargs)
        def execute(request, cancel):
            computed.set()
            return compute(request, cancel)
        link.worker_tcp.send_data, link.compute = slow_accept, execute
        rows, errors = [], []
        def run():
            try:
                rows.append(link.sample(0, b"\x00\x00\x80\x3f" * 16))
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            assert entered.wait(3)
            assert not computed.is_set()
            release.set()
            thread.join(4)
            assert not thread.is_alive() and not errors
            assert computed.is_set() and rows[0]["output"] == {"token_argmax": 42}
        finally:
            release.set()
            thread.join(4)
            sender = getattr(link.worker, "_task_worker_io_sender", None)
            link.close()
            if sender is not None:
                sender.close(wait=True)
                assert not sender.snapshot()["thread_alive"]


@pytest.mark.parametrize("mode", ["0", "1"])
def test_result_write_holds_active_stage_only_on_legacy_path(monkeypatch, mode):
    from scripts.comm_merge_bench import Loopback, binary_switch
    monkeypatch.setenv("QLH_TASK_WORKER_IO_ISOLATION", mode)
    entered, release = threading.Event(), threading.Event()
    with binary_switch(True):
        link = Loopback(binary=True, compute_delay_ms=0, master_accept_delay_ms=0, timeout=5)
        send = link.worker_tcp.send_data
        def slow_result(data, kind, **kwargs):
            if data.get("message_type") == "stage_result":
                entered.set()
                assert release.wait(4)
            return send(data, kind, **kwargs)
        link.worker_tcp.send_data = slow_result
        rows, errors = [], []
        def run():
            try:
                rows.append(link.sample(0, b"\x00\x00\x80\x3f" * 16))
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            assert entered.wait(3)
            if mode == "1":
                assert link.stage_done.wait(3)
                assert not link.worker._task_worker_active_attempts
            else:
                assert not link.stage_done.is_set()
                assert link.worker._task_worker_active_attempts
            # Both paths already release model locks before reply transport.
            for lock in (link.worker._host.full_chat_execution_lock, link.worker._inference_lock):
                assert lock.acquire(blocking=False)
                lock.release()
            release.set()
            thread.join(4)
            assert not thread.is_alive() and not errors
            assert rows[0]["output"] == {"token_argmax": 42}
        finally:
            release.set()
            thread.join(4)
            sender = getattr(link.worker, "_task_worker_io_sender", None)
            link.close()
            if sender is not None:
                sender.close(wait=True)
                assert not sender.snapshot()["thread_alive"]

@pytest.mark.parametrize("retire", [False, True])
def test_result_publication_waits_for_physical_completion_or_retirement(retire):
    worker = Worker()
    waiting, completed = threading.Event(), threading.Event()
    class Ready:
        def wait(self, timeout):
            waiting.set()
            return completed.wait(timeout)
    future = worker._send_task_worker_responses("request", [response()], wait=False, ready_event=Ready())
    sender = worker._task_worker_io_sender
    try:
        assert waiting.wait(3)
        assert worker._tcp_client.sock.packets == []
        assert sender.snapshot()["pending_items"] == 1
        if retire:
            worker._cleanup_task_worker_prepare("local_inference_mode")
            with pytest.raises(ConnectionError, match="retired"):
                future.result(3)
            assert worker._tcp_client.sock.packets == []
        else:
            completed.set()
            future.result(3)
            assert len(worker._tcp_client.sock.packets) == 1
    finally:
        completed.set()
        sender.close(wait=True)
        worker.close()


def test_pending_accept_is_not_replayable_until_write_completes():
    worker = Worker()
    row = response("stage_accept")
    worker._prepare_task_worker_request(row)
    sock = worker._tcp_client.sock
    sock.release.clear()
    done, errors = threading.Event(), []
    def send():
        try:
            worker._send_task_worker_response(row.message_id, row)
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()
    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    try:
        assert sock.entered.wait(3)
        duplicate, cached = worker._prepare_task_worker_request(row)
        assert duplicate and cached == []
        assert not done.is_set()
        sock.release.set()
        assert done.wait(3) and not errors
        assert worker._prepare_task_worker_request(row)[1] == [row.snapshot()]
    finally:
        sock.release.set()
        thread.join(3)
        worker.close()


def test_retired_stage_keeps_terminal_cache_without_promoting_generation():
    worker = Worker()
    row = response()
    worker._prepare_task_worker_request(row)
    owner, generation = worker._task_worker_response_owner()
    results = []
    def finish(_message):
        worker._cleanup_task_worker_prepare("local_inference_mode")
        try:
            worker._send_task_worker_responses(row.message_id, [row], wait=False)
        except ConnectionError:
            results.append("retired")
    worker._handle_task_worker_stage_offer = finish
    try:
        worker._run_task_worker_stage_offer(row, owner, generation)
        assert results == ["retired"]
        assert worker._task_worker_seen_messages[row.message_id][1] == [row.snapshot()]
        assert worker._tcp_client.sock.packets == []
        worker._send_task_worker_response(row.message_id, response("stage_cancelled"))
        worker._send_task_worker_responses(None, [response("stage_accept")], wait=True)
        assert worker._tcp_client.sock.packets[0]["message_type"] == "stage_cancelled"
    finally:
        worker.close()


def test_result_batch_cache_publication_is_atomic_to_snapshot_observers():
    worker = Worker()
    request = response()
    worker._prepare_task_worker_request(request)
    observed = []
    class Observed:
        def __init__(self, message):
            self.message = message
        def __getattr__(self, name):
            return getattr(self.message, name)
        def snapshot(self):
            observed.append(len(worker._task_worker_seen_messages[request.message_id][1]))
            return self.message.snapshot()
    rows = [Observed(response(index=i)) for i in range(3)]
    try:
        future = worker._send_task_worker_responses(request.message_id, rows, wait=False)
        future.result(3)
        assert set(observed) <= {0, 3}
        assert len(worker._task_worker_seen_messages[request.message_id][1]) == 3
    finally:
        worker.close()


def test_cached_result_replay_waits_for_active_attempt_physical_completion():
    worker = Worker()
    fields = payload()
    active = _TaskWorkerActiveAttempt(**fields, lease_expires_at_ms=int(time.time()*1000)+5000,
                                     lease_deadline_monotonic=time.monotonic()+5)
    completed, waiting = threading.Event(), threading.Event()
    class Ready:
        def wait(self, timeout):
            waiting.set()
            return completed.wait(timeout)
        def set(self):
            completed.set()
        def is_set(self):
            return completed.is_set()
    active.done_event = Ready()
    worker._task_worker_active_attempts[fields["attempt_id"]] = active
    row = response()
    worker._prepare_task_worker_request(row)
    worker._cache_task_worker_response(row.message_id, row)
    try:
        worker._replay_task_worker_responses([row.snapshot()])
        assert waiting.wait(3), "cached result must enter the physical completion gate"
        # A second direct result shares the same gate, making a FIFO barrier
        # observable only after the first cached replay has been published.
        future = worker._send_task_worker_responses(None, [response(index=1)], wait=False)
        assert not future.done()
        assert worker._tcp_client.sock.packets == []
        with worker._task_worker_stage_lock:
            worker._task_worker_active_attempts.pop(fields["attempt_id"])
            active.done_event.set()
        future.result(3)
        assert len(worker._tcp_client.sock.packets) == 2
    finally:
        active.done_event.set()
        worker.close()


def test_scheduler_start_reopens_worker_io_admission_after_stop(monkeypatch):
    from scheduler import Scheduler
    scheduler = Scheduler()
    scheduler._task_worker_io_stopped = True
    observed = []
    monkeypatch.setattr(scheduler._worker_prepare, "reopen", lambda: None)
    monkeypatch.setattr(scheduler, "_start_impl", lambda **kw: observed.append(scheduler._task_worker_io_stopped))
    try:
        scheduler.start()
        assert observed == [False]
    finally:
        scheduler._running = False
