"""Real stage-lock contention must not starve TCP HELLO/heartbeat processing."""
from contextlib import contextmanager
from copy import deepcopy
import threading
import time
from types import SimpleNamespace

import pytest

from scheduler import NodeInfo, NodeRole, NodeState, Scheduler
from task_worker_protocol import PROTOCOL_VERSION, build_message
from tcp_comm import ClientConn, TCPServer
from tests.test_task_worker_hello_owner import RecordingSocket


@pytest.fixture
def master(monkeypatch):
    sched = Scheduler()
    sched._role_override = "master"
    sched._running = True
    sched._distributed_inference_enabled = True
    server = TCPServer()
    server._running = True
    sched._tcp_server = server
    sched.nodes["worker"] = NodeInfo(
        node_id="worker", role=NodeRole.CLIENT, node_type="pc", state=NodeState.ONLINE,
    )
    server.clients["worker"] = ClientConn(
        "worker", RecordingSocket(), ("127.0.0.1", 1),
        registration_confirmed=True, registration_epoch=11,
    )
    sched._task_worker_control.mark_worker_connection_pending(
        "worker", connection_generation=11,
    )
    monkeypatch.setattr(sched, "get_effective_node_id", lambda: "master")
    monkeypatch.setattr(sched, "_is_relay_host", lambda _: False)
    # Drive the real retry scan explicitly; only the lifecycle test starts it.
    monkeypatch.setattr(sched, "_start_layer_config_retry_monitor", lambda: None)
    sched.effects = []
    monkeypatch.setattr(sched, "push_layer_config_to_clients",
                        lambda: sched.effects.append("push"))
    monkeypatch.setattr(sched, "request_authoritative_layer_sync",
                        lambda **_: sched.effects.append("recover"))
    monkeypatch.setattr(sched, "_handle_layer_worker_opt_out",
                        lambda *_, **_kwargs: sched.effects.append("opt_out"))
    yield sched
    sched._startup_cancel_event.set()
    sched._running = server._running = False
    thread = sched._layer_config_retry_thread
    if thread is not None:
        thread.join(2)
        assert not thread.is_alive()


def hello(sched, *, message_id="msg_hello_first", full=False, layer=False, node_id="worker"):
    capabilities = {"stage_types": ["aggregate"], "engines": ["pytorch"],
                    "models": [], "max_concurrency": 1}
    if full:
        capabilities["models"] = [{"model_id": "test-model", "revision": "1",
                                   "sha256": "a" * 64, "engine": "pytorch",
                                   "format": "safetensors"}]
    if layer:
        capabilities["stage_types"] = ["layer_forward"]
        capabilities["layer_ranges"] = [[4, 8]]
    message = build_message("hello", {
        "node_id": node_id, "worker_kind": "pc_full_worker",
        "min_version": PROTOCOL_VERSION, "max_version": PROTOCOL_VERSION,
        "capabilities": capabilities,
    }, message_id=message_id, sent_at_ms=int(time.time() * 1000))
    sched._handle_task_worker_message(node_id, {"data": message.snapshot()})


def replace(sched, *, accepted=False):
    server = sched._tcp_server
    server.clients["worker"] = ClientConn(
        "worker", RecordingSocket(), ("127.0.0.1", 2),
        registration_confirmed=True, registration_epoch=12,
    )
    sched._task_worker_control.mark_worker_connection_pending(
        "worker", connection_generation=12,
    )
    if accepted:
        hello(sched, message_id="msg_hello_replacement")


@contextmanager
def gc_admission(sched):
    entered, release = threading.Event(), threading.Event()
    failures = []

    def gc_owner():
        try:
            # Exactly the admission boundary held by production collect_garbage;
            # no files are collected, deleted, or mocked in this regression.
            with sched._model_supply_worker._admission():
                entered.set()
                if not release.wait(5):
                    failures.append("GC release was not signalled")
        except Exception as exc:
            failures.append(repr(exc))
            entered.set()

    thread = threading.Thread(target=gc_owner, name="test-supply-gc")
    thread.start()
    try:
        assert entered.wait(2)
        assert not failures
        yield
    finally:
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert not failures


def test_gc_stage_lock_does_not_delay_ack_fence_or_heartbeat(master):
    server = master._tcp_server
    with gc_admission(master):
        returned = threading.Event()
        receiver = threading.Thread(target=lambda: (hello(master), returned.set()))
        receiver.start()
        try:
            assert returned.wait(0.5), "HELLO callback waited behind supply GC"
            packets = server.clients["worker"].sock.packets
            assert packets[0]["message_type"] == "hello_ack"
            assert packets[0]["payload"]["accepted"] is True
            assert "worker" not in master._task_worker_control.pending_worker_ids()
            # The same receive path can now service the following heartbeat.
            server._handle_heartbeat("worker", {"data": {"t_send": 123.0}})
            assert packets[-1] == {"t_send": 123.0}
            master._retry_pending_layer_configs()
            assert master._remote_task_worker_providers == {}
            assert master.effects == []
        finally:
            receiver.join(0.5)
    master._retry_pending_layer_configs()
    assert "worker" in master._remote_task_worker_providers
    assert master.effects == ["push"]


@pytest.mark.parametrize("successor_hello", [False, True])
def test_only_current_registration_is_processed_after_gc(master, successor_hello):
    with gc_admission(master):
        hello(master, full=True)
        master._retry_pending_layer_configs()
        replace(master, accepted=successor_hello)
        master._retry_pending_layer_configs()
        assert master.effects == []
    master._retry_pending_layer_configs()
    assert "opt_out" not in master.effects
    assert bool(master._remote_task_worker_providers) is successor_hello
    assert master.effects == (["push"] if successor_hello else [])
    assert master._task_worker_hello_followups == {}


def test_repeated_hellos_coalesce_without_losing_changed_capabilities(master):
    with gc_admission(master):
        for index in range(100):
            hello(master, message_id=f"msg_hello_repeat_{index}")
        assert len(master._task_worker_hello_followups) == 1
        assert master._layer_config_retry_thread is None
        assert len(master._tcp_server.clients["worker"].sock.packets) == 100
    master._retry_pending_layer_configs()
    assert master.effects == ["push"]
    assert len(master._remote_task_worker_providers) == 1
    assert master._task_worker_hello_followups == {}


@pytest.mark.parametrize("change", ["epoch", "server", "stop", "local", "role"])
def test_queued_hello_is_discarded_after_owner_or_lifecycle_change(master, change):
    with gc_admission(master):
        hello(master, full=True)
        if change == "epoch":
            replace(master)
        elif change == "server":
            master._tcp_server = SimpleNamespace(_running=True)
        elif change == "stop":
            master._startup_cancel_event.set()
        elif change == "local":
            master._distributed_inference_enabled = False
        else:
            master._role_override = "client"
        master._drain_task_worker_hello_followups()
    master._drain_task_worker_hello_followups()
    assert master._remote_task_worker_providers == {}
    assert master.effects == []
    assert master._task_worker_hello_followups == {}


def test_change_during_publish_cannot_apply_old_full_worker_opt_out(master, monkeypatch):
    hello(master, full=True)
    def publish():
        master.effects.append("push")
        replace(master)
    monkeypatch.setattr(master, "push_layer_config_to_clients", publish)
    master._retry_pending_layer_configs()
    assert master.effects == ["push"]
    assert "worker" in master._task_worker_control.pending_worker_ids()


def test_shared_retry_monitor_is_single_and_exits_with_scheduler(master, monkeypatch):
    monkeypatch.setattr(master, "_start_layer_config_retry_monitor",
                        Scheduler._start_layer_config_retry_monitor.__get__(master))
    with gc_admission(master):
        hello(master)
        monitor = master._layer_config_retry_thread
        assert monitor is not None and monitor.is_alive()
        for index in range(30):
            hello(master, message_id=f"msg_hello_bounded_{index}")
            assert master._layer_config_retry_thread is monitor
        assert len(master._task_worker_hello_followups) == 1
        master._startup_cancel_event.set()
        master._running = False
        monitor.join(2)
        assert not monitor.is_alive(), "retry monitor remained blocked on stage lock"
        assert master._remote_task_worker_providers == {}
    master._cleanup_task_worker_prepare("worker_prepare_stopped", close=True)
    assert master._task_worker_hello_followups == {}


@pytest.fixture
def hello_clock(monkeypatch):
    import task_worker_adapter
    epoch = [time.time()]
    monkeypatch.setattr(task_worker_adapter, "time", SimpleNamespace(
        time=lambda: epoch[0], monotonic=time.monotonic, perf_counter=time.perf_counter,
    ))
    def advance():
        epoch[0] += 1
    return advance


@pytest.mark.parametrize("interleave", ["drain", "peer_queue", "apply_return"])
def test_snapshot_update_before_queue_preserves_unapplied_changed(master, monkeypatch, hello_clock, interleave):
    hello(master)
    original = master._task_worker_hello_followups["worker"]
    assert original[3] is True
    paused, resume = threading.Event(), threading.Event()
    failures = []
    queue = master._queue_task_worker_hello_followup

    def paused_queue(node, server, generation):
        if threading.current_thread().name == "same-owner-refresh":
            paused.set()
            assert resume.wait(5)
        return queue(node, server, generation)

    monkeypatch.setattr(master, "_queue_task_worker_hello_followup", paused_queue)

    def refresh():
        try:
            hello_clock()
            hello(master, message_id="msg_same_owner_refresh")
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=refresh, name="same-owner-refresh")
    if interleave == "apply_return":
        # The snapshot can also change after drain's entry check and before an
        # early-returning apply. Its CAS removal must retain changed intent.
        apply = master._apply_task_worker_hello_followup
        def interleaved_apply(node, work):
            thread.start()
            assert paused.wait(2)
            return apply(node, work)
        monkeypatch.setattr(master, "_apply_task_worker_hello_followup", interleaved_apply)
        master._drain_task_worker_hello_followups()
        monkeypatch.setattr(master, "_apply_task_worker_hello_followup", apply)
    else:
        thread.start()
    try:
        assert paused.wait(2)
        snapshot = master._task_worker_control.worker_snapshot("worker")
        assert snapshot["capabilities_changed"] is False
        assert snapshot["hello_received_at"] != original[2]
        assert len(master._tcp_server.clients["worker"].sock.packets) == 2
        if interleave == "drain":
            master._drain_task_worker_hello_followups()
        elif interleave == "peer_queue":
            master.nodes["peer"] = NodeInfo(
                node_id="peer", role=NodeRole.CLIENT, node_type="pc", state=NodeState.ONLINE,
            )
            master._tcp_server._set_client("peer", ClientConn(
                "peer", RecordingSocket(), ("127.0.0.1", 3),
                registration_confirmed=True, registration_epoch=21,
            ))
            master._task_worker_control.mark_worker_connection_pending(
                "peer", connection_generation=21,
            )
            hello(master, node_id="peer", message_id="msg_peer_during_refresh")
        assert master._task_worker_hello_followups["worker"] is original
        assert master.effects == []
    finally:
        resume.set()
        thread.join(2)
    assert not thread.is_alive()
    assert failures == []
    assert master._task_worker_hello_followups["worker"][3] is True
    master._drain_task_worker_hello_followups()
    assert master.effects == (["push", "push"] if interleave == "peer_queue" else ["push"])
    assert master._task_worker_hello_followups == {}


class ObservedRLock:
    """A real RLock with an event at one chosen contender's acquire edge."""

    def __init__(self, thread_name):
        self.lock = threading.RLock()
        self.thread_name = thread_name
        self.waiting = threading.Event()

    def acquire(self, *args, **kwargs):
        if threading.current_thread().name == self.thread_name:
            self.waiting.set()
        return self.lock.acquire(*args, **kwargs)

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


def ready_successor(master, *, generation=12):
    """Install a ready control fixture through the actual assignment owner."""
    with master._layer_config_lock:
        expected = {"node_id": "worker", "config_id": "successor-config",
                    "assignment_id": "successor-assignment", "generation": 42}
        master._worker_assignments.begin(
            "worker", config_id=expected["config_id"], connection_generation=generation,
            expected_config=expected,
        )
        master._worker_assignments.transition("worker", phase="ready")
        transaction = {"phase": "ready", "config_id": expected["config_id"],
                       "worker_ids": {"worker"}, "expected": {"worker": expected}}
        master._pipeline_load_transaction = transaction
        master._active_pipeline_capacity_plan = {"config_id": expected["config_id"]}
        master._layer_config_retry_state["worker"] = {"attempts": 2}
    return transaction


@pytest.mark.parametrize("change", ["registration", "same_owner_hello", "same_owner_same_tick"])
def test_opt_out_rechecks_owner_after_waiting_for_config_lock(master, monkeypatch, hello_clock, change):
    hello(master, full=True)
    work = master._task_worker_hello_followups["worker"]
    transaction = ready_successor(master, generation=11)
    config_lock = ObservedRLock("old-opt-out")
    monkeypatch.setattr(master, "_layer_config_lock", config_lock)
    real_opt_out = Scheduler._handle_layer_worker_opt_out.__get__(master)
    monkeypatch.setattr(master, "_handle_layer_worker_opt_out", real_opt_out)
    entered = threading.Event()
    errors = []

    def opt_out_at_boundary(node, msg, *, hello_work):
        assert hello_work is work
        entered.set()
        return real_opt_out(node, msg, hello_work=hello_work)

    monkeypatch.setattr(master, "_handle_layer_worker_opt_out", opt_out_at_boundary)
    # Hold the lock only when apply reaches the real opt-out helper; its earlier
    # cleanup lookup also uses the config lock and must not be the paused edge.
    pending_cleanup = master._pending_pipeline_model_cleanup_for_node
    monkeypatch.setattr(master, "_pending_pipeline_model_cleanup_for_node", lambda _: [])
    monkeypatch.setattr(master, "push_layer_config_to_clients", lambda: None)

    def old_followup():
        try:
            master._apply_task_worker_hello_followup("worker", work)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=old_followup, name="old-opt-out")
    with config_lock:
        thread.start()
        try:
            assert entered.wait(2)
            assert config_lock.waiting.wait(2)
            assert master._task_worker_hello_followup_current("worker", work)
            if change == "registration":
                replace(master)
            if change != "same_owner_same_tick":
                hello_clock()
            hello(master, message_id="msg_successor_layer", layer=True)
            transaction = ready_successor(master, generation=12 if change == "registration" else 11)
            assignment = deepcopy(master._worker_assignments.snapshot())
            plan = deepcopy(master._active_pipeline_capacity_plan)
            retries = deepcopy(master._layer_config_retry_state)
            assert master._task_worker_hello_followup_current("worker", work) is (change == "same_owner_same_tick")
        finally:
            # The context releases the actual config lock before joining.
            pass
    thread.join(2)
    assert not thread.is_alive()
    assert errors == []
    assert "worker" not in master._pipeline_worker_opt_out
    assert master._pipeline_load_transaction is transaction
    assert transaction["phase"] == "ready"
    assert master._worker_assignments.snapshot() == assignment
    assert master._active_pipeline_capacity_plan == plan
    assert master._layer_config_retry_state == retries
    assert master._pipeline_model_cleanup_ledger == {}
    assert pending_cleanup("worker") == []


@pytest.mark.parametrize("phase", ["preparing", "ready"])
def test_opt_out_identity_and_state_changes_are_serialized_before_dispatch(master, monkeypatch, phase):
    hello(master, full=True)
    work = master._task_worker_hello_followups["worker"]
    transaction = ready_successor(master, generation=11)
    transaction.update(phase=phase, candidate_id="old-candidate", expected={})
    entered, resume = threading.Event(), threading.Event()
    errors, dispatched = [], []

    class PausedOptOutSet(set):
        def add(self, node):
            entered.set()
            assert resume.wait(5)
            super().add(node)

    monkeypatch.setattr(master, "_pipeline_worker_opt_out", PausedOptOutSet())
    monkeypatch.setattr(master, "_persist_pipeline_lifecycle_locked", lambda: None)
    # No host or wire work: observe actual cleanup dispatch after state mutation.
    def dispatch(record):
        acquired = []
        def verify_locks_are_free():
            for lock in (master._layer_config_lock, master._tcp_server._clients_lock,
                         master._task_worker_control._lock):
                free = lock.acquire(blocking=False)
                acquired.append(free)
                if free:
                    lock.release()
        probe = threading.Thread(target=verify_locks_are_free)
        probe.start()
        probe.join(2)
        assert not probe.is_alive()
        assert acquired == [True, True, True]
        assert transaction["phase"] == ("aborted" if phase == "preparing" else "invalidated")
        assert master._worker_assignments.state("worker").terminal
        dispatched.append(record)
        return True

    monkeypatch.setattr(master, "_dispatch_pipeline_model_cleanup", dispatch)
    # Ready retirement with expected={} owes no native cleanup; observe its
    # deferred callback via a real native identity only for the ready branch.
    if phase == "ready":
        from tests.test_worker_prepare_coordinator import identity
        transaction["expected"] = {"worker": {"worker_prepare_identity": identity()}}

    def opt_out():
        try:
            Scheduler._handle_layer_worker_opt_out(
                master, "worker", {"data": {"node_id": "worker"}}, hello_work=work,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=opt_out, name="serialized-opt-out")
    thread.start()
    try:
        assert entered.wait(2)
        # These are the exact existing ownership locks used by TCP REGISTER and
        # control HELLO, tested from a different thread than the guarded change.
        for lock in (master._tcp_server._clients_lock, master._task_worker_control._lock):
            acquired = lock.acquire(blocking=False)
            if acquired:
                lock.release()
            assert acquired is False
    finally:
        resume.set()
        thread.join(2)
    assert not thread.is_alive()
    assert errors == []
    assert dispatched
    assert "worker" in master._pipeline_worker_opt_out
