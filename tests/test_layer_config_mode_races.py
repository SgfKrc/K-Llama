"""Deterministic mode/publication and lifecycle lock races; no model I/O."""

from types import SimpleNamespace
import threading
from threading import Thread as RealThread

import pytest

import scheduler_pipeline
from scheduler_pipeline import SchedulerPipelineMixin
from task_worker_adapter import TaskWorkerControlPlane


WAIT_SECONDS = 5


class ObservedRLock:
    """Expose an acquisition attempt before blocking on a real reentrant lock."""

    def __init__(self):
        self.lock = threading.RLock()
        self.attempted = threading.Event()

    def __enter__(self):
        self.attempted.set()
        self.lock.acquire()
        return self

    def __exit__(self, *_exc):
        self.lock.release()


class AdmissionGate:
    """Pause the enqueue thread just after its outer mode check."""

    def __init__(self):
        self.lock = threading.Lock()
        self.arrived = threading.Event()
        self.resume = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "enqueue-race":
            self.arrived.set()
            assert self.resume.wait(WAIT_SECONDS), "enqueue gate was not released"
        self.lock.acquire()
        return self

    def __exit__(self, *_exc):
        self.lock.release()


class LayerHost:
    def __init__(self):
        self.full_chat_execution_lock = ObservedRLock()
        self._engine_type = "pytorch"
        self.is_loaded = True
        self.is_pipeline_prepared = False
        self.model_loaded = True
        self.layer_range = None
        self.quant_type = "fp16"
        self.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2"))
        self.runtime = "local-full"
        self.commits = []

    def commit_pipeline_candidate(self, transaction_id):
        # Preserve the real ModelHost contract: commit reenters full-chat.
        with self.full_chat_execution_lock:
            self.commits.append(transaction_id)
            self.runtime = "layer"
            self.layer_range = (4, 8)
            self.is_pipeline_prepared = True


class Worker(SchedulerPipelineMixin):
    def __init__(self):
        self._host = LayerHost()
        self._task_worker_control = TaskWorkerControlPlane()
        self._distributed_inference_enabled = True
        self._layer_config_lock = threading.Lock()
        self._layer_execution_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._pipeline_worker_opted_out = False
        self._pipeline_worker_reserved = False
        self._active_layer_config = None
        self._last_layer_config_ack_payload = None
        self._layer_config_inflight = set()
        self._layer_config_receive_sequence = 0
        self._latest_layer_config_receive_sequence = 0
        self._latest_layer_config_generation = 0
        self._latest_layer_assignment_id = ""
        self._active_pipeline_task_ids = set()
        self._pending_layer_config = None
        self._local_pipeline_steps = {}
        self._prepared_layer_configs = {}
        self._pipeline_assignment_rollbacks = {}
        self._tcp_client = None
        self.acks = []

    def get_distributed_inference_enabled(self):
        return self._distributed_inference_enabled

    def _effective_role(self):
        return "client"

    def get_effective_node_id(self):
        return "worker"

    def _send_layer_config_ack(self, payload):
        self.acks.append(dict(payload))

    def exit_to_local(self):
        # The mode API holds full-chat through opt-out and model replacement.
        with self._host.full_chat_execution_lock:
            self._distributed_inference_enabled = False
            self.release_pipeline_worker_for_local_model()
            self._host.runtime = "local-full"


def assignment(**changes):
    return {
        "node_id": "worker", "config_id": "cfg-authoritative",
        "assignment_id": "assignment-1", "generation": 1,
        "authoritative_sync": True, "phase": "commit",
        "start_layer": 4, "end_layer": 8, "total_layers": 12,
        "model_id": "test-model", "model_sha256": "test-sha",
        "model_type": "qwen2", "engine": "pytorch",
        **changes,
    }


def start_checked(target, *, name):
    errors = []

    def run():
        try:
            target()
        except BaseException as exc:
            errors.append(exc)

    thread = RealThread(target=run, name=name, daemon=True)
    thread.start()
    return thread, errors


@pytest.fixture
def deferred_loads(monkeypatch):
    queued = []

    class DeferredThread:
        def __init__(self, *, target, name, daemon):
            assert name.startswith("layer-config-")
            self.target = target

        def start(self):
            queued.append(self.target)

    monkeypatch.setattr(scheduler_pipeline.threading, "Thread", DeferredThread)
    return queued


def test_authoritative_admission_rechecks_mode_after_outer_check(deferred_loads):
    worker = Worker()
    gate = AdmissionGate()
    worker._layer_config_lock = gate
    thread, errors = start_checked(
        lambda: worker._schedule_layer_config("master", assignment()),
        name="enqueue-race",
    )
    try:
        assert gate.arrived.wait(WAIT_SECONDS), "enqueue never reached its state lock"
        worker.exit_to_local()
    finally:
        gate.resume.set()
        thread.join(WAIT_SECONDS)
    assert not thread.is_alive()
    assert errors == []
    assert deferred_loads == []
    assert worker._pipeline_worker_opted_out
    assert not worker._pipeline_worker_reserved
    assert worker._layer_config_inflight == set()
    assert worker._layer_config_receive_sequence == 0
    assert worker.acks == []
    assert worker._host.runtime == "local-full"


@pytest.mark.parametrize("phase", ["prepare", "commit"])
def test_queued_authoritative_apply_rechecks_mode(deferred_loads, monkeypatch, phase):
    worker = Worker()
    applied = []
    monkeypatch.setattr(worker, "_handle_layer_config_locked", lambda *_a, **_kw: applied.append(phase))
    worker._schedule_layer_config("master", assignment(phase=phase))
    assert len(deferred_loads) == 1
    assert worker._pipeline_worker_reserved

    # Leave sequence unchanged: the mode guard must work independently of
    # TCP disconnect's sequence invalidation, including direct/embedded use.
    worker.exit_to_local()
    deferred_loads.pop()()
    assert applied == []
    assert worker._pipeline_worker_opted_out
    assert not worker._pipeline_worker_reserved
    assert worker._layer_config_inflight == set()
    assert worker._active_layer_config is None
    assert worker.acks == []
    assert worker._host.runtime == "local-full"


@pytest.mark.parametrize("chat_runtime", [True, False])
def test_local_mode_still_applies_release_cleanup(chat_runtime):
    worker = Worker()
    worker.exit_to_local()
    if not chat_runtime:
        del worker._host.full_chat_execution_lock
    worker._pipeline_worker_reserved = True
    worker._active_layer_config = {"config_id": "old"}
    worker._handle_layer_config("master", assignment(release=True))
    assert not worker._pipeline_worker_reserved
    assert worker._active_layer_config is None
    assert worker.acks[-1]["status"] == "released"
    assert worker._host.runtime == "local-full"
    assert worker._host.commits == []


def test_authoritative_admission_can_rejoin_temporary_opt_out(deferred_loads, monkeypatch):
    worker = Worker()
    worker._pipeline_worker_opted_out = True
    applied = []
    monkeypatch.setattr(worker, "_handle_layer_config_locked", lambda *_a, **_kw: applied.append("layer"))
    worker._schedule_layer_config("master", assignment())
    assert not worker._pipeline_worker_opted_out
    assert worker._pipeline_worker_reserved
    assert len(deferred_loads) == 1
    deferred_loads.pop()()
    assert applied == ["layer"]
    assert worker._layer_config_inflight == set()


def test_legacy_commit_waits_for_full_chat_without_holding_layer_lock():
    worker = Worker()
    payload = assignment(plan_id="plan-1")
    worker._prepared_layer_configs[payload["config_id"]] = {
        "plan_id": "plan-1", "assignment_id": payload["assignment_id"],
        "layer_range": [4, 8], "model_sha256": "test-sha",
        "model_path": "unused-model-path",
    }
    full_chat = worker._host.full_chat_execution_lock
    thread = None
    try:
        with full_chat:
            full_chat.attempted.clear()
            thread, errors = start_checked(
                lambda: worker._handle_layer_config("master", payload),
                name="legacy-commit",
            )
            assert full_chat.attempted.wait(WAIT_SECONDS), "commit did not try full-chat"
            # A model change/chat already owning full-chat must be able to take
            # layer-execution. The old layer -> full order makes this fail.
            layer_available = worker._layer_execution_lock.acquire(blocking=False)
            if layer_available:
                worker._layer_execution_lock.release()
            assert worker._host.runtime == "local-full"
    finally:
        if thread is not None:
            thread.join(WAIT_SECONDS)
    assert not thread.is_alive()
    assert errors == []
    assert layer_available, "legacy commit held layer-execution while waiting for full-chat"
    assert worker._host.commits == [payload["config_id"]]
    assert worker._host.runtime == "layer"
    assert worker.acks[-1]["status"] == "ready"
    assert worker._active_layer_config["layer_range"] == [4, 8]


def test_apply_waiting_for_full_chat_rechecks_local_mode():
    worker = Worker()
    payload = assignment(plan_id="plan-1")
    worker._prepared_layer_configs[payload["config_id"]] = {
        "plan_id": "plan-1", "assignment_id": payload["assignment_id"],
        "layer_range": [4, 8], "model_sha256": "test-sha",
        "model_path": "unused-model-path",
    }
    full_chat = worker._host.full_chat_execution_lock
    thread = None
    try:
        with full_chat:
            full_chat.attempted.clear()
            thread, errors = start_checked(
                lambda: worker._handle_layer_config("master", payload),
                name="legacy-apply-race",
            )
            assert full_chat.attempted.wait(WAIT_SECONDS), "apply did not try full-chat"
            # The apply has already started. The mode owner wins the lifecycle
            # lock and opts out before letting that apply continue.
            worker.exit_to_local()
    finally:
        if thread is not None:
            thread.join(WAIT_SECONDS)
    assert not thread.is_alive()
    assert errors == []
    assert worker._host.commits == []
    assert worker._host.runtime == "local-full"
    assert worker._pipeline_worker_opted_out
    assert not worker._pipeline_worker_reserved
    assert worker._active_layer_config is None
    assert worker.acks == []


@pytest.fixture
def pipeline_master(monkeypatch):
    from model_host import ModelHost, _PipelineCandidateTransaction
    from worker_assignment_state import WorkerAssignmentRegistry

    def manager(runtime, prepared):
        result = SimpleNamespace(
            runtime=runtime, is_loaded=True, is_pipeline_prepared=prepared,
            engine_type="pytorch", quant_type="fp16", closed=0,
        )

        def unload():
            result.closed += 1
            result.is_loaded = False

        result.unload_model = unload
        return result

    master = Worker()
    active = manager("local-full", False)
    candidate = manager("layer", True)
    master._host = ModelHost(manager=active)
    master._host.model_loaded = True
    master._host.full_chat_execution_lock = ObservedRLock()
    candidate_txn = _PipelineCandidateTransaction(
        transaction_id="candidate-pending", model_id="test-model",
        model_path="unused-model-path", quant_type="fp16", layer_range=(0, 4),
        model_sha256=None, manager=candidate, descriptor={}, phase="materialized",
    )
    object.__setattr__(master._host, "_pipeline_candidate_transaction", candidate_txn)
    master._pipeline_load_transaction = {
        "config_id": "cfg-pending", "candidate_id": "candidate-pending",
        "phase": "committing_global", "generation": 1,
        "plan": {"model_id": "test-model", "plan_id": "plan-pending"},
        "worker_ids": {"worker"},
        "expected": {"worker": {
            "assignment_id": "assignment-pending", "config_id": "cfg-pending",
        }},
        "acks": {}, "retire_configs": {},
    }
    master._layer_config_generation = 1
    master._pipeline_model_cleanup_ledger = {}
    master._layer_config_retry_state = {}
    master._worker_assignments = WorkerAssignmentRegistry()
    master._active_pipeline_capacity_plan = None
    events = []
    master._tcp_server = SimpleNamespace(
        _running=True,
        send_layer_config=lambda node_id, payload: events.append(
            ("send", node_id, dict(payload))
        ),
    )
    master._pipeline_load_transaction["on_commit"] = lambda: events.append(("callback",))
    monkeypatch.setattr(master, "_effective_role", lambda: "master")
    monkeypatch.setattr(master, "_persist_pipeline_lifecycle_locked", lambda: None)
    monkeypatch.setattr(master, "_maybe_finish_pipeline_recovery", lambda: None)
    monkeypatch.setattr(
        master, "_commit_ready_pipeline_reshard",
        lambda plan: events.append(("reshard", dict(plan))) or True,
        raising=False,
    )
    return master, active, candidate, candidate_txn, events


@pytest.mark.parametrize("isolated_candidate", [True, False])
def test_master_commit_waiting_for_full_chat_rejects_local_mode(
    pipeline_master, isolated_candidate,
):
    master, active, candidate, candidate_txn, events = pipeline_master
    if not isolated_candidate:
        master._pipeline_load_transaction.pop("candidate_id")
        # A legacy abort would unload the current runtime. Rejection must only
        # invalidate its control state after the user has selected local mode.
        active.abort_pipeline_materialization = active.unload_model
    full_chat = master._host.full_chat_execution_lock
    thread = None
    try:
        with full_chat:
            full_chat.attempted.clear()
            thread, errors = start_checked(
                lambda: master._complete_pipeline_load_transaction("cfg-pending"),
                name="master-publication-mode-race",
            )
            assert full_chat.attempted.wait(WAIT_SECONDS), "commit did not try full-chat"
            # Keep the transaction intact to test the mode fence independently
            # of API invalidation, which may already have removed the snapshot.
            master.exit_to_local()
    finally:
        if thread is not None:
            thread.join(WAIT_SECONDS)
    assert not thread.is_alive()
    assert errors == []
    transaction = master._pipeline_load_transaction
    assert transaction["phase"] == ("aborted" if isolated_candidate else "invalidated")
    assert transaction["reason_code"] == "distributed_inference_disabled"
    assert master._host.peek_manager() is active
    assert master._host.runtime == "local-full"
    assert active.closed == 0
    assert master._host.model_loaded
    assert master._active_pipeline_capacity_plan is None
    assert master._worker_assignments.state("worker") is None
    assert not any(event[0] in {"reshard", "callback"} for event in events)
    if isolated_candidate:
        assert candidate_txn.phase == "aborted"
        assert candidate.closed == 1
        assert len(events) == 1
        assert events[0][0:2] == ("send", "worker")
        assert events[0][2]["cleanup_action"] == "abort"
        assert events[0][2]["aborted_config_id"] == "cfg-pending"
        assert all(
            record["node_id"] == "worker" and record["action"] == "abort"
            for record in master._pipeline_model_cleanup_ledger.values()
        )
    else:
        assert events == []
        assert candidate.closed == 0


@pytest.mark.parametrize("clear_transaction", [False, True])
def test_master_commit_waiting_for_full_chat_discards_invalidated_snapshot(
    pipeline_master, monkeypatch, clear_transaction,
):
    master, active, candidate, candidate_txn, events = pipeline_master
    # Keep disposal pending: publication must be fenced by control state even
    # if the isolated candidate has not yet been physically discarded.
    monkeypatch.setattr(master, "_dispatch_pipeline_model_cleanup", lambda _record: False)
    full_chat = master._host.full_chat_execution_lock
    thread = None
    try:
        with full_chat:
            full_chat.attempted.clear()
            thread, errors = start_checked(
                lambda: master._complete_pipeline_load_transaction("cfg-pending"),
                name="master-publication-invalidation-race",
            )
            assert full_chat.attempted.wait(WAIT_SECONDS), "commit did not try full-chat"
            master._invalidate_pipeline_load_transaction()
            if clear_transaction:
                with master._layer_config_lock:
                    master._pipeline_load_transaction = None
            # Enabled mode models a subsequent rejoin; the old snapshot must
            # still be rejected before commit, reshard, or the success callback.
            assert master.get_distributed_inference_enabled()
    finally:
        if thread is not None:
            thread.join(WAIT_SECONDS)
    assert not thread.is_alive()
    assert errors == []
    if clear_transaction:
        assert master._pipeline_load_transaction is None
    else:
        assert master._pipeline_load_transaction["phase"] == "invalidated"
    assert candidate_txn.phase == "materialized"
    assert master._host.peek_manager() is active
    assert active.closed == candidate.closed == 0
    assert master._active_pipeline_capacity_plan is None
    assert master._worker_assignments.state("worker") is None
    assert events == []


def test_master_commit_still_publishes_when_distributed_enabled(pipeline_master):
    master, active, candidate, candidate_txn, events = pipeline_master
    master._complete_pipeline_load_transaction("cfg-pending")
    assert master._pipeline_load_transaction["phase"] == "ready"
    assert candidate_txn.phase == "finalized"
    assert master._host.peek_manager() is candidate
    assert active.closed == 1
    assert candidate.closed == 0
    assert master._worker_assignments.state("worker").phase == "ready"
    assert [event[0] for event in events] == ["reshard", "callback", "send"]
    assert events[-1][2]["cleanup_action"] == "finalize"


@pytest.mark.parametrize("operation", ["commit", "abort_materialized", "abort_committed"])
def test_model_host_waits_for_full_chat_without_holding_candidate_lock(
    pipeline_master, operation,
):
    master, active, candidate, candidate_txn, _events = pipeline_master
    host = master._host
    if operation == "abort_committed":
        host.commit_pipeline_candidate("candidate-pending")
    method = (
        host.commit_pipeline_candidate if operation == "commit"
        else host.abort_pipeline_candidate
    )
    full_chat = host.full_chat_execution_lock
    candidate_lock = object.__getattribute__(host, "_pipeline_candidate_lock")
    thread = None
    try:
        with full_chat:
            full_chat.attempted.clear()
            thread, errors = start_checked(
                lambda: method("candidate-pending"), name="model-host-lock-order",
            )
            assert full_chat.attempted.wait(WAIT_SECONDS), "operation did not try full-chat"
            # An API unload holding full-chat can acquire candidate immediately;
            # a direct operation waiting for full-chat must not already own it.
            candidate_available = candidate_lock.acquire(blocking=False)
            if candidate_available:
                candidate_lock.release()
    finally:
        if thread is not None:
            thread.join(WAIT_SECONDS)
    assert not thread.is_alive()
    assert errors == []
    assert candidate_available, "candidate lock was held while waiting for full-chat"
    if operation == "commit":
        assert candidate_txn.phase == "committed"
        assert host.peek_manager() is candidate
        assert active.closed == candidate.closed == 0
    else:
        assert candidate_txn.phase == "aborted"
        assert host.peek_manager() is active
        assert candidate.closed == 1
        assert active.closed == 0
