"""Legacy deferred abort and successor commit share the real execution boundary."""
from copy import deepcopy
import threading
from types import MethodType, SimpleNamespace

import pytest

from model_host import ModelHost
from scheduler import Scheduler
from tests.test_task_worker_hello_followup import master, hello, ObservedRLock  # noqa: F401


def assert_control_locks_free(master):
    acquired = []
    def probe():
        for lock in (master._layer_config_lock, master._tcp_server._clients_lock,
                     master._task_worker_control._lock):
            free = lock.acquire(blocking=False)
            acquired.append(free)
            if free:
                lock.release()
    thread = threading.Thread(target=probe)
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    assert acquired == [True, True, True]


@pytest.fixture
def legacy_case(master, monkeypatch):
    import model_module
    manager = model_module.ModelManager()
    manager._engine_type = "pytorch"
    manager._pipeline_descriptor = {"model_id": "legacy-model", "total_layers": 8}
    manager._pipeline_distributed_only = True
    manager._full_model_path = "virtual-legacy-model"
    manager.model = object()
    manager.tokenizer = object()
    manager.layer_range = (0, 2)
    manager.quant_type = "fp16"
    host = ModelHost(manager=manager)
    host.model_loaded = True
    master._host = host
    loads, aborts, published = [], [], []
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(master, "_persist_pipeline_lifecycle_locked", lambda: None)
    monkeypatch.setattr(master, "_publish_layer_configs", lambda configs: published.append(deepcopy(configs)))

    # Only weight/tokenizer I/O is replaced. The manager's actual serialized
    # abort and scheduler's actual opt-out/abort/commit methods remain active.
    def tokenizer():
        with manager._lock:
            manager.tokenizer = object()
    monkeypatch.setattr(manager, "prepare_pipeline_tokenizer", tokenizer)

    def load(self, start_layer, end_layer, **kwargs):
        assert_control_locks_free(master)
        assert (start_layer, end_layer) == (0, 4)
        assert kwargs["has_embedding"] is True
        assert kwargs["has_lm_head"] is False
        self.model = object()
        self.tokenizer = object()
        self.layer_range = (start_layer, end_layer)
        host.model_loaded = True
        loads.append(self.model)
    monkeypatch.setattr(manager, "load_layer_range", MethodType(
        model_module._serialized_model_access(load), manager,
    ))
    actual_abort = manager.abort_pipeline_materialization
    def abort():
        assert_control_locks_free(master)
        actual_abort()
        aborts.append(True)
    monkeypatch.setattr(manager, "abort_pipeline_materialization", abort)
    hello(master, full=True)
    old = {"config_id": "old-config", "generation": 1, "phase": "committing",
           "candidate_id": "", "worker_ids": {"worker"},
           "plan": {"model_id": "legacy-model"},
           "expected": {"worker": {"config_id": "old-config", "assignment_id": "old-assignment"}}}
    master._pipeline_load_transaction = old
    return SimpleNamespace(master=master, host=host, manager=manager, old=old,
                           work=master._task_worker_hello_followups["worker"],
                           loads=loads, aborts=aborts, published=published)


def install_successor(case):
    master = case.master
    with master._layer_config_lock:
        expected = {"node_id": "successor-worker", "config_id": "successor-config",
                    "assignment_id": "successor-assignment", "generation": 2,
                    "phase": "prepare", "start_layer": 4, "end_layer": 8}
        master._worker_assignments.begin(
            "successor-worker", config_id=expected["config_id"],
            expected_config=expected, connection_generation=12,
        )
        transaction = {"config_id": "successor-config", "generation": 2,
                       "phase": "preparing", "candidate_id": "",
                       "worker_ids": {"successor-worker"},
                       "plan": {"model_id": "legacy-model", "total_layers": 8,
                                "assignments": [{"node_id": "master", "start_layer": 0,
                                                 "end_layer": 4, "has_embedding": True,
                                                 "has_lm_head": False}]}}
        master._pipeline_load_transaction = transaction
        return transaction


def start_old_opt_out(case):
    errors = []
    def run():
        try:
            Scheduler._handle_layer_worker_opt_out(
                case.master, "worker", {"data": {"node_id": "worker"}},
                hello_work=case.work,
            )
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run, name="old-legacy-opt-out")
    thread.start()
    return thread, errors


class DispatchBoundaryLock(ObservedRLock):
    """Pause the old handler after its actual outer config/owner guard exits."""
    def __init__(self):
        super().__init__("unused-contender")
        self.depth = threading.local()
        self.exited, self.resume = threading.Event(), threading.Event()
        self.fired = False

    def __enter__(self):
        super().__enter__()
        self.depth.value = getattr(self.depth, "value", 0) + 1
        return self

    def __exit__(self, *_):
        self.depth.value -= 1
        super().__exit__()
        if (threading.current_thread().name == "old-legacy-opt-out"
                and self.depth.value == 0 and not self.fired):
            self.fired = True
            self.exited.set()
            assert self.resume.wait(5)


@pytest.mark.parametrize("window", ["before_dispatch", "execution_lock_wait"])
def test_old_deferred_abort_cannot_clear_already_materialized_successor(legacy_case, monkeypatch, window):
    case = legacy_case
    master = case.master
    thread = None
    errors = []
    execution_lock = ObservedRLock("old-legacy-opt-out")
    master._host.full_chat_execution_lock = execution_lock
    boundary = DispatchBoundaryLock()
    if window == "before_dispatch":
        monkeypatch.setattr(master, "_layer_config_lock", boundary)
    try:
        # Same thread may commit reentrantly while the old cleanup contender
        # waits for the real execution lock. No sleep decides the ordering.
        with execution_lock:
            thread, errors = start_old_opt_out(case)
            if window == "before_dispatch":
                assert boundary.exited.wait(2)
            else:
                assert execution_lock.waiting.wait(2)
            assert case.old["phase"] == "aborted"
            successor = install_successor(case)
            master._commit_pipeline_load_transaction("successor-config")
            assert successor["phase"] == "committing"
            assert len(case.loads) == 1
            model = case.manager.model
            assignments = deepcopy(master._worker_assignments.snapshot())
            boundary.resume.set()
    finally:
        boundary.resume.set()
        if thread is not None:
            thread.join(3)
    assert not thread.is_alive()
    assert errors == []
    assert case.aborts == []
    assert case.manager.model is model
    assert case.host.model_loaded is True
    assert case.manager.layer_range == (0, 4)
    assert master._pipeline_load_transaction is successor
    assert successor["phase"] == "committing"
    assert master._worker_assignments.snapshot() == assignments
    assert case.published[0]["successor-worker"]["phase"] == "commit"


def test_successor_commit_cannot_materialize_between_abort_check_and_native_cleanup(legacy_case, monkeypatch):
    case = legacy_case
    master = case.master
    execution_lock = ObservedRLock("successor-legacy-commit")
    case.host.full_chat_execution_lock = execution_lock
    native_entered, resume_native = threading.Event(), threading.Event()
    actual_abort = case.manager.abort_pipeline_materialization

    def paused_native_abort():
        native_entered.set()
        assert resume_native.wait(5)
        actual_abort()
    monkeypatch.setattr(case.manager, "abort_pipeline_materialization", paused_native_abort)
    old_thread, old_errors = start_old_opt_out(case)
    commit_thread = None
    commit_errors = []
    def commit():
        try:
            master._commit_pipeline_load_transaction("successor-config")
        except BaseException as exc:
            commit_errors.append(exc)
    try:
        assert native_entered.wait(2)
        # Old identity has passed its check; native cleanup has not run yet.
        # A new control transaction may be installed, but its load MUST wait.
        successor = install_successor(case)
        commit_thread = threading.Thread(target=commit, name="successor-legacy-commit")
        commit_thread.start()
        assert execution_lock.waiting.wait(2)
        assert case.loads == []
        assert_control_locks_free(master)
    finally:
        resume_native.set()
        old_thread.join(3)
        if commit_thread is not None:
            commit_thread.join(3)
    assert not old_thread.is_alive()
    assert not commit_thread.is_alive()
    assert old_errors == commit_errors == []
    assert case.aborts == [True]
    assert len(case.loads) == 1
    assert case.manager.model is case.loads[0]
    assert case.manager.layer_range == (0, 4)
    assert case.host.model_loaded is True
    assert master._pipeline_load_transaction is successor
    assert successor["phase"] == "committing"
    assert master._worker_assignments.state("successor-worker").assignment_id == "successor-assignment"


def test_superseded_legacy_commit_rechecks_after_execution_lock_wait(legacy_case):
    case = legacy_case
    master = case.master
    execution_lock = ObservedRLock("superseded-legacy-commit")
    case.host.full_chat_execution_lock = execution_lock
    successor = install_successor(case)
    errors = []
    def commit():
        try:
            master._commit_pipeline_load_transaction("successor-config")
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=commit, name="superseded-legacy-commit")
    with execution_lock:
        thread.start()
        assert execution_lock.waiting.wait(2)
        assert successor["phase"] == "committing_local"
        # The waiting commit must recheck the exact transaction after the wait.
        replacement = {**successor, "config_id": "newest-config", "phase": "ready"}
        master._pipeline_load_transaction = replacement
    thread.join(3)
    assert not thread.is_alive()
    assert errors == []
    assert case.loads == []
    assert case.published == []
    assert master._pipeline_load_transaction is replacement


def test_current_legacy_deferred_cleanup_still_unloads_its_own_host(legacy_case):
    case = legacy_case
    thread, errors = start_old_opt_out(case)
    thread.join(3)
    assert not thread.is_alive()
    assert errors == []
    assert case.aborts == [True]
    assert case.manager.model is None
    assert case.manager.layer_range is None
    assert case.manager.is_pipeline_prepared is True
    assert case.host.model_loaded is False
    assert case.old["phase"] == "aborted"
