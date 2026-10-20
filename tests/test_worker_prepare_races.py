"""Deterministic regressions for WORKER-PREPARE master races and recovery.

Real transaction, ModelHost and cleanup code runs against the small managers in
stage_case. Only transport/storage I/O and the named scheduling boundaries are
substituted; no native runtime, sleeps or background retry threads are needed.
"""
from copy import deepcopy
import json
from threading import Event

import pytest

from tests.test_pipeline_retirement import retirement_case, _Manager  # noqa: F401
from tests.test_worker_prepare_coordinator import (
    ack,
    prepare_both,
    publish,
    stage_case,  # noqa: F401 - fixture dependency of race_case
)


@pytest.fixture
def race_case(stage_case, monkeypatch):
    import local_store
    from scheduler_types import NodeInfo, NodeRole, NodeState

    case = stage_case
    # Exercise the real lifecycle persistence path without a shared SQLite DB.
    store = {}
    monkeypatch.setattr(
        local_store, "get_local_setting",
        lambda key, default=None: deepcopy(store.get(key, default)),
    )
    monkeypatch.setattr(
        local_store, "set_local_setting",
        lambda key, value: store.__setitem__(key, deepcopy(value)),
    )
    monkeypatch.setattr(case.master, "_running", True)
    case.lifecycle_store = store
    for node, platform in (("worker", "pc"), ("android", "android")):
        case.master.nodes[node] = NodeInfo(
            node_id=node, role=NodeRole.CLIENT, node_type=platform,
            state=NodeState.ONLINE,
        )
    # Disconnect still runs the real retirement/registry/deregister paths. These
    # two adapters prevent unrelated node broadcasts and automatic replanning.
    monkeypatch.setattr(case.master, "_push_node_update_to_all_clients", lambda *a, **k: None)
    monkeypatch.setattr(case.master, "_stage_pipeline_reshard_after_disconnect", lambda _node: None)
    return case


def _assignments(master):
    return deepcopy(master._pipeline_load_transaction["expected"])


def _receipt(assignment, phase="commit", **changes):
    result = {
        "action": "ack", "phase": phase,
        "identity": deepcopy(assignment["worker_prepare_identity"]),
        "status": "prepared" if phase == "prepare" else "ready", "loaded": True,
    }
    result.update(changes)
    return result


def _replay_ready(master, assignments):
    for node, assignment in assignments.items():
        master._handle_worker_prepare_ack(node, _receipt(assignment))


def _assert_unpublished(master, assignments):
    transaction = master._pipeline_load_transaction
    assert not transaction or transaction.get("phase") != "ready"
    assert master._active_pipeline_capacity_plan is None
    for node, assignment in assignments.items():
        loaded, reason = master._worker_prepare_assignment_loaded(node, assignment)
        assert loaded is False, (node, reason)


def _assert_registry_not_ready(master, nodes):
    with master._layer_config_lock:
        for node in nodes:
            state = master._worker_assignments.state(node)
            assert master._worker_assignments.is_ready(node) is False, (
                node, None if state is None else state.phase,
            )


def _assert_debts(master, assignments, action):
    records = [record for record in master._pipeline_model_cleanup_ledger.values()
               if record["node_id"] != "master" and record["action"] == action]
    assert len(records) == len(assignments), [(r["node_id"], r["action"]) for r in records]
    assert {r["node_id"] for r in records} == set(assignments)
    for record in records:
        identity = assignments[record["node_id"]]["worker_prepare_identity"]
        assert record["payload"]["worker_prepare_identity"] == identity
        assert record["candidate_config_id"] == identity["config_id"]
        assert record["assignment_id"] == identity["assignment_id"]
    return {record["cleanup_id"] for record in records}


def _finish_finalizes(master):
    for record in list(master._pipeline_model_cleanup_ledger.values()):
        assert record["action"] == "finalize"
        if record["node_id"] == "master":
            master._dispatch_pipeline_model_cleanup(record)
            continue
        master._handle_worker_prepare_ack(record["node_id"], {
            "action": "ack", "phase": "finalize", "status": "finalized", "loaded": True,
            "identity": deepcopy(record["payload"]["worker_prepare_identity"]),
            "cleanup_id": record["cleanup_id"],
        })
    assert master._pipeline_model_cleanup_ledger == {}


class _AfterUnlock:
    """Schedule one cancellation at the materialization/check unlock edge."""

    def __init__(self, lock, should_fire, callback):
        self.lock = lock
        self.should_fire = should_fire
        self.callback = callback
        self.fired = False

    def acquire(self, *args, **kwargs):
        return self.lock.acquire(*args, **kwargs)

    def release(self):
        self.lock.release()
        if not self.fired and self.should_fire():
            self.fired = True  # Cleanup reenters this lock; never fire twice.
            self.callback()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()


@pytest.mark.parametrize("boundary", ["local_commit", "reshard_cas"])
def test_reconnect_during_publication_cannot_rebind_old_loaded_ack(race_case, monkeypatch, boundary):
    case = race_case
    master = case.master
    epochs = {"worker": 11, "android": 11}
    monkeypatch.setattr(master, "_assignment_connection_generation", lambda node: epochs[node])
    case.start()
    prepare_both(case)
    assignments = _assignments(master)
    reached = []

    def replace_connection():
        reached.append(boundary)
        assert master._active_pipeline_capacity_plan is None
        epochs["worker"] = 12

    if boundary == "local_commit":
        # ModelHost forwards instance setattr to its manager: patch the class.
        original_commit = type(case.host).commit_pipeline_candidate

        def commit_then_reconnect(host, candidate_id):
            result = original_commit(host, candidate_id)
            replace_connection()
            return result

        monkeypatch.setattr(type(case.host), "commit_pipeline_candidate", commit_then_reconnect)
    else:
        def cas_then_reconnect(_plan):
            replace_connection()
            return True

        monkeypatch.setattr(master, "_commit_ready_pipeline_reshard", cas_then_reconnect)

    ack(master, "worker", "commit")
    ack(master, "android", "commit")

    assert reached == [boundary]
    _assert_unpublished(master, assignments)
    assert master._pipeline_load_transaction["phase"] == "aborted"
    assert case.host.pipeline_candidate_status()["phase"] == "aborted"
    assert case.host.peek_manager() is case.active
    _assert_debts(master, assignments, "abort")
    # A delayed success from connection 11 must not readmit connection 12.
    _replay_ready(master, assignments)
    _assert_unpublished(master, assignments)


@pytest.mark.parametrize("boundary", ["materialization_unlock", "commit_delivery"])
def test_cancel_after_materialization_cannot_resurrect_commit(race_case, monkeypatch, boundary):
    case = race_case
    master = case.master
    case.start()
    assignments = _assignments(master)
    config_id = master._pipeline_load_transaction["config_id"]
    ack(master, "worker", "prepare")
    reached = []

    def cancel():
        reached.append(boundary)
        master._abort_pipeline_load_transaction(config_id, "test_cancel_after_materialization")

    if boundary == "materialization_unlock":
        lock = _AfterUnlock(
            master._layer_config_lock,
            lambda: master._pipeline_load_transaction.get("phase") == "committing_local"
            and case.host.pipeline_candidate_status()["phase"] == "materialized",
            cancel,
        )
        monkeypatch.setattr(master, "_layer_config_lock", lock)
    else:
        original_publish = master._publish_pipeline_candidate_configs

        def cancel_then_deliver(configs):
            if configs and all(config.get("phase") == "commit" for config in configs.values()):
                cancel()
            return original_publish(configs)

        monkeypatch.setattr(master, "_publish_pipeline_candidate_configs", cancel_then_deliver)

    first_new_message = len(case.sent)
    ack(master, "android", "prepare")

    assert reached == [boundary]
    assert master._pipeline_load_transaction["phase"] == "aborted"
    assert case.host.pipeline_candidate_status()["phase"] == "aborted"
    assert case.host.peek_manager() is case.active
    assert not any(payload.get("action") == "commit"
                   for _node, payload, *_rest in case.sent[first_new_message:])
    _assert_debts(master, assignments, "abort")
    _replay_ready(master, assignments)
    _assert_unpublished(master, assignments)


@pytest.mark.parametrize("phase", ["committing_global", "ready"])
def test_native_error_revokes_collected_loaded_ready(race_case, monkeypatch, phase):
    case = race_case
    master = case.master
    if phase == "ready":
        publish(case)
        _finish_finalizes(master)
        assignments = _assignments(master)
        complete = None
    else:
        case.start()
        prepare_both(case)
        assignments = _assignments(master)
        complete = master._complete_pipeline_load_transaction
        deferred = []
        monkeypatch.setattr(master, "_complete_pipeline_load_transaction", deferred.append)
        ack(master, "worker", "commit")
        ack(master, "android", "commit")
        assert deferred == [master._pipeline_load_transaction["config_id"]]
    assert master._pipeline_load_transaction["phase"] == phase
    old_registry_nodes = tuple(master._worker_assignments.snapshot())
    if phase == "ready":
        assert set(assignments) <= set(old_registry_nodes)
        assert all(master._worker_assignments.is_ready(node) for node in assignments)

    # The error belongs to this exact operation/generation and connection.
    error = _receipt(assignments["worker"], status="error", loaded=False, error="native_runtime_revoked")
    master._handle_worker_prepare_ack("worker", error)

    assert master._pipeline_load_transaction["phase"] == ("aborted" if complete else "invalidated")
    _assert_unpublished(master, assignments)
    if phase == "ready":
        _assert_registry_not_ready(master, old_registry_nodes)
    _assert_debts(master, assignments, "abort" if complete else "release")
    if complete:
        # A callback already queued before revocation must not publish later.
        complete(master._pipeline_load_transaction["config_id"])
        _assert_unpublished(master, assignments)
    _replay_ready(master, assignments)
    _assert_unpublished(master, assignments)
    if phase == "ready":
        _assert_registry_not_ready(master, old_registry_nodes)


@pytest.mark.parametrize("phase", ["preparing", "ready"])
def test_worker_opt_out_aborts_or_retires_its_current_operation(race_case, phase):
    case = race_case
    master = case.master
    if phase == "ready":
        publish(case)
        _finish_finalizes(master)
    else:
        case.start()
        ack(master, "worker", "prepare")
        assert master._pipeline_load_transaction["prepared_nodes"] == {"worker"}
    assignments = _assignments(master)
    old_registry_nodes = tuple(master._worker_assignments.snapshot())
    if phase == "ready":
        assert set(assignments) <= set(old_registry_nodes)
        assert all(master._worker_assignments.is_ready(node) for node in assignments)

    master._handle_layer_worker_opt_out("worker", {"data": {"node_id": "worker"}})

    assert "worker" in master._pipeline_worker_opt_out
    assert master._pipeline_load_transaction["phase"] == ("aborted" if phase == "preparing" else "invalidated")
    _assert_unpublished(master, assignments)
    if phase == "ready":
        _assert_registry_not_ready(master, old_registry_nodes)
    _assert_debts(master, assignments, "abort" if phase == "preparing" else "release")
    # Completing the other prepare and replaying success cannot undo opt-out.
    master._handle_worker_prepare_ack("android", _receipt(assignments["android"], "prepare"))
    _replay_ready(master, assignments)
    _assert_unpublished(master, assignments)
    if phase == "ready":
        _assert_registry_not_ready(master, old_registry_nodes)


def test_disconnect_then_unload_keeps_every_remote_release_debt(race_case):
    from scheduler_pipeline import _PIPELINE_LIFECYCLE_STATE_KEY

    case = race_case
    master = case.master
    publish(case)
    _finish_finalizes(master)
    assignments = _assignments(master)

    case.connected.remove("worker")
    master._on_tcp_disconnect("worker")

    assert master._active_pipeline_capacity_plan is None
    assert master._pipeline_load_transaction["phase"] == "invalidated"
    assert master._worker_prepare_assignment_loaded("worker", assignments["worker"])[0] is False
    debt_ids = _assert_debts(master, assignments, "release")
    snapshot = deepcopy(master._model_control.snapshot()["lifecycle"])
    assert snapshot["published"] is True
    assert {record["cleanup_id"] for record in snapshot["cleanup_ledger"]} == debt_ids

    # The real unload path clears the live transaction, but must keep its debts.
    result = case.unload()
    assert result["success"] is True
    assert _assert_debts(master, assignments, "release") == debt_ids
    persisted = master._model_control.snapshot()["lifecycle"]
    assert {record["cleanup_id"] for record in persisted["cleanup_ledger"]} == debt_ids
    _assert_unpublished(master, assignments)
    # A reconnected peer receives the original operation identity for retirement.
    case.connected.add("worker")
    first_replay = len(case.sent)
    assert master._replay_pipeline_model_cleanup_for_node("worker") is True
    replayed = case.sent[first_replay:]
    assert len(replayed) == 1
    assert replayed[0][1]["action"] == "release"
    assert replayed[0][1]["identity"] == assignments["worker"]["worker_prepare_identity"]
    assert replayed[0][1]["cleanup_id"] in debt_ids
    assert _assert_debts(master, assignments, "release") == debt_ids


def test_cancel_available_supply_aborts_prepare_and_preserves_verified_assets(race_case, monkeypatch):
    case = race_case
    master = case.master
    supply = master._model_supply
    operation_id = "supply_" + "a" * 32
    available = supply.get(operation_id)
    operation = {
        **available, "operation_id": operation_id,
        "intent": {"model_id": "retirement-model", "engine": "llama_cpp", "quant_type": "gguf"},
        "master_model_path": str(case.model_path),
    }
    supply._operations[operation_id] = deepcopy(operation)
    supply._events[operation_id] = Event()
    supply._active = operation_id
    # Use the real in-memory/disk operation lookup instead of stage_case's stub.
    monkeypatch.setattr(supply, "get", lambda op: type(supply).get(supply, op))
    supply._persist(operation)
    segment = case.model_path / "verified-segment.bin"
    segment.write_bytes(b"verified segment must survive cancellation")
    before = segment.read_bytes()
    candidate = "candidate-cancel-supplied"
    result = master.start_pipeline_model_transaction(candidate_id=candidate,
        descriptor=case.host.prepare_pipeline_candidate(candidate, "retirement-model", str(case.model_path),
            quant_type="fp16", model_sha256="a" * 64),
        model_path=str(case.model_path), quant_type="fp16", operation_id=operation_id)
    assert result["transaction_phase"] == "preparing"
    ack(master, "worker", "prepare")
    assignments = _assignments(master)

    result = supply.cancel(operation_id)

    assert master._pipeline_load_transaction["phase"] == "aborted"
    assert case.host.pipeline_candidate_status()["phase"] == "aborted"
    _assert_unpublished(master, assignments)
    _assert_debts(master, assignments, "abort")
    assert result["status"] == "available"
    assert segment.read_bytes() == before
    persisted_operation = master._model_control.snapshot()["supplies"][operation_id]
    assert persisted_operation["status"] == "available"
    _replay_ready(master, assignments)
    _assert_unpublished(master, assignments)


@pytest.mark.parametrize("phase", ["preparing", "committing", "committing_global"])
def test_restart_rebuilds_abort_debt_without_losing_model_recovery_fence(race_case, monkeypatch, phase):
    from model_host import ModelHost
    from scheduler_pipeline import _PIPELINE_LIFECYCLE_STATE_KEY

    case = race_case
    master = case.master
    case.start()
    if phase == "preparing":
        ack(master, "worker", "prepare")
    else:
        prepare_both(case)
        if phase == "committing_global":
            deferred = []
            monkeypatch.setattr(master, "_complete_pipeline_load_transaction", deferred.append)
            ack(master, "worker", "commit")
            ack(master, "android", "commit")
            assert deferred == [master._pipeline_load_transaction["config_id"]]
    assert master._pipeline_load_transaction["phase"] == phase
    assignments = _assignments(master)
    original = deepcopy(master._model_control.snapshot()["lifecycle"])
    assert original["phase"] == phase
    assert original["cleanup_ledger"] == []

    def restart():
        recovered = case.new_master(ModelHost(manager=_Manager())).master
        monkeypatch.setattr(recovered, "_running", True)
        recovered._load_pipeline_recovery_state()
        return recovered

    recovered = restart()
    assert recovered._pipeline_recovery_pending is True
    assert recovered._pipeline_recovery_failure != "pipeline_recovery_state_invalid"
    _assert_unpublished(recovered, assignments)
    debt_ids = _assert_debts(recovered, assignments, "abort")
    persisted = deepcopy(recovered._model_control.snapshot()["lifecycle"])
    assert {record["cleanup_id"] for record in persisted["cleanup_ledger"] if record["node_id"] != "master"} == debt_ids
    if phase != "preparing":
        local, = [record for record in persisted["cleanup_ledger"] if record["node_id"] == "master"]
        assert local["process_boot_id"] == master._model_control.boot_id
        assert recovered._dispatch_pipeline_model_cleanup(local) is False
    # Rebuilding debts with no live transaction must preserve the raw model and
    # generation fence; otherwise a second crash permits a different model.
    recovery_keys = (
        "config_id", "generation", "phase", "published", "model_id", "model_type",
        "model_sha256", "total_layers", "worker_ids", "assignments",
    )
    for key in recovery_keys:
        assert persisted[key] == original[key], key
    _replay_ready(recovered, assignments)
    _assert_unpublished(recovered, assignments)

    recovered_again = restart()
    assert recovered_again._pipeline_recovery_pending is True
    assert recovered_again._pipeline_recovery_failure != "pipeline_recovery_state_invalid"
    assert _assert_debts(recovered_again, assignments, "abort") == debt_ids
    persisted_again = recovered_again._model_control.snapshot()["lifecycle"]
    for key in recovery_keys:
        assert recovered_again._pipeline_recovery_state[key] == original[key], key
        assert persisted_again[key] == original[key], key
    _assert_unpublished(recovered_again, assignments)

@pytest.mark.parametrize("caller,finalizes", [
    ("operation_cancel", "not_published"),
    ("captured_abort", "pending"),
    ("captured_abort", "acknowledged"),
])
def test_cancel_racing_ready_retires_every_published_assignment(race_case, monkeypatch, caller, finalizes):
    from scheduler_pipeline import _PIPELINE_LIFECYCLE_STATE_KEY

    case = race_case
    master = case.master
    case.start()
    prepare_both(case)
    assignments = _assignments(master)
    config_id = master._pipeline_load_transaction["config_id"]
    complete = master._complete_pipeline_load_transaction
    deferred = []
    monkeypatch.setattr(master, "_complete_pipeline_load_transaction", deferred.append)
    ack(master, "worker", "commit")
    ack(master, "android", "commit")
    assert deferred == [config_id]
    assert master._pipeline_load_transaction["phase"] == "committing_global"
    monkeypatch.setattr(master, "_complete_pipeline_load_transaction", complete)
    reached = []
    published_nodes = []
    completion_abort_ids = []
    completion_generations = []
    cas_calls = []

    def cas_spy(plan):
        cas_calls.append(deepcopy(plan))
        return True

    monkeypatch.setattr(master, "_commit_ready_pipeline_reshard", cas_spy)

    def complete_before_abort():
        reached.append(caller)
        complete(config_id)
        if caller == "operation_cancel":
            # Operation cancellation already marked its intent inside the lookup
            # lock. The queued completion must honor that fence before CAS.
            assert master._pipeline_load_transaction["cancel_requested"] is True
            assert master._pipeline_load_transaction["phase"] == "aborted"
            completion_abort_ids.extend(_assert_debts(master, assignments, "abort"))
            completion_generations.append(master._layer_config_generation)
            return
        assert master._pipeline_load_transaction["phase"] == "ready"
        if finalizes == "acknowledged":
            _finish_finalizes(master)
        assert all(master._worker_assignments.is_ready(node) for node in assignments)
        assert all(master._worker_prepare_assignment_loaded(node, assignment)[0]
                   for node, assignment in assignments.items())
        published_nodes.extend(master._worker_assignments.snapshot())

    # Schedule completion precisely after the caller captures its config/phase
    # and releases the lock, before its selected abort reacquires that lock.
    lock = _AfterUnlock(
        master._layer_config_lock,
        lambda: master._pipeline_load_transaction["phase"] == "committing_global",
        complete_before_abort,
    )
    monkeypatch.setattr(master, "_layer_config_lock", lock)
    if caller == "operation_cancel":
        assert master.cancel_worker_prepare_operation("supply_test") is True
    else:
        with master._layer_config_lock:
            captured = master._pipeline_load_transaction["config_id"]
            assert master._pipeline_load_transaction["phase"] == "committing_global"
        master._abort_pipeline_load_transaction(captured, "pipeline_prepare_cancelled")

    assert reached == [caller]
    _assert_unpublished(master, assignments)
    snapshot = master._model_control.snapshot()["lifecycle"]
    if caller == "operation_cancel":
        assert cas_calls == []
        assert master._pipeline_load_transaction["phase"] == "aborted"
        assert case.host.pipeline_candidate_status()["phase"] == "aborted"
        assert case.host.peek_manager() is case.active
        _assert_registry_not_ready(master, assignments)
        abort_ids = _assert_debts(master, assignments, "abort")
        # The outer abort resumes after completion already aborted. It must be
        # idempotent, retaining the same cleanup debts and generation fence.
        assert abort_ids == set(completion_abort_ids)
        assert completion_generations == [master._layer_config_generation]
        assert snapshot["phase"] == "aborted"
        assert snapshot["published"] is False
        assert not any(record["action"] == "finalize"
                       for record in master._pipeline_model_cleanup_ledger.values())
        _replay_ready(master, assignments)
        _assert_unpublished(master, assignments)
        _assert_registry_not_ready(master, assignments)
        assert _assert_debts(master, assignments, "abort") == abort_ids
        return

    assert len(cas_calls) == 1
    assert master._pipeline_load_transaction["phase"] == "invalidated"
    assert master._pipeline_load_transaction["published"] is True
    _assert_registry_not_ready(master, published_nodes)
    release_ids = _assert_debts(master, assignments, "release")
    assert not any(record["action"] == "abort"
                   for record in master._pipeline_model_cleanup_ledger.values())
    # Publication finalized the host candidate. Retirement must not attempt to
    # roll it back to the old runtime while remote release ACKs are still owed.
    assert case.host.pipeline_candidate_status()["phase"] == "finalized"
    assert case.host.peek_manager() is case.managers[-1]
    assert case.managers[-1].closed == 0
    assert snapshot["phase"] == "invalidated"
    assert snapshot["published"] is True
    assert {record["cleanup_id"] for record in snapshot["cleanup_ledger"]
            if record["action"] == "release"} == release_ids
    _replay_ready(master, assignments)
    _assert_unpublished(master, assignments)
    _assert_registry_not_ready(master, published_nodes)
    assert _assert_debts(master, assignments, "release") == release_ids


@pytest.mark.parametrize("cancel_path", ["operation_cancel", "direct_abort"])
def test_abort_before_local_commit_returns_never_crosses_reshard_cas(race_case, monkeypatch, cancel_path):
    case = race_case
    master = case.master
    case.start()
    prepare_both(case)
    assignments = _assignments(master)
    config_id = master._pipeline_load_transaction["config_id"]
    original_commit = type(case.host).commit_pipeline_candidate
    reached = []
    cas_calls = []

    def commit_then_abort(host, candidate_id):
        result = original_commit(host, candidate_id)
        assert host.pipeline_candidate_status()["phase"] == "committed"
        assert master._pipeline_load_transaction["phase"] == "committing_global"
        reached.append(cancel_path)
        if cancel_path == "operation_cancel":
            assert master.cancel_worker_prepare_operation("supply_test") is True
        else:
            master._abort_pipeline_load_transaction(config_id, "pipeline_prepare_cancelled")
        return result

    def cas_spy(plan):
        cas_calls.append((master._pipeline_load_transaction["phase"], deepcopy(plan)))
        return True

    # Patch the ModelHost class: its instance setattr proxies to the manager.
    monkeypatch.setattr(type(case.host), "commit_pipeline_candidate", commit_then_abort)
    monkeypatch.setattr(master, "_commit_ready_pipeline_reshard", cas_spy)
    ack(master, "worker", "commit")
    ack(master, "android", "commit")

    assert reached == [cancel_path]
    assert cas_calls == [], "an aborted candidate must never advance the reshard epoch"
    assert master._pipeline_load_transaction["phase"] == "aborted"
    assert case.host.pipeline_candidate_status()["phase"] == "aborted"
    assert case.host.peek_manager() is case.active
    abort_ids = _assert_debts(master, assignments, "abort")
    assert not any(record["action"] == "finalize"
                   for record in master._pipeline_model_cleanup_ledger.values())
    _replay_ready(master, assignments)
    _assert_unpublished(master, assignments)
    assert _assert_debts(master, assignments, "abort") == abort_ids


@pytest.mark.parametrize("schema_version", [1, 2, 3])
@pytest.mark.parametrize("action", ["abort", "finalize"])
def test_legacy_offline_cleanup_survives_new_transaction_and_restart(race_case, monkeypatch, schema_version, action):
    from model_host import ModelHost
    from scheduler_pipeline import _PIPELINE_LIFECYCLE_STATE_KEY

    case = race_case
    master = case.master
    legacy_node = "offline-legacy-worker"
    cleanup_id = "legacy-cleanup"
    terminal_key = "aborted_config_id" if action == "abort" else "finalized_config_id"
    case.lifecycle_store[_PIPELINE_LIFECYCLE_STATE_KEY] = {
        "schema_version": schema_version,
        "config_id": "legacy-config", "generation": 17,
        "phase": "aborted" if action == "abort" else "invalidated",
        "model_id": "legacy-model", "model_type": "qwen2", "model_sha256": "f" * 64,
        "assignments": [], "cleanup_ledger": [{
            "cleanup_id": cleanup_id, "node_id": legacy_node,
            "candidate_config_id": "legacy-config", "generation": 18,
            "action": action, "created_at": 1.0,
            "payload": {action: True, terminal_key: "legacy-config",
                        **({"release": True} if action == "abort" else {})},
        }],
    }
    assert legacy_node not in case.connected
    master._load_pipeline_recovery_state()
    assert master._pipeline_lifecycle_persist_ok is True
    assert master._pipeline_recovery_failure != "pipeline_recovery_state_invalid"
    legacy = deepcopy(master._pipeline_model_cleanup_ledger[cleanup_id])
    assert legacy["legacy_identity"] is True
    assert legacy["assignment_id"] == ""
    master._persist_pipeline_lifecycle_locked()
    # Without a live transaction, compatibility is retained at the old schema.
    assert master._model_control.snapshot()["lifecycle"]["schema_version"] == schema_version

    result = case.start("candidate-with-legacy-debt")
    assert result["transaction_phase"] == "preparing"
    assignments = _assignments(master)
    migrated = deepcopy(master._model_control.snapshot()["lifecycle"])
    assert migrated["schema_version"] == 4
    assert migrated["config_id"] == master._pipeline_load_transaction["config_id"]
    assert cleanup_id in master._pipeline_model_cleanup_ledger

    restarted_case = case.new_master(ModelHost(manager=_Manager()))
    restarted = restarted_case.master
    monkeypatch.setattr(restarted, "_running", True)
    restarted._load_pipeline_recovery_state()
    assert restarted._pipeline_lifecycle_persist_ok is True, restarted._pipeline_recovery_failure
    assert restarted._pipeline_recovery_failure == "pipeline_recovery_pending"
    assert restarted._pipeline_recovery_pending is True
    _assert_unpublished(restarted, assignments)
    restored = restarted._pipeline_model_cleanup_ledger[cleanup_id]
    for key in ("cleanup_id", "node_id", "candidate_config_id", "generation", "action", "payload"):
        assert restored[key] == legacy[key], key
    assert restored["legacy_identity"] is True
    assert restored["assignment_id"] == ""
    persisted_legacy = next(record for record in migrated["cleanup_ledger"]
                            if record["cleanup_id"] == cleanup_id)
    assert persisted_legacy["legacy_identity"] is True
    native_debts = {key: record for key, record in restarted._pipeline_model_cleanup_ledger.items()
                    if key != cleanup_id}
    assert {record["node_id"] for record in native_debts.values()} == set(assignments)
    for record in native_debts.values():
        assert record["action"] == "abort"
        assert record.get("legacy_identity", False) is False
        assert record["payload"]["worker_prepare_identity"] == assignments[record["node_id"]]["worker_prepare_identity"]

    restarted_case.connected.add(legacy_node)
    assert restarted._replay_pipeline_model_cleanup_for_node(legacy_node) is True
    replay_node, replay = restarted_case.sent[-1]
    assert replay_node == legacy_node
    assert replay == restored["payload"]
    assert replay["config_id"] == cleanup_id
    assert replay["assignment_id"] == ""
    status = "released" if action == "abort" else "finalized"
    assert restarted._ack_pipeline_model_cleanup(legacy_node, {
        **replay, "status": status, "generation": replay["generation"] + 1,
    }) is False
    assert cleanup_id in restarted._pipeline_model_cleanup_ledger
    assert restarted._ack_pipeline_model_cleanup(legacy_node, {**replay, "status": status}) is True
    assert cleanup_id not in restarted._pipeline_model_cleanup_ledger
    assert set(restarted._pipeline_model_cleanup_ledger) == set(native_debts)
    persisted = restarted._model_control.snapshot()["lifecycle"]
    assert {record["cleanup_id"] for record in persisted["cleanup_ledger"]} == set(native_debts)
    assert all(record["legacy_identity"] is False for record in persisted["cleanup_ledger"])

    restarted_again = case.new_master(ModelHost(manager=_Manager())).master
    monkeypatch.setattr(restarted_again, "_running", True)
    restarted_again._load_pipeline_recovery_state()
    assert restarted_again._pipeline_lifecycle_persist_ok is True
    assert restarted_again._pipeline_recovery_failure == "pipeline_recovery_pending"
    assert set(restarted_again._pipeline_model_cleanup_ledger) == set(native_debts)
    assert cleanup_id not in restarted_again._pipeline_model_cleanup_ledger
    _assert_unpublished(restarted_again, assignments)


@pytest.mark.parametrize("phase", ["prepare", "commit"])
def test_tcp_heartbeat_and_duplicate_ack_progress_during_native_barrier(stage_case, monkeypatch, phase):
    """A blocked real host boundary must leave its inbound TCP stream usable."""
    import threading
    import time
    import config
    import tcp_comm
    from scheduler import Scheduler

    case = stage_case
    master = case.master
    case.start()
    if phase == "prepare":
        ack(master, "worker", "prepare")
        host_method = "materialize_pipeline_candidate"
    else:
        prepare_both(case)
        ack(master, "worker", "commit")
        host_method = "commit_pipeline_candidate"
    entered, release, finished, returned, duplicate, heartbeat = [Event() for _ in range(6)]
    calls = []
    original = getattr(type(case.host), host_method)

    def blocked(host, *args, **kwargs):
        calls.append(threading.current_thread().name)
        entered.set()
        assert release.wait(10), "native test boundary was not released"
        return original(host, *args, **kwargs)

    monkeypatch.setattr(type(case.host), host_method, blocked)
    transition_name = "_commit_pipeline_load_transaction" if phase == "prepare" else "_complete_pipeline_load_transaction"
    transition = getattr(master, transition_name)

    def complete(config_id):
        try:
            return transition(config_id)
        finally:
            finished.set()

    monkeypatch.setattr(master, transition_name, complete)
    monkeypatch.setattr(config, "CLUSTER_SECRET", "r6-native-heartbeat-regression")
    monkeypatch.setattr(tcp_comm.TCPClient, "_compute_local_model_sha256", staticmethod(lambda: "test"))
    monkeypatch.setattr(tcp_comm.TCPClient, "_heartbeat_loop", lambda *_args: None)
    monkeypatch.setattr(tcp_comm, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm, "detect_lan_ip", lambda: "127.0.0.1")
    server = tcp_comm.TCPServer(host="127.0.0.1", port=1)
    server.port = 0
    received = []

    def inbound(node, message):
        if message["type"] == "register":
            server.confirm_registration(node)
        elif message["type"] == "model_prepare":
            Scheduler._on_tcp_message(master, node, message)
            received.append(message)
            returned.set()
            if len(received) == 2:
                duplicate.set()

    server.start(on_message=inbound)
    client = tcp_comm.TCPClient(server_host="127.0.0.1", server_port=server.sock.getsockname()[1], client_id="android", role="client")
    handle_ack = client._handle_heartbeat_ack

    def heartbeat_ack(message, connection_generation=None):
        handle_ack(message, connection_generation)
        heartbeat.set()

    client._handle_heartbeat_ack = heartbeat_ack
    expected = master._pipeline_load_transaction["expected"]["android"]
    payload = _receipt(expected, phase)
    payload["runtime_receipt"] = {"load_spec": deepcopy(expected["load_spec"]), "materialized_bytes": 1,
        "memory_scope": "weight_tensors", "layer_range": deepcopy(payload["identity"]["layer_range"]),
        "artifact_sha256": payload["identity"]["artifact_sha256"]}
    try:
        assert client.connect() is True
        client.send_data(payload, tcp_comm.MessageType.MODEL_PREPARE)
        assert entered.wait(3), "native barrier not entered"
        client.send_data({"t_send": time.time()}, tcp_comm.MessageType.HEARTBEAT)
        assert heartbeat.wait(2), "native barrier blocked TCP heartbeat ACK"
        assert returned.wait(2), "TCP callback is still running native work"
        client.send_data(payload, tcp_comm.MessageType.MODEL_PREPARE)
        assert duplicate.wait(2), "duplicate ACK blocked transport"
        assert len(calls) == 1
        assert master._active_pipeline_capacity_plan is None
        assert client.is_registered
    finally:
        release.set()
        assert finished.wait(5)
        client.disconnect()
        server.stop()
    assert master._pipeline_load_transaction["phase"] == ("committing" if phase == "prepare" else "ready")


def test_reconnect_during_local_materialization_aborts_before_commit_delivery(stage_case, monkeypatch):
    case = stage_case
    master = case.master
    epochs = {"worker": 11, "android": 11}
    monkeypatch.setattr(master, "_assignment_connection_generation", lambda node: epochs[node])
    original = type(case.host).materialize_pipeline_candidate

    def materialize_then_reconnect(host, *args, **kwargs):
        result = original(host, *args, **kwargs)
        epochs["worker"] = 12
        return result

    monkeypatch.setattr(type(case.host), "materialize_pipeline_candidate", materialize_then_reconnect)
    case.start()
    ack(master, "worker", "prepare")
    ack(master, "android", "prepare")
    assert master._pipeline_load_transaction["phase"] == "aborted"
    assert master._pipeline_load_transaction["reason_code"] == "pipeline_worker_connection_changed"
    assert case.host.pipeline_candidate_status()["phase"] == "aborted"
    assert master._active_pipeline_capacity_plan is None
    assert not any(payload.get("action") == "commit" for _, payload, *_ in case.sent)
