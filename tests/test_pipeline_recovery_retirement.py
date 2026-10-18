"""Explicit model changes retire recovery identity without forgiving native debt."""

import asyncio
from copy import deepcopy
import json
import threading
from unittest.mock import Mock

import pytest

import api_server
import local_store
from api import routes_model_intent
from model_intent import ModelIntentCoordinator
from model_supply import ModelSupplyError
from tests.test_pipeline_retirement import _Manager, retirement_case  # noqa: F401
from tests.test_worker_prepare_coordinator import identity


LIFECYCLE_KEY = "pipeline_config_lifecycle_v1"
OLD_MODEL = {"model_id": "old-9b", "model_type": "qwen2", "model_sha256": "a" * 64}
NEW_MODEL = {"model_id": "new-2b", "model_type": "qwen3_5", "model_sha256": "b" * 64}


@pytest.fixture
def recovery_case(retirement_case, monkeypatch):
    """Real Scheduler/ModelHost/API owners; only persistence and model I/O are local."""
    case = retirement_case
    master = case.master
    # This suite exercises import/retirement of the frozen legacy JSON schema.
    # R3 SQLite crash/commit behavior is covered by test_refactor_deployment_owner.
    master._model_control = None
    legacy_new_master = case.new_master
    def new_legacy_master(host):
        restored = legacy_new_master(host)
        restored.master._model_control = None
        return restored
    case.new_master = new_legacy_master
    master._running = True
    # The fixture's connected worker is healthy; both old participants are offline.
    assignments = []
    for node, span in (("y700", [4, 6]), ("surface", [6, 8])):
        native = identity(node, span)
        native.update(config_id="cfg-old", generation=7, model_id=OLD_MODEL["model_id"],
                      source_model_id=OLD_MODEL["model_id"])
        assignments.append({"node_id": node, "start_layer": span[0], "end_layer": span[1],
                            "worker_prepare_identity": native})
        master._record_pipeline_model_cleanup_locked(
            node_id=node, candidate_config_id="cfg-old", generation=7, action="release",
            assignment_id=native["assignment_id"],
            payload={"worker_prepare_identity": native, "release": True},
        )
    master._pipeline_load_transaction = {
        "config_id": "cfg-old", "generation": 7, "phase": "ready", "published": True,
        "worker_ids": {"y700", "surface"}, "ready_nodes": {"y700", "surface"},
        "plan": {**OLD_MODEL, "total_layers": 8, "quant_type": "fp16", "assignments": assignments},
    }
    case.settings = {LIFECYCLE_KEY: master._pipeline_lifecycle_snapshot_locked()}
    case.saved = []

    def save(key, value):
        case.settings[key] = deepcopy(value)
        if key == LIFECYCLE_KEY:
            case.saved.append(deepcopy(value))

    monkeypatch.setattr(local_store, "get_local_setting",
                        lambda key, default=None: deepcopy(case.settings.get(key, default)))
    monkeypatch.setattr(local_store, "set_local_setting", save)
    # Reconstruct the fence and exact native debts through the restart loader.
    master._pipeline_load_transaction = None
    master._pipeline_model_cleanup_ledger.clear()
    master._load_pipeline_recovery_state()
    assert master._pipeline_recovery_pending is True
    assert master._pipeline_lifecycle_persist_ok is True
    assert master._pipeline_load_transaction is None
    assert master._active_pipeline_capacity_plan is None
    case.debts = deepcopy(master._pipeline_model_cleanup_ledger)
    assert {item["node_id"] for item in case.debts.values()} == {"y700", "surface"}
    assert all(item["state"] == "await_reconnect" for item in case.debts.values())
    assert master._connected_client_ids().isdisjoint({"y700", "surface"})
    case.saved.clear()
    case.durable_debts = deepcopy(case.settings[LIFECYCLE_KEY]["cleanup_ledger"])
    return case


def test_successful_unload_retires_recovery_before_refresh(recovery_case, monkeypatch):
    case = recovery_case
    master = case.master
    events = []
    retire = master.retire_pipeline_recovery_for_model_change
    clear = master._clear_pipeline_recovery_fence
    save = local_store.set_local_setting
    # Include a stale registry entry to observe that the API clears it before retirement.
    master._worker_assignments.begin("worker", expected_config={"config_id": "cfg-old", "generation": 7})
    assert master._recovery_model_matches(NEW_MODEL) is False

    def retire_after_local_change():
        events.append("retire")
        assert case.active.closed == 1 and not case.host.model_loaded
        assert master._running and master._effective_role() == "master"
        assert master._pipeline_load_transaction is None
        assert master._active_pipeline_capacity_plan is None
        assert master._worker_assignments.snapshot() == {}
        assert master._pipeline_recovery_pending is True
        return retire()

    def persist_before_clear(key, value):
        assert key == LIFECYCLE_KEY
        assert master._layer_config_lock._is_owned()
        assert master._pipeline_recovery_pending is True
        assert value["phase"] == "invalidated" and value["published"] is False
        assert master._pipeline_model_cleanup_ledger == case.debts
        events.append("persist")
        save(key, value)

    def clear_after_persist():
        assert master._layer_config_lock._is_owned()
        assert case.settings[LIFECYCLE_KEY]["phase"] == "invalidated"
        assert case.settings[LIFECYCLE_KEY]["published"] is False
        events.append("clear")
        clear()

    def refresh_after_retirement():
        assert master._pipeline_recovery_pending is False
        assert master._recovery_model_matches(NEW_MODEL) is True
        events.append("refresh")

    monkeypatch.setattr(master, "retire_pipeline_recovery_for_model_change", retire_after_local_change)
    monkeypatch.setattr(local_store, "set_local_setting", persist_before_clear)
    monkeypatch.setattr(master, "_clear_pipeline_recovery_fence", clear_after_persist)
    monkeypatch.setattr(api_server, "_refresh_pipeline_layer_config", refresh_after_retirement)

    assert case.unload()["unloaded"] is True

    assert events == ["retire", "persist", "clear", "refresh"]
    assert master._pipeline_recovery_pending is False
    assert master._pipeline_recovery_failure == ""
    assert master._pipeline_model_cleanup_ledger == case.debts
    assert case.saved[-1]["cleanup_ledger"] == case.durable_debts


@pytest.mark.parametrize("field,value", [("config_id", "cfg-newer"), ("generation", 8), ("phase", "preparing")])
def test_route_retirement_waiting_for_model_lock_rejects_new_recovery_identity(
        recovery_case, monkeypatch, field, value):
    case = recovery_case
    master = case.master
    monkeypatch.setattr(routes_model_intent, "_api_module", api_server)
    unload = Mock(wraps=api_server._unload_model_under_model_lock)
    monkeypatch.setattr(api_server, "_unload_model_under_model_lock", unload)
    entered = threading.Event()

    def before_model_lock(reason):
        assert reason == "local_model_load"
        entered.set()

    monkeypatch.setattr(api_server, "_wait_for_worker_prepare_retirement", before_model_lock)
    durable_before = deepcopy(case.settings[LIFECYCLE_KEY])
    newer = _Manager(loaded=True)
    newer_recovery = {**master._pipeline_recovery_state, field: value}
    assert master._pipeline_load_transaction is None
    assert master._active_pipeline_capacity_plan is None

    async def scenario():
        retirement = None
        try:
            # The callback captures recovery identity before its worker waits on
            # this real owner lock. Replace only recovery identity while it waits.
            with case.host.full_chat_execution_lock:
                retirement = asyncio.create_task(routes_model_intent._retire_native_plan())
                assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), timeout=4)
                assert not retirement.done()
                with master._layer_config_lock:
                    master._pipeline_recovery_state = deepcopy(newer_recovery)
                case.host._manager = newer
                case.host.model_loaded = True
                case.host.current_quant = "fp16"
            with pytest.raises(ModelSupplyError) as error:
                await asyncio.wait_for(retirement, timeout=3)
            assert error.value.code == "INTENT_RETIRE_SUPERSEDED"
        finally:
            if retirement is not None:
                await asyncio.gather(retirement, return_exceptions=True)

    asyncio.run(scenario())

    unload.assert_not_called()
    assert case.host.peek_manager() is newer and newer.is_loaded
    assert newer.unload_attempts == 0 and case.active.unload_attempts == 0
    assert case.host.model_loaded and case.host.current_quant == "fp16"
    assert master._pipeline_recovery_pending is True
    assert master._pipeline_recovery_state == newer_recovery
    assert master._pipeline_load_transaction is None
    assert master._active_pipeline_capacity_plan is None
    assert master._layer_config_model_change_depth == 0
    assert master._pipeline_model_cleanup_ledger == case.debts
    assert case.saved == [] and case.settings[LIFECYCLE_KEY] == durable_before


def test_retirement_tombstone_survives_second_restart_with_all_native_debts(recovery_case, monkeypatch):
    case = recovery_case
    assert case.unload()["unloaded"] is True
    # Round-trip the captured setting as JSON, then give a fresh Scheduler only that record.
    persisted = json.loads(json.dumps(case.settings[LIFECYCLE_KEY]))
    assert persisted["phase"] == "invalidated" and persisted["published"] is False
    monkeypatch.setattr(local_store, "get_local_setting",
                        lambda key, default=None: deepcopy(persisted) if key == LIFECYCLE_KEY else default)
    restarted = case.new_master(case.host).master
    restarted._running = True
    restore_model = Mock(side_effect=AssertionError("retired 9B must not be restored"))
    monkeypatch.setattr(case.host, "prepare_pipeline_model", restore_model)

    restarted._load_pipeline_recovery_state()

    assert restarted._pipeline_recovery_pending is False
    assert restarted._pipeline_lifecycle_persist_ok is True
    assert restarted._pipeline_load_transaction is None
    assert restarted._active_pipeline_capacity_plan is None
    assert restarted._pipeline_model_cleanup_ledger == case.debts
    assert restarted._restore_pipeline_model_for_recovery() is True
    restore_model.assert_not_called()
    assert restarted._recovery_model_matches(NEW_MODEL) is True


def test_failed_tombstone_persistence_keeps_recovery_fence_and_native_debts(recovery_case, monkeypatch):
    case = recovery_case
    master = case.master
    durable_before = deepcopy(case.settings[LIFECYCLE_KEY])
    attempted = []
    clear = Mock(wraps=master._clear_pipeline_recovery_fence)
    monkeypatch.setattr(master, "_clear_pipeline_recovery_fence", clear)

    def fail_write(key, value):
        assert key == LIFECYCLE_KEY and master._layer_config_lock._is_owned()
        attempted.append(deepcopy(value))
        raise OSError("injected lifecycle store failure")

    monkeypatch.setattr(local_store, "set_local_setting", fail_write)
    assert case.unload()["unloaded"] is True

    assert attempted and attempted[-1]["phase"] == "invalidated"
    assert attempted[-1]["published"] is False
    assert attempted[-1]["cleanup_ledger"] == case.durable_debts
    assert master._pipeline_recovery_pending is True
    assert master._pipeline_lifecycle_persist_ok is False
    assert master._pipeline_recovery_failure == "pipeline_lifecycle_persist_failed"
    clear.assert_not_called()
    assert case.settings[LIFECYCLE_KEY] == durable_before
    assert master._pipeline_model_cleanup_ledger == case.debts
    assert master._recovery_model_matches(NEW_MODEL) is False


@pytest.mark.parametrize("failure", ["return_false", "unload_exception"])
def test_failed_local_change_preserves_recovery_identity_and_digest_check(recovery_case, monkeypatch, failure):
    case = recovery_case
    master = case.master
    recovery_before = deepcopy(master._pipeline_recovery_state)
    durable_before = deepcopy(case.settings[LIFECYCLE_KEY])
    retire = Mock(wraps=master.retire_pipeline_recovery_for_model_change)
    refresh = Mock()
    monkeypatch.setattr(master, "retire_pipeline_recovery_for_model_change", retire)
    monkeypatch.setattr(api_server, "_refresh_pipeline_layer_config", refresh)

    if failure == "return_false":
        result = api_server._run_exclusive_model_change(lambda: {"success": False, "error": "load failed"})
        assert result["success"] is False
    else:
        case.active.fail_unload = True
        with pytest.raises(RuntimeError, match="previous runtime cleanup unavailable"):
            case.unload()

    retire.assert_not_called()
    refresh.assert_not_called()
    assert case.active.closed == 0 and case.host.model_loaded
    assert master._pipeline_recovery_pending is True
    assert master._pipeline_recovery_state == recovery_before
    assert master._pipeline_model_cleanup_ledger == case.debts
    assert case.saved == [] and case.settings[LIFECYCLE_KEY] == durable_before
    assert master._recovery_model_matches(OLD_MODEL) is True
    assert master._recovery_model_matches({**OLD_MODEL, "model_sha256": "b" * 64}) is False
    assert master._pipeline_recovery_failure == "pipeline_recovery_model_mismatch"
    assert master._recovery_model_matches(NEW_MODEL) is False


@pytest.mark.parametrize("reason", ["pipeline_recovery_state_invalid", "pipeline_recovery_state_unavailable"])
def test_unload_cannot_clear_an_invalid_or_unavailable_recovery_fence(recovery_case, monkeypatch, reason):
    case = recovery_case
    master = case.master
    master._pipeline_lifecycle_persist_ok = False
    master._pipeline_recovery_failure = reason
    recovery_before = deepcopy(master._pipeline_recovery_state)
    durable_before = deepcopy(case.settings[LIFECYCLE_KEY])
    clear = Mock(wraps=master._clear_pipeline_recovery_fence)
    monkeypatch.setattr(master, "_clear_pipeline_recovery_fence", clear)

    assert case.unload()["unloaded"] is True

    clear.assert_not_called()
    assert master._pipeline_recovery_pending is True
    assert master._pipeline_lifecycle_persist_ok is False
    assert master._pipeline_recovery_failure == reason
    assert master._pipeline_recovery_state == recovery_before
    assert master._pipeline_model_cleanup_ledger == case.debts
    assert case.saved == [] and case.settings[LIFECYCLE_KEY] == durable_before

@pytest.mark.parametrize("corruption", ["late_assignment_identity", "ledger_structure", "ledger_identity"])
def test_invalid_recovery_survives_partial_ack_manual_persist_unload_and_restart(
        recovery_case, monkeypatch, corruption):
    case = recovery_case
    raw = deepcopy(case.settings[LIFECYCLE_KEY])
    if corruption == "late_assignment_identity":
        # No pre-existing debts: the first assignment creates the partial ledger
        # before validation of the second assignment fails.
        raw["cleanup_ledger"] = []
        raw["assignments"][1]["worker_prepare_identity"]["manifest_sha256"] = "invalid-digest"
    elif corruption == "ledger_structure":
        raw["cleanup_ledger"] = {"unparsed_obligations": raw["cleanup_ledger"]}
    else:
        raw["cleanup_ledger"][1]["payload"]["assignment_id"] = "wrong-assignment"

    # Keep serialized bytes as the durable owner, so attempted writes and loss
    # of the unrecovered obligation are both observable across a second restart.
    original_bytes = json.dumps(raw, indent=2).encode("utf-8")
    durable = {"bytes": original_bytes}

    def read(key, default=None):
        return json.loads(durable["bytes"]) if key == LIFECYCLE_KEY else default

    def write(key, value):
        assert key == LIFECYCLE_KEY
        durable["bytes"] = json.dumps(value).encode("utf-8")

    save = Mock(side_effect=write)
    monkeypatch.setattr(local_store, "get_local_setting", read)
    monkeypatch.setattr(local_store, "set_local_setting", save)
    master = case.new_master(case.host).master
    master._running = True
    monkeypatch.setattr(api_server, "scheduler", master)
    master._load_pipeline_recovery_state()

    def assert_invalid_record_preserved(scheduler):
        save.assert_not_called()
        assert durable["bytes"] == original_bytes
        assert scheduler._pipeline_recovery_state == raw
        assert scheduler._pipeline_recovery_validated is False
        assert scheduler._pipeline_recovery_pending is True
        assert scheduler._pipeline_lifecycle_persist_ok is False
        assert scheduler._pipeline_recovery_failure == "pipeline_recovery_state_invalid"

    assert_invalid_record_preserved(master)
    if corruption == "late_assignment_identity":
        partial, = master._pipeline_model_cleanup_ledger.values()
        assert partial["node_id"] == "y700" and partial["action"] == "release"
        assert partial["payload"]["worker_prepare_identity"] == raw["assignments"][0]["worker_prepare_identity"]
        # The real native ACK owner may settle the known debt in memory, but
        # its persistence call must not replace the still-unvalidated raw record.
        master._handle_worker_prepare_ack("y700", {
            "action": "ack", "phase": "release", "status": "released", "loaded": False,
            "identity": deepcopy(partial["payload"]["worker_prepare_identity"]),
            "cleanup_id": partial["cleanup_id"],
        })
        assert master._pipeline_model_cleanup_ledger == {}
        assert_invalid_record_preserved(master)
    else:
        assert master._pipeline_model_cleanup_ledger == {}

    with master._layer_config_lock:
        master._persist_pipeline_lifecycle_locked()
    assert_invalid_record_preserved(master)
    assert master._restore_pipeline_model_for_recovery() is False
    assert_invalid_record_preserved(master)

    clear = Mock(wraps=master._clear_pipeline_recovery_fence)
    retire = Mock(wraps=master.retire_pipeline_recovery_for_model_change)
    monkeypatch.setattr(master, "_clear_pipeline_recovery_fence", clear)
    monkeypatch.setattr(master, "retire_pipeline_recovery_for_model_change", retire)
    assert case.unload()["unloaded"] is True
    assert case.active.closed == 1
    retire.assert_called_once_with()
    clear.assert_not_called()
    assert master._pipeline_model_cleanup_ledger == {}
    assert_invalid_record_preserved(master)

    restarted = case.new_master(case.host).master
    restarted._running = True
    restarted._load_pipeline_recovery_state()
    assert_invalid_record_preserved(restarted)
    if corruption == "late_assignment_identity":
        # Even the valid debt is reconstructed from the unchanged evidence;
        # the invalid second identity must still prevent recovery admission.
        assert {item["node_id"] for item in restarted._pipeline_model_cleanup_ledger.values()} == {"y700"}

