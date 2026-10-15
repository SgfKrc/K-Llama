"""Native-load barriers, identity fencing and durable retirement."""
from copy import deepcopy
from types import SimpleNamespace
import pytest

from tests.test_pipeline_retirement import retirement_case, _capacity_plan
from worker_prepare_contract import validate_identity, validate_command, validate_ack, stage_identity, WorkerPrepareError


def identity(node="worker", span=None):
    span = span or [4, 8]
    return {"operation_id": "supply_test", "config_id": "config-test", "generation": 7,
            "assignment_id": "assignment-" + node, "node_id": node,
            "model_id": "retirement-model", "model_sha256": "a" * 64,
            "stage_model_identity": {"model_id": "artifact-" + node, "engine": "llama_cpp", "format": "gguf", "revision": "1", "sha256": "b" * 64},
            "artifact_sha256": "b" * 64, "manifest_sha256": "c" * 64,
            "source_model_id": "retirement-model", "source_model_sha256": "d" * 64,
            "tokenizer_sha256": "e" * 64, "layer_range": span, "total_layers": 8,
            "hidden_size": 4, "segment_mode": "tail" if span[1] == 8 else "middle"}


@pytest.fixture
def stage_case(retirement_case, monkeypatch):
    case = retirement_case
    master = case.master
    case.connected.clear(); case.connected.update({"worker", "android"})
    master.nodes["android"] = SimpleNamespace(node_type="android")
    master._tcp_server.send_to_client = lambda node, payload, msg_type: case.sent.append((node, deepcopy(payload), msg_type.value))
    monkeypatch.setattr(master, "_task_worker_layer_stage_ids", lambda _ids: {"worker", "android"})
    monkeypatch.setattr(master, "_assignment_connection_generation", lambda _node: 11)
    def capacity(_ids, *, descriptor, require_distributed, capacity_evidence=None):
        plan = _capacity_plan(descriptor)
        plan["assignments"] = [plan["assignments"][0]]
        for node, span in (("worker", [4, 6]), ("android", [6, 8])):
            value = identity(node, span)
            plan["assignments"].append({"node_id": node, "start_layer": span[0], "end_layer": span[1],
                "layers_count": span[1] - span[0], "has_embedding": False, "has_lm_head": span[1] == 8,
                "stage_model_identity": value["stage_model_identity"],
                "layer_artifact": {"artifact_sha256": value["artifact_sha256"], "layer_range": span,
                    "model_id": value["stage_model_identity"]["model_id"], "segment_mode": value["segment_mode"],
                    "source_model_id": value["source_model_id"], "source_model_sha256": value["source_model_sha256"],
                    "tokenizer_sha256": value["tokenizer_sha256"], "hidden_size": 4}})
        return plan
    monkeypatch.setattr(master, "get_pipeline_capacity_plan", capacity)
    monkeypatch.setattr(master._model_supply, "get", lambda _op: {"status": "available", "nodes": {
        node: {"status": "available", "artifact_sha256": "b" * 64, "manifest_sha256": "c" * 64, "layer_range": span}
        for node, span in (("worker", [4, 6]), ("android", [6, 8]))}})
    case.start = lambda candidate="candidate-stage": master.start_pipeline_model_transaction(
        candidate_id=candidate,
        descriptor=case.host.prepare_pipeline_candidate(candidate, "retirement-model", str(case.model_path), quant_type="fp16", model_sha256="a" * 64),
        model_path=str(case.model_path), quant_type="fp16", operation_id="supply_test")
    return case


def ack(master, node, phase, **changes):
    value = deepcopy(master._pipeline_load_transaction["expected"][node]["worker_prepare_identity"])
    payload = {"action": "ack", "phase": phase, "identity": value,
               "status": "prepared" if phase == "prepare" else "ready", "loaded": True}
    payload.update(changes)
    master._handle_worker_prepare_ack(node, payload)
    return payload


def prepare_both(case):
    ack(case.master, "worker", "prepare")
    assert case.host.pipeline_candidate_status()["phase"] == "prepared"
    assert case.master._active_pipeline_capacity_plan is None
    ack(case.master, "android", "prepare")
    assert case.master._pipeline_load_transaction["phase"] == "committing"
    assert case.host.pipeline_candidate_status()["phase"] == "materialized"


def publish(case):
    assert case.start()["transaction_phase"] == "preparing"
    prepare_both(case)
    ack(case.master, "worker", "commit")
    assert case.master._active_pipeline_capacity_plan is None
    ack(case.master, "android", "commit")
    assert case.master._pipeline_load_transaction["phase"] == "ready"


def test_stage_only_plan_waits_for_every_native_prepare_and_commit(stage_case):
    case = stage_case
    assert case.start()["transaction_phase"] == "preparing"
    assert case.master._pipeline_load_transaction["worker_ids"] == {"worker", "android"}
    assert len(case.sent) == 2 and all(item[2] == "model_prepare" for item in case.sent)
    assert case.host.pipeline_candidate_status()["phase"] == "prepared"
    prepare_both(case)
    ack(case.master, "worker", "commit")
    assert case.master._active_pipeline_capacity_plan is None
    ack(case.master, "android", "commit")
    assert case.master._active_pipeline_capacity_plan["transaction_phase"] == "ready"
    assignment = case.master._active_pipeline_capacity_plan["assignments"][1]
    assert case.master._worker_prepare_assignment_loaded("worker", assignment) == (True, "ready")


@pytest.mark.parametrize("key,changed", [("operation_id", "supply_old"), ("config_id", "old-config"),
    ("generation", 6), ("assignment_id", "old-asg"), ("artifact_sha256", "f" * 64),
    ("manifest_sha256", "f" * 64), ("layer_range", [3, 6])])
def test_old_or_wrong_receipt_cannot_advance_candidate(stage_case, key, changed):
    case = stage_case; case.start()
    value = deepcopy(case.master._pipeline_load_transaction["expected"]["worker"]["worker_prepare_identity"])
    value[key] = changed
    ack(case.master, "worker", "prepare", identity=value)
    assert case.master._pipeline_load_transaction["prepared_nodes"] == set()
    assert case.master._pipeline_load_transaction["phase"] == "preparing"


def test_legacy_ack_cannot_substitute_native_ready(stage_case):
    case = stage_case; case.start()
    expected = case.master._pipeline_load_transaction["expected"]["worker"]
    case.master._handle_layer_config_ack("worker", {"data": {**expected, "status": "prepared", "loaded": True}})
    assert case.master._pipeline_load_transaction["prepared_nodes"] == set()


def test_inventory_and_unloaded_ack_do_not_grant_execution(stage_case):
    case = stage_case; case.start()
    ack(case.master, "worker", "prepare", loaded=False)
    assignment = case.master._pipeline_load_transaction["plan"]["assignments"][1]
    assert case.master._worker_prepare_assignment_loaded("worker", assignment)[0] is False
    assert case.master._pipeline_load_transaction["prepared_nodes"] == set()


def test_connection_replacement_discards_loaded_receipt(stage_case, monkeypatch):
    case = stage_case; case.start()
    monkeypatch.setattr(case.master, "_assignment_connection_generation", lambda _node: 12)
    ack(case.master, "worker", "prepare")
    assert case.master._pipeline_load_transaction["prepared_nodes"] == set()


def test_native_failure_aborts_all_candidates_and_records_cleanup(stage_case):
    case = stage_case; case.start()
    ack(case.master, "worker", "prepare")
    ack(case.master, "android", "prepare", status="error", loaded=False, error="native_load_failed")
    assert case.master._pipeline_load_transaction["phase"] == "aborted"
    assert case.master._active_pipeline_capacity_plan is None
    assert {record["node_id"] for record in case.master._pipeline_model_cleanup_ledger.values()} == {"worker", "android"}
    assert all(record["action"] == "abort" for record in case.master._pipeline_model_cleanup_ledger.values())


def test_ready_unload_requires_real_remote_release_ack(stage_case):
    case = stage_case; publish(case)
    # Clear finalize via the exact identity-bearing ACK, then retire the plan.
    for record in list(case.master._pipeline_model_cleanup_ledger.values()):
        case.master._handle_worker_prepare_ack(record["node_id"], {"action": "ack", "phase": "finalize",
            "identity": record["payload"]["worker_prepare_identity"], "cleanup_id": record["cleanup_id"], "status": "finalized", "loaded": True})
    assert case.master._pipeline_model_cleanup_ledger == {}
    case.master._invalidate_pipeline_load_transaction("test_unload")
    records = list(case.master._pipeline_model_cleanup_ledger.values())
    assert {record["action"] for record in records} == {"release"}
    for record in records:
        payload = {"action": "ack", "phase": "release", "identity": record["payload"]["worker_prepare_identity"],
            "cleanup_id": record["cleanup_id"], "status": "released", "loaded": True}
        case.master._handle_worker_prepare_ack(record["node_id"], payload)
        assert record["cleanup_id"] in case.master._pipeline_model_cleanup_ledger
        payload["loaded"] = False
        case.master._handle_worker_prepare_ack(record["node_id"], payload)
    assert case.master._pipeline_model_cleanup_ledger == {}


def test_prepare_contract_rejects_full_model_and_extra_fields():
    value = identity(); value["layer_range"] = [0, 8]; value["segment_mode"] = "head"
    with pytest.raises(WorkerPrepareError): validate_identity(value)
    value = identity(); value["invented"] = True
    with pytest.raises(WorkerPrepareError): validate_identity(value)
    with pytest.raises(WorkerPrepareError): validate_command({"action": "abort", "identity": identity()})
    with pytest.raises(WorkerPrepareError): validate_ack({"action": "ack", "phase": "prepare", "identity": identity(), "status": "prepared", "loaded": False})



def test_legacy_cleanup_ack_cannot_clear_native_debt(stage_case):
    case = stage_case; publish(case)
    record = next(iter(case.master._pipeline_model_cleanup_ledger.values()))
    payload = {**record["payload"], "status": "finalized"}
    case.master._handle_layer_config_ack(record["node_id"], {"data": payload})
    assert record["cleanup_id"] in case.master._pipeline_model_cleanup_ledger


def test_connection_changes_at_publication_roll_back_materialized_candidate(stage_case, monkeypatch):
    case = stage_case; case.start(); prepare_both(case)
    original_commit = case.host.commit_pipeline_candidate
    def commit_and_disconnect(_host, candidate):
        original_commit(candidate)
        monkeypatch.setattr(case.master, "_assignment_connection_generation", lambda _node: 12)
    monkeypatch.setattr(type(case.host), "commit_pipeline_candidate", commit_and_disconnect)
    ack(case.master, "worker", "commit")
    ack(case.master, "android", "commit")
    assert case.master._pipeline_load_transaction["phase"] == "aborted"
    assert case.master._active_pipeline_capacity_plan is None
    assert case.host.is_loaded is True  # The prior runtime is restored.
    assert case.host.pipeline_candidate_status()["phase"] == "aborted"


def test_retirement_cleanup_survives_master_restart(stage_case, monkeypatch):
    import local_store
    case = stage_case; publish(case)
    case.master._invalidate_pipeline_load_transaction("test_unload")
    snapshot = case.master._pipeline_lifecycle_snapshot_locked()
    monkeypatch.setattr(local_store, "get_local_setting", lambda _key, _default=None: snapshot)
    restored = case.new_master(case.host).master
    restored._load_pipeline_recovery_state()
    records = list(restored._pipeline_model_cleanup_ledger.values())
    assert {record["action"] for record in records} == {"finalize", "release"}
    assert all(record["payload"]["worker_prepare_identity"] for record in records)
    assert restored._pipeline_recovery_failure != "pipeline_recovery_state_invalid"


def test_hello_refresh_cannot_supersede_inflight_model_operation(stage_case):
    case = stage_case; case.start()
    before = case.master._pipeline_load_transaction
    count = len(case.sent)
    result = case.master._push_layer_config_to_clients_locked()
    assert result["transaction_phase"] == "preparing"
    assert case.master._pipeline_load_transaction is before
    assert len(case.sent) == count


def test_supply_registration_does_not_automatically_load_stage_workers(stage_case):
    case = stage_case
    case.master._model_supply._active = "supply_test"
    result = case.master._push_layer_config_to_clients_locked()
    assert result == {}
    assert case.master._pipeline_load_transaction is None
    assert case.sent == []


def test_recovery_ledger_updates_preserve_model_identity_across_restarts(stage_case, monkeypatch):
    import local_store
    case = stage_case
    case.start()
    snapshot = case.master._pipeline_lifecycle_snapshot_locked()
    persisted = []
    monkeypatch.setattr(local_store, "get_local_setting", lambda _key, _default=None: snapshot)
    monkeypatch.setattr(local_store, "set_local_setting", lambda _key, value: persisted.append(deepcopy(value)))
    restored = case.new_master(case.host).master
    restored._running = True
    restored._load_pipeline_recovery_state()
    assert persisted
    assert persisted[-1]["model_id"] == snapshot["model_id"]
    assert persisted[-1]["model_sha256"] == snapshot["model_sha256"]
    assert persisted[-1]["assignments"] == snapshot["assignments"]
    for record in list(restored._pipeline_model_cleanup_ledger.values()):
        restored._handle_worker_prepare_ack(record["node_id"], {
            "action": "ack", "phase": "abort", "identity": record["payload"]["worker_prepare_identity"],
            "cleanup_id": record["cleanup_id"], "status": "released", "loaded": False,
        })
    assert restored._pipeline_model_cleanup_ledger == {}
    latest = persisted[-1]
    assert latest["cleanup_ledger"] == []
    assert latest["config_id"] == snapshot["config_id"]
    assert latest["assignments"] == snapshot["assignments"]
    monkeypatch.setattr(local_store, "get_local_setting", lambda _key, _default=None: latest)
    restarted = case.new_master(case.host).master
    restarted._load_pipeline_recovery_state()
    assert restarted._pipeline_recovery_state["model_id"] == snapshot["model_id"]
    assert restarted._pipeline_recovery_failure == "pipeline_recovery_pending"


def test_supply_cancel_before_transaction_creation_fences_candidate_and_allows_retry(stage_case, monkeypatch):
    from model_supply import ModelSupplyError
    case = stage_case
    supply = case.master._model_supply
    supply._operations["supply_test"] = {"operation_id": "supply_test", "status": "available", "nodes": {}}
    monkeypatch.setattr(supply, "prepared_source", lambda _op: {})
    with supply.prepare_guard("supply_test"):
        supply.cancel("supply_test")
        with pytest.raises(ModelSupplyError) as error:
            case.start()
        assert error.value.code == "SUPPLY_PREPARE_CANCELLED"
        assert case.master._pipeline_load_transaction is None
        assert case.sent == []
        case.host.abort_pipeline_candidate("candidate-stage")
    # Cancellation retires this attempt, while verified disk assets are reusable.
    with supply.prepare_guard("supply_test"):
        assert case.start("candidate-retry")["transaction_phase"] == "preparing"


def test_source_host_budget_constrains_gpu_plan_and_intersects_reuse_preferences():
    from model_supply import plan_segment_supply
    descriptor = {"schema_version": 1, "model_id": "host-budget", "model_type": "qwen2", "model_sha256": "a" * 64,
                  "total_layers": 8, "hidden_size": 4, "layer_weight_bytes": [100] * 8,
                  "component_weight_bytes": {"embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0},
                  "pipeline_runtime_supported": True}
    nodes = [{"node_id": node, "engine": engine, "capacity_bytes": 10000, "reserve_bytes": 0,
              "runtime_multiplier": 1, "score": score}
             for node, engine, score in (("master", "pytorch", 100), ("worker", "llama_cpp", 50), ("android", "llama_cpp", 10))]
    host_ranges = {"master": [[0, 2]], "worker": [[2, 4], [2, 6]]}
    plan = plan_segment_supply({"pytorch": descriptor, "llama_cpp": descriptor}, nodes,
        master_engine="pytorch", alignment=2, runtime_host_ranges=host_ranges,
        preferred_ranges={"worker": [[2, 4]]})
    assert plan["admitted"] is True
    assert [(a["node_id"], a["start_layer"], a["end_layer"]) for a in plan["assignments"]] == [
        ("master", 0, 2), ("worker", 2, 4), ("android", 4, 8)]
    rejected = plan_segment_supply({"pytorch": descriptor, "llama_cpp": descriptor}, nodes,
        master_engine="pytorch", alignment=2, runtime_host_ranges=host_ranges,
        preferred_ranges={"worker": [[4, 6]]})
    assert all(a["node_id"] != "worker" for a in rejected.get("assignments", []))
    assert nodes[0].get("layer_ranges") is None
    assert host_ranges == {"master": [[0, 2]], "worker": [[2, 4], [2, 6]]}


def test_retry_does_not_recreate_state_for_confirmed_native_receipts(stage_case):
    import time
    case = stage_case
    case.start()
    ack(case.master, "worker", "prepare")
    case.sent.clear()
    case.master._retry_pending_layer_configs(now=time.monotonic() + 10)
    assert all(node != "worker" for node, *_rest in case.sent)
    prepare_both(case)
    ack(case.master, "worker", "commit")
    case.sent.clear()
    case.master._retry_pending_layer_configs(now=time.monotonic() + 10)
    assert all(node != "worker" for node, *_rest in case.sent)
    assert any(node == "android" for node, *_rest in case.sent)
