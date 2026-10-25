"""R3 transactional publication and crash matrix; temporary SQLite and tiny receipts."""
from copy import deepcopy
from contextlib import contextmanager
import sqlite3

import pytest

from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
from qlh_core.control.deployment import DeploymentService
from qlh_core.contracts.deployment import ModelControlError, MODEL_CONTROL_DOMAIN

pytestmark = pytest.mark.refactor_contract


@pytest.fixture
def control(tmp_path):
    path = tmp_path / "model-control.sqlite3"
    def connect():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection
    @contextmanager
    def transaction():
        connection = connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
    repository = SqliteModelControlRepository(connect=connect, transaction=transaction)
    owner = DeploymentService(repository, boot_id="boot-1", clock=lambda: 1.0)
    owner.start()
    return owner, repository, connect


def candidate(phase="preparing", *, config="deployment-1", cleanup=()):
    identity = {"config_id": config, "assignment_id": "assignment-1"}
    expected = {"worker": {"worker_prepare_identity": identity,
                           "phase": "commit" if phase in {"committing", "committing_global", "ready"} else "prepare"}}
    lifecycle = {"config_id": config, "phase": phase, "generation": 1,
                 "published": phase == "ready", "cleanup_ledger": list(cleanup)}
    return lifecycle, expected


def save(owner, phase, *, config="deployment-1", cleanup=(), authority=None, connections=None):
    lifecycle, expected = candidate(phase, config=config, cleanup=cleanup)
    receipts = {"worker": {"phase": "commit" if phase in {"committing_global", "ready"} else "prepare",
                            "loaded": True, "status": "ready", "identity": expected["worker"]["worker_prepare_identity"]}}
    return owner.save_execution(lifecycle, expected=expected, acknowledgements=receipts,
                                authority=authority or {"term": 1}, connections=connections or {"worker": 1})


def debt(action="release"):
    return {"cleanup_id": "cleanup-1", "node_id": "worker", "candidate_config_id": "deployment-1",
            "action": action, "payload": {"worker_prepare_identity": {
                "config_id": "deployment-1", "assignment_id": "assignment-1"},
                "config_id": "cleanup-1", "cleanup_action": action, "release": True}}


def test_prepare_and_commands_commit_in_same_transaction_and_queries_are_readonly(control):
    owner, repository, connect = control
    snapshot = save(owner, "preparing")
    assert snapshot["deployments"]["deployment-1"]["status"] == "preparing"
    assert len(snapshot["outbox"]) == 1
    command = next(iter(snapshot["outbox"].values()))
    assert owner.authorize_command("worker", command["payload"], authority={"term": 1}, connection_generation=1)
    before = repository.recovery_status()
    for _ in range(3):
        owner.snapshot()
        owner.runnable("deployment-1", authority={"term": 1}, connections={"worker": 1})
    assert repository.recovery_status() == before
    with connect() as connection:
        assert {r[0] for r in connection.execute("SELECT DISTINCT domain_id FROM model_control_events")} == {MODEL_CONTROL_DOMAIN}


def test_failed_transaction_exposes_neither_deployment_nor_outbox(control, monkeypatch):
    owner, repository, _ = control
    before = owner.snapshot()
    original = repository._transaction
    @contextmanager
    def full_disk():
        with original() as connection:
            yield connection
            raise OSError("injected disk full before commit")
    monkeypatch.setattr(repository, "_transaction", full_disk)
    with pytest.raises(OSError):
        save(owner, "preparing")
    assert owner.snapshot() == before
    assert owner.runnable("deployment-1", authority={"term": 1}, connections={"worker": 1}) is False


@pytest.mark.parametrize("phase", ["preparing", "committing_local", "committing", "committing_global", "ready", "aborted", "invalidated"])
def test_crash_never_restores_readiness_and_preserves_exact_debt(control, phase):
    owner, repository, _ = control
    save(owner, "preparing")
    save(owner, phase, cleanup=[debt()])
    restarted = DeploymentService(repository, boot_id="boot-2", clock=lambda: 2.0)
    after = restarted.start()
    assert after["active_deployment"] == ""
    assert after["cleanup"]["cleanup-1"] == debt()
    assert not restarted.runnable("deployment-1", authority={"term": 1}, connections={"worker": 1})
    assert all(c["state"] == "superseded" for c in after["outbox"].values() if c["action"] in {"prepare", "commit"})
    with pytest.raises(ModelControlError, match="boot_superseded"):
        save(owner, "preparing")


@pytest.mark.parametrize("changed", ["authority", "connection"])
def test_changed_inputs_cannot_commit_or_publish(control, changed):
    owner, _, _ = control
    before = save(owner, "preparing")
    args = {"authority": {"term": 2}} if changed == "authority" else {"connections": {"worker": 2}}
    with pytest.raises(ModelControlError, match="input_changed"):
        save(owner, "committing", **args)
    assert owner.snapshot() == before


def test_publish_requires_every_exact_commit_receipt(control):
    owner, _, _ = control
    save(owner, "preparing")
    lifecycle, expected = candidate("ready")
    with pytest.raises(ModelControlError, match="receipts_missing"):
        owner.save_execution(lifecycle, expected=expected, acknowledgements={}, authority={"term": 1}, connections={"worker": 1})
    assert owner.snapshot()["active_deployment"] == ""
    save(owner, "committing")
    save(owner, "ready")
    assert owner.runnable("deployment-1", authority={"term": 1}, connections={"worker": 1})
    assert not owner.runnable("deployment-1", authority={"term": 2}, connections={"worker": 1})
    assert not owner.runnable("deployment-1", authority={"term": 1}, connections={"worker": 2})


def test_new_selection_and_empty_execution_ledger_cannot_erase_cleanup(control):
    owner, _, _ = control
    save(owner, "preparing", cleanup=[debt("abort")])
    owner.save_operation({"operation_id": "op-1", "status": "preparing"}, select=True, expected_selected="")
    owner.save_operation({"operation_id": "op-2", "status": "accepted"}, select=True, expected_selected="op-1")
    save(owner, "aborted", cleanup=[])
    after = owner.snapshot()
    assert after["operations"]["op-1"]["status"] == "superseded"
    assert after["cleanup"]["cleanup-1"] == debt("abort")
    with pytest.raises(ModelControlError, match="receipt_mismatch"):
        owner.settle_cleanup("cleanup-1", evidence={"kind": "receipt", "action": "abort", "identity": {"assignment_id": "old"}})
    owner.settle_cleanup("cleanup-1", evidence={"kind": "receipt", "action": "abort", "identity": debt()["payload"]["worker_prepare_identity"]})
    assert not owner.snapshot()["cleanup"]
    owner.settle_cleanup("cleanup-1", evidence={"kind": "receipt"})  # Duplicate release is idempotent.


def test_selection_cas_and_terminal_history(control):
    owner, repository, _ = control
    owner.save_operation({"operation_id": "op-1", "status": "ready"}, select=True, expected_selected="", idempotency_key="key-1")
    before = owner.snapshot()
    with pytest.raises(ModelControlError, match="selection_conflict"):
        owner.save_operation({"operation_id": "op-2", "status": "accepted"}, select=True, expected_selected="")
    assert owner.snapshot() == before
    with pytest.raises(ModelControlError, match="terminal_conflict"):
        owner.save_operation({"operation_id": "op-1", "status": "superseded"})
    restarted = DeploymentService(repository, boot_id="boot-2", clock=lambda: 2.0)
    assert restarted.start()["operations"]["op-1"]["status"] == "ready"


def test_journal_corruption_fails_closed(control):
    owner, repository, connect = control
    save(owner, "preparing")
    assert repository.recovery_status()["verified"]
    with connect() as connection:
        connection.execute("UPDATE model_control_events SET previous_digest=? WHERE sequence=2", ("f" * 64,))
    with pytest.raises(ModelControlError, match="journal_corrupt"):
        repository.recovery_status()


@pytest.mark.parametrize("domain", ["configured", "removed_from_configuration"])
def test_required_floor_never_disappears_after_read_failure(domain):
    from types import SimpleNamespace
    from cluster_auto_role_runtime import JournalPrefixAuthority, AutoRoleRuntimeError
    from cluster_control_contract import CommitWatermark
    floor = CommitWatermark(workflow_id="model-control:v1", last_sequence=3, checkpoint_digest="a" * 64)
    prefix = JournalPrefixAuthority(SimpleNamespace(recovery_status=lambda _: {"verified": False}),
                                    workflows=(floor.workflow_id,) if domain == "configured" else ())
    prefix.floor_provider = lambda: (floor,)
    with pytest.raises(AutoRoleRuntimeError, match="floor is unavailable"):
        prefix.propose()


def test_model_recovery_floor_never_uses_workflow_replica(control):
    from types import SimpleNamespace
    from cluster_auto_role_runtime import JournalPrefixAuthority
    owner, repository, _ = control
    def forbidden(*_args, **_kwargs):
        pytest.fail("model control was routed to graph recovery")
    prefix = JournalPrefixAuthority(SimpleNamespace(recovery_status=forbidden, durable_prefix=forbidden))
    prefix.model_control = owner
    held = prefix.held(MODEL_CONTROL_DOMAIN)
    assert held is not None and held.last_sequence == repository.recovery_status()["durable_sequence"]
    prefix.enforce(SimpleNamespace(commit_watermark=(held,)))


def test_supply_acceptance_and_selected_pointer_commit_before_dispatch(control):
    owner, _, _ = control
    after = owner.save_supply({"operation_id": "op-1", "status": "queued", "intent": {"model_id": "tiny"},
                               "created_at": 1, "wait_for_nodes": True})
    assert after["selected_operation"] == "op-1"
    assert after["operations"]["op-1"]["phase"] == "accepted"
    assert after["operations"]["op-1"]["deadline"] == 3601


def test_missing_runtime_receipt_cannot_advance_production_candidate(stage_case):
    from tests.test_worker_prepare_coordinator import ack
    case = stage_case
    case.start()
    node = "worker"
    identity = deepcopy(case.master._pipeline_load_transaction["expected"][node]["worker_prepare_identity"])
    assert identity["load_spec_digest"]
    case.master._handle_worker_prepare_ack(node, {"action": "ack", "phase": "prepare", "identity": identity,
        "status": "prepared", "loaded": True})
    assert case.master._pipeline_load_transaction["prepared_nodes"] == set()
    ack(case.master, node, "prepare")
    assert case.master._pipeline_load_transaction["prepared_nodes"] == {node}


@pytest.mark.parametrize("field", ["device", "dtype", "quant_type", "context_size", "kv_dtype", "components", "materialization"])
def test_changed_actual_policy_receipt_is_rejected_before_commit(stage_case, field):
    from tests.test_worker_prepare_coordinator import ack
    case = stage_case
    case.start()
    expected = case.master._pipeline_load_transaction["expected"]["worker"]
    receipt = {"load_spec": deepcopy(expected["load_spec"]), "materialized_bytes": 1,
        "memory_scope": "weight_tensors", "layer_range": expected["worker_prepare_identity"]["layer_range"], "artifact_sha256": "b" * 64}
    receipt["load_spec"][field] = {"device": "cuda:0", "dtype": "float32", "quant_type": "fp32", "context_size": 32,
                                   "kv_dtype": "float32", "components": ["layers", "embedding"], "materialization": "other"}[field]
    ack(case.master, "worker", "prepare", runtime_receipt=receipt)
    assert case.master._pipeline_load_transaction["prepared_nodes"] == set()


def test_oversized_actual_materialization_aborts_instead_of_publishing(stage_case):
    from tests.test_worker_prepare_coordinator import ack
    case = stage_case
    case.start()
    expected = case.master._pipeline_load_transaction["expected"]["worker"]
    expected["required_bytes"] = 4
    receipt = {"load_spec": deepcopy(expected["load_spec"]), "materialized_bytes": 5,
        "memory_scope": "weight_tensors", "layer_range": expected["worker_prepare_identity"]["layer_range"], "artifact_sha256": "b" * 64}
    ack(case.master, "worker", "prepare", runtime_receipt=receipt)
    assert case.master._pipeline_load_transaction["phase"] == "aborted"
    assert case.master._active_pipeline_capacity_plan is None


def test_native_stage_identity_includes_load_spec_digest(stage_case):
    case = stage_case
    case.start()
    expected = case.master._pipeline_load_transaction["expected"]["worker"]
    assert expected["prepare_identity"]["load_spec_digest"] == expected["worker_prepare_identity"]["load_spec_digest"]


from tests.test_pipeline_retirement import retirement_case
from tests.test_worker_prepare_coordinator import stage_case


@pytest.mark.parametrize("kind", ["operation", "supply"])
def test_stale_boot_cannot_update_selection_or_supply(control, kind):
    owner, repository, _ = control
    DeploymentService(repository, boot_id="replacement", clock=lambda: 2).start()
    before = repository.read()
    with pytest.raises(ModelControlError, match="boot_superseded"):
        if kind == "operation":
            owner.save_operation({"operation_id": "stale", "status": "queued"}, select=True)
        else:
            owner.save_supply({"operation_id": "stale", "status": "queued"})
    assert repository.read() == before


def test_publish_checks_exact_identity_even_with_loaded_commit_flag(control):
    owner, _, _ = control
    save(owner, "preparing")
    lifecycle, expected = candidate("ready")
    before = owner.snapshot()
    with pytest.raises(ModelControlError, match="identity_mismatch"):
        owner.save_execution(lifecycle, expected=expected, acknowledgements={"worker": {
            "phase": "commit", "loaded": True, "status": "ready", "identity": {"assignment_id": "other"}}},
            authority={"term": 1}, connections={"worker": 1})
    assert owner.snapshot() == before


def test_frozen_native_assignment_cannot_change_between_phases(stage_case):
    case = stage_case; case.start()
    owner = case.master._model_control
    before = owner.snapshot()
    lifecycle = case.master._pipeline_lifecycle_snapshot_locked()
    expected = deepcopy(case.master._pipeline_load_transaction["expected"])
    expected["worker"]["required_bytes"] = 999
    with pytest.raises(ModelControlError, match="assignment_changed"):
        owner.save_execution(lifecycle, expected=expected, acknowledgements={},
            authority=case.master._model_control_authority(), connections=case.master._model_control_connections())
    assert owner.snapshot() == before


def test_native_owner_rejects_publish_before_prepare_and_commit(stage_case):
    case = stage_case; case.start()
    owner = case.master._model_control
    lifecycle = case.master._pipeline_lifecycle_snapshot_locked()
    lifecycle.update(phase="ready", published=True,
                     input_revisions=owner.snapshot()["lifecycle"]["input_revisions"])
    with pytest.raises(ModelControlError, match="transition_invalid"):
        owner.save_execution(lifecycle, expected=case.master._pipeline_load_transaction["expected"], acknowledgements={},
            authority=case.master._model_control_authority(), connections=case.master._model_control_connections())
    assert owner.snapshot()["active_deployment"] == ""


def test_local_materialization_receipt_is_exact_before_remote_commit(stage_case):
    from tests.test_worker_prepare_coordinator import prepare_both
    case = stage_case; case.start(); prepare_both(case)
    state = case.master._model_control.snapshot()
    local = state["lifecycle"]["local_assignment"]
    receipt = state["lifecycle"]["local_receipt"]
    assert local["load_spec"]["quant_type"] == "fp32"
    assert receipt["load_spec"] == local["load_spec"]
    assert receipt["materialized_bytes"] == 16
    assert receipt["layer_range"] == [0, 4]
    assert receipt["artifact_sha256"] == "a" * 64
    assert state["active_deployment"] == ""


def test_delivery_observation_is_durable_and_old_connection_cannot_replay(control):
    owner, repository, _ = control
    state = save(owner, "preparing")
    command = next(iter(state["outbox"].values()))
    assert not owner.authorize_command("worker", command["payload"], authority={"term": 1}, connection_generation=2)
    owner.note_delivery("worker", command["payload"], delivered=False)
    observed = repository.read()["outbox"][command["command_id"]]
    assert observed["attempts"] == 1 and observed["delivered"] is False
    assert owner.authorize_command("worker", command["payload"], authority={"term": 1}, connection_generation=1)


def test_disk_failure_fence_is_owned_readonly_and_explicitly_volatile(control):
    owner, repository, _ = control
    owner.save_operation({"operation_id": "op", "status": "queued"}, select=True)
    before = repository.read()
    owner.fence_operation_failure({"operation_id": "op", "status": "cancelled", "reason_code": "disk_full"})
    assert owner.snapshot()["operations"]["op"]["durable"] is False
    assert repository.read() == before
    owner.save_operation({"operation_id": "op", "status": "cancelled", "reason_code": "disk_full"})
    assert owner.snapshot()["operations"]["op"]["reason_code"] == "disk_full"
    assert "durable" not in repository.read()["operations"]["op"]


def test_native_local_policy_keeps_one_weight_owner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from llama_engine import LlamaCppEngine
    from qlh_core.planning.load_spec import assignment_load_spec
    import qlh_core.adapters.llama_vocabulary as vocabulary
    artifact = tmp_path / "tiny-head.gguf"
    artifact.write_bytes(b"tiny mock native boundary")
    engine = LlamaCppEngine()
    engine._pipeline_descriptor = {"model_path": str(artifact), "total_layers": 8, "pipeline_runtime_supported": True}
    engine._pipeline_distributed_only = True
    monkeypatch.setattr(engine, "_verify_catalog_artifact", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_find_layer_artifact", lambda *a: str(artifact))
    monkeypatch.setattr(engine, "_load_model_runtime", lambda *a, **k: pytest.fail("second native weight context loaded"))
    monkeypatch.setattr(vocabulary, "LlamaVocabulary", lambda path: SimpleNamespace(close=lambda: None, metadata={}))
    observed = []
    upstream = SimpleNamespace(is_available=True, close=lambda: None)
    def runner():
        observed.append(engine._runtime_load_spec)
        engine._keep_head = upstream
        return upstream
    monkeypatch.setattr(engine, "_keep_head_upstream", runner)
    policy = assignment_load_spec({"has_embedding": True}, artifact_format="gguf")
    engine.load_layer_range(0, 4, has_embedding=True, load_spec=policy.to_dict())
    assert engine.is_loaded and observed == [policy]
    assert engine._pipeline_descriptor["assignment_layer_range"] == [0, 4]
    engine.unload()
    assert engine._keep_head is None and not engine.is_loaded


@pytest.mark.parametrize("changed", ["inputs", "local_receipt"])
def test_changed_commit_inputs_abort_before_reshard_publication(stage_case, monkeypatch, changed):
    from tests.test_worker_prepare_coordinator import prepare_both, ack
    case = stage_case; case.start(); prepare_both(case)
    tx = case.master._pipeline_load_transaction
    calls = []
    monkeypatch.setattr(case.master, "_commit_ready_pipeline_reshard", lambda plan: calls.append(plan))
    if changed == "local_receipt":
        tx["local_receipt"]["load_spec"]["dtype"] = "float16"
    else:
        frozen = deepcopy(tx["model_control_inputs"])
        frozen["catalog"] += 1
        monkeypatch.setattr(case.master, "_model_control_input_revisions", lambda _: frozen)
    ack(case.master, "worker", "commit"); ack(case.master, "android", "commit")
    assert calls == []
    assert tx["phase"] == "aborted"
    assert case.master._model_control.snapshot()["active_deployment"] == ""


def test_expired_operation_cannot_be_published(stage_case):
    from tests.test_worker_prepare_coordinator import prepare_both, ack
    case = stage_case
    owner = case.master._model_control
    owner.start()
    owner.save_supply({"operation_id": "supply_test", "status": "queued", "intent": {"timeout_seconds": 1},
                       "created_at": 1, "wait_for_nodes": True})
    case.start(); prepare_both(case)
    ack(case.master, "worker", "commit"); ack(case.master, "android", "commit")
    assert case.master._pipeline_load_transaction["phase"] == "aborted"
    assert owner.snapshot()["active_deployment"] == ""


@pytest.mark.parametrize("same_intent", [True, False])
def test_supply_idempotency_is_atomic_across_two_owners(control, same_intent):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    owner, repository, _ = control
    second = DeploymentService(repository, boot_id=owner.boot_id, clock=lambda: 1.0)
    second.start()
    barrier = threading.Barrier(2)
    def accept(index):
        value = {"operation_id": f"operation-{index}", "status": "queued", "wait_for_nodes": True,
                 "idempotency_key": "one-key", "created_at": 1.0,
                 "intent": {"model_id": "tiny" if same_intent else f"tiny-{index}"}}
        barrier.wait()
        try:
            (owner if index == 0 else second).save_supply(value)
            return "accepted"
        except ModelControlError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(accept, range(2)))
    expected = "operation_idempotency_reused" if same_intent else "operation_idempotency_conflict"
    assert sorted(outcomes) == sorted(["accepted", expected])
    state = owner.snapshot()
    assert len(state["supplies"]) == len(state["operations"]) == 1
    assert state["selected_operation"] in state["operations"]
