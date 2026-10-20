"""Late Surface cleanup receipts preserve durable deployment retirement."""
from copy import deepcopy

import pytest

from qlh_core.contracts.deployment import ModelControlError
from qlh_core.control.deployment import DeploymentService
from tests.test_pipeline_retirement import retirement_case
from tests.test_worker_prepare_coordinator import stage_case, publish, prepare_both, ack

pytestmark = pytest.mark.refactor_contract


def cleanup_ack(master, record, **changes):
    action = record["action"]
    payload = {"action": "ack", "phase": action,
               "identity": deepcopy(record["payload"]["worker_prepare_identity"]),
               "cleanup_id": record["cleanup_id"],
               "status": "finalized" if action == "finalize" else "released",
               "loaded": action == "finalize"}
    payload.update(changes)
    master._handle_worker_prepare_ack(record["node_id"], payload)


def finish_finalizes(master):
    for record in list(master._pipeline_model_cleanup_ledger.values()):
        cleanup_ack(master, record)
    assert master._pipeline_model_cleanup_ledger == {}
    assert master._pipeline_lifecycle_persist_ok


def terminal_case(case, kind):
    master = case.master
    if kind == "release":
        publish(case)
        finish_finalizes(master)
        master._invalidate_pipeline_load_transaction("pipeline_worker_opted_out")
    else:
        case.start()
        master._abort_pipeline_load_transaction(master._pipeline_load_transaction["config_id"], "test_abort")
    assert master._pipeline_lifecycle_persist_ok
    assert not master._pipeline_recovery_pending
    records = list(master._pipeline_model_cleanup_ledger.values())
    assert len(records) == 2
    assert {record["action"] for record in records} == {kind}
    return records


def save_current(owner, master, lifecycle=None, **changes):
    transaction = master._pipeline_load_transaction
    arguments = {"expected": transaction["expected"], "acknowledgements": transaction.get("acks", {}),
                 "authority": master._model_control_authority(),
                 "connections": master._model_control_connections(transaction)}
    arguments.update(changes)
    return owner.save_execution(lifecycle or owner.snapshot()["lifecycle"], **arguments)


@pytest.mark.parametrize("kind,terminal", [("release", "retired"), ("abort", "aborted")])
def test_owner_terminal_cleanup_snapshot_is_monotonic_and_durable(stage_case, kind, terminal):
    master = stage_case.master
    records = terminal_case(stage_case, kind)
    owner = master._model_control
    config = master._pipeline_load_transaction["config_id"]
    for record in records:
        owner.settle_cleanup(record["cleanup_id"], evidence={"kind": "receipt", "action": kind,
                             "identity": record["payload"]["worker_prepare_identity"]})
    settled = owner.snapshot()
    assert settled["deployments"][config]["status"] == terminal
    assert settled["lifecycle"]["cleanup_ledger"] == []

    saved = save_current(owner, master)

    assert saved["deployments"][config]["status"] == terminal
    assert saved["active_deployment"] == ""
    assert saved["cleanup"] == {}
    reopened = DeploymentService(owner.repository, boot_id=owner.boot_id, clock=owner.clock)
    assert reopened.start() == saved
    assert owner.repository.recovery_status()["verified"]
    assert not reopened.runnable(config, authority=master._model_control_authority(),
                                 connections=master._model_control_connections())


@pytest.mark.parametrize("kind", ["release", "abort"])
def test_last_cleanup_ack_does_not_poison_next_scheduler_transaction(stage_case, kind):
    case = stage_case
    master = case.master
    records = terminal_case(case, kind)
    old_config = master._pipeline_load_transaction["config_id"]
    for record in sorted(records, key=lambda item: item["node_id"] == "worker"):
        cleanup_ack(master, record)
        assert master._pipeline_lifecycle_persist_ok
        assert not master._pipeline_recovery_pending
    assert master._model_control.snapshot()["cleanup"] == {}
    assert master._model_control.snapshot()["lifecycle"]["cleanup_ledger"] == []

    # A new cut of the same model/range may have different artifact bytes.
    # It is a fresh deployment, so a completed cleanup must not create a
    # restart-recovery digest target for the old artifact.
    candidate = "candidate-after-surface-rejoin"
    descriptor = case.host.prepare_pipeline_candidate(
        candidate, "retirement-model", str(case.model_path),
        quant_type="fp16", model_sha256="f" * 64,
    )
    result = master.start_pipeline_model_transaction(
        candidate_id=candidate, descriptor=descriptor,
        model_path=str(case.model_path), quant_type="fp16", operation_id="supply_test",
    )
    assert result["transaction_phase"] == "preparing"
    assert master._pipeline_load_transaction["plan"]["model_sha256"] == "f" * 64
    assert master._pipeline_load_transaction["plan"]["assignments"][0]["start_layer"] == 0
    assert master._pipeline_load_transaction["plan"]["assignments"][0]["end_layer"] == 4
    new_config = master._pipeline_load_transaction["config_id"]
    assert new_config != old_config
    before = master._model_control.snapshot()
    # An old ACK and its duplicate may not alter the new generation.
    for record in records:
        cleanup_ack(master, record)
        ack(master, record["node_id"], "prepare", identity=record["payload"]["worker_prepare_identity"])
    assert master._model_control.snapshot() == before
    prepare_both(case)
    ack(master, "worker", "commit")
    ack(master, "android", "commit")
    assert master._pipeline_load_transaction["phase"] == "ready"
    assert master._model_control.snapshot()["active_deployment"] == new_config
    assert master._pipeline_lifecycle_persist_ok
    assert not master._pipeline_recovery_pending


def test_surface_opt_out_reconnect_release_keeps_old_generation_fenced(stage_case, monkeypatch):
    case = stage_case
    master = case.master
    publish(case)
    finish_finalizes(master)
    old_config = master._pipeline_load_transaction["config_id"]
    case.connected.remove("worker")
    master._handle_layer_worker_opt_out("worker", {"data": {"node_id": "worker"}})
    assert master._pipeline_load_transaction["phase"] == "invalidated"
    records = list(master._pipeline_model_cleanup_ledger.values())
    for record in records:
        wrong = deepcopy(record["payload"]["worker_prepare_identity"])
        wrong["generation"] -= 1
        before = master._model_control.snapshot()
        cleanup_ack(master, record, identity=wrong)
        assert master._model_control.snapshot() == before
        assert record["cleanup_id"] in master._pipeline_model_cleanup_ledger
    case.connected.add("worker")
    monkeypatch.setattr(master, "_assignment_connection_generation", lambda node: 12 if node == "worker" else 11)
    master._pipeline_worker_opt_out.discard("worker")
    for record in sorted(records, key=lambda item: item["node_id"] == "worker"):
        cleanup_ack(master, record)
    assert master._pipeline_lifecycle_persist_ok
    assert not master._pipeline_recovery_pending
    assert master._model_control.snapshot()["deployments"][old_config]["status"] == "retired"
    assert case.start("candidate-after-surface-reconnect")["transaction_phase"] == "preparing"


@pytest.mark.parametrize("changed", ["ready", "generation", "assignment"])
def test_settled_deployment_replay_cannot_publish_or_change_generation(stage_case, changed):
    master = stage_case.master
    records = terminal_case(stage_case, "release")
    owner = master._model_control
    for record in records:
        owner.settle_cleanup(record["cleanup_id"], evidence={"kind": "receipt", "action": "release",
                             "identity": record["payload"]["worker_prepare_identity"]})
    before = owner.snapshot()
    lifecycle = deepcopy(before["lifecycle"])
    expected = deepcopy(master._pipeline_load_transaction["expected"])
    if changed == "ready":
        lifecycle["phase"] = "ready"
        code = "deployment_retired"
    elif changed == "generation":
        lifecycle["generation"] -= 1
        code = "deployment_transition_invalid"
    else:
        expected["worker"]["worker_prepare_identity"]["generation"] -= 1
        code = "deployment_assignment_changed"
    with pytest.raises(ModelControlError, match=code):
        save_current(owner, master, lifecycle, expected=expected)
    assert owner.snapshot() == before


def test_recovery_cleanup_snapshot_cannot_reopen_retired_deployment(stage_case):
    master = stage_case.master
    records = terminal_case(stage_case, "release")
    owner = master._model_control
    for record in records:
        owner.settle_cleanup(record["cleanup_id"], evidence={"kind": "receipt", "action": "release",
                             "identity": record["payload"]["worker_prepare_identity"]})
    config = master._pipeline_load_transaction["config_id"]

    saved = save_current(owner, master, expected={}, acknowledgements={}, recovery_input=True)

    assert saved["deployments"][config]["status"] == "retired"
    assert saved["cleanup"] == {}
    assert saved["active_deployment"] == ""


def test_explicit_retirement_after_reboot_preserves_settled_deployment(stage_case):
    master = stage_case.master
    records = terminal_case(stage_case, "release")
    for record in records:
        cleanup_ack(master, record)
    owner = master._model_control
    settled = owner.snapshot()
    config = settled["lifecycle"]["config_id"]
    assert settled["deployments"][config]["status"] == "retired"
    # Reboot restores historical publication before an explicit model change
    # retires the recovery target. Unrelated native cleanup debts stay owned.
    master._pipeline_recovery_state = deepcopy(settled["lifecycle"])
    master._pipeline_load_transaction = None
    master._active_pipeline_capacity_plan = None
    master._pipeline_recovery_validated = True
    master._pipeline_recovery_pending = True
    master._pipeline_recovery_failure = "pipeline_lifecycle_persist_failed"
    master._pipeline_lifecycle_persist_ok = False
    master._running = True

    assert master.retire_pipeline_recovery_for_model_change()

    saved = owner.snapshot()
    assert saved["deployments"][config]["status"] == "retired"
    assert saved["active_deployment"] == ""
    assert saved["cleanup"] == settled["cleanup"]
    assert saved["lifecycle"]["published"] is False
    assert saved["lifecycle"]["generation"] == settled["lifecycle"]["generation"]
    assert saved["lifecycle"]["model_sha256"] == settled["lifecycle"]["model_sha256"]
    assert master._pipeline_lifecycle_persist_ok
    assert not master._pipeline_recovery_pending
    assert owner.repository.recovery_status()["verified"]


@pytest.mark.parametrize("changed", ["generation", "phase", "live_abort", "model_sha256", "assignment"])
def test_explicit_retirement_cannot_reclassify_a_different_terminal_transaction(stage_case, changed):
    master = stage_case.master
    records = terminal_case(stage_case, "release")
    for record in records:
        cleanup_ack(master, record)
    owner = master._model_control
    before = owner.snapshot()
    lifecycle = deepcopy(before["lifecycle"])
    lifecycle["published"] = False
    if changed == "generation":
        lifecycle["generation"] -= 1
    elif changed == "phase":
        lifecycle["phase"] = "aborted"
    elif changed == "model_sha256":
        lifecycle["model_sha256"] = "f" * 64
    elif changed == "assignment":
        lifecycle["assignments"][0]["end_layer"] -= 1
    recovery_input = changed != "live_abort"
    if changed == "live_abort":
        lifecycle["phase"] = "aborted"
    with pytest.raises(ModelControlError, match="deployment_transition_invalid"):
        save_current(owner, master, lifecycle, expected={}, acknowledgements={}, recovery_input=recovery_input)
    assert owner.snapshot() == before


@pytest.mark.parametrize("completion", ["explicit_retirement", "fresh_ready"])
def test_completed_restart_identity_cannot_poison_second_model_retirement(stage_case, monkeypatch, completion):
    case = stage_case
    master = case.master
    records = terminal_case(case, "release")
    for record in records:
        cleanup_ack(master, record)
    owner = master._model_control
    recovered = deepcopy(owner.snapshot()["lifecycle"])
    master._pipeline_recovery_state = recovered
    master._worker_assignments.clear()  # A reboot has no live transport assignments.
    master._pipeline_recovery_validated = True
    master._pipeline_recovery_pending = True
    master._running = True
    master._pipeline_load_transaction = None
    master._active_pipeline_capacity_plan = None
    if completion == "explicit_retirement":
        assert master.retire_pipeline_recovery_for_model_change()
    # Publish a genuinely new generation through the normal prepare/commit ACKs.
    assert case.start("candidate-second-generation")["transaction_phase"] == "preparing"
    prepare_both(case)
    ack(master, "worker", "commit")
    ack(master, "android", "commit")
    current = master._pipeline_load_transaction["config_id"]
    assert current != recovered["config_id"]
    finish_finalizes(master)
    if completion == "fresh_ready":
        # This native-ACK fixture has no remote stage-offer provider. The fence
        # still waits for that independent readiness observation after commit.
        master._maybe_finish_pipeline_recovery()
        assert master._pipeline_recovery_pending
        monkeypatch.setattr(master, "_stage_offer_assignment_ready", lambda _node, _assignment: (True, "ready"))
    master._maybe_finish_pipeline_recovery()
    assert not master._pipeline_recovery_pending, {
        "reason": master._pipeline_recovery_failure,
        "registry": {n: {k: v.get(k) for k in ("phase", "reason_code", "expected_config")} for n, v in master._worker_assignments.snapshot().items()},
        "stage": [(v["node_id"], master._stage_offer_assignment_ready(v["node_id"], v)) for v in master._pipeline_load_transaction["plan"]["assignments"] if v.get("execution") == "stage_offer_v3"],
        "phase": master._pipeline_load_transaction["phase"],
        "persisted": master._pipeline_lifecycle_persist_ok,
    }
    master._invalidate_pipeline_load_transaction("next_model_change")
    for record in list(master._pipeline_model_cleanup_ledger.values()):
        cleanup_ack(master, record)
    master._pipeline_load_transaction = None
    master._active_pipeline_capacity_plan = None
    before = owner.snapshot()

    assert master.retire_pipeline_recovery_for_model_change()

    assert master._pipeline_recovery_state is None
    assert master._pipeline_lifecycle_persist_ok
    assert not master._pipeline_recovery_pending
    assert owner.snapshot() == before
    assert before["lifecycle"]["config_id"] == current
    assert before["deployments"][current]["status"] == "retired"


@pytest.mark.parametrize("write_fails", [False, True])
def test_recovery_identity_completion_preserves_unsettled_native_debts(stage_case, monkeypatch, write_fails):
    master = stage_case.master
    terminal_case(stage_case, "release")
    owner = master._model_control
    recovered = deepcopy(owner.snapshot()["lifecycle"])
    debts = deepcopy(master._pipeline_model_cleanup_ledger)
    assert debts
    master._pipeline_recovery_state = recovered
    master._pipeline_recovery_validated = True
    master._pipeline_recovery_pending = True
    master._pipeline_load_transaction = None
    master._active_pipeline_capacity_plan = None
    master._running = True
    if write_fails:
        def reject(_state):
            raise OSError("injected owner persistence failure")
        monkeypatch.setattr(master, "_save_model_control_lifecycle", reject)

    assert master.retire_pipeline_recovery_for_model_change() is not write_fails

    assert master._pipeline_model_cleanup_ledger == debts
    assert master._pipeline_recovery_pending is write_fails
    if write_fails:
        assert master._pipeline_recovery_state["config_id"] == recovered["config_id"]
        assert owner.snapshot()["lifecycle"] == recovered
    else:
        assert master._pipeline_recovery_state is None
        # A subsequent debt-only persistence cannot resurrect the retired model.
        master._persist_pipeline_lifecycle_locked()
        assert master._pipeline_lifecycle_persist_ok
        saved = owner.snapshot()
        assert not saved["lifecycle"]["config_id"]
        assert set(saved["cleanup"]) == set(debts)
        assert {item["cleanup_id"] for item in saved["lifecycle"]["cleanup_ledger"]} == set(debts)
