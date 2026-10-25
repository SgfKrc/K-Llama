"""Prepare barriers validate independent CPU/CUDA budgets without native I/O."""
from copy import deepcopy
import threading
from types import SimpleNamespace as NS

import pytest

from qlh_core.planning.load_spec import assignment_load_spec
from tests.test_worker_prepare import identity
from worker_prepare_contract import WorkerPrepareError, validate_ack, validate_runtime_receipt
from worker_prepare_coordinator import WorkerPrepareCoordinatorMixin


def case(*, phase="prepare", offload=True):
    assignment = {"execution_device": "cuda", "required_bytes": 100, "capacity_bytes": 120,
                  "host_required_bytes": 330, "host_capacity_bytes": 600, "kv_cache_bytes": 20}
    if offload:
        assignment["runtime_materialization"] = "safetensors_layer_offload"
    spec = assignment_load_spec(assignment, artifact_format="safetensors", context_size=64)
    value = identity()
    value["load_spec_digest"] = spec.digest
    assignment.update(load_spec=spec.to_dict(), worker_prepare_identity=value, phase=phase)
    receipt = {"load_spec": spec.to_dict(), "materialized_bytes": 300 if offload else 80,
               "memory_scope": "weight_tensors", "layer_range": value["layer_range"],
               "artifact_sha256": value["artifact_sha256"]}
    if offload:
        receipt.update(host_source_bytes=300, host_conversion_bytes=30, device_weight_bytes=80)
    payload = {"action": "ack", "phase": phase, "identity": deepcopy(value), "loaded": True,
               "status": "prepared" if phase == "prepare" else "ready", "runtime_receipt": receipt}
    advanced, aborted = [], []
    transaction = {"expected": {"pc": assignment}, "config_id": value["config_id"],
                   "phase": "preparing" if phase == "prepare" else "committing",
                   "connection_generations": {"pc": 11}, "worker_ids": {"pc"}}
    host = NS(_layer_config_lock=threading.RLock(), _pipeline_load_transaction=transaction,
              _assignment_connection_generation=lambda node: 11, _pipeline_worker_opt_out=set(),
              _persist_pipeline_lifecycle_locked=lambda: None,
              _advance_worker_prepare_transaction=lambda *args, **kwargs: advanced.append(args),
              _abort_pipeline_load_transaction=lambda *args: aborted.append(args))
    return NS(host=host, assignment=assignment, receipt=receipt, payload=payload,
              transaction=transaction, advanced=advanced, aborted=aborted)


def handle(value):
    WorkerPrepareCoordinatorMixin._handle_worker_prepare_ack(value.host, "pc", value.payload)


@pytest.mark.parametrize("phase", ["prepare", "commit"])
def test_real_host_total_larger_than_vram_passes_at_independent_exact_boundaries(phase):
    value = case(phase=phase)
    assert value.receipt["materialized_bytes"] > value.assignment["required_bytes"]
    handle(value)
    key = "prepared_nodes" if phase == "prepare" else "ready_nodes"
    assert value.transaction[key] == {"pc"}
    assert value.advanced == [(value.payload["identity"]["config_id"], phase)]
    assert not value.aborted


@pytest.mark.parametrize("field,amount", [("device_weight_bytes", 81), ("host_conversion_bytes", 31)])
@pytest.mark.parametrize("phase", ["prepare", "commit"])
def test_shortage_in_either_domain_aborts_before_native_barrier(field, amount, phase):
    value = case(phase=phase)
    value.receipt[field] = amount
    handle(value)
    assert not value.advanced
    assert value.aborted[0][2] == "deployment_materialization_budget_exceeded"
    assert value.transaction["acks"]["pc"]["status"] == "error"


@pytest.mark.parametrize("field,amount", [("host_required_bytes", None), ("host_capacity_bytes", None),
                                          ("host_required_bytes", 601), ("host_required_bytes", True)])
def test_offload_requires_valid_frozen_host_admission(field, amount):
    value = case()
    value.assignment[field] = amount
    handle(value)
    assert not value.advanced
    assert value.aborted[0][2] == "deployment_host_budget_missing"


def test_complete_frozen_policy_is_checked_in_coordinator():
    value = case()
    value.assignment["load_spec"]["context_size"] += 1
    handle(value)
    assert not value.advanced
    assert value.aborted[0][2] == "deployment_runtime_receipt_mismatch"


@pytest.mark.parametrize("field,amount", [("device_weight_bytes", -1), ("device_weight_bytes", 0),
    ("device_weight_bytes", True), ("host_source_bytes", 299), ("host_conversion_bytes", True),
    ("unknown", 1)])
def test_malformed_offload_domain_fields_cannot_cross_wire_or_advance(field, amount):
    value = case()
    value.receipt[field] = amount
    with pytest.raises(WorkerPrepareError, match="prepare_runtime_receipt_invalid"):
        validate_ack(value.payload)
    handle(value)
    assert not value.advanced and not value.aborted


@pytest.mark.parametrize("field", ["device_weight_bytes", "host_source_bytes", "host_conversion_bytes"])
def test_offload_wire_requires_all_three_observed_weight_domains(field):
    value = case()
    value.receipt.pop(field)
    with pytest.raises(WorkerPrepareError, match="prepare_runtime_receipt_invalid"):
        validate_runtime_receipt(value.receipt, identity=value.payload["identity"])


@pytest.mark.parametrize("materialized,accepted", [(80, True), (81, False)])
def test_resident_scalar_weights_plus_kv_check_remains_strict(materialized, accepted):
    value = case(offload=False)
    value.receipt["materialized_bytes"] = materialized
    handle(value)
    assert bool(value.advanced) == accepted
    if not accepted:
        assert value.aborted[0][2] == "prepare_materialization_budget_exceeded"


def test_resident_wire_still_rejects_extra_offload_fields():
    value = case(offload=False)
    value.receipt["device_weight_bytes"] = 80
    with pytest.raises(WorkerPrepareError, match="prepare_runtime_receipt_invalid"):
        validate_ack(value.payload)
