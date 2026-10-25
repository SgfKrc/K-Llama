"""Offloaded host residency and CUDA execution obey independent frozen budgets."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from qlh_core.adapters.runtime_observation import observe_runtime
from qlh_core.contracts.deployment import ModelControlError, require_runtime_receipt
from qlh_core.planning.load_spec import assignment_load_spec


def assignment():
    value = {"execution_device": "cuda", "runtime_materialization": "safetensors_layer_offload",
             "has_embedding": True, "has_lm_head": False, "required_bytes": 100,
             "capacity_bytes": 120, "host_required_bytes": 500, "host_capacity_bytes": 600}
    policy = assignment_load_spec(value, artifact_format="safetensors", context_size=64)
    value.update(load_spec=policy.to_dict(), identity={"layer_range": [0, 2],
                 "artifact_sha256": "a" * 64, "load_spec_digest": policy.digest})
    return value


def manager(**changes):
    observed = {"execution_device": "cuda:0", "execution_dtype": "float16",
                "resident_source_bytes": 300, "device_weight_bytes": 60,
                "host_conversion_bytes": 30}
    observed.update(changes)
    return SimpleNamespace(layer_range=(0, 2), is_loaded=True, quant_type="fp16",
                           layer_offload_observation=lambda: deepcopy(observed))


def test_actual_offload_observation_keeps_host_total_larger_than_vram():
    value = assignment()
    receipt = observe_runtime(manager(), value)
    assert receipt["materialized_bytes"] == receipt["host_source_bytes"] == 300
    assert receipt["device_weight_bytes"] == 60
    assert receipt["host_conversion_bytes"] == 30
    assert receipt["load_spec"]["materialization"] == "safetensors_layer_offload"
    require_runtime_receipt(value, receipt)


@pytest.mark.parametrize("changes", [{"device_weight_bytes": 101},
                                     {"resident_source_bytes": 490, "host_conversion_bytes": 11}])
def test_real_shortage_in_either_memory_domain_rejects(changes):
    with pytest.raises(ModelControlError, match="budget_exceeded"):
        observe_runtime(manager(**changes), assignment())


@pytest.mark.parametrize("field,value", [("host_required_bytes", 601),
                                         ("host_capacity_bytes", None),
                                         ("host_required_bytes", True)])
def test_host_budget_cannot_be_missing_or_larger_than_sample(field, value):
    frozen = assignment()
    frozen[field] = value
    with pytest.raises(ModelControlError, match="host_budget_missing"):
        observe_runtime(manager(), frozen)


@pytest.mark.parametrize("field,value", [("device_weight_bytes", -1),
                                         ("host_conversion_bytes", True),
                                         ("host_source_bytes", 299)])
def test_forged_memory_domain_receipts_are_rejected(field, value):
    frozen = assignment()
    receipt = observe_runtime(manager(), frozen)
    receipt[field] = value
    with pytest.raises(ModelControlError, match="receipt_mismatch"):
        require_runtime_receipt(frozen, receipt)


def test_offload_policy_changes_digest_and_cannot_be_applied_to_cpu():
    frozen = assignment()
    resident = dict(frozen)
    resident.pop("load_spec")
    resident["runtime_materialization"] = "safetensors_key_filtered"
    assert assignment_load_spec(resident, artifact_format="safetensors").digest != frozen["identity"]["load_spec_digest"]
    resident.update(execution_device="cpu", runtime_materialization="safetensors_layer_offload")
    with pytest.raises(ValueError, match="device_unsupported"):
        assignment_load_spec(resident, artifact_format="safetensors")
