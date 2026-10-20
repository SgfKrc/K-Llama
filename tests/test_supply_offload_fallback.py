"""The real supply coordinator selects offload only after resident admission fails."""
from copy import deepcopy

import pytest

from tests.test_model_supply import startup_pool_case
from tests.test_model_supply_runtime_capacity import _write_shard
from model_supply import ModelSupplyError
from model_supply_capacity import bind_supplied_offload_capacity, project_inventory_capacity


def configure(case, tmp_path, *, supported=True, cuda_available=20):
    case.c._operations[case.op]["intent"].update(engine="pytorch", timeout_seconds=1)
    case.engines = {"master": ["pytorch"], "worker": ["llama_cpp"]}
    case.budgets = {"master": 1000, "worker": 60}
    case.runtime_source = tmp_path / "hf"
    case.runtime_source.mkdir()
    _write_shard(case.runtime_source, {f"model.layers.{i}.weight": ("F16", [5]) for i in range(6)})
    original = case.scheduler._model_supply_worker.inventory
    def inventory(**kwargs):
        result = original(**kwargs)
        result["pytorch_runtime"].update(execution_device="cuda", available_bytes=cuda_available,
            materializations=["safetensors_key_filtered"] + (["safetensors_layer_offload"] if supported else []))
        return result
    case.scheduler._model_supply_worker.inventory = inventory
    case.release_source.set()
    return inventory


def test_resident_failure_uses_real_host_and_unit_device_quotes(startup_pool_case, tmp_path):
    case = startup_pool_case
    inventory = configure(case, tmp_path)
    runner = case.start()
    runner.join(3)
    assert not runner.is_alive()
    operation = case.c.get(case.op)
    assert operation["status"] == "available", operation
    plan = operation["plan"]
    assert plan["resident_capacity_diagnostic"]["reason_code"]
    master = plan["assignments"][0]
    assert master["load_spec"]["materialization"] == "safetensors_layer_offload"
    assert master["device_weight_bytes"] == 10
    assert master["raw_weight_bytes"] == 10
    assert master["host_source_bytes"] == master["layers_count"] * 10
    assert master["host_conversion_bytes"] == 0
    assert master["host_required_bytes"] <= master["host_capacity_bytes"] == 1000
    assert master["required_bytes"] <= master["capacity_bytes"] == 20
    frozen = deepcopy(master)
    record = {"node_id": "master", "capacity_bytes": 999999, "reserve_bytes": 0,
              "score": 100, "execution_device": "cuda:0", "engine": "pytorch"}
    current = inventory(model_id="")
    bound = bind_supplied_offload_capacity(record, master, current)
    refreshed = project_inventory_capacity(bound, current, "pytorch")
    assert refreshed["capacity_bytes"] == 20
    assert refreshed["range_weight_bytes"][0]["weight_bytes"] == 10
    current["memory_available_bytes"] = master["host_required_bytes"] - 1
    with pytest.raises(ModelSupplyError) as error:
        bind_supplied_offload_capacity(record, master, current)
    assert error.value.code == "SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT"
    assert master == frozen


@pytest.mark.parametrize("supported,cuda_available,expected", [(False, 20, "failed"), (True, 1000, "available")])
def test_unsupported_offload_rejects_and_resident_fit_keeps_resident_path(startup_pool_case, tmp_path,
                                                                       supported, cuda_available, expected):
    case = startup_pool_case
    configure(case, tmp_path, supported=supported, cuda_available=cuda_available)
    runner = case.start()
    runner.join(3)
    assert not runner.is_alive()
    operation = case.c.get(case.op)
    assert operation["status"] == expected, operation
    if expected == "failed":
        assert operation["plan"] is None and not case.build_calls
    else:
        master = operation["plan"]["assignments"][0]
        assert master["runtime_materialization"] == "safetensors_key_filtered"
        assert master["load_spec"]["materialization"] == "safetensors_key_filtered"
        assert master["load_spec"]["dtype"] == master["runtime_target_dtype"] == "float16"
        assert master["device_weight_bytes"] == master["target_weight_bytes"] == master["layers_count"] * 10
        assert "resident_capacity_diagnostic" not in operation["plan"]
        # Available supply must freeze a policy that can survive fresh ready
        # admission, including resident target precision and exact range cost.
        record = {"node_id": "master", "capacity_bytes": 999999, "reserve_bytes": 0,
                  "score": 100, "execution_device": "cuda:0", "engine": "pytorch"}
        current = case.scheduler._model_supply_worker.inventory(model_id="")
        refreshed = project_inventory_capacity(
            bind_supplied_offload_capacity(record, master, current), current, "pytorch")
        assert refreshed["load_spec_digest"] == master["load_spec_digest"]
        assert refreshed["range_weight_bytes"] == [{
            "layer_range": [master["start_layer"], master["end_layer"]],
            "weight_bytes": master["target_weight_bytes"],
            "has_embedding": master["has_embedding"], "has_lm_head": master["has_lm_head"],
        }]


def test_fresh_cuda_budget_preserves_uuid_global_authority_and_evidence():
    inventory = {"distributed": True, "engines": ["pytorch"], "memory_available_bytes": 1000,
                 "pytorch_runtime": {"execution_device": "cuda", "target_element_size": 2,
                     "available_bytes": 50, "budget_source": "cuda.min_mem_get_info_nvidia_smi_uuid",
                     "budget_observations": {"torch_free_bytes": 100,
                         "nvidia_smi_free_bytes": 50, "nvidia_smi_status": "matched"}}}
    record = {"node_id": "master", "reserve_bytes": 0}
    projected = project_inventory_capacity(record, inventory, "pytorch")
    assert projected["capacity_bytes"] == 50
    assert projected["capacity_source"] == "inventory.cuda.min_mem_get_info_nvidia_smi_uuid"
    assert projected["capacity_observations"] == inventory["pytorch_runtime"]["budget_observations"]
    projected["capacity_observations"]["torch_free_bytes"] = 0
    assert inventory["pytorch_runtime"]["budget_observations"]["torch_free_bytes"] == 100
