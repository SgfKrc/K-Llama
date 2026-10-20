"""Supply, execution and shortage explanations use one placement preference."""
from copy import deepcopy

import pytest

from model_supply import ModelSupplyError, plan_segment_supply
from model_supply_diagnostics import _capacity_candidate
from model_supply_runtime_capacity import project_supply_capacity_nodes
from pipeline_capacity import solve_pipeline_capacity
from tests.test_model_supply import descriptor, node


@pytest.mark.parametrize("prefer_all,expected", [(False, 2), (True, 3)])
def test_supply_and_execution_share_configured_node_preference(monkeypatch, prefer_all, expected):
    import config
    monkeypatch.setattr(config, "PIPELINE_PREFER_ALL_WORKERS", prefer_all)
    source = descriptor()
    records = [node(name, 1000) for name in ("master", "middle", "tail")]
    before = deepcopy(records)
    supply = plan_segment_supply({"llama_cpp": source}, records, master_engine="llama_cpp")
    grouped, projected, _ = project_supply_capacity_nodes(
        {"llama_cpp": source}, records, master_engine="llama_cpp")
    execution = solve_pipeline_capacity(grouped, projected, require_distributed=True,
                                       required_node_id="master", prefer_all_workers=prefer_all)
    spans = lambda plan: [(a["node_id"], a["start_layer"], a["end_layer"])
                          for a in plan["assignments"]]
    assert supply["admitted"] and execution["admitted"]
    assert len(supply["assignments"]) == expected
    assert spans(supply) == spans(execution)
    assert records == before


@pytest.mark.parametrize("prefer_all,expected", [(False, 2), (True, 3)])
def test_explicit_supply_preference_and_diagnostic_use_same_policy(monkeypatch, prefer_all, expected):
    import config
    monkeypatch.setattr(config, "PIPELINE_PREFER_ALL_WORKERS", not prefer_all)
    source = descriptor()
    records = [node(name, 1000) for name in ("master", "middle", "tail")]
    plan = plan_segment_supply({"llama_cpp": source}, records, master_engine="llama_cpp",
                               prefer_all_workers=prefer_all)
    assert plan["admitted"] and len(plan["assignments"]) == expected
    grouped, projected, _ = project_supply_capacity_nodes(
        {"llama_cpp": source}, records, master_engine="llama_cpp")
    candidate = _capacity_candidate(grouped, projected, 1.2, prefer_all_workers=prefer_all)
    assert candidate is not None and len(candidate[0]) == expected


@pytest.mark.parametrize("invalid", [0, 1, "false", [], {}])
def test_invalid_supply_preference_is_named(invalid):
    with pytest.raises(ModelSupplyError) as error:
        plan_segment_supply({"llama_cpp": descriptor()},
                            [node("master"), node("tail")], master_engine="llama_cpp",
                            prefer_all_workers=invalid)
    assert error.value.code == "SUPPLY_PLACEMENT_POLICY_INVALID"
