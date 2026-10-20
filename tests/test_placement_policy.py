"""Placement regressions: capability, byte admission and exact topology together."""
import copy
import itertools
import math

import pytest

from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity
from scheduler_layer_plan import compute_node_weight


def _descriptor(sizes=None, count=20):
    sizes = sizes or [100] * count
    return {"model_id": "placement", "model_type": "qwen2", "model_sha256": "a" * 64,
            "total_layers": len(sizes), "layer_weight_bytes": sizes,
            "component_weight_bytes": {"embedding": 0, "final_norm": 0, "lm_head": 0}}


def _node(name, score, *, capacity=100000, reserve=100):
    return {"node_id": name, "role": "master" if name == "master" else "client",
            "score": score, "capacity_bytes": capacity, "reserve_bytes": reserve,
            "runtime_multiplier": 1.0, "execution_device": "cpu"}


def _profile(available, total=32):
    return {"ram": {"total_gb": total, "available_gb": available},
            "cpu": {"physical_cores": 8, "freq_max_mhz": 3200}}


def _counts(plan):
    assert plan["admitted"], plan.get("reason_code")
    return {item["node_id"]: item["layers_count"] for item in plan["assignments"]}


def test_free_ram_is_absolute_and_installed_capacity_does_not_replace_it():
    large, small = _profile(19.2, 32), _profile(7.2, 8)
    assert compute_node_weight(large) > compute_node_weight(small)
    assert compute_node_weight(_profile(7.2, 128)) == compute_node_weight(small)
    unknown = {"ram": {"total_gb": 512}, "cpu": small["cpu"]}
    assert compute_node_weight(unknown) == compute_node_weight(_profile(0))
    assert compute_node_weight(_profile(128)) > compute_node_weight(_profile(64))


def test_android_absolute_bytes_and_pc_free_gib_have_identical_ram_weight():
    android = {"memory": {"available_bytes": 8 * 1024 ** 3}, "cpu": _profile(8)["cpu"]}
    assert compute_node_weight(android) == compute_node_weight(_profile(8))


@pytest.mark.parametrize("bad", [None, "bad", float("nan"), float("inf"), -2])
def test_bad_free_memory_never_turns_into_installed_capacity(bad):
    assert math.isfinite(compute_node_weight(_profile(bad)))
    assert compute_node_weight(_profile(bad)) == compute_node_weight(_profile(0))


def test_cuda_hardware_only_boosts_the_selected_cuda_execution_path():
    profile = {**_profile(16), "gpu": {"name": "RTX 4060", "is_integrated": False,
               "cuda_available": True, "vram_total_gb": 8}}
    cpu = compute_node_weight(profile, execution_device="cpu")
    cuda = compute_node_weight(profile, execution_device="cuda:0")
    assert cpu == compute_node_weight(_profile(16))
    assert cuda > cpu
    profile["gpu"]["is_integrated"] = True
    assert compute_node_weight(profile, execution_device="cuda") == cpu


def test_strong_master_gets_proportional_work_instead_of_unused_headroom():
    plan = solve_pipeline_capacity(_descriptor(), [_node("master", 4), _node("weak", 1)],
                                   require_distributed=True, required_node_id="master", safety_margin=1)
    assert _counts(plan) == {"master": 16, "weak": 4}
    assert plan["assignments"][0]["start_layer"] == 0
    assert plan["placement_policy"]["objective"] == "capability_weighted_work_square"


def test_master_prefix_contract_does_not_entitle_a_weak_master_to_extra_layers():
    plan = solve_pipeline_capacity(_descriptor(), [_node("master", 1), _node("strong", 4)],
                                   require_distributed=True, required_node_id="master", safety_margin=1)
    assert _counts(plan) == {"master": 4, "strong": 16}
    assert plan["assignments"][0]["node_id"] == "master"
    assert plan["assignments"][0]["has_embedding"] is True


def test_same_nodes_different_scores_change_layer_amount_not_just_order():
    nodes = [_node("master", 1), _node("worker", 1)]
    balanced = solve_pipeline_capacity(_descriptor(), nodes, require_distributed=True, safety_margin=1)
    nodes[0]["score"] = 4
    strong = solve_pipeline_capacity(_descriptor(), nodes, require_distributed=True, safety_margin=1)
    assert _counts(balanced) == {"master": 10, "worker": 10}
    assert _counts(strong) == {"master": 16, "worker": 4}


def test_node_count_preference_precedes_capability_and_is_input_order_invariant():
    nodes = [_node("master", 16), _node("middle", 3), _node("edge", 1)]
    for prefer, count in ((False, 2), (True, 3)):
        plan = solve_pipeline_capacity(_descriptor(), nodes, require_distributed=True,
                                       required_node_id="master", prefer_all_workers=prefer, safety_margin=1)
        reversed_plan = solve_pipeline_capacity(_descriptor(), list(reversed(nodes)), require_distributed=True,
                                                required_node_id="master", prefer_all_workers=prefer, safety_margin=1)
        assert plan["participating_node_count"] == count
        assert plan["plan_id"] == reversed_plan["plan_id"]
        assert _counts(plan)["master"] > max(value for name, value in _counts(plan).items() if name != "master")
    local = solve_pipeline_capacity(_descriptor(), nodes, safety_margin=1)
    assert _counts(local) == {"master": 20}


def test_advertised_unused_cache_cannot_force_extra_nodes_with_fewest_policy():
    nodes = [_node("master", 8), {**_node("cached", 2), "layer_ranges": [[4, 12]]},
             _node("tail", 1)]
    plan = solve_pipeline_capacity(_descriptor(), nodes, safety_margin=1, require_distributed=True,
                                   required_node_id="master", prefer_all_workers=False)
    assert plan["participating_node_count"] == 2
    assert "cached" in plan["control_only_nodes"]


def test_exact_artifact_and_candidate_host_ranges_still_override_soft_capability():
    nodes = [{**_node("master", 100), "candidate_layer_ranges": [[0, 4]]},
             {**_node("tail", 1), "layer_ranges": [[4, 20]]}]
    before = copy.deepcopy(nodes)
    plan = solve_pipeline_capacity(_descriptor(), nodes, safety_margin=1,
                                   require_distributed=True, required_node_id="master")
    assert _counts(plan) == {"master": 4, "tail": 16}
    assert nodes == before


def test_gpu_score_cannot_invent_room_beyond_reserve_multiplier_and_safety():
    nodes = [_node("master", 1000000, capacity=1000, reserve=100), _node("tail", 1)]
    nodes[0]["runtime_multiplier"] = 2
    plan = solve_pipeline_capacity(_descriptor(), nodes, safety_margin=1.5,
                                   require_distributed=True, required_node_id="master")
    assert _counts(plan) == {"master": 3, "tail": 17}
    first = plan["assignments"][0]
    assert first["required_bytes"] == first["capacity_bytes"] == 1000
    assert all(item["shortfall_bytes"] == 0 for item in plan["assignments"])


def test_reference_decoder_work_is_shared_across_different_engine_byte_quotes():
    nodes = [_node("master", 4), _node("worker", 1)]
    nodes[0]["layer_weight_bytes"] = [400] * 20  # FP32 quote, not four times the FLOPs.
    plan = solve_pipeline_capacity(_descriptor(), nodes, safety_margin=1, require_distributed=True)
    assert _counts(plan) == {"master": 16, "worker": 4}
    assert plan["assignments"][0]["raw_weight_bytes"] == 6400


@pytest.mark.parametrize("prefer", [False, True])
@pytest.mark.parametrize("sizes", [[100] * 7, [40, 200, 50, 300, 70, 120, 60]])
def test_solver_matches_exhaustive_discrete_objective(prefer, sizes):
    nodes = [_node("master", 5, capacity=1000), _node("middle", 3, capacity=650),
             _node("edge", 1, capacity=500)]
    plan = solve_pipeline_capacity(_descriptor(sizes), nodes, safety_margin=1,
                                   require_distributed=True, required_node_id="master",
                                   prefer_all_workers=prefer)
    candidates = []
    for count in (2, 3):
        for rest in itertools.combinations(nodes[1:], count - 1):
            selected = [nodes[0], *rest]
            for cuts in itertools.combinations(range(1, 7), count - 1):
                bounds = (0, *cuts, 7)
                work = [sum(sizes[a:b]) for a, b in zip(bounds, bounds[1:])]
                if any(amount + node["reserve_bytes"] > node["capacity_bytes"]
                       for amount, node in zip(work, selected)):
                    continue
                candidates.append(((-count if prefer else count),
                                   sum((amount / sum(sizes)) ** 2 / node["score"]
                                       for amount, node in zip(work, selected))))
    optimum = min(candidates)
    assert plan["participating_node_count"] == abs(optimum[0])
    assert plan["placement_objective"]["weighted_work_cost"] == pytest.approx(optimum[1])


def test_different_memory_domains_do_not_starve_the_stronger_executor():
    # A smaller free-VRAM byte pool used to lose layers to a slower large-RAM
    # node because the old objective maximized minimum absolute free bytes.
    nodes = [_node("master", 4, capacity=4000), _node("worker", 1, capacity=20000)]
    plan = solve_pipeline_capacity(_descriptor(), nodes, safety_margin=1,
                                   require_distributed=True, required_node_id="master")
    assert _counts(plan) == {"master": 16, "worker": 4}


def _range_quote(start, end, weight=100, *, total=20):
    return {"layer_range": [start, end], "weight_bytes": weight,
            "has_embedding": start == 0, "has_lm_head": end == total}


def test_nonadditive_device_peak_replaces_layer_sum_and_component_charges():
    master = _node("master", 9, capacity=250)
    master.update(runtime_multiplier=1.0, retained_component_bytes=500,
                  range_weight_bytes=[_range_quote(0, end) for end in range(1, 20)])
    before = copy.deepcopy(master)
    plan = solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)], safety_margin=1.5,
                                   require_distributed=True, required_node_id="master")
    assert _counts(plan) == {"master": 18, "tail": 2}
    first = plan["assignments"][0]
    assert first["raw_weight_bytes"] == 100  # Exact range peak, neither 1800 nor 2300.
    assert first["required_bytes"] == 250  # Reserve 100 + ceil(100 * 1.5).
    assert first["placement_work_fraction"] == pytest.approx(0.9)
    assert first["range_weight_quote"] == _range_quote(0, 18)
    assert plan["raw_model_bytes"] == 2000  # Complete model weight remains meaningful.
    assert plan["capacity_cost_mode"] == "exact_range"
    assert master == before


def test_range_peak_still_charges_runtime_multiplier_margin_and_reserve():
    master = {**_node("master", 100, capacity=249), "runtime_multiplier": 2.0,
              "range_weight_bytes": [_range_quote(0, 18)]}
    plan = solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)], safety_margin=1.5,
                                   require_distributed=True, required_node_id="master")
    assert not plan["admitted"] and not plan["assignments"]
    check = next(item for item in plan["capacity_checks"] if item["layer_range"] == [0, 18])
    assert check["required_bytes"] == 400 and check["shortfall_bytes"] == 151
    assert "raw_capacity_deficit_bytes" not in plan  # Sum of weights is not a device peak.


@pytest.mark.parametrize("quotes", [None, {}, [None],
    [{"layer_range": [0, 18], "weight_bytes": True, "has_embedding": True, "has_lm_head": False}],
    [_range_quote(0, 18, 0)], [_range_quote(0, 18, -1)], [_range_quote(0, 18, 1.5)],
    [{**_range_quote(0, 18), "has_embedding": 1}],
    [{key: value for key, value in _range_quote(0, 18).items() if key != "has_lm_head"}],
    [_range_quote(0, 21)], [_range_quote(2, 2)], [_range_quote(0, 18), _range_quote(0, 18)],
])
def test_exact_range_quote_validation_is_strict(quotes):
    master = {**_node("master", 9), "range_weight_bytes": quotes}
    with pytest.raises(PipelineCapacityError, match="range_weight_bytes"):
        solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)])


@pytest.mark.parametrize("quote", [_range_quote(0, 18),
    {**_range_quote(0, 4), "has_embedding": False},
    {**_range_quote(0, 4), "has_lm_head": True},
])
def test_missing_or_wrong_boundary_peak_never_falls_back_to_additive_cost(quote):
    master = {**_node("master", 9), "range_weight_bytes": [quote],
              "candidate_layer_ranges": [[0, 4]]}
    plan = solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)], safety_margin=1,
                                   require_distributed=True, required_node_id="master")
    assert not plan["admitted"] and plan["assignments"] == []
    assert any(item["reason_code"] in {"node_range_weight_quote_missing", "node_range_weight_components_mismatch"}
               for item in plan["candidate_exclusions"])


def test_full_model_fit_uses_only_the_exact_full_range_peak():
    master = {**_node("master", 9, capacity=200),
              "range_weight_bytes": [_range_quote(0, 20)]}
    plan = solve_pipeline_capacity(_descriptor(), [master], safety_margin=1)
    assert _counts(plan) == {"master": 20}
    assert plan["single_node_full_model_candidates"] == ["master"]
    assert plan["raw_model_bytes"] == 2000
    master["range_weight_bytes"] = [_range_quote(0, 18)]
    distributed = solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)], safety_margin=1,
                                          require_distributed=True, required_node_id="master")
    assert _counts(distributed) == {"master": 18, "tail": 2}
    assert "master" not in distributed["single_node_full_model_candidates"]


def test_range_quote_is_bound_to_plan_identity_even_when_rounded_budget_matches():
    master = {**_node("master", 9), "runtime_multiplier": 0.1,
              "range_weight_bytes": [_range_quote(0, 18, 101)]}
    first = solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)], safety_margin=1,
                                    require_distributed=True, required_node_id="master")
    master["range_weight_bytes"] = [_range_quote(0, 18, 102)]
    second = solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)], safety_margin=1,
                                     require_distributed=True, required_node_id="master")
    assert first["assignments"][0]["required_bytes"] == second["assignments"][0]["required_bytes"]
    assert first["plan_id"] != second["plan_id"]


def test_offload_load_spec_requires_cuda_and_exact_range_quotes_and_keeps_digest():
    from qlh_core.contracts.load import RuntimeLoadSpec
    spec = RuntimeLoadSpec(device="cuda:0", dtype="float16", quant_type="none", context_size=16,
                           kv_device="cuda:0", kv_dtype="float16", kv_policy="disabled",
                           components=("layers", "embedding"), materialization="safetensors_layer_offload")
    master = {**_node("master", 9), "engine": "pytorch", "execution_device": "cuda:0",
              "load_spec": spec.to_dict(), "range_weight_bytes": [_range_quote(0, 18)],
              "host_required_bytes": 2000, "host_capacity_bytes": 3000}
    plan = solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)], safety_margin=1,
                                   require_distributed=True, required_node_id="master")
    assert _counts(plan) == {"master": 18, "tail": 2}
    assert plan["assignments"][0]["load_spec_digest"] == spec.digest
    master.pop("range_weight_bytes")
    with pytest.raises(PipelineCapacityError, match="offload requires CUDA and exact range_weight_bytes"):
        solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)])
    spec_dict = spec.to_dict()
    spec_dict.update(device="cpu", kv_device="cpu")
    master.update(load_spec=spec_dict, execution_device="cpu", range_weight_bytes=[_range_quote(0, 18)])
    with pytest.raises(PipelineCapacityError, match="offload requires CUDA"):
        solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)])


@pytest.mark.parametrize("fields", [
    {}, {"host_required_bytes": 1}, {"host_capacity_bytes": 2},
    {"host_required_bytes": 0, "host_capacity_bytes": 2},
    {"host_required_bytes": 1, "host_capacity_bytes": 0},
    {"host_required_bytes": True, "host_capacity_bytes": 2},
    {"host_required_bytes": 1, "host_capacity_bytes": 2.0},
    {"host_required_bytes": 3, "host_capacity_bytes": 2},
])
def test_offload_requires_a_complete_positive_admitted_host_budget(fields):
    from qlh_core.contracts.load import RuntimeLoadSpec
    spec = RuntimeLoadSpec(device="cuda:0", dtype="float16", quant_type="none", context_size=16,
                           kv_device="cuda:0", kv_dtype="float16", kv_policy="disabled",
                           components=("layers", "embedding"), materialization="safetensors_layer_offload")
    master = {**_node("master", 9), "engine": "pytorch", "execution_device": "cuda:0",
              "load_spec": spec.to_dict(), "range_weight_bytes": [_range_quote(0, 18)], **fields}
    with pytest.raises(PipelineCapacityError, match="host_.*bytes"):
        solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)])


def test_host_domain_is_preserved_and_identity_frozen_independently_of_device_budget():
    master = {**_node("master", 9, capacity=200),
              "range_weight_bytes": [_range_quote(0, 18)],
              "host_required_bytes": 2000, "host_capacity_bytes": 3000}
    plans = []
    for field, value in (("host_capacity_bytes", 3000), ("host_capacity_bytes", 3100),
                         ("host_required_bytes", 2100)):
        master[field] = value
        plan = solve_pipeline_capacity(_descriptor(), [master, _node("tail", 1)], safety_margin=1,
                                       require_distributed=True, required_node_id="master")
        assert _counts(plan) == {"master": 18, "tail": 2}
        item = plan["assignments"][0]
        assert item["host_required_bytes"] == master["host_required_bytes"]
        assert item["host_capacity_bytes"] == master["host_capacity_bytes"]
        assert item["required_bytes"] == 200  # Host quote never enters the CUDA pool.
        plans.append(plan)
    assert len({plan["plan_id"] for plan in plans}) == 3


@pytest.mark.parametrize("fields", [{"host_required_bytes": 1},
                                   {"host_required_bytes": 10, "host_capacity_bytes": 9}])
def test_optional_host_fields_on_resident_nodes_are_validated_together(fields):
    with pytest.raises(PipelineCapacityError, match="host_.*bytes"):
        solve_pipeline_capacity(_descriptor(), [{**_node("master", 9), **fields}])


@pytest.mark.parametrize("host_capacity", [0, 3000])
def test_resident_cuda_host_capacity_observation_does_not_require_host_admission(host_capacity):
    from qlh_core.contracts.load import RuntimeLoadSpec
    spec = RuntimeLoadSpec(device="cuda:0", dtype="float16", quant_type="none", context_size=16,
                           kv_device="cuda:0", kv_dtype="float16", kv_policy="disabled",
                           components=("layers", "embedding", "lm_head"),
                           materialization="safetensors_key_filtered")
    master = {**_node("master", 9), "engine": "pytorch", "execution_device": "cuda:0",
              "load_spec": spec.to_dict(), "host_capacity_bytes": host_capacity}
    baseline = solve_pipeline_capacity(_descriptor(), [
        {key: value for key, value in master.items() if key != "host_capacity_bytes"}
    ], safety_margin=1)
    plan = solve_pipeline_capacity(_descriptor(), [master], safety_margin=1)
    assert _counts(plan) == _counts(baseline) == {"master": 20}
    item = plan["assignments"][0]
    assert item["host_capacity_bytes"] == host_capacity
    assert "host_required_bytes" not in item
    assert item["required_bytes"] == baseline["assignments"][0]["required_bytes"]
