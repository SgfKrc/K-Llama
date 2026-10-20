"""PC generation candidates are separate from registered executable segments."""
from copy import deepcopy

import pytest

from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity


def descriptor(sizes=(50, 50, 50, 50)):
    return {"model_id": "capacity-pc", "model_type": "qwen2", "model_sha256": "a" * 64,
            "source_model_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
            "pipeline_runtime_supported": True, "hidden_size": 16,
            "total_layers": len(sizes), "layer_weight_bytes": list(sizes),
            "component_weight_bytes": {"embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0}}


def node(name, available=35, **extra):
    return {"node_id": name, "role": "master" if name == "master" else "client",
            "capacity_bytes": available, "reserve_bytes": 5, "runtime_multiplier": 1.0,
            "capacity_source": "test.runtime", "engine": "pytorch",
            "execution_device": "cpu", "score": 10, **extra}


def solve(nodes, value=None, **kwargs):
    return solve_pipeline_capacity(value or descriptor(), nodes, safety_margin=1.0,
                                   require_distributed=True, required_node_id="master", **kwargs)


def test_nonuniform_runtime_quotes_find_pc_only_split_without_worst_layer_ratio():
    nodes = [node("master", layer_weight_bytes=[10, 10, 500, 500]),
             node("surface", layer_weight_bytes=[500, 500, 10, 10])]
    before = deepcopy(nodes)
    plan = solve(nodes)
    assert plan["admitted"], plan
    assert [(a["node_id"], a["start_layer"], a["end_layer"], a["required_bytes"])
            for a in plan["assignments"]] == [("master", 0, 2, 25), ("surface", 2, 4, 25)]
    assert nodes == before


def test_new_candidate_ranges_are_hard_even_with_local_cut():
    budget = {"available_bytes": 10000, "per_layer_bytes": 1, "max_layers": 4, "local_cut": True}
    nodes = [node("master", 10000, candidate_layer_ranges=[[0, 1]], layer_budget=budget),
             node("surface", 10000, candidate_layer_ranges=[[2, 4]], layer_budget=budget)]
    plan = solve(nodes)
    assert not plan["admitted"], plan
    assert any(e["reason_code"] == "node_candidate_layer_range_unavailable"
               for e in plan["candidate_exclusions"])


@pytest.mark.parametrize("sizes", [[1, 2], [1, 2, 3, 0], [1, 2, 3, -1],
                                    [1, 2, 3, True], [1, 2, 3, 1.5]])
def test_invalid_node_runtime_quotes_cannot_be_silently_ignored(sizes):
    with pytest.raises(PipelineCapacityError, match="layer_weight_bytes"):
        solve([node("master", 10000, layer_weight_bytes=sizes), node("surface", 10000)])


def test_exact_node_quotes_keep_margin_reserve_and_real_shortage():
    nodes = [node("master", 33, layer_weight_bytes=[10] * 4),
             node("surface", 32, layer_weight_bytes=[10] * 4)]
    plan = solve_pipeline_capacity(descriptor(), nodes, safety_margin=1.4,
                                   require_distributed=True, required_node_id="master")
    assert not plan["admitted"]
    nodes[1]["capacity_bytes"] = 33
    plan = solve_pipeline_capacity(descriptor(), nodes, safety_margin=1.4,
                                   require_distributed=True, required_node_id="master")
    assert plan["admitted"], plan
    assert all(a["required_bytes"] == 33 for a in plan["assignments"])


def test_ready_artifact_byte_quote_wins_over_node_projection():
    artifact = {"layer_range": [2, 4], "segment_mode": "tail", "model_id": "ready-tail",
                "artifact_sha256": "c" * 64, "source_model_sha256": "a" * 64,
                "layer_weight_bytes": [100, 100]}
    nodes = [node("master", 10000), node("surface", 35, layer_weight_bytes=[1] * 4,
              layer_ranges=[[2, 4]], layer_artifacts=[artifact])]
    assert not solve(nodes)["admitted"]


def project(nodes, value=None, **kwargs):
    from model_supply_runtime_capacity import project_supply_capacity_nodes
    model = value or descriptor()
    return project_supply_capacity_nodes({"pytorch": model}, nodes, master_engine="pytorch", **kwargs)


def test_supply_candidates_ignore_old_segments_without_granting_execution_readiness():
    nodes = [node("master", 1000, layer_ranges=[[0, 1]], segment_mode="head",
                  layer_budget={"available_bytes": 1000, "per_layer_bytes": 50,
                                "max_layers": 1, "local_cut": False}),
             node("surface", 1000, layer_ranges=[[1, 4]], segment_mode="tail")]
    before = deepcopy(nodes)
    grouped, projected, _ = project(nodes)
    assert [0, 3] in projected[0]["candidate_layer_ranges"]
    assert [3, 4] in projected[1]["candidate_layer_ranges"]
    assert "layer_ranges" not in projected[0] and "layer_budget" not in projected[0]
    assert "layer_ranges" not in projected[1] and "layer_artifacts" not in projected[1]
    # Actual execution admission still honors the exact registered ranges.
    execution = solve(nodes)
    assert [(a["start_layer"], a["end_layer"]) for a in execution["assignments"]] == [(0, 1), (1, 4)]
    candidate = solve(projected, grouped)
    assert [(a["start_layer"], a["end_layer"]) for a in candidate["assignments"]] == [(0, 2), (2, 4)]
    assert nodes == before


@pytest.mark.parametrize("change", [
    {"partial_assignment": True, "assignment_layer_range": [0, 1]},
    {"layer_weight_bytes": [50, 0, 0, 0]},
    {"pipeline_runtime_supported": False},
])
def test_old_segment_or_incomplete_source_never_authorizes_new_candidate_space(change):
    from model_supply_runtime_capacity import RuntimeCapacityProjectionError
    value = {**descriptor(), **change}
    with pytest.raises(RuntimeCapacityProjectionError):
        project([node("master", 1000), node("surface", 1000)], value)


def test_exact_projection_uses_each_node_dtype_and_preserves_configured_reserve():
    master = {**descriptor(), "layer_weight_bytes": [10, 10, 500, 500]}
    surface = {**descriptor(), "layer_weight_bytes": [500, 500, 10, 10]}
    nodes = [node("master"), node("surface")]
    grouped, projected, _ = project(nodes, runtime_descriptors={"master": master, "surface": surface})
    assert all(a["runtime_multiplier"] == 1 and a["reserve_bytes"] == 5 for a in projected)
    assert solve(projected, grouped)["admitted"]


def test_cpu_boundary_and_margin_are_charged_once_in_addition_to_explicit_reserve():
    value = descriptor((10, 10, 10, 10))
    value["component_weight_bytes"] = {"embedding": 7, "final_norm": 3, "lm_head": 11, "other": 2}
    grouped, projected, _ = project([node("master", 1000, runtime_multiplier=2),
                                     node("surface", 1000, runtime_multiplier=2)], value)
    import math
    assert [a["retained_component_reserve_bytes"] for a in projected] == [math.ceil(12 * 2 * 1.2), math.ceil(23 * 2 * 1.2)]
    assert [a["configured_reserve_bytes"] for a in projected] == [5, 5]
    plan = solve_pipeline_capacity(grouped, projected, safety_margin=1.2,
                                   require_distributed=True, required_node_id="master")
    import math
    for item in plan["assignments"]:
        boundary = 12 if item["node_id"] == "master" else 23
        assert item["required_bytes"] == math.ceil(item["layers_count"] * 10 * 2 * 1.2) + math.ceil(boundary * 2 * 1.2) + 5


def test_host_ranges_remain_hard_after_pc_ready_hints_are_removed():
    nodes = [node("master", 10000, layer_ranges=[[0, 1]]),
             node("surface", 10000, layer_ranges=[[1, 4]])]
    grouped, projected, _ = project(nodes, runtime_host_ranges={"master": [[0, 1]], "surface": [[2, 4]]})
    assert not solve(projected, grouped)["admitted"]
    assert projected[0]["candidate_layer_ranges"] == [[0, 1]]


def load_spec(device="cuda:0", dtype="float16", **extra):
    from qlh_core.contracts.load import RuntimeLoadSpec
    return RuntimeLoadSpec(device=device, dtype=dtype, quant_type="fp16", context_size=32,
                           kv_device=device, kv_dtype=dtype, kv_policy="dynamic",
                           components=("layers", "final_norm", "embedding"),
                           materialization="safetensors_key_filtered", **extra)


def test_inventory_uses_actual_engine_spec_and_keeps_host_ram_out_of_vram():
    from model_supply_capacity import project_inventory_capacity
    policy = load_spec(kv_cache_bytes=5)
    raw = node("master", 10**15, load_spec=policy.to_dict(), load_spec_digest=policy.digest)
    inventory = {"distributed": True, "engines": ["pytorch"], "memory_available_bytes": 10**15,
                 "local_cut": True, "pytorch_runtime": {"execution_device": "cuda",
                    "target_element_size": 2, "available_bytes": 33}}
    result = project_inventory_capacity(raw, inventory, "pytorch")
    assert result["capacity_bytes"] == 33 and result["host_capacity_bytes"] == 10**15
    assert result["execution_device"] == "cuda:0" and result["runtime_multiplier"] == 1
    assert result["reserve_bytes"] == 5 and result["load_spec_digest"] == policy.digest
    assert result["capacity_source"] == "inventory.cuda.mem_get_info"
    assert "layer_budget" not in result


@pytest.mark.parametrize("change", [
    {"engines": ["llama_cpp"]},
    {"pytorch_runtime": None},
    {"pytorch_runtime": {"execution_device": "cpu", "target_element_size": 2, "available_bytes": 10000}},
    {"pytorch_runtime": {"execution_device": "cuda", "target_element_size": 4, "available_bytes": 10000}},
])
def test_large_ram_or_local_cut_cannot_override_engine_device_dtype_contract(change):
    from model_supply import ModelSupplyError
    from model_supply_capacity import project_inventory_capacity
    policy = load_spec()
    raw = node("master", load_spec=policy.to_dict(), load_spec_digest=policy.digest)
    inventory = {"distributed": True, "engines": ["pytorch"], "memory_available_bytes": 10**15,
                 "local_cut": True, "pytorch_runtime": {"execution_device": "cuda", "target_element_size": 2,
                                                        "available_bytes": 10000}, **change}
    with pytest.raises(ModelSupplyError):
        project_inventory_capacity(raw, inventory, "pytorch")


def test_actual_selected_head_0_4_does_not_lock_the_second_full_source_solve(tmp_path, monkeypatch):
    """Build/register a real GGUF head, then enlarge its next candidate budget."""
    import json
    import config
    import gguf
    import numpy as np
    from llama_engine import LlamaCppEngine
    from model_supply_assets import build_gguf_segment, sha256_file
    from model_supply_worker import ModelSupplyWorker, get_selected_model
    from model_supply_runtime_capacity import project_supply_capacity_nodes
    from relay_segment_info import read_gguf_tokenizer_sha256
    from scheduler import Scheduler
    from tests.test_model_supply_worker import Scheduler as RegistrationScheduler

    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    source = tmp_path / "full.gguf"
    writer = gguf.GGUFWriter(str(source), "qwen2")
    writer.add_block_count(8)
    writer.add_embedding_length(8)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["hello", "world"])
    writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    for i in range(8):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(8, dtype=np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    engine = LlamaCppEngine()
    engine._model_path = str(source)
    engine.active_model_id = "issue95-full"
    full = engine.get_pipeline_descriptor()
    reserve = 7
    components = full["component_weight_bytes"]
    overhead = sum(components.get(k, 0) for k in ("embedding", "final_norm", "lm_head", "other"))
    four_cost = sum(full["layer_weight_bytes"][:4]) + overhead + reserve
    records = [node("master", four_cost, engine="llama_cpp"), node("surface", four_cost, engine="llama_cpp")]
    for record in records:
        record["reserve_bytes"] = reserve
    def candidate():
        grouped, projected, _ = project_supply_capacity_nodes(
            {"llama_cpp": full}, records, master_engine="llama_cpp", safety_margin=1.0)
        return solve(projected, grouped)
    first = candidate()
    assert [(a["start_layer"], a["end_layer"]) for a in first["assignments"]] == [(0, 4), (4, 8)]
    owner = RegistrationScheduler("master")
    worker = ModelSupplyWorker(owner, tmp_path / "supply" / "received")
    owner.worker = worker
    directory = worker.root.parent / "operations" / "first-actual-head"
    manifest = build_gguf_segment(source, directory, source_model_id="issue95-full",
        source_model_sha256=sha256_file(source), tokenizer_sha256=read_gguf_tokenizer_sha256(source),
        start_layer=0, end_layer=4, total_layers=8, operation_id="issue95-first", node_id="master")
    registered = worker.activate_local(manifest, directory)
    assert registered["status"] == "available", registered
    selected = get_selected_model(worker.root)
    assert selected["manifest"]["layer_range"] == [0, 4]
    actual = Scheduler.__new__(Scheduler)
    actual._model_supply_worker = worker
    artifact = actual._configured_layer_artifact()
    assert (artifact["start"], artifact["end"]) == (0, 4)
    records[0].update(layer_ranges=[[artifact["start"], artifact["end"]]],
                      segment_mode=artifact["segment_mode"], capacity_bytes=four_cost + 4 * 32)
    records[1]["layer_ranges"] = [[4, 8]]
    records[1]["capacity_bytes"] = four_cost - 2 * 32
    pointer = (worker.root / "current.json").read_bytes()
    second = candidate()
    assert second["admitted"], second
    head, tail = second["assignments"]
    assert head["node_id"] == "master" and head["start_layer"] == 0
    assert 4 < head["end_layer"] < 8
    assert tail["start_layer"] == head["end_layer"] and tail["end_layer"] == 8
    assert all(item["required_bytes"] <= item["capacity_bytes"] for item in second["assignments"])
    assert second["placement_policy"]["objective"] == "capability_weighted_work_square"
    assert (worker.root / "current.json").read_bytes() == pointer
    assert str(worker.selected_model_path()) == selected["model_path"]
    print(json.dumps({"registered_range": selected["manifest"]["layer_range"],
                      "first": [[a["start_layer"], a["end_layer"]] for a in first["assignments"]],
                      "second": [[a["start_layer"], a["end_layer"]] for a in second["assignments"]],
                      "native_loaded": False, "pointer_unchanged": True}, sort_keys=True))


def test_candidate_plan_never_claims_whole_model_eligibility_from_reference_bytes():
    grouped, projected, _ = project([node("master", 10000), node("surface", 10000)])
    plan = solve(projected, grouped)
    assert plan["admitted"] and plan["single_node_full_model_candidates"] == []


def test_ready_solver_keeps_engine_load_spec_gate_with_local_cut():
    policy = load_spec(device="cpu")
    policy = {**policy.to_dict(), "materialization": "gguf_segment"}
    budget = {"available_bytes": 10000, "per_layer_bytes": 50, "max_layers": 4, "local_cut": True}
    with pytest.raises(PipelineCapacityError, match="engine mismatch"):
        solve([node("master", 10000, load_spec=policy, layer_budget=budget), node("surface", 10000)])


def test_supply_projection_cannot_change_runtime_source_identity():
    from model_supply_runtime_capacity import RuntimeCapacityProjectionError
    with pytest.raises(RuntimeCapacityProjectionError, match="changed source_model_sha256"):
        project([node("master"), node("surface")], runtime_descriptors={
            "master": {**descriptor(), "source_model_sha256": "c" * 64}})


def test_supply_diagnostics_boost_exact_node_quotes_without_ignoring_host_ranges():
    from model_supply_diagnostics import explain_supply_capacity
    value = descriptor((1, 1, 1, 1))
    nodes = [node("master", 10, layer_weight_bytes=[500] * 4),
             node("surface", 10, layer_weight_bytes=[1000] * 4)]
    rejected = solve(nodes, value)
    diagnostic = explain_supply_capacity(rejected, grouped_descriptor=value, projected_nodes=nodes,
        alignment=1, total_layers=4, safety_margin=1.0)
    assert all(row["deficit_bytes"] > 0 for row in diagnostic["nodes"])
    for record in nodes:
        record["candidate_layer_ranges"] = []
    rejected = solve(nodes, value)
    diagnostic = explain_supply_capacity(rejected, grouped_descriptor=value, projected_nodes=nodes,
        alignment=1, total_layers=4, safety_margin=1.0)
    assert all("required_bytes" not in row for row in diagnostic["nodes"])
