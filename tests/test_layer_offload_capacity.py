"""Header-derived host/device quotes for the explicit layer offload contract."""
from copy import deepcopy
import math

import pytest

from model_supply_host_capacity import HostCapacityProjectionError, allowed_host_layer_ranges
from model_supply_runtime_capacity import (
    RuntimeCapacityProjectionError, project_runtime_descriptor,
    project_supply_capacity_nodes, runtime_range_capacity,
)
from pipeline_capacity import solve_pipeline_capacity
from qlh_core.planning.load_spec import assignment_load_spec
from tests.test_model_supply_runtime_capacity import _descriptor, _write_shard


@pytest.fixture
def projected(tmp_path):
    _write_shard(tmp_path, {
        "model.layers.0.a.weight": ("BF16", [5]),
        "model.layers.0.b.weight": ("BF16", [15]),
        "model.layers.1.weight": ("BF16", [25]),
        "model.layers.2.weight": ("BF16", [100]),
        "model.layers.3.weight": ("BF16", [10]),
        "model.embed_tokens.weight": ("BF16", [30]),
        "model.norm.weight": ("F32", [2]),
        "lm_head.weight": ("BF16", [40]),
    })
    return project_runtime_descriptor(tmp_path, _descriptor(layers=4), 2, target_dtype="float16")


def test_bf16_to_fp16_counts_cpu_conversion_even_at_equal_element_width(projected):
    assert projected["source_layer_weight_bytes"] == [40, 50, 200, 20]
    assert projected["runtime_layer_weight_bytes"] == [40, 50, 200, 20]
    assert projected["runtime_conversion_layer_max_bytes"] == [30, 50, 200, 20]
    assert projected["runtime_cpu_layer_weight_bytes"] == [80, 100, 400, 40]
    head = runtime_range_capacity(projected, 0, 2, has_embedding=True, has_lm_head=False)
    assert head == {"host_source_bytes": 158, "host_conversion_bytes": 60,
                    "host_weight_peak_bytes": 218, "device_weight_bytes": 60,
                    "target_weight_bytes": 154}
    resident = runtime_range_capacity(projected, 0, 2, has_embedding=True, has_lm_head=False,
                                      materialization="safetensors_key_filtered")
    assert resident["device_weight_bytes"] == 154
    assert resident["host_weight_peak_bytes"] == head["host_weight_peak_bytes"]


def test_matching_source_dtype_keeps_cpu_mmap_alias_cost_once(tmp_path):
    _write_shard(tmp_path, {"model.layers.0.weight": ("F16", [32])})
    source = _descriptor(layers=1)
    legacy = project_runtime_descriptor(tmp_path, source, 2)
    exact = project_runtime_descriptor(tmp_path, source, 2, target_dtype="float16")
    assert legacy["layer_weight_bytes"] == exact["layer_weight_bytes"] == [128]
    assert legacy["runtime_cpu_layer_weight_bytes"] == [128]
    assert exact["runtime_cpu_layer_weight_bytes"] == [64]
    assert exact["runtime_conversion_layer_max_bytes"] == [0]
    quote = runtime_range_capacity(exact, 0, 1, has_embedding=False, has_lm_head=False)
    assert quote["host_source_bytes"] == quote["host_weight_peak_bytes"] == 64
    assert quote["device_weight_bytes"] == 64


def test_tail_tied_head_loads_embedding_once_and_no_unused_embedding_on_middle(projected):
    value = deepcopy(projected)
    value["tie_word_embeddings"] = True
    for name in ("source_component_weight_bytes", "runtime_component_weight_bytes",
                 "runtime_conversion_component_max_bytes"):
        value[name]["lm_head"] = 0
    middle = runtime_range_capacity(value, 1, 2, has_embedding=False, has_lm_head=False)
    tail = runtime_range_capacity(value, 3, 4, has_embedding=False, has_lm_head=True)
    assert middle["host_source_bytes"] == 58
    assert middle["device_weight_bytes"] == 50
    assert tail["host_source_bytes"] == 88
    assert tail["device_weight_bytes"] == 60
    assert tail["host_conversion_bytes"] == 60


def test_host_candidate_bound_charges_one_largest_conversion_not_the_sum(projected):
    # Prefix [0,2): source 158, one conversion 60. A single byte below
    # ceil((158 + 60) * 1.2) + 7 must fail the same admission.
    required = math.ceil(218 * 1.2) + 7
    args = dict(node_id="master", reserve_bytes=7, safety_margin=1.2, alignment=1)
    assert [0, 2] not in allowed_host_layer_ranges(projected, available_bytes=required - 1, **args)
    assert [0, 2] in allowed_host_layer_ranges(projected, available_bytes=required, **args)


def offload_node(**changes):
    # Supply selects a range before freezing its boundary components and
    # exact host requirement. A planning record therefore carries intent only.
    return {"node_id": "master", "capacity_bytes": 100, "host_capacity_bytes": 300,
            "reserve_bytes": 7, "runtime_multiplier": 1.0, "execution_device": "cuda:0",
            "engine": "pytorch", "materialization": "safetensors_layer_offload",
            "score": 10, **changes}


def test_offload_projection_and_solver_use_max_unit_with_independent_host_hard_bound(projected):
    master = offload_node()
    surface = {"node_id": "surface", "capacity_bytes": 10000, "reserve_bytes": 7,
               "runtime_multiplier": 1.0, "engine": "pytorch", "execution_device": "cpu"}
    before = deepcopy((projected, master, surface))
    grouped, nodes, by_id = project_supply_capacity_nodes(
        {"pytorch": projected}, [master, surface], master_engine="pytorch", safety_margin=1.2,
        runtime_descriptors={"master": projected})
    assert nodes[0]["reserve_bytes"] == nodes[0]["configured_reserve_bytes"] == 7
    assert nodes[0]["retained_component_reserve_bytes"] == 0
    assert nodes[0]["candidate_layer_ranges"] == [[0, 1], [0, 2]]
    assert nodes[0]["range_weight_bytes"] == [
        {"layer_range": [0, 1], "weight_bytes": 60, "has_embedding": True, "has_lm_head": False},
        {"layer_range": [0, 2], "weight_bytes": 60, "has_embedding": True, "has_lm_head": False},
    ]
    plan = solve_pipeline_capacity(grouped, nodes, safety_margin=1.2, require_distributed=True,
                                   required_node_id="master")
    assert plan["admitted"], plan
    head = next(a for a in plan["assignments"] if a["node_id"] == "master")
    assert head["required_bytes"] == math.ceil(60 * 1.2) + 7
    assert head["range_weight_quote"]["weight_bytes"] == 60
    # ★ 2026-10-26（LOCAL-EXECUTORS-01）：`by_id` 的键是 `(node_id, executor_id)` ——
    #   同一物理机的多个执行器各持有不同工件，必须能分别查表。主执行器的 `executor_id`
    #   是空串 ⇒ 既有的 `by_id["master"]` 写法在这里显式写成 `("master", "")`。
    assert by_id[("master", "")]["materialization"] == "safetensors_layer_offload"
    selected = {**head, "runtime_materialization": by_id[("master", "")]["materialization"]}
    policy = assignment_load_spec(selected, artifact_format="safetensors", context_size=32)
    assert policy.materialization == "safetensors_layer_offload"
    assert set(policy.components) == {"layers", "final_norm", "embedding"}
    assert (projected, master, surface) == before


def test_host_constraint_is_required_and_cannot_be_added_to_vram(projected):
    master = offload_node()
    del master["host_capacity_bytes"]
    with pytest.raises(RuntimeCapacityProjectionError) as exc:
        project_supply_capacity_nodes({"pytorch": projected}, [master], master_engine="pytorch")
    assert exc.value.code == "SUPPLY_RUNTIME_HOST_BUDGET_MISSING"
    master.update(capacity_bytes=10**12, host_capacity_bytes=1)
    _, nodes, _ = project_supply_capacity_nodes({"pytorch": projected}, [master], master_engine="pytorch")
    assert nodes[0]["candidate_layer_ranges"] == []
    assert nodes[0]["range_weight_bytes"] == []
    master.update(capacity_bytes=1, host_capacity_bytes=10**12)
    grouped, nodes, _ = project_supply_capacity_nodes({"pytorch": projected}, [master], master_engine="pytorch")
    assert not solve_pipeline_capacity(grouped, nodes, required_node_id="master")["admitted"]


def test_alignment_and_explicit_host_ranges_are_preserved_in_exact_quotes(projected):
    master = offload_node(host_capacity_bytes=10**6)
    _, nodes, _ = project_supply_capacity_nodes({"pytorch": projected}, [master], master_engine="pytorch",
        alignment=2, runtime_host_ranges={"master": [[0, 2]]})
    assert nodes[0]["candidate_layer_ranges"] == [[0, 1]]
    assert nodes[0]["range_weight_bytes"][0]["layer_range"] == [0, 1]
    assert nodes[0]["range_weight_bytes"][0]["weight_bytes"] == 60


@pytest.mark.parametrize("changes", [
    {"source_layer_weight_bytes": [40, 50]},
    {"runtime_layer_weight_bytes": [40, 50, True, 20]},
    {"runtime_conversion_layer_max_bytes": [0, -1, 0, 0]},
    {"runtime_conversion_component_max_bytes": {}},
    {"model_type": "qwen"},
])
def test_malformed_offload_projection_fails_closed(projected, changes):
    with pytest.raises(RuntimeCapacityProjectionError):
        runtime_range_capacity({**projected, **changes}, 0, 2,
                               has_embedding=True, has_lm_head=False)


@pytest.mark.parametrize("dtype,size", [("bfloat16", 2), ("float16", 4), ("float32", 2), ("unknown", 2)])
def test_projection_rejects_unbound_or_inconsistent_target_dtype(tmp_path, dtype, size):
    with pytest.raises(RuntimeCapacityProjectionError) as exc:
        project_runtime_descriptor(tmp_path, _descriptor(), size, target_dtype=dtype)
    assert exc.value.code == "SUPPLY_RUNTIME_DTYPE_UNSUPPORTED"


@pytest.mark.parametrize("changes", [
    {"runtime_conversion_layer_max_bytes": [1]},
    {"runtime_conversion_layer_max_bytes": [True] * 4},
    {"runtime_conversion_component_max_bytes": {}},
    {"runtime_conversion_component_max_bytes": None},
])
def test_host_ranges_reject_partial_or_corrupt_conversion_metadata(projected, changes):
    with pytest.raises(HostCapacityProjectionError):
        allowed_host_layer_ranges({**projected, **changes}, "master", 10000, 0, 1.2, 1)
