"""Capacity explanations stay factual without granting runtime admission."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

import model_supply_diagnostics as diagnostics
from model_supply_diagnostics import explain_host_capacity, explain_supply_capacity
from model_supply_host_capacity import HostCapacityProjectionError, allowed_host_layer_ranges
from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity


def grouped(layers=(100, 100, 100, 100)):
    return {
        "total_layers": len(layers), "layer_weight_bytes": list(layers),
        "component_weight_bytes": dict.fromkeys(("embedding", "final_norm", "lm_head", "other"), 0),
    }


def node(node_id, capacity, *, reserve=0, multiplier=1.0, score=0, device="cpu", **extra):
    return {
        "node_id": node_id, "role": "master" if node_id == "master" else "client",
        "capacity_bytes": capacity, "reserve_bytes": reserve,
        "runtime_multiplier": multiplier, "score": score,
        "engine": "pytorch" if device.startswith("cuda") else "llama_cpp",
        "execution_device": device, **extra,
    }


def rejected(descriptor, nodes, margin=1.0, *, prefer_all_workers=False):
    result = solve_pipeline_capacity(
        descriptor, nodes, safety_margin=margin,
        require_distributed=True, prefer_all_workers=prefer_all_workers,
    )
    assert result["admitted"] is False
    return result


def explain(descriptor, nodes, *, plan=None, alignment=1, margin=1.0, prefer_all_workers=False):
    return explain_supply_capacity(
        rejected(descriptor, nodes, margin, prefer_all_workers=prefer_all_workers) if plan is None else plan,
        grouped_descriptor=descriptor, projected_nodes=nodes,
        alignment=alignment, total_layers=descriptor["total_layers"] * alignment,
        safety_margin=margin, prefer_all_workers=prefer_all_workers,
    )


def rows(result):
    return {item["node_id"]: item for item in result["nodes"]}


def host_projection(layers=(101, 200, 300, 400), **extra):
    return {
        "total_layers": len(layers), "source_layer_weight_bytes": list(layers),
        "source_component_weight_bytes": {
            "embedding": 30, "final_norm": 10, "lm_head": 60, "other": 10,
            "layers": sum(layers), "mtp": 10**9,
        },
        "layer_weight_bytes": [1] * len(layers),
        "runtime_layer_weight_bytes": [1] * len(layers),
        "runtime_component_weight_bytes": {"embedding": 1},
        "runtime_multiplier": 1000, **extra,
    }


def host_explain(projection, node_id="master", *, available=0, reserve=23, margin=1.2,
                 alignment=1, layer_range=None):
    return explain_host_capacity(
        projection, node_id, available, reserve, margin, alignment, layer_range,
    )


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_exact_runtime_one_byte_boundary_and_resource_identity(offset):
    descriptor = grouped((101, 37))
    master_required = 19 + math.ceil(101 * 1.5 * 1.2)
    nodes = [
        node("master", master_required + offset, reserve=19, multiplier=1.5,
             device="cuda", layer_ranges=[[0, 1]]),
        node("worker", 1, reserve=10, multiplier=2.5, layer_ranges=[[1, 2]]),
    ]
    plan = rejected(descriptor, nodes, 1.2)
    result = explain(descriptor, nodes, plan=plan, alignment=2, margin=1.2)
    by_id = rows(result)
    assert result["reason_code"] == plan["reason_code"]
    assert result["scope"] == "candidate_plan"
    assert by_id["master"] == {
        "node_id": "master", "engine": "pytorch", "execution_device": "cuda",
        "resource": "vram", "available_bytes": master_required + offset,
        "required_bytes": master_required, "deficit_bytes": max(0, -offset),
        "reserve_bytes": 19, "layer_range": [0, 2],
    }
    worker_required = 10 + math.ceil(37 * 2.5 * 1.2)
    assert by_id["worker"]["required_bytes"] == worker_required
    assert by_id["worker"]["deficit_bytes"] == worker_required - 1
    assert by_id["worker"]["layer_range"] == [2, 4]
    assert by_id["worker"]["resource"] == "ram"
    assert "不代表最小缺口" in result["suggested_action"]
    assert "不保证整体准入" in result["suggested_action"]
    assert str(master_required) in result["reason"]
    assert "显存" in result["reason"] and "内存" in result["reason"]
    assert "assignments" not in result and "loaded_ready" not in result


def test_already_projected_boundary_and_conservative_ratio_are_not_recomputed():
    descriptor = grouped((20, 100, 40))
    margin = 1.2
    original_multiplier, original_reserve = 2.0, 7
    boundary_bytes = 31
    projected_reserve = original_reserve + math.ceil(boundary_bytes * original_multiplier * margin)
    # The per-group ratio is conservative for every group, even though the
    # cheapest individual group has a lower actual engine/source ratio.
    ratio = max(a / b for a, b in zip((10, 300, 50), descriptor["layer_weight_bytes"]))
    multiplier = original_multiplier * ratio
    nodes = [
        node("master", 10, reserve=projected_reserve, multiplier=multiplier,
             layer_ranges=[[0, 1]], device="cuda"),
        node("tail", 10000, layer_ranges=[[1, 3]]),
    ]
    plan = rejected(descriptor, nodes, margin)
    plan["raw_capacity_deficit_bytes"] = 0  # Aggregate bytes say nothing about the head.
    result = explain(descriptor, nodes, margin=margin, plan=plan)
    master = rows(result)["master"]
    required = projected_reserve + math.ceil(20 * multiplier * margin)
    assert master["reserve_bytes"] == projected_reserve
    assert master["required_bytes"] == required
    assert master["deficit_bytes"] == required - 10 > 0
    # Changing unrelated estimates cannot overwrite the projected solver facts.
    descriptor.update(runtime_layer_weight_bytes=[1, 1, 1], weight_bytes=1)
    assert rows(explain(descriptor, nodes, margin=margin, plan=plan))["master"] == master


@pytest.mark.parametrize("prefer_all_workers", [False, True])
@pytest.mark.parametrize("first_options,second_options", [
    ({"capacity": 200, "reserve": 100}, {"capacity": 80, "reserve": 0}),
    ({"capacity": 100, "score": 9}, {"capacity": 400, "score": 8}),
    ({"capacity": 200, "reserve": 100}, {"capacity": 100, "reserve": 0}),
])
def test_solver_relative_order_and_host_ranges_survive_capacity_inflation(
    monkeypatch, first_options, second_options, prefer_all_workers,
):
    descriptor = grouped((500, 500, 500))
    # The reversed input list must not become the execution order. These exact
    # ranges can also be source-host RAM bounds intersected by the supply planner.
    nodes = [
        node("z-second", layer_ranges=[[2, 3]], **second_options),
        node("master", 600, layer_ranges=[[0, 1]]),
        node("a-first", layer_ranges=[[1, 2]], **first_options),
    ]
    plan = rejected(descriptor, nodes, prefer_all_workers=prefer_all_workers)
    before = deepcopy((plan, descriptor, nodes))
    captured = []

    def checked_solver(value, inflated, **kwargs):
        captured.append(deepcopy(inflated))
        original = {item["node_id"]: item for item in nodes}
        increments = set()
        for item in inflated:
            source = original[item["node_id"]]
            increments.add(item["capacity_bytes"] - source["capacity_bytes"])
            assert {k: v for k, v in item.items() if k != "capacity_bytes"} == {
                k: v for k, v in source.items() if k != "capacity_bytes"
            }
        assert len(increments) == 1 and next(iter(increments)) > 0
        assert kwargs == {"safety_margin": 1.0, "require_distributed": True,
                          "prefer_all_workers": prefer_all_workers, "required_node_id": "master"}
        return solve_pipeline_capacity(value, inflated, **kwargs)

    monkeypatch.setattr(diagnostics, "solve_pipeline_capacity", checked_solver)
    result = explain(descriptor, nodes, plan=plan, alignment=2, prefer_all_workers=prefer_all_workers)
    assert len(captured) == 1
    assert [(item["node_id"], item["layer_range"]) for item in result["nodes"]] == [
        ("master", [0, 2]), ("a-first", [2, 4]), ("z-second", [4, 6]),
    ]
    assert (plan, descriptor, nodes) == before
    assert solve_pipeline_capacity(descriptor, nodes, safety_margin=1,
                                   require_distributed=True, prefer_all_workers=prefer_all_workers) == plan


@pytest.mark.parametrize("prefer_all_workers", [False, True])
def test_joint_shortage_is_visible_when_each_node_can_fit_its_cheapest_segment(prefer_all_workers):
    descriptor = grouped((100,) * 6)
    nodes = [node("b", 150), node("master", 150), node("a", 150)]
    plan = rejected(descriptor, nodes, prefer_all_workers=prefer_all_workers)
    plan["raw_capacity_deficit_bytes"] = 1  # Must not be copied into per-node gaps.
    result = explain(descriptor, nodes, plan=plan, alignment=2, prefer_all_workers=prefer_all_workers)
    assert all(100 <= item["available_bytes"] for item in result["nodes"])
    selected = [item for item in result["nodes"] if "required_bytes" in item]
    if prefer_all_workers:
        assert [item["required_bytes"] for item in selected] == [200, 200, 200]
        assert [item["deficit_bytes"] for item in selected] == [50, 50, 50]
        assert [item["layer_range"] for item in selected] == [[0, 4], [4, 8], [8, 12]]
    else:
        assert [item["node_id"] for item in selected] == ["master", "a"]
        assert [item["required_bytes"] for item in selected] == [300, 300]
        assert [item["deficit_bytes"] for item in selected] == [150, 150]
        assert [item["layer_range"] for item in selected] == [[0, 6], [6, 12]]
        assert not {"required_bytes", "deficit_bytes", "layer_range"}.intersection(rows(result)["b"])
    assert result["reason_code"] == plan["reason_code"]
    assert "assignments" not in result and "loaded_ready" not in result


def test_topologically_unavailable_reordering_cannot_invent_a_candidate():
    descriptor = grouped((500, 500, 500))
    nodes = [
        node("master", 1, layer_ranges=[[0, 1]]),
        node("first-by-score", 20, score=20, layer_ranges=[[2, 3]]),
        node("second-by-score", 1000, score=10, layer_ranges=[[1, 2]]),
    ]
    plan = rejected(descriptor, nodes)
    plan["reason"] = "保留原始拓扑拒绝原因：节点顺序和 host 范围不相容"
    before = deepcopy((plan, descriptor, nodes))
    result = explain(descriptor, nodes, plan=plan)
    assert result["reason"] == plan["reason"]
    assert result["reason_code"] == plan["reason_code"]
    for row, source in zip(result["nodes"], nodes):
        assert row["available_bytes"] == source["capacity_bytes"]
        assert row["reserve_bytes"] == source["reserve_bytes"]
        assert not {"required_bytes", "deficit_bytes", "layer_range"}.intersection(row)
    assert (plan, descriptor, nodes) == before


@pytest.mark.parametrize("constraint", [
    {"layer_ranges": []},
    {"layer_ranges": [[1, 3]], "layer_budget": {
        "max_layers": 1, "available_bytes": 1, "per_layer_bytes": 1, "local_cut": False,
    }},
    {"layer_ranges": [[1, 3]], "segment_mode": "middle"},
    {"layer_ranges": [[1, 3]], "layer_artifacts": [{
        "layer_range": [1, 3], "segment_mode": "tail", "model_id": "tail",
        "artifact_sha256": "b" * 64, "source_model_sha256": "b" * 64,
        "layer_weight_bytes": [100, 100],
    }]},
])
def test_no_candidate_does_not_remove_any_non_capacity_constraints(constraint):
    descriptor = grouped((100, 100, 100))
    descriptor["source_model_sha256"] = "a" * 64
    nodes = [node("master", 1, layer_ranges=[[0, 1]]), node("worker", 1, **constraint)]
    plan = rejected(descriptor, nodes)
    plan["reason"] = "原始拒绝原因"
    before = deepcopy((descriptor, nodes, plan))
    result = explain(descriptor, nodes, plan=plan)
    assert result["reason"] == "原始拒绝原因"
    assert all("required_bytes" not in item for item in result["nodes"])
    assert (descriptor, nodes, plan) == before


def test_excluded_zero_capacity_nodes_can_be_explained_but_not_admitted():
    descriptor = grouped((100, 100))
    nodes = [node("master", 0, reserve=20), node("worker", 0, reserve=10)]
    plan = rejected(descriptor, nodes)
    result = explain(descriptor, nodes, plan=plan)
    assert result["reason_code"] == "pipeline_distributed_workers_unavailable"
    assert rows(result)["master"]["required_bytes"] == 120
    assert rows(result)["master"]["deficit_bytes"] == 120
    assert rows(result)["worker"]["deficit_bytes"] == 110
    assert all(item["capacity_bytes"] == 0 for item in nodes)
    assert rejected(descriptor, nodes) == plan


@pytest.mark.parametrize("nodes", [
    [node("master", 1)],
    [node("master", 1), node("worker", 1)],
    [node("master", 1), node("relay", 0, capacity_source="relay_exempt")],
])
def test_no_proper_group_never_creates_a_full_model_example(nodes):
    descriptor = grouped((100,))
    result = explain(descriptor, nodes, alignment=4)
    assert all("layer_range" not in item for item in result["nodes"])
    assert "assignments" not in result and "loaded_ready" not in result


def test_unused_nodes_report_only_known_capacity_and_output_owns_its_ranges():
    descriptor = grouped((100, 100))
    nodes = [
        node("master", 1, layer_ranges=[[0, 1]]),
        node("chosen", 80, score=10, layer_ranges=[[1, 2]]),
        node("control-only", 1, layer_ranges=[[1, 2]]),
    ]
    before = deepcopy((descriptor, nodes))
    result = explain(descriptor, nodes)
    assert "required_bytes" in rows(result)["chosen"]
    assert "required_bytes" not in rows(result)["control-only"]
    rows(result)["master"]["layer_range"][0] = 999
    result["nodes"].clear()
    assert (descriptor, nodes) == before
    assert rows(explain(descriptor, nodes))["master"]["layer_range"] == [0, 1]


@pytest.mark.parametrize("device,expected", [("cuda", "vram"), ("cuda:1", "vram"), ("cpu", "ram")])
def test_resource_follows_selected_device_rather_than_engine_label(device, expected):
    descriptor = grouped((100, 100))
    nodes = [node("master", 1, device=device), node("worker", 1)]
    nodes[0]["engine"] = "pytorch"
    assert rows(explain(descriptor, nodes))["master"]["resource"] == expected


@pytest.mark.parametrize("alignment,total,margin", [
    (0, 4, 1), (True, 4, 1), (3, 4, 1), (1, 5, 1),
    (1, 4, 0.9), (1, 4, True), (1, 4, float("nan")), (1, 4, float("inf")),
])
def test_invalid_grouping_or_margin_cannot_give_an_explanation(alignment, total, margin):
    with pytest.raises(PipelineCapacityError):
        explain_supply_capacity({}, grouped_descriptor=grouped(), projected_nodes=[],
                                alignment=alignment, total_layers=total, safety_margin=margin)


@pytest.mark.parametrize("node_id,boundary", [("master", 50), ("worker", 50)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_host_exact_source_budget_agrees_with_real_admission_at_one_byte(node_id, boundary, offset):
    projection = host_projection()
    required = math.ceil((101 + boundary) * 1.2) + 23
    before = deepcopy(projection)
    result = host_explain(projection, node_id, available=required + offset)
    row = result["nodes"][0]
    assert row["required_bytes"] == required
    assert row["deficit_bytes"] == max(0, -offset)
    assert row["layer_range"] == [0, 1]
    assert row["resource"] == "host_ram"
    assert row["reserve_bytes"] == 23
    assert result["scope"] == "minimum_segment"
    admitted = allowed_host_layer_ranges(projection, node_id, required + offset, 23, 1.2, 1)
    assert (row["layer_range"] in admitted) == (offset >= 0)
    assert projection == before


@pytest.mark.parametrize("node_id,expected_range,boundary", [
    ("master", [0, 2], 50), ("pc-worker", [2, 4], 20), ("android-worker", [2, 4], 20),
])
def test_minimum_host_span_uses_nonuniform_aligned_source_groups(node_id, expected_range, boundary):
    projection = host_projection((80, 20, 9, 11, 30, 50))
    result = host_explain(projection, node_id, alignment=2)
    row = result["nodes"][0]
    start, end = expected_range
    assert row["layer_range"] == expected_range
    assert row["required_bytes"] == math.ceil((sum(projection["source_layer_weight_bytes"][start:end]) + boundary) * 1.2) + 23
    assert row["required_bytes"] == row["deficit_bytes"]


def test_host_explicit_candidate_uses_its_exact_source_range_and_not_minimum():
    projection = host_projection((80, 20, 9, 11, 30, 50))
    minimum = host_explain(projection, "worker", alignment=2, available=200)
    explicit = host_explain(projection, "worker", alignment=2, available=200, layer_range=[0, 4])
    assert minimum["scope"] == "minimum_segment"
    assert explicit["scope"] == "candidate_plan"
    assert minimum["nodes"][0]["deficit_bytes"] == 0
    assert explicit["nodes"][0]["layer_range"] == [0, 4]
    required = math.ceil((120 + 50) * 1.2) + 23
    assert explicit["nodes"][0]["required_bytes"] == required
    assert explicit["nodes"][0]["deficit_bytes"] == required - 200


@pytest.mark.parametrize("component,node_id,charge", [
    ("embedding", "master", 7), ("final_norm", "master", 7),
    ("other", "master", 7), ("lm_head", "master", 0),
    ("embedding", "worker", 0), ("final_norm", "worker", 7),
    ("other", "worker", 7), ("lm_head", "worker", 0),
])
def test_host_boundaries_match_postcheck_including_remote_middle(component, node_id, charge):
    projection = host_projection((10,) * 4)
    projection["source_component_weight_bytes"] = {
        **dict.fromkeys(("embedding", "final_norm", "lm_head", "other"), 0), component: 7,
    }
    layer_range = [0, 1] if node_id == "master" else [1, 2]
    result = host_explain(projection, node_id, reserve=3, margin=1, layer_range=layer_range)
    assert result["nodes"][0]["required_bytes"] == 10 + charge + 3


@pytest.mark.parametrize("node_id", ["master", "worker"])
@pytest.mark.parametrize("layers,alignment", [((10,), 1), ((10, 10, 10, 10), 4)])
def test_host_no_proper_range_reports_unknown_instead_of_whole_model(node_id, layers, alignment):
    result = host_explain(host_projection(layers), node_id, available=10**12, alignment=alignment)
    assert result["scope"] == "minimum_segment"
    assert result["nodes"][0]["available_bytes"] == 10**12
    assert not {"required_bytes", "deficit_bytes", "layer_range"}.intersection(result["nodes"][0])
    assert "没有合法的非整模源层段" in result["reason"]


@pytest.mark.parametrize("node_id,span", [
    ("master", [1, 2]), ("worker", [0, 4]), ("worker", [-1, 2]),
    ("worker", [2, 2]), ("worker", [0, 5]), ("worker", [0, True]),
    ("worker", [0.0, 2]), ("worker", [0]), ("worker", "0,2"),
])
def test_host_invalid_or_full_selected_ranges_do_not_become_candidates(node_id, span):
    with pytest.raises(HostCapacityProjectionError):
        host_explain(host_projection(), node_id, layer_range=span)


def test_host_selected_range_must_respect_alignment():
    with pytest.raises(HostCapacityProjectionError):
        host_explain(host_projection(), "worker", alignment=2, layer_range=[1, 2])


def test_host_input_and_output_are_independent_with_immutable_source_metadata():
    source = host_projection((10, 10, 20, 20))
    immutable = MappingProxyType({
        **source, "source_layer_weight_bytes": tuple(source["source_layer_weight_bytes"]),
        "source_component_weight_bytes": MappingProxyType(source["source_component_weight_bytes"]),
    })
    before = deepcopy(source)
    span = [0, 2]
    result = host_explain(immutable, layer_range=span)
    result["nodes"][0]["layer_range"][0] = 999
    assert span == [0, 2] and source == before
    assert host_explain(immutable)["nodes"][0]["layer_range"] == [0, 1]


@pytest.mark.parametrize("field,value", [
    ("source_layer_weight_bytes", [0, 1, 1, 1]),
    ("source_layer_weight_bytes", None), ("source_component_weight_bytes", {}),
    ("total_layers", True),
])
def test_host_does_not_fall_back_from_invalid_source_metadata(field, value):
    projection = host_projection()
    projection[field] = value
    with pytest.raises(HostCapacityProjectionError):
        host_explain(projection)


def test_import_and_both_calculations_need_no_weights_heavy_runtime_or_file_io():
    source_path = str(Path(__file__).resolve().parents[1] / "src")
    code = r'''import builtins
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "safetensors", "model_supply", "pipeline_model_descriptor"}:
        raise AssertionError("heavy import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from model_supply_diagnostics import explain_supply_capacity, explain_host_capacity
def forbidden(*args, **kwargs):
    raise AssertionError("diagnostics must not read files")
builtins.open = forbidden
pathlib.Path.open = forbidden
descriptor = {"total_layers": 2, "layer_weight_bytes": [100, 100],
              "component_weight_bytes": {"embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0}}
nodes = [{"node_id": name, "role": "master" if name == "master" else "client",
          "capacity_bytes": 1, "reserve_bytes": 0} for name in ("master", "worker")]
result = explain_supply_capacity({"reason_code": "rejected"}, grouped_descriptor=descriptor,
                                 projected_nodes=nodes, alignment=1, total_layers=2, safety_margin=1)
assert result["nodes"][0]["required_bytes"] == 100
projection = {"total_layers": 2, "source_layer_weight_bytes": [100, 100],
              "source_component_weight_bytes": descriptor["component_weight_bytes"]}
assert explain_host_capacity(projection, "master", 1, 0, 1, 1)["nodes"][0]["deficit_bytes"] == 99
assert "torch" not in sys.modules and "safetensors" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, source_path],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("span", [[0, 1], [1, 2], [3, 4]])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_host_owned_boundaries_match_runtime_range_quote_and_real_admission(tied, span, offset):
    from model_supply_runtime_capacity import runtime_range_capacity

    keys = ("layers", "embedding", "final_norm", "lm_head", "visual", "mtp", "multimodal", "other")
    value = host_projection((10, 20, 30, 40))
    value.update(tie_word_embeddings=tied, runtime_layer_weight_bytes=[5, 10, 15, 20],
                 runtime_conversion_layer_max_bytes=[2, 4, 6, 8],
                 source_component_weight_bytes={**dict.fromkeys(keys, 0),
                     "embedding": 7, "final_norm": 2, "lm_head": 13, "other": 3},
                 runtime_component_weight_bytes=dict.fromkeys(keys, 1),
                 runtime_conversion_component_max_bytes={**dict.fromkeys(keys, 0),
                     "embedding": 9, "lm_head": 11, "final_norm": 3, "other": 1})
    start, end = span
    quote = runtime_range_capacity(value, start, end, has_embedding=start == 0, has_lm_head=end == 4,
                                   materialization="safetensors_key_filtered")
    required = math.ceil(quote["host_weight_peak_bytes"] * 1.2) + 5
    available = required + offset
    row = host_explain(value, "worker", available=available, reserve=5, margin=1.2,
                       layer_range=span)["nodes"][0]
    assert row["host_source_bytes"] == quote["host_source_bytes"]
    assert row["host_conversion_bytes"] == quote["host_conversion_bytes"]
    assert row["required_bytes"] == required
    assert row["deficit_bytes"] == max(0, -offset)
    assert row["layer_range"] == span
    assert (span in allowed_host_layer_ranges(value, "worker", available, 5, 1.2, 1)) == (offset >= 0)
    if tied and span == [1, 2]:
        assert row["host_source_bytes"] == 25
        assert row["host_conversion_bytes"] == 4
    if tied and span == [3, 4]:
        assert row["host_source_bytes"] == 52
        assert row["host_conversion_bytes"] == 9


@pytest.mark.parametrize("cost_domain", ["owned_boundaries", "cpu_conversion"])
def test_minimum_host_span_compares_complete_candidate_quotes(cost_domain):
    value = host_projection((1, 10, 20, 1))
    value["source_component_weight_bytes"] = dict.fromkeys(("embedding", "final_norm", "lm_head", "other"), 0)
    if cost_domain == "owned_boundaries":
        value["source_component_weight_bytes"].update(embedding=100, lm_head=200)
    else:
        value.update(runtime_conversion_layer_max_bytes=[100, 0, 0, 200],
                     runtime_conversion_component_max_bytes=dict.fromkeys(
                         ("embedding", "final_norm", "lm_head", "other"), 0))
    row = host_explain(value, "worker", available=0, reserve=0, margin=1)["nodes"][0]
    assert row["layer_range"] == [1, 2]
    assert row["required_bytes"] == row["host_source_bytes"] == 10
    assert row["host_conversion_bytes"] == 0
    assert [1, 2] not in allowed_host_layer_ranges(value, "worker", 9, 0, 1, 1)
    assert [1, 2] in allowed_host_layer_ranges(value, "worker", 10, 0, 1, 1)
