"""Host RAM bounds remain independent from CUDA/runtime capacity estimates."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

from model_supply_host_capacity import (
    HostCapacityProjectionError,
    allowed_host_layer_ranges,
)


def projection(layers=(100,) * 8, *, components=None):
    return {
        "total_layers": len(layers),
        "source_layer_weight_bytes": list(layers),
        "source_component_weight_bytes": {
            "layers": sum(layers), "embedding": 30, "final_norm": 10,
            "lm_head": 60, "other": 10, "visual": 0, "mtp": 0, "multimodal": 0,
            **(components or {}),
        },
        # Source staging must not fall back to any of these target/peak fields.
        "layer_weight_bytes": [125] * len(layers),
        "runtime_layer_weight_bytes": [25] * len(layers),
        "component_weight_bytes": {"embedding": 1000},
        "runtime_component_weight_bytes": {"embedding": 5},
        "runtime_weight_bytes": 25 * len(layers) + 5,
        "runtime_target_element_size": 2,
        "execution_device": "cuda",
    }


def allowed(value, node_id="master", *, available=350, reserve=20, margin=1.2, alignment=1):
    return allowed_host_layer_ranges(
        value, node_id=node_id, available_bytes=available,
        reserve_bytes=reserve, safety_margin=margin, alignment=alignment,
    )


def test_cuda_budget_keeps_feasible_small_master_head_instead_of_late_rejection():
    value = projection()
    # VRAM permits a large head, but host RAM only permits the first 1-2 layers.
    assert value["runtime_weight_bytes"] < 1000
    before = deepcopy(value)
    assert allowed(value) == [[0, 1], [0, 2]]
    assert math.ceil((300 + 50) * 1.2) + 20 > 350
    assert value == before


@pytest.mark.parametrize("node_id", ["master", "pc-worker", "android-worker"])
def test_valid_zero_capacity_returns_no_ranges(node_id):
    assert allowed(projection(), node_id, available=0) == []
    assert allowed(projection(), node_id, available=20, reserve=20) == []
    assert allowed(projection(), node_id, available=10, reserve=20) == []
    assert allowed(projection(), node_id, available=100, reserve=20) == []


@pytest.mark.parametrize("node_id,expected", [
    ("master", [[0, 2], [0, 4]]),
    ("pc-worker", [[0, 2], [0, 4], [2, 4], [4, 6], [6, 8]]),
])
def test_nonuniform_source_cost_is_exact_for_each_aligned_interval(node_id, expected):
    value = projection((10, 10, 40, 40, 200, 10, 10, 10),
                       components={"embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0})
    # Group [4,6) fits exactly, [2,6) does not. Neither average nor worst-layer
    # scaling can reproduce both decisions or the cheaper starts after it.
    assert allowed(value, node_id, available=210, reserve=0, margin=1, alignment=2) == expected


@pytest.mark.parametrize("node_id,expected", [
    ("master", [[0, 1], [0, 2], [0, 3]]),
    ("pc-worker", [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]),
])
def test_even_unlimited_ram_never_returns_a_full_model_or_unaligned_segment(node_id, expected):
    assert allowed(projection((1, 1, 1, 1)), node_id, available=10000,
                   reserve=0, margin=1) == expected
    aligned = allowed(projection((1, 1, 1, 1)), node_id, available=10000,
                      reserve=0, margin=1, alignment=2)
    assert all(start % 2 == end % 2 == 0 and 0 < end - start < 4 for start, end in aligned)
    if node_id == "master":
        assert aligned == [[0, 2]]
    else:
        assert aligned == [[0, 2], [2, 4]]


@pytest.mark.parametrize("layers,alignment", [((1,), 1), ((1, 1, 1, 1), 4)])
@pytest.mark.parametrize("node_id", ["master", "worker"])
def test_one_aligned_group_has_no_proper_segment(layers, alignment, node_id):
    assert allowed(projection(layers), node_id, available=10000,
                   reserve=0, margin=1, alignment=alignment) == []


@pytest.mark.parametrize("component", ["embedding", "final_norm", "other"])
@pytest.mark.parametrize("node_id", ["master", "worker"])
def test_required_source_boundary_is_charged_even_for_remote_middle(component, node_id):
    components = dict.fromkeys(("embedding", "final_norm", "lm_head", "other"), 0)
    components[component] = 7
    value = projection((10, 10, 10, 10), components=components)
    assert allowed(value, node_id, available=16, reserve=0, margin=1) == []
    spans = allowed(value, node_id, available=17, reserve=0, margin=1)
    assert spans == ([[0, 1]] if node_id == "master" else [[0, 1], [1, 2], [2, 3], [3, 4]])


@pytest.mark.parametrize("node_id", ["worker", "android"])
def test_lm_head_is_reserved_on_every_remote_range_but_not_master(node_id):
    value = projection((10, 10, 10, 10), components={
        "embedding": 0, "final_norm": 0, "other": 0, "lm_head": 7,
    })
    assert allowed(value, "master", available=10, reserve=0, margin=1) == [[0, 1]]
    assert allowed(value, node_id, available=16, reserve=0, margin=1) == []
    assert allowed(value, node_id, available=17, reserve=0, margin=1) == [[0, 1], [1, 2], [2, 3], [3, 4]]


@pytest.mark.parametrize("node_id,boundary", [("master", 50), ("worker", 110)])
def test_margin_and_reserve_match_postcheck_at_the_last_byte(node_id, boundary):
    value = projection((101, 200, 300))
    required = math.ceil((101 + boundary) * 1.2) + 23
    assert [0, 1] not in allowed(value, node_id, available=required - 1, reserve=23)
    assert [0, 1] in allowed(value, node_id, available=required, reserve=23)
    assert [0, 1] in allowed(value, node_id, available=required + 1, reserve=23)


def test_only_source_fields_and_current_postcheck_boundary_keys_affect_ram_admission():
    value = projection()
    expected = allowed(value)
    value.update(
        weight_bytes=1, runtime_weight_bytes=1,
        layer_weight_bytes=[1] * 8, runtime_layer_weight_bytes=[10**12] * 8,
        component_weight_bytes={"embedding": 10**12},
        runtime_component_weight_bytes={"lm_head": 10**12},
        layer_ranges=[[0, 7]], runtime_multiplier=1000,
    )
    value["source_component_weight_bytes"].update(visual=10**12, mtp=10**12, multimodal=10**12)
    assert allowed(value) == expected
    value["source_layer_weight_bytes"][0] = 10000
    assert allowed(value) == []


def test_projection_total_layers_is_authoritative_and_input_and_results_are_independent():
    value = projection((1, 1, 1, 1))
    value.update(config={"num_hidden_layers": 32}, layer_ranges=[[0, 32]])
    before = deepcopy(value)
    immutable = MappingProxyType({
        **value,
        "source_layer_weight_bytes": tuple(value["source_layer_weight_bytes"]),
        "source_component_weight_bytes": MappingProxyType(value["source_component_weight_bytes"]),
    })
    first = allowed(immutable, "worker", available=10000, reserve=0, margin=1)
    assert max(end for _, end in first) == 4
    first[0][0] = 99
    first.append([0, 32])
    assert allowed(immutable, "worker", available=10000, reserve=0, margin=1)[0] == [0, 1]
    assert value == before


@pytest.mark.parametrize("field,invalid", [
    ("total_layers", True), ("total_layers", 0), ("total_layers", -1),
    ("total_layers", 8.0), ("total_layers", "8"),
    ("source_layer_weight_bytes", None), ("source_layer_weight_bytes", {}),
    ("source_layer_weight_bytes", [100] * 7), ("source_layer_weight_bytes", [100] * 9),
    ("source_layer_weight_bytes", [100] * 7 + [0]),
    ("source_layer_weight_bytes", [100] * 7 + [-1]),
    ("source_layer_weight_bytes", [100] * 7 + [True]),
    ("source_layer_weight_bytes", [100] * 7 + [100.0]),
    ("source_layer_weight_bytes", [100] * 7 + ["100"]),
    ("source_component_weight_bytes", None), ("source_component_weight_bytes", []),
    ("source_component_weight_bytes", {"embedding": 1}),
    ("source_component_weight_bytes", {"embedding": True, "final_norm": 0, "lm_head": 0, "other": 0}),
    ("source_component_weight_bytes", {"embedding": 0, "final_norm": -1, "lm_head": 0, "other": 0}),
    ("source_component_weight_bytes", {"embedding": 0, "final_norm": 0, "lm_head": 1.0, "other": 0}),
    ("source_component_weight_bytes", {"embedding": 0, "final_norm": 0, "lm_head": 0, "other": "1"}),
])
def test_malformed_source_projection_is_rejected_without_mutation(field, invalid):
    value = projection()
    value[field] = invalid
    before = deepcopy(value)
    with pytest.raises(HostCapacityProjectionError) as exc:
        allowed(value)
    assert exc.value.code == "SUPPLY_RUNTIME_HOST_CAPACITY_INVALID"
    assert value == before


@pytest.mark.parametrize("field", ["total_layers", "source_layer_weight_bytes", "source_component_weight_bytes"])
def test_missing_source_metadata_never_falls_back_to_runtime_estimates(field):
    value = projection()
    del value[field]
    with pytest.raises(HostCapacityProjectionError):
        allowed(value)


@pytest.mark.parametrize("kwargs", [
    {"available": True}, {"available": -1}, {"available": 350.0}, {"available": "350"},
    {"reserve": True}, {"reserve": -1}, {"reserve": 20.0},
    {"alignment": True}, {"alignment": 0}, {"alignment": -1}, {"alignment": 3}, {"alignment": 2.0},
    {"margin": True}, {"margin": 0}, {"margin": 0.99}, {"margin": float("nan")},
    {"margin": float("inf")}, {"margin": float("-inf")}, {"margin": "1.2"},
    {"node_id": ""}, {"node_id": " master"}, {"node_id": "master "}, {"node_id": None},
])
def test_invalid_budget_or_alignment_cannot_bypass_admission(kwargs):
    with pytest.raises(HostCapacityProjectionError):
        allowed(projection(), **kwargs)


@pytest.mark.parametrize("value", [None, [], "projection"])
def test_projection_requires_a_mapping(value):
    with pytest.raises(HostCapacityProjectionError):
        allowed(value)


def test_import_and_calculation_need_no_torch_weight_reader_or_file_io():
    source = str(Path(__file__).resolve().parents[1] / "src")
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
from model_supply_host_capacity import allowed_host_layer_ranges
def forbid_file(*args, **kwargs):
    raise AssertionError("capacity enumeration must not read model files")
builtins.open = forbid_file
pathlib.Path.open = forbid_file
projection = {"total_layers": 4, "source_layer_weight_bytes": [10, 10, 10, 10],
              "source_component_weight_bytes": {"embedding": 1, "final_norm": 1, "lm_head": 2, "other": 1}}
assert allowed_host_layer_ranges(projection, "master", 23, 0, 1, 1) == [[0, 1], [0, 2]]
assert "torch" not in sys.modules and "safetensors" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, source],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
