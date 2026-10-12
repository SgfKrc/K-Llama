"""Real Safetensors header projections without tensor materialization."""
from __future__ import annotations

import copy
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import safetensors

from model_supply_runtime_capacity import (
    RuntimeCapacityProjectionError,
    project_runtime_descriptor,
)
from pipeline_model_descriptor import _ARCHITECTURE_LAYOUTS, inspect_pipeline_model


_MIB = 1024**2
_SIZES = {"F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
          "BOOL": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U8": 1}


def _write_shard(root, tensors, name="model.safetensors"):
    header, offset = {}, 0
    for key, (dtype, shape) in tensors.items():
        nbytes = math.prod(shape) * _SIZES[dtype]
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with (root / name).open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.truncate(8 + len(raw) + offset)
    return offset


def _descriptor(model_type="qwen2", layers=2):
    return {"model_type": model_type, "total_layers": layers, "hidden_size": 16,
            "model_id": "fixture", "source_model_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64, "pipeline_runtime_supported": True,
            "layer_weight_bytes": [99] * layers,
            "component_weight_bytes": {"embedding": 99}, "weight_bytes": 99,
            "architectures": ["fixture"], "partial_assignment": False,
            "runtime_ignored_components": list(_ARCHITECTURE_LAYOUTS[model_type]["runtime_ignored_components"])}


def test_mixed_headers_count_elements_and_preserve_source_descriptor(tmp_path):
    tensors = {
        "model.layers.0.a.weight": ("F8_E4M3", [3, 4]),
        "model.layers.0.b.weight": ("F16", [5]),
        "model.layers.1.weight": ("F32", [7]),
        "model.embed_tokens.weight": ("F16", [3, 8]),
        "model.norm.weight": ("F32", [8]),
        "lm_head.weight": ("F8_E5M2", [3, 8]),
        "model.rotary_emb.inv_freq": ("F32", [2]),
    }
    source_bytes = _write_shard(tmp_path, tensors)
    source = _descriptor()
    before = copy.deepcopy(source)
    projected = project_runtime_descriptor(tmp_path, source, 4)
    assert source == before
    assert projected["source_layer_weight_bytes"] == [22, 28]
    assert projected["runtime_layer_weight_bytes"] == [68, 28]
    assert projected["layer_weight_bytes"] == [90, 56]
    assert projected["source_weight_bytes"] == source_bytes == 162
    assert projected["runtime_weight_bytes"] == 328
    assert projected["weight_bytes"] == 490
    assert projected["runtime_component_weight_bytes"] == {
        "layers": 96, "embedding": 96, "final_norm": 32, "lm_head": 96,
        "visual": 0, "mtp": 0, "multimodal": 0, "other": 8,
    }
    assert projected["component_weight_bytes"]["embedding"] == 144
    assert projected["component_weight_bytes"]["lm_head"] == 120
    assert projected["source_dtype_statistics"]["F8_E4M3"] == {
        "tensor_count": 1, "element_count": 12, "weight_bytes": 12,
    }
    assert projected["source_dtype_statistics"]["F16"]["tensor_count"] == 2
    assert projected["source_dtype_statistics"]["F32"]["element_count"] == 17
    assert projected["runtime_multiplier"] == 1
    assert projected["indexed_tensor_count"] == 7
    assert projected["weight_file_count"] == 1
    assert projected["source_model_sha256"] == before["source_model_sha256"]
    projected["architectures"].append("changed")
    projected["runtime_ignored_components"].append("changed")
    assert source == before


@pytest.mark.parametrize("dtype,size,target", [("F8_E4M3", 1, 4), ("F8_E5M2", 1, 4),
                                               ("F16", 2, 2), ("F32", 4, 2),
                                               ("BF16", 2, 4), ("F64", 8, 4)])
def test_one_mib_source_layer_load_peak(dtype, size, target, tmp_path):
    _write_shard(tmp_path, {"model.layers.0.weight": (dtype, [_MIB // size])})
    projected = project_runtime_descriptor(tmp_path, _descriptor(layers=1), target)
    runtime = _MIB // size * target
    assert projected["source_layer_weight_bytes"] == [_MIB]
    assert projected["runtime_layer_weight_bytes"] == [runtime]
    assert projected["layer_weight_bytes"] == [_MIB + runtime]
    assert projected["runtime_weight_bytes"] == runtime
    assert projected["weight_bytes"] == _MIB + runtime


@pytest.mark.parametrize("model_type", tuple(_ARCHITECTURE_LAYOUTS))
def test_architecture_components_match_real_source_inspector(model_type, tmp_path):
    layout = _ARCHITECTURE_LAYOUTS[model_type]
    tensors = {
        f"{layout['layer_prefix']}0.weight": ("F16", [8]),
        f"{layout['embedding_prefixes'][0]}weight": ("F16", [32, 8]),
        f"{layout['final_norm_prefixes'][0]}weight": ("F32", [8]),
        f"{layout['lm_head_prefixes'][0]}weight": ("F16", [32, 8]),
        "unclassified.scalar": ("F32", []),
        "unclassified.empty": ("F32", [0, 8]),
    }
    for component in ("visual", "mtp", "multimodal"):
        for prefix in layout[f"{component}_prefixes"]:
            tensors[f"{prefix}weight"] = ("F16", [16])
    _write_shard(tmp_path, tensors)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": model_type, "num_hidden_layers": 1, "hidden_size": 8,
    }), encoding="utf-8")
    source = inspect_pipeline_model(tmp_path)
    before = copy.deepcopy(source)
    projected = project_runtime_descriptor(tmp_path, source, 4)
    assert projected["source_component_weight_bytes"] == source["component_weight_bytes"]
    assert projected["source_layer_weight_bytes"] == source["layer_weight_bytes"]
    assert projected["source_weight_bytes"] == source["weight_bytes"]
    assert projected["runtime_component_weight_bytes"]["other"] == 4
    for component in ("visual", "mtp", "multimodal"):
        ignored = component in layout["runtime_ignored_components"]
        raw = source["component_weight_bytes"][component]
        assert projected["runtime_component_weight_bytes"][component] == (0 if ignored else raw * 2)
        assert projected["component_weight_bytes"][component] == (0 if ignored else raw * 3)
    assert source == before


def test_large_boundaries_and_ignored_components_are_separate(tmp_path):
    _write_shard(tmp_path, {
        "model.language_model.layers.0.weight": ("F8_E4M3", [_MIB]),
        "model.language_model.embed_tokens.weight": ("F16", [_MIB]),
        "model.language_model.norm.weight": ("F32", [8]),
        "lm_head.weight": ("F8_E5M2", [_MIB]),
        "model.rotary_emb.inv_freq": ("F32", [16]),
        "model.visual.weight": ("F16", [2 * _MIB]),
        "mtp.weight": ("F16", [3 * _MIB]),
    })
    projected = project_runtime_descriptor(tmp_path, _descriptor("qwen3_5", layers=1), 4)
    assert projected["source_weight_bytes"] == 14 * _MIB + 96
    assert projected["weight_bytes"] == 16 * _MIB + 192
    assert projected["runtime_weight_bytes"] == 12 * _MIB + 96
    assert projected["component_weight_bytes"]["embedding"] == 6 * _MIB
    assert projected["component_weight_bytes"]["lm_head"] == 5 * _MIB
    assert projected["runtime_ignored_component_bytes"] == {"visual": 4 * _MIB, "mtp": 6 * _MIB}
    for name in ("visual", "mtp"):
        assert projected["runtime_component_weight_bytes"][name] == 0
        assert projected["component_weight_bytes"][name] == 0


def test_partial_f8_headers_keep_global_layer_coordinates(tmp_path):
    _write_shard(tmp_path, {"model.layers.2.weight": ("F8_E4M3", [32]),
                            "model.layers.3.weight": ("F8_E5M2", [64]),
                            "lm_head.weight": ("F16", [8])})
    source = _descriptor(layers=5)
    source.update(partial_assignment=True, assignment_layer_range=[2, 4])
    projected = project_runtime_descriptor(tmp_path, source, 4)
    assert projected["source_layer_weight_bytes"] == [0, 0, 32, 64, 0]
    assert projected["runtime_layer_weight_bytes"] == [0, 0, 128, 256, 0]
    assert projected["layer_weight_bytes"] == [0, 0, 160, 320, 0]
    assert projected["assignment_layer_range"] == [2, 4]


def test_each_node_dtype_has_an_independent_projection(tmp_path):
    _write_shard(tmp_path, {"model.layers.0.weight": ("F8_E4M3", [16]),
                            "model.layers.1.weight": ("F16", [16])})
    source = _descriptor()
    fp16 = project_runtime_descriptor(tmp_path, source, 2)
    fp32 = project_runtime_descriptor(tmp_path, source, 4)
    assert fp16["runtime_layer_weight_bytes"] == [32, 32]
    assert fp32["runtime_layer_weight_bytes"] == [64, 64]
    assert fp16["layer_weight_bytes"] == [48, 64]
    assert fp32["layer_weight_bytes"] == [80, 96]
    fp16["source_layer_weight_bytes"][0] = 0
    assert fp32["source_layer_weight_bytes"] == [16, 32]
    assert source["layer_weight_bytes"] == [99, 99]


def test_index_selects_tensor_names_from_real_shards(tmp_path):
    first = _write_shard(tmp_path, {"model.layers.0.weight": ("F8_E4M3", [8]),
                                   "unindexed.weight": ("F32", [10])}, "first.safetensors")
    _write_shard(tmp_path, {"model.layers.1.weight": ("F16", [8])}, "second.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": first - 40 + 16},
        "weight_map": {"model.layers.0.weight": "first.safetensors",
                       "model.layers.1.weight": "second.safetensors"},
    }), encoding="utf-8")
    projected = project_runtime_descriptor(tmp_path, _descriptor(), 4)
    assert projected["source_weight_bytes"] == 24
    assert projected["runtime_weight_bytes"] == 64
    assert projected["weight_file_count"] == 2
    assert projected["indexed_tensor_count"] == 2
    assert projected["source_component_weight_bytes"]["other"] == 0


@pytest.mark.parametrize("dtype", ["BOOL", "I8", "I16", "I32", "I64", "U8"])
def test_nonfloating_source_dtype_is_a_named_error(dtype, tmp_path):
    _write_shard(tmp_path, {"model.layers.0.weight": (dtype, [8])})
    with pytest.raises(RuntimeCapacityProjectionError, match=dtype) as exc:
        project_runtime_descriptor(tmp_path, _descriptor(layers=1), 4)
    assert exc.value.code == "SUPPLY_RUNTIME_DTYPE_UNSUPPORTED"
    assert "model.layers.0.weight" in exc.value.message


@pytest.mark.parametrize("target", [True, False, 0, 1, 3, 8, -1, 2.0, "4", None])
def test_invalid_target_dtype_size_is_a_named_error(target, tmp_path):
    with pytest.raises(RuntimeCapacityProjectionError) as exc:
        project_runtime_descriptor(tmp_path, _descriptor(), target)
    assert exc.value.code == "SUPPLY_RUNTIME_DTYPE_UNSUPPORTED"


@pytest.mark.parametrize("key", ["model.layers.1.weight", "model.layers.2.weight"])
def test_missing_and_out_of_bounds_decoder_layers_fail(key, tmp_path):
    _write_shard(tmp_path, {key: ("F16", [8])})
    with pytest.raises(RuntimeCapacityProjectionError) as exc:
        project_runtime_descriptor(tmp_path, _descriptor(), 4)
    assert exc.value.code == "SUPPLY_RUNTIME_SOURCE_INVALID"


def test_unknown_architecture_and_forged_ignore_declarations_fail(tmp_path):
    source = _descriptor()
    source["model_type"] = "unknown"
    with pytest.raises(RuntimeCapacityProjectionError) as exc:
        project_runtime_descriptor(tmp_path, source, 4)
    assert exc.value.code == "SUPPLY_RUNTIME_ARCHITECTURE_UNSUPPORTED"
    source = _descriptor()
    source["runtime_ignored_components"] = ["visual"]
    with pytest.raises(RuntimeCapacityProjectionError, match="runtime_ignored_components") as exc:
        project_runtime_descriptor(tmp_path, source, 4)
    assert exc.value.code == "SUPPLY_RUNTIME_DESCRIPTOR_INVALID"


def test_malformed_headers_fail_with_named_source_error(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"invalid header")
    with pytest.raises(RuntimeCapacityProjectionError) as exc:
        project_runtime_descriptor(tmp_path, _descriptor(), 4)
    assert exc.value.code == "SUPPLY_RUNTIME_SOURCE_INVALID"


def test_get_tensor_and_slice_materialization_are_never_used(tmp_path, monkeypatch):
    _write_shard(tmp_path, {"model.layers.0.weight": ("F8_E4M3", [8])})
    real_open = safetensors.safe_open

    class HeaderOnly:
        def __init__(self, *args, **kwargs):
            assert kwargs == {"framework": "numpy"}
            self.inner = real_open(*args, **kwargs)
        def __enter__(self):
            self.inner.__enter__()
            return self
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
        def keys(self):
            return self.inner.keys()
        def get_tensor(self, key):
            pytest.fail("get_tensor must never be used")
        def get_slice(self, key):
            inner = self.inner.get_slice(key)
            class MetadataOnly:
                def get_shape(self):
                    return inner.get_shape()
                def get_dtype(self):
                    return inner.get_dtype()
                def __getitem__(self, item):
                    pytest.fail("slice must never be materialized")
            return MetadataOnly()

    monkeypatch.setattr(safetensors, "safe_open", HeaderOnly)
    assert project_runtime_descriptor(tmp_path, _descriptor(layers=1), 4)["layer_weight_bytes"] == [40]


def test_helper_runs_without_importing_torch(tmp_path):
    _write_shard(tmp_path, {"model.layers.0.weight": ("F8_E4M3", [8])})
    script = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
real_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'accelerate'}:
        raise AssertionError('heavy runtime import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from model_supply_runtime_capacity import project_runtime_descriptor
result = project_runtime_descriptor(sys.argv[2], {'model_type': 'qwen2', 'total_layers': 1}, 4)
assert result['layer_weight_bytes'] == [40]
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve().parents[1] / "src"),
                             str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
