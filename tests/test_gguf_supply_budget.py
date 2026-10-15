"""Exact GGUF decoder budgets agree with real, verified supply artifacts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import model_supply_assets as assets
from gguf_tensor_selection import select_gguf_tensors
from llama_engine import LlamaCppEngine
from relay_segment_info import read_gguf_tokenizer_sha256


def _write_source(path, *, mtp=0, extras=(), omit_layer=None,
                  scalar_overrides=None, omit_metadata=()):
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), "qwen2")
    scalars = {"block_count": (4 + mtp, 4), "embedding_length": (8, 4)}
    if mtp:
        scalars["nextn_predict_layers"] = (mtp, 4)
    scalars.update(scalar_overrides or {})
    for name, (value, kind) in scalars.items():
        if name not in omit_metadata:
            if kind == 9:
                writer.add_key_value(f"qwen2.{name}", value, gguf.GGUFValueType.ARRAY,
                                     gguf.GGUFValueType.UINT32)
            else:
                writer.add_key_value(f"qwen2.{name}", value, gguf.GGUFValueType(kind))
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["hello", "world", "!"])
    tensors = {
        "token_embd.weight": np.arange(64, dtype=np.float32).reshape(8, 8),
        "output_norm.weight": np.ones(8, dtype=np.float32),
        "output.weight": np.ones((8, 8), dtype=np.float32),
    }
    tensors.update({f"blk.{layer}.attn_norm.weight": np.full(8 * (layer + 1), layer, dtype=np.float32)
                    for layer in range(4) if layer != omit_layer})
    # Even an unfamiliar suffix in a declared MTP block is excluded by index;
    # an MTP-looking name outside blk.* must remain conservatively included.
    for index in range(mtp):
        tensors[f"blk.{4 + index}.future_predictor.weight"] = np.ones(8 * (index + 1), dtype=np.float32)
        tensors[f"blk.{4 + index}.nextn.weight"] = np.ones(16, dtype=np.float32)
    tensors.update({name: np.ones(elements, dtype=np.float32) for name, elements in extras})
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {name: value.nbytes for name, value in tensors.items()}


def _engine(path):
    engine = LlamaCppEngine()
    engine._model_path = str(path)
    engine.active_model_id = "tiny:qwen2"
    return engine


def _build_args(source, span):
    return {
        "source_model_id": "tiny:qwen2", "source_model_sha256": assets.sha256_file(source),
        "tokenizer_sha256": read_gguf_tokenizer_sha256(source), "start_layer": span[0],
        "end_layer": span[1], "total_layers": 4, "operation_id": "budget-1", "node_id": "worker-1",
    }


@pytest.mark.parametrize("span", [(0, 2), (1, 3), (2, 4)])
@pytest.mark.parametrize("mtp", [0, 2])
@pytest.mark.parametrize("unknowns", [(), (("unknown_aux.weight", 12), ("mtp.weight", 16))])
def test_real_head_middle_tail_budget_matches_verified_builder(tmp_path, span, mtp, unknowns):
    source = tmp_path / "source.gguf"
    sizes = _write_source(source, mtp=mtp, extras=unknowns)
    engine = _engine(source)
    descriptor = engine.get_pipeline_descriptor()
    assert descriptor["total_layers"] == 4
    assert descriptor["layer_weight_bytes"] == [32, 64, 96, 128]
    assert descriptor["component_weight_bytes"] == {
        "embedding": 256, "final_norm": 32, "lm_head": 256,
        "other": sum(elements * 4 for _, elements in unknowns),
    }
    ignored = {name: size for name, size in sizes.items() if name.startswith(("blk.4.", "blk.5."))}
    assert descriptor["runtime_ignored_tensor_bytes"] == ignored
    assert descriptor["runtime_ignored_component_bytes"] == ({"mtp": sum(ignored.values())} if mtp else {})
    assert descriptor["runtime_ignored_components"] == (["mtp"] if mtp else [])
    assert descriptor["indexed_tensor_count"] == len(sizes)
    assert descriptor["source_weight_bytes"] == sum(sizes.values())
    assert descriptor["pipeline_weight_bytes"] + sum(ignored.values()) == sum(sizes.values())
    # Preserve the original file-size identity separately from payload budgets.
    assert descriptor["weight_bytes"] == source.stat().st_size
    assert engine._model is None and engine.is_loaded is False

    result = assets.build_gguf_segment(source, tmp_path / "out", **_build_args(source, span))
    target = Path(result["local_files"]["segment.gguf"])
    _, actual = assets._gguf_table(target)
    actual_bytes = sum(entry[1] for entry in actual.values())
    selection = select_gguf_tensors(sizes, block_count=4 + mtp, nextn_predict_layers=mtp,
                                   start_layer=span[0], end_layer=span[1])
    predicted = sum(descriptor["layer_weight_bytes"][span[0]:span[1]]) + sum(
        descriptor["component_weight_bytes"].values())
    assert actual_bytes == predicted == selection.selected_weight_bytes
    assert actual_bytes == assets.gguf_segment_disk_weight_bytes(source, *span)
    assert set(actual) == set(selection.tensor_sources)
    assert all(actual[target_name][1] == sizes[source_name]
               for target_name, source_name in selection.tensor_sources.items())
    assert {name for name, _ in unknowns} <= actual.keys()
    assert not any("future_predictor" in name or "nextn" in name for name in actual)
    assert result["layer_weight_bytes"] == [size if span[0] <= layer < span[1] else 0
                                            for layer, size in enumerate([32, 64, 96, 128])]

    # Precut descriptors still use global source indices, not local blk.* indices.
    cropped = _engine(target).get_pipeline_descriptor()
    assert cropped["total_layers"] == 4
    assert cropped["layer_weight_bytes"] == result["layer_weight_bytes"]
    assert cropped["component_weight_bytes"] == descriptor["component_weight_bytes"]
    assert cropped["source_weight_bytes"] == cropped["pipeline_weight_bytes"] == actual_bytes
    assert cropped["runtime_ignored_tensor_bytes"] == {}


def test_without_mtp_all_payload_bytes_remain_in_budget(tmp_path):
    path = tmp_path / "plain.gguf"
    sizes = _write_source(path, extras=(("unknown.nextn_predictor", 24),))
    descriptor = _engine(path).get_pipeline_descriptor()
    assert descriptor["source_weight_bytes"] == descriptor["pipeline_weight_bytes"] == sum(sizes.values())
    assert descriptor["component_weight_bytes"]["other"] == 96
    assert descriptor["runtime_ignored_components"] == []
    assert descriptor["runtime_ignored_tensor_bytes"] == {}


@pytest.mark.parametrize("name,mtp", [
    ("blk.4.unexpected.weight", 0), ("blk.6.unexpected.weight", 2),
    ("blk.-1.attn.weight", 0), ("blk.+1.attn.weight", 0),
    ("blk.01.attn.weight", 0), ("blk..attn.weight", 0),
    ("blk.0.", 0), ("blk.bad.weight", 2),
])
def test_real_invalid_block_inventory_is_rejected_before_cut(tmp_path, monkeypatch, name, mtp):
    source = tmp_path / "invalid.gguf"
    _write_source(source, mtp=mtp, extras=((name, 8),))
    def forbidden_cut(*args, **kwargs):
        pytest.fail("invalid block inventory reached native cutter")
    monkeypatch.setattr(assets, "_run_cut", forbidden_cut)
    assert _engine(source).get_pipeline_descriptor() == {}
    with pytest.raises(assets.SupplyError) as error:
        assets.gguf_segment_disk_weight_bytes(source, 0, 2)
    assert error.value.code == "invalid_gguf"
    with pytest.raises(assets.SupplyError) as error:
        assets.build_gguf_segment(source, tmp_path / "out", **_build_args(source, (0, 2)))
    assert error.value.code == "invalid_gguf"
    assert not (tmp_path / "out").exists()


def test_missing_assigned_decoder_never_uses_mtp_to_fill_gap(tmp_path, monkeypatch):
    path = tmp_path / "gap.gguf"
    _write_source(path, mtp=2, omit_layer=1)
    monkeypatch.setattr(assets, "_run_cut", lambda *args, **kwargs: pytest.fail("must reject before cut"))
    assert _engine(path).get_pipeline_descriptor() == {}
    with pytest.raises(assets.SupplyError, match="missing an assigned decoder layer"):
        assets.build_gguf_segment(path, tmp_path / "out", **_build_args(path, (0, 2)))


_BASE = {"blk.0.attn.weight": 8, "blk.1.attn.weight": 16, "token_embd.weight": 32}


@pytest.mark.parametrize("block_count", [0, -1, True, 2.0, "2", None, 10**12])
def test_selection_rejects_invalid_or_unbounded_decoder_count(block_count):
    with pytest.raises(ValueError):
        select_gguf_tensors(_BASE, block_count=block_count)


@pytest.mark.parametrize("nextn", [-1, 2, 3, True, 1.0, "1", None])
def test_selection_rejects_invalid_mtp_metadata(nextn):
    with pytest.raises(ValueError):
        select_gguf_tensors(_BASE, block_count=2, nextn_predict_layers=nextn)


@pytest.mark.parametrize("span", [(-1, 1), (0, 0), (1, 1), (2, 1), (0, 3), (True, 2), (0, False), (0.0, 1), (0, "1")])
def test_selection_rejects_invalid_ranges(span):
    with pytest.raises(ValueError):
        select_gguf_tensors(_BASE, block_count=2, start_layer=span[0], end_layer=span[1])


@pytest.mark.parametrize("entry", [
    ("blk.0.bad", 0), ("blk.0.bad", -1), ("unknown", True), ("unknown", 1.0),
    ("unknown", "8"), ("", 8), (None, 8), ("blk.\u0660.attn.weight", 8),
])
def test_selection_rejects_invalid_names_and_sizes(entry):
    with pytest.raises(ValueError):
        select_gguf_tensors({**_BASE, entry[0]: entry[1]}, block_count=2)


@pytest.mark.parametrize("inventory", [{}, None, [], {"other.weight": 8}])
def test_selection_requires_a_complete_assigned_decoder_inventory(inventory):
    with pytest.raises(ValueError):
        select_gguf_tensors(inventory, block_count=2)


@pytest.mark.parametrize("span", [(0, 4), (-1, 2), (0, 5), (1, 1), (True, 2), (0, 2.0)])
def test_disk_budget_preserves_proper_segment_guard(tmp_path, span):
    path = tmp_path / "source.gguf"
    _write_source(path, mtp=2)
    with pytest.raises(assets.SupplyError) as error:
        assets.gguf_segment_disk_weight_bytes(path, *span)
    assert error.value.code == "invalid_range"


def test_builder_continues_to_forbid_whole_model_segments(tmp_path, monkeypatch):
    path = tmp_path / "source.gguf"
    _write_source(path, mtp=2)
    monkeypatch.setattr(assets, "_run_cut", lambda *args, **kwargs: pytest.fail("full model cut attempted"))
    with pytest.raises(assets.SupplyError) as error:
        assets.build_gguf_segment(path, tmp_path / "out", **_build_args(path, (0, 4)))
    assert error.value.code == "full_model_forbidden"


def test_descriptor_and_shared_rule_need_no_native_or_weight_loader(tmp_path):
    path = tmp_path / "cold.gguf"
    _write_source(path, mtp=2)
    script = r'''
import builtins, json, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def bounded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'gguf', 'numpy', 'llama_cpp', 'torch', 'transformers', 'safetensors'}:
        raise AssertionError('header inspection tried to import ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = bounded_import
from llama_engine import LlamaCppEngine
engine = LlamaCppEngine()
engine._model_path = sys.argv[2]
result = engine.get_pipeline_descriptor()
assert result['layer_weight_bytes'] == [32, 64, 96, 128]
assert result['runtime_ignored_component_bytes'] == {'mtp': 224}
assert engine._model is None and engine.is_loaded is False
assert not ({'gguf', 'numpy', 'llama_cpp', 'torch', 'transformers', 'safetensors'} & sys.modules.keys())
print(json.dumps(result))
'''
    result = subprocess.run([sys.executable, "-B", "-c", script,
                             str(Path(__file__).resolve().parents[1] / "src"), str(path)],
                            capture_output=True, text=True, encoding="utf-8", timeout=30,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["source_weight_bytes"] == 1088


@pytest.mark.parametrize("field,value,kind", [
    ("block_count", 0, 5), ("block_count", -1, 5),
    ("block_count", True, 7), ("block_count", False, 7),
    ("block_count", 5.0, 6), ("block_count", 5.5, 12),
    ("block_count", "5", 8), ("block_count", [5], 9),
    ("nextn_predict_layers", -1, 5), ("nextn_predict_layers", 5, 5),
    ("nextn_predict_layers", 6, 5), ("nextn_predict_layers", True, 7),
    ("nextn_predict_layers", False, 7), ("nextn_predict_layers", 1.5, 6),
    ("nextn_predict_layers", 0.0, 12), ("nextn_predict_layers", "1", 8),
    ("nextn_predict_layers", [1], 9),
    ("embedding_length", 0, 5), ("embedding_length", -1, 5),
    ("embedding_length", True, 7), ("embedding_length", False, 7),
    ("embedding_length", 8.0, 6), ("embedding_length", 8.5, 12),
    ("embedding_length", "8", 8), ("embedding_length", [8], 9),
])
def test_real_gguf_invalid_decoder_metadata_is_rejected_by_reader_and_builder(
    tmp_path, monkeypatch, field, value, kind,
):
    from relay_segment_info import read_gguf_layer_info

    path = tmp_path / "invalid-scalar.gguf"
    _write_source(path, mtp=1, scalar_overrides={field: (value, kind)})
    monkeypatch.setattr(assets, "_run_cut", lambda *args, **kwargs: pytest.fail("invalid scalar reached cut"))
    assert read_gguf_layer_info(path) is None
    assert _engine(path).get_pipeline_descriptor() == {}
    with pytest.raises(assets.SupplyError):
        assets.gguf_segment_disk_weight_bytes(path, 0, 2)
    with pytest.raises(assets.SupplyError):
        assets.build_gguf_segment(path, tmp_path / "out", **_build_args(path, (0, 2)))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4, 5, 10, 11])
def test_real_gguf_integer_decoder_metadata_types_remain_supported(tmp_path, kind):
    from relay_segment_info import read_gguf_layer_info

    path = tmp_path / "integer-scalars.gguf"
    _write_source(path, mtp=1, scalar_overrides={
        "block_count": (5, kind), "nextn_predict_layers": (1, kind),
        "embedding_length": (8, kind),
    })
    assert read_gguf_layer_info(path) == {
        "architecture": "qwen2", "block_count": 5, "nextn_predict_layers": 1,
        "n_layer": 4, "n_embd": 8,
    }
    assert _engine(path).get_pipeline_descriptor()["layer_weight_bytes"] == [32, 64, 96, 128]


def test_absent_embedding_is_unknown_in_diagnostics_and_not_executable(tmp_path):
    from relay_segment_info import read_gguf_layer_info

    path = tmp_path / "missing-embedding.gguf"
    _write_source(path, omit_metadata=("embedding_length",))
    assert read_gguf_layer_info(path)["n_embd"] == 0
    assert _engine(path).get_pipeline_descriptor() == {}
    with pytest.raises(assets.SupplyError):
        assets.gguf_segment_disk_weight_bytes(path, 0, 2)


def test_supply_capacity_matches_no_mtp_source_without_changing_margins(tmp_path):
    from model_supply import plan_segment_supply

    descriptors = []
    for mtp in (0, 2):
        path = tmp_path / f"capacity-mtp-{mtp}.gguf"
        _write_source(path, mtp=mtp, extras=(("unknown_aux.weight", 12),))
        descriptors.append(_engine(path).get_pipeline_descriptor())
    nodes = [{"node_id": name, "role": role, "capacity_bytes": 2000,
              "reserve_bytes": 11, "runtime_multiplier": 2.0, "engine": "llama_cpp"}
             for name, role in (("master", "master"), ("tail", "client"))]
    plans = [plan_segment_supply({"llama_cpp": descriptor}, nodes,
                                 master_engine="llama_cpp", safety_margin=1.2)
             for descriptor in descriptors]
    assert all(plan["admitted"] for plan in plans)
    for plan in plans:
        assert plan["safety_margin"] == 1.2
        assert len(plan["assignments"]) == 2
        assert all(0 < assignment["layers_count"] < 4 for assignment in plan["assignments"])
    projection = lambda plan: [(item["node_id"], item["start_layer"], item["end_layer"], item["required_bytes"])
                               for item in plan["assignments"]]
    assert projection(plans[0]) == projection(plans[1])
    # Reintroducing the old overcharge at the exact same capacity/margins
    # makes the true decoder-only placement fail, reproducing the defect.
    old = {**descriptors[1], "component_weight_bytes": {
        **descriptors[1]["component_weight_bytes"], "other": 48 + 224}}
    assert not plan_segment_supply({"llama_cpp": old}, nodes, master_engine="llama_cpp",
                                   safety_margin=1.2)["admitted"]
