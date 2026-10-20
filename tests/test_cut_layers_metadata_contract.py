"""Cutter metadata declarations describe the crop and verify without native imports."""
from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gguf_metadata_contract import read_gguf_metadata

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cut_layers.py"


def _source(path, *, mtp=False, arrays=False):
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), "qwen2")
    blocks = 4 + int(mtp)
    writer.add_block_count(blocks)
    writer.add_embedding_length(8)
    if mtp:
        writer.add_key_value("qwen2.nextn_predict_layers", 1, gguf.GGUFValueType.UINT32)
    if arrays:
        writer.add_key_value("qwen2.attention.head_count_kv", list(range(1, blocks + 1)),
                             gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
        writer.add_key_value("qwen2.attention.head_count", list(range(5, blocks + 5)),
                             gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
    else:
        writer.add_key_value("qwen2.attention.head_count_kv", 4, gguf.GGUFValueType.UINT32)
    writer.add_key_value("qwen2.attention.key_length", 256, gguf.GGUFValueType.UINT32)
    for layer in range(blocks):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", np.ones(8, dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True,
                          encoding="utf-8", errors="replace", check=False)


@pytest.mark.parametrize("args,span", [
    (("--keep-head", "2"), (0, 2)),
    (("--k", "1", "--end", "3"), (1, 3)),
    (("--k", "2"), (2, 4)),
])
def test_cut_manifest_uses_kept_count_and_preserves_integer_arrays(tmp_path, args, span):
    source = _source(tmp_path / "source.gguf", arrays=True)
    dst, native_path = tmp_path / "cut.gguf", tmp_path / "cut.json"
    result = _run("--src", source, "--dst", dst, "--manifest", native_path, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(native_path.read_text(encoding="utf-8"))
    actual = read_gguf_metadata(dst)
    assert native["gguf_metadata"] == actual
    assert native["block_count"] == 4
    assert native["gguf_metadata"]["block_count"] == span[1] - span[0]
    assert actual["head_count_kv"] == [1, 2, 3, 4][span[0]:span[1]]
    assert actual["full_attention_interval"] is None
    assert _run("--src", dst, "--verify-manifest", native_path).returncode == 0
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(str(dst))
    assert list(reader.fields["qwen2.attention.head_count"].contents()) == [5, 6, 7, 8][span[0]:span[1]]


def test_mtp_source_and_segment_metadata_are_distinct(tmp_path):
    source = _source(tmp_path / "source.gguf", mtp=True, arrays=True)
    dst, native_path = tmp_path / "cut.gguf", tmp_path / "cut.json"
    result = _run("--src", source, "--dst", dst, "--manifest", native_path, "--k", "2", "--end", "4")
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(native_path.read_text(encoding="utf-8"))
    assert native["block_count"] == 5
    assert native["nextn_predict_layers"] == 1
    assert native["gguf_metadata"]["block_count"] == 2
    assert native["gguf_metadata"]["nextn_predict_layers"] == 0
    assert native["gguf_metadata"]["head_count_kv"] == [3, 4]


@pytest.fixture
def cut_pair(tmp_path):
    source = _source(tmp_path / "source.gguf")
    dst, native_path = tmp_path / "cut.gguf", tmp_path / "cut.json"
    result = _run("--src", source, "--dst", dst, "--manifest", native_path, "--keep-head", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    return dst, native_path


@pytest.mark.parametrize("field", ["block_count", "head_count_kv", "key_length", "full_attention_interval"])
@pytest.mark.parametrize("missing", [False, True])
def test_verify_manifest_requires_all_four_exact_metadata_declarations(cut_pair, field, missing):
    dst, native_path = cut_pair
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if missing:
        native["gguf_metadata"].pop(field)
    else:
        native["gguf_metadata"][field] = 999
    native_path.write_text(json.dumps(native), encoding="utf-8")
    result = _run("--src", dst, "--verify-manifest", native_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert field in result.stdout


@pytest.mark.parametrize("legacy", [False, True])
def test_verify_reads_no_gguf_or_native_modules(cut_pair, monkeypatch, legacy):
    dst, native_path = cut_pair
    if legacy:
        native = json.loads(native_path.read_text(encoding="utf-8"))
        native.pop("gguf_metadata")
        native_path.write_text(json.dumps(native), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("ticket19_cut_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"gguf", "llama_cpp", "numpy", "torch"}
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--src", str(dst), "--verify-manifest", str(native_path)])
    assert cli.main() == (1 if legacy else 0)


@pytest.mark.parametrize("short_writes", [False, True])
def test_cutter_tensor_stream_uses_bounded_views_and_preserves_exact_bytes(short_writes):
    import io
    import numpy as np
    spec = importlib.util.spec_from_file_location("r6_cut_stream", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    tensor = np.arange(3 * 1024**2 + 17, dtype=np.uint8)
    class Sink(io.BytesIO):
        sizes = []
        def write(self, value):
            assert isinstance(value, memoryview) and value.obj is tensor
            self.sizes.append(len(value))
            assert len(value) <= 1024**2
            if short_writes:
                value = value[:min(len(value), 65537)]
            return super().write(value)
    sink = Sink()
    payload = cli._TensorWriteView(tensor)
    assert payload.shape == tensor.shape and payload.dtype == tensor.dtype
    assert payload.nbytes == tensor.nbytes
    payload.tofile(sink)
    assert sink.getvalue() == tensor.tobytes()
    assert len(sink.sizes) >= 4


def test_cutter_tensor_stream_fails_if_output_stops_accepting_bytes():
    import numpy as np
    spec = importlib.util.spec_from_file_location("r6_cut_stream", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    class NoSpace:
        def write(self, value):
            raise OSError(28, "No space left on device")
    with pytest.raises(OSError) as error:
        cli._TensorWriteView(np.ones(8, dtype=np.float32)).tofile(NoSpace())
    assert error.value.errno == 28
