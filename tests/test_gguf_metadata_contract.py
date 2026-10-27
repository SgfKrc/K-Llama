"""The metadata contract is strict and reads only bounded GGUF headers."""
from __future__ import annotations

import builtins
import os
import struct
from pathlib import Path

import pytest

from gguf_metadata_contract import (
    GGUF_METADATA_KEYS, GGUF_METADATA_REQUIRED_KEYS, normalize_gguf_metadata,
    project_gguf_metadata, read_gguf_header, read_gguf_metadata, verify_gguf_metadata,
)


def _string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _header(path, extra=(), *, architecture_last=False):
    fields = [("general.architecture", 8, _string("qwen2")),
              ("qwen2.block_count", 4, struct.pack("<I", 4))]
    fields.extend(extra)
    if architecture_last:
        fields.append(fields.pop(0))
    # Intentionally no tensor payload/table: metadata reads must stop at KV end.
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, len(fields)) + b"".join(
        _string(key) + struct.pack("<I", kind) + data for key, kind, data in fields))
    return path


def _critical(**updates):
    return dict(block_count=4, head_count_kv=None, key_length=None,
                full_attention_interval=None, **updates)


def test_metadata_read_has_explicit_optional_absence_without_native_imports(tmp_path, monkeypatch):
    path = _header(tmp_path / "header.gguf", architecture_last=True)
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"gguf", "llama_cpp", "numpy", "torch"}
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    actual = read_gguf_metadata(path)
    assert set(actual) == set(GGUF_METADATA_KEYS)
    assert actual["architecture"] == "qwen2"
    assert actual["block_count"] == 4
    assert all(actual[key] is None for key in GGUF_METADATA_KEYS
               if key not in {"architecture", "block_count"})


def test_reader_preserves_integer_arrays_and_skips_unrelated_arrays(tmp_path):
    path = _header(tmp_path / "arrays.gguf", [
        ("qwen2.attention.head_count_kv", 9, struct.pack("<IQ4I", 4, 4, 1, 2, 3, 4)),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _string("x") + _string("y")),
        ("qwen2.attention.key_length", 4, struct.pack("<I", 128)),
    ], architecture_last=True)
    actual = read_gguf_metadata(path)
    assert actual["head_count_kv"] == [1, 2, 3, 4]
    assert actual["key_length"] == 128
    raw, tensors = read_gguf_header(path, include_tensors=False)
    assert tensors == {}
    assert normalize_gguf_metadata(raw) == actual
    assert project_gguf_metadata(actual, 1, 3)["head_count_kv"] == [2, 3]


def test_header_parse_is_reused_across_hardlink_paths_and_invalidated_on_change(
        tmp_path, monkeypatch):
    import gguf_metadata_contract as contract

    source = _header(tmp_path / "source.gguf")
    alias = tmp_path / "alias.gguf"
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    contract._HEADER_CACHE.clear()
    calls = []
    original = contract._read_gguf_header

    def counted(path, *, include_tensors=True):
        calls.append(Path(path))
        return original(path, include_tensors=include_tensors)

    monkeypatch.setattr(contract, "_read_gguf_header", counted)
    first = contract.read_gguf_header(source, include_tensors=False)
    second = contract.read_gguf_header(alias, include_tensors=False)
    assert first == second and calls == [source]

    original_bytes = source.read_bytes()
    old_value = _string("qwen2.block_count") + struct.pack("<II", 4, 4)
    new_value = _string("qwen2.block_count") + struct.pack("<II", 4, 5)
    assert original_bytes.count(old_value) == 1
    source.write_bytes(original_bytes.replace(old_value, new_value))
    changed, _ = contract.read_gguf_header(alias, include_tensors=False)
    assert changed["qwen2.block_count"] == 5
    assert calls == [source, alias]


@pytest.mark.parametrize("key", GGUF_METADATA_REQUIRED_KEYS)
@pytest.mark.parametrize("side", ["actual", "expected"])
def test_missing_critical_key_never_equals_explicit_none(key, side):
    actual = _critical()
    expected = _critical()
    (actual if side == "actual" else expected).pop(key)
    problems = verify_gguf_metadata(actual, expected)
    assert any(key in item and side in item and "missing" in item for item in problems)


@pytest.mark.parametrize("left,right", [
    (True, 1), (1, True), (1.0, 1), ([True], [1]), ([1.0], [1]),
    ((1, 2), [1, 2]), ([1, 2], 1), (None, 0), (None, []),
    ([True], [True]), ("1", "1"),
])
def test_values_require_exact_integer_list_or_none_types(left, right):
    actual, expected = _critical(), _critical()
    actual["head_count_kv"], expected["head_count_kv"] = left, right
    assert any("head_count_kv" in item for item in verify_gguf_metadata(actual, expected))


@pytest.mark.parametrize("value", [None, 0, 4, [], [1, 2]])
def test_exact_values_are_valid(value):
    actual = _critical()
    actual["head_count_kv"] = value
    assert verify_gguf_metadata(actual, actual) == []


def test_optional_declared_key_also_cannot_skip_missing_actual():
    expected = {**_critical(), "n_embd": None}
    assert any("n_embd" in item and "missing" in item
               for item in verify_gguf_metadata(_critical(), expected))


@pytest.mark.parametrize("data", [b"", b"NOTGGUF", b"GGUF" + struct.pack("<IQQ", 3, 0, 100_001)])
def test_invalid_headers_fail_closed(tmp_path, data):
    path = tmp_path / "bad.gguf"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        read_gguf_metadata(path)


def test_duplicate_and_oversized_headers_are_rejected(tmp_path):
    path = _header(tmp_path / "duplicate.gguf", [
        ("qwen2.block_count", 4, struct.pack("<I", 2)),
    ])
    with pytest.raises(ValueError, match="duplicate"):
        read_gguf_metadata(path)
    path.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 65 * 1024 * 1024))
    with pytest.raises(ValueError, match="limit"):
        read_gguf_metadata(path)


def test_projection_drops_source_mtp_and_uses_kept_blocks():
    source = {**_critical(), "block_count": 5, "nextn_predict_layers": 1}
    result = project_gguf_metadata(source, 2, 4)
    assert result["block_count"] == 2
    assert result["nextn_predict_layers"] == 0
    assert source["block_count"] == 5
    assert project_gguf_metadata(source, 2, 5)["nextn_predict_layers"] == 1


def test_projection_rejects_wrong_per_layer_array_length():
    source = {**_critical(), "head_count_kv": [1, 2, 3]}
    with pytest.raises(ValueError, match="head_count_kv"):
        project_gguf_metadata(source, 1, 3)
