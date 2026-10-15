"""Bounded GGUF headers and a strict metadata contract, using only stdlib.

read_gguf_metadata raises OSError/ValueError for unreadable or malformed headers.
An absent optional KV is returned as None; absence from a contract dictionary is
an error, never a wildcard. No tensor payload or native model loader is used.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Mapping

GGUF_METADATA_KEYS = (
    "architecture", "block_count", "head_count_kv", "key_length",
    "full_attention_interval", "n_embd", "nextn_predict_layers",
)
GGUF_METADATA_REQUIRED_KEYS = (
    "block_count", "head_count_kv", "key_length", "full_attention_interval",
)
_MAX_HEADER = 64 * 1024 * 1024
_CONTRACT_SUFFIXES = (
    ".block_count", ".attention.head_count", ".attention.head_count_kv",
    ".attention.key_length", ".full_attention_interval", ".embedding_length",
    ".nextn_predict_layers",
)


def _invalid(message: str) -> None:
    raise ValueError(message)


def read_gguf_header(path: str | Path, *, include_tensors: bool = True) -> tuple[dict, dict[str, tuple[int, int, tuple, int]]]:
    """Read bounded metadata and tensor ranges, never tensor payloads."""
    from relay_segment_info import _GGML_TYPE_SIZES, _GGUF_SCALAR_TYPES
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        def read(count):
            if count < 0 or handle.tell() + count > min(size, _MAX_HEADER):
                _invalid("GGUF header is truncated")
            data = handle.read(count)
            if len(data) != count:
                _invalid("GGUF header is truncated")
            return data

        def unpack(fmt):
            return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]

        def string():
            count = unpack("<Q")
            if count > _MAX_HEADER:
                _invalid("GGUF string exceeds metadata limit")
            return read(count).decode("utf-8")

        def value(kind, depth=0, *, retain=False):
            if depth > 2:
                _invalid("GGUF arrays are nested too deeply")
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = unpack("<I"), unpack("<Q")
                if count > 10_000_000:
                    _invalid("GGUF array exceeds metadata limit")
                if retain:
                    return [value(subtype, depth + 1, retain=True) for _ in range(count)]
                if subtype in _GGUF_SCALAR_TYPES:
                    length = count * _GGUF_SCALAR_TYPES[subtype][1]
                    if handle.tell() + length > min(size, _MAX_HEADER):
                        _invalid("GGUF array is truncated or exceeds metadata limit")
                    handle.seek(length, 1)
                else:
                    for _ in range(count):
                        value(subtype, depth + 1)
                return None
            if kind not in _GGUF_SCALAR_TYPES:
                _invalid("GGUF metadata type is unsupported")
            return unpack(_GGUF_SCALAR_TYPES[kind][0])

        if read(4) != b"GGUF" or unpack("<I") not in {2, 3}:
            _invalid("source is not a supported GGUF")
        count, metadata_count = unpack("<Q"), unpack("<Q")
        if count > 1_000_000 or metadata_count > 100_000:
            _invalid("GGUF header exceeds metadata limits")
        metadata = {}
        for _ in range(metadata_count):
            key = string()
            if key in metadata:
                _invalid("GGUF contains duplicate metadata")
            metadata[key] = value(unpack("<I"), retain=key.endswith(_CONTRACT_SUFFIXES))
        if not include_tensors:
            return metadata, {}
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or alignment <= 0 or alignment > _MAX_HEADER:
            _invalid("GGUF alignment is invalid")
        tensors = {}
        for _ in range(count):
            name, dimensions = string(), unpack("<I")
            if name in tensors or not 1 <= dimensions <= 4:
                _invalid("GGUF tensor name or dimensions are invalid")
            shape = tuple(unpack("<Q") for _ in range(dimensions))
            kind, offset = unpack("<I"), unpack("<Q")
            if kind not in _GGML_TYPE_SIZES or any(dim <= 0 for dim in shape):
                _invalid("GGUF tensor type or shape is unsupported")
            block, width = _GGML_TYPE_SIZES[kind]
            elements = math.prod(shape)
            if elements % block or offset % alignment:
                _invalid("GGUF tensor has invalid block alignment")
            tensors[name] = (offset, elements // block * width, shape, kind)
        data_start = (handle.tell() + alignment - 1) // alignment * alignment
        last_end = data_start
        for name, (offset, length, shape, kind) in sorted(tensors.items(), key=lambda item: item[1][0]):
            absolute = data_start + offset
            if absolute < last_end or absolute + length > size:
                _invalid("GGUF tensor range overlaps or is truncated")
            tensors[name] = (absolute, length, shape, kind)
            last_end = absolute + length
    return metadata, tensors


def normalize_gguf_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize raw GGUF KVs, preserving integer arrays and explicit absence."""
    arch = metadata.get("general.architecture")
    if type(arch) is not str or not arch:
        raise ValueError("GGUF architecture is missing or invalid")
    names = {
        "block_count": "block_count", "head_count_kv": "attention.head_count_kv",
        "key_length": "attention.key_length", "full_attention_interval": "full_attention_interval",
        "n_embd": "embedding_length", "nextn_predict_layers": "nextn_predict_layers",
    }
    result = {"architecture": arch}
    result.update({key: metadata.get(f"{arch}.{suffix}") for key, suffix in names.items()})
    return result


def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
    """Read seven normalized keys from the metadata header, without native imports."""
    raw, _ = read_gguf_header(path, include_tensors=False)
    result = normalize_gguf_metadata(raw)
    problems = verify_gguf_metadata(result, result)
    if problems:
        raise ValueError("invalid GGUF metadata: " + "; ".join(problems))
    if type(result["block_count"]) is not int or result["block_count"] <= 0:
        raise ValueError("GGUF block_count must be a positive integer")
    return result


def _metadata_value(value: Any) -> bool:
    return (value is None or type(value) is int
            or (type(value) is list and all(type(item) is int for item in value)))


def verify_gguf_metadata(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """Return named missing/type/value errors. All four critical keys are mandatory.

    Optional keys outside the four critical keys are compared when expected
    declares them. bool, float, tuple and arrays containing non-ints never equal
    an integer/list contract, even if Python's ordinary == would accept them.
    """
    problems = []
    if not isinstance(actual, Mapping):
        return ["actual gguf_metadata must be a mapping"]
    if not isinstance(expected, Mapping):
        return ["expected gguf_metadata must be a mapping"]
    keys = list(GGUF_METADATA_REQUIRED_KEYS)
    keys.extend(key for key in GGUF_METADATA_KEYS if key not in keys and key in expected)
    for key in keys:
        absent = False
        for side, values in (("actual", actual), ("expected", expected)):
            if key not in values:
                problems.append(f"{key}: missing from {side} metadata")
                absent = True
        if absent:
            continue
        left, right = actual[key], expected[key]
        valid = (type(left) is str and type(right) is str) if key == "architecture" else (
            _metadata_value(left) and _metadata_value(right))
        if not valid or type(left) is not type(right) or left != right:
            problems.append(f"{key}: actual={left!r} ({type(left).__name__}), "
                            f"expected={right!r} ({type(right).__name__})")
    return problems


def project_gguf_metadata(metadata: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    """Project a source header onto a half-open block range, including array KVs."""
    blocks = metadata["block_count"]
    if type(blocks) is not int or not 0 <= start < end <= blocks:
        raise ValueError("block_count: invalid segment range")
    result = dict(metadata)
    result["block_count"] = end - start
    for key in ("head_count_kv", "key_length", "full_attention_interval"):
        value = metadata[key]
        if type(value) is list:
            if len(value) != blocks or any(type(item) is not int for item in value):
                raise ValueError(f"{key}: integer array length must equal source block_count={blocks}")
            result[key] = value[start:end]
    nextn = metadata.get("nextn_predict_layers")
    if type(nextn) is int and nextn > 0:
        result["nextn_predict_layers"] = max(0, end - max(start, blocks - nextn))
    return result
