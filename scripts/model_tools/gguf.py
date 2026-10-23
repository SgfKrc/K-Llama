"""Small, dependency-free GGUF header and integrity reader."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

MAGIC = b"GGUF"
MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ITEMS = 2_000_000

VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}

_SCALAR_STRUCTS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


class GGUFError(ValueError):
    """Raised when a GGUF file cannot be safely parsed."""


@dataclass(frozen=True)
class _Tensor:
    name: str
    shape: tuple[int, ...]
    type_code: int
    type_name: str
    offset: int
    byte_size: int | None


class _Reader:
    def __init__(self, handle: BinaryIO, file_size: int):
        self.handle = handle
        self.file_size = file_size

    def tell(self) -> int:
        return int(self.handle.tell())

    def exact(self, length: int) -> bytes:
        if length < 0 or length > self.file_size - self.tell():
            raise GGUFError(f"truncated GGUF at offset {self.tell()}")
        data = self.handle.read(length)
        if len(data) != length:
            raise GGUFError(f"truncated GGUF at offset {self.tell()}")
        return data

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.exact(size))[0]

    def u32(self) -> int:
        return int(self.unpack("<I"))

    def u64(self) -> int:
        return int(self.unpack("<Q"))

    def string(self) -> str:
        length = self.u64()
        if length > MAX_STRING_BYTES:
            raise GGUFError(f"GGUF string is too large: {length} bytes")
        try:
            return self.exact(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError(f"invalid UTF-8 metadata at offset {self.tell()}") from exc


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _read_value(reader: _Reader, value_type: int, depth: int = 0) -> Any:
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        if depth > 4:
            raise GGUFError("nested GGUF metadata arrays are too deep")
        element_type = reader.u32()
        count = reader.u64()
        if element_type not in VALUE_TYPES or element_type == 9 and depth >= 4:
            raise GGUFError(f"unsupported GGUF array element type: {element_type}")
        if count > MAX_ARRAY_ITEMS:
            raise GGUFError(f"GGUF metadata array is too large: {count} items")
        preview: list[Any] = []
        for index in range(count):
            item = _read_value(reader, element_type, depth + 1)
            if index < 8:
                preview.append(item)
        return {
            "type": VALUE_TYPES[element_type],
            "count": count,
            "preview": preview,
        }
    fmt = _SCALAR_STRUCTS.get(value_type)
    if fmt is None:
        raise GGUFError(f"unsupported GGUF metadata type: {value_type}")
    return reader.unpack(fmt)


def _tensor_size(shape: tuple[int, ...], type_code: int) -> int | None:
    descriptor = GGML_TYPES.get(type_code)
    if descriptor is None:
        return None
    _, block_size, bytes_per_block = descriptor
    elements = math.prod(shape) if shape else 1
    if block_size > 1 and elements % block_size:
        return None
    return (elements // block_size) * bytes_per_block


def _tensor_dict(tensor: _Tensor) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "type": tensor.type_name,
        "type_code": tensor.type_code,
        "offset": tensor.offset,
        "byte_size": tensor.byte_size,
    }


def inspect_gguf(path: str | Path) -> dict[str, Any]:
    """Parse GGUF metadata and tensor descriptors without loading tensor data."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise GGUFError(f"GGUF file does not exist: {target}")
    file_size = target.stat().st_size
    tensors: list[_Tensor] = []
    metadata: dict[str, Any] = {}
    errors: list[str] = []
    with target.open("rb") as handle:
        reader = _Reader(handle, file_size)
        if reader.exact(4) != MAGIC:
            raise GGUFError("invalid GGUF magic")
        version = reader.u32()
        if version not in {1, 2, 3}:
            raise GGUFError(f"unsupported GGUF version: {version}")
        tensor_count = reader.u64()
        metadata_count = reader.u64()
        if tensor_count > MAX_ARRAY_ITEMS or metadata_count > MAX_ARRAY_ITEMS:
            raise GGUFError("GGUF header count is unreasonably large")
        for _ in range(metadata_count):
            key = reader.string()
            if key in metadata:
                raise GGUFError(f"duplicate GGUF metadata key: {key}")
            value_type = reader.u32()
            metadata[key] = _read_value(reader, value_type)
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = reader.u32()
            if dimensions > 64:
                raise GGUFError(f"tensor has too many dimensions: {dimensions}")
            shape = tuple(reader.u64() for _ in range(dimensions))
            type_code = reader.u32()
            offset = reader.u64()
            descriptor = GGML_TYPES.get(type_code)
            type_name = descriptor[0] if descriptor else f"TYPE_{type_code}"
            tensors.append(_Tensor(name, shape, type_code, type_name, offset, _tensor_size(shape, type_code)))
        alignment_value = metadata.get("general.alignment", 32)
        alignment = int(alignment_value) if isinstance(alignment_value, (int, float)) else 32
        if alignment < 1 or alignment > 1024 * 1024 or alignment & (alignment - 1):
            errors.append(f"invalid alignment: {alignment}")
            alignment = 32
        data_offset = _align(reader.tell(), alignment)

    if data_offset > file_size:
        errors.append("tensor data offset is beyond end of file")
    ranges: list[tuple[int, int, str]] = []
    for tensor in tensors:
        file_offset = data_offset + tensor.offset
        if file_offset >= file_size and file_size > 0:
            errors.append(f"tensor offset beyond end of file: {tensor.name}")
        if file_offset % alignment:
            errors.append(f"tensor offset is not aligned: {tensor.name}")
        if tensor.byte_size is not None:
            end = file_offset + tensor.byte_size
            if end > file_size:
                errors.append(f"tensor data truncated: {tensor.name}")
            ranges.append((file_offset, end, tensor.name))
        elif tensor.type_code not in GGML_TYPES:
            errors.append(f"unknown tensor type: {tensor.type_code} ({tensor.name})")
        else:
            errors.append(f"tensor shape is not block-aligned: {tensor.name}")
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            errors.append(f"overlapping tensor data: {previous[2]} / {current[2]}")

    type_counts: dict[str, int] = {}
    for tensor in tensors:
        type_counts[tensor.type_name] = type_counts.get(tensor.type_name, 0) + 1
    architecture = metadata.get("general.architecture")
    derived: dict[str, Any] = {
        "architecture": architecture,
        "name": metadata.get("general.name"),
        "context_length": metadata.get(f"{architecture}.context_length") if isinstance(architecture, str) else None,
        "block_count": metadata.get(f"{architecture}.block_count") if isinstance(architecture, str) else None,
        "vocab_size": metadata.get(f"{architecture}.vocab_size") if isinstance(architecture, str) else None,
        "tokenizer_model": metadata.get("tokenizer.ggml.model"),
        "tensor_types": type_counts,
    }
    return {
        "path": str(target.resolve()),
        "file_size_bytes": file_size,
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "data_offset": data_offset,
        "metadata": metadata,
        "tensors": [_tensor_dict(tensor) for tensor in tensors],
        "derived": derived,
        "valid": not errors,
        "errors": errors,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_candidates(path: Path) -> list[Path]:
    candidates = [path.with_name(path.name + ".sha256"), path.with_suffix(".sha256")]
    return list(dict.fromkeys(candidate for candidate in candidates if candidate.is_file()))


def _read_sidecar(sidecar: Path, target: Path) -> str | None:
    pattern = re.compile(r"\b([0-9a-fA-F]{64})\b")
    lines = sidecar.read_text(encoding="utf-8", errors="replace").splitlines()
    fallback: str | None = None
    for line in lines:
        match = pattern.search(line)
        if not match:
            continue
        value = match.group(1).lower()
        fallback = fallback or value
        filename = line[match.end():].strip().lstrip("* ").split()[0] if line[match.end():].strip() else ""
        if filename and Path(filename).name == target.name:
            return value
    return fallback


def verify_gguf(path: str | Path, *, full_hash: bool = False,
                digest: Callable[[Path], str] | None = None) -> dict[str, Any]:
    """Verify GGUF structure and an adjacent sha256 sidecar when present.

    ``digest`` optionally supplies the content digest implementation; callers
    that already hold a size/mtime-guarded digest may inject it so one file is
    not re-read. The default reads the whole file, unchanged.
    """
    target = Path(path).expanduser()
    try:
        inspection = inspect_gguf(target)
    except (OSError, GGUFError) as exc:
        return {"path": str(target), "valid": False, "errors": [str(exc)], "structure_valid": False}
    sidecars = _sidecar_candidates(target)
    expected: str | None = None
    sidecar_path: Path | None = None
    sidecar_errors: list[str] = []
    for candidate in sidecars:
        try:
            value = _read_sidecar(candidate, target)
        except OSError as exc:
            sidecar_errors.append(f"cannot read sidecar {candidate.name}: {exc}")
            continue
        if value:
            if expected and expected != value:
                sidecar_errors.append("conflicting sha256 sidecars")
            expected = expected or value
            sidecar_path = candidate
    actual = (digest or _sha256)(target) if (full_hash or expected) else None
    errors = list(inspection.get("errors", [])) + sidecar_errors
    if expected and actual and expected != actual:
        errors.append(f"sha256 mismatch: expected {expected}, actual {actual}")
    return {
        "path": str(target.resolve()),
        "valid": not errors,
        "structure_valid": bool(inspection.get("valid")),
        "sha256": actual,
        "sha256_expected": expected,
        "sha256_checked": actual is not None,
        "sidecar": str(sidecar_path.resolve()) if sidecar_path else None,
        "inspection": inspection,
        "errors": errors,
    }
