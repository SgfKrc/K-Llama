"""Shared fail-closed validation and manifest helpers for local model assets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

MODEL_TYPES = {"safetensors", "gguf", "both"}
SAFE_SUFFIXES = (".safetensors", ".bin")


def normalize_model_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in MODEL_TYPES:
        raise ValueError("model_type must be safetensors, gguf, or both")
    return normalized


def _path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().absolute().resolve(strict=False)


def _weight_files(root: Path) -> list[Path]:
    return sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.name.lower().endswith(SAFE_SUFFIXES)
        and item.stat().st_size > 0
    )


def validate_model_artifact(
    model_type: str,
    model_path: str = "",
    gguf_path: str = "",
    *,
    require_exists: bool = True,
) -> dict[str, Any]:
    """Validate layout before a registry write and return normalized artifacts."""
    model_type = normalize_model_type(model_type)
    safe_root = _path(model_path) if model_path else None
    gguf = _path(gguf_path) if gguf_path else None
    safe_files: list[Path] = []

    if model_type in {"safetensors", "both"}:
        if safe_root is None:
            raise ValueError("safetensors model_path is required")
        if require_exists and not safe_root.is_dir():
            raise ValueError(f"safetensors model_path is not a directory: {model_path}")
        if safe_root.exists() and not safe_root.is_dir():
            raise ValueError(f"safetensors model_path is not a directory: {model_path}")
        config = safe_root / "config.json"
        if require_exists and not config.is_file():
            raise ValueError(f"safetensors config.json is missing: {model_path}")
        safe_files = _weight_files(safe_root) if safe_root.is_dir() else []
        if require_exists and not safe_files:
            raise ValueError(f"safetensors weights are missing or empty: {model_path}")

    if model_type in {"gguf", "both"}:
        if gguf is None:
            raise ValueError("gguf_path is required")
        if require_exists and not gguf.is_file():
            raise ValueError(f"GGUF path is not a file: {gguf_path}")
        if gguf.exists() and not gguf.is_file():
            raise ValueError(f"GGUF path is not a file: {gguf_path}")
        if gguf.suffix.lower() != ".gguf":
            raise ValueError(f"GGUF path must end with .gguf: {gguf_path}")
        if require_exists and gguf.stat().st_size <= 0:
            raise ValueError(f"GGUF file is empty: {gguf_path}")

    if model_type == "gguf" and model_path:
        raise ValueError("gguf model_type cannot include model_path")
    return {
        "model_type": model_type,
        "model_path": str(safe_root) if safe_root else "",
        "gguf_path": str(gguf) if gguf else "",
        "safetensors_files": safe_files,
        "files": [*safe_files, *([gguf] if gguf else [])],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_external_files(root: Path, files: Iterable[Path]) -> dict[str, str]:
    """Bind v1 external aliases to explicitly registered files, never guessed paths."""
    root = _path(root)
    result: dict[str, str] = {}
    for item in files:
        item = _path(item)
        if item.is_relative_to(root):
            continue
        alias = f"external/{item.name}"
        if alias in result and result[alias] != str(item):
            raise ValueError("ambiguous external manifest filename")
        result[alias] = str(item)
    return result


def build_manifest(
    root: Path,
    files: Iterable[Path],
    *,
    model_type: str,
    revision: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Build a stable manifest with relative paths and per-file hashes."""
    root = _path(root)
    selected = {Path(path).absolute().resolve(strict=False) for path in files}
    # Bind the loader metadata as well as weights so config/tokenizer drift is visible.
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors.index.json"):
        candidate = root / name
        if candidate.is_file():
            selected.add(candidate)
    manifest_external_files(root, selected)
    entries: list[dict[str, Any]] = []
    for item in sorted(selected, key=lambda value: str(value).lower()):
        item = item.absolute().resolve(strict=False)
        try:
            relative = item.relative_to(root).as_posix()
        except ValueError:
            # A both-format registration may keep a GGUF beside the HF directory.
            relative = f"external/{item.name}"
        entries.append({"path": relative, "size_bytes": item.stat().st_size, "sha256": sha256_file(item)})
    metadata: dict[str, Any] = {
        "schema": 1,
        "model_type": normalize_model_type(model_type),
        "revision": revision or "",
        "source": source or "",
        "files": entries,
    }
    canonical = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata["artifact_sha256"] = hashlib.sha256(
        "\n".join(f"{entry['path']}\0{entry['size_bytes']}\0{entry['sha256']}" for entry in entries).encode("utf-8")
    ).hexdigest()
    metadata["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return metadata


def write_manifest(root: Path, manifest: dict[str, Any], filename: str = "model.manifest.json") -> Path:
    """Atomically publish a manifest next to the staged model files."""
    root = _path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    from atomic_json import atomic_write_json

    atomic_write_json(destination, manifest, ensure_ascii=True, sort_keys=True,
                      indent=2, trailing_newline=True)
    return destination
