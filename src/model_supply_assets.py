"""Assignment-sized, verified model supply artifacts (MODEL-SUPPLY-01).

No model loader is used: GGUF cutting runs in a child process and Safetensors
weights are copied by byte range. Builders/receivers atomically publish an
immutable directory. Layer bytes use absolute source indices, with zero for
unassigned layers. Canonical JSON is ASCII, sorted-key, compact JSON.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from gguf_metadata_contract import (
    normalize_gguf_metadata, project_gguf_metadata, verify_gguf_metadata,
)

MANIFEST_KIND = "qlh.segment_supply.v1"
_MARKER = "supply.manifest.json"
_RESUME_MARKER = ".supply-resume.json"
_RESUME_INTERVAL = 16 * 1024 * 1024
_CHUNK = 1024 * 1024
_MAX_HEADER = 64 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_FIELDS = {
    "manifest_kind", "operation_id", "node_id", "source_model_id",
    "source_model_sha256", "tokenizer_sha256", "total_layers", "layer_range",
    "segment_mode", "layer_weight_bytes", "hidden_size", "artifact_sha256",
    "conversion_proof", "files", "manifest_sha256", "format", "model_type",
    "has_embedding", "has_lm_head", "tie_word_embeddings",
}
_MODEL_SUFFIXES = (
    ".safetensors", ".bin", ".json", ".py", ".tiktoken", ".model", ".txt",
    ".jinja", ".spm", ".vocab",
)
__all__ = [
    "SupplyError", "sha256_file", "build_gguf_segment",
    "build_safetensors_segment", "receive_segment", "validate_supply_manifest",
]


class SupplyError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _fail(code: str, message: str) -> None:
    raise SupplyError(code, message)


def sha256_file(path: str | Path) -> str:
    """Hash live bytes, bypassing mtime-based hash caches."""
    return _hash_file(path)


def _hash_file(path: str | Path, cancelled: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    try:
        path = Path(path)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                if cancelled is not None:
                    _cancelled(cancelled)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        current = path.stat()
        # Windows fstat/path.stat use different ctime semantics; compare live
        # file identity, size and nanosecond mtime consistently across APIs.
        def signature(stat):
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature(before) != signature(after) or signature(after) != signature(current):
            _fail("file_digest_mismatch", "artifact changed while hashing")
        if cancelled is not None:
            _cancelled(cancelled)
    except OSError as exc:
        raise SupplyError("file_unavailable", "artifact cannot be read") from exc
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items()
                if isinstance(key, str) and key not in {"local_files", "private"}
                and not key.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return value


def _check_public(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _check_public(key)
            _check_public(item)
    elif isinstance(value, list):
        for item in value:
            _check_public(item)
    elif isinstance(value, str):
        if value.startswith(("/", "\\", "~/")) or re.search(r"[A-Za-z]:[\\/]", value):
            _fail("unsafe_manifest", "public metadata contains an absolute path")
    elif value is not None and type(value) not in (int, float, bool):
        _fail("invalid_manifest", "manifest is not JSON data")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail("invalid_manifest", f"{name} must be an integer >= {minimum}")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        _fail("invalid_manifest", f"{name} must be a lowercase SHA256")
    return value


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or value in {".", ".."}:
        _fail("invalid_identity", f"invalid {name}")
    return value


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail("unsafe_path", "invalid artifact path")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
    }
    for part in value.split("/"):
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part.startswith(".")
                or part.endswith((".", " ")) or part.lower().endswith(".part")
                or part.split(".")[0].upper() in reserved):
            _fail("unsafe_path", "unsafe artifact path")
    if value.lower() == _MARKER:
        _fail("unsafe_path", "artifact collides with supply marker")
    return value


def _link(path: Path) -> bool:
    # Refuse every Windows reparse type, including cloud placeholders.
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _safe_path(root: Path, relative: str) -> Path:
    _relative(relative)
    current = root
    if _link(root):
        _fail("unsafe_path", "artifact root is a link")
    for part in relative.split("/"):
        current = current / part
        if _link(current):
            _fail("unsafe_path", "artifact path is a link")
    try:
        current.resolve().relative_to(root.resolve())
    except ValueError:
        _fail("unsafe_path", "artifact escapes destination")
    return current


def _weight_digest(files: list[dict]) -> str:
    return _digest([item for item in files if item["kind"] == "weights"])


def _mode(start: int, end: int, total: int) -> str:
    if start == 0:
        return "head"
    return "tail" if end == total else "middle"


def validate_supply_manifest(payload: dict) -> dict:
    """Return a detached, private-free manifest after strict validation."""
    if not isinstance(payload, dict):
        _fail("invalid_manifest", "manifest must be an object")
    try:
        public = json.loads(_json_bytes(_public(payload)))
    except (ValueError, TypeError, RecursionError) as exc:
        raise SupplyError("invalid_manifest", "manifest is not finite JSON") from exc
    optional = {"model_type", "has_embedding", "has_lm_head", "tie_word_embeddings"}
    if set(public) - _FIELDS or not (_FIELDS - optional).issubset(public):
        _fail("invalid_manifest", "manifest fields are incomplete or unknown")
    _check_public(public)
    if public["manifest_kind"] != MANIFEST_KIND:
        _fail("invalid_manifest", "unsupported supply manifest kind")
    for name in ("operation_id", "node_id", "source_model_id"):
        _identity(public[name], name)
    for name in ("source_model_sha256", "tokenizer_sha256", "artifact_sha256", "manifest_sha256"):
        _hash(public[name], name)
    total = _integer(public["total_layers"], "total_layers", 1)
    if total > 100_000:
        _fail("invalid_manifest", "total_layers exceeds metadata limit")
    span = public["layer_range"]
    if not isinstance(span, list) or len(span) != 2:
        _fail("invalid_manifest", "layer_range must contain two integers")
    start, end = [_integer(item, "layer_range") for item in span]
    if start == 0 and end == total:
        _fail("full_model_forbidden", "distributed supply requires proper segments")
    if not 0 <= start < end <= total or public["segment_mode"] != _mode(start, end, total):
        _fail("invalid_manifest", "layer range or segment mode is inconsistent")
    _integer(public["hidden_size"], "hidden_size", 1)
    layer_bytes = public["layer_weight_bytes"]
    if not isinstance(layer_bytes, list) or len(layer_bytes) != total:
        _fail("invalid_manifest", "layer_weight_bytes must use absolute layer indices")
    for index, size in enumerate(layer_bytes):
        _integer(size, "layer_weight_bytes")
        if (start <= index < end and size == 0) or (not start <= index < end and size != 0):
            _fail("invalid_manifest", "layer bytes differ from the assigned range")
    _proof_source(public)
    proof = public["conversion_proof"]
    if proof is not None and not isinstance(proof, dict):
        _fail("invalid_manifest", "conversion_proof must be an object or null")
    for name in ("has_embedding", "has_lm_head", "tie_word_embeddings"):
        if name in public and type(public[name]) is not bool:
            _fail("invalid_manifest", f"{name} must be a boolean")
    files = public["files"]
    if not isinstance(files, list) or not files or len(files) > 100_000:
        _fail("invalid_manifest", "manifest has no valid file inventory")
    paths = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256", "kind"}:
            _fail("invalid_manifest", "invalid artifact file entry")
        relative = _relative(item["path"])
        if relative.lower() in paths:
            _fail("unsafe_path", "duplicate artifact path")
        paths.add(relative.lower())
        _integer(item["size_bytes"], "size_bytes")
        _hash(item["sha256"], "file sha256")
        if item["kind"] not in {"weights", "support", "manifest"}:
            _fail("invalid_manifest", "invalid artifact file kind")
    for path in paths:
        parts = path.split("/")
        if any("/".join(parts[:index]) in paths for index in range(1, len(parts))):
            _fail("unsafe_path", "artifact paths collide with directories")
    weights = [item for item in files if item["kind"] == "weights"]
    if public["format"] == "gguf":
        if len(weights) != 1 or not weights[0]["path"].endswith(".gguf"):
            _fail("invalid_manifest", "GGUF supply must contain one weight file")
        paired = Path(weights[0]["path"]).with_suffix(".manifest.json").as_posix()
        if not any(item["path"] == paired and item["kind"] == "manifest" for item in files):
            _fail("invalid_manifest", "GGUF supply lacks its native manifest")
        expected_artifact = weights[0]["sha256"]
    elif public["format"] == "safetensors":
        if not weights or any(not item["path"].endswith(".safetensors") for item in weights):
            _fail("invalid_manifest", "Safetensors weight inventory is invalid")
        if not {"config.json", "model.safetensors.index.json", "tokenizer.json"}.issubset(paths):
            _fail("invalid_manifest", "Safetensors supply lacks config, index or tokenizer")
        expected_artifact = _weight_digest(files)
    else:
        _fail("invalid_manifest", "unsupported supply artifact format")
    if public["artifact_sha256"] != expected_artifact:
        _fail("artifact_digest_mismatch", "artifact digest differs from file inventory")
    unsigned = dict(public)
    unsigned.pop("manifest_sha256")
    if _digest(unsigned) != public["manifest_sha256"]:
        _fail("manifest_digest_mismatch", "supply manifest digest mismatch")
    return public


def _base_manifest(*, source_model_id, source_model_sha256, tokenizer_sha256,
                   start_layer, end_layer, total_layers, operation_id, node_id,
                   conversion_proof=None) -> dict:
    payload = {
        "manifest_kind": MANIFEST_KIND, "operation_id": operation_id, "node_id": node_id,
        "source_model_id": source_model_id, "source_model_sha256": source_model_sha256,
        "tokenizer_sha256": tokenizer_sha256, "total_layers": total_layers,
        "layer_range": [start_layer, end_layer], "conversion_proof": _public(conversion_proof),
    }
    for name in ("operation_id", "node_id", "source_model_id"):
        _identity(payload[name], name)
    for name in ("source_model_sha256", "tokenizer_sha256"):
        _hash(payload[name], name)
    total = _integer(total_layers, "total_layers", 1)
    start = _integer(start_layer, "start_layer")
    end = _integer(end_layer, "end_layer", 1)
    if not 0 <= start < end <= total or total > 100_000:
        _fail("invalid_range", "layer range is outside the source model")
    if start == 0 and end == total:
        _fail("full_model_forbidden", "distributed supply requires proper segments")
    if conversion_proof is not None and not isinstance(conversion_proof, dict):
        _fail("invalid_manifest", "conversion_proof must be an object or null")
    _check_public(payload)
    try:
        _json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise SupplyError("invalid_manifest", "conversion proof is not finite JSON") from exc
    payload["segment_mode"] = _mode(start, end, total)
    return payload


def _read_json(path: Path) -> dict:
    try:
        if not 0 < path.stat().st_size <= _MAX_HEADER:
            _fail("invalid_artifact", "artifact metadata size is invalid")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise SupplyError("invalid_artifact", "artifact metadata cannot be read") from exc
    if not isinstance(payload, dict):
        _fail("invalid_artifact", "artifact metadata must be an object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _inventory(root: Path, names: dict[str, str]) -> list[dict]:
    return [{"path": name, "size_bytes": _safe_path(root, name).stat().st_size,
             "sha256": sha256_file(_safe_path(root, name)), "kind": kind}
            for name, kind in sorted(names.items())]


def _attach_local(payload: dict, root: Path) -> dict:
    return {**payload, "local_files": {
        item["path"]: str(_safe_path(root, item["path"]).absolute())
        for item in payload["files"]
    }}


def _verify_files(root: Path, payload: dict, *, cancelled=None) -> None:
    for item in payload["files"]:
        path = _safe_path(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            _fail("file_size_mismatch", "segment file is absent or has the wrong size")
        if _hash_file(path, cancelled) != item["sha256"]:
            _fail("file_digest_mismatch", "segment file digest mismatch")
    expected = {item["path"] for item in payload["files"]} | {_MARKER}
    directories = {parent.as_posix() for name in expected
                   for parent in Path(name).parents if parent != Path(".")}
    for path in root.rglob("*"):
        if _link(path):
            _fail("unsafe_path", "segment directory contains a filesystem link")
        relative = path.relative_to(root).as_posix()
        if ((path.is_file() and relative not in expected)
                or (path.is_dir() and relative not in directories)
                or not (path.is_file() or path.is_dir())):
            _fail("unexpected_file", "segment directory contains an unlisted member")


def _storage_root(value: str | Path) -> Path:
    path = Path(value).absolute()
    for ancestor in (path, *path.parents):
        if _link(ancestor):
            _fail("unsafe_path", "destination contains a filesystem link")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _ensure_space(root: Path, required: int) -> None:
    if shutil.disk_usage(root).free < required:
        _fail("insufficient_disk", "insufficient disk space for segment staging")


def _generation(root: Path, payload: dict) -> Path:
    return root / ("segment-" + _digest(payload))


def _reuse(final: Path, identity: dict, *, cancelled=None) -> dict | None:
    if not final.exists():
        return None
    if _link(final):
        _fail("unsafe_path", "segment generation is a filesystem link")
    payload = validate_supply_manifest(_read_json(final / _MARKER))
    if any(payload.get(key) != value for key, value in identity.items()):
        _fail("identity_mismatch", "existing segment belongs to another assignment")
    _verify_files(final, payload, cancelled=cancelled)
    _verify_segment(final, payload)
    return _attach_local(payload, final)


# Cache hints are bounded and private. Names never establish identity.
_CACHE_SCAN_LIMIT = 512
_CACHE_GENERATION_LIMIT = 128


def _segment_candidates(root: Path, cancelled=lambda: False):
    root = Path(root).absolute()
    if not root.is_dir() or any(_link(p) for p in (root, *root.parents)):
        return
    pending = [(root, 0)]
    examined = generations = 0
    while pending and examined < _CACHE_SCAN_LIMIT:
        directory, depth = pending.pop(0)
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    _cancelled(cancelled)
                    examined += 1
                    if examined > _CACHE_SCAN_LIMIT:
                        break
                    path = Path(entry.path)
                    if _link(path) or not entry.is_dir(follow_symlinks=False):
                        continue
                    if re.fullmatch(r"segment-[0-9a-f]{64}", entry.name):
                        generations += 1
                        yield path
                        if generations >= _CACHE_GENERATION_LIMIT:
                            return
                    elif depth < 2 and _ID.fullmatch(entry.name) and not entry.name.startswith("."):
                        # Supports receiver/local-cut and operation/node layouts.
                        pending.append((path, depth + 1))
        except OSError:
            continue


def _candidate_layout(root: Path, directory: Path, manifest: dict) -> bool:
    parts = directory.relative_to(Path(root).resolve()).parts
    return (len(parts) == 1 or
            (len(parts) == 2 and re.fullmatch(r"local-[0-9a-f]{32}", parts[0]) is not None) or
            (len(parts) == 3 and parts[:2] == (manifest["operation_id"], manifest["node_id"])))


def _same_segment(left: dict, right: dict, *, same_node=True) -> bool:
    keys = ("node_id", "source_model_id", "source_model_sha256", "tokenizer_sha256",
            "total_layers", "layer_range", "segment_mode", "format", "conversion_proof")
    return all(left.get(key) == right.get(key) for key in keys if same_node or key != "node_id")


def _link_verified(source: Path, target: Path, entry: dict, cancelled) -> bool:
    if any(_link(p) for p in (source, *source.parents)) or not source.is_file():
        return False
    before = source.stat()
    if before.st_size != entry["size_bytes"] or _hash_file(source, cancelled) != entry["sha256"]:
        return False
    _cancelled(cancelled)
    target.parent.mkdir(parents=True, exist_ok=True)
    _safe_path(target.parent, target.name)
    try:
        os.link(source, target)
    except OSError:
        # Never mutate a published source; copy if hardlinks are unsupported.
        shutil.copyfile(source, target)
    after = source.stat()
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (stamp(before) != stamp(after) or _hash_file(target, cancelled) != entry["sha256"]
            or any(_link(p) for p in (source, *source.parents))):
        target.unlink(missing_ok=True)
        return False
    return True


def _rebind_cached_segment(root: Path, final: Path, identity: dict, reuse_root: Path,
                           cancelled=lambda: False, support_files=None,
                           gguf_metadata=None, gguf_source_metadata=None) -> dict | None:
    for candidate in _segment_candidates(reuse_root, cancelled):
        if candidate == final:
            continue
        stage = None
        try:
            marker = candidate / _MARKER
            if _link(marker):
                continue
            old = validate_supply_manifest(_read_json(marker))
            if not _candidate_layout(reuse_root, candidate, old) or not _same_segment(old, identity):
                continue
            if support_files is not None:
                support = {e["path"]: e for e in old["files"] if e["kind"] == "support"}
                support.pop("model.safetensors.index.json", None)
                if set(support) != set(support_files) or any(
                        path.stat().st_size != support[name]["size_bytes"]
                        or _hash_file(path, cancelled) != support[name]["sha256"]
                        for name, path in support_files.items()):
                    continue
            _verify_files(candidate, old, cancelled=cancelled)
            upgraded_native = None
            if gguf_metadata is not None:
                weight = next(e for e in old["files"] if e["kind"] == "weights")
                native_name = Path(weight["path"]).with_suffix(".manifest.json").as_posix()
                native = _read_json(_safe_path(candidate, native_name))
                raw, _ = _gguf_table(_safe_path(candidate, weight["path"]))
                declaration = _native_gguf_metadata(
                    raw, native, source_expected=gguf_metadata,
                    source_metadata=gguf_source_metadata, upgrade=True,
                )
                if native.get("gguf_metadata") != declaration:
                    upgraded_native = {**native, "gguf_metadata": declaration}
                else:
                    # A complete declaration needs no migration. Preserve its
                    # immutable file identity and count all live reused bytes.
                    _verify_segment(candidate, old)
            else:
                _verify_segment(candidate, old)
            payload = dict(old)
            payload["operation_id"] = identity["operation_id"]
            payload.pop("manifest_sha256")
            stage = Path(tempfile.mkdtemp(prefix=".supply-build-", dir=root))
            for entry in old["files"]:
                if upgraded_native is not None and entry["path"] == native_name:
                    # Never modify a hard-linked native manifest in the cache.
                    _write_json(_safe_path(stage, native_name), upgraded_native)
                    continue
                if not _link_verified(_safe_path(candidate, entry["path"]),
                                      _safe_path(stage, entry["path"]), entry, cancelled):
                    _fail("file_digest_mismatch", "cache changed while rebinding")
            result = _finish(stage, final, payload, {e["path"]: e["kind"] for e in old["files"]})
            result["_reused_bytes"] = sum(e["size_bytes"] for e in old["files"]
                                          if upgraded_native is None or e["path"] != native_name)
            return result
        except (OSError, SupplyError) as exc:
            if isinstance(exc, SupplyError) and exc.code == "cancelled":
                raise
            if (gguf_metadata is not None and isinstance(exc, SupplyError)
                    and exc.code == "gguf_metadata_mismatch"):
                raise
            # A corrupt hint is never a receipt. Build from verified source.
            continue
        finally:
            if stage is not None:
                _remove_stage(stage, root)
    return None


def segment_cache_hint(root: Path, identity: dict) -> bool:
    """Metadata hint only: the builder must still verify all live bytes."""
    for directory in _segment_candidates(root):
        try:
            marker = directory / _MARKER
            if _link(marker):
                continue
            payload = validate_supply_manifest(_read_json(marker))
            if _candidate_layout(root, directory, payload) and _same_segment(payload, identity):
                return True
        except (SupplyError, OSError):
            continue
    return False


def reuse_segment(manifest: dict, destination_root: Path, *, cancelled=lambda: False) -> dict | None:
    """Rebind an exact verified local inventory to a fresh operation receipt."""
    payload = validate_supply_manifest(manifest)
    root = _storage_root(destination_root)
    final = root / ("segment-" + payload["manifest_sha256"])
    existing = _reuse(final, payload, cancelled=cancelled)
    if existing:
        return _received_result(payload, final, reused=True,
                                reused_bytes=sum(e["size_bytes"] for e in payload["files"]))
    for candidate in _segment_candidates(root, cancelled):
        stage = None
        try:
            marker = candidate / _MARKER
            if _link(marker):
                continue
            old = validate_supply_manifest(_read_json(marker))
            if (not _candidate_layout(root, candidate, old)
                    or not _same_segment(old, payload, same_node=False)
                    or old["files"] != payload["files"]):
                continue
            _verify_files(candidate, old, cancelled=cancelled)
            _verify_segment(candidate, old)
            stage = Path(tempfile.mkdtemp(prefix=".supply-build-", dir=root))
            for entry in payload["files"]:
                if not _link_verified(_safe_path(candidate, entry["path"]),
                                      _safe_path(stage, entry["path"]), entry, cancelled):
                    _fail("file_digest_mismatch", "cache changed while rebinding")
            _verify_segment(stage, payload)
            _write_json(stage / _MARKER, payload)
            _cancelled(cancelled)
            os.replace(stage, final)
            return _received_result(payload, final, reused=True,
                                    reused_bytes=sum(e["size_bytes"] for e in payload["files"]))
        except (SupplyError, OSError) as exc:
            if isinstance(exc, SupplyError) and exc.code == "cancelled":
                raise
        finally:
            if stage is not None:
                _remove_stage(stage, root)
    return None


def _collect_covered_partials(root: Path, protected, *, max_remove, max_hash_bytes, cancelled):
    """Retire checkpoints only under the receiver's exclusive execution owner.

    A validated, live complete copy must cover every byte in the old staging
    inventory. Incomplete/unowned data stays resumable. Directory age is never
    ownership evidence, and partial files are never hardlinked.
    """
    report = {"removed": 0, "partial_removed": 0, "reclaimed_bytes": 0,
              "hashed_bytes": 0, "examined": 0, "skipped": {}}
    def skip(reason):
        report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
    def pinned(directory):
        return any(path == directory or path.is_relative_to(directory)
                   or directory.is_relative_to(path) for path in protected)
    def stamp(path):
        if any(_link(p) for p in (path, *path.parents)):
            _fail("unsafe_path", "retirement member is a filesystem link")
        value = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode):
            _fail("unsafe_path", "retirement member is not a regular file")
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
                value.st_ctime_ns, value.st_nlink)
    def inventory(stage, manifest):
        allowed = {_RESUME_MARKER}
        for entry in manifest["files"]:
            allowed.update((entry["path"], entry["path"] + ".part"))
        directories = {parent.as_posix() for name in allowed
                       for parent in Path(name).parents if parent != Path(".")}
        result = {}
        pending = [stage]
        examined = 0
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    _cancelled(cancelled)
                    examined += 1
                    if examined > 4096:
                        _fail("invalid_artifact", "partial member bound exceeded")
                    path = Path(entry.path)
                    relative = path.relative_to(stage).as_posix()
                    if _link(path):
                        _fail("unsafe_path", "partial contains a filesystem link")
                    if entry.is_dir(follow_symlinks=False) and relative in directories:
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False) and relative in allowed:
                        result[relative] = stamp(path)
                    else:
                        _fail("unexpected_file", "partial contains an unowned member")
        return result
    candidates = list(_segment_candidates(root, cancelled))
    verified = {}
    with os.scandir(root) as entries:
        for ordinal, entry in enumerate(entries):
            _cancelled(cancelled)
            if ordinal >= _CACHE_SCAN_LIMIT or report["removed"] >= max_remove:
                break
            if not re.fullmatch(r"\.supply-receive-[0-9a-f]{64}", entry.name):
                continue
            stage = Path(entry.path)
            report["examined"] += 1
            if pinned(stage):
                skip("partial_referenced")
                continue
            try:
                if _link(stage) or not entry.is_dir(follow_symlinks=False) or stage.resolve().parent != root:
                    _fail("unsafe_path", "partial staging is outside storage")
                marker = stage / _RESUME_MARKER
                if not marker.is_file() or marker.stat().st_size > 4 * 1024**2:
                    _fail("invalid_artifact", "partial checkpoint is unavailable")
                journal = _read_resume_journal(stage)
                old = journal["manifest"]
                before = inventory(stage, old)
                records = {record["path"]: record for record in journal["prefixes"]}
                if any(record["path"] + ".part" not in before for record in records.values()):
                    _fail("invalid_artifact", "recorded partial is absent")
                sizes = 0
                for member in old["files"]:
                    name = member["path"]
                    if name in before and name + ".part" in before:
                        _fail("invalid_artifact", "partial has ambiguous completed members")
                    if name + ".part" in before:
                        length = before[name + ".part"][2]
                        if (name not in records or not records[name]["prefix_bytes"] <= length <= member["size_bytes"]
                                or before[name + ".part"][5] != 1):
                            _fail("invalid_artifact", "partial prefix has no bounded owner record")
                        sizes += 2 * length
                    elif name in before:
                        if before[name][2] != member["size_bytes"]:
                            _fail("file_size_mismatch", "completed staging member has wrong size")
                        sizes += member["size_bytes"]
                witness = None
                witness_stamp = None
                for directory in candidates:
                    if any(_link(path) for path in (directory, directory / _MARKER)):
                        continue
                    complete = validate_supply_manifest(_read_json(directory / _MARKER))
                    if (not _candidate_layout(root, directory, complete)
                            or not _same_segment(old, complete) or old["files"] != complete["files"]):
                        continue
                    key = str(directory)
                    snapshots = {e["path"]: stamp(_safe_path(directory, e["path"])) for e in complete["files"]}
                    snapshots[_MARKER] = stamp(directory / _MARKER)
                    cost = 0 if verified.get(key) == snapshots else sum(e["size_bytes"] for e in complete["files"])
                    if report["hashed_bytes"] + cost + sizes > max_hash_bytes:
                        continue
                    if cost:
                        report["hashed_bytes"] += cost
                        _verify_files(directory, complete, cancelled=cancelled)
                        _verify_segment(directory, complete)
                        if snapshots != {name: stamp(directory / name) for name in snapshots}:
                            _fail("file_digest_mismatch", "complete witness changed during verification")
                        verified[key] = snapshots
                    witness, witness_stamp = directory, snapshots
                    break
                if witness is None:
                    skip("partial_uncovered_or_budget")
                    continue
                report["hashed_bytes"] += sizes
                for member in old["files"]:
                    name = member["path"]
                    if name in before:
                        if _hash_file(_safe_path(stage, name), cancelled) != member["sha256"]:
                            _fail("file_digest_mismatch", "completed partial member changed")
                    elif name + ".part" in before:
                        record = records[name]
                        remaining = before[name + ".part"][2]
                        prefix_remaining = record["prefix_bytes"]
                        digest = hashlib.sha256()
                        with _safe_partial(stage, name).open("rb") as partial, _safe_path(witness, name).open("rb") as complete:
                            while remaining:
                                _cancelled(cancelled)
                                data = partial.read(min(_CHUNK, remaining))
                                if not data or complete.read(len(data)) != data:
                                    _fail("file_digest_mismatch", "partial bytes differ from completed copy")
                                digest.update(data[:prefix_remaining])
                                prefix_remaining = max(0, prefix_remaining - len(data))
                                remaining -= len(data)
                        if digest.hexdigest() != record["prefix_sha256"]:
                            _fail("file_digest_mismatch", "retired checkpoint digest changed")
                _cancelled(cancelled)
                if (pinned(stage) or any(_link(path) for path in (stage, *stage.parents))
                        or before != inventory(stage, old) or journal != _read_resume_journal(stage)
                        or witness_stamp != {name: stamp(witness / name) for name in witness_stamp}):
                    _fail("file_digest_mismatch", "retirement proof changed before deletion")
                freed = sum(value[2] for value in before.values() if value[5] == 1)
                # The checked absolute target is a direct child of root. Both
                # receive and prefix-donor access are excluded by the caller.
                if stage.resolve().parent != root:
                    _fail("unsafe_path", "retirement target changed")
                _cancelled(cancelled)
                shutil.rmtree(stage)
                report["removed"] += 1
                report["partial_removed"] += 1
                report["reclaimed_bytes"] += freed
            except (OSError, SupplyError) as exc:
                if isinstance(exc, SupplyError) and exc.code == "cancelled":
                    raise
                skip(exc.code if isinstance(exc, SupplyError) else "artifact_io_error")
    return report


def collect_supply_segments(root: Path, *, protected_paths=(), native_paths=None,
                            keep=1, max_remove=16, max_hash_bytes=16 * 1024**3,
                            cancelled=lambda: False, receive_idle=False) -> dict:
    """Collect only verified generations while the caller holds owner admission.

    None is an unavailable native ownership proof, rather than an empty proof.
    All deletion targets are checked within root, with exact manifest members,
    live SHA and no reparse at any level. Work is bounded in directories/bytes.
    """
    report = {"removed": 0, "reclaimed_bytes": 0, "skipped": {}, "examined": 0}
    def skip(reason):
        report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
    if native_paths is None:
        skip("native_references_unknown")
        return report
    if type(keep) is not int or keep < 0 or type(max_remove) is not int or not 0 <= max_remove <= 32:
        _fail("invalid_retention", "invalid segment retention bound")
    root = Path(root).absolute()
    if any(_link(path) for path in (root, *root.parents)):
        skip("unsafe_path")
        return report
    root = root.resolve()
    protected = [Path(path).absolute().resolve() for path in (*protected_paths, *native_paths)]
    def pinned(directory):
        return any(path == directory or path.is_relative_to(directory)
                   or directory.is_relative_to(path) for path in protected)
    if receive_idle:
        partials = _collect_covered_partials(root, protected, max_remove=max_remove,
            max_hash_bytes=max_hash_bytes, cancelled=cancelled)
        report.update({key: partials[key] for key in ("removed", "reclaimed_bytes", "examined", "partial_removed")})
        for reason, count in partials["skipped"].items():
            report["skipped"][reason] = report["skipped"].get(reason, 0) + count
        hashed = partials["hashed_bytes"]
    else:
        hashed = 0
    candidates = sorted(_segment_candidates(root, cancelled),
                        key=lambda path: path.stat().st_mtime_ns, reverse=True)
    retained = 0
    for directory in candidates:
        _cancelled(cancelled)
        report["examined"] += 1
        if pinned(directory):
            skip("referenced")
            continue
        if retained < keep:
            retained += 1
            continue
        if report["removed"] >= max_remove:
            skip("removal_budget")
            break
        try:
            # Resolve before entering or removing any candidate, never rmtree a
            # computed destination that has not been checked under the root.
            if any(_link(path) for path in (directory, *directory.parents)):
                skip("unsafe_path")
                continue
            resolved = directory.resolve()
            if not resolved.is_relative_to(root) or resolved == root:
                skip("unsafe_path")
                continue
            marker = directory / _MARKER
            if _link(marker):
                skip("unsafe_path")
                continue
            manifest = validate_supply_manifest(_read_json(marker))
            if not _candidate_layout(root, directory, manifest):
                skip("unowned_layout")
                continue
            if len(manifest["files"]) > 1024:
                skip("member_budget")
                continue
            size = sum(entry["size_bytes"] for entry in manifest["files"])
            if hashed + size > max_hash_bytes:
                skip("hash_budget")
                break
            hashed += size
            _verify_files(directory, manifest, cancelled=cancelled)
            _verify_segment(directory, manifest)
            _cancelled(cancelled)
            # Recheck all members after hashing. Owner locks prevent another
            # legitimate prepare/receive from acquiring new references here.
            if pinned(resolved) or directory.resolve() != resolved:
                skip("referenced")
                continue
            if any(_link(path) for path in (directory, *directory.parents)) or any(
                    _link(path) for path in directory.rglob("*")):
                skip("unsafe_path")
                continue
            # Hardlinks share storage; removing an extra receipt does not free
            # its linked payload yet. Report only files losing their last link.
            freed = sum(path.stat().st_size for path in directory.rglob("*")
                        if path.is_file() and path.stat().st_nlink == 1)
            _cancelled(cancelled)
            shutil.rmtree(directory)
            report["removed"] += 1
            report["reclaimed_bytes"] += freed
        except (OSError, SupplyError) as exc:
            if isinstance(exc, SupplyError) and exc.code == "cancelled":
                raise
            skip(exc.code if isinstance(exc, SupplyError) else "artifact_io_error")
    return report


def _finish(stage: Path, final: Path, payload: dict, names: dict[str, str]) -> dict:
    for name in names:
        with _safe_path(stage, name).open("r+b") as handle:
            os.fsync(handle.fileno())
    payload["files"] = _inventory(stage, names)
    payload["artifact_sha256"] = (
        next(item["sha256"] for item in payload["files"] if item["kind"] == "weights")
        if payload["format"] == "gguf" else _weight_digest(payload["files"])
    )
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = _digest(payload)
    payload = validate_supply_manifest(payload)
    _verify_segment(stage, payload)
    _write_json(stage / _MARKER, payload)
    try:
        os.replace(stage, final)
    except OSError:
        existing = _reuse(final, payload)
        if existing is None:
            raise
        return existing
    return _attach_local(payload, final)


def _remove_stage(stage: Path, root: Path) -> None:
    if stage.exists():
        resolved = stage.resolve()
        if resolved.parent != root.resolve() or not stage.name.startswith(".supply-build-"):
            _fail("unsafe_path", "refusing to remove an unrelated staging directory")
        if _link(stage):
            _fail("unsafe_path", "refusing to remove a staging link")
        shutil.rmtree(stage)


def _gguf_table(path: Path) -> tuple[dict, dict[str, tuple[int, int, tuple, int]]]:
    """Read bounded metadata and tensor ranges, never tensor payloads."""
    from gguf_metadata_contract import read_gguf_header
    try:
        return read_gguf_header(path)
    except (ValueError, struct.error, UnicodeError) as exc:
        raise SupplyError("invalid_gguf", str(exc)) from exc


def _require_gguf_metadata(actual: dict, expected: dict) -> None:
    problems = verify_gguf_metadata(actual, expected)
    if problems:
        code = "gguf_metadata_missing" if any("missing" in item for item in problems) else "gguf_metadata_mismatch"
        _fail(code, "GGUF metadata: " + "; ".join(problems))


def _native_gguf_metadata(metadata: dict, native: dict, *, source_expected=None,
                          source_metadata=None, upgrade=False) -> dict:
    """Verify declarations before using weights; only a known source can fill gaps."""
    actual = normalize_gguf_metadata(metadata)
    if source_expected is not None:
        _require_gguf_metadata(actual, source_expected)
    declared = native.get("gguf_metadata", {})
    if not isinstance(declared, dict):
        _fail("gguf_metadata_missing", "native gguf_metadata must declare block_count, head_count_kv, key_length, full_attention_interval")
    if upgrade:
        if source_expected is None:
            _fail("gguf_metadata_missing", "legacy GGUF metadata requires an authoritative source")
        declared = {**source_expected, **declared}
    _require_gguf_metadata(actual, declared)
    if "kept_block_count" in native:
        _require_gguf_metadata(actual, {**declared, "block_count": native["kept_block_count"]})
    # Existing top-level block_count describes the SOURCE. Never compare it
    # against the crop's kept count. Other legacy declarations still bind.
    legacy = dict(declared)
    for key in ("architecture", "head_count_kv", "key_length", "full_attention_interval"):
        if key in native:
            legacy[key] = native[key]
    _require_gguf_metadata(actual, legacy)
    if source_metadata is not None and "block_count" in native:
        if (type(native["block_count"]) is not int
                or native["block_count"] != source_metadata["block_count"]):
            _fail("gguf_metadata_mismatch", "GGUF block_count: native source declaration differs from source")
    if "block_count" in native:
        source_nextn = native.get("nextn_predict_layers", 0)
        source_layers = native.get("n_layer")
        if (type(source_nextn) is not int or type(source_layers) is not int
                or type(native["block_count"]) is not int
                or native["block_count"] != source_layers + source_nextn):
            _fail("gguf_metadata_mismatch", "GGUF block_count: native source block_count differs from n_layer + nextn_predict_layers")
    return declared


def _gguf_layers(metadata: dict) -> tuple[str, int, int]:
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        _fail("invalid_gguf", "GGUF architecture is missing")
    blocks = _integer(metadata.get(f"{arch}.block_count"), "GGUF block_count", 1)
    nextn = _integer(metadata.get(f"{arch}.nextn_predict_layers", 0), "GGUF MTP count")
    hidden = _integer(metadata.get(f"{arch}.embedding_length"), "GGUF hidden_size", 1)
    if nextn >= blocks:
        _fail("invalid_gguf", "GGUF has no decoder layers")
    return arch, blocks - nextn, hidden


def _layer_sizes(tensors: dict, total: int, start: int = 0, end: int | None = None,
                 *, local: bool = False) -> list[int]:
    end = total if end is None else end
    result = [0] * total
    for name, entry in tensors.items():
        if not name.startswith("blk."):
            continue
        match = re.match(r"blk\.(\d+)\..+", name)
        if not match:
            _fail("invalid_gguf", "GGUF has a malformed layer tensor name")
        index = int(match[1]) + (start if local else 0)
        if start <= index < end:
            result[index] += entry[1]
        elif local:
            _fail("layer_range_mismatch", "segment contains an unassigned layer")
    if any(result[index] <= 0 for index in range(start, end)):
        _fail("layer_range_mismatch", "segment is missing an assigned layer")
    return result


def _range_hash(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        while length:
            chunk = handle.read(min(_CHUNK, length))
            if not chunk:
                _fail("invalid_artifact", "tensor payload is truncated")
            length -= len(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _run_cut(source: Path, artifact: Path, native: Path, payload: dict, nextn: int, *, timeout_seconds=3600) -> None:
    args = ["--src", str(source), "--dst", str(artifact), "--manifest", str(native),
            "--source-model-id", payload["source_model_id"],
            "--tokenizer-sha256", payload["tokenizer_sha256"]]
    start, end = payload["layer_range"]
    if start == 0:
        args += ["--keep-head", str(end)]
    else:
        args += ["--k", str(start)]
        if end < payload["total_layers"] or nextn:
            args += ["--end", str(end)]
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--qlh-cut-layers", *args]
    else:
        script = Path(__file__).resolve().parents[1] / "scripts" / "cut_layers.py"
        if not script.is_file():
            _fail("cut_tool_unavailable", "GGUF cutting tool is unavailable")
        command = [sys.executable, str(script), *args]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout_seconds,
                                encoding="utf-8", errors="replace",
                                creationflags=(subprocess.CREATE_NO_WINDOW
                                               if sys.platform == "win32" else 0))
    except subprocess.TimeoutExpired as exc:
        raise SupplyError("cut_timeout", "GGUF cutting timed out") from exc
    except OSError as exc:
        raise SupplyError("cut_tool_unavailable", "GGUF cutting process could not start") from exc
    if result.returncode:
        code = "cut_tool_unavailable" if "No module named" in result.stderr or "ImportError" in result.stderr else "cut_failed"
        _fail(code, f"GGUF cutting failed (exit {result.returncode})")



def _selected_gguf_weight_bytes(tensors: dict, start_layer: int, end_layer: int) -> int:
    return sum(entry[1] for name, entry in tensors.items()
               if not name.startswith("blk.")
               or start_layer <= int(name.split(".")[1]) < end_layer)


def gguf_segment_disk_weight_bytes(source: Path, start_layer: int, end_layer: int) -> int:
    """Header-only weight budget using the builder's exact selection rule.

    MTP blk entries outside the decoder range are excluded even when a runtime
    descriptor groups their bytes into its 'other' component.
    """
    metadata, tensors = _gguf_table(Path(source))
    _, total_layers, _ = _gguf_layers(metadata)
    if not 0 <= start_layer < end_layer <= total_layers or end_layer - start_layer == total_layers:
        _fail("invalid_range", "distributed disk budget requires proper decoder segments")
    return _selected_gguf_weight_bytes(tensors, start_layer, end_layer)


def build_gguf_segment(source: Path, destination_dir: Path, *, source_model_id,
                       source_model_sha256, tokenizer_sha256, start_layer,
                       end_layer, total_layers, operation_id, node_id,
                       conversion_proof: dict | None = None, reuse_root: Path | None = None,
                       cut_timeout_seconds: int | None = None) -> dict:
    payload = _base_manifest(
        source_model_id=source_model_id, source_model_sha256=source_model_sha256,
        tokenizer_sha256=tokenizer_sha256, start_layer=start_layer, end_layer=end_layer,
        total_layers=total_layers, operation_id=operation_id, node_id=node_id,
        conversion_proof=conversion_proof,
    )
    payload["format"] = "gguf"
    source = Path(source).resolve()
    if not source.is_file():
        _fail("source_missing", "source GGUF is missing")
    if sha256_file(source) != source_model_sha256:
        _fail("source_digest_mismatch", "source GGUF digest mismatch")
    from relay_segment_info import read_gguf_tokenizer_sha256
    try:
        metadata, tensors = _gguf_table(source)
        arch, actual_total, hidden = _gguf_layers(metadata)
    except (OSError, UnicodeError, struct.error) as exc:
        raise SupplyError("invalid_gguf", "source GGUF header is invalid") from exc
    if actual_total != total_layers:
        _fail("layer_range_mismatch", "source GGUF layer count differs from assignment")
    if read_gguf_tokenizer_sha256(source) != tokenizer_sha256:
        _fail("tokenizer_digest_mismatch", "source GGUF tokenizer fingerprint mismatch")
    layer_bytes = _layer_sizes(tensors, total_layers, start_layer, end_layer)
    try:
        source_metadata = normalize_gguf_metadata(metadata)
        expected_metadata = project_gguf_metadata(source_metadata, start_layer, end_layer)
        _require_gguf_metadata(source_metadata, source_metadata)
    except ValueError as exc:
        if isinstance(exc, SupplyError):
            raise
        raise SupplyError("gguf_metadata_mismatch", str(exc)) from exc
    root = _storage_root(destination_dir)
    final = _generation(root, payload)
    try:
        reused = _reuse(final, payload)
    except SupplyError as exc:
        if exc.code != "gguf_metadata_missing":
            raise
        # Keep the old ready generation intact. Publish a separate contract
        # generation so a legacy receipt cannot permanently block rebuilding.
        final = _generation(root, {**payload, "_gguf_metadata_contract": 1})
        reused = _reuse(final, payload)
    if reused:
        raw, _ = _gguf_table(Path(reused["local_files"]["segment.gguf"]))
        _require_gguf_metadata(normalize_gguf_metadata(raw), expected_metadata)
        return reused
    rebound = _rebind_cached_segment(root, final, payload, reuse_root or root,
                                    gguf_metadata=expected_metadata,
                                    gguf_source_metadata=source_metadata)
    if rebound:
        return rebound
    selected_size = _selected_gguf_weight_bytes(tensors, start_layer, end_layer)
    _ensure_space(root, selected_size + len(tensors) * 32 + _MAX_HEADER)
    stage = Path(tempfile.mkdtemp(prefix=".supply-build-", dir=root))
    try:
        artifact = stage / "segment.gguf"
        native = stage / "segment.manifest.json"
        nextn = metadata.get(f"{arch}.nextn_predict_layers", 0)
        if cut_timeout_seconds is None:
            _run_cut(source, artifact, native, payload, nextn)
        else:
            if type(cut_timeout_seconds) is not int or not 1 <= cut_timeout_seconds <= 3600:
                _fail("cut_timeout", "local cutter budget is exhausted")
            _run_cut(source, artifact, native, payload, nextn, timeout_seconds=cut_timeout_seconds)
        generated = _read_json(native)
        if (generated.get("source_layer_range") != payload["layer_range"]
                or generated.get("n_layer") != total_layers
                or generated.get("source_model_sha256") != source_model_sha256
                or generated.get("tokenizer_sha256") != tokenizer_sha256
                or generated.get("source_model_id") != source_model_id
                or generated.get("hidden_size") != hidden
                or generated.get("kept_block_count") != end_layer - start_layer
                or generated.get("artifact_sha256") != sha256_file(artifact)):
            _fail("cut_manifest_mismatch", "cut manifest differs from actual assignment")
        # --end excludes source MTP blocks at a decoder tail. The cutter calls
        # this CLI shape "middle" even when the verified decoder range ends at
        # total_layers. Accept only that precise legacy shape; all other role
        # disagreement is an invalid cutter result.
        decoder_only_tail = (payload["segment_mode"] == "tail" and nextn > 0
                             and generated.get("mode") == "middle"
                             and end_layer == total_layers)
        if generated.get("mode") != payload["segment_mode"] and not decoder_only_tail:
            _fail("cut_manifest_mismatch", "cut manifest role differs from assignment")
        out_metadata, out_tensors = _gguf_table(artifact)
        generated["gguf_metadata"] = _native_gguf_metadata(
            out_metadata, generated, source_expected=expected_metadata,
            source_metadata=source_metadata, upgrade=True,
        )
        if _gguf_layers(out_metadata) != (arch, end_layer - start_layer, hidden):
            _fail("layer_range_mismatch", "cut GGUF header differs from assignment")
        expected = {}
        for name, entry in tensors.items():
            if name.startswith("blk."):
                match = re.match(r"blk\.(\d+)\.(.+)", name)
                if not match:
                    _fail("invalid_gguf", "source GGUF has malformed layer names")
                index = int(match[1])
                if not start_layer <= index < end_layer:
                    continue
                target = f"blk.{index - start_layer}.{match[2]}"
            else:
                target = name
            expected[target] = entry
        if set(expected) != set(out_tensors):
            _fail("layer_range_mismatch", "cut GGUF tensor set differs from source range")
        for name, entry in expected.items():
            output = out_tensors[name]
            if (entry[1:] != output[1:] or _range_hash(source, entry[0], entry[1])
                    != _range_hash(artifact, output[0], output[1])):
                _fail("tensor_digest_mismatch", "cut GGUF tensor bytes differ from source")
        if _layer_sizes(out_tensors, total_layers, start_layer, end_layer, local=True) != layer_bytes:
            _fail("layer_bytes_mismatch", "cut GGUF per-layer bytes differ from source")
        if sha256_file(source) != source_model_sha256:
            _fail("source_digest_mismatch", "source GGUF changed during segment generation")
        if decoder_only_tail:
            # The role comes from the verified source range/tensor set, rather
            # than the presence of --end. Keep the exact GGUF bytes unchanged.
            generated["cutter_mode"] = generated["mode"]
            generated["mode"] = payload["segment_mode"]
            generated["contract"] = {"skipped": "decoder-only tail excludes source MTP; verified by model supply range and tensor checks"}
        generated.pop("source", None)
        generated["artifact"] = artifact.name
        generated["conversion_proof"] = payload["conversion_proof"]
        generated["layer_weight_bytes"] = layer_bytes
        _check_public(generated)
        _write_json(native, generated)
        payload.update(layer_weight_bytes=layer_bytes, hidden_size=hidden, model_type=arch)
        return _finish(stage, final, payload, {artifact.name: "weights", native.name: "manifest"})
    except OSError as exc:
        code = "insufficient_disk" if exc.errno == 28 else "artifact_io_error"
        raise SupplyError(code, "GGUF segment publication failed") from exc
    finally:
        _remove_stage(stage, root)


def _safetensors_header(path: Path) -> tuple[dict, int]:
    from pipeline_model_descriptor import _DTYPE_BYTES
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            _fail("invalid_safetensors", "Safetensors length header is truncated")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > _MAX_HEADER or length + 8 > size:
            _fail("invalid_safetensors", "Safetensors JSON header length is invalid")
        try:
            def unique(pairs):
                result = {}
                for key, item in pairs:
                    if key in result:
                        _fail("invalid_safetensors", "Safetensors contains duplicate keys")
                    result[key] = item
                return result
            header = json.loads(handle.read(length), object_pairs_hook=unique)
        except (ValueError, UnicodeError) as exc:
            raise SupplyError("invalid_safetensors", "Safetensors JSON header is invalid") from exc
    if not isinstance(header, dict):
        _fail("invalid_safetensors", "Safetensors header must be an object")
    tensors = {}
    for key, item in header.items():
        if key == "__metadata__":
            if not isinstance(item, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in item.items()):
                _fail("invalid_safetensors", "Safetensors metadata is invalid")
            continue
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            _fail("invalid_safetensors", "Safetensors tensor entry is invalid")
        shape, offsets, dtype = item["shape"], item["data_offsets"], item["dtype"]
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            _fail("invalid_safetensors", "Safetensors tensor shape is invalid")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            _fail("invalid_safetensors", "Safetensors dtype is unsupported")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets):
            _fail("invalid_safetensors", "Safetensors byte offsets are invalid")
        lo, hi = offsets
        if hi < lo or hi > size - 8 - length or hi - lo != math.prod(shape) * _DTYPE_BYTES[dtype]:
            _fail("invalid_safetensors", "Safetensors bytes differ from declared shape")
        tensors[key] = item
    position = 0
    for item in sorted(tensors.values(), key=lambda value: tuple(value["data_offsets"])):
        lo, hi = item["data_offsets"]
        if lo != position:
            _fail("invalid_safetensors", "Safetensors payload overlaps or has gaps")
        position = hi
    if position != size - 8 - length:
        _fail("invalid_safetensors", "Safetensors has undeclared payload bytes")
    return tensors, 8 + length


def _model_digest(root: Path) -> str:
    """model_sync's aggregate recipe, without modifying source/cache files."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if (not path.is_file() or path.name in {"model.sha256", "model.sha256.meta.json"}
                or path.name.endswith(".part") or not path.name.lower().endswith(_MODEL_SUFFIXES)
                or any(part.startswith(".") for part in path.relative_to(root).parts)):
            continue
        safe = _safe_path(root, relative)
        digest.update(f"{relative}\0{safe.stat().st_size}\0{sha256_file(safe)}\n".encode("utf-8"))
    return digest.hexdigest()


def _source_safetensors(root: Path) -> tuple[dict, dict, dict]:
    from pipeline_model_descriptor import _ARCHITECTURE_LAYOUTS, _decoder_config
    config = _read_json(root / "config.json")
    model_type = str(config.get("model_type", "")).lower()
    if model_type not in _ARCHITECTURE_LAYOUTS:
        _fail("unsupported_architecture", "source model has no assignment tensor layout")
    index = root / "model.safetensors.index.json"
    headers, weight_map = {}, {}
    if index.is_file():
        weight_map = _read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            _fail("invalid_safetensors", "source index has no weight_map")
        for key, filename in weight_map.items():
            if not isinstance(key, str) or not isinstance(filename, str) or not filename.endswith(".safetensors"):
                _fail("invalid_safetensors", "index contains an invalid tensor or shard")
            _safe_path(root, filename)
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(path.name for path in root.glob("*.safetensors"))
    seen = set()
    for filename in shards:
        path = _safe_path(root, filename)
        if not path.is_file():
            _fail("source_missing", "source Safetensors shard is missing")
        headers[filename] = _safetensors_header(path)
        for key in headers[filename][0]:
            if key in seen:
                _fail("invalid_safetensors", "source shards contain duplicate tensor keys")
            seen.add(key)
            if not index.is_file():
                weight_map[key] = filename
    if not weight_map:
        _fail("invalid_safetensors", "source has no Safetensors weights")
    for key, filename in weight_map.items():
        if key not in headers[filename][0]:
            _fail("invalid_safetensors", "source index refers to a missing tensor")
    decoder = _decoder_config(config)
    _integer(decoder.get("num_hidden_layers"), "config num_hidden_layers", 1)
    _integer(decoder.get("hidden_size"), "config hidden_size", 1)
    return config, weight_map, headers


def _copy_selected(source: Path, destination: Path, header: dict, data_start: int,
                   keys: list[str]) -> None:
    offset, output = 0, {"__metadata__": {"format": "pt"}}
    for key in keys:
        item = header[key]
        count = item["data_offsets"][1] - item["data_offsets"][0]
        output[key] = {**item, "data_offsets": [offset, offset + count]}
        offset += count
    encoded = _json_bytes(output)
    encoded += b" " * (-len(encoded) % 8)
    with source.open("rb") as reader, destination.open("wb") as writer:
        writer.write(struct.pack("<Q", len(encoded)))
        writer.write(encoded)
        for key in keys:
            lo, hi = header[key]["data_offsets"]
            reader.seek(data_start + lo)
            remaining = hi - lo
            while remaining:
                chunk = reader.read(min(_CHUNK, remaining))
                if not chunk:
                    _fail("invalid_safetensors", "source tensor bytes are truncated")
                writer.write(chunk)
                remaining -= len(chunk)
        writer.flush()
        os.fsync(writer.fileno())


def _proof_source(payload: dict) -> dict | None:
    """Verify portable conversion identity, without following the old target path."""
    proof = payload.get("conversion_proof")
    if not isinstance(proof, dict) or proof.get("kind") != "qlh.cross_format_pipeline_identity.v1":
        return None
    if (set(proof) != {"schema_version", "kind", "source", "conversion", "target", "provenance_sha256"}
            or type(proof["schema_version"]) is not int or proof["schema_version"] != 1):
        _fail("conversion_proof_invalid", "conversion proof fields are invalid")
    unsigned = dict(proof)
    declared = unsigned.pop("provenance_sha256")
    if not isinstance(declared, str) or not _SHA.fullmatch(declared) or _digest(unsigned) != declared:
        _fail("conversion_proof_invalid", "conversion proof digest mismatch")
    source, target, conversion = proof["source"], proof["target"], proof["conversion"]
    if (not isinstance(source, dict) or set(source) != {"artifact_sha256", "manifest_sha256", "tokenizer_sha256"}
            or not isinstance(target, dict) or set(target) != {"path", "artifact_sha256", "tokenizer_sha256"}
            or not isinstance(conversion, dict) or set(conversion) != {
                "output_type", "converter_revision", "converter_sha256", "quantizer_revision", "quantizer_sha256"}):
        _fail("conversion_proof_invalid", "conversion proof identities are incomplete")
    for identity in (source, target, conversion):
        for name, value in identity.items():
            if name.endswith("sha256") and (not isinstance(value, str) or not _SHA.fullmatch(value)):
                _fail("conversion_proof_invalid", "conversion proof contains an invalid digest")
    for name in ("converter_revision", "quantizer_revision"):
        if not isinstance(conversion[name], str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", conversion[name]):
            _fail("conversion_proof_invalid", "conversion tool revision is invalid")
    if (not isinstance(conversion["output_type"], str)
            or not re.fullmatch(r"[A-Za-z0-9_]{2,32}", conversion["output_type"])
            or not isinstance(target["path"], str) or not target["path"]):
        _fail("conversion_proof_invalid", "conversion target or output type is invalid")
    _check_public(proof)
    if (target["artifact_sha256"] != payload["source_model_sha256"]
            or target["tokenizer_sha256"] != payload["tokenizer_sha256"]):
        _fail("conversion_identity_mismatch", "supply identity differs from the converted GGUF")
    return source


def _verify_cross_format_source(root: Path, payload: dict, config: dict, weight_map: dict) -> None:
    """Reverify full source inventory and existing HF-to-GGUF provenance."""
    from pipeline_model_descriptor import (
        PipelineModelDescriptorError, cross_format_pipeline_identity, _decoder_config,
    )
    if _proof_source(payload) is None:
        _fail("conversion_proof_required", "GGUF identity for Safetensors requires conversion proof")
    if _read_json(root / "cross_format.provenance.json") != payload["conversion_proof"]:
        _fail("conversion_proof_invalid", "supplied proof differs from source provenance")
    try:
        identity = cross_format_pipeline_identity(root)
    except (PipelineModelDescriptorError, OSError, ValueError) as exc:
        raise SupplyError("conversion_proof_invalid", "source conversion binding could not be verified") from exc
    if (identity.get("source_model_sha256") != payload["source_model_sha256"]
            or identity.get("tokenizer_sha256") != payload["tokenizer_sha256"]):
        _fail("conversion_identity_mismatch", "source provenance differs from supply identity")
    manifest = _read_json(root / "model.manifest.json")
    declared = set()
    for entry in manifest["files"]:
        relative = _relative(entry["path"])
        if relative in declared:
            _fail("conversion_proof_invalid", "full source inventory has duplicate paths")
        declared.add(relative)
        path = _safe_path(root, relative)
        if (not path.is_file() or path.stat().st_size != entry["size_bytes"]
                or sha256_file(path) != entry["sha256"]):
            _fail("source_digest_mismatch", "full source file differs from conversion proof")
    required = set(weight_map.values()) | {"config.json", "tokenizer.json"}
    if (root / "tokenizer_config.json").is_file():
        required.add("tokenizer_config.json")
    if (root / "model.safetensors.index.json").is_file():
        required.add("model.safetensors.index.json")
    if not required.issubset(declared):
        _fail("conversion_proof_invalid", "conversion proof does not cover full source artifacts")
    target = (root / payload["conversion_proof"]["target"]["path"]).resolve()
    metadata, _ = _gguf_table(target)
    decoder = _decoder_config(config)
    if _gguf_layers(metadata)[1:] != (decoder["num_hidden_layers"], decoder["hidden_size"]):
        _fail("conversion_identity_mismatch", "converted GGUF dimensions differ from Safetensors")


def build_safetensors_segment(source: Path, destination_dir: Path, *, source_model_id,
                              source_model_sha256, tokenizer_sha256, start_layer,
                              end_layer, total_layers, operation_id, node_id,
                              conversion_proof: dict | None = None, reuse_root: Path | None = None) -> dict:
    """Repack selected tensors; source is a HF directory (or its shard).

    Source SHA is model_sync's aggregate recipe, or a full GGUF target identity
    verified through conversion_proof plus the full source manifest. Native
    tokenizer SHA hashes tokenizer.json; cross-format SHA fingerprints GGUF
    metadata, with the source tokenizer separately bound in proof["source"].
    """
    payload = _base_manifest(
        source_model_id=source_model_id, source_model_sha256=source_model_sha256,
        tokenizer_sha256=tokenizer_sha256, start_layer=start_layer, end_layer=end_layer,
        total_layers=total_layers, operation_id=operation_id, node_id=node_id,
        conversion_proof=conversion_proof,
    )
    payload["format"] = "safetensors"
    root_source = Path(source).resolve()
    if root_source.is_file() and root_source.suffix == ".safetensors":
        root_source = root_source.parent
    if not root_source.is_dir():
        _fail("source_missing", "source Safetensors directory is missing")
    from pipeline_assignment_manifest import _selected_keys, _support_files
    from pipeline_model_descriptor import _ARCHITECTURE_LAYOUTS, _component_for_key, _decoder_config
    config, weight_map, headers = _source_safetensors(root_source)
    decoder = _decoder_config(config)
    actual_source = _model_digest(root_source)
    if actual_source != source_model_sha256:
        if not isinstance(conversion_proof, dict) or conversion_proof.get("kind") != "qlh.cross_format_pipeline_identity.v1":
            _fail("source_digest_mismatch", "source model aggregate digest mismatch")
        _verify_cross_format_source(root_source, payload, config, weight_map)
    proof_source = _proof_source(payload)
    expected_tokenizer = proof_source["tokenizer_sha256"] if proof_source else tokenizer_sha256
    tokenizer_path = _safe_path(root_source, "tokenizer.json")
    if not tokenizer_path.is_file() or sha256_file(tokenizer_path) != expected_tokenizer:
        _fail("tokenizer_digest_mismatch", "source tokenizer.json digest mismatch")
    if decoder["num_hidden_layers"] != total_layers:
        _fail("layer_range_mismatch", "source config layer count differs from assignment")
    model_type = str(config["model_type"]).lower()
    tied = bool(decoder.get("tie_word_embeddings", False))
    selected = _selected_keys(model_type, weight_map, start_layer, end_layer,
                              start_layer == 0, end_layer == total_layers, tied)
    layer_bytes = [0] * total_layers
    selected_bytes = 0
    for filename, keys in selected.items():
        for key in keys:
            component, index = _component_for_key(key, _ARCHITECTURE_LAYOUTS[model_type])
            lo, hi = headers[filename][0][key]["data_offsets"]
            selected_bytes += hi - lo
            if component == "layers":
                layer_bytes[index] += hi - lo
    if any(layer_bytes[index] <= 0 for index in range(start_layer, end_layer)):
        _fail("layer_range_mismatch", "source is missing an assigned layer")
    root = _storage_root(destination_dir)
    try:
        root.relative_to(root_source)
    except ValueError:
        pass
    else:
        _fail("unsafe_path", "segment destination must be outside source model tree")
    final = _generation(root, payload)
    reused = _reuse(final, payload)
    if reused:
        return reused
    support = [relative for relative in _support_files(root_source)
               if Path(relative).name not in {
                   "model.manifest.json", "cross_format.provenance.json", ".qlh-model-asset.json",
               }]
    rebound = _rebind_cached_segment(root, final, payload, reuse_root or root,
        support_files={name: _safe_path(root_source, name) for name in ["config.json", *support]})
    if rebound:
        return rebound
    support_bytes = sum(_safe_path(root_source, name).stat().st_size for name in support)
    _ensure_space(root, selected_bytes + support_bytes + _MAX_HEADER)
    stage = Path(tempfile.mkdtemp(prefix=".supply-build-", dir=root))
    try:
        names = {"config.json": "support", "model.safetensors.index.json": "support"}
        filtered, total_size = {}, 0
        for number, (filename, keys) in enumerate(sorted(selected.items()), 1):
            target = f"assignment-{number:05d}-of-{len(selected):05d}.safetensors"
            header, data_start = headers[filename]
            _copy_selected(root_source / filename, stage / target, header, data_start, keys)
            output_header, output_start = _safetensors_header(stage / target)
            for key in keys:
                original, copied = header[key], output_header[key]
                lo, hi = original["data_offsets"]
                out_lo, out_hi = copied["data_offsets"]
                if (original["dtype"] != copied["dtype"] or original["shape"] != copied["shape"]
                        or hi - lo != out_hi - out_lo
                        or _range_hash(root_source / filename, data_start + lo, hi - lo)
                        != _range_hash(stage / target, output_start + out_lo, out_hi - out_lo)):
                    _fail("tensor_digest_mismatch", "repacked tensor differs from source")
                filtered[key] = target
                total_size += hi - lo
            names[target] = "weights"
        _write_json(stage / "model.safetensors.index.json", {
            "metadata": {"total_size": total_size}, "weight_map": filtered,
        })
        # Keep full num_hidden_layers/layer_types for loading absolute indices.
        shutil.copyfile(root_source / "config.json", stage / "config.json")
        for relative in support:
            if relative in names:
                _fail("unsafe_path", "support file collides with generated assignment")
            original = _safe_path(root_source, relative)
            destination = _safe_path(stage, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
            names[relative] = "support"
        if _model_digest(root_source) != actual_source:
            _fail("source_digest_mismatch", "source model changed during segment generation")
        payload.update(
            model_type=model_type, hidden_size=decoder["hidden_size"],
            layer_weight_bytes=layer_bytes, has_embedding=start_layer == 0,
            has_lm_head=end_layer == total_layers, tie_word_embeddings=tied,
        )
        return _finish(stage, final, payload, names)
    except OSError as exc:
        code = "insufficient_disk" if exc.errno == 28 else "artifact_io_error"
        raise SupplyError(code, "Safetensors segment publication failed") from exc
    finally:
        _remove_stage(stage, root)


def _verify_segment(root: Path, payload: dict) -> None:
    """Verify actual headers/index/layer bytes, on publication, receive and reuse."""
    start, end = payload["layer_range"]
    total = payload["total_layers"]
    if payload["format"] == "gguf":
        from relay_segment_info import read_gguf_tokenizer_sha256
        weight = next(item for item in payload["files"] if item["kind"] == "weights")
        path = _safe_path(root, weight["path"])
        metadata, tensors = _gguf_table(path)
        native_name = Path(weight["path"]).with_suffix(".manifest.json").as_posix()
        native = _read_json(_safe_path(root, native_name))
        _check_public(native)
        _native_gguf_metadata(metadata, native)
        arch, actual_layers, hidden = _gguf_layers(metadata)
        if actual_layers != end - start or hidden != payload["hidden_size"] or arch != payload.get("model_type", arch):
            _fail("layer_range_mismatch", "received GGUF header differs from assignment")
        if _layer_sizes(tensors, total, start, end, local=True) != payload["layer_weight_bytes"]:
            _fail("layer_bytes_mismatch", "GGUF layer bytes differ from manifest")
        if read_gguf_tokenizer_sha256(path) != payload["tokenizer_sha256"]:
            _fail("tokenizer_digest_mismatch", "GGUF tokenizer differs from manifest")
        for field, expected in {
            "source_layer_range": payload["layer_range"], "n_layer": total,
            "source_model_sha256": payload["source_model_sha256"],
            "source_model_id": payload["source_model_id"],
            "tokenizer_sha256": payload["tokenizer_sha256"],
            "artifact_sha256": payload["artifact_sha256"], "hidden_size": payload["hidden_size"],
            "kept_block_count": end - start,
            "mode": payload["segment_mode"], "artifact": weight["path"],
            "conversion_proof": payload["conversion_proof"],
        }.items():
            if native.get(field) != expected:
                _fail("native_manifest_mismatch", "native GGUF manifest differs from supply")
        if "layer_weight_bytes" in native and native["layer_weight_bytes"] != payload["layer_weight_bytes"]:
            _fail("layer_bytes_mismatch", "native GGUF layer bytes differ from supply")
    else:
        from pipeline_assignment_manifest import _selected_keys
        from pipeline_model_descriptor import _ARCHITECTURE_LAYOUTS, _component_for_key, _decoder_config
        config, weight_map, headers = _source_safetensors(root)
        decoder = _decoder_config(config)
        model_type = str(config["model_type"]).lower()
        if decoder["num_hidden_layers"] != total or decoder["hidden_size"] != payload["hidden_size"]:
            _fail("layer_range_mismatch", "received config differs from assignment")
        if model_type != payload.get("model_type", model_type):
            _fail("identity_mismatch", "received architecture differs from manifest")
        listed = {item["path"] for item in payload["files"] if item["kind"] == "weights"}
        if set(headers) != listed:
            _fail("invalid_safetensors", "filtered index differs from weight inventory")
        tied = bool(decoder.get("tie_word_embeddings", False))
        selected = _selected_keys(model_type, weight_map, start, end, start == 0, end == total, tied)
        if {key for keys in selected.values() for key in keys} != set(weight_map):
            _fail("layer_range_mismatch", "filtered index contains unassigned tensors")
        for name, expected in {"has_embedding": start == 0, "has_lm_head": end == total,
                               "tie_word_embeddings": tied}.items():
            if name in payload and payload[name] != expected:
                _fail("identity_mismatch", "assignment role differs from model config")
        actual_bytes, total_size = [0] * total, 0
        for filename, (header, _) in headers.items():
            if set(header) != {key for key, shard in weight_map.items() if shard == filename}:
                _fail("layer_range_mismatch", "shard has tensors absent from filtered index")
            for key, item in header.items():
                component, index = _component_for_key(key, _ARCHITECTURE_LAYOUTS[model_type])
                size = item["data_offsets"][1] - item["data_offsets"][0]
                total_size += size
                if component == "layers":
                    if index is None or not start <= index < end:
                        _fail("layer_range_mismatch", "weight shard contains unassigned layers")
                    actual_bytes[index] += size
        if actual_bytes != payload["layer_weight_bytes"]:
            _fail("layer_bytes_mismatch", "Safetensors layer bytes differ from manifest")
        index = _read_json(root / "model.safetensors.index.json")
        if not isinstance(index.get("metadata"), dict) or index["metadata"].get("total_size") != total_size:
            _fail("layer_bytes_mismatch", "index total_size differs from tensors")
        tokenizer = _safe_path(root, "tokenizer.json")
        proof_source = _proof_source(payload)
        expected_tokenizer = proof_source["tokenizer_sha256"] if proof_source else payload["tokenizer_sha256"]
        if not tokenizer.is_file() or sha256_file(tokenizer) != expected_tokenizer:
            _fail("tokenizer_digest_mismatch", "received tokenizer.json differs from manifest")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _fail("http_redirect_rejected", "supply downloads must remain on the master endpoint")


def _cancelled(callback: Callable[[], bool]) -> None:
    if callback():
        _fail("cancelled", "segment reception was cancelled")


def _published_file_inventory(root: Path, cancelled: Callable[[], bool], identity: dict | None = None) -> dict:
    """Bounded published-marker hints inside one node's private receiver root.

    Node labels may change when rebinding; this root is the physical node
    boundary. Source/range/role/proof and every individual file digest still
    have to match. Changed optional support never invalidates reusable weights.
    """
    inventory = {}
    for generation in _segment_candidates(root, cancelled):
        marker = generation / _MARKER
        if _link(marker) or not marker.is_file():
            continue
        try:
            published = validate_supply_manifest(_read_json(marker))
            if not _candidate_layout(root, generation, published):
                continue
            if identity is not None and not _same_segment(published, identity, same_node=False):
                continue
            for item in published["files"]:
                _safe_path(generation, item["path"])
        except (SupplyError, OSError):
            continue
        for item in published["files"]:
            key = (item["sha256"], item["size_bytes"], item["kind"])
            inventory.setdefault(key, []).append((generation, item["path"]))
    return inventory


def _copy_published_file(inventory: dict, entry: dict, target: Path,
                         cancelled: Callable[[], bool]) -> bool:
    """Reuse live verified immutable bytes, hardlinking where supported."""
    key = (entry["sha256"], entry["size_bytes"], entry["kind"])
    for generation, relative in inventory.get(key, []):
        _cancelled(cancelled)
        try:
            if _link_verified(_safe_path(generation, relative), target, entry, cancelled):
                return True
        except (SupplyError, OSError) as exc:
            target.unlink(missing_ok=True)
            if isinstance(exc, SupplyError) and exc.code == "cancelled":
                raise
    return False


def _safe_partial(stage: Path, relative: str) -> Path:
    if _link(stage.parent):
        _fail("unsafe_path", "partial staging parent is a filesystem link")
    target = _safe_path(stage, relative)
    partial = target.with_name(target.name + ".part")
    if _link(partial):
        _fail("unsafe_path", "partial download is a filesystem link")
    if not partial.resolve().is_relative_to(stage.resolve()):
        _fail("unsafe_path", "partial download escapes staging")
    return partial


def _read_resume_journal(stage: Path) -> dict:
    marker = stage / _RESUME_MARKER
    if _link(marker):
        _fail("unsafe_path", "resume journal is a filesystem link")
    journal = _read_json(marker)
    if set(journal) != {"schema_version", "manifest", "prefixes", "journal_sha256"}:
        _fail("invalid_artifact", "resume journal fields are invalid")
    unsigned = {key: value for key, value in journal.items() if key != "journal_sha256"}
    if journal["schema_version"] != 1 or _digest(unsigned) != journal["journal_sha256"]:
        _fail("invalid_artifact", "resume journal digest mismatch")
    payload = validate_supply_manifest(journal["manifest"])
    if stage.name != ".supply-receive-" + payload["manifest_sha256"]:
        _fail("identity_mismatch", "resume journal does not match its staging identity")
    files = {entry["path"]: entry for entry in payload["files"]}
    prefixes = journal["prefixes"]
    if not isinstance(prefixes, list) or len(prefixes) > len(files):
        _fail("invalid_artifact", "resume journal prefix inventory is invalid")
    seen = set()
    for record in prefixes:
        if not isinstance(record, dict) or set(record) != {"path", "prefix_bytes", "prefix_sha256"}:
            _fail("invalid_artifact", "resume journal prefix record is invalid")
        relative = _relative(record["path"])
        if relative not in files or relative in seen:
            _fail("invalid_artifact", "resume journal prefix is outside the file inventory")
        seen.add(relative)
        length = _integer(record["prefix_bytes"], "prefix_bytes", 1)
        if length > files[relative]["size_bytes"]:
            _fail("invalid_artifact", "resume prefix exceeds its bound file size")
        _hash(record["prefix_sha256"], "prefix_sha256")
        _safe_partial(stage, relative)
    return journal


def _prefix_digest(path: Path, length: int, cancelled: Callable[[], bool]):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while length:
            _cancelled(cancelled)
            chunk = handle.read(min(_CHUNK, length))
            if not chunk:
                _fail("file_digest_mismatch", "resume prefix is truncated")
            digest.update(chunk)
            length -= len(chunk)
    return digest


class _ResumeJournal:
    """Private staging checkpoints, bound to manifest files and live prefixes.

    Records are created from the HTTP stream or a reverified existing record.
    Legacy, unrecorded .part bytes can resume only their own manifest and are
    never promoted into a cross-operation candidate before full SHA validation.
    """
    def __init__(self, stage: Path, payload: dict):
        self.stage, self.payload = stage, payload
        self.records = {}
        marker = stage / _RESUME_MARKER
        if _link(marker):
            _fail("unsafe_path", "resume journal is a filesystem link")
        if marker.exists():
            try:
                journal = _read_resume_journal(stage)
                if journal["manifest"] == payload:
                    self.records = {record["path"]: record for record in journal["prefixes"]}
            except (SupplyError, OSError):
                pass

    def checkpoint(self, entry: dict, length: int, digest: str) -> None:
        if not length:
            return
        partial = _safe_partial(self.stage, entry["path"])
        if not partial.is_file() or partial.stat().st_size < length:
            return
        marker = self.stage / _RESUME_MARKER
        if _link(marker):
            _fail("unsafe_path", "resume checkpoint path is a filesystem link")
        record = {"path": entry["path"], "prefix_bytes": length, "prefix_sha256": digest}
        if self.records.get(entry["path"]) == record and marker.is_file():
            return
        records = {**self.records, entry["path"]: record}
        journal = {"schema_version": 1, "manifest": self.payload,
                   "prefixes": [records[name] for name in sorted(records)]}
        journal["journal_sha256"] = _digest(journal)
        temporary = None
        try:
            with partial.open("r+b") as handle:
                os.fsync(handle.fileno())
            # Exclusive, unique temporary names prevent overlapping checkpoint
            # writes from truncating or moving another writer's temporary file.
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".supply-checkpoint-", suffix=".tmp",
                dir=self.stage, delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(_json_bytes(journal))
                handle.flush()
                os.fsync(handle.fileno())
            # Windows readers without FILE_SHARE_DELETE (including filesystem
            # scanners) can transiently deny replacement with WinError 5/32/33.
            # Both our handles are closed. Retry only these Windows errors,
            # preserving the previous durable journal until replace succeeds.
            for attempt in range(8):
                if _link(self.stage) or _link(marker) or _link(temporary):
                    _fail("unsafe_path", "resume checkpoint path is a filesystem link")
                try:
                    os.replace(temporary, marker)
                    temporary = None
                    self.records = records
                    break
                except OSError as exc:
                    if (sys.platform != "win32" or getattr(exc, "winerror", None) not in {5, 32, 33}
                            or attempt == 7):
                        raise SupplyError("artifact_io_error", "resume checkpoint could not be published") from exc
                    time.sleep(0.01 * 2 ** attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def trusted_prefix(self, entry: dict, partial: Path, offset: int,
                       cancelled: Callable[[], bool]):
        if not offset:
            return 0, hashlib.sha256()
        record = self.records.get(entry["path"])
        if record is None or record["prefix_bytes"] > offset:
            return offset, None
        length = record["prefix_bytes"]
        digest = _prefix_digest(partial, length, cancelled)
        if digest.hexdigest() != record["prefix_sha256"]:
            partial.unlink()
            self.records.pop(entry["path"], None)
            return 0, hashlib.sha256()
        # A crash may leave bytes beyond the last durable checkpoint. Keep
        # only the prefix that has an independent recorded digest.
        if length < offset:
            with partial.open("r+b") as handle:
                handle.truncate(length)
                handle.flush()
                os.fsync(handle.fileno())
        return length, digest

    def discard(self) -> None:
        for name in (_RESUME_MARKER, _RESUME_MARKER + ".tmp"):
            path = self.stage / name
            if _link(path):
                _fail("unsafe_path", "resume checkpoint path is a filesystem link")
            path.unlink(missing_ok=True)


def _partial_file_inventory(root: Path, cancelled: Callable[[], bool]) -> dict:
    inventory = {}
    for stage in sorted(root.iterdir()):
        _cancelled(cancelled)
        if (not re.fullmatch(r"\.supply-receive-[0-9a-f]{64}", stage.name)
                or _link(stage) or not stage.is_dir() or stage.resolve().parent != root):
            continue
        try:
            journal = _read_resume_journal(stage)
        except (SupplyError, OSError):
            continue
        files = {entry["path"]: entry for entry in journal["manifest"]["files"]}
        for record in journal["prefixes"]:
            entry = files[record["path"]]
            key = (entry["sha256"], entry["size_bytes"], entry["kind"])
            inventory.setdefault(key, []).append((stage, record))
    for candidates in inventory.values():
        candidates.sort(key=lambda candidate: candidate[1]["prefix_bytes"], reverse=True)
    return inventory


def _copy_partial_prefix(inventory: dict, entry: dict, partial: Path,
                         cancelled: Callable[[], bool]):
    """Copy a checkpointed prefix; hash copied bytes before making it resumable."""
    existing = partial.stat().st_size if partial.is_file() else 0
    key = (entry["sha256"], entry["size_bytes"], entry["kind"])
    for stage, record in inventory.get(key, []):
        _cancelled(cancelled)
        length = record["prefix_bytes"]
        if length <= existing:
            continue
        try:
            source = _safe_partial(stage, record["path"])
            if not source.is_file() or source.stat().st_size < length:
                continue
            reader = source.open("rb")
        except (SupplyError, OSError):
            continue
        temporary = None
        try:
            with reader, tempfile.NamedTemporaryFile(
                mode="wb", prefix=".supply-prefix-", suffix=".part",
                dir=partial.parent, delete=False,
            ) as writer:
                temporary = Path(writer.name)
                digest, remaining = hashlib.sha256(), length
                while remaining:
                    _cancelled(cancelled)
                    chunk = reader.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    writer.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                valid = not remaining and digest.hexdigest() == record["prefix_sha256"]
                if valid:
                    writer.flush()
                    os.fsync(writer.fileno())
            if not valid:
                continue
            _cancelled(cancelled)
            if _link(partial):
                _fail("unsafe_path", "partial destination became a filesystem link")
            os.replace(temporary, partial)
            temporary = None
            return length, digest
        except OSError as exc:
            raise SupplyError("artifact_io_error", "checkpointed supply prefix could not be copied") from exc
        finally:
            reader.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return None


def _download(opener, url: str, target: Path, entry: dict, token: str,
              cancelled: Callable[[], bool], notify: Callable[[int, int, int, int], None],
              inventory: dict, prefix_inventory: dict,
              journal: _ResumeJournal) -> tuple[int, int, int]:
    size = entry["size_bytes"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if _link(partial):
        _fail("unsafe_path", "partial download is a filesystem link")
    if target.is_file():
        if target.stat().st_size == size and _hash_file(target, cancelled) == entry["sha256"]:
            partial.unlink(missing_ok=True)
            _cancelled(cancelled)
            notify(size, 0, size, 0)
            return 0, size, 0
        target.unlink()
    if _copy_published_file(inventory, entry, target, cancelled):
        partial.unlink(missing_ok=True)
        notify(size, 0, size, 0)
        return 0, size, 0
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    offset, running_digest = journal.trusted_prefix(entry, partial, offset, cancelled)
    imported = _copy_partial_prefix(prefix_inventory, entry, partial, cancelled)
    if imported is not None:
        offset, running_digest = imported
        journal.checkpoint(entry, offset, running_digest.hexdigest())
    if size == 0 and not partial.exists():
        partial.write_bytes(b"")
    if offset == size and partial.is_file():
        if _hash_file(partial, cancelled) != entry["sha256"]:
            partial.unlink()
            _fail("file_digest_mismatch", "complete partial download digest mismatch")
        _cancelled(cancelled)
        os.replace(partial, target)
        notify(size, 0, 0, size)
        return 0, 0, size
    headers = {"X-QLH-Supply-Token": token, "Accept-Encoding": "identity",
               "Range": f"bytes={offset}-"}
    request = urllib.request.Request(url, headers=headers)
    received, checkpoint_at = offset, offset
    wrote_prefix = False
    try:
        _cancelled(cancelled)
        with opener.open(request, timeout=30) as response:
            if response.status != 206:
                _fail("http_range_mismatch", "supply endpoint must return HTTP 206")
            if response.headers.get("Content-Range") != f"bytes {offset}-{size - 1}/{size}":
                _fail("http_range_mismatch", "supply Content-Range differs from requested bytes")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                _fail("http_range_mismatch", "encoded supply responses are forbidden")
            if response.headers.get("Content-Length") != str(size - offset):
                _fail("http_size_mismatch", "supply response length differs from inventory")
            if response.geturl() != url:
                _fail("http_redirect_rejected", "supply response changed endpoint")
            notify(offset, 0, 0, offset)
            with partial.open("ab" if offset else "wb") as handle:
                received = offset
                while received < size:
                    _cancelled(cancelled)
                    chunk = response.read(min(_CHUNK, size - received))
                    if not chunk:
                        _fail("download_incomplete", "supply download ended before declared size")
                    handle.write(chunk)
                    received += len(chunk)
                    wrote_prefix = True
                    if running_digest is not None:
                        running_digest.update(chunk)
                        if received < size and received - checkpoint_at >= _RESUME_INTERVAL:
                            handle.flush()
                            journal.checkpoint(entry, received, running_digest.hexdigest())
                            checkpoint_at = received
                    notify(received, received - offset, 0, offset)
                _cancelled(cancelled)
                if response.read(1):
                    _fail("http_size_mismatch", "supply response exceeded declared size")
                handle.flush()
                os.fsync(handle.fileno())
    except urllib.error.HTTPError as exc:
        raise SupplyError("http_error", f"supply endpoint returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise SupplyError("download_failed", "supply download transport failed") from exc
    finally:
        original_error = sys.exception()
        unwinding = original_error is not None
        if (unwinding and wrote_prefix and running_digest is not None and received > checkpoint_at
                and (received < size or running_digest.hexdigest() == entry["sha256"])):
            # Save interrupted HTTP bytes after the handle closes. Successful
            # complete files immediately receive full SHA verification instead
            # of a redundant checkpoint for every tiny support file.
            try:
                journal.checkpoint(entry, received, running_digest.hexdigest())
            except (OSError, SupplyError) as checkpoint_error:
                # Keep cancellation/transport as the terminal outcome while
                # making the secondary persistence failure available to logs.
                original_error.add_note(f"resume checkpoint failed: {checkpoint_error}")
    if partial.stat().st_size != size:
        _fail("file_size_mismatch", "downloaded size differs from manifest")
    if _hash_file(partial, cancelled) != entry["sha256"]:
        partial.unlink()
        _fail("file_digest_mismatch", "downloaded digest differs from manifest")
    _cancelled(cancelled)
    os.replace(partial, target)
    return size - offset, 0, offset


def receive_segment(manifest: dict, destination_root: Path, *, base_url: str, token: str,
                    cancelled: Callable[[], bool] = lambda: False,
                    progress: Callable[[dict], None] | None = None) -> dict:
    """Download/verify/publish one segment; progress receives event dictionaries.

    downloaded_bytes measures completed bytes, including reuse and resume;
    completed_bytes is its compatibility alias. actual_downloaded_bytes counts
    only HTTP body bytes received during this call. reused_bytes counts complete
    verified local files; resumed_bytes counts pre-existing .part prefixes.
    The three independent counters sum to downloaded_bytes in progress/results.
    Each event also has path, total_bytes, file_bytes and file_size_bytes. Lookup
    stays inside published segment-* inventories under destination_root. Every
    HTTP request requires a matching Range/206.
    """
    payload = validate_supply_manifest(manifest)
    if not isinstance(base_url, str):
        _fail("invalid_base_url", "base_url must be the master's HTTP(S) origin")
    try:
        parsed = urllib.parse.urlsplit(base_url)
        parsed.port
    except ValueError as exc:
        raise SupplyError("invalid_base_url", "master origin is invalid") from exc
    if any(ord(char) <= 32 for char in base_url):
        _fail("invalid_base_url", "master origin contains whitespace")
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        _fail("invalid_base_url", "base_url must be the master's HTTP(S) origin")
    if not isinstance(token, str) or not token or any(ord(char) < 32 or ord(char) > 126 for char in token):
        _fail("invalid_token", "supply token must be a nonempty HTTP header value")
    _cancelled(cancelled)
    root = _storage_root(destination_root)
    final = root / ("segment-" + payload["manifest_sha256"])
    total_size = sum(item["size_bytes"] for item in payload["files"])
    existing = _reuse(final, payload, cancelled=cancelled)
    if existing:
        _cancelled(cancelled)
        completed = 0
        for entry in payload["files"]:
            completed += entry["size_bytes"]
            if progress:
                progress({"path": entry["path"], "downloaded_bytes": completed,
                          "actual_downloaded_bytes": 0, "reused_bytes": completed,
                          "resumed_bytes": 0, "completed_bytes": completed,
                          "total_bytes": total_size, "file_bytes": entry["size_bytes"],
                          "file_size_bytes": entry["size_bytes"]})
            _cancelled(cancelled)
        return _received_result(payload, final, reused=True,
                                actual_downloaded_bytes=0, reused_bytes=total_size)
    stage = root / (".supply-receive-" + payload["manifest_sha256"])
    if _link(stage):
        _fail("unsafe_path", "receive staging directory is a filesystem link")
    stage.mkdir(exist_ok=True)
    already = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file() and not _link(path))
    _ensure_space(root, max(0, total_size - already) + _MAX_HEADER)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    base = base_url.rstrip("/")
    prefix = (f"{base}/api/models/supply/"
              f"{urllib.parse.quote(payload['operation_id'], safe='')}/files/"
              f"{urllib.parse.quote(payload['node_id'], safe='')}/")
    journal = _ResumeJournal(stage, payload)
    inventory = _published_file_inventory(root, cancelled, payload)
    prefix_inventory = _partial_file_inventory(root, cancelled)
    completed = actual_downloaded_bytes = reused_bytes = resumed_bytes = 0
    for entry in payload["files"]:
        _cancelled(cancelled)
        target = _safe_path(stage, entry["path"])

        def notify(received, transferred, local, resumed):
            if progress:
                progress({"path": entry["path"], "downloaded_bytes": completed + received,
                          "actual_downloaded_bytes": actual_downloaded_bytes + transferred,
                          "reused_bytes": reused_bytes + local,
                          "resumed_bytes": resumed_bytes + resumed,
                          "completed_bytes": completed + received,
                          "total_bytes": total_size, "file_bytes": received,
                          "file_size_bytes": entry["size_bytes"]})
        url = prefix + urllib.parse.quote(entry["path"], safe="/")
        transferred, local, resumed = _download(
            opener, url, target, entry, token, cancelled, notify,
            inventory, prefix_inventory, journal,
        )
        actual_downloaded_bytes += transferred
        reused_bytes += local
        resumed_bytes += resumed
        completed += entry["size_bytes"]
    _cancelled(cancelled)
    journal.discard()
    _verify_files(stage, payload, cancelled=cancelled)
    _verify_segment(stage, payload)
    _cancelled(cancelled)
    try:
        _write_json(stage / _MARKER, payload)
        _cancelled(cancelled)
        os.replace(stage, final)
    except OSError as exc:
        if stage.exists():
            (stage / _MARKER).unlink(missing_ok=True)
        if _reuse(final, payload) is None:
            raise SupplyError("publication_failed", "verified segment could not be published") from exc
    except BaseException:
        if stage.exists():
            (stage / _MARKER).unlink(missing_ok=True)
        raise
    return _received_result(payload, final, reused=actual_downloaded_bytes == 0,
                            actual_downloaded_bytes=actual_downloaded_bytes,
                            reused_bytes=reused_bytes, resumed_bytes=resumed_bytes)


def _received_result(payload: dict, root: Path, *, reused: bool,
                     actual_downloaded_bytes: int = 0, reused_bytes: int = 0,
                     resumed_bytes: int = 0) -> dict:
    total_size = sum(item["size_bytes"] for item in payload["files"])
    if payload["format"] == "gguf":
        name = next(item["path"] for item in payload["files"] if item["kind"] == "weights")
        model_path = root / name
    else:
        model_path = root
    return {
        "model_path": str(model_path.absolute()), "manifest_path": str((root / _MARKER).absolute()),
        "manifest": payload, "local_files": _attach_local(payload, root)["local_files"],
        "reused": reused, "downloaded_bytes": total_size,
        "actual_downloaded_bytes": actual_downloaded_bytes,
        "reused_bytes": reused_bytes, "resumed_bytes": resumed_bytes,
        "completed_bytes": total_size, "total_bytes": total_size,
    }
