"""Tailnet model synchronization for PyTorch pipeline workers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import model_config as mc


_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_ARTIFACT_SUFFIXES = (
    ".safetensors", ".bin", ".json", ".py", ".tiktoken",
    ".model", ".txt", ".jinja", ".spm", ".vocab",
)
_HASH_META_NAME = "model.sha256.meta.json"
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def compute_file_sha256(path: str | Path) -> str:
    """Hash one file and cache only while its size and mtime stay unchanged."""
    value_path = Path(path)
    stat = value_path.stat()
    key = (str(value_path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with value_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


def compute_model_sha256(model_path: str, *, use_cache: bool = True, digest=None) -> str:
    """Hash every executable PyTorch model artifact in stable path order."""
    root = Path(model_path)
    if not root.is_dir():
        return ""

    cache_path = root / "model.sha256"
    metadata_path = root / _HASH_META_NAME
    # Supply directories have an exact member whitelist; hashing must not add files.
    cache_writable = not (root / "supply.manifest.json").exists()
    file_digest = digest or compute_file_sha256
    if digest is None and not cache_writable:
        from model_supply_assets import _hash_file_cached
        file_digest = _hash_file_cached
    artifact_files = sorted(
        path for path in root.rglob("*")
        if (
            path.is_file()
            and path.name not in {"model.sha256", _HASH_META_NAME}
            and not path.name.endswith(".part")
            and path.name.lower().endswith(_ARTIFACT_SUFFIXES)
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    )
    if not artifact_files:
        return ""

    fingerprint = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in artifact_files
    ]
    if cache_writable and use_cache and cache_path.is_file() and metadata_path.is_file():
        try:
            cached = cache_path.read_text(encoding="utf-8").strip().split()[0]
            cached_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if len(cached) == 64 and cached_meta == fingerprint:
                return cached
        except (OSError, IndexError, json.JSONDecodeError):
            pass

    model_digest = hashlib.sha256()
    for path in artifact_files:
        relative = path.relative_to(root).as_posix()
        file_sha = file_digest(path)
        model_digest.update(
            f"{relative}\0{path.stat().st_size}\0{file_sha}\n".encode("utf-8")
        )
    value = model_digest.hexdigest()
    if cache_writable:
        try:
            cache_path.write_text(f"{value}  {root.name}\n", encoding="utf-8")
            metadata_path.write_text(
                json.dumps(fingerprint, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass
    return value


def resolve_worker_model_path(model_id: str) -> str:
    """Resolve a registry model path, or a safe local external-model directory."""
    if not _MODEL_ID_RE.fullmatch(model_id or ""):
        raise ValueError(f"invalid model_id: {model_id!r}")
    config = mc.get_model_config(model_id)
    if config and config.model_path:
        return os.path.abspath(mc.resolve_model_path(config.model_path))
    return os.path.abspath(mc.resolve_model_path(os.path.join("models", model_id)))


def _read_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model manifest HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"model manifest request failed: {exc.reason}") from exc


def _safe_destination(root: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"unsafe model file path: {relative_path!r}")
    destination = (root / normalized).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"model file escapes destination: {relative_path!r}") from exc
    return destination


def _file_sha256(path: Path) -> str:
    # 同步路径必须验证当前内容；compute_file_sha256 的 stat key 会让已替换文件失效。
    return compute_file_sha256(path)


def _download_file(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if 0 < offset < expected_size else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            append = offset > 0 and getattr(response, "status", None) == 206
            with partial.open("ab" if append else "wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"model file download failed: {exc.reason}") from exc

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"download size mismatch for {destination.name}: "
            f"actual={actual_size}, expected={expected_size}"
        )
    os.replace(partial, destination)


def ensure_model_available(
    master_host: str,
    master_api_port: int,
    model_id: str,
    expected_sha256: str,
    *,
    timeout: float = 10.0,
) -> str:
    """Ensure the worker owns the master's exact active PyTorch model."""
    encoded_model_id = urllib.parse.quote(model_id, safe="")
    from network_address import build_url

    base_url = build_url("http", master_host, int(master_api_port))
    manifest = _read_json(
        f"{base_url}/api/models/downloadable?model_id={encoded_model_id}",
        timeout,
    )
    if manifest.get("model_id") != model_id:
        raise RuntimeError("master returned a different model manifest")
    manifest_sha256 = str(manifest.get("sha256", ""))
    if expected_sha256 and manifest_sha256 != expected_sha256:
        raise RuntimeError("master model manifest changed while configuring pipeline")

    destination_root = Path(resolve_worker_model_path(model_id))
    destination_root.mkdir(parents=True, exist_ok=True)
    files = manifest.get("files") or []
    if not files:
        raise RuntimeError("master model manifest contains no files")

    def _sync_entry(entry: dict[str, Any], force: bool = False) -> None:
        relative_path = str(entry.get("path", ""))
        expected_size = int(entry.get("size_bytes", -1))
        expected_file_sha = str(entry.get("sha256", ""))
        if expected_size < 0:
            raise RuntimeError(f"invalid model file size: {relative_path}")
        destination = _safe_destination(destination_root, relative_path)
        valid = destination.is_file() and destination.stat().st_size == expected_size
        if valid and expected_file_sha:
            valid = _file_sha256(destination) == expected_file_sha
        if force or not valid:
            encoded_path = urllib.parse.quote(relative_path.replace("\\", "/"), safe="/")
            _download_file(
                f"{base_url}/api/models/files/{encoded_model_id}/{encoded_path}",
                destination,
                expected_size,
            )
            if expected_file_sha and _file_sha256(destination) != expected_file_sha:
                raise RuntimeError(f"SHA256 mismatch after download: {relative_path}")

    for entry in files:
        _sync_entry(entry)

    # The worker directory is an exact mirror of the active model. Old shards
    # left by a previous revision participate in the aggregate digest and can
    # otherwise make synchronization fail forever.
    manifest_artifacts = {
        str(entry.get("path", "")).replace("\\", "/").lstrip("/")
        for entry in files
        if str(entry.get("path", "")).lower().endswith(_ARTIFACT_SUFFIXES)
    }
    for local_artifact in destination_root.rglob("*"):
        relative_path = local_artifact.relative_to(destination_root)
        if (not local_artifact.is_file()
                or local_artifact.name in {"model.sha256", _HASH_META_NAME}
                or not local_artifact.name.lower().endswith(_ARTIFACT_SUFFIXES)
                or any(part.startswith(".") for part in relative_path.parts)):
            continue
        relative = relative_path.as_posix()
        if relative not in manifest_artifacts:
            local_artifact.unlink()

    expected = expected_sha256 or manifest_sha256
    actual_sha256 = compute_model_sha256(str(destination_root), use_cache=False)
    if expected and actual_sha256 != expected:
        # A file may have changed after the first validation pass; force an exact mirror.
        for entry in files:
            if str(entry.get("path", "")).lower().endswith(_ARTIFACT_SUFFIXES):
                _sync_entry(entry, force=True)
        actual_sha256 = compute_model_sha256(str(destination_root), use_cache=False)
    if expected and actual_sha256 != expected:
        raise RuntimeError(
            f"model SHA256 mismatch after synchronization: "
            f"local={actual_sha256[:16]}..., master={expected[:16]}..."
        )
    return str(destination_root)


def _assignment_destination_root(model_id: str, config_id: str, node_id: str) -> Path:
    """Use a generation-scoped cache so partial assignments never replace a model."""
    base = Path(resolve_worker_model_path(model_id))
    safe_config = re.sub(r"[^A-Za-z0-9._-]", "_", str(config_id or "")) or "config"
    safe_node = re.sub(r"[^A-Za-z0-9._-]", "_", str(node_id or "")) or "node"
    return base / ".pipeline_assignments" / f"{safe_config}-{safe_node}"


def assignment_cache_root(model_id: str) -> Path:
    """Return the model-scoped assignment cache without creating it."""
    return Path(resolve_worker_model_path(model_id)) / ".pipeline_assignments"


def _assignment_cache_limits() -> tuple[int, int, float]:
    try:
        import config

        max_bytes = int(float(config.PIPELINE_ASSIGNMENT_CACHE_MAX_MB) * 1024 * 1024)
        min_free = int(float(config.PIPELINE_ASSIGNMENT_MIN_FREE_MB) * 1024 * 1024)
        stale_seconds = float(config.PIPELINE_ASSIGNMENT_STALE_SECONDS)
    except (AttributeError, TypeError, ValueError):
        max_bytes, min_free, stale_seconds = 4 * 1024**3, 512 * 1024**2, 86400.0
    return max(0, max_bytes), max(0, min_free), max(60.0, stale_seconds)


def _assignment_dir_size(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        return total
    return total


def _assignment_dir_config_id(path: Path) -> str:
    marker = path / "assignment.manifest.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return str(payload.get("config_id", "") or "")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""


def remove_pipeline_assignment_cache(
    model_id: str,
    config_id: str,
    node_id: str = "",
) -> bool:
    """Remove exactly one assignment generation after an abort/release."""
    root = assignment_cache_root(model_id)
    if not root.is_dir():
        return False
    removed = False
    for child in list(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        marker_config = _assignment_dir_config_id(child)
        marker_node = ""
        try:
            marker_node = str(json.loads(
                (child / "assignment.manifest.json").read_text(encoding="utf-8")
            ).get("node_id", "") or "")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        if marker_config != str(config_id) or (node_id and marker_node != str(node_id)):
            continue
        try:
            shutil.rmtree(child)
            removed = True
        except OSError:
            pass
    return removed


def reconcile_pipeline_assignment_cache(
    model_id: str,
    *,
    active_config_ids: set[str] | None = None,
    active_assignment_dirs: set[str] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Remove abandoned partials and old generations inside the cache root.

    Only directories created by assignment sync are touched.  Active config
    IDs are protected even when their mtime is old because a long load may be
    legitimately in progress.
    """
    root = assignment_cache_root(model_id)
    active = {str(value) for value in (active_config_ids or set()) if value}
    active_dirs = {
        str(value) for value in (active_assignment_dirs or set()) if value
    }
    current_time = time.time() if now is None else float(now)
    _max_bytes, _min_free, stale_seconds = _assignment_cache_limits()
    result = {
        "model_id": model_id,
        "root": str(root),
        "removed": [],
        "protected": [],
        "bytes_before": 0,
        "bytes_after": 0,
        "status": "ok",
    }
    if not root.is_dir():
        return result
    entries: list[tuple[Path, int, float, str]] = []
    try:
        children = list(root.iterdir())
    except OSError:
        result["status"] = "unavailable"
        return result
    for child in children:
        if not child.is_dir() or child.name.startswith("."):
            continue
        size = _assignment_dir_size(child)
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = current_time
        result["bytes_before"] += size
        config_id = _assignment_dir_config_id(child)
        if not config_id:
            # A directory without a valid marker is an incomplete generation;
            # it is never considered active and can be reclaimed by policy.
            config_id = ""
        if config_id in active or child.name in active_dirs:
            result["protected"].append(child.name)
            entries.append((child, size, mtime, config_id))
            continue
        stale = current_time - mtime >= stale_seconds
        has_part = any(item.is_file() and item.name.endswith(".part") for item in child.rglob("*"))
        if stale or has_part:
            try:
                shutil.rmtree(child)
                result["removed"].append(child.name)
                continue
            except OSError:
                result["status"] = "partial"
        entries.append((child, size, mtime, config_id))

    result["bytes_after"] = sum(_assignment_dir_size(item[0]) for item in entries)
    max_bytes, _min_free, _stale = _assignment_cache_limits()
    if max_bytes > 0 and result["bytes_after"] > max_bytes:
        for child, size, _mtime, config_id in sorted(entries, key=lambda item: item[2]):
            if config_id in active or child.name in active_dirs:
                continue
            try:
                shutil.rmtree(child)
            except OSError:
                result["status"] = "partial"
                continue
            result["removed"].append(child.name)
            result["bytes_after"] -= size
            if result["bytes_after"] <= max_bytes:
                break
    return result


def _ensure_assignment_disk_budget(destination_root: Path, expected_bytes: int) -> None:
    max_bytes, min_free_bytes, _stale = _assignment_cache_limits()
    try:
        free_bytes = shutil.disk_usage(destination_root.parent).free
    except OSError as exc:
        raise RuntimeError("assignment disk capacity is unavailable") from exc
    if free_bytes < expected_bytes + min_free_bytes:
        raise RuntimeError(
            f"assignment disk capacity insufficient: required={expected_bytes}, "
            f"free={free_bytes}, reserve={min_free_bytes}"
        )
    root = destination_root.parent
    existing = _assignment_dir_size(root) if root.is_dir() else 0
    current = _assignment_dir_size(destination_root) if destination_root.is_dir() else 0
    projected = max(0, existing - current) + expected_bytes
    if max_bytes > 0 and projected > max_bytes:
        raise RuntimeError(
            f"assignment cache budget exceeded: current={existing - current}, "
            f"incoming={expected_bytes}, limit={max_bytes}"
        )


def ensure_pipeline_assignment_available(
    master_host: str,
    master_api_port: int,
    assignment: dict[str, Any],
    *,
    timeout: float = 10.0,
) -> tuple[str, dict[str, Any]]:
    """Fetch and verify one assignment manifest without mirroring the model."""
    model_id = str(assignment.get("model_id", "") or "")
    config_id = str(assignment.get("config_id", "") or "")
    plan_id = str(assignment.get("plan_id", "") or "")
    node_id = str(assignment.get("node_id", "") or "")
    if not model_id or not config_id or not plan_id or not node_id:
        raise RuntimeError("pipeline assignment identity is incomplete")
    encoded_model_id = urllib.parse.quote(model_id, safe="")
    query = urllib.parse.urlencode({
        "config_id": config_id,
        "plan_id": plan_id,
        "node_id": node_id,
        "start_layer": int(assignment.get("start_layer", 0)),
        "end_layer": int(assignment.get("end_layer", 0)),
        "total_layers": int(assignment.get("total_layers", 0)),
        "has_embedding": int(bool(assignment.get("has_embedding", False))),
        "has_lm_head": int(bool(assignment.get("has_lm_head", False))),
    })
    from network_address import build_url

    base_url = build_url("http", master_host, int(master_api_port))
    manifest = _read_json(
        f"{base_url}/api/models/pipeline-assignment/{encoded_model_id}?{query}",
        timeout,
    )
    expected_revision = str(assignment.get("model_sha256", "") or "")
    if manifest.get("model_sha256") != expected_revision:
        raise RuntimeError("pipeline assignment model revision changed")
    if manifest.get("config_id") != config_id or manifest.get("plan_id") != plan_id:
        raise RuntimeError("pipeline assignment transaction identity changed")
    manifest_sha = str(manifest.get("manifest_sha256", "") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
        raise RuntimeError("pipeline assignment manifest digest is invalid")
    check = dict(manifest)
    check.pop("manifest_sha256", None)
    if hashlib.sha256(
        json.dumps(check, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() != manifest_sha:
        raise RuntimeError("pipeline assignment manifest digest mismatch")

    destination_root = _assignment_destination_root(model_id, config_id, node_id)
    destination_root.mkdir(parents=True, exist_ok=True)
    files = manifest.get("files") or []
    if not isinstance(files, list) or not files:
        raise RuntimeError("pipeline assignment manifest contains no files")
    expected_manifest_bytes = sum(
        int(entry.get("size_bytes", 0) or 0)
        for entry in files if isinstance(entry, dict)
    )
    reconcile_pipeline_assignment_cache(
        model_id,
        active_config_ids={config_id},
        active_assignment_dirs={destination_root.name},
    )
    _ensure_assignment_disk_budget(destination_root, expected_manifest_bytes)
    for entry in files:
        if not isinstance(entry, dict):
            raise RuntimeError("pipeline assignment file entry is invalid")
        relative_path = str(entry.get("path", ""))
        expected_size = int(entry.get("size_bytes", -1))
        expected_sha = str(entry.get("sha256", ""))
        destination = _safe_destination(destination_root, relative_path)
        if relative_path == "model.safetensors.index.json" and entry.get("filtered_weight_map"):
            filtered = {
                "metadata": {"total_size": 0},
                "weight_map": dict(entry["filtered_weight_map"]),
            }
            encoded = json.dumps(filtered, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if len(encoded) != expected_size or hashlib.sha256(encoded).hexdigest() != expected_sha:
                raise RuntimeError("filtered assignment index size or SHA mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(encoded)
            continue
        if entry.get("kind") == "weights" and entry.get("keys"):
            expected_key_digest = hashlib.sha256(
                "\n".join(sorted(str(key) for key in entry["keys"])).encode("utf-8")
            ).hexdigest()
            if expected_key_digest != str(entry.get("key_set_sha256", "")):
                raise RuntimeError(f"pipeline assignment key-set digest mismatch: {relative_path}")
        valid = destination.is_file() and destination.stat().st_size == expected_size
        if valid and expected_sha:
            valid = _file_sha256(destination) == expected_sha
        if not valid:
            encoded_path = urllib.parse.quote(relative_path.replace("\\", "/"), safe="/")
            _download_file(
                f"{base_url}/api/models/files/{encoded_model_id}/{encoded_path}",
                destination,
                expected_size,
            )
            if expected_sha and _file_sha256(destination) != expected_sha:
                raise RuntimeError(f"pipeline assignment SHA256 mismatch: {relative_path}")

    marker = destination_root / "assignment.manifest.json"
    marker.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    return str(destination_root), manifest
