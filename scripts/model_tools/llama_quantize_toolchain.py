"""Pinned llama-quantize package validation and runtime discovery."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path(__file__).with_name("llama_quantize.lock.json")
MANIFEST_NAME = "manifest.json"
MAX_MANIFEST_BYTES = 64 * 1024


class LlamaQuantizeToolchainError(ValueError):
    """Raised when the pinned toolchain contract is invalid."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_platform(value: str | None = None) -> str:
    name = (value or platform.system()).strip().lower()
    if name in {"win32", "windows"}:
        return "windows"
    if name.startswith("linux"):
        return "linux"
    raise LlamaQuantizeToolchainError(f"unsupported platform: {name}")


def normalize_architecture(value: str | None = None) -> str:
    name = (value or platform.machine()).strip().lower()
    if name in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    raise LlamaQuantizeToolchainError(f"unsupported architecture: {name}")


def host_target_id(*, platform_name: str | None = None, architecture: str | None = None) -> str:
    return f"{normalize_platform(platform_name)}-{normalize_architecture(architecture)}"


def expected_source_revision(lock: dict[str, Any]) -> str:
    """The source HEAD pin differs from the upstream baseline for committed patches."""
    upstream = lock.get("upstream")
    if not isinstance(upstream, dict):
        raise LlamaQuantizeToolchainError("invalid llama-quantize upstream contract")
    revision = upstream.get("revision")
    local = lock.get("revision_local")
    for value in (revision, local) if "revision_local" in lock else (revision,):
        if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise LlamaQuantizeToolchainError("invalid pinned llama.cpp source revision")
    return local if "revision_local" in lock else revision


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LlamaQuantizeToolchainError("cannot read llama-quantize lock") from exc
    if not isinstance(lock, dict) or lock.get("schema_version") != 1 or lock.get("tool") != "llama-quantize":
        raise LlamaQuantizeToolchainError("invalid llama-quantize lock header")
    upstream = lock.get("upstream")
    targets = lock.get("targets")
    if not isinstance(upstream, dict) or not isinstance(targets, dict):
        raise LlamaQuantizeToolchainError("invalid llama-quantize lock structure")
    expected_source_revision(lock)
    for target_id, target in targets.items():
        if not isinstance(target_id, str) or not isinstance(target, dict):
            raise LlamaQuantizeToolchainError("invalid llama-quantize target")
        if target_id != f"{target.get('platform')}-{target.get('architecture')}":
            raise LlamaQuantizeToolchainError("llama-quantize target ID mismatch")
        executable = target.get("executable")
        if not isinstance(executable, str) or Path(executable).name != executable:
            raise LlamaQuantizeToolchainError("invalid llama-quantize executable name")
    return lock


def default_package_root(project_root: Path = ROOT) -> Path:
    return project_root / "build" / "model-tools" / "llama-quantize"


def managed_package_dir(
    project_root: Path = ROOT,
    *,
    target_id: str | None = None,
) -> Path:
    return default_package_root(project_root) / "packages" / (target_id or host_target_id())


def llama_cpp_source(project_root: Path = ROOT) -> Path:
    """Resolve the Android-owned llama.cpp checkout used by core model tools."""
    roots: list[Path] = []
    configured_root = os.environ.get("QLH_ANDROID_ROOT")
    if configured_root:
        roots.append(Path(configured_root).expanduser())
    roots.extend((project_root.parent / "qlh-android", project_root / "android"))
    for android_root in roots:
        candidate = android_root / "app" / "src" / "main" / "cpp" / "llama.cpp"
        if candidate.is_dir():
            return candidate
    return roots[0] / "app" / "src" / "main" / "cpp" / "llama.cpp"


def verify_managed_package(
    package_dir: Path,
    *,
    expected_target: str | None = None,
    lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lock = lock or load_lock()
    target_id = expected_target or host_target_id()
    errors: list[dict[str, str]] = []

    def fail(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    manifest_path = package_dir / MANIFEST_NAME
    manifest: dict[str, Any] = {}
    if package_dir.is_symlink() or not package_dir.is_dir():
        fail("package_missing", "managed package directory is missing or unsafe")
    elif manifest_path.is_symlink() or not manifest_path.is_file():
        fail("manifest_missing", "managed package manifest is missing or unsafe")
    else:
        try:
            if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError("manifest too large")
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("manifest must be an object")
            manifest = parsed
        except (OSError, ValueError, json.JSONDecodeError):
            fail("manifest_invalid", "managed package manifest is invalid")

    revision: str | None = None
    source_dirty: bool | None = None
    source_identity = "unproven"
    expected_revision = expected_source_revision(lock)
    target = lock["targets"].get(target_id)
    if target is None:
        fail("target_unsupported", "managed package target is not supported by the lock")
    if manifest:
        if manifest.get("schema_version") != 1 or manifest.get("tool") != lock["tool"]:
            fail("manifest_contract_mismatch", "managed package manifest header does not match the lock")
        if manifest.get("target_id") != target_id:
            fail("target_mismatch", "managed package target does not match this host")
        upstream = manifest.get("upstream")
        if not isinstance(upstream, dict) or upstream.get("repository") != lock["upstream"]["repository"] or upstream.get("revision") != lock["upstream"]["revision"]:
            fail("revision_mismatch", "managed package llama.cpp revision does not match the lock")
        if "source_revision" not in manifest:
            if "revision_local" in lock:
                fail("source_revision_missing", "managed package lacks the locked committed-patch source identity; rebuild it")
            elif isinstance(upstream, dict) and upstream.get("revision") == expected_revision:
                # Legacy schema 1 recorded HEAD in upstream only. Dirty/patch facts
                # were not recorded, so never synthesize them from the current lock.
                revision = upstream["revision"]
                source_identity = "legacy_upstream_only"
        else:
            observed = manifest.get("source_revision")
            if isinstance(observed, str) and len(observed) == 40 and all(char in "0123456789abcdef" for char in observed):
                revision = observed
            if revision != expected_revision:
                fail("source_revision_mismatch", "managed package source HEAD does not match the lock")
            dirty = manifest.get("source_dirty")
            if type(dirty) is not bool:
                fail("source_contract_invalid", "managed package source dirty fact is missing or invalid")
            else:
                source_dirty = dirty
                if "revision_local" in lock and dirty:
                    fail("source_contract_invalid", "locked committed-patch source must not contain worktree patches")
            receipts = manifest.get("source_patches")
            expected_patches = {
                item.get("id") or item["path"]: item["sha256"].lower()
                for item in lock.get("patches", [])
            }
            observed_patches = {}
            states = []
            if isinstance(receipts, list):
                for item in receipts:
                    if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                            or item["id"] in observed_patches or item.get("state") not in {"applied", "applied_committed"}):
                        fail("source_patches_invalid", "managed package patch receipt is invalid")
                        continue
                    observed_patches[item["id"]] = item.get("patch_sha256")
                    states.append(item["state"])
            if not isinstance(receipts, list) or observed_patches != expected_patches:
                fail("source_patches_invalid", "managed package patch identities differ from the lock")
            if type(dirty) is bool and dirty != any(state == "applied" for state in states):
                fail("source_contract_invalid", "managed package patch states disagree with source dirty fact")
            source_identity = "explicit_source"
        if target is not None and manifest.get("executable") != target["executable"]:
            fail("executable_mismatch", "managed package executable does not match the lock")
        smoke = manifest.get("smoke")
        if (
            not isinstance(smoke, dict)
            or smoke.get("help") is not True
            or smoke.get("q4_k_m_listed") is not True
            or smoke.get("runtime_dependencies_verified") is not True
        ):
            fail("smoke_missing", "managed package was not proven by the required help smoke")

        files = manifest.get("files")
        listed_names: set[str] = set()
        if not isinstance(files, list) or not files:
            fail("files_invalid", "managed package file list is invalid")
        else:
            for item in files:
                if not isinstance(item, dict):
                    fail("files_invalid", "managed package file entry is invalid")
                    continue
                name = item.get("path")
                if not isinstance(name, str) or Path(name).name != name or name in listed_names:
                    fail("file_path_invalid", "managed package contains an unsafe or duplicate file name")
                    continue
                listed_names.add(name)
                file_path = package_dir / name
                if file_path.is_symlink() or not file_path.is_file():
                    fail("file_missing", f"managed package file is missing: {name}")
                    continue
                try:
                    size = file_path.stat().st_size
                    digest = file_sha256(file_path)
                except OSError:
                    fail("file_unreadable", f"managed package file is unreadable: {name}")
                    continue
                if item.get("size_bytes") != size or item.get("sha256") != digest:
                    fail("file_digest_mismatch", f"managed package file digest does not match: {name}")
            try:
                actual_names = {item.name for item in package_dir.iterdir() if item.name != MANIFEST_NAME}
            except OSError:
                actual_names = set()
                fail("package_unreadable", "managed package directory is unreadable")
            if actual_names != listed_names:
                fail("file_set_mismatch", "managed package contains unlisted or missing files")

    executable_name = target["executable"] if target is not None else None
    executable_path = package_dir / executable_name if executable_name else None
    if executable_path is not None and normalize_platform(target["platform"]) == "linux":
        if executable_path.is_file() and not os.access(executable_path, os.X_OK):
            fail("executable_permission_missing", "managed Linux executable is not executable")
    executable_record = next(
        (item for item in manifest.get("files", []) if isinstance(item, dict) and item.get("path") == executable_name),
        {},
    ) if manifest else {}
    return {
        "valid": not errors,
        "status": "verified" if not errors else "invalid",
        "package": package_dir.name,
        "target_id": target_id,
        "revision": revision,
        "upstream_revision": manifest.get("upstream", {}).get("revision") if isinstance(manifest.get("upstream"), dict) else None,
        "expected_revision": expected_revision,
        "source_dirty": source_dirty,
        "source_identity": source_identity,
        "executable": executable_name,
        "sha256": executable_record.get("sha256"),
        "size_bytes": executable_record.get("size_bytes"),
        "errors": errors,
    }


def managed_package_candidates(
    *,
    project_root: Path = ROOT,
    target_id: str | None = None,
) -> list[Path]:
    target_id = target_id or host_target_id()
    roots: list[Path] = []
    configured = os.environ.get("QLH_MODEL_TOOLS_ROOT")
    if configured:
        roots.append(Path(configured).expanduser())
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.append(Path(frozen_root) / "model-tools")
    roots.extend([
        Path(sys.executable).resolve().parent / "model-tools",
        project_root / "build" / "model-tools",
    ])
    if normalize_platform() == "linux":
        roots.append(Path("/opt/qlh-edge-inference/model-tools"))
    candidates: list[Path] = []
    for root in roots:
        candidate = root / "llama-quantize"
        if root == project_root / "build" / "model-tools":
            candidate /= "packages"
        candidate /= target_id
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _unmanaged_summary(path: Path, provenance: str) -> dict[str, Any]:
    try:
        digest = file_sha256(path)
        size = path.stat().st_size
    except OSError:
        return {"label": path.name, "status": "missing", "provenance": provenance, "verification": "unmanaged"}
    return {
        "label": path.name,
        "status": "available",
        "provenance": provenance,
        "verification": "unmanaged",
        "sha256": digest,
        "size_bytes": size,
        "revision": None,
    }


def resolve_quantizer(
    explicit: Path | None = None,
    *,
    project_root: Path = ROOT,
) -> tuple[Path | None, dict[str, Any]]:
    if explicit is not None:
        candidate = explicit.expanduser().absolute()
        if candidate.is_file() and not candidate.is_symlink():
            return candidate, _unmanaged_summary(candidate, "explicit")
        return None, {"label": candidate.name, "status": "missing", "provenance": "explicit", "verification": "unmanaged"}

    configured = os.environ.get("QLH_LLAMA_QUANTIZE")
    if configured:
        candidate = Path(configured).expanduser().absolute()
        if candidate.is_file() and not candidate.is_symlink():
            return candidate, _unmanaged_summary(candidate, "environment")
        return None, {"label": candidate.name, "status": "missing", "provenance": "environment", "verification": "unmanaged"}

    lock = load_lock()
    target_id = host_target_id()
    for package_dir in managed_package_candidates(project_root=project_root, target_id=target_id):
        if not package_dir.exists() and not package_dir.is_symlink():
            continue
        verification = verify_managed_package(package_dir, expected_target=target_id, lock=lock)
        if not verification["valid"]:
            return None, {
                "label": verification.get("executable"),
                "status": "invalid",
                "provenance": "managed_package",
                "verification": "failed",
                "revision": verification.get("revision"),
                "errors": verification.get("errors", []),
            }
        executable = package_dir / str(verification["executable"])
        return executable, {
            "label": executable.name,
            "status": "available",
            "provenance": "managed_package",
            "verification": "verified",
            "revision": verification["revision"],
            "upstream_revision": verification["upstream_revision"],
            "source_dirty": verification["source_dirty"],
            "source_identity": verification["source_identity"],
            "sha256": verification["sha256"],
            "size_bytes": verification["size_bytes"],
            "target_id": target_id,
        }

    for name in ("llama-quantize", "llama-quantize.exe"):
        found = shutil.which(name)
        if found:
            candidate = Path(found).absolute()
            return candidate, _unmanaged_summary(candidate, "path")
    repo = llama_cpp_source(project_root)
    for relative in ("build/bin/llama-quantize", "build/bin/llama-quantize.exe", "build/bin/Release/llama-quantize.exe"):
        candidate = repo / relative
        if candidate.is_file() and not candidate.is_symlink():
            return candidate, _unmanaged_summary(candidate, "source_build")
    return None, {"label": None, "status": "missing", "provenance": None, "verification": None}


__all__ = [
    "LlamaQuantizeToolchainError",
    "LOCK_PATH",
    "ROOT",
    "default_package_root",
    "file_sha256",
    "expected_source_revision",
    "host_target_id",
    "load_lock",
    "llama_cpp_source",
    "managed_package_candidates",
    "managed_package_dir",
    "normalize_architecture",
    "normalize_platform",
    "resolve_quantizer",
    "verify_managed_package",
]
