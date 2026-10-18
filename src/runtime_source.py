"""Opt-in immutable startup provenance for local product observations.

This module does not load models, read credentials or expose a service. A process
records its source bytes once at startup and diagnostics rehash that same file
set. Git-less deployments consume an observed checkout manifest;
revision/cleanliness cannot be supplied by environment strings or probe callers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

_LOCK = threading.Lock()
_STARTUP = None
_ERROR = None
_PRUNE = {".git", "__pycache__", "build", "dist", "node_modules", ".venv", ".venv-test", "venv"}


class SourceObservationError(ValueError):
    pass


def _require(value, reason):
    if not value:
        raise SourceObservationError(reason)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate source manifest key")
        result[key] = value
    return result


def _read_object(path):
    def invalid(_):
        raise SourceObservationError("invalid source manifest number")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_json_object,
                      parse_constant=invalid)


def _git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                                capture_output=True, env=env, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceObservationError("startup source Git observation unavailable") from exc
    _require(result.returncode == 0, "startup source Git identity unavailable")
    return result.stdout


def _git_identity(root):
    observed = Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
    _require(observed == root, "startup source root is not the main checkout")
    revision = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", revision), "invalid startup source revision")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "-z")
    return revision, status


def _inventory(root):
    names = {"release-contract.json"}
    for directory in ("src", "scripts"):
        base = root / directory
        _require(base.is_dir(), "startup source directories missing")
        for parent, directories, files in os.walk(base, followlinks=False):
            for name in directories:
                if name not in _PRUNE:
                    _require(not (Path(parent) / name).is_symlink(), "source directory symlink refused")
            directories[:] = [name for name in directories if name not in _PRUNE]
            for name in files:
                if name.endswith(".py"):
                    path = Path(parent) / name
                    _require(not path.is_symlink(), "source file symlink refused")
                    names.add(path.relative_to(root).as_posix())
    _require("src/api_server.py" in names and "src/runtime_source.py" in names,
             "main executable source files missing")
    return sorted(names)


def _rehash(root, names):
    records = []
    for name in names:
        path = root / name
        _require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink(),
                 "startup source file missing or escaped")
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                 and len(data) == after.st_size, "source changed while hashing")
        records.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
    return records


def checkout_manifest(root):
    """Observe real checkout identity and source bytes; also used by deployment export."""
    root = Path(root).resolve()
    before = _git_identity(root)
    names = _inventory(root)
    first = _rehash(root, names)
    second = _rehash(root, _inventory(root))
    tracked = {os.fsdecode(name).replace("\\", "/") for name in
               _git(root, "ls-files", "-z", "--", "src", "scripts", "release-contract.json").split(b"\0") if name}
    _require(first == second and _git_identity(root) == before, "startup source changed during capture")
    # Broad ignore rules must never hide executable Python from cleanliness.
    return {"schema_version": 1, "source_revision": before[0],
            "source_dirty": bool(before[1]) or not set(names).issubset(tracked), "files": second}


def _export_manifest(root, path):
    payload = _read_object(path)
    _require(isinstance(payload, dict) and payload.get("schema_version") == 1,
             "invalid deployment source manifest")
    manifest = payload.get("manifest")
    _require(isinstance(manifest, dict) and type(manifest.get("source_dirty")) is bool
             and isinstance(manifest.get("source_revision"), str)
             and re.fullmatch(r"[0-9a-f]{40}", manifest["source_revision"]),
             "deployment source manifest must preserve observed Git revision and boolean cleanliness")
    entries = manifest.get("files")
    _require(isinstance(entries, list) and entries, "deployment source file inventory missing")
    names = []
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == {"path", "sha256"}, "invalid deployment file record")
        name, digest = entry["path"], entry["sha256"]
        _require(isinstance(name, str) and name and "\\" not in name and not Path(name).is_absolute()
                 and not re.match(r"^[A-Za-z]:", name) and ".." not in Path(name).parts,
                 "unsafe deployment source path")
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid deployment source digest")
        names.append(name)
    _require(names == sorted(set(names)) == _inventory(root), "deployment source inventory incomplete or has extra files")
    _require(_rehash(root, names) == entries == _rehash(root, names), "deployment source content mismatch")
    return manifest


def initialize(root=None):
    """Record startup provenance once. Disabled by default; failure affects probes only."""
    global _STARTUP, _ERROR
    if os.environ.get("QLH_PRODUCT_OBSERVATION") != "1":
        return
    with _LOCK:
        if _STARTUP is not None or _ERROR is not None:
            return
        try:
            import psutil
            process = psutil.Process(os.getpid())
            if getattr(sys, "frozen", False):
                from runtime_frozen_source import capture_startup
                _STARTUP = capture_startup(process, root=root)
                return
            source_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
            identity = {"pid": process.pid, "created_at": process.create_time()}
            _require(Path(process.cwd()).resolve() == source_root, "running process cwd differs from source root")
            configured = os.environ.get("QLH_RUNTIME_SOURCE_MANIFEST")
            if configured:
                configured_path = Path(configured)
                _require(configured_path.is_absolute(), "deployment manifest path must be absolute")
                manifest = _export_manifest(source_root, configured_path)
            else:
                manifest = checkout_manifest(source_root)
            _STARTUP = {**identity, "source_root": str(source_root), "manifest": manifest,
                        "manifest_sha256": canonical_hash(manifest)}
        except Exception as exc:
            # Opt-in diagnostic capture must not prevent the application startup.
            # Keep every failure terminal for this process rather than recapturing.
            _ERROR = type(exc).__name__ + ": " + str(exc)


def startup_source():
    with _LOCK:
        _require(_STARTUP is not None, _ERROR or "startup source was not captured before runtime imports")
        return deepcopy(_STARTUP)


def observe_source(platform):
    """Rehash precisely the startup inventory and bind it to this live process."""
    _require(platform in ("master", "pc"), "unsupported runtime source role")
    snapshot = startup_source()
    import psutil
    process = psutil.Process(os.getpid())
    if snapshot.get("kind") == "installed_exe":
        from runtime_frozen_source import observe_source as observe_frozen_source
        return observe_frozen_source(snapshot, process)
    _require((process.pid, process.create_time()) == (snapshot["pid"], snapshot["created_at"]),
             "startup receipt belongs to another process")
    root = Path(snapshot["source_root"])
    _require(Path(process.cwd()).resolve() == root, "runtime cwd changed after startup")
    names = [entry["path"] for entry in snapshot["manifest"]["files"]]
    _require(names == _inventory(root), "runtime source inventory changed after startup")
    rehashed = _rehash(root, names)
    _require(rehashed == snapshot["manifest"]["files"] == _rehash(root, names),
             "running source files differ from startup receipt")
    return {**snapshot, "kind": "runtime_manifest" if platform == "master" else "synced_manifest",
            "source_revision": snapshot["manifest"]["source_revision"],
            "source_dirty": snapshot["manifest"]["source_dirty"],
            "runtime_manifest_sha256": snapshot["manifest_sha256"], "rehashed_files": rehashed}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export observed source bytes for a trusted Git-less lab deployment")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--include-dirty", action="store_true", help="Export observed dirty provenance for development diagnostics; strict gates still reject it")
    args = parser.parse_args(argv)
    try:
        manifest = checkout_manifest(args.root)
        _require(manifest["source_dirty"] is False or args.include_dirty, "cannot export modified/unverifiable source as clean; use --include-dirty for explicit development provenance")
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "manifest": manifest}, stream, sort_keys=True, indent=2)
            stream.write("\n")
        print(json.dumps({"exported": True, "manifest_sha256": canonical_hash(manifest),
                          "source_revision": manifest["source_revision"], "source_dirty": manifest["source_dirty"]}))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"exported": False, "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
