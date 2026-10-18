"""Unsigned, opt-in provenance for a complete installed frozen application.

The builder calls export_package_manifest only after staging the actual
build-info.json and the complete application tree. This is a Gitless lab
observation contract, not a release signature or compiler attestation. Runtime
capture pins the manifest once; later observations only verify those same bytes.
No model, native runtime, credential contents or release signing API is imported.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path
import re
import stat
import sys

MANIFEST_NAME = "product-package-observation.json"
BUILD_INFO_NAME = "build-info.json"
# Created after this export; excluding only this exact file breaks the signing
# cycle while every other manifest/ program file remains in the inventory.
INSTALL_MANIFEST_NAME = "manifest/install-manifest.json"
# Only top-level runtime data is pruned. The retention contract owns models,
# logs, chat_history, config and local_docs; local_store also has a state fallback.
EXCLUDED_TOP_LEVEL_DIRECTORIES = frozenset({
    "models", "logs", "state", "chat_history", "config", "local_docs",
})
# config.py reads .env; node_config migrates the install-root config and keeps a
# sibling secret store. The launcher owns .tailscale_joined. local_store can use
# an old install-root SQLite database (including SQLite's mutable sidecars).
EXCLUDED_TOP_LEVEL_FILES = frozenset({
    ".env", ".tailscale_joined", "node_config.json", "node_secrets.json",
    "qlh-control.sqlite3", "qlh-control.sqlite3-wal", "qlh-control.sqlite3-shm",
    "qlh-control.sqlite3-journal",
})
_BUILD_INFO_LIMIT = 64 * 1024
_MANIFEST_LIMIT = 64 * 1024 * 1024
_MAX_FILES = 250_000
_CHUNK = 1024 * 1024
_MANIFEST_FIELDS = {"schema_version", "source_revision", "source_dirty",
                    "executable_relative", "build_info_sha256", "files"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_WINDOWS_DEVICE = re.compile(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", re.I)


class FrozenSourceObservationError(ValueError):
    """Missing, changed or unsafe installed-application evidence."""


def _require(condition, message):
    if not condition:
        raise FrozenSourceObservationError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _relative(name):
    _require(isinstance(name, str) and bool(name) and "\\" not in name
             and not ntpath.isabs(name) and not ntpath.splitdrive(name)[0],
             "unsafe package relative path")
    parts = name.split("/")
    _require(all(part and part not in {".", ".."} and not part.endswith((" ", "."))
                 and not _WINDOWS_DEVICE.fullmatch(part)
                 and not any(ord(c) < 32 or c in '<>:"|?*' for c in part)
                 for part in parts), "unsafe package relative path")
    return name


def _excluded(name):
    parts = name.casefold().split("/")
    return (name == INSTALL_MANIFEST_NAME
            or parts[0] in EXCLUDED_TOP_LEVEL_DIRECTORIES
            or len(parts) == 1 and parts[0] in EXCLUDED_TOP_LEVEL_FILES)


def _plain_stat(path):
    try:
        result = path.lstat()
    except OSError as exc:
        raise FrozenSourceObservationError("package path missing or unreadable") from exc
    _require(not stat.S_ISLNK(result.st_mode)
             and not getattr(result, "st_file_attributes", 0) & 0x400,
             "package link or reparse point refused")
    return result


def _stamp(metadata):
    # Windows lstat synthesizes executable permission bits from the extension;
    # fstat does not. Compare the file type, actual identity, size and timestamps.
    return (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode), metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _root(value):
    candidate = Path(value).absolute()
    _require(".." not in candidate.parts, "unsafe package root")
    for parent in reversed((candidate, *candidate.parents)):
        _require(stat.S_ISDIR(_plain_stat(parent).st_mode), "package root is not a directory")
    return candidate.resolve(strict=True)


def _member(root, name):
    current = root
    parts = _relative(name).split("/")
    for offset, part in enumerate(parts):
        current = current / part
        info = _plain_stat(current)
        if offset + 1 < len(parts):
            _require(stat.S_ISDIR(info.st_mode), "package member parent is not a directory")
        else:
            _require(stat.S_ISREG(info.st_mode), "package member is not a regular file")
    _require(current.resolve(strict=True).is_relative_to(root), "package member escaped root")
    return current, info


def _hash_member(root, name, *, limit=None):
    """Stream bytes with path/handle identity and before/after stat checks."""
    path, before = _member(root, name)
    _require(limit is None or before.st_size <= limit, "package metadata exceeds size limit")
    digest, count, chunks = hashlib.sha256(), 0, [] if limit is not None else None
    with path.open("rb") as stream:
        # Windows fstat/lstat can expose different ctime (creation/change)
        # semantics. Path-to-path still compares ctime; handle-to-path compares
        # the shared file identity, type, size and nanosecond mtime fields.
        _require(_stamp(os.fstat(stream.fileno()))[:-1] == _stamp(before)[:-1],
                 "package file changed before hashing")
        while chunk := stream.read(_CHUNK):
            count += len(chunk)
            _require(count <= before.st_size, "package file changed while hashing")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after_handle = os.fstat(stream.fileno())
    _, after = _member(root, name)
    _require(count == before.st_size and _stamp(before) == _stamp(after)
             and _stamp(before)[:-1] == _stamp(after_handle)[:-1],
             "package file changed while hashing")
    record = {"path": name, "size_bytes": count, "sha256": digest.hexdigest()}
    return record, _stamp(after), b"".join(chunks) if chunks is not None else None


def _inventory(root):
    files, directories, seen = {}, {}, set()

    def visit(directory, prefix=""):
        info = _plain_stat(directory)
        _require(stat.S_ISDIR(info.st_mode), "package directory is not regular")
        directories[prefix] = (info.st_dev, info.st_ino, info.st_mode)
        with os.scandir(directory) as entries:
            members = sorted(entries, key=lambda item: item.name)
        for entry in members:
            name = _relative(prefix + entry.name)
            _require(name.casefold() not in seen, "duplicate package path")
            seen.add(name.casefold())
            path = directory / entry.name
            info = _plain_stat(path)
            if not prefix:
                lower = entry.name.casefold()
                if lower in EXCLUDED_TOP_LEVEL_DIRECTORIES:
                    _require(stat.S_ISDIR(info.st_mode), "runtime data root is not a directory")
                    continue
                if lower in EXCLUDED_TOP_LEVEL_FILES or lower == MANIFEST_NAME.casefold():
                    _require(stat.S_ISREG(info.st_mode), "package metadata/runtime data is not a regular file")
                    continue
            if name == INSTALL_MANIFEST_NAME:
                _require(stat.S_ISREG(info.st_mode), "install manifest is not a regular file")
                continue
            if stat.S_ISDIR(info.st_mode):
                visit(path, name + "/")
            else:
                _require(stat.S_ISREG(info.st_mode), "package contains a special file")
                files[name] = _stamp(info)
                _require(len(files) <= _MAX_FILES, "package file inventory exceeds limit")

    visit(root)
    return files, directories


def _rehash_package(root):
    before = _inventory(root)
    records = []
    for name in sorted(before[0]):
        record, stamp, _ = _hash_member(root, name)
        _require(stamp == before[0][name], "package changed during inventory hashing")
        records.append(record)
    _require(before == _inventory(root), "package inventory changed while hashing")
    return records, before


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate package JSON key")
        value[key] = item
    return value


def _invalid_constant(value):
    raise FrozenSourceObservationError("nonfinite package JSON number")


def _finite_float(value):
    number = float(value)
    _require(math.isfinite(number), "nonfinite package JSON number")
    return number


def _read_json(data):
    try:
        result = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_json_object,
                            parse_constant=_invalid_constant, parse_float=_finite_float)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise FrozenSourceObservationError("invalid package JSON: " + str(exc)) from exc
    _require(isinstance(result, dict), "package JSON must be an object")
    return result


def _source_identity(value):
    _require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
             "invalid package/build-info schema_version")
    _require(isinstance(value.get("source_revision"), str)
             and _REVISION.fullmatch(value["source_revision"])
             and type(value.get("source_dirty")) is bool,
             "package/build-info requires observed revision and boolean source_dirty")


def _executable_name(name):
    _relative(name)
    _require("/" not in name and name.lower().endswith(".exe") and not _excluded(name),
             "package executable must be an install-root EXE")
    return name


def _validate_manifest(value):
    _require(set(value) == _MANIFEST_FIELDS, "invalid package manifest fields")
    _source_identity(value)
    executable = _executable_name(value["executable_relative"])
    _require(isinstance(value["build_info_sha256"], str)
             and _SHA256.fullmatch(value["build_info_sha256"]), "invalid build-info SHA256")
    entries = value["files"]
    _require(isinstance(entries, list) and 0 < len(entries) <= _MAX_FILES,
             "package file inventory missing or too large")
    names, folded, by_name = [], set(), {}
    for item in entries:
        _require(isinstance(item, dict) and set(item) == {"path", "size_bytes", "sha256"},
                 "invalid package file record")
        name = _relative(item["path"])
        _require(not _excluded(name) and name.casefold() != MANIFEST_NAME.casefold(),
                 "package inventory contains excluded metadata/runtime data")
        _require(name.casefold() not in folded, "duplicate package file record")
        _require(type(item["size_bytes"]) is int and item["size_bytes"] >= 0,
                 "invalid package file size")
        _require(isinstance(item["sha256"], str) and _SHA256.fullmatch(item["sha256"]),
                 "invalid package file SHA256")
        names.append(name)
        folded.add(name.casefold())
        by_name[name] = item
    _require(names == sorted(names), "package file inventory must be sorted")
    _require(executable in by_name and BUILD_INFO_NAME in by_name,
             "package inventory must contain executable and build-info.json")
    _require(by_name[executable]["size_bytes"] > 0 and by_name[BUILD_INFO_NAME]["size_bytes"] > 0,
             "empty executable or build-info.json")
    _require(by_name[BUILD_INFO_NAME]["sha256"] == value["build_info_sha256"],
             "package build-info digest differs from inventory")
    return value


def _package_bytes(root):
    first, first_state = _rehash_package(root)
    info_record, info_stamp, data = _hash_member(root, BUILD_INFO_NAME, limit=_BUILD_INFO_LIMIT)
    _require(info_record in first and first_state[0][BUILD_INFO_NAME] == info_stamp,
             "build-info changed during package capture")
    info = _read_json(data)
    _source_identity(info)
    second, second_state = _rehash_package(root)
    _require(first == second and first_state == second_state,
             "package bytes or inventory changed during capture")
    return info, second, second_state


def export_package_manifest(app_root, *, executable_relative="QLH-Edge-Inference.exe"):
    """Create fixed unsigned metadata from the staged bytes; never overwrite.

    No revision/cleanliness override is accepted. Dirty build-info is preserved
    as dirty evidence, which the strict product gate must reject. Export again
    into a fresh application tree if packaging changes any program file.
    """
    root = _root(app_root)
    name = _executable_name(executable_relative)
    destination = root / MANIFEST_NAME
    _require(not destination.exists() and not destination.is_symlink(),
             "package observation metadata already exists; refusing overwrite")
    info, files, state = _package_bytes(root)
    entries = {item["path"]: item for item in files}
    _require(name in entries, "package executable missing")
    manifest = _validate_manifest({
        "schema_version": 1, "source_revision": info["source_revision"],
        "source_dirty": info["source_dirty"], "executable_relative": name,
        "build_info_sha256": entries[BUILD_INFO_NAME]["sha256"], "files": files,
    })
    data = (json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _require(len(data) <= _MANIFEST_LIMIT, "package observation metadata exceeds size limit")
    with destination.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _require(state == _inventory(root), "package changed while exporting metadata")
    record, _, captured = _hash_member(root, MANIFEST_NAME, limit=_MANIFEST_LIMIT)
    _require(captured == data and record["sha256"] == hashlib.sha256(data).hexdigest(),
             "package observation metadata changed during export")
    return deepcopy(manifest)


def _verify_package(root):
    metadata, metadata_stamp, data = _hash_member(root, MANIFEST_NAME, limit=_MANIFEST_LIMIT)
    manifest = _validate_manifest(_read_json(data))
    info, records, _ = _package_bytes(root)
    after, after_stamp, _ = _hash_member(root, MANIFEST_NAME, limit=_MANIFEST_LIMIT)
    _require(metadata == after and metadata_stamp == after_stamp,
             "package observation metadata changed during verification")
    _require(records == manifest["files"], "package inventory/content differs from exported manifest")
    _require(info["source_revision"] == manifest["source_revision"]
             and info["source_dirty"] is manifest["source_dirty"],
             "package manifest revision/cleanliness differs from actual build-info")
    entries = {entry["path"]: entry for entry in records}
    return {"manifest": manifest, "manifest_sha256": canonical_hash(manifest),
            "runtime_manifest_sha256": canonical_hash(manifest),
            "build_info": info, "build_info_sha256": entries[BUILD_INFO_NAME]["sha256"],
            "executable_sha256": entries[manifest["executable_relative"]]["sha256"],
            "rehashed_files": records, "source_revision": info["source_revision"],
            "source_dirty": info["source_dirty"],
            # Pin raw metadata too, so reformatting/replacing it cannot recapture
            # a startup receipt. This private field is omitted from observations.
            "_manifest_file_sha256": metadata["sha256"]}


def _process_binding(process):
    pid, created = process.pid, process.create_time()
    _require(type(pid) is int and pid > 0 and type(created) in (int, float)
             and math.isfinite(created) and created > 0, "invalid frozen process identity")
    executable, actual = Path(sys.executable), Path(process.exe())
    _require(executable.is_absolute() and actual.is_absolute(), "frozen executable path must be absolute")
    root = _root(executable.parent)
    member, _ = _member(root, _executable_name(executable.name))
    _require(actual == member, "OS process executable differs from sys.executable")
    _require(Path(process.cwd()).is_absolute() and _root(process.cwd()) == root,
             "frozen process cwd differs from installation root")
    return {"pid": pid, "created_at": created, "source_root": str(root), "executable": str(member)}


def _fresh_binding(process):
    # psutil can cache create_time/exe on a Process instance. Reopen the PID for
    # the closing bracket instead of comparing a cached identity to itself.
    import psutil
    return _process_binding(psutil.Process(process.pid))


def capture_startup(process, *, root=None):
    """Pin a frozen installation once; runtime_source owns opt-in and locking."""
    binding = _process_binding(process)
    install_root = Path(binding["source_root"])
    _require(root is None or _root(root) == install_root,
             "configured frozen root differs from actual executable installation")
    source = _verify_package(install_root)
    _require(install_root / source["manifest"]["executable_relative"] == Path(binding["executable"]),
             "manifest executable differs from running process")
    _require(binding == _fresh_binding(process), "frozen process changed during startup capture")
    return {"kind": "installed_exe", **binding, **source}


def observe_source(snapshot, process):
    """Verify the original package and live PID; never adopt new package bytes."""
    binding = _process_binding(process)
    _require(all(snapshot.get(key) == value for key, value in binding.items()),
             "frozen process identity/executable/cwd changed after startup")
    current = _verify_package(Path(binding["source_root"]))
    _require(all(snapshot.get(key) == value for key, value in current.items()),
             "frozen package differs from immutable startup receipt")
    _require(binding == _fresh_binding(process), "frozen process changed during observation")
    return deepcopy({key: value for key, value in snapshot.items() if not key.startswith("_")})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export unsigned frozen-package lab observation metadata")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--executable-relative", default="QLH-Edge-Inference.exe")
    args = parser.parse_args(argv)
    try:
        manifest = export_package_manifest(args.root, executable_relative=args.executable_relative)
    except (OSError, ValueError) as exc:
        print(json.dumps({"exported": False, "reason": str(exc)}))
        return 1
    print(json.dumps({"exported": True, "manifest_sha256": canonical_hash(manifest),
                      "source_revision": manifest["source_revision"], "source_dirty": manifest["source_dirty"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
