#!/usr/bin/env python3
"""Observe QLH build sources from Git and working-tree bytes, never environment revisions.

The fixed repository graph is part of this contract. A source export, a missing
submodule, an unexpected Git root or an unreadable source fails closed. Generated
output belongs under build/, and must be generated before packaging/signing. This
is source evidence, not a signature or proof that a compiler used a snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUBMODULES = {
    "": ("android", "frontend_cybergothic", "harness_workbench", "packaging"),
    "android": ("app/src/main/cpp/llama.cpp",),
}
# These are generated/runtime directories, never exclusions for tracked files.
GENERATED_DIRS = {
    ".git", ".gradle", ".kotlin", ".cxx", "__pycache__", "node_modules",
    "build", "dist", ".venv", ".venv-test", ".venv-packaging", "venv",
    "models", "models_pc", "models_android", "runtime", "chat_history", "test-results", "logs",
}
EVIDENCE_SUFFIXES = {".log", ".out", ".err", ".xml", ".json", ".patch", ".pyc"}
# One legacy source snapshot is evidence, not an Android compilation input.
LEGACY_EVIDENCE = {"worker-reconnect-evidence/tcp_comm.recovered.py"}
# Non-build documents with local storage/key-shaped filenames are represented by
# their tracked blob identity. Never follow model storage or open a .key document.
# These exceptions cannot conceal production sources; modifications still dirty Git.
NON_BUILD_GIT_ONLY = {
    "": {"models/README.md"},
    "android/app/src/main/cpp/llama.cpp": {"docs/development/llama-star/idea-arch.key"},
}
IGNORED_SOURCE_DIRS = {
    "": ("src", "scripts", "tests"),
    "android": ("app/src",),
    "packaging": ("packaging",),
    "frontend_cybergothic": ("frontend_cybergothic/src",),
    "harness_workbench": ("src", "mcp_server", "image", "ui_react/src"),
}


class ProvenanceError(ValueError):
    """The source identity cannot be established safely."""


def _git(root: Path, *args: str, missing_ok: bool = False, input_data: bytes | None = None) -> bytes:
    # An inherited GIT_DIR/GIT_WORK_TREE/index/config must not select other sources.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            input=input_data, capture_output=True, env=env, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError("Git unavailable or timed out while observing build sources") from exc
    if result.returncode == 1 and missing_ok:
        return b""
    if result.returncode:
        # Do not echo Git stderr/config/remotes: they can contain credentials.
        raise ProvenanceError(f"Git source identity/read failed ({args[0]})")
    return result.stdout


def _paths(data: bytes) -> list[str]:
    return [os.fsdecode(part) for part in data.split(b"\0") if part]


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not value:
        raise ProvenanceError("Unsafe source path in Git inventory")
    return value


def _sensitive(value: str) -> bool:
    path = PurePosixPath(value)
    name = path.name.lower()
    return (
        ".signing-keys" in path.parts
        or name in {".env", "keystore.properties", "keystore-lite.properties", "node_secrets.json"}
        or (name.startswith(".env.") and name not in {".env.example", ".env.sample", ".env.template"})
        or path.suffix.lower() in {".key", ".pem", ".jks", ".keystore", ".p12", ".pfx"}
    )


def _generated(value: str) -> bool:
    path = PurePosixPath(value)
    return any(part in GENERATED_DIRS or part.startswith(".venv-") for part in path.parts[:-1])


def _evidence(component: str, value: str) -> bool:
    path = PurePosixPath(value)
    if _generated(value):
        return True
    if path.suffix.lower() in {".log", ".err", ".out"}:
        return True
    if component == "" and value.startswith("tests/simulation/results/"):
        return True
    if component == "android" and value in LEGACY_EVIDENCE:
        return True
    return (
        component == "android" and path.parts[0] == "worker-reconnect-evidence"
        and path.suffix.lower() in EVIDENCE_SUFFIXES
    )


def _identity(root: Path) -> str:
    if not (root / ".git").exists():
        raise ProvenanceError("Git checkout required; source exports cannot prove build identity")
    actual_root = Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
    if actual_root != root:
        raise ProvenanceError("Wrong Git root; main checkout and each submodule must be distinct repositories")
    revision = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ProvenanceError("Git did not return a full commit identity")
    return revision


def _inventory(root: Path) -> tuple[dict[str, tuple[str, str]], dict[str, str], dict[str, str]]:
    tracked: dict[str, tuple[str, str]] = {}
    links: dict[str, str] = {}
    indexed_links: dict[str, str] = {}
    for entry in _git(root, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split()
        path = _safe_relative(os.fsdecode(name))
        if mode == "160000" and kind == "commit":
            links[path] = oid
        elif kind == "blob":
            tracked[path] = (mode, oid)
        else:
            raise ProvenanceError("Unsupported Git source entry")
    # Include staged additions and preserve staged deletions in the HEAD inventory.
    for entry in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, oid, stage = header.decode("ascii").split()
        if stage != "0":
            raise ProvenanceError("Unmerged source index cannot prove build identity")
        if mode == "160000":
            indexed_links[_safe_relative(os.fsdecode(name))] = oid
        else:
            tracked[_safe_relative(os.fsdecode(name))] = (mode, oid)
    return tracked, links, indexed_links


def _file_record(root: Path, value: str, mode: str, expected_oid: str = "",
                 normalize_crlf: bool = False) -> dict[str, Any]:
    path = root / value
    if _sensitive(value):
        if mode != "untracked":
            raise ProvenanceError("Sensitive file is tracked; refusing to read it for provenance")
        return {"path": value, "mode": mode, "kind": "sensitive-unread"}
    # Git-tracked model manifests may live behind a Windows directory junction.
    # Hash only the inventoried file bytes; never enumerate a resolved target.
    try:
        before = path.lstat()
    except FileNotFoundError:
        return {"path": value, "mode": mode, "kind": "missing"}
    if stat.S_ISLNK(before.st_mode):
        # Hash the link text; never read its target (which could be a secret).
        data = os.fsencode(os.readlink(path))
        digest, kind = hashlib.sha256(data).hexdigest(), "symlink"
    elif stat.S_ISREG(before.st_mode):
        with path.open("rb") as stream:
            data = stream.read()
        digest, kind = hashlib.sha256(data).hexdigest(), "file"
    else:
        raise ProvenanceError("Source is not a regular file or symbolic link")
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ProvenanceError("Source changed while hashing; retry with a stable checkout")
    record = {"path": value, "mode": mode, "kind": kind, "sha256": digest}
    if expected_oid:
        algorithm = "sha256" if len(expected_oid) == 64 else "sha1"

        def blob_oid(content: bytes) -> str:
            header = b"blob " + str(len(content)).encode("ascii") + b"\0"
            return hashlib.new(algorithm, header + content).hexdigest()

        matches = blob_oid(data) == expected_oid
        if not matches and normalize_crlf and b"\0" not in data:
            matches = blob_oid(data.replace(b"\r\n", b"\n")) == expected_oid
        # Internal comparison bypasses Git stat caches/assume-unchanged flags.
        record["matches_index_blob"] = matches
    return record


def _canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def _observe(root: Path) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    tree: list[dict[str, Any]] = []

    def visit(component: str, expected_revision: str | None = None) -> None:
        checkout = (root / component).resolve()
        if not checkout.is_relative_to(root):
            raise ProvenanceError("Submodule checkout escapes the main checkout")
        revision = _identity(checkout)
        tracked, links, indexed_links = _inventory(checkout)
        expected_links = SUBMODULES.get(component, ())
        if set(links) != set(expected_links) or set(indexed_links) != set(expected_links):
            raise ProvenanceError("Fixed submodule graph does not match this checkout")
        if component == "" and not {"release-contract.json", ".gitmodules"}.issubset(tracked):
            raise ProvenanceError("Wrong main source root; QLH release contract and submodule graph required")
        git_only = NON_BUILD_GIT_ONLY.get(component, set())
        if any(_sensitive(value) and value not in git_only for value in tracked):
            # Refuse before diff/attribute commands could inspect tracked secrets.
            raise ProvenanceError("Sensitive file is tracked; refusing to read it for provenance")
        changed = _paths(_git(checkout, "diff", "--no-ext-diff", "--no-textconv", "--name-only",
                              "--ignore-submodules=all", "-z", "HEAD", "--"))
        changed.extend(path for path in links if links[path] != indexed_links[path])
        untracked = set(_paths(_git(checkout, "ls-files", "--others", "--exclude-standard", "-z")))
        # Broad .gitignore rules must not conceal actual untracked production code.
        source_dirs = IGNORED_SOURCE_DIRS.get(component, ())
        if source_dirs:
            untracked.update(_paths(_git(checkout, "ls-files", "--others", "--ignored",
                                         "--exclude-standard", "-z", "--", *source_dirs)))
        untracked = {value for value in untracked if not _evidence(component, value)}
        autocrlf = _git(checkout, "config", "--get", "core.autocrlf", missing_ok=True).strip().lower()
        # check-attr reads attributes, not source bytes or key-shaped documents.
        attribute_paths = b"\0".join(os.fsencode(value) for value in sorted(tracked)) + b"\0"
        attribute_parts = _git(checkout, "check-attr", "-z", "--stdin", "text", "eol",
                               input_data=attribute_paths).split(b"\0")
        eol_attributes: dict[str, dict[str, str]] = {}
        for offset in range(0, len(attribute_parts) - 1, 3):
            name, attribute, value = attribute_parts[offset:offset + 3]
            eol_attributes.setdefault(os.fsdecode(name), {})[attribute.decode("ascii")] = value.decode("ascii")
        files = []
        for value in sorted(set(tracked) | untracked):
            mode, oid = tracked.get(value, ("untracked", ""))
            if value in git_only and oid:
                files.append({"path": value, "mode": mode, "kind": "non-build-git-blob", "git_blob_oid": oid})
            else:
                attributes = eol_attributes.get(value, {})
                normalize = attributes.get("text") != "unset" and (
                    autocrlf in {b"true", b"input"} or attributes.get("text") in {"set", "auto"}
                    or attributes.get("eol") in {"lf", "crlf"}
                )
                record = _file_record(checkout, value, mode, oid, normalize)
                if record.pop("matches_index_blob", True) is False or (oid and record["kind"] == "missing"):
                    changed.append(value)
                files.append(record)
        drift = expected_revision is not None and revision != expected_revision
        components[component or "."] = {
            "source_revision": revision,
            "expected_revision": expected_revision,
            "source_dirty": bool(changed or untracked or drift),
            "revision_matches_pin": not drift,
            "tracked_changes": sorted(set(changed)),
            "untracked_sources": sorted(untracked),
            "source_tree_sha256": _canonical_hash(files),
            "source_file_count": len(files),
            "non_build_git_only_files": sorted(set(tracked) & git_only),
        }
        tree.append({"component": component or ".", "files": files})
        for child in expected_links:
            visit(f"{component}/{child}" if component else child, links[child])

    visit("")
    return {
        "schema_version": 1,
        "source_revision": components["."]["source_revision"],
        "source_dirty": any(value["source_dirty"] for value in components.values()),
        "source_tree_sha256": _canonical_hash(tree),
        "source_tree_hash_algorithm": "sha256-canonical-json-source-bytes-and-nonbuild-git-blobs-v1",
        "components": components,
        "submodules": {path: value for path, value in components.items() if path != "."},
    }


def collect_build_info(root: Path = ROOT) -> dict[str, Any]:
    """Return fresh source facts; detect concurrent source/index/identity changes."""
    root = Path(root).resolve()
    try:
        first = _observe(root)
        second = _observe(root)
    except (OSError, UnicodeError) as exc:
        raise ProvenanceError("Cannot read build source inventory safely") from exc
    if first != second:
        raise ProvenanceError("Build sources changed during provenance observation; retry when stable")
    return second


def write_build_info(root: Path, output: Path) -> dict[str, Any]:
    """Regenerate atomically under a build directory; never relabel existing dist."""
    root, output = Path(root).resolve(), Path(output).resolve()
    try:
        relative = output.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError("build-info output must be inside the main checkout") from exc
    if "build" not in relative.parts[:-1] or "dist" in relative.parts or output.name != "build-info.json":
        raise ProvenanceError("build-info output must be named build-info.json under build/, outside dist")
    # Observe before writing; a failure cannot be presented as successful regeneration.
    info = collect_build_info(root)
    data = (json.dumps(info, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("ascii")
    if len(data) > 65536:
        raise ProvenanceError("Build provenance exceeds the APK metadata size limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".build-info-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return info


def _snapshot_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError("build-info snapshot invalid: duplicate JSON key")
        result[key] = value
    return result


def _snapshot_number(value: str) -> None:
    # Generated provenance has integers and booleans only. Reject both JSON's
    # nonstandard NaN/Infinity constants and floats (including 1e999 overflow).
    raise ProvenanceError("build-info snapshot invalid: non-integer or nonfinite JSON number")


def _read_snapshot(snapshot: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        with snapshot.open("rb") as stream:
            raw = stream.read(65537)
    except FileNotFoundError as exc:
        raise ProvenanceError("build-info snapshot missing; generation is required before packaging") from exc
    except OSError as exc:
        raise ProvenanceError("build-info snapshot unreadable") from exc
    if not raw or len(raw) > 65536:
        raise ProvenanceError("build-info snapshot invalid: empty or exceeds 64 KiB")
    try:
        info = json.loads(raw.decode("utf-8"), object_pairs_hook=_snapshot_object,
                          parse_constant=_snapshot_number, parse_float=_snapshot_number)
    except ProvenanceError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProvenanceError("build-info snapshot invalid: malformed UTF-8 JSON") from exc
    if type(info) is not dict:
        raise ProvenanceError("build-info snapshot invalid: root must be an object")
    return raw, info


def _snapshot_differences(saved: Any, current: Any, field: str = "$") -> list[str]:
    # Python considers False == 0 and True == 1; provenance must not do so.
    if type(saved) is not type(current):
        return [field + " (type)"]
    if isinstance(current, dict):
        differences = []
        for key in sorted(saved.keys() | current.keys()):
            location = field + "." + key
            if key not in saved or key not in current:
                differences.append(location + " (missing/extra field)")
            else:
                differences.extend(_snapshot_differences(saved[key], current[key], location))
        return differences
    if isinstance(current, list):
        if len(saved) != len(current):
            return [field + " (length)"]
        return [location for index, (left, right) in enumerate(zip(saved, current))
                for location in _snapshot_differences(left, right, f"{field}[{index}]")]
    return [] if saved == current else [field]


def verify_build_info(root: Path, snapshot: Path) -> dict[str, Any]:
    """Read-only package-time check; never regenerate/relabel the original snapshot.

    Stable dirty development sources may package, retaining their original dirty
    fact for release rejection. ANY fact changed since generation aborts packaging.
    """
    snapshot = Path(snapshot).resolve()
    original, saved = _read_snapshot(snapshot)
    current = collect_build_info(root)
    differences = _snapshot_differences(saved, current)
    if differences:
        raise ProvenanceError("build-info snapshot mismatch: " + ", ".join(differences[:12])
                              + "; sources or metadata changed since generation; original snapshot preserved")
    # The snapshot itself must remain stable while the source inventory is read.
    if _read_snapshot(snapshot)[0] != original:
        raise ProvenanceError("build-info snapshot changed during verification; packaging refused")
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Exact main QLH Git checkout, never Android's Git root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="Generated build-info.json path under a build/ directory")
    mode.add_argument("--print", action="store_true", dest="print_info", help="Read-only provenance JSON (Gradle input)")
    mode.add_argument("--verify", type=Path, metavar="SNAPSHOT", help="Verify original build-info without writing it")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.verify or args.output or root / "build" / "provenance" / "build-info.json"
    try:
        if args.verify is not None:
            info = verify_build_info(root, args.verify)
        else:
            info = collect_build_info(root) if args.print_info else write_build_info(root, output)
    except ProvenanceError as exc:
        print(f"build_provenance: {exc}", file=sys.stderr)
        return 2
    result = info if args.print_info else {
        "schema_version": info["schema_version"], "source_revision": info["source_revision"],
        "source_dirty": info["source_dirty"], "source_tree_sha256": info["source_tree_sha256"],
        "output": str(output.resolve()),
    }
    if args.verify is not None:
        result["verified"] = True
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
