"""Check or apply the QLH llama.cpp version and patch contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path(__file__).with_name("llama_quantize.lock.json")


class SyncError(RuntimeError):
    """Raised when the checked-out llama.cpp cannot satisfy the contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(
    source: Path, *args: str, check: bool = True, env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        env=env,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise SyncError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.rstrip("\r\n")


def _android_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().absolute()
    configured = os.environ.get("QLH_ANDROID_ROOT")
    if configured:
        return Path(configured).expanduser().absolute()
    return ROOT / "android"


def _contract(lock_path: Path = LOCK_PATH) -> tuple[str, list[dict[str, Any]]]:
    """Read the lock and return (revision, patches).

    ★ 2026-09-20：契约由「**恰好一个**补丁」放宽为「**一个或多个**补丁」。
    每个补丁条目声明：
      * ``path``    —— 补丁文件（相对仓库根）
      * ``targets`` —— 它改动的文件（相对 submodule 根）；兼容旧字段 ``target``（单数）
      * ``marker``  —— 仅供诊断的说明或标记串，不参与补丁状态判定
      * ``sha256``  —— 补丁文件的摘要（防止内容漂移）
    """
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError("cannot read llama.cpp lock") from exc
    if not isinstance(lock, dict):
        raise SyncError("llama.cpp lock must be an object")
    upstream = lock.get("upstream")
    patch_list = lock.get("patches")
    if not isinstance(upstream, dict) or not isinstance(patch_list, list) or not patch_list:
        raise SyncError("llama.cpp lock must contain one upstream and at least one patch contract")
    revision = upstream.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise SyncError("invalid llama.cpp revision in lock")
    # ★ 2026-09-20：允许 lock 声明一个「本地补丁 commit」（submodule HEAD 应当是它）。
    #   未声明时回落到 upstream.revision（即「补丁只在工作树、不提交」的形态）。
    local_revision = lock.get("revision_local")
    if local_revision is not None and (
        not isinstance(local_revision, str) or len(local_revision) != 40
    ):
        raise SyncError("invalid llama.cpp local revision in lock")
    expected_revision = local_revision or revision

    patches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in patch_list:
        if not isinstance(entry, dict):
            raise SyncError("invalid llama.cpp patch contract")
        if any(not isinstance(entry.get(key), str) for key in ("path", "marker", "sha256")):
            raise SyncError("invalid llama.cpp patch contract")
        targets = _patch_targets(entry)
        if not targets:
            raise SyncError("llama.cpp patch contract needs target or targets")
        digest = entry["sha256"].lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SyncError("invalid llama.cpp patch SHA-256 in lock")
        patch_id = entry.get("id") or entry["path"]
        if patch_id in seen_ids:
            raise SyncError(f"duplicate llama.cpp patch id in lock: {patch_id}")
        seen_ids.add(patch_id)
        patches.append({**entry, "id": patch_id, "targets": targets})
    return expected_revision, patches


def _patch_targets(entry: dict[str, Any]) -> list[str]:
    """Normalize a patch entry's targets, accepting both `targets` and legacy `target`."""
    raw = entry.get("targets")
    if raw is None:
        single = entry.get("target")
        raw = [single] if isinstance(single, str) and single else []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item]


def _safe_relative(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SyncError(f"contract path escapes repository: {value}") from exc
    return candidate


def _status_paths(source: Path) -> set[str]:
    output = _git(source, "status", "--porcelain", "-z", "--untracked-files=all")
    paths: set[str] = set()
    records = iter(output.split("\0"))
    for record in records:
        if not record:
            continue
        paths.add(record[3:])
        if "R" in record[:2] or "C" in record[:2]:
            paths.add(next(records))
    return paths


def _patch_check(
    source: Path, patch_path: Path, *, reverse: bool = False,
    cached: bool = False, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    flags = (["--reverse"] if reverse else []) + (["--cached"] if cached else [])
    return subprocess.run(
        ["git", "-C", str(source), "apply", *flags, "--check", str(patch_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, env=env,
    )


def _check_worktree_diff(source: Path, applied: list[dict[str, Any]], targets: set[str]) -> None:
    """Compare dirty targets with HEAD plus managed patches using disposable Git state."""
    with tempfile.TemporaryDirectory(prefix="qlh-llama-sync-") as directory:
        scratch = Path(directory)
        objects = scratch / "objects"
        objects.mkdir()
        # Keep the real checkout, index and object database untouched, including
        # in --dry-run. Git still supplies its normal content/line-ending filters.
        env = {
            **os.environ,
            "GIT_INDEX_FILE": str(scratch / "index"),
            "GIT_OBJECT_DIRECTORY": str(objects),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": _git(
                source, "rev-parse", "--path-format=absolute", "--git-path", "objects",
            ),
        }
        _git(source, "read-tree", "HEAD", env=env)
        for entry in applied:
            if _patch_check(source, entry["patch_path"], reverse=True, cached=True, env=env).returncode == 0:
                continue  # The pinned HEAD already contains this complete patch.
            if _patch_check(source, entry["patch_path"], cached=True, env=env).returncode != 0:
                raise SyncError(f"llama.cpp unmanaged diff for patch {entry['id']}")
            _git(source, "apply", "--cached", str(entry["patch_path"]), env=env)
        _git(
            source, "diff", "--quiet", "--no-ext-diff", "--", *sorted(targets), env=env,
        )


def inspect_sync(
    *,
    android_root: Path | None = None,
    lock_path: Path = LOCK_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate the llama.cpp checkout against **every** patch in the contract.

    ★ 2026-09-20：由「单补丁」改为「多补丁」。每个补丁独立判定 applied /
    ready_to_apply，并逐个校验：
      * 补丁文件存在且 sha256 与 lock 一致；
      * submodule 的 HEAD 等于 lock 的 revision；
      * **worktree 中允许出现的改动，恰好是所有补丁 targets 的并集**
        （原先要求「只有唯一 target 有改动」⇒ 多补丁必然误报 unrelated changes）；
      * 仅按 apply / reverse --check 判定完整补丁状态，marker 只用于诊断；
      * dirty targets 的内容必须恰好匹配 HEAD 加受管补丁，拒绝混入其他 diff。
    """
    revision, patch_contracts = _contract(lock_path)
    source = _android_root(android_root) / "app" / "src" / "main" / "cpp" / "llama.cpp"
    if not source.is_dir():
        raise SyncError("Android llama.cpp submodule is missing")

    checked_revision = _git(source, "rev-parse", "HEAD")
    if checked_revision != revision:
        raise SyncError(
            f"llama.cpp revision mismatch: expected {revision}, got {checked_revision}"
        )

    # 先把所有补丁的路径/target 解析并校验，再统一比对 worktree 改动集合。
    entries: list[dict[str, Any]] = []
    allowed_paths: set[str] = set()
    for contract in patch_contracts:
        patch_path = _safe_relative(ROOT, contract["path"])
        if not patch_path.is_file():
            raise SyncError(f"managed llama.cpp patch is missing: {contract['path']}")
        if _sha256(patch_path) != contract["sha256"].lower():
            raise SyncError(
                f"managed llama.cpp patch SHA-256 does not match the lock: {contract['path']}"
            )
        relative_targets: list[str] = []
        for raw_target in contract["targets"]:
            target = (source / raw_target).resolve()
            try:
                target.relative_to(source.resolve())
            except ValueError as exc:
                raise SyncError("llama.cpp patch target escapes the submodule") from exc
            if not target.is_file():
                raise SyncError(
                    f"llama.cpp patch target is missing: {raw_target} (patch {contract['id']})"
                )
            relative_targets.append(target.relative_to(source).as_posix())
        stats = _git(source, "apply", "--numstat", "-z", str(patch_path))
        actual_targets = set()
        for record in stats.split("\0"):
            if not record:
                continue
            fields = record.split("\t", 2)
            if len(fields) != 3:
                raise SyncError(f"llama.cpp patch has unsupported targets: {contract['id']}")
            actual_targets.add(fields[2])
        if actual_targets != set(relative_targets):
            raise SyncError(f"llama.cpp patch targets do not match the lock: {contract['id']}")
        allowed_paths.update(relative_targets)
        entries.append({**contract, "patch_path": patch_path, "relative_targets": relative_targets})

    status_paths = _status_paths(source)
    unexpected = sorted(path for path in status_paths if path not in allowed_paths)
    if unexpected:
        raise SyncError(f"llama.cpp worktree has unrelated changes: {', '.join(unexpected[:8])}")

    results: list[dict[str, Any]] = []
    for entry in entries:
        reverse = _patch_check(source, entry["patch_path"], reverse=True)
        forward = _patch_check(source, entry["patch_path"])
        if reverse.returncode == 0 and forward.returncode != 0:
            dirty = any(rel in status_paths for rel in entry["relative_targets"])
            state = "applied" if dirty else "applied_committed"
        elif forward.returncode == 0 and reverse.returncode != 0:
            state = "ready_to_apply"
        else:
            marker_present = any(
                entry["marker"] in (source / rel).read_text(encoding="utf-8", errors="replace")
                for rel in entry["relative_targets"]
            )
            detail = (
                f"forward: {(forward.stderr or forward.stdout).strip()[:500]}; "
                f"reverse: {(reverse.stderr or reverse.stdout).strip()[:500]}"
            )
            problem = (
                "has ambiguous apply/reverse checks"
                if forward.returncode == reverse.returncode == 0
                else "is neither applicable nor reversible"
            )
            raise SyncError(
                f"llama.cpp patch {entry['id']} {problem} "
                f"(marker_present={marker_present}): {detail}"
            )
        results.append(
            {
                "id": entry["id"],
                "patch": entry["path"],
                "patch_sha256": entry["sha256"].lower(),
                "targets": entry["relative_targets"],
                "state": state,
            }
        )

    if status_paths:
        applied = [
            entry for entry, result in zip(entries, results)
            if result["state"].startswith("applied")
        ]
        try:
            _check_worktree_diff(source, applied, allowed_paths)
        except SyncError as exc:
            raise SyncError(f"llama.cpp worktree has unmanaged diff: {exc}") from exc

    # 顶层字段保留首个补丁的信息，兼容既有消费方；新增 `patches` 给出全量。
    first = results[0]
    result: dict[str, Any] = {
        "revision": checked_revision,
        "patch": first["patch"],
        "patch_sha256": first["patch_sha256"],
        "target": first["targets"][0],
        "state": ("applied" if all(item["state"].startswith("applied") for item in results)
                  else "ready_to_apply"),
        "patches": results,
        "dry_run": dry_run,
    }
    if dry_run:
        pending = [item["id"] for item in results if item["state"] == "ready_to_apply"]
        if pending:
            result["action"] = "would_apply"
            result["pending"] = pending
    return result


def apply_sync(*, android_root: Path | None = None, lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    result = inspect_sync(android_root=android_root, lock_path=lock_path, dry_run=False)
    pending = [item for item in result["patches"] if item["state"] == "ready_to_apply"]
    if not pending:
        result["action"] = "already_applied"
        return result
    source = _android_root(android_root) / "app" / "src" / "main" / "cpp" / "llama.cpp"
    applied_ids: list[str] = []
    for item in pending:
        _git(source, "apply", str(_safe_relative(ROOT, item["patch"])))
        applied_ids.append(item["id"])
    result = inspect_sync(android_root=android_root, lock_path=lock_path, dry_run=False)
    result["action"] = "applied"
    result["applied"] = applied_ids
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android-root", type=Path, help="path to the qlh-android checkout")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    parser.add_argument("--apply", action="store_true", help="apply the managed patch")
    args = parser.parse_args(argv)
    if args.dry_run and args.apply:
        parser.error("--dry-run and --apply are mutually exclusive")
    try:
        result = (
            apply_sync(android_root=args.android_root)
            if args.apply
            else inspect_sync(android_root=args.android_root, dry_run=args.dry_run)
        )
    except SyncError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps({"ok": True, **result}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
