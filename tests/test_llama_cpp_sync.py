from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.model_tools import sync_llama_cpp as sync


def _write_lock(tmp_path: Path, patch_path: str, digest: str) -> Path:
    lock = {
        "upstream": {"revision": "a" * 40},
        "patches": [{
            "path": patch_path,
            "target": "conversion/base.py",
            "marker": "QLH patch marker",
            "sha256": digest,
        }],
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def test_contract_rejects_malformed_patch_digest(tmp_path: Path):
    lock = _write_lock(tmp_path, "patch.diff", "g" * 64)

    with pytest.raises(sync.SyncError, match="SHA-256"):
        sync._contract(lock)


def test_contract_requires_at_least_one_patch(tmp_path: Path):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"upstream": {"revision": "a" * 40}, "patches": []}), encoding="utf-8")

    with pytest.raises(sync.SyncError, match="one patch"):
        sync._contract(lock)


@pytest.mark.skipif(sys.platform != "win32", reason="基准路径用 Windows 盘符（G:/repo），Linux 上会退化成相对路径")
def test_safe_relative_rejects_escape():
    with pytest.raises(sync.SyncError, match="escapes"):
        sync._safe_relative(Path("G:/repo"), "../outside.patch")


MARKER = "QLH patch marker"
DESCRIPTION = "★ QLH 2026-09-20（2026-09-23 新增部分用 ★ QLH 2026-09-23）"


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args], check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    return result.stdout.strip()


def _write_sources(source: Path, contents: dict[str, str]) -> None:
    for name, content in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def _update_lock(checkout, **changes) -> None:
    data = json.loads(checkout.lock.read_text(encoding="utf-8"))
    data.update(changes)
    checkout.lock.write_text(json.dumps(data), encoding="utf-8")


def _commit(checkout, *, local: bool = True) -> str:
    _git(checkout.source, "add", ".")
    _git(checkout.source, "commit", "-qm", "fixture revision")
    revision = _git(checkout.source, "rev-parse", "HEAD")
    if local:
        _update_lock(checkout, revision_local=revision)
    else:
        _update_lock(checkout, upstream={"revision": revision})
    return revision


@pytest.fixture
def patch_checkout(tmp_path: Path, monkeypatch):
    # Real two-target patch, with paths containing spaces and enough unchanged
    # lines to place an unmanaged edit beyond all of the patch's contexts.
    root = tmp_path / "project with spaces"
    source = root / "android/app/src/main/cpp/llama.cpp"
    source.mkdir(parents=True)
    _git(source, "init", "-q")
    for key, value in (
        ("user.name", "Sync test"), ("user.email", "sync@example.invalid"),
        ("core.autocrlf", "false"), ("commit.gpgsign", "false"),
    ):
        _git(source, "config", key, value)
    before = {
        "conversion/base.py": "enabled = False\n" + "# context\n" * 20,
        "src/model.cpp": "int enabled = 0;\n" + "// context\n" * 20,
    }
    after = {
        "conversion/base.py": f"# {MARKER}\nenabled = True\n" + "# context\n" * 20,
        "src/model.cpp": "int enabled = 1;\n" + "// context\n" * 20,
    }
    _write_sources(source, before)
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "upstream fixture")
    revision = _git(source, "rev-parse", "HEAD")
    _write_sources(source, after)
    patch = root / "managed patch.diff"
    diff = subprocess.run(
        ["git", "-C", str(source), "diff", "--binary"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    patch.write_bytes(diff)
    _write_sources(source, before)
    lock = root / "lock.json"
    lock.write_text(json.dumps({
        "upstream": {"revision": revision},
        "patches": [{
            "id": "fixture", "path": patch.name, "targets": list(before),
            "marker": DESCRIPTION, "sha256": sync._sha256(patch),
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(sync, "ROOT", root)
    return SimpleNamespace(
        root=root, source=source, patch=patch, lock=lock, before=before,
        after=after, revision=revision, android=root / "android",
    )


def _inspect(checkout):
    return sync.inspect_sync(android_root=checkout.android, lock_path=checkout.lock, dry_run=True)


@pytest.mark.parametrize("staged", [False, True])
def test_missing_marker_accepts_complete_reversible_patch(patch_checkout, staged):
    checkout = patch_checkout
    _write_sources(checkout.source, checkout.after)
    if staged:
        _git(checkout.source, "add", ".")
    status = _git(checkout.source, "status", "--porcelain")
    index = (checkout.source / ".git/index").read_bytes()
    objects = _git(checkout.source, "count-objects", "-v")

    result = _inspect(checkout)

    assert result["state"] == "applied"
    assert result["patches"][0]["state"] == "applied"
    assert "pending" not in result and "action" not in result
    assert _git(checkout.source, "status", "--porcelain") == status
    assert (checkout.source / ".git/index").read_bytes() == index
    assert _git(checkout.source, "count-objects", "-v") == objects
    assert all((checkout.source / name).read_text() == text for name, text in checkout.after.items())


def test_clean_pinned_committed_patch_requires_and_passes_reverse_check(patch_checkout):
    checkout = patch_checkout
    _write_sources(checkout.source, checkout.after)
    revision = _commit(checkout)

    result = _inspect(checkout)

    assert result["revision"] == revision
    assert result["state"] == "applied"
    assert result["patches"][0]["state"] == "applied_committed"
    assert _git(checkout.source, "status", "--porcelain") == ""
    assert sync.apply_sync(android_root=checkout.android, lock_path=checkout.lock)["action"] == "already_applied"


@pytest.mark.parametrize("committed", [False, True])
def test_marker_present_does_not_accept_partial_patch(patch_checkout, committed):
    checkout = patch_checkout
    _write_sources(checkout.source, {"conversion/base.py": checkout.after["conversion/base.py"]})
    data = json.loads(checkout.lock.read_text())
    data["patches"][0]["marker"] = MARKER
    _update_lock(checkout, patches=data["patches"])
    if committed:
        _commit(checkout)

    with pytest.raises(sync.SyncError, match="neither.*appl|not reversible"):
        _inspect(checkout)


@pytest.mark.parametrize("marker_present", [False, True])
def test_forward_applicable_patch_is_pending_regardless_of_marker(patch_checkout, marker_present):
    checkout = patch_checkout
    if marker_present:
        content = checkout.before["conversion/base.py"] + f"# {MARKER}\n"
        _write_sources(checkout.source, {"conversion/base.py": content})
        _commit(checkout, local=False)
        data = json.loads(checkout.lock.read_text())
        data["patches"][0]["marker"] = MARKER
        _update_lock(checkout, patches=data["patches"])

    result = _inspect(checkout)

    assert result["state"] == "ready_to_apply"
    assert result["patches"][0]["state"] == "ready_to_apply"
    assert result["action"] == "would_apply"
    assert result["pending"] == ["fixture"]
    assert _git(checkout.source, "status", "--porcelain") == ""


@pytest.mark.parametrize("committed", [False, True])
def test_neither_forward_nor_reverse_rejects_wrong_diff(patch_checkout, committed):
    checkout = patch_checkout
    _write_sources(checkout.source, checkout.after)
    _write_sources(checkout.source, {"src/model.cpp": "int enabled = 42;\n" + "// context\n" * 20})
    if committed:
        _commit(checkout)

    with pytest.raises(sync.SyncError, match="neither.*appl"):
        _inspect(checkout)


def test_revision_must_match_pin_even_if_patch_is_reversible(patch_checkout):
    checkout = patch_checkout
    _write_sources(checkout.source, checkout.after)
    _update_lock(checkout, revision_local="f" * 40)

    with pytest.raises(sync.SyncError, match="revision mismatch"):
        _inspect(checkout)


def test_raw_patch_digest_must_match(patch_checkout):
    checkout = patch_checkout
    checkout.patch.write_bytes(checkout.patch.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(sync.SyncError, match="SHA-256 does not match"):
        _inspect(checkout)


@pytest.mark.parametrize("target", ["missing.py", "../outside.py"])
def test_missing_or_escaping_target_is_rejected(patch_checkout, target):
    checkout = patch_checkout
    data = json.loads(checkout.lock.read_text())
    data["patches"][0]["targets"] = [target]
    _update_lock(checkout, patches=data["patches"])

    with pytest.raises(sync.SyncError, match="target.*(missing|escapes)"):
        _inspect(checkout)


@pytest.mark.parametrize("tracked", [False, True])
def test_unmanaged_file_is_rejected(patch_checkout, tracked):
    checkout = patch_checkout
    extra = checkout.source / "unmanaged.txt"
    extra.write_text("baseline\n")
    if tracked:
        _commit(checkout, local=False)
    extra.write_text("unmanaged change\n")

    with pytest.raises(sync.SyncError, match="unrelated changes"):
        _inspect(checkout)


@pytest.mark.parametrize("state", ["pending", "applied", "committed"])
def test_extra_diff_in_managed_target_is_rejected_even_when_patch_checks_pass(patch_checkout, state):
    checkout = patch_checkout
    if state != "pending":
        _write_sources(checkout.source, checkout.after)
    if state == "committed":
        _commit(checkout)
    target = checkout.source / "conversion/base.py"
    with target.open("a", encoding="utf-8") as stream:
        stream.write("# unmanaged extra edit beyond patch context\n")

    with pytest.raises(sync.SyncError, match="unmanaged.*diff|does not match.*patch"):
        _inspect(checkout)


def test_multiple_patches_preserve_mixed_states_and_apply_only_pending_in_fixture(patch_checkout):
    checkout = patch_checkout
    _write_sources(checkout.source, checkout.after)
    data = json.loads(checkout.lock.read_text())
    data["patches"] = []
    for number, target in enumerate(checkout.before):
        patch = checkout.root / f"patch-{number}.diff"
        patch.write_bytes(subprocess.run(
            ["git", "-C", str(checkout.source), "diff", "--binary", "--", target],
            check=True, capture_output=True, timeout=30,
        ).stdout)
        data["patches"].append({
            "id": f"patch-{number}", "path": patch.name, "target": target,
            "marker": DESCRIPTION, "sha256": sync._sha256(patch),
        })
    _update_lock(checkout, patches=data["patches"])
    _write_sources(checkout.source, {"src/model.cpp": checkout.before["src/model.cpp"]})

    result = _inspect(checkout)

    assert [item["state"] for item in result["patches"]] == ["applied", "ready_to_apply"]
    assert result["pending"] == ["patch-1"]
    applied = sync.apply_sync(android_root=checkout.android, lock_path=checkout.lock)
    assert applied["action"] == "applied"
    assert applied["applied"] == ["patch-1"]
    assert all(item["state"] == "applied" for item in applied["patches"])
    assert sync.apply_sync(android_root=checkout.android, lock_path=checkout.lock)["action"] == "already_applied"


@pytest.mark.parametrize("declared", [["conversion/base.py"], ["conversion/base.py", "src/model.cpp", "extra.py"]])
def test_patch_targets_must_match_declared_scope(patch_checkout, declared):
    checkout = patch_checkout
    (checkout.source / "extra.py").write_text("# extra allowed target\n")
    _commit(checkout, local=False)
    data = json.loads(checkout.lock.read_text())
    data["patches"][0]["targets"] = declared
    _update_lock(checkout, patches=data["patches"])

    with pytest.raises(sync.SyncError, match="targets do not match"):
        _inspect(checkout)


def test_both_forward_and_reverse_applicable_is_ambiguous(patch_checkout):
    checkout = patch_checkout
    # Interior hunks can match an original block and an already-patched block
    # in the same file. Both successful directions must not imply a safe state.
    before = {name: "# prefix\n" * 10 + value for name, value in checkout.before.items()}
    after = {name: "# prefix\n" * 10 + value for name, value in checkout.after.items()}
    _write_sources(checkout.source, before)
    _commit(checkout, local=False)
    _write_sources(checkout.source, after)
    checkout.patch.write_bytes(subprocess.run(
        ["git", "-C", str(checkout.source), "diff", "--binary"],
        check=True, capture_output=True, timeout=30,
    ).stdout)
    data = json.loads(checkout.lock.read_text())
    data["patches"][0]["sha256"] = sync._sha256(checkout.patch)
    _update_lock(checkout, patches=data["patches"])
    _write_sources(checkout.source, {name: after[name] + before[name] for name in before})
    _commit(checkout)
    _git(checkout.source, "apply", "--check", str(checkout.patch))
    _git(checkout.source, "apply", "--reverse", "--check", str(checkout.patch))

    with pytest.raises(sync.SyncError, match="ambiguous"):
        _inspect(checkout)


def test_status_scope_includes_both_sides_of_a_rename(patch_checkout):
    checkout = patch_checkout
    old = checkout.source / "old unmanaged.py"
    old.write_text(checkout.before["conversion/base.py"])
    _commit(checkout, local=False)
    _git(checkout.source, "mv", "-f", old.name, "conversion/base.py")

    with pytest.raises(sync.SyncError, match="unrelated changes.*old unmanaged"):
        _inspect(checkout)


def test_status_scope_preserves_untracked_unicode_and_spaces(patch_checkout):
    checkout = patch_checkout
    (checkout.source / "额外 file.py").write_text("# unmanaged\n")

    with pytest.raises(sync.SyncError, match="unrelated changes.*额外 file.py"):
        _inspect(checkout)



def test_exact_worktree_check_uses_git_line_ending_filters(patch_checkout):
    checkout = patch_checkout
    _git(checkout.source, "config", "core.autocrlf", "true")
    for name, content in checkout.after.items():
        (checkout.source / name).write_bytes(content.replace("\n", "\r\n").encode("utf-8"))

    assert _inspect(checkout)["patches"][0]["state"] == "applied"


@pytest.mark.parametrize("field", ["upstream", "revision_local"])
@pytest.mark.parametrize("revision", ["g" * 40, "A" * 40, "a" * 39, None])
def test_contract_rejects_noncanonical_revision_pins(tmp_path: Path, field, revision):
    lock = _write_lock(tmp_path, "patch.diff", "a" * 64)
    data = json.loads(lock.read_text(encoding="utf-8"))
    if field == "upstream":
        data["upstream"]["revision"] = revision
    else:
        data[field] = revision
    lock.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(sync.SyncError, match="revision"):
        sync._contract(lock)


@pytest.mark.parametrize("pin_kind", ["missing", "tree", "unrelated_commit"])
def test_local_pin_does_not_bypass_upstream_commit_validation(patch_checkout, pin_kind):
    checkout = patch_checkout
    _write_sources(checkout.source, checkout.after)
    _commit(checkout)
    if pin_kind == "missing":
        upstream = "f" * 40
    elif pin_kind == "tree":
        upstream = _git(checkout.source, "rev-parse", f"{checkout.revision}^{{tree}}")
    else:
        tree = _git(checkout.source, "rev-parse", f"{checkout.revision}^{{tree}}")
        upstream = _git(checkout.source, "commit-tree", tree, "-m", "unrelated upstream")
    _update_lock(checkout, upstream={"revision": upstream})

    with pytest.raises(sync.SyncError, match="upstream"):
        _inspect(checkout)


@pytest.mark.parametrize("target", ["conversion/base.py", "src/unmanaged.cpp"])
def test_clean_local_pin_rejects_committed_changes_outside_managed_patch(patch_checkout, target):
    checkout = patch_checkout
    _write_sources(checkout.source, checkout.after)
    changed = checkout.source / target
    with changed.open("a", encoding="utf-8") as stream:
        stream.write("// unmanaged committed change beyond patch context\n")
    _commit(checkout)
    assert _git(checkout.source, "status", "--porcelain") == ""
    _git(checkout.source, "apply", "--reverse", "--check", str(checkout.patch))

    with pytest.raises(sync.SyncError, match="unmanaged.*diff|pinned source.*upstream"):
        _inspect(checkout)


@pytest.mark.parametrize("target", ["conversion/base.py", "src/unmanaged.cpp"])
def test_skip_worktree_cannot_hide_source_drift(patch_checkout, target):
    checkout = patch_checkout
    if target not in checkout.before:
        _write_sources(checkout.source, {target: "// unchanged upstream file\n"})
        _commit(checkout, local=False)
    _write_sources(checkout.source, checkout.after)
    _commit(checkout)
    _git(checkout.source, "update-index", "--skip-worktree", "--", target)
    with (checkout.source / target).open("a", encoding="utf-8") as stream:
        stream.write("// unmanaged hidden worktree change beyond patch context\n")
    assert _git(checkout.source, "status", "--porcelain") == ""

    with pytest.raises(sync.SyncError, match="unmanaged.*diff"):
        _inspect(checkout)


def test_committed_source_verification_preserves_checkout_and_object_database(patch_checkout):
    checkout = patch_checkout
    _write_sources(checkout.source, checkout.after)
    _commit(checkout)
    before = {
        "index": (checkout.source / ".git/index").read_bytes(),
        "objects": _git(checkout.source, "count-objects", "-v"),
        "status": _git(checkout.source, "status", "--porcelain"),
        "files": {name: (checkout.source / name).read_bytes() for name in checkout.after},
    }

    assert _inspect(checkout)["patches"][0]["state"] == "applied_committed"

    assert (checkout.source / ".git/index").read_bytes() == before["index"]
    assert _git(checkout.source, "count-objects", "-v") == before["objects"]
    assert _git(checkout.source, "status", "--porcelain") == before["status"]
    assert {name: (checkout.source / name).read_bytes() for name in checkout.after} == before["files"]


def test_managed_patch_checkout_preserves_locked_bytes_with_autocrlf(tmp_path: Path):
    """Windows checkout filters must preserve the raw SHA-256 contract."""
    root = Path(__file__).resolve().parents[1]
    relative = Path("scripts/model_tools/patches/llama-cpp-layer-forward-api.patch")
    repository = tmp_path / "patch checkout repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    for key, value in (("user.name", "Sync test"), ("user.email", "sync@example.invalid"),
                       ("core.autocrlf", "true"), ("commit.gpgsign", "false")):
        _git(repository, "config", key, value)
    (repository / ".gitattributes").write_bytes((root / ".gitattributes").read_bytes())
    patch = repository / relative
    patch.parent.mkdir(parents=True)
    expected = (root / relative).read_bytes()
    patch.write_bytes(expected)
    _git(repository, "add", ".gitattributes", relative.as_posix())
    _git(repository, "commit", "-qm", "byte-pinned patch fixture")
    exported = tmp_path / "filtered checkout"
    prefix = exported.as_posix() + "/"
    _git(repository, "checkout-index", "--all", "--prefix=" + prefix)

    assert (exported / relative).read_bytes() == expected


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
@pytest.mark.parametrize("local_pin", [False, True])
def test_hidden_worktree_patch_is_never_reported_as_committed(patch_checkout, flag, local_pin):
    checkout = patch_checkout
    if local_pin:
        _update_lock(checkout, revision_local=checkout.revision)
    _write_sources(checkout.source, checkout.after)
    _git(checkout.source, "update-index", flag, "--", *checkout.before)
    assert _git(checkout.source, "status", "--porcelain") == ""
    original_index = (checkout.source / ".git/index").read_bytes()

    result = _inspect(checkout)

    assert result["state"] == "applied"
    assert result["patches"][0]["state"] == "applied"
    assert _git(checkout.source, "show", "HEAD:src/model.cpp").startswith("int enabled = 0;")
    assert (checkout.source / ".git/index").read_bytes() == original_index
