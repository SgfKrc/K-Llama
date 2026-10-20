"""Quantize source gates using small real Git checkouts; no native build."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import build_llama_quantize as build
from scripts.model_tools import sync_llama_cpp as sync


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args], check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    return result.stdout.rstrip("\r\n")


def _write_sources(source: Path, contents: dict[str, str]) -> None:
    for name, content in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def _save_lock(checkout) -> None:
    checkout.lock_path.write_text(json.dumps(checkout.lock), encoding="utf-8")


def _apply(checkout, count: int = 2) -> None:
    _write_sources(checkout.source, dict(list(checkout.after.items())[:count]))


def _commit_local(checkout) -> str:
    _git(checkout.source, "add", ".")
    _git(checkout.source, "commit", "-qm", "local fixture")
    revision = _git(checkout.source, "rev-parse", "HEAD")
    checkout.lock["revision_local"] = revision
    _save_lock(checkout)
    return revision


def _inspect(checkout):
    return build._source_contract(checkout.source, checkout.root, checkout.lock)


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch):
    for name in tuple(os.environ):
        if name.upper().startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "0")
    root = tmp_path / "project with spaces"
    source = root / "android/app/src/main/cpp/llama.cpp"
    source.mkdir(parents=True)
    _git(source, "init", "-q")
    for name, value in (
        ("user.name", "Quantize source fixture"), ("user.email", "source@example.invalid"),
        ("core.autocrlf", "false"), ("commit.gpgsign", "false"),
    ):
        _git(source, "config", name, value)
    before = {
        "include/llama.h": "int managed_header = 0;\n" + "// context\n" * 20,
        "src/model.cpp": "int managed_model = 0;\n" + "// context\n" * 20,
    }
    after = {name: content.replace("= 0;", "= 1;") for name, content in before.items()}
    _write_sources(source, {**before, "src/unmanaged.cpp": "int unchanged = 0;\n"})
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "upstream fixture")
    revision = _git(source, "rev-parse", "HEAD")
    patches = []
    for number, target in enumerate(before):
        _write_sources(source, {target: after[target]})
        patch = root / f"scripts/model_tools/patches/fixture-{number}.patch"
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_bytes(subprocess.run(
            ["git", "-C", str(source), "diff", "--binary", "--", target],
            check=True, capture_output=True, timeout=30,
        ).stdout)
        patches.append({
            "id": f"fixture-{number}", "path": patch.relative_to(root).as_posix(),
            "targets": [target], "marker": "diagnostic description",
            "sha256": sync._sha256(patch),
        })
        _write_sources(source, {target: before[target]})
    lock_path = root / "scripts/model_tools/llama_quantize.lock.json"
    lock = {
        "schema_version": 1, "tool": "llama-quantize", "cmake_target": "llama-quantize",
        "upstream": {"repository": "fixture", "revision": revision},
        "patches": patches,
        "targets": {
            "windows-x86_64": {"platform": "windows", "architecture": "x86_64", "executable": "llama-quantize.exe"},
            "linux-x86_64": {"platform": "linux", "architecture": "x86_64", "executable": "llama-quantize"},
        },
    }
    fixture = SimpleNamespace(root=root, source=source, lock_path=lock_path, lock=lock,
                              before=before, after=after, revision=revision)
    _save_lock(fixture)
    monkeypatch.setattr(sync, "ROOT", root)
    monkeypatch.setenv("QLH_ANDROID_ROOT", str(root / "android"))
    return fixture


def _expected_receipt(checkout, revision: str, *, dirty: bool):
    return {
        "source_revision": revision, "source_dirty": dirty,
        "source_patches": [
            {"id": item["id"], "patch_sha256": item["sha256"],
             "state": "applied" if dirty else "applied_committed"}
            for item in checkout.lock["patches"]
        ],
    }


def test_committed_source_preserves_receipt_and_repeat_verification(checkout):
    _apply(checkout)
    revision = _commit_local(checkout)

    assert _inspect(checkout) == _expected_receipt(checkout, revision, dirty=False)
    assert _inspect(checkout) == _expected_receipt(checkout, revision, dirty=False)
    assert _git(checkout.source, "status", "--porcelain") == ""


@pytest.mark.parametrize("staged", [False, True])
def test_valid_worktree_patch_form_preserves_receipt(checkout, staged):
    _apply(checkout)
    if staged:
        _git(checkout.source, "add", ".")

    assert _inspect(checkout) == _expected_receipt(checkout, checkout.revision, dirty=True)
    assert _git(checkout.source, "rev-parse", "HEAD") == checkout.revision
    assert "revision_local" not in checkout.lock


@pytest.mark.parametrize("applied_count", [0, 1])
def test_pending_or_mixed_pending_patches_are_rejected_without_applying(checkout, applied_count):
    _apply(checkout, applied_count)
    before = {name: (checkout.source / name).read_bytes() for name in checkout.before}
    index = _git(checkout.source, "ls-files", "--stage")

    with pytest.raises(build.BuildError, match="pending"):
        _inspect(checkout)

    assert {name: (checkout.source / name).read_bytes() for name in checkout.before} == before
    assert _git(checkout.source, "ls-files", "--stage") == index


@pytest.mark.parametrize("target", ["include/llama.h", "src/unmanaged.cpp"])
def test_extra_committed_changes_are_rejected_even_with_clean_pinned_head(checkout, target):
    _apply(checkout)
    with (checkout.source / target).open("a", encoding="utf-8") as stream:
        stream.write("// unrelated committed edit beyond patch context\n")
    _commit_local(checkout)
    assert _git(checkout.source, "status", "--porcelain") == ""

    with pytest.raises(build.BuildError, match="unmanaged.*diff"):
        _inspect(checkout)


@pytest.mark.parametrize("kind", ["missing", "tree", "unrelated_commit"])
def test_local_revision_cannot_bypass_fake_upstream(checkout, kind):
    _apply(checkout)
    _commit_local(checkout)
    tree = _git(checkout.source, "rev-parse", f"{checkout.revision}^{{tree}}")
    if kind == "missing":
        upstream = "f" * 40
    elif kind == "tree":
        upstream = tree
    else:
        upstream = _git(checkout.source, "commit-tree", tree, "-m", "unrelated fixture")
    checkout.lock["upstream"]["revision"] = upstream
    _save_lock(checkout)

    with pytest.raises(build.BuildError, match="upstream"):
        _inspect(checkout)


@pytest.mark.parametrize("mode", ["skip-edit", "assume-edit", "skip-delete"])
def test_hidden_worktree_drift_is_rejected(checkout, mode):
    _apply(checkout)
    _commit_local(checkout)
    flag = "--assume-unchanged" if mode == "assume-edit" else "--skip-worktree"
    target = "src/unmanaged.cpp"
    _git(checkout.source, "update-index", flag, "--", target)
    path = checkout.source / target
    if mode == "skip-delete":
        path.unlink()
    else:
        path.write_text("int unchanged = 99;\n", encoding="utf-8")
    assert _git(checkout.source, "status", "--porcelain") == ""

    with pytest.raises(build.BuildError, match="unmanaged.*diff"):
        _inspect(checkout)


def test_committed_pin_still_rejects_uncommitted_patch_form(checkout):
    checkout.lock["revision_local"] = checkout.revision
    _save_lock(checkout)
    _apply(checkout)

    with pytest.raises(build.BuildError, match="committed-patch source has worktree"):
        _inspect(checkout)


@pytest.mark.parametrize("local_pin", [False, True])
@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_hidden_complete_worktree_patch_cannot_claim_committed_source(checkout, local_pin, flag):
    if local_pin:
        checkout.lock["revision_local"] = checkout.revision
        _save_lock(checkout)
    _apply(checkout)
    _git(checkout.source, "update-index", flag, "--", *checkout.before)
    assert _git(checkout.source, "status", "--porcelain") == ""
    assert _git(checkout.source, "show", "HEAD:include/llama.h").startswith("int managed_header = 0;")

    if local_pin:
        with pytest.raises(build.BuildError, match="committed-patch source has worktree"):
            _inspect(checkout)
    else:
        assert _inspect(checkout) == _expected_receipt(checkout, checkout.revision, dirty=True)


def test_head_mismatch_still_fails_before_sync(checkout):
    checkout.lock["revision_local"] = "f" * 40
    _save_lock(checkout)

    with pytest.raises(build.BuildError, match="HEAD does not match"):
        _inspect(checkout)


def test_caller_lock_must_match_current_project_lock(checkout):
    checkout.lock_path.write_text(
        json.dumps({**checkout.lock, "note": "changed after initial load"}), encoding="utf-8",
    )

    with pytest.raises(build.BuildError, match="lock changed"):
        _inspect(checkout)


def test_patch_raw_digest_still_has_to_match(checkout):
    _apply(checkout)
    _commit_local(checkout)
    patch = checkout.root / checkout.lock["patches"][0]["path"]
    patch.write_bytes(patch.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(build.BuildError, match="SHA-256 does not match"):
        _inspect(checkout)


def test_second_verification_rejects_hidden_source_change(checkout):
    _apply(checkout)
    _commit_local(checkout)
    assert _inspect(checkout)["source_dirty"] is False
    _git(checkout.source, "update-index", "--skip-worktree", "--", "src/unmanaged.cpp")
    (checkout.source / "src/unmanaged.cpp").write_text("int unchanged = 99;\n", encoding="utf-8")

    with pytest.raises(build.BuildError, match="unmanaged.*diff"):
        _inspect(checkout)


def test_sync_uses_selected_android_root_and_dry_run(checkout, monkeypatch):
    _apply(checkout)
    _commit_local(checkout)
    monkeypatch.setenv("QLH_ANDROID_ROOT", str(checkout.root / "wrong android"))
    calls = []
    inspect = build.inspect_sync

    def observed(**kwargs):
        calls.append(kwargs)
        return inspect(**kwargs)

    monkeypatch.setattr(build, "inspect_sync", observed)

    assert _inspect(checkout)["source_dirty"] is False
    assert calls == [{"android_root": checkout.root / "android",
                      "lock_path": checkout.lock_path, "dry_run": True}]


def test_lock_changed_during_sync_is_rejected(checkout, monkeypatch):
    _apply(checkout)
    _commit_local(checkout)
    inspect = build.inspect_sync

    def changed(**kwargs):
        result = inspect(**kwargs)
        checkout.lock_path.write_text(
            json.dumps({**checkout.lock, "note": "changed during sync"}), encoding="utf-8",
        )
        return result

    monkeypatch.setattr(build, "inspect_sync", changed)

    with pytest.raises(build.BuildError, match="lock changed"):
        _inspect(checkout)


def test_source_rejection_precedes_tool_selection_or_output_creation(checkout, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid source reached tool selection")

    monkeypatch.setattr(build, "find_cmake", unexpected)
    output = checkout.root / "never-created-output"

    with pytest.raises(build.BuildError, match="pending"):
        build.build_and_package(project_root=checkout.root, output_root=output)

    assert not output.exists()
