"""Startup source receipts use real temporary Git; no production state or models."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import runtime_source as source


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    for name, data in {"src/api_server.py": "value = 1\n", "src/runtime_source.py": "pass\n",
                       "scripts/probe.py": "pass\n", "release-contract.json": "{}\n",
                       ".gitignore": "build/\n"}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
    git("init", "-q")
    git("config", "user.name", "offline test")
    git("config", "user.email", "offline@example.invalid")
    git("add", ".")
    git("commit", "-qm", "synthetic fixture")
    return root


@pytest.fixture
def opted_in(monkeypatch, checkout):
    import psutil
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION", "1")
    monkeypatch.delenv("QLH_RUNTIME_SOURCE_MANIFEST", raising=False)
    monkeypatch.setattr(source, "_STARTUP", None)
    monkeypatch.setattr(source, "_ERROR", None)
    monkeypatch.setattr(psutil, "Process", lambda *_: SimpleNamespace(pid=123, create_time=lambda: 10.0,
                                                                    cwd=lambda: str(checkout)))
    return checkout


def test_default_disabled_never_observes_git_or_source(monkeypatch):
    monkeypatch.delenv("QLH_PRODUCT_OBSERVATION", raising=False)
    monkeypatch.setattr(source, "_STARTUP", None)
    monkeypatch.setattr(source, "_ERROR", None)
    monkeypatch.setattr(source, "checkout_manifest", lambda *_: pytest.fail("disabled observer touched source"))
    source.initialize()
    with pytest.raises(source.SourceObservationError, match="not captured"):
        source.startup_source()


def test_real_git_clean_and_dirty_are_observed_without_override(checkout, monkeypatch):
    clean = source.checkout_manifest(checkout)
    assert clean["source_dirty"] is False and len(clean["source_revision"]) == 40
    (checkout / "src/hidden.py").write_text("changed = True\n")
    monkeypatch.setenv("SOURCE_REVISION", "b"*40)
    dirty = source.checkout_manifest(checkout)
    assert dirty["source_dirty"] is True and dirty["source_revision"] == clean["source_revision"]
    assert "src/hidden.py" in {entry["path"] for entry in dirty["files"]}


def test_startup_receipt_is_pid_bound_immutable_and_rehashed(opted_in):
    source.initialize(opted_in)
    first = source.startup_source()
    first["manifest"]["files"].clear()
    observed = source.observe_source("master")
    assert observed["pid"] == 123 and observed["created_at"] == 10
    assert observed["kind"] == "runtime_manifest" and observed["source_dirty"] is False
    assert observed["manifest_sha256"] == observed["runtime_manifest_sha256"] == source.canonical_hash(observed["manifest"])
    assert observed["rehashed_files"] == observed["manifest"]["files"]
    assert source.observe_source("pc")["kind"] == "synced_manifest"


@pytest.mark.parametrize("mutation", ["content", "extra", "removed"])
def test_changed_runtime_source_cannot_be_recaptured_as_original(opted_in, mutation):
    source.initialize(opted_in)
    original = source.startup_source()
    if mutation == "content":
        (opted_in / "src/api_server.py").write_text("value = 2\n")
    elif mutation == "extra":
        (opted_in / "src/extra.py").write_text("pass\n")
    else:
        (opted_in / "scripts/probe.py").unlink()
    source.initialize(opted_in)
    assert source.startup_source() == original
    with pytest.raises(source.SourceObservationError, match="changed|differ"):
        source.observe_source("master")


def test_process_identity_or_cwd_change_is_rejected(opted_in, monkeypatch):
    import psutil
    source.initialize(opted_in)
    monkeypatch.setattr(psutil, "Process", lambda *_: SimpleNamespace(pid=123, create_time=lambda: 11.0))
    with pytest.raises(source.SourceObservationError, match="another process"):
        source.observe_source("master")


def test_export_observes_clean_checkout_and_refuses_overwrite(checkout):
    target = checkout / "build/source-manifest.json"
    assert source.main(["--root", str(checkout), "--export", str(target)]) == 0
    original = target.read_bytes()
    assert source.main(["--root", str(checkout), "--export", str(target)]) == 1
    assert target.read_bytes() == original
    (checkout / "src/api_server.py").write_text("changed = True\n")
    assert source.main(["--root", str(checkout), "--export", str(target.parent / 'dirty.json')]) == 1
    assert not (target.parent / "dirty.json").exists()


def test_gitless_deployment_is_bound_to_exported_bytes(opted_in, monkeypatch):
    target = opted_in / "build/source-manifest.json"
    assert source.main(["--root", str(opted_in), "--export", str(target)]) == 0
    monkeypatch.setenv("QLH_RUNTIME_SOURCE_MANIFEST", str(target))
    monkeypatch.setattr(source, "_git", lambda *_: pytest.fail("deployment should use full manifest rehash"))
    source.initialize(opted_in)
    assert source.observe_source("pc")["source_dirty"] is False
    (opted_in / "src/api_server.py").write_text("altered = 1\n")
    with pytest.raises(source.SourceObservationError, match="differ"):
        source.observe_source("pc")


@pytest.mark.parametrize("mutation", ["dirty-string", "duplicate", "missing", "unsafe", "extra"])
def test_deployment_manifest_missing_or_forged_inventory_fails(opted_in, monkeypatch, mutation):
    manifest = source.checkout_manifest(opted_in)
    if mutation == "dirty-string":
        manifest["source_dirty"] = "false"
    elif mutation == "duplicate":
        manifest["files"].append(deepcopy(manifest["files"][0]))
    elif mutation == "missing":
        manifest["files"].pop()
    elif mutation == "unsafe":
        manifest["files"][0]["path"] = "../outside.py"
    else:
        manifest["files"].append({"path": "src/unobserved.py", "sha256": "a"*64})
    target = opted_in / "build/source-manifest.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"schema_version": 1, "manifest": manifest}))
    monkeypatch.setenv("QLH_RUNTIME_SOURCE_MANIFEST", str(target))
    source.initialize(opted_in)
    with pytest.raises(source.SourceObservationError):
        source.startup_source()


def test_source_capture_failure_stays_a_diagnostic_failure(opted_in, monkeypatch):
    monkeypatch.setattr(source, "checkout_manifest", lambda *_: (_ for _ in ()).throw(source.SourceObservationError("unreadable")))
    source.initialize(opted_in)
    monkeypatch.setattr(source, "checkout_manifest", lambda *_: pytest.fail("must not silently recapture later"))
    source.initialize(opted_in)
    with pytest.raises(source.SourceObservationError, match="unreadable"):
        source.observe_source("master")


def test_ignored_executable_source_still_dirties_startup_manifest(checkout):
    with (checkout / ".gitignore").open("a") as stream:
        stream.write("scripts/ignored.py\n")
    subprocess.check_call(["git", "-C", str(checkout), "add", ".gitignore"], stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "-C", str(checkout), "commit", "-qm", "synthetic ignore rule"], stdout=subprocess.DEVNULL)
    (checkout / "scripts/ignored.py").write_text("pass\n")
    assert not subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"])
    manifest = source.checkout_manifest(checkout)
    assert manifest["source_dirty"] is True
    assert "scripts/ignored.py" in {entry["path"] for entry in manifest["files"]}


def test_psutil_failure_never_prevents_application_startup(opted_in, monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "Process", lambda *_: (_ for _ in ()).throw(psutil.AccessDenied(pid=1)))
    source.initialize(opted_in)
    with pytest.raises(source.SourceObservationError, match="AccessDenied"):
        source.startup_source()


def test_explicit_dirty_export_remains_dirty_in_gitless_runtime(opted_in, monkeypatch, tmp_path):
    (opted_in / "src/api_server.py").write_text("# dirty development source\n", encoding="utf-8")
    target = tmp_path / "outside-source.json"
    assert source.main(["--root", str(opted_in), "--export", str(target)]) == 1
    assert not target.exists()
    assert source.main(["--root", str(opted_in), "--export", str(target), "--include-dirty"]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["manifest"]["source_dirty"] is True
    monkeypatch.setenv("QLH_RUNTIME_SOURCE_MANIFEST", str(target))
    source.initialize(opted_in)
    observed = source.observe_source("pc")
    assert observed["source_dirty"] is True and observed["manifest"]["source_dirty"] is True
    assert observed["manifest_sha256"] == source.canonical_hash(payload["manifest"])
