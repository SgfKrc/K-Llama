"""Synthetic local Git histories exercise source evidence, never release acceptance."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/package_contract/build_provenance.py"
spec = importlib.util.spec_from_file_location("qlh_build_provenance_tests", SCRIPT)
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


def git(path, *args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(GIT_AUTHOR_NAME="Provenance Fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="Provenance Fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid",
               GIT_AUTHOR_DATE="2026-10-01T00:00:00Z", GIT_COMMITTER_DATE="2026-10-01T00:00:00Z")
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, env=env, check=True)
    return result.stdout.decode("utf-8").strip()


def put(path, name, data):
    target = path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def commit(path, *names):
    git(path, "add", "--", *names)
    git(path, "-c", "commit.gpgsign=false", "-c", "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null",
        "commit", "-q", "-m", "synthetic source fixture")
    return git(path, "rev-parse", "HEAD")


@pytest.fixture(scope="session")
def seed_checkout(tmp_path_factory):
    root = tmp_path_factory.mktemp("provenance-seed")

    def repository(component):
        checkout = root / component
        checkout.mkdir(parents=True, exist_ok=True)
        git(checkout, "init", "-q")
        git(checkout, "config", "core.autocrlf", "false")
        put(checkout, ".gitignore", "build/\ndist/\n__pycache__/\n.signing-keys/\n.env\n")
        put(checkout, "source.txt", b"source bytes\n")
        children = provenance.SUBMODULES.get(component, ())
        names = [".gitignore", "source.txt"]
        if component == "":
            put(checkout, "release-contract.json", '{"schema_version":1}\n')
            put(checkout, "src/core.py", b"VALUE = 1\n")
            names += ["release-contract.json", "src/core.py"]
        if component == "android":
            put(checkout, "app/src/main/Production.kt", b"class Production\n")
            names += ["app/src/main/Production.kt"]
        if children:
            modules = []
            for child in children:
                child_component = f"{component}/{child}" if component else child
                oid = repository(child_component)
                git(checkout, "update-index", "--add", "--cacheinfo", f"160000,{oid},{child}")
                modules.append(f'[submodule "{child}"]\n\tpath = {child}\n\turl = https://example.invalid/{child}.git\n')
            put(checkout, ".gitmodules", "".join(modules))
            names.append(".gitmodules")
        return commit(checkout, *names)

    repository("")
    return root


@pytest.fixture
def checkout(seed_checkout, tmp_path):
    root = tmp_path / "checkout"
    shutil.copytree(seed_checkout, root)
    return root


def test_clean_fixed_graph_is_full_deterministic_identity(checkout, monkeypatch):
    monkeypatch.setenv("QLH_SOURCE_REVISION", "f" * 40)
    monkeypatch.setenv("SOURCE_REVISION", "f" * 40)
    monkeypatch.setenv("GIT_DIR", str(checkout / "android/.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(checkout / "android"))
    info = provenance.collect_build_info(checkout)
    assert info == provenance.collect_build_info(checkout)
    assert info["schema_version"] == 1
    assert info["source_revision"] == git(checkout, "rev-parse", "HEAD")
    assert len(info["source_revision"]) == 40 and info["source_dirty"] is False
    assert len(info["source_tree_sha256"]) == 64
    assert set(info["submodules"]) == {
        "android", "frontend_cybergothic", "harness_workbench", "packaging",
        "android/app/src/main/cpp/llama.cpp",
    }
    assert all(item["source_dirty"] is False and item["revision_matches_pin"] is True
               for item in info["submodules"].values())
    assert "timestamp" not in info


@pytest.mark.parametrize("component,name", [
    (".", "src/core.py"),
    ("android", "app/src/main/Production.kt"),
    ("android/app/src/main/cpp/llama.cpp", "source.txt"),
    ("packaging", "source.txt"),
])
def test_actual_source_change_marks_component_and_aggregate_dirty(checkout, component, name):
    before = provenance.collect_build_info(checkout)
    put(checkout / component, name, b"different production bytes\n")
    after = provenance.collect_build_info(checkout)
    assert after["source_revision"] == before["source_revision"]
    assert after["source_dirty"] is True
    assert after["components"][component]["source_dirty"] is True
    assert name in after["components"][component]["tracked_changes"]
    assert after["source_tree_sha256"] != before["source_tree_sha256"]


@pytest.mark.parametrize("component,name", [
    (".", "src/new_worker.py"), (".", "scripts/package_contract/new_source.py"),
    ("android", "app/src/main/NewWorker.kt"),
    ("android", "worker-reconnect-evidence/NewProduction.kt"),
])
def test_untracked_production_sources_are_hashed_and_dirty(checkout, component, name):
    before = provenance.collect_build_info(checkout)
    put(checkout / component, name, b"untracked production source\n")
    after = provenance.collect_build_info(checkout)
    assert after["source_dirty"] is True
    assert name in after["components"][component]["untracked_sources"]
    assert after["source_tree_sha256"] != before["source_tree_sha256"]


def test_gitignore_cannot_hide_untracked_source_in_production_directories(checkout):
    with (checkout / ".gitignore").open("a", encoding="utf-8") as stream:
        stream.write("scripts/*.py\n")
    commit(checkout, ".gitignore")
    before = provenance.collect_build_info(checkout)
    put(checkout, "scripts/hidden_worker.py", b"print('real new source')\n")
    after = provenance.collect_build_info(checkout)
    assert before["source_dirty"] is False and after["source_dirty"] is True
    assert "scripts/hidden_worker.py" in after["components"]["."]["untracked_sources"]


def test_old_logs_build_outputs_and_known_android_evidence_do_not_dirty_sources(checkout):
    before = provenance.collect_build_info(checkout)
    for name in ("pc-reconnect-after.log", "worker-reconnect-full-jvm.log",
                 "worker-reconnect-evidence/jvm-results.json",
                 "worker-reconnect-evidence/TEST-fixture.xml",
                 "worker-reconnect-evidence/tcp_comm.recovered.py",
                 "worker-reconnect-evidence/pc-working.patch",
                 "app/build/generated/old-source.kt"):
        put(checkout / "android", name, b"old build evidence\n")
    put(checkout, "build/provenance/build-info.json", b"stale identity")
    put(checkout, "dist/old.exe", b"old binary")
    put(checkout, "src/import_time.log", b"runtime import evidence")
    put(checkout, "tests/simulation/results/old-result.json", b"old simulation evidence")
    assert provenance.collect_build_info(checkout) == before


def test_tracked_files_are_never_excluded_as_build_evidence(checkout):
    put(checkout / "android", "tracked.log", b"tracked build input\n")
    commit(checkout / "android", "tracked.log")
    # Update the synthetic superproject pin so the baseline is clean.
    git(checkout, "add", "android")
    commit(checkout, "android")
    before = provenance.collect_build_info(checkout)
    put(checkout / "android", "tracked.log", b"changed tracked input\n")
    after = provenance.collect_build_info(checkout)
    assert before["source_dirty"] is False and after["source_dirty"] is True
    assert "tracked.log" in after["components"]["android"]["tracked_changes"]


def test_submodule_revision_drift_records_both_actual_and_fixed_revision(checkout):
    before = provenance.collect_build_info(checkout)
    put(checkout / "android", "source.txt", b"new Android revision\n")
    actual = commit(checkout / "android", "source.txt")
    info = provenance.collect_build_info(checkout)
    android = info["submodules"]["android"]
    assert android["source_revision"] == actual
    assert android["expected_revision"] == before["submodules"]["android"]["source_revision"]
    assert android["revision_matches_pin"] is False and android["source_dirty"] is True
    assert info["source_dirty"] is True
    assert info["source_revision"] == before["source_revision"]


def test_staged_gitlink_change_is_dirty_even_with_original_checkout_revision(checkout):
    before = provenance.collect_build_info(checkout)
    # Stage a different commit pointer without moving the actual Android checkout.
    other = before["submodules"]["packaging"]["source_revision"]
    git(checkout, "update-index", "--cacheinfo", f"160000,{other},android")
    after = provenance.collect_build_info(checkout)
    assert after["source_dirty"] is True
    assert "android" in after["components"]["."]["tracked_changes"]


def test_missing_tracked_source_is_dirty_and_changes_byte_hash(checkout):
    before = provenance.collect_build_info(checkout)
    (checkout / "src/core.py").unlink()
    after = provenance.collect_build_info(checkout)
    assert after["source_dirty"] is True
    assert after["source_tree_sha256"] != before["source_tree_sha256"]


def test_hash_uses_actual_bytes_even_with_same_size_and_mtime(checkout):
    target = checkout / "src/core.py"
    before = provenance.collect_build_info(checkout)
    original_stat = target.stat()
    target.write_bytes(b"VALUE = 2\n")
    os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    after = provenance.collect_build_info(checkout)
    assert after["source_dirty"] is True
    assert after["source_tree_sha256"] != before["source_tree_sha256"]
    assert after["components"]["."]["source_tree_sha256"] != before["components"]["."]["source_tree_sha256"]
    record = provenance._file_record(checkout, "src/core.py", "100644")
    assert record["sha256"] == hashlib.sha256(b"VALUE = 2\n").hexdigest()


@pytest.mark.parametrize("bad_root", ["android", "src", "export"])
def test_wrong_root_and_gitless_export_fail_explicitly(checkout, tmp_path, bad_root):
    path = checkout / bad_root
    path.mkdir(exist_ok=True)
    with pytest.raises(provenance.ProvenanceError, match="Git|checkout|graph|root"):
        provenance.collect_build_info(path)


def test_uninitialized_submodule_cannot_be_mistaken_for_parent_git_root(checkout):
    original = (checkout / "android/.git").resolve()
    saved = (checkout / "android/.saved-git-fixture").resolve()
    assert original.is_relative_to(checkout.resolve()) and saved.is_relative_to(checkout.resolve())
    original.rename(saved)
    with pytest.raises(provenance.ProvenanceError, match="Git checkout required"):
        provenance.collect_build_info(checkout)


def test_missing_fixed_gitlink_fails_instead_of_silently_omitting_component(checkout):
    git(checkout, "update-index", "--force-remove", "android")
    with pytest.raises(provenance.ProvenanceError, match="Fixed submodule graph"):
        provenance.collect_build_info(checkout)


def test_git_failure_does_not_default_to_clean_or_echo_credentials(checkout, monkeypatch, capsys):
    def fail(*args, **kwargs):
        return subprocess.CompletedProcess(args, 128, b"", b"password=DO-NOT-PRINT")
    monkeypatch.setattr(provenance.subprocess, "run", fail)
    assert provenance.main(["--root", str(checkout), "--print"]) == 2
    captured = capsys.readouterr()
    assert "Git source identity/read failed" in captured.err
    assert "DO-NOT-PRINT" not in captured.err and not captured.out


def test_sensitive_material_is_never_opened_or_emitted(checkout, monkeypatch):
    put(checkout / "packaging", ".signing-keys/release.key", b"DO-NOT-READ-PRIVATE-KEY")
    put(checkout / "android", "keystore.properties", b"storePassword=DO-NOT-READ")
    put(checkout, ".env", b"TOKEN=DO-NOT-READ")
    original_open = Path.open

    def refuse_secret(path, *args, **kwargs):
        assert path.name not in {"release.key", "keystore.properties", ".env"}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_secret)
    info = provenance.collect_build_info(checkout)
    assert "DO-NOT-READ" not in json.dumps(info)


def test_changed_sources_during_observation_fail_closed(checkout, monkeypatch):
    original_observe = provenance._observe
    count = 0

    def mutate(root):
        nonlocal count
        result = original_observe(root)
        count += 1
        if count == 1:
            put(root, "src/core.py", b"changed during observation\n")
        return result

    monkeypatch.setattr(provenance, "_observe", mutate)
    with pytest.raises(provenance.ProvenanceError, match="changed during"):
        provenance.collect_build_info(checkout)


def test_generation_refreshes_stale_identity_and_never_relabels_dist(checkout):
    output = checkout / "android/app/build/generated/qlhBuildInfo/assets/build-info.json"
    first = provenance.write_build_info(checkout, output)
    put(checkout, "src/core.py", b"current source\n")
    second = provenance.write_build_info(checkout, output)
    assert first["source_revision"] == second["source_revision"]
    assert second["source_dirty"] is True
    assert json.loads(output.read_text(encoding="ascii")) == second
    assert first["source_tree_sha256"] != second["source_tree_sha256"]
    assert output.stat().st_size <= 65536
    stale = checkout / "dist/build-info.json"
    put(checkout, "dist/build-info.json", b"old shipped identity")
    with pytest.raises(provenance.ProvenanceError, match="build/"):
        provenance.write_build_info(checkout, stale)
    assert stale.read_bytes() == b"old shipped identity"


def test_cli_print_only_is_read_only_and_output_is_json(checkout):
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--root", str(checkout), "--print"],
                            capture_output=True, check=True, text=True)
    assert json.loads(result.stdout)["source_dirty"] is False
    assert not (checkout / "build").exists()



def test_non_build_key_document_uses_git_identity_without_reading_bytes(checkout, monkeypatch):
    vendor = checkout / "android/app/src/main/cpp/llama.cpp"
    name = "docs/development/llama-star/idea-arch.key"
    put(vendor, name, b"opaque non-build document")
    commit(vendor, name)
    git(checkout / "android", "add", "app/src/main/cpp/llama.cpp")
    commit(checkout / "android", "app/src/main/cpp/llama.cpp")
    git(checkout, "add", "android")
    commit(checkout, "android")
    original_open = Path.open

    def refuse_key(path, *args, **kwargs):
        assert path.name != "idea-arch.key"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_key)
    info = provenance.collect_build_info(checkout)
    assert info["source_dirty"] is False
    assert info["submodules"]["android/app/src/main/cpp/llama.cpp"]["non_build_git_only_files"] == [name]


def test_tracked_secret_fails_without_opening_it(checkout, monkeypatch):
    put(checkout, "private.key", b"synthetic forbidden secret")
    commit(checkout, "private.key")
    original_open = Path.open

    def refuse_key(path, *args, **kwargs):
        assert path.name != "private.key"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_key)
    with pytest.raises(provenance.ProvenanceError, match="Sensitive file is tracked"):
        provenance.collect_build_info(checkout)



def test_assume_unchanged_cannot_hide_actual_source_bytes(checkout):
    git(checkout, "update-index", "--assume-unchanged", "src/core.py")
    put(checkout, "src/core.py", b"VALUE = 9\n")
    assert git(checkout, "diff", "--name-only", "HEAD", "--", "src/core.py") == ""
    info = provenance.collect_build_info(checkout)
    assert info["source_dirty"] is True
    assert "src/core.py" in info["components"]["."]["tracked_changes"]


def test_windows_text_checkout_crlf_is_clean_but_hashes_actual_bytes(checkout):
    git(checkout, "config", "core.autocrlf", "true")
    before = provenance.collect_build_info(checkout)
    put(checkout, "src/core.py", b"VALUE = 1\r\n")
    after = provenance.collect_build_info(checkout)
    assert after["source_dirty"] is False
    assert after["source_tree_sha256"] != before["source_tree_sha256"]



@pytest.fixture(scope="session")
def seed_info(seed_checkout):
    return provenance.collect_build_info(seed_checkout)


@pytest.fixture
def saved_snapshot(tmp_path, seed_info, monkeypatch):
    snapshot = tmp_path / "build-info.json"
    snapshot.write_text(json.dumps(seed_info), encoding="utf-8")
    monkeypatch.setattr(provenance, "collect_build_info", lambda root: copy.deepcopy(seed_info))
    return snapshot


def test_verify_is_read_only_and_compares_stable_dirty_snapshots(saved_snapshot, seed_info, monkeypatch):
    info = copy.deepcopy(seed_info)
    info["source_dirty"] = True
    info["components"]["android"]["source_dirty"] = True
    saved_snapshot.write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(provenance, "collect_build_info", lambda root: copy.deepcopy(info))
    before, before_stat = saved_snapshot.read_bytes(), saved_snapshot.stat()
    assert provenance.verify_build_info(saved_snapshot.parent, saved_snapshot) == info
    assert saved_snapshot.read_bytes() == before
    assert saved_snapshot.stat().st_mtime_ns == before_stat.st_mtime_ns


@pytest.mark.parametrize("component,name", [
    (".", "src/core.py"),
    ("android", "app/src/main/Production.kt"),
    ("android/app/src/main/cpp/llama.cpp", "source.txt"),
])
def test_verify_refuses_compile_time_source_changes_without_overwriting_snapshot(checkout, component, name):
    snapshot = checkout / "build/provenance/build-info.json"
    provenance.write_build_info(checkout, snapshot)
    before, before_stat = snapshot.read_bytes(), snapshot.stat()
    put(checkout / component, name, b"changed while compile/native work ran\n")
    with pytest.raises(provenance.ProvenanceError, match="snapshot mismatch.*source_tree_sha256"):
        provenance.verify_build_info(checkout, snapshot)
    assert snapshot.read_bytes() == before
    assert snapshot.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_verify_refuses_new_production_source_after_generation(checkout):
    snapshot = checkout / "build/provenance/build-info.json"
    provenance.write_build_info(checkout, snapshot)
    before = snapshot.read_bytes()
    put(checkout / "android", "app/src/main/NewDuringCompilation.kt", b"class NewDuringCompilation\n")
    with pytest.raises(provenance.ProvenanceError, match="snapshot mismatch"):
        provenance.verify_build_info(checkout, snapshot)
    assert snapshot.read_bytes() == before


@pytest.mark.parametrize("component", [".", "android"])
def test_verify_compares_git_identity_even_if_source_tree_bytes_are_unchanged(checkout, component):
    snapshot = checkout / "build/provenance/build-info.json"
    before = provenance.write_build_info(checkout, snapshot)
    git(checkout / component, "-c", "commit.gpgsign=false", "-c",
        "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null",
        "commit", "--allow-empty", "-q", "-m", "synthetic identity-only change")
    current = provenance.collect_build_info(checkout)
    assert current["source_tree_sha256"] == before["source_tree_sha256"]
    assert current["components"][component]["source_revision"] != before["components"][component]["source_revision"]
    original = snapshot.read_bytes()
    with pytest.raises(provenance.ProvenanceError, match="snapshot mismatch.*source_revision"):
        provenance.verify_build_info(checkout, snapshot)
    assert snapshot.read_bytes() == original


@pytest.mark.parametrize("change,field", [
    (lambda info: info.update(schema_version=True), "schema_version"),
    (lambda info: info.update(source_dirty=0), "source_dirty"),
    (lambda info: info.update(source_dirty=None), "source_dirty"),
    (lambda info: info.update(source_dirty="false"), "source_dirty"),
    (lambda info: info.update(source_dirty=True), "source_dirty"),
    (lambda info: info.update(source_revision="f" * 40), "source_revision"),
    (lambda info: info.update(source_revision=0), "source_revision"),
    (lambda info: info.update(source_tree_sha256="f" * 64), "source_tree_sha256"),
    (lambda info: info.update(source_tree_hash_algorithm="unknown"), "source_tree_hash_algorithm"),
    (lambda info: info["components"]["android"].update(source_dirty=0), "components.android.source_dirty"),
    (lambda info: info["components"]["android"].update(source_revision="f" * 40), "components.android.source_revision"),
    (lambda info: info["components"]["android"].update(source_file_count=True), "source_file_count"),
    (lambda info: info["components"]["android"].update(revision_matches_pin=1), "revision_matches_pin"),
    (lambda info: info["submodules"]["android"].update(expected_revision="f" * 40), "expected_revision"),
    (lambda info: info["submodules"].pop("android"), "submodules.android"),
    (lambda info: info.pop("source_dirty"), "source_dirty"),
    (lambda info: info.update(extra_field=False), "extra_field"),
])
def test_verify_compares_all_snapshot_facts_and_exact_types(saved_snapshot, change, field):
    info = json.loads(saved_snapshot.read_text(encoding="utf-8"))
    original_hash = info["source_tree_sha256"]
    change(info)
    saved_snapshot.write_text(json.dumps(info), encoding="utf-8")
    original = saved_snapshot.read_bytes()
    if field != "source_tree_sha256":
        assert info["source_tree_sha256"] == original_hash
    with pytest.raises(provenance.ProvenanceError, match="snapshot mismatch") as error:
        provenance.verify_build_info(saved_snapshot.parent, saved_snapshot)
    assert field in str(error.value)
    assert saved_snapshot.read_bytes() == original


@pytest.mark.parametrize("raw", [
    b"", b"not JSON", b"\xff", b"{} trailing", b"{} {}",
    b"[]", b"null", b"true", b'"string"',
    b'{"schema_version":1,"schema_version":1}',
    b'{"components":{"android":{"source_dirty":false,"source_dirty":true}}}',
    b'{"source_dirty":NaN}', b'{"source_dirty":Infinity}', b'{"source_dirty":-Infinity}',
    b'{"source_dirty":1e999}', b'{"schema_version":1.0}', pytest.param(b" " * 65537, id="oversized-snapshot"),
])
def test_verify_rejects_malformed_duplicate_nonfinite_or_wrong_root_json_before_git(tmp_path, monkeypatch, raw):
    snapshot = tmp_path / "build-info.json"
    snapshot.write_bytes(raw)
    monkeypatch.setattr(provenance, "collect_build_info", lambda root: pytest.fail("invalid snapshot must fail before Git"))
    with pytest.raises(provenance.ProvenanceError, match="snapshot invalid"):
        provenance.verify_build_info(tmp_path, snapshot)
    assert snapshot.read_bytes() == raw


def test_verify_missing_snapshot_is_named_and_never_regenerated(tmp_path, monkeypatch, capsys):
    snapshot = tmp_path / "build-info.json"
    monkeypatch.setattr(provenance, "write_build_info", lambda *args: pytest.fail("verification must never generate"))
    assert provenance.main(["--root", str(tmp_path), "--verify", str(snapshot)]) == 2
    captured = capsys.readouterr()
    assert "build-info snapshot missing" in captured.err and not captured.out
    assert not snapshot.exists()


def test_verify_detects_snapshot_replacement_during_source_observation(saved_snapshot, seed_info, monkeypatch):
    def replace_during_observation(root):
        saved_snapshot.write_bytes(saved_snapshot.read_bytes() + b"\n")
        return copy.deepcopy(seed_info)
    monkeypatch.setattr(provenance, "collect_build_info", replace_during_observation)
    with pytest.raises(provenance.ProvenanceError, match="snapshot changed during verification"):
        provenance.verify_build_info(saved_snapshot.parent, saved_snapshot)


def test_cli_verify_success_and_mismatch_do_not_write_original_snapshot(checkout):
    snapshot = checkout / "build/provenance/build-info.json"
    provenance.write_build_info(checkout, snapshot)
    original, before_stat = snapshot.read_bytes(), snapshot.stat()
    args = [sys.executable, "-B", str(SCRIPT), "--root", str(checkout), "--verify", str(snapshot)]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["verified"] is True
    put(checkout / "android", "app/src/main/Production.kt", b"class ChangedAfterCompile\n")
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2 and "build-info snapshot mismatch" in result.stderr and not result.stdout
    assert snapshot.read_bytes() == original
    assert snapshot.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_cli_verify_is_mutually_exclusive_with_generation_or_print(saved_snapshot):
    for other in (["--print"], ["--output", str(saved_snapshot)]):
        with pytest.raises(SystemExit) as error:
            provenance.main(["--verify", str(saved_snapshot), *other])
        assert error.value.code == 2
