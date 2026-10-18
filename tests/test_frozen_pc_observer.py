"""Independent external controller tests with temporary packages; no model loads."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import psutil
import runtime_frozen_source as frozen

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frozen_pc_observer_test", ROOT / "scripts/package_contract/pc_product_observer.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)
REVISION = "a" * 40


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "frozen application"
    root.mkdir()
    info = {"schema_version": 1, "source_revision": REVISION, "source_dirty": False,
            "source_tree_sha256": "c" * 64, "components": {}}
    contents = {"QLH.exe": b"offline fake executable", "build-info.json": json.dumps(info, indent=2).encode(),
                "_internal/PYZ.pyz": b"frozen python", "_internal/native.pyd": b"native extension",
                "_internal/python312.dll": b"runtime DLL", "scripts/probe.py": b"pass\n",
                "_internal/config/program.json": b"{}", "manifest/runtime-profile.json": b"{}"}
    for name, data in contents.items():
        member = root / name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(data)
    manifest = frozen.export_package_manifest(root, executable_relative="QLH.exe")
    executable = root / "QLH.exe"
    process = SimpleNamespace(pid=101, create_time=lambda: 10.0, exe=lambda: str(executable),
                              cwd=lambda: str(root))
    monkeypatch.setattr(psutil, "Process", lambda *_: process)
    source = {"kind": "installed_exe", "pid": 101, "created_at": 10.0, "source_root": str(root),
              "executable": str(executable), **frozen._verify_package(root)}
    source.pop("_manifest_file_sha256")
    return SimpleNamespace(root=root, exe=executable, manifest=manifest, source=source,
                           process=process, os_root={"pid": 101, "created_at": 10.0, "executable": str(executable), "cwd": str(root)})


def test_actual_full_package_rehash_retains_raw_text_and_external_controller_identity(package):
    original = {p.relative_to(package.root).as_posix(): p.read_bytes() for p in package.root.rglob("*") if p.is_file()}
    fresh = observer.rehash_installed_exe(package.os_root)
    assert fresh["method"] == "independent_frozen_package_rehash"
    assert fresh["observer"] == "external_python_controller"
    assert fresh["rehashed_files"] == package.manifest["files"]
    assert fresh["manifest_text"].encode() == (package.root / frozen.MANIFEST_NAME).read_bytes()
    assert fresh["build_info_text"].encode() == (package.root / "build-info.json").read_bytes()
    assert fresh["build_info_sha256"] == hashlib.sha256(fresh["build_info_text"].encode()).hexdigest()
    # Pretty JSON deliberately has a different canonical SHA from its file SHA.
    assert fresh["build_info_sha256"] != observer._frozen_contract()._canonical_hash(fresh["build_info"])
    assert {p.relative_to(package.root).as_posix(): p.read_bytes() for p in package.root.rglob("*") if p.is_file()} == original


@pytest.mark.parametrize("name", ["QLH.exe", "build-info.json", "_internal/PYZ.pyz", "_internal/native.pyd",
                                  "_internal/python312.dll", "scripts/probe.py", "manifest/runtime-profile.json"])
def test_same_size_same_mtime_changes_are_read_as_bytes_instead_of_cached_inventory(package, name):
    member = package.root / name
    before = member.stat()
    data = member.read_bytes()
    member.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    os.utime(member, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        observer.rehash_installed_exe(package.os_root)


@pytest.mark.parametrize("name", ["extra.dll", "scripts/extra.py", "manifest/extra.json", "manifest/Install-Manifest.json.extra"])
def test_extra_program_files_are_never_pruned_by_directory_or_extension(package, name):
    member = package.root / name
    member.parent.mkdir(parents=True, exist_ok=True)
    member.write_bytes(b"unlisted program file")
    with pytest.raises(ValueError, match="inventory|content"):
        observer.rehash_installed_exe(package.os_root)


@pytest.mark.parametrize("name", ["logs/current.log", "models/private.gguf", "state/session.sqlite3",
    "config/preferences.json", "chat_history/chat.json", "local_docs/private.txt", ".env", ".tailscale_joined",
    "node_config.json", "node_secrets.json", "qlh-control.sqlite3", "qlh-control.sqlite3-wal",
    "qlh-control.sqlite3-shm", "qlh-control.sqlite3-journal", "manifest/install-manifest.json"])
def test_only_fixed_runtime_data_and_exact_cyclic_metadata_are_excluded(package, name):
    member = package.root / name
    member.parent.mkdir(parents=True, exist_ok=True)
    member.write_bytes(b"user bytes must not be inventoried")
    assert observer.rehash_installed_exe(package.os_root)["rehashed_files"] == package.manifest["files"]


def test_pure_gate_and_filesystem_helper_share_the_fixed_exclusion_contract():
    contract = observer._frozen_contract()
    assert contract._FROZEN_USER_ROOTS == frozen.EXCLUDED_TOP_LEVEL_DIRECTORIES
    assert contract._FROZEN_USER_FILES == frozen.EXCLUDED_TOP_LEVEL_FILES
    assert contract._frozen_excluded("manifest/install-manifest.json")
    for name in ["manifest/runtime-profile.json", "_internal/config/default.json", "_internal/models/program.bin",
                 "scripts/__pycache__/probe.pyc", "other/product-package-observation.json"]:
        assert not contract._frozen_excluded(name)


@pytest.mark.parametrize("change", ["missing", "relative", "caller-path", "pid", "created-at", "os-path"])
def test_fresh_root_is_bound_to_os_process_executable_not_arbitrary_caller_configuration(package, change, monkeypatch):
    root = deepcopy(package.os_root)
    if change == "missing": root.pop("executable")
    if change == "relative": root["executable"] = "QLH.exe"
    if change == "caller-path": root["executable"] = str(package.root / "other.exe")
    if change == "pid": package.process.pid = 102
    if change == "created-at": package.process.create_time = lambda: 11.0
    if change == "os-path": package.process.exe = lambda: str(package.root / "other.exe")
    with pytest.raises(ValueError):
        observer.rehash_installed_exe(root)


def test_metadata_cannot_change_between_filesystem_verification_and_raw_text_capture(package, monkeypatch):
    original = frozen._verify_package
    def verify(root):
        captured = original(root)
        marker = root / frozen.MANIFEST_NAME
        marker.write_bytes(marker.read_bytes() + b"\n")
        return captured
    monkeypatch.setattr(frozen, "_verify_package", verify)
    with pytest.raises(ValueError, match="metadata changed"):
        observer.rehash_installed_exe(package.os_root)


def test_os_identity_is_queried_with_fresh_process_objects_before_and_after_hashing(package, monkeypatch):
    calls = []
    def process(pid):
        calls.append(pid)
        return SimpleNamespace(pid=pid, create_time=lambda: 10.0 if len(calls) == 1 else 11.0,
                               exe=lambda: str(package.exe), cwd=lambda: str(package.root))
    monkeypatch.setattr(psutil, "Process", process)
    with pytest.raises(ValueError, match="identity/executable changed"):
        observer.rehash_installed_exe(package.os_root)
    assert calls == [101, 101]


def test_full_filesystem_rehash_detects_mutation_during_capture(package, monkeypatch):
    original = frozen._hash_member
    mutated = False
    def changed(root, name, **kwargs):
        nonlocal mutated
        result = original(root, name, **kwargs)
        if name == "QLH.exe" and not mutated:
            mutated = True
            (root / "_internal/PYZ.pyz").write_bytes(b"changed while hashing")
        return result
    monkeypatch.setattr(frozen, "_hash_member", changed)
    with pytest.raises(ValueError, match="changed|differs"):
        observer.rehash_installed_exe(package.os_root)


@pytest.fixture
def collector(package, monkeypatch):
    calls = []
    def request(base, nonce):
        calls.append(nonce)
        return {"nonce": nonce, "pid": 101, "created_at": 10.0, "captured_at": 100.0, "errors": [],
                "source": deepcopy(package.source),
                "runtime": {"node_id": "master", "operation_id": "op", "config_id": "cfg", "generation": 3,
                            "captured_at": 100.0, "artifact": {"path": "C:/segment.gguf", "sha256": "b" * 64}}}
    monkeypatch.setattr(observer, "_request", request)
    monkeypatch.setattr(observer, "rehash_artifact", lambda runtime: {"sha256": "b" * 64, "files": []})
    facts = {"started_at": 100.0, "finished_at": 101.0, "identity_before": {"pid": 101, "created_at": 10.0},
             "identity_after": {"pid": 101, "created_at": 10.0},
             "process_tree": {"complete": True, "processes": [deepcopy(package.os_root)]}}
    monkeypatch.setattr(observer, "process_observation", lambda *_: deepcopy(facts))
    return SimpleNamespace(calls=calls, request=request, facts=facts)


def test_collector_retains_api_receipts_and_publishes_independent_full_package_facts(package, collector):
    result = observer.observe_pc(node_id="master", platform="master")
    assert result["collection_errors"] == []
    assert len(collector.calls) == 2 and collector.calls[0] != collector.calls[1]
    assert result["facts"]["source"] == package.source
    assert result["receipts"][0]["source"] == result["receipts"][1]["source"] == package.source
    assert result["facts"]["package_rehash"]["rehashed_files"] == package.manifest["files"]
    assert [c["command"][0] for c in result["commands"]] == ["HTTP GET", "psutil", "external-python-observer",
                                                              "independent-artifact-rehash", "HTTP GET"]
    command = result["commands"][2]
    assert json.loads(command["stdout"]) == result["facts"]["package_rehash"]
    assert command["returncode"] == 0


def test_collector_cannot_trust_api_rehashed_files_when_disk_bytes_differ(package, collector):
    (package.root / "_internal/native.pyd").write_bytes(b"changed frozen program")
    result = observer.observe_pc(node_id="master", platform="master")
    assert result["collection_errors"]
    assert result["commands"][-1]["returncode"] == 1
    assert result["commands"][-1]["command"][1] == "independent-installed-exe-rehash"
    assert result["receipts"][0]["source"] == package.source
    assert "package_rehash" not in result["facts"]


def test_collector_never_uses_api_source_root_to_select_filesystem_path(package, collector):
    package.source["source_root"] = "C:/attacker/elsewhere"
    result = observer.observe_pc(node_id="master", platform="master")
    assert result["collection_errors"] == []
    assert result["facts"]["package_rehash"]["source_root"] == str(package.root)
    assert result["facts"]["source"]["source_root"] == "C:/attacker/elsewhere"
    # Pure validation rejects these contradictory facts; the controller keeps
    # the endpoint receipt verbatim instead of silently repairing its source.
    with pytest.raises(ValueError, match="source root"):
        observer._frozen_contract()._installed_exe_source(result["facts"]["source"], package.os_root,
            REVISION, result["facts"]["package_rehash"], (result["started_at"], result["finished_at"]))


def test_frozen_collector_requires_executable_without_strengthening_legacy_mock_contract(collector):
    collector.facts["process_tree"]["processes"][0].pop("executable")
    result = observer.observe_pc(node_id="master", platform="master")
    assert result["collection_errors"] and "path" in str(result["collection_errors"])


@pytest.mark.parametrize("change", ["source", "bool-to-int", "pid"])
def test_frozen_source_is_unchanged_across_api_bracketing_receipts(package, collector, monkeypatch, change):
    count = 0
    def request(base, nonce):
        nonlocal count
        count += 1
        value = collector.request(base, nonce)
        if count == 2:
            if change == "source": value["source"]["runtime_manifest_sha256"] = "f" * 64
            if change == "bool-to-int": value["source"]["source_dirty"] = 0
            if change == "pid": value["pid"] += 1
        return value
    monkeypatch.setattr(observer, "_request", request)
    result = observer.observe_pc(node_id="master", platform="master")
    assert result["collection_errors"]
    assert "changed" in str(result["collection_errors"]) or "restarted" in str(result["collection_errors"])
    assert len(result["receipts"]) == 2


def test_os_process_observation_records_executable_from_psutil(monkeypatch):
    process = SimpleNamespace(pid=101, create_time=lambda: 10.0, exe=lambda: "C:/OS-observed/QLH.exe",
        ppid=lambda: 1, cmdline=lambda: ["arbitrary-command-is-not-executable-proof"], cwd=lambda: "C:/other",
        children=lambda **kwargs: [], memory_maps=lambda **kwargs: [],
        net_connections=lambda **kwargs: [SimpleNamespace(status="LISTEN", laddr=("127.0.0.1", 8000), raddr=())])
    monkeypatch.setattr(psutil, "Process", lambda *_: process)
    result = observer.process_observation(101, 8000)
    assert result["process_tree"]["complete"]
    assert result["process_tree"]["processes"][0]["executable"] == "C:/OS-observed/QLH.exe"


def test_legacy_process_observation_records_exe_denial_without_requiring_frozen_provenance(monkeypatch):
    def denied():
        raise psutil.AccessDenied(pid=101)
    process = SimpleNamespace(pid=101, create_time=lambda: 10.0, exe=denied,
        ppid=lambda: 1, cmdline=lambda: ["source-runtime"], cwd=lambda: "C:/source",
        children=lambda **kwargs: [], memory_maps=lambda **kwargs: [],
        net_connections=lambda **kwargs: [SimpleNamespace(status="LISTEN", laddr=("127.0.0.1", 8000), raddr=())])
    monkeypatch.setattr(psutil, "Process", lambda *_: process)
    result = observer.process_observation(101, 8000)
    assert result["process_tree"]["complete"]
    root = result["process_tree"]["processes"][0]
    assert "executable" not in root and "AccessDenied" in root["executable_error"]
    with pytest.raises(ValueError, match="path"):
        observer.rehash_installed_exe(root)


@pytest.mark.parametrize("text", ['{"source":{"pid":101,"pid":102}}', '{"source":{"value":NaN}}'])
def test_http_parser_does_not_erase_duplicate_source_keys_or_nonfinite_values(monkeypatch, text):
    class Response(io.StringIO):
        def __enter__(self): return self
        def __exit__(self, *_): self.close()
    monkeypatch.setattr(observer.urllib.request, "urlopen", lambda *args, **kwargs: Response(text))
    with pytest.raises(ValueError, match="duplicate|non-finite"):
        observer._request("http://127.0.0.1:8000", "test-nonce")


def test_actual_builder_and_filesystem_observer_outputs_satisfy_the_pure_source_gate(package):
    fresh = observer.rehash_installed_exe(package.os_root)
    contract = observer._frozen_contract()
    root = deepcopy(package.os_root)
    info = contract._source(package.source, root, "master", REVISION, fresh,
                            (fresh["started_at"], fresh["finished_at"]))
    assert info == package.source["build_info"]
    assert package.source["manifest_sha256"] == fresh["manifest_sha256"]


@pytest.mark.parametrize("cwd", ["C:/different-os-working-directory", "relative-directory"])
def test_independent_rehash_rejects_initial_actual_process_cwd_before_package_reads(package, monkeypatch, cwd):
    package.process.cwd = lambda: cwd
    monkeypatch.setattr(frozen, "_verify_package", lambda *_: pytest.fail("invalid process cwd reached package hashing"))
    with pytest.raises(ValueError, match="cwd|absolute"):
        observer.rehash_installed_exe(package.os_root)


def test_independent_rehash_rejects_actual_cwd_drift_during_package_hashing(package, monkeypatch):
    calls = []
    current = {"cwd": str(package.root)}
    def process(pid):
        calls.append(pid)
        return SimpleNamespace(pid=pid, create_time=lambda: 10.0, exe=lambda: str(package.exe),
                               cwd=lambda: current["cwd"])
    verify = frozen._verify_package
    def verify_then_move_cwd(root):
        result = verify(root)
        current["cwd"] = str(package.root.parent)
        return result
    monkeypatch.setattr(psutil, "Process", process)
    monkeypatch.setattr(frozen, "_verify_package", verify_then_move_cwd)
    with pytest.raises(ValueError, match="OS process cwd differs from executable parent"):
        observer.rehash_installed_exe(package.os_root)
    assert calls == [101, 101]


def test_independent_rehash_normalizes_actual_process_cwd_before_comparing_to_executable_parent(package):
    package.process.cwd = lambda: str(package.root).upper().replace("/", "\\")
    assert observer.rehash_installed_exe(package.os_root)["source_root"] == str(package.root)


def test_collector_records_cwd_drift_as_failed_external_rehash(package, collector, monkeypatch):
    verify = frozen._verify_package
    def verify_then_move_cwd(root):
        result = verify(root)
        package.process.cwd = lambda: str(package.root.parent)
        return result
    monkeypatch.setattr(frozen, "_verify_package", verify_then_move_cwd)
    result = observer.observe_pc(node_id="master", platform="master")
    assert "cwd differs" in str(result["collection_errors"])
    assert result["commands"][-1]["returncode"] == 1
    assert result["commands"][-1]["command"][1] == "independent-installed-exe-rehash"
    assert "package_rehash" not in result["facts"]
