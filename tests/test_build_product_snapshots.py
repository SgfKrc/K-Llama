"""Offline regressions for the observer producer, never physical acceptance."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


producer = load_module("snapshot_producer_test", ROOT / "scripts/build_product_snapshots.py")
observations = __import__("package_contract.product_observations", fromlist=["derive_node_observation"])
fixtures = load_module("snapshot_fixture_test", ROOT / "tests/product_observation_fixtures.py")
REVISION = "a" * 40
IDENTITY = {"operation_id": "supply_observer", "config_id": "config_observer", "generation": 1792124484513344701}


def example(mode="LL"):
    assignments = [{"node_id": node, "engine": "pytorch" if mode == "DL" and node == "master" else "llama_cpp",
                    "start_layer": lo, "end_layer": hi}
                   for node, lo, hi in [("master", 0, 4), ("rog-5060", 4, 8), ("android-d83bf4d0", 8, 32)]]
    intent = {**IDENTITY, "status": "ready", "loaded_ready": True,
              "plan": {"total_layers": 32, "assignments": assignments},
              "nodes": {a["node_id"]: {"artifact_sha256": "b"*64} for a in assignments}}
    raws = {}
    for assignment in assignments:
        raw, _, _, addresses = fixtures.node_observation(IDENTITY, assignment, assignments, REVISION, 32)
        raws[assignment["node_id"]] = raw
    return intent, raws, addresses


def expected(intent, node, addresses):
    assignment = next(a for a in intent["plan"]["assignments"] if a["node_id"] == node)
    return dict(expected_revision=REVISION, expected_node_id=node, expected_identity=IDENTITY,
                expected_assignment={**assignment, "_total_layers": 32, "artifact_sha256": "b"*64},
                node_addresses=addresses)


@pytest.mark.parametrize("mode", ["LL", "DL"])
def test_producer_preserves_derived_facts_raw_refs_and_original_windows(tmp_path, mode):
    intent, raws, addresses = example(mode)
    report = producer.produce_snapshots(intent=intent, raw_nodes=raws, node_addresses=addresses,
                                       revision=REVISION, out_dir=tmp_path, mode=mode)
    assert report["passed"], report
    topology = json.loads((tmp_path / "topology.json").read_text(encoding="utf-8"))
    native = json.loads((tmp_path / "native.json").read_text(encoding="utf-8"))
    assert topology["node_addresses"] == native["node_addresses"] == addresses
    assert topology["captured_at"] == native["captured_at"] == 106.0
    for top, nat in zip(topology["nodes"], native["nodes"]):
        node = top["node_id"]
        derived = observations.derive_node_observation(raws[node], **expected(intent, node, addresses))
        assert top["raw_evidence"] == nat["raw_evidence"]
        ref = top["raw_evidence"]
        payload = (tmp_path / ref["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == ref["sha256"]
        assert json.loads(payload) == raws[node]
        assert {k:v for k,v in top.items() if k != "raw_evidence"} == derived["topology"]
        assert {k:v for k,v in nat.items() if k != "raw_evidence"} == derived["native"]


@pytest.mark.parametrize("legacy", [
    {"role": "client", "ssh_returncode": 0, "probe_stdout_tail": '"source_files_rehashed_by_this_probe": false\n}',
     "synced_revision": REVISION, "sync_verification": "ref_total=197 match=197 mismatch=0 missing=0"},
    {"role": "client", "probe_returncode": 0, "probe": {"passed": True, "observation": {
        "pid": 14352, "model": {"loaded": True, "full_model_loaded": False, "layer_range": [8,32]},
        "process": {"maps_complete": True, "children_complete": False}}},
     "installed_apk_build_info": {"source_revision": REVISION, "source_dirty": True}},
    {"role": "master", "git_head": REVISION, "processes": [
        {"pid": 13216, "model_mappings": [], "maps_complete": True, "identity_stable": True}]},
])
def test_actual_legacy_failure_combinations_never_become_green(tmp_path, legacy):
    intent, raws, addresses = example()
    raws["rog-5060"] = legacy
    original = copy.deepcopy(legacy)
    report = producer.produce_snapshots(intent=intent, raw_nodes=raws, node_addresses=addresses,
                                       revision=REVISION, out_dir=tmp_path, mode="LL")
    assert report["passed"] is False and "unsupported raw schema" in str(report["errors"])
    assert not (tmp_path / "topology.json").exists()
    assert not (tmp_path / "native.json").exists()
    assert legacy == original
    assert json.loads((tmp_path / "ll-nodes/rog-5060.json").read_text()) == legacy


@pytest.mark.parametrize("change,reason", [
    (lambda r: r["facts"]["source"].pop("source_dirty"), "source_dirty"),
    (lambda r: r["facts"]["source"].update(source_dirty=True), "source_dirty"),
    (lambda r: r["facts"]["source"].update(runtime_manifest_sha256="0"*64), "running process"),
    (lambda r: r["facts"]["source"].update(rehashed_files=[]), "rehash"),
    (lambda r: r["facts"]["source"].update(rehashed_files=[{**r["facts"]["source"]["rehashed_files"][0], "sha256": "c"*64}]), "rehash"),
    (lambda r: r["facts"].update(identity_after={**r["facts"]["identity_after"], "created_at": 11}), "identity changed"),
    (lambda r: r["facts"]["runtime"].update(operation_id="another"), "another operation"),
    (lambda r: r["facts"]["runtime"].update(generation=True), "generation"),
    (lambda r: r["facts"]["runtime"].update(total_layers=64), "total_layers"),
    (lambda r: r["facts"]["process_tree"].update(processes=[]), "processes"),
    (lambda r: r["facts"]["process_tree"]["processes"][0].update(children_complete=False), "children_complete"),
    (lambda r: r["facts"]["process_tree"]["processes"][0].update(child_pids=[999]), "child process"),
    (lambda r: r["facts"]["process_tree"]["processes"][0].update(maps_complete=False), "maps_complete"),
    (lambda r: r["facts"]["process_tree"]["processes"][0].update(connections_complete=False), "connections_complete"),
    (lambda r: r["facts"]["process_tree"]["processes"][0].update(connections=[]), "OS cluster peer"),
    (lambda r: r["facts"]["runtime"].update(captured_at=1000), "window"),
    (lambda r: r["commands"][0].update(returncode=1), "command failed"),
    (lambda r: r["commands"][0].pop("stderr"), "stdout/stderr"),
    (lambda r: r.update(collection_errors=["probe permission denied"]), "collection reported"),
])
def test_valid_schema_cannot_hide_missing_or_contradictory_evidence(change, reason):
    intent, raws, addresses = example()
    raw = copy.deepcopy(raws["rog-5060"])
    change(raw)
    with pytest.raises(observations.ObservationError, match=reason):
        observations.derive_node_observation(raw, **expected(intent, "rog-5060", addresses))


@pytest.mark.parametrize("stdout", [
    'starting probe\n{\n  "pid": 2268,\n  "source_manifest": {\n    "source_files_rehashed_by_this_probe": false\n  }\n}\nprobe finished\n',
    '{"pid":2268,"source_manifest":{"source_files_rehashed_by_this_probe":false}}',
])
def test_rog_keeps_full_multiline_json_and_stderr_without_revision_fallback(monkeypatch, stdout):
    execution = {"command": ["ssh"], "started_at": 100, "finished_at": 110, "returncode": 0,
                 "stdout": stdout, "stderr": "full remote diagnostic\n"}
    monkeypatch.setattr(producer, "run", lambda *a, **kw: execution)
    result = producer.rog_evidence("supply_test", "DL-ready", "ROG@100.116.17.12", "C:/Python/python.exe", "C:/qlh/probe.py")
    assert result["probe"]["pid"] == 2268
    assert result["probe"]["source_manifest"]["source_files_rehashed_by_this_probe"] is False
    assert result["ssh"]["stdout"] == stdout and result["ssh"]["stderr"] == execution["stderr"]
    assert "synced_revision" not in result and "sync_method" not in result


@pytest.mark.parametrize("stdout", [
    '{\n  "pid": 2268,\n', '{"pid":1}\n{"pid":2}', '{"pid":1,"pid":2}', '{"at":NaN}', 'not JSON',
])
def test_bad_or_ambiguous_json_is_not_silently_replaced(stdout):
    with pytest.raises(observations.ObservationError):
        observations.parse_json_output(stdout)


def test_snapshot_booleans_cannot_override_raw_and_times():
    intent, raws, addresses = example()
    kw = expected(intent, "rog-5060", addresses)
    raw = raws["rog-5060"]
    snapshot = observations.derive_node_observation(raw, **kw)
    snapshot["native"]["source_dirty"] = 0
    snapshot["topology"]["captured_at"] = 999
    assert observations.validate_node_snapshot(raw, snapshot, **kw) == [
        "topology.captured_at: snapshot differs from raw observation",
        "native.source_dirty: snapshot differs from raw observation"]


def test_no_manifest_cli_revision_or_empty_mmap_can_prove_dl():
    intent, raws, addresses = example("DL")
    raw = raws["master"]
    raw["facts"]["runtime"].pop("materialization")
    with pytest.raises(observations.ObservationError, match="selective tensor names"):
        observations.derive_node_observation(raw, **expected(intent, "master", addresses))


def test_native_child_is_observed_and_bound_to_loader_not_only_master():
    intent, raws, addresses = example()
    raw = copy.deepcopy(raws["master"])
    root = raw["facts"]["process_tree"]["processes"][0]
    child = copy.deepcopy(root)
    child.update(pid=201, parent_pid=root["pid"], connections=[], child_pids=[])
    child["identity_after"] = {"pid": 201, "created_at": child["created_at"]}
    root.update(child_pids=[201], model_mappings=[])
    raw["facts"]["process_tree"]["processes"].append(child)
    raw["facts"]["runtime"]["loader_pid"] = 201
    result = observations.derive_node_observation(raw, **expected(intent, "master", addresses))
    assert result["native"]["complete"] is True
    child["maps_complete"] = False
    with pytest.raises(observations.ObservationError, match="PID 201 maps_complete"):
        observations.derive_node_observation(raw, **expected(intent, "master", addresses))


@pytest.mark.parametrize("mode", ["LL", "DL"])
def test_cli_expected_revision_does_not_replace_running_source(tmp_path, mode):
    intent, raws, addresses = example(mode)
    report = producer.produce_snapshots(intent=intent, raw_nodes=raws, node_addresses=addresses,
                                       revision="f"*40, out_dir=tmp_path, mode=mode)
    assert not report["passed"] and "source revision" in str(report["errors"])
    assert not (tmp_path / "topology.json").exists()


def test_cannot_overwrite_or_relabel_existing_evidence(tmp_path):
    intent, raws, addresses = example()
    existing = tmp_path / "topology.json"
    existing.write_bytes(b"historical evidence")
    with pytest.raises(observations.ObservationError, match="refusing existing"):
        producer.produce_snapshots(intent=intent, raw_nodes=raws, node_addresses=addresses,
                                   revision=REVISION, out_dir=tmp_path, mode="LL")
    assert existing.read_bytes() == b"historical evidence"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["topology.json"]


def test_run_retains_nonzero_and_complete_stdout_stderr(monkeypatch):
    text = "probe output\n" * 1000
    monkeypatch.setattr(producer.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=7, stdout=text, stderr=text))
    record = producer.run(["synthetic-observer"])
    assert record["returncode"] == 7 and record["stdout"] == text and record["stderr"] == text
    assert record["started_at"] <= record["finished_at"]


@pytest.mark.parametrize("mutate", ["none", "path", "pid", "sha"])
@pytest.mark.parametrize("probe_kind", ["legacy", "product", "wrong-generation", "partial-tree"])
def test_android_build_info_comes_from_installed_bytes_under_stable_identity(tmp_path, monkeypatch, mutate, probe_kind):
    package = "com.qlh.inference.debug"
    path = "/data/app/com.qlh.inference.debug-test/base.apk"
    info = {"source_revision": REVISION, "source_dirty": True}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as apk:
        apk.writestr("assets/build-info.json", json.dumps(info))
    apk_bytes = archive.getvalue()
    sha = hashlib.sha256(apk_bytes).hexdigest()
    raw_path = tmp_path / "probe.json"
    seen = []
    counters = {"pid": 0, "path": 0, "sha": 0}

    def fake_run(command, **kw):
        seen.append(command)
        output = ""
        if str(producer.ANDROID_PROBE) in command:
            assert "--product" in command and "--require-process-tree" in command
            assert kw.get("timeout") == 150
            observation = {"pid": 14352, "package_name": package, "process": {"children_complete": False}}
            if probe_kind != "legacy":
                runtime = {"pid": 14352, "created_at": 12345, "node_id": "android-d83bf4d0",
                           **IDENTITY, "artifact": {"path": "/data/user/0/segment.gguf", "sha256": "b"*64}}
                identity = {"pid": 14352, "created_at": 999 if probe_kind == "wrong-generation" else 12345}
                observation.update(runtime_receipt=runtime, process={"identity_before": identity,
                    "identity_after": identity, "complete": probe_kind != "partial-tree", "processes": [{"pid": 14352}],
                    "errors": ["descendant inaccessible"] if probe_kind == "partial-tree" else []})
            raw_path.write_text(json.dumps({"passed": True, "observation": observation}))
        elif command[3:5] == ["shell", "pidof"]:
            counters["pid"] += 1
            output = "999\n" if mutate == "pid" and counters["pid"] == 2 else "14352\n"
        elif "stat" in command[-1] and command[3:5] == ["shell", "run-as"]:
            output = "14352 (worker process) " + " ".join(["S"] + ["0"]*18 + ["12345"]) + "\n"
        elif command[3:6] == ["shell", "pm", "path"]:
            counters["path"] += 1
            output = "package:" + (path.replace("test", "changed") if mutate == "path" and counters["path"] == 2 else path) + "\n"
        elif command[3:5] == ["shell", "sha256sum"]:
            counters["sha"] += 1
            output = ("0"*64 if mutate == "sha" and counters["sha"] == 2 else sha) + "  " + path + "\n"
        elif command[3] == "pull":
            Path(command[-1]).write_bytes(apk_bytes)
        else:
            raise AssertionError(command)
        return {"command": command, "started_at": 100, "finished_at": 101, "returncode": 0, "stdout": output, "stderr": ""}

    monkeypatch.setattr(producer, "run", fake_run)
    evidence = producer.android_evidence("HA26PYZQ", "synthetic-adb", raw_path)
    if mutate == "none":
        binding = evidence["installed_package_binding"]
        assert binding["build_info"] == info and binding["pulled_sha256"] == sha
        assert binding["build_info"]["source_dirty"] is True
        assert binding["path_before"] == binding["path_after"] == path
    else:
        assert "installed_package_binding" not in evidence
        assert any("changed" in e or "identity" in e for e in evidence["collection_errors"])
    if mutate == "none" and probe_kind == "product":
        assert evidence["collection_errors"] == []
        assert evidence["kind"] == "product-node-observation"
        assert evidence["platform"] == "android" and evidence["node_id"] == "android-d83bf4d0"
        assert evidence["facts"]["runtime"]["operation_id"] == IDENTITY["operation_id"]
        assert evidence["facts"]["source"] == evidence["installed_package_binding"]
        assert evidence["facts"]["process_tree"]["complete"] is True
    else:
        assert evidence["collection_errors"]
        if mutate == "none" and probe_kind == "wrong-generation":
            assert "another process generation" in str(evidence["collection_errors"])
        if mutate == "none" and probe_kind == "partial-tree":
            assert "descendant inaccessible" in str(evidence["collection_errors"])
    assert all("install" not in command and "force-stop" not in command for command in seen)
    assert any("pull" in command for command in seen)


def test_intent_request_uses_observer_bearer_header_without_exposing_token(monkeypatch):
    import io
    captured = []
    class Response(io.StringIO):
        def __enter__(self): return self
        def __exit__(self, *_): self.close()
    def urlopen(request, **kwargs):
        captured.append(request)
        return Response('{"operation_id":"actual"}')
    monkeypatch.setenv("QLH_PRODUCT_AUTH_TOKEN", "synthetic-secret")
    monkeypatch.setattr(producer.urllib.request, "urlopen", urlopen)
    assert producer.api("models/intent/current") == {"operation_id": "actual"}
    assert captured[0].get_header("Authorization") == "Bearer synthetic-secret"
    assert "synthetic-secret" not in captured[0].full_url
