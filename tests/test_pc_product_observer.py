"""Collector tests bind OS facts to actual runtime receipts; no synthetic product pass."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pc_observer_test", ROOT / "scripts/package_contract/pc_product_observer.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def receipt(nonce, **runtime):
    return {"nonce": nonce, "pid": 101, "created_at": 10.0, "captured_at": 101,
            "runtime": {"node_id": "master", "operation_id": "op", "config_id": "cfg", "generation": 3,
                        "captured_at": 101, "artifact": {"path": "C:/segment.gguf", "sha256": "b"*64}, **runtime},
            "source": {"pid": 101, "created_at": 10, "manifest_sha256": "a"*64}, "errors": []}


@pytest.fixture
def observed(monkeypatch):
    calls = []
    monkeypatch.setattr(observer, "_request", lambda base, nonce: calls.append(nonce) or receipt(nonce))
    monkeypatch.setattr(observer, "rehash_artifact", lambda runtime: {"sha256": "b"*64, "files": []})
    facts = {"started_at": 102, "finished_at": 103, "identity_before": {"pid":101,"created_at":10},
             "identity_after": {"pid":101,"created_at":10}, "process_tree":{"complete": True, "processes":[{"pid":101}]}}
    monkeypatch.setattr(observer, "process_observation", lambda *_: deepcopy(facts))
    return facts, calls


def test_receipts_bracket_os_observation_without_relabelling_sources(observed):
    facts, calls = observed
    result = observer.observe_pc(node_id="master", platform="master", operation_id="op")
    assert result["collection_errors"] == [] and len(calls) == 2 and calls[0] != calls[1]
    assert result["facts"]["runtime"]["generation"] == 3
    assert result["facts"]["source"]["manifest_sha256"] == "a"*64
    assert [c["command"][0] for c in result["commands"]] == ["HTTP GET", "psutil", "independent-artifact-rehash", "HTTP GET"]
    assert result["facts"]["process_tree"] == facts["process_tree"]
    assert len(result["receipts"]) == 2


@pytest.mark.parametrize("change,reason", [("nonce","identity"),("errors","identity"),("node","node ID"),
    ("operation","operation"),("generation","runtime/source changed"),("source","runtime/source changed"),
    ("pid","restarted"),("os-partial","incomplete"),("denied","denied")])
def test_changed_or_missing_receipt_and_partial_os_capture_rejected(observed, monkeypatch, change, reason):
    facts, calls = observed
    index = 0
    def request(base, nonce):
        nonlocal index
        index += 1
        value = receipt(nonce)
        if change == "nonce": value["nonce"] = "bad"
        if change == "errors": value["errors"] = ["unavailable"]
        if change == "node": value["runtime"]["node_id"] = "other"
        if change == "operation": value["runtime"]["operation_id"] = "other"
        if index == 2:
            if change == "generation": value["runtime"]["generation"] += 1
            if change == "source": value["source"]["manifest_sha256"] = "b"*64
            if change == "pid": value["pid"] += 1
        if change == "denied": raise PermissionError("denied")
        return value
    if change == "os-partial": facts["process_tree"]["complete"] = False
    monkeypatch.setattr(observer,"_request", request)
    result = observer.observe_pc(node_id="master", platform="master", operation_id="op")
    assert reason in str(result["collection_errors"])
    assert len(result["commands"]) >= 1


@pytest.mark.parametrize("base", ["https://127.0.0.1:8000","http://100.1.2.3:8000", "http://127.0.0.1:8000/extra"])
def test_only_loopback_endpoint_is_contacted(base, monkeypatch):
    monkeypatch.setattr(observer, "_request", lambda *_: pytest.fail("unsafe address contacted"))
    with pytest.raises(ValueError, match="loopback"):
        observer.observe_pc(node_id="master",platform="master",base=base)


def test_actual_process_tree_includes_native_style_child_and_owned_listener(tmp_path):
    import os
    import subprocess
    import sys
    code = '''import os, socket, subprocess, sys
listener=socket.socket()
listener.bind(("127.0.0.1",0))
listener.listen()
child=subprocess.Popen([sys.executable,"-B","-c","import sys; print('ready', flush=True); sys.stdin.readline()"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
assert child.stdout.readline().strip()=="ready"
print(os.getpid(), listener.getsockname()[1], flush=True)
try: sys.stdin.readline()
finally:
 child.stdin.close()
 child.wait(timeout=5)
 child.stdout.close()
'''
    process = subprocess.Popen([sys.executable,"-B","-c",code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    try:
        runtime_pid, port = map(int, process.stdout.readline().split())
        result = observer.process_observation(runtime_pid,port)
        assert result["process_tree"]["complete"], result
        pids = {entry["pid"] for entry in result["process_tree"]["processes"]}
        assert runtime_pid in pids and len(pids) >= 2
        assert result["identity_before"] == result["identity_after"]
        assert all(entry["maps_complete"] and entry["connections_complete"] and entry["children_complete"]
                   for entry in result["process_tree"]["processes"])
        wrong_port = observer.process_observation(runtime_pid,1)
        assert "does not own" in wrong_port["error"]
    finally:
        process.stdin.close()
        process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()


def test_live_gguf_rehash_rejects_expected_hash_without_reading_matching_bytes(tmp_path):
    import hashlib
    artifact = tmp_path / "segment.gguf"
    artifact.write_bytes(b"proper segment bytes")
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    runtime = {"artifact": {"path": str(artifact), "sha256": sha}}
    assert observer.rehash_artifact(runtime)["sha256"] == sha
    artifact.write_bytes(b"changed segment bytes")
    with pytest.raises(ValueError, match="loaded identity"):
        observer.rehash_artifact(runtime)


def test_collection_retains_failed_independent_hash_command(observed, monkeypatch):
    def fail(_):
        raise ValueError("artifact changed")
    monkeypatch.setattr(observer, "rehash_artifact", fail)
    result = observer.observe_pc(node_id="master", platform="master")
    assert "artifact changed" in str(result["collection_errors"])
    assert result["commands"][-1]["returncode"] == 1
    assert result["facts"].get("runtime") is None


def test_authorization_is_sent_in_header_without_recording_secret(monkeypatch):
    import io
    captured = []
    class Response(io.StringIO):
        def __enter__(self): return self
        def __exit__(self, *_): self.close()
    def urlopen(request, **kwargs):
        captured.append(request)
        return Response('{"observed":true}')
    monkeypatch.setenv("QLH_PRODUCT_AUTH_TOKEN", "local-secret")
    monkeypatch.setattr(observer.urllib.request, "urlopen", urlopen)
    assert observer._request("http://127.0.0.1:8000", "a"*32) == {"observed": True}
    assert captured[0].get_header("Authorization") == "Bearer local-secret"
    assert "local-secret" not in captured[0].full_url


def test_safetensors_rehash_uses_supply_weight_inventory_not_model_cache(tmp_path):
    import hashlib
    import json
    import model_supply_assets as supply
    # Minimal valid supply contract; weights are only read as bytes here.
    names = {"segment.safetensors": b"layer tensor bytes", "config.json": b"{}",
             "model.safetensors.index.json": b"{}", "tokenizer.json": b"{}"}
    for name, content in names.items():
        (tmp_path / name).write_bytes(content)
    files = [{"path": name, "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
              "kind": "weights" if name.endswith(".safetensors") else "support"}
             for name, content in sorted(names.items())]
    manifest = supply._base_manifest(source_model_id="tiny", source_model_sha256="a"*64,
        tokenizer_sha256=hashlib.sha256(b"{}").hexdigest(), start_layer=0, end_layer=1,
        total_layers=2, operation_id="op", node_id="master")
    manifest.update(format="safetensors", hidden_size=4, segment_mode="head", layer_weight_bytes=[18,0],
                    files=files, artifact_sha256=supply._weight_digest(files))
    manifest["manifest_sha256"] = supply._digest(manifest)
    marker = tmp_path / "supply.manifest.json"
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = {"total_layers": 2, "artifact": {"path": str(tmp_path), "sha256": manifest["artifact_sha256"],
                                               "layer_range": [0,1]}}
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert observer.rehash_artifact(runtime)["sha256"] == manifest["artifact_sha256"]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    (tmp_path / "segment.safetensors").write_bytes(b"wrong tensor bytes")
    with pytest.raises(ValueError, match="size|digest"):
        observer.rehash_artifact(runtime)


def test_unlisted_file_in_segment_cannot_be_hidden_from_rehash(tmp_path):
    # Supply validation failure is explicit rather than falling back to cached hashes.
    (tmp_path / "supply.manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        observer.rehash_artifact({"artifact": {"path": str(tmp_path), "sha256": "a"*64}})


def test_http_failure_retains_endpoint_body_in_raw_command(observed, monkeypatch):
    import io
    body = b'{"detail":{"code":"PRODUCT_OBSERVATION_UNAVAILABLE","message":"worker is fenced"}}'
    def request(*_):
        raise observer.urllib.error.HTTPError("http://127.0.0.1:8000", 409, "Conflict", {}, io.BytesIO(body))
    monkeypatch.setattr(observer, "_request", request)
    result = observer.observe_pc(node_id="master", platform="master")
    assert result["commands"][0]["returncode"] == 1
    assert result["commands"][0]["stdout"] == body.decode()
    assert "409" in str(result["collection_errors"])


@pytest.fixture
def multi_file_supply(tmp_path):
    """Real v1 supply schema with the four-shard/auxiliary-file lab layout.

    Byte payloads are deliberately small; this fixture tests the digest contract,
    not tensor loading (covered by the real selective-load integration test).
    """
    import hashlib
    import json
    import model_supply_assets as supply
    import model_sync
    # Autouse logging owns tmp_path/logs; keep it outside the exact artifact inventory.
    tmp_path = tmp_path / "artifact"
    tmp_path.mkdir()
    contents = {f"assignment-{i:05d}-of-00004.safetensors": f"shard {i} payload".encode()
                for i in range(1, 5)}
    contents.update({"config.json": b'{"num_hidden_layers":4}',
                     "model.safetensors.index.json": b'{"weight_map":{}}',
                     "tokenizer.json": b'{"version":"1.0"}',
                     "tokenizer_config.json": b'{"model_max_length":64}',
                     "chat_template.jinja": b"{{ messages }}",
                     "merges.txt": b"#version: 0.2\na b\n",
                     "vocab.json": b'{"a":0,"b":1}'})
    for name, content in contents.items():
        (tmp_path / name).write_bytes(content)
    files = [{"path": name, "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
              "kind": "weights" if name.endswith(".safetensors") else "support"}
             for name, content in sorted(contents.items())]
    manifest = supply._base_manifest(source_model_id="tiny", source_model_sha256="a"*64,
        tokenizer_sha256=hashlib.sha256(contents["tokenizer.json"]).hexdigest(), start_layer=0, end_layer=2,
        total_layers=4, operation_id="op", node_id="master")
    manifest.update(format="safetensors", hidden_size=4, segment_mode="head", layer_weight_bytes=[18,18,0,0],
                    files=files, artifact_sha256=supply._weight_digest(files))
    manifest["manifest_sha256"] = supply._digest(manifest)
    marker = tmp_path / "supply.manifest.json"
    marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assert supply.validate_supply_manifest(manifest) == manifest
    directory_sha = model_sync.compute_model_sha256(str(tmp_path), use_cache=False)
    assert directory_sha != manifest["artifact_sha256"]
    return SimpleNamespace(root=tmp_path, manifest=manifest, directory_sha=directory_sha, marker=marker)


def _supply_runtime(supply, *, directory=True):
    digest = supply.directory_sha if directory else supply.manifest["artifact_sha256"]
    return {"engine": "pytorch", "total_layers": 4,
            "artifact": {"path": str(supply.root), "sha256": digest, "layer_range": [0, 2]},
            "materialization": {"artifact_path": str(supply.root), "artifact_sha256": digest,
                                "layer_range": [0, 2]}}


@pytest.mark.parametrize("directory", [False, True], ids=["worker-supply-digest", "master-directory-digest"])
def test_multi_file_supply_rehash_binds_both_real_load_contracts_without_cache(multi_file_supply, monkeypatch, directory):
    import model_sync
    def cache_used(*args, **kwargs):
        pytest.fail("independent observer called the model/file hash cache")
    monkeypatch.setattr(model_sync, "compute_model_sha256", cache_used)
    monkeypatch.setattr(model_sync, "compute_file_sha256", cache_used)
    runtime = _supply_runtime(multi_file_supply, directory=directory)
    original = deepcopy(runtime)
    before = {p.name: p.read_bytes() for p in multi_file_supply.root.iterdir()}
    digest = observer.rehash_artifact(runtime)
    assert digest["sha256"] == multi_file_supply.manifest["artifact_sha256"]
    assert digest["directory_sha256"] == multi_file_supply.directory_sha
    assert len(digest["files"]) == 11
    assert digest["loaded_digest_scheme"] == ("model_sync_directory_v1" if directory else "safetensors_supply_weights_v1")
    derived = observer.bind_artifact_rehash(runtime, digest)
    assert derived["artifact"]["sha256"] == derived["artifact"]["rehashed_sha256"] == digest["sha256"]
    assert derived["materialization"]["artifact_sha256"] == digest["sha256"]
    if directory:
        assert derived["artifact"]["load_identity"] == {"sha256": multi_file_supply.directory_sha,
                                                          "scheme": "model_sync_directory_v1"}
        assert derived["materialization"]["loaded_artifact_sha256"] == multi_file_supply.directory_sha
    assert runtime == original
    assert {p.name: p.read_bytes() for p in multi_file_supply.root.iterdir()} == before


@pytest.mark.parametrize("member", ["assignment-00001-of-00004.safetensors", "assignment-00004-of-00004.safetensors",
                                   "config.json", "tokenizer.json", "merges.txt"])
def test_multi_file_rehash_detects_changed_bytes_even_with_same_size_mtime_and_warm_cache(multi_file_supply, member):
    import os
    path = multi_file_supply.root / member
    info = path.stat()
    content = path.read_bytes()
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert path.stat().st_size == info.st_size and path.stat().st_mtime_ns == info.st_mtime_ns
    with pytest.raises(ValueError, match="member digest"):
        observer.rehash_artifact(_supply_runtime(multi_file_supply))


def test_directory_load_digest_covers_marker_bytes_not_only_parsed_manifest(multi_file_supply):
    # Same manifest identity and weight inventory, different directory bytes.
    marker = multi_file_supply.marker
    marker.write_bytes(marker.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="loaded identity"):
        observer.rehash_artifact(_supply_runtime(multi_file_supply))


def test_directory_load_digest_rejects_resigned_support_with_unchanged_supply_weights(multi_file_supply):
    import hashlib
    import json
    import model_supply_assets as supply
    manifest = deepcopy(multi_file_supply.manifest)
    content = b'{"num_hidden_layers":999}'
    (multi_file_supply.root / "config.json").write_bytes(content)
    entry = next(item for item in manifest["files"] if item["path"] == "config.json")
    entry.update(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = supply._digest(manifest)
    assert supply.validate_supply_manifest(manifest)["artifact_sha256"] == multi_file_supply.manifest["artifact_sha256"]
    multi_file_supply.marker.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="loaded identity"):
        observer.rehash_artifact(_supply_runtime(multi_file_supply))


def test_directory_rehash_never_accepts_source_model_or_arbitrary_identity(multi_file_supply):
    runtime = _supply_runtime(multi_file_supply)
    runtime["artifact"]["sha256"] = multi_file_supply.manifest["source_model_sha256"]
    with pytest.raises(ValueError, match="loaded identity"):
        observer.rehash_artifact(runtime)


def test_directory_rehash_retains_exact_inventory_and_stability_guards(multi_file_supply, monkeypatch):
    import model_supply_assets as supply
    extra = multi_file_supply.root / "model.sha256"
    extra.write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        observer.rehash_artifact(_supply_runtime(multi_file_supply))
    extra.unlink()
    original_hash = supply.sha256_file
    def changing_hash(path):
        value = original_hash(path)
        if Path(path).name == "assignment-00001-of-00004.safetensors":
            multi_file_supply.marker.write_bytes(multi_file_supply.marker.read_bytes() + b"\n")
        return value
    monkeypatch.setattr(supply, "sha256_file", changing_hash)
    with pytest.raises(ValueError, match="changed during"):
        observer.rehash_artifact(_supply_runtime(multi_file_supply))


@pytest.mark.parametrize("field", ["artifact_sha256", "artifact_path", "layer_range", "engine"])
def test_supply_digest_conversion_requires_the_same_materialization_proof(multi_file_supply, field):
    runtime = _supply_runtime(multi_file_supply)
    rehashed = observer.rehash_artifact(runtime)
    if field == "engine":
        runtime[field] = "llama_cpp"
    else:
        runtime["materialization"][field] = [2, 4] if field == "layer_range" else "unbound"
    with pytest.raises(ValueError, match="materialization proof"):
        observer.bind_artifact_rehash(runtime, rehashed)


def test_collector_keeps_verbatim_directory_receipts_and_publishes_proven_supply_identity(multi_file_supply, observed, monkeypatch):
    facts, calls = observed
    loaded = _supply_runtime(multi_file_supply)
    monkeypatch.setattr(observer, "_request", lambda base, nonce: receipt(nonce, **deepcopy(loaded)))
    # The observed fixture stubs hashing; restore the real implementation for this capture.
    spec = importlib.util.spec_from_file_location("pc_observer_fresh", ROOT / "scripts/package_contract/pc_product_observer.py")
    real_observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real_observer)
    monkeypatch.setattr(observer, "rehash_artifact", real_observer.rehash_artifact)
    result = observer.observe_pc(node_id="master", platform="master")
    assert result["collection_errors"] == []
    derived = result["facts"]["runtime"]
    assert derived["artifact"]["sha256"] == multi_file_supply.manifest["artifact_sha256"]
    assert derived["materialization"]["loaded_artifact_sha256"] == multi_file_supply.directory_sha
    for value in result["receipts"]:
        assert value["runtime"]["artifact"]["sha256"] == multi_file_supply.directory_sha
        assert value["runtime"]["materialization"]["artifact_sha256"] == multi_file_supply.directory_sha
    import json
    assert json.loads(result["commands"][0]["stdout"]) == result["receipts"][0]
    assert json.loads(result["commands"][3]["stdout"]) == result["receipts"][1]


def test_unexpected_windows_mapping_retains_os_name_and_independent_proof_command(monkeypatch):
    import json
    observed = {"process_tree": {"processes": [{"pid": 101, "created_at": 10.0,
                "model_mappings": ["C:/retired/segment.gguf"]}]}}
    original_name = observed["process_tree"]["processes"][0]["model_mappings"].copy()
    runtime = {"engine": "llama_cpp", "artifact": {"path": "C:/active/segment.gguf"}}
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return {"mapped_path": kwargs["mapped_path"], "sha256": kwargs["expected_sha256"]}
    monkeypatch.setattr(observer, "_capture_windows_mapping", capture)
    commands = []
    observer.observe_mapping_aliases(observed, runtime, {"sha256": "b"*64, "size_bytes": 8193}, commands)
    assert len(calls) == 1 and calls[0]["expected_size"] == 8193
    assert calls[0]["artifact_path"] == "C:/active/segment.gguf"
    process = observed["process_tree"]["processes"][0]
    assert process["model_mappings"] == original_name
    assert process["mapped_artifact_proofs"][0]["mapped_path"] == original_name[0]
    assert commands[0]["returncode"] == 0
    assert json.loads(commands[0]["stdout"]) == process["mapped_artifact_proofs"][0]


@pytest.mark.parametrize("failure", ["denied", "partial", "different-bytes"])
def test_failed_memory_alias_verification_retains_command_and_cannot_be_replaced_by_disk_hash(monkeypatch, failure):
    observed = {"process_tree": {"processes": [{"pid": 101, "created_at": 10,
                "model_mappings": ["C:/retired/segment.gguf"]}]}}
    def fail(**kwargs):
        raise ValueError(failure)
    monkeypatch.setattr(observer, "_capture_windows_mapping", fail)
    commands = []
    with pytest.raises(ValueError, match=failure):
        observer.observe_mapping_aliases(observed, {"engine": "llama_cpp", "artifact": {"path": "C:/active/segment.gguf"}},
                                        {"sha256": "b"*64, "size_bytes": 8193}, commands)
    assert commands[0]["returncode"] == 1 and failure in commands[0]["stderr"]
    assert "mapped_artifact_proofs" not in observed["process_tree"]["processes"][0]


def test_direct_matching_mapping_does_not_read_memory(monkeypatch):
    monkeypatch.setattr(observer, "_capture_windows_mapping", lambda **_: pytest.fail("unneeded memory read"))
    observed = {"process_tree": {"processes": [{"model_mappings": ["C:/Active/segment.gguf"]}]}}
    commands = []
    observer.observe_mapping_aliases(observed, {"engine": "llama_cpp", "artifact": {"path": "c:\\active\\segment.gguf"}},
                                    {"sha256": "b"*64, "size_bytes": 8193}, commands)
    assert commands == []


def test_gguf_independent_rehash_binds_size_and_rejects_during_hash_change(tmp_path, monkeypatch):
    import hashlib
    import model_supply_assets as assets
    path = tmp_path / "segment.gguf"
    path.write_bytes(b"proper segment")
    runtime = {"artifact": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    digest = observer.rehash_artifact(runtime)
    assert digest["size_bytes"] == len(b"proper segment")
    assert observer.bind_artifact_rehash(runtime, digest)["artifact"]["size_bytes"] == len(b"proper segment")
    original = assets.sha256_file
    def changed(member):
        value = original(member)
        path.write_bytes(b"different size during hash")
        return value
    monkeypatch.setattr(assets, "sha256_file", changed)
    with pytest.raises(ValueError, match="changed during independent"):
        observer.rehash_artifact(runtime)
