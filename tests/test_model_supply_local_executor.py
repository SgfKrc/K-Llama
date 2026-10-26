"""Local executors supply the bound GGUF, never an unused recut or prepare ACK."""
import copy
import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import model_supply as supply
import model_supply_assets as assets
import model_supply_digest_cache as digests
import model_supply_plan_reuse as reuse
import relay_segment_info as relay


def _string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf(path, layers=16):
    metadata = (_string("general.architecture") + struct.pack("<I", 8) + _string("qwen2")
                + _string("qwen2.block_count") + struct.pack("<II", 4, layers)
                + _string("qwen2.embedding_length") + struct.pack("<II", 4, 8))
    table = b"".join(_string(f"blk.{layer}.attn_norm.weight")
                     + struct.pack("<IQIQ", 1, 1, 0, layer * 32) for layer in range(layers))
    header = b"GGUF" + struct.pack("<IQQ", 3, layers, 3) + metadata + table
    path.write_bytes(header + b"\0" * (-len(header) % 32)
                     + b"".join(struct.pack("<f", layer + 1) + b"\0" * 28 for layer in range(layers)))


def _write_manifest(case):
    case.marker.write_text(json.dumps(case.manifest), encoding="utf-8")


def _verify(case):
    return case.coordinator._local_resident_view(
        case.operation_id, case.sources, case.plan, case.assignment, time.monotonic() + 30)


@pytest.fixture
def resident_case(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    artifact = tmp_path / "resident.gguf"
    _gguf(artifact)
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {"source_model_id": "fixture", "source_model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64, "source_layer_range": [8, 24],
                "artifact_sha256": artifact_sha, "n_layer": 32}
    assignment = {"node_id": "master", "executor_id": "cpu", "engine": "llama_cpp",
                  "start_layer": 8, "end_layer": 24, "segment_mode": "middle",
                  "layer_artifact": {"model_id": "segment.gguf", "artifact_sha256": artifact_sha,
                                     "source_model_sha256": "a" * 64, "layer_range": [2, 6],
                                     "segment_mode": "middle"}}
    assignments = [{"node_id": "master", "engine": "llama_cpp", "start_layer": 0, "end_layer": 8},
                   assignment,
                   {"node_id": "remote", "engine": "llama_cpp", "start_layer": 24, "end_layer": 32}]
    plan = {"admitted": True, "assignments": assignments, "total_layers": 32, "cut_alignment": 4,
            "model_id": "fixture", "model_sha256": "a" * 64, "loaded_ready": False, "supply_only": True}
    sources = {"gguf": tmp_path / "full-source.gguf", "hf": None, "master_engine": "llama_cpp",
               "descriptors": {"llama_cpp": {"total_layers": 32, "layer_weight_bytes": [1] * 32}},
               "alignment": 4, "source_model_id": "fixture", "source_model_sha256": "a" * 64,
               "tokenizer_sha256": "b" * 64, "conversion_proof": None}
    inventory = {"distributed": True, "engines": ["llama_cpp"], "disk_free_bytes": 10**9,
                 "memory_available_bytes": 10**9, "sources": []}
    activated, sent, built = [], [], []
    scheduler = SimpleNamespace(_effective_role=lambda: "master",
        get_distributed_inference_enabled=lambda: True, _get_pipeline_capacity_nodes=lambda *args: [],
        _tcp_server=None,
        _model_supply_worker=SimpleNamespace(inventory=lambda **kwargs: copy.deepcopy(inventory)))
    coordinator = supply.ModelSupplyCoordinator(scheduler, tmp_path / "operations")
    operation_id = "supply_" + "a" * 32
    coordinator._active = operation_id
    coordinator._events[operation_id] = threading.Event()
    coordinator._operations[operation_id] = {"operation_id": operation_id, "status": "inspecting",
        "created_at": time.time(), "nodes": {}, "intent": {"model_id": "fixture", "engine": "llama_cpp",
        "node_ids": ["remote"], "strategy": "distribute", "timeout_seconds": 30}}
    monkeypatch.setattr(coordinator, "_sources", lambda *args, **kwargs: sources)
    monkeypatch.setattr(coordinator, "_worker_connections", lambda selected: {"remote": {"connected_at": 1}})
    monkeypatch.setattr(coordinator, "_worker_handshake_ready", lambda *args: True)
    monkeypatch.setattr(coordinator, "_send", lambda operation, node, payload: sent.append((node, payload)))
    monkeypatch.setattr(reuse, "plan_segment_supply_reuse", lambda *args, **kwargs: copy.deepcopy(plan))
    monkeypatch.setattr(supply, "_segment_disk_weight_bytes", lambda *args: 10)
    monkeypatch.setattr(supply, "_segment_disk_overhead_bytes", lambda *args: 1)
    monkeypatch.setattr(supply.shutil, "disk_usage", lambda path: SimpleNamespace(free=22))
    monkeypatch.setattr(assets, "segment_cache_hint", lambda *args: False)
    manifests = {}

    def build(source, directory, **options):
        built.append((options["node_id"], options["start_layer"], options["end_layer"]))
        assert options["start_layer"] != 8
        assert not options["cancelled"]()
        segment = directory / "segment.gguf"
        segment.write_bytes(b"segment")
        result = {"layer_range": [options["start_layer"], options["end_layer"]],
                  "segment_mode": "head" if options["start_layer"] == 0 else "tail",
                  "artifact_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                  "source_model_sha256": sources["source_model_sha256"],
                  "files": [{"path": segment.name, "size_bytes": 7}],
                  "local_files": {segment.name: str(segment)}}
        manifests[options["node_id"]] = copy.deepcopy(result)
        return result

    def activate(manifest, directory, **options):
        assert not options["cancelled"]()
        activated.append(directory)
        return {"status": "available", "model_path": str(directory)}

    def wait(operation, node, action, timeout):
        if action == "inventory_result":
            return copy.deepcopy(inventory)
        return {**manifests[node], "status": "available", "actual_downloaded_bytes": 7}

    monkeypatch.setattr(assets, "build_gguf_segment", build)
    monkeypatch.setattr(assets, "build_safetensors_segment", build)
    monkeypatch.setattr(scheduler._model_supply_worker, "activate_local", activate, raising=False)
    monkeypatch.setattr(coordinator, "_wait", wait)
    case = SimpleNamespace(coordinator=coordinator, operation_id=operation_id, sources=sources, plan=plan,
        assignment=assignment, artifact=artifact, marker=artifact.with_suffix(".manifest.json"),
        manifest=manifest, built=built, sent=sent, activated=activated)
    _write_manifest(case)
    monkeypatch.setenv("QLH_LAYER_GGUF", str(artifact))
    return case


def test_supply_uses_resident_identity_and_raw_bytes_without_prepare(resident_case):
    case = resident_case
    before = copy.deepcopy(case.plan)
    case.coordinator._execute(case.operation_id)
    operation = case.coordinator.get(case.operation_id)
    view = operation["nodes"]["master#cpu"]
    assert view["artifact_sha256"] == case.manifest["artifact_sha256"]
    assert view["source_model_id"] == "fixture" != case.assignment["layer_artifact"]["model_id"]
    assert view["source_model_sha256"] == case.sources["source_model_sha256"]
    assert view["tokenizer_sha256"] == case.sources["tokenizer_sha256"]
    assert view["layer_range"] == [8, 24]
    assert view["model_path"] == str(case.artifact.resolve())
    assert view["resident_manifest_path"] == str(case.marker.resolve())
    assert view["total_bytes"] == view["reused_bytes"] == case.artifact.stat().st_size
    assert view["actual_downloaded_bytes"] == 0 and view["strategy"] == "local_resident"
    assert "manifest_sha256" not in view
    assert operation["status"] == "available" and operation["requires_prepare"]
    assert operation["loaded_ready"] is False
    assert operation["plan"]["assignments"] == before["assignments"] and case.plan == before
    assert sorted(case.built) == [("master", 0, 8), ("remote", 24, 32)]
    assert len(case.activated) == 1
    assert [node for node, payload in case.sent if payload["action"] == "ensure"] == ["remote"]
    assert not case.coordinator._leases


def test_operation_retention_precedes_available_metadata_admission(resident_case, monkeypatch):
    case = resident_case
    phases = []

    def collect(**kwargs):
        phases.append(("operations", case.coordinator.get(case.operation_id)["status"]))
        assert kwargs["keep_operation_ids"] == (case.operation_id,)
        return {"removed": 0}

    def collect_segments(**kwargs):
        phases.append(("segments", case.coordinator.get(case.operation_id)["status"]))
        assert kwargs["root"] == case.coordinator.root
        assert case.coordinator.root / case.operation_id in kwargs["protected_paths"]
        assert not kwargs["cancelled"]()
        return {"removed": 0}

    monkeypatch.setattr(case.coordinator, "collect_operation_directories", collect)
    monkeypatch.setattr(case.coordinator.scheduler._model_supply_worker, "collect_garbage", collect_segments,
                        raising=False)
    case.coordinator._run(case.operation_id)
    operation = case.coordinator.get(case.operation_id)
    assert operation["status"] == "available"
    assert operation["retention"] == {"removed": 0}
    assert phases == [("operations", "supplying"), ("segments", "supplying")]


def test_segment_gc_cannot_change_published_hardlink_version(resident_case, monkeypatch, tmp_path):
    case = resident_case
    worker = case.coordinator.scheduler._model_supply_worker
    original_activate, original_update = worker.activate_local, case.coordinator._update
    segment = case.coordinator.root / case.operation_id / "master" / "segment.gguf"
    old_link = tmp_path / "old-segment.gguf"
    published_versions = []

    def activate(*args, **kwargs):
        os.link(segment, old_link)
        return original_activate(*args, **kwargs)

    def collect(**kwargs):
        old_link.unlink()
        return {"removed": 1}

    def update(operation_id, **values):
        original_update(operation_id, **values)
        if values.get("status") == "available":
            published_versions.append(digests.file_stamp(segment))

    monkeypatch.setattr(worker, "activate_local", activate)
    monkeypatch.setattr(worker, "collect_garbage", collect, raising=False)
    monkeypatch.setattr(case.coordinator, "_update", update)
    case.coordinator._run(case.operation_id)
    assert case.coordinator.get(case.operation_id)["status"] == "available"
    assert not old_link.exists()
    assert published_versions == [digests.file_stamp(segment)]


def test_prepare_admission_waits_for_segment_maintenance_completion(resident_case, monkeypatch):
    case = resident_case
    worker = case.coordinator.scheduler._model_supply_worker
    entered, release = threading.Event(), threading.Event()
    owner_admission = threading.Lock()

    def collect(**kwargs):
        with owner_admission:
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release maintenance")
            return {"removed": 0}

    monkeypatch.setattr(worker, "collect_garbage", collect, raising=False)
    monkeypatch.setattr(worker, "selected_model_path",
                        lambda: case.coordinator.root / case.operation_id / "master", raising=False)
    runner = threading.Thread(target=case.coordinator._run, args=(case.operation_id,))
    runner.start()
    try:
        assert entered.wait(3)
        assert case.coordinator.get(case.operation_id)["status"] == "supplying"
        with pytest.raises(supply.ModelSupplyError) as error:
            with case.coordinator.prepare_guard(case.operation_id):
                pytest.fail("prepare entered while maintenance held owner admission")
        assert error.value.code == "SUPPLY_ASSETS_NOT_AVAILABLE"
        assert not case.coordinator._preparing
    finally:
        release.set()
        runner.join(3)
    assert not runner.is_alive()
    assert case.coordinator.get(case.operation_id)["status"] == "available"
    assert owner_admission.acquire(blocking=False)
    owner_admission.release()
    with case.coordinator.prepare_guard(case.operation_id) as source:
        assert source["model_path"] == case.coordinator.get(case.operation_id)["master_model_path"]


@pytest.mark.parametrize("failure", ["operations", "segments", "both"])
def test_maintenance_failures_do_not_block_available(resident_case, monkeypatch, caplog, failure):
    case = resident_case
    calls = []

    def collect(stage):
        calls.append((stage, case.coordinator.get(case.operation_id)["status"]))
        if failure in {stage, "both"}:
            raise OSError("maintenance unavailable")
        return {"removed": 0}

    monkeypatch.setattr(case.coordinator, "collect_operation_directories", lambda **kwargs: collect("operations"))
    monkeypatch.setattr(case.coordinator.scheduler._model_supply_worker, "collect_garbage",
                        lambda **kwargs: collect("segments"), raising=False)
    case.coordinator._run(case.operation_id)
    operation = case.coordinator.get(case.operation_id)
    assert operation["status"] == "available" and operation["requires_prepare"]
    assert operation["loaded_ready"] is False
    assert not case.coordinator._events[case.operation_id].is_set()
    assert calls == [("operations", "supplying"), ("segments", "supplying")]
    if failure in {"operations", "both"}:
        assert "event=supply_operations_reclaim_failed" in caplog.text
    if failure in {"segments", "both"}:
        assert "event=supply_cache_collection_failed" in caplog.text


@pytest.mark.parametrize("stage", ["operations", "segments"])
@pytest.mark.parametrize("damage", ["artifact", "sidecar", "cancel"])
def test_maintenance_failure_cannot_hide_final_identity_or_cancel_failure(
        resident_case, monkeypatch, stage, damage):
    case = resident_case

    def collect(**kwargs):
        if damage == "artifact":
            before = case.artifact.stat()
            data = case.artifact.read_bytes()
            case.artifact.write_bytes(data[:-1] + b"!")
            os.utime(case.artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif damage == "sidecar":
            case.marker.write_text("{broken", encoding="utf-8")
        else:
            case.coordinator._events[case.operation_id].set()
        raise OSError("maintenance failed after invalidation")

    if stage == "operations":
        monkeypatch.setattr(case.coordinator, "collect_operation_directories", collect)
    else:
        monkeypatch.setattr(case.coordinator.scheduler._model_supply_worker, "collect_garbage", collect,
                            raising=False)
    with pytest.raises(supply.ModelSupplyError) as error:
        case.coordinator._execute(case.operation_id)
    assert error.value.code == ("SUPPLY_CANCELLED" if damage == "cancel" else "SUPPLY_LOCAL_RESIDENT_CHANGED")
    assert case.coordinator.get(case.operation_id)["status"] != "available"


def test_builds_skip_resident_before_directory_cache_hint_and_disk_reservation(resident_case, monkeypatch):
    case = resident_case

    def forbidden(*args, **kwargs):
        pytest.fail("resident executor entered the segment build or reservation path")

    for name in ("_segment_disk_weight_bytes", "_segment_disk_overhead_bytes"):
        monkeypatch.setattr(supply, name, forbidden)
    for name in ("segment_cache_hint", "build_gguf_segment", "build_safetensors_segment"):
        monkeypatch.setattr(assets, name, forbidden)
    monkeypatch.setattr(supply.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    plan = {**case.plan, "assignments": [case.assignment]}
    with case.coordinator._segment_builds(case.operation_id, case.sources, plan, time.monotonic() + 30) as builds:
        assert list(builds) == []
    assert not (case.coordinator.root / case.operation_id / "master").exists()


def test_warm_supply_shares_freeze_asset_digest_and_sidecar_cache(resident_case, monkeypatch):
    case = resident_case
    calls, reads = [], []
    original_hash, original_read = digests._hash_file, relay.read_artifact_manifest

    def counted_hash(path, cancelled=None):
        calls.append(path)
        return original_hash(path, cancelled)

    def counted_read(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(digests, "_hash_file", counted_hash)
    monkeypatch.setattr(relay, "read_artifact_manifest", counted_read)
    assert digests.cached_sha256(case.artifact) == case.manifest["artifact_sha256"]
    assert assets._hash_file_cached(case.artifact) == case.manifest["artifact_sha256"]
    for attempt in range(2):
        view, unchanged = _verify(case)
        unchanged()
        assert view["artifact_sha256"] == case.manifest["artifact_sha256"]
    assert calls == [case.artifact.resolve()] and reads == [case.marker]


@pytest.mark.parametrize("damage", [
    "unset_path", "missing_file", "missing_sidecar", "malformed_sidecar", "foreign_source_id",
    "foreign_source_sha", "foreign_tokenizer", "wrong_range", "coerced_range", "wrong_sidecar_sha",
    "missing_selected", "empty_selected_sha", "wrong_selected_sha", "empty_selected_source",
    "wrong_selected_range", "wrong_alignment", "wrong_plan_source", "wrong_header_layers",
    "changed_bytes", "unsupported_engine", "conflicting_sidecars",
])
def test_invalid_resident_rejected_before_any_build_or_fallback(resident_case, monkeypatch, damage):
    case = resident_case
    if damage == "unset_path":
        monkeypatch.delenv("QLH_LAYER_GGUF")
    elif damage == "missing_file":
        case.artifact.unlink()
    elif damage == "missing_sidecar":
        case.marker.unlink()
    elif damage == "malformed_sidecar":
        case.marker.write_text("{broken", encoding="utf-8")
        Path(str(case.artifact) + ".manifest.json").write_text(json.dumps(case.manifest), encoding="utf-8")
    elif damage.startswith("foreign_"):
        field = {"foreign_source_id": "source_model_id", "foreign_source_sha": "source_model_sha256",
                 "foreign_tokenizer": "tokenizer_sha256"}[damage]
        case.manifest[field] = "segment.gguf" if field == "source_model_id" else "e" * 64
        _write_manifest(case)
    elif damage in {"wrong_range", "coerced_range"}:
        case.manifest["source_layer_range"] = [4, 20] if damage == "wrong_range" else [8.5, 24]
        _write_manifest(case)
    elif damage == "wrong_sidecar_sha":
        case.manifest["artifact_sha256"] = "e" * 64
        _write_manifest(case)
    elif damage == "missing_selected":
        case.assignment.pop("layer_artifact")
    elif damage in {"empty_selected_sha", "wrong_selected_sha", "empty_selected_source", "wrong_selected_range"}:
        field, value = {"empty_selected_sha": ("artifact_sha256", ""),
                        "wrong_selected_sha": ("artifact_sha256", "e" * 64),
                        "empty_selected_source": ("source_model_sha256", ""),
                        "wrong_selected_range": ("layer_range", [1, 5])}[damage]
        case.assignment["layer_artifact"][field] = value
    elif damage == "wrong_alignment":
        case.plan["cut_alignment"] = 2
    elif damage == "wrong_plan_source":
        case.plan["model_sha256"] = "e" * 64
    elif damage == "wrong_header_layers":
        _gguf(case.artifact, layers=8)
        digest = hashlib.sha256(case.artifact.read_bytes()).hexdigest()
        case.assignment["layer_artifact"]["artifact_sha256"] = digest
        case.manifest["artifact_sha256"] = digest
        _write_manifest(case)
    elif damage == "changed_bytes":
        assert digests.cached_sha256(case.artifact) == case.manifest["artifact_sha256"]
        original_stat = case.artifact.stat()
        data = case.artifact.read_bytes()
        case.artifact.write_bytes(data[:-1] + b"!")
        os.utime(case.artifact, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    elif damage == "unsupported_engine":
        case.assignment["engine"] = "pytorch"
    elif damage == "conflicting_sidecars":
        Path(str(case.artifact) + ".manifest.json").write_text(
            json.dumps({**case.manifest, "source_model_sha256": "e" * 64}), encoding="utf-8")
    with pytest.raises(supply.ModelSupplyError) as error:
        case.coordinator._execute(case.operation_id)
    assert error.value.code.startswith("SUPPLY_LOCAL_RESIDENT_")
    assert not case.built and not case.activated and not case.coordinator._leases
    assert "master#cpu" not in case.coordinator.get(case.operation_id)["nodes"]
    assert not (case.coordinator.root / case.operation_id / "master").exists()


@pytest.mark.parametrize("change", ["artifact", "sidecar", "binding", "cancel"])
def test_changes_during_verification_cannot_publish_resident(resident_case, monkeypatch, change):
    case = resident_case
    original = digests.cached_sha256

    def changed_after_digest(*args, **kwargs):
        digest = original(*args, **kwargs)
        if change == "artifact":
            case.artifact.write_bytes(case.artifact.read_bytes() + b"!")
        elif change == "sidecar":
            case.marker.write_text(case.marker.read_text(encoding="utf-8") + " ", encoding="utf-8")
        elif change == "binding":
            monkeypatch.setenv("QLH_LAYER_GGUF", str(case.artifact.with_name("another.gguf")))
        else:
            case.coordinator._events[case.operation_id].set()
        return digest

    monkeypatch.setattr(digests, "cached_sha256", changed_after_digest)
    with pytest.raises(supply.ModelSupplyError) as error:
        case.coordinator._execute(case.operation_id)
    assert error.value.code == ("SUPPLY_CANCELLED" if change == "cancel" else "SUPPLY_LOCAL_RESIDENT_CHANGED")
    assert not case.built
    assert "master#cpu" not in case.coordinator.get(case.operation_id)["nodes"]


@pytest.mark.parametrize("change", ["artifact", "sidecar", "cancel", "receipt"])
def test_final_publication_keeps_resident_and_remote_receipt_fences(resident_case, monkeypatch, change):
    case = resident_case
    original_wait = case.coordinator._wait

    def wait(*args):
        result = original_wait(*args)
        if args[2] == "result":
            if change == "artifact":
                case.artifact.write_bytes(case.artifact.read_bytes() + b"!")
            elif change == "sidecar":
                case.marker.unlink()
            elif change == "cancel":
                case.coordinator._events[case.operation_id].set()
            else:
                result["artifact_sha256"] = "e" * 64
        return result

    monkeypatch.setattr(case.coordinator, "_wait", wait)
    with pytest.raises(supply.ModelSupplyError) as error:
        case.coordinator._execute(case.operation_id)
    expected = {"artifact": "SUPPLY_LOCAL_RESIDENT_CHANGED", "sidecar": "SUPPLY_LOCAL_RESIDENT_CHANGED",
                "cancel": "SUPPLY_CANCELLED", "receipt": "SUPPLY_RECEIPT_MISMATCH"}[change]
    assert error.value.code == expected
    assert case.coordinator.get(case.operation_id)["status"] != "available"


def test_cancel_and_deadline_reject_even_warm_identity(resident_case):
    case = resident_case
    _verify(case)
    with pytest.raises(supply.ModelSupplyError) as error:
        case.coordinator._local_resident_view(case.operation_id, case.sources, case.plan,
                                               case.assignment, time.monotonic() - 1)
    assert error.value.code == "SUPPLY_LOCAL_RESIDENT_TIMEOUT"
    case.coordinator._events[case.operation_id].set()
    with pytest.raises(supply.ModelSupplyError) as error:
        _verify(case)
    assert error.value.code == "SUPPLY_CANCELLED"
