"""Supply registration, admission, cancellation and live-file cache fences."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time
from types import SimpleNamespace

import pytest

import config
import model_supply_assets as assets
import model_supply_worker as module
from model_supply_worker import ModelSupplyWorker, get_selected_model, selected_gguf_path
from relay_segment_info import read_gguf_tokenizer_sha256
from transport_port import MessageType


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def gguf(path, layers=2):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    metadata = [
        ("general.architecture", 8, string("qwen2")),
        ("qwen2.block_count", 4, struct.pack("<I", layers)),
        ("qwen2.embedding_length", 4, struct.pack("<I", 2)),
        ("tokenizer.ggml.model", 8, string("gpt2")),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + string("test")),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, layers, len(metadata))
    for name, kind, value in metadata:
        header += string(name) + struct.pack("<I", kind) + value
    for index in range(layers):
        header += string(f"blk.{index}.attn_norm.weight") + struct.pack("<IQIQ", 1, 2, 0, index * 32)
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + (struct.pack("<ff", 1., 2.) + b"\0" * 24) * layers)
    return path


def segment(directory, operation="op-one", node="worker-one", fmt="gguf", source_sha="a" * 64):
    directory.mkdir(parents=True, exist_ok=True)
    if fmt == "gguf":
        model = gguf(directory / "segment.gguf")
        tokenizer = read_gguf_tokenizer_sha256(model)
        native = {"artifact": "segment.gguf", "mode": "middle", "source_layer_range": [1, 3], "n_layer": 4,
                  "source_model_id": "source-one", "source_model_sha256": source_sha,
                  "tokenizer_sha256": tokenizer, "artifact_sha256": assets.sha256_file(model),
                  "hidden_size": 2, "kept_block_count": 2,
                  "gguf_metadata": {"block_count": 2, "head_count_kv": None,
                                    "key_length": None, "full_attention_interval": None}}
        write_json(directory / "segment.manifest.json", native)
        kinds = {"segment.gguf": "weights", "segment.manifest.json": "manifest"}
    else:
        tokenizer = hashlib.sha256(b"{}").hexdigest()
        (directory / "tokenizer.json").write_bytes(b"{}")
        write_json(directory / "config.json", {"model_type": "qwen2", "num_hidden_layers": 4,
                                               "hidden_size": 2, "tie_word_embeddings": False})
        tensors, weight_map = {}, {}
        for index in (1, 2):
            key = f"model.layers.{index}.input_layernorm.weight"
            offset = (index - 1) * 8
            tensors[key] = {"dtype": "F32", "shape": [2], "data_offsets": [offset, offset + 8]}
            weight_map[key] = "segment.safetensors"
        encoded = json.dumps(tensors).encode()
        (directory / "segment.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 16)
        write_json(directory / "model.safetensors.index.json", {"metadata": {"total_size": 16}, "weight_map": weight_map})
        kinds = {"config.json": "support", "tokenizer.json": "support",
                 "model.safetensors.index.json": "support", "segment.safetensors": "weights"}
    files = [{"path": name, "size_bytes": (directory / name).stat().st_size,
              "sha256": assets.sha256_file(directory / name), "kind": kind}
             for name, kind in sorted(kinds.items())]
    manifest = {"manifest_kind": "qlh.segment_supply.v1", "operation_id": operation,
                "node_id": node, "source_model_id": "source-one", "source_model_sha256": source_sha,
                "tokenizer_sha256": tokenizer, "total_layers": 4, "layer_range": [1, 3],
                "segment_mode": "middle", "layer_weight_bytes": [0, 8, 8, 0], "hidden_size": 2,
                "conversion_proof": None, "format": fmt, "model_type": "qwen2", "files": files,
                "artifact_sha256": next(item["sha256"] for item in files if item["kind"] == "weights")
                if fmt == "gguf" else digest([item for item in files if item["kind"] == "weights"])}
    manifest["manifest_sha256"] = digest(manifest)
    assets.validate_supply_manifest(manifest)
    assets._verify_segment(directory, manifest)
    write_json(directory / "supply.manifest.json", manifest)
    return manifest


class Client:
    server_host = "100.64.0.9"
    is_registered = True
    registration_epoch = 1
    _connection_generation = 1

    def __init__(self):
        self.messages, self.condition = [], threading.Condition()

    def send_data(self, data, message_type):
        assert message_type == MessageType.MODEL_SUPPLY
        with self.condition:
            self.messages.append(deepcopy(data))
            self.condition.notify_all()

    def wait(self, action="result", operation=None):
        def find():
            return next((item for item in self.messages if item["action"] == action
                         and (operation is None or item.get("operation_id") == operation)), None)
        with self.condition:
            assert self.condition.wait_for(lambda: find() is not None, 5), self.messages
            return find()


class Scheduler:
    def __init__(self, role="client"):
        self._tcp_client = Client() if role == "client" else None
        self.enabled, self.role, self.refreshed = True, role, 0
        self.on_refresh = self.worker = None
        self._task_worker_stage_lock = threading.RLock()
        self._task_worker_active_attempts = {}
        self._active_pipeline_task_ids = set()
        self._host = SimpleNamespace(full_chat_execution_lock=threading.RLock())

    def assert_unlocked(self):
        if self.worker is not None:
            # Another thread may briefly own the lock while accepting/joining
            # a job. A callback owning it itself cannot release it while waiting.
            assert self.worker._lock.acquire(timeout=1), "scheduler called holding worker lock"
            self.worker._lock.release()

    def get_effective_node_id(self):
        self.assert_unlocked()
        return "worker-one" if self.role == "client" else "machine-hostname"

    def _effective_role(self):
        self.assert_unlocked()
        return self.role

    def get_distributed_inference_enabled(self):
        self.assert_unlocked()
        return self.enabled

    def refresh_task_worker_capabilities(self):
        self.assert_unlocked()
        self.refreshed += 1
        if self.on_refresh:
            self.on_refresh()


def finish(worker):
    with worker._lock:
        runner = worker._runner
    if runner is not None:
        runner.join(5)
        assert not runner.is_alive()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler()
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter(()))
    yield worker, scheduler, scheduler._tcp_client
    worker.cancel()
    finish(worker)


def command(manifest, **overrides):
    return {"action": "ensure", "operation_id": manifest["operation_id"], "node_id": manifest["node_id"],
            "strategy": "distribute", "manifest": deepcopy(manifest), "api_port": 8000,
            "token": "private-supply-token", **overrides}


def receiving(monkeypatch, source, *, progress_events=None, result_counts=None, progress_kwargs=False):
    calls = []
    def receive(manifest, root, **kwargs):
        calls.append(kwargs)
        destination = root / ("segment-" + manifest["manifest_sha256"])
        shutil.copytree(source, destination)
        events = progress_events if progress_events is not None else [
            {"downloaded_bytes": 16, "actual_downloaded_bytes": 16,
             "reused_bytes": 0, "resumed_bytes": 0, "total_bytes": 16}]
        for event in events:
            data = {**event, "path": kwargs["token"], "reason": kwargs["token"]}
            if progress_kwargs:
                kwargs["progress"](**data)
            else:
                kwargs["progress"](data)
        return {"manifest": manifest, "model_path": str(destination / "segment.gguf"),
                "local_files": {entry["path"]: str(destination / entry["path"]) for entry in manifest["files"]},
                **(result_counts or {})}
    monkeypatch.setattr(assets, "receive_segment", receive)
    return calls


def test_supply_available_does_not_run_retention_before_prepare(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    receiving(monkeypatch, source)
    calls = []
    monkeypatch.setattr(worker, "collect_garbage", lambda **kwargs: calls.append(kwargs))
    worker.handle(command(manifest), client)
    assert client.wait()["status"] == "available"
    finish(worker)
    assert calls == []
    assert get_selected_model(worker.root)["artifact_sha256"] == manifest["artifact_sha256"]


def test_receive_register_and_cached_selection(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = tmp_path / "master-segment"
    manifest = segment(source)
    calls = receiving(monkeypatch, source)
    worker.handle({"type": MessageType.MODEL_SUPPLY, "data": command(manifest)}, client)
    reply = client.wait()
    finish(worker)
    assert reply == {"action": "result", "operation_id": "op-one", "node_id": "worker-one",
                     "strategy": "distribute", "strategy_reason_code": "SUPPLY_DISTRIBUTE_REQUESTED",
                     "status": "available", "reason_code": "", "manifest_sha256": manifest["manifest_sha256"],
                     "artifact_sha256": manifest["artifact_sha256"], "layer_range": [1, 3], "source_model_sha256": "a" * 64,
                     "downloaded_bytes": 16, "actual_downloaded_bytes": 16, "reused_bytes": 0, "resumed_bytes": 0}
    assert scheduler.refreshed == 1 and calls[0]["base_url"] == "http://100.64.0.9:8000"
    monkeypatch.setattr(module, "_file_sha", lambda *a, **kw: pytest.fail("unchanged selection rehashed"))
    path = worker.selected_artifact_path()
    assert selected_gguf_path(worker.root) == str(path)
    assert path.read_bytes() == (source / "segment.gguf").read_bytes()
    pointer = json.loads((worker.root / "current.json").read_text())
    assert set(pointer) == {"kind", "directory", "manifest", "manifest_sha256", "artifact_sha256"}
    assert "private-supply-token" not in repr(client.messages) + repr(pointer)



@pytest.mark.parametrize("returned_counts,progress_kwargs", [(False, False), (True, True)])
def test_progress_and_final_result_use_cumulative_operation_snapshots(setup, tmp_path, monkeypatch,
                                                                     returned_counts, progress_kwargs):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    first = {"downloaded_bytes": 32, "actual_downloaded_bytes": 12, "reused_bytes": 16, "resumed_bytes": 4}
    last = {"downloaded_bytes": 64, "actual_downloaded_bytes": 32, "reused_bytes": 24, "resumed_bytes": 8}
    final = {"downloaded_bytes": 80, "actual_downloaded_bytes": 40, "reused_bytes": 32, "resumed_bytes": 8}
    receiving(monkeypatch, source, progress_events=[
        {"phase": "receiving", "total_bytes": 80, **first},
        {"phase": "receiving", "total_bytes": 80, **first},
        {"stage": "verify", "total_bytes": 80, **last}],
        result_counts=final if returned_counts else None, progress_kwargs=progress_kwargs)
    worker.handle(command(manifest))
    reply = client.wait()
    finish(worker)
    assert reply["status"] == "available" and scheduler.refreshed == 1
    expected = final if returned_counts else last
    assert {key: reply[key] for key in expected} == expected
    progress = [message for message in client.messages if message["action"] == "progress"]
    assert [{key: message[key] for key in first} for message in progress[:3]] == [first, first, last]
    assert [message["phase"] for message in progress] == ["receiving", "receiving", "verify", "verifying"]
    assert progress[0]["total_bytes"] == 80
    assert all(message["operation_id"] == manifest["operation_id"] for message in client.messages)
    assert "private-supply-token" not in repr(client.messages)
    assert not any(key in message for message in client.messages for key in ("path", "reason", "local_files"))


def test_final_result_records_receiver_counts_without_progress(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    counts = {"downloaded_bytes": 7, "actual_downloaded_bytes": 7, "reused_bytes": 19, "resumed_bytes": 3}
    receiving(monkeypatch, source, progress_events=[], result_counts=counts)
    worker.handle(command(manifest))
    reply = client.wait()
    finish(worker)
    assert reply["status"] == "available"
    assert {key: reply[key] for key in counts} == counts


@pytest.mark.parametrize("invalid", [-1, True, False, 1.5, "17", None])
def test_progress_and_result_reject_invalid_transfer_counts(setup, tmp_path, monkeypatch, invalid):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    valid = {"downloaded_bytes": 17, "actual_downloaded_bytes": 5, "reused_bytes": 9, "resumed_bytes": 3}
    rejected = dict.fromkeys(valid, invalid)
    receiving(monkeypatch, source, progress_events=[valid, rejected], result_counts=rejected)
    worker.handle(command(manifest))
    reply = client.wait()
    finish(worker)
    assert reply["status"] == "available"
    assert {key: reply[key] for key in valid} == valid
    progress = [message for message in client.messages if message["action"] == "progress"]
    assert {key: progress[0][key] for key in valid} == valid
    assert all(key not in progress[1] for key in valid)
    assert all(type(reply[key]) is int and reply[key] >= 0 for key in valid)


@pytest.mark.parametrize("same_operation", [False, True])
def test_failure_counts_and_next_attempt_counts_stay_with_their_operation(setup, tmp_path, monkeypatch,
                                                                        same_operation):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    first = {"downloaded_bytes": 8, "actual_downloaded_bytes": 2, "reused_bytes": 4, "resumed_bytes": 2}
    latest = {"downloaded_bytes": 16, "actual_downloaded_bytes": 6, "reused_bytes": 8, "resumed_bytes": 2}
    def fail(manifest, root, **kwargs):
        kwargs["progress"](first)
        kwargs["progress"](latest)
        raise assets.SupplyError("download_incomplete", "peer disconnected")
    monkeypatch.setattr(assets, "receive_segment", fail)
    worker.handle(command(manifest))
    failed = client.wait()
    finish(worker)
    assert failed["reason_code"] == "download_incomplete"
    assert {key: failed[key] for key in latest} == latest
    if not same_operation:
        manifest = segment(tmp_path / "next-source", operation="op-next")
        source = tmp_path / "next-source"
    # A fresh call without metrics must start at zero, including retries of the
    # same operation_id. The finished operation's snapshots are never reused.
    receiving(monkeypatch, source, progress_events=[])
    worker.handle(command(manifest))
    finish(worker)
    available = next(message for message in reversed(client.messages) if message.get("status") == "available")
    assert available["operation_id"] == manifest["operation_id"]
    assert {key: available[key] for key in latest} == dict.fromkeys(latest, 0)
    assert {key: failed[key] for key in latest} == latest


@pytest.mark.parametrize("field,value", [("schema", 1), ("engine", "llama_cpp"), ("manifest_kind", "wrong")])
def test_invalid_manifest_never_transfers(setup, tmp_path, monkeypatch, field, value):
    worker, scheduler, client = setup
    manifest = segment(tmp_path / "source")
    manifest[field] = value
    monkeypatch.setattr(assets, "receive_segment", lambda *a, **kw: pytest.fail("invalid manifest transferred"))
    worker.handle(command(manifest))
    assert client.wait()["status"] == "failed"
    assert not (worker.root / "current.json").exists() and scheduler.refreshed == 0


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_master_activation_registers_parent_generation_and_literal_master(tmp_path, monkeypatch, fmt):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    parent = worker.root.parent / "operations" / "op-one" / "master"
    directory = parent / "segment-built"
    manifest = segment(directory, node="master", fmt=fmt)
    reply = worker.activate_local(manifest, parent)
    assert reply["status"] == "available", reply
    expected = directory / "segment.gguf" if fmt == "gguf" else directory
    assert reply["node_id"] == "master" and reply["model_path"] == str(expected)
    assert scheduler.refreshed == 1 and worker.selected_model_path() == expected
    assert (worker.selected_artifact_path() is not None) == (fmt == "gguf")
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter(()))
    assert worker.inventory()["node_id"] == "master"
    assert ModelSupplyWorker(scheduler, worker.root).selected_model_path() == expected
    scheduler.enabled = False
    assert worker.selected_model_path() is None
    assert get_selected_model(worker.root) is None and selected_gguf_path(worker.root) == ""
    monkeypatch.setattr(config, "RUN_MODE", "local")
    assert get_selected_model(worker.root) is None and selected_gguf_path(worker.root) == ""


@pytest.mark.parametrize("role,enabled", [("master", False), ("client", True)])
def test_local_activation_requires_master_and_current_mode(tmp_path, monkeypatch, role, enabled):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler(role)
    scheduler.enabled = enabled
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    directory = worker.root.parent / "built"
    result = worker.activate_local(segment(directory, node="master"), directory)
    assert result["status"] == "failed" and scheduler.refreshed == 0
    assert not (worker.root / "current.json").exists()


@pytest.mark.parametrize("change", ["cancel", "wrong_cancel", "disabled", "disconnected", "replacement", "epoch", "generation"])
def test_transfer_completion_rechecks_cancellation_and_connection(setup, tmp_path, monkeypatch, change):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    receiving(monkeypatch, source)
    receiver = assets.receive_segment
    entered, release = threading.Event(), threading.Event()
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return receiver(*args, **kwargs)
    monkeypatch.setattr(assets, "receive_segment", blocked)
    returned = threading.Event()
    def submit():
        worker.handle(command(manifest))
        returned.set()
    caller = threading.Thread(target=submit, daemon=True)
    caller.start()
    try:
        assert entered.wait(2)
        # Prove acceptance returns before releasing the transfer. A wall-clock
        # threshold also measures unrelated scheduler contention in full runs.
        assert returned.wait(2), "handle blocked on the unfinished transfer"
        assert not release.is_set()
        if change in {"cancel", "wrong_cancel"}:
            worker.handle({"action": "cancel", "operation_id": "op-one" if change == "cancel" else "other-op"})
        elif change == "disabled":
            scheduler.enabled = False
        elif change == "disconnected":
            client.is_registered = False
        elif change == "replacement":
            scheduler._tcp_client = Client()
        elif change == "epoch":
            client.registration_epoch += 1
        else:
            client._connection_generation += 1
    finally:
        release.set()
        caller.join(5)
    assert not caller.is_alive()
    finish(worker)
    if change == "wrong_cancel":
        assert client.wait()["status"] == "available" and scheduler.refreshed == 1
    else:
        assert not (worker.root / "current.json").exists() and scheduler.refreshed == 0
        assert not any(item.get("status") == "available" for item in client.messages)
        if change == "cancel":
            assert client.wait()["reason_code"] == "cancelled"
        if change == "replacement":
            assert not scheduler._tcp_client.messages


def test_replacement_serializes_jobs_and_discards_old_results(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    sources = {op: tmp_path / op for op in ("op-old", "op-new")}
    manifests = {op: segment(source, operation=op) for op, source in sources.items()}
    entered, release = threading.Event(), threading.Event()
    running, maximum = 0, 0
    def receive(manifest, root, **kwargs):
        nonlocal running, maximum
        running += 1
        maximum = max(maximum, running)
        try:
            if manifest["operation_id"] == "op-old":
                entered.set()
                assert release.wait(5)
            target = root / ("segment-" + manifest["manifest_sha256"])
            shutil.copytree(sources[manifest["operation_id"]], target)
            return {"manifest": manifest, "model_path": str(target / "segment.gguf")}
        finally:
            running -= 1
    monkeypatch.setattr(assets, "receive_segment", receive)
    worker.handle(command(manifests["op-old"]))
    try:
        assert entered.wait(2)
        worker.handle(command(manifests["op-new"]))
    finally:
        release.set()
    assert client.wait(operation="op-new")["status"] == "available"
    finish(worker)
    assert maximum == scheduler.refreshed == 1
    assert not any(item.get("operation_id") == "op-old" for item in client.messages)
    assert get_selected_model(worker.root)["manifest"]["operation_id"] == "op-new"


@pytest.mark.parametrize("moment", ["replace", "refresh"])
def test_mode_transition_at_registration_revokes_selection(setup, tmp_path, monkeypatch, moment):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    receiving(monkeypatch, source)
    if moment == "replace":
        replace = module.os.replace
        def switching(source_path, dest):
            result = replace(source_path, dest)
            if Path(dest).name == "current.json":
                scheduler.enabled = False
            return result
        monkeypatch.setattr(module.os, "replace", switching)
    else:
        scheduler.on_refresh = lambda: setattr(scheduler, "enabled", False)
    worker.handle(command(manifest))
    finish(worker)
    assert not (worker.root / "current.json").exists()
    assert not any(item.get("status") == "available" for item in client.messages)


@pytest.mark.parametrize("local", [False, True])
def test_busy_stage_blocks_current_pointer_switch(tmp_path, monkeypatch, local):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master" if local else "client")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master" if local else "worker-one")
    scheduler._task_worker_active_attempts["attempt"] = object()
    if local:
        reply = worker.activate_local(manifest, directory)
    else:
        receiving(monkeypatch, directory)
        worker.handle(command(manifest))
        reply = scheduler._tcp_client.wait()
        finish(worker)
    assert reply["status"] == "failed" and reply["reason_code"] == "SUPPLY_WORKER_BUSY"
    assert not (worker.root / "current.json").exists() and scheduler.refreshed == 0


def test_inventory_async_source_verification_and_old_client_fence(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = gguf(tmp_path / "local-source.gguf", 4)
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter([("source-one", source)]))
    entered, release = threading.Event(), threading.Event()
    original = module._file_sha
    def slow(path, *args):
        entered.set()
        assert release.wait(5)
        return original(path, *args)
    monkeypatch.setattr(module, "_file_sha", slow)
    start = time.monotonic()
    worker.handle({"action": "inventory", "operation_id": "inventory-one"})
    assert time.monotonic() - start < 0.5
    try:
        assert entered.wait(2)
    finally:
        release.set()
    reply = client.wait("inventory_result")
    assert reply["disk_free_bytes"] > 0 and reply["memory_available_bytes"] >= 0
    assert "engines" in reply and "execution_device" in reply
    assert reply["sources"] == [{"source_model_id": "source-one", "source_model_sha256": original(source),
                                 "tokenizer_sha256": read_gguf_tokenizer_sha256(source), "size_bytes": source.stat().st_size}]
    assert str(tmp_path) not in repr(reply) and list(worker._sources.values()) == [source]



@pytest.mark.parametrize("selected_id", ["selected-model", "selected-model-gguf"])
def test_inventory_filters_real_registry_and_discovery_before_weight_io(setup, tmp_path, monkeypatch, selected_id):
    import local_model_assets
    import local_store
    import model_config
    import relay_segment_info
    worker, scheduler, client = setup
    registered = gguf(tmp_path / "unrelated-name.gguf", 2)
    builtin = gguf(tmp_path / "opaque-builtin.gguf", 3)
    unrelated_registered = gguf(tmp_path / "selected-model.gguf", 4)
    unrelated_builtin = gguf(tmp_path / "selected-model-Q4_K_M.gguf", 4)
    assert local_store.save_local_experimental_model("selected-model", {"model_id": "selected-model", "gguf_path": str(registered)})
    assert local_store.save_local_experimental_model("other-registry", {"model_id": "other-registry", "gguf_path": str(unrelated_registered)})
    monkeypatch.setattr(model_config, "get_builtin_models", lambda: [
        SimpleNamespace(model_id="selected-model-gguf", gguf_path=str(builtin)),
        SimpleNamespace(model_id="other-builtin", gguf_path=str(unrelated_builtin))])
    monkeypatch.setattr(local_model_assets, "_app_root", lambda: tmp_path)
    discovered_paths = []
    for directory_name, asset_id, filename in [
        ("opaque-layout", "selected-model-gguf", "different-family.gguf"),
        ("other-layout", "other-asset-gguf", "selected-model.gguf")]:
        directory = tmp_path / "models" / directory_name
        directory.mkdir(parents=True)
        source = gguf(directory / filename, 4)
        write_json(directory / ".qlh-model-asset.json", {
            "schema_version": 1, "artifact_kind": "llama.cpp_gguf",
            "asset": {"asset_id": asset_id},
            "files": [{"path": filename, "size_bytes": source.stat().st_size,
                       "sha256": assets.sha256_file(source)}]})
        discovered_paths.append(source)
    discovered = discovered_paths[0]
    expected_paths = {registered, builtin, discovered}
    monkeypatch.setattr(worker, "_source_candidates", ModelSupplyWorker._source_candidates.__get__(worker))
    calls = {"layers": [], "tokenizer": [], "hash": []}
    def recording(kind, original):
        def read(path, *args, **kwargs):
            assert path in expected_paths, f"unselected source reached {kind}: {path}"
            calls[kind].append(path)
            return original(path, *args, **kwargs)
        return read
    monkeypatch.setattr(relay_segment_info, "read_gguf_layer_info",
                        recording("layers", relay_segment_info.read_gguf_layer_info))
    monkeypatch.setattr(relay_segment_info, "read_gguf_tokenizer_sha256",
                        recording("tokenizer", relay_segment_info.read_gguf_tokenizer_sha256))
    monkeypatch.setattr(module, "_file_sha", recording("hash", module._file_sha))
    monkeypatch.setattr(local_model_assets, "_sha256_file",
                        lambda *args: pytest.fail("discovery hashed an unselected asset"))
    start = time.monotonic()
    worker.handle({"action": "inventory", "operation_id": "filtered-inventory", "model_id": selected_id})
    assert time.monotonic() - start < 0.5
    reply = client.wait("inventory_result")
    assert "reason_code" not in reply, reply
    assert {item["source_model_id"] for item in reply["sources"]} == {"selected-model", "selected-model-gguf"}
    assert len(reply["sources"]) == 3
    assert all(set(paths) == expected_paths and len(paths) == 3 for paths in calls.values())
    with worker._lock:
        assert set(worker._sources.values()) == expected_paths
    assert str(tmp_path) not in repr(reply)


@pytest.mark.parametrize("selected_id", ["not-registered", True, {}, ["source-one"]])
def test_inventory_unknown_or_invalid_id_skips_hash_and_clears_prior_sources(setup, tmp_path, monkeypatch, selected_id):
    import relay_segment_info
    worker, scheduler, client = setup
    source = gguf(tmp_path / "not-registered.gguf", 4)
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter([("source-one", source)]))
    worker._sources[("previous-source", "a" * 64, "b" * 64)] = source
    def reject(*args, **kwargs):
        pytest.fail("unselected model reached weight I/O")
    monkeypatch.setattr(module, "_file_sha", reject)
    monkeypatch.setattr(relay_segment_info, "read_gguf_layer_info", reject)
    monkeypatch.setattr(relay_segment_info, "read_gguf_tokenizer_sha256", reject)
    assert worker.inventory(model_id=selected_id)["sources"] == []
    assert worker._sources == {}


def test_resource_inventory_skips_source_discovery_preserves_registration_and_resamples_memory(setup, monkeypatch):
    worker, scheduler, client = setup
    registered = {("source-one", "a" * 64, "b" * 64): Path("existing.gguf")}
    worker._sources = registered
    monkeypatch.setattr(worker, "_source_candidates", lambda: pytest.fail("resource refresh discovered sources"))
    monkeypatch.setattr("torch_runtime.torch_available", lambda: False)
    memory = SimpleNamespace(available=100, total=500)
    monkeypatch.setattr("psutil.virtual_memory", lambda: memory)
    first = worker.inventory(model_id="")
    memory.available = 200
    second = worker.inventory(model_id="")
    assert first["sources"] == second["sources"] == []
    assert first["memory_available_bytes"] == 100
    assert second["memory_available_bytes"] == 200
    assert second["memory_total_bytes"] == 500
    assert worker._sources is registered


def test_inventory_without_model_id_keeps_all_registered_sources(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    paths = [gguf(tmp_path / "first.gguf", 2), gguf(tmp_path / "second.gguf", 4)]
    candidates = [("first-model", paths[0]), ("second-model", paths[1])]
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter(candidates))
    expected = [{"source_model_id": model_id, "source_model_sha256": assets.sha256_file(path),
                 "tokenizer_sha256": read_gguf_tokenizer_sha256(path), "size_bytes": path.stat().st_size}
                for model_id, path in candidates]
    assert worker.inventory()["sources"] == expected
    assert worker.inventory(model_id=None)["sources"] == expected


@pytest.mark.parametrize("mutation", ["bytes", "pointer", "new_file"])
def test_selection_cache_invalidates_on_live_file_or_pointer_change(setup, tmp_path, monkeypatch, mutation):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    receiving(monkeypatch, source)
    worker.handle(command(manifest))
    assert client.wait()["status"] == "available"
    finish(worker)
    selected = worker.selected_artifact_path()
    assert get_selected_model(worker.root) is not None
    if mutation == "bytes":
        stat = selected.stat()
        payload = selected.read_bytes()
        selected.write_bytes(payload[:-1] + b"x")
        os.utime(selected, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert selected.stat().st_mtime_ns == stat.st_mtime_ns
    elif mutation == "pointer":
        pointer_path = worker.root / "current.json"
        pointer = json.loads(pointer_path.read_text())
        pointer["directory"] = "../source"
        write_json(pointer_path, pointer)
    else:
        (selected.parent / "unlisted.safetensors").write_bytes(b"new weight")
    assert get_selected_model(worker.root) is None


def test_local_cut_uses_only_inventory_verified_source(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    local_source = gguf(tmp_path / "trusted-source.gguf", 4)
    source_sha = assets.sha256_file(local_source)
    expected = tmp_path / "expected"
    manifest = segment(expected, source_sha=source_sha)
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter([("source-one", local_source)]))
    worker.inventory()
    calls = []
    def build(source, destination, **kwargs):
        calls.append((source, kwargs))
        generation = destination / "segment-built"
        shutil.copytree(expected, generation)
        return {**manifest, "local_files": {entry["path"]: str(generation / entry["path"]) for entry in manifest["files"]}}
    monkeypatch.setattr(assets, "build_gguf_segment", build)
    worker.handle(command(manifest, strategy="local_cut", source_path=str(tmp_path / "external.gguf")))
    assert client.wait()["status"] == "available"
    assert calls[0][0] == local_source and calls[0][1]["start_layer"] == 1 and calls[0][1]["end_layer"] == 3
    assert not calls[0][1]["cancelled"]()
    worker.cancel(manifest["operation_id"])
    assert calls[0][1]["cancelled"]()


def test_local_cut_rejects_external_path_and_changed_verified_source(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = gguf(tmp_path / "local-source.gguf", 4)
    manifest = segment(tmp_path / "expected", source_sha=assets.sha256_file(source))
    monkeypatch.setattr(assets, "build_gguf_segment", lambda *a, **kw: pytest.fail("unverified source cut"))
    worker.handle(command(manifest, strategy="local_cut", source_path=str(source)))
    assert client.wait()["reason_code"] == "local_source_unavailable"
    finish(worker)
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter([("source-one", source)]))
    worker.inventory()
    source.write_bytes(source.read_bytes() + b"changed")
    manifest["operation_id"] = "op-two"
    manifest["manifest_sha256"] = digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    worker.handle(command(manifest, strategy="local_cut"))
    assert client.wait(operation="op-two")["reason_code"] == "source_model_sha256_mismatch"


def test_asset_errors_do_not_echo_authorization(setup, tmp_path, monkeypatch, caplog):
    worker, scheduler, client = setup
    manifest = segment(tmp_path / "source")
    def fail(*args, **kwargs):
        raise RuntimeError("authorization URL " + kwargs["token"])
    monkeypatch.setattr(assets, "receive_segment", fail)
    worker.handle(command(manifest))
    assert client.wait()["reason_code"] == "transfer_failed"
    finish(worker)
    assert "private-supply-token" not in repr(client.messages) + caplog.text
    assert worker._active.token == ""


def test_default_root_uses_application_state_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path / "state"))
    worker = ModelSupplyWorker(Scheduler())
    assert worker.root == tmp_path / "state" / "model-supply" / "received"
    assert not worker.root.exists()



def test_inventory_reply_is_not_written_to_replacement_connection(setup, monkeypatch):
    worker, scheduler, old_client = setup
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def inventory():
        entered.set()
        assert release.wait(5)
        return {"sources": [], "distributed": True}
    original = worker._send
    def send(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(worker, "inventory", inventory)
    monkeypatch.setattr(worker, "_send", send)
    worker.handle({"action": "inventory", "operation_id": "old-inventory"}, old_client)
    try:
        assert entered.wait(2)
        scheduler._tcp_client = Client()
    finally:
        release.set()
    assert finished.wait(5)
    assert not old_client.messages and not scheduler._tcp_client.messages


def test_busy_is_rechecked_after_waiting_for_stage_admission(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    receiving(monkeypatch, source)
    verified = threading.Event()
    original = module._verify_files
    def verify(*args, **kwargs):
        result = original(*args, **kwargs)
        verified.set()
        return result
    monkeypatch.setattr(module, "_verify_files", verify)
    with scheduler._task_worker_stage_lock:
        worker.handle(command(manifest))
        assert verified.wait(5)
        scheduler._task_worker_active_attempts["arrived-during-verification"] = object()
    assert client.wait()["reason_code"] == "SUPPLY_WORKER_BUSY"
    finish(worker)
    assert not (worker.root / "current.json").exists()


def test_changed_pointer_bytes_revalidate_without_rehash_and_detach_cached_results(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    receiving(monkeypatch, source)
    worker.handle(command(manifest))
    assert client.wait()["status"] == "available"
    finish(worker)
    first = get_selected_model(worker.root)
    first["manifest"]["source_model_id"] = "mutated-return-value"
    assert get_selected_model(worker.root)["manifest"]["source_model_id"] == "source-one"
    calls = []
    original = assets._hash_file
    def record(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(assets, "_hash_file", record)
    pointer = worker.root / "current.json"
    pointer.write_bytes(pointer.read_bytes() + b"\n")
    assert get_selected_model(worker.root) is not None
    assert not any(path.suffix == ".gguf" for path in calls)
    count = len(calls)
    assert get_selected_model(worker.root) is not None and len(calls) == count


@pytest.mark.parametrize("strategy", ["distribute", "local_cut"])
def test_worker_reuses_verified_segment_before_transfer_or_local_cut(setup, tmp_path, monkeypatch, strategy):
    worker, scheduler, client = setup
    old = segment(worker.root / ("segment-" + "a" * 64))
    manifest = deepcopy(old)
    manifest["operation_id"] = "op-two"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = digest(manifest)
    for entry in manifest["files"]:
        assets._hash_file_cached(worker.root / ("segment-" + "a" * 64) / entry["path"])
    monkeypatch.setattr(assets, "receive_segment", lambda *args, **kwargs: pytest.fail("verified cache started HTTP transfer"))
    monkeypatch.setattr(worker, "_local_cut", lambda *args, **kwargs: pytest.fail("verified cache started local cutting"))
    original = assets._hash_file
    def no_weight_reads(path, *args, **kwargs):
        assert Path(path).suffix != ".gguf", "verified cache reread weight bytes"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(assets, "_hash_file", no_weight_reads)
    worker.handle(command(manifest, strategy=strategy))
    result = client.wait(operation="op-two")
    assert result["status"] == "available" and result["strategy"] == "reuse"
    assert result["actual_downloaded_bytes"] == 0
    assert result["reused_bytes"] == sum(entry["size_bytes"] for entry in manifest["files"])
    finish(worker)
    selected = get_selected_model(worker.root)
    assert selected["manifest"] == manifest
    assert os.path.samefile(selected["model_path"], worker.root / ("segment-" + "a" * 64) / "segment.gguf")


def test_registration_reuses_verified_immutable_file_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    assert worker.activate_local(manifest, directory)["status"] == "available"
    assert get_selected_model(worker.root) is not None
    calls = []
    original = assets._hash_file
    def record(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(assets, "_hash_file", record)
    assert worker.activate_local(manifest, directory)["status"] == "available"
    assert not any(path.suffix == ".gguf" for path in calls)


def test_model_identity_reuses_registration_digests_without_changing_bundle_hash(tmp_path, monkeypatch):
    from model_sync import compute_model_sha256
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master", fmt="safetensors")
    expected = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        expected.update(f"{path.name}\0{path.stat().st_size}\0{assets.sha256_file(path)}\n".encode())
    assert worker.activate_local(manifest, directory)["status"] == "available"
    original = assets._hash_file
    def no_weight_reads(path, *args, **kwargs):
        assert Path(path).suffix != ".safetensors", "prepare rehashed an already verified weight"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(assets, "_hash_file", no_weight_reads)
    assert compute_model_sha256(str(directory)) == expected.hexdigest()
    assert not (directory / "model.sha256").exists()


@pytest.mark.parametrize("preserve_mtime", [False, True])
def test_registration_rejects_changed_previously_selected_weights(tmp_path, monkeypatch, preserve_mtime):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    assert worker.activate_local(manifest, directory)["status"] == "available"
    weight = directory / "segment.gguf"
    before = weight.stat()
    original = weight.read_bytes()
    weight.write_bytes(original[:-1] + b"x")
    if preserve_mtime:
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
    result = worker.activate_local(manifest, directory)
    assert result["status"] == "failed" and result["reason_code"] == "artifact_sha256_mismatch"


def test_local_cut_real_builder_contract(setup, tmp_path, monkeypatch):
    pytest.importorskip("gguf")
    worker, scheduler, client = setup
    source = gguf(tmp_path / "real-source.gguf", 4)
    kwargs = {"source_model_id": "source-one", "source_model_sha256": assets.sha256_file(source),
              "tokenizer_sha256": read_gguf_tokenizer_sha256(source), "start_layer": 1, "end_layer": 3,
              "total_layers": 4, "operation_id": "op-real", "node_id": "worker-one", "conversion_proof": None}
    built = assets.build_gguf_segment(source, tmp_path / "master-built", **kwargs)
    manifest = assets.validate_supply_manifest(built)
    # The local registry ID can differ; the verified source digest is identity.
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter([("local-alias", source)]))
    worker.inventory(model_id="local-alias")
    worker.handle(command(manifest, strategy="local_cut"))
    reply = client.wait(operation="op-real")
    assert reply["status"] == "available", reply
    assert reply["artifact_sha256"] == manifest["artifact_sha256"]
    assert {key: reply[key] for key in ("downloaded_bytes", "actual_downloaded_bytes", "reused_bytes", "resumed_bytes")} == {
        "downloaded_bytes": 0, "actual_downloaded_bytes": 0, "reused_bytes": 0, "resumed_bytes": 0}
    assert scheduler.refreshed == 1


def test_invalid_client_cannot_cancel_current_operation(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    receiving(monkeypatch, source)
    receiver = assets.receive_segment
    entered, release = threading.Event(), threading.Event()
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return receiver(*args, **kwargs)
    monkeypatch.setattr(assets, "receive_segment", blocked)
    worker.handle(command(manifest))
    try:
        assert entered.wait(2)
        worker.handle({"action": "cancel", "operation_id": "op-one"}, Client())
    finally:
        release.set()
    assert client.wait()["status"] == "available"



def test_master_cancel_during_verification_preserves_previous_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    old_dir, new_dir = worker.root.parent / "old", worker.root.parent / "new"
    old = segment(old_dir, operation="op-old", node="master")
    new = segment(new_dir, operation="op-new", node="master")
    assert worker.activate_local(old, old_dir)["status"] == "available"
    previous = (worker.root / "current.json").read_bytes()
    entered, release = threading.Event(), threading.Event()
    original = module._verify_files
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_verify_files", blocked)
    result = {}
    thread = threading.Thread(target=lambda: result.update(worker.activate_local(new, new_dir)))
    thread.start()
    try:
        assert entered.wait(2)
        worker.cancel("op-new")
    finally:
        release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert result["status"] == "failed" and result["reason_code"] == "cancelled"
    assert (worker.root / "current.json").read_bytes() == previous
    assert scheduler.refreshed == 1


def test_busy_activation_keeps_existing_registration_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    old_dir, new_dir = worker.root.parent / "old", worker.root.parent / "new"
    old = segment(old_dir, operation="op-old", node="master")
    new = segment(new_dir, operation="op-new", node="master")
    assert worker.activate_local(old, old_dir)["status"] == "available"
    previous = (worker.root / "current.json").read_bytes()
    scheduler._task_worker_active_attempts["attempt-running"] = object()
    assert worker.activate_local(new, new_dir)["reason_code"] == "SUPPLY_WORKER_BUSY"
    assert (worker.root / "current.json").read_bytes() == previous
    assert worker.selected_model_path() == old_dir / "segment.gguf"
    assert scheduler.refreshed == 1


def test_manifest_cannot_persist_authorization_as_conversion_metadata(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    manifest = segment(tmp_path / "source")
    manifest["conversion_proof"] = {"converter_revision": "private-supply-token"}
    manifest["manifest_sha256"] = digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    monkeypatch.setattr(assets, "receive_segment", lambda *a, **kw: pytest.fail("credential-bearing manifest transferred"))
    worker.handle(command(manifest))
    assert client.wait()["reason_code"] == "invalid_manifest"
    assert not (worker.root / "current.json").exists()



def test_same_operation_can_retry_after_transfer_failure(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    source = tmp_path / "source"
    manifest = segment(source)
    def fail(*args, **kwargs):
        raise assets.SupplyError("download_incomplete", "peer disconnected")
    monkeypatch.setattr(assets, "receive_segment", fail)
    worker.handle(command(manifest))
    assert client.wait()["reason_code"] == "download_incomplete"
    finish(worker)
    receiving(monkeypatch, source)
    worker.handle(command(manifest))
    finish(worker)
    assert any(reply.get("status") == "available" for reply in client.messages), json.dumps(client.messages)
    assert scheduler.refreshed == 1 and worker.selected_artifact_path() is not None



@contextmanager
def held_by_another_thread(lock):
    entered, release = threading.Event(), threading.Event()
    def hold():
        with lock:
            entered.set()
            assert release.wait(5)
    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert entered.wait(2)
        yield
    finally:
        release.set()
        holder.join(5)
        assert not holder.is_alive()


@pytest.mark.parametrize("local", [False, True])
def test_other_thread_full_chat_blocks_registration_without_waiting(tmp_path, monkeypatch, local):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master" if local else "client")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master" if local else "worker-one")
    receiving(monkeypatch, directory)
    with held_by_another_thread(scheduler._host.full_chat_execution_lock):
        if local:
            reply = worker.activate_local(manifest, directory)
        else:
            worker.handle(command(manifest))
            reply = scheduler._tcp_client.wait()
            finish(worker)
        assert reply["status"] == "failed" and reply["reason_code"] == "SUPPLY_WORKER_BUSY"
        # Nonblocking full acquisition leaves cancellation and heartbeat state
        # accessible; the holder is still alive here.
        assert worker._lock.acquire(timeout=1)
        worker._lock.release()
        assert not (worker.root / "current.json").exists()
        assert scheduler.refreshed == 0


def test_master_active_pipeline_preserves_registration_and_blocks_revoke(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    old_dir, new_dir = worker.root.parent / "old", worker.root.parent / "new"
    old = segment(old_dir, operation="old-pipeline", node="master")
    new = segment(new_dir, operation="new-pipeline", node="master")
    assert worker.activate_local(old, old_dir)["status"] == "available"
    previous_op = worker._active
    previous = (worker.root / "current.json").read_bytes()
    scheduler._active_pipeline_task_ids.add("active-master-pipeline")
    reply = worker.activate_local(new, new_dir)
    assert reply["reason_code"] == "SUPPLY_WORKER_BUSY"
    assert (worker.root / "current.json").read_bytes() == previous
    with pytest.raises(module.SupplyError, match="SUPPLY_WORKER_BUSY"):
        worker._revoke(previous_op)
    assert (worker.root / "current.json").read_bytes() == previous
    assert scheduler.refreshed == 1


def test_revoke_cannot_change_pointer_during_other_thread_full_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    assert worker.activate_local(manifest, directory)["status"] == "available"
    previous = (worker.root / "current.json").read_bytes()
    with held_by_another_thread(scheduler._host.full_chat_execution_lock):
        with pytest.raises(module.SupplyError, match="SUPPLY_WORKER_BUSY"):
            worker._revoke(worker._active)
        assert (worker.root / "current.json").read_bytes() == previous
    worker._revoke(worker._active)
    assert not (worker.root / "current.json").exists()
    assert str(worker.root) not in module._SELECTION_CACHE


@pytest.mark.parametrize("cancel_on_refresh", [False, True])
def test_hash_outside_locks_publish_refresh_and_revoke_inside_ordered_locks(tmp_path, monkeypatch, cancel_on_refresh):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    full = scheduler._host.full_chat_execution_lock
    stage = scheduler._task_worker_stage_lock
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    events = []
    class OrderedStage:
        def __enter__(self):
            assert full._is_owned(), "stage lock acquired before full chat"
            stage.acquire()
            events.append("stage_enter")
            return self
        def __exit__(self, *args):
            assert full._is_owned(), "full chat released before stage"
            events.append("stage_exit")
            stage.release()
    scheduler._task_worker_stage_lock = OrderedStage()
    original_verify = module._verify_files
    def verify(*args, **kwargs):
        assert not full._is_owned() and not stage._is_owned()
        events.append("hash")
        return original_verify(*args, **kwargs)
    monkeypatch.setattr(module, "_verify_files", verify)
    original_replace = module.os.replace
    def replace(source, dest):
        if Path(dest).name == "current.json":
            assert full._is_owned() and stage._is_owned()
            events.append("publish")
        return original_replace(source, dest)
    monkeypatch.setattr(module.os, "replace", replace)
    original_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        if path.name == "current.json":
            assert full._is_owned() and stage._is_owned()
            events.append("revoke")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    def on_refresh():
        assert full._is_owned() and stage._is_owned()
        events.append("refresh")
        if cancel_on_refresh:
            worker.cancel(manifest["operation_id"])
    scheduler.on_refresh = on_refresh
    result = worker.activate_local(manifest, directory)
    if cancel_on_refresh:
        assert result["reason_code"] == "cancelled"
        assert not (worker.root / "current.json").exists()
        assert events == ["hash", "stage_enter", "publish", "refresh", "revoke", "stage_exit"]
    else:
        assert result["status"] == "available"
        worker._revoke(worker._active)
        assert events == ["hash", "stage_enter", "publish", "refresh", "stage_exit",
                          "stage_enter", "revoke", "stage_exit"]


@pytest.mark.parametrize("pause_admission", [False, True])
@pytest.mark.parametrize("public_cancel", [False, True])
def test_cancel_before_ensure_never_starts_or_replaces_active_registration(
        setup, tmp_path, monkeypatch, pause_admission, public_cancel):
    worker, scheduler, client = setup
    source = tmp_path / "existing"
    existing = segment(source, operation="op-existing")
    receiving(monkeypatch, source)
    worker.handle(command(existing))
    assert client.wait(operation="op-existing")["status"] == "available"
    finish(worker)
    previous_op = worker._active
    previous = (worker.root / "current.json").read_bytes()
    late = segment(tmp_path / "late", operation="op-late")
    entered, release = threading.Event(), threading.Event()
    original_copy = module.deepcopy
    def blocked_copy(value):
        if value is late_payload["manifest"] and pause_admission:
            entered.set()
            assert release.wait(5)
        return original_copy(value)
    late_payload = command(late)
    monkeypatch.setattr(module, "deepcopy", blocked_copy)
    monkeypatch.setattr(assets, "receive_segment", lambda *a, **kw: pytest.fail("cancelled ensure transferred"))
    thread = None
    try:
        if pause_admission:
            thread = threading.Thread(target=worker.handle, args=(late_payload, client))
            thread.start()
            assert entered.wait(2)
        if public_cancel:
            worker.cancel("op-late")
        else:
            worker.handle({"action": "cancel", "operation_id": "op-late"}, client)
        if not pause_admission:
            worker.handle(late_payload, client)
    finally:
        release.set()
        if thread is not None:
            thread.join(5)
            assert not thread.is_alive()
    assert worker._active is previous_op and not previous_op.cancelled.is_set()
    assert worker._runner is None and worker._pending is None
    assert (worker.root / "current.json").read_bytes() == previous
    assert scheduler.refreshed == 1
    worker.handle({"action": "inventory", "operation_id": "op-late"}, client)
    assert client.wait(action="inventory_result", operation="op-late")["distributed"]
    other = segment(tmp_path / "other", operation="op-other")
    receiving(monkeypatch, tmp_path / "other")
    worker.handle(command(other), client)
    assert client.wait(operation="op-other")["status"] == "available"
    finish(worker)
    assert scheduler.refreshed == 2


@pytest.mark.parametrize("change", ["connection", "epoch", "generation"])
def test_cancel_tombstone_is_confined_to_captured_master_connection(setup, tmp_path, monkeypatch, change):
    worker, scheduler, client = setup
    manifest = segment(tmp_path / "source")
    worker.handle({"action": "cancel", "operation_id": manifest["operation_id"]}, client)
    worker.handle(command(manifest), client)
    assert worker._runner is None and worker._active is None
    if change == "connection":
        scheduler._tcp_client = Client()
        current = scheduler._tcp_client
        # A delayed message from the old connection cannot poison the new scope.
        worker.handle({"action": "cancel", "operation_id": manifest["operation_id"]}, client)
    else:
        current = client
        if change == "epoch":
            current.registration_epoch += 1
        else:
            current._connection_generation += 1
    receiving(monkeypatch, tmp_path / "source")
    worker.handle(command(manifest), current)
    assert current.wait()["status"] == "available"
    finish(worker)
    assert scheduler.refreshed == 1


def test_cancel_tombstones_are_bounded_and_newest_remains_fenced(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    monkeypatch.setattr(module, "_MAX_CANCELLED_OPERATIONS", 3)
    for operation in ("evicted", "kept-one", "kept-two", "kept-three"):
        worker.handle({"action": "cancel", "operation_id": operation}, client)
    assert len(worker._cancelled_operations) == 3
    newest = segment(tmp_path / "newest", operation="kept-three")
    worker.handle(command(newest), client)
    assert worker._active is None and worker._runner is None
    oldest = segment(tmp_path / "oldest", operation="evicted")
    receiving(monkeypatch, tmp_path / "oldest")
    worker.handle(command(oldest), client)
    assert client.wait(operation="evicted")["status"] == "available"
    finish(worker)


def test_invalid_client_cannot_tombstone_a_future_operation(setup, tmp_path, monkeypatch):
    worker, scheduler, client = setup
    manifest = segment(tmp_path / "source")
    worker.handle({"action": "cancel", "operation_id": manifest["operation_id"]}, Client())
    assert not worker._cancelled_operations
    receiving(monkeypatch, tmp_path / "source")
    worker.handle(command(manifest), client)
    assert client.wait()["status"] == "available"
    finish(worker)


@pytest.mark.parametrize("local", [False, True])
def test_global_cancel_only_cancels_active_work_and_allows_same_id_again(tmp_path, monkeypatch, local):
    errors = []
    original_error_code = module._error_code
    def error_code(error, stage):
        code = original_error_code(error, stage)
        errors.append({"type": type(error).__name__, "stage": stage, "code": code,
                       "errno": getattr(error, "errno", None), "winerror": getattr(error, "winerror", None)})
        return code
    monkeypatch.setattr(module, "_error_code", error_code)
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master" if local else "client")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built" if local else worker.root / "built"
    manifest = segment(directory, node="master" if local else "worker-one")
    if not local:
        monkeypatch.setattr(assets, "receive_segment", lambda *a, **kw: {"manifest": manifest, "model_path": str(directory / "segment.gguf")})
    if local:
        assert worker.activate_local(manifest, directory)["status"] == "available"
    else:
        worker.handle(command(manifest))
        reply = scheduler._tcp_client.wait()
        assert reply["status"] == "available", (reply, errors)
        finish(worker)
    previous_op = worker._active
    worker.cancel()
    assert previous_op.cancelled.is_set() and not worker._cancelled_operations
    if local:
        assert worker.activate_local(manifest, directory)["status"] == "available"
    else:
        scheduler._tcp_client.messages.clear()
        worker.handle(command(manifest))
        reply = scheduler._tcp_client.wait()
        assert reply["status"] == "available", (reply, errors)
        finish(worker)
    assert scheduler.refreshed == 2


@contextmanager
def windows_reader_without_delete_sharing(path):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes, close.restype = [wintypes.HANDLE], wintypes.BOOL
    handle = create(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    assert handle != wintypes.HANDLE(-1).value, ctypes.get_last_error()
    try:
        yield
    finally:
        assert close(handle), ctypes.get_last_error()



@pytest.fixture(params=[False, True], ids=["remote", "local"])
def repeat_registration(tmp_path, monkeypatch, request):
    local = request.param
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master" if local else "client")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built" if local else worker.root / "built"
    manifest = segment(directory, node="master" if local else "worker-one")
    monkeypatch.setattr(assets, "receive_segment", lambda *a, **kw: {
        "manifest": manifest, "model_path": str(directory / "segment.gguf")})
    def activate():
        if local:
            return worker.activate_local(manifest, directory)
        worker.handle(command(manifest))
        reply = scheduler._tcp_client.wait()
        finish(worker)
        return reply
    try:
        reply = activate()
        assert reply["status"] == "available", reply
        worker.cancel()
        if not local:
            scheduler._tcp_client.messages.clear()
        yield worker, scheduler, directory, manifest, activate
    finally:
        worker.cancel()
        finish(worker)


@pytest.mark.skipif(os.name != "nt", reason="native Windows delete-sharing contract")
def test_registration_retries_real_windows_reader_without_delete_sharing(repeat_registration, monkeypatch):
    worker, scheduler, directory, manifest, activate = repeat_registration
    current = worker.root / "current.json"
    previous = current.read_bytes()
    real_replace = module.os.replace
    attempts, denied = [], []
    def replace(source, target):
        if Path(target) != current:
            return real_replace(source, target)
        attempts.append(Path(source).name)
        assert scheduler._host.full_chat_execution_lock._is_owned()
        assert scheduler._task_worker_stage_lock._is_owned()
        if not denied:
            with windows_reader_without_delete_sharing(current):
                try:
                    return real_replace(source, target)
                except OSError as error:
                    denied.append({"type": type(error).__name__, "winerror": error.winerror,
                                   "fallback": module._error_code(error, "registration_failed")})
                    raise
        return real_replace(source, target)
    monkeypatch.setattr(module.os, "replace", replace)
    reply = activate()
    assert reply["status"] == "available", (reply, denied, attempts)
    assert denied and denied[0]["winerror"] in {5, 32, 33}
    assert denied[0]["fallback"] == "registration_failed"
    assert len(attempts) == 2
    assert current.read_bytes() == previous and scheduler.refreshed == 2
    assert not list(worker.root.glob(".current-*.tmp"))


@pytest.mark.parametrize("platform,winerror,expected_attempts", [
    ("win32", 5, 8), ("win32", 32, 8), ("win32", 33, 8),
    ("win32", 87, 1), ("linux", 32, 1), ("win32", None, 1),
])
def test_registration_replace_failure_is_bounded_and_preserves_pointer(
        repeat_registration, monkeypatch, platform, winerror, expected_attempts):
    worker, scheduler, directory, manifest, activate = repeat_registration
    current = worker.root / "current.json"
    previous = current.read_bytes()
    real_replace = module.os.replace
    calls, delays = [], []
    def replace(source, target):
        if Path(target) != current:
            return real_replace(source, target)
        calls.append(Path(source).name)
        error = PermissionError("simulated Windows pointer conflict")
        error.winerror = winerror
        raise error
    original_wait = threading.Event.wait
    def wait(event, delay=None):
        if event is not getattr(worker._active, "cancelled", None):
            return original_wait(event, delay)
        assert worker._lock.acquire(blocking=False), "retry waited holding operation mutex"
        worker._lock.release()
        assert scheduler._host.full_chat_execution_lock._is_owned()
        assert scheduler._task_worker_stage_lock._is_owned()
        delays.append(delay)
        return event.is_set()
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module.os, "replace", replace)
    monkeypatch.setattr(threading.Event, "wait", wait)
    reply = activate()
    assert reply["status"] == "failed" and reply["reason_code"] == "registration_failed", reply
    assert len(calls) == expected_attempts and len(delays) == expected_attempts - 1
    assert sum(delays) < 1.3
    assert current.read_bytes() == previous and scheduler.refreshed == 1
    assert not list(worker.root.glob(".current-*.tmp"))


@pytest.mark.skipif(
    os.name != "nt",
    reason="该重试路径是 Windows 专用的文件替换冲突（winerror ∈ {5,32,33}），"
           "见 src/model_supply_worker.py:1070 —— 非 Windows 上这段代码不执行",
)
@pytest.mark.parametrize("repeat_registration,change", [
    (local, change)
    for local in (False, True)
    for change in ("cancel", "active", "mode", "disabled", "role", "client", "epoch", "generation", "artifact")
    if not (local and change in {"client", "epoch", "generation"}) and not (not local and change == "role")
], indirect=["repeat_registration"])
def test_registration_retry_rechecks_cancellation_context_and_artifact(
        repeat_registration, monkeypatch, change):
    worker, scheduler, directory, manifest, activate = repeat_registration
    local = scheduler._tcp_client is None
    current = worker.root / "current.json"
    previous = current.read_bytes()
    real_replace = module.os.replace
    calls, waits, errors = [], [], []
    original_error_code = module._error_code
    def error_code(error, stage):
        code = original_error_code(error, stage)
        errors.append(code)
        return code
    monkeypatch.setattr(module, "_error_code", error_code)
    def replace(source, target):
        if Path(target) != current:
            return real_replace(source, target)
        calls.append(Path(source).name)
        error = PermissionError("simulated Windows pointer conflict")
        error.winerror = 32
        raise error
    original_wait = threading.Event.wait
    def wait(event, delay=None):
        if event is not getattr(worker._active, "cancelled", None):
            return original_wait(event, delay)
        waits.append(delay)
        assert worker._lock.acquire(blocking=False), "retry waited holding operation mutex"
        worker._lock.release()
        if change == "cancel":
            worker.cancel()
        elif change == "active":
            with worker._lock:
                worker._active = None
        elif change == "mode":
            config.RUN_MODE = "local"
        elif change == "disabled":
            scheduler.enabled = False
        elif change == "role":
            scheduler.role = "client"
        elif change == "client":
            scheduler._tcp_client = Client()
        elif change == "epoch":
            scheduler._tcp_client.registration_epoch += 1
        elif change == "generation":
            scheduler._tcp_client._connection_generation += 1
        else:
            with (directory / "segment.gguf").open("r+b") as file:
                file.seek(-1, 2)
                file.write(b"x")
        return event.is_set()
    monkeypatch.setattr(module.os, "replace", replace)
    monkeypatch.setattr(threading.Event, "wait", wait)
    # Remote context retirement correctly suppresses a reply to the stale
    # connection. Inspect the worker's recorded error after joining instead.
    if local:
        reply = activate()
    else:
        client = scheduler._tcp_client
        worker.handle(command(manifest))
        finish(worker)
        replies = [item for item in client.messages if item["action"] == "result"]
        reply = replies[-1] if replies else None
    expected = "artifact_changed" if change == "artifact" else "cancelled"
    assert errors == [expected]
    if reply is not None:
        assert reply["status"] == "failed" and reply["reason_code"] == expected, reply
    assert len(calls) == len(waits) == 1
    assert current.read_bytes() == previous and scheduler.refreshed == 1
    assert not list(worker.root.glob(".current-*.tmp"))


@pytest.mark.parametrize("cancel_kind", ["explicit", "callback"])
def test_local_cancel_before_activate_preserves_registration_without_hashing(tmp_path, monkeypatch, cancel_kind):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    old_dir, new_dir = worker.root.parent / "old", worker.root.parent / "new"
    old = segment(old_dir, operation="op-old", node="master")
    new = segment(new_dir, operation="op-new", node="master")
    assert worker.activate_local(old, old_dir)["status"] == "available"
    previous_op = worker._active
    previous = (worker.root / "current.json").read_bytes()
    event = threading.Event()
    if cancel_kind == "explicit":
        worker.cancel("op-new")
    else:
        event.set()
    monkeypatch.setattr(module, "_verify_files", lambda *a, **kw: pytest.fail("cancelled local activation hashed"))
    result = worker.activate_local(new, new_dir, cancelled=lambda: event.is_set())
    assert result["status"] == "failed" and result["reason_code"] == "cancelled"
    assert worker._active is previous_op and not previous_op.cancelled.is_set()
    assert (worker.root / "current.json").read_bytes() == previous
    assert scheduler.refreshed == 1


@pytest.mark.parametrize("stage", ["manifest", "fingerprint", "verify", "marker", "temporary", "publish", "cache", "refresh"])
def test_local_activation_checks_live_coordinator_event_at_each_registration_stage(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    old_dir, new_dir = worker.root.parent / "old", worker.root.parent / "new"
    old = segment(old_dir, operation="op-old", node="master")
    new = segment(new_dir, operation="op-new", node="master")
    assert worker.activate_local(old, old_dir)["status"] == "available"
    previous = (worker.root / "current.json").read_bytes()
    event = threading.Event()
    hook_name = {"manifest": "_manifest", "fingerprint": "_selection_key", "verify": "_verify_files",
                 "marker": "_json", "cache": "_remember_selection"}.get(stage)
    if hook_name:
        original = getattr(module, hook_name)
        def cancelled_after_stage(*args, **kwargs):
            result = original(*args, **kwargs)
            if stage != "marker" or args[0] == new_dir / "supply.manifest.json":
                event.set()
            return result
        monkeypatch.setattr(module, hook_name, cancelled_after_stage)
    elif stage == "temporary":
        original = Path.write_text
        def write_then_cancel(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.name.startswith(".current-"):
                event.set()
            return result
        monkeypatch.setattr(Path, "write_text", write_then_cancel)
    elif stage == "publish":
        original = module.os.replace
        def publish_then_cancel(source, dest):
            result = original(source, dest)
            if Path(dest).name == "current.json":
                event.set()
            return result
        monkeypatch.setattr(module.os, "replace", publish_then_cancel)
    else:
        scheduler.on_refresh = event.set
    result = worker.activate_local(new, new_dir, cancelled=lambda: event.is_set())
    assert event.is_set()
    assert result["status"] == "failed" and result["reason_code"] == "cancelled"
    assert (worker.root / "current.json").read_bytes() == previous
    assert scheduler.refreshed == (2 if stage == "refresh" else 1)
    assert worker.selected_model_path() == old_dir / "segment.gguf"


def test_local_coordinator_event_cancels_while_waiting_for_stage_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    entered, event = threading.Event(), threading.Event()
    original_admission = worker._admission
    @contextmanager
    def admission():
        entered.set()
        with original_admission():
            yield
    monkeypatch.setattr(worker, "_admission", admission)
    result = {}
    with scheduler._task_worker_stage_lock:
        thread = threading.Thread(target=lambda: result.update(worker.activate_local(manifest, directory, cancelled=lambda: event.is_set())))
        thread.start()
        assert entered.wait(2)
        event.set()
    thread.join(5)
    assert not thread.is_alive()
    assert result["reason_code"] == "cancelled"
    assert not (worker.root / "current.json").exists() and scheduler.refreshed == 0


@pytest.mark.skipif(os.name != "nt", reason="native Windows pointer sharing")
def test_selection_reader_does_not_race_pointer_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    assert worker.activate_local(manifest, directory)["status"] == "available"
    current = worker.root / "current.json"
    opened, release, admitted, replaced = (threading.Event() for _ in range(4))
    real_read, real_replace, real_idle = Path.read_bytes, module.os.replace, worker._assert_idle
    outcomes = {}
    def read(path):
        if path == current and threading.current_thread().name == "held-selection-reader":
            with path.open("rb") as handle:
                opened.set()
                assert release.wait(5), "reader was not released"
                return handle.read()
        return real_read(path)
    def replace(source, target):
        if Path(target) == current:
            replaced.set()
        return real_replace(source, target)
    def idle(op):
        real_idle(op)
        admitted.set()
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(module.os, "replace", replace)
    monkeypatch.setattr(worker, "_assert_idle", idle)
    reader = threading.Thread(target=lambda: outcomes.update(selected=get_selected_model(worker.root)),
                              name="held-selection-reader")
    writer = threading.Thread(target=lambda: outcomes.update(reply=worker.activate_local(manifest, directory)),
                              name="selection-writer")
    reader.start()
    try:
        assert opened.wait(5)
        writer.start()
        assert admitted.wait(5)
        assert not replaced.wait(.15), "writer attempted replacement while a product reader held current.json"
    finally:
        release.set()
        reader.join(5)
        if writer.ident is not None:
            writer.join(5)
    assert not reader.is_alive() and not writer.is_alive()
    assert outcomes["reply"]["status"] == "available"
    assert outcomes["selected"] is not None


def test_registration_failure_has_local_exception_context(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    def fail_refresh():
        raise OSError(13, "capability refresh denied")
    scheduler.on_refresh = fail_refresh
    reply = worker.activate_local(manifest, directory)
    assert reply["status"] == "failed" and reply["reason_code"] == "registration_failed"
    records = [r for r in caplog.records if getattr(r, "supply_phase", None) == "registration"]
    assert len(records) == 1
    record = records[0]
    assert record.supply_operation_id == manifest["operation_id"]
    assert record.supply_node_id == "master" and isinstance(record.exc_info[1], OSError)
    assert record.supply_errno == 13
    assert "capability refresh denied" not in json.dumps(reply)


def test_failed_rollback_is_reported_and_selection_recovers(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    old_dir, new_dir = worker.root.parent / "old", worker.root.parent / "new"
    old = segment(old_dir, operation="op-old", node="master")
    new = segment(new_dir, operation="op-new", node="master")
    assert worker.activate_local(old, old_dir)["status"] == "available"
    previous = (worker.root / "current.json").read_bytes()
    real_replace = module.os.replace
    def replace(source, target):
        if Path(source).name.startswith(".restore-"):
            raise PermissionError(13, "restore blocked")
        return real_replace(source, target)
    monkeypatch.setattr(module.os, "replace", replace)
    scheduler.on_refresh = lambda: (_ for _ in ()).throw(RuntimeError("refresh failed"))
    reply = worker.activate_local(new, new_dir)
    assert reply["status"] == "failed" and reply["reason_code"] == "registration_rollback_failed"
    assert get_selected_model(worker.root) is None
    assert worker._active.committed is not None
    assert any(getattr(r, "supply_phase", None) == "rollback" and r.exc_info for r in caplog.records)
    monkeypatch.setattr(module.os, "replace", real_replace)
    worker._revoke(worker._active)
    assert (worker.root / "current.json").read_bytes() == previous
    assert get_selected_model(worker.root)["manifest"]["operation_id"] == "op-old"


@pytest.mark.parametrize("master,keep,byte_cap", [(False, 2, 8 * 1024**3), (True, 6, 8 * 1024**3),
                                                  (False, 0, 8 * 1024**3), (True, 6, 0)])
def test_worker_applies_runtime_cache_budget(tmp_path, monkeypatch, master, keep, byte_cap):
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    root = worker.root.parent / "operations" if master else worker.root
    root.mkdir(parents=True)
    policy = {"MODEL_SUPPLY_CACHE_KEEP": keep if not master else 2,
              "MODEL_SUPPLY_CACHE_MASTER_KEEP": keep if master else 6,
              "MODEL_SUPPLY_CACHE_MAX_BYTES": byte_cap}
    scheduler._runtime_setting = policy.__getitem__
    @contextmanager
    def native_paths(root):
        yield []
    monkeypatch.setattr(worker, "_native_gc_paths", native_paths)
    calls = []
    def collect(root, **kwargs):
        calls.append((root, kwargs))
        return {"removed": 0, "reclaimed_bytes": 0, "skipped": {}, "examined": 0}
    monkeypatch.setattr(assets, "collect_supply_segments", collect)
    assert worker.collect_garbage(root=root)["skipped"] == {}
    assert calls[0][0] == root
    assert calls[0][1]["keep"] == keep and calls[0][1]["max_retained_bytes"] == byte_cap


def test_supply_cache_defaults_and_environment_are_bounded():
    settings = config.build_settings({"QLH_CLUSTER_SECRET": "a" * 32})
    assert settings["MODEL_SUPPLY_CACHE_KEEP"] == 2
    assert settings["MODEL_SUPPLY_CACHE_MASTER_KEEP"] == 6
    assert settings["MODEL_SUPPLY_CACHE_MAX_BYTES"] == 8 * 1024**3
    settings = config.build_settings({"QLH_CLUSTER_SECRET": "a" * 32,
        "QLH_MODEL_SUPPLY_CACHE_KEEP": "0", "QLH_MODEL_SUPPLY_CACHE_MASTER_KEEP": "128",
        "QLH_MODEL_SUPPLY_CACHE_MAX_BYTES": "0"})
    assert settings["MODEL_SUPPLY_CACHE_KEEP"] == settings["MODEL_SUPPLY_CACHE_MAX_BYTES"] == 0
    assert settings["MODEL_SUPPLY_CACHE_MASTER_KEEP"] == 128


def test_registration_rejects_replace_success_without_pointer_visibility(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    old_dir, new_dir = worker.root.parent / "old", worker.root.parent / "new"
    old = segment(old_dir, operation="op-old", node="master")
    new = segment(new_dir, operation="op-new", node="master")
    assert worker.activate_local(old, old_dir)["status"] == "available"
    previous = (worker.root / "current.json").read_bytes()
    monkeypatch.setattr(module.os, "replace", lambda *args: None)
    reply = worker.activate_local(new, new_dir)
    assert reply["status"] == "failed" and reply["reason_code"] == "registration_pointer_visibility_failed"
    assert (worker.root / "current.json").read_bytes() == previous
    assert get_selected_model(worker.root) is None and scheduler.refreshed == 1
