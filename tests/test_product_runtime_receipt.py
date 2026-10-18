"""Runtime receipts use real load proof and existing owners; no devices or services."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors import safe_open

import model_module
from product_runtime_receipt import (
    LayerMaterializationProof, NativeLayerLoadProof, RuntimeObservationError,
    capture_runtime,
)
from tests.test_safetensors_supply_worker import tiny_supply
from tests.test_worker_prepare import identity as worker_identity, Client, command
from tests.test_worker_prepare_runtime import supply_identity
from worker_prepare import LayerRuntime, WorkerPrepareController


def _api(manager=None, *, role="master", node_id="master"):
    host = NS(full_chat_execution_lock=threading.RLock(), _pipeline_candidate_lock=threading.RLock())
    selected = {"operation_id": "op-real", "config_id": "config-real", "generation": 3}
    transaction = {**selected, "candidate_id": "candidate-real", "phase": "ready", "published": True,
                   "expected": {"pc": {"worker_prepare_identity": deepcopy(selected)}}}
    scheduler = NS(_inference_lock=threading.Lock(), _layer_execution_lock=threading.RLock(),
                   _layer_config_lock=threading.RLock(), _pipeline_load_transaction=transaction,
                   _effective_role=lambda: role, get_effective_node_id=lambda: node_id)
    host._pipeline_candidate_transaction = NS(transaction_id="candidate-real", phase="finalized",
                                               manager=manager, previous_manager=None)
    return NS(model_host=host, scheduler=scheduler, _peek_model_manager=lambda: manager)


@pytest.fixture
def loaded(tiny_supply):
    import model_supply_worker
    manifest = tiny_supply.activate(2)
    native_host = tiny_supply.host()
    value = supply_identity(manifest, 1)
    native_host.prepare_worker_runtime(command("prepare", value), model_supply_worker.get_selected_model(), threading.Event())
    native_host.commit_worker_runtime(value)
    native_host.finalize_worker_runtime(value)
    return NS(manager=native_host._worker_runtime_active.manager, manifest=manifest, host=native_host)


def _proof(manager):
    return manager._layer_load_metrics["materialization_proof"]


def test_tracker_records_actual_shape_source_and_target_bytes_and_freezes(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tracker = model_module._LayerRangeLoadTracker(
        architecture="qwen2", start_layer=1, end_layer=2, layer_prefix="model.layers.",
        selected_prefixes=["model.layers.1."], target_dtype=torch.float16,
    )
    key = "model.layers.1.weight"
    handle = NS(get_tensor=lambda name: torch.ones((2, 3), dtype=torch.float32))
    tracker.materialize(handle, key, source_file=str(tmp_path / "part.safetensors"))
    metrics = tracker.finish()
    proof = metrics["materialization_proof"]
    assert proof.tensors[0].shape == (2, 3)
    assert proof.tensors[0].element_size == 2
    assert proof.tensors[0].source_element_size == 4
    assert proof.tensors[0].source_file == str(tmp_path / "part.safetensors")
    assert metrics["source_tensor_bytes"] == 24
    assert proof.materialized_bytes == metrics["materialized_tensor_bytes"] == 12
    with pytest.raises(FrozenInstanceError):
        proof.tensors[0].source_file = "forged"
    with pytest.raises(FrozenInstanceError):
        proof.materialized_bytes = 1
    detached = proof.as_dict()
    detached["tensor_metadata"][key]["shape"].append(99)
    detached["loaded_tensor_names"].clear()
    tracker.loaded_keys.clear()
    assert proof.as_dict()["tensor_metadata"][key]["shape"] == [2, 3]
    assert proof.as_dict()["loaded_tensor_names"] == [key]


def test_tracker_rejects_duplicate_before_fetch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tracker = model_module._LayerRangeLoadTracker(
        architecture="qwen2", start_layer=1, end_layer=2, layer_prefix="model.layers.",
        selected_prefixes=["model.layers.1."], target_dtype=torch.float32,
    )
    requested = []
    handle = NS(get_tensor=lambda key: requested.append(key) or torch.ones(2))
    tracker.materialize(handle, "model.layers.1.weight")
    with pytest.raises(RuntimeError, match="duplicate"):
        tracker.materialize(handle, "model.layers.1.weight")
    assert requested == ["model.layers.1.weight"]


def test_real_selective_load_sources_shapes_and_detached_public_metrics(loaded):
    proof = _proof(loaded.manager)
    assert isinstance(proof, LayerMaterializationProof)
    assert proof.artifact_sha256 == loaded.manifest["artifact_sha256"]
    assert proof.total_layers == loaded.manifest["total_layers"] == 4
    assert proof.layer_range == (1, 3)
    for tensor in proof.tensors:
        assert Path(tensor.source_file).is_absolute()
        with safe_open(tensor.source_file, framework="pt", device="cpu") as shard:
            actual = shard.get_tensor(tensor.name)
        assert tuple(actual.shape) == tensor.shape
        assert actual.element_size() == tensor.source_element_size
        assert tensor.element_size == 4  # actual CPU materialization uses fp32
    public = loaded.manager.get_model_info()["layer_load_metrics"]["materialization_proof"]
    json.dumps(public, allow_nan=False)
    public["tensor_metadata"][proof.tensors[0].name]["source_file"] = "forged"
    assert _proof(loaded.manager).tensors[0].source_file != "forged"


def test_master_receipt_uses_loaded_facts_and_is_stable_without_capture_time(loaded, monkeypatch):
    api = _api(loaded.manager)
    # Deliberately contradictory plans must have no influence on actual facts.
    api.scheduler._active_pipeline_capacity_plan = {"engine": "llama_cpp", "layer_range": [0, 4], "total_layers": 999}
    monkeypatch.setattr(loaded.manager, "get_pipeline_descriptor", lambda: pytest.fail("diagnostic inspected/hashed model"))
    monkeypatch.setattr(loaded.manager, "load_model", lambda *a, **k: pytest.fail("diagnostic loaded model"))
    first = capture_runtime(api)
    second = capture_runtime(api)
    assert first["node_id"] == "master"
    assert first["layer_range"] == [1, 3] and first["total_layers"] == 4
    assert first["engine"] == "pytorch" and first["full_model_loaded"] is False
    assert first["artifact"]["sha256"] == loaded.manifest["artifact_sha256"]
    assert "rehashed_sha256" not in first["artifact"]
    assert first["loader_pid"] == first["pid"] == first["materialization"]["pid"]
    assert first["created_at"] == first["materialization"]["created_at"]
    for key in ("operation_id", "config_id", "generation"):
        assert first[key] == first["materialization"][key]
    first.pop("captured_at")
    second.pop("captured_at")
    assert first == second


@pytest.mark.parametrize("change,reason", [
    ("no_proof", "immutable selective"),
    ("unloaded", "no loaded model"),
    ("full", "full decoder"),
    ("range", "range/total"),
    ("count", "decoder count"),
    ("hash", "artifact SHA256"),
    ("bytes", "byte accounting"),
    ("source", "source shard"),
    ("shard_escape", "outside the loaded artifact"),
    ("unowned", "outside owned boundaries"),
    ("no_manager_lock", "manager lock"),
])
def test_master_rejects_missing_or_contradictory_real_load_facts(loaded, change, reason, tmp_path):
    manager = loaded.manager
    proof = _proof(manager)
    if change == "no_proof":
        manager._layer_load_metrics = None
    elif change == "unloaded":
        manager.model = None
    elif change == "full":
        manager.layer_range = (0, 4)
    elif change == "range":
        manager.layer_range = (0, 2)
    elif change == "count":
        manager.model.model.layers = torch.nn.ModuleList(list(manager.model.model.layers)[:1])
    elif change == "hash":
        manager._layer_load_metrics["materialization_proof"] = replace(proof, artifact_sha256="")
    elif change == "bytes":
        manager._layer_load_metrics["materialization_proof"] = replace(proof, materialized_bytes=proof.materialized_bytes + 1)
    elif change in {"source", "shard_escape", "unowned"}:
        first = proof.tensors[0]
        replacement = replace(first, source_file="" if change == "source" else str(tmp_path / "unloaded.safetensors"))
        if change == "unowned":
            replacement = replace(first, name="visual.full_model.weight")
        manager._layer_load_metrics["materialization_proof"] = replace(proof, tensors=(replacement, *proof.tensors[1:]))
    elif change == "no_manager_lock":
        manager._lock = None
    with pytest.raises(RuntimeObservationError, match=reason):
        capture_runtime(_api(manager))
    # Keep the shared fixture cleanup independent of deliberately broken locks.
    if change == "no_manager_lock":
        manager._lock = threading.RLock()


@pytest.mark.parametrize("field,value,reason", [
    ("phase", "preparing", "ready/published"),
    ("published", False, "ready/published"),
    ("cancel_requested", True, "ready/published"),
    ("candidate_id", "other", "installed manager"),
    ("generation", 4, "operation/config/generation"),
    ("expected", {}, "operation identity"),
])
def test_master_requires_committed_ssot_identity(loaded, field, value, reason):
    api = _api(loaded.manager)
    api.scheduler._pipeline_load_transaction[field] = value
    with pytest.raises(RuntimeObservationError, match=reason):
        capture_runtime(api)


def test_master_rejects_pending_previous_owner_and_wrong_candidate_manager(loaded):
    api = _api(loaded.manager)
    api.model_host._pipeline_candidate_transaction.previous_manager = object()
    with pytest.raises(RuntimeObservationError, match="pending retirement"):
        capture_runtime(api)
    api.model_host._pipeline_candidate_transaction.previous_manager = None
    api.model_host._pipeline_candidate_transaction.manager = object()
    with pytest.raises(RuntimeObservationError, match="installed manager"):
        capture_runtime(api)


def test_master_never_constructs_lazy_manager():
    api = _api(None)
    api.model_host.factory = lambda: pytest.fail("lazy factory used")
    with pytest.raises(RuntimeObservationError, match="existing model manager"):
        capture_runtime(api)


@pytest.mark.parametrize("method", ["unload_model", "abort_pipeline_materialization"])
def test_real_manager_retirement_clears_immutable_load_proof(loaded, method):
    manager = loaded.manager
    assert _proof(manager).tensors
    getattr(manager, method)()
    assert manager._layer_load_metrics is None
    with pytest.raises(RuntimeObservationError, match="no loaded model|engine is absent"):
        capture_runtime(_api(manager))


def _worker_api(*, manager=None, node_id="pc", child=True):
    api = _api(role="client", node_id=node_id)
    value = worker_identity(fmt="safetensors" if manager is not None else "gguf", node_id=node_id)
    path = Path("C:/observed/segment.gguf")
    if manager is not None:
        value["artifact_sha256"] = _proof(manager).artifact_sha256
        value["stage_model_identity"]["sha256"] = value["artifact_sha256"]
        value["layer_range"] = list(manager.layer_range)
        value["total_layers"] = manager._total_model_layers
    selected = {"manifest": deepcopy(value), "model_path": str(path),
                "directory": manager._model_path if manager is not None else str(path.parent)}
    upstream = None if manager is not None else NS(model_path=path, n_layer=2, is_available=True,
        _worker=NS(pid=42424, poll=lambda: None) if child else None,
        _lib=object(), _handle=object())
    owner = NS(_worker_runtime_lock=threading.RLock(), _worker_runtime_phase="finalized",
               _worker_runtime_active=LayerRuntime(deepcopy(value), selected, upstream=upstream, manager=manager),
               _worker_runtime_candidate=None, _worker_runtime_previous=None, _worker_runtime_orphans=[],
               _worker_runtime_fenced=threading.Event())
    scheduler = api.scheduler
    scheduler._tcp_client = Client()
    scheduler._control_fence = None
    scheduler.get_distributed_inference_enabled = lambda: True
    controller = WorkerPrepareController(scheduler)
    controller._runtime = owner
    controller._active_identity = deepcopy(value)
    controller._active_scope = controller._scope(None, scheduler._tcp_client, None)
    controller._stage_fenced = False
    scheduler._worker_prepare = controller
    return api, owner, controller


@pytest.mark.parametrize("child", [True, False])
def test_worker_native_uses_actual_upstream_path_count_and_loader_pid(child):
    api, owner, controller = _worker_api(child=child)
    api.scheduler._active_layer_config = {"layer_range": [0, 4], "total_layers": 100}
    runtime = capture_runtime(api)
    assert runtime["node_id"] == "pc" and runtime["total_layers"] == 4
    assert runtime["layer_range"] == [1, 3] and runtime["native_layer_count"] == 2
    assert runtime["loader_pid"] == (42424 if child else runtime["pid"])
    assert runtime["artifact"]["path"] == str(owner._worker_runtime_active.upstream.model_path)


def test_worker_pytorch_uses_real_materialization_receipt(loaded):
    api, owner, controller = _worker_api(manager=loaded.manager)
    result = capture_runtime(api)
    assert result["engine"] == "pytorch" and result["node_id"] == "pc"
    assert result["materialization"]["artifact_sha256"] == loaded.manifest["artifact_sha256"]
    assert result["materialization"]["operation_id"] == controller._active_identity["operation_id"]


@pytest.mark.parametrize("change,reason", [
    ("fence", "fenced"), ("disconnect", "disconnected"), ("registration", "disconnected"),
    ("epoch", "fenced"), ("owner_identity", "changed"), ("native_fence", "fenced"),
    ("manifest_total", "total_layers"), ("manifest_sha", "artifact_sha256"),
    ("native_count", "artifact/count"), ("wrong_path", "artifact/count"),
    ("dead_child", "not alive"), ("previous", "additional"), ("candidate", "additional"),
    ("orphan", "additional"),
])
def test_worker_fails_closed_for_stale_identity_or_incomplete_loader(change, reason):
    api, owner, controller = _worker_api()
    active = owner._worker_runtime_active
    if change == "fence":
        controller._stage_fenced = True
    elif change == "disconnect":
        api.scheduler._tcp_client.is_registered = False
    elif change == "registration":
        api.scheduler._tcp_client._connection_generation += 1
    elif change == "epoch":
        controller._epoch += 1
    elif change == "owner_identity":
        active.identity["operation_id"] = "different"
    elif change == "native_fence":
        owner._worker_runtime_fenced.set()
    elif change == "manifest_total":
        active.selected["manifest"]["total_layers"] = 8
    elif change == "manifest_sha":
        active.selected["manifest"]["artifact_sha256"] = "e" * 64
    elif change == "native_count":
        active.upstream.n_layer = 4
    elif change == "wrong_path":
        active.upstream.model_path = Path("C:/full.gguf")
    elif change == "dead_child":
        active.upstream._worker.poll = lambda: 1
    else:
        setattr(owner, "_worker_runtime_orphans" if change == "orphan" else "_worker_runtime_" + change,
                [active] if change == "orphan" else active)
    with pytest.raises(RuntimeObservationError, match=reason):
        capture_runtime(api)


def test_worker_never_holds_controller_lock_while_entering_native_owner():
    api, owner, controller = _worker_api()
    real_lock = owner._worker_runtime_lock
    class GuardLock:
        def acquire(self, **kwargs):
            assert not controller._lock._is_owned()
            return real_lock.acquire(**kwargs)
        def release(self):
            real_lock.release()
    owner._worker_runtime_lock = GuardLock()
    assert capture_runtime(api)["loaded"] is True


def test_worker_rechecks_scope_after_native_read():
    api, owner, controller = _worker_api()
    class ChangingUpstream:
        def __init__(self, values):
            self.__dict__.update(values)
        @property
        def is_available(self):
            controller._epoch += 1
            return True
    owner._worker_runtime_active.upstream = ChangingUpstream(vars(owner._worker_runtime_active.upstream))
    with pytest.raises(RuntimeObservationError, match="fenced"):
        capture_runtime(api)


def test_master_rechecks_transaction_after_manager_read(loaded):
    api = _api(loaded.manager)
    def peek():
        api.scheduler._pipeline_load_transaction["expected"]["pc"]["worker_prepare_identity"]["operation_id"] = "changed"
        return loaded.manager
    api._peek_model_manager = peek
    with pytest.raises(RuntimeObservationError, match="identity changed"):
        capture_runtime(api)


def test_runtime_capture_serializes_with_existing_model_change_lock(loaded):
    api = _api(loaded.manager)
    acquired = []
    started = threading.Event()
    writer = None
    def replace_model():
        started.set()
        with api.model_host.full_chat_execution_lock:
            acquired.append(True)
    def peek():
        nonlocal writer
        writer = threading.Thread(target=replace_model)
        writer.start()
        assert started.wait(1)
        assert not acquired
        return loaded.manager
    api._peek_model_manager = peek
    assert capture_runtime(api)["artifact"]["sha256"] == loaded.manifest["artifact_sha256"]
    writer.join(timeout=2)
    assert not writer.is_alive() and acquired == [True]


def _llama_fixture(tmp_path, monkeypatch, *, changed_on_load=False):
    from llama_engine import LlamaCppEngine
    path = tmp_path / "head.gguf"
    path.write_bytes(b"GGUF observed cropped file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"source_layer_range": [0, 2], "n_layer": 4, "artifact_sha256": digest,
                "source_model_sha256": "e" * 64}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    engine = LlamaCppEngine()
    engine._pipeline_descriptor = {"model_path": str(tmp_path / "whole.gguf"), "model_sha256": "e" * 64,
                                   "total_layers": 4}
    engine._pipeline_distributed_only = True
    monkeypatch.setattr(engine, "_find_layer_artifact", lambda start, end: str(path))
    def load(path_text):
        engine._model_path = path_text
        engine._pipeline_descriptor = None
        engine._layer_load_proof = None
        engine._model = NS(_model=NS(model="native-handle"))
        engine._loaded = True
        if changed_on_load:
            path.write_bytes(b"different artifact during loading")
    monkeypatch.setattr(engine, "_load_model_runtime", load)
    def native_count(pointer):
        assert pointer == "native-handle"
        return 2
    binding = NS(llama_model_n_layer=native_count)
    monkeypatch.setitem(sys.modules, "llama_cpp", NS(Llama=NS, llama_cpp=binding))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", binding)
    engine.load_layer_range(0, 2)
    return engine, path, digest


def test_llama_load_bound_proof_survives_source_descriptor_restoration(tmp_path, monkeypatch):
    engine, path, digest = _llama_fixture(tmp_path, monkeypatch)
    assert engine._pipeline_descriptor["model_path"].endswith("whole.gguf")
    assert engine._pipeline_descriptor["model_sha256"] != digest
    assert isinstance(engine._layer_load_proof, NativeLayerLoadProof)
    result = capture_runtime(_api(engine))
    assert result["artifact"]["path"] == str(path)
    assert result["artifact"]["sha256"] == digest
    assert result["layer_range"] == [0, 2] and result["total_layers"] == 4
    assert result["native_layer_count"] == 2 and result["loader_pid"] == result["pid"]
    engine.unload()
    assert engine._layer_load_proof is None


def test_llama_does_not_publish_load_bound_proof_if_file_changed_during_load(tmp_path, monkeypatch):
    engine, path, digest = _llama_fixture(tmp_path, monkeypatch, changed_on_load=True)
    assert engine._layer_load_proof is None
    with pytest.raises(RuntimeObservationError, match="load-bound GGUF"):
        capture_runtime(_api(engine))


def test_llama_missing_native_count_is_not_filled_from_assignment(tmp_path, monkeypatch):
    engine, path, digest = _llama_fixture(tmp_path, monkeypatch)
    monkeypatch.delitem(sys.modules, "llama_cpp.llama_cpp")
    with pytest.raises(RuntimeObservationError, match="native decoder count"):
        capture_runtime(_api(engine))


def test_llama_reports_actual_keep_head_child_and_rejects_different_artifact(tmp_path, monkeypatch):
    engine, path, digest = _llama_fixture(tmp_path, monkeypatch)
    engine._keep_head = NS(model_path=path, n_layer=2, is_available=True,
                           _worker=NS(pid=4444, poll=lambda: None))
    result = capture_runtime(_api(engine))
    assert result["loader_pid"] == 4444
    engine._keep_head.model_path = tmp_path / "whole.gguf"
    with pytest.raises(RuntimeObservationError, match="different artifacts/counts"):
        capture_runtime(_api(engine))
