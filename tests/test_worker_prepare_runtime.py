"""Real filtered segment loading, rollback, mandatory stage gate, and child retirement."""
import base64
from copy import deepcopy
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.engine_host import EngineHost
from scheduler_task_worker import SchedulerTaskWorkerMixin
from task_worker_adapter import remote_provider_id
from worker_prepare import LayerRuntime, WorkerPrepareController, close_layer_runtime
from worker_prepare_contract import WorkerPrepareError, stage_identity
from tests.test_worker_prepare import command, identity
from tests.test_safetensors_supply_worker import tiny_supply, _request


def supply_identity(manifest, generation):
    value = identity(generation, node_id="master")
    for key in ("operation_id", "node_id", "source_model_id", "source_model_sha256",
                "artifact_sha256", "manifest_sha256", "tokenizer_sha256", "layer_range",
                "total_layers", "hidden_size", "segment_mode"):
        value[key] = deepcopy(manifest[key])
    value["model_id"] = manifest["source_model_id"]
    value["model_sha256"] = manifest["source_model_sha256"]
    value["stage_model_identity"] = {"model_id": "supply-" + manifest["artifact_sha256"][:24],
                                     "engine": "pytorch", "format": "safetensors", "revision": "1",
                                     "sha256": manifest["artifact_sha256"]}
    return value


def request(manifest, value, hidden, **kw):
    result = _request(manifest, hidden, **kw)
    result.stage_fields["prepare_identity"] = stage_identity(value)
    return result


def test_real_prepare_filters_only_segment_and_commit_never_reloads(tiny_supply, monkeypatch):
    import model_module
    from model_supply_worker import get_selected_model
    manifest = tiny_supply.activate(0)
    value = supply_identity(manifest, 1)
    host = tiny_supply.host()
    materialized = []
    original = model_module._LayerRangeLoadTracker.materialize

    def track(tracker, handle, key, **kwargs):
        materialized.append(key)
        assert kwargs.get("source_file"), "real loader must record the source shard"
        return original(tracker, handle, key, **kwargs)

    def forbidden(*args, **kw):
        pytest.fail("worker prepare attempted whole-model loading")

    monkeypatch.setattr(model_module._LayerRangeLoadTracker, "materialize", track)
    monkeypatch.setattr(model_module.ModelManager, "load_model", forbidden)
    monkeypatch.setattr(model_module.ModelManager, "_load_pytorch", forbidden)
    monkeypatch.setattr(model_module.AutoModelForCausalLM, "from_pretrained", forbidden)
    host.prepare_worker_runtime(command("prepare", value), get_selected_model(), threading.Event())
    assert host.worker_runtime_loaded(value)
    candidate = host._worker_runtime_candidate.manager
    assert candidate.layer_range == (0, 2)
    assert candidate._layer_load_metrics["mode"] == "safetensors_key_filtered"
    assert materialized and all(not key.startswith("model.layers.") or 0 <= int(key.split(".")[2]) < 2 for key in materialized)
    count = len(materialized)
    host.commit_worker_runtime(value)
    assert host._supplied_layer_manager is candidate and len(materialized) == count
    out = host._execute_layer_forward_stage(
        request(manifest, value, np.ones((2, 16), dtype=np.float32)), threading.Event(),
    )
    assert out["hidden_out_f32"] and len(materialized) == count
    host.release_worker_runtime(value)
    assert candidate.model is None and host._supplied_layer_caches == {}


def test_real_abort_restores_old_segment_and_decode_kv_after_selection_changed(tiny_supply, monkeypatch):
    import model_supply_worker
    old_manifest = tiny_supply.activate(0)
    old_id = supply_identity(old_manifest, 1)
    host = tiny_supply.host()
    host.prepare_worker_runtime(command("prepare", old_id), model_supply_worker.get_selected_model(), threading.Event())
    host.commit_worker_runtime(old_id)
    host.finalize_worker_runtime(old_id)
    hidden = np.ones((2, 16), dtype=np.float32)
    host._execute_layer_forward_stage(request(old_manifest, old_id, hidden), threading.Event())
    old_manager, old_cache = host._supplied_layer_manager, host._supplied_layer_caches
    new_manifest = tiny_supply.activate(2)
    new_id = supply_identity(new_manifest, 2)
    host.prepare_worker_runtime(command("prepare", new_id), model_supply_worker.get_selected_model(), threading.Event())
    host.commit_worker_runtime(new_id)
    candidate = host._supplied_layer_manager
    host.abort_worker_runtime(new_id)
    assert candidate.model is None
    assert host._supplied_layer_manager is old_manager and host._supplied_layer_caches is old_cache
    monkeypatch.setattr(model_supply_worker, "get_selected_model", lambda: pytest.fail("committed stage re-resolved registration"))
    result = host._execute_layer_forward_stage(
        request(old_manifest, old_id, hidden[:1], pos_base=2), threading.Event(),
    )
    assert result["hidden_out_f32"] and host._supplied_layer_caches[0][1] == 3
    host.release_worker_runtime()


def test_real_cancel_after_segment_load_unloads_candidate(tiny_supply, monkeypatch):
    import model_module
    from model_supply_worker import get_selected_model
    manifest = tiny_supply.activate(0)
    value = supply_identity(manifest, 1)
    host, cancel = tiny_supply.host(), threading.Event()
    original = model_module.ModelManager.load_layer_range
    managers = []

    def load(manager, *args, **kw):
        original(manager, *args, **kw)
        managers.append(manager)
        cancel.set()

    monkeypatch.setattr(model_module.ModelManager, "load_layer_range", load)
    with pytest.raises(WorkerPrepareError, match="cancelled"):
        host.prepare_worker_runtime(command("prepare", value), get_selected_model(), cancel)
    assert len(managers) == 1 and managers[0].model is None
    assert not host.worker_runtime_present()


class Upstream:
    made = []

    def __init__(self, shim, path, **kw):
        self.model_path = path
        self.n_layer = 2
        self.n_embd = 16
        self.is_available = True
        self.resets = 0
        self._worker = None
        self.options = kw
        self.made.append(self)

    def reset(self):
        self.resets += 1

    def forward_hidden_to_hidden(self, hidden, **kw):
        return hidden + 1

    def forward_hidden_to_token(self, hidden, **kw):
        return 11

    def close(self):
        self.is_available = False


@pytest.fixture
def gguf(monkeypatch):
    import keep_head_shim
    import llama_keep_head
    monkeypatch.setattr(keep_head_shim, "resolve_keep_head_shim", lambda: "shim")
    monkeypatch.setattr(llama_keep_head, "KeepHeadUpstream", Upstream)
    Upstream.made = []
    value = identity(fmt="gguf")
    selected = {"model_path": "segment.gguf", "directory": "segment",
                "manifest": {**deepcopy(value), "format": "gguf"}}
    host = EngineHost.__new__(EngineHost)
    host._layer_upstream = None
    yield host, value, selected
    host.release_worker_runtime()


def gguf_request(value):
    hidden = np.ones((2, 16), dtype=np.float32)
    return SimpleNamespace(stage_fields={"prepare_identity": stage_identity(value), "layer_range": value["layer_range"],
                                        "hidden_spec": {"n_tokens": 2, "n_embd": 16}},
                           root_input={"hidden_f32": base64.b64encode(hidden.tobytes()).decode("ascii"), "want_hidden": True},
                           workflow_id="wf-native", model_identity=value["stage_model_identity"])


def test_gguf_prepare_probes_real_entry_commit_promotes_without_lazy_reload(gguf, monkeypatch):
    host, value, selected = gguf
    host.prepare_worker_runtime(command("prepare", value), selected, threading.Event())
    upstream = Upstream.made[0]
    assert upstream.resets == 1 and upstream.options["n_ctx"] == 64
    host.commit_worker_runtime(value)
    monkeypatch.setattr(host, "_build_worker_runtime", lambda *args: pytest.fail("stage reloaded native runtime"))
    assert host._execute_layer_forward_stage(gguf_request(value), threading.Event())["hidden_out_f32"]
    assert host._layer_upstream is upstream and len(Upstream.made) == 1
    upstream.is_available = False
    with pytest.raises(WorkerPrepareError):
        host._execute_layer_forward_stage(gguf_request(value), threading.Event())
    assert len(Upstream.made) == 1


def test_managed_runtime_never_accepts_missing_identity_or_prepared_only(gguf):
    host, value, selected = gguf
    host.prepare_worker_runtime(command("prepare", value), selected, threading.Event())
    with pytest.raises(WorkerPrepareError):
        host._execute_layer_forward_stage(gguf_request(value), threading.Event())
    host.commit_worker_runtime(value)
    req = gguf_request(value)
    req.stage_fields.pop("prepare_identity")
    with pytest.raises(WorkerPrepareError):
        host._execute_layer_forward_stage(req, threading.Event())


def test_tail_prepare_checks_optional_native_token_entry_before_ready(gguf, monkeypatch):
    host, value, selected = gguf
    value["layer_range"] = [2, 4]
    value["segment_mode"] = "tail"
    selected["manifest"].update(layer_range=[2, 4], segment_mode="tail")
    monkeypatch.setattr(Upstream, "forward_hidden_to_token", lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("missing tail symbol")))
    with pytest.raises(RuntimeError, match="tail symbol"):
        host.prepare_worker_runtime(command("prepare", value), selected, threading.Event())
    assert not Upstream.made[0].is_available and not host.worker_runtime_present()


def test_retirement_reaps_real_child_even_without_close_response():
    from llama_keep_head import KeepHeadUpstream
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys,time; sys.stdin.readline(); time.sleep(60)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    upstream = KeepHeadUpstream.__new__(KeepHeadUpstream)
    upstream._worker = process
    try:
        close_layer_runtime(LayerRuntime(None, upstream=upstream))
        assert process.poll() is not None and upstream._worker is None
        assert process.stdout.closed and process.stdin.closed
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def test_engine_close_releases_legacy_gguf_as_well_as_supplied_manager(gguf):
    host, value, selected = gguf
    upstream = Upstream("shim", "legacy.gguf")
    host._layer_upstream = upstream
    host._host = SimpleNamespace(full_chat_execution_lock=threading.RLock())
    host._task_graph_coordinator = None
    host._model_change_lock = threading.RLock()
    host.close()
    assert not upstream.is_available and host._layer_upstream is None


def test_scheduler_rejects_no_identity_before_accept_or_execution():
    class Worker(SchedulerTaskWorkerMixin):
        def get_distributed_inference_enabled(self):
            return True
        def get_effective_node_id(self):
            return "pc"
        def _scheduler_facade_global(self, name):
            return True
        def _send_task_worker_stage_accept(self, message_id, offer, **kw):
            self.accepts.append(kw)
    worker = Worker()
    worker._task_worker_stage_lock = threading.RLock()
    worker._task_worker_active_attempts = {}
    worker._task_worker_control = SimpleNamespace(coordinator_snapshot=lambda: {})
    worker.accepts = []
    worker._worker_prepare = WorkerPrepareController(worker)
    worker._handle_task_worker_stage_offer(SimpleNamespace(message_id="offer", payload={
        "attempt_id": "attempt", "provider_id": remote_provider_id("pc"), "stage_type": "layer_forward",
    }))
    assert worker.accepts == [{"accepted": False, "reason_code": "invalid_prepare_identity", "retryable": False}]
    assert worker._task_worker_active_attempts == {}


def test_scheduler_outer_dispatch_enqueues_controller_only_for_master_client():
    from scheduler import Scheduler
    calls = []
    client = object()
    worker = SimpleNamespace(_effective_role=lambda: "client", _tcp_client=client,
                             _worker_prepare=SimpleNamespace(handle=lambda *args: calls.append(args)))
    payload = command("prepare", identity())
    Scheduler._on_tcp_message(worker, "master", {"type": "model_prepare", "data": payload})
    assert calls == [(payload, client)]
    Scheduler._on_tcp_message(worker, "other", {"type": "model_prepare", "data": payload})
    assert len(calls) == 1


def test_public_executor_cannot_use_unprepared_native_lazy_path(gguf, monkeypatch):
    host, value, selected = gguf
    req = gguf_request(value)
    req.stage_type = "layer_forward"
    req.stage_fields.pop("prepare_identity")
    monkeypatch.setattr(host, "_layer_forward_upstream", lambda: pytest.fail("public executor reached lazy load"))
    with pytest.raises(WorkerPrepareError, match="invalid_prepare_identity"):
        host.execute_task_worker_stage(req, threading.Event())


def test_real_controller_and_public_stage_share_exact_prepared_runtime(tiny_supply, monkeypatch):
    from tests.test_worker_prepare import Client, run
    import model_supply_worker
    manifest = tiny_supply.activate(0)
    value = supply_identity(manifest, 1)
    host = tiny_supply.host()
    scheduler = SimpleNamespace(
        _host=SimpleNamespace(full_chat_execution_lock=threading.RLock()),
        _tcp_client=Client(), _control_fence=None, _effective_role=lambda: "client",
        get_effective_node_id=lambda: "master", get_distributed_inference_enabled=lambda: True,
        _model_supply_worker=tiny_supply.scheduler._model_supply_worker,
        _configured_layer_artifact=lambda: value["stage_model_identity"],
        _task_worker_layer_artifact_identity=lambda artifact: artifact,
    )
    controller = WorkerPrepareController(scheduler)
    controller.configure_runtime(lambda: host)
    env = SimpleNamespace(controller=controller, scheduler=scheduler)
    try:
        assert run(env, "prepare", value)["status"] == "prepared"
        manager = host._worker_runtime_candidate.manager
        assert run(env, "commit", value)["status"] == "ready"
        assert host._supplied_layer_manager is manager
        monkeypatch.setattr(host, "_build_worker_runtime", lambda *args: pytest.fail("committed stage reloaded"))
        monkeypatch.setattr(model_supply_worker, "get_selected_model", lambda *args: pytest.fail("committed stage reselected"))
        req = request(manifest, value, np.ones((2, 16), dtype=np.float32))
        controller.check_stage({"prepare_identity": stage_identity(value), "layer_range": value["layer_range"],
                                "model_identity": value["stage_model_identity"]})
        assert host.execute_task_worker_stage(req, threading.Event())["hidden_out_f32"]
        assert not run(env, "release", value, cleanup_id="retire-stage")["loaded"]
        assert manager.model is None
    finally:
        controller.close()


def test_failed_native_build_close_retains_owner_until_retry(gguf, monkeypatch):
    host, value, selected = gguf
    original_close = Upstream.close
    monkeypatch.setattr(Upstream, "forward_hidden_to_hidden", lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("native probe failed")))
    monkeypatch.setattr(Upstream, "close", lambda *args: (_ for _ in ()).throw(RuntimeError("native close failed")))
    with pytest.raises(RuntimeError, match="close failed"):
        host.prepare_worker_runtime(command("prepare", value), selected, threading.Event())
    assert host.worker_runtime_present()
    assert value in host.worker_runtime_owned_identities()
    monkeypatch.setattr(Upstream, "close", original_close)
    host.release_worker_runtime(value)
    assert not host.worker_runtime_present()


def test_gguf_layer_prepare_ignores_cuda_server_auto_resolver(gguf, monkeypatch):
    import inference_service.engine_host as engine
    host, value, selected = gguf
    monkeypatch.setattr(engine, "resolve_model_load", lambda *args, **kw: pytest.fail("layer prepare entered whole-model auto/server resolver"))
    monkeypatch.setattr(engine, "effective_pytorch_cuda_available", lambda *args, **kw: True)
    host.prepare_worker_runtime(command("prepare", value), selected, threading.Event())
    assert host._worker_runtime_candidate.upstream is Upstream.made[0]
    assert Upstream.made[0].model_path == selected["model_path"]


@pytest.fixture
def scoped_api_runtime(monkeypatch):
    import api_server
    from qlh_core.adapters.settings import SettingsStore
    from qlh_core.bootstrap.runtime import AppRuntime

    runtime = AppRuntime(SettingsStore().load(), host=api_server.model_host,
                         resources_factory=api_server._build_default_resources)
    monkeypatch.setattr(api_server, "_app_runtime", runtime)
    monkeypatch.setattr(api_server, "_LAYER_STAGE_HOST", None)
    try:
        yield runtime
    finally:
        runtime.close()


def test_real_api_factory_and_layer_stage_share_single_runtime_through_retry_and_cleanup(tiny_supply, monkeypatch, scoped_api_runtime):
    import api_server
    import inference_service.engine_host as engine_module
    import model_module
    import model_supply_worker
    from task_graph import TaskGraphError
    from tests.test_worker_prepare import Client, run

    # Assert the real composition-root registration before isolating native
    # ownership. Use its registered factory, not a hand-written host lambda.
    registered_factory = api_server.scheduler._worker_prepare._runtime_factory
    assert registered_factory is api_server._get_layer_stage_host
    monkeypatch.setattr(api_server, "_LAYER_STAGE_HOST", None)
    created, loaded_keys = [], []
    original_materialize = model_module._LayerRangeLoadTracker.materialize

    def construct(**kwargs):
        host = EngineHost(**kwargs)
        created.append(host)
        return host

    def materialize(tracker, handle, key, **kwargs):
        loaded_keys.append(key)
        assert kwargs.get("source_file"), "real loader must record the source shard"
        return original_materialize(tracker, handle, key, **kwargs)

    monkeypatch.setattr(engine_module, "EngineHost", construct)
    monkeypatch.setattr(model_module._LayerRangeLoadTracker, "materialize", materialize)
    manifest = tiny_supply.activate(0)
    value = supply_identity(manifest, 1)
    scheduler = SimpleNamespace(
        _host=SimpleNamespace(full_chat_execution_lock=threading.RLock()),
        _tcp_client=Client(), _control_fence=None, _effective_role=lambda: "client",
        get_effective_node_id=lambda: "master", get_distributed_inference_enabled=lambda: True,
        _model_supply_worker=tiny_supply.scheduler._model_supply_worker,
        _configured_layer_artifact=lambda: current_identity["stage_model_identity"],
        _task_worker_layer_artifact_identity=lambda artifact: artifact,
    )
    current_identity = value
    controller = WorkerPrepareController(scheduler, runtime_factory=registered_factory)
    env = SimpleNamespace(controller=controller, scheduler=scheduler)
    try:
        assert created == [] and api_server._LAYER_STAGE_HOST is None
        assert run(env, "prepare", value)["status"] == "prepared"
        host = controller._runtime
        assert created == [host]
        assert api_server._get_layer_stage_host() is host is api_server._LAYER_STAGE_HOST
        manager = host._worker_runtime_candidate.manager
        assert manager._layer_load_metrics["mode"] == "safetensors_key_filtered"
        req = request(manifest, value, np.ones((2, 16), dtype=np.float32))
        with pytest.raises(WorkerPrepareError):
            api_server._execute_task_worker_stage(req, threading.Event())
        assert run(env, "commit", value)["status"] == "ready"
        assert run(env, "finalize", value)["loaded"]
        loaded_count = len(loaded_keys)

        def no_reload(*args, **kwargs):
            pytest.fail("API layer stage recreated or lazily loaded its prepared runtime")

        with monkeypatch.context() as stage_checks:
            stage_checks.setattr(host, "_build_worker_runtime", no_reload)
            stage_checks.setattr(model_module.ModelManager, "load_layer_range", no_reload)
            stage_checks.setattr(model_supply_worker, "get_selected_model", no_reload)
            first = api_server._execute_task_worker_stage(req, threading.Event())
            assert api_server._execute_task_worker_stage(req, threading.Event()) == first
            cancelled = threading.Event()
            cancelled.set()
            with pytest.raises(TaskGraphError):
                api_server._execute_task_worker_stage(req, cancelled)
            # Cancellation during real forward must clear partial KV and keep
            # the committed manager so a prefill retry uses the same runtime.
            original_forward = manager.forward_layers
            def cancel_after_forward(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                cancelled.set()
                return result
            stage_checks.setattr(manager, "forward_layers", cancel_after_forward)
            cancelled.clear()
            with pytest.raises(TaskGraphError, match="supplied_stage_cancelled"):
                api_server._execute_task_worker_stage(req, cancelled)
            assert host._supplied_layer_caches == {}
            stage_checks.setattr(manager, "forward_layers", original_forward)
            assert api_server._execute_task_worker_stage(req, threading.Event()) == first
            assert host._supplied_layer_manager is manager and len(loaded_keys) == loaded_count

        # Finalize must retire previous resources on this same API-owned host.
        new_manifest = tiny_supply.activate(2)
        current_identity = new_value = supply_identity(new_manifest, 2)
        assert run(env, "prepare", new_value)["status"] == "prepared"
        assert run(env, "commit", new_value)["status"] == "ready"
        current = host._supplied_layer_manager
        assert current is not manager and host._worker_runtime_previous.manager is manager
        assert run(env, "finalize", new_value)["loaded"]
        assert manager.model is None and current.model is not None
        loaded_count = len(loaded_keys)
        with monkeypatch.context() as cleanup_checks:
            cleanup_checks.setattr(host, "_build_worker_runtime", no_reload)
            cleanup_checks.setattr(model_module.ModelManager, "load_layer_range", no_reload)
            cleanup_checks.setattr(model_supply_worker, "get_selected_model", no_reload)
            new_req = request(new_manifest, new_value, np.ones((2, 16), dtype=np.float32))
            assert api_server._execute_task_worker_stage(new_req, threading.Event())["hidden_out_f32"]
            assert not run(env, "release", new_value, cleanup_id="api-retire")["loaded"]
            assert current.model is None and not host.worker_runtime_present()
            ack = run(env, "finalize", new_value, cleanup_id="api-finalize-replay")
            assert ack["status"] == "finalized" and not ack["loaded"]
            with pytest.raises(WorkerPrepareError):
                api_server._execute_task_worker_stage(new_req, threading.Event())
            assert api_server._get_layer_stage_host() is host
            assert created == [host] and len(loaded_keys) == loaded_count
    finally:
        controller.close()


@pytest.fixture
def api_retirement_runtime(tiny_supply, monkeypatch, scoped_api_runtime):
    import api_server
    from tests.test_worker_prepare import Client, run

    registered_factory = api_server.scheduler._worker_prepare._runtime_factory
    assert registered_factory is api_server._get_layer_stage_host
    monkeypatch.setattr(api_server, "_LAYER_STAGE_HOST", None)
    manifest = tiny_supply.activate(0)
    value = supply_identity(manifest, 1)
    full_host = SimpleNamespace(full_chat_execution_lock=threading.RLock(), model_loaded=False, current_quant=None)
    scheduler = SimpleNamespace(
        _host=full_host, _tcp_client=Client(), _control_fence=None,
        _effective_role=lambda: "client", get_effective_node_id=lambda: "master",
        _model_supply_worker=tiny_supply.scheduler._model_supply_worker,
        _configured_layer_artifact=lambda: value["stage_model_identity"],
        _task_worker_layer_artifact_identity=lambda artifact: artifact,
        _distributed_inference_enabled=True, _task_worker_stage_lock=threading.RLock(),
        _task_worker_refresh_lock=threading.RLock(),
        _task_worker_active_attempts={}, _inference_lock=threading.RLock(),
        _layer_execution_lock=threading.RLock(), _layer_config_lock=threading.RLock(),
        _worker_assignments={}, _layer_config_retry_state={}, _local_pipeline_steps={},
        refresh_task_worker_capabilities=lambda: None, release_pipeline_worker_for_local_model=lambda: True,
    )
    scheduler.get_distributed_inference_enabled = lambda: scheduler._distributed_inference_enabled
    controller = WorkerPrepareController(scheduler, runtime_factory=registered_factory)
    scheduler._worker_prepare = controller
    enabled = []
    def set_enabled(value):
        if not value:
            SchedulerTaskWorkerMixin._cleanup_task_worker_prepare(scheduler, "local_inference_mode")
        scheduler._distributed_inference_enabled = bool(value)
        enabled.append(bool(value))
        return {"enabled": bool(value)}
    scheduler.set_distributed_inference_enabled = set_enabled
    monkeypatch.setattr(api_server, "scheduler", scheduler)
    monkeypatch.setattr(api_server, "model_host", full_host)
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    monkeypatch.setattr(api_server, "_local_llm_is_loaded", lambda: False)
    monkeypatch.setattr(api_server, "_reset_runtime_conversation_state", lambda **kw: None)
    monkeypatch.setattr(api_server, "_refresh_pipeline_layer_config", lambda: None)
    env = SimpleNamespace(controller=controller, scheduler=scheduler, api=api_server, enabled=enabled,
                          value=value, manifest=manifest)
    try:
        run(env, "prepare", value)
        run(env, "commit", value)
        run(env, "finalize", value)
        env.host = api_server._get_layer_stage_host()
        env.manager = env.host._supplied_layer_manager
        assert env.host is controller._runtime and env.host._host is not full_host
        yield env
    finally:
        controller.close()


def async_call(call):
    outcome = {}
    def invoke():
        try:
            outcome["result"] = call()
        except Exception as exc:
            outcome["error"] = exc
    thread = threading.Thread(target=invoke)
    thread.start()
    return thread, outcome


def block_manager_close(env, monkeypatch):
    entered, complete = threading.Event(), threading.Event()
    original = env.manager.unload_model
    def block():
        entered.set()
        assert complete.wait(5)
        original()
    monkeypatch.setattr(env.manager, "unload_model", block)
    return entered, complete


def test_api_full_model_change_waits_for_real_layer_unload_before_loading(api_retirement_runtime, monkeypatch):
    env = api_retirement_runtime
    entered, complete = block_manager_close(env, monkeypatch)
    loaded = []
    def change():
        assert env.manager.model is None
        assert not env.host.worker_runtime_present()
        loaded.append("full-model")
        return {"success": True}
    thread, outcome = async_call(lambda: env.api._run_exclusive_model_change(change, release_worker_reservation=True))
    try:
        assert entered.wait(3)
        assert outcome == {} and loaded == [] and thread.is_alive()
        assert env.manager.model is not None
        assert not env.controller.handle(command("prepare", {**env.value, "generation": 2}))
    finally:
        complete.set()
        thread.join(timeout=5)
    assert outcome == {"result": {"success": True}} and loaded == ["full-model"]
    assert env.api._get_layer_stage_host() is env.host


def test_api_full_model_change_close_failure_returns_409_before_loading(api_retirement_runtime, monkeypatch):
    env = api_retirement_runtime
    original = env.manager.unload_model
    monkeypatch.setattr(env.manager, "unload_model", lambda: (_ for _ in ()).throw(RuntimeError("physical unload failed")))
    loaded = []
    with pytest.raises(env.api.HTTPException) as failed:
        env.api._run_exclusive_model_change(lambda: loaded.append("full-model"), release_worker_reservation=True)
    assert failed.value.status_code == 409
    assert failed.value.error_code == "worker_prepare_retire_failed"
    assert loaded == [] and env.host.worker_runtime_present()
    with pytest.raises(WorkerPrepareError):
        env.controller.check_stage({"prepare_identity": stage_identity(env.value), "layer_range": env.value["layer_range"],
                                    "model_identity": env.value["stage_model_identity"]})
    monkeypatch.setattr(env.manager, "unload_model", original)
    assert env.controller.retire_and_join("local_model_load", timeout=2)


def test_api_mode_change_returns_only_after_layer_unload_outside_full_chat_lock(api_retirement_runtime, monkeypatch):
    env = api_retirement_runtime
    entered, complete = block_manager_close(env, monkeypatch)
    thread, outcome = async_call(lambda: env.api._set_node_inference_mode("local"))
    try:
        assert entered.wait(3)
        assert env.scheduler._distributed_inference_enabled is False
        assert outcome == {} and thread.is_alive()
        assert env.manager.model is not None
        assert env.controller._stage_fenced
    finally:
        complete.set()
        thread.join(timeout=5)
    assert "error" not in outcome
    assert outcome["result"]["mode"] == "local" and outcome["result"]["changed"] is True
    assert env.manager.model is None and not env.host.worker_runtime_present()


def test_api_mode_change_cleanup_failure_never_reenables_distributed(api_retirement_runtime, monkeypatch):
    env = api_retirement_runtime
    env.scheduler._distributed_inference_enabled = False
    original = env.manager.unload_model
    monkeypatch.setattr(env.manager, "unload_model", lambda: (_ for _ in ()).throw(RuntimeError("physical unload failed")))
    try:
        with pytest.raises(env.api.HTTPException) as failed:
            env.api._set_node_inference_mode("distributed")
        assert failed.value.status_code == 409
        assert failed.value.error_code == "worker_prepare_retire_failed"
        assert env.scheduler._distributed_inference_enabled is False
        assert True not in env.enabled
        assert env.host.worker_runtime_present()
    finally:
        monkeypatch.setattr(env.manager, "unload_model", original)
        env.controller.retire_and_join("cleanup-retry", timeout=2)


def test_api_full_model_change_retirement_timeout_never_loads_over_live_segment(api_retirement_runtime, monkeypatch):
    env = api_retirement_runtime
    entered, complete = block_manager_close(env, monkeypatch)
    original_retire = env.controller.retire_and_join
    requested = []
    def short_deadline(reason_code, timeout=120):
        requested.append((reason_code, timeout))
        return original_retire(reason_code, timeout=0.05)
    monkeypatch.setattr(env.controller, "retire_and_join", short_deadline)
    loaded = []
    try:
        with pytest.raises(env.api.HTTPException) as failed:
            env.api._run_exclusive_model_change(lambda: loaded.append("full-model"), release_worker_reservation=True)
        assert entered.is_set()
        assert failed.value.status_code == 409
        assert failed.value.error_code == "worker_prepare_retire_timeout"
        assert requested == [("local_model_load", 120)]
        assert loaded == [] and env.manager.model is not None
    finally:
        complete.set()
        from tests.test_worker_prepare import drain
        drain(env.controller)
    assert env.manager.model is None


def test_retire_and_join_reaps_real_layer_child_before_confirming_unloaded(monkeypatch):
    from llama_keep_head import KeepHeadUpstream
    from tests.test_worker_prepare import Client
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys,time; sys.stdin.readline(); time.sleep(60)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    upstream = KeepHeadUpstream.__new__(KeepHeadUpstream)
    upstream._worker = process
    host = EngineHost.__new__(EngineHost)
    host._layer_upstream = upstream
    scheduler = SimpleNamespace(_host=SimpleNamespace(full_chat_execution_lock=threading.RLock()),
                                _tcp_client=Client(), _control_fence=None)
    controller = WorkerPrepareController(scheduler, runtime_factory=lambda: host)
    try:
        assert host.worker_runtime_present()
        assert controller.retire_and_join("local_model_load", timeout=8) is True
        assert process.poll() is not None and upstream._worker is None
        assert process.stdin.closed and process.stdout.closed
        assert not host.worker_runtime_present()
    finally:
        controller.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def test_failed_initializer_is_retained_until_real_runtime_cleanup(gguf, monkeypatch):
    from llama_keep_head import KeepHeadCleanupError
    host, value, selected = gguf
    upstream = Upstream("shim", selected["model_path"])
    upstream.is_available = False
    fail = [True]

    def close():
        if fail[0]:
            raise OSError("failed initializer still owns weights")
        upstream.closed = True

    upstream.close = close
    def initialize(*args, **kwargs):
        raise KeepHeadCleanupError(upstream, RuntimeError("init failed"), OSError("close failed"))

    monkeypatch.setattr("llama_keep_head.KeepHeadUpstream", initialize)
    with pytest.raises(OSError, match="failed initializer still owns weights"):
        host.prepare_worker_runtime(command("prepare", value), selected, threading.Event())
    assert host.worker_runtime_present()
    assert host._worker_runtime_orphans[0].upstream is upstream
    assert host._worker_runtime_phase == "failed"
    fail[0] = False
    host.release_worker_runtime()
    assert upstream.closed and not host.worker_runtime_present()
