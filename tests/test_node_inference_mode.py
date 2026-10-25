import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api_server
from model_host import ModelHost
from model_load_resolver import ModelLoadResolution
from scheduler import Scheduler


@pytest.fixture
def node(monkeypatch):
    manager = SimpleNamespace(is_loaded=False, is_pipeline_prepared=False)
    host = ModelHost(manager)
    scheduler = Scheduler(host=host)
    monkeypatch.setattr(api_server, "model_host", host)
    monkeypatch.setattr(api_server, "model_manager", host)
    monkeypatch.setattr(api_server, "scheduler", scheduler)
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    monkeypatch.setattr(scheduler, "_distributed_inference_enabled", True)
    monkeypatch.setattr(api_server, "_reset_runtime_conversation_state", lambda **_kw: None)
    return scheduler, host, manager


@pytest.mark.parametrize("engine", ["llama_cpp", "pytorch"])
def test_distributed_load_never_uses_full_model_capacity(node, monkeypatch, engine):
    monkeypatch.setattr(api_server, "_estimate_full_model_load_bytes", lambda *_args: pytest.fail("full model estimate must not authorize loading"))
    assert api_server._decide_pipeline_load_mode("large", engine, "int4")[0] == "pipeline"


@pytest.mark.parametrize("handler,request_type", [
    ("load_model", "LoadModelRequest"),
    ("switch_model", "SwitchModelRequest"),
])
def test_default_model_actions_prepare_segments(node, monkeypatch, handler, request_type):
    from api import routes_models

    resolution = ModelLoadResolution(
        model_id="large", requested_engine="llama_cpp", engine="llama_cpp",
        artifact_kind="gguf", model_path="large.gguf", requested_quant="gguf",
        quant_type="gguf", runtime="llama_cpp_native", runtime_quant="gguf",
        layer_range_mode="precut_artifact", reason_code="TEST", reason="fixture",
    )
    monkeypatch.setattr(api_server, "_resolve_model_load_plan", lambda *_args, **_kw: resolution)
    calls = []

    async def prepare(selected):
        calls.append(selected)
        return {"loaded": False, "pipeline_prepared": True}

    monkeypatch.setattr(routes_models, "_prepare_pipeline_model_resolution", prepare)
    result = asyncio.run(getattr(api_server, handler)(getattr(api_server, request_type)(model_id="large")))
    assert result == {"loaded": False, "pipeline_prepared": True}
    assert calls == [resolution]
    assert node[1].model_loaded is False


def test_local_mode_releases_pipeline_before_full_load(node, monkeypatch):
    scheduler, host, manager = node
    manager.is_loaded = True
    manager.is_pipeline_prepared = True
    scheduler._pipeline_worker_reserved = True
    unloaded = []

    def unload():
        unloaded.append(scheduler.get_distributed_inference_enabled())
        manager.is_loaded = False
        manager.is_pipeline_prepared = False

    manager.unload_model = unload
    response = TestClient(api_server.app).put("/api/models/inference-mode", json={"mode": "local"})
    assert response.status_code == 200
    assert response.json()["mode"] == "local"
    assert unloaded == [False]
    assert not scheduler.has_pipeline_worker_reservation()
    assert scheduler._pipeline_worker_opted_out
    assert not host.model_loaded
    assert api_server._decide_pipeline_load_mode("large", "llama_cpp", "gguf")[0] == "full"
    assert TestClient(api_server.app).get("/api/models/current").json()["inference_mode"] == "local"


def test_rejoin_unloads_full_model_before_enabling_worker(node, monkeypatch):
    scheduler, host, manager = node
    scheduler._distributed_inference_enabled = False
    manager.is_loaded = True
    host.model_loaded = True
    calls = []

    def unload():
        calls.append(("unload", scheduler.get_distributed_inference_enabled()))
        manager.is_loaded = False

    manager.unload_model = unload
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "client")
    monkeypatch.setattr(scheduler, "_request_pipeline_worker_opt_in", lambda: calls.append(("opt_in", manager.is_loaded)))
    result = api_server._set_node_inference_mode("distributed")
    assert result["mode"] == "distributed"
    assert calls == [("unload", False), ("opt_in", False)]
    assert not host.model_loaded
    assert scheduler.get_distributed_inference_enabled()


def test_reselecting_local_mode_keeps_baseline_loaded(node):
    scheduler, host, manager = node
    scheduler._distributed_inference_enabled = False
    host.model_loaded = True
    manager.is_loaded = True
    manager.unload_model = lambda: pytest.fail("idempotent mode change unloaded baseline")
    assert api_server._set_node_inference_mode("local")["changed"] is False
    assert host.model_loaded


def test_active_worker_prevents_mode_change(node):
    scheduler, host, _manager = node
    scheduler._task_worker_active_attempts["active"] = object()
    response = TestClient(api_server.app).put("/api/models/inference-mode", json={"mode": "local"})
    assert response.status_code == 409
    assert "INFERENCE_MODE_BUSY" in response.text
    assert scheduler.get_distributed_inference_enabled()


def test_authoritative_push_cannot_override_local_mode(node, monkeypatch):
    scheduler, _host, _manager = node
    scheduler._distributed_inference_enabled = False
    scheduler._pipeline_worker_opted_out = True
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "client")
    scheduler._schedule_layer_config("master", {
        "config_id": "stale", "authoritative_sync": True,
        "generation": 1, "start_layer": 0, "end_layer": 4,
    })
    assert scheduler._pipeline_worker_opted_out
    assert not scheduler._layer_config_inflight
    assert scheduler._active_layer_config is None
    assert scheduler.refresh_task_worker_capabilities() is False
    assert scheduler._send_task_worker_hello() is False


def test_local_mode_refuses_pipeline_prepare(node):
    node[0]._distributed_inference_enabled = False
    response = TestClient(api_server.app).post("/api/models/prepare-pipeline", json={"model_id": "large"})
    assert response.status_code == 409
    assert "LOCAL_INFERENCE_MODE" in response.text


def test_distributed_cold_chat_does_not_auto_load_full_model(node, monkeypatch):
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: pytest.fail("unexpected implicit model discovery"))
    with pytest.raises(api_server.HTTPException) as failure:
        api_server._auto_load_default_model()
    assert failure.value.status_code == 409
    assert failure.value.error_code == "DISTRIBUTED_MODEL_SELECTION_REQUIRED"


def test_worker_disconnects_and_stays_out_until_explicit_rejoin(node, monkeypatch):
    scheduler, _host, _manager = node
    calls = []
    scheduler._tcp_client = SimpleNamespace(disconnect=lambda: calls.append("disconnect"), _running=False)
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "client")
    scheduler.set_distributed_inference_enabled(False)
    assert calls == ["disconnect"]
    assert scheduler._client_reconnect_enabled is False
    assert scheduler.connect_to_master("127.0.0.1", 8888)["reason"] == "local_inference_mode"


@pytest.mark.parametrize("stream", [False, True])
def test_unprepared_segment_cannot_reload_full_model_in_distributed_mode(node, stream):
    scheduler, _host, manager = node
    manager.is_loaded = True
    manager.layer_range = (0, 4)
    manager.ensure_full_model = lambda: pytest.fail("distributed fallback loaded a full model")
    if stream:
        result = next(scheduler._run_full_model_inference_stream("hi"))
    else:
        result = scheduler._run_full_model_inference("hi")
    assert result["reason_code"] == "distributed_full_model_forbidden"


def test_direct_authoritative_push_rejects_local_master(node):
    scheduler, _host, _manager = node
    scheduler._distributed_inference_enabled = False
    result = scheduler._push_layer_config_to_clients_locked(candidate={"model_id": "large"})
    assert result["reason_code"] == "local_inference_mode"
    assert not result["admitted"]


@pytest.mark.parametrize("handler,request_type", [
    ("load_model", "LoadModelRequest"),
    ("switch_model", "SwitchModelRequest"),
])
def test_full_load_waiting_for_lock_cannot_cross_rejoin(node, monkeypatch, handler, request_type):
    scheduler, _host, manager = node
    scheduler._distributed_inference_enabled = False
    resolution = ModelLoadResolution(
        model_id="large", requested_engine="llama_cpp", engine="llama_cpp",
        artifact_kind="gguf", model_path="large.gguf", requested_quant="gguf",
        quant_type="gguf", runtime="llama_cpp_native", runtime_quant="gguf",
        layer_range_mode="precut_artifact", reason_code="TEST", reason="fixture",
    )
    monkeypatch.setattr(api_server, "_resolve_model_load_plan", lambda *_a, **_k: resolution)
    manager.switch_model = lambda **_kw: pytest.fail("queued full load crossed explicit rejoin")
    original = api_server._run_exclusive_model_change

    def rejoin_before_lock(*args, **kwargs):
        scheduler._distributed_inference_enabled = True
        return original(*args, **kwargs)

    monkeypatch.setattr(api_server, "_run_exclusive_model_change", rejoin_before_lock)
    with pytest.raises(api_server.HTTPException) as failure:
        asyncio.run(getattr(api_server, handler)(getattr(api_server, request_type)(model_id="large")))
    assert failure.value.error_code == "INFERENCE_MODE_CHANGED"


def test_legacy_control_setter_cannot_rejoin_with_local_model(node):
    scheduler, host, manager = node
    scheduler._distributed_inference_enabled = False
    manager.is_loaded = True
    host.model_loaded = True
    result = scheduler.set_distributed_inference_enabled(True)
    assert result["reason_code"] == "local_model_still_loaded"
    assert not scheduler.get_distributed_inference_enabled()


def test_worker_offer_cannot_register_after_local_mode_wins_admission_lock(node, monkeypatch):
    scheduler, _host, _manager = node
    responses = []

    class SwitchBeforeAdmission:
        def __enter__(self):
            scheduler._distributed_inference_enabled = False
        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(scheduler, "_task_worker_stage_lock", SwitchBeforeAdmission())
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "test-worker")
    monkeypatch.setattr(scheduler, "_runtime_setting", lambda _name: True)
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", lambda: {"stage_types": ["inference"]})
    monkeypatch.setattr(scheduler, "_task_worker_offer_model_identity_matches", lambda *_args: True)
    monkeypatch.setattr(scheduler._task_worker_control, "coordinator_snapshot", lambda: {"manual_stage_dispatch_enabled": True})
    monkeypatch.setattr(scheduler, "_send_task_worker_stage_accept", lambda *_a, **kw: responses.append(kw))
    from task_worker_adapter import remote_provider_id
    scheduler._handle_task_worker_stage_offer(SimpleNamespace(
        message_id="offer", sent_at_ms=1,
        payload={"attempt_id": "late", "provider_id": remote_provider_id("test-worker"),
                 "stage_type": "inference", "model_identity": {}},
    ))
    assert not scheduler._task_worker_active_attempts
    assert responses[0]["reason_code"] == "local_inference_mode"


@pytest.mark.parametrize("enabled", [None, False, True])
def test_thinking_choice_reaches_native_chat_template(enabled):
    seen = []

    class NativeTemplate:
        def apply_chat_template(self, messages, **options):
            seen.append(options)
            return "assistant<think></think>" if options.get("enable_thinking") is False else "assistant<think>"

    result = api_server._build_model_chat_prompt(
        NativeTemplate(), [{"role": "user", "content": "1+1?"}],
        enable_thinking=enabled,
    )
    if enabled is None:
        assert "enable_thinking" not in seen[0]
    else:
        assert seen[0]["enable_thinking"] is enabled
    assert ("</think>" in result) is (enabled is False)
