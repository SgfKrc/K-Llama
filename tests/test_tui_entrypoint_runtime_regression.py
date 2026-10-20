"""Normal TUI startup and first local load, without native models or listeners."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import qlh
from model_host import ModelHost
from qlh_core.bootstrap.runtime import AppRuntime


class NativeManager:
    def __init__(self):
        self.active_model_id = None
        self.quant_type = "fp16"
        self.model_path = ""
        self._engine_type = "pytorch"
        self.is_loaded = False

    def switch_model(self, **values):
        self.active_model_id = values["model_id"]
        self.model_path = values["model_path"]
        self.quant_type = values["quant_type"]
        self.is_loaded = True
        return {"success": True, "model_name": self.active_model_id}

    def get_model_info(self):
        return {"model_id": self.active_model_id, "model_name": self.active_model_id,
                "engine": "pytorch", "model_path": self.model_path}

    def get_memory_usage(self):
        return {}

    def get_pipeline_descriptor(self):
        return {}

    def unload_model(self):
        self.active_model_id = None
        self.is_loaded = False


def test_metadata_observes_created_lazy_owner_without_warming_cold_owner(monkeypatch):
    made = []
    def create():
        made.append(NativeManager())
        return made[-1]
    monkeypatch.setitem(sys.modules, "model_module", SimpleNamespace(ModelManager=create))
    host = ModelHost()
    assert host.peek_manager() is None
    assert getattr(host, "active_model_id", None) is None
    assert made == []
    for model_id in ("first", "second"):
        host.switch_model(model_id=model_id, engine="pytorch", quant_type="fp16", model_path=model_id)
        assert host.active_model_id == model_id
        assert host.model_path == model_id
        assert host.quant_type == "fp16"
    assert len(made) == 1
    host.active_model_id = "published"
    assert host.peek_manager().active_model_id == "published"
    assert host.active_model_id == "published"
    host.unload_model()
    assert host.active_model_id is None
    assert made == [host.peek_manager()]


@pytest.mark.parametrize("model_id", ["qw3.5-2b", "another-local-model"])
def test_first_local_load_keeps_api_and_model_catalog_available(monkeypatch, model_id):
    import api_server
    from scheduler import Scheduler
    from model_load_resolver import ModelLoadResolution

    monkeypatch.setitem(sys.modules, "model_module", SimpleNamespace(ModelManager=NativeManager))
    host = ModelHost()
    scheduler = Scheduler(host=host)
    scheduler._distributed_inference_enabled = False
    monkeypatch.setattr(api_server, "model_host", host)
    monkeypatch.setattr(api_server, "model_manager", host)
    monkeypatch.setattr(api_server, "scheduler", scheduler)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)
    monkeypatch.setattr(api_server, "_refresh_pipeline_layer_config", lambda: None)
    monkeypatch.setattr(scheduler, "refresh_task_worker_capabilities", lambda: None)
    monkeypatch.setattr(api_server, "_get_all_model_configs", lambda: [])
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [])
    monkeypatch.setattr(api_server, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(api_server, "_resolve_model_load_plan", lambda *args, **kwargs: ModelLoadResolution(
        model_id=model_id, requested_engine="pytorch", engine="pytorch", artifact_kind="safetensors",
        model_path="fixture/model", requested_quant="fp16", quant_type="fp16", runtime="pytorch_cpu",
        runtime_quant="fp16", layer_range_mode="none", reason_code="TEST", reason="test"))
    client = TestClient(api_server.app, raise_server_exceptions=False)
    assert host.peek_manager() is None
    loaded = client.post("/api/models/load", json={"model_id": model_id, "engine": "pytorch", "quant_type": "fp16"})
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["active_model_id"] == model_id
    for endpoint in ("/api/status", "/api/models", "/api/models/current"):
        response = client.get(endpoint)
        assert response.status_code == 200, (endpoint, response.text)
        assert response.json().get("active_model_id", response.json().get("model_id")) == model_id
    assert client.get("/api/models/current").json()["runnable"] is True
    unloaded = client.post("/api/models/unload")
    assert unloaded.status_code == 200
    assert client.get("/api/models/current").json()["runnable"] is False


@pytest.mark.parametrize("role_source, role", [
    ("none", "master"), ("unconfirmed-master", "master"),
    ("environment", "client"), ("confirmed", "client"),
    ("bootstrap", "client"), ("auto", "auto"), ("dotenv", "client"),
])
def test_interactive_entrypoint_uses_product_default_and_keeps_explicit_roles(monkeypatch, tmp_path, role_source, role):
    import node_config
    from qlh_core.adapters.settings import SettingsStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(node_config, "get_app_root", lambda: tmp_path)
    monkeypatch.delenv("QLH_NODE_ROLE", raising=False)
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "0")
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node.json"))
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(tmp_path / "secrets.json"))
    monkeypatch.setenv("QLH_STATE_DIR", str(tmp_path / "state"))
    dot = tmp_path / ".env"
    dot.write_text("QLH_NODE_ROLE=client\n" if role_source == "dotenv" else "", encoding="utf-8")
    if role_source in {"environment", "auto"}:
        monkeypatch.setenv("QLH_NODE_ROLE", role)
    elif role_source in {"confirmed", "bootstrap", "unconfirmed-master"}:
        (tmp_path / "node.json").write_text(json.dumps({
            "bootstrapped": role_source == "bootstrap",
            "node": {"role": "master" if role_source == "unconfirmed-master" else role,
                     "role_confirmed": role_source == "confirmed"},
        }), encoding="utf-8")
    before = dict(__import__("os").environ)
    options = qlh._shell_options(["--auto-start"])
    assert options["environment"]["QLH_NODE_ROLE"] == role
    runtime = AppRuntime()
    runtime.configure_listener(18191, environment=options["environment"])
    assert runtime.config.node_role == role
    assert runtime.config.api_port == 18191
    assert dict(__import__("os").environ) == before
    assert runtime._services is None
    runtime.close()
    if role_source == "none":
        # Non-interactive compositions retain the development default.
        assert SettingsStore().load_listener()["QLH_NODE_ROLE"] == "client"


@pytest.mark.parametrize("role", ["master", "client"])
def test_single_node_role_binding_preserves_real_admission(monkeypatch, tmp_path, role):
    import asyncio
    import threading
    import time
    import httpx
    import api_server
    from api import routes_model_intent
    from cluster_fence import ControlFence
    from model_supply import ModelSupplyCoordinator
    from qlh_core.adapters.settings import SettingsStore
    from scheduler import Scheduler

    config = SettingsStore({
        "QLH_STATE_DIR": str(tmp_path / "state"), "QLH_NODE_CONFIG_PATH": str(tmp_path / "node.json"),
        "QLH_NODE_SECRET_STORE_PATH": str(tmp_path / "secrets.json"),
        "QLH_NODE_ROLE": role, "QLH_RUN_MODE": "distributed", "QLH_RELEASE_PROFILE_ENFORCE": "0",
        "QLH_DISTRIBUTED_INFERENCE_ENABLED": "1",
    }, dotenv_path=tmp_path / "absent.env").load()
    host = ModelHost()
    scheduler = Scheduler(host=host, runtime_config=config)
    scheduler._distributed_inference_enabled = True
    scheduler._model_supply = ModelSupplyCoordinator(scheduler, tmp_path / "supply")
    monkeypatch.setattr(api_server, "scheduler", scheduler)
    monkeypatch.setattr(api_server, "model_host", host)
    monkeypatch.setattr(api_server, "model_manager", host)
    monkeypatch.setattr(api_server, "API_PORT", 18191)
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    monkeypatch.setattr(api_server, "control_fence", ControlFence(required=False))
    monkeypatch.setattr(api_server, "_get_all_model_configs", lambda: [])
    monkeypatch.setattr(routes_model_intent, "_coordinator", None)
    monkeypatch.setattr(routes_model_intent, "_preflight_selection", lambda intent: None)
    monkeypatch.setattr(scheduler._model_supply, "_sources",
                        lambda *_a, **_k: pytest.fail("single-node selection must not inspect or load model weights"))
    prior_threads = set(threading.enumerate())

    async def scenario():
        transport = httpx.ASGITransport(app=api_server.app, client=("127.0.0.1", 45000))
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                catalog = await client.get("/api/models/intent/catalog")
                if role == "client":
                    assert catalog.status_code == 503
                    assert catalog.json()["detail"]["code"] == "INTENT_MASTER_DISCONNECTED"
                    selection = await client.post("/api/models/intent", json={"model_id": "fixture"})
                    assert selection.status_code == 503
                    assert selection.json()["detail"]["code"] == "INTENT_MASTER_DISCONNECTED"
                    assert routes_model_intent._coordinator is None
                    return
                assert catalog.status_code == 200, catalog.text
                binding = catalog.json()["binding"]
                assert binding["status"] == "connected" and binding["is_master"] is True
                assert binding["master_api_port"] == 18191
                selection = await client.post("/api/models/intent", json={"model_id": "fixture"})
                assert selection.status_code == 200, selection.text
                operation_id = selection.json()["operation_id"]
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    value = (await client.get("/api/models/intent/current")).json()
                    if value["status"] == "waiting_nodes":
                        break
                    await asyncio.sleep(0.01)
                assert value["operation_id"] == operation_id
                assert value["status"] == "waiting_nodes"
                assert value["reason_code"] == "SUPPLY_WAITING_NODES"
                assert value["loaded_ready"] is False and value["runnable"] is False
                assert value["plan"] is None and host.peek_manager() is None
                cancelled = await client.delete("/api/models/intent/" + operation_id)
                assert cancelled.status_code == 200, cancelled.text
                assert cancelled.json()["status"] == "cancelled"
                assert not cancelled.json()["runnable"]
        finally:
            if routes_model_intent._coordinator is not None:
                await routes_model_intent._coordinator.close()
            scheduler._model_supply.cancel_active()
            for thread in set(threading.enumerate()) - prior_threads:
                if thread.name == "model-supply":
                    await asyncio.to_thread(thread.join, 3)
                    assert not thread.is_alive(), "single-node supply worker leaked"
    asyncio.run(scenario())
    assert scheduler._tcp_server is None and scheduler._tcp_client is None
    assert host.peek_manager() is None
