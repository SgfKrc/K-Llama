import os
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from worker_assignment_state import WorkerAssignmentRegistry


def test_unload_model_is_idempotent_and_keeps_lazy_manager_cold(monkeypatch):
    from model_host import ModelHost

    host = ModelHost()
    scheduler = SimpleNamespace(
        _inference_lock=threading.RLock(),
        _layer_execution_lock=threading.RLock(),
        _layer_config_lock=threading.RLock(),
        # ★ 2026-10-07（DIST-NEXT-3）：assignment 权威视图是卸载路径依赖的**真实契约**
        #   （`_layer_config_pushed` 已降级为它的派生视图），fake scheduler 必须提供它。
        _worker_assignments=WorkerAssignmentRegistry(),
        _layer_config_pushed=set(),
        _layer_config_retry_state={},
        _pipeline_load_transaction=None,
        _active_pipeline_capacity_plan=None,
        _active_layer_config=None,
        _last_layer_config_ack_payload=None,
        _local_pipeline_steps={},
        get_distributed_inference_enabled=lambda: True,
        _effective_role=lambda: "client",
        release_pipeline_worker_for_local_model=lambda: None,
        refresh_task_worker_capabilities=lambda: None,
    )
    monkeypatch.setattr(api_server, "model_host", host)
    monkeypatch.setattr(api_server, "model_manager", host)
    monkeypatch.setattr(api_server, "scheduler", scheduler)
    assert host._manager._instance is None
    result = api_server._unload_model_under_model_lock()

    assert result == {
        "success": True,
        "loaded": False,
        "unloaded": False,
        "message": "当前没有已加载的模型",
    }
    assert host._manager._instance is None


def test_unload_model_releases_engine_runtime_and_worker_reservation(monkeypatch):
    calls = []
    manager = SimpleNamespace(
        is_loaded=True,
        unload_model=lambda: calls.append("unload"),
    )
    host = SimpleNamespace(
        model_loaded=True,
        current_quant="int4",
        full_chat_execution_lock=threading.RLock(),
        has_loaded_model=lambda: True,
    )
    scheduler = SimpleNamespace(
        _inference_lock=threading.RLock(),
        _layer_execution_lock=threading.RLock(),
        _layer_config_lock=threading.RLock(),
        # ★ 2026-10-07（DIST-NEXT-3）：同上，卸载路径要求 assignment 权威视图存在。
        _worker_assignments=WorkerAssignmentRegistry(),
        _layer_config_pushed=set(),
        _layer_config_retry_state={},
        _pipeline_load_transaction=None,
        _active_pipeline_capacity_plan=None,
        _active_layer_config=None,
        _last_layer_config_ack_payload=None,
        _local_pipeline_steps={},
        get_distributed_inference_enabled=lambda: True,
        _effective_role=lambda: "client",
        release_pipeline_worker_for_local_model=lambda: calls.append("release"),
        refresh_task_worker_capabilities=lambda: calls.append("refresh"),
    )
    monkeypatch.setattr(api_server, "model_host", host)
    monkeypatch.setattr(api_server, "model_manager", manager)
    monkeypatch.setattr(api_server, "scheduler", scheduler)
    monkeypatch.setattr(
        api_server,
        "_reset_runtime_conversation_state",
        lambda **_kwargs: calls.append("reset"),
    )

    result = api_server._unload_model_under_model_lock()

    assert result["unloaded"] is True
    assert calls == ["unload", "reset", "release", "refresh"]
    assert host.model_loaded is False
    assert host.current_quant is None
