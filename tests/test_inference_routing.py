"""
inference-svc 执行段 T9.5 同步契约测试
====================================
覆盖 routes.chat_stream 的 routing gate（distributed_required）与 interactive
事件序列（start→token*→done），以及 EngineHost 决策方法的 local_only /
distributed_required 语义（转发/流水线跳过、强制本地）。
"""

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference_svc_main import build_app  # noqa: E402
from inference_service.engine_host import EngineHost  # noqa: E402
from inference_service.protocol import ChatRequest  # noqa: E402
from tests.test_inference_service_protocol import FakeEngineHost  # noqa: E402


def _sse_events(response):
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            import json
            events.append(json.loads(line[len("data: "):]))
    return events


def _req(**overrides):
    base = dict(
        routing_preference="auto",
        execution_mode="auto",
        message="hi",
        session_id=None,
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
        show_thinking=False,
        allow_external=False,
        prefer_external=False,
        generation_id="gen_x",
        client_node_id=None,
        client_node_type=None,
        client_mode=None,
        client_app_variant=None,
    )
    base.update(overrides)
    return ChatRequest(**base)


@pytest.fixture
def svc_client():
    app = build_app("master", engine_host=FakeEngineHost())
    with TestClient(app) as client:
        yield client


class TestInteractiveContract:
    """inference-svc 的 interactive 事件序列（T9 契约 §9.4.1）。"""

    def test_start_then_tokens_then_done(self, svc_client):
        response = svc_client.post("/v1/chat/stream", json={
            "message": "你好",
            "streaming_mode": "interactive",
            "session_id": "sess_iv",
            "routing_preference": "local_only",
        })
        assert response.status_code == 200
        events = _sse_events(response)
        assert events[0]["start"] is True
        assert events[0]["session_id"] == "sess_iv"
        assert events[0]["routing_preference"] == "local_only"
        tokens = [e["token"] for e in events if "token" in e]
        assert tokens == ["你", "好"]
        done = events[-1]
        assert done["done"] is True
        assert done["generation_id"].startswith("gen_")
        assert done["session_id"] == "sess_iv"
        assert done["history_committed"] is False  # 薄实现如实上报
        assert done["metrics"]["routing_preference"] == "local_only"
        assert done["metrics"]["distributed_used"] is False

    def test_distributed_required_rejected_without_path(self, svc_client):
        # FakeEngineHost 无 scheduler → 无分布式路径
        response = svc_client.post("/v1/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
            "routing_preference": "distributed_required",
        })
        events = _sse_events(response)
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "distributed_required" in error[0]["error"]

    def test_full_mode_also_gated(self, svc_client):
        response = svc_client.post("/v1/chat/stream", json={
            "message": "hi",
            "streaming_mode": "full",
            "routing_preference": "distributed_required",
        })
        events = _sse_events(response)
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "distributed_required" in error[0]["error"]


class TestRoutingGates:
    """EngineHost 决策方法的 local_only / distributed_required 语义（单元级）。"""

    def test_external_decision_local_only_override(self):
        host = FakeEngineHost()
        decision = host._external_route_decision(_req(
            routing_preference="local_only",
            allow_external=True,
            prefer_external=True,
        ))
        assert decision.use_external is False

    def test_external_decision_auto_unchanged(self):
        host = FakeEngineHost()
        decision = host._external_route_decision(_req(
            routing_preference="auto",
            allow_external=False,
            prefer_external=False,
        ))
        # 未启用外部配置时 auto 不触发外部（保持原语义）
        assert decision.use_external is False

    def test_distributed_path_available(self, monkeypatch):
        host = FakeEngineHost()
        assert host._distributed_path_available() is False  # 无 scheduler

        fake = SimpleNamespace(
            get_distributed_inference_enabled=lambda: True,
            _effective_role=lambda: "master",
        )
        host._scheduler = fake
        host._run_mode = "distributed"
        assert host._distributed_path_available() is True

        fake_role = SimpleNamespace(
            get_distributed_inference_enabled=lambda: True,
            _effective_role=lambda: "client",
        )
        host._scheduler = fake_role
        assert host._distributed_path_available() is True  # 从节点可转发

    def test_routing_gate_required_fails_without_path(self):
        host = FakeEngineHost()
        gate = host._routing_gate_error(_req(
            routing_preference="distributed_required",
        ))
        assert gate is not None and "distributed_required" in gate
        assert host._routing_gate_error(_req(
            routing_preference="local_only",
        )) is None
        assert host._routing_gate_error(_req()) is None

    def test_routing_gate_required_allowed_with_path(self):
        host = FakeEngineHost()
        host._scheduler = SimpleNamespace(
            get_distributed_inference_enabled=lambda: True,
            _effective_role=lambda: "master",
        )
        host._run_mode = "distributed"
        assert host._routing_gate_error(_req(
            routing_preference="distributed_required",
        )) is None


class TestChatFullLocalOnly:
    """EngineHost.chat_full 基类方法的 local_only 决策（从节点场景）。"""

    def _host_with_client_scheduler(self):
        forwarded = []
        fake_scheduler = SimpleNamespace(
            get_distributed_inference_enabled=lambda: True,
            _effective_role=lambda: "client",
            forward_inference_to_master=lambda **kw:
                forwarded.append(kw) or
                {"status": "ok", "content": "转发结果", "metrics": {}},
            run_pipeline_safe=lambda **kw:
                {"status": "ok", "response": "p", "metrics": {}},
            refresh_task_worker_capabilities=lambda: None,
        )
        host = FakeEngineHost()
        host._scheduler = fake_scheduler
        host._run_mode = "distributed"
        return host, forwarded

    def test_chat_full_auto_forwards_on_client(self, monkeypatch):
        host, forwarded = self._host_with_client_scheduler()
        result = EngineHost.chat_full(host, _req(), None)
        assert len(forwarded) == 1
        assert result["content"] == "转发结果"

    def test_chat_full_local_only_skips_forward(self, monkeypatch):
        host, forwarded = self._host_with_client_scheduler()
        result = EngineHost.chat_full(
            host, _req(routing_preference="local_only"), None,
        )
        assert len(forwarded) == 0
        assert result["content"] == "候选答案内容"  # 本地 FakeModel.chat 输出
