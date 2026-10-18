"""
T9 interactive 流式模式契约测试
==============================
覆盖 /api/chat/stream?streaming_mode=interactive 的 SSE 事件序列：
start → token* → done（含 generation_id/request_id/session_id/history_committed），
取消 → cancelled，失败 → error，以及路由偏好 metrics 与事务提交。
"""

import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from api_server import ChatGenerationCancelled


@pytest.fixture
def interactive_env(monkeypatch):
    """把调度/模型状态 mock 到 llama.cpp 假流式路径（run_pipeline_safe）。"""
    calls = {"run_pipeline_safe": None, "commits": [],
             "pipeline_stream": 0, "full_model_stream": 0, "forward": 0}

    def fake_run_pipeline_safe(message, **kwargs):
        calls["run_pipeline_safe"] = (message, kwargs)
        if calls.get("raise_cancelled"):
            raise ChatGenerationCancelled("gen_x")
        if calls.get("raise_error"):
            raise RuntimeError("engine exploded")
        return {
            "status": "ok",
            "response": "你好，世界！",
            "metrics": {"engine": "llama_cpp", "tokens_generated": 6},
            "error": None,
        }

    def fake_pipeline_stream(message, **kwargs):
        calls["pipeline_stream"] += 1
        return iter([
            {"token": "流"},
            {"token": "式"},
            {"done": True, "response": "流式",
             "metrics": {"engine": "pytorch_pipeline"}},
        ])

    def fake_full_model_stream(message, **kwargs):
        calls["full_model_stream"] += 1
        return iter([
            {"token": "单机"},
            {"done": True, "response": "单机",
             "metrics": {"engine": "pytorch"}},
        ])

    def fake_forward(message, **kwargs):
        calls["forward"] += 1
        return {"status": "ok", "content": "转发结果",
                "metrics": {"engine": "distributed_forward"}}

    def fake_chat(messages, **kwargs):
        calls["chat"] = (calls.get("chat") or 0) + 1
        return {
            "content": "full 本地回复",
            "tokens_per_second": 12.5,
            "usage": {"completion_tokens": 7},
            "followups": [],
        }

    fake_scheduler = SimpleNamespace(
        get_distributed_inference_enabled=lambda: calls.get(
            "distributed_enabled", False,
        ),
        _effective_role=lambda: calls.get("role", "master"),
        run_pipeline_safe=fake_run_pipeline_safe,
        run_pipeline_stream=fake_pipeline_stream,
        _run_full_model_inference_stream=fake_full_model_stream,
        forward_inference_to_master=fake_forward,
        record_task_complete=lambda success=True: None,
        get_effective_node_id=lambda: "master",
    )
    fake_model_manager = SimpleNamespace(
        is_loaded=True,
        _engine_type="llama_cpp",
        chat=fake_chat,
    )

    monkeypatch.setattr(api_server, "scheduler", fake_scheduler)
    monkeypatch.setattr(api_server, "model_manager", fake_model_manager)
    monkeypatch.setattr(
        api_server, "model_host",
        SimpleNamespace(
            model_loaded=True,
            full_chat_execution_lock=threading.RLock(),
        ),
    )
    monkeypatch.setattr(api_server, "RUN_MODE", "local")
    monkeypatch.setattr(api_server._chat_context, "active_session_id", None)
    monkeypatch.setattr(api_server._chat_context, "histories", {})
    monkeypatch.setattr(
        api_server._chat_context,
        "_load_history",
        lambda _session_id: [],
    )
    api_server._chat_context.invalidate(clear_histories=True)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    def fake_commit(session_id, user_message, response_text, metrics):
        calls["commits"].append(
            (session_id, user_message, response_text, dict(metrics)),
        )
        return True

    monkeypatch.setattr(api_server, "_persist_conversation_turn", fake_commit)
    monkeypatch.setattr(api_server._chat_context, "_persist_turn", fake_commit)
    monkeypatch.setattr(api_server, "_external_route_decision",
                        lambda req: SimpleNamespace(use_external=False))

    client = TestClient(api_server.app)
    return client, calls


def _sse_events(response):
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


class TestInteractiveContract:
    """事件序列与字段契约。"""

    def test_start_then_tokens_then_done(self, interactive_env):
        client, calls = interactive_env
        response = client.post("/api/chat/stream", json={
            "message": "你好",
            "streaming_mode": "interactive",
            "session_id": "sess_t9",
            "routing_preference": "local_only",
        })
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _sse_events(response)
        # start → token → done
        assert events[0]["start"] is True
        assert events[0]["generation_id"].startswith("gen_")
        assert events[0]["session_id"] == "sess_t9"
        assert events[0]["routing_preference"] == "local_only"

        tokens = [e["token"] for e in events if "token" in e]
        assert tokens == ["你好，世界！"]

        done = events[-1]
        assert done["done"] is True
        assert done["response"] == "你好，世界！"
        assert done["generation_id"] == events[0]["generation_id"]
        assert done["session_id"] == "sess_t9"
        assert done["history_committed"] is True
        assert done["metrics"]["routing_preference"] == "local_only"
        assert done["metrics"]["distributed_requested"] is False
        # ★ DIST-NEXT-8：完成路径带统一相位与互斥终态（本地执行 ⇒ 非 admitted）。
        assert done["metrics"]["outcome"] == "completed"
        assert done["metrics"]["started"] is True
        assert done["metrics"]["completed"] is True
        assert done["metrics"]["fallback"] is False
        assert done["metrics"]["refused"] is False
        assert done["metrics"]["cancelled"] is False
        assert done["metrics"]["failed"] is False

    def test_cancel_emits_cancelled_event(self, interactive_env):
        client, calls = interactive_env
        calls["raise_cancelled"] = True
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
            "generation_id": "gen_t9_cancel",
        })
        events = _sse_events(response)
        assert events[0]["start"] is True
        cancelled = [e for e in events if e.get("cancelled")]
        assert len(cancelled) == 1
        assert cancelled[0]["generation_id"] == "gen_t9_cancel"
        assert "partial" in cancelled[0]
        # ★ DIST-NEXT-8：取消是独立终态（带相位与稳定 reason），不被 fallback/error 覆盖。
        assert cancelled[0]["outcome"] == "cancelled"
        assert cancelled[0]["cancelled"] is True
        assert cancelled[0]["started"] is True
        assert cancelled[0]["fallback"] is False
        assert cancelled[0]["failed"] is False
        assert cancelled[0]["outcome_reason"] == "generation_cancelled"
        assert not [e for e in events if e.get("done")]
        # 取消不提交历史
        assert calls["commits"] == []

    def test_error_emits_error_event(self, interactive_env):
        client, calls = interactive_env
        calls["raise_error"] = True
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
        })
        events = _sse_events(response)
        assert events[0]["start"] is True
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "engine exploded" in error[0]["error"]
        # ★ DIST-NEXT-8：链路错误是 `failed` 终态，不是 `refused`、也不靠 fallback 表达。
        assert error[0]["outcome"] == "failed"
        assert error[0]["failed"] is True
        assert error[0]["refused"] is False
        assert error[0]["outcome_reason"] == "request_failed"
        assert calls["commits"] == []

    def test_commit_receives_user_and_response_together(self, interactive_env):
        client, calls = interactive_env
        response = client.post("/api/chat/stream", json={
            "message": "提交测试",
            "streaming_mode": "interactive",
            "session_id": "sess_commit",
        })
        assert response.status_code == 200
        assert len(calls["commits"]) == 1
        session_id, user_message, response_text, metrics = calls["commits"][0]
        assert session_id == "sess_commit"
        assert user_message == "提交测试"
        assert response_text == "你好，世界！"
        assert metrics["engine"] == "llama_cpp"
        assert api_server._chat_context.histories["sess_commit"] == [
            {"role": "user", "content": "提交测试"},
            {"role": "assistant", "content": "你好，世界！"},
        ]

    def test_interactive_uses_and_extends_canonical_history(self, interactive_env):
        client, calls = interactive_env
        api_server._chat_context.set_history("sess_context", [
            {"role": "user", "content": "记住颜色"},
            {"role": "assistant", "content": "蓝色"},
        ])

        response = client.post("/api/chat/stream", json={
            "message": "颜色是什么",
            "streaming_mode": "interactive",
            "session_id": "sess_context",
        })

        assert response.status_code == 200
        _message, kwargs = calls["run_pipeline_safe"]
        assert kwargs["messages"] == [
            {"role": "user", "content": "记住颜色"},
            {"role": "assistant", "content": "蓝色"},
            {"role": "user", "content": "颜色是什么"},
        ]
        assert api_server._chat_context.histories["sess_context"][-2:] == [
            {"role": "user", "content": "颜色是什么"},
            {"role": "assistant", "content": "你好，世界！"},
        ]

    def test_distributed_required_marks_requested(self, interactive_env, monkeypatch):
        client, calls = interactive_env
        calls["distributed_enabled"] = True
        calls["role"] = "master"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        monkeypatch.setattr(
            api_server, "model_manager",
            SimpleNamespace(is_loaded=True, engine_type="pytorch",
                            runtime_kind="pytorch_cpu", native_abi="torch_module.v1"),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
            "routing_preference": "distributed_required",
        })
        done = _sse_events(response)[-1]
        assert done["metrics"]["distributed_requested"] is True
        assert done["metrics"]["distributed_used"] is True

    def test_invalid_routing_preference_rejected(self, interactive_env):
        client, _calls = interactive_env
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
            "routing_preference": "sideways",
        })
        assert response.status_code == 422

    def test_client_node_guides_to_master(self, interactive_env, monkeypatch):
        client, _calls = interactive_env
        # 从节点场景：转发条件成立
        fake_scheduler = SimpleNamespace(
            get_distributed_inference_enabled=lambda: True,
            _effective_role=lambda: "client",
            run_pipeline_safe=lambda **kw: {},
        )
        monkeypatch.setattr(api_server, "scheduler", fake_scheduler)
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
        })
        events = _sse_events(response)
        assert events[0]["start"] is True
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "主节点" in error[0]["error"]


class TestRoutingPreference:
    """T9.5 请求级路由偏好：local_only / distributed_required / 回退 metrics。"""

    def _set_client_scene(self, env, monkeypatch):
        """从节点场景：分布式启用 + role=client。"""
        client, calls = env
        calls["distributed_enabled"] = True
        calls["role"] = "client"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        return client, calls

    @pytest.mark.parametrize('streaming_mode', ['full', 'interactive', 'fast'])
    def test_required_cold_start_never_auto_loads_full_model(
        self, interactive_env, monkeypatch, streaming_mode,
    ):
        client, calls = interactive_env
        calls['distributed_enabled'] = True
        calls['role'] = 'master'
        monkeypatch.setattr(api_server, 'RUN_MODE', 'distributed')
        monkeypatch.setattr(api_server.model_host, 'model_loaded', False)
        monkeypatch.setattr(
            api_server, 'model_manager',
            SimpleNamespace(is_loaded=False, is_pipeline_prepared=False),
        )
        monkeypatch.setattr(api_server, 'runtime_supports', lambda *_: False)
        auto_load_calls = []
        monkeypatch.setattr(
            api_server, '_auto_load_default_model',
            lambda: auto_load_calls.append(True),
        )

        response = client.post('/api/chat/stream', json={
            'message': 'hi',
            'streaming_mode': streaming_mode,
            'routing_preference': 'distributed_required',
        })

        assert not auto_load_calls
        assert response.status_code == 200
        errors = [event for event in _sse_events(response) if event.get('error')]
        assert errors
        if streaming_mode == 'full':
            assert errors[0]['outcome_reason'] == 'DISTRIBUTED_PIPELINE_NOT_READY'

    def test_client_without_local_only_guides_to_master(self, interactive_env, monkeypatch):
        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
        })
        events = _sse_events(response)
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "主节点" in error[0]["error"]
        assert calls["forward"] == 0

    def test_local_only_on_client_executes_locally(self, interactive_env, monkeypatch):
        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "local_only",
        })
        assert response.status_code == 200
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert events[-1]["response"] == "你好，世界！"
        assert calls["forward"] == 0  # 未转发主节点
        assert calls["run_pipeline_safe"] is not None  # 本地路径执行
        assert events[-1]["metrics"]["distributed_used"] is False

    def test_local_only_skips_pipeline_on_master(self, interactive_env, monkeypatch):
        client, calls = interactive_env
        calls["distributed_enabled"] = True
        calls["role"] = "master"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        monkeypatch.setattr(
            api_server, "model_manager",
            SimpleNamespace(is_loaded=True, engine_type="pytorch",
                            runtime_kind="pytorch_cpu", native_abi="torch_module.v1"),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "local_only",
        })
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert calls["pipeline_stream"] == 0   # 流水线被跳过
        assert calls["full_model_stream"] == 1  # 单机 PyTorch 执行
        assert events[-1]["metrics"]["distributed_used"] is False

    def test_auto_uses_pipeline_when_available(self, interactive_env, monkeypatch):
        client, calls = interactive_env
        calls["distributed_enabled"] = True
        calls["role"] = "master"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        monkeypatch.setattr(
            api_server, "model_manager",
            SimpleNamespace(is_loaded=True, engine_type="pytorch",
                            runtime_kind="pytorch_cpu", native_abi="torch_module.v1"),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
        })
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert calls["pipeline_stream"] == 1
        assert events[-1]["metrics"]["distributed_used"] is True

    def test_distributed_required_fails_without_path(self, interactive_env):
        client, _calls = interactive_env  # 分布式未启用
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "distributed_required",
        })
        events = _sse_events(response)
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "distributed_required" in error[0]["error"]
        # ★ DIST-NEXT-8：路由门是「不可恢复、dispatch 前的具名拒绝」⇒ `refused`，
        #   与链路失败（`failed`）在 metrics 层面分开。
        assert error[0]["outcome"] == "refused"
        assert error[0]["refused"] is True
        assert error[0]["failed"] is False
        assert error[0]["outcome_reason"] == "request_refused"

    def test_distributed_required_allowed_with_path(self, interactive_env, monkeypatch):
        client, calls = interactive_env
        calls["distributed_enabled"] = True
        calls["role"] = "master"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        monkeypatch.setattr(
            api_server, "model_manager",
            SimpleNamespace(is_loaded=True, engine_type="pytorch",
                            runtime_kind="pytorch_cpu", native_abi="torch_module.v1"),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "distributed_required",
        })
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert calls["pipeline_stream"] == 1
        assert events[-1]["metrics"]["distributed_used"] is True

    def test_distributed_preferred_fallback_metrics(self, interactive_env):
        client, _calls = interactive_env  # 分布式未启用 → 本地回退
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "distributed_preferred",
        })
        events = _sse_events(response)
        done = events[-1]
        assert done["done"] is True
        assert done["metrics"]["distributed_used"] is False
        assert done["metrics"]["fallback"] is True
        assert "fallback_reason" in done["metrics"]

    def test_full_mode_local_only_executes_locally(self, interactive_env, monkeypatch):
        class _Lock:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        monkeypatch.setattr(
            api_server, "model_host",
            SimpleNamespace(model_loaded=True,
                            full_chat_execution_lock=_Lock()),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "full",
            "routing_preference": "local_only",
        })
        assert response.status_code == 200
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert calls["forward"] == 0

    @pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
    def test_required_worker_forward_allows_returning_stage_lock(
        self, interactive_env, monkeypatch, endpoint,
    ):
        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        acquired = threading.Event()
        stage_threads = []

        def returning_stage():
            lock = api_server.model_host.full_chat_execution_lock
            if lock.acquire(timeout=1.0):
                try:
                    acquired.set()
                finally:
                    lock.release()

        def forward(message, **kwargs):
            assert kwargs["routing_preference"] == "distributed_required"
            stage = threading.Thread(target=returning_stage)
            stage_threads.append(stage)
            stage.start()
            stage.join(timeout=2.0)
            assert acquired.is_set(), "worker held model lock while awaiting master"
            return {
                "status": "ok", "content": "2",
                "metrics": {"distributed_used": True, "workers_used": ["worker"]},
            }

        monkeypatch.setattr(api_server.scheduler, "forward_inference_to_master", forward)
        monkeypatch.setattr(
            api_server, "_auto_load_default_model",
            lambda: pytest.fail("required worker forwarding loaded a full model"),
        )
        response = client.post(endpoint, json={
            "message": "one plus one", "streaming_mode": "full",
            "routing_preference": "distributed_required",
        })
        for stage in stage_threads:
            stage.join(timeout=2.0)
        assert response.status_code == 200
        result = (_sse_events(response)[-1] if endpoint.endswith("/stream")
                  else response.json())
        assert (result.get("response") or result.get("content")) == "2"
        assert result["metrics"]["distributed_used"] is True
        assert result["metrics"]["fallback"] is False
        assert calls.get("chat", 0) == 0
        assert acquired.is_set()

    @pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
    @pytest.mark.parametrize("transition", ["master", "disabled"])
    def test_unlocked_forward_role_change_never_enters_native_execution(
        self, interactive_env, monkeypatch, endpoint, transition,
    ):
        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        execute = api_server._execute_requested_chat

        def change_before_execution(req, cancel_event=None, request_deadline=None, **kwargs):
            assert kwargs["forward_only"] is True
            calls["role"] = "master" if transition == "master" else "client"
            calls["distributed_enabled"] = transition != "disabled"
            return execute(req, cancel_event, request_deadline, **kwargs)

        monkeypatch.setattr(api_server, "_execute_requested_chat", change_before_execution)
        monkeypatch.setattr(
            api_server, "_auto_load_default_model",
            lambda: pytest.fail("forward-only request auto-loaded a full model"),
        )
        response = client.post(endpoint, json={
            "message": "hi", "streaming_mode": "full",
            "routing_preference": "distributed_required",
        })
        if endpoint.endswith("/stream"):
            assert response.status_code == 200
            error = next(event for event in _sse_events(response) if event.get("error"))
            assert error["outcome_reason"] == "DISTRIBUTED_FORWARD_CONTEXT_CHANGED"
        else:
            assert response.status_code == 409
            assert response.json()["error_code"] == "DISTRIBUTED_FORWARD_CONTEXT_CHANGED"
        assert calls.get("chat", 0) == 0
        assert calls["forward"] == 0
        assert calls["run_pipeline_safe"] is None

    @pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
    @pytest.mark.parametrize("transition", ["master", "disabled"])
    def test_role_change_after_forward_guard_stays_forward_only(
        self, interactive_env, monkeypatch, endpoint, transition,
    ):
        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        prepare = api_server._prepare_chat_context
        forwarded = []

        def change_during_context(*args, **kwargs):
            context = prepare(*args, **kwargs)
            calls["role"] = "master" if transition == "master" else "client"
            calls["distributed_enabled"] = transition != "disabled"
            return context

        def denied_by_forward_owner(**kwargs):
            assert not api_server._should_forward_chat_to_master()
            forwarded.append(True)
            return {"status": "disconnected", "error": "worker forwarding role changed"}

        monkeypatch.setattr(api_server, "_prepare_chat_context", change_during_context)
        monkeypatch.setattr(api_server.scheduler, "forward_inference_to_master", denied_by_forward_owner)
        response = client.post(endpoint, json={
            "message": "hi", "streaming_mode": "full",
            "routing_preference": "distributed_required",
        })
        if endpoint.endswith("/stream"):
            assert response.status_code == 200
            assert any(event.get("error") for event in _sse_events(response))
        else:
            assert response.status_code == 503
        assert forwarded == [True]
        assert calls.get("chat", 0) == 0
        assert calls["run_pipeline_safe"] is None

    @pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
    def test_required_worker_forward_failure_never_runs_local_model(
        self, interactive_env, monkeypatch, endpoint,
    ):
        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        monkeypatch.setattr(
            api_server.scheduler, "forward_inference_to_master",
            lambda **kwargs: {"status": "disconnected", "error": "master unavailable"},
        )
        monkeypatch.setattr(
            api_server, "_auto_load_default_model",
            lambda: pytest.fail("required forwarding failure loaded a full model"),
        )
        response = client.post(endpoint, json={
            "message": "hi", "streaming_mode": "full",
            "routing_preference": "distributed_required",
        })
        if endpoint.endswith("/stream"):
            assert response.status_code == 200
            events = _sse_events(response)
            assert any(event.get("error") for event in events)
            assert not any(event.get("done") and not event.get("error") for event in events)
        else:
            assert response.status_code == 503
        assert calls.get("chat", 0) == 0
        assert calls["run_pipeline_safe"] is None

    @pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
    def test_local_full_chat_still_serializes_model_execution(
        self, interactive_env, monkeypatch, endpoint,
    ):
        client, calls = interactive_env
        acquired = []

        def chat(messages, **kwargs):
            def competing_stage():
                lock = api_server.model_host.full_chat_execution_lock
                owns = lock.acquire(timeout=0.1)
                acquired.append(owns)
                if owns:
                    lock.release()
            stage = threading.Thread(target=competing_stage)
            stage.start()
            stage.join(timeout=2.0)
            assert acquired and not any(acquired)
            return {"content": "local", "usage": {"completion_tokens": 1}}

        monkeypatch.setattr(api_server.model_manager, "chat", chat)
        response = client.post(endpoint, json={
            "message": "hi", "streaming_mode": "full", "routing_preference": "local_only",
        })
        assert response.status_code == 200
        result = (_sse_events(response)[-1] if endpoint.endswith("/stream")
                  else response.json())
        assert (result.get("response") or result.get("content")) == "local"
        assert acquired and not any(acquired)
        assert calls["forward"] == 0

    def test_full_mode_distributed_required_rejected(self, interactive_env):
        client, _calls = interactive_env  # 分布式未启用
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "full",
            "routing_preference": "distributed_required",
        })
        assert response.status_code == 200
        events = _sse_events(response)
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "distributed_required" in error[0]["error"]


class TestFastContextContract:
    def test_fast_uses_and_commits_canonical_history(self, interactive_env):
        client, calls = interactive_env
        api_server._chat_context.set_history("sess_fast", [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ])

        response = client.post("/api/chat/stream", json={
            "message": "third",
            "streaming_mode": "fast",
            "session_id": "sess_fast",
        })

        assert response.status_code == 200
        _message, kwargs = calls["run_pipeline_safe"]
        assert kwargs["messages"] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        done = _sse_events(response)[-1]
        assert done["done"] is True
        assert done["history_committed"] is True
        assert api_server._chat_context.histories["sess_fast"][-2:] == [
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": "你好，世界！"},
        ]


class TestContextReload:
    def test_same_active_session_reloads_after_model_state_reset(
        self, interactive_env, monkeypatch,
    ):
        client, calls = interactive_env
        rows = [
            {"role": "user", "content": "persisted question"},
            {"role": "assistant", "content": "persisted answer"},
        ]
        monkeypatch.setattr(
            api_server._chat_context,
            "_load_history",
            lambda session_id: list(rows) if session_id == "sess_reload" else [],
        )
        api_server._chat_context.active_session_id = "sess_reload"
        api_server._chat_context.histories = {
            "sess_reload": [{"role": "user", "content": "stale"}],
        }

        api_server._reset_runtime_conversation_state(clear_histories=True)
        response = client.post("/api/chat/stream", json={
            "message": "continue",
            "streaming_mode": "interactive",
            "session_id": "sess_reload",
        })

        assert response.status_code == 200
        _message, kwargs = calls["run_pipeline_safe"]
        assert kwargs["messages"] == [
            *rows,
            {"role": "user", "content": "continue"},
        ]


class TestSessionCrudContextContract:
    def test_activate_same_stale_session_reloads_canonical_history(
        self, interactive_env, monkeypatch,
    ):
        client, _calls = interactive_env
        rows = [
            {"role": "user", "content": "persisted question"},
            {"role": "assistant", "content": "persisted answer"},
        ]
        api_server._chat_context.active_session_id = "sess_activate"
        api_server._chat_context.histories = {
            "sess_activate": [{"role": "user", "content": "stale"}],
        }
        api_server._chat_context.invalidate_session("sess_activate")
        monkeypatch.setattr(
            api_server._chat_context,
            "_load_history",
            lambda session_id: list(rows) if session_id == "sess_activate" else [],
        )

        response = client.post("/api/sessions/sess_activate/activate")

        assert response.status_code == 200
        assert response.json()["messages"] == rows
        assert api_server._chat_context.histories["sess_activate"] == rows

    def test_delete_active_session_clears_kv_before_dropping_facade(
        self, interactive_env, monkeypatch,
    ):
        client, _calls = interactive_env
        cleared = []
        initialized = []
        api_server._chat_context.active_session_id = "sess_delete"
        api_server._chat_context.histories = {
            "sess_delete": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        monkeypatch.setattr(
            api_server,
            "kv_cache",
            SimpleNamespace(clear=lambda: cleared.append(True)),
        )
        monkeypatch.setattr(
            api_server._local_store,
            "delete_local_session",
            lambda session_id: 2 if session_id == "sess_delete" else 0,
        )
        monkeypatch.setattr(
            api_server, "_init_kv_cache", lambda: initialized.append(True),
        )

        response = client.delete("/api/sessions/sess_delete")

        assert response.status_code == 200
        assert api_server._chat_context.active_session_id is None
        assert "sess_delete" not in api_server._chat_context.histories
        assert cleared == [True]
        assert initialized == [True]

    def test_delete_old_persisted_turn_invalidates_window_instead_of_splicing_it(
        self, interactive_env, monkeypatch,
    ):
        client, _calls = interactive_env
        warm_window = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": str(index),
            }
            for index in range(20, 220)
        ]
        reloaded_window = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": str(index),
            }
            for index in range(18, 218)
        ]
        api_server._chat_context.active_session_id = "sess_window"
        api_server._chat_context.histories = {"sess_window": warm_window}
        monkeypatch.setattr(
            api_server._local_store,
            "get_local_conversation_count",
            lambda session_id: 220 if session_id == "sess_window" else 0,
        )
        deleted = []
        monkeypatch.setattr(
            api_server._local_store,
            "delete_local_message_range",
            lambda session_id, turn_index: deleted.append(
                (session_id, turn_index)
            ) or 2,
        )
        monkeypatch.setattr(
            api_server._chat_context,
            "_load_history",
            lambda session_id: (
                list(reloaded_window) if session_id == "sess_window" else []
            ),
        )

        response = client.delete("/api/sessions/sess_window/turns/0")

        assert response.status_code == 200
        assert response.json()["remaining_turns"] == 109
        assert deleted == [("sess_window", 0)]
        assert api_server._chat_context.histories["sess_window"] == reloaded_window
        assert api_server._chat_context.histories["sess_window"][0]["content"] == "18"


class TestCommitFunction:
    """真实 _commit_interactive_history 的本地存储分支。"""

    def test_local_store_branch(self, monkeypatch):
        saved = []

        fake_store = SimpleNamespace(
            get_local_save_history=lambda: True,
            load_local_conversation=lambda _session_id: [],
            save_local_conversation_turn=(
                lambda sid, user, assistant, metrics=None, **kwargs:
                saved.append((sid, user, assistant, metrics)) or True
            ),
        )
        monkeypatch.setattr(api_server, "_local_store", fake_store)
        monkeypatch.setattr(
            api_server, "model_host",
            SimpleNamespace(),
        )
        committed = api_server._commit_interactive_history(
            "sess_l", "问", "答", {"engine": "llama_cpp"},
        )
        assert committed is True
        assert saved == [("sess_l", "问", "答", {"engine": "llama_cpp"})]


def test_full_stream_flushes_before_context_and_generation(interactive_env, monkeypatch):
    """A proxy can flush SSE headers even while context/native work is blocked."""
    import asyncio
    from starlette.requests import Request

    _client, calls = interactive_env
    acquired = []
    entered = threading.Event()
    release = threading.Event()
    acquire = api_server._chat_context.acquire_transaction

    def capture_acquire(**kwargs):
        acquired.append(True)
        return acquire(**kwargs)

    def blocked_execute(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3.0), "stream consumer never released execution"
        return {"content": "2", "followups": [], "metrics": {"model_generation": 1791914962549556600}}

    monkeypatch.setattr(api_server._chat_context, "acquire_transaction", capture_acquire)
    monkeypatch.setattr(api_server, "_execute_requested_chat", blocked_execute)

    async def consume():
        api_server._request_id_ctx.set("full-flush-contract")
        request = Request({"type": "http", "method": "POST", "path": "/api/chat/stream",
                           "headers": [], "client": ("127.0.0.1", 1234), "scheme": "http"})
        response = await api_server.chat_stream(
            api_server.ChatRequest(message="one plus one", streaming_mode="full", routing_preference="local_only"),
            request,
        )
        iterator = response.body_iterator
        first = await asyncio.wait_for(anext(iterator), timeout=0.5)
        assert first == ": processing\n\n"
        assert not acquired and not entered.is_set()
        pending = asyncio.create_task(anext(iterator))
        try:
            assert await asyncio.to_thread(entered.wait, 1.0)
            assert not pending.done(), "completion arrived while generation was blocked"
            release.set()
            final = await asyncio.wait_for(pending, timeout=2.0)
            assert json.loads(final.removeprefix("data: ")) == {
                "done": True, "response": "2", "thinking_content": None,
                "followups": [], "metrics": {"model_generation": 1791914962549556600}, "request_id": "full-flush-contract",
            }
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
        finally:
            release.set()
            if not pending.done():
                await pending
            await iterator.aclose()

    asyncio.run(consume())
    assert acquired == [True]
    assert not calls["commits"]


def test_full_stream_disconnect_after_first_comment_retires_generation(interactive_env, monkeypatch):
    import asyncio
    from starlette.requests import Request

    _client, calls = interactive_env
    monkeypatch.setattr(api_server._chat_context, "acquire_transaction",
                        lambda **_kwargs: pytest.fail("disconnected stream acquired a context transaction"))
    monkeypatch.setattr(api_server, "_execute_requested_chat",
                        lambda *_args, **_kwargs: pytest.fail("disconnected stream started generation"))

    async def disconnect():
        request = Request({"type": "http", "method": "POST", "path": "/api/chat/stream",
                           "headers": [], "client": ("127.0.0.1", 1234), "scheme": "http"})
        response = await api_server.chat_stream(
            api_server.ChatRequest(message="hi", streaming_mode="full", routing_preference="local_only"), request)
        generation = response.headers["x-generation-id"]
        cancel = api_server._generation_cancel_events[generation]
        assert await anext(response.body_iterator) == ": processing\n\n"
        await response.body_iterator.aclose()
        assert cancel.is_set()
        assert generation not in api_server._generation_cancel_events

    asyncio.run(disconnect())
    assert not calls["commits"]
