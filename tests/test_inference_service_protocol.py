"""inference-svc 契约测试（微服务架构改造计划 §1.1 / §4.1）。

锁定 /v1/* 契约：端点存在性、协议字段校验、错误码、SSE 事件格式
（对齐 api_server /api/chat/stream）、KV 生命周期、张量 roundtrip。
全部测试不加载真实模型（FakeEngineHost 注入；KVHost 用真实实现，
PagedKVCache 为纯 torch 轻量对象）。

并行共存：本测试只 import inference_service 包，不触碰 api_server。
"""
import base64
import json
import os
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api_errors import install_http_error_handler
from inference_service.engine_host import ChatGenerationCancelled, EngineHost
from inference_service.kv_host import KVHost
from inference_service.protocol import ChatRequest, LoadModelRequest
from inference_service.routes import router
from inference_service.tensor_transport import deserialize_tensor, serialize_tensor


@pytest.fixture(autouse=True)
def isolated_engine_local_store(monkeypatch):
    """Keep EngineHost protocol tests away from the user's main-node SQLite."""
    import inference_service.engine_host as engine_host_module

    messages = {}
    titles = {}

    def save_message(session_id, role, content, metrics=None):
        messages.setdefault(session_id, []).append({
            "role": role,
            "content": content,
            **({"metrics": metrics} if metrics is not None else {}),
        })

    def save_turn(session_id, user_message, assistant_message, metrics=None,
                  *, operation_id=None):
        save_message(session_id, "user", user_message)
        save_message(session_id, "assistant", assistant_message, metrics)
        return True

    fake_store = SimpleNamespace(
        initialize_local_store=lambda: "memory://protocol-test",
        get_local_save_history=lambda: False,
        save_local_message=save_message,
        save_local_conversation_turn=save_turn,
        increment_local_session_message_count=lambda _sid: None,
        load_local_conversation=lambda sid: list(messages.get(sid, [])),
        update_local_session_title=lambda sid, title: titles.__setitem__(sid, title),
    )
    monkeypatch.setattr(engine_host_module, "_local_store", fake_store)
    return fake_store


class FakeModel:
    """假模型宿主：有 chat / active_model_id，行为可断言。"""

    active_model_id = "qwen-1.8b"
    _engine_type = "llama_cpp"
    model_loaded = True
    is_loaded = True
    generation_config = {
        "max_new_tokens": 1024,
        "tier_max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    def __init__(self):
        self.calls = []
        self.layer_load_calls = []
        self.pipeline_prepare_calls = []
        self.candidate_calls = []
        self._candidate_prepared = {}

    def chat(self, messages, max_tokens=None, temperature=None, top_p=None,
             **_kw):
        self.calls.append(
            (list(messages), max_tokens, temperature, top_p)
        )
        return {
            "content": "候选答案内容",
            "usage": {"total_tokens": 8},
            "tokens_per_second": 12.5,
            "model": "qwen-1.8b",
        }

    def load_layer_range(self, start_layer=0, end_layer=24,
                         has_embedding=False, has_lm_head=False,
                          model_path=None, quant_type=None, model_id=None,
                          profile=None, total_layers=None):
        """层段加载桩：1.2 复制完成后接入真实实现。"""
        self.layer_load_calls.append({
            "start_layer": start_layer,
            "end_layer": end_layer,
            "model_path": model_path,
            "quant_type": quant_type,
            "model_id": model_id,
            "profile": profile,
            "total_layers": total_layers,
        })
        return {"success": True, "layer_range": f"{start_layer}-{end_layer}"}

    def prepare_pipeline_model(self, **kwargs):
        self.pipeline_prepare_calls.append(dict(kwargs))
        return {"pipeline_runtime_supported": True, **kwargs}

    def prepare_pipeline_candidate(self, transaction_id, **kwargs):
        self.candidate_calls.append(("prepare", transaction_id, dict(kwargs)))
        self.pipeline_prepare_calls.append(dict(kwargs))
        self._candidate_prepared[transaction_id] = dict(kwargs)
        return {"pipeline_runtime_supported": True, **kwargs}

    def materialize_pipeline_candidate(self, transaction_id, **kwargs):
        self.candidate_calls.append(
            ("materialize", transaction_id, dict(kwargs))
        )
        prepared = self._candidate_prepared[transaction_id]
        return self.load_layer_range(
            model_path=prepared.get("model_path"),
            quant_type=prepared.get("quant_type"),
            **kwargs,
        )

    def commit_pipeline_candidate(self, transaction_id):
        self.candidate_calls.append(("commit", transaction_id))

    def finalize_pipeline_candidate(self, transaction_id):
        self.candidate_calls.append(("finalize", transaction_id))

    def abort_pipeline_candidate(self, transaction_id):
        self.candidate_calls.append(("abort", transaction_id))


class FakeEngineHost(EngineHost):
    """轻量假宿主：注入 FakeModel 替代真实 ModelHost（不触发 model_module）。"""

    def __init__(self):
        super().__init__()
        self._host = FakeModel()  # 替换真实 ModelHost

    def load_model(
        self,
        engine=None,
        quant_type=None,
        use_compile=False,
        model_id=None,
        model_path=None,
        resolution=None,
    ):
        return {"success": True, "engine": engine or "pytorch", "model_id": model_id}

    def unload_model(self):
        return {"success": True, "message": "模型已卸载"}

    def switch_model(self, model_id, engine=None, quant_type=None, model_path=None,
                     resolution=None):
        return {"success": True, "model_id": model_id, "engine": engine}

    def current_model(self):
        # 对齐真实 EngineHost.current_model 完整字段（10 字段 loaded 形状）
        return {
            "loaded": True,
            "model_id": "qwen-1.8b",
            "quant_type": "int4",
            "model_name": "Qwen-1.8B-Chat",
            "model_path": "models/qwen-1_8b-chat",
            "engine": "pytorch",
            "total_params": "N/A",
            "device": "cpu",
            "gpu_allocated_gb": 0,
            "gpu_reserved_gb": 0,
        }

    def list_local_model_assets(self):
        return {
            "assets": [{"model_id": "fixture-local"}],
            "summary": {"total": 1, "total_bytes": 1},
        }

    def preflight_local_model_asset(self, model_id):
        return {"model_id": model_id, "gate_passed": True, "status": "ready_for_qwen3_smoke"}

    def chat_full(self, req, cancel_event=None):
        # 返回结构与真实 EngineHost.chat_full 一致（content 而非 response，
        # 见 engine_host.py:678）：routes 取 result["content"]
        return {
            "content": "你好，我是 QLH。",
            "thinking_content": None,
            "followups": [],
            "metrics": {"tokens_per_second": 42.0},
        }

    def chat_stream_events(self, req, cancel_event):
        for chunk in ("你", "好"):
            if cancel_event is not None and cancel_event.is_set():
                break
            yield {"token": chunk}
        yield {
            "done": True,
            "response": "你好",
            "followups": [],
            "metrics": {},
            "request_id": "-",
        }

    def forward_layers(self, layer_range, hidden, past_key_values=None, **kw):
        return hidden  # identity：验证张量 roundtrip


def make_app(engine_host=None, kv_host=None) -> FastAPI:
    app = FastAPI()
    app.state.engine_host = engine_host if engine_host is not None else FakeEngineHost()
    app.state.kv_host = kv_host if kv_host is not None else KVHost()
    install_http_error_handler(app)
    app.include_router(router)
    return app


@pytest.fixture()
def client():
    return TestClient(make_app())


# ----------------------------------------------------------------------
# 1. 契约：端点存在性（§4.1 全清单）
# ----------------------------------------------------------------------
EXPECTED_ENDPOINTS = [
    ("GET", "/v1/health"),
    ("GET", "/v1/ready"),
    ("GET", "/v1/status"),
    ("POST", "/v1/models/load"),
    ("POST", "/v1/models/unload"),
    ("POST", "/v1/models/switch"),
    ("GET", "/v1/models/current"),
    ("GET", "/v1/models/local-assets"),
    ("POST", "/v1/models/local-assets/{model_id}/preflight"),
    ("POST", "/v1/chat"),
    ("POST", "/v1/chat/stream"),
    ("POST", "/v1/chat/cancel"),
    ("POST", "/v1/speculative/run"),
    ("POST", "/v1/layers/load"),
    ("POST", "/v1/layers/unload"),
    ("POST", "/v1/layers/forward"),
    ("POST", "/v1/layers/embedding"),
    ("POST", "/v1/layers/lm_head"),
    ("POST", "/v1/kv/init"),
    ("POST", "/v1/kv/free"),
]


def test_contract_endpoint_existence():
    # FastAPI 0.139 keeps included routers as lazy containers on app.routes.
    paths = {route.path for route in router.routes}
    for method, path in EXPECTED_ENDPOINTS:
        assert path in paths, f"缺少端点 {method} {path}"


def test_models_local_assets_is_read_only_inventory(client):
    response = client.get("/v1/models/local-assets")

    assert response.status_code == 200
    assert response.json() == {
        "assets": [{"model_id": "fixture-local"}],
        "summary": {"total": 1, "total_bytes": 1},
    }


def test_models_local_asset_preflight_is_explicit_and_read_only(client):
    response = client.post("/v1/models/local-assets/fixture-local/preflight")

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "fixture-local",
        "gate_passed": True,
        "status": "ready_for_qwen3_smoke",
    }


# ----------------------------------------------------------------------
# 2. 协议字段校验（pydantic）
# ----------------------------------------------------------------------
def test_protocol_chat_request_validation():
    with pytest.raises(ValidationError):
        ChatRequest(message="")  # min_length=1
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", max_new_tokens=0)  # ge=1
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", temperature=2.5)  # le=2.0
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", streaming_mode="bogus")  # pattern
    req = ChatRequest(message="hi")
    assert req.max_new_tokens == 1024
    assert req.streaming_mode == "full"


def test_protocol_load_model_request_defaults():
    req = LoadModelRequest()
    assert req.engine is None
    assert req.quant_type is None
    assert req.use_compile is False
    assert req.layer_range is None


def test_protocol_endpoint_422_on_missing_field(client):
    resp = client.post("/v1/chat", json={})
    assert resp.status_code == 422
    assert "detail" in resp.json()


# ----------------------------------------------------------------------
# 3. 存活/就绪/状态
# ----------------------------------------------------------------------
def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "inference-svc"


def test_ready(client):
    resp = client.get("/v1/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert {"ready", "model_loaded", "layers"} <= set(body.keys())
    assert isinstance(body["layers"], list)


def test_status_includes_kv_cache(client):
    resp = client.get("/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "kv_cache" in body
    assert body["kv_cache"]["task_count"] == 0


def test_uninitialized_host_returns_503():
    app = FastAPI()  # 不设置 app.state
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/v1/ready")
    assert resp.status_code == 503
    assert "detail" in resp.json()


# ----------------------------------------------------------------------
# 4. 对话 JSON / SSE 事件格式（对齐 api_server /api/chat/stream）
# ----------------------------------------------------------------------
def test_chat_json(client):
    resp = client.post("/v1/chat", json={"message": "你好"})
    assert resp.status_code == 200
    body = resp.json()
    # 真实 EngineHost.chat_full 返回 content（对齐 api_server）
    assert body["content"] == "你好，我是 QLH。"
    assert body["request_id"] == "-"
    assert body["generation_id"].startswith("gen_")
    assert "followups" in body
    assert "metrics" in body


def test_chat_stream_fast_event_format(client):
    resp = client.post(
        "/v1/chat/stream",
        json={"message": "你好", "streaming_mode": "fast"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # 纯 data 事件：data: {...}\n\n，无 event: 行
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
        else:
            assert line == "", f"SSE 中出现非 data 行: {line!r}"
    assert len(events) >= 2
    assert "token" in events[0]
    assert events[-1]["done"] is True
    assert "response" in events[-1]
    # 薄实现 done 事件的 "-" 占位被 routes 覆盖为真实 request_id/generation_id
    assert events[-1]["request_id"] == "-"  # 无 X-QLH-Request-ID 头时默认 "-"
    assert events[-1]["generation_id"].startswith("gen_")


def test_chat_stream_full_single_done(client):
    resp = client.post(
        "/v1/chat/stream",
        json={"message": "你好", "streaming_mode": "full"},
    )
    events = [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 1
    assert events[0]["done"] is True
    # 真实 chat_full 返回 content（修复前取 response 恒为空）
    assert events[0]["response"] == "你好，我是 QLH。"
    assert events[0]["thinking_content"] is None
    assert events[0]["generation_id"].startswith("gen_")


def test_chat_json_deadline_returns_504_with_stable_code():
    from request_deadline import RequestDeadlineExceeded

    class DeadlineHost(FakeEngineHost):
        def chat_full_with_operation_id(
                self, req, cancel_event, operation_id, request_deadline=None):
            del req, cancel_event, operation_id, request_deadline
            raise RequestDeadlineExceeded("request deadline exceeded")

    with TestClient(make_app(engine_host=DeadlineHost())) as deadline_client:
        resp = deadline_client.post("/v1/chat", json={"message": "deadline"})

    assert resp.status_code == 504
    assert resp.json()["error_code"] == "request_deadline_exceeded"
    assert resp.headers["X-QLH-Error-Code"] == "request_deadline_exceeded"


def test_chat_stream_deadline_sse_preserves_reason_code():
    from request_deadline import RequestDeadlineExceeded

    class DeadlineHost(FakeEngineHost):
        def chat_stream_events_with_operation_id(
                self, req, cancel_event, operation_id, request_deadline=None):
            del req, cancel_event, operation_id, request_deadline
            raise RequestDeadlineExceeded("request deadline exceeded")
            yield  # pragma: no cover - keep this method a generator

    with TestClient(make_app(engine_host=DeadlineHost())) as deadline_client:
        resp = deadline_client.post(
            "/v1/chat/stream",
            json={"message": "deadline", "streaming_mode": "fast"},
        )

    events = [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert resp.status_code == 200
    assert events == [{
        "done": True,
        "error": "request deadline exceeded",
        "request_id": "-",
        "reason_code": "request_deadline_exceeded",
    }]


def test_chat_cancel_semantics(client):
    # 格式无效 → 400（对齐 api_server generation_id 格式校验）
    resp = client.post("/v1/chat/cancel", json={"generation_id": "bad!"})
    assert resp.status_code == 400
    assert "detail" in resp.json()
    # 未知但格式合法 → 200 cancel_pending（对齐 api_server，不 404）
    resp = client.post("/v1/chat/cancel", json={"generation_id": "gen_nope_12345678"})
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "cancel_pending",
        "generation_id": "gen_nope_12345678",
    }
    # 已注册 → 200 cancel_requested
    host = client.app.state.engine_host
    gid, ev = host.register_generation("gen_test_123456")
    resp = client.post("/v1/chat/cancel", json={"generation_id": "gen_test_123456"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancel_requested"
    assert ev.is_set()
    host.unregister_generation(gid)


def test_chat_cancel_during_stream(monkeypatch):
    """生成期间 cancel 可命中（threadpool/线程桥接后事件循环不被阻塞）。

    修复前：async 端点内同步迭代生成器，流式生成期间 /v1/chat/cancel
    得不到处理（事件循环被占），取消功能实际失效。"""
    import time as _time

    host = FakeEngineHost()
    app = make_app(engine_host=host)
    # TestClient 单 portal 不支持并发：流式与 cancel 各用独立 TestClient，
    # 共享同一 app 实例（app.state.engine_host 是同一个 host）
    stream_client = TestClient(app)
    cancel_client = TestClient(app)

    def _slow_stream(self, req, cancel_event):
        for i in range(50):
            if cancel_event is not None and cancel_event.is_set():
                yield {"token": "[cancelled]"}
                return
            yield {"token": f"t{i}"}
            _time.sleep(0.05)  # 模拟慢生成：期间事件循环必须仍可服务 cancel
        yield {"done": True, "response": "ok", "followups": [], "metrics": {}}

    # FakeEngineHost 自带 chat_stream_events 覆盖基类，patch 必须作用于子类
    monkeypatch.setattr(FakeEngineHost, "chat_stream_events", _slow_stream)

    tokens = []
    statuses = []
    cancelled_ok = []
    errors = []

    def _consumer():
        try:
            with stream_client.stream(
                "POST",
                "/v1/chat/stream",
                json={"message": "hi", "streaming_mode": "fast"},
            ) as resp:
                statuses.append(resp.status_code)
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        ev = json.loads(line[6:])
                        tokens.append(ev)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t = threading.Thread(target=_consumer, daemon=True)
    t.start()
    # 等生成开始（已注册 gid）：轮询而非固定 sleep（TestClient 启动有开销）
    gids = []
    deadline = _time.time() + 5.0
    while _time.time() < deadline:
        gids = host.generation_status()["active_generation_ids"]
        if gids:
            break
        _time.sleep(0.02)
    assert len(gids) == 1, (
        f"流式期间应有 1 个已注册 generation，实际: {gids}；"
        f"consumer 线程存活: {t.is_alive()}；statuses: {statuses}；"
        f"tokens: {tokens}；errors: {errors}"
    )
    resp = cancel_client.post("/v1/chat/cancel", json={"generation_id": gids[0]})
    cancelled_ok.append(resp.status_code)
    t.join(timeout=10)
    assert not errors
    assert cancelled_ok == [200], f"cancel 应 200，实际 {cancelled_ok}"
    # 流被 cancel 中断：出现 [cancelled] 且没有正常 done
    assert any(ev.get("token") == "[cancelled]" for ev in tokens)
    assert not any(ev.get("done") for ev in tokens)
    # 注册表已清空（无泄漏）
    assert host.generation_status()["active_generation_count"] == 0


# ----------------------------------------------------------------------
# 5. KV 生命周期（真实 KVHost）
# ----------------------------------------------------------------------
def test_kv_init_reuse_free(client):
    r = client.post("/v1/kv/init", json={"task_id": "t1"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "t1"
    assert r.json()["reused"] is False
    assert r.json()["cold_enabled"] is False

    r = client.post("/v1/kv/init", json={"task_id": "t1"})  # 幂等复用
    assert r.json()["reused"] is True

    r = client.post("/v1/kv/free", json={"task_id": "t1"})
    assert r.json()["freed"] is True

    r = client.post("/v1/kv/free", json={"task_id": "t1"})  # 已释放 → 404
    assert r.status_code == 404

    r = client.get("/v1/status")
    assert r.json()["kv_cache"]["task_count"] == 0


def test_kv_auto_task_id(client):
    r = client.post("/v1/kv/init", json={})
    assert r.status_code == 200
    assert r.json()["task_id"]


def test_kv_init_accepts_cold_tier_configuration(client, tmp_path):
    managed_root = tmp_path / "managed-kv"
    app = make_app(kv_host=KVHost(cold_cache_root=managed_root))
    with TestClient(app) as managed_client:
        rejected = managed_client.post(
            "/v1/kv/init",
            json={
                "task_id": "caller-path-task",
                "cold_cache_dir": str(tmp_path / "caller-controlled"),
                "cold_max_pages": 1,
            },
        )
        assert rejected.status_code == 400
        assert not (tmp_path / "caller-controlled").exists()

        r = managed_client.post(
            "/v1/kv/init",
            json={
                "task_id": "cold-api-task",
                "device": "cpu",
                "page_size": 4,
                "max_pages": 1,
                "cold_max_pages": 1,
                "cache_unit_size": 2,
            },
        )
        assert r.status_code == 200
        assert r.json()["cold_enabled"] is True
        assert r.json()["cache_unit_size"] == 2
        assert list(managed_root.glob("session-*"))

        status = managed_client.get("/v1/status").json()["kv_cache"]

    task = next(item for item in status["tasks"] if item["task_id"] == "cold-api-task")
    assert task["cold_enabled"] is True
    assert task["cold_pages"] == 0


def test_kv_endpoints_reject_untrusted_remote(tmp_path):
    app = make_app(kv_host=KVHost(cold_cache_root=tmp_path / "managed-kv"))
    with TestClient(app, client=("10.0.0.2", 12345)) as remote_client:
        assert remote_client.post("/v1/kv/init", json={}).status_code == 403
        assert remote_client.post("/v1/kv/free", json={"task_id": "missing"}).status_code == 403
    assert not (tmp_path / "managed-kv").exists()


def test_kv_free_reports_cleanup_failure_and_preserves_handle(tmp_path, monkeypatch):
    kv_host = KVHost(cold_cache_root=tmp_path / "managed-kv")
    app = make_app(kv_host=kv_host)
    kv_host.init(task_id="cleanup-api-task")
    cache = kv_host.get("cleanup-api-task")

    def fail_close():
        raise OSError("cleanup unavailable")

    monkeypatch.setattr(cache, "close", fail_close)
    with TestClient(app) as local_client:
        failed = local_client.post("/v1/kv/free", json={"task_id": "cleanup-api-task"})
        assert failed.status_code == 500
        assert failed.json()["detail"]["code"] == "KV_CLEANUP_FAILED"
    assert kv_host.get("cleanup-api-task") is cache


def test_kv_init_uses_server_managed_directory_without_caller_path(client):
    r = client.post("/v1/kv/init", json={"task_id": "managed-cold-task", "cold_max_pages": 1})
    assert r.status_code == 200
    assert r.json()["cold_enabled"] is True


def test_kv_init_rejects_unaligned_cache_unit(client):
    r = client.post(
        "/v1/kv/init",
        json={"page_size": 4, "max_pages": 1, "cache_unit_size": 3},
    )
    assert r.status_code == 400


# ----------------------------------------------------------------------
# 6. 层段张量传输 roundtrip（tensor_transport）
# ----------------------------------------------------------------------
def test_tensor_transport_roundtrip():
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    blob = serialize_tensor(hidden)
    restored = deserialize_tensor(blob)
    torch.testing.assert_close(restored, hidden)


def test_layers_forward_roundtrip(client):
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    tensor_ref = base64.b64encode(serialize_tensor(hidden)).decode("ascii")
    resp = client.post(
        "/v1/layers/forward",
        json={"layer_range": "0-12", "tensor_ref": tensor_ref},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "output_ref" in body
    out = deserialize_tensor(base64.b64decode(body["output_ref"]))
    torch.testing.assert_close(out, hidden)


def test_layers_forward_bad_tensor_ref(client):
    resp = client.post(
        "/v1/layers/forward",
        json={"layer_range": "0-12", "tensor_ref": "!!!not-base64!!!"},
    )
    assert resp.status_code in (400, 500)  # base64 失败或反序列化失败均被捕获
    assert "detail" in resp.json()


def test_layers_forward_unknown_kv_ref(client):
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    tensor_ref = base64.b64encode(serialize_tensor(hidden)).decode("ascii")
    resp = client.post(
        "/v1/layers/forward",
        json={
            "layer_range": "0-12",
            "tensor_ref": tensor_ref,
            "past_key_values_ref": "task_missing",
        },
    )
    assert resp.status_code == 404


# ----------------------------------------------------------------------
# ★ #31 M6：hybrid 在 /v1/layers/forward 上**明确 fail-closed**
# ----------------------------------------------------------------------
class _ConfigStub:
    def __init__(self, layer_types):
        self.layer_types = layer_types


class _ModelStub:
    def __init__(self, layer_types):
        self.config = _ConfigStub(layer_types)


def _app_with_layer_types(layer_types):
    """造 app 并把 engine_host 底下的 `_host.model` 换成带 `layer_types` 的桩。"""
    app = make_app()
    app.state.engine_host._host.model = (
        None if layer_types is None else _ModelStub(layer_types)
    )
    return app


def test_layers_forward_rejects_hybrid_with_explicit_fail_closed():
    """★ M6：hybrid 的 recurrent state **不在** KV 分页里 ⇒ 必须**明确拒绝**（501）。

    否则会拿 `PagedKVCache` 当载体"看起来能跑" —— 那正是最坏的一种：同一轮不报错、
    数值却错（静默丢状态），事后极难归因。
    """
    client = TestClient(_app_with_layer_types(["linear_attention", "full_attention"]))
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    tensor_ref = base64.b64encode(serialize_tensor(hidden)).decode("ascii")

    resp = client.post(
        "/v1/layers/forward",
        json={"layer_range": "0-12", "tensor_ref": tensor_ref},
    )

    assert resp.status_code == 501
    detail = resp.json()["detail"]
    assert "混合层型" in detail or "recurrent" in detail


def test_layers_forward_pure_full_attention_is_not_gated():
    """★ 回归：纯 `full_attention`（含 transformers 5.x 给 Qwen2 也加 `layer_types` 的形态）
    **不受**这条闸门影响 —— 必须能透过闸门（FakeModel 无 `forward_layers`，故只断言非 501）。"""
    for layer_types in (["full_attention"] * 4, None):
        client = TestClient(_app_with_layer_types(layer_types))
        hidden = torch.randn(2, 2048, dtype=torch.float16)
        tensor_ref = base64.b64encode(serialize_tensor(hidden)).decode("ascii")

        resp = client.post(
            "/v1/layers/forward",
            json={"layer_range": "0-12", "tensor_ref": tensor_ref},
        )

        assert resp.status_code != 501, layer_types


def test_kv_state_kind_uses_layer_types_not_model_type_allowlist():
    """★ 判据是**层类型**：含 `linear_attention` 才算 hybrid；无 `layer_types` ⇒ 老架构。"""
    host = EngineHost.__new__(EngineHost)      # 不跑 __init__（避免拉起真实 ModelHost）

    class _Holder:
        model = None

    host._host = _Holder()
    assert host.kv_state_kind() == "unknown"   # 没模型 ⇒ unknown（**不**误判成 hybrid/tuple）

    host._host.model = _ModelStub(["linear_attention", "full_attention"])
    assert host.kv_state_kind() == "hybrid"

    host._host.model = _ModelStub(["full_attention"] * 24)
    assert host.kv_state_kind() == "tuple"

    host._host.model = _ModelStub(None)
    assert host.kv_state_kind() == "tuple"


# ----------------------------------------------------------------------
# 7. 实验端点门控
# ----------------------------------------------------------------------
def test_speculative_run_gated_off(client):
    """SPEC_ENABLED 未启用（默认）→ 404 门控（复制 api_server 语义）。"""
    resp = client.post(
        "/v1/speculative/run", json={"message": "你好"}
    )
    assert resp.status_code == 404
    assert "detail" in resp.json()


def test_speculative_run_enabled_path(client, monkeypatch):
    """SPEC_ENABLED=true 时启用路径：真实 SpeculativeRunRequest 全字段
    透传到 host.speculative_run（修复前协议缺字段 → AttributeError 500）。"""
    import config as _cfg

    monkeypatch.setattr(_cfg, "SPEC_ENABLED", True)

    captured = {}

    def _fake_speculative_run(self, req):
        captured["message"] = req.message
        captured["gamma"] = req.gamma
        captured["max_rounds"] = req.max_rounds
        captured["seed"] = req.seed
        captured["draft_hint"] = req.draft_hint
        captured["allow_external"] = req.allow_external
        captured["max_new_tokens"] = req.max_new_tokens
        return {
            "content": "草稿验证结果",
            "finish_reason": "stop",
            "metrics": {},
            "rounds": 1,
        }

    from inference_service.engine_host import EngineHost

    monkeypatch.setattr(EngineHost, "speculative_run", _fake_speculative_run)
    resp = client.post(
        "/v1/speculative/run",
        json={
            "message": "你好",
            "gamma": 4,
            "max_rounds": 2,
            "seed": 42,
            "draft_hint": "你好，",
            "allow_external": True,
            "max_new_tokens": 32,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "草稿验证结果"
    assert captured == {
        "message": "你好",
        "gamma": 4,
        "max_rounds": 2,
        "seed": 42,
        "draft_hint": "你好，",
        "allow_external": True,
        "max_new_tokens": 32,
    }


# ----------------------------------------------------------------------
# 8. 层段加载/卸载
# ----------------------------------------------------------------------
def test_layers_load_unload(client):
    resp = client.post(
        "/v1/layers/load",
        json={"layer_range": "0-12", "embed": True, "lm_head": False},
    )
    assert resp.status_code == 200
    r = client.get("/v1/status")
    assert "0-12" in r.json()["layers"]
    resp = client.post("/v1/layers/unload", json={"layer_range": "0-12"})
    assert resp.status_code == 200
    r = client.get("/v1/status")
    assert "0-12" not in r.json()["layers"]


# ----------------------------------------------------------------------
# 9. 1.2a 数据面执行段：task-worker Stage（复制自 api_server
#    _execute_task_worker_stage；宿主适配 model_manager → self._host）
# ----------------------------------------------------------------------
from task_provider import StageRequest as ProviderStageRequest
from task_graph import TaskGraphError


def _stage(stage_type, stage_id, root_input, dependencies=None):
    return ProviderStageRequest(
        workflow_id="w1",
        request_id="r1",
        stage_id=stage_id,
        stage_type=stage_type,
        provider_id="local-full-model",
        dependencies=dependencies or {},
        root_input=root_input,
    )


def test_worker_stage_full_inference():
    host = FakeEngineHost()
    req = _stage(
        "full_inference", "candidate_a",
        {"messages": [{"role": "user", "content": "1+1=?"}],
         "task_options": {"candidate_max_tokens": 128, "temperature": 0.5}},
    )
    result = host.execute_task_worker_stage(req, threading.Event())
    assert result["content"] == "候选答案内容"
    assert result["tokens_per_second"] == 12.5
    assert result["model"] == "qwen-1.8b"
    # 系统提示注入 + 用户消息
    messages = host._host.calls[0][0]
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "1+1=?"
    assert host._host.calls[0][1] == 128  # candidate_budget
    assert host._host.calls[0][2] == 0.5  # temperature


def test_worker_stage_aggregate():
    host = FakeEngineHost()
    req = _stage(
        "aggregate", "aggregate",
        {"message": "原始问题", "task_options": {"final_max_tokens": 256}},
        dependencies={
            "candidate_a": {"content": "候选一"},
            "candidate_b": {"content": "候选二"},
        },
    )
    result = host.execute_task_worker_stage(req, threading.Event())
    assert result["content"] == "候选答案内容"
    messages = host._host.calls[0][0]
    assert "候选一" in messages[-1]["content"]
    assert "候选二" in messages[-1]["content"]


def test_worker_stage_unsupported_type():
    host = FakeEngineHost()
    req = _stage("bogus", "x", {"task_options": {}})
    with pytest.raises(TaskGraphError):
        host.execute_task_worker_stage(req, threading.Event())


def test_worker_stage_invalid_options():
    host = FakeEngineHost()
    req = _stage(
        "full_inference", "candidate_a",
        {"messages": [], "task_options": {"temperature": "不是数字"}},
    )
    with pytest.raises(TaskGraphError):
        host.execute_task_worker_stage(req, threading.Event())


def test_worker_stage_cancel_stops_inference():
    host = FakeEngineHost()
    cancel = threading.Event()
    cancel.set()  # 已取消 → run_model 直接返回空
    req = _stage(
        "full_inference", "candidate_a",
        {"messages": [{"role": "user", "content": "hi"}],
         "task_options": {}},
    )
    result = host.execute_task_worker_stage(req, cancel)
    assert result["content"] == ""
    assert host._host.calls == []  # 未触发模型调用


def test_worker_stage_route_generation_no_leak(monkeypatch):
    """路由级：/v1/worker/stage 结束时 generation 注册表无泄漏。
    （修复前：req.request_id 默认空时 finally 用 header 默认值 "-" 注销，
    注册表条目永不释放，且该 generation 永远无法 cancel。）"""
    from inference_service.engine_host import EngineHost

    host = FakeEngineHost()
    app = make_app(engine_host=host)
    client = TestClient(app)
    captured = {}

    def _fake_execute(self, stage_request, provider_cancel_event):
        captured["cancel_event"] = provider_cancel_event
        captured["stage_type"] = stage_request.stage_type
        return {"success": True, "content": "ok"}

    monkeypatch.setattr(EngineHost, "execute_task_worker_stage", _fake_execute)
    resp = client.post(
        "/v1/worker/stage",
        json={
            "workflow_id": "wf-1",
            "request_id": "",  # 空 request_id：注册表用自动生成 gid
            "stage_id": "s1",
            "stage_type": "candidate_a",
            "provider_id": "local",
            "dependencies": {},
            "root_input": {"task_options": {}},
            "model_identity": None,
            "runtime_context": {},
        },
    )
    assert resp.status_code == 200
    assert captured["stage_type"] == "candidate_a"
    # 请求结束后注册表必须清空（无泄漏）
    assert host.generation_status()["active_generation_count"] == 0
    # 且注册期间 cancel 可命中（用注册时的 gid）
    assert captured["cancel_event"] is not None


# ----------------------------------------------------------------------
# 10. 1.2a KV from_profile（复制自 api_server._init_kv_cache 自适应逻辑）
# ----------------------------------------------------------------------
def test_kv_from_profile_tier_sizing():
    kv = KVHost()
    edge = kv.from_profile(profile={"tier": "edge", "gpu": {}})
    assert edge["reused"] is False
    edge_pages = edge["max_pages"]

    kv2 = KVHost()
    workstation = kv2.from_profile(profile={"tier": "workstation", "gpu": {}})
    assert workstation["max_pages"] > edge_pages  # 工作站档位更大


def test_kv_from_profile_with_model_heads():
    kv = KVHost()
    r = kv.from_profile(profile={"tier": "laptop", "gpu": {}},
                        num_heads=16, head_dim=64)
    assert r["total_pages"] == 0  # 初始未分配页
    assert r["max_pages"] > 0


# ----------------------------------------------------------------------
# 11. 1.2a 辅助纯函数行为（复制自 api_server，等价性由一次性对比脚本验证）
# ----------------------------------------------------------------------
def test_aux_strip_thinking_tags():
    from inference_service.engine_host import _strip_native_thinking_tags as strip
    assert strip("答案<think>思考</think>") == "答案"
    assert strip("<think>仅思考</think>") == ""
    assert strip("a\n\n\n\nb") == "a\n\nb"
    assert strip("") == ""


def test_aux_parse_thinking_response():
    from inference_service.engine_host import _parse_thinking_response as parse
    answer, thinking = parse("【思考】推理过程\n【思考结束】最终答案")
    assert thinking == "推理过程"
    assert answer == "最终答案"
    answer, thinking = parse("无标记直接回答")
    assert answer == "无标记直接回答"
    assert thinking is None


# ----------------------------------------------------------------------
# 12. 1.2b 追问生成（复制自 api_server._generate_followups_llama +
#     _is_question / _fallback_followups）
# ----------------------------------------------------------------------
def test_followups_history_too_short():
    host = FakeEngineHost()
    assert host.generate_followups_llama([]) == []
    assert host.generate_followups_llama([{"role": "user", "content": "hi"}]) == []


def test_followups_cancel_event():
    host = FakeEngineHost()
    cancel = threading.Event()
    cancel.set()
    history = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩技术。"},
    ]
    assert host.generate_followups_llama(history, cancel) == []


def test_followups_model_generates_valid_questions():
    host = FakeEngineHost()

    class ChatModel(FakeModel):
        def chat(self, messages, **_kw):
            return {"content": "1. 量化有哪些方法？\n2. INT8量化相比INT4有哪些优势？\n3. 陈述句不是问题"}

    host._host = ChatModel()
    history = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩技术。"},
    ]
    questions = host.generate_followups_llama(history)
    assert "量化有哪些方法？" in questions
    assert "INT8量化相比INT4有哪些优势？" in questions
    assert "陈述句不是问题" not in questions
    assert len(questions) <= 3


def test_followups_fallback_on_garbage():
    host = FakeEngineHost()

    class GarbageModel(FakeModel):
        def chat(self, messages, **_kw):
            return {"content": "完全不相关的内容没有问号"}

    host._host = GarbageModel()
    history = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩技术。"},
    ]
    questions = host.generate_followups_llama(history)
    assert len(questions) >= 2  # 关键词模板兜底
    assert any("量化" in q or "模型" in q for q in questions)


# ----------------------------------------------------------------------
# 13. P2-B Q14：EngineHost 关键实现的真实执行覆盖
# ----------------------------------------------------------------------
def test_engine_host_initializes_profiled_kv_cache_from_model_config(monkeypatch):
    from paged_kv_cache import PagedKVCache

    host = FakeEngineHost()
    host._host = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(
            num_attention_heads=12,
            hidden_size=768,
        )),
        get_device=lambda: torch.device("cpu"),
    )
    host._device_profile = {"tier": "edge", "gpu": {}}
    monkeypatch.setattr(
        host, "_ensure_device_profile", lambda: host._device_profile,
    )

    original_from_profile = PagedKVCache.from_profile
    captured = []

    def observe_from_profile(cls, **kwargs):
        captured.append(kwargs)
        return original_from_profile(**kwargs)

    monkeypatch.setattr(
        PagedKVCache, "from_profile", classmethod(observe_from_profile),
    )

    cache = host._init_kv_cache()

    assert cache.page_size == 64
    assert cache.max_pages == 64
    assert cache.device == "cpu"
    assert captured == [{
        "profile": {"tier": "edge", "gpu": {}},
        "device": "cpu",
        "dtype": torch.float16,
        "num_heads": 12,
        "head_dim": 64,
    }]


def test_engine_host_auto_load_selects_sorted_gguf_candidate(tmp_path, monkeypatch):
    import config as cfg

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    selected = models_dir / "a-first.gguf"
    selected.write_bytes(b"test-gguf")
    (models_dir / "z-second.gguf").write_bytes(b"test-gguf")

    class RecordingModelHost:
        def __init__(self):
            self.full_chat_execution_lock = threading.RLock()
            self.model = None
            self.model_loaded = False
            self.quant_type = None
            self.calls = []

        @staticmethod
        def get_device():
            return torch.device("cpu")

        def load_model(self, **kwargs):
            self.calls.append(kwargs)
            self.quant_type = kwargs["quant_type"]

    host = FakeEngineHost()
    recorder = RecordingModelHost()
    host._host = recorder
    host._device_profile = {"tier": "edge", "gpu": {}}
    monkeypatch.setattr(
        host, "_ensure_device_profile", lambda: host._device_profile,
    )
    monkeypatch.setattr(cfg, "GGUF_MODEL_PATH", str(models_dir / "missing.gguf"))
    monkeypatch.setattr(cfg, "MODEL_PATH", str(tmp_path / "missing-pytorch"))
    # ★ 2026-09-19：`_auto_load_default_model` 现在优先按**设备画像**解析默认模型路径
    #   （用户裁定：边缘 <1B / PC ~2B）。本用例验的是「无画像模型时回退到扫目录 +
    #   **按文件名排序**取第一个」，故把画像解析结果置空以保持该路径可达。
    monkeypatch.setattr(cfg, "get_active_model_paths", lambda: {})
    monkeypatch.setattr(cfg, "INFERENCE_ENGINE", "pytorch")
    monkeypatch.setattr(cfg, "QUANT_TYPE", "fp16")
    monkeypatch.setattr(cfg, "USE_COMPILE", True)

    host._auto_load_default_model()

    assert len(recorder.calls) == 1
    call = dict(recorder.calls[0])
    resolution = call.pop("resolution")
    assert call == {
        "model_path": str(selected),
        "quant_type": "gguf",
        "profile": {"tier": "edge", "gpu": {}},
        "model_id": None,
        "engine": "llama_cpp",
    }
    assert resolution.engine == "llama_cpp"
    assert resolution.model_path == str(selected)
    assert host.current_model()["loaded"] is True
    assert recorder.current_quant == "gguf"
    assert host.kv_cache_status()["page_size"] == 64


def test_engine_host_auto_load_safetensors_only_uses_pytorch_quant(tmp_path, monkeypatch):
    import config as cfg

    model_dir = tmp_path / "safetensors-model"
    model_dir.mkdir()

    class RecordingModelHost:
        def __init__(self):
            self.full_chat_execution_lock = threading.RLock()
            self.model = None
            self.model_loaded = False
            self.quant_type = None
            self.calls = []

        @staticmethod
        def get_device():
            return torch.device("cpu")

        def load_model(self, **kwargs):
            self.calls.append(kwargs)
            self.quant_type = kwargs["quant_type"]

    host = FakeEngineHost()
    recorder = RecordingModelHost()
    host._host = recorder
    host._device_profile = {"tier": "edge", "gpu": {"cuda_available": False}}
    monkeypatch.setattr(host, "_ensure_device_profile", lambda: host._device_profile)
    monkeypatch.setattr(cfg, "get_active_model_paths", lambda: {})
    monkeypatch.setattr(cfg, "GGUF_MODEL_PATH", str(tmp_path / "missing.gguf"))
    monkeypatch.setattr(cfg, "MODEL_PATH", str(model_dir))
    monkeypatch.setattr(cfg, "QUANT_TYPE", "int4")
    monkeypatch.setattr(cfg, "PREFER_PYTORCH", False)

    host._auto_load_default_model()

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["engine"] == "pytorch"
    assert call["model_path"] == str(model_dir)
    assert call["quant_type"] == "fp32"
    assert call["resolution"].runtime == "pytorch_cpu"


def test_engine_host_auto_load_never_rebinds_active_id_to_global_artifact(
    tmp_path, monkeypatch,
):
    import config as cfg

    foreign = tmp_path / "foreign.gguf"
    foreign.write_bytes(b"gguf")
    host = FakeEngineHost()
    host._device_profile = {"tier": "edge", "gpu": {"cuda_available": False}}
    monkeypatch.setattr(host, "_ensure_device_profile", lambda: host._device_profile)
    monkeypatch.setattr(
        cfg,
        "get_active_model_paths",
        lambda: {
            "model_id": "model-a",
            "gguf_path": str(tmp_path / "missing-a.gguf"),
            "model_path": str(tmp_path / "missing-a"),
        },
    )
    monkeypatch.setattr(cfg, "GGUF_MODEL_PATH", str(foreign))
    monkeypatch.setattr(cfg, "MODEL_PATH", str(tmp_path / "foreign-safetensors"))

    with pytest.raises(FileNotFoundError, match="未找到可自动加载的模型文件"):
        host._auto_load_default_model()


def test_engine_host_remote_worker_selection_enforces_feature_health_model_and_load(
        monkeypatch):
    import config as cfg

    class Provider:
        def __init__(self, provider_id, supports=True):
            self.provider_id = provider_id
            self._supports = supports

        def supports_model_identity(self, model_identity, stage_type):
            return self._supports and model_identity == "model-a" and stage_type == "full_inference"

    class Coordinator:
        def __init__(self, statuses):
            self._statuses = statuses
            self.providers = {}

        def has_provider(self, provider_id):
            return provider_id in self.providers

        def register_provider(self, provider):
            self.providers[provider.provider_id] = provider

        def provider_status(self):
            return self._statuses

    providers = [
        Provider("remote-alpha"),
        Provider("remote-beta"),
        Provider("remote-wrong-model", supports=False),
        Provider("remote-unavailable"),
    ]
    statuses = [
        {
            "provider_id": "remote-alpha",
            "provider_kind": "remote_full_worker",
            "healthy": True,
            "available": True,
            "supported_stage_types": ["full_inference"],
            "max_concurrency": 2,
            "active_reservations": 1,
        },
        {
            "provider_id": "remote-beta",
            "provider_kind": "remote_full_worker",
            "healthy": True,
            "available": True,
            "supported_stage_types": ["full_inference"],
            "max_concurrency": 2,
            "active_reservations": 0,
        },
        {
            "provider_id": "remote-wrong-model",
            "provider_kind": "remote_full_worker",
            "healthy": True,
            "available": True,
            "supported_stage_types": ["full_inference"],
            "max_concurrency": 1,
            "active_reservations": 0,
        },
        {
            "provider_id": "remote-unavailable",
            "provider_kind": "remote_full_worker",
            "healthy": False,
            "available": False,
            "supported_stage_types": ["full_inference"],
            "max_concurrency": 1,
            "active_reservations": 0,
        },
    ]
    host = FakeEngineHost()
    host._scheduler = SimpleNamespace(
        remote_task_worker_providers=lambda: providers,
    )
    coordinator = Coordinator(statuses)
    host._task_graph_coordinator = coordinator
    monkeypatch.setattr(cfg, "TASK_WORKER_EXPERIMENTAL_ENABLED", False)
    assert host._eligible_remote_task_worker_provider_ids(
        "model-a", "full_inference",
    ) == []

    monkeypatch.setattr(cfg, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    assert host._sync_remote_task_worker_providers() == [
        "remote-alpha",
        "remote-beta",
        "remote-unavailable",
        "remote-wrong-model",
    ]
    assert host._eligible_remote_task_worker_provider_ids(
        "model-a", "full_inference", limit=2,
    ) == ["remote-beta", "remote-alpha"]


def test_engine_host_pipeline_worker_reservation_requires_live_scheduler_check():
    host = FakeEngineHost()
    assert host._pipeline_worker_is_reserved() is False

    host._scheduler = SimpleNamespace(
        has_pipeline_worker_reservation=lambda: True,
    )
    assert host._pipeline_worker_is_reserved() is True

    host._scheduler = SimpleNamespace()
    assert host._pipeline_worker_is_reserved() is False


def test_followups_generation_failure_non_fatal():
    host = FakeEngineHost()

    class BrokenModel(FakeModel):
        def chat(self, messages, **_kw):
            raise RuntimeError("模型调用失败")

    host._host = BrokenModel()
    history = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩技术。"},
    ]
    questions = host.generate_followups_llama(history)
    assert len(questions) >= 2  # 异常非致命 → 模板兜底


def test_is_question_aux():
    from inference_service.engine_host import _is_question
    assert _is_question("量化有哪些方法？")
    assert _is_question("为什么？")
    assert not _is_question("量化很重要")  # 无问号
    assert not _is_question("有以下几点：")  # 陈述句式
    assert not _is_question("1. 首先")  # 列举开头
    assert not _is_question("")


# ----------------------------------------------------------------------
# 13. 1.2c 外部推理整请求路由（复制自 api_server._execute_external_chat）
# ----------------------------------------------------------------------
class FakeExternalClient:
    masked_base_url = "https://***"
    model_name = "gpt-4o-mini"

    def __init__(self):
        self.last = None

    def ensure_connected(self):
        pass

    def chat(self, history, **kw):
        self.last = (list(history), kw)
        return {
            "content": "外部回答",
            "usage": {"completion_tokens": 3},
            "tokens_per_second": 9.0,
            "model": "gpt-4o-mini",
        }


def test_external_chat_routes_and_persists(monkeypatch):
    import local_store
    import external_provider

    fake_client = FakeExternalClient()
    monkeypatch.setattr(external_provider, "get_external_chat_client", lambda: fake_client)
    monkeypatch.setattr(local_store, "get_local_save_history", lambda: False)  # 测试不落盘

    host = FakeEngineHost()
    req = ChatRequest(message="你好", allow_external=True, client_node_type="pc")
    result = host.execute_external_chat(
        req,
        [{"role": "user", "content": "之前的问题"}],
        target_session_id="s1",
    )
    assert result["content"] == "外部回答"
    assert result["metrics"]["engine"] == "external_api"
    assert result["metrics"]["request_origin"] == "pc_http"
    assert result["metrics"]["completion_tokens"] == 3
    stats = host.conversation_status()["stats"]
    assert stats["rounds"] == 1
    assert stats["total_generated_tokens"] == 3
    # 请求历史 = 原历史 + 新消息
    assert fake_client.last[0][-1] == {"role": "user", "content": "你好"}
    assert fake_client.last[1]["allow_external"] is True


def test_external_chat_strips_thinking_when_disabled(monkeypatch):
    import local_store
    import external_provider

    class ThinkingClient(FakeExternalClient):
        def chat(self, history, **kw):
            return {"content": "<think>内部</think>外部回答"}

    monkeypatch.setattr(external_provider, "get_external_chat_client", lambda: ThinkingClient())
    monkeypatch.setattr(local_store, "get_local_save_history", lambda: False)

    host = FakeEngineHost()
    req = ChatRequest(message="hi", allow_external=True, show_thinking=False)
    result = host.execute_external_chat(req, [], "s2")
    assert "内部" not in result["content"]


def test_chat_origin_aux():
    from inference_service.engine_host import _chat_origin
    assert _chat_origin(ChatRequest(message="x", client_node_type="android")) == "android_http"
    assert _chat_origin(ChatRequest(message="x", client_node_type="pc")) == "pc_http"
    assert _chat_origin(ChatRequest(message="x")) == "web_http"


def make_full_host(fake_model_cls=FakeModel):
    """真实 EngineHost + 假模型注入（测试 chat_full 复制实现本体）。"""
    host = EngineHost()
    host._host = fake_model_cls()
    return host


def test_chat_full_llama_cpp_does_not_import_torch(monkeypatch):
    """The torch-free package must still serve the direct GGUF path."""
    import builtins

    real_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    host = make_full_host()

    result = host.chat_full(ChatRequest(message="hello"))

    assert result["content"] == "候选答案内容"
    assert result["metrics"]["engine"] == "llama_cpp"


@pytest.mark.parametrize("engine", ["pytorch", "llama_cpp"])
def test_chat_full_partial_pipeline_local_only_fails_closed(engine):
    host = make_full_host()
    host._host._engine_type = engine
    host._host.is_pipeline_prepared = True

    with pytest.raises(HTTPException) as exc_info:
        host.chat_full(ChatRequest(
            message="hello",
            routing_preference="local_only",
        ))

    assert exc_info.value.status_code == 409
    assert "分布式流水线模式准备" in str(exc_info.value.detail)
    assert host._host.calls == []


def test_engine_host_pipeline_deadline_never_enters_local_fallback(monkeypatch):
    import inference_service.engine_host as engine_host_module
    from request_deadline import RequestDeadline

    calls = {"chat": 0, "ensure": 0}

    class DeadlineModel(FakeModel):
        _engine_type = "pytorch"

        def ensure_full_model(self):
            calls["ensure"] += 1
            pytest.fail("deadline terminal must not restore a full model")

        def chat(self, *_args, **_kwargs):
            calls["chat"] += 1
            pytest.fail("deadline terminal must not enter local chat")

    host = make_full_host(DeadlineModel)
    host._run_mode = "distributed"
    host._scheduler = SimpleNamespace(
        get_distributed_inference_enabled=lambda: True,
        _effective_role=lambda: "master",
        has_pipeline_worker_reservation=lambda: False,
        run_pipeline_safe=lambda *_a, **_k: {
            "response": "",
            "error": "request deadline exceeded",
            "reason_code": "request_deadline_exceeded",
            "metrics": {"fallback": False},
        },
    )
    monkeypatch.setattr(
        engine_host_module,
        "runtime_supports",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        host,
        "_prepare_chat_context",
        lambda *_args, **_kwargs: SimpleNamespace(
            session_id="default", history=[], revision=0,
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        host.chat_full(
            ChatRequest(message="deadline fence"),
            request_deadline=RequestDeadline.start(1.0),
        )

    assert exc_info.value.status_code == 504
    assert getattr(exc_info.value, "error_code", "") == (
        "request_deadline_exceeded"
    )
    assert calls == {"chat": 0, "ensure": 0}


# ----------------------------------------------------------------------
# 14.5 1.2d task_graph 执行段（复制自 api_server._execute_task_graph_chat
#      + _execute_task_graph_chat_with_slot + 5 个辅助函数）
# ----------------------------------------------------------------------
def test_task_graph_gate_off(monkeypatch):
    """TASK_GRAPH_ENABLED 默认关闭 → chat_full(task_graph) 409 门控
    （对齐 api_server 语义；不加载模型、不创建 journal）。
    显式强制关闭：node_config 的 feature-gates 持久化会把
    QLH_TASK_GRAPH_ENABLED 注入环境，测试须自包含不受用户配置影响。"""
    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", False)
    # 真实 EngineHost（FakeEngineHost 覆盖了基类 chat_full，分支不会触发）
    host = EngineHost()
    req = ChatRequest(message="hi", execution_mode="task_graph")
    with pytest.raises(Exception) as excinfo:
        host.chat_full(req)
    assert excinfo.value.status_code == 409
    assert excinfo.value.error_code == "TASK_GRAPH_DISABLED"


def test_task_graph_chat_full_dispatches(monkeypatch):
    """execution_mode=task_graph → chat_full 分派到 execute_task_graph_chat
    （对齐 api_server._execute_requested_chat）。"""
    from inference_service.engine_host import EngineHost

    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    host = EngineHost()  # 真实 EngineHost：基类 chat_full 才有 task_graph 分支
    captured = {}

    def _fake_execute(self, req, cancel_event=None):
        captured["mode"] = req.execution_mode
        return {"content": "task-graph-ok", "thinking_content": None,
                "metrics": {"execution_mode": "task_graph"}, "followups": []}

    monkeypatch.setattr(EngineHost, "execute_task_graph_chat", _fake_execute)
    req = ChatRequest(message="hi", execution_mode="task_graph")
    result = host.chat_full(req)
    assert captured["mode"] == "task_graph"
    assert result["content"] == "task-graph-ok"


def test_task_graph_slot_exclusive(monkeypatch):
    """task_graph 执行槽互斥：占用中再来 → 429（对齐 api_server 语义）。"""
    from inference_service.engine_host import EngineHost

    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    host = EngineHost()
    captured = []

    def _fake_execute(self, req, cancel_event=None):
        captured.append("entered")
        return {"content": "ok", "thinking_content": None,
                "metrics": {}, "followups": []}

    monkeypatch.setattr(EngineHost, "execute_task_graph_chat_with_slot", _fake_execute)
    # gate 的 journal 检查需 available：伪造 coordinator
    class _FakeCoordinator:
        def journal_status(self):
            return {"available": True}

    monkeypatch.setattr(EngineHost, "_ensure_task_graph_coordinator",
                        lambda self: _FakeCoordinator())
    # 先占用槽位
    assert host._task_graph_execution_slot.acquire(blocking=False) is True
    req = ChatRequest(message="hi", execution_mode="task_graph")
    with pytest.raises(Exception) as excinfo:
        host.chat_full(req)
    assert excinfo.value.status_code == 429
    assert captured == []  # 未进入执行体
    host._task_graph_execution_slot.release()


def test_task_graph_coordinator_lazy_idempotent():
    """_ensure_task_graph_coordinator 惰性创建且幂等（单实例缓存）。"""
    host = FakeEngineHost()
    c1 = host._ensure_task_graph_coordinator()
    c2 = host._ensure_task_graph_coordinator()
    assert c1 is c2


def test_task_graph_active_identity_none_without_model():
    """未加载模型 → _active_task_graph_model_identity 返回 None。"""
    host = EngineHost()  # 真实 ModelHost：model_loaded=False
    assert host._active_task_graph_model_identity() is None


def test_task_graph_active_identity_uses_prepared_pipeline(monkeypatch):
    """distributed-only 已准备模型（未物化权重）也要能给出身份。

    `prepare-pipeline` 按设计不加载权重，而 Route A 要求活动模型身份；身份从它
    inspect 出的描述符取，否则两者互斥、跨框架阶段链无法准入。
    """
    host = EngineHost()  # 真实 ModelHost：model_loaded=False
    target = host._host
    monkeypatch.setattr(target, "_pipeline_distributed_only", True, raising=False)
    monkeypatch.setattr(
        target,
        "_pipeline_descriptor",
        {
            "model_id": "qwen2.5-0.5b-instruct",
            "model_sha256": "596062f9" + "0" * 56,
        },
        raising=False,
    )
    monkeypatch.setattr(
        target, "_active_model_id", "qwen2.5-0.5b-instruct", raising=False
    )

    identity = host._active_task_graph_model_identity()

    assert identity is not None
    assert identity.model_id == "qwen2.5-0.5b-instruct"
    assert identity.engine == "pytorch"
    assert identity.format == "safetensors"
    assert identity.revision == "prepared"
    assert identity.sha256.startswith("596062f9")


def test_task_graph_with_slot_real_execution(monkeypatch):
    """真实执行 execute_task_graph_chat_with_slot 主体（不 monkeypatch
    执行体）：本地模板分支全链路——会话切换/历史维护/run_template/
    持久化/metrics。修复前此路径 6 处 NameError 必崩。"""
    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(_cfg, "TASK_WORKER_EXPERIMENTAL_ENABLED", False)

    host = make_full_host()
    host._host.full_chat_execution_lock = threading.RLock()  # FakeModel 缺锁
    host._active_session_id = "s1"
    host._session_histories["s1"] = []

    class FakeCoordinator:
        def __init__(self):
            self.providers = []

        def journal_status(self):
            return {"available": True}

        def has_provider(self, pid):
            return pid in self.providers

        def register_provider(self, provider):
            self.providers.append(provider.provider_id)

        def run_template(self, **kw):
            captured["run_template"] = kw
            return (
                {"content": "任务链答案", "thinking_content": None},
                {
                    "workflow_id": "wf-1",
                    "template": "dual_candidate",
                    "state": "completed",
                    "partial_result": False,
                    "stages": [],
                    "stage_count": 0,
                    "attempt_count": 0,
                    "duration_seconds": 0.5,
                },
            )

        def commit_result(self, workflow_id):
            return {"workflow_id": workflow_id, "stages": [],
                    "template": "dual_candidate", "state": "completed",
                    "partial_result": False, "stage_count": 0,
                    "attempt_count": 0, "duration_seconds": 0.5}

        def discard_result(self, workflow_id, **_kwargs):
            return None

        def provider_status(self):
            return []

    fake_coord = FakeCoordinator()
    captured = {}
    monkeypatch.setattr(
        EngineHost, "_ensure_task_graph_coordinator",
        lambda self: fake_coord,
    )

    req = ChatRequest(message="hi", session_id="s1", execution_mode="task_graph")
    result = host.chat_full(req)

    assert result["content"] == "任务链答案"
    assert result["metrics"]["execution_mode"] == "task_graph"
    assert result["metrics"]["workflow_id"] == "wf-1"
    # run_template 收到正确入参（root_input 带消息与历史）
    assert captured["run_template"]["root_input"]["message"] == "hi"
    assert captured["run_template"]["session_id"] == "s1"
    # 历史已追加（user + assistant）
    history = host.session_history("s1")
    assert [message["role"] for message in history] == ["user", "assistant"]
    # 本地 provider 注册成功（self._dispatch_local_task_provider 引用有效）
    assert "local_full_model" in fake_coord.providers


def test_engine_host_release_profile_discards_task_graph_required_before_commit(
    monkeypatch,
):
    """The inference service cannot revive task_graph as a release route."""
    import config as _cfg
    from task_provider import ModelIdentity

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(_cfg, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")

    host = make_full_host()
    identity = ModelIdentity(
        model_id="qwen-1.8b",
        engine="llama_cpp",
        format="gguf",
        revision="release-test",
        sha256="a" * 64,
    )
    monkeypatch.setattr(
        host, "_active_task_graph_model_identity", lambda: identity,
    )
    monkeypatch.setattr(
        host,
        "_eligible_remote_task_worker_provider_ids",
        lambda *_args, **_kwargs: ["remote_release_worker"],
    )

    class FakeCoordinator:
        def __init__(self):
            self.providers = []
            self.discarded = []

        def has_provider(self, provider_id):
            return provider_id in self.providers

        def register_provider(self, provider):
            self.providers.append(provider.provider_id)

        def run(self, **_kwargs):
            return (
                {"content": "forbidden task graph result"},
                {
                    "workflow_id": "wf_releaseenginehost01",
                    "state": "result_ready",
                    "stages": [{
                        "attempts": [{
                            "state": "completed",
                            "provider_kind": "remote_full_worker",
                            "provider_node_id": "worker-release",
                        }],
                    }],
                },
            )

        def discard_result(self, workflow_id, **kwargs):
            self.discarded.append((workflow_id, kwargs))
            return {"workflow_id": workflow_id, "state": "failed"}

        def commit_result(self, _workflow_id):
            pytest.fail("release route rejection must happen before commit")

    coordinator = FakeCoordinator()
    monkeypatch.setattr(
        host, "_ensure_task_graph_coordinator", lambda: coordinator,
    )

    with pytest.raises(HTTPException) as captured:
        host.execute_task_graph_chat_with_slot(ChatRequest(
            message="release route policy",
            session_id="s1",
            execution_mode="task_graph",
            routing_preference="distributed_required",
            workflow_id="wf_releaseenginehost01",
        ))

    assert captured.value.status_code == 503
    assert (
        captured.value.error_code
        == "DISTRIBUTED_REQUIRED_RELEASE_ROUTE_FORBIDDEN"
    )
    assert coordinator.discarded == [(
        "wf_releaseenginehost01",
        {"terminal_reason_code": "request_refused"},
    )]


def test_task_graph_auto_remote_identity_path(monkeypatch):
    """auto_remote 分支调用 _active_task_graph_model_identity（修复前
    1597 行裸 model_manager NameError）：无可用远端 → 本地降级执行。"""
    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(_cfg, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)

    host = make_full_host()
    host._host.full_chat_execution_lock = threading.RLock()
    host._active_session_id = "s1"
    host._session_histories["s1"] = []

    class FakeCoordinator:
        def __init__(self):
            self.providers = []

        def journal_status(self):
            return {"available": True}

        def has_provider(self, pid):
            return pid in self.providers

        def register_provider(self, provider):
            self.providers.append(provider.provider_id)

        def run_template(self, **kw):
            return (
                {"content": "远端不可用本地降级", "thinking_content": None},
                {
                    "workflow_id": "wf-2",
                    "template": "dual_candidate",
                    "state": "completed",
                    "partial_result": False,
                    "stages": [],
                    "stage_count": 0,
                    "attempt_count": 0,
                    "duration_seconds": 0.3,
                },
            )

        def commit_result(self, workflow_id):
            return {"workflow_id": workflow_id, "stages": [],
                    "template": "dual_candidate", "state": "completed",
                    "partial_result": False, "stage_count": 0,
                    "attempt_count": 0, "duration_seconds": 0.3}

        def discard_result(self, workflow_id, **_kwargs):
            return None

        def provider_status(self):
            return []

    monkeypatch.setattr(
        EngineHost, "_ensure_task_graph_coordinator",
        lambda self: FakeCoordinator(),
    )
    # 不 stub _active_task_graph_model_identity：真实执行方法体
    # （FakeModel 有 _engine_type/active_model_id 但缺 _model_path →
    # 返回 None → model_identity_unavailable → 本地降级；修复前 1597
    # 行裸 model_manager 在此路径 NameError）

    req = ChatRequest(
        message="hi", session_id="s1", execution_mode="task_graph",
        task_graph_auto_remote=True,
    )
    result = host.chat_full(req)
    assert result["content"] == "远端不可用本地降级"
    # 无可用远端 → 降级标记
    assert result["metrics"]["fallback"] is True
    assert result["metrics"]["auto_remote_enabled"] is True
    assert result["metrics"]["auto_remote_providers"] == []
    assert result["metrics"]["fallback_reason"] == "model_identity_unavailable"


def test_chat_full_llama_cpp_path(monkeypatch):
    """llama.cpp 引擎整请求路径：历史维护 + metrics + followups 兜底。"""
    import local_store

    monkeypatch.setattr(local_store, "get_local_save_history", lambda: False)  # 不落盘

    host = make_full_host()
    host._active_session_id = "s1"
    host._session_histories["s1"] = [
        {"role": "user", "content": "之前的问题"},
        {"role": "assistant", "content": "之前的回答"},
    ]

    req = ChatRequest(message="新问题", session_id="s1")
    result = host.chat_full(req)

    assert result["content"] == "候选答案内容"
    assert result["metrics"]["engine"] == "llama_cpp"
    assert result["metrics"]["execution_mode"] == "local_llama_cpp"
    assert result["metrics"]["request_origin"] == "web_http"
    # 历史已追加两轮
    assert host.session_history("s1")[-2:] == [
        {"role": "user", "content": "新问题"},
        {"role": "assistant", "content": "候选答案内容"},
    ]
    # followups：模型无合格问句 → 模板兜底 ≥2
    assert len(result["followups"]) >= 2
    assert host.conversation_status()["stats"]["rounds"] == 1


def test_chat_full_local_native_image_path_removes_temp_file(monkeypatch):
    import config as _cfg

    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    observed = {}

    class NativeImageModel(FakeModel):
        def native_vision_available(self):
            return True

        def chat_image(self, **kwargs):
            observed.update(kwargs)
            observed["exists_during_call"] = os.path.isfile(kwargs["image_path"])
            return {
                "content": "本地图片描述",
                "usage": {"completion_tokens": 3, "total_tokens": 10},
                "tokens_per_second": 7.5,
                "native_mtmd": True,
            }

    monkeypatch.setattr(_cfg, "EXTERNAL_ENABLED", False)
    host = make_full_host(NativeImageModel)
    req = ChatRequest(
        message="描述图片",
        image_data_urls=[image],
        routing_preference="local_only",
        max_new_tokens=24,
    )

    result = host.chat_full(req)

    assert result["content"] == "本地图片描述"
    assert result["metrics"]["execution_mode"] == "local_llama_cpp_mtmd"
    assert result["metrics"]["native_mtmd"] is True
    assert observed["exists_during_call"] is True
    assert not os.path.exists(observed["image_path"])


def test_chat_full_local_native_image_rejects_multi_image_fallback(monkeypatch):
    import config as _cfg

    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    class NativeImageModel(FakeModel):
        def native_vision_available(self):
            return True

    monkeypatch.setattr(_cfg, "EXTERNAL_ENABLED", False)
    host = make_full_host(NativeImageModel)
    req = ChatRequest(
        message="鎻忚堪鍥剧墖",
        image_data_urls=[image, image],
        allow_external=True,
        prefer_external=True,
    )

    with pytest.raises(HTTPException, match="不能静默丢弃"):
        host.chat_full(req)


def test_chat_full_session_switch(monkeypatch):
    """session_id 切换 → _switch_session 加载目标会话。"""
    import local_store

    monkeypatch.setattr(local_store, "get_local_save_history", lambda: False)

    host = make_full_host()
    host._active_session_id = "s_old"
    host._session_histories["s_old"] = [{"role": "user", "content": "旧会话"}]

    req = ChatRequest(message="hello", session_id="s_new")
    result = host.chat_full(req)
    assert host.conversation_status()["active_session_id"] == "s_new"
    assert result["content"] == "候选答案内容"
    # 新会话历史 = 本次对话
    assert host.session_history("s_new")[-1]["role"] == "assistant"


def test_chat_full_same_session_reloads_after_runtime_reset(monkeypatch):
    import inference_service.engine_host as engine_host_module

    rows = [
        {"role": "user", "content": "persisted question"},
        {"role": "assistant", "content": "persisted answer"},
    ]
    monkeypatch.setattr(
        engine_host_module._local_store,
        "load_local_conversation",
        lambda session_id: list(rows) if session_id == "s1" else [],
    )
    monkeypatch.setattr(
        engine_host_module._local_store,
        "get_local_save_history",
        lambda: False,
    )

    host = make_full_host()
    host._active_session_id = "s1"
    host._session_histories["s1"] = [{"role": "user", "content": "stale"}]
    host._reset_runtime_conversation_state(clear_histories=True)

    host.chat_full(ChatRequest(message="continue", session_id="s1"))

    assert host.session_history("s1")[:2] == rows


def test_engine_host_close_releases_task_graph_coordinator():
    host = FakeEngineHost()
    closed = []
    host._task_graph_coordinator = SimpleNamespace(
        close=lambda: closed.append(True),
    )

    host.close()
    host.close()

    assert closed == [True]
    assert host._task_graph_coordinator is None


def test_chat_full_first_message_auto_title(monkeypatch):
    """首条消息自动标题：截取前 30 字。"""
    titles = []
    import inference_service.engine_host as engine_host_module
    monkeypatch.setattr(
        engine_host_module._local_store,
        "update_local_session_title",
        lambda sid, title: titles.append((sid, title)),
    )

    host = make_full_host()
    long_msg = "这是一个非常非常非常非常非常非常非常非常非常非常长的首条消息啊"
    req = ChatRequest(message=long_msg, session_id="s_title")
    host.chat_full(req)
    assert titles and titles[0][0] == "s_title"
    assert titles[0][1].endswith("...")


def test_chat_full_external_fallback(monkeypatch):
    """外部路由失败（无本地模型）→ prefer_external 时 502。"""
    import config as _cfg
    import local_store
    import external_provider

    monkeypatch.setattr(local_store, "get_local_save_history", lambda: False)
    _cfg.EXTERNAL_ENABLED = True
    _cfg.EXTERNAL_BASE_URL = "https://example.com/v1"
    _cfg.EXTERNAL_DATA_SCOPE = "opt_in"

    class FailingClient(FakeExternalClient):
        def chat(self, history, **kw):
            raise RuntimeError("external down")

    monkeypatch.setattr(external_provider, "get_external_chat_client", lambda: FailingClient())

    host = make_full_host()
    host._host.model_loaded = False
    host._host.is_loaded = False

    from fastapi import HTTPException
    req = ChatRequest(message="hi", session_id="s1",
                      allow_external=True, prefer_external=True)
    with pytest.raises(HTTPException) as exc:
        host.chat_full(req)
    assert exc.value.status_code == 502


class FakeTokenizer:
    eos_token_id = 151643

    def __call__(self, prompt, return_tensors="pt"):
        import torch
        return {"input_ids": torch.zeros(1, 8, dtype=torch.long),
                "attention_mask": torch.ones(1, 8, dtype=torch.long)}

    def apply_chat_template(self, messages, tokenize=False, **kw):
        return "fake prompt"

    def decode(self, ids, skip_special_tokens=True):
        return "PyTorch本地回答"


class FakePyTorchModel(FakeModel):
    """PyTorch 引擎：带 tokenizer + model.generate 的假实现。"""

    _engine_type = "pytorch"
    tokenizer = FakeTokenizer()

    def __init__(self):
        super().__init__()
        import torch
        self._model = torch.nn.Linear(4, 4)  # 仅占位
        self.device_str = "cpu"

    @property
    def model(self):
        return self

    def ensure_full_model(self):
        return None

    def get_device(self):
        return "cpu"

    def _merge_stop_sequences(self, seqs):
        return None

    def _get_generation_eos_token_ids(self, stop_sequences):
        return None

    def _build_stop_criteria(self, seqs, prompt_len, **kw):
        return None

    def _decode_generated_ids(self, ids, stop_sequences):
        return "PyTorch本地回答"

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.generate_called = True
        return input_ids  # 形状 [1, prompt_len] → generated_ids 为空


def test_chat_full_pytorch_path(monkeypatch):
    """PyTorch 引擎路径：generate + 解码 + 追问 + 持久化。"""
    import local_store

    monkeypatch.setattr(local_store, "get_local_save_history", lambda: False)

    host = make_full_host(FakePyTorchModel)
    host._active_session_id = "s1"
    host._session_histories["s1"] = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩。"},
    ]

    req = ChatRequest(message="详细讲讲", session_id="s1")
    result = host.chat_full(req)

    assert result["content"] == "PyTorch本地回答"
    assert result["metrics"]["engine"] == "pytorch"
    assert result["metrics"]["execution_mode"] == "local_pytorch"
    assert host.conversation_status()["stats"]["rounds"] == 1
    # followups 走 PyTorch 追问路径（模型无问句 → 兜底）
    assert len(result["followups"]) >= 2
    # 会话历史追加
    assert host.session_history("s1")[-1]["role"] == "assistant"


# ----------------------------------------------------------------------
# 15. 1.3 服务入口：build_app + 角色感知门控
# ----------------------------------------------------------------------
def test_engine_host_persists_locally_without_legacy_export(monkeypatch):
    import inference_service.engine_host as engine_host_module

    events = []
    fake_local_store = SimpleNamespace(
        get_local_save_history=lambda: True,
        save_local_conversation_turn=(
            lambda sid, user, assistant, metrics=None, *, operation_id=None:
            events.append(("local", sid, user, assistant, metrics, operation_id))
            or True
        ),
    )
    monkeypatch.setattr(engine_host_module, "_local_store", fake_local_store)

    host = EngineHost()
    result = host._persist_conversation_turn("s-local", "q", "a", {})
    assert result.value == "committed"
    assert events == [("local", "s-local", "q", "a", {}, None)]


def test_inference_http_request_id_replay_survives_host_restart(monkeypatch):
    import inference_service.engine_host as engine_host_module

    durable: dict[str, list[dict]] = {"default": []}
    receipts: dict[str, tuple[str, str, str]] = {}

    def save_turn(
        session_id, user, assistant, metrics=None, *, operation_id=None,
    ):
        turn = (session_id, user, assistant)
        if operation_id in receipts:
            if receipts[operation_id] != turn:
                raise ValueError("operation_id has conflicting conversation turn")
            return False
        receipts[operation_id] = turn
        durable.setdefault(session_id, []).extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])
        return True

    monkeypatch.setattr(
        engine_host_module._local_store, "get_local_save_history", lambda: True,
    )
    monkeypatch.setattr(
        engine_host_module._local_store, "save_local_conversation_turn", save_turn,
    )
    monkeypatch.setattr(
        engine_host_module._local_store,
        "load_local_conversation",
        lambda session_id: list(durable.get(session_id, [])),
    )
    monkeypatch.setattr(
        engine_host_module._local_store,
        "update_local_session_title",
        lambda *_args, **_kwargs: None,
    )

    headers = {"X-QLH-Request-ID": "req-restart-replay"}
    first_host = make_full_host()
    with TestClient(make_app(engine_host=first_host)) as first_client:
        first = first_client.post(
            "/v1/chat", json={"message": "same question"}, headers=headers,
        )
    assert first.status_code == 200
    assert len(durable["default"]) == 2

    restarted_host = make_full_host()
    with TestClient(make_app(engine_host=restarted_host)) as second_client:
        replay = second_client.post(
            "/v1/chat", json={"message": "same question"}, headers=headers,
        )

    assert replay.status_code == 200
    assert len(durable["default"]) == 2
    assert restarted_host.session_history("default") == durable["default"]


def test_inference_stream_producer_does_not_commit_before_terminal_consumption(
    monkeypatch,
):
    import inference_service.engine_host as engine_host_module

    writes = []
    monkeypatch.setattr(
        engine_host_module._local_store, "get_local_save_history", lambda: True,
    )
    monkeypatch.setattr(
        engine_host_module._local_store,
        "save_local_conversation_turn",
        lambda *args, **kwargs: writes.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        engine_host_module._local_store,
        "load_local_conversation",
        lambda _session_id: [],
    )

    class StreamingModel(FakeModel):
        def chat_stream(self, messages, **_kwargs):
            del messages
            yield "first"
            yield "second"

    host = make_full_host(StreamingModel)
    cancel_event = threading.Event()
    events = host.chat_stream_events_with_operation_id(
        ChatRequest(message="stream", session_id="stream-session"),
        cancel_event,
        "req-stream-consume",
    )

    assert next(events) == {"token": "first"}
    cancel_event.set()
    with pytest.raises(ChatGenerationCancelled):
        next(events)

    assert writes == []
    assert host.session_history("stream-session") == []


def test_inference_stream_blocked_backend_returns_deadline_but_retains_lock():
    import time

    from request_deadline import RequestDeadline

    class BlockingStreamingModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.full_chat_execution_lock = threading.Lock()
            self.started = threading.Event()
            self.release = threading.Event()

        def chat_stream(self, messages, **_kwargs):
            del messages
            self.started.set()
            assert self.release.wait(5)
            yield "late"

    host = make_full_host(BlockingStreamingModel)
    model = host._host
    events = host.chat_stream_events_with_operation_id(
        ChatRequest(message="blocked stream", session_id="blocked-session"),
        threading.Event(),
        "req-blocked-stream",
        RequestDeadline.start(0.08),
    )

    started_at = time.monotonic()
    terminal = next(events)
    elapsed = time.monotonic() - started_at
    events.close()

    assert model.started.is_set()
    assert elapsed < 0.5
    assert terminal["done"] is True
    assert terminal["reason_code"] == "request_deadline_exceeded"
    assert model.full_chat_execution_lock.acquire(blocking=False) is False

    model.release.set()
    released = False
    wait_until = time.monotonic() + 2.0
    while time.monotonic() < wait_until:
        if model.full_chat_execution_lock.acquire(blocking=False):
            model.full_chat_execution_lock.release()
            released = True
            break
        time.sleep(0.01)
    assert released is True


def test_build_app_master_role(monkeypatch, tmp_path):
    from inference_svc_main import build_app

    monkeypatch.setenv("QLH_SQLITE_PATH", str(tmp_path / "control.sqlite3"))
    app = build_app("master")
    assert app.state.engine_host.role == "master"
    assert app.state.node_role == "master"
    with TestClient(app) as c:
        assert c.get("/v1/health").status_code == 200
        # master 角色放行 chat（无模型文件环境返回 500 模型未加载；有模型则 200）
        assert c.post("/v1/chat", json={"message": "你好"}).status_code in (200, 500, 507)


def test_build_app_client_role_gates_chat():
    from inference_svc_main import build_app

    app = build_app("client")
    assert app.state.engine_host.role == "client"
    fake_host = FakeEngineHost()
    fake_host.role = "client"
    app.state.engine_host = fake_host
    with TestClient(app) as c:
        # client 角色：chat 端点 404，层段/KV 端点仍可用
        assert c.post("/v1/chat", json={"message": "你好"}).status_code == 404
        assert c.post("/v1/chat/stream", json={"message": "你好"}).status_code == 404
        assert c.get("/v1/health").status_code == 200
        assert c.post("/v1/kv/init", json={}).status_code == 200
        r = c.post("/v1/layers/load", json={"layer_range": "0-12"})
        assert r.status_code in (200, 500)  # FakeModel 桩可加载


def test_entry_module_no_heavy_imports():
    """1.3 验收：入口顶层不 import model_module/transformers/sklearn。"""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'src'); import inference_svc_main; "
        "mods = set(sys.modules); "
        "heavy = [m for m in ('model_module', 'transformers', 'sklearn', 'pandas') "
        "         if any(k == m or k.startswith(m + '.') for k in mods)]; "
        "print('HEAVY:', heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert "HEAVY: []" in result.stdout, f"顶层拉入了重依赖: {result.stdout} {result.stderr}"


# ----------------------------------------------------------------------
# 16. 1.4 InferenceClient（HTTP 客户端，真实 uvicorn + FakeEngineHost）
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_inference_svc():
    """起真实 inference-svc（随机端口），engine_host 换为 FakeEngineHost。"""
    import threading

    import requests
    import uvicorn

    from inference_svc_main import build_app

    app = build_app("master", engine_host=FakeEngineHost())

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # 等待就绪
    import time
    deadline = time.time() + 15
    last_error = None
    port = None
    while time.time() < deadline:
        try:
            for running_server in getattr(server, "servers", []):
                sockets = getattr(running_server, "sockets", None) or []
                if sockets:
                    port = sockets[0].getsockname()[1]
                    break
            if port is None:
                time.sleep(0.1)
                continue
            response = requests.get(
                f"http://127.0.0.1:{port}/v1/health", timeout=1
            )
            response.raise_for_status()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail(f"inference-svc 未在 15 秒内就绪: {last_error}")
    assert port is not None
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_inference_client_lifecycle(live_inference_svc):
    from inference_client import InferenceClient

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    # 模型生命周期
    r = client.load_model(engine="pytorch")
    assert r["success"] is True
    assert client.model_loaded is True  # 状态缓存
    r = client.unload_model()
    assert r["success"] is True


def test_inference_client_chat(live_inference_svc, monkeypatch):
    from inference_client import InferenceClient

    # 验证 session_id 透传（修复前丢 session_id → 多轮历史语义漂移）
    captured = {}

    def _chat_full(self, req, cancel_event=None):
        captured["session_id"] = req.session_id
        return {
            "content": "你好，我是 QLH。",
            "thinking_content": None,
            "followups": [],
            "metrics": {"tokens_per_second": 42.0},
        }

    # FakeEngineHost 自带 chat_full 覆盖基类，patch 必须作用于子类
    monkeypatch.setattr(FakeEngineHost, "chat_full", _chat_full)
    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    result = client.chat(
        [{"role": "user", "content": "你好"}], session_id="sess-1"
    )
    assert captured["session_id"] == "sess-1"
    assert result["content"] == "你好，我是 QLH。"
    assert result["metrics"]["tokens_per_second"] == 42.0


def test_inference_client_chat_stream(live_inference_svc):
    from inference_client import InferenceClient

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    events = list(client.chat_stream([{"role": "user", "content": "你好"}]))
    tokens = [e["token"] for e in events if "token" in e]
    assert "".join(tokens) == "你好"
    assert events[-1]["done"] is True


def test_inference_client_forward_layers_loopback(live_inference_svc):
    from inference_client import InferenceClient

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    out = client.forward_layers("0-12", hidden)
    torch.testing.assert_close(out, hidden)  # FakeEngineHost identity


def test_inference_client_worker_stage(live_inference_svc):
    from inference_client import InferenceClient
    from task_provider import StageRequest as ProviderStageRequest

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    stage = ProviderStageRequest(
        workflow_id="w1", request_id="r1", stage_id="candidate_a",
        stage_type="full_inference", provider_id="local-full-model",
        dependencies={},
        root_input={"messages": [{"role": "user", "content": "1+1=?"}],
                    "task_options": {}},
    )
    result = client.scheduler_callbacks.execute_task_worker_stage(stage)
    assert result["content"] == "候选答案内容"


def test_inference_client_exposes_complete_scheduler_callback_bundle():
    from inference_client import InferenceClient

    client = InferenceClient(base_url="http://127.0.0.1:8010")
    callbacks = client.scheduler_callbacks
    assert callbacks is client.scheduler_callbacks
    assert callbacks.execute_task_worker_stage == client.execute_task_worker_stage
    assert callbacks.active_task_graph_model_identity == client.active_task_graph_model_identity


def test_inference_client_kv_and_cancel(live_inference_svc):
    from inference_client import InferenceClient

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    r = client.kv_init(task_id="t_remote")
    assert r["task_id"] == "t_remote"
    r = client.kv_free(task_id="t_remote")
    assert r["freed"] is True
    # 未知 generation cancel → 200 cancel_pending（对齐 api_server 语义，2026-08-05）
    r = client.cancel_generation("gen_nonexistent")
    assert r["status"] == "cancel_pending"


# ----------------------------------------------------------------------
# 17. 1.5 从节点 PeerClient（复制自 scheduler client 分支）
# ----------------------------------------------------------------------
class FakeTCPClient:
    def __init__(self):
        self.sent = []
        self._running = True
        self.device_info = {}
        self.server_host = "127.0.0.1"
        self.server_port = 8888
        self.is_registered = True
        self.sock = object()

    def send_data(self, payload, msg_type):
        self.sent.append((msg_type, dict(payload)))


def make_peer():
    from inference_service.peer import PeerClient

    peer = PeerClient(master_host="127.0.0.1", master_port=8888,
                      node_id="test_client")
    peer._host._ensure_device_profile = lambda: {
        "tier": "edge",
        "gpu": {"cuda_available": False},
    }
    fake = FakeModel()
    fake.forward_layers_called = []

    def _forward_layers(input_ids=None, hidden_states=None, attention_mask=None,
                        position_ids=None, past_key_values=None, use_cache=True,
                        apply_lm_head=False, **_kw):
        fake.forward_layers_called.append(
            (input_ids, hidden_states, past_key_values, apply_lm_head))
        hs = hidden_states if hidden_states is not None else torch.zeros(1, 8, 128)
        pkv = (torch.zeros(1, 2, 8, 64), torch.zeros(1, 2, 8, 64))
        out = {"hidden_states": hs}
        if apply_lm_head:
            out["logits"] = torch.zeros(1, 8, 32000)
        if use_cache:
            out["past_key_values"] = pkv
        return out

    fake.forward_layers = _forward_layers
    fake.full_load_calls = []
    fake.load_model = lambda **kw: fake.full_load_calls.append(dict(kw))
    fake.is_loaded = True
    fake._engine_type = "pytorch"
    fake.model = type("M", (), {"config": type("C", (), {"model_type": "qwen"})()})()
    peer._host._host = fake
    peer._client = FakeTCPClient()
    return peer


def test_peer_layer_config_new_format(monkeypatch):
    peer = make_peer()
    monkeypatch.setattr("model_sync.resolve_worker_model_path", lambda mid: "/fake/path")
    monkeypatch.setattr("model_sync.ensure_model_available", lambda mid: None)

    peer._handle_layer_config({
        "config_id": "cfg-1",
        "node_id": "test_client",
        "start_layer": 0, "end_layer": 12,
        "has_embedding": True, "has_lm_head": False,
        "model_id": "qwen-1.8b", "model_sha256": "abc123",
        "model_type": "qwen", "total_layers": 24,
        "master_quant_type": "fp32",
    })
    ack = [p for t, p in peer._client.sent if t.value == "layer_config_ack"]
    assert ack and ack[0]["status"] == "ready"
    assert ack[0]["config_id"] == "cfg-1"
    assert peer._active_layer_config["config_id"] == "cfg-1"
    assert peer._host._host.full_load_calls == []
    assert peer._host._host.layer_load_calls[0]["quant_type"] == "fp32"


def test_peer_disconnect_marks_client_for_reregistration():
    peer = make_peer()
    peer._running = True
    peer._pending_layer_config = ("cfg-old", {})
    peer._active_pipeline_task_ids.add("task-old")
    peer._local_pipeline_cancelled.add("task-old")
    peer._kv_cache["task-old"] = ("stale",)
    peer._close_all_relay_sessions = lambda: None
    peer._handle_disconnect()

    assert peer._running is False
    assert peer._active_layer_config is None
    assert peer._local_pipeline_steps == {}
    assert peer._pending_layer_config is None
    assert peer._active_pipeline_task_ids == set()
    assert peer._local_pipeline_cancelled == set()
    assert peer._kv_cache == {}


def test_peer_layer_config_missing_contract(monkeypatch):
    peer = make_peer()
    peer._handle_layer_config({
        "config_id": "cfg-x", "node_id": "test_client",
        "start_layer": 0, "end_layer": 12,
        "model_id": "", "model_sha256": "", "model_type": "qwen",
        "total_layers": 24,
    })
    ack = [p for t, p in peer._client.sent if t.value == "layer_config_ack"]
    assert ack and ack[0]["status"] == "error"


def test_peer_layer_forward_prefill_and_decode():
    from tcp_comm import MessageType

    peer = make_peer()
    peer._active_layer_config = {
        "config_id": "cfg-1", "model_sha256": "abc123",
        "model_type": "qwen", "model_id": "qwen-1.8b",
    }
    # prefill step 0
    peer._handle_layer_forward({
        "task_id": "t1", "step": 0, "use_kv_cache": False,
        "config_id": "cfg-1", "model_sha256": "abc123", "model_type": "qwen",
        "input_ids": [1, 2, 3], "apply_lm_head": False,
    })
    results = [p for t, p in peer._client.sent if t == MessageType.LAYER_RESULT]
    assert results and "error" not in results[0]
    assert results[0]["step"] == 0
    assert "hidden_states" in results[0]  # base64 序列化
    # decode step 1（用 KV cache）
    peer._client.sent.clear()
    peer._handle_layer_forward({
        "task_id": "t1", "step": 1, "use_kv_cache": True,
        "config_id": "cfg-1", "model_sha256": "abc123", "model_type": "qwen",
        "hidden_states": None, "apply_lm_head": True,
    })
    results = [p for t, p in peer._client.sent if t == MessageType.LAYER_RESULT]
    assert results and "error" not in results[0]
    assert "logits" in results[0]


def test_peer_layer_forward_step_out_of_order():
    from tcp_comm import MessageType

    peer = make_peer()
    peer._active_layer_config = {
        "config_id": "cfg-1", "model_sha256": "abc123",
        "model_type": "qwen", "model_id": "qwen-1.8b",
    }
    # 直接 decode step 2（无 prefill）→ 越序错误
    peer._handle_layer_forward({
        "task_id": "t9", "step": 2, "use_kv_cache": True,
        "config_id": "cfg-1", "model_sha256": "abc123", "model_type": "qwen",
    })
    results = [p for t, p in peer._client.sent if t == MessageType.LAYER_RESULT]
    assert results and "error" in results[0]


def test_peer_pipeline_done_abort_cleanup():
    peer = make_peer()
    peer._kv_cache["t1"] = object()
    peer._local_pipeline_steps["t1"] = 3
    peer._handle_pipeline_done({"task_id": "t1"})
    assert "t1" not in peer._kv_cache
    assert "t1" not in peer._local_pipeline_steps

    peer._kv_cache["t2"] = object()
    peer._handle_pipeline_abort({"task_id": "t2"})
    assert "t2" not in peer._kv_cache
    assert "t2" in peer._local_pipeline_cancelled


def test_peer_client_no_heavy_imports():
    """1.5 验收：peer 路径不 import fastapi/scheduler/api_server。"""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "import inference_service.peer; "
        "mods = set(sys.modules); "
        "bad = [m for m in ('fastapi', 'scheduler', 'api_server', 'uvicorn') "
        "       if any(k == m or k.startswith(m + '.') for k in mods)]; "
        "print('BAD:', bad)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert "BAD: []" in result.stdout, f"从节点入口拉入了重依赖: {result.stdout} {result.stderr}"


# ----------------------------------------------------------------------
# 17.5 scheduler-svc HTTP 壳（§4.2 透传路径契约，阶段 2 起点实现）
# ----------------------------------------------------------------------
@pytest.fixture()
def sched_http_client():
    """真实 Scheduler 实例 + HTTP 壳应用（不启动 TCP，TestClient 直连）。"""
    import scheduler as sched_mod
    import scheduler_svc_http as http_mod

    sched = sched_mod.Scheduler()
    app = http_mod.build_scheduler_app(sched)
    yield TestClient(app), sched
    http_mod.reset_scheduler()  # 测试隔离：清空注入实例


def test_sched_http_status(sched_http_client):
    """GET /cluster/status：run_mode/node_role/node_id/max_nodes 字段。"""
    client, _ = sched_http_client
    r = client.get("/cluster/status")
    assert r.status_code == 200
    body = r.json()
    assert "run_mode" in body and "node_role" in body
    assert "node_id" in body and "max_nodes" in body


def test_sched_http_nodes(sched_http_client):
    """GET /cluster/nodes：nodes/count/online_count/offline_count。"""
    client, _ = sched_http_client
    r = client.get("/cluster/nodes")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"nodes", "count", "online_count", "offline_count"}
    assert isinstance(body["count"], int)


def test_sched_http_my_role(sched_http_client):
    client, _ = sched_http_client
    r = client.get("/cluster/my-role")
    assert r.status_code == 200
    body = r.json()
    assert "is_master" in body and "node_role" in body and "node_id" in body


def test_sched_http_queue_lifecycle(sched_http_client):
    """queue 域：detail/strategy/pause/resume/clear。"""
    client, _ = sched_http_client
    r = client.get("/cluster/queue")
    assert r.status_code == 200
    body = r.json()
    for key in ("paused", "strategy", "queue_size", "max_size", "q0", "q1", "q2"):
        assert key in body, f"queue 缺字段 {key}"

    r = client.post("/cluster/queue/strategy", json={"strategy": "mlfq"})
    assert r.status_code == 200
    assert r.json()["strategy"] == "mlfq"

    r = client.post("/cluster/queue/pause")
    assert r.status_code == 200 and r.json()["paused"] is True
    r = client.post("/cluster/queue/resume")
    assert r.status_code == 200 and r.json()["paused"] is False
    r = client.post("/cluster/queue/clear")
    assert r.status_code == 200 and "cleared" in r.json()


def test_sched_http_config_and_layers(sched_http_client):
    client, _ = sched_http_client
    r = client.get("/cluster/config")
    assert r.status_code == 200
    assert "run_mode" in r.json() and "max_nodes" in r.json()

    r = client.get("/cluster/layers")
    assert r.status_code == 200
    body = r.json()
    assert "total" in body and "assignments" in body

    r = client.get("/cluster/config/distributed-inference")
    assert r.status_code == 200
    assert "enabled" in r.json()


def test_sched_http_nodes_deregister_404(sched_http_client):
    """deregister 不存在节点 → 404（对齐 api_server 语义）。"""
    client, _ = sched_http_client
    r = client.post("/cluster/nodes/nope/deregister")
    assert r.status_code == 404
    assert "detail" in r.json()


def test_sched_http_queue_task_cancel_404_semantics(sched_http_client):
    """cancel 不存在任务 → 200 + success:false（对齐 api_server.py:5715-5732）。"""
    client, _ = sched_http_client
    r = client.delete("/cluster/queue/task/nonexistent-task")
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is False
    assert "message" in body


# ----------------------------------------------------------------------
# 17.6 scheduler-svc device 域（画像 profile / auto-configure / select-gpu）
# ----------------------------------------------------------------------

FAKE_DEVICE_PROFILE = {
    "tier": "laptop",
    "tier_label": "笔记本",
    "tier_icon": "💻",
    "score_total": 85.5,
    "score_breakdown": {"gpu": 50.0, "ram": 24.0, "cpu": 11.5},
    "cpu": {
        "model_name": "Intel(R) Core(TM) i5-12400F",
        "physical_cores": 6,
        "logical_cores": 12,
        "freq_mhz": 2496.0,
        "freq_max_mhz": 4400.0,
        "architecture": "x86_64",
        "usage_percent": 3.2,
    },
    "ram": {"total_gb": 16.0, "available_gb": 8.5, "used_gb": 7.5, "percent_used": 46.9},
    "gpu": None,
    "gpus": [{
        "name": "NVIDIA GeForce RTX 3060",
        "vram_total_gb": 12.0,
        "vram_free_gb": 10.0,
        "cuda_available": True,
        "compute_capability": "8.6",
        "is_integrated": False,
        "gpu_type": "discrete",
        "driver_version": "566.36",
        "mps_available": False,
        "index": 0,
    }],
    "selected_gpu_index": 0,
    "disk": {"free_gb": 100.0, "total_gb": 512.0},
    "platform": {
        "os": "Windows",
        "os_version": "10.0.22631",
        "architecture": "AMD64",
        "machine": "AMD64",
        "hostname": "test-pc",
        "python_version": "3.12.10",
    },
    "recommendations": ["推荐使用 INT4 量化档位"],
    "warnings": [],
    "android_ready": False,
    "island": False,
}


def test_sched_http_device_profile(sched_http_client, monkeypatch):
    """GET /device/profile：真实 to_dict 字段 + TUI 兼容别名字段。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", FAKE_DEVICE_PROFILE)
    r = client.get("/device/profile")
    assert r.status_code == 200
    body = r.json()
    # 真实字段（device_profiler.to_dict 形状）
    assert body["tier"] == "laptop"
    assert body["score_total"] == 85.5
    assert body["platform"]["hostname"] == "test-pc"
    assert body["cpu"]["model_name"] == "Intel(R) Core(TM) i5-12400F"
    assert body["gpus"][0]["gpu_type"] == "discrete"
    # 兼容字段（TUI tui_admin.py:1141-1151 消费）
    assert body["hostname"] == "test-pc"
    assert body["os"] == {"system": "Windows", "release": "10.0.22631"}
    assert body["cpu"]["model"] == "Intel(R) Core(TM) i5-12400F"
    assert body["cpu"]["brand"] == "Intel(R) Core(TM) i5-12400F"
    assert body["memory"]["total_gb"] == 16.0


def test_sched_http_device_profile_detect_failure(sched_http_client, monkeypatch):
    """GET /device/profile：检测失败 → 500。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", None)

    def _boom():
        raise RuntimeError("detect failed")

    monkeypatch.setattr(http_mod, "_detect_device_profile", _boom)
    r = client.get("/device/profile")
    assert r.status_code == 500
    assert r.json()["error_code"] == "DEVICE_PROFILE_DETECTION_FAILED"


def test_sched_http_device_auto_configure(sched_http_client, monkeypatch):
    """POST /device/auto-configure：应用推荐配置。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", FAKE_DEVICE_PROFILE)

    class FakeProfiler:
        def recommend_config(self):
            return {
                "quant_type": "int4",
                "page_size": 128,
                "max_pages": 256,
                "max_seq_len": 2048,
                "max_new_tokens": 512,
                "use_compile": False,
                "device": "cuda:0",
                "description": "笔记本档：INT4 量化 + 中等上下文",
            }

    monkeypatch.setattr("device_profiler.get_profile", lambda: FakeProfiler())
    r = client.post("/device/auto-configure")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "configured"
    assert body["tier"] == "laptop"
    assert body["score"] == 85.5
    assert body["applied_config"]["page_size"] == 128
    assert body["recommendations"] == ["推荐使用 INT4 量化档位"]
    assert body["warnings"] == []


def test_sched_http_device_select_gpu_not_ready(sched_http_client, monkeypatch):
    """POST /device/select-gpu：画像未就绪 → 400。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", None)
    r = client.post("/device/select-gpu", json={"gpu_index": 0})
    assert r.status_code == 400
    assert r.json()["error_code"] == "DEVICE_PROFILE_NOT_READY"


def test_sched_http_device_select_gpu_out_of_range(sched_http_client, monkeypatch):
    """POST /device/select-gpu：序号越界 → 400。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", FAKE_DEVICE_PROFILE)
    r = client.post("/device/select-gpu", json={"gpu_index": 5})
    assert r.status_code == 400
    assert r.json()["error_code"] == "DEVICE_GPU_INDEX_INVALID"


def test_sched_http_device_select_gpu_ok(sched_http_client, monkeypatch):
    """POST /device/select-gpu：正常切换 → 200 switched。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", FAKE_DEVICE_PROFILE)

    class FakeProfiler:
        def __init__(self):
            self.selected = None

        def select_gpu(self, index):
            self.selected = index
            return True

        def to_dict(self):
            return FAKE_DEVICE_PROFILE

        def recommend_config(self):
            return {"device": "cuda:0", "description": "x"}

    fake = FakeProfiler()
    monkeypatch.setattr("device_profiler.get_profile", lambda: fake)
    r = client.post("/device/select-gpu", json={"gpu_index": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "switched"
    assert body["selected_gpu_index"] == 0
    assert body["selected_gpu"]["name"] == "NVIDIA GeForce RTX 3060"
    assert body["selected_gpu"]["gpu_type"] == "discrete"
    assert body["device"] == "cuda:0"
    assert fake.selected == 0


# ----------------------------------------------------------------------
# 17.7 /v1/models 注册表 / available / current 完整契约（推理面缺口修复）
# ----------------------------------------------------------------------

def test_v1_models_registry_shape(live_inference_svc):
    """GET /v1/models：内置模型 + 文件状态 payload + active_model_id。"""
    import requests
    r = requests.get(f"{live_inference_svc}/v1/models", timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and "active_model_id" in body
    assert isinstance(body["models"], list) and len(body["models"]) >= 1
    m = body["models"][0]
    # payload 完整字段（对齐 api_server _model_api_payload）
    for key in (
        "model_id", "name", "is_builtin", "model_type", "is_experimental",
        "recommended_vram_gb", "max_context", "quant_types", "description",
        "huggingface_id", "location", "model_path", "gguf_path",
        "is_available", "unavailable_reason", "available_formats",
        "has_safetensors", "has_gguf", "expected_paths",
        "supported_engines", "preferred_engine", "default_quant_type",
        "requires_cuda",
    ):
        assert key in m, f"payload 缺字段 {key}"
    # 至少一个内置模型 is_builtin
    assert any(x["is_builtin"] for x in body["models"])


def test_v1_models_available_shape(live_inference_svc):
    """GET /v1/models/available：quant 选项 + available_engines。"""
    import requests
    r = requests.get(f"{live_inference_svc}/v1/models/available", timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and "available_engines" in body
    assert "current" in body and "current_engine" in body
    for quant in body["models"]:
        for key in ("id", "name", "engine", "is_available", "memory_gb",
                    "speed_tok_s", "compile_support", "description"):
            assert key in quant, f"quant 缺字段 {key}"
    for engine in body["available_engines"]:
        for key in ("id", "name", "description", "model_size_gb", "requires_cuda"):
            assert key in engine, f"engine 缺字段 {key}"


def test_v1_models_current_loaded_shape(live_inference_svc):
    """GET /v1/models/current 已加载：完整 10 字段（FakeEngineHost 对齐新契约）。"""
    import requests
    r = requests.get(f"{live_inference_svc}/v1/models/current", timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True
    for key in (
        "model_id", "quant_type", "model_name", "model_path", "engine",
        "total_params", "device", "gpu_allocated_gb", "gpu_reserved_gb",
    ):
        assert key in body, f"current 缺字段 {key}"


def test_v1_models_current_unloaded_shape_real_host():
    """真实 EngineHost 未加载：{loaded:False, quant_type:None, model_id:None}
    （不触发模型加载，EngineHost.__init__ 仅 import model_host）。"""
    from inference_service.engine_host import EngineHost

    host = EngineHost()
    assert host.current_model() == {
        "loaded": False, "quant_type": None, "model_id": None,
    }


# ----------------------------------------------------------------------
# 14.7 模型生命周期校验（contract_diff 2026-08-05 复测修复回归）
#    - /v1/models/load：引擎白名单 400（此前 500）
#    - /v1/models/switch：未注册模型 404 detail（此前 200 success:false）
#    - 引擎大小写经统一 resolver 归一化后执行（校验与执行同值）
# ----------------------------------------------------------------------
def test_models_load_invalid_engine_400(client):
    # ★ 2026-09-19：`qwen-1_8b` 已从内置列表移除 ⇒ 改用 `qwen3-0.6b`
    #   （本用例验的是「引擎白名单」，与具体模型无关）。
    resp = client.post("/v1/models/load", json={
        "engine": "tensorrt", "model_id": "qwen3-0.6b"})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "MODEL_ENGINE_UNSUPPORTED"


def test_models_load_engine_case_normalized(client):
    """大写的 LLAMA_CPP：白名单校验用小写归一化，执行侧也必须用小写
    （修复前校验通过但执行传原始大小写会走错引擎分支）。"""
    resp = client.post("/v1/models/load", json={
        "engine": "LLAMA_CPP", "model_id": "qwen3-0.6b"})
    assert resp.status_code == 200


def test_models_load_torch_alias_uses_pytorch(client):
    resp = client.post("/v1/models/load", json={
        "engine": "torch", "quant_type": "int4", "model_id": "qwen3-0.6b"})
    assert resp.status_code == 200
    assert resp.json()["engine"] == "pytorch"


def test_models_load_layer_range_never_loads_full_model(monkeypatch):
    from inference_service import routes as routes_module
    from model_load_resolver import ModelLoadResolution

    host = FakeEngineHost()
    full_load_calls = []
    host.load_model = lambda **kwargs: full_load_calls.append(dict(kwargs))
    resolution = ModelLoadResolution(
        model_id="qwen3-0.6b",
        requested_engine="pytorch",
        engine="pytorch",
        artifact_kind="safetensors",
        model_path="C:/models/qwen3-0.6b",
        requested_quant="fp32",
        quant_type="fp32",
        runtime="pytorch_cpu",
        runtime_quant="fp32",
        layer_range_mode="dynamic",
        reason_code="ENGINE_EXPLICIT",
        reason="test",
    )
    monkeypatch.setattr(
        routes_module,
        "_resolve_service_model_load",
        lambda **_kwargs: resolution,
    )

    response = TestClient(make_app(engine_host=host)).post(
        "/v1/models/load",
        json={
            "engine": "pytorch",
            "quant_type": "fp32",
            "model_id": "qwen3-0.6b",
            "layer_range": "0-12",
        },
    )

    assert response.status_code == 200
    assert full_load_calls == []
    assert host._host.layer_load_calls == [{
        "start_layer": 0,
        "end_layer": 12,
        "model_path": "C:/models/qwen3-0.6b",
        "quant_type": "fp32",
        "model_id": "qwen3-0.6b",
        "profile": None,
        "total_layers": None,
    }]


def test_engine_host_precut_layer_range_prepares_descriptor_before_loading():
    from model_load_resolver import ModelLoadResolution

    host = FakeEngineHost()
    resolution = ModelLoadResolution(
        model_id="qwen3-0.6b",
        requested_engine="llama_cpp",
        engine="llama_cpp",
        artifact_kind="gguf",
        model_path="C:/models/qwen3-0.6b.gguf",
        requested_quant="Q4_K_M",
        quant_type="gguf",
        runtime="llama_cpp_native",
        runtime_quant="gguf",
        layer_range_mode="precut_artifact",
        reason_code="ENGINE_EXPLICIT",
        reason="test",
    )

    host.load_layer_range(layer_range="12-24", resolution=resolution)

    assert host._host.pipeline_prepare_calls == [{
        "model_id": "qwen3-0.6b",
        "model_path": "C:/models/qwen3-0.6b.gguf",
        "quant_type": "gguf",
        "layer_range": (12, 24),
        "model_sha256": None,
    }]
    assert host._host.layer_load_calls[0]["start_layer"] == 12
    assert host._host.layer_load_calls[0]["end_layer"] == 24
    assert [call[0] for call in host._host.candidate_calls] == [
        "prepare", "materialize", "commit", "finalize",
    ]


def test_models_switch_invalid_engine_400(client):
    resp = client.post("/v1/models/switch", json={
        "model_id": "qwen3-0.6b", "engine": "tensorrt"})
    assert resp.status_code == 400
    assert resp.json()["error_code"] == "MODEL_ENGINE_UNSUPPORTED"


def test_models_switch_unregistered_404(client):
    resp = client.post("/v1/models/switch", json={
        "model_id": "no-such-model", "engine": "llama_cpp"})
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "MODEL_NOT_REGISTERED"


def test_models_load_unregistered_404(client):
    resp = client.post("/v1/models/load", json={
        "model_id": "no-such-model", "engine": "llama_cpp"})
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "MODEL_NOT_REGISTERED"


class _FakeLoadHost:
    """记录 load/switch 调用的假宿主（无 ModelHost 依赖）。"""

    def __init__(self):
        self.model_loaded = False
        self.current_quant = "int4"
        self.quant_type = None
        self.load_calls = []
        self.switch_calls = []

    def load_model(self, engine=None, quant_type=None, use_compile=False,
                   model_id=None, model_path=None, profile=None, resolution=None):
        self.load_calls.append((engine, quant_type, use_compile, model_id))
        # 模拟 manager 归一化：llama_cpp 引擎忽略请求量化（GGUF 自带）
        self.quant_type = "gguf" if engine == "llama_cpp" else (quant_type or "int4")
        return None

    def switch_model(self, model_id=None, engine=None, quant_type=None,
                     model_path=None, profile=None, resolution=None):
        self.switch_calls.append((model_id, engine))
        self.quant_type = "int8"
        return {"success": True, "model_id": model_id, "quant_type": "int8"}

    def unload_model(self):
        self.model_loaded = False


def test_engine_host_load_sets_runtime_state():
    """load 成功后 model_loaded=True 且 current_quant 取 manager 生效值
    （对齐 api_server.py:2264-2265；2026-08-05 真实加载复测暴露 status
    恒 false）。"""
    host = EngineHost()
    fake = _FakeLoadHost()
    host._host = fake
    host.load_model(engine="pytorch", quant_type="int4", model_id="qwen-1_8b")
    assert fake.model_loaded is True
    assert fake.current_quant == "int4"  # manager.quant_type 优先
    # 请求值与 manager 归一化值不一致时取 manager（GGUF 引擎忽略 int4 场景）
    fake2 = _FakeLoadHost()
    host._host = fake2
    host.load_model(engine="llama_cpp", quant_type="int4", model_id="qwen-1_8b")
    assert fake2.current_quant == "gguf"


def test_engine_host_switch_sets_runtime_state():
    """switch 成功（result.success）后置位 model_loaded/current_quant
    （对齐 api_server.py:5146-5147；修复前 switch 路径漏置位）。"""
    host = EngineHost()
    fake = _FakeLoadHost()
    host._host = fake
    r = host.switch_model(model_id="qwen-1_8b", engine="llama_cpp")
    assert r["success"] is True
    assert fake.model_loaded is True
    assert fake.current_quant == "int8"


def test_engine_host_unload_clears_state():
    host = EngineHost()
    fake = _FakeLoadHost()
    fake.model_loaded = True
    host._host = fake
    host.unload_model()
    assert fake.model_loaded is False
