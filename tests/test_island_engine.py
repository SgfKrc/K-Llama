"""
单元测试 — TP 孤岛推理引擎（island_engine.IslandEngine）
========================================================
使用标准库 http.server 起一个线程化的 mock OpenAI 兼容后端，覆盖:

1. 非流式对话补全（内容 + usage/tok/s 指标）
2. SSE 流式补全（chunk 拼接 + [DONE] 处理）
3. /v1/models 模型自动发现
4. 后端不可达 / 超时 / HTTP 错误 / 流中断的中文错误分类
5. api_key 配置时发送 Bearer 头
6. ModelManager 引擎选择与 island 引擎接线（stub torch.nn/transformers）
"""

import json
import os
import socket
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from network_test_support import (  # noqa: E402
    release_reserved_unreachable_ports,
    reserve_unreachable_port,
)

from island_engine import (
    IslandEngine,
    IslandEngineError,
    IslandHTTPError,
    IslandStreamInterruptedError,
    IslandTimeoutError,
    IslandUnreachableError,
    _classify_httpx_error,
    mask_island_url,
)


# ================================================================
# Mock OpenAI 兼容后端（标准库 http.server，线程化）
# ================================================================

class MockOpenAIHandler(BaseHTTPRequestHandler):
    """最小 OpenAI 兼容端点：/v1/models + /v1/chat/completions。"""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # 静默访问日志
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        behavior = self.server.behavior
        self.server.requests.append({
            "method": "GET",
            "path": self.path,
            "headers": dict(self.headers),
        })
        if self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": behavior["model_id"], "object": "model"}],
            })
        else:
            self._send_json({"error": {"message": "not found"}}, status=404)

    def do_POST(self):
        behavior = self.server.behavior
        length = int(self.headers.get("Content-Length", 0) or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append({
            "method": "POST",
            "path": self.path,
            "headers": dict(self.headers),
            "payload": payload,
        })

        if behavior.get("post_sleep"):
            time.sleep(behavior["post_sleep"])

        if self.path != "/v1/chat/completions":
            self._send_json({"error": {"message": "not found"}}, status=404)
            return
        if behavior.get("post_status", 200) != 200:
            self._send_json(
                {"error": {"message": behavior.get("post_error", "backend error")}},
                status=behavior["post_status"],
            )
            return

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = behavior.get("stream_chunks", ["你好", "，", "世界"])
            for text in chunks:
                event = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": behavior["model_id"],
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }],
                }
                try:
                    self.wfile.write(
                        ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8")
                    )
                except (BrokenPipeError, ConnectionResetError):
                    return
            if behavior.get("truncate_stream"):
                return  # 不发 finish_reason / [DONE]，模拟流中断
            final = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": behavior["model_id"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": len(chunks),
                    "total_tokens": 12 + len(chunks),
                },
            }
            try:
                self.wfile.write(
                    ("data: " + json.dumps(final, ensure_ascii=False) + "\n\n").encode("utf-8")
                )
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send_json({
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": behavior["model_id"],
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": behavior.get("content", "这是孤岛的回复"),
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            })


@pytest.fixture()
def mock_island_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    server.behavior = {"model_id": "qwen2.5-7b-instruct"}
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _base_url(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def _make_engine(server, **kwargs) -> IslandEngine:
    engine = IslandEngine()
    load_kwargs = dict(
        base_url=_base_url(server),
        api_key="",
        model="",
        timeout=10,
        connect_timeout=3,
    )
    load_kwargs.update(kwargs)
    engine.load_model(**load_kwargs)
    return engine


# ================================================================
# 非流式 / 流式补全
# ================================================================

def test_chat_returns_content_and_metrics(mock_island_server):
    engine = _make_engine(mock_island_server)
    result = engine.chat(
        [{"role": "user", "content": "你好"}],
        max_tokens=64,
        temperature=0.5,
        top_p=0.8,
    )

    assert result["content"] == "这是孤岛的回复"
    assert result["finish_reason"] == "stop"
    assert result["model"] == "qwen2.5-7b-instruct"
    assert result["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }
    assert result["tokens_per_second"] > 0

    # 请求体应携带映射后的生成参数
    posts = [r for r in mock_island_server.requests if r["method"] == "POST"]
    assert posts[-1]["payload"]["max_tokens"] == 64
    assert posts[-1]["payload"]["temperature"] == 0.5
    assert posts[-1]["payload"]["top_p"] == 0.8
    assert posts[-1]["payload"]["stream"] is False
    engine.unload()


def test_chat_optional_reasoning_effort_is_added_to_payload(mock_island_server):
    engine = _make_engine(mock_island_server)

    engine.chat(
        [{"role": "user", "content": "hi"}],
        reasoning_effort="none",
    )

    posts = [r for r in mock_island_server.requests if r["method"] == "POST"]
    assert posts[-1]["payload"]["reasoning_effort"] == "none"
    engine.unload()


def test_chat_stream_assembles_chunks_until_done(mock_island_server):
    mock_island_server.behavior["stream_chunks"] = ["分", "布", "式", "推理"]
    engine = _make_engine(mock_island_server)

    chunks = list(engine.chat_stream([{"role": "user", "content": "你好"}]))

    assert chunks == ["分", "布", "式", "推理"]
    posts = [r for r in mock_island_server.requests if r["method"] == "POST"]
    assert posts[-1]["payload"]["stream"] is True
    engine.unload()


def test_stream_interrupted_raises_chinese_error(mock_island_server):
    mock_island_server.behavior["stream_chunks"] = ["半", "截"]
    mock_island_server.behavior["truncate_stream"] = True
    engine = _make_engine(mock_island_server)

    with pytest.raises(IslandStreamInterruptedError) as excinfo:
        list(engine.chat_stream([{"role": "user", "content": "你好"}]))
    # 消息文案不纳入断言（异常类型即契约）
    engine.unload()


# ================================================================
# 模型自动发现 / 鉴权头
# ================================================================

def test_model_auto_discovery_from_v1_models(mock_island_server):
    mock_island_server.behavior["model_id"] = "auto-discovered-model"
    engine = _make_engine(mock_island_server, model="")

    assert engine.model_name == "auto-discovered-model"
    # 未显式配置模型时，chat 请求应使用发现的模型名
    result = engine.chat([{"role": "user", "content": "hi"}])
    posts = [r for r in mock_island_server.requests if r["method"] == "POST"]
    assert posts[-1]["payload"]["model"] == "auto-discovered-model"
    assert result["model"] == "auto-discovered-model"
    engine.unload()


def test_api_key_sends_bearer_header(mock_island_server):
    engine = _make_engine(mock_island_server, api_key="secret-island-key")
    engine.chat([{"role": "user", "content": "hi"}])

    for request in mock_island_server.requests:
        assert request["headers"].get("Authorization") == "Bearer secret-island-key"
    engine.unload()


def test_no_api_key_omits_authorization_header(mock_island_server):
    engine = _make_engine(mock_island_server, api_key="")
    engine.chat([{"role": "user", "content": "hi"}])
    for request in mock_island_server.requests:
        assert "Authorization" not in request["headers"]
    engine.unload()


# ================================================================
# 错误分类：不可达 / 超时 / HTTP 错误
# ================================================================

def test_transport_error_classification_is_platform_independent():
    unreachable = _classify_httpx_error(
        httpx.ConnectError("connection refused"),
        "http://127.0.0.1:1",
    )
    timeout = _classify_httpx_error(
        httpx.ConnectTimeout("timed out"),
        "http://127.0.0.1:1",
    )

    assert isinstance(unreachable, IslandUnreachableError)
    # 消息文案不纳入断言（异常类型即契约）
    assert isinstance(timeout, IslandTimeoutError)
    # 消息文案不纳入断言（异常类型即契约，P2-D 后消息仅是给人读的）


def test_backend_down_raises_clean_transport_error(monkeypatch):
    # Loopback must never be routed through a developer/CI HTTP proxy. A proxy
    # can return a synthetic HTTP status and hide the transport failure.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    engine = IslandEngine()
    port = reserve_unreachable_port()
    try:
        with pytest.raises((IslandUnreachableError, IslandTimeoutError)) as excinfo:
            engine.load_model(
                base_url=f"http://127.0.0.1:{port}",
                timeout=3,
                connect_timeout=1,
            )
        message = str(excinfo.value)
        if isinstance(excinfo.value, IslandUnreachableError):
            assert "孤岛后端不可达" in message
        else:
            assert "孤岛后端超时" in message
        # 错误消息不应泄露原始 traceback / 内部异常 repr
        assert "Traceback" not in message
        assert "ConnectError(" not in message
    finally:
        release_reserved_unreachable_ports()
        engine.unload()

    assert not engine.is_loaded


def test_timeout_raises_timeout_error(mock_island_server):
    engine = _make_engine(mock_island_server, timeout=1)
    mock_island_server.behavior["post_sleep"] = 3

    with pytest.raises((IslandTimeoutError, IslandStreamInterruptedError)) as excinfo:
        engine.chat([{"role": "user", "content": "hi"}])
    # 消息文案不纳入断言（异常类型即契约）
    mock_island_server.behavior["post_sleep"] = 0
    engine.unload()


def test_http_error_raises_with_status_code(mock_island_server):
    engine = _make_engine(mock_island_server)
    mock_island_server.behavior["post_status"] = 503

    with pytest.raises(IslandHTTPError) as excinfo:
        engine.chat([{"role": "user", "content": "hi"}])
    assert "503" in str(excinfo.value)
    assert "孤岛后端 HTTP 错误" in str(excinfo.value)
    mock_island_server.behavior["post_status"] = 200
    engine.unload()


# ================================================================
# 协作取消 / URL 脱敏
# ================================================================

def test_chat_pre_cancelled_does_not_call_backend(mock_island_server):
    engine = _make_engine(mock_island_server)
    cancel_event = threading.Event()
    cancel_event.set()
    before = len([r for r in mock_island_server.requests if r["method"] == "POST"])

    result = engine.chat(
        [{"role": "user", "content": "hi"}], _cancel_event=cancel_event,
    )

    after = len([r for r in mock_island_server.requests if r["method"] == "POST"])
    assert before == after
    assert result["content"] == ""
    assert result["finish_reason"] == "cancelled"
    assert result["usage_estimated"] is True
    engine.unload()


def test_url_userinfo_stripped_and_sent_as_basic_auth(mock_island_server):
    """URL 内嵌账号密码：内部 URL 必须剥离凭据（防 httpx INFO 日志泄露），
    同时以 Basic 认证头继续发送，线上行为与 httpx userinfo 语义一致。"""
    import base64

    port = mock_island_server.server_address[1]
    engine = IslandEngine()
    engine.load_model(
        base_url=f"http://islanduser:islandp%40ss@127.0.0.1:{port}",
        timeout=10,
        connect_timeout=3,
    )
    # 内部 base_url 不得再含凭据（httpx 会把请求 URL 原样打进 INFO 日志）
    assert "islanduser" not in engine._base_url
    assert "islandp" not in engine._base_url
    engine.chat([{"role": "user", "content": "hi"}])

    expected = "Basic " + base64.b64encode(b"islanduser:islandp@ss").decode()
    for request in mock_island_server.requests:
        assert request["headers"].get("Authorization") == expected
    engine.unload()


def test_mask_island_url_strips_credentials():
    assert mask_island_url("http://user:pass@10.0.0.2:8000/v1?key=abc") == (
        "http://10.0.0.2:8000/v1"
    )
    assert mask_island_url("http://10.0.0.2:8000/") == "http://10.0.0.2:8000"
    assert mask_island_url("") == ""


# ================================================================
# ModelManager 引擎选择与接线（stub torch.nn / transformers）
# ================================================================

def _import_model_module():
    """容器内 torch stub 无 torch.nn / 无 transformers，注入最小桩后导入。

    ⚠️ 这里注入的是**进程级全局**（`sys.modules` 与 `torch.nn` 属性），因此必须在
    finally 里复原 —— 否则会污染同进程的后续测试。实测：若本用例先跑，它把
    `sys.modules["transformers"]` 换成 stub 且不还原，后续
    `test_inference_service_protocol::test_build_app_client_role_gates_chat` 就会在
    「假 transformers（没有 DisjunctiveConstraint）」上失败（与 transformers 版本无关，
    属既存的测试卫生问题，5.x 只是让它更容易暴露）。
    """
    import torch

    had_torch_nn_module = "torch.nn" in sys.modules
    saved_torch_nn = getattr(torch, "nn", None)
    had_transformers = "transformers" in sys.modules
    try:
        if not had_torch_nn_module:
            nn_stub = types.ModuleType("torch.nn")

            class _Module:  # noqa: N801 - 与 torch.nn.Module 同名
                pass

            nn_stub.Module = _Module
            nn_stub.ModuleList = list
            sys.modules["torch.nn"] = nn_stub
            torch.nn = nn_stub

        if not had_transformers:
            tf_stub = types.ModuleType("transformers")
            for name in (
                "AutoConfig", "AutoModelForCausalLM",
                "AutoTokenizer", "BitsAndBytesConfig",
            ):
                setattr(tf_stub, name, type(name, (), {}))
            sys.modules["transformers"] = tf_stub

        import model_module
        return model_module
    finally:
        # 复原注入的桩（只还原**本次**注入的那些）
        if not had_transformers:
            sys.modules.pop("transformers", None)
            # ⚠️ model_module 是在**假 transformers**下导入的 ⇒ 它把桩类绑定进了自己的命名空间。
            #    若不一并移除，后续测试（例如 test_inference_service_protocol 经
            #    inference_svc_main → model_module）会拿到这个「桩版 model_module」，
            #    从而在看起来像「transformers 缺符号」的地方失败。
            sys.modules.pop("model_module", None)
        if not had_torch_nn_module:
            sys.modules.pop("torch.nn", None)
            if saved_torch_nn is not None:
                torch.nn = saved_torch_nn


def test_select_engine_prefers_island_when_enabled(mock_island_server, monkeypatch):
    model_module = _import_model_module()
    import config

    monkeypatch.setattr(config, "ISLAND_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ISLAND_BASE_URL", _base_url(mock_island_server), raising=False)

    assert model_module.ModelManager.select_engine() == "island"


def test_select_engine_honors_runtime_island_override(monkeypatch):
    model_module = _import_model_module()
    import config

    monkeypatch.setattr(config, "INFERENCE_ENGINE", "island", raising=False)
    monkeypatch.setattr(config, "ISLAND_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "ISLAND_BASE_URL", "", raising=False)

    assert model_module.ModelManager.select_engine() == "island"


def test_select_engine_ignores_island_when_disabled(monkeypatch):
    model_module = _import_model_module()
    import config

    monkeypatch.setattr(config, "ISLAND_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "ISLAND_BASE_URL", "", raising=False)

    assert model_module.ModelManager.select_engine() != "island"


def test_model_manager_loads_and_routes_island_engine(mock_island_server, monkeypatch):
    model_module = _import_model_module()
    import config

    monkeypatch.setattr(config, "ISLAND_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ISLAND_BASE_URL", _base_url(mock_island_server), raising=False)
    monkeypatch.setattr(config, "ISLAND_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "ISLAND_MODEL", "", raising=False)
    monkeypatch.setattr(config, "ISLAND_TIMEOUT", 10, raising=False)
    monkeypatch.setattr(config, "ISLAND_CONNECT_TIMEOUT", 3, raising=False)

    mgr = model_module.ModelManager()
    mgr.load_model(engine="island")

    assert mgr.is_loaded is True
    assert mgr.engine_type == "island"
    assert mgr.is_island is True
    assert mgr.is_llama_cpp is False

    # chat 应路由到孤岛引擎并拿到 mock 后端的回复
    result = mgr.chat([{"role": "user", "content": "你好"}], max_tokens=32)
    assert result["content"] == "这是孤岛的回复"
    assert result["usage"]["completion_tokens"] == 8

    # chat_stream 应逐 chunk 产出
    chunks = list(mgr.chat_stream([{"role": "user", "content": "你好"}]))
    assert "".join(chunks) == "你好，世界"

    # get_model_info 上报 island 引擎与脱敏端点
    info = mgr.get_model_info()
    assert info["engine"] == "island"
    assert info["base_url"].startswith("http://127.0.0.1:")
    assert "@" not in info["base_url"]

    mgr.unload_model()
    assert mgr.is_loaded is False
    assert mgr.engine_type == ""


def test_device_profile_contains_island_section(monkeypatch):
    """孤岛环境变量应体现在设备画像的 island 段与评分/档位中。"""
    monkeypatch.setenv("QLH_ISLAND_ENABLED", "1")
    monkeypatch.setenv("QLH_ISLAND_BASE_URL", "http://user:pass@10.0.0.2:8000")
    monkeypatch.setenv("QLH_ISLAND_BACKEND", "vllm-tp2")
    monkeypatch.setenv("QLH_ISLAND_TP_SIZE", "2")
    monkeypatch.setenv("QLH_ISLAND_GPU_COUNT", "2")
    monkeypatch.setenv("QLH_ISLAND_VRAM_GB", "48")

    from device_profiler import DeviceProfiler, detect_island_profile

    island = detect_island_profile()
    assert island["enabled"] is True
    assert island["backend"] == "vllm-tp2"
    assert island["gpu_count"] == 2
    assert island["vram_gb"] == 48.0
    # 凭据脱敏
    assert "pass" not in island["base_url"]
    assert island["base_url"] == "http://10.0.0.2:8000"

    profile = DeviceProfiler().to_dict()
    assert profile["island"]["enabled"] is True
    assert profile["island"]["tp_size"] == 2
    # 聚合能力附加分: min(50, 48/24*50)=50 + (2-1)*2.5 = 52.5
    assert profile["score_breakdown"]["island"] == 52.5
    # 孤岛网关档位按聚合能力判定
    assert profile["tier"] == "workstation"


def test_scheduler_node_weight_boosted_and_layer_split_excluded(monkeypatch):
    """主节点侧：孤岛节点权重上浮，且被排除出层拆分候选。"""
    monkeypatch.delenv("QLH_ISLAND_ENABLED", raising=False)
    import scheduler as scheduler_module

    sched = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)

    from tests.test_scheduler import _engine_contract
    plain_info = {
        **_engine_contract("pytorch", "pytorch_cpu"),
        "ram": {"total_gb": 16},
        "cpu": {"physical_cores": 8, "freq_max_mhz": 3200},
    }
    island_info = dict(plain_info)
    island_info["island"] = {
        "enabled": True,
        "backend": "vllm-tp2",
        "tp_size": 2,
        "gpu_count": 2,
        "vram_gb": 48.0,
    }

    plain_weight = sched._compute_node_weight(plain_info)
    island_weight = sched._compute_node_weight(island_info)
    # 公式: min(48,96)/24*50 + min(2,8)*5 + 60 = 100 + 10 + 60 = 170
    assert island_weight == pytest.approx(plain_weight + 170.0)

    assert sched._node_is_island_gateway(island_info) is True
    assert sched._node_is_island_gateway(plain_info) is False

    # compute_layer_assignment 的显式 nodes 分支必须过滤孤岛网关
    sched._pipeline_worker_opt_out = set()

    class _FakeLock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    sched._layer_config_lock = _FakeLock()
    monkeypatch.setattr(
        scheduler_module.Scheduler, "_get_total_model_layers", lambda self: 24,
    )
    assignments = sched.compute_layer_assignment(nodes=[
        {"node_id": "master", "role": "master", "node_type": "pc",
         "device_info": plain_info},
        {"node_id": "island-gw", "role": "client", "node_type": "pc",
         "device_info": island_info},
    ])
    assigned_nodes = {a["node_id"] for a in assignments}
    assert "island-gw" not in assigned_nodes
    assert assigned_nodes == {"master"}
