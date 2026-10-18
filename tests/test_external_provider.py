"""
单元测试 — 外部推理服务 Provider（路线 B PoC，external_provider）
================================================================
使用标准库 http.server 起一个线程化的 mock OpenAI 兼容后端
（复用 test_island_engine 的测试模式），覆盖:

1. 路由决策纯函数：数据作用域矩阵（deny / opt_in±flag / allow_all）+ 长度阈值
2. ExternalChatClient 非流式 / 流式补全、chunk 边界取消
3. 数据作用域最后关口：拒绝时不发出任何对外请求
4. 错误分类中文化（不可达 / HTTP 状态码 / retryable 判定），凭据脱敏
5. ExternalOpenAIProvider 经 ProviderRegistry 注册-预约-执行-取消全链路
6. ModelIdentity engine="external_api" 被 execution_provider / task_worker_protocol
   扩展后的校验集接受
7. API 层：/api/status external 段、/api/chat 外部路由 / 作用域拒绝 /
   后端故障回退本地 / prefer_external 无本地引擎的 502、/api/chat/stream SSE
"""

import json
import logging
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from model_host import model_host
from network_test_support import (
    release_reserved_unreachable_ports,
    reserve_unreachable_port,
)

import external_provider as ep
from external_provider import (
    ExternalChatClient,
    ExternalHTTPError,
    ExternalOpenAIProvider,
    ExternalScopeDeniedError,
    ExternalServiceError,
    ExternalTimeoutError,
    ExternalUnreachableError,
    decide_external_route,
    ensure_external_scope_allowed,
    external_model_identity,
    mask_external_url,
)
from execution_provider import (
    ModelIdentity,
    ProviderExecutionError,
    ProviderRegistry,
    StageAttempt,
    StageRequest,
)


# ================================================================
# Mock OpenAI 兼容后端（与 test_island_engine 相同模式 + 可配置逐 chunk 延迟）
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
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
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
            chunks = behavior.get("stream_chunks", ["外部", "服务", "回复"])
            chunk_sleep = behavior.get("stream_chunk_sleep", 0)
            first_chunk_written = behavior.get("first_chunk_written")
            for text in chunks:
                if chunk_sleep:
                    time.sleep(chunk_sleep)
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
                        ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n")
                        .encode("utf-8")
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return  # 客户端已取消断连（best-effort 取消语义）
                if first_chunk_written is not None:
                    first_chunk_written.set()
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
                    ("data: " + json.dumps(final, ensure_ascii=False) + "\n\n")
                    .encode("utf-8")
                )
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
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
                        "content": behavior.get("content", "这是外部服务的回复"),
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
def mock_external_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    server.behavior = {"model_id": "qwen2.5-7b-external"}
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


def _closed_port() -> int:
    return reserve_unreachable_port()


@pytest.fixture(autouse=True)
def _reset_external_singletons():
    """每个用例独享共享客户端与健康检查缓存，避免测试间串联。"""
    ep._shared_client = None
    ep.reset_reachable_cache()
    yield
    client = ep._shared_client
    ep._shared_client = None
    ep.reset_reachable_cache()
    if client is not None:
        client.close()
    release_reserved_unreachable_ports()


def _patch_external_config(
    monkeypatch,
    server=None,
    *,
    base_url=None,
    enabled=True,
    data_scope="opt_in",
    model="",
    api_key="",
    min_prompt_chars=0,
    label="测试外部服务",
    timeout=10,
    connect_timeout=3,
    reasoning_effort="",
):
    import config

    resolved = base_url if base_url is not None else (
        _base_url(server) if server is not None else ""
    )
    monkeypatch.setattr(config, "EXTERNAL_ENABLED", enabled, raising=False)
    monkeypatch.setattr(config, "EXTERNAL_BASE_URL", resolved, raising=False)
    monkeypatch.setattr(config, "EXTERNAL_API_KEY", api_key, raising=False)
    monkeypatch.setattr(config, "EXTERNAL_MODEL", model, raising=False)
    monkeypatch.setattr(config, "EXTERNAL_TIMEOUT", timeout, raising=False)
    monkeypatch.setattr(
        config, "EXTERNAL_CONNECT_TIMEOUT", connect_timeout, raising=False,
    )
    monkeypatch.setattr(config, "EXTERNAL_DATA_SCOPE", data_scope, raising=False)
    monkeypatch.setattr(
        config, "EXTERNAL_MIN_PROMPT_CHARS", min_prompt_chars, raising=False,
    )
    monkeypatch.setattr(config, "EXTERNAL_LABEL", label, raising=False)
    monkeypatch.setattr(
        config, "EXTERNAL_REASONING_EFFORT", reasoning_effort, raising=False,
    )
    return resolved


# ================================================================
# 路由决策纯函数：数据作用域矩阵 + 长度阈值
# ================================================================

@pytest.mark.parametrize(
    "scope,allow,prefer,expect_use,expect_reason",
    [
        # deny：硬禁用，携带 flag 也拒绝
        ("deny", True, True, False, "scope_deny"),
        ("deny", False, False, False, "scope_deny"),
        # opt_in（默认）：无 flag 不出集群
        ("opt_in", False, True, False, "scope_opt_in_missing_flag"),
        ("opt_in", False, False, False, "scope_opt_in_missing_flag"),
        # opt_in + flag + prefer → 外部
        ("opt_in", True, True, True, "prefer_external"),
        # opt_in + flag 无触发条件 → eligible 但仍本地
        ("opt_in", True, False, False, "no_trigger"),
        # allow_all：无需 flag
        ("allow_all", False, True, True, "prefer_external"),
        ("allow_all", False, False, False, "no_trigger"),
        # 未知档位按 deny 处理（默认 DENY 姿态）
        ("garbage", True, True, False, "scope_deny"),
    ],
)
def test_decide_external_route_scope_matrix(
    scope, allow, prefer, expect_use, expect_reason,
):
    decision = decide_external_route(
        enabled=True,
        base_url="http://10.0.0.9:8000",
        data_scope=scope,
        allow_external=allow,
        prefer_external=prefer,
        prompt_chars=10,
        min_prompt_chars=0,
    )
    assert decision.use_external is expect_use
    assert decision.reason == expect_reason


def test_decide_external_route_disabled_and_missing_url():
    assert decide_external_route(
        enabled=False, base_url="http://x", data_scope="allow_all",
        allow_external=True, prefer_external=True,
        prompt_chars=1, min_prompt_chars=0,
    ).reason == "disabled"
    assert decide_external_route(
        enabled=True, base_url="", data_scope="allow_all",
        allow_external=True, prefer_external=True,
        prompt_chars=1, min_prompt_chars=0,
    ).reason == "no_base_url"


def test_decide_external_route_prompt_length_threshold():
    common = dict(
        enabled=True, base_url="http://x", data_scope="opt_in",
        allow_external=True, prefer_external=False,
    )
    # 阈值 0 = 关闭长上下文卸载
    assert decide_external_route(
        prompt_chars=99999, min_prompt_chars=0, **common,
    ).use_external is False
    # 达到阈值 → 外部
    long_route = decide_external_route(
        prompt_chars=512, min_prompt_chars=512, **common,
    )
    assert long_route.use_external is True
    assert long_route.reason == "long_prompt"
    # 未达阈值 → 本地
    assert decide_external_route(
        prompt_chars=511, min_prompt_chars=512, **common,
    ).use_external is False


# ================================================================
# ExternalChatClient：非流式 / 流式 / 取消 / 作用域最后关口
# ================================================================

def _make_client(monkeypatch, server, **config_kwargs) -> ExternalChatClient:
    _patch_external_config(monkeypatch, server, **config_kwargs)
    client = ExternalChatClient()
    client.ensure_connected()
    return client


def test_client_chat_returns_content_and_metrics(monkeypatch, mock_external_server):
    client = _make_client(monkeypatch, mock_external_server)
    result = client.chat(
        [{"role": "user", "content": "你好"}],
        max_tokens=64,
        temperature=0.5,
        top_p=0.8,
        allow_external=True,
    )
    assert result["content"] == "这是外部服务的回复"
    assert result["usage"]["completion_tokens"] == 8
    assert result["model"] == "qwen2.5-7b-external"
    posts = [r for r in mock_external_server.requests if r["method"] == "POST"]
    assert posts[-1]["payload"]["max_tokens"] == 64
    assert posts[-1]["payload"]["stream"] is False
    client.close()


def test_client_preserves_structured_image_content_and_reasoning_effort(
    monkeypatch, mock_external_server,
):
    from multimodal import build_openai_user_content

    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    client = _make_client(
        monkeypatch,
        mock_external_server,
        reasoning_effort="none",
    )
    client.chat(
        [{
            "role": "user",
            "content": build_openai_user_content("描述图片", [image]),
        }],
        allow_external=True,
    )

    posts = [r for r in mock_external_server.requests if r["method"] == "POST"]
    payload = posts[-1]["payload"]
    assert payload["reasoning_effort"] == "none"
    assert payload["messages"][0]["content"][0] == {
        "type": "text",
        "text": "描述图片",
    }
    assert payload["messages"][0]["content"][1]["image_url"]["url"] == image
    client.close()


def test_client_chat_stream_yields_chunks(monkeypatch, mock_external_server):
    mock_external_server.behavior["stream_chunks"] = ["长", "上", "下", "文"]
    client = _make_client(monkeypatch, mock_external_server)
    chunks = list(client.chat_stream(
        [{"role": "user", "content": "你好"}], allow_external=True,
    ))
    assert chunks == ["长", "上", "下", "文"]
    posts = [r for r in mock_external_server.requests if r["method"] == "POST"]
    assert posts[-1]["payload"]["stream"] is True
    client.close()


def test_client_stream_cancel_stops_at_chunk_boundary(
    monkeypatch, mock_external_server,
):
    mock_external_server.behavior["stream_chunks"] = ["A"] * 50
    mock_external_server.behavior["stream_chunk_sleep"] = 0.05
    client = _make_client(monkeypatch, mock_external_server)
    cancel_event = threading.Event()
    received = []
    for chunk in client.chat_stream(
        [{"role": "user", "content": "hi"}],
        allow_external=True,
        cancel_event=cancel_event,
    ):
        received.append(chunk)
        if len(received) == 3:
            cancel_event.set()
    # chunk 边界断流：取消后最多再收到 0 个 chunk
    assert len(received) == 3
    client.close()


def test_client_chat_cancel_mid_stream_returns_partial(
    monkeypatch, mock_external_server,
):
    """非流式接口携带 cancel_event 时内部走流式，取消返回已收到的部分内容。"""
    mock_external_server.behavior["stream_chunks"] = ["片", "段"] * 20
    mock_external_server.behavior["stream_chunk_sleep"] = 0.05
    client = _make_client(monkeypatch, mock_external_server)
    cancel_event = threading.Event()
    result_box = {}

    def _run():
        result_box["result"] = client.chat(
            [{"role": "user", "content": "hi"}],
            allow_external=True,
            cancel_event=cancel_event,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    # 确定性等待：轮询服务器已收到流式请求再取消（替代固定 sleep(0.4)，
    # 避免慢机 0.4s 内一个 chunk 都没发出导致 content 为空的 flaky）
    deadline = time.time() + 5.0
    while not mock_external_server.requests and time.time() < deadline:
        time.sleep(0.01)
    cancel_event.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    result = result_box["result"]
    assert result["finish_reason"] == "cancelled"
    assert 0 < len(result["content"]) < len("片段" * 20)
    client.close()


def test_scope_gate_blocks_before_any_request(monkeypatch, mock_external_server):
    """数据作用域最后关口：opt_in 未带 flag → 拒绝且不发出任何补全请求。"""
    client = _make_client(monkeypatch, mock_external_server, data_scope="opt_in")
    posts_before = len(
        [r for r in mock_external_server.requests if r["method"] == "POST"]
    )
    with pytest.raises(ExternalScopeDeniedError) as excinfo:
        client.chat([{"role": "user", "content": "机密内容"}], allow_external=False)
    posts_after = len(
        [r for r in mock_external_server.requests if r["method"] == "POST"]
    )
    assert posts_before == posts_after
    # 消息文案不纳入断言（类型 ExternalScopeDeniedError 即契约）
    # deny 档位下带 flag 同样拒绝
    with pytest.raises(ExternalScopeDeniedError):
        ensure_external_scope_allowed(True, data_scope="deny")
    # allow_all 放行
    ensure_external_scope_allowed(False, data_scope="allow_all")
    client.close()


def test_backend_down_raises_clean_transport_error(monkeypatch):
    # Keep the closed-loopback-port assertion independent from CI/system proxies.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    _patch_external_config(
        monkeypatch, base_url=f"http://127.0.0.1:{_closed_port()}",
        connect_timeout=1, timeout=2,
    )
    client = ExternalChatClient()
    with pytest.raises((ExternalUnreachableError, ExternalTimeoutError)) as excinfo:
        client.ensure_connected()
    message = str(excinfo.value)
    if isinstance(excinfo.value, ExternalUnreachableError):
        assert "外部推理服务不可达" in message
    else:
        assert "外部推理服务超时" in message
    assert "孤岛" not in message
    assert "Traceback" not in message
    assert not client.is_connected


def test_http_error_maps_status_and_retryable(monkeypatch, mock_external_server):
    client = _make_client(monkeypatch, mock_external_server)
    for status, retryable in ((429, True), (503, True), (404, False)):
        mock_external_server.behavior["post_status"] = status
        with pytest.raises(ExternalHTTPError) as excinfo:
            client.chat([{"role": "user", "content": "hi"}], allow_external=True)
        assert str(status) in str(excinfo.value)
        # 消息文案不纳入断言（类型 ExternalHTTPError + status_code 即契约）
        assert excinfo.value.status_code == status
        assert ep._error_retryable(excinfo.value) is retryable
    mock_external_server.behavior["post_status"] = 200
    client.close()


def test_credential_masking_and_basic_auth(monkeypatch, mock_external_server):
    """URL 内嵌账号密码：状态展示脱敏，请求以 Basic 认证头继续发送。"""
    import base64

    port = mock_external_server.server_address[1]
    _patch_external_config(
        monkeypatch,
        base_url=f"http://extuser:extp%40ss@127.0.0.1:{port}",
    )
    client = ExternalChatClient()
    client.ensure_connected()
    assert "extuser" not in client.masked_base_url
    assert "extp" not in client.masked_base_url
    client.chat([{"role": "user", "content": "hi"}], allow_external=True)
    expected = "Basic " + base64.b64encode(b"extuser:extp@ss").decode()
    for request in mock_external_server.requests:
        assert request["headers"].get("Authorization") == expected
    assert mask_external_url("http://u:p@10.0.0.9:8000/v1?key=a") == (
        "http://10.0.0.9:8000/v1"
    )
    client.close()


# ================================================================
# ExternalOpenAIProvider：ProviderRegistry 注册-预约-执行-取消
# ================================================================

def _stage_request(stage_type="full_inference", **root_extra) -> StageRequest:
    root_input = {
        "message": "介绍一下张量并行",
        "messages": [{"role": "user", "content": "介绍一下张量并行"}],
        "task_options": {
            "candidate_max_tokens": 64,
            "final_max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }
    root_input.update(root_extra)
    return StageRequest(
        workflow_id="wf_external",
        request_id="req_external",
        stage_id="candidate_a" if stage_type == "full_inference" else "aggregate",
        stage_type=stage_type,
        provider_id="external_openai",
        dependencies={},
        root_input=root_input,
    )


def test_provider_executes_stage_via_registry(monkeypatch, mock_external_server):
    # 执行器携带 cancel_event → 传输层走内部流式（chunk 边界可取消）
    mock_external_server.behavior["stream_chunks"] = ["这是", "外部", "服务", "的回复"]
    _patch_external_config(monkeypatch, mock_external_server)
    provider = ExternalOpenAIProvider()
    registry = ProviderRegistry()
    registry.register(provider)

    capabilities = registry.inspect()[0]
    assert capabilities["provider_kind"] == "external_openai"
    assert set(capabilities["supported_stage_types"]) == {
        "full_inference", "aggregate",
    }

    request = _stage_request(allow_external=True)
    reservation = registry.reserve(request)
    attempt = StageAttempt(
        attempt_id="attempt_ext_1",
        request=request,
        provider_id="external_openai",
    )
    result = registry.execute(attempt, reservation, threading.Event())
    assert result.output["content"] == "这是外部服务的回复"
    assert result.output["model"] == "qwen2.5-7b-external"
    assert len(result.output["endpoint_fingerprint"]) == 64
    assert result.metadata["usage"]["completion_tokens"] == 4
    registry.release(reservation.reservation_id)
    registry.close()


def test_provider_stage_preserves_validated_structured_content():
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    request = _stage_request(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述图片"},
            {"type": "image_url", "image_url": {"url": image}},
        ],
    }])

    messages = ExternalOpenAIProvider._build_stage_messages(request)

    assert messages[0]["content"][0]["text"] == "描述图片"
    assert messages[0]["content"][1]["image_url"]["url"] == image


def test_provider_scope_denied_maps_to_provider_error(
    monkeypatch, mock_external_server,
):
    """未携带 allow_external 的 Stage：拒绝且连健康检查都不对外发出。"""
    _patch_external_config(monkeypatch, mock_external_server, data_scope="opt_in")
    provider = ExternalOpenAIProvider()
    request = _stage_request()  # root_input 无 allow_external
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="attempt_ext_denied",
        request=request,
        provider_id="external_openai",
    )
    with pytest.raises(ProviderExecutionError) as excinfo:
        provider.execute(attempt, reservation, threading.Event())
    assert excinfo.value.code == "external_scope_denied"
    assert mock_external_server.requests == []  # 零对外请求
    provider.release(reservation.reservation_id)
    provider.close()


def test_provider_cancel_mid_stream_returns_cancelled_partial(
    monkeypatch, mock_external_server,
):
    mock_external_server.behavior["stream_chunks"] = ["块"] * 40
    mock_external_server.behavior["stream_chunk_sleep"] = 0.05
    first_chunk_written = threading.Event()
    mock_external_server.behavior["first_chunk_written"] = first_chunk_written
    _patch_external_config(monkeypatch, mock_external_server)
    provider = ExternalOpenAIProvider()
    request = _stage_request(allow_external=True)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="attempt_ext_cancel",
        request=request,
        provider_id="external_openai",
    )
    result_box = {}

    def _run():
        result_box["result"] = provider.execute(
            attempt, reservation, threading.Event(),
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert first_chunk_written.wait(timeout=5), "mock backend did not write the first SSE chunk"
    provider.cancel("attempt_ext_cancel")
    thread.join(timeout=10)
    assert not thread.is_alive()
    output = result_box["result"].output
    assert output["finish_reason"] == "cancelled"
    assert 0 < len(output["content"]) < len("块" * 40)
    provider.release(reservation.reservation_id)
    provider.close()


def test_provider_backend_down_raises_retryable_error(monkeypatch):
    _patch_external_config(
        monkeypatch, base_url=f"http://127.0.0.1:{_closed_port()}",
        connect_timeout=1, timeout=2,
    )
    provider = ExternalOpenAIProvider()
    request = _stage_request(allow_external=True)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="attempt_ext_down",
        request=request,
        provider_id="external_openai",
    )
    with pytest.raises(ProviderExecutionError) as excinfo:
        provider.execute(attempt, reservation, threading.Event())
    assert excinfo.value.code == "external_backend_failed"
    assert excinfo.value.retryable is True
    provider.release(reservation.reservation_id)
    provider.close()


# ================================================================
# ModelIdentity engine="external_api" 校验集扩展
# ================================================================

def test_model_identity_external_api_accepted(monkeypatch, mock_external_server):
    _patch_external_config(
        monkeypatch, mock_external_server, model="qwen2.5-7b-external",
    )
    identity = external_model_identity()
    assert identity.engine == "external_api"
    assert identity.format == "openai_api"
    assert len(identity.sha256) == 64
    assert identity.revision.startswith("external-")

    # ModelIdentity 直接构造同样接受 external_api（execution_provider 扩展）
    ModelIdentity(
        model_id="external-api",
        engine="external_api",
        format="openai_api",
        revision=identity.revision,
        sha256=identity.sha256,
    )
    with pytest.raises(ValueError):
        ModelIdentity(
            model_id="external-api",
            engine="bogus_engine",
            format="openai_api",
            revision=identity.revision,
            sha256=identity.sha256,
        )

    # task_worker_protocol 的模型身份 / 能力校验接受 external_api
    from task_worker_protocol import (
        _validate_capabilities,
        _validate_model_identity,
    )

    _validate_model_identity(identity.snapshot(), "payload.model")
    _validate_capabilities({
        "stage_types": ["full_inference", "aggregate"],
        "engines": ["external_api"],
        "models": [identity.snapshot()],
        "max_concurrency": 2,
    }, version=2)


# ================================================================
# API 层：/api/status、/api/chat、/api/chat/stream
# ================================================================

@pytest.fixture()
def api_env(monkeypatch, tmp_path):
    """TestClient + mock scheduler + 无本地模型的干净 api_server 环境。"""
    from fastapi.testclient import TestClient

    import api_server
    import config

    with patch("api_server.scheduler", MagicMock()) as mock_sched:
        mock_sched.get_effective_node_id.return_value = "test-node"
        mock_sched._effective_role.return_value = "master"
        mock_sched.get_distributed_inference_enabled.return_value = False
        mock_sched.has_pipeline_worker_reservation.return_value = False
        mock_sched._max_nodes = 3
        monkeypatch.setattr(model_host, "model_loaded", False)
        monkeypatch.setattr(api_server, "_local_store", MagicMock())
        # 本地无可自动加载的模型（指向空目录）
        monkeypatch.setattr(
            config, "GGUF_MODEL_PATH", str(tmp_path / "no-model.gguf"),
        )
        monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "no-model-dir"))
        # ★ 2026-09-19：`_auto_load_default_model` 现在走**设备画像**解析
        #   （`config.get_active_model_paths()`），不再读上面的两个静态常量 ⇒
        #   必须一并 patch 成「空」，否则本机画像会在测试里**真的自动加载** 2B 模型，
        #   使「本地无模型 ⇒ 400」的断言失效（实测返回 200）。
        monkeypatch.setattr(config, "get_active_model_paths", lambda: {})
        monkeypatch.setattr(config, "ISLAND_ENABLED", False, raising=False)
        client = TestClient(api_server.app)
        yield {
            "client": client,
            "api_server": api_server,
            "scheduler": mock_sched,
        }


class _FakeLocalLlamaManager:
    """最小本地 llama_cpp 假引擎（外部故障回退本地用）。"""

    is_loaded = True
    _engine_type = "llama_cpp"
    active_model_id = "qwen-1_8b"

    def chat(self, messages=None, **kwargs):
        return {
            "content": "本地引擎回复",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 4,
                "total_tokens": 9,
            },
            "tokens_per_second": 12.5,
            "model": "qwen-1_8b-local",
        }


def test_status_reports_external_section_with_masked_url(
    monkeypatch, mock_external_server, api_env,
):
    port = mock_external_server.server_address[1]
    _patch_external_config(
        monkeypatch,
        base_url=f"http://statususer:statuspass@127.0.0.1:{port}",
        data_scope="opt_in",
        model="qwen2.5-7b-external",
        min_prompt_chars=2048,
    )
    response = api_env["client"].get("/api/status")
    assert response.status_code == 200
    external = response.json()["external"]
    assert external["enabled"] is True
    assert external["label"] == "测试外部服务"
    assert external["data_scope"] == "opt_in"
    assert external["reasoning_effort"] == ""
    assert external["min_prompt_chars"] == 2048
    assert external["model"] == "qwen2.5-7b-external"
    assert external["reachable"] is True
    assert "statuspass" not in external["base_url"]
    assert "statususer" not in external["base_url"]
    assert external["base_url"] == f"http://127.0.0.1:{port}"

    # 健康检查结果被缓存：停掉后端后 30s 内 /api/status 仍报 reachable=True
    mock_external_server.shutdown()
    response = api_env["client"].get("/api/status")
    assert response.json()["external"]["reachable"] is True


def test_status_omits_external_section_when_disabled(monkeypatch, api_env):
    _patch_external_config(monkeypatch, base_url="", enabled=False)
    response = api_env["client"].get("/api/status")
    assert response.status_code == 200
    assert response.json()["external"] is None


def test_models_available_lists_external_option(
    monkeypatch, mock_external_server, api_env,
):
    _patch_external_config(monkeypatch, mock_external_server)
    response = api_env["client"].get("/api/models/available")
    assert response.status_code == 200
    payload = response.json()
    engine_ids = [item["id"] for item in payload["available_engines"]]
    assert "external_api" in engine_ids
    external_models = [
        item for item in payload["models"] if item["engine"] == "external_api"
    ]
    assert len(external_models) == 1
    assert external_models[0]["is_available"] is True


def test_chat_routes_external_with_flags(
    monkeypatch, mock_external_server, api_env,
):
    """opt_in + allow_external + prefer_external：无本地模型也可整请求外发。"""
    # 聊天路径携带取消事件 → 传输层内部流式，content 为 chunk 拼接
    mock_external_server.behavior["stream_chunks"] = ["这是", "外部", "服务", "的回复"]
    _patch_external_config(monkeypatch, mock_external_server, data_scope="opt_in")
    response = api_env["client"].post("/api/chat", json={
        "message": "介绍一下张量并行",
        "allow_external": True,
        "prefer_external": True,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "这是外部服务的回复"
    assert body["metrics"]["engine"] == "external_api"
    assert body["metrics"]["execution_mode"] == "external_api"
    assert body["metrics"]["route"].endswith("_to_external_api")
    assert body["metrics"]["fallback"] is False
    posts = [r for r in mock_external_server.requests if r["method"] == "POST"]
    assert posts[-1]["payload"]["messages"][-1]["content"] == "介绍一下张量并行"


def test_chat_routes_bounded_image_content_without_persisting_image(
    monkeypatch, mock_external_server, api_env,
):
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    mock_external_server.behavior["stream_chunks"] = ["图", "片", "描", "述"]
    _patch_external_config(
        monkeypatch,
        mock_external_server,
        data_scope="opt_in",
        reasoning_effort="none",
    )
    persisted = []
    monkeypatch.setattr(
        api_env["api_server"],
        "_persist_conversation_turn",
        lambda session_id, user, assistant, metrics: persisted.append(
            (session_id, user, assistant, metrics)
        ),
    )

    response = api_env["client"].post("/api/chat", json={
        "message": "描述这张图片",
        "image_data_urls": [image],
        "allow_external": True,
        "prefer_external": True,
    })

    assert response.status_code == 200
    assert response.json()["content"] == "图片描述"
    posts = [r for r in mock_external_server.requests if r["method"] == "POST"]
    payload = posts[-1]["payload"]
    assert payload["reasoning_effort"] == "none"
    content = payload["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "描述这张图片"}
    assert content[1]["image_url"]["url"] == image
    assert persisted[0][1] == "描述这张图片"
    assert persisted[0][2] == "图片描述"
    assert image not in json.dumps(persisted, ensure_ascii=False)


def test_multimodal_external_failure_never_falls_back_without_image(
    monkeypatch, api_env,
):
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    _patch_external_config(
        monkeypatch,
        base_url=f"http://127.0.0.1:{_closed_port()}",
        data_scope="opt_in",
        connect_timeout=1,
        timeout=2,
    )
    api_server = api_env["api_server"]
    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "model_manager", _FakeLocalLlamaManager())

    response = api_env["client"].post("/api/chat", json={
        "message": "描述这张图片",
        "image_data_urls": [image],
        "allow_external": True,
        "prefer_external": True,
    })

    assert response.status_code == 502
    assert "禁止丢弃图片后回退" in response.json()["detail"]


def test_chat_without_flag_stays_local_under_opt_in(
    monkeypatch, mock_external_server, api_env,
):
    """opt_in 未带 flag：走原本地路径（本容器无本地模型 → 原 400 行为不变）。"""
    _patch_external_config(monkeypatch, mock_external_server, data_scope="opt_in")
    response = api_env["client"].post("/api/chat", json={"message": "你好"})
    assert response.status_code == 400
    assert "模型未加载" in response.json()["detail"]
    # 消息内容绝不外发
    posts = [r for r in mock_external_server.requests if r["method"] == "POST"]
    assert posts == []


def test_chat_scope_deny_blocks_flagged_request_and_logs(
    monkeypatch, mock_external_server, api_env, caplog,
):
    """deny + flag：拒绝外发（INFO 记录一次，不含正文），落回本地路径。"""
    _patch_external_config(monkeypatch, mock_external_server, data_scope="deny")
    with caplog.at_level(logging.INFO, logger="api_server"):
        response = api_env["client"].post("/api/chat", json={
            "message": "机密数据内容",
            "allow_external": True,
            "prefer_external": True,
        })
    assert response.status_code == 400  # 本地无模型的原行为
    posts = [r for r in mock_external_server.requests if r["method"] == "POST"]
    assert posts == []
    denial_logs = [
        record for record in caplog.records
        if "数据作用域拒绝外部路由" in record.getMessage()
    ]
    assert len(denial_logs) == 1
    assert "机密数据内容" not in denial_logs[0].getMessage()


def test_multimodal_scope_deny_fails_instead_of_dropping_image(
    monkeypatch, mock_external_server, api_env,
):
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    _patch_external_config(monkeypatch, mock_external_server, data_scope="deny")
    api_server = api_env["api_server"]
    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "model_manager", _FakeLocalLlamaManager())

    response = api_env["client"].post("/api/chat", json={
        "message": "描述这张图片",
        "image_data_urls": [image],
        "allow_external": True,
        "prefer_external": True,
    })

    assert response.status_code == 400
    assert "scope_deny" in response.json()["detail"]
    posts = [r for r in mock_external_server.requests if r["method"] == "POST"]
    assert posts == []


def test_chat_falls_back_to_local_on_backend_down(monkeypatch, api_env):
    """外部后端不可达 + 本地引擎可用：回退本地并记录 fallback_reason。"""
    _patch_external_config(
        monkeypatch,
        base_url=f"http://127.0.0.1:{_closed_port()}",
        data_scope="opt_in",
        min_prompt_chars=1,
        connect_timeout=1,
        timeout=2,
    )
    api_server = api_env["api_server"]
    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "model_manager", _FakeLocalLlamaManager())
    response = api_env["client"].post("/api/chat", json={
        "message": "触发长上下文外发的消息",
        "allow_external": True,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "本地引擎回复"
    assert body["metrics"]["engine"] == "llama_cpp"
    assert body["metrics"]["fallback"] is True
    assert body["metrics"]["fallback_reason"].startswith("external_api_failed:")
    # An unused loopback port may be reported directly, time out, or be
    # intercepted by a host proxy and return a gateway error. The stable
    # contract is the external_api_failed prefix and successful local fallback.


def test_chat_prefer_external_without_local_engine_returns_502(
    monkeypatch, api_env,
):
    _patch_external_config(
        monkeypatch,
        base_url=f"http://127.0.0.1:{_closed_port()}",
        data_scope="opt_in",
        connect_timeout=1,
        timeout=2,
    )
    response = api_env["client"].post("/api/chat", json={
        "message": "你好",
        "allow_external": True,
        "prefer_external": True,
    })
    assert response.status_code == 502
    assert "外部推理服务调用失败" in response.json()["detail"]
    assert "本地无可用推理引擎" in response.json()["detail"]


def test_chat_stream_external_returns_sse_tokens(
    monkeypatch, mock_external_server, api_env,
):
    mock_external_server.behavior["stream_chunks"] = ["流", "式", "外", "发"]
    _patch_external_config(monkeypatch, mock_external_server, data_scope="opt_in")
    response = api_env["client"].post("/api/chat/stream", json={
        "message": "介绍一下张量并行",
        "streaming_mode": "fast",
        "allow_external": True,
        "prefer_external": True,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    tokens = [event["token"] for event in events if "token" in event]
    assert tokens == ["流", "式", "外", "发"]
    done_events = [event for event in events if event.get("done")]
    assert len(done_events) == 1
    assert done_events[0]["response"] == "流式外发"
    assert done_events[0]["metrics"]["engine"] == "external_api"
    assert done_events[0]["metrics"]["execution_mode"] == "external_api"


def test_chat_stream_full_mode_external_single_done_event(
    monkeypatch, mock_external_server, api_env,
):
    """full 模式（默认）：走完整 chat 流程，SSE 单 done 事件（孤岛同款语义）。"""
    mock_external_server.behavior["stream_chunks"] = ["这是", "外部", "服务", "的回复"]
    _patch_external_config(monkeypatch, mock_external_server, data_scope="opt_in")
    response = api_env["client"].post("/api/chat/stream", json={
        "message": "介绍一下张量并行",
        "streaming_mode": "full",
        "allow_external": True,
        "prefer_external": True,
    })
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 1
    assert events[0]["done"] is True
    assert events[0]["response"] == "这是外部服务的回复"
    assert events[0]["metrics"]["engine"] == "external_api"
