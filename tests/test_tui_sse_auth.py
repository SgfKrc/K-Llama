"""TUI SSE authentication and transport-boundary regression tests."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tui_api import ApiClient, ApiError, auth_login, iter_chat_payloads  # noqa: E402


class _AuthenticatedSseHandler(BaseHTTPRequestHandler):
    seen_authorization = []

    def do_POST(self):  # noqa: N802 - stdlib handler API
        self.__class__.seen_authorization.append(self.headers.get("Authorization"))
        if self.headers.get("Authorization") != "Bearer stream-token":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"auth_required"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"start":true}\n\n')
        self.wfile.write(b'data: {"token":"ok"}\n\n')
        self.wfile.write(b'data: {"done":true,"response":"ok"}\n\n')

    def log_message(self, *_args):
        return


@pytest.fixture
def authenticated_sse_server():
    _AuthenticatedSseHandler.seen_authorization = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthenticatedSseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_sse_reuses_bearer_header_and_consumes_authenticated_stream(
    authenticated_sse_server,
):
    port = authenticated_sse_server.server_address[1]
    api = ApiClient(host="127.0.0.1", port=port, auth_token="stream-token")

    payloads = list(iter_chat_payloads(api, "hello", read_timeout=3))

    assert payloads == [
        {"start": True}, {"token": "ok"}, {"done": True, "response": "ok"},
    ]
    assert _AuthenticatedSseHandler.seen_authorization == ["Bearer stream-token"]


def test_sse_without_bearer_surfaces_authentication_failure(
    authenticated_sse_server,
):
    port = authenticated_sse_server.server_address[1]
    api = ApiClient(host="127.0.0.1", port=port)

    with pytest.raises(ApiError) as caught:
        list(iter_chat_payloads(api, "hello", read_timeout=3))

    assert caught.value.status == 401
    assert _AuthenticatedSseHandler.seen_authorization == [None]


@pytest.fixture
def live_authenticated_api(monkeypatch, tmp_path):
    """Run the real FastAPI stream route behind a local uvicorn socket."""
    import api_server
    import auth_service
    import uvicorn

    monkeypatch.setenv("QLH_AUTH_REQUIRED", "1")
    monkeypatch.setattr(
        auth_service, "_auth_db_path", lambda: str(tmp_path / "auth.sqlite"),
    )
    auth_service._reset_for_tests()
    auth_service.get_auth_store().create_user(
        "stream-admin", "password123", role="admin",
    )

    def run_pipeline_safe(message, **_kwargs):
        return {
            "status": "ok", "response": "real-stream", "error": None,
            "metrics": {"engine": "llama_cpp", "tokens_generated": 1},
        }

    monkeypatch.setattr(
        api_server, "scheduler", SimpleNamespace(
            get_distributed_inference_enabled=lambda: False,
            _effective_role=lambda: "master",
            run_pipeline_safe=run_pipeline_safe,
            record_task_complete=lambda success=True: None,
            configure_callbacks=lambda *_: None,
            start=lambda: None,
            stop=lambda: None,
            _running=True,
        ),
    )
    monkeypatch.setattr(
        api_server, "model_manager", SimpleNamespace(
            is_loaded=True, _engine_type="test",
        ),
    )
    monkeypatch.setattr(
        api_server, "model_host", SimpleNamespace(
            model_loaded=True,
            current_quant=None,
            generation_config={},
            full_chat_execution_lock=contextlib.nullcontext(),
            configure_runtime_environment=lambda *_: None,
            configure_scheduler_callbacks=lambda *_: None,
            unload_model=lambda: None,
        ),
    )
    monkeypatch.setattr(
        api_server, "_external_route_decision",
        lambda _request: SimpleNamespace(use_external=False),
    )
    monkeypatch.setattr(api_server, "_commit_interactive_history", lambda *args: True)

    config = uvicorn.Config(
        api_server.app, host="127.0.0.1", port=0,
        log_level="warning", access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    port = None
    while time.monotonic() < deadline:
        for running_server in getattr(server, "servers", []):
            sockets = getattr(running_server, "sockets", None) or []
            if sockets:
                port = sockets[0].getsockname()[1]
                break
        if port is not None:
            try:
                ApiClient(host="127.0.0.1", port=port, timeout=1).get("/health")
                break
            except ApiError:
                pass
        time.sleep(0.05)
    if port is None:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("real api_server uvicorn did not become ready")
    try:
        yield ApiClient(host="127.0.0.1", port=port, timeout=3), server, thread
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        auth_service._reset_for_tests()


def test_authenticated_sse_reaches_real_api_stream_route(live_authenticated_api):
    api, _server, _thread = live_authenticated_api

    with pytest.raises(ApiError) as rejected:
        list(iter_chat_payloads(api, "hello", read_timeout=3))
    assert rejected.value.status == 401

    login = auth_login(api, "stream-admin", "password123")
    api.auth_token = str(login["token"])
    payloads = list(iter_chat_payloads(api, "hello", read_timeout=3))

    assert payloads[0]["start"] is True
    assert payloads[-1].get("done") is True, payloads
    assert payloads[-1]["response"] == "real-stream"
