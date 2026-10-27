"""Registered Android text chat crosses the real source and session boundary."""
from __future__ import annotations

import asyncio
from copy import copy
import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from tests.test_api_model_intent import api, registered_api  # noqa: F401


CHAT_PATHS = ("/api/chat", "/api/chat/stream")
SOURCE_ERROR = "MODEL_API_SOURCE_UNTRUSTED"
REPLY = "registered distributed reply"


@pytest.fixture
def registered_chat(registered_api, monkeypatch):
    """Keep the real TCP/source owners; replace only loaded-model execution."""
    owner = registered_api
    owner.scheduler = owner.server.scheduler
    owner.tcp.clients[owner.node_id].node_type = "android"
    owner.calls = []
    owner.after_execute = None
    owner.response_started = None
    monkeypatch.setattr(owner.server, "model_host", SimpleNamespace(
        model_loaded=True, full_chat_execution_lock=threading.RLock(),
    ))
    monkeypatch.setattr(owner.server, "model_manager", SimpleNamespace(is_loaded=True))
    monkeypatch.setattr(owner.server, "_pipeline_model_is_prepared", lambda: False)

    def execute(req, cancel_event=None, request_deadline=None, *, forward_only=False):
        owner.calls.append((req, cancel_event, request_deadline, forward_only))
        if owner.response_started is not None:
            assert owner.response_started.wait(10), "SSE response did not start before execution completed"
        if owner.after_execute is not None:
            owner.after_execute()
        return {
            "content": REPLY,
            "thinking_content": None,
            "metrics": {"execution_mode": "distributed_pipeline", "workers_used": [owner.node_id]},
            "followups": [],
        }

    monkeypatch.setattr(owner.server, "_execute_requested_chat", execute)
    return owner


def _payload(owner, **extra):
    return {"message": "Android text chat", "routing_preference": "distributed_required", **extra}


def _client(owner, *, host=None, app=None):
    # No lifespan: importing the API does not start its server or journal.
    return TestClient(app or owner.server.app, client=(host or owner.peer, 50000))


def _events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def _error_code(response):
    body = response.json()
    # Middleware's existing HTTPException format differs from coded_http_error.
    if "error_code" in body:
        assert response.headers["X-QLH-Error-Code"] == body["error_code"]
        return body["error_code"]
    assert isinstance(body.get("detail"), dict), body
    return body["detail"]["code"]


def _assert_source_rejected(response):
    assert response.status_code == 403, response.text
    assert _error_code(response) == SOURCE_ERROR


def _assert_success(response, path):
    assert response.status_code == 200, response.text
    if path.endswith("/stream"):
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response)
        successes = [event for event in events if event.get("done") and not event.get("error")]
        assert len(successes) == 1, events
        assert not any(event.get("error") for event in events), events
        assert successes[0]["response"] == REPLY
        assert successes[0]["metrics"]["execution_mode"] == "distributed_pipeline"
    else:
        assert response.json()["content"] == REPLY
        assert response.json()["metrics"]["execution_mode"] == "distributed_pipeline"


@pytest.mark.parametrize("path", CHAT_PATHS)
@pytest.mark.parametrize("shape", ["minimal", "explicit_text", "android_metadata"])
def test_registered_android_required_text_chat_and_full_stream_succeed(registered_chat, path, shape):
    owner = registered_chat
    extra = {}
    if shape == "explicit_text":
        extra = {"execution_mode": "auto", "image_data_urls": [],
                 "allow_external": False, "prefer_external": False, "streaming_mode": "full"}
    elif shape == "android_metadata":
        extra = {"client_node_id": owner.node_id, "client_node_type": "android", "client_mode": "full",
                 "client_app_variant": "full", "generation_id": "gen_registered_chat",
                 "max_new_tokens": 32, "temperature": 0.5, "top_p": 0.9,
                 "show_thinking": False, "enable_thinking": False, "session_id": "android-session"}
    response = _client(owner).post(path, json=_payload(owner, **extra))
    _assert_success(response, path)
    assert len(owner.calls) == 1
    assert owner.calls[0][0].routing_preference == "distributed_required"
    assert owner.calls[0][1] is not None and owner.calls[0][2] is not None


@pytest.mark.parametrize("path", CHAT_PATHS)
def test_execution_mock_supports_required_forward_only_kwargs(registered_chat, path, monkeypatch):
    owner = registered_chat
    # Only the execution route is substituted; source still sees the real master TCP owner.
    monkeypatch.setattr(owner.server, "_should_forward_chat_to_master", lambda: True)
    _assert_success(_client(owner).post(path, json=_payload(owner)), path)
    assert len(owner.calls) == 1 and owner.calls[0][3] is True


FORBIDDEN_SHAPES = [
    pytest.param({"routing_preference": "auto"}, id="auto"),
    pytest.param({"routing_preference": "local_only"}, id="local"),
    pytest.param({"routing_preference": "distributed_preferred"}, id="preferred"),
    pytest.param({"execution_mode": "task_graph"}, id="taskgraph"),
    pytest.param({"task_graph_template": "dual_candidate"}, id="taskgraph-template"),
    pytest.param({"task_graph_remote_stage": "candidate_a"}, id="taskgraph-stage"),
    pytest.param({"task_graph_remote_provider_id": "worker"}, id="taskgraph-provider"),
    pytest.param({"task_graph_auto_remote": False}, id="taskgraph-auto-field"),
    pytest.param({"task_graph_inferred": False}, id="taskgraph-inferred-field"),
    pytest.param({"workflow_id": "wf_registered_chat"}, id="workflow"),
    pytest.param({"model_path": "G:/private/model.gguf"}, id="model-path"),
    pytest.param({"path": "G:/private/text.txt"}, id="path"),
    pytest.param({"input_path": "G:/private/text.txt"}, id="input-path"),
    pytest.param({"unknown_key": "ignored-by-schema"}, id="unknown-key"),
    pytest.param({"allow_external": True}, id="allow-external"),
    pytest.param({"prefer_external": True}, id="prefer-external"),
    pytest.param({"allow_external": "false"}, id="external-string"),
    pytest.param({"allow_external": 0}, id="external-integer-is-not-false"),
    pytest.param({"prefer_external": 0.0}, id="external-number-is-not-false"),
    pytest.param({"image_data_urls": ["data:image/png;base64,AA=="]}, id="image"),
    pytest.param({"image_data_urls": None}, id="image-null-is-not-empty-list"),
    pytest.param({"execution_mode": None}, id="execution-null-is-not-auto"),
]


@pytest.mark.parametrize("path", CHAT_PATHS)
@pytest.mark.parametrize("extra", FORBIDDEN_SHAPES)
def test_registered_peer_cannot_expand_text_chat_source_exception(registered_chat, path, extra):
    owner = registered_chat
    _assert_source_rejected(_client(owner).post(path, json=_payload(owner, **extra)))
    assert not owner.calls


@pytest.mark.parametrize("path", CHAT_PATHS)
@pytest.mark.parametrize("payload", [None, [], ["text"], "text", {"message": "routing missing"},
                                    {"message": "", "routing_preference": "distributed_required"}])
def test_source_shape_rejection_precedes_schema_or_execution(registered_chat, path, payload):
    owner = registered_chat
    response = _client(owner).post(path, content=json.dumps(payload), headers={"Content-Type": "application/json"})
    _assert_source_rejected(response)
    assert not owner.calls


@pytest.mark.parametrize("path", CHAT_PATHS)
def test_unregistered_tailnet_and_spoofed_headers_cannot_chat(registered_chat, path):
    owner = registered_chat
    response = _client(owner, host="100.64.10.21").post(path, json=_payload(owner), headers={
        "Host": owner.peer, "X-Forwarded-For": owner.peer, "X-Real-IP": owner.peer,
        "X-QLH-Intent-Hop": "1", "X-QLH-Node-ID": owner.node_id,
        "X-QLH-Registered-Distributed-Chat": "true",
    })
    _assert_source_rejected(response)
    assert not owner.calls


@pytest.mark.parametrize("path", CHAT_PATHS)
@pytest.mark.parametrize("state", ["unconfirmed", "pending"])
def test_connected_android_without_current_registration_ack_cannot_chat(registered_chat, path, state):
    owner = registered_chat
    connection = owner.tcp.clients[owner.node_id]
    if state == "pending":
        owner.tcp._pending_registrations[owner.node_id] = connection
    else:
        connection.registration_confirmed = False
    _assert_source_rejected(_client(owner).post(path, json=_payload(owner)))
    assert not owner.calls


@pytest.mark.parametrize("path", CHAT_PATHS)
def test_valid_registered_text_shape_ignores_retired_account_auth(registered_chat, path, monkeypatch):
    owner = registered_chat
    monkeypatch.setenv("QLH_AUTH_REQUIRED", "1")
    response = _client(owner).post(
        path, json=_payload(owner), headers={"Authorization": "Bearer invalid"})
    _assert_success(response, path)
    assert len(owner.calls) == 1


def _replace_peer(owner, change):
    with owner.tcp._clients_lock:
        old = owner.tcp.clients[owner.node_id]
        if change == "connection":
            owner.tcp.clients[owner.node_id] = copy(old)
            assert owner.tcp.clients[owner.node_id] is not old
        else:
            old.registration_epoch += 1
        assert owner.tcp.clients[owner.node_id].registration_confirmed is True


@pytest.mark.parametrize("path", CHAT_PATHS)
@pytest.mark.parametrize("change", ["connection", "epoch"])
def test_middleware_to_handler_reconnect_cannot_execute_or_adopt_new_session(registered_chat, path, change, monkeypatch):
    owner = registered_chat
    original_gate = owner.server.require_model_intent_source

    def admit_then_replace(request, scheduler):
        original_gate(request, scheduler)
        _replace_peer(owner, change)

    monkeypatch.setattr(owner.server, "require_model_intent_source", admit_then_replace)
    _assert_source_rejected(_client(owner).post(path, json=_payload(owner)))
    assert not owner.calls
    monkeypatch.setattr(owner.server, "require_model_intent_source", original_gate)
    _assert_success(_client(owner).post(path, json=_payload(owner)), path)
    assert len(owner.calls) == 1


@pytest.mark.parametrize("change", ["connection", "epoch"])
def test_full_sse_reconnect_after_execution_never_emits_success_done(registered_chat, change):
    owner = registered_chat
    owner.response_started = threading.Event()
    owner.after_execute = lambda: _replace_peer(owner, change)

    async def observe_response_start(scope, receive, send):
        async def observed_send(message):
            if message["type"] == "http.response.start":
                owner.response_started.set()
            await send(message)
        await owner.server.app(scope, receive, observed_send)

    response = _client(owner, app=observe_response_start).post(
        "/api/chat/stream", json=_payload(owner, streaming_mode="full"))
    assert response.status_code == 200, response.text
    assert len(owner.calls) == 1
    events = _events(response)
    assert events, response.text
    assert not any(event.get("done") and not event.get("error") for event in events), events
    assert any(event.get("done") and event.get("error") and event.get("outcome_reason") == SOURCE_ERROR
               for event in events), events
    assert not any(event.get("response") == REPLY for event in events), events


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/chat"), ("PUT", "/api/chat"), ("POST", "/api/chat/"),
    ("POST", "/api/chat/stream/"), ("POST", "/api/chat/stream/unknown"),
    ("POST", "/api/models/load"), ("POST", "/api/models/unload"),
    ("POST", "/api/models/prepare-pipeline"), ("GET", "/api/models/current"),
    ("GET", "/api/models/registry"),
])
def test_chat_marker_never_admits_other_methods_paths_or_model_management(registered_chat, method, path):
    owner = registered_chat
    _assert_source_rejected(_client(owner).request(method, path, json=_payload(owner)))
    assert not owner.calls
    assert owner.coordinator.create.await_count == 0


def test_unregistered_source_is_rejected_before_body_receive(registered_chat):
    import model_intent_access as access
    owner = registered_chat
    received = []

    async def receive():
        received.append(True)
        raise AssertionError("unregistered source body must not be read")

    request = Request({"type": "http", "method": "POST", "scheme": "http", "path": "/api/chat",
                       "query_string": b"", "client": ("100.64.10.21", 50000), "headers": []}, receive)
    assert asyncio.run(access.mark_registered_chat_request(request, owner.scheduler)) is False
    assert not access.is_registered_chat_request(request)
    assert not received and not owner.calls


@pytest.mark.parametrize("path", CHAT_PATHS)
def test_malformed_json_from_registered_android_is_source_rejected(registered_chat, path):
    owner = registered_chat
    response = _client(owner).post(path, content=b'{"message":', headers={"Content-Type": "application/json"})
    _assert_source_rejected(response)
    assert not owner.calls


@pytest.mark.parametrize("field", ["allow_external", "prefer_external"])
def test_null_external_flag_is_shape_admitted_without_weakening_schema(registered_chat, field):
    import model_intent_access as access
    owner = registered_chat
    body = json.dumps(_payload(owner, **{field: None})).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "POST", "scheme": "http", "path": "/api/chat",
                       "query_string": b"", "client": (owner.peer, 50000), "headers": []}, receive)
    assert asyncio.run(access.mark_registered_chat_request(request, owner.scheduler)) is True
    assert access.is_registered_chat_request(request)
    assert access.require_model_intent_source(request, owner.scheduler) == owner.peer
    assert not owner.calls
