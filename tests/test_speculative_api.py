"""Request validation and production API handoffs for explicit speculation."""
from contextlib import nullcontext
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import api_server
from qlh_core.contracts.deadline import RequestDeadline
from api import routes_chat
from api_errors import install_http_error_handler
from scheduler_cluster import SchedulerClusterMixin
from scheduler_pipeline import SchedulerPipelineMixin


@pytest.fixture
def schema_client():
    app = FastAPI()

    def receive(req: api_server.ChatRequest):
        return {"options": req.pipeline_speculation(), "sampling": req.pipeline_sampling()}

    app.post("/api/chat")(receive)
    app.post("/api/chat/stream")(receive)
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("gamma", [1, 2, 3, 4, 5, 6, 7, 8])
def test_http_accepts_explicit_gamma_without_changing_default_greedy(schema_client, endpoint, gamma):
    response = schema_client.post(endpoint, json={"message": "hello", "speculative_gamma": gamma})
    assert response.status_code == 200
    assert response.json() == {"options": {"speculative_gamma": gamma}, "sampling": [0.0, 1.0]}


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("value", [0, -1, 9, True, False, 2.0, 2.5, "4", "", [], {}])
def test_http_rejects_invalid_gamma_with_422(schema_client, endpoint, value):
    response = schema_client.post(endpoint, json={"message": "hello", "speculative_gamma": value})
    assert response.status_code == 422
    assert any(error["loc"] == ["body", "speculative_gamma"] for error in response.json()["detail"])


@pytest.mark.parametrize("options", [{}, {"speculative_gamma": None}])
def test_http_omission_and_null_leave_default_resolution_to_execution_owner(schema_client, monkeypatch, options):
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "4")
    response = schema_client.post("/api/chat", json={"message": "hello", **options})
    assert response.status_code == 200
    assert response.json()["options"] == {}


@pytest.fixture
def chat_handoff(monkeypatch):
    calls, commits = [], []
    state = {"role": "master", "error": None}
    prepared = SimpleNamespace(session_id="temporary", history=[], revision=7)

    def pipeline(message, **options):
        calls.append(("pipeline", message, options))
        if state["error"]:
            return {"execution_owner": "linear", "error": state["error"], "reason_code": state["error"]}
        return {"execution_owner": "linear", "response": "verified", "metrics": {"distributed_used": True}}

    def forward(**options):
        calls.append(("forward", options["message"], options))
        if state.get("forward_error"):
            return {"status": "error", "error": state["forward_error"],
                    "reason_code": state["forward_error"]}
        return {"status": "ok", "content": "verified", "metrics": {"distributed_used": True}}

    def forbidden_full_chat(**_options):
        pytest.fail("API handoff invoked a full model")

    scheduler = SimpleNamespace(
        _effective_role=lambda: state["role"],
        get_distributed_inference_enabled=lambda: True,
        _execution_app=SimpleNamespace(handles=lambda: True),
        run_pipeline_safe=pipeline,
        forward_inference_to_master=forward,
        record_task_complete=lambda **_options: None,
    )
    for key, value in {
        "scheduler": scheduler,
        "model_manager": SimpleNamespace(is_loaded=True, chat=forbidden_full_chat),
        "RUN_MODE": "distributed",
        "_routing_gate_error": lambda _req: None,
        "_prepare_chat_context": lambda *_args, **_options: prepared,
        "_external_route_decision": lambda _req: SimpleNamespace(use_external=False, reason=""),
        "_pipeline_worker_is_reserved": lambda: False,
        "runtime_supports": lambda *_args: True,
        "_augment_chat_metrics": lambda metrics, *_args, **_options: dict(metrics),
        "_commit_chat_context_turn": lambda *args, **options: commits.append((args, options)),
        "_fallback_followups": lambda *_args: [],
        "conversation_stats": {"rounds": 0},
    }.items():
        monkeypatch.setattr(api_server, key, value)
    return calls, commits, state


@pytest.mark.parametrize("role", ["master", "client"])
@pytest.mark.parametrize("options", [{}, {"speculative_gamma": None}, {"speculative_gamma": 1},
                                    {"speculative_gamma": 2}, {"speculative_gamma": 4}, {"speculative_gamma": 8}])
def test_production_chat_preserves_gamma_at_pipeline_and_forward_handoffs(chat_handoff, monkeypatch, role, options):
    calls, commits, state = chat_handoff
    state["role"] = role
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "4")
    req = api_server.ChatRequest(message="hello", **options)
    result = api_server._execute_requested_chat(req, request_deadline=RequestDeadline.start(3))
    assert result["content"] == "verified"
    assert len(calls) == 1
    kind, message, handed = calls[0]
    assert (kind, message) == ("pipeline" if role == "master" else "forward", "hello")
    if options.get("speculative_gamma") is None:
        assert "speculative_gamma" not in handed
    else:
        assert handed["speculative_gamma"] == options["speculative_gamma"]
    assert (handed["temperature"], handed["top_p"]) == (0.0, 1.0)
    assert len(commits) == 1
    assert commits[0][1]["expected_revision"] == 7


@pytest.mark.parametrize("role", ["master", "client"])
def test_gamma_handoff_preserves_explicit_sampling_for_execution_admission(chat_handoff, role):
    calls, _, state = chat_handoff
    state["role"] = role
    req = api_server.ChatRequest(message="hello", speculative_gamma=4, temperature=0.7, top_p=0.2)
    api_server._execute_requested_chat(req, request_deadline=RequestDeadline.start(3))
    assert (calls[0][2]["temperature"], calls[0][2]["top_p"]) == (0.7, 0.2)
    assert calls[0][2]["speculative_gamma"] == 4


def test_speculative_admission_error_is_terminal_and_does_not_commit(chat_handoff):
    calls, commits, state = chat_handoff
    state["error"] = "speculative_multi_position_required"
    req = api_server.ChatRequest(message="hello", speculative_gamma=4)
    with pytest.raises(HTTPException) as raised:
        api_server._execute_requested_chat(req, request_deadline=RequestDeadline.start(3))
    assert raised.value.status_code == 503
    assert raised.value.error_code == state["error"]
    assert len(calls) == 1
    assert calls[0][2]["speculative_gamma"] == 4
    assert commits == []


@pytest.fixture
def production_client(chat_handoff, monkeypatch):
    """Real chat routes and SSE serialization; only model/transport work is fake."""
    calls, commits, state = chat_handoff

    def stream(message, **options):
        calls.append(("stream", message, options))
        if state["error"]:
            yield {"done": True, "error": state["error"], "reason_code": state["error"]}
            return
        yield {"token": "verified"}
        yield {"done": True, "response": "verified", "metrics": {"distributed_used": True}}

    def native_stream(message, **options):
        calls.append(("native_stream", message, options))
        yield {"token": "native"}
        yield {"done": True, "response": "native", "metrics": {"distributed_used": False}}

    def external_stream(*_args, **_options):
        calls.append(("external_stream", "hello", {}))
        yield {"token": "external"}
        yield {"done": True, "response": "external", "metrics": {"distributed_used": False}}

    def external_chat(*_args, **_options):
        calls.append(("external", "hello", {}))
        return {"content": "external", "metrics": {}, "followups": []}

    def forbid_load(*_args, **_options):
        pytest.fail("speculation route unexpectedly loaded a whole model")

    api_server.scheduler.run_pipeline_stream = stream
    api_server.scheduler._run_full_model_inference_stream = native_stream
    monkeypatch.setattr(api_server, "model_host", SimpleNamespace(
        model_loaded=True, full_chat_execution_lock=threading.RLock()))
    monkeypatch.setattr(api_server, "backend_id_for", lambda _manager: "llama_cpp")
    api_server.model_manager.chat_stream = lambda **_options: None
    monkeypatch.setattr(api_server, "_chat_context", SimpleNamespace(
        resolve_session=lambda *_args, **_options: "temporary",
        transaction=lambda **_options: nullcontext(),
        acquire_transaction=lambda **_options: None,
        release_transaction=lambda: None,
    ))
    monkeypatch.setattr(api_server, "is_registered_chat_request", lambda _request: False)
    monkeypatch.setattr(api_server, "_register_generation", lambda _id: ("gamma-test", threading.Event()))
    monkeypatch.setattr(api_server, "_unregister_generation", lambda *_args: None)
    monkeypatch.setattr(api_server, "_pipeline_model_is_prepared", lambda: False)
    monkeypatch.setattr(api_server, "_auto_load_default_model", forbid_load)
    monkeypatch.setattr(api_server, "_ensure_chat_model_or_forwarding", forbid_load)
    monkeypatch.setattr(api_server, "_external_stream_events", external_stream)
    monkeypatch.setattr(api_server, "_execute_external_chat", external_chat)
    # Keep the external decision controllable without mocking the speculation gate.
    monkeypatch.setattr(api_server, "_external_route_decision", lambda _req: SimpleNamespace(
        use_external=state.get("external", False), eligible=True, reason=""))
    app = FastAPI()
    install_http_error_handler(app)
    app.post("/api/chat")(routes_chat.chat)
    app.post("/api/chat/stream")(routes_chat.chat_stream)
    with TestClient(app) as client:
        yield client, calls, commits, state


def _events(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.mark.parametrize("mode", ["interactive", "fast", "full"])
@pytest.mark.parametrize("options", [{}, {"speculative_gamma": None}, {"speculative_gamma": 1},
                                    {"speculative_gamma": 2}, {"speculative_gamma": 4}, {"speculative_gamma": 8}])
def test_actual_sse_pipeline_preserves_gamma(production_client, monkeypatch, mode, options):
    client, calls, commits, _ = production_client
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "invalid-on-transit")
    events = _events(client.post("/api/chat/stream", json={
        "message": "hello", "streaming_mode": mode,
        "routing_preference": "distributed_required", **options,
    }))
    assert events[-1]["response"] == "verified"
    assert not any(event.get("error") for event in events)
    assert len(calls) == len(commits) == 1
    kind, _, handed = calls[0]
    assert kind == ("pipeline" if mode == "full" else "stream")
    assert {k: v for k, v in handed.items() if k == "speculative_gamma"} == (
        {} if options.get("speculative_gamma") is None else options)
    assert handed["routing_preference"] == "distributed_required"
    assert (handed["temperature"], handed["top_p"]) == (0.0, 1.0)
    assert handed["_request_deadline"] is not None


@pytest.mark.parametrize("mode", ["fast", "full"])
@pytest.mark.parametrize("options", [{}, {"speculative_gamma": None}, {"speculative_gamma": 1},
                                    {"speculative_gamma": 4}])
def test_actual_sse_worker_forward_preserves_gamma(production_client, monkeypatch, mode, options):
    client, calls, commits, state = production_client
    state["role"] = "client"
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "8")
    events = _events(client.post("/api/chat/stream", json={
        "message": "hello", "streaming_mode": mode, **options,
    }))
    assert events[-1]["response"] == "verified"
    assert len(calls) == len(commits) == 1
    assert calls[0][0] == "forward"
    assert {k: v for k, v in calls[0][2].items() if k == "speculative_gamma"} == (
        {} if options.get("speculative_gamma") is None else options)
    assert calls[0][2]["_request_deadline"] is not None


@pytest.mark.parametrize("mode", ["interactive", "fast", "full"])
def test_sse_speculative_execution_refusal_keeps_reason_and_does_not_commit(production_client, mode):
    client, calls, commits, state = production_client
    state["error"] = "speculative_multi_position_required"
    events = _events(client.post("/api/chat/stream", json={
        "message": "hello", "streaming_mode": mode, "speculative_gamma": 4,
    }))
    assert events[-1]["error"] == state["error"]
    assert (events[-1].get("reason_code") or events[-1].get("outcome_reason")) == state["error"]
    assert len(calls) == 1
    assert not commits


@pytest.mark.parametrize("mode", [None, "fast", "full"])
def test_speculative_forward_refusal_never_falls_back(production_client, mode):
    client, calls, commits, state = production_client
    state.update(role="client", forward_error="speculative_load_policy_unsupported")
    payload = {"message": "hello", "speculative_gamma": 4}
    if mode:
        response = client.post("/api/chat/stream", json={**payload, "streaming_mode": mode})
        assert _events(response)[-1]["outcome_reason"] == state["forward_error"]
    else:
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 503
        assert response.json()["error_code"] == state["forward_error"]
    assert len(calls) == 1 and calls[0][0] == "forward"
    assert not commits


@pytest.mark.parametrize("route", ["local_only", "external", "local_full", "no_layers", "no_linear"])
@pytest.mark.parametrize("mode", [None, "interactive", "fast", "full"])
def test_actual_routes_refuse_explicit_speculation_before_model_or_external_work(
        production_client, monkeypatch, route, mode):
    client, calls, commits, state = production_client
    payload = {"message": "hello", "speculative_gamma": 4}
    if route == "local_only":
        payload["routing_preference"] = "local_only"
    elif route == "external":
        state["external"] = True
    elif route == "local_full":
        monkeypatch.setattr(api_server, "RUN_MODE", "local")
    elif route == "no_layers":
        monkeypatch.setattr(api_server, "runtime_supports", lambda *_args: False)
    else:
        api_server.scheduler._execution_app.handles = lambda: False
    if mode:
        response = client.post("/api/chat/stream", json={**payload, "streaming_mode": mode})
        terminal = _events(response)[-1]
        assert terminal["outcome_reason"] == "speculative_route_unsupported"
        assert terminal["outcome"] == "refused"
    else:
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 400
        assert response.json()["error_code"] == "speculative_route_unsupported"
    assert not calls and not commits


@pytest.mark.parametrize("mode", ["interactive", "fast"])
def test_gamma_one_allows_actual_local_native_sse(production_client, monkeypatch, mode):
    client, calls, commits, _ = production_client
    monkeypatch.setattr(api_server, "RUN_MODE", "local")
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "8")
    events = _events(client.post("/api/chat/stream", json={
        "message": "hello", "routing_preference": "local_only",
        "streaming_mode": mode, "speculative_gamma": 1,
    }))
    assert events[-1]["response"] == "native"
    assert len(calls) == len(commits) == 1
    assert calls[0][0] == "native_stream"


@pytest.mark.parametrize("mode", [None, "interactive", "fast", "full"])
def test_gamma_one_allows_actual_external_route(production_client, mode):
    client, calls, _, state = production_client
    state["external"] = True
    payload = {"message": "hello", "speculative_gamma": 1}
    if mode:
        events = _events(client.post("/api/chat/stream", json={**payload, "streaming_mode": mode}))
        assert events[-1]["response"] == "external"
    else:
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        assert response.json()["content"] == "external"
    assert len(calls) == 1
    assert calls[0][0].startswith("external")


def test_interactive_worker_keeps_existing_master_endpoint_requirement(production_client):
    client, calls, commits, state = production_client
    state["role"] = "client"
    events = _events(client.post("/api/chat/stream", json={
        "message": "hello", "streaming_mode": "interactive", "speculative_gamma": 4,
    }))
    assert "interactive 模式要求连接主节点" in events[-1]["error"]
    assert not calls and not commits


@pytest.fixture
def forward_receiver():
    """Execute the production handler, including its real background task/cleanup."""
    completed = threading.Event()
    calls, replies, tasks = [], [], []
    slots = threading.BoundedSemaphore(1)
    state = {"error": None}

    def pipeline(**options):
        calls.append(options)
        if state["error"]:
            return {"error": state["error"], "reason_code": state["error"], "metrics": {}}
        # Empty content avoids writing a real conversation from this transport test.
        return {"response": "", "metrics": {"distributed_used": True}}

    def send(*args, **options):
        replies.append((args, options))

    class CompletionSlots:
        def acquire(self, **options):
            return slots.acquire(**options)

        def release(self):
            slots.release()
            completed.set()

    owner = SimpleNamespace(
        _forward_infer_slots=CompletionSlots(),
        _forward_cancel_lock=threading.Lock(), _forward_cancel_events={},
        start_infer_task=lambda *args, **options: tasks.append((args, options)) or "task",
        fail_infer_task=lambda *_args: None,
        complete_infer_task=lambda *_args: None,
        run_pipeline_safe=pipeline, _send_infer_result=send,
    )
    return owner, calls, replies, tasks, completed, state


@pytest.mark.parametrize("options", [{}, {"speculative_gamma": None}, *[
    {"speculative_gamma": gamma} for gamma in range(1, 9)]])
def test_production_forward_receiver_validates_only_explicit_gamma(
        forward_receiver, monkeypatch, options):
    import qlh_core.contracts.speculation as speculation
    owner, calls, replies, tasks, completed, _ = forward_receiver
    resolved = []
    original = speculation.resolve_speculative_gamma

    def resolve(value=None, **kwargs):
        resolved.append(value)
        assert value is not None, "handler resolved an omitted sender default"
        return original(value, **kwargs)

    monkeypatch.setattr(speculation, "resolve_speculative_gamma", resolve)
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "invalid-on-receiving-node")
    SchedulerPipelineMixin.handle_infer_forward(owner, "worker", {"data": {
        "prompt": "hello", "temperature": 0.0, "top_p": 1.0,
        "routing_preference": "distributed_required", "forward_request_id": "forward-1", **options,
    }})
    assert completed.wait(3), "production forward task did not clean up"
    assert len(calls) == len(replies) == len(tasks) == 1
    expected = options.get("speculative_gamma")
    assert resolved == ([] if expected is None else [expected])
    assert {k: v for k, v in calls[0].items() if k == "speculative_gamma"} == (
        {} if expected is None else options)
    assert calls[0]["routing_preference"] == "distributed_required"
    assert calls[0]["_require_distributed"] is True
    assert calls[0]["_force_distributed_assignment"] is True
    assert replies[0][1]["forward_request_id"] == "forward-1"
    assert not owner._forward_cancel_events


@pytest.mark.parametrize("value", [0, -1, 9, True, False, 2.0, 2.5, "4", "", [], {}])
def test_production_forward_receiver_rejects_invalid_gamma_before_task(forward_receiver, value):
    owner, calls, replies, tasks, completed, _ = forward_receiver
    SchedulerPipelineMixin.handle_infer_forward(owner, "worker", {"data": {
        "prompt": "hello", "forward_request_id": "invalid", "speculative_gamma": value,
    }})
    assert not calls and not tasks and not completed.is_set()
    assert replies[0][0][3]["reason_code"] == "speculative_gamma_invalid"
    assert replies[0][1]["status"] == "error"
    assert replies[0][1]["forward_request_id"] == "invalid"
    assert not owner._forward_cancel_events


@pytest.mark.parametrize("gamma", [1, 4])
def test_production_forward_receiver_local_only_gamma_policy(forward_receiver, gamma):
    owner, calls, replies, tasks, completed, _ = forward_receiver
    SchedulerPipelineMixin.handle_infer_forward(owner, "worker", {"data": {
        "prompt": "hello", "routing_preference": "local_only", "speculative_gamma": gamma,
    }})
    if gamma == 1:
        assert completed.wait(3)
        assert calls[0]["speculative_gamma"] == 1
        assert calls[0]["routing_preference"] == "local_only"
        assert calls[0]["_force_distributed_assignment"] is False
    else:
        assert not calls and not tasks
        assert replies[0][0][3]["reason_code"] == "speculative_route_unsupported"
        assert replies[0][1]["status"] == "error"


def test_production_forward_receiver_preserves_execution_refusal(forward_receiver):
    owner, _, replies, _, completed, state = forward_receiver
    state["error"] = "speculative_multi_position_required"
    SchedulerPipelineMixin.handle_infer_forward(owner, "worker", {"data": {
        "prompt": "hello", "speculative_gamma": 4,
    }})
    assert completed.wait(3)
    assert replies[0][1]["status"] == "error"
    assert replies[0][0][3]["reason_code"] == state["error"]


@pytest.fixture
def forward_sender():
    payloads = []
    owner = SimpleNamespace(
        _effective_role=lambda: "client", _client_pending_lock=threading.Lock(),
        _client_pending_events={}, _client_pending_results={},
    )

    def send(data, kind):
        payloads.append((data, kind))
        request_id = data["forward_request_id"]
        with owner._client_pending_lock:
            owner._client_pending_results[request_id] = {
                "status": "ok", "content": "verified", "metrics": {"distributed_used": True},
            }
            owner._client_pending_events[request_id].set()

    owner._tcp_client = SimpleNamespace(_running=True, is_registered=True, sock=object(), send_data=send)
    return owner, payloads


@pytest.mark.parametrize("options", [{}, {"speculative_gamma": None}, *[
    {"speculative_gamma": gamma} for gamma in range(1, 9)]])
def test_production_forward_sender_wire_preserves_explicit_gamma(forward_sender, monkeypatch, options):
    from transport_port import MessageType
    owner, payloads = forward_sender
    monkeypatch.setenv("QLH_SPECULATIVE_GAMMA", "invalid-on-sending-node")
    result = SchedulerClusterMixin.forward_inference_to_master(owner, "hello", timeout=1, **options)
    assert result["status"] == "ok"
    assert len(payloads) == 1 and payloads[0][1] == MessageType.INFER_FORWARD
    assert {k: v for k, v in payloads[0][0].items() if k == "speculative_gamma"} == (
        {} if options.get("speculative_gamma") is None else options)
    assert not owner._client_pending_events and not owner._client_pending_results


@pytest.mark.parametrize("value", [0, -1, 9, True, False, 2.0, 2.5, "4", "", [], {}])
def test_production_forward_sender_rejects_invalid_gamma_without_sending(forward_sender, value):
    owner, payloads = forward_sender
    result = SchedulerClusterMixin.forward_inference_to_master(owner, "hello", speculative_gamma=value)
    assert result["status"] == "error"
    assert result["reason_code"] == "speculative_gamma_invalid"
    assert not payloads and not owner._client_pending_events


@pytest.mark.parametrize("gamma", [1, 4])
def test_production_forward_sender_local_only_gamma_policy(forward_sender, gamma):
    owner, payloads = forward_sender
    result = SchedulerClusterMixin.forward_inference_to_master(
        owner, "hello", routing_preference="local_only", speculative_gamma=gamma)
    if gamma == 1:
        assert result["status"] == "ok" and payloads[0][0]["speculative_gamma"] == 1
    else:
        assert result["reason_code"] == "speculative_route_unsupported"
        assert not payloads


@pytest.mark.parametrize("mode", ["interactive", "fast"])
@pytest.mark.parametrize("options", [{}, {"speculative_gamma": None}, {"speculative_gamma": 1}])
def test_actual_sse_safe_fallback_handoff_preserves_omission_and_one(
        production_client, monkeypatch, mode, options):
    client, calls, commits, _ = production_client
    monkeypatch.setattr(api_server, "RUN_MODE", "local")
    monkeypatch.setattr(api_server, "backend_id_for", lambda _manager: "other")
    events = _events(client.post("/api/chat/stream", json={
        "message": "hello", "streaming_mode": mode, **options,
    }))
    assert events[-1]["response"] == "verified"
    assert len(calls) == len(commits) == 1 and calls[0][0] == "pipeline"
    assert {k: v for k, v in calls[0][2].items() if k == "speculative_gamma"} == (
        {} if options.get("speculative_gamma") is None else options)


@pytest.mark.parametrize("mode", [None, "interactive", "fast", "full"])
@pytest.mark.parametrize("value", [True, 2.0, "4", 0, 9])
def test_actual_chat_routes_reject_invalid_gamma_at_http_boundary(production_client, mode, value):
    client, calls, commits, _ = production_client
    payload = {"message": "hello", "speculative_gamma": value}
    endpoint = "/api/chat"
    if mode:
        endpoint += "/stream"
        payload["streaming_mode"] = mode
    response = client.post(endpoint, json=payload)
    assert response.status_code == 422
    assert not calls and not commits
