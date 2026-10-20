"""R4 composition tests exercise the production frozen-deployment entry with tiny native ports."""
from copy import deepcopy
from dataclasses import replace
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from model_host import ModelHost
from qlh_core.bootstrap.execution import ExecutionApplication
from qlh_core.contracts.deployment import canonical
from qlh_core.contracts.execution import ExecutionError
from qlh_core.contracts.deadline import RequestDeadline
from qlh_core.contracts.load import RuntimeLoadSpec
from qlh_core.execution.requests import RequestService
from tests.test_refactor_execution import repository, wait_for
from tests.test_worker_prepare_runtime import gguf, gguf_request
from tests.test_worker_prepare import command

pytestmark = pytest.mark.refactor_contract


def spec(*components):
    return RuntimeLoadSpec(device="cpu", dtype="float32", quant_type="fp32", context_size=128,
        kv_device="cpu", kv_dtype="float32", kv_policy="dynamic", components=("layers", *components),
        materialization="segment").to_dict()


class Tokenizer:
    eos_token_id = 999
    def __call__(self, *_args, **_kwargs):
        return {"input_ids": np.array([[1]], dtype=np.int64)}
    def decode(self, tokens, **_kwargs):
        return "".join({42: "A", 43: "B", 999: ""}[int(t)] for t in tokens)


@pytest.fixture
def application(repository):
    calls = []
    identity = {"operation_id": "supply_" + "a" * 32, "config_id": "deployment", "generation": 1,
        "assignment_id": "assignment", "artifact_sha256": "a" * 64, "hidden_size": 2,
        "layer_range": [2, 4], "stage_model_identity": {"model_id": "tail", "engine": "llama_cpp",
        "format": "gguf", "revision": "1", "sha256": "b" * 64}}
    local = {"node_id": "master", "start_layer": 0, "end_layer": 2, "load_spec": spec("embedding"),
             "identity": {"assignment_id": "local", "layer_range": [0, 2]}}
    tail = {"node_id": "worker", "start_layer": 2, "end_layer": 4, "load_spec": spec("lm_head", "final_norm"),
            "worker_prepare_identity": identity}
    state = {"active_deployment": "deployment", "lifecycle": {"total_layers": 4}, "deployments": {
        "deployment": {"published": True, "status": "active", "boot_id": "boot", "authority": {"term": 1},
                       "connections": {"worker": 1}, "local_assignment": local, "assignments": {"worker": tail}}}}
    def forward(**kwargs):
        calls.append(("prefix", kwargs))
        return {"hidden_states": np.zeros((1, 1, 2), dtype=np.float32), "cache": 1}
    manager = SimpleNamespace(layer_range=(0, 2), tokenizer=Tokenizer(), engine_type="llama_cpp", forward_layers=forward,
                              clear_pipeline_kv=lambda: calls.append(("local_cleanup",)))
    host = ModelHost(manager=manager)
    callbacks = SimpleNamespace(thinking_system_prompt="", build_model_chat_prompt=lambda *_a, **_kw: "prompt",
                                format_model_response=lambda text, *_a, **_kw: (text, None))
    def offer(**kwargs):
        calls.append(("offer", kwargs))
        if kwargs.get("cleanup_only"):
            return {"kv_cleared": True, "kind": "cleanup"}
        return {"kind": "token", "token_argmax": 42}
    scheduler = SimpleNamespace(_host=host,
        _model_control=SimpleNamespace(boot_id="boot", snapshot=lambda: deepcopy(state), runnable=lambda *a, **kw: state["deployments"]["deployment"]["status"] == "active"),
        _model_control_authority=lambda: {"term": 1}, _model_control_connections=lambda: {"worker": 1},
        _task_worker_control=SimpleNamespace(worker_snapshot=lambda _n: {"capabilities": {"stage_kv_cleanup": True, "node_incarnation": "worker-boot"}}),
        get_effective_node_id=lambda: "master", _execute_layer_stage_offer=offer,
        _require_callbacks=lambda: callbacks)
    requests = RequestService(repository[0], boot_id="boot")
    app = ExecutionApplication(scheduler, requests)
    yield app, calls, state
    requests.close()


@pytest.mark.parametrize("engine", ["llama_cpp", "pytorch"])
def test_production_entry_consumes_publication_without_planning_graph_or_full_load(application, engine):
    app, calls, _ = application
    app.scheduler._host._manager.engine_type = engine
    result = app.run("hello", max_new_tokens=2)
    assert result["response"] == "AA"
    assert result["execution_owner"] == "linear"
    assert result["metrics"]["request"]["outcome"] == "completed"
    assert result["metrics"]["request"]["execution_state"] == "stopped"
    assert [c[0] for c in calls] == ["prefix", "offer", "prefix", "offer", "offer", "local_cleanup"]
    assert calls[1][1]["positions"] == [0]
    assert calls[3][1]["positions"] == [1]
    assert calls[1][1]["assignment"]["worker_prepare_identity"]["assignment_id"] == "assignment"
    assert calls[-2][1]["cleanup_only"] is True


def test_production_sampling_refusal_precedes_tokenization_and_native_work(application):
    app, calls, _ = application
    result = app.run("hello", temperature=0.7, top_p=0.2)
    assert result["reason_code"] == "route_a_sampling_unsupported"
    assert not calls
    assert app.requests.snapshot()["requests"] == {}


def test_old_peer_missing_cleanup_capability_is_rejected_before_native_work(application):
    app, calls, _ = application
    app.scheduler._task_worker_control.worker_snapshot = lambda _n: {"capabilities": {"node_incarnation": "worker-boot"}}
    assert app.run("hello")["reason_code"] == "execution_kv_cleanup_unsupported"
    assert not calls


def test_retired_deployment_is_refused_without_implicit_whole_model(application):
    app, calls, state = application
    state["deployments"]["deployment"]["status"] = "retired"
    assert app.run("hello")["reason_code"] == "execution_deployment_unavailable"
    assert not calls


def test_stage_revalidates_authority_after_return_and_drops_stale_output(application):
    app, calls, state = application
    original = app.scheduler._execute_layer_stage_offer
    def stale(**kwargs):
        result = original(**kwargs)
        if not kwargs.get("cleanup_only"):
            state["deployments"]["deployment"]["status"] = "invalidated"
        return result
    app.scheduler._execute_layer_stage_offer = stale
    emitted = []
    result = app.run("hello", max_new_tokens=1, _stream_callback=emitted.append)
    assert result["reason_code"] == "execution_deployment_fenced"
    assert emitted == []


def test_native_cancel_returns_terminal_but_unload_waits_for_native(application):
    app, calls, _ = application
    entered, release = threading.Event(), threading.Event()
    manager = app.scheduler._host._manager
    def forward(**_kwargs):
        entered.set()
        assert release.wait(4)
        return {"hidden_states": np.zeros((1, 1, 2), dtype=np.float32), "cache": 1}
    manager.forward_layers = forward
    manager.unload_model = lambda: calls.append(("unload",))
    cancel = threading.Event()
    result = {}
    request = threading.Thread(target=lambda: result.update(app.run("hello", max_new_tokens=1, _cancel_event=cancel)))
    request.start()
    assert entered.wait(3)
    cancel.set()
    request.join(2)
    assert result["reason_code"] == "generation_cancelled"
    unload = threading.Thread(target=app.scheduler._host.unload_model)
    unload.start()
    wait_for(lambda: unload.is_alive())
    assert not any(c[0] == "unload" for c in calls)
    release.set()
    unload.join(3)
    assert not unload.is_alive()
    assert [c[0] for c in calls][-2:] == ["local_cleanup", "unload"]


def test_http_default_pipeline_sampling_is_greedy_but_explicit_parameters_are_preserved():
    from api_server import ChatRequest
    assert ChatRequest(message="hello").pipeline_sampling() == (0.0, 1.0)
    assert ChatRequest(message="hello", temperature=0.7).pipeline_sampling() == (0.7, 0.9)
    assert ChatRequest(message="hello", temperature=0).pipeline_sampling() == (0.0, 0.9)


def test_runtime_instances_keep_request_and_native_owners_isolated(tmp_path):
    from qlh_core.bootstrap.runtime import AppRuntime
    from qlh_core.contracts.runtime import RuntimeConfig
    from tests.test_refactor_bootstrap import settings
    first = AppRuntime(settings(tmp_path / "one"))
    second = AppRuntime(settings(tmp_path / "two"))
    try:
        assert first.services.requests is not second.services.requests
        assert first.services.runtime_controller is not second.services.runtime_controller
        first.services.requests.pause()
        assert second.services.requests.snapshot()["paused"] is False
    finally:
        first.close()
        second.close()


def test_pc_worker_cleanup_returns_receipt_after_reset_and_ignores_old_registry(gguf, monkeypatch):
    host, value, selected = gguf
    host.prepare_worker_runtime(command("prepare", value), selected, threading.Event())
    host.commit_worker_runtime(value)
    active = host._worker_runtime_active
    before = active.upstream.resets
    monkeypatch.setattr(active.upstream, "forward_hidden_to_hidden", lambda *_a, **_kw: pytest.fail("cleanup ran layers"))
    import model_supply_worker
    monkeypatch.setattr(model_supply_worker, "get_selected_model", lambda: pytest.fail("cleanup consulted registry"))
    request = gguf_request(value)
    request.stage_type = "layer_forward"
    request.root_input["cleanup_only"] = True
    host._supplied_layer_caches = {0: (object(), 2)}
    assert host.execute_task_worker_stage(request, threading.Event()) == {"kv_cleared": True}
    assert active.upstream.resets == before + 1
    assert host._supplied_layer_caches == {}


def test_pc_worker_cleanup_failure_propagates_without_receipt(gguf, monkeypatch):
    host, value, selected = gguf
    host.prepare_worker_runtime(command("prepare", value), selected, threading.Event())
    host.commit_worker_runtime(value)
    request = gguf_request(value)
    request.stage_type = "layer_forward"
    request.root_input["cleanup_only"] = True
    monkeypatch.setattr(host._worker_runtime_active.upstream, "reset", lambda: (_ for _ in ()).throw(OSError("native reset failed")))
    with pytest.raises(OSError, match="native reset failed"):
        host.execute_task_worker_stage(request, threading.Event())


def test_worker_incarnation_change_invalidates_frozen_attempt_before_forward(application):
    app, calls, _ = application
    placement = app.freeze()
    app.scheduler._task_worker_control.worker_snapshot = lambda _n: {
        "capabilities": {"stage_kv_cleanup": True, "node_incarnation": "restarted-worker"}}
    assert app.current(placement) is False
    assert not calls


def full_boundary(name, app, monkeypatch):
    import api_server
    import inference_service.engine_host as engine_module
    from inference_service.protocol import ChatRequest as ServiceChatRequest
    scheduler = app.scheduler
    scheduler._execution_app = app
    scheduler.get_distributed_inference_enabled = lambda: True
    scheduler._effective_role = lambda: "master"
    scheduler.record_task_complete = lambda **_kw: None
    prepared = SimpleNamespace(session_id="temporary", history=[
        {"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}], revision=7)
    committed, captured = [], []
    def pipeline(prompt, **options):
        captured.append(options)
        return app.run(prompt, **options)
    scheduler.run_pipeline_safe = pipeline
    app.scheduler._host._manager.chat = lambda **_kw: pytest.fail("linear result invoked native full chat")
    app.scheduler._host._manager.ensure_full_model = lambda: pytest.fail("linear result restored a full model")
    decision = lambda _req: SimpleNamespace(use_external=False, reason="")
    commit = lambda *args, **kwargs: committed.append((args, kwargs))
    if name == "api":
        for key, value in {"scheduler": scheduler, "model_host": scheduler._host, "model_manager": scheduler._host,
            "RUN_MODE": "distributed", "_routing_gate_error": lambda _req: None,
            "_prepare_chat_context": lambda *_a, **_kw: prepared, "_external_route_decision": decision,
            "_pipeline_worker_is_reserved": lambda: False, "_pipeline_model_is_prepared": lambda: False,
            "runtime_supports": lambda *_a: True, "_augment_chat_metrics": lambda metrics, *_a, **_kw: dict(metrics),
            "_commit_chat_context_turn": commit, "conversation_stats": {"rounds": 0},
            "_generate_followups_llama": lambda *_a: pytest.fail("linear result invoked native followups")}.items():
            monkeypatch.setattr(api_server, key, value)
        return api_server._execute_chat_full, api_server.ChatRequest, committed, captured
    host = engine_module.EngineHost()
    host._host, host._scheduler, host._run_mode = scheduler._host, scheduler, "distributed"
    host._routing_gate_error = lambda _req: None
    host._prepare_chat_context = lambda *_a, **_kw: prepared
    host._external_route_decision = decision
    host._pipeline_worker_is_reserved = lambda: False
    host._commit_chat_context_turn = commit
    host._record_task_complete = lambda: None
    host.generate_followups_llama = lambda *_a: pytest.fail("linear result invoked native followups")
    monkeypatch.setattr(engine_module, "runtime_supports", lambda *_a: True)
    monkeypatch.setattr(engine_module, "_augment_chat_metrics", lambda metrics, *_a, **_kw: dict(metrics))
    return host.chat_full, ServiceChatRequest, committed, captured


@pytest.mark.parametrize("boundary", ["api", "service"])
@pytest.mark.parametrize("scenario", ["completed", "empty_eos", "sampling_refused", "unexpected_error", "context_conflict"])
def test_full_product_boundaries_keep_linear_terminal_and_never_load_whole_model(application, monkeypatch, boundary, scenario):
    from fastapi import HTTPException
    app, calls, _ = application
    run, request_type, committed, captured = full_boundary(boundary, app, monkeypatch)
    options = {"temperature": 0.7} if scenario == "sampling_refused" else {}
    if scenario == "empty_eos":
        original = app.scheduler._execute_layer_stage_offer
        app.scheduler._execute_layer_stage_offer = lambda **kw: (original(**kw) if kw.get("cleanup_only")
                                                                else {"kind": "token", "token_argmax": 999})
    elif scenario == "unexpected_error":
        app.scheduler.run_pipeline_safe = lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("transport failed"))
    elif scenario == "context_conflict":
        def conflict(*_a, **_kw):
            raise ValueError("conversation revision changed")
        if boundary == "api":
            import api_server
            monkeypatch.setattr(api_server, "_commit_chat_context_turn", conflict)
        else:
            run.__self__._commit_chat_context_turn = conflict
    request = request_type(message="hello", max_new_tokens=1)
    if options:
        request = request_type(message="hello", max_new_tokens=1, **options)
    if scenario in {"sampling_refused", "unexpected_error", "context_conflict"}:
        with pytest.raises(HTTPException) as raised:
            run(request, request_deadline=RequestDeadline.start(3))
        assert raised.value.status_code == 503
        assert raised.value.error_code == ("route_a_sampling_unsupported" if scenario == "sampling_refused" else "request_execution_failed")
        assert committed == []
    else:
        result = run(request, request_deadline=RequestDeadline.start(3))
        assert result["content"] == ("" if scenario == "empty_eos" else "A")
        assert result["metrics"]["request"]["outcome"] == "completed"
        assert committed[0][1]["expected_revision"] == 7
    if captured:
        assert captured[0]["temperature"] == (0.7 if options else 0.0)
        assert captured[0]["_context_revision"] == 7
    if scenario == "sampling_refused":
        assert calls == []


@pytest.mark.parametrize("boundary", ["api", "service"])
@pytest.mark.parametrize("explicit", [False, True])
def test_forwarded_product_request_preserves_explicit_sampling_and_projects_default_greedy(application, monkeypatch, boundary, explicit):
    app, _, _ = application
    run, request_type, committed, _ = full_boundary(boundary, app, monkeypatch)
    app.scheduler._effective_role = lambda: "client"
    forwarded = []
    app.scheduler.forward_inference_to_master = lambda **kw: forwarded.append(kw) or {"status": "ok", "content": "forwarded", "metrics": {}}
    request = request_type(message="hello", **({"temperature": 0.7} if explicit else {}))
    assert run(request)["content"] == "forwarded"
    assert (forwarded[0]["temperature"], forwarded[0]["top_p"]) == ((0.7, 0.9) if explicit else (0.0, 1.0))
    assert len(committed) == 1


def test_primary_chat_published_execution_skips_outer_model_lock(application, monkeypatch):
    import asyncio
    import api_server
    from api import routes_chat
    app, _, _ = application
    _, request_type, _, _ = full_boundary("api", app, monkeypatch)
    monkeypatch.setattr(api_server, "hold_lock_until_deadline", lambda *_a, **_kw: pytest.fail("FIFO admission waited for outer native lock"))
    monkeypatch.setattr(api_server, "_maybe_log_external_scope_denial", lambda *_a: None)
    monkeypatch.setattr(api_server, "_execute_requested_chat", api_server._execute_chat_full)
    result = asyncio.run(routes_chat.chat(request_type(message="hello", max_new_tokens=1)))
    assert result.content == "A"
    assert result.metrics["request"]["outcome"] == "completed"


def test_runtime_close_timeout_preserves_native_owner_until_requests_physically_stop(application, monkeypatch):
    from qlh_core.bootstrap.services import RuntimeServices
    app, _, _ = application
    entered, release = threading.Event(), threading.Event()
    app.scheduler._host._manager.forward_layers = lambda **_kw: (entered.set(), release.wait(4),
        {"hidden_states": np.zeros((1, 1, 2), dtype=np.float32), "cache": 1})[-1]
    result = {}
    producer = threading.Thread(target=lambda: result.update(app.run("hello", max_new_tokens=1)))
    producer.start()
    assert entered.wait(3)
    services = object.__new__(RuntimeServices)
    services._stage_host_lock = threading.RLock()
    services._closed = services._closing = False
    services._owns_host = True
    services._stage_host = services.auto_role = services.journal = None
    services.host = app.scheduler._host
    services.scheduler = SimpleNamespace(stop=lambda: app.requests.close(timeout=0))
    unloaded = []
    monkeypatch.setattr(ModelHost, "unload_model", lambda _host: unloaded.append(True))
    try:
        with pytest.raises(ExceptionGroup, match="runtime cleanup failed") as raised:
            services.close()
        assert isinstance(raised.value.exceptions[0], TimeoutError)
        assert not services._closed and services._closing
        assert unloaded == []
        producer.join(2)
        assert result["reason_code"] == "generation_cancelled"
    finally:
        release.set()
        producer.join(3)
    wait_for(lambda: app.requests.snapshot()["active"] is None)
    app.requests.close()
    services.close()
    assert services._closed and unloaded == [True]


@pytest.mark.parametrize("suffix,generated,expected", [
    ("<think>\n\n</think>\n\n", [42, 999], "A"),
    ("", [42, 999], "A"),
    ("<think>\n", [42, 999], ""),
    ("<think>\n", [42, 43, 999], "B"),
])
def test_linear_closed_prompt_thinking_preserves_answer_and_stream(application, monkeypatch, suffix, generated, expected):
    from api_server import _format_model_response
    app, calls, _ = application
    callbacks = app.scheduler._require_callbacks()
    callbacks.build_model_chat_prompt = lambda *_a, **_kw: "<|im_start|>assistant\n" + suffix
    callbacks.format_model_response = _format_model_response
    manager = app.scheduler._host._manager
    if suffix == "<think>\n":
        manager.tokenizer.decode = lambda tokens, **_kw: "".join(
            {42: "private", 43: "</think>B", 999: ""}[int(token)] for token in tokens)
    tokens = iter(generated)
    def offer(**kwargs):
        if kwargs.get("cleanup_only"):
            return {"kv_cleared": True, "kind": "cleanup"}
        return {"kind": "token", "token_argmax": next(tokens)}
    monkeypatch.setattr(app.scheduler, "_execute_layer_stage_offer", offer)
    events = []
    result = app.run("hello", max_new_tokens=8, enable_thinking=False,
                     _stream_callback=events.append)
    assert result["response"] == expected
    assert "".join(event.get("token", "") for event in events) == expected
    assert result["metrics"]["stop_reason"] == "eos_token"
    assert result["metrics"]["request"]["execution_state"] == "stopped"


@pytest.mark.parametrize("tensor_kind", ["numpy", "torch"])
def test_linear_three_nodes_passes_real_wire_decoded_middle_to_tail(tmp_path, monkeypatch, tensor_kind):
    import base64
    import hashlib
    import scheduler_pipeline as pipeline
    from tests.helpers.linear_application import make_application
    if tensor_kind == "torch":
        import torch
        monkeypatch.setattr(pipeline, "loaded_torch", lambda: torch)
    else:
        monkeypatch.setattr(pipeline, "loaded_torch", lambda: None)
    wire_hidden = np.arange(4, dtype=np.float32).reshape(1, 4)
    raw = wire_hidden.tobytes()
    wire_output = {"hidden_out_f32": base64.b64encode(raw).decode("ascii"),
                   "hidden_out_sha256": hashlib.sha256(raw).hexdigest()}
    seen = []
    def offer(**kwargs):
        if kwargs.get("cleanup_only"):
            return {"kind": "cleanup", "kv_cleared": True}
        seen.append(kwargs["node_id"])
        if kwargs["node_id"] == "middle":
            return pipeline._layer_stage_result_to_pipeline_value(
                wire_output, hidden_spec={"n_tokens": 1, "n_embd": 4})
        np.testing.assert_array_equal(kwargs["hidden_states"], wire_hidden)
        return {"kind": "token", "token_argmax": 42}
    manager = SimpleNamespace(tokenizer=Tokenizer(), engine_type="llama_cpp",
        forward_layers=lambda **_kw: {"hidden_states": np.zeros((1, 1, 4), dtype=np.float32), "cache": 1},
        clear_pipeline_kv=lambda: None)
    callbacks = SimpleNamespace(thinking_system_prompt="",
        build_model_chat_prompt=lambda *_a, **_kw: "prompt",
        format_model_response=lambda text, *_a, **_kw: (text, None))
    app, _, _ = make_application(tmp_path / "requests.sqlite3", manager, offer, callbacks)
    try:
        result = app.run("hello", max_new_tokens=1)
        assert result.get("response") == "A", result
        assert seen == ["middle", "tail"]
        assert result["metrics"]["workers_used"] == ["middle", "tail"]
        assert result["metrics"]["request"]["execution_state"] == "stopped"
    finally:
        app.requests.close()


def test_linear_records_physical_stage_completion_before_provider_cache_expires(application, monkeypatch):
    from scheduler_pipeline import SchedulerPipelineMixin
    app, _, _ = application
    outputs = iter(({"token_argmax": 42}, {"token_argmax": 999}, {"kv_cleared": True}))
    queries = []
    provider = SimpleNamespace(reserve=lambda _r: SimpleNamespace(reservation_id="res_fixture"),
        execute=lambda *_a: SimpleNamespace(output=next(outputs)), release=lambda _r: None,
        execution_drained=lambda attempt: queries.append(attempt) or False)
    # The provider has already discarded its old successful result cache. A
    # validated synchronous StageResult is the physical completion proof.
    monkeypatch.setattr(app.scheduler, "_execute_layer_stage_offer",
        lambda **kwargs: SchedulerPipelineMixin._execute_layer_stage_offer(app.scheduler, **kwargs))
    app.scheduler._worker_prepare_assignment_loaded = lambda *_a: (True, "")
    app.scheduler._ensure_remote_task_worker_provider = lambda _n: provider
    result = app.run("hello", max_new_tokens=3, _request_timeout_seconds=1)
    assert result.get("response") == "A", result
    assert queries == []
    assert result["metrics"]["request"]["outcome"] == "completed"
    assert result["metrics"]["request"]["execution_state"] == "stopped"


def test_production_linear_completion_is_allowed_in_frozen_release_profile(application, monkeypatch):
    from distributed_completion import validate_distributed_completion
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")
    app, _, _ = application
    result = app.run("hello", max_new_tokens=2)
    assert result["metrics"]["execution_mode"] == "distributed_pipeline"
    assert result["metrics"]["request"]["execution_state"] == "stopped"
    validate_distributed_completion("distributed_required", result["metrics"])
