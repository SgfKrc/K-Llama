import asyncio
import os
import sys
import threading
import time
from typing import Any, cast

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from model_host import model_host
from task_graph import StageSpec, TaskGraphCoordinator, TaskGraphUnavailable
from task_journal import JournalEvent, SQLiteTaskJournal
from task_provider import (
    LocalFullModelProvider,
    ModelIdentity,
    ProviderBusy,
    ProviderExecutionError,
    ProviderRegistry,
    StageRequest,
)
from task_worker_adapter import RemoteFullWorkerProvider
from task_worker_protocol import build_message, canonical_sha256


class FakeTaskGraphModelManager:
    is_loaded = True
    _engine_type = "llama_cpp"
    active_model_id = "fake-model"
    model = None

    def __init__(self):
        self.calls = []

    def chat(self, messages, max_tokens, temperature, top_p, **kwargs):
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "cancel_event": kwargs.get("_cancel_event"),
        })
        call_no = len(self.calls)
        content = {
            1: "candidate A",
            2: "candidate B",
            3: "final answer",
        }[call_no]
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "tokens_per_second": 2.5,
            "model": "fake-model",
        }


@pytest.fixture
def task_graph_api(monkeypatch):
    manager = FakeTaskGraphModelManager()
    coordinator = TaskGraphCoordinator(max_records=20)
    monkeypatch.setattr(api_server, "model_manager", manager)
    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(api_server, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    monkeypatch.setattr(
        sys.modules["scheduler"],
        "TASK_WORKER_EXPERIMENTAL_ENABLED",
        True,
    )
    monkeypatch.setattr(api_server, "task_graph_coordinator", coordinator)
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")
    monkeypatch.setattr(
        api_server.scheduler, "get_effective_node_id", lambda: "test-master",
    )
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: False,
    )
    monkeypatch.setattr(
        api_server.scheduler, "record_task_complete", lambda **kwargs: True,
    )
    monkeypatch.setattr(api_server, "_auto_title_session", lambda *a, **k: None)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(
        api_server._local_store, "save_local_message", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        api_server._local_store,
        "save_local_conversation_turn",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        api_server._local_store,
        "increment_local_session_message_count",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        api_server._local_store,
        "clear_local_conversation",
        lambda *a, **k: 0,
    )
    monkeypatch.setattr(api_server, "active_session_id", "task-session")
    monkeypatch.setattr(api_server, "session_histories", {"task-session": []})
    monkeypatch.setattr(
        api_server,
        "conversation_stats",
        {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        },
    )
    return manager, coordinator


def test_task_graph_config_can_open_local_experiment_without_physical_worker(
    task_graph_api, monkeypatch,
):
    """The UI switch controls the experiment gate, not physical admission."""
    import config as config_module
    import scheduler as scheduler_module
    monkeypatch.setattr(config_module, "TASK_GRAPH_ENABLED", config_module.TASK_GRAPH_ENABLED)
    monkeypatch.setattr(
        config_module,
        "TASK_WORKER_EXPERIMENTAL_ENABLED",
        config_module.TASK_WORKER_EXPERIMENTAL_ENABLED,
    )
    monkeypatch.setattr(
        scheduler_module,
        "TASK_WORKER_EXPERIMENTAL_ENABLED",
        scheduler_module.TASK_WORKER_EXPERIMENTAL_ENABLED,
    )
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", False)
    monkeypatch.setattr(api_server, "TASK_WORKER_EXPERIMENTAL_ENABLED", False)
    monkeypatch.setattr(api_server, "load_node_config", lambda: {})
    saved = {}
    monkeypatch.setattr(api_server, "write_node_config", lambda data: saved.update(data))
    monkeypatch.setattr(
        api_server.scheduler,
        "get_task_worker_protocol_status",
        lambda: {
            "admission_state": "n2_4_experiment_enabled_not_connected",
        },
    )

    result = asyncio.run(api_server.set_task_graph_config(
        api_server.TaskGraphConfigRequest(
            enabled=True,
            worker_experimental_enabled=True,
        )
    ))

    assert result["task_graph_enabled"] is True
    assert result["task_worker_experimental_enabled"] is True
    assert result["physical_validation_pending"] is False
    live_scheduler_module = sys.modules[
        type(api_server.scheduler).__module__
    ]
    assert live_scheduler_module.TASK_WORKER_EXPERIMENTAL_ENABLED is True
    assert saved["features"] == {
        "task_graph_enabled": True,
        "task_worker_experimental_enabled": True,
    }


def test_task_graph_config_is_master_only(monkeypatch):
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "client")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.set_task_graph_config(
            api_server.TaskGraphConfigRequest(enabled=True)
        ))
    assert exc_info.value.status_code == 403


def _remote_worker_provider(
    identity: ModelIdentity,
    node_id: str,
    *,
    content: str = "remote candidate",
    error_code: str = "",
) -> RemoteFullWorkerProvider:
    holder = {}

    def peer_snapshot():
        return {
            "node_id": node_id,
            "healthy": True,
            "selected_version": 2,
            "manual_stage_dispatch_enabled": True,
            "capabilities": {
                "stage_types": ["full_inference", "aggregate"],
                "engines": [identity.engine],
                "models": [identity.snapshot()],
                "max_concurrency": 1,
            },
        }

    def respond(message):
        if message.message_type != "stage_offer":
            return
        offer = message.payload
        response_identity = {
            key: offer[key]
            for key in (
                "workflow_id", "stage_id", "attempt_id", "lease_id",
                "lease_epoch", "provider_id",
            )
        }
        provider = holder["provider"]
        provider.handle_message(build_message(
            "stage_accept",
            {
                **response_identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id=f"msg_accept_{node_id}_{offer['attempt_id']}",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        if error_code:
            provider.handle_message(build_message(
                "stage_error",
                {
                    **response_identity,
                    "error_code": error_code,
                    "retryable": True,
                },
                message_id=f"msg_error_{node_id}_{offer['attempt_id']}",
                sent_at_ms=int(time.time() * 1000),
                version=2,
            ).snapshot())
            return
        output = {
            "content": f"{content}:{offer['stage_id']}",
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
            "model": identity.model_id,
        }
        provider.handle_message(build_message(
            "stage_result",
            {
                **response_identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {
                    "usage": output["usage"],
                    "usage_estimated": False,
                    "model": identity.model_id,
                },
            },
            message_id=f"msg_result_{node_id}_{offer['attempt_id']}",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

    provider = RemoteFullWorkerProvider(
        node_id=node_id,
        peer_snapshot=peer_snapshot,
        send_message=respond,
    )
    holder["provider"] = provider
    return provider


def test_task_graph_chat_runs_three_local_stages_with_honest_metrics(task_graph_api):
    manager, coordinator = task_graph_api
    req = api_server.ChatRequest(
        message="question",
        session_id="task-session",
        max_new_tokens=64,
        execution_mode="task_graph",
        workflow_id="wf_api12345",
    )

    response = asyncio.run(api_server.chat(req))

    assert response.content == "final answer"
    assert len(manager.calls) == 3
    assert all(call["cancel_event"] is not None for call in manager.calls)
    metrics = response.metrics
    assert metrics["execution_mode"] == "task_graph"
    assert metrics["provider"] == "local_full_model"
    assert metrics["orchestrator"] == "task_graph"
    assert metrics["subproviders"] == ["local_full_model"]
    assert metrics["workflow_id"] == "wf_api12345"
    assert metrics["workflow_state"] == "completed"
    assert metrics["partial_result"] is False
    assert metrics["stage_count"] == 3
    assert metrics["stage_attempt_count"] == 3
    assert metrics["nodes_planned"] == 1
    assert metrics["nodes_participated"] == 1
    assert metrics["participating_nodes"] == ["test-master"]
    assert metrics["distributed_used"] is False
    assert metrics["distributed_kind"] == "task_graph_local_poc"
    assert metrics["total_tokens"] == 45

    workflow = coordinator.get("wf_api12345")
    assert workflow["state"] == "completed"
    assert workflow["completed_stage_count"] == 3
    attempts = [
        attempt
        for stage in workflow["stages"]
        for attempt in stage["attempts"]
    ]
    assert {attempt["provider"] for attempt in attempts} == {
        "local_full_model",
    }
    assert all(attempt["result_metadata"]["usage"] for attempt in attempts)
    history = api_server.session_histories["task-session"]
    assert history == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "final answer"},
    ]


def test_task_graph_chat_reports_partial_candidate_degradation(
    task_graph_api,
    monkeypatch,
):
    manager, coordinator = task_graph_api
    prompts = []

    def partial_chat(messages, max_tokens, temperature, top_p, **kwargs):
        prompts.append(messages)
        call_no = len(prompts)
        if call_no == 1:
            raise RuntimeError("candidate A failed with private detail")
        content = "candidate B" if call_no == 2 else "partial final answer"
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "tokens_per_second": 2.5,
            "model": "fake-model",
        }

    monkeypatch.setattr(manager, "chat", partial_chat)
    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="question",
        session_id="task-session",
        max_new_tokens=64,
        execution_mode="task_graph",
        workflow_id="wf_apipartial1",
    )))

    assert response.content == "partial final answer"
    assert response.metrics["partial_result"] is True
    assert response.metrics["workflow_state"] == "completed"
    assert "未完成候选摘要" in prompts[-1][-1]["content"]
    assert "private detail" not in prompts[-1][-1]["content"]
    workflow = coordinator.get("wf_apipartial1")
    assert workflow["state"] == "completed"
    assert workflow["partial_result"] is True
    assert workflow["failed_stage_count"] == 1
    assert workflow["completed_stage_count"] == 2


def test_task_graph_aggregate_retry_is_not_reported_as_fallback(
    task_graph_api,
    monkeypatch,
):
    manager, coordinator = task_graph_api
    calls = 0

    def retry_aggregate(messages, max_tokens, temperature, top_p, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("transient aggregate timeout")
        content = (
            "candidate A" if calls == 1
            else "candidate B" if calls == 2
            else "recovered final answer"
        )
        return {
            "content": content,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "tokens_per_second": 2.5,
            "model": "fake-model",
        }

    monkeypatch.setattr(manager, "chat", retry_aggregate)
    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="question",
        session_id="task-session",
        max_new_tokens=64,
        execution_mode="task_graph",
        workflow_id="wf_apiaggretry",
    )))

    assert response.content == "recovered final answer"
    assert response.metrics["fallback"] is False
    assert response.metrics["fallback_reason"] == ""
    assert response.metrics["stage_retry_count"] == 1
    assert response.metrics["same_provider_retry_count"] == 1
    assert response.metrics["reassignment_count"] == 0
    assert response.metrics["retry_reason"] == "provider_execution_failed"
    workflow = coordinator.get("wf_apiaggretry")
    assert workflow["partial_result"] is False
    assert workflow["same_provider_retry_count"] == 1


def test_aggregate_runtime_error_is_not_marked_for_same_provider_retry(
    task_graph_api,
    monkeypatch,
):
    manager, _coordinator = task_graph_api
    monkeypatch.setattr(
        manager,
        "chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("permanent aggregate runtime failure")
        ),
    )
    request = StageRequest(
        workflow_id="wf_apipermanent1",
        request_id="request-api-permanent",
        stage_id="aggregate",
        stage_type="aggregate",
        provider_id="local_full_model",
        dependencies={"candidate_a": {"content": "candidate"}},
        root_input={"message": "question", "task_options": {}},
    )

    with pytest.raises(RuntimeError, match="permanent aggregate runtime failure"):
        api_server._execute_task_worker_stage(request, threading.Event())


def test_empty_aggregate_output_is_explicitly_same_provider_retryable(
    task_graph_api,
    monkeypatch,
):
    manager, _coordinator = task_graph_api
    monkeypatch.setattr(
        manager,
        "chat",
        lambda *args, **kwargs: {"content": "", "usage": {}},
    )
    request = StageRequest(
        workflow_id="wf_apiemptyagg1",
        request_id="request-api-empty-aggregate",
        stage_id="aggregate",
        stage_type="aggregate",
        provider_id="local_full_model",
        dependencies={"candidate_a": {"content": "candidate"}},
        root_input={"message": "question", "task_options": {}},
    )

    with pytest.raises(ProviderExecutionError) as captured:
        api_server._execute_task_worker_stage(request, threading.Event())

    assert captured.value.code == "empty_provider_output"
    assert captured.value.retryable is False
    assert captured.value.same_provider_retryable is True


def test_task_graph_manual_remote_stage_uses_explicit_provider_without_fallback(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="3" * 64,
    )
    provider_holder = {}

    def peer_snapshot():
        return {
            "node_id": "worker_01",
            "healthy": True,
            "selected_version": 2,
            "manual_stage_dispatch_enabled": True,
            "capabilities": {
                "stage_types": ["full_inference", "aggregate"],
                "engines": ["llama_cpp"],
                "models": [identity.snapshot()],
                "max_concurrency": 1,
            },
        }

    def respond(message):
        offer = message.payload
        response_identity = {
            key: offer[key]
            for key in (
                "workflow_id", "stage_id", "attempt_id", "lease_id",
                "lease_epoch", "provider_id",
            )
        }
        provider = provider_holder["provider"]
        provider.handle_message(build_message(
            "stage_accept",
            {
                **response_identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_apiaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        output = {
            "content": "remote candidate",
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
            "model": "fake-model",
        }
        provider.handle_message(build_message(
            "stage_result",
            {
                **response_identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {
                    "usage": output["usage"],
                    "usage_estimated": False,
                    "model": "fake-model",
                },
            },
            message_id="msg_apiresult01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=peer_snapshot,
        send_message=respond,
    )
    provider_holder["provider"] = provider
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity,
    )
    req = api_server.ChatRequest(
        message="question",
        session_id="task-session",
        max_new_tokens=64,
        execution_mode="task_graph",
        workflow_id="wf_apiremote01",
        task_graph_remote_stage="candidate_a",
        task_graph_remote_provider_id=provider.provider_id,
    )

    response = asyncio.run(api_server.chat(req))

    assert len(manager.calls) == 2
    assert response.metrics["distributed_requested"] is True
    assert response.metrics["distributed_used"] is True
    assert response.metrics["distributed_kind"] == "task_graph_remote_manual"
    assert response.metrics["workers_used"] == ["worker_01"]
    assert response.metrics["nodes_planned"] == 2
    assert response.metrics["nodes_participated"] == 2
    assert response.metrics["manual_remote_stage"] == "candidate_a"
    assert response.metrics["fallback"] is False
    workflow = coordinator.get("wf_apiremote01")
    candidate = next(
        stage for stage in workflow["stages"]
        if stage["stage_id"] == "candidate_a"
    )
    assert candidate["requested_provider"] == provider.provider_id
    assert candidate["fallback_providers"] == []
    assert candidate["attempts"][0]["provider_node_id"] == "worker_01"


def test_task_graph_manual_remote_stage_rejects_wrong_model_before_run(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    active_identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="a" * 64,
    )
    wrong_identity = ModelIdentity(
        model_id="other-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-other",
        sha256="b" * 64,
    )
    provider = _remote_worker_provider(wrong_identity, "worker_01")
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    monkeypatch.setattr(
        api_server,
        "_active_task_graph_model_identity",
        lambda: active_identity,
    )
    request = api_server.ChatRequest(
        message="wrong manual model",
        session_id="task-session",
        execution_mode="task_graph",
        workflow_id="wf_manualwrongmodel01",
        task_graph_remote_stage="candidate_a",
        task_graph_remote_provider_id=provider.provider_id,
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(api_server.chat(request))

    assert captured.value.status_code == 409
    assert captured.value.detail["reason_code"] == "model_identity_mismatch"
    assert captured.value.detail["provider_id"] == provider.provider_id
    assert manager.calls == []
    assert coordinator.list() == []


def test_task_graph_manual_remote_stage_requires_both_explicit_fields(
    task_graph_api,
):
    req = api_server.ChatRequest(
        message="question",
        execution_mode="task_graph",
        task_graph_remote_stage="candidate_a",
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(api_server.chat(req))

    assert captured.value.status_code == 400
    assert captured.value.error_code == "TASK_GRAPH_MANUAL_REMOTE_FIELDS_INCOMPLETE"


def test_registered_remote_provider_is_never_selected_without_explicit_fields(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="4" * 64,
    )
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: {
            "node_id": "worker_01",
            "healthy": True,
            "selected_version": 2,
            "manual_stage_dispatch_enabled": True,
            "capabilities": {
                "stage_types": ["full_inference", "aggregate"],
                "engines": ["llama_cpp"],
                "models": [identity.snapshot()],
                "max_concurrency": 1,
            },
        },
        send_message=lambda _message: pytest.fail(
            "remote Provider must not receive automatic Stage work"
        ),
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    listed = asyncio.run(api_server.list_workflows(limit=5))
    assert provider.provider_id in listed["providers"]

    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="local only",
        session_id="task-session",
        execution_mode="task_graph",
        workflow_id="wf_noautoremt01",
    )))

    assert response.metrics["distributed_used"] is False
    assert response.metrics["subproviders"] == ["local_full_model"]
    assert len(manager.calls) == 3
    workflow = coordinator.get("wf_noautoremt01")
    assert {
        attempt["provider"]
        for stage in workflow["stages"]
        for attempt in stage["attempts"]
    } == {"local_full_model"}


def test_task_graph_auto_remote_splits_one_candidate_to_one_worker(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="5" * 64,
    )
    provider = _remote_worker_provider(identity, "worker_01")
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity,
    )

    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="auto split",
        session_id="task-session",
        execution_mode="task_graph",
        task_graph_auto_remote=True,
        workflow_id="wf_autooneworker01",
    )))

    assert len(manager.calls) == 2
    assert response.metrics["distributed_requested"] is True
    assert response.metrics["distributed_used"] is True
    assert response.metrics["distributed_kind"] == "task_graph_remote_auto"
    assert response.metrics["workers_used"] == ["worker_01"]
    assert response.metrics["auto_remote_enabled"] is True
    assert response.metrics["auto_remote_providers"] == [provider.provider_id]
    assert response.metrics["fallback"] is False
    workflow = coordinator.get("wf_autooneworker01")
    stages = {stage["stage_id"]: stage for stage in workflow["stages"]}
    assert stages["candidate_a"]["requested_provider"] == provider.provider_id
    assert stages["candidate_a"]["fallback_providers"] == [
        "local_full_model"
    ]
    assert stages["candidate_a"]["pure"] is True
    assert stages["candidate_b"]["requested_provider"] == "local_full_model"
    assert stages["candidate_b"]["pure"] is True
    assert stages["aggregate"]["requested_provider"] == "local_full_model"
    assert stages["aggregate"]["pure"] is False


def test_task_graph_distributed_required_auto_assigns_remote_worker(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="d" * 64,
    )
    provider = _remote_worker_provider(identity, "worker_01")
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity,
    )

    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="required task split",
        session_id="task-session",
        execution_mode="task_graph",
        routing_preference="distributed_required",
        workflow_id="wf_requiredtaskauto01",
    )))

    assert response.metrics["distributed_used"] is True
    assert response.metrics["distributed_kind"] == "task_graph_remote_auto"
    assert response.metrics["workers_used"] == ["worker_01"]
    workflow = coordinator.get("wf_requiredtaskauto01")
    stages = {stage["stage_id"]: stage for stage in workflow["stages"]}
    assert stages["candidate_a"]["requested_provider"] == provider.provider_id
    assert stages["candidate_a"]["fallback_providers"] == []
    assert len(manager.calls) == 2


def test_release_profile_discards_task_graph_required_result_before_commit(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="f" * 64,
    )
    provider = _remote_worker_provider(identity, "worker_release")
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity,
    )
    monkeypatch.setenv("QLH_RELEASE_PROFILE_ENFORCE", "1")

    with pytest.raises(HTTPException) as captured:
        asyncio.run(api_server.chat(api_server.ChatRequest(
            message="release route policy",
            session_id="task-session",
            execution_mode="task_graph",
            routing_preference="distributed_required",
            workflow_id="wf_releasetaskroute01",
        )))

    assert captured.value.status_code == 503
    assert (
        captured.value.error_code
        == "DISTRIBUTED_REQUIRED_RELEASE_ROUTE_FORBIDDEN"
    )
    rejected = coordinator.get("wf_releasetaskroute01")
    assert rejected["state"] == "failed"
    assert rejected["cancel_requested"] is False
    assert rejected["terminal_reason"]["reason_code"] == "request_refused"
    assert len(manager.calls) == 2


def test_task_graph_distributed_required_rejects_without_eligible_worker(
    task_graph_api, monkeypatch,
):
    manager, _coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="e" * 64,
    )
    monkeypatch.setattr(
        api_server.scheduler, "remote_task_worker_providers", lambda: [],
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity,
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(api_server.chat(api_server.ChatRequest(
            message="required task no worker",
            session_id="task-session",
            execution_mode="task_graph",
            routing_preference="distributed_required",
            workflow_id="wf_requiredtasknone01",
        )))

    assert captured.value.status_code == 503
    assert captured.value.error_code == "TASK_WORKER_NO_ELIGIBLE_REMOTE_PROVIDER"
    assert manager.calls == []


def test_task_graph_auto_remote_distributes_candidates_across_two_workers(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="6" * 64,
    )
    providers = [
        _remote_worker_provider(identity, "worker_01", content="one"),
        _remote_worker_provider(identity, "worker_02", content="two"),
    ]
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: providers,
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity,
    )

    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="auto two workers",
        session_id="task-session",
        execution_mode="task_graph",
        task_graph_auto_remote=True,
        workflow_id="wf_autotwoworkers01",
    )))

    assert len(manager.calls) == 1
    assert response.metrics["workers_used"] == ["worker_01", "worker_02"]
    assert response.metrics["nodes_participated"] == 3
    assert response.metrics["fallback"] is False
    workflow = coordinator.get("wf_autotwoworkers01")
    stages = {stage["stage_id"]: stage for stage in workflow["stages"]}
    candidate_a = stages["candidate_a"]
    candidate_b = stages["candidate_b"]
    assert candidate_a["requested_provider"] == providers[0].provider_id
    assert candidate_b["requested_provider"] == providers[1].provider_id
    assert candidate_a["attempts"][0]["provider_node_id"] == "worker_01"
    assert candidate_b["attempts"][0]["provider_node_id"] == "worker_02"


def test_task_graph_auto_remote_filters_each_stage_by_its_model_identity(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    identity_a = ModelIdentity(
        model_id="qwen-1_8b",
        engine="llama_cpp",
        format="gguf",
        revision="stage-a",
        sha256="a" * 64,
    )
    identity_b = ModelIdentity(
        model_id="qwen-3b",
        engine="llama_cpp",
        format="gguf",
        revision="stage-b",
        sha256="b" * 64,
    )
    providers = [
        _remote_worker_provider(identity_a, "worker_a", content="a"),
        _remote_worker_provider(identity_b, "worker_b", content="b"),
    ]
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: providers,
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity_a,
    )
    monkeypatch.setattr(
        api_server,
        "dual_candidate_template",
        lambda: ([
            StageSpec(
                "candidate_a",
                "full_inference",
                model_identity=identity_a,
            ),
            StageSpec(
                "candidate_b",
                "full_inference",
                model_identity=identity_b,
            ),
            StageSpec(
                "aggregate",
                "aggregate",
                depends_on=("candidate_a", "candidate_b"),
                minimum_successful_dependencies=1,
            ),
        ], "aggregate"),
    )

    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="auto heterogeneous",
        session_id="task-session",
        execution_mode="task_graph",
        task_graph_auto_remote=True,
        workflow_id="wf_autohetero01",
    )))

    assert len(manager.calls) == 1
    workflow = coordinator.get("wf_autohetero01")
    stages = {stage["stage_id"]: stage for stage in workflow["stages"]}
    assert stages["candidate_a"]["requested_provider"] == providers[0].provider_id
    assert stages["candidate_b"]["requested_provider"] == providers[1].provider_id
    assert stages["candidate_a"]["model_identity"] == identity_a.snapshot()
    assert stages["candidate_b"]["model_identity"] == identity_b.snapshot()
    assert response.metrics["workers_used"] == ["worker_a", "worker_b"]


def test_task_graph_auto_remote_retries_pure_stage_on_local_provider(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="7" * 64,
    )
    provider = _remote_worker_provider(
        identity,
        "worker_01",
        error_code="remote_worker_disconnected",
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity,
    )

    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="fallback locally",
        session_id="task-session",
        execution_mode="task_graph",
        task_graph_auto_remote=True,
        workflow_id="wf_autofallback01",
    )))

    assert len(manager.calls) == 3
    assert response.metrics["distributed_requested"] is True
    assert response.metrics["distributed_used"] is False
    assert response.metrics["distributed_kind"] == "task_graph_local_fallback"
    assert response.metrics["nodes_planned"] == 2
    assert response.metrics["nodes_participated"] == 1
    assert response.metrics["fallback"] is True
    assert response.metrics["fallback_reason"] == "remote_worker_disconnected"
    workflow = coordinator.get("wf_autofallback01")
    candidate = next(
        stage for stage in workflow["stages"]
        if stage["stage_id"] == "candidate_a"
    )
    assert candidate["retry_count"] == 1
    assert candidate["last_retry_error_code"] == "remote_worker_disconnected"
    assert [attempt["provider"] for attempt in candidate["attempts"]] == [
        provider.provider_id, "local_full_model",
    ]
    assert candidate["attempts"][-1]["state"] == "completed"


def test_task_graph_auto_remote_reports_reservation_fallback_reason(
    task_graph_api, monkeypatch,
):
    manager, _coordinator = task_graph_api
    identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="a" * 64,
    )
    provider = _remote_worker_provider(identity, "worker_01")

    def reject_reservation(_request):
        raise ProviderBusy(
            "worker became busy",
            code="remote_worker_busy",
            provider_id=provider.provider_id,
            retryable=True,
        )

    monkeypatch.setattr(provider, "reserve", reject_reservation)
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    monkeypatch.setattr(
        api_server, "_active_task_graph_model_identity", lambda: identity,
    )

    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="reservation fallback",
        session_id="task-session",
        execution_mode="task_graph",
        task_graph_auto_remote=True,
        workflow_id="wf_autoreservefail01",
    )))

    assert len(manager.calls) == 3
    assert response.metrics["fallback"] is True
    assert response.metrics["fallback_reason"] == "remote_worker_busy"


def test_task_graph_auto_remote_filters_wrong_model_and_falls_back_local(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    active_identity = ModelIdentity(
        model_id="fake-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-test",
        sha256="8" * 64,
    )
    wrong_identity = ModelIdentity(
        model_id="other-model",
        engine="llama_cpp",
        format="gguf",
        revision="local-other",
        sha256="9" * 64,
    )
    provider = _remote_worker_provider(wrong_identity, "worker_01")
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [provider],
    )
    monkeypatch.setattr(
        api_server,
        "_active_task_graph_model_identity",
        lambda: active_identity,
    )

    response = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="wrong model",
        session_id="task-session",
        execution_mode="task_graph",
        task_graph_auto_remote=True,
        workflow_id="wf_autowrongmodel01",
    )))

    assert len(manager.calls) == 3
    assert response.metrics["distributed_used"] is False
    assert response.metrics["fallback"] is True
    assert response.metrics["fallback_reason"] == (
        "no_eligible_remote_provider"
    )
    assert response.metrics["auto_remote_providers"] == []
    workflow = coordinator.get("wf_autowrongmodel01")
    assert {
        attempt["provider"]
        for stage in workflow["stages"]
        for attempt in stage["attempts"]
    } == {"local_full_model"}


def test_task_graph_auto_remote_rejects_manual_remote_fields(task_graph_api):
    req = api_server.ChatRequest(
        message="conflicting policy",
        execution_mode="task_graph",
        task_graph_auto_remote=True,
        task_graph_remote_stage="candidate_a",
        task_graph_remote_provider_id="remote_worker_01",
    )

    with pytest.raises(HTTPException) as captured:
        asyncio.run(api_server.chat(req))

    assert captured.value.status_code == 400
    assert captured.value.error_code == "TASK_GRAPH_REMOTE_POLICY_CONFLICT"


def test_task_worker_experiment_gate_falls_back_auto_and_rejects_manual(
    task_graph_api, monkeypatch,
):
    manager, _coordinator = task_graph_api
    monkeypatch.setattr(api_server, "TASK_WORKER_EXPERIMENTAL_ENABLED", False)

    automatic = asyncio.run(api_server.chat(api_server.ChatRequest(
        message="gate disabled auto",
        session_id="task-session",
        execution_mode="task_graph",
        task_graph_auto_remote=True,
        workflow_id="wf_gateautooff01",
    )))

    assert len(manager.calls) == 3
    assert automatic.metrics["distributed_requested"] is True
    assert automatic.metrics["distributed_used"] is False
    assert automatic.metrics["fallback"] is True
    assert automatic.metrics["fallback_reason"] == (
        "task_worker_experiment_disabled"
    )

    manual = api_server.ChatRequest(
        message="gate disabled manual",
        execution_mode="task_graph",
        task_graph_remote_stage="candidate_a",
        task_graph_remote_provider_id="remote_worker_01",
    )
    with pytest.raises(HTTPException) as captured:
        asyncio.run(api_server.chat(manual))
    assert captured.value.status_code == 409
    assert captured.value.error_code == "TASK_WORKER_EXPERIMENT_DISABLED"


def test_task_graph_mode_is_rejected_while_feature_flag_is_disabled(
    task_graph_api, monkeypatch,
):
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", False)
    req = api_server.ChatRequest(
        message="question",
        execution_mode="task_graph",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.chat(req))

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_code == "TASK_GRAPH_DISABLED"


def test_local_provider_dispatch_validates_runtime_return_type():
    request = api_server.ProviderStageRequest(
        workflow_id="wf_dispatch01",
        request_id="req-dispatch",
        stage_id="candidate_a",
        stage_type="full_inference",
        provider_id="local_full_model",
        dependencies={},
        root_input={},
        runtime_context={
            "local_provider_executor": lambda *args: object(),
        },
    )

    with pytest.raises(api_server.TaskGraphError, match="必须是 dict"):
        api_server._dispatch_local_task_provider(
            request,
            threading.Event(),
        )


def test_disabled_task_graph_does_not_initialize_sqlite(monkeypatch):
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", False)

    def unexpected_journal(path):
        raise AssertionError("SQLite journal must not initialize while disabled")

    monkeypatch.setattr(api_server, "SQLiteTaskJournal", unexpected_journal)
    coordinator = api_server._create_task_graph_coordinator()

    assert coordinator.journal_status()["backend"] == "memory"


def test_enabled_task_graph_initializes_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(
        api_server,
        "TASK_GRAPH_JOURNAL_PATH",
        str(tmp_path / "task_graph.sqlite3"),
    )

    coordinator = api_server._create_task_graph_coordinator()

    status = coordinator.journal_status()
    assert status["available"] is True
    assert status["backend"] == "sqlite"
    assert (tmp_path / "task_graph.sqlite3").is_file()
    coordinator.close()


def test_enabled_task_graph_recovers_nonterminal_snapshot_on_startup(
    tmp_path, monkeypatch,
):
    path = str(tmp_path / "task_graph.sqlite3")
    seed = SQLiteTaskJournal(path)
    seed.append_event(
        JournalEvent(
            event_id="evt_factory_recovery",
            workflow_id="wf_factoryrec",
            sequence=1,
            entity_type="workflow",
            entity_id="wf_factoryrec",
            event_type="workflow_state_changed",
            occurred_at=100.0,
            payload={"to_state": "running"},
        ),
        {
            "workflow_id": "wf_factoryrec",
            "last_sequence": 1,
            "state": "running",
            "created_at": 90.0,
            "started_at": 95.0,
            "stages": [],
        },
    )
    seed.close()
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(api_server, "TASK_GRAPH_JOURNAL_PATH", path)
    monkeypatch.setattr(api_server, "TASK_GRAPH_RETENTION_DAYS", 0)
    monkeypatch.setattr(api_server, "TASK_GRAPH_RETENTION_MAX_RECORDS", 0)

    coordinator = api_server._create_task_graph_coordinator()

    recovered = coordinator.get("wf_factoryrec")
    assert recovered["state"] == "failed"
    assert recovered["error_code"] == "coordinator_restarted_during_execution"
    assert coordinator.journal_status()["last_recovery"][
        "recovered_workflows"
    ] == 1
    monkeypatch.setattr(api_server, "task_graph_coordinator", coordinator)
    public = asyncio.run(api_server.get_workflow("wf_factoryrec"))
    assert public["observability"]["recovered_after_restart"] is True
    assert public["observability"]["recovery_reason"] == (
        "coordinator_restarted_during_execution"
    )
    assert public["journal"]["backend"] == "sqlite"
    assert "path" not in public["journal"]
    coordinator.close()


def test_task_journal_cleanup_api_applies_terminal_only_policy(
    tmp_path, monkeypatch,
):
    journal = SQLiteTaskJournal(str(tmp_path / "task_graph.sqlite3"))
    for workflow_id, state, occurred_at in (
        ("wf_apicleanold", "completed", 100.0),
        ("wf_apicleannew", "failed", 200.0),
        ("wf_apicleanrun", "running", 50.0),
    ):
        journal.append_event(
            JournalEvent(
                event_id=f"evt_{workflow_id}",
                workflow_id=workflow_id,
                sequence=1,
                entity_type="workflow",
                entity_id=workflow_id,
                event_type="workflow_state_changed",
                occurred_at=occurred_at,
                payload={"to_state": state},
            ),
            {
                "workflow_id": workflow_id,
                "last_sequence": 1,
                "state": state,
                "created_at": occurred_at,
                "stages": [],
            },
        )
    coordinator = TaskGraphCoordinator(journal=journal)
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(api_server, "task_graph_coordinator", coordinator)
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")

    response = asyncio.run(api_server.cleanup_task_journal(
        max_age_days=0,
        max_records=1,
    ))

    assert response["status"] == "completed"
    assert response["result"]["deleted_workflows"] == 1
    assert journal.get_snapshot("wf_apicleanold") is None
    assert journal.get_snapshot("wf_apicleannew") is not None
    assert journal.get_snapshot("wf_apicleanrun")["state"] == "running"
    coordinator.close()


def test_task_graph_mode_is_rejected_when_journal_is_unavailable(
    task_graph_api, monkeypatch,
):
    coordinator = TaskGraphCoordinator(
        availability_error="task journal unavailable: read-only path",
    )
    monkeypatch.setattr(api_server, "task_graph_coordinator", coordinator)
    req = api_server.ChatRequest(
        message="question",
        execution_mode="task_graph",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.chat(req))

    assert exc_info.value.status_code == 503
    assert "journal" in str(exc_info.value.detail)

    listed = asyncio.run(api_server.list_workflows(limit=5))
    assert listed["available"] is False
    assert listed["journal"]["available"] is False
    assert "path" not in listed["journal"]


def test_workflow_list_is_unavailable_when_local_provider_inspection_fails(
    monkeypatch,
):
    class BrokenInspectionProvider(LocalFullModelProvider):
        inspection_broken = False

        def inspect(self):
            if self.inspection_broken:
                raise RuntimeError("inspection failed")
            return super().inspect()

    registry = ProviderRegistry()
    provider = BrokenInspectionProvider(
        lambda request, cancel_event: {"content": "unused"},
    )
    registry.register(provider)
    provider.inspection_broken = True
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(api_server, "task_graph_coordinator", coordinator)
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")

    listed = asyncio.run(api_server.list_workflows(limit=5))

    assert listed["available"] is False
    assert listed["provider_status"][0]["healthy"] is False
    assert listed["provider_status"][0]["error_code"] == (
        "provider_inspection_failed"
    )
    coordinator.close()


def test_second_task_graph_request_is_rejected_without_queueing(task_graph_api):
    req = api_server.ChatRequest(
        message="question",
        execution_mode="task_graph",
    )
    assert api_server._task_graph_execution_slot.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(api_server.chat(req))
    finally:
        api_server._task_graph_execution_slot.release()

    assert exc_info.value.status_code == 429


def test_task_graph_respects_small_token_budget(task_graph_api):
    manager, _coordinator = task_graph_api
    req = api_server.ChatRequest(
        message="question",
        max_new_tokens=1,
        execution_mode="task_graph",
    )

    asyncio.run(api_server.chat(req))

    assert [call["max_tokens"] for call in manager.calls] == [1, 1, 1]


def test_persistent_local_provider_uses_each_requests_execution_context(
    task_graph_api,
):
    manager, _coordinator = task_graph_api
    first = api_server.ChatRequest(
        message="first question",
        max_new_tokens=1,
        execution_mode="task_graph",
        workflow_id="wf_context01",
    )
    asyncio.run(api_server.chat(first))
    assert [call["max_tokens"] for call in manager.calls] == [1, 1, 1]

    manager.calls.clear()
    second = api_server.ChatRequest(
        message="second question",
        max_new_tokens=7,
        execution_mode="task_graph",
        workflow_id="wf_context02",
    )
    asyncio.run(api_server.chat(second))

    assert [call["max_tokens"] for call in manager.calls] == [7, 7, 7]
    assert any(
        message.get("content") == "second question"
        for message in manager.calls[0]["messages"]
    )


def test_workflow_query_list_and_cancel_api(task_graph_api):
    _manager, coordinator = task_graph_api

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        session_id="session-a",
        workflow_id="wf_query123",
    )
    coordinator.commit_result("wf_query123")

    listed = asyncio.run(api_server.list_workflows(limit=5))
    assert listed["enabled"] is True
    assert listed["available"] is True
    assert listed["role"] == "master"
    assert listed["templates"] == ["dual_candidate"]
    assert listed["providers"] == ["local_full_model"]
    assert listed["provider_status"][0]["provider_kind"] == "local_full_model"
    assert listed["provider_status"][0]["max_concurrency"] == 1
    assert listed["worker_protocol"]["schema_ready"] is True
    assert listed["worker_protocol"]["adapter_connected"] is False
    assert listed["worker_protocol"]["admission_state"] == (
        "n2_4_experiment_enabled_not_connected"
    )
    assert listed["worker_protocol"]["experiment_enabled"] is True
    assert listed["worker_protocol"]["experimental_dispatch_enabled"] is False
    assert listed["worker_protocol"]["control_plane_ready"] is True
    assert listed["worker_protocol"]["task_dispatch_enabled"] is False
    assert listed["worker_protocol"]["manual_stage_dispatch_enabled"] is False
    assert listed["journal"]["backend"] == "memory"
    assert listed["journal"]["available"] is True
    assert listed["workflows"][0]["workflow_id"] == "wf_query123"
    assert listed["workflows"][0]["session_id"] == "session-a"
    assert listed["workflows"][0]["observability"] == {
        "state": "completed",
        "result_ready": False,
        "terminal": True,
        "partial_result": False,
        "recovered_after_restart": False,
        "recovery_reason": "",
        "retry_count": 0,
        "same_provider_retry_count": 0,
        "reassignment_count": 0,
        "retrying": False,
        "result_rejection_count": 0,
        "last_result_rejection_reason": "",
        "last_result_rejected_at": None,
        "winner_count": 3,
        "actual_providers": ["local_full_model"],
        "actual_nodes": [],
    }
    assert listed["workflows"][0]["journal"]["backend"] == "memory"

    summary = asyncio.run(api_server.list_workflows(limit=50, summary=True))
    assert summary["enabled"] is True
    assert summary["workflows"][0]["workflow_id"] == "wf_query123"
    assert summary["workflows"][0]["stages"]
    assert "session_id" not in summary["workflows"][0]
    assert "error" not in summary["workflows"][0]
    assert "model_identity" not in summary["workflows"][0]
    assert "input_bindings" not in summary["workflows"][0]["stages"][0]
    assert "result_metadata" not in summary["workflows"][0]["stages"][0]["attempts"][0]

    fetched = asyncio.run(api_server.get_workflow("wf_query123"))
    assert fetched["state"] == "completed"
    assert fetched["observability"]["winner_count"] == 3
    assert fetched["journal"]["available"] is True

    coordinator.run_template(
        "dual_candidate",
        {"message": "other"},
        execute,
        session_id="session-b",
        workflow_id="wf_queryother",
    )
    coordinator.commit_result("wf_queryother")
    filtered = asyncio.run(api_server.list_workflows(
        limit=5, session_id="session-a",
    ))
    assert [item["workflow_id"] for item in filtered["workflows"]] == [
        "wf_query123",
    ]

    cancelled = asyncio.run(api_server.cancel_workflow("wf_query123"))
    assert cancelled["status"] == "completed"
    assert cancelled["workflow"]["cancel_requested"] is False


def test_workflow_observability_reports_retry_and_result_rejection():
    observability = api_server._workflow_observability({
        "state": "result_ready",
        "retry_count": 1,
        "same_provider_retry_count": 1,
        "result_rejection_count": 1,
        "partial_result": True,
        "stages": [
            {
                "retry_count": 1,
                "result_rejection_count": 1,
                "last_result_rejection_reason": "winner_already_committed",
                "last_result_rejected_at": 123.0,
                "winner_attempt_id": "att_winner01",
                "attempts": [{
                    "provider": "worker_a",
                    "provider_node_id": "node-a",
                }],
            },
        ],
    })

    assert observability["result_ready"] is True
    assert observability["terminal"] is False
    assert observability["partial_result"] is True
    assert observability["retry_count"] == 1
    assert observability["same_provider_retry_count"] == 1
    assert observability["reassignment_count"] == 0
    assert observability["result_rejection_count"] == 1
    assert observability["last_result_rejection_reason"] == (
        "winner_already_committed"
    )
    assert observability["actual_providers"] == ["worker_a"]
    assert observability["actual_nodes"] == ["node-a"]


def test_workflow_api_fences_unknown_valid_id_and_rejects_invalid_id(task_graph_api):
    with pytest.raises(HTTPException) as get_error:
        asyncio.run(api_server.get_workflow("wf_missing1"))
    assert get_error.value.status_code == 404

    cancel_result = asyncio.run(api_server.cancel_workflow("wf_missing1"))
    assert cancel_result["status"] == "cancel_pending"
    assert cancel_result["workflow"]["cancel_requested"] is True

    with pytest.raises(HTTPException) as cancel_error:
        asyncio.run(api_server.cancel_workflow("unsafe"))
    assert cancel_error.value.status_code == 400


def test_auto_role_refuses_unreplicated_task_graph_writes(task_graph_api, monkeypatch):
    monkeypatch.setattr(api_server, "NODE_ROLE", "auto")
    listed = asyncio.run(api_server.list_workflows())
    assert listed["local_available"] is False
    request = api_server.ChatRequest(message="unreplicated", use_task_graph=True)
    with pytest.raises(HTTPException) as rejected:
        api_server._execute_task_graph_chat(request)
    assert rejected.value.status_code == 503
    with pytest.raises(HTTPException) as cancelled:
        asyncio.run(api_server.cancel_workflow("wf_pending_auto"))
    assert cancelled.value.status_code == 503
    with pytest.raises(HTTPException) as cleaned:
        asyncio.run(api_server.cleanup_task_journal())
    assert cleaned.value.status_code == 503


def test_chat_threadpool_allows_concurrent_workflow_cancellation(
    task_graph_api, monkeypatch,
):
    manager, coordinator = task_graph_api
    started = threading.Event()

    def wait_for_cancel(messages, max_tokens, temperature, top_p, **kwargs):
        cancel_event = kwargs["_cancel_event"]
        started.set()
        assert cancel_event.wait(5)
        return {
            "content": "late result",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(manager, "chat", wait_for_cancel)
    req = api_server.ChatRequest(
        message="cancel me",
        execution_mode="task_graph",
        workflow_id="wf_cancelapi",
    )

    async def scenario():
        chat_task = asyncio.create_task(api_server.chat(req))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        cancel_result = await api_server.cancel_workflow("wf_cancelapi")
        assert cancel_result["status"] == "cancel_requested"
        assert cancel_result["workflow"]["cancel_requested"] is True
        with pytest.raises(HTTPException) as exc_info:
            await chat_task
        assert exc_info.value.status_code == 409

    asyncio.run(scenario())
    workflow = coordinator.get("wf_cancelapi")
    assert workflow["state"] == "cancelled"
    assert workflow["completed_stage_count"] == 0


def test_standard_chat_generation_cancel_discards_late_result(
    task_graph_api, monkeypatch,
):
    manager, _coordinator = task_graph_api
    started = threading.Event()

    def wait_for_cancel(messages, max_tokens, temperature, top_p, **kwargs):
        cancel_event = kwargs["_cancel_event"]
        started.set()
        assert cancel_event.wait(5)
        return {
            "content": "late result",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(manager, "chat", wait_for_cancel)
    req = api_server.ChatRequest(
        message="cancel standard chat",
        session_id="task-session",
        generation_id="gen_standard123",
    )

    async def scenario():
        chat_task = asyncio.create_task(api_server.chat(req))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        cancel_result = await api_server.cancel_chat_generation(
            "gen_standard123",
        )
        assert cancel_result["status"] == "cancel_requested"
        with pytest.raises(HTTPException) as exc_info:
            await chat_task
        assert exc_info.value.status_code == 409

    asyncio.run(scenario())
    assert api_server.session_histories["task-session"] == []
    assert "gen_standard123" not in api_server._generation_cancel_events


def test_generation_cancel_before_registration_fences_request(task_graph_api):
    result = asyncio.run(
        api_server.cancel_chat_generation("gen_before1234"),
    )
    assert result["status"] == "cancel_pending"

    req = api_server.ChatRequest(
        message="must not run",
        session_id="task-session",
        generation_id="gen_before1234",
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.chat(req))

    assert exc_info.value.status_code == 409
    assert api_server.session_histories["task-session"] == []


def test_clear_chat_waits_for_inflight_generation_then_removes_history(
    task_graph_api, monkeypatch,
):
    manager, _coordinator = task_graph_api
    started = threading.Event()
    release = threading.Event()

    def slow_chat(messages, max_tokens, temperature, top_p, **kwargs):
        started.set()
        assert release.wait(5)
        return {
            "content": "completed answer",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(manager, "chat", slow_chat)
    monkeypatch.setattr(api_server, "_generate_followups_llama", lambda *a, **k: [])
    req = api_server.ChatRequest(
        message="finish then clear",
        session_id="task-session",
        generation_id="gen_clearwait1",
    )

    async def scenario():
        chat_task = asyncio.create_task(api_server.chat(req))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        clear_task = asyncio.create_task(
            asyncio.to_thread(api_server.clear_chat, "task-session"),
        )
        await asyncio.sleep(0.05)
        assert not clear_task.done()
        release.set()
        await chat_task
        result = await clear_task
        assert result["status"] == "cleared"

    asyncio.run(scenario())
    assert api_server.session_histories["task-session"] == []


def test_fast_stream_bridge_keeps_event_loop_responsive(
    task_graph_api, monkeypatch,
):
    manager, _coordinator = task_graph_api
    manager._engine_type = "pytorch"

    def slow_stream(*args, **kwargs):
        time.sleep(0.3)
        yield {
            "done": True,
            "response": "streamed",
            "metrics": {},
        }

    monkeypatch.setattr(
        api_server.scheduler,
        "_run_full_model_inference_stream",
        slow_stream,
    )
    req = api_server.ChatRequest(
        message="stream",
        streaming_mode="fast",
        generation_id="gen_stream1234",
    )

    async def scenario():
        response = await api_server.chat_stream(req, cast(Any, None))
        body_iterator = cast(Any, response.body_iterator)
        started_at = time.monotonic()
        chunk_task = asyncio.create_task(body_iterator.__anext__())
        await asyncio.sleep(0.03)
        assert time.monotonic() - started_at < 0.15
        chunk = await chunk_task
        assert '"response": "streamed"' in chunk
        await body_iterator.aclose()

    asyncio.run(scenario())
    assert "gen_stream1234" not in api_server._generation_cancel_events


def test_task_graph_parses_thinking_without_leaking_tags(
    task_graph_api, monkeypatch,
):
    manager, _coordinator = task_graph_api
    contents = iter([
        "【思考】候选甲思考【思考结束】候选甲答案",
        "【思考】候选乙思考【思考结束】候选乙答案",
        "【思考】最终思考【思考结束】最终答案",
    ])

    def thinking_chat(messages, max_tokens, temperature, top_p, **kwargs):
        manager.calls.append({"messages": messages, "max_tokens": max_tokens})
        return {
            "content": next(contents),
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }

    monkeypatch.setattr(manager, "chat", thinking_chat)
    req = api_server.ChatRequest(
        message="think",
        session_id="task-session",
        execution_mode="task_graph",
        workflow_id="wf_thinking1",
        show_thinking=True,
    )

    response = asyncio.run(api_server.chat(req))

    assert response.content == "最终答案"
    assert response.thinking_content == "最终思考"
    assert api_server.session_histories["task-session"][-1]["content"] == "最终答案"
    aggregate_prompt = manager.calls[2]["messages"][-1]["content"]
    assert "候选甲答案" in aggregate_prompt
    assert "候选乙答案" in aggregate_prompt
    assert "候选甲思考" not in aggregate_prompt
    assert "候选乙思考" not in aggregate_prompt


def test_task_graph_persists_followups_with_assistant_metrics(
    task_graph_api, monkeypatch,
):
    saved = []
    monkeypatch.setattr(
        api_server._local_store,
        "save_local_conversation_turn",
        lambda session_id, user_message, assistant_message, metrics=None, **kwargs: (
            saved.extend([
                (session_id, "user", user_message, None),
                (session_id, "assistant", assistant_message, metrics),
            ])
            or True
        ),
    )
    req = api_server.ChatRequest(
        message="persist followups",
        session_id="task-session",
        execution_mode="task_graph",
        workflow_id="wf_followups1",
    )

    response = asyncio.run(api_server.chat(req))

    assistant = next(item for item in saved if item[1] == "assistant")
    assert assistant[3]["followups"] == response.followups
    assert len(response.followups) >= 2


def test_generation_cancel_after_graph_compute_marks_workflow_discarded(
    task_graph_api, monkeypatch,
):
    _manager, coordinator = task_graph_api
    computed = threading.Event()
    release = threading.Event()
    original_run_template = coordinator.run_template

    def pause_after_compute(*args, **kwargs):
        result = original_run_template(*args, **kwargs)
        computed.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(coordinator, "run_template", pause_after_compute)
    req = api_server.ChatRequest(
        message="cancel before commit",
        session_id="task-session",
        execution_mode="task_graph",
        workflow_id="wf_latecancel",
        generation_id="gen_latecancel",
    )

    async def scenario():
        chat_task = asyncio.create_task(api_server.chat(req))
        for _ in range(100):
            if computed.is_set():
                break
            await asyncio.sleep(0.01)
        assert computed.is_set()
        result = await api_server.cancel_chat_generation("gen_latecancel")
        assert result["status"] == "cancel_requested"
        release.set()
        with pytest.raises(HTTPException) as exc_info:
            await chat_task
        assert exc_info.value.status_code == 409

    asyncio.run(scenario())
    snapshot = coordinator.get("wf_latecancel")
    assert snapshot["state"] == "cancelled"
    assert snapshot["completed_stage_count"] == 3
    assert api_server.session_histories["task-session"] == []


def test_workflow_cancel_during_result_commit_rolls_back_history(
    task_graph_api, monkeypatch,
):
    _manager, coordinator = task_graph_api
    commit_started = threading.Event()
    release_commit = threading.Event()
    original_commit_result = coordinator.commit_result

    def pause_before_commit(workflow_id):
        commit_started.set()
        assert release_commit.wait(5)
        return original_commit_result(workflow_id)

    monkeypatch.setattr(coordinator, "commit_result", pause_before_commit)
    req = api_server.ChatRequest(
        message="cancel during commit",
        session_id="task-session",
        execution_mode="task_graph",
        workflow_id="wf_commitcancel",
        generation_id="gen_commitcancel",
    )

    async def scenario():
        chat_task = asyncio.create_task(api_server.chat(req))
        for _ in range(100):
            if commit_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert commit_started.is_set()
        result = await api_server.cancel_workflow("wf_commitcancel")
        assert result["status"] == "cancelled"
        release_commit.set()
        with pytest.raises(HTTPException) as exc_info:
            await chat_task
        assert exc_info.value.status_code == 409

    asyncio.run(scenario())
    snapshot = coordinator.get("wf_commitcancel")
    assert snapshot["state"] == "cancelled"
    assert snapshot["completed_stage_count"] == 3
    assert api_server.session_histories["task-session"] == []


def test_journal_failure_during_result_commit_rolls_back_history(
    task_graph_api, monkeypatch,
):
    _manager, coordinator = task_graph_api

    def fail_commit(workflow_id):
        raise TaskGraphUnavailable("task journal unavailable: disk full")

    monkeypatch.setattr(coordinator, "commit_result", fail_commit)
    req = api_server.ChatRequest(
        message="journal failure",
        session_id="task-session",
        execution_mode="task_graph",
        workflow_id="wf_journalfail",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.chat(req))

    assert exc_info.value.status_code == 503
    assert api_server.session_histories["task-session"] == []


def test_normal_task_graph_stream_does_not_set_cancel_requested(task_graph_api):
    _manager, coordinator = task_graph_api
    req = api_server.ChatRequest(
        message="stream workflow",
        execution_mode="task_graph",
        workflow_id="wf_streamdone",
        generation_id="gen_streamdone",
    )

    async def scenario():
        response = await api_server.chat_stream(req, cast(Any, None))
        chunks = []
        async for chunk in cast(Any, response.body_iterator):
            chunks.append(chunk)
        assert any('"done": true' in chunk for chunk in chunks)

    asyncio.run(scenario())
    snapshot = coordinator.get("wf_streamdone")
    assert snapshot["state"] == "completed"
    assert snapshot["cancel_requested"] is False


def test_model_change_waits_until_all_task_graph_stages_finish(
    task_graph_api, monkeypatch,
):
    manager, _coordinator = task_graph_api
    first_stage_started = threading.Event()
    release_first_stage = threading.Event()
    prepare_started = threading.Event()
    observed_models = []

    def staged_chat(messages, max_tokens, temperature, top_p, **kwargs):
        call_no = len(observed_models) + 1
        observed_models.append(manager.active_model_id)
        if call_no == 1:
            first_stage_started.set()
            assert release_first_stage.wait(5)
        return {
            "content": f"stage {call_no}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def change_model():
        manager.active_model_id = "new-model"
        return {"success": True}

    monkeypatch.setattr(manager, "chat", staged_chat)
    monkeypatch.setattr(api_server, "_refresh_pipeline_layer_config", lambda: None)
    req = api_server.ChatRequest(
        message="model boundary",
        execution_mode="task_graph",
        workflow_id="wf_modelguard",
    )

    async def scenario():
        chat_task = asyncio.create_task(api_server.chat(req))
        for _ in range(100):
            if first_stage_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert first_stage_started.is_set()
        change_task = asyncio.create_task(asyncio.to_thread(
            api_server._run_exclusive_model_change,
            change_model,
            lambda: prepare_started.set(),
        ))
        await asyncio.sleep(0.05)
        assert not prepare_started.is_set()
        release_first_stage.set()
        await chat_task
        await change_task

    asyncio.run(scenario())
    assert observed_models == ["fake-model", "fake-model", "fake-model"]
    assert manager.active_model_id == "new-model"
    assert prepare_started.is_set()
