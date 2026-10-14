import copy
import base64
import hashlib
import sys
import os
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tcp_comm as tcp_comm_mod
import task_worker_adapter as task_worker_adapter_mod
from request_deadline import RequestDeadline
from task_worker_adapter import TaskWorkerControlPlane
from task_worker_protocol import (
    WorkerProtocolError,
    build_message,
    canonical_sha256,
    stage_input_sha256,
)
from task_provider import (
    CallbackExecutionProvider,
    DEPENDENCY_FAILURES_KEY,
    ModelIdentity,
    ProviderExecutionError,
    ProviderUnavailable,
    StageAttempt,
    StageRequest,
)
from task_worker_adapter import RemoteFullWorkerProvider, remote_provider_id
from tcp_comm import MessageType, TCPClient, TCPServer


@pytest.fixture(autouse=True)
def _enable_task_worker_experiment(monkeypatch):
    import scheduler as scheduler_mod

    monkeypatch.setattr(
        scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True,
    )


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    wake = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return True
        wake.wait(min(0.01, max(0.0, deadline - time.monotonic())))
    return bool(predicate())


def _capabilities():
    return {
        "stage_types": ["full_inference", "aggregate"],
        "engines": ["pytorch"],
        "models": [{
            "model_id": "qwen-1_8b",
            "engine": "pytorch",
            "format": "safetensors",
            "revision": "local-rev1",
            "sha256": "1" * 64,
        }],
        "max_concurrency": 1,
    }


def _scheduler_callbacks(*, identity=None, execute=None):
    from model_host import SchedulerCallbackSet

    def unexpected_execute(*args, **kwargs):
        raise AssertionError("unexpected task-worker stage execution")

    return SchedulerCallbackSet(
        active_task_graph_model_identity=identity or (lambda: None),
        execute_task_worker_stage=execute or unexpected_execute,
        build_model_chat_prompt=lambda tokenizer, messages, **kwargs: "",
        thinking_system_prompt="thinking",
        snapshot_recent_logs=lambda: ([], 0),
        filter_recent_logs=lambda entries, **kwargs: list(entries),
        format_model_response=lambda text, **kwargs: (text, None),
    )


def _android_capabilities(*, resource_admitted=True):
    return {
        "stage_types": ["full_inference"],
        "engines": ["llama_cpp"],
        "models": [{
            "model_id": "qwen_1_8b",
            "engine": "llama_cpp",
            "format": "gguf",
            "revision": "local-v1",
            "sha256": "a" * 64,
        }],
        "max_concurrency": 1,
        "resource_gate": {
            "admitted": resource_admitted,
            "reason_code": "" if resource_admitted else "thermal_gate",
        },
    }


def _android_layer_capabilities(*, resource_admitted=True):
    capabilities = _android_capabilities(resource_admitted=resource_admitted)
    capabilities["stage_types"] = ["layer_forward"]
    capabilities["layer_ranges"] = [[4, 8]]
    return capabilities


def _admitted_control_plane():
    worker = TaskWorkerControlPlane()
    coordinator = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    worker.receive_on_worker(ack.snapshot())
    return coordinator, worker


def _admitted_android_control_plane():
    worker = TaskWorkerControlPlane()
    coordinator = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="android_worker_01",
        worker_kind="android_full_worker",
        capabilities=_android_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "android_worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    assert ack.payload["accepted"] is True
    worker.receive_on_worker(ack.snapshot())
    return coordinator, worker


def test_android_layer_worker_is_admitted_without_full_model_dispatch_gate():
    coordinator = TaskWorkerControlPlane()
    # Negotiate a layer-only peer and keep the assertion focused on the
    # dedicated layer gate rather than the full-model dispatch state.
    layer_worker = TaskWorkerControlPlane()
    hello = layer_worker.begin_worker_hello(
        node_id="android_layer_01",
        worker_kind="android_full_worker",
        capabilities=_android_layer_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "android_layer_01", hello.snapshot(), coordinator_node_id="master",
    )
    assert ack.payload["accepted"] is True
    snapshot = coordinator.worker_snapshot("android_layer_01")
    assert snapshot["manual_stage_dispatch_enabled"] is True
    assert snapshot["layer_stage_dispatch_enabled"] is True
    assert snapshot["capabilities"]["models"]


def test_recovery_candidate_uses_accepted_capabilities_without_health_overlay():
    coordinator = TaskWorkerControlPlane()
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="android_layer_01",
        worker_kind="android_full_worker",
        capabilities=_android_layer_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "android_layer_01", hello.snapshot(), coordinator_node_id="master",
    )
    assert ack.payload["selected_version"] >= 2

    candidates = coordinator.connected_layer_stage_workers({"android_layer_01"})
    assert [item["node_id"] for item in candidates] == ["android_layer_01"]

    coordinator.disconnect_worker("android_layer_01")
    assert coordinator.connected_layer_stage_workers() == []


def test_pending_worker_fence_is_resolved_by_hello_and_disconnect():
    control = TaskWorkerControlPlane()
    control.mark_worker_connection_pending("worker_pending")
    assert control.pending_worker_ids() == {"worker_pending"}
    control.resolve_worker_connection_pending("worker_pending")
    assert control.pending_worker_ids() == set()
    control.mark_worker_connection_pending("worker_pending")
    control.disconnect_worker("worker_pending")
    assert control.pending_worker_ids() == set()


def test_remote_provider_uses_layer_gate_for_layer_forward_stage():
    coordinator = TaskWorkerControlPlane()
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="android_layer_01",
        worker_kind="android_full_worker",
        capabilities=_android_layer_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "android_layer_01", hello.snapshot(), coordinator_node_id="master",
    )
    assert ack.payload["accepted"] is True
    snapshot = coordinator.worker_snapshot("android_layer_01")
    snapshot["manual_stage_dispatch_enabled"] = False
    snapshot["layer_stage_dispatch_enabled"] = True
    provider = RemoteFullWorkerProvider(
        node_id="android_layer_01",
        peer_snapshot=lambda: snapshot,
        send_message=lambda _message: None,
    )
    status = provider.inspect()
    assert status.healthy is True
    assert "layer_forward" in status.supported_stage_types


def test_layer_worker_matches_physical_identity_but_not_full_model_alias():
    worker = TaskWorkerControlPlane()
    coordinator = TaskWorkerControlPlane()
    capabilities = _android_layer_capabilities()
    capabilities["layer_worker"] = True
    capabilities["models"] = [{
        "model_id": "layer-aaaaaaaaaaaaaaaa",
        "engine": "llama_cpp",
        "format": "gguf",
        "revision": "local-aaaaaaaaaaaa",
        "sha256": "a" * 64,
    }]
    hello = worker.begin_worker_hello(
        node_id="android_layer_alias_01",
        worker_kind="android_full_worker",
        capabilities=capabilities,
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "android_layer_alias_01", hello.snapshot(), coordinator_node_id="master",
    )
    assert ack.payload["accepted"] is True
    snapshot = coordinator.worker_snapshot("android_layer_alias_01")
    snapshot["manual_stage_dispatch_enabled"] = False
    snapshot["layer_stage_dispatch_enabled"] = True
    provider = RemoteFullWorkerProvider(
        node_id="android_layer_alias_01",
        peer_snapshot=lambda: snapshot,
        send_message=lambda _message: None,
    )
    requested = ModelIdentity(
        model_id="qwen3-5-2b",
        engine="llama_cpp",
        format="gguf",
        revision="coordinator-rev",
        sha256="a" * 64,
    )
    assert provider.supports_model_identity(requested, "layer_forward") is True
    assert provider.supports_model_identity(requested, "full_inference") is False


def test_worker_artifact_matching_normalizes_local_engine_aliases_and_rejects_unknown():
    requested = ModelIdentity(
        model_id="model-a", engine="llama.cpp", format="gguf",
        revision="rev-a", sha256="a" * 64,
    )
    exact_alias = [{
        "model_id": "model-a", "engine": "gguf", "format": "gguf",
        "revision": "rev-a", "sha256": "a" * 64,
    }]
    layer_alias = [{
        "model_id": "layer-a", "engine": "llama-cpp", "format": "gguf",
        "revision": "layer-rev", "sha256": "a" * 64,
    }]
    unknown = [{
        "model_id": "model-a", "engine": "llama_magic", "format": "gguf",
        "revision": "rev-a", "sha256": "a" * 64,
    }]

    assert RemoteFullWorkerProvider._model_matches(requested, exact_alias) is True
    assert RemoteFullWorkerProvider._layer_model_matches(requested, layer_alias) is True
    assert RemoteFullWorkerProvider._model_matches(requested, unknown) is False
    assert RemoteFullWorkerProvider._layer_model_matches(requested, unknown) is False

    torch_requested = ModelIdentity(
        model_id="model-b", engine="pytorch", format="safetensors",
        revision="rev-b", sha256="b" * 64,
    )
    torch_alias = [{
        "model_id": "model-b", "engine": "torch", "format": "safetensors",
        "revision": "rev-b", "sha256": "b" * 64,
    }]
    assert RemoteFullWorkerProvider._model_matches(torch_requested, torch_alias) is True

    external_requested = ModelIdentity(
        model_id="model-c", engine="external_api", format="remote",
        revision="rev-c", sha256="c" * 64,
    )
    assert RemoteFullWorkerProvider._model_matches(
        external_requested, [external_requested.snapshot()],
    ) is True


def test_route_a_stage_result_maps_hidden_bytes_back_to_pipeline_tensor():
    import struct
    from scheduler_pipeline import _layer_stage_result_to_pipeline_value

    raw = struct.pack("<4f", 0.25, -1.5, 3.0, 4.5)
    output = {
        "hidden_out_f32": base64.b64encode(raw).decode("ascii"),
        "hidden_out_sha256": hashlib.sha256(raw).hexdigest(),
        "token_argmax": 7,
    }
    mapped = _layer_stage_result_to_pipeline_value(
        output, hidden_spec={"n_tokens": 2, "n_embd": 2},
    )
    assert mapped["kind"] == "hidden"
    assert mapped["token_argmax"] == 7
    assert mapped["hidden_states"].shape == (2, 2)
    assert mapped["hidden_states"].flatten().tolist() == [0.25, -1.5, 3.0, 4.5]


def test_route_a_stage_result_rejects_hidden_digest_mismatch():
    from scheduler_pipeline import _layer_stage_result_to_pipeline_value

    with pytest.raises(ValueError, match="digest"):
        _layer_stage_result_to_pipeline_value(
            {
                "hidden_out_f32": base64.b64encode(b"\x00" * 8).decode("ascii"),
                "hidden_out_sha256": "0" * 64,
            },
            hidden_spec={"n_tokens": 1, "n_embd": 2},
        )


def _model_identity():
    return ModelIdentity(**_capabilities()["models"][0])


def _remote_request(provider_id):
    return StageRequest(
        workflow_id="wf_remote001",
        request_id="request-remote-01",
        stage_id="candidate_a",
        stage_type="full_inference",
        provider_id=provider_id,
        dependencies={},
        root_input={"message": "hello", "messages": []},
        model_identity=_model_identity(),
    )


def _response_identity(offer):
    return {
        "workflow_id": offer["workflow_id"],
        "stage_id": offer["stage_id"],
        "attempt_id": offer["attempt_id"],
        "lease_id": offer["lease_id"],
        "lease_epoch": offer["lease_epoch"],
        "provider_id": offer["provider_id"],
    }


def _start_accepted_remote_attempt(suffix):
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id=f"att_terminal_{suffix}",
        request=request,
        provider_id=provider.provider_id,
        lease_id=f"lease_terminal_{suffix}",
        lease_epoch=1,
        lease_expires_at=time.time() + 5.0,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert _wait_until(
        lambda: any(message.message_type == "stage_offer" for message in sent)
    )
    offer = next(
        message for message in sent if message.message_type == "stage_offer"
    )
    identity = _response_identity(offer.payload)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id=f"msg_terminal_accept_{suffix}",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())
    return (
        provider, reservation, attempt, identity, outcome, finished, thread, sent,
    )


def test_remote_provider_request_deadline_caps_accept_wait():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    deadline = RequestDeadline.start(0.08)
    attempt = StageAttempt(
        attempt_id="att_deadline_accept01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_deadline_accept01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5.0,
        request_deadline=deadline,
    )

    started = time.monotonic()
    with pytest.raises(ProviderExecutionError) as captured:
        provider.execute(attempt, reservation, threading.Event())

    assert captured.value.code == "request_deadline_exceeded"
    assert captured.value.retryable is False
    assert time.monotonic() - started < 0.5
    assert sent[0].message_type == "stage_offer"
    assert sent[0].payload["lease_expires_at_ms"] == int(
        attempt.lease_expires_at * 1000
    )
    assert sent[0].payload["request_deadline_ms"] == (
        deadline.expires_at_epoch_ms
    )
    assert (
        sent[0].payload["lease_expires_at_ms"]
        > sent[0].payload["request_deadline_ms"]
    )
    assert _wait_until(
        lambda: any(message.message_type == "stage_cancel" for message in sent)
    )
    provider.release(reservation.reservation_id)


def test_remote_offer_timestamp_is_sampled_after_chunk_preparation(monkeypatch):
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_offer_timestamp01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_offer_timestamp01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5.0,
    )
    observed = {}

    def prepare_chunks(_attempt):
        time.sleep(0.03)
        observed["completed_at_ms"] = int(time.time() * 1000)
        return dict(request.root_input)

    monkeypatch.setattr(provider, "_maybe_send_stage_chunks", prepare_chunks)
    finished = threading.Event()

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except ProviderExecutionError:
            pass
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert _wait_until(
        lambda: any(message.message_type == "stage_offer" for message in sent)
    )
    offer = next(
        message for message in sent if message.message_type == "stage_offer"
    )
    assert offer.sent_at_ms >= observed["completed_at_ms"]

    provider.cancel(attempt.attempt_id)
    assert finished.wait(1.0)
    provider.release(reservation.reservation_id)


def test_remote_wait_is_not_shortened_by_wall_clock_jump(monkeypatch):
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_wall_jump01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_wall_jump01",
        lease_epoch=1,
        lease_expires_at=time.time() + 1.0,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert _wait_until(lambda: bool(sent))
    offer = sent[0].payload
    identity = _response_identity(offer)
    real_time = time.time
    monkeypatch.setattr(
        task_worker_adapter_mod.time,
        "time",
        lambda: real_time() + 10_000.0,
    )
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_wall_jump_accept01",
        sent_at_ms=int(real_time() * 1000),
        version=2,
    ).snapshot())
    output = {"content": "monotonic wait survived"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {},
        },
        message_id="msg_wall_jump_result01",
        sent_at_ms=int(real_time() * 1000),
        version=2,
    ).snapshot())

    assert finished.wait(1.0)
    assert "error" not in outcome
    assert outcome["result"].output == output
    provider.release(reservation.reservation_id)


def _v1_hello(node_id="worker_01"):
    return build_message(
        "hello",
        {
            "node_id": node_id,
            "worker_kind": "pc_full_worker",
            "min_version": 1,
            "max_version": 1,
            "capabilities": _capabilities(),
        },
        message_id="msg_v1hello01",
        sent_at_ms=1000,
        version=1,
    )


def _stage_offer():
    root_input = {"message": "hello"}
    dependencies = {}
    return build_message(
        "stage_offer",
        {
            "workflow_id": "wf_adapter001",
            "request_id": "request-adapter-01",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_adapter001",
            "lease_id": "lease_adapter001",
            "lease_epoch": 1,
            "lease_expires_at_ms": 2000,
            "provider_id": "worker_01",
            "root_input": root_input,
            "dependencies": dependencies,
            "input_sha256": stage_input_sha256(root_input, dependencies),
            "model_identity": _capabilities()["models"][0],
        },
        message_id="msg_offer001",
        sent_at_ms=1000,
        version=2,
    )


def test_stage_offer_request_deadline_wire_field_is_optional_and_distinct():
    without_deadline = _stage_offer()
    assert "request_deadline_ms" not in without_deadline.payload

    payload = {
        **without_deadline.payload,
        "request_deadline_ms": 1500,
    }
    with_deadline = build_message(
        "stage_offer",
        payload,
        message_id="msg_offer_deadline01",
        sent_at_ms=1000,
        version=2,
    )

    assert with_deadline.payload["request_deadline_ms"] == 1500
    assert with_deadline.payload["lease_expires_at_ms"] == 2000

    payload["request_deadline_ms"] = 0
    with pytest.raises(WorkerProtocolError) as captured:
        build_message(
            "stage_offer",
            payload,
            message_id="msg_offer_deadline02",
            sent_at_ms=1000,
            version=2,
        )
    assert captured.value.code == "invalid_integer"
    assert captured.value.field == "payload.request_deadline_ms"


def test_v2_hello_negotiates_and_reports_control_plane_only():
    worker = TaskWorkerControlPlane()
    coordinator = TaskWorkerControlPlane()

    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(), sent_at_ms=1000,
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(),
        coordinator_node_id="master", sent_at_ms=1001,
    )
    worker.receive_on_worker(ack.snapshot())

    master_status = coordinator.status(role="master")
    worker_status = worker.status(role="client")
    # ★ 2026-09-20：协议升到 v3（层段）。`begin_worker_hello` 用默认 min/max ⇒ 协商出 3。
    #   本用例验的是「协商成功 + 控制面已连、数据面未接」，与具体版本号无关。
    assert ack.payload == {
        "coordinator_node_id": "master",
        "accepted": True,
        "selected_version": 3,
        "reason_code": "",
    }
    assert master_status["control_plane_connected"] is True
    assert master_status["connected_worker_count"] == 1
    assert master_status["workers"][0]["capabilities"] == _capabilities()
    assert master_status["adapter_connected"] is False
    assert master_status["task_dispatch_enabled"] is False
    assert worker_status["coordinator"]["selected_version"] == 3


def test_android_full_worker_provider_requires_resource_gate_and_exact_model():
    worker = TaskWorkerControlPlane()
    coordinator = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="android_worker_01",
        worker_kind="android_full_worker",
        capabilities=_android_capabilities(),
        sent_at_ms=1000,
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "android_worker_01", hello.snapshot(), coordinator_node_id="master",
        sent_at_ms=1001,
    )
    assert ack.payload["accepted"] is True
    provider = RemoteFullWorkerProvider(
        node_id="android_worker_01",
        peer_snapshot=lambda: coordinator.worker_snapshot("android_worker_01"),
        send_message=lambda _message: None,
    )
    status = provider.inspect()
    assert status.healthy is True
    assert status.max_concurrency == 1
    assert provider.supports_model_identity(
        ModelIdentity(**_android_capabilities()["models"][0]),
        "full_inference",
    )

    denied_worker = TaskWorkerControlPlane()
    denied_coordinator = TaskWorkerControlPlane()
    denied_hello = denied_worker.begin_worker_hello(
        node_id="android_worker_denied",
        worker_kind="android_full_worker",
        capabilities=_android_capabilities(resource_admitted=False),
    )
    assert denied_hello is not None
    denied_coordinator.receive_on_coordinator(
        "android_worker_denied", denied_hello.snapshot(), coordinator_node_id="master",
    )
    denied_provider = RemoteFullWorkerProvider(
        node_id="android_worker_denied",
        peer_snapshot=lambda: denied_coordinator.worker_snapshot("android_worker_denied"),
        send_message=lambda _message: None,
    )
    assert denied_provider.inspect().healthy is False
    assert denied_provider.supports_model_identity(
        ModelIdentity(**_android_capabilities()["models"][0]),
        "full_inference",
    ) is False


def test_scheduler_admits_android_worker_only_for_android_node(monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    assert scheduler.register_node(
        "android_worker_01", role="client", node_type="android",
    ) is True
    sent = []
    monkeypatch.setattr(
        scheduler, "_send_task_worker_to_node",
        lambda _node_id, message: sent.append(message),
    )
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="android_worker_01",
        worker_kind="android_full_worker",
        capabilities=_android_capabilities(),
    )
    assert hello is not None
    scheduler._handle_task_worker_message(
        "android_worker_01", {"data": hello.snapshot()},
    )
    assert sent and sent[0].message_type == "hello_ack"
    assert sent[0].payload["accepted"] is True
    providers = scheduler.remote_task_worker_providers()
    assert len(providers) == 1
    assert providers[0].inspect().healthy is True


def test_android_fake_provider_task_graph_journal_records_attempt_identity(tmp_path):
    from task_graph import StageSpec, TaskGraphCoordinator
    from task_journal import SQLiteTaskJournal

    coordinator_control, _worker_control = _admitted_android_control_plane()
    sent = []
    provider_holder = {}

    def send(message):
        sent.append(message.message_type)
        if message.message_type != "stage_offer":
            return
        identity = _response_identity(message.payload)
        provider = provider_holder["provider"]
        provider.handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_android_journal_accept",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        output = {"content": "android fake journal result"}
        provider.handle_message(build_message(
            "stage_result",
            {
                **identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {"model": "qwen_1_8b"},
            },
            message_id="msg_android_journal_result",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

    provider = RemoteFullWorkerProvider(
        node_id="android_worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("android_worker_01"),
        send_message=send,
    )
    provider_holder["provider"] = provider
    journal = SQLiteTaskJournal(str(tmp_path / "android-task-graph.sqlite3"))
    graph = TaskGraphCoordinator(journal=journal)
    graph.register_provider(provider)
    try:
        output, snapshot = graph.run(
            stages=[StageSpec(
                "android_stage",
                "full_inference",
                provider=provider.provider_id,
                model_identity=ModelIdentity(**_android_capabilities()["models"][0]),
                lease_timeout_seconds=5.0,
            )],
            final_stage_id="android_stage",
            root_input={"message": "hello"},
            model_identity=ModelIdentity(**_android_capabilities()["models"][0]),
            workflow_id="wf_android_journal01",
        )
        assert output == {"content": "android fake journal result"}
        attempt = snapshot["stages"][0]["attempts"][0]
        assert attempt["provider_kind"] == "remote_full_worker"
        assert attempt["provider_node_id"] == "android_worker_01"
        # The fake worker feeds accept/result back into the provider; only the
        # coordinator-to-worker offer traverses the outbound callback.
        assert sent == ["stage_offer"]
        graph.commit_result("wf_android_journal01")
        persisted = journal.get_snapshot("wf_android_journal01")
        assert persisted["state"] == "completed"
        assert persisted["stages"][0]["attempts"][0]["provider_node_id"] == "android_worker_01"
    finally:
        graph.close()
        journal.close()
        provider.close()


def test_v1_and_transport_identity_mismatch_are_stably_rejected():
    coordinator = TaskWorkerControlPlane()

    old_ack = coordinator.receive_on_coordinator(
        "worker_01", _v1_hello().snapshot(), coordinator_node_id="master",
    )
    mismatch_ack = coordinator.receive_on_coordinator(
        "different_worker", _v1_hello("worker_01").snapshot(),
        coordinator_node_id="master",
    )

    assert old_ack.payload["accepted"] is False
    assert old_ack.payload["reason_code"] == "protocol_v2_required"
    assert mismatch_ack.payload["accepted"] is False
    assert mismatch_ack.payload["reason_code"] == "node_identity_mismatch"


def test_stage_messages_are_fenced_before_provider_dispatch():
    coordinator = TaskWorkerControlPlane()

    with pytest.raises(WorkerProtocolError) as captured:
        coordinator.receive_on_coordinator(
            "worker_01", _stage_offer().snapshot(), coordinator_node_id="master",
        )

    assert captured.value.code == "control_plane_only"
    assert coordinator.status(role="master")["rejected_message_count"] == 1


def test_duplicate_hello_is_idempotent_but_message_id_conflict_is_rejected():
    coordinator = TaskWorkerControlPlane()
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(), sent_at_ms=1000,
    )
    assert hello is not None

    first = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
        sent_at_ms=1001,
    )
    duplicate = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
        sent_at_ms=2000,
    )
    assert duplicate.snapshot() == first.snapshot()

    conflicting = copy.deepcopy(hello.snapshot())
    conflicting["payload"]["capabilities"]["max_concurrency"] = 2
    with pytest.raises(WorkerProtocolError) as captured:
        coordinator.receive_on_coordinator(
            "worker_01", conflicting, coordinator_node_id="master",
        )
    assert captured.value.code == "message_id_conflict"


def test_reconnected_hello_reopens_config_sync_even_when_capabilities_match():
    coordinator = TaskWorkerControlPlane()
    worker = TaskWorkerControlPlane()

    first = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(), sent_at_ms=1000,
    )
    assert first is not None
    coordinator.mark_worker_connection_pending("worker_01")
    coordinator.receive_on_coordinator(
        "worker_01", first.snapshot(), coordinator_node_id="master",
        sent_at_ms=1001,
    )
    coordinator.resolve_worker_connection_pending("worker_01")
    coordinator.disconnect_worker("worker_01")

    worker.disconnect_coordinator()
    second = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(), sent_at_ms=2000,
    )
    assert second is not None
    coordinator.mark_worker_connection_pending("worker_01")
    coordinator.receive_on_coordinator(
        "worker_01", second.snapshot(), coordinator_node_id="master",
        sent_at_ms=2001,
    )

    snapshot = coordinator.worker_snapshot("worker_01")
    assert snapshot["capabilities_changed"] is True
    assert snapshot["connection_rebound"] is True


def test_disconnect_clears_health_and_pending_hello_fence():
    worker = TaskWorkerControlPlane()
    first = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert first is not None
    assert worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    ) is None

    worker.disconnect_coordinator()
    assert worker.status(role="client")["control_plane_connected"] is False
    assert worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    ) is not None


def test_remote_provider_requires_exact_model_and_returns_fenced_identity():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_remote001",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_remote001",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert _wait_until(lambda: bool(sent))
    assert sent and sent[0].message_type == "stage_offer"
    offer = sent[0].payload
    identity = _response_identity(offer)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_acceptremote01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    output = {"content": "remote answer", "usage": {"total_tokens": 2}}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {
                "usage": {"total_tokens": 2},
                "usage_estimated": False,
                "model": "qwen-1_8b",
            },
        },
        message_id="msg_resultremote01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())

    assert finished.wait(2)
    assert "error" not in outcome
    result = outcome["result"]
    assert result.output == output
    assert result.provider_id == provider.provider_id
    assert result.attempt_id == attempt.attempt_id
    assert result.lease_epoch == 1
    provider.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0

    wrong_model_request = StageRequest(
        **{
            **request.__dict__,
            "model_identity": ModelIdentity(
                model_id="other-model",
                engine="pytorch",
                format="safetensors",
                revision="local-other",
                sha256="2" * 64,
            ),
        }
    )
    with pytest.raises(ProviderUnavailable) as mismatch:
        provider.reserve(wrong_model_request)
    assert mismatch.value.code == "model_identity_mismatch"


def test_remote_provider_accept_race_records_one_terminal_acceptance():
    """Concurrent duplicate accepts must not wake the same attempt twice."""
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_accept_race01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_accept_race01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert _wait_until(lambda: bool(sent))
    identity = _response_identity(sent[0].payload)
    start = threading.Barrier(3)
    results = []

    def accept(message_id):
        try:
            start.wait(timeout=2)
            provider.handle_message(build_message(
                "stage_accept",
                {**identity, "accepted": True, "reason_code": "", "retryable": False},
                message_id=message_id,
                sent_at_ms=int(time.time() * 1000),
                version=2,
            ).snapshot())
            results.append("accepted")
        except BaseException as exc:
            results.append(exc)

    threads = [
        threading.Thread(target=accept, args=(f"msg_accept_race0{index}",))
        for index in (1, 2)
    ]
    for thread in threads:
        thread.start()
    start.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=2)

    assert results.count("accepted") == 1
    errors = [item for item in results if isinstance(item, WorkerProtocolError)]
    assert len(errors) == 1
    assert errors[0].code == "duplicate_stage_response"

    output = {"content": "accepted once"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {},
        },
        message_id="msg_result_accept_race01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    assert finished.wait(2)
    assert "error" not in outcome
    assert outcome["result"].output == output
    provider.release(reservation.reservation_id)


def test_remote_v2_provider_rejects_partial_dependencies_before_offer():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = StageRequest(
        workflow_id="wf_partialremote1",
        request_id="request-partial-remote",
        stage_id="aggregate",
        stage_type="aggregate",
        provider_id=provider.provider_id,
        dependencies={
            "candidate_a": {"content": "surviving candidate"},
            DEPENDENCY_FAILURES_KEY: {
                "candidate_b": {
                    "state": "failed",
                    "error_code": "provider_execution_failed",
                },
            },
        },
        root_input={"message": "hello", "messages": []},
        model_identity=_model_identity(),
    )

    with pytest.raises(ProviderUnavailable) as captured:
        provider.reserve(request)

    assert captured.value.code == "partial_dependencies_not_supported"
    assert captured.value.retryable is True
    assert sent == []
    assert provider.inspect().active_reservations == 0


def test_partial_dependencies_fallback_from_v2_remote_to_local_provider():
    from task_graph import StageSpec, TaskGraphCoordinator

    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    coordinator = TaskGraphCoordinator()
    coordinator.register_provider(provider)

    def execute(request, cancel_event):
        if request.stage_id == "candidate_b":
            raise RuntimeError("candidate failed")
        if request.stage_type == "aggregate":
            assert DEPENDENCY_FAILURES_KEY in request.dependencies
            return {
                "content": request.dependencies["candidate_a"]["content"]
            }
        return {"content": "surviving candidate"}

    coordinator.register_provider(CallbackExecutionProvider(
        provider_id="local_full_model",
        executor=execute,
        supported_stage_types=("full_inference", "aggregate"),
    ))

    try:
        output, snapshot = coordinator.run(
            [
                StageSpec("candidate_a", "full_inference"),
                StageSpec("candidate_b", "full_inference"),
                StageSpec(
                    "aggregate",
                    "aggregate",
                    depends_on=("candidate_a", "candidate_b"),
                    provider=provider.provider_id,
                    fallback_providers=("local_full_model",),
                    pure=True,
                    minimum_successful_dependencies=1,
                ),
            ],
            "aggregate",
            {"message": "question"},
            model_identity=_model_identity(),
            workflow_id="wf_partialfallback1",
        )
    finally:
        coordinator.close()

    aggregate = next(
        stage for stage in snapshot["stages"]
        if stage["stage_id"] == "aggregate"
    )
    assert output == {"content": "surviving candidate"}
    assert snapshot["partial_result"] is True
    assert aggregate["retry_count"] == 1
    assert aggregate["same_provider_retry_count"] == 0
    assert aggregate["last_retry_error_code"] == (
        "partial_dependencies_not_supported"
    )
    assert aggregate["attempts"][0]["provider"] == "local_full_model"
    assert sent == []


def test_remote_provider_rejects_wrong_epoch_without_waking_attempt():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_remote002",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_remote002",
        lease_epoch=2,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert _wait_until(lambda: bool(sent))
    offer = sent[0].payload
    identity = _response_identity(offer)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_acceptremote02",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    output = {"content": "wrong epoch"}
    with pytest.raises(WorkerProtocolError) as wrong_epoch:
        provider.handle_message(build_message(
            "stage_result",
            {
                **identity,
                "lease_epoch": 1,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {},
            },
            message_id="msg_resultremote02",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
    assert wrong_epoch.value.code == "attempt_identity_mismatch"
    assert not finished.wait(0.1)

    correct = {"content": "correct epoch"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": correct,
            "output_sha256": canonical_sha256(correct),
            "metadata": {},
        },
        message_id="msg_resultremote03",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    assert finished.wait(2)
    assert outcome["result"].output == correct
    provider.release(reservation.reservation_id)


def test_remote_provider_disconnect_unblocks_pending_attempt():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_disconnect01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_disconnect01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert _wait_until(lambda: bool(sent))
    assert sent

    provider.notify_disconnect()

    assert finished.wait(2)
    assert outcome["error"].code == "remote_worker_disconnected"
    # ★ 2026-10-05（DIST-2）：断连现在会**主动回收** reservation，不再需要调用方
    #   补一次 `release()`。此前这里必须显式 `provider.release(...)` 才能让
    #   `active_reservations` 归零 —— 而真实路径上那次释放依赖上层
    #   `task_graph._run_stage` 的 `finally`；上层一旦没走到（异常路径、外层取消、
    #   进程卡住）就永久泄漏「已预留未执行」的槽位。
    assert provider.inspect().active_reservations == 0


def test_cancel_event_propagates_stage_cancel_to_worker():
    """★ #47（2026-10-05 三机实测）：本地取消必须**向 worker 传播** `stage_cancel`。

    此前等结果阶段的 `_wait` 在 `cancel_event` 置位时直接抛 `provider_cancelled`，
    **从不调用 `cancel()`**（本类里早已实现构造与发送 `stage_cancel` 的逻辑，却只在
    `remote_accept_timeout` 那条路径被调过）⇒ worker 白跑完整个 Stage，回传结果时
    本端已无对应 pending（实测 `event=task_worker_message_rejected
    reason=unknown_attempt`）。
    """
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_cancel01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_cancel01",
        lease_epoch=1,
        lease_expires_at=time.time() + 30,
    )
    cancel_event = threading.Event()
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            provider.execute(attempt, reservation, cancel_event)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    # 等 stage_offer 发出（说明 attempt 已登记在 `_pending`）。
    assert _wait_until(lambda: bool(sent))
    before = len(sent)

    cancel_event.set()

    assert finished.wait(3)
    assert outcome["error"].code == "provider_cancelled"
    # 关键断言：取消之后对端**应当收到** `stage_cancel`。
    assert _wait_until(
        lambda: any(
            "stage_cancel" in _message_text(item) for item in sent[before:]
        )
    ), f"取消后未向 worker 发送 stage_cancel，实际发送: {sent[before:]}"


def _message_text(raw) -> str:
    """把 provider 发出的原始消息解成可搜索的文本（bytes/str 都兼容）。"""
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", "replace")
    return str(raw)


def test_task_graph_commits_remote_result_through_existing_winner_gate():
    from task_graph import StageSpec, TaskGraphCoordinator

    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}

    def respond(message):
        offer = message.payload
        identity = _response_identity(offer)
        provider = provider_holder["provider"]
        provider.handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_graphaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        output = {"content": "remote graph result"}
        provider.handle_message(build_message(
            "stage_result",
            {
                **identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {},
            },
            message_id="msg_graphresult01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=respond,
    )
    provider_holder["provider"] = provider
    coordinator = TaskGraphCoordinator()
    coordinator.register_provider(provider)

    output, workflow = coordinator.run(
        stages=[StageSpec(
            "candidate_a",
            "full_inference",
            provider=provider.provider_id,
            fallback_providers=(),
            pure=False,
        )],
        final_stage_id="candidate_a",
        root_input={"message": "hello", "messages": []},
        model_identity=_model_identity(),
        workflow_id="wf_remotegraph01",
    )

    assert output == {"content": "remote graph result"}
    stage = workflow["stages"][0]
    assert stage["winner_attempt_id"]
    assert stage["attempts"][0]["provider_kind"] == (
        "remote_full_worker"
    )
    assert stage["attempts"][0]["provider_node_id"] == "worker_01"
    assert stage["fallback_providers"] == []
    coordinator.close()


def test_hello_round_trips_over_authenticated_loopback_tcp(monkeypatch):
    import config as cfg

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    monkeypatch.setattr(cfg, "CLUSTER_SECRET", "n2-control-plane-test-secret")
    monkeypatch.setattr(
        TCPClient, "_compute_local_model_sha256", staticmethod(lambda: "")
    )
    monkeypatch.setattr(
        TCPClient,
        "_heartbeat_loop",
        lambda self, connection_generation=None, connection_sock=None: None,
    )
    monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")

    coordinator = TaskWorkerControlPlane()
    worker = TaskWorkerControlPlane()
    completed = threading.Event()
    server = TCPServer(host="127.0.0.1", port=port)

    def on_server_message(client_id, message):
        if message.get("type") == MessageType.REGISTER.value:
            server.confirm_registration(client_id)
        elif message.get("type") == MessageType.TASK_WORKER.value:
            ack = coordinator.receive_on_coordinator(
                client_id,
                message["data"],
                coordinator_node_id="master",
            )
            server.send_to_client(
                client_id, ack.snapshot(), MessageType.TASK_WORKER,
            )

    def on_client_message(message):
        if message.get("type") == MessageType.TASK_WORKER.value:
            worker.receive_on_worker(message["data"])
            completed.set()

    client = TCPClient(
        server_host="127.0.0.1",
        server_port=port,
        client_id="worker_01",
        role="client",
    )
    try:
        server.start(on_message=on_server_message)
        assert client.connect(on_message=on_client_message) is True
        hello = worker.begin_worker_hello(
            node_id="worker_01", capabilities=_capabilities(),
        )
        assert hello is not None
        client.send_data(hello.snapshot(), MessageType.TASK_WORKER)
        assert completed.wait(3.0)
        assert coordinator.status(role="master")["connected_worker_count"] == 1
        assert worker.status(role="client")["control_plane_connected"] is True
    finally:
        client.disconnect()
        server.stop()


def test_single_stage_round_trips_over_authenticated_loopback_tcp(monkeypatch):
    import config as cfg

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setattr(cfg, "CLUSTER_SECRET", "n2-stage-loopback-test-secret")
    monkeypatch.setattr(
        TCPClient, "_compute_local_model_sha256", staticmethod(lambda: "")
    )
    monkeypatch.setattr(
        TCPClient,
        "_heartbeat_loop",
        lambda self, connection_generation=None, connection_sock=None: None,
    )
    monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")

    coordinator_control = TaskWorkerControlPlane()
    worker_control = TaskWorkerControlPlane()
    hello_completed = threading.Event()
    provider_holder = {}
    server = TCPServer(host="127.0.0.1", port=port)

    def on_server_message(client_id, outer):
        if outer.get("type") == MessageType.REGISTER.value:
            server.confirm_registration(client_id)
            return
        if outer.get("type") != MessageType.TASK_WORKER.value:
            return
        inner = outer["data"]
        if inner["message_type"] == "hello":
            ack = coordinator_control.receive_on_coordinator(
                client_id, inner, coordinator_node_id="master",
            )
            server.send_to_client(
                client_id, ack.snapshot(), MessageType.TASK_WORKER,
            )
        else:
            provider_holder["provider"].handle_message(inner)

    client = TCPClient(
        server_host="127.0.0.1",
        server_port=port,
        client_id="worker_01",
        role="client",
    )

    def on_client_message(outer):
        if outer.get("type") != MessageType.TASK_WORKER.value:
            return
        inner = outer["data"]
        if inner["message_type"] == "hello_ack":
            worker_control.receive_on_worker(inner)
            hello_completed.set()
            return
        offer = inner["payload"]
        identity = _response_identity(offer)
        accepted = build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_tcpaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        )
        client.send_data(accepted.snapshot(), MessageType.TASK_WORKER)
        output = {"content": "tcp remote result"}
        result = build_message(
            "stage_result",
            {
                **identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {},
            },
            message_id="msg_tcpresult01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        )
        client.send_data(result.snapshot(), MessageType.TASK_WORKER)

    try:
        server.start(on_message=on_server_message)
        assert client.connect(on_message=on_client_message)
        hello = worker_control.begin_worker_hello(
            node_id="worker_01", capabilities=_capabilities(),
        )
        assert hello is not None
        client.send_data(hello.snapshot(), MessageType.TASK_WORKER)
        assert hello_completed.wait(2)

        provider = RemoteFullWorkerProvider(
            node_id="worker_01",
            peer_snapshot=lambda: coordinator_control.worker_snapshot(
                "worker_01"
            ),
            send_message=lambda message: server.send_to_client(
                "worker_01", message.snapshot(), MessageType.TASK_WORKER,
            ),
        )
        provider_holder["provider"] = provider
        request = _remote_request(provider.provider_id)
        reservation = provider.reserve(request)
        attempt = StageAttempt(
            attempt_id="att_tcpstage01",
            request=request,
            provider_id=provider.provider_id,
            lease_id="lease_tcpstage01",
            lease_epoch=1,
            lease_expires_at=time.time() + 5,
        )

        result = provider.execute(
            attempt, reservation, threading.Event(),
        )

        assert result.output == {"content": "tcp remote result"}
        assert result.attempt_id == "att_tcpstage01"
        provider.release(reservation.reservation_id)
    finally:
        client.disconnect()
        server.stop()


def test_scheduler_binds_full_worker_hello_to_registered_pc_client(monkeypatch):
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["worker_01"] = NodeInfo(
        node_id="worker_01",
        role=NodeRole.CLIENT,
        node_type="pc",
        state=NodeState.ONLINE,
    )
    sent = []
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, node_id, data, message_type: sent.append(
            (node_id, data, message_type)
        ),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None

    scheduler._handle_task_worker_message(
        "worker_01", {"data": hello.snapshot()},
    )

    assert sent[0][0] == "worker_01"
    assert sent[0][2] == MessageType.TASK_WORKER
    assert sent[0][1]["message_type"] == "hello_ack"
    assert sent[0][1]["payload"]["accepted"] is True
    assert scheduler.get_task_worker_protocol_status()[
        "connected_worker_count"
    ] == 1


def test_recovery_hello_does_not_publish_two_layer_generations(monkeypatch):
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler._pipeline_recovery_pending = True
    scheduler.nodes["android_layer_01"] = NodeInfo(
        node_id="android_layer_01",
        role=NodeRole.CLIENT,
        node_type="android",
        state=NodeState.ONLINE,
    )
    sent = []
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, node_id, data, message_type: sent.append(
            (node_id, data, message_type)
        ),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    normal_pushes = []
    authoritative_pushes = []
    monkeypatch.setattr(
        scheduler, "push_layer_config_to_clients",
        lambda: normal_pushes.append(True),
    )
    monkeypatch.setattr(
        scheduler,
        "request_authoritative_layer_sync",
        lambda **kwargs: authoritative_pushes.append(kwargs) or True,
    )
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="android_layer_01",
        worker_kind="android_full_worker",
        capabilities=_android_layer_capabilities(),
    )
    assert hello is not None

    scheduler._handle_task_worker_message(
        "android_layer_01", {"data": hello.snapshot()},
    )

    assert sent[0][1]["payload"]["accepted"] is True
    assert normal_pushes == []
    assert authoritative_pushes == [{"require_distributed": True}]


def test_scheduler_full_worker_hello_releases_layer_reservation(monkeypatch):
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["worker_01"] = NodeInfo(
        node_id="worker_01",
        role=NodeRole.CLIENT,
        node_type="pc",
        state=NodeState.ONLINE,
    )
    sent = []
    opted_out = []
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, node_id, data, message_type: sent.append(
            (node_id, data, message_type)
        ),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    monkeypatch.setattr(
        scheduler,
        "_handle_layer_worker_opt_out",
        lambda node_id, msg: opted_out.append((node_id, msg)),
    )
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None

    scheduler._handle_task_worker_message(
        "worker_01", {"data": hello.snapshot()},
    )

    assert sent[0][1]["payload"]["accepted"] is True
    assert opted_out == [("worker_01", {"data": {"node_id": "worker_01"}})]


def test_full_task_worker_takes_precedence_over_automatic_layer_assignment(
    monkeypatch,
):
    import scheduler as scheduler_mod
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    monkeypatch.setattr(
        scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True,
    )
    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["worker_01"] = NodeInfo(
        node_id="worker_01",
        role=NodeRole.CLIENT,
        node_type="pc",
        state=NodeState.ONLINE,
    )
    sent = []
    scheduler._tcp_server = type("Server", (), {
        "_running": True,
        "clients": {"worker_01": object()},
        "get_client_ids": lambda self: ["worker_01"],
        "send_layer_config": lambda self, node_id, payload: sent.append(
            (node_id, payload)
        ),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    monkeypatch.setattr(
        scheduler,
        "get_layer_assignments",
        lambda: {"assignments": [
            {"node_id": "master", "start_layer": 0, "end_layer": 21},
            {"node_id": "worker_01", "start_layer": 21, "end_layer": 24},
        ]},
    )
    monkeypatch.setattr(
        scheduler,
        "_get_active_pipeline_model_info",
        lambda: {
            "model_id": "qwen-1_8b",
            "model_sha256": "1" * 64,
            "model_type": "qwen",
            "total_layers": 24,
            "quant_type": "fp16",
        },
    )
    monkeypatch.setattr(
        scheduler, "_task_worker_full_model_ids", lambda: {"worker_01"},
    )

    scheduler.push_layer_config_to_clients()

    assert sent
    assert sent[-1][0] == "worker_01"
    assert sent[-1][1]["release"] is True
    assert "start_layer" not in sent[-1][1]
    assert "worker_01" in scheduler._pipeline_worker_opt_out


def test_route_a_android_assignment_never_publishes_legacy_layer_config(monkeypatch):
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["android_01"] = NodeInfo(
        node_id="android_01",
        role=NodeRole.CLIENT,
        node_type="android",
        state=NodeState.ONLINE,
    )
    sent = []
    scheduler._tcp_server = type("Server", (), {
        "_running": True,
        "clients": {"android_01": object()},
        "get_client_ids": lambda self: ["android_01"],
        "send_layer_config": lambda self, node_id, payload: sent.append(
            (node_id, payload)
        ),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    monkeypatch.setattr(
        scheduler,
        "_task_worker_layer_stage_ids",
        lambda connected: {"android_01"},
    )
    monkeypatch.setattr(
        scheduler,
        "get_layer_assignments",
        lambda: {"assignments": [
            {"node_id": "master", "start_layer": 0, "end_layer": 4},
            {"node_id": "android_01", "start_layer": 4, "end_layer": 8},
        ]},
    )
    monkeypatch.setattr(
        scheduler,
        "_get_active_pipeline_model_info",
        lambda: {
            "model_id": "qwen-1_8b",
            "model_sha256": "1" * 64,
            "model_type": "qwen",
            "total_layers": 8,
            "quant_type": "fp16",
        },
    )

    scheduler.push_layer_config_to_clients()

    assert sent == []
    assert scheduler._worker_assignments.expected_configs() == {}


@pytest.mark.parametrize("previously_opted_out", [False, True])
def test_configured_relay_worker_precedes_full_task_worker_opt_out(
        monkeypatch, previously_opted_out):
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    # This covers the configured legacy relay path; do not inherit a
    # metadata-only pipeline preparation from the process-wide ModelHost.
    monkeypatch.setattr(
        scheduler, "_host", type("Host", (), {"is_pipeline_prepared": False})(),
    )
    scheduler.nodes["worker_01"] = NodeInfo(
        node_id="worker_01",
        role=NodeRole.CLIENT,
        node_type="pc",
        state=NodeState.ONLINE,
    )
    relay_spec = {
        "role": "middle",
        "host": "127.0.0.1",
        "port": 50183,
        "n_embd": 896,
        "timeout": 5.0,
        "layer_start": 8,
        "layer_end": 16,
    }
    sent = []
    scheduler._tcp_server = type("Server", (), {
        "_running": True,
        "clients": {"worker_01": object()},
        "get_client_ids": lambda self: ["worker_01"],
        "send_layer_config": lambda self, node_id, payload: sent.append(
            (node_id, payload)
        ),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    monkeypatch.setattr(
        scheduler,
        "get_layer_assignments",
        lambda: {"assignments": [
            {"node_id": "master", "start_layer": 0, "end_layer": 21},
            {"node_id": "worker_01", "start_layer": 21, "end_layer": 24},
        ]},
    )
    monkeypatch.setattr(
        scheduler,
        "_get_active_pipeline_model_info",
        lambda: {
            "model_id": "qwen-1_8b",
            "model_sha256": "1" * 64,
            "model_type": "qwen",
            "total_layers": 24,
            "quant_type": "fp16",
        },
    )
    monkeypatch.setattr(
        scheduler, "_task_worker_full_model_ids", lambda: {"worker_01"},
    )
    monkeypatch.setattr(
        scheduler,
        "_relay_segment_for_worker",
        lambda node_id, routing_preference="auto": (
            relay_spec if node_id == "worker_01" else None
        ),
    )
    monkeypatch.setattr(scheduler, "_start_layer_config_retry_monitor", lambda: None)
    if previously_opted_out:
        scheduler._pipeline_worker_opt_out.add("worker_01")

    scheduler.push_layer_config_to_clients()
    scheduler._handle_layer_worker_opt_out("worker_01", {"data": {
        "node_id": "worker_01",
    }})

    assert "worker_01" not in scheduler._pipeline_worker_opt_out
    assert len(sent) == 2
    for node_id, config in sent:
        assert node_id == "worker_01"
        assert config.get("release") is not True
        assert config["engine"] == "relay_middle"
        assert config["relay_segment"] == relay_spec


def test_scheduler_experimental_gate_reports_physical_validation_pending(
    monkeypatch,
):
    import scheduler as scheduler_mod
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    monkeypatch.setattr(
        scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True,
    )
    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["worker_01"] = NodeInfo(
        node_id="worker_01",
        role=NodeRole.CLIENT,
        node_type="pc",
        state=NodeState.ONLINE,
    )
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, *args: None,
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    scheduler._handle_task_worker_message(
        "worker_01", {"data": hello.snapshot()},
    )

    status = scheduler.get_task_worker_protocol_status()
    assert status["experiment_enabled"] is True
    assert status["experimental_dispatch_enabled"] is True
    assert status["adapter_connected"] is False
    assert status["task_dispatch_enabled"] is False
    assert status["admission_state"] == (
        "n2_4_experimental_physical_validation_pending"
    )


def test_scheduler_does_not_advertise_worker_without_full_model_as_dispatchable(
        monkeypatch):
    import scheduler as scheduler_mod
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    monkeypatch.setattr(
        scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True,
    )
    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["worker_01"] = NodeInfo(
        node_id="worker_01",
        role=NodeRole.CLIENT,
        node_type="pc",
        state=NodeState.ONLINE,
    )
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, *args: None,
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    worker = TaskWorkerControlPlane()
    capabilities = {**_capabilities(), "models": []}
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=capabilities,
    )
    assert hello is not None
    scheduler._handle_task_worker_message(
        "worker_01", {"data": hello.snapshot()},
    )

    status = scheduler.get_task_worker_protocol_status()
    assert status["connected_worker_count"] == 1
    assert status["full_model_worker_count"] == 0
    assert status["workers_missing_full_model"] == ["worker_01"]
    assert status["experimental_dispatch_enabled"] is False
    assert status["worker_readiness_reason"] == (
        "task_worker_full_model_not_advertised"
    )


def test_scheduler_rejects_android_client_full_worker_claim(monkeypatch):
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["android_01"] = NodeInfo(
        node_id="android_01",
        role=NodeRole.CLIENT,
        node_type="android",
        state=NodeState.ONLINE,
    )
    sent = []
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, *args: sent.append(args),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="android_01", capabilities=_capabilities(),
    )
    assert hello is not None

    scheduler._handle_task_worker_message(
        "android_01", {"data": hello.snapshot()},
    )

    assert sent == []
    status = scheduler.get_task_worker_protocol_status()
    assert status["connected_worker_count"] == 0
    assert status["rejected_message_count"] == 1


def test_scheduler_does_not_advertise_a_layer_partition_as_full_model(
        monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    partition_manager = SimpleNamespace(
        is_loaded=True,
        layer_range=(0, 12),
    )
    fake_api = SimpleNamespace(
        model_loaded=True,
        model_manager=SimpleNamespace(_instance=partition_manager),
    )
    monkeypatch.setattr(scheduler, "_host", fake_api)

    capabilities = scheduler._task_worker_capabilities()

    assert capabilities["models"] == []
    assert capabilities["stage_types"] == ["full_inference", "aggregate"]


def test_scheduler_advertises_endpoint_relay_as_layer_worker_without_local_model(
        monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    monkeypatch.setattr(
        scheduler, "_host", SimpleNamespace(
            model_loaded=False, is_loaded=False, layer_range=None,
        ),
    )
    scheduler._active_layer_config = {
        "engine": "relay_middle",
        "config_id": "cfg-relay",
        "layer_range": [8, 16],
    }

    capabilities = scheduler._task_worker_capabilities()

    assert capabilities["models"] == []
    assert capabilities["layer_worker"] is True
    assert capabilities["relay_middle"] is True


def test_scheduler_advertises_modelhost_loaded_model_without_legacy_is_loaded(
    monkeypatch,
):
    """The ModelHost loaded-state proxy is sufficient for Full Worker hello."""
    from scheduler import Scheduler

    identity = ModelIdentity(
        model_id="qwen-1_8b",
        engine="pytorch",
        format="safetensors",
        revision="local-rev1",
        sha256="2" * 64,
    )
    fake_host = SimpleNamespace(
        has_loaded_model=lambda: True,
        layer_range=None,
    )
    scheduler = Scheduler()
    monkeypatch.setattr(scheduler, "_host", fake_host)
    scheduler.configure_callbacks(_scheduler_callbacks(identity=lambda: identity))

    capabilities = scheduler._task_worker_capabilities()

    assert capabilities["models"] == [identity.snapshot()]


def test_scheduler_worker_gate_blocks_hello_and_rejects_stale_offer(
    monkeypatch,
):
    import scheduler as scheduler_mod
    from scheduler import Scheduler

    monkeypatch.setattr(
        scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", False,
    )
    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    sent = []

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))

    scheduler._tcp_client = Client()
    assert scheduler._send_task_worker_hello() is False
    assert sent == []

    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    ))
    scheduler.configure_callbacks(_scheduler_callbacks(
        execute=lambda *args: pytest.fail("a disabled worker must not execute a Stage"),
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_gateworker01",
            "request_id": "request-gate-worker",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_gateworker01",
            "lease_id": "lease_gateworker01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_gateworkeroffer01",
        sent_at_ms=now_ms,
        version=2,
    )

    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})

    assert _wait_until(lambda: bool(sent))
    response = sent[-1][0]
    assert response["message_type"] == "stage_accept"
    assert response["payload"]["accepted"] is False
    assert response["payload"]["reason_code"] == "worker_experiment_disabled"
    assert response["payload"]["retryable"] is True
    assert scheduler._task_worker_active_attempts == {}


def test_scheduler_worker_executes_one_admitted_stage_and_returns_result(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator_control = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator_control.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())

    sent = []
    completed = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data.get("message_type") in {"stage_result", "stage_error"}:
                completed.set()

    scheduler._tcp_client = Client()
    fake_api = SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    )
    monkeypatch.setattr(scheduler, "_host", fake_api)
    scheduler.configure_callbacks(_scheduler_callbacks(
        execute=lambda request, cancel_event: {
            "content": f"remote:{request.stage_id}",
            "usage": {"total_tokens": 3},
            "model": "qwen-1_8b",
        },
    ))
    root_input = {"message": "hello", "messages": []}
    dependencies = {}
    now_ms = int(time.time() * 1000)
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_workerexec01",
            "request_id": "request-worker-exec",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_workerexec01",
            "lease_id": "lease_workerexec01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": dependencies,
            "input_sha256": stage_input_sha256(root_input, dependencies),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_workerexec01",
        sent_at_ms=now_ms,
        version=2,
    )

    scheduler._handle_task_worker_message(
        "master", {"data": offer.snapshot()},
    )

    assert completed.wait(2)
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result",
    ]
    assert sent[0][0]["payload"]["accepted"] is True
    assert sent[1][0]["payload"]["output"]["content"] == (
        "remote:candidate_a"
    )
    assert sent[1][0]["payload"]["attempt_id"] == "att_workerexec01"


@pytest.mark.parametrize(
    (
        "blocked_lock_name", "request_budget_ms", "lease_budget_ms",
        "expected_reason",
    ),
    [
        (
            "full_chat_execution_lock", 120, 600,
            "request_deadline_exceeded",
        ),
        ("_inference_lock", 600, 120, "lease_expired"),
    ],
)
def test_scheduler_worker_lock_wait_stops_at_earliest_attempt_boundary(
    monkeypatch,
    blocked_lock_name,
    request_budget_ms,
    lease_budget_ms,
    expected_reason,
):
    """Lock queues consume one request/lease budget and never call executor late."""
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())

    full_chat_lock = threading.Lock()
    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=full_chat_lock,
    ))
    executor_calls = []

    def execute(request, _cancel_event):
        executor_calls.append(request.stage_id)
        return {"content": f"completed:{request.stage_id}"}

    scheduler.configure_callbacks(_scheduler_callbacks(execute=execute))
    sent = []
    first_terminal = threading.Event()
    followup_terminal = threading.Event()
    delayed_terminal = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if (
                data["payload"].get("attempt_id") == "att_lockwait01"
                and data["message_type"] == "stage_error"
            ):
                first_terminal.set()
            if (
                data["payload"].get("attempt_id") == "att_lockfollow01"
                and data["message_type"] == "stage_result"
            ):
                followup_terminal.set()
            if (
                data["payload"].get("attempt_id") == "att_delaydeadline01"
                and data["message_type"] == "stage_error"
            ):
                delayed_terminal.set()

    scheduler._tcp_client = Client()

    def make_offer(
        *,
        attempt_id,
        message_id,
        lease_id,
        lease_ms,
        request_deadline_ms=None,
    ):
        sent_at_ms = int(time.time() * 1000)
        root_input = {"message": "hello"}
        payload = {
            "workflow_id": f"wf_{attempt_id}",
            "request_id": f"request-{attempt_id}",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": attempt_id,
            "lease_id": lease_id,
            "lease_epoch": 1,
            "lease_expires_at_ms": sent_at_ms + lease_ms,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        }
        if request_deadline_ms is not None:
            payload["request_deadline_ms"] = request_deadline_ms
        return build_message(
            "stage_offer", payload,
            message_id=message_id,
            sent_at_ms=sent_at_ms,
            version=2,
        )

    first_sent_at_ms = int(time.time() * 1000)
    first_offer = make_offer(
        attempt_id="att_lockwait01",
        message_id="msg_lockwaitoffer01",
        lease_id="lease_lockwait01",
        lease_ms=lease_budget_ms,
        request_deadline_ms=first_sent_at_ms + request_budget_ms,
    )
    blocked_lock = (
        full_chat_lock
        if blocked_lock_name == "full_chat_execution_lock"
        else scheduler._inference_lock
    )
    blocked_lock.acquire()
    blocked_lock_held = True
    try:
        scheduler._handle_task_worker_message(
            "master", {"data": first_offer.snapshot()},
        )
        assert first_terminal.wait(2)
        assert _wait_until(lambda: not scheduler._task_worker_active_attempts)
    finally:
        if blocked_lock_held:
            blocked_lock.release()
            blocked_lock_held = False

    first_error = next(
        data for data, _message_type in sent
        if data["message_type"] == "stage_error"
        and data["payload"]["attempt_id"] == "att_lockwait01"
    )
    assert first_error["payload"]["error_code"] == expected_reason
    assert first_error["payload"]["retryable"] is (
        expected_reason == "lease_expired"
    )
    assert executor_calls == []

    # When the second lock timed out, the first lock must also have been
    # released before the attempt leaves the active registry.
    assert full_chat_lock.acquire(blocking=False)
    full_chat_lock.release()

    followup_offer = make_offer(
        attempt_id="att_lockfollow01",
        message_id="msg_lockfollowoffer01",
        lease_id="lease_lockfollow01",
        lease_ms=1500,
    )
    scheduler._handle_task_worker_message(
        "master", {"data": followup_offer.snapshot()},
    )

    assert followup_terminal.wait(2)
    assert executor_calls == ["candidate_a"]
    followup_types = [
        data["message_type"] for data, _message_type in sent
        if data["payload"].get("attempt_id") == "att_lockfollow01"
    ]
    assert followup_types == ["stage_accept", "stage_result"]

    if expected_reason == "request_deadline_exceeded":
        assert _wait_until(lambda: not scheduler._task_worker_active_attempts)
        delayed_deadline_ms = int(time.time() * 1000) + 80
        delayed_offer = make_offer(
            attempt_id="att_delaydeadline01",
            message_id="msg_delaydeadlineoffer01",
            lease_id="lease_delaydeadline01",
            lease_ms=1500,
            request_deadline_ms=delayed_deadline_ms,
        )
        threading.Event().wait(0.12)
        scheduler._handle_task_worker_message(
            "master", {"data": delayed_offer.snapshot()},
        )

        assert delayed_terminal.wait(2)
        delayed_error = next(
            data for data, _message_type in sent
            if data["message_type"] == "stage_error"
            and data["payload"]["attempt_id"] == "att_delaydeadline01"
        )
        assert delayed_error["payload"]["error_code"] == (
            "request_deadline_exceeded"
        )
        assert executor_calls == ["candidate_a"]


def test_scheduler_worker_rechecks_model_identity_when_offer_arrives(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    initial_capabilities = _capabilities()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=initial_capabilities,
    )
    assert hello is not None
    coordinator = TaskWorkerControlPlane()
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    changed_capabilities = _capabilities()
    changed_capabilities["models"] = []
    monkeypatch.setattr(
        scheduler, "_task_worker_capabilities", lambda: changed_capabilities,
    )
    sent = []
    rejected = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            rejected.set()

    scheduler._tcp_client = Client()
    root_input = {"message": "hello", "messages": []}
    now_ms = int(time.time() * 1000)
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_modelcheck01",
            "request_id": "request-model-check",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_modelcheck01",
            "lease_id": "lease_modelcheck01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_modelcheck01",
        sent_at_ms=now_ms,
        version=2,
    )

    scheduler._handle_task_worker_message(
        "master", {"data": offer.snapshot()},
    )

    assert rejected.wait(2)
    assert len(sent) == 1
    assert sent[0][0]["message_type"] == "stage_accept"
    assert sent[0][0]["payload"]["accepted"] is False
    assert sent[0][0]["payload"]["reason_code"] == (
        "model_identity_mismatch"
    )


def test_remote_provider_renews_lease_and_finishes_after_original_deadline():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    original_deadline = time.time() + 0.2
    attempt = StageAttempt(
        attempt_id="att_renewremote01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_renewremote01",
        lease_epoch=1,
        lease_expires_at=original_deadline,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert _wait_until(lambda: bool(sent))
    offer = sent[0].payload
    identity = _response_identity(offer)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_renewaccept01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())

    renewed_deadline = time.time() + 1.0
    assert provider.renew_lease(
        attempt.attempt_id,
        attempt.lease_id,
        attempt.lease_epoch,
        renewed_deadline,
    ) is True
    assert _wait_until(
        lambda: any(message.message_type == "lease_renew" for message in sent)
    )
    assert sent[-1].message_type == "lease_renew"
    threading.Event().wait(max(0.0, original_deadline - time.time()) + 0.05)
    assert not finished.is_set()

    output = {"content": "after renewal"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {},
        },
        message_id="msg_renewresult01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    assert finished.wait(2)
    assert outcome["result"].output == output
    provider.release(reservation.reservation_id)


def test_remote_provider_sends_cancel_and_accepts_cancel_ack():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_cancelremote01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_cancelremote01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert _wait_until(lambda: bool(sent))
    identity = _response_identity(sent[0].payload)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_cancelaccept01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())

    provider.cancel(attempt.attempt_id)
    assert _wait_until(
        lambda: any(message.message_type == "stage_cancel" for message in sent)
    )
    assert sent[-1].message_type == "stage_cancel"
    provider.handle_message(build_message(
        "stage_cancelled",
        {**identity, "reason_code": "coordinator_cancelled"},
        message_id="msg_cancelledremote01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    assert finished.wait(2)
    assert outcome["error"].code == "provider_cancelled"
    provider.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0


def test_remote_provider_absorbs_stage_responses_crossing_cancel():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_cancellate01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_cancellate01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert _wait_until(lambda: bool(sent))
    identity = _response_identity(sent[0].payload)

    provider.cancel(attempt.attempt_id)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_cancellateaccept01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    output = {"content": "finished while cancel crossed the wire"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {},
        },
        message_id="msg_cancellateresult01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    provider.handle_message(build_message(
        "stage_cancelled",
        {**identity, "reason_code": "coordinator_cancelled"},
        message_id="msg_cancellateack01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())

    thread.join(2)
    assert not thread.is_alive()
    assert outcome["error"].code == "provider_cancelled"
    provider.release(reservation.reservation_id)
    provider.close()


def test_remote_cancel_is_deferred_until_blocked_offer_send_completes():
    coordinator_control, _worker_control = _admitted_control_plane()
    offer_send_started = threading.Event()
    finish_offer_send = threading.Event()
    cancel_sent = threading.Event()
    send_order = []
    queued_types = []

    def send(message):
        if message.message_type == "stage_offer":
            send_order.append("stage_offer_started")
            offer_send_started.set()
            assert finish_offer_send.wait(2)
            send_order.append("stage_offer_finished")
            return
        send_order.append(message.message_type)
        if message.message_type == "stage_cancel":
            cancel_sent.set()

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=send,
    )
    original_queue = provider._queue_outbound_message

    def record_queue(message, on_error):
        queued_types.append(message.message_type)
        return original_queue(message, on_error)

    provider._queue_outbound_message = record_queue
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_cancelorder01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_cancelorder01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    assert offer_send_started.wait(2)

    provider.cancel(attempt.attempt_id)

    # Cancellation wakes local waiters immediately, but its wire message must
    # not be queued while the synchronous offer transport call is in flight.
    assert queued_types == []
    assert send_order == ["stage_offer_started"]

    finish_offer_send.set()
    assert cancel_sent.wait(2)
    assert finished.wait(2)
    assert queued_types == ["stage_cancel"]
    assert send_order == [
        "stage_offer_started", "stage_offer_finished", "stage_cancel",
    ]
    assert outcome["error"].code == "provider_cancelled"
    provider.release(reservation.reservation_id)
    provider.close()


def test_remote_cancel_racing_failed_offer_retains_terminal_tombstone():
    coordinator_control, _worker_control = _admitted_control_plane()
    offer_send_started = threading.Event()
    finish_offer_send = threading.Event()

    def send(message):
        if message.message_type == "stage_offer":
            offer_send_started.set()
            assert finish_offer_send.wait(2)
            raise ConnectionError("disconnected before offer was sent")

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=send,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_cancelfail01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_cancelfail01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert offer_send_started.wait(2)
    provider.cancel(attempt.attempt_id)
    finish_offer_send.set()
    thread.join(2)

    assert outcome["error"].code == "provider_cancelled"
    provider.release(reservation.reservation_id)
    pending = provider._pending[attempt.attempt_id]
    assert pending.released is True
    assert pending.terminal_snapshot()[0] == "provider_cancelled"
    provider.close()


def test_remote_cancel_and_renew_do_not_block_on_network_send():
    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}
    outbound_block = threading.Event()
    offer_sent = threading.Event()
    sent_types = []

    def send(message):
        sent_types.append(message.message_type)
        if message.message_type == "stage_offer":
            offer_sent.set()
            identity = _response_identity(message.payload)
            provider_holder["provider"].handle_message(build_message(
                "stage_accept",
                {
                    **identity,
                    "accepted": True,
                    "reason_code": "",
                    "retryable": False,
                },
                message_id="msg_nonblockingaccept01",
                sent_at_ms=int(time.time() * 1000),
                version=2,
            ).snapshot())
            return
        outbound_block.wait(1.0)

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=send,
    )
    provider_holder["provider"] = provider
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_nonblocking01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_nonblocking01",
        lease_epoch=1,
        lease_expires_at=time.time() + 2.0,
    )
    outcome = {}

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert offer_sent.wait(2)

    started = time.perf_counter()
    assert provider.renew_lease(
        attempt.attempt_id,
        attempt.lease_id,
        attempt.lease_epoch,
        time.time() + 3.0,
    ) is True
    renew_elapsed = time.perf_counter() - started
    assert _wait_until(lambda: "lease_renew" in sent_types)

    started = time.perf_counter()
    provider.cancel(attempt.attempt_id)
    cancel_elapsed = time.perf_counter() - started

    assert renew_elapsed < 0.1
    assert cancel_elapsed < 0.1
    thread.join(2)
    assert not thread.is_alive()
    assert outcome["error"].code == "provider_cancelled"
    outbound_block.set()
    assert _wait_until(lambda: "stage_cancel" in sent_types)
    provider.release(reservation.reservation_id)
    provider.close()


def test_task_graph_auto_renews_remote_lease_before_result():
    from task_graph import StageSpec, TaskGraphCoordinator

    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}
    sent_types = []

    def send(message):
        sent_types.append(message.message_type)
        if message.message_type != "stage_offer":
            return
        offer = message.payload
        identity = _response_identity(offer)
        provider = provider_holder["provider"]
        provider.handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_autorenewaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

        def finish_later():
            threading.Event().wait(0.35)
            output = {"content": "renewed graph result"}
            provider.handle_message(build_message(
                "stage_result",
                {
                    **identity,
                    "output": output,
                    "output_sha256": canonical_sha256(output),
                    "metadata": {},
                },
                message_id="msg_autorenewresult01",
                sent_at_ms=int(time.time() * 1000),
                version=2,
            ).snapshot())

        threading.Thread(target=finish_later, daemon=True).start()

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=send,
    )
    provider_holder["provider"] = provider
    coordinator = TaskGraphCoordinator()
    coordinator.register_provider(provider)

    output, workflow = coordinator.run(
        stages=[StageSpec(
            "candidate_a",
            "full_inference",
            provider=provider.provider_id,
            fallback_providers=(),
            pure=False,
            lease_timeout_seconds=0.15,
        )],
        final_stage_id="candidate_a",
        root_input={"message": "hello", "messages": []},
        model_identity=_model_identity(),
        workflow_id="wf_autorenewremote01",
    )

    assert output == {"content": "renewed graph result"}
    assert sent_types.count("lease_renew") >= 2
    assert workflow["stages"][0]["attempts"][0]["state"] == "completed"
    assert provider.inspect().active_reservations == 0
    coordinator.close()


def test_task_graph_disconnect_and_accept_timeout_release_remote_slot():
    from task_graph import StageSpec, TaskGraphCoordinator, WorkflowExecutionError

    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}
    offered = threading.Event()

    def accept_only(message):
        if message.message_type != "stage_offer":
            return
        identity = _response_identity(message.payload)
        provider_holder["provider"].handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_disconnectaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        offered.set()

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=accept_only,
    )
    provider_holder["provider"] = provider
    coordinator = TaskGraphCoordinator()
    coordinator.register_provider(provider)
    errors = []

    def run_disconnected():
        try:
            coordinator.run(
                stages=[StageSpec(
                    "candidate_a",
                    "full_inference",
                    provider=provider.provider_id,
                    fallback_providers=(),
                    pure=False,
                    lease_timeout_seconds=1.0,
                )],
                final_stage_id="candidate_a",
                root_input={"message": "hello"},
                model_identity=_model_identity(),
                workflow_id="wf_disconnectremote01",
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_disconnected, daemon=True)
    thread.start()
    assert offered.wait(2)
    provider.notify_disconnect()
    thread.join(2)
    assert not thread.is_alive()
    assert isinstance(errors[0], WorkflowExecutionError)
    assert provider.inspect().active_reservations == 0
    coordinator.close()

    timeout_control, _ = _admitted_control_plane()
    timeout_sent = []
    timeout_provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: timeout_control.worker_snapshot("worker_01"),
        send_message=timeout_sent.append,
    )
    timeout_coordinator = TaskGraphCoordinator()
    timeout_coordinator.register_provider(timeout_provider)
    with pytest.raises(WorkflowExecutionError):
        timeout_coordinator.run(
            stages=[StageSpec(
                "candidate_a",
                "full_inference",
                provider=timeout_provider.provider_id,
                fallback_providers=(),
                pure=False,
                accept_timeout_seconds=0.1,
                lease_timeout_seconds=1.0,
            )],
            final_stage_id="candidate_a",
            root_input={"message": "hello"},
            model_identity=_model_identity(),
            workflow_id="wf_accepttimeout01",
        )
    assert _wait_until(
        lambda: any(
            message.message_type == "stage_cancel"
            for message in timeout_sent
        )
    )
    assert [message.message_type for message in timeout_sent] == [
        "stage_offer", "stage_cancel",
    ]
    assert timeout_provider.inspect().active_reservations == 0
    timeout_coordinator.close()


def test_scheduler_worker_replays_duplicate_offer_without_second_execution(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    completed = threading.Event()
    calls = []

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if len(sent) >= 2 and data["message_type"] == "stage_result":
                completed.set()

    scheduler._tcp_client = Client()

    def execute(request, cancel_event):
        calls.append(request.stage_id)
        return {"content": "once"}

    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    ))
    scheduler.configure_callbacks(_scheduler_callbacks(execute=execute))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_replayoffer01",
            "request_id": "request-replay-offer",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_replayoffer01",
            "lease_id": "lease_replayoffer01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_replayoffer01",
        sent_at_ms=now_ms,
        version=2,
    )

    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert completed.wait(2)
    first_responses = [item[0] for item in sent]
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})

    assert calls == ["candidate_a"]
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result", "stage_accept", "stage_result",
    ]
    assert sent[2][0] == first_responses[0]
    assert sent[3][0] == first_responses[1]


def test_scheduler_worker_lease_renew_extends_active_execution(monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    accepted = threading.Event()
    completed = threading.Event()
    release_execution = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] == "stage_accept":
                accepted.set()
            if data["message_type"] in {"stage_result", "stage_error"}:
                completed.set()

    scheduler._tcp_client = Client()

    def execute(request, cancel_event):
        while not release_execution.wait(0.01):
            if cancel_event.is_set():
                raise RuntimeError("cancelled")
        return {"content": "renewed worker result"}

    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    ))
    scheduler.configure_callbacks(_scheduler_callbacks(execute=execute))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_workerrenew01",
            "request_id": "request-worker-renew",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_workerrenew01",
            "lease_id": "lease_workerrenew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 250,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_workerrenewoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert accepted.wait(2)

    renew_now_ms = int(time.time() * 1000)
    renewal = build_message(
        "lease_renew",
        {
            "workflow_id": "wf_workerrenew01",
            "stage_id": "candidate_a",
            "attempt_id": "att_workerrenew01",
            "lease_id": "lease_workerrenew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": renew_now_ms + 1000,
        },
        message_id="msg_workerrenew01",
        sent_at_ms=renew_now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message(
        "master", {"data": renewal.snapshot()},
    )
    threading.Event().wait(max(0.0, (now_ms + 300) / 1000.0 - time.time()))
    assert not completed.is_set()

    release_execution.set()
    assert completed.wait(2)
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result",
    ]
    assert sent[-1][0]["payload"]["output"]["content"] == (
        "renewed worker result"
    )


def test_scheduler_worker_cancel_ack_is_replayed_without_stage_error(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    accepted = threading.Event()
    cancelled = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] == "stage_accept":
                accepted.set()
            if data["message_type"] == "stage_cancelled":
                cancelled.set()

    scheduler._tcp_client = Client()

    def execute(request, cancel_event):
        assert cancel_event.wait(2)
        return {"content": "must not be committed"}

    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    ))
    scheduler.configure_callbacks(_scheduler_callbacks(execute=execute))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_workercancel01",
            "request_id": "request-worker-cancel",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_workercancel01",
            "lease_id": "lease_workercancel01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_workercanceloffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert accepted.wait(2)

    cancel = build_message(
        "stage_cancel",
        {
            "workflow_id": "wf_workercancel01",
            "stage_id": "candidate_a",
            "attempt_id": "att_workercancel01",
            "lease_id": "lease_workercancel01",
            "lease_epoch": 1,
            "reason_code": "coordinator_cancelled",
        },
        message_id="msg_workercancel01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": cancel.snapshot()})
    assert cancelled.wait(2)
    assert _wait_until(lambda: not scheduler._task_worker_active_attempts)
    scheduler._handle_task_worker_message("master", {"data": cancel.snapshot()})

    message_types = [item[0]["message_type"] for item in sent]
    assert message_types == [
        "stage_accept", "stage_cancelled", "stage_cancelled",
    ]
    assert "stage_error" not in message_types
    assert sent[1][0] == sent[2][0]


def test_scheduler_worker_acknowledges_cancel_after_stage_completed(monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    completed = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] == "stage_result":
                completed.set()

    scheduler._tcp_client = Client()
    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    ))
    scheduler.configure_callbacks(_scheduler_callbacks(
        execute=lambda _request, _cancel_event: {"content": "already complete"},
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer_payload = {
        "workflow_id": "wf_workercancellate01",
        "request_id": "request-worker-cancel-late",
        "stage_id": "candidate_a",
        "stage_type": "full_inference",
        "attempt_id": "att_workercancellate01",
        "lease_id": "lease_workercancellate01",
        "lease_epoch": 1,
        "lease_expires_at_ms": now_ms + 5000,
        "provider_id": remote_provider_id("worker_01"),
        "root_input": root_input,
        "dependencies": {},
        "input_sha256": stage_input_sha256(root_input, {}),
        "model_identity": _model_identity().snapshot(),
    }
    offer = build_message(
        "stage_offer", offer_payload,
        message_id="msg_workercancellateoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert completed.wait(2)
    assert _wait_until(lambda: not scheduler._task_worker_active_attempts)

    cancel = build_message(
        "stage_cancel",
        {
            key: offer_payload[key]
            for key in (
                "workflow_id", "stage_id", "attempt_id", "lease_id", "lease_epoch",
            )
        } | {"reason_code": "coordinator_cancelled"},
        message_id="msg_workercancellate01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": cancel.snapshot()})

    assert _wait_until(
        lambda: sent and sent[-1][0]["message_type"] == "stage_cancelled",
    )
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result", "stage_cancelled",
    ]


def test_scheduler_worker_replays_cached_result_after_send_failure(monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    first_sent = []
    result_failed = threading.Event()
    calls = []

    class FailingResultClient:
        is_registered = True

        def send_data(self, data, message_type):
            first_sent.append((data, message_type))
            if data["message_type"] == "stage_result":
                result_failed.set()
                raise ConnectionError("result transport failed")

    scheduler._tcp_client = FailingResultClient()

    def execute(request, cancel_event):
        calls.append(request.stage_id)
        return {"content": "cached result"}

    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    ))
    scheduler.configure_callbacks(_scheduler_callbacks(execute=execute))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_resultretry01",
            "request_id": "request-result-retry",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_resultretry01",
            "lease_id": "lease_resultretry01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_resultretryoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert result_failed.wait(2)
    assert _wait_until(lambda: not scheduler._task_worker_active_attempts)

    replayed = []

    class ReconnectedClient:
        is_registered = True

        def send_data(self, data, message_type):
            replayed.append((data, message_type))

    scheduler._tcp_client = ReconnectedClient()
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})

    assert calls == ["candidate_a"]
    assert [item[0]["message_type"] for item in replayed] == [
        "stage_accept", "stage_result",
    ]
    assert replayed[-1][0]["payload"]["output"] == {
        "content": "cached result"
    }


def test_scheduler_worker_does_not_resurrect_an_expired_lease(monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    accepted = threading.Event()
    completed = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] == "stage_accept":
                accepted.set()
            if data["message_type"] == "stage_error":
                completed.set()

    scheduler._tcp_client = Client()
    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    ))
    scheduler.configure_callbacks(_scheduler_callbacks(
        execute=lambda request, cancel_event: (
            threading.Event().wait(0.3) or {"content": "too late"}
        ),
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_expiredrenew01",
            "request_id": "request-expired-renew",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_expiredrenew01",
            "lease_id": "lease_expiredrenew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 100,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_expiredrenewoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert accepted.wait(2)
    threading.Event().wait(0.15)
    renew_now_ms = int(time.time() * 1000)
    renewal = build_message(
        "lease_renew",
        {
            "workflow_id": "wf_expiredrenew01",
            "stage_id": "candidate_a",
            "attempt_id": "att_expiredrenew01",
            "lease_id": "lease_expiredrenew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": renew_now_ms + 1000,
        },
        message_id="msg_expiredrenew01",
        sent_at_ms=renew_now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message(
        "master", {"data": renewal.snapshot()},
    )

    assert completed.wait(2)
    assert sent[-1][0]["message_type"] == "stage_error"
    assert sent[-1][0]["payload"]["error_code"] == "lease_expired"


def test_scheduler_worker_lease_uses_duration_across_wall_clock_skew(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    completed = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] in {"stage_result", "stage_error"}:
                completed.set()

    scheduler._tcp_client = Client()
    monkeypatch.setattr(scheduler, "_host", SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
    ))
    scheduler.configure_callbacks(_scheduler_callbacks(
        execute=lambda request, cancel_event: {"content": "clock independent"},
    ))
    now_ms = 1000
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_clockskew01",
            "request_id": "request-clock-skew",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_clockskew01",
            "lease_id": "lease_clockskew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 1000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_clockskewoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})

    assert completed.wait(2)
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result",
    ]
    assert sent[-1][0]["payload"]["output"] == {
        "content": "clock independent"
    }


def test_full_worker_stage_round_trips_through_independent_process(
    monkeypatch, tmp_path,
):
    import config as cfg
    import node_config as _node_config
    from task_graph import StageSpec, TaskGraphCoordinator

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    secret = "n2-independent-process-secret"
    # ★ 票9 SEC-BOUNDARY-01 起，集群长期凭据的唯一事实源是 LocalSecretStore：
    #   `config` 导入时执行 `node_config.apply_node_config_to_env()`，会用 store 里的
    #   (secret, epoch) 覆盖 `QLH_CLUSTER_SECRET`（防旧 .env 回滚已轮换凭据）。
    #   因此只给子进程传 env 是无效的（两侧不同源 ⇒ TCP 注册 v2 HMAC 签名不匹配）。
    #   这里把同一凭据写进一个临时 store，并让子进程指向它。
    secret_store = tmp_path / "node-secret-store"
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(secret_store))
    _node_config._store_cluster_credential(secret, 1)
    monkeypatch.setattr(cfg, "CLUSTER_SECRET", secret)
    monkeypatch.setattr(cfg, "CLUSTER_SECRET_EPOCH", 1)
    control = TaskWorkerControlPlane()
    identity = ModelIdentity(
        model_id="qwen-1_8b",
        engine="pytorch",
        format="safetensors",
        revision="process-test",
        sha256="b" * 64,
    )
    hello_received = threading.Event()
    provider_holder = {}
    server = TCPServer(host="127.0.0.1", port=port)

    def on_server_message(client_id, outer):
        if outer.get("type") == MessageType.REGISTER.value:
            server.confirm_registration(client_id)
            return
        if outer.get("type") != MessageType.TASK_WORKER.value:
            return
        inner = outer["data"]
        if inner["message_type"] == "hello":
            ack = control.receive_on_coordinator(
                client_id,
                inner,
                coordinator_node_id="master",
            )
            server.send_to_client(
                client_id, ack.snapshot(), MessageType.TASK_WORKER,
            )
            hello_received.set()
            return
        provider_holder["provider"].handle_message(inner)

    provider = RemoteFullWorkerProvider(
        node_id="worker_process_01",
        peer_snapshot=lambda: control.worker_snapshot("worker_process_01"),
        send_message=lambda message: server.send_to_client(
            "worker_process_01",
            message.snapshot(),
            MessageType.TASK_WORKER,
        ),
    )
    provider_holder["provider"] = provider
    helper = os.path.join(
        os.path.dirname(__file__), "helpers", "task_worker_process.py",
    )
    env = dict(os.environ)
    env.update({
        # ★ 凭据以 store 为唯一来源（见上）；这里显式指向同一临时 store，
        #   env 里的 QLH_CLUSTER_SECRET 反而会被 config 用 store 覆盖，故清除。
        "QLH_NODE_SECRET_STORE_PATH": str(secret_store),
        "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": "true",
        "PYTHONUTF8": "1",
    })
    env.pop("QLH_CLUSTER_SECRET", None)
    process = None
    coordinator = TaskGraphCoordinator()
    try:
        server.start(on_message=on_server_message)
        process = subprocess.Popen(
            [
                sys.executable,
                helper,
                "--port", str(port),
                "--node-id", "worker_process_01",
                "--model-id", identity.model_id,
                "--model-sha256", identity.sha256,
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # ★ Windows 中文 locale 下 `text=True` 默认按 GBK 解码子进程输出；
            #   子进程一旦打印非 GBK 字节（UTF-8 中文/符号），subprocess 的读取
            #   线程会抛 UnicodeDecodeError 静默死掉 ⇒ `communicate()` 返回
            #   `(None, None)`，握手诊断也一并失效（实测崩在 byte 0xa5）。
            #   与本仓既有修法一致（见 `scripts/android_validation.py`）。
            encoding="utf-8",
            errors="replace",
        )
        # ★ 子进程要 import 整套生产代码，冷启动在负载高时（全量并行跑）会明显
        #   超过 8 秒 —— 实测全量耗时从 891s 涨到 1261s 时这里就 8s 超时假红，
        #   而单跑 5/5 全绿。改为最长 30 秒轮询，并区分两种失败：
        #   子进程**提前退出**（给出 stderr）与单纯**握手过慢**。
        deadline = time.time() + 30.0
        while not hello_received.wait(0.5):
            if process.poll() is not None:
                out, err = process.communicate(timeout=5)
                # 读取线程可能已因解码失败死掉 ⇒ communicate() 会给 (None, None)
                raise AssertionError(
                    "worker process exited before hello handshake: "
                    f"rc={process.returncode}\n"
                    f"stdout={(out or '')[-2000:]}\nstderr={(err or '')[-2000:]}"
                )
            if time.time() >= deadline:
                raise AssertionError(
                    "worker process did not complete hello handshake within 30s"
                )
        coordinator.register_provider(provider)
        output, workflow = coordinator.run(
            stages=[StageSpec(
                "candidate_a",
                "full_inference",
                provider=provider.provider_id,
                fallback_providers=(),
                pure=False,
                lease_timeout_seconds=5.0,
            )],
            final_stage_id="candidate_a",
            root_input={"message": "independent process"},
            model_identity=identity,
            workflow_id="wf_processworker01",
        )

        assert output["content"] == "process:candidate_a"
        assert output["worker_pid"] != os.getpid()
        assert output["worker_node_id"] == "worker_process_01"
        assert workflow["stages"][0]["attempts"][0][
            "provider_node_id"
        ] == "worker_process_01"
        assert process.wait(timeout=10) == 0
        assert provider.inspect().active_reservations == 0
    finally:
        coordinator.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        server.stop()


def test_remote_worker_releases_slot_across_success_and_failure_cycles():
    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}

    def respond(message):
        if message.message_type != "stage_offer":
            return
        offer = message.payload
        identity = _response_identity(offer)
        provider = provider_holder["provider"]
        suffix = offer["attempt_id"].removeprefix("att_stress")
        provider.handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id=f"msg_stressaccept_{suffix}",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        if int(suffix) % 5 == 0:
            provider.handle_message(build_message(
                "stage_error",
                {
                    **identity,
                    "error_code": "remote_worker_disconnected",
                    "retryable": True,
                },
                message_id=f"msg_stresserror_{suffix}",
                sent_at_ms=int(time.time() * 1000),
                version=2,
            ).snapshot())
            return
        output = {"content": f"stress:{suffix}"}
        provider.handle_message(build_message(
            "stage_result",
            {
                **identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {},
            },
            message_id=f"msg_stressresult_{suffix}",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=respond,
    )
    provider_holder["provider"] = provider

    for index in range(1, 31):
        suffix = f"{index:08d}"
        request = StageRequest(
            workflow_id=f"wf_stress{suffix}",
            request_id=f"request-stress-{suffix}",
            stage_id="candidate_a",
            stage_type="full_inference",
            provider_id=provider.provider_id,
            dependencies={},
            root_input={"message": "stress"},
            model_identity=_model_identity(),
        )
        reservation = provider.reserve(request)
        attempt = StageAttempt(
            attempt_id=f"att_stress{suffix}",
            request=request,
            provider_id=provider.provider_id,
            lease_id=f"lease_stress{suffix}",
            lease_epoch=1,
            lease_expires_at=time.time() + 2,
        )
        try:
            if index % 5 == 0:
                with pytest.raises(ProviderExecutionError) as captured:
                    provider.execute(
                        attempt, reservation, threading.Event(),
                    )
                assert captured.value.code == "remote_worker_disconnected"
            else:
                result = provider.execute(
                    attempt, reservation, threading.Event(),
                )
                assert result.output == {"content": f"stress:{suffix}"}
        finally:
            provider.release(reservation.reservation_id)
        assert provider.inspect().active_reservations == 0


def _cancelled_ack(identity, *, reason_code="coordinator_cancelled",
                   execution_state=None, message_id="msg_cancelacked0001"):
    payload = {**identity, "reason_code": reason_code}
    if execution_state is not None:
        payload["execution_state"] = execution_state
    return build_message(
        "stage_cancelled",
        payload,
        message_id=message_id,
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot()


def test_remote_provider_distinguishes_cancel_ack_from_execution_stopped():
    """★ DIST-NEXT-1：ACK 只证明「请求已送达」，`execution_stopped` 才证明「执行已停止」。"""
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_cancelstate01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_cancelstate01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert _wait_until(lambda: bool(sent))
    identity = _response_identity(sent[0].payload)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_cancelstateaccept01",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())

    provider.cancel(attempt.attempt_id)
    provider.handle_message(_cancelled_ack(
        identity, execution_state="execution_in_flight",
    ))
    in_flight = provider.cancel_diagnostics()
    assert in_flight["cancel_ack_execution_states"] == {"execution_in_flight": 1}
    assert in_flight["cancel_execution_stopped_confirmed"] == 0
    assert in_flight["pending_cancel_ack_in_flight"] == 1

    # 执行真正停止后，对端补一条终态 ACK；本地状态必须升级而不是被当成重复丢弃。
    provider.handle_message(_cancelled_ack(
        identity, execution_state="execution_stopped",
        message_id="msg_cancelacked0002",
    ))
    stopped = provider.cancel_diagnostics()
    assert stopped["cancel_execution_stopped_confirmed"] == 1
    assert stopped["cancel_ack_execution_states"]["execution_stopped"] == 1
    assert stopped["pending_cancel_ack_in_flight"] == 0

    thread.join(2)
    assert not thread.is_alive()
    assert outcome["error"].code == "provider_cancelled"
    provider.release(reservation.reservation_id)
    provider.close()


def test_remote_provider_accepts_remote_initiated_cancellation():
    """★ DIST-NEXT-1：worker 主动取消（本地用户取消 / Service 回收）不再按协议违例拒绝。

    此前该消息一律抛 `unexpected_stage_cancelled`，master 只能等到租约/步骤超时。
    """
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_cancelremote02",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_cancelremote02",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert _wait_until(lambda: bool(sent))
    identity = _response_identity(sent[0].payload)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_cancelremoteaccept02",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())

    # 本端**没有**请求取消，worker 自己取消并发来 ACK。
    provider.handle_message(_cancelled_ack(
        identity, reason_code="user_cancelled",
        execution_state="execution_stopped",
        message_id="msg_cancelremoteack02",
    ))

    thread.join(2)
    assert not thread.is_alive()
    assert outcome["error"].code == "user_cancelled"
    assert outcome["error"].retryable is True
    diagnostics = provider.cancel_diagnostics()
    assert diagnostics["cancel_remote_initiated"] == 1
    assert diagnostics["cancel_execution_stopped_confirmed"] == 1
    provider.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0
    provider.close()


def test_remote_cancel_terminal_absorbs_late_result_and_error_tombstones():
    (
        provider, reservation, attempt, identity, outcome, finished, _thread, _sent,
    ) = _start_accepted_remote_attempt("remote_cancel_late01")

    provider.handle_message(_cancelled_ack(
        identity,
        reason_code="user_cancelled",
        execution_state="execution_stopped",
        message_id="msg_remote_cancel_first01",
    ))
    assert finished.wait(2.0)
    assert outcome["error"].code == "user_cancelled"

    # release() leaves a short-lived known tombstone so distinct late terminal
    # messages are absorbed instead of becoming duplicate/unknown violations.
    provider.release(reservation.reservation_id)
    output = {"content": "too late"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {},
        },
        message_id="msg_remote_cancel_late_result01",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())
    provider.handle_message(build_message(
        "stage_error",
        {
            **identity,
            "error_code": "remote_worker_disconnected",
            "retryable": True,
            "reason": "late error after remote cancellation",
        },
        message_id="msg_remote_cancel_late_error01",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())

    pending = provider._pending[attempt.attempt_id]
    terminal_kind, terminal_result, terminal_error = pending.terminal_snapshot()
    assert terminal_kind == "remote_cancel"
    assert terminal_result is None
    assert terminal_error is outcome["error"]
    assert provider.cancel_diagnostics()["late_stage_responses_absorbed"] == 2
    provider.close()


def test_remote_result_is_not_overwritten_by_disconnect():
    (
        provider, _reservation, attempt, identity, outcome, finished, _thread, _sent,
    ) = _start_accepted_remote_attempt("result_disconnect01")
    output = {"content": "result won"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {},
        },
        message_id="msg_result_before_disconnect01",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())
    provider.notify_disconnect()

    assert finished.wait(2.0)
    assert "error" not in outcome
    assert outcome["result"].output == output
    terminal_kind, terminal_result, terminal_error = (
        provider._pending[attempt.attempt_id].terminal_snapshot()
    )
    assert terminal_kind == "stage_result"
    assert terminal_result is outcome["result"]
    assert terminal_error is None

    provider.handle_message(build_message(
        "stage_error",
        {
            **identity,
            "error_code": "remote_worker_disconnected",
            "retryable": True,
            "reason": "late after disconnect cleanup",
        },
        message_id="msg_result_disconnect_late_error01",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())
    assert provider._pending[attempt.attempt_id].terminal_snapshot() == (
        terminal_kind, terminal_result, terminal_error,
    )
    provider.close()


def test_remote_stage_error_is_not_overwritten_by_renew_send_failure(monkeypatch):
    (
        provider, reservation, attempt, identity, outcome, finished, _thread, _sent,
    ) = _start_accepted_remote_attempt("error_renew_failure01")
    queued = {}

    def capture_renew(_message, on_error):
        queued["on_error"] = on_error
        return True

    monkeypatch.setattr(provider, "_queue_outbound_message", capture_renew)
    assert provider.renew_lease(
        attempt.attempt_id,
        attempt.lease_id,
        attempt.lease_epoch,
        attempt.lease_expires_at + 1.0,
    ) is True
    assert "on_error" in queued

    provider.handle_message(build_message(
        "stage_error",
        {
            **identity,
            "error_code": "remote_stage_failed",
            "retryable": False,
            "reason": "authoritative worker failure",
        },
        message_id="msg_stage_error_before_renew_failure01",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())
    queued["on_error"](ConnectionError("renew send failed later"))

    assert finished.wait(2.0)
    assert outcome["error"].code == "remote_stage_failed"
    pending = provider._pending[attempt.attempt_id]
    terminal_kind, terminal_result, terminal_error = pending.terminal_snapshot()
    assert terminal_kind == "stage_error"
    assert terminal_result is None
    assert terminal_error is outcome["error"]
    provider.release(reservation.reservation_id)
    provider.close()


def test_remote_cancel_result_disconnect_race_has_one_terminal():
    (
        provider, _reservation, attempt, identity, outcome, finished, _thread, _sent,
    ) = _start_accepted_remote_attempt("terminal_race01")
    output = {"content": "racing result"}
    result_message = build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {},
        },
        message_id="msg_terminal_race_result01",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot()
    barrier = threading.Barrier(4)
    race_errors = []

    def race(action):
        try:
            barrier.wait()
            action()
        except BaseException as exc:
            race_errors.append(exc)

    racers = [
        threading.Thread(
            target=race, args=(lambda: provider.cancel(attempt.attempt_id),),
            daemon=True,
        ),
        threading.Thread(
            target=race,
            args=(lambda: provider.handle_message(result_message),),
            daemon=True,
        ),
        threading.Thread(
            target=race, args=(provider.notify_disconnect,), daemon=True,
        ),
    ]
    for racer in racers:
        racer.start()
    barrier.wait()
    for racer in racers:
        racer.join(2.0)

    assert all(not racer.is_alive() for racer in racers)
    assert race_errors == []
    assert finished.wait(2.0)
    pending = provider._pending[attempt.attempt_id]
    first_snapshot = pending.terminal_snapshot()
    terminal_kind, terminal_result, terminal_error = first_snapshot
    assert terminal_kind in {"provider_cancelled", "stage_result", "disconnect"}
    assert (terminal_result is None) != (terminal_error is None)
    if terminal_kind == "stage_result":
        assert outcome["result"].output == output
        assert "error" not in outcome
    else:
        assert outcome["error"] is terminal_error
        assert outcome["error"].code in {
            "provider_cancelled", "remote_worker_disconnected",
        }

    provider.cancel(attempt.attempt_id)
    provider.notify_disconnect()
    provider.handle_message(build_message(
        "stage_error",
        {
            **identity,
            "error_code": "remote_stage_failed",
            "retryable": False,
            "reason": "late terminal after three-way race",
        },
        message_id="msg_terminal_race_late_error01",
        sent_at_ms=int(time.time() * 1000),
        version=3,
    ).snapshot())
    assert pending.terminal_snapshot() == first_snapshot
    assert provider.cancel_diagnostics()["late_stage_responses_absorbed"] >= 1
    provider.close()
