import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import (
    StageSpec,
    TaskGraphCoordinator,
    TaskGraphError,
    TaskGraphUnavailable,
    WorkflowDeadlineExceeded,
    WorkflowExecutionError,
)
from request_deadline import RequestDeadline
from task_journal import SQLiteTaskJournal, TaskJournalError
from task_provider import (
    DeterministicFakeProvider,
    LocalFullModelProvider,
    ModelIdentity,
    ProviderRegistry,
)


def test_hung_provider_is_reassigned_when_lease_expires():
    never_release = threading.Event()
    fallback_started = threading.Event()
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider(
        "primary",
        block_event=never_release,
    ))
    registry.register(DeterministicFakeProvider(
        "fallback",
        output_factory=lambda request, cancel_event: (
            fallback_started.set() or {"content": "fallback"}
        ),
    ))
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    errors = []

    def run_workflow():
        try:
            coordinator.run(
                [StageSpec(
                    "answer",
                    "full_inference",
                    provider="primary",
                    fallback_providers=("fallback",),
                    pure=True,
                    lease_timeout_seconds=0.03,
                )],
                "answer",
                {},
                workflow_id="wf_reviewhung1",
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    thread.start()
    try:
        assert fallback_started.wait(0.25)
        never_release.set()
        thread.join(2)
        if thread.is_alive():
            coordinator.cancel("wf_reviewhung1")
            thread.join(2)
    finally:
        never_release.set()
        coordinator.close()
    assert not thread.is_alive()
    assert not errors


class _SlowMemoryJournal:
    def __init__(self, delay_seconds=0.03):
        self.delay_seconds = delay_seconds
        self.snapshots = {}

    def append_event(self, event, snapshot):
        time.sleep(self.delay_seconds)
        self.snapshots[event.workflow_id] = snapshot
        return True

    def get_snapshot(self, workflow_id):
        return self.snapshots.get(workflow_id)

    def list_snapshots(self, limit=20):
        return list(self.snapshots.values())[:limit]

    def list_events(self, workflow_id):
        return []

    def list_nonterminal_snapshots(self, limit=100):
        return []

    def cleanup_terminal(self, **kwargs):
        return {}

    def health(self):
        return {
            "enabled": True,
            "available": True,
            "backend": "slow_memory",
            "error": "",
        }

    def close(self):
        return None


class _FailingCancelJournal(_SlowMemoryJournal):
    def append_event(self, event, snapshot):
        if event.event_type == "workflow_cancel_requested":
            raise TaskJournalError("cancel journal write failed")
        return super().append_event(event, snapshot)


class _CancelTrackingProvider(LocalFullModelProvider):
    def __init__(self, executor):
        super().__init__(executor, provider_id="cancel_tracking")
        self.cancelled_attempts = []

    def cancel(self, attempt_id):
        self.cancelled_attempts.append(attempt_id)
        super().cancel(attempt_id)


def test_cancel_side_effect_waits_for_durable_journal_event():
    started = threading.Event()
    release = threading.Event()

    def execute(request, cancel_event):
        started.set()
        assert release.wait(2)
        return {"content": request.stage_id}

    provider = _CancelTrackingProvider(execute)
    registry = ProviderRegistry()
    registry.register(provider)
    coordinator = TaskGraphCoordinator(
        journal=_FailingCancelJournal(delay_seconds=0),
        provider_registry=registry,
    )
    errors = []

    def run_workflow():
        try:
            coordinator.run(
                [StageSpec(
                    "answer",
                    "full_inference",
                    provider=provider.provider_id,
                )],
                "answer",
                {},
                workflow_id="wf_canceljournal01",
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow, daemon=True)
    thread.start()
    assert started.wait(2)
    try:
        with pytest.raises(TaskGraphUnavailable):
            coordinator.cancel("wf_canceljournal01")
        assert provider.cancelled_attempts == []
        assert coordinator.get("wf_canceljournal01")["cancel_requested"] is False
    finally:
        release.set()
        thread.join(2)
        coordinator.close()
    assert not thread.is_alive()
    assert errors and isinstance(errors[0], TaskGraphUnavailable)


def test_journal_latency_does_not_consume_execution_lease():
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider("instant"))
    coordinator = TaskGraphCoordinator(
        journal=_SlowMemoryJournal(),
        provider_registry=registry,
    )
    try:
        output, workflow = coordinator.run(
            [StageSpec(
                "answer",
                "full_inference",
                provider="instant",
                lease_timeout_seconds=0.04,
            )],
            "answer",
            {},
            workflow_id="wf_reviewslow1",
        )
        assert output == {"content": "answer"}
        assert workflow["state"] == "result_ready"
    finally:
        coordinator.close()


class _FailingReleaseProvider(LocalFullModelProvider):
    def release(self, reservation_id):
        super().release(reservation_id)
        raise RuntimeError("release failed")


class _AbortReserveProvider(LocalFullModelProvider):
    def reserve(self, request):
        raise TaskGraphUnavailable("simulated coordinator stop")


def test_batch_cleanup_releases_every_prepared_reservation_after_error():
    registry = ProviderRegistry()
    first = _FailingReleaseProvider(
        lambda request, cancel_event: {"content": "a"},
        provider_id="provider_a",
    )
    second = LocalFullModelProvider(
        lambda request, cancel_event: {"content": "b"},
        provider_id="provider_b",
    )
    aborting = _AbortReserveProvider(
        lambda request, cancel_event: {"content": "c"},
        provider_id="provider_c",
    )
    for provider in (first, second, aborting):
        registry.register(provider)
    coordinator = TaskGraphCoordinator(
        provider_registry=registry,
        max_parallel_stages=3,
    )
    try:
        with pytest.raises(TaskGraphError):
            coordinator.run(
                [
                    StageSpec("a", "full_inference", provider="provider_a"),
                    StageSpec("b", "full_inference", provider="provider_b"),
                    StageSpec("c", "full_inference", provider="provider_c"),
                ],
                "a",
                {},
                workflow_id="wf_reviewrelease",
            )
        assert second.inspect().active_reservations == 0
    finally:
        coordinator.close()


def test_graceful_close_terminalizes_result_ready_snapshot(tmp_path):
    path = str(tmp_path / "close-result-ready.sqlite3")
    coordinator = TaskGraphCoordinator(journal=SQLiteTaskJournal(path))

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {},
        execute,
        workflow_id="wf_reviewclose1",
    )
    coordinator.close()

    reopened = SQLiteTaskJournal(path)
    try:
        persisted = reopened.get_snapshot("wf_reviewclose1")
        assert persisted["state"] in {"failed", "cancelled"}
    finally:
        reopened.close()


def test_lease_renewal_extends_running_attempt_without_fallback():
    release_primary = threading.Event()
    registry = ProviderRegistry()
    primary = DeterministicFakeProvider(
        "primary",
        block_event=release_primary,
        output_factory=lambda request, cancel_event: {"content": "primary"},
    )
    fallback = DeterministicFakeProvider("fallback")
    registry.register(primary)
    registry.register(fallback)
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    outcome = {}

    def run_workflow():
        outcome["value"] = coordinator.run(
            [StageSpec(
                "answer",
                "full_inference",
                provider="primary",
                fallback_providers=("fallback",),
                pure=True,
                lease_timeout_seconds=0.08,
            )],
            "answer",
            {},
            workflow_id="wf_reviewrenew1",
        )

    thread = threading.Thread(target=run_workflow)
    thread.start()
    deadline = time.time() + 1.0
    attempt = None
    while time.time() < deadline:
        try:
            snapshot = coordinator.get("wf_reviewrenew1")
        except TaskGraphError:
            time.sleep(0.005)
            continue
        attempts = snapshot["stages"][0]["attempts"]
        if attempts:
            attempt = attempts[0]
            break
        time.sleep(0.005)
    assert attempt is not None
    coordinator.renew_stage_lease(
        "wf_reviewrenew1",
        "answer",
        attempt["attempt_id"],
        attempt["lease_id"],
        attempt["lease_epoch"],
        time.time() + 0.3,
    )
    time.sleep(0.1)
    release_primary.set()
    thread.join(2)

    assert not thread.is_alive()
    output, workflow = outcome["value"]
    assert output == {"content": "primary"}
    assert workflow["stages"][0]["retry_count"] == 0
    assert fallback.call_records() == []
    coordinator.close()


def test_request_deadline_stops_local_provider_and_latches_terminal_reason():
    stopped = threading.Event()
    registry = ProviderRegistry()

    def execute(_request, cancel_event):
        assert cancel_event.wait(1.0)
        stopped.set()
        raise RuntimeError("provider observed cooperative stop")

    registry.register(LocalFullModelProvider(execute))
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    with pytest.raises(WorkflowExecutionError) as captured:
        coordinator.run(
            [StageSpec("answer", "full_inference")],
            "answer",
            {},
            workflow_id="wf_requestdeadline01",
            request_deadline=RequestDeadline.start(0.08),
        )

    assert captured.value.code == "request_deadline_exceeded"
    assert stopped.wait(0.5)
    snapshot = coordinator.get("wf_requestdeadline01")
    assert snapshot["state"] == "failed"
    assert snapshot["stages"][0]["attempts"][0]["state"] == "expired"
    assert snapshot["terminal_reason"]["reason_code"] == (
        "request_deadline_exceeded"
    )
    coordinator.close()


def test_worker_lease_remains_independent_from_request_deadline():
    release = threading.Event()
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider(
        "remote-shaped",
        block_event=release,
        output_factory=lambda _request, _cancel: {"content": "done"},
    ))
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    request_deadline = RequestDeadline.start(1.0)
    outcome = {}

    def run_workflow():
        outcome["value"] = coordinator.run(
            [StageSpec(
                "answer",
                "full_inference",
                provider="remote-shaped",
                lease_timeout_seconds=0.2,
            )],
            "answer",
            {},
            workflow_id="wf_renewdeadline01",
            request_deadline=request_deadline,
        )

    thread = threading.Thread(target=run_workflow)
    thread.start()
    attempt = None
    poll_deadline = time.monotonic() + 1.0
    while time.monotonic() < poll_deadline:
        try:
            attempts = coordinator.get(
                "wf_renewdeadline01",
            )["stages"][0]["attempts"]
        except TaskGraphError:
            attempts = []
        if attempts:
            attempt = attempts[0]
            break
        time.sleep(0.005)
    assert attempt is not None

    renewed = coordinator.renew_stage_lease(
        "wf_renewdeadline01",
        "answer",
        attempt["attempt_id"],
        attempt["lease_id"],
        attempt["lease_epoch"],
        request_deadline.expires_at_epoch + 1.0,
    )
    assert renewed["lease_expires_at"] > request_deadline.expires_at_epoch

    release.set()
    thread.join(2)
    assert not thread.is_alive()
    assert outcome["value"][0] == {"content": "done"}
    coordinator.close()


def test_result_ready_cannot_commit_after_request_deadline():
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider("instant"))
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    deadline = RequestDeadline.start(0.2)

    output, workflow = coordinator.run(
        [StageSpec(
            "answer", "full_inference", provider="instant",
        )],
        "answer",
        {},
        workflow_id="wf_commitdeadline01",
        request_deadline=deadline,
    )
    assert output == {"content": "answer"}
    assert workflow["state"] == "result_ready"

    time.sleep(deadline.remaining() + 0.02)
    with pytest.raises(WorkflowDeadlineExceeded):
        coordinator.commit_result("wf_commitdeadline01")

    snapshot = coordinator.get("wf_commitdeadline01")
    assert snapshot["state"] == "failed"
    assert snapshot["terminal_reason"]["reason_code"] == (
        "request_deadline_exceeded"
    )
    coordinator.cancel("wf_commitdeadline01")
    snapshot = coordinator.get("wf_commitdeadline01")
    assert snapshot["terminal_reason"]["reason_code"] == (
        "request_deadline_exceeded"
    )
    assert any(
        item["reason_code"] == "generation_cancelled"
        for item in snapshot["terminal_reason"]["secondary_observations"]
    )
    coordinator.close()


@pytest.mark.parametrize("cancel_method", [
    "cancel", "request_cancel", "discard_result",
])
def test_result_ready_cancel_paths_latch_generation_cancelled(cancel_method):
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider("instant"))
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    workflow_id = f"wf_readycancel_{cancel_method}"
    coordinator.run(
        [StageSpec(
            "answer", "full_inference", provider="instant",
        )],
        "answer",
        {},
        workflow_id=workflow_id,
    )

    getattr(coordinator, cancel_method)(workflow_id)
    snapshot = coordinator.get(workflow_id)
    assert snapshot["state"] == "cancelled"
    assert snapshot["terminal_reason"]["reason_code"] == (
        "generation_cancelled"
    )
    coordinator.close()


def test_workflow_snapshot_and_stage_request_keep_model_and_runtime_separate():
    captured = {}
    registry = ProviderRegistry()

    def execute(request, cancel_event):
        captured["model"] = request.model_identity
        captured["runtime"] = request.runtime_context
        captured["root_input"] = request.root_input
        return {"content": "done"}

    registry.register(LocalFullModelProvider(execute))
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    identity = ModelIdentity(
        model_id="qwen-1_8b",
        engine="pytorch",
        format="safetensors",
        revision="local-rev1",
        sha256="0" * 64,
    )
    output, workflow = coordinator.run(
        [StageSpec("answer", "full_inference")],
        "answer",
        {"message": "question"},
        session_id="session-a",
        model_identity=identity,
        runtime_context={"local_only": object()},
        workflow_id="wf_reviewmodel1",
    )

    assert output == {"content": "done"}
    assert captured["model"] == identity
    assert captured["runtime"]["local_only"] is not None
    assert captured["root_input"] == {"message": "question"}
    assert workflow["model_identity"] == identity.snapshot()
    assert workflow["stages"][0]["model_identity"] == identity.snapshot()
    assert workflow["session_id"] == "session-a"
    coordinator.close()


def test_stage_model_identity_routes_heterogeneous_fake_workers():
    worker_a_identity = ModelIdentity(
        model_id="qwen-1_8b",
        engine="pytorch",
        format="safetensors",
        revision="local-rev1",
        sha256="1" * 64,
    )
    worker_b_identity = ModelIdentity(
        model_id="qwen-3b",
        engine="pytorch",
        format="safetensors",
        revision="local-rev2",
        sha256="2" * 64,
    )
    observed = {}
    registry = ProviderRegistry()

    def candidate_output(worker_id, expected_identity):
        def execute(request, cancel_event):
            del cancel_event
            assert request.model_identity == expected_identity
            observed[worker_id] = request.model_identity
            return {"content": worker_id}

        return execute

    registry.register(DeterministicFakeProvider(
        "fake_worker_a",
        output_factory=candidate_output("fake_worker_a", worker_a_identity),
    ))
    registry.register(DeterministicFakeProvider(
        "fake_worker_b",
        output_factory=candidate_output("fake_worker_b", worker_b_identity),
    ))
    registry.register(DeterministicFakeProvider(
        "local_aggregate",
        supported_stage_types=("aggregate",),
        output_factory=lambda request, cancel_event: {
            "content": "+".join(sorted(
                item["content"] for item in request.dependencies.values()
            )),
        },
    ))
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    try:
        output, workflow = coordinator.run(
            [
                StageSpec(
                    "candidate_a",
                    "full_inference",
                    provider="fake_worker_a",
                    model_identity=worker_a_identity,
                ),
                StageSpec(
                    "candidate_b",
                    "full_inference",
                    provider="fake_worker_b",
                    model_identity=worker_b_identity,
                ),
                StageSpec(
                    "aggregate",
                    "aggregate",
                    depends_on=("candidate_a", "candidate_b"),
                    provider="local_aggregate",
                ),
            ],
            "aggregate",
            {"message": "question"},
            model_identity=worker_a_identity,
            workflow_id="wf_stageidentity1",
        )
    finally:
        coordinator.close()

    stages = {stage["stage_id"]: stage for stage in workflow["stages"]}
    assert output == {"content": "fake_worker_a+fake_worker_b"}
    assert observed == {
        "fake_worker_a": worker_a_identity,
        "fake_worker_b": worker_b_identity,
    }
    assert stages["candidate_a"]["model_identity"] == worker_a_identity.snapshot()
    assert stages["candidate_b"]["model_identity"] == worker_b_identity.snapshot()
    assert stages["aggregate"]["model_identity"] == worker_a_identity.snapshot()
