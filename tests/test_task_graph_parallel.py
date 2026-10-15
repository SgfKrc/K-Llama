import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import (
    StageSpec,
    TaskGraphCoordinator,
    WorkflowCancelled,
    WorkflowExecutionError,
)
from task_provider import (
    DeterministicFakeProvider,
    InProcessWorkerProvider,
    ProviderRegistry,
)
from task_journal import SQLiteTaskJournal


def _parallel_graph(candidate_providers=("fake_a", "fake_b")):
    return [
        StageSpec("candidate_a", "full_inference", provider=candidate_providers[0]),
        StageSpec("candidate_b", "full_inference", provider=candidate_providers[1]),
        StageSpec(
            "aggregate",
            "aggregate",
            depends_on=("candidate_a", "candidate_b"),
            provider="aggregate_worker",
        ),
    ]


def _aggregate_output(request, cancel_event):
    return {
        "content": "+".join(
            request.dependencies[stage_id]["content"]
            for stage_id in sorted(request.dependencies)
        ),
    }


def test_independent_ready_stages_overlap_and_fan_in_once():
    start_barrier = threading.Barrier(2)
    registry = ProviderRegistry()
    candidate_a = DeterministicFakeProvider(
        "fake_a",
        delay_seconds=0.25,
        start_barrier=start_barrier,
        node_id="fake-node-a",
    )
    candidate_b = DeterministicFakeProvider(
        "fake_b",
        delay_seconds=0.25,
        start_barrier=start_barrier,
        node_id="fake-node-b",
    )
    aggregate_calls = []

    def aggregate(request, cancel_event):
        aggregate_calls.append(request.stage_id)
        return _aggregate_output(request, cancel_event)

    registry.register(candidate_a)
    registry.register(candidate_b)
    registry.register(InProcessWorkerProvider(
        "aggregate_worker",
        aggregate,
        node_id="aggregate-node",
    ))
    coordinator = TaskGraphCoordinator(
        provider_registry=registry,
        max_parallel_stages=2,
    )

    started_at = time.perf_counter()
    output, workflow = coordinator.run(
        _parallel_graph(),
        "aggregate",
        {"message": "question"},
        workflow_id="wf_parallel01",
    )
    elapsed = time.perf_counter() - started_at

    assert output["content"] == "candidate_a+candidate_b"
    assert elapsed < 0.45
    record_a = candidate_a.call_records()[0]
    record_b = candidate_b.call_records()[0]
    assert max(record_a["started_at"], record_b["started_at"]) < min(
        record_a["finished_at"], record_b["finished_at"],
    )
    assert aggregate_calls == ["aggregate"]
    assert workflow["completed_stage_count"] == 3
    assert {
        stage["provider"] for stage in workflow["stages"]
    } == {"fake_a", "fake_b", "aggregate_worker"}
    coordinator.commit_result("wf_parallel01")
    coordinator.close()


def test_same_single_slot_provider_remains_strictly_serial():
    registry = ProviderRegistry()
    state_lock = threading.Lock()
    active = 0
    max_active = 0
    order = []

    def execute(request, cancel_event):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            order.append(request.stage_id)
        try:
            assert not cancel_event.wait(0.04)
            if request.stage_type == "aggregate":
                return _aggregate_output(request, cancel_event)
            return {"content": request.stage_id}
        finally:
            with state_lock:
                active -= 1

    registry.register(InProcessWorkerProvider(
        "shared_local",
        execute,
        max_concurrency=1,
    ))
    coordinator = TaskGraphCoordinator(
        provider_registry=registry,
        max_parallel_stages=4,
    )
    stages = [
        StageSpec("candidate_a", "full_inference", provider="shared_local"),
        StageSpec("candidate_b", "full_inference", provider="shared_local"),
        StageSpec(
            "aggregate",
            "aggregate",
            depends_on=("candidate_a", "candidate_b"),
            provider="shared_local",
        ),
    ]

    output, _workflow = coordinator.run(
        stages,
        "aggregate",
        {"message": "question"},
        workflow_id="wf_serialprov",
    )

    assert output["content"] == "candidate_a+candidate_b"
    assert max_active == 1
    assert order == ["candidate_a", "candidate_b", "aggregate"]
    coordinator.commit_result("wf_serialprov")
    coordinator.close()


def test_global_parallel_limit_batches_three_independent_stages():
    registry = ProviderRegistry()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def execute(request, cancel_event):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            assert not cancel_event.wait(0.06)
            if request.stage_type == "aggregate":
                return _aggregate_output(request, cancel_event)
            return {"content": request.stage_id}
        finally:
            with state_lock:
                active -= 1

    for provider_id in ("worker_a", "worker_b", "worker_c", "aggregate_worker"):
        registry.register(InProcessWorkerProvider(provider_id, execute))
    coordinator = TaskGraphCoordinator(
        provider_registry=registry,
        max_parallel_stages=2,
    )
    stages = [
        StageSpec("a", "full_inference", provider="worker_a"),
        StageSpec("b", "full_inference", provider="worker_b"),
        StageSpec("c", "full_inference", provider="worker_c"),
        StageSpec(
            "aggregate",
            "aggregate",
            depends_on=("a", "b", "c"),
            provider="aggregate_worker",
        ),
    ]

    output, workflow = coordinator.run(
        stages,
        "aggregate",
        {"message": "question"},
        workflow_id="wf_boundedpar",
    )

    assert output["content"] == "a+b+c"
    assert max_active == 2
    assert workflow["completed_stage_count"] == 4
    coordinator.commit_result("wf_boundedpar")
    coordinator.close()


def test_parallel_failure_waits_for_other_future_and_releases_all_slots():
    registry = ProviderRegistry()
    failed = DeterministicFakeProvider(
        "fake_a",
        delay_seconds=0.02,
        fail_stage_ids=("candidate_a",),
    )
    completed = DeterministicFakeProvider(
        "fake_b",
        delay_seconds=0.08,
    )
    aggregate = InProcessWorkerProvider(
        "aggregate_worker",
        _aggregate_output,
    )
    registry.register(failed)
    registry.register(completed)
    registry.register(aggregate)
    coordinator = TaskGraphCoordinator(
        provider_registry=registry,
        max_parallel_stages=2,
    )

    with pytest.raises(WorkflowExecutionError) as captured:
        coordinator.run(
            _parallel_graph(),
            "aggregate",
            {"message": "question"},
            workflow_id="wf_parallelfail",
        )

    assert captured.value.stage_id == "candidate_a"
    snapshot = coordinator.get("wf_parallelfail")
    stages = {stage["stage_id"]: stage for stage in snapshot["stages"]}
    assert snapshot["state"] == "failed"
    assert stages["candidate_a"]["state"] == "failed"
    assert stages["candidate_b"]["state"] == "completed"
    assert stages["aggregate"]["state"] == "skipped"
    assert all(
        attempt["state"] != "running"
        for stage in snapshot["stages"]
        for attempt in stage["attempts"]
    )
    assert all(
        item["active_reservations"] == 0
        for item in coordinator.provider_status()
    )
    coordinator.close()


def test_parallel_cancellation_reaches_all_providers_and_releases_slots():
    class CancellationGate(threading.Event):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.allow_observation = threading.Event()

        def wait(self, timeout=None):
            self.entered.set()
            assert self.allow_observation.wait(20), "provider gate not released"
            return super().wait(timeout)

    class CompletionTrackedFakeProvider(DeterministicFakeProvider):
        def __init__(self, provider_id, **kwargs):
            self.execution_finished = threading.Event()
            super().__init__(provider_id, **kwargs)

        def execute(self, attempt, reservation, cancel_event):
            try:
                return super().execute(attempt, reservation, cancel_event)
            finally:
                # super().execute has finalized the fake record and active attempt.
                self.execution_finished.set()

    gates = [CancellationGate(), CancellationGate()]
    registry = ProviderRegistry()
    candidate_a = CompletionTrackedFakeProvider("fake_a", block_event=gates[0])
    candidate_b = CompletionTrackedFakeProvider("fake_b", block_event=gates[1])
    providers = (candidate_a, candidate_b)
    registry.register(candidate_a)
    registry.register(candidate_b)
    registry.register(InProcessWorkerProvider(
        "aggregate_worker",
        _aggregate_output,
    ))
    coordinator = TaskGraphCoordinator(
        provider_registry=registry,
        max_parallel_stages=2,
    )
    errors = []

    def run_workflow():
        try:
            coordinator.run(
                _parallel_graph(),
                "aggregate",
                {"message": "question"},
                workflow_id="wf_parallelcancel",
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    thread.start()
    try:
        for gate in gates:
            assert gate.entered.wait(20), "provider did not reach execution gate"
        requested = coordinator.cancel("wf_parallelcancel")
        assert requested["cancel_requested"] is True
        thread.join(20)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], WorkflowCancelled)
        snapshot = coordinator.get("wf_parallelcancel")
        assert snapshot["state"] == "cancelled"
        assert {
            stage["state"] for stage in snapshot["stages"][:2]
        } == {"cancelled"}
        assert all(
            item["active_reservations"] == 0
            for item in coordinator.provider_status()
        )
        # A terminal workflow acknowledges cancellation without waiting for the
        # provider execution thread. Force that ordering before allowing either
        # fake to observe cancellation and finalize its call record.
        for provider in providers:
            assert not provider.execution_finished.is_set()
            assert provider.call_records()[0]["state"] == "running"
        for gate in gates:
            gate.allow_observation.set()
        for provider in providers:
            assert provider.execution_finished.wait(20), "provider did not finish"
            record = provider.call_records()[0]
            assert record["state"] == "cancelled"
            assert record["finished_at"] is not None
    finally:
        for gate in gates:
            gate.allow_observation.set()
        coordinator.cancel("wf_parallelcancel")
        thread.join(20)
        coordinator.close()
        for provider, gate in zip(providers, gates):
            if gate.entered.is_set():
                assert provider.execution_finished.wait(20)


def test_parallel_journal_sequence_is_contiguous(tmp_path):
    path = str(tmp_path / "parallel.sqlite3")
    start_barrier = threading.Barrier(2)
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider(
        "fake_a",
        delay_seconds=0.03,
        start_barrier=start_barrier,
    ))
    registry.register(DeterministicFakeProvider(
        "fake_b",
        delay_seconds=0.03,
        start_barrier=start_barrier,
    ))
    registry.register(InProcessWorkerProvider(
        "aggregate_worker",
        _aggregate_output,
    ))
    journal = SQLiteTaskJournal(path)
    coordinator = TaskGraphCoordinator(
        journal=journal,
        provider_registry=registry,
        max_parallel_stages=2,
    )

    coordinator.run(
        _parallel_graph(),
        "aggregate",
        {"message": "question"},
        workflow_id="wf_parjournal",
    )
    committed = coordinator.commit_result("wf_parjournal")
    events = journal.list_events("wf_parjournal")

    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1),
    )
    assert committed["last_sequence"] == events[-1]["sequence"]
    assert committed["state"] == "completed"
    coordinator.close()

    reopened = SQLiteTaskJournal(path)
    persisted = reopened.get_snapshot("wf_parjournal")
    assert persisted["state"] == "completed"
    assert {
        attempt["provider"]
        for stage in persisted["stages"]
        for attempt in stage["attempts"]
    } == {"fake_a", "fake_b", "aggregate_worker"}
    reopened.close()


def test_close_waits_for_parallel_cancellation_to_reach_journal(tmp_path):
    path = str(tmp_path / "close-parallel.sqlite3")
    start_barrier = threading.Barrier(3)
    never_release = threading.Event()
    registry = ProviderRegistry()
    for provider_id in ("fake_a", "fake_b"):
        registry.register(DeterministicFakeProvider(
            provider_id,
            start_barrier=start_barrier,
            block_event=never_release,
        ))
    registry.register(InProcessWorkerProvider(
        "aggregate_worker",
        _aggregate_output,
    ))
    coordinator = TaskGraphCoordinator(
        journal=SQLiteTaskJournal(path),
        provider_registry=registry,
        max_parallel_stages=2,
    )
    errors = []

    def run_workflow():
        try:
            coordinator.run(
                _parallel_graph(),
                "aggregate",
                {"message": "question"},
                workflow_id="wf_closepar1",
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    thread.start()
    start_barrier.wait(timeout=5)

    coordinator.close()
    thread.join(5)

    assert not thread.is_alive()
    assert isinstance(errors[0], WorkflowCancelled)
    reopened = SQLiteTaskJournal(path)
    persisted = reopened.get_snapshot("wf_closepar1")
    assert persisted["state"] == "cancelled"
    assert all(
        attempt["state"] != "running"
        for stage in persisted["stages"]
        for attempt in stage["attempts"]
    )
    reopened.close()
