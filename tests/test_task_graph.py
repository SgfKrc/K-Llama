import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import (
    DEPENDENCY_FAILURES_KEY,
    StageSpec,
    TaskGraphCoordinator,
    TaskGraphError,
    WorkflowCancelled,
    WorkflowExecutionError,
)
from task_provider import ProviderExecutionError


def test_dual_candidate_template_executes_dependencies_in_order():
    coordinator = TaskGraphCoordinator()
    order = []

    def execute(stage, dependencies, root_input, cancel_event):
        assert not cancel_event.is_set()
        order.append(stage.stage_id)
        if stage.stage_type == "aggregate":
            assert set(dependencies) == {"candidate_a", "candidate_b"}
            return {
                "content": dependencies["candidate_a"]["content"]
                + dependencies["candidate_b"]["content"]
            }
        assert dependencies == {}
        return {"content": stage.stage_id + root_input["message"]}

    output, workflow = coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        request_id="req-1",
        workflow_id="wf_12345678",
    )

    assert order == ["candidate_a", "candidate_b", "aggregate"]
    assert output["content"] == "candidate_aquestioncandidate_bquestion"
    assert workflow["state"] == "result_ready"
    assert workflow["result_ready_at"] is not None
    assert workflow["finished_at"] is None
    assert workflow["stage_count"] == 3
    assert workflow["completed_stage_count"] == 3
    assert workflow["attempt_count"] == 3
    assert all(stage["output_available"] for stage in workflow["stages"])
    assert all("content" not in stage for stage in workflow["stages"])

    committed = coordinator.commit_result("wf_12345678")
    assert committed["state"] == "completed"
    assert committed["finished_at"] is not None


@pytest.mark.parametrize(
    "stages, final_stage, message",
    [
        ([StageSpec("a", "x"), StageSpec("a", "y")], "a", "unique"),
        ([StageSpec("a", "x", ("missing",))], "a", "missing"),
        ([StageSpec("a", "x", ("a",))], "a", "itself"),
        (
            [StageSpec("a", "x", ("b",)), StageSpec("b", "x", ("a",))],
            "a",
            "acyclic",
        ),
        (
            [StageSpec("a", "x", minimum_successful_dependencies=1)],
            "a",
            "minimum successful dependencies",
        ),
        (
            [StageSpec(
                "a", "x", max_same_provider_retries=4,
            )],
            "a",
            "same-Provider retries",
        ),
        (
            [StageSpec(
                "a", "x", max_same_provider_retries=1,
            )],
            "a",
            "retry_safe=true",
        ),
    ],
)
def test_validate_rejects_invalid_graphs(stages, final_stage, message):
    with pytest.raises(TaskGraphError, match=message):
        TaskGraphCoordinator.validate(stages, final_stage)


def test_strict_dependency_failure_marks_downstream_skipped_and_workflow_failed():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        if stage.stage_id == "candidate_b":
            raise RuntimeError("candidate failed")
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowExecutionError) as exc_info:
        coordinator.run(
            [
                StageSpec("candidate_a", "full_inference"),
                StageSpec("candidate_b", "full_inference"),
                StageSpec(
                    "aggregate",
                    "aggregate",
                    depends_on=("candidate_a", "candidate_b"),
                ),
            ],
            "aggregate",
            {"message": "question"},
            execute_stage=execute,
            workflow_id="wf_failure1",
        )

    workflow = coordinator.get(exc_info.value.workflow_id)
    by_id = {stage["stage_id"]: stage for stage in workflow["stages"]}
    assert workflow["state"] == "failed"
    assert by_id["candidate_a"]["state"] == "completed"
    assert by_id["candidate_b"]["state"] == "failed"
    assert by_id["aggregate"]["state"] == "skipped"
    assert by_id["candidate_b"]["attempts"][0]["state"] == "failed"


def test_dual_candidate_partial_failure_runs_aggregate_with_safe_summary():
    coordinator = TaskGraphCoordinator()
    aggregate_inputs = []

    def execute(stage, dependencies, root_input, cancel_event):
        if stage.stage_id == "candidate_b":
            raise RuntimeError("sensitive candidate failure detail")
        if stage.stage_type == "aggregate":
            aggregate_inputs.append(dependencies)
            return {"content": dependencies["candidate_a"]["content"]}
        return {"content": "surviving candidate"}

    output, workflow = coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_partialok1",
    )

    assert output == {"content": "surviving candidate"}
    assert workflow["state"] == "result_ready"
    assert workflow["partial_result"] is True
    assert workflow["completed_stage_count"] == 2
    assert workflow["failed_stage_count"] == 1
    assert aggregate_inputs == [{
        "candidate_a": {"content": "surviving candidate"},
        DEPENDENCY_FAILURES_KEY: {
            "candidate_b": {
                "state": "failed",
                "error_code": "provider_execution_failed",
            },
        },
    }]
    assert "sensitive candidate failure detail" not in str(aggregate_inputs)
    committed = coordinator.commit_result("wf_partialok1")
    assert committed["state"] == "completed"
    assert committed["partial_result"] is True
    coordinator.close()


def test_dual_candidate_both_fail_still_fails_without_aggregate():
    coordinator = TaskGraphCoordinator()
    calls = []

    def execute(stage, dependencies, root_input, cancel_event):
        calls.append(stage.stage_id)
        if stage.stage_type == "full_inference":
            raise RuntimeError("candidate failed")
        return {"content": "must not run"}

    with pytest.raises(WorkflowExecutionError):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_partialno1",
        )

    snapshot = coordinator.get("wf_partialno1")
    stages = {stage["stage_id"]: stage for stage in snapshot["stages"]}
    assert snapshot["state"] == "failed"
    assert snapshot["partial_result"] is False
    assert stages["candidate_a"]["state"] == "failed"
    assert stages["candidate_b"]["state"] == "failed"
    assert stages["aggregate"]["state"] == "skipped"
    assert "aggregate" not in calls
    coordinator.close()


def test_dual_candidate_aggregate_retries_once_on_same_local_provider():
    coordinator = TaskGraphCoordinator()
    aggregate_calls = 0

    def execute(stage, dependencies, root_input, cancel_event):
        nonlocal aggregate_calls
        if stage.stage_type == "aggregate":
            aggregate_calls += 1
            if aggregate_calls == 1:
                raise ProviderExecutionError(
                    "transient aggregate failure",
                    code="provider_execution_failed",
                    same_provider_retryable=True,
                )
            return {"content": "recovered aggregate"}
        return {"content": stage.stage_id}

    output, workflow = coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_aggretry1",
    )

    aggregate = next(
        stage for stage in workflow["stages"]
        if stage["stage_id"] == "aggregate"
    )
    assert output == {"content": "recovered aggregate"}
    assert aggregate_calls == 2
    assert aggregate["retry_count"] == 1
    assert aggregate["same_provider_retry_count"] == 1
    assert workflow["retry_count"] == 1
    assert workflow["same_provider_retry_count"] == 1
    assert [attempt["state"] for attempt in aggregate["attempts"]] == [
        "failed", "completed",
    ]
    assert [attempt["lease_epoch"] for attempt in aggregate["attempts"]] == [
        1, 2,
    ]
    coordinator.close()


def test_same_provider_retry_rejects_permanent_provider_output_error():
    coordinator = TaskGraphCoordinator()
    aggregate_calls = 0

    def execute(stage, dependencies, root_input, cancel_event):
        nonlocal aggregate_calls
        if stage.stage_type == "aggregate":
            aggregate_calls += 1
            return "invalid provider output"
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowExecutionError) as captured:
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_noretrypermanent",
        )

    assert captured.value.stage_id == "aggregate"
    snapshot = coordinator.get("wf_noretrypermanent")
    aggregate = next(
        stage for stage in snapshot["stages"]
        if stage["stage_id"] == "aggregate"
    )
    assert aggregate_calls == 1
    assert aggregate["retry_count"] == 0
    assert len(aggregate["attempts"]) == 1
    coordinator.close()


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("permanent runtime failure"),
        ProviderExecutionError(
            "explicit permanent provider failure",
            code="provider_execution_failed",
            retryable=False,
        ),
        ProviderExecutionError(
            "cross-provider-only failure",
            code="provider_execution_failed",
            retryable=True,
            same_provider_retryable=False,
        ),
    ],
)
def test_same_provider_retry_requires_explicit_error_marker(failure):
    coordinator = TaskGraphCoordinator()
    aggregate_calls = 0

    def execute(stage, dependencies, root_input, cancel_event):
        nonlocal aggregate_calls
        if stage.stage_type == "aggregate":
            aggregate_calls += 1
            raise failure
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowExecutionError):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_explicitretryflag",
        )

    aggregate = next(
        stage for stage in coordinator.get("wf_explicitretryflag")["stages"]
        if stage["stage_id"] == "aggregate"
    )
    assert aggregate_calls == 1
    assert aggregate["retry_count"] == 0
    assert len(aggregate["attempts"]) == 1
    coordinator.close()


def test_tolerant_side_branch_does_not_mask_strict_final_path_failure():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        if stage.stage_id == "source_a":
            raise RuntimeError("strict source failed")
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowExecutionError) as captured:
        coordinator.run(
            [
                StageSpec("source_a", "full_inference"),
                StageSpec("source_b", "full_inference"),
                StageSpec(
                    "tolerant_side",
                    "aggregate",
                    depends_on=("source_a", "source_b"),
                    minimum_successful_dependencies=1,
                ),
                StageSpec(
                    "strict_path", "aggregate", depends_on=("source_a",),
                ),
                StageSpec(
                    "final", "aggregate", depends_on=("strict_path",),
                ),
            ],
            "final",
            {"message": "question"},
            execute_stage=execute,
            workflow_id="wf_strictpathfailure",
        )

    assert captured.value.stage_id == "source_a"
    assert "strict source failed" in str(captured.value)
    assert "final stage did not complete" not in str(captured.value)
    coordinator.close()


def test_unrelated_tolerated_failure_does_not_mark_final_result_partial():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        if stage.stage_id == "side_failure":
            raise RuntimeError("side branch failed")
        return {"content": stage.stage_id}

    output, snapshot = coordinator.run(
        [
            StageSpec("side_failure", "full_inference"),
            StageSpec("final", "full_inference"),
            StageSpec(
                "side_join",
                "aggregate",
                depends_on=("side_failure", "final"),
                minimum_successful_dependencies=1,
            ),
        ],
        "final",
        {"message": "question"},
        execute_stage=execute,
        workflow_id="wf_unrelatedfailure",
    )

    assert output == {"content": "final"}
    assert snapshot["failed_stage_count"] == 1
    assert snapshot["partial_result"] is False
    coordinator.close()


def test_cancel_running_workflow_discards_stage_result():
    coordinator = TaskGraphCoordinator()
    started = threading.Event()
    release = threading.Event()
    error = []

    def execute(stage, dependencies, root_input, cancel_event):
        started.set()
        assert release.wait(5)
        return {"content": "late"}

    def run():
        try:
            coordinator.run_template(
                "dual_candidate",
                {"message": "question"},
                execute,
                workflow_id="wf_cancel12",
            )
        except Exception as exc:
            error.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(5)
    snapshot = coordinator.cancel("wf_cancel12")
    assert snapshot["cancel_requested"] is True
    release.set()
    thread.join(5)

    assert not thread.is_alive()
    assert isinstance(error[0], WorkflowCancelled)
    workflow = coordinator.get("wf_cancel12")
    assert workflow["state"] == "cancelled"
    assert workflow["completed_stage_count"] == 0
    assert workflow["cancelled_stage_count"] == 3
    candidate = workflow["stages"][0]
    assert candidate["output_available"] is False
    assert candidate["attempts"][0]["state"] == "cancelled"


def test_cancel_before_registration_fences_future_workflow():
    coordinator = TaskGraphCoordinator()
    called = []

    assert coordinator.request_cancel("wf_cancelbefore") is None

    def execute(stage, dependencies, root_input, cancel_event):
        called.append(stage.stage_id)
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowCancelled):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_cancelbefore",
        )

    assert called == []
    assert coordinator.get("wf_cancelbefore")["state"] == "cancelled"


def test_duplicate_or_unsafe_workflow_id_is_rejected():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_duplicate",
    )
    with pytest.raises(TaskGraphError, match="already exists"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_duplicate",
        )
    with pytest.raises(TaskGraphError, match="workflow_id"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="../unsafe",
        )


def test_registry_prunes_old_terminal_workflows():
    coordinator = TaskGraphCoordinator(max_records=2)

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    for workflow_id in ("wf_record01", "wf_record02", "wf_record03"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id=workflow_id,
        )
        coordinator.commit_result(workflow_id)

    workflows = coordinator.list(limit=10)
    assert len(workflows) == 2
    assert {item["workflow_id"] for item in workflows} == {
        "wf_record02", "wf_record03",
    }


def test_workflow_list_filters_by_session():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    for workflow_id, session_id in (
        ("wf_sessiona1", "session-a"),
        ("wf_sessionb1", "session-b"),
    ):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            session_id=session_id,
            workflow_id=workflow_id,
        )
        coordinator.commit_result(workflow_id)

    workflows = coordinator.list(limit=10, session_id="session-a")
    assert [item["workflow_id"] for item in workflows] == ["wf_sessiona1"]
    coordinator.close()


def test_cancel_during_final_stage_wins_over_completed_state():
    coordinator = TaskGraphCoordinator()
    aggregate_calls = 0

    def execute(stage, dependencies, root_input, cancel_event):
        nonlocal aggregate_calls
        if stage.stage_id == "aggregate":
            aggregate_calls += 1
            cancel_event.set()
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowCancelled):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_finalcancel",
        )

    snapshot = coordinator.get("wf_finalcancel")
    assert snapshot["state"] == "cancelled"
    assert snapshot["stages"][-1]["output_available"] is False
    assert aggregate_calls == 1


def test_provider_error_after_cancel_is_reported_as_cancelled():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        cancel_event.set()
        raise RuntimeError("provider stopped while cancelling")

    with pytest.raises(WorkflowCancelled):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_errorcancel",
        )

    snapshot = coordinator.get("wf_errorcancel")
    assert snapshot["state"] == "cancelled"
    assert snapshot["stages"][0]["state"] == "cancelled"
    assert snapshot["stages"][0]["attempts"][0]["state"] == "cancelled"


def test_result_ready_workflow_can_be_discarded_before_result_commit():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_discarded1",
    )
    assert coordinator.get("wf_discarded1")["state"] == "result_ready"
    snapshot = coordinator.discard_result("wf_discarded1")

    assert snapshot["state"] == "cancelled"
    assert snapshot["cancel_requested"] is True
    assert snapshot["completed_stage_count"] == 3
    assert snapshot["error"] == "cancelled before result commit"


def test_completed_workflow_is_idempotent_and_cannot_be_discarded():
    coordinator = TaskGraphCoordinator()
    cancel_event = threading.Event()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_terminal1",
        cancel_event=cancel_event,
    )
    first = coordinator.commit_result("wf_terminal1")
    second = coordinator.commit_result("wf_terminal1")

    assert first["state"] == "completed"
    assert first["terminal_reason"]["reason_code"] == "completed"
    assert second["state"] == "completed"
    assert second["finished_at"] == first["finished_at"]
    with pytest.raises(TaskGraphError, match="cannot be discarded"):
        coordinator.discard_result("wf_terminal1")
    late_cancel = coordinator.cancel("wf_terminal1")
    assert late_cancel["cancel_requested"] is False
    assert late_cancel["terminal_reason"]["reason_code"] == "completed"
    assert [
        item["reason_code"]
        for item in late_cancel["terminal_reason"]["secondary_observations"]
    ] == ["generation_cancelled"]
    cancel_event.set()
    assert coordinator.get("wf_terminal1")["cancel_requested"] is False
    assert coordinator.get("wf_terminal1")["state"] == "completed"


def test_cancel_result_ready_is_terminal_and_prevents_commit():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_terminal2",
    )
    cancelled = coordinator.cancel("wf_terminal2")

    assert cancelled["state"] == "cancelled"
    assert cancelled["cancel_requested"] is True
    assert coordinator.discard_result("wf_terminal2")["state"] == "cancelled"
    with pytest.raises(WorkflowCancelled):
        coordinator.commit_result("wf_terminal2")


def test_refused_result_ready_is_failed_without_claiming_user_cancel():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_refused01",
    )
    refused = coordinator.discard_result(
        "wf_refused01",
        reason="distributed route refused",
        terminal_reason_code="request_refused",
    )

    assert refused["state"] == "failed"
    assert refused["error"] == "request_refused"
    assert refused["cancel_requested"] is False
    assert refused["terminal_reason"]["reason_code"] == "request_refused"
