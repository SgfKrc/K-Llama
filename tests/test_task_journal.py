import json
import os
import subprocess
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import (
    TaskGraphCoordinator,
    TaskGraphUnavailable,
)
from task_journal import (
    JournalEvent,
    SQLiteTaskJournal,
    TaskJournalConflict,
    TaskJournalError,
    TaskJournalInUse,
)
from task_provider import ProviderExecutionError


def _event(event_id="evt_one", sequence=1, occurred_at=100.0, payload=None):
    return JournalEvent(
        event_id=event_id,
        workflow_id="wf_journal1",
        sequence=sequence,
        entity_type="workflow",
        entity_id="wf_journal1",
        event_type="workflow_state_changed",
        occurred_at=occurred_at,
        payload=payload or {"from_state": "created", "to_state": "running"},
    )


def _snapshot(sequence=1, state="running"):
    return {
        "workflow_id": "wf_journal1",
        "last_sequence": sequence,
        "state": state,
        "created_at": 90.0,
        "stages": [],
    }


def _append_workflow_snapshot(
    journal,
    workflow_id,
    state,
    occurred_at,
    *,
    stages=None,
    sequence=1,
    **snapshot_fields,
):
    journal.append_event(
        JournalEvent(
            event_id=f"evt_{workflow_id}_{sequence}",
            workflow_id=workflow_id,
            sequence=sequence,
            entity_type="workflow",
            entity_id=workflow_id,
            event_type="workflow_state_changed",
            occurred_at=occurred_at,
            payload={"to_state": state},
        ),
        {
            "workflow_id": workflow_id,
            "last_sequence": sequence,
            "state": state,
            "created_at": occurred_at - 10,
            "started_at": occurred_at - 5,
            "finished_at": occurred_at if state in {
                "completed", "failed", "cancelled",
            } else None,
            "stages": list(stages or []),
            **snapshot_fields,
        },
    )


def test_sqlite_journal_initializes_in_nested_directory(tmp_path):
    path = tmp_path / "nested" / "task_graph.sqlite3"
    journal = SQLiteTaskJournal(str(path))

    status = journal.health()

    assert path.is_file()
    assert status["available"] is True
    assert status["backend"] == "sqlite"
    assert status["journal_mode"] == "wal"
    assert status["schema_version"] == 1


def test_sqlite_journal_has_single_process_owner_until_closed(tmp_path):
    path = str(tmp_path / "task_graph.sqlite3")
    first = SQLiteTaskJournal(path)

    with pytest.raises(TaskJournalInUse, match="another process"):
        SQLiteTaskJournal(path)

    first.close()
    with pytest.raises(TaskJournalError, match="closed"):
        first.get_snapshot("wf_unused123")

    second = SQLiteTaskJournal(path)
    assert second.health()["available"] is True
    second.close()


def test_state_dir_override_is_user_configurable(tmp_path):
    configured = tmp_path / "custom-state"
    # conftest deliberately replaces cached paths in the test process. Probe
    # real import-time Config resolution in a fresh interpreter instead.
    node_config_path = tmp_path / "probe-node-config.json"
    node_config_path.write_text("{}", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "QLH_STATE_DIR": str(configured),
        "QLH_NODE_CONFIG_PATH": str(node_config_path),
        "QLH_NODE_SECRET_STORE_PATH": str(tmp_path / "probe-node-secrets.json"),
        "QLH_SQLITE_PATH": str(tmp_path / "probe-local-store.sqlite3"),
        "QLH_TASK_GRAPH_INSTANCE_ID": "state-dir-probe",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (src_root, env.get("PYTHONPATH", "")) if path
    )
    probe = """
import json
import config
print(json.dumps({
    "resolved_state_dir": config._get_state_dir(),
    "state_dir": config.STATE_DIR,
    "journal_path": config.TASK_GRAPH_JOURNAL_PATH,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    resolved = json.loads(completed.stdout)
    assert resolved["resolved_state_dir"] == str(configured.resolve())
    assert resolved["state_dir"] == str(configured.resolve())
    assert os.path.dirname(resolved["journal_path"]) == str(configured / "task_graph")
    filename = os.path.basename(resolved["journal_path"])
    assert filename.startswith("instance-state-dir-probe-")
    assert filename.endswith(".sqlite3")


def test_event_and_snapshot_commit_atomically_and_replay_is_idempotent(tmp_path):
    journal = SQLiteTaskJournal(str(tmp_path / "task_graph.sqlite3"))
    event = _event()
    snapshot = _snapshot()

    assert journal.append_event(event, snapshot) is True
    assert journal.append_event(event, snapshot) is False
    assert journal.get_snapshot("wf_journal1") == snapshot
    events = journal.list_events("wf_journal1")
    assert len(events) == 1
    assert events[0]["event_id"] == "evt_one"
    assert events[0]["payload"]["to_state"] == "running"

    second_event = _event("evt_two", sequence=2, occurred_at=101.0)
    second_snapshot = _snapshot(sequence=2, state="result_ready")
    assert journal.append_event(second_event, second_snapshot) is True
    assert journal.append_event(event, snapshot) is False
    assert journal.get_snapshot("wf_journal1") == second_snapshot


def test_event_id_conflict_and_sequence_gap_do_not_advance_snapshot(tmp_path):
    journal = SQLiteTaskJournal(str(tmp_path / "task_graph.sqlite3"))
    journal.append_event(_event(), _snapshot())

    with pytest.raises(TaskJournalConflict, match="conflicting content"):
        journal.append_event(
            _event(payload={"from_state": "created", "to_state": "failed"}),
            _snapshot(),
        )
    with pytest.raises(TaskJournalConflict, match="conflicting snapshot"):
        journal.append_event(
            _event(),
            {**_snapshot(), "state": "failed"},
        )
    with pytest.raises(TaskJournalConflict, match="expected sequence 2"):
        journal.append_event(
            _event("evt_gap", sequence=3, occurred_at=101.0),
            _snapshot(sequence=3),
        )

    assert journal.get_snapshot("wf_journal1")["last_sequence"] == 1
    assert len(journal.list_events("wf_journal1")) == 1


def test_journal_rejects_mismatched_or_non_json_snapshot(tmp_path):
    journal = SQLiteTaskJournal(str(tmp_path / "task_graph.sqlite3"))

    with pytest.raises(TaskJournalConflict, match="workflow_id"):
        journal.append_event(
            _event(),
            {**_snapshot(), "workflow_id": "wf_other123"},
        )
    with pytest.raises(TaskJournalError, match="not JSON serializable"):
        journal.append_event(
            _event(),
            {**_snapshot(), "bad": object()},
        )

    assert journal.get_snapshot("wf_journal1") is None


def test_coordinator_terminal_snapshot_is_queryable_from_new_instance(tmp_path):
    path = str(tmp_path / "task_graph.sqlite3")
    first = TaskGraphCoordinator(journal=SQLiteTaskJournal(path))

    def execute(stage, dependencies, root_input, cancel_event):
        if stage.stage_type == "aggregate":
            return {"content": "secret final answer"}
        return {"content": f"secret {stage.stage_id}"}

    first.run_template(
        "dual_candidate",
        {"message": "secret prompt"},
        execute,
        workflow_id="wf_durable1",
    )
    committed = first.commit_result("wf_durable1")
    assert committed["state"] == "completed"
    first.close()

    second_journal = SQLiteTaskJournal(path)
    second = TaskGraphCoordinator(journal=second_journal)
    recovered = second.get("wf_durable1")

    assert recovered["state"] == "completed"
    assert recovered["recovery_pending"] is False
    assert recovered["runtime_status"] == "terminal"
    assert recovered["last_sequence"] == committed["last_sequence"]
    assert second.list(limit=10)[0]["workflow_id"] == "wf_durable1"
    durable_text = str(recovered)
    assert recovered["error"] == ""
    assert "secret prompt" not in durable_text
    assert "secret final answer" not in durable_text
    assert "secret candidate" not in durable_text
    events_text = str(second_journal.list_events("wf_durable1"))
    sequences = [
        event["sequence"]
        for event in second_journal.list_events("wf_durable1")
    ]
    assert sequences == list(range(1, len(sequences) + 1))
    assert recovered["last_sequence"] == sequences[-1]
    assert "secret prompt" not in events_text
    assert "secret final answer" not in events_text
    persisted_cancel = second.request_cancel("wf_durable1")
    assert persisted_cancel is not None
    assert persisted_cancel["state"] == "completed"
    second.close()


def test_coordinator_failure_snapshot_redacts_exception_text(tmp_path):
    path = str(tmp_path / "task_graph.sqlite3")
    journal = SQLiteTaskJournal(path)
    coordinator = TaskGraphCoordinator(journal=journal)

    def execute(stage, dependencies, root_input, cancel_event):
        raise RuntimeError("secret credential text")

    with pytest.raises(Exception):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_redacted1",
        )

    durable_text = str(journal.get_snapshot("wf_redacted1"))
    events_text = str(journal.list_events("wf_redacted1"))
    assert "secret credential text" not in durable_text
    assert "secret credential text" not in events_text
    assert "error_present" in durable_text


class _FailingJournal:
    def append_event(self, event, snapshot):
        raise TaskJournalError("disk full")

    def get_snapshot(self, workflow_id):
        return None

    def list_snapshots(self, limit=20):
        return []

    def list_events(self, workflow_id):
        return []

    def health(self):
        return {"enabled": True, "available": True, "backend": "fake"}

    def close(self):
        return None


def test_journal_write_failure_disables_task_graph_instead_of_using_memory():
    coordinator = TaskGraphCoordinator(journal=_FailingJournal())

    with pytest.raises(TaskGraphUnavailable, match="disk full"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            lambda *args: {"content": "answer"},
            workflow_id="wf_diskfull1",
        )
    assert coordinator.list(limit=10) == []
    with pytest.raises(Exception):
        coordinator.get("wf_diskfull1")

    status = coordinator.journal_status()
    assert status["available"] is False
    assert "disk full" in status["error"]
    with pytest.raises(TaskGraphUnavailable, match="disk full"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            lambda *args: {"content": "answer"},
            workflow_id="wf_diskfull2",
        )


def test_completed_stage_snapshot_never_contains_running_attempt(tmp_path):
    journal = SQLiteTaskJournal(str(tmp_path / "task_graph.sqlite3"))
    coordinator = TaskGraphCoordinator(journal=journal)

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_ordering1",
    )

    snapshot = journal.get_snapshot("wf_ordering1")
    assert snapshot is not None
    for stage in snapshot["stages"]:
        if stage["state"] == "completed":
            assert all(
                attempt["state"] != "running"
                for attempt in stage["attempts"]
            )


def test_persisted_nonterminal_is_marked_pending_and_cannot_be_cancelled(
    tmp_path,
):
    path = str(tmp_path / "task_graph.sqlite3")
    journal = SQLiteTaskJournal(path)
    event = JournalEvent(
        event_id="evt_pending",
        workflow_id="wf_pending1",
        sequence=1,
        entity_type="workflow",
        entity_id="wf_pending1",
        event_type="workflow_state_changed",
        occurred_at=100.0,
        payload={"from_state": "created", "to_state": "running"},
    )
    journal.append_event(
        event,
        {
            "workflow_id": "wf_pending1",
            "last_sequence": 1,
            "state": "running",
            "created_at": 90.0,
            "stages": [],
        },
    )
    journal.close()
    coordinator = TaskGraphCoordinator(journal=SQLiteTaskJournal(path))

    snapshot = coordinator.get("wf_pending1")
    assert snapshot["recovery_pending"] is True
    assert snapshot["runtime_status"] == "persisted_unrecovered"
    with pytest.raises(TaskGraphUnavailable, match="requires restart recovery"):
        coordinator.request_cancel("wf_pending1")
    coordinator.close()


def test_restart_recovery_expires_attempts_and_terminalizes_workflow(tmp_path):
    path = str(tmp_path / "task_graph.sqlite3")
    seed = SQLiteTaskJournal(path)
    _append_workflow_snapshot(
        seed,
        "wf_recovery01",
        "running",
        100.0,
        stages=[
            {
                "stage_id": "candidate",
                "state": "running",
                "started_at": 96.0,
                "finished_at": None,
                "attempts": [{
                    "attempt_id": "attempt_running",
                    "state": "running",
                    "reservation_id": "res_before_restart",
                    "reservation_active": True,
                    "started_at": 97.0,
                    "finished_at": None,
                }],
            },
            {
                "stage_id": "aggregate",
                "state": "blocked",
                "started_at": None,
                "finished_at": None,
                "attempts": [],
            },
        ],
    )
    seed.close()

    journal = SQLiteTaskJournal(path)
    coordinator = TaskGraphCoordinator(journal=journal)
    summary = coordinator.recover_persisted_workflows()
    recovered = coordinator.get("wf_recovery01")
    stages = {stage["stage_id"]: stage for stage in recovered["stages"]}

    assert summary == {
        "recovered_workflows": 1,
        "expired_attempts": 1,
        "failed_stages": 1,
        "skipped_stages": 1,
    }
    assert recovered["state"] == "failed"
    assert recovered["partial_result"] is False
    assert recovered["skipped_stage_count"] == 1
    assert recovered["last_sequence"] == 2
    assert recovered["recovery_pending"] is False
    assert recovered["runtime_status"] == "terminal"
    assert recovered["error_code"] == "coordinator_restarted_during_execution"
    assert stages["candidate"]["state"] == "failed"
    assert stages["candidate"]["attempts"][0]["state"] == "expired"
    assert stages["candidate"]["attempts"][0]["reservation_active"] is False
    assert stages["aggregate"]["state"] == "skipped"
    events = journal.list_events("wf_recovery01")
    assert [event["sequence"] for event in events] == [1, 2]
    assert events[-1]["event_type"] == "workflow_recovered_after_restart"

    assert coordinator.recover_persisted_workflows() == {
        "recovered_workflows": 0,
        "expired_attempts": 0,
        "failed_stages": 0,
        "skipped_stages": 0,
    }
    assert len(journal.list_events("wf_recovery01")) == 2
    coordinator.close()


def test_result_ready_recovery_fails_without_replaying_completed_model_work(tmp_path):
    path = str(tmp_path / "task_graph.sqlite3")
    seed = SQLiteTaskJournal(path)
    _append_workflow_snapshot(
        seed,
        "wf_readyrec01",
        "result_ready",
        100.0,
        stages=[{
            "stage_id": "aggregate",
            "state": "completed",
            "started_at": 96.0,
            "finished_at": 99.0,
            "attempts": [{
                "attempt_id": "attempt_completed",
                "state": "completed",
                "started_at": 96.0,
                "finished_at": 99.0,
            }],
        }],
    )
    seed.close()

    coordinator = TaskGraphCoordinator(journal=SQLiteTaskJournal(path))
    summary = coordinator.recover_persisted_workflows()
    recovered = coordinator.get("wf_readyrec01")

    assert summary["recovered_workflows"] == 1
    assert summary["expired_attempts"] == 0
    assert summary["failed_stages"] == 0
    assert recovered["state"] == "failed"
    assert recovered["error_code"] == "coordinator_restarted_before_result_commit"
    assert recovered["stages"][0]["state"] == "completed"
    assert recovered["stages"][0]["attempts"][0]["state"] == "completed"
    coordinator.close()


def test_partial_result_and_retry_counters_persist_without_error_text(tmp_path):
    path = str(tmp_path / "partial-result.sqlite3")
    coordinator = TaskGraphCoordinator(journal=SQLiteTaskJournal(path))
    aggregate_calls = 0

    def execute(stage, dependencies, root_input, cancel_event):
        nonlocal aggregate_calls
        if stage.stage_id == "candidate_b":
            raise RuntimeError("private candidate failure")
        if stage.stage_type == "aggregate":
            aggregate_calls += 1
            if aggregate_calls == 1:
                raise ProviderExecutionError(
                    "private aggregate retry detail",
                    code="provider_execution_failed",
                    same_provider_retryable=True,
                )
            return {"content": dependencies["candidate_a"]["content"]}
        return {"content": "surviving candidate"}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_partialpersist",
    )
    coordinator.commit_result("wf_partialpersist")
    coordinator.close()

    reopened = TaskGraphCoordinator(journal=SQLiteTaskJournal(path))
    try:
        snapshot = reopened.get("wf_partialpersist")
        assert snapshot["state"] == "completed"
        assert snapshot["partial_result"] is True
        assert snapshot["failed_stage_count"] == 1
        assert snapshot["same_provider_retry_count"] == 1
        encoded = str(snapshot)
        assert "private candidate failure" not in encoded
        assert "private aggregate retry detail" not in encoded
    finally:
        reopened.close()


def test_terminal_cleanup_by_age_never_deletes_nonterminal_workflows(tmp_path):
    journal = SQLiteTaskJournal(str(tmp_path / "task_graph.sqlite3"))
    _append_workflow_snapshot(journal, "wf_oldterm01", "completed", 100.0)
    _append_workflow_snapshot(journal, "wf_newterm01", "failed", 900.0)
    _append_workflow_snapshot(journal, "wf_oldrunning", "running", 100.0)

    result = journal.cleanup_terminal(max_age_seconds=200.0, now=1000.0)

    assert result == {
        "deleted_workflows": 1,
        "deleted_events": 1,
        "deleted_by_age": 1,
        "deleted_by_limit": 0,
        "remaining_terminal": 1,
    }
    assert journal.get_snapshot("wf_oldterm01") is None
    assert journal.list_events("wf_oldterm01") == []
    assert journal.get_snapshot("wf_newterm01")["state"] == "failed"
    assert journal.get_snapshot("wf_oldrunning")["state"] == "running"
    journal.close()


def test_terminal_cleanup_record_limit_keeps_newest_terminal_snapshots(tmp_path):
    journal = SQLiteTaskJournal(str(tmp_path / "task_graph.sqlite3"))
    _append_workflow_snapshot(journal, "wf_limitold1", "completed", 100.0)
    _append_workflow_snapshot(journal, "wf_limitmid1", "failed", 200.0)
    _append_workflow_snapshot(journal, "wf_limitnew1", "cancelled", 300.0)
    _append_workflow_snapshot(journal, "wf_limitrun1", "running", 50.0)

    result = journal.cleanup_terminal(max_records=2)

    assert result["deleted_workflows"] == 1
    assert result["deleted_by_limit"] == 1
    assert result["remaining_terminal"] == 2
    assert journal.get_snapshot("wf_limitold1") is None
    assert journal.get_snapshot("wf_limitmid1") is not None
    assert journal.get_snapshot("wf_limitnew1") is not None
    assert journal.get_snapshot("wf_limitrun1")["state"] == "running"
    journal.close()


def test_list_returns_current_memory_snapshot_when_journal_is_unavailable():
    coordinator = TaskGraphCoordinator()

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        lambda stage, *args: {"content": stage.stage_id},
        workflow_id="wf_memoryview",
    )
    coordinator._journal_error = "task journal unavailable: disk full"

    snapshots = coordinator.list(limit=10)
    assert snapshots[0]["workflow_id"] == "wf_memoryview"
    assert coordinator.journal_status()["available"] is False
