import json
import os
import subprocess
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recovery_journal import (
    JournalEvent,
    SQLiteTaskJournal,
    TaskJournalConflict,
    TaskJournalError,
    TaskJournalInUse,
)
from execution_provider import ProviderExecutionError


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
        "QLH_LEGACY_RECOVERY_INSTANCE_ID": "state-dir-probe",
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
    "journal_path": config.LEGACY_RECOVERY_JOURNAL_PATH,
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
