"""Journal recovery policy gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cluster_recovery import decide_recovery  # noqa: E402
from recovery_journal import JournalEvent, SQLiteTaskJournal  # noqa: E402


def _append(journal, workflow_id, snapshot, event_type="workflow_state_changed"):
    journal.append_event(
        JournalEvent(
            event_id=f"evt_{workflow_id}_{snapshot['last_sequence']}",
            workflow_id=workflow_id,
            sequence=snapshot["last_sequence"],
            entity_type="workflow",
            entity_id=workflow_id,
            event_type=event_type,
            occurred_at=float(snapshot.get("updated_at", 100.0)),
            payload={"state": snapshot["state"]},
        ),
        snapshot,
    )


def _running_snapshot(workflow_id: str, **changes):
    snapshot = {
        "workflow_id": workflow_id,
        "last_sequence": 1,
        "state": "running",
        "created_at": 90.0,
        "started_at": 95.0,
        "stages": [{
            "stage_id": "pure-stage",
            "state": "running",
            "pure": True,
            "retry_safe": True,
            "attempts": [{
                "attempt_id": "attempt-1",
                "state": "running",
                "started_at": 96.0,
                "reservation_active": True,
            }],
        }],
    }
    snapshot.update(changes)
    return snapshot


def test_recovery_policy_retries_only_explicitly_safe_stage():
    decision = decide_recovery(_running_snapshot("wf_retry01"))
    assert decision.action == "retry"
    assert decision.retry_stage_ids == ("pure-stage",)
    assert decision.expired_attempts == 1


def test_recovery_policy_prefers_cancel_request_over_retry():
    snapshot = _running_snapshot("wf_cancel01", cancel_requested=True)
    decision = decide_recovery(snapshot)
    assert decision.action == "cancel"
    assert decision.reason == "recovered_cancel_request"


def test_recovery_policy_can_continue_an_explicitly_continuable_projection():
    snapshot = _running_snapshot(
        "wf_continue01", recovery_continuable=True,
    )
    decision = decide_recovery(snapshot)
    assert decision.action == "continue"
    assert decision.reason == "journal_state_continuable"
