"""Journal replica truncation keeps recovery bounded to a real hash-chain point.

After truncation, the snapshot still refers to the removed tail, so the replica is
intentionally unverified until a later checkpoint restores a consistent prefix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journal_replication import (  # noqa: E402
    DurableJournalReplica,
    JournalReplicationConflict,
    JournalReplicationGap,
    build_checkpoint,
)
from recovery_journal import JournalEvent, SQLiteTaskJournal  # noqa: E402

SOURCE = "node-a"
STREAM = "journal-1"


def _snapshot(workflow_id: str, sequence: int, *, state: str = "running") -> dict:
    return {
        "workflow_id": workflow_id,
        "last_sequence": sequence,
        "state": state,
        "created_at": 100.0,
        "updated_at": 100.0 + sequence,
        "stages": [
            {
                "stage_id": "stage-1",
                "state": "running",
                "retry_safe": True,
                "pure": True,
                "attempts": [{"state": "running"}],
            }
        ],
    }


def _append(journal: SQLiteTaskJournal, workflow_id: str, sequence: int) -> None:
    journal.append_event(
        JournalEvent(
            event_id=f"evt-{workflow_id}-{sequence}",
            workflow_id=workflow_id,
            sequence=sequence,
            entity_type="workflow",
            entity_id=workflow_id,
            event_type="workflow_state_changed",
            occurred_at=100.0 + sequence,
            payload={"state": "running", "sequence": sequence},
        ),
        _snapshot(workflow_id, sequence),
    )


def _replica_with(tmp_path: Path, workflow_id: str, sequences: int):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    replica = DurableJournalReplica(
        str(tmp_path / "replica.sqlite3"), source_node_id=SOURCE, stream_id=STREAM,
    )
    for sequence in range(1, sequences + 1):
        _append(source, workflow_id, sequence)
    checkpoint = build_checkpoint(
        source, source_node_id=SOURCE, stream_id=STREAM,
        workflow_id=workflow_id, created_at=1.0,
    )
    replica.apply_checkpoint(checkpoint)
    return source, replica, checkpoint


def test_truncate_drops_uncommitted_tail_keeps_prefix(tmp_path):
    source, replica, _ = _replica_with(tmp_path, "wf-t1", 5)
    result = replica.truncate_to("wf-t1", last_sequence=3)
    assert result["accepted"] is True
    assert result["durable_sequence"] == 3
    assert result["removed_events"] == 2
    prefix = replica.durable_prefix("wf-t1")
    assert prefix["last_sequence"] == 3
    assert prefix["event_count"] == 3
    # 截断后不再自洽 ⇒ 不允许被当成已提交前缀（必须靠追赶重建）
    assert replica.recovery_status("wf-t1")["verified"] is False
    with pytest.raises(Exception):
        replica.recovery_projection("wf-t1")
    source.close()
    replica.close()


def test_truncate_rejects_point_the_replica_does_not_hold(tmp_path):
    """不允许截到一个本地没有的点（否则等于凭空造前缀）。"""
    source, replica, _ = _replica_with(tmp_path, "wf-t2", 3)
    with pytest.raises(JournalReplicationGap):
        replica.truncate_to("wf-t2", last_sequence=5)
    assert replica.durable_prefix("wf-t2")["last_sequence"] == 3
    source.close()
    replica.close()


def test_truncate_rejects_mismatched_chain_head(tmp_path):
    source, replica, _ = _replica_with(tmp_path, "wf-t3", 4)
    with pytest.raises(JournalReplicationConflict):
        replica.truncate_to("wf-t3", last_sequence=2, event_chain_head="0" * 64)
    assert replica.durable_prefix("wf-t3")["last_sequence"] == 4
    source.close()
    replica.close()


def test_truncate_accepts_matching_chain_head(tmp_path):
    source, replica, checkpoint = _replica_with(tmp_path, "wf-t7", 4)
    head = checkpoint.records[1]["event_digest"]  # sequence == 2
    result = replica.truncate_to("wf-t7", last_sequence=2, event_chain_head=head)
    assert result["accepted"] is True
    prefix = replica.durable_prefix("wf-t7")
    assert prefix["last_sequence"] == 2
    assert prefix["event_chain_head"] == head
    source.close()
    replica.close()


def test_truncate_to_zero_clears_the_chain(tmp_path):
    source, replica, _ = _replica_with(tmp_path, "wf-t4", 3)
    result = replica.truncate_to("wf-t4", last_sequence=0)
    assert result["removed_events"] == 3
    prefix = replica.durable_prefix("wf-t4")
    assert prefix["event_count"] == 0
    assert prefix["last_sequence"] == 0
    source.close()
    replica.close()


def test_truncate_records_stale_snapshot_marker(tmp_path):
    """快照内容仍指向被截掉的高点 ⇒ 必须留痕，不能假装它就是已提交前缀。"""
    source, replica, _ = _replica_with(tmp_path, "wf-t5", 4)
    result = replica.truncate_to("wf-t5", last_sequence=2)
    assert result["snapshot_stale"] is True
    assert replica.truncated_watermark("wf-t5").startswith("2@")
    source.close()
    replica.close()


def test_truncate_to_current_point_is_a_noop(tmp_path):
    source, replica, _ = _replica_with(tmp_path, "wf-t6", 3)
    result = replica.truncate_to("wf-t6", last_sequence=3)
    assert result["removed_events"] == 0
    assert result["snapshot_stale"] is False
    assert replica.truncated_watermark("wf-t6") == ""
    assert replica.recovery_status("wf-t6")["verified"] is True
    source.close()
    replica.close()
