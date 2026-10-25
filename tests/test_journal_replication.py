"""Durable cross-node journal replication contract tests."""

from __future__ import annotations

import multiprocessing
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journal_replication import (  # noqa: E402
    DurableJournalReplica,
    JournalCheckpoint,
    JournalReplicationConflict,
    JournalReplicationError,
    JournalReplicationGap,
    build_checkpoint,
)
from journal_replication_transport import (  # noqa: E402
    JournalReplicationTcpServer,
    JournalReplicationTransportError,
    send_checkpoint,
)
from recovery_journal import JournalEvent, SQLiteTaskJournal  # noqa: E402


def _serve_replica_child(replica_path, ready_queue, stop_event, secret):
    replica = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    server = JournalReplicationTcpServer(replica, secret=secret)
    try:
        ready_queue.put(server.start())
        stop_event.wait(15)
    finally:
        server.close()
        replica.close()


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


def _append(journal: SQLiteTaskJournal, workflow_id: str, sequence: int, *, state: str = "running") -> None:
    journal.append_event(
        JournalEvent(
            event_id=f"evt-{workflow_id}-{sequence}",
            workflow_id=workflow_id,
            sequence=sequence,
            entity_type="workflow",
            entity_id=workflow_id,
            event_type="workflow_state_changed",
            occurred_at=100.0 + sequence,
            payload={"state": state, "sequence": sequence},
        ),
        _snapshot(workflow_id, sequence, state=state),
    )


def test_checkpoint_round_trip_and_incremental_idempotent_apply(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-1", 1)
    replica = DurableJournalReplica(
        str(tmp_path / "replica.sqlite3"), source_node_id="node-a", stream_id="journal-1"
    )

    first = build_checkpoint(
        source, source_node_id="node-a", stream_id="journal-1", workflow_id="wf-1", created_at=1.0
    )
    assert replica.apply_checkpoint(first)["durable_sequence"] == 1
    assert replica.apply_checkpoint(first)["idempotent"] is True

    _append(source, "wf-1", 2)
    second = build_checkpoint(
        source, source_node_id="node-a", stream_id="journal-1", workflow_id="wf-1", created_at=2.0
    )
    result = replica.apply_checkpoint(second)
    assert result["idempotent"] is False
    assert result["durable_sequence"] == 2
    snapshot, events, status = replica.recovery_projection("wf-1")
    assert snapshot["last_sequence"] == 2
    assert [event["sequence"] for event in events] == [1, 2]
    assert status["verified"] is True

    source.close()
    replica.close()


def test_replica_rejects_source_fork_and_keeps_previous_durable_state(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-2", 1)
    replica = DurableJournalReplica(
        str(tmp_path / "replica.sqlite3"), source_node_id="node-a", stream_id="journal-1"
    )
    original = build_checkpoint(
        source, source_node_id="node-a", stream_id="journal-1", workflow_id="wf-2"
    )
    replica.apply_checkpoint(original)

    fork_event = dict(original.records[0]["event"])
    fork_event["payload"] = {"state": "forked", "sequence": 1}
    fork_record = dict(original.records[0])
    fork_record["event"] = fork_event
    fork_records = (fork_record,)
    with pytest.raises(JournalReplicationConflict):
        # The constructor recalculates and rejects a digest that does not match.
        JournalCheckpoint(
            source_node_id=original.source_node_id,
            stream_id=original.stream_id,
            workflow_id=original.workflow_id,
            created_at=original.created_at,
            snapshot=original.snapshot,
            records=fork_records,
            event_chain_head=original.event_chain_head,
            checkpoint_digest=original.checkpoint_digest,
        )
    assert replica.recovery_status("wf-2")["durable_sequence"] == 1
    source.close()
    replica.close()


def test_replica_rejects_validly_rehashed_fork_and_wrong_source(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-fork", 1)
    replica = DurableJournalReplica(
        str(tmp_path / "replica.sqlite3"), source_node_id="node-a", stream_id="journal-1"
    )
    first = build_checkpoint(
        source,
        source_node_id="node-a",
        stream_id="journal-1",
        workflow_id="wf-fork",
    )
    replica.apply_checkpoint(first)
    _append(source, "wf-fork", 2)

    fork_journal = SQLiteTaskJournal(str(tmp_path / "fork.sqlite3"))
    _append(fork_journal, "wf-fork", 1)
    _append(fork_journal, "wf-fork", 2)
    fork_snapshot = fork_journal.get_snapshot("wf-fork")
    fork_events = fork_journal.list_events("wf-fork")
    fork_events[0]["payload"] = {"state": "forked", "sequence": 1}
    fork_records = []
    from journal_replication import _event_digest, _checkpoint_digest

    previous_digest = "0" * 64
    for event in fork_events:
        digest = _event_digest(previous_digest, event)
        fork_records.append({
            "event": event,
            "event_digest": digest,
            "previous_digest": previous_digest,
        })
        previous_digest = digest
    forked = JournalCheckpoint(
        source_node_id="node-a",
        stream_id="journal-1",
        workflow_id="wf-fork",
        created_at=2.0,
        snapshot=fork_snapshot,
        records=tuple(fork_records),
        event_chain_head=previous_digest,
        checkpoint_digest=_checkpoint_digest(
            "node-a", "journal-1", "wf-fork", previous_digest, fork_snapshot
        ),
    )
    with pytest.raises(JournalReplicationConflict):
        replica.apply_checkpoint(forked)

    wrong_source = build_checkpoint(
        source,
        source_node_id="node-other",
        stream_id="journal-1",
        workflow_id="wf-fork",
    )
    with pytest.raises(JournalReplicationConflict):
        replica.apply_checkpoint(wrong_source)
    assert replica.recovery_status("wf-fork")["durable_sequence"] == 1
    fork_journal.close()
    source.close()
    replica.close()


def test_replica_rejects_sequence_gap_before_writing(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-3", 1)
    _append(source, "wf-3", 2)
    checkpoint = build_checkpoint(
        source, source_node_id="node-a", stream_id="journal-1", workflow_id="wf-3"
    )
    raw = checkpoint.to_dict()
    raw["records"] = raw["records"][1:]
    with pytest.raises(JournalReplicationGap):
        JournalCheckpoint.from_dict(raw)

    replica = DurableJournalReplica(
        str(tmp_path / "replica.sqlite3"), source_node_id="node-a", stream_id="journal-1"
    )
    assert replica.recovery_status("wf-3")["verified"] is False
    source.close()
    replica.close()


def test_durable_replica_gates_recovery_and_survives_reopen(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-4", 1)
    replica_path = str(tmp_path / "replica.sqlite3")
    replica = DurableJournalReplica(replica_path, source_node_id="node-a", stream_id="journal-1")
    checkpoint = build_checkpoint(
        source, source_node_id="node-a", stream_id="journal-1", workflow_id="wf-4"
    )
    replica.apply_checkpoint(checkpoint)
    assert replica.decide_recovery("wf-4").action == "retry"
    replica.close()

    reopened = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    assert reopened.decide_recovery("wf-4").reason == "coordinator_restarted_during_execution"
    with pytest.raises(JournalReplicationError):
        reopened.recovery_projection("missing-workflow")
    source.close()
    reopened.close()


def test_snapshot_digest_tamper_fails_closed(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-tamper", 1)
    replica_path = str(tmp_path / "replica.sqlite3")
    replica = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    replica.apply_checkpoint(
        build_checkpoint(
            source,
            source_node_id="node-a",
            stream_id="journal-1",
            workflow_id="wf-tamper",
        )
    )
    replica.close()

    import sqlite3

    connection = sqlite3.connect(replica_path)
    try:
        connection.execute(
            "UPDATE replicated_snapshots SET snapshot_json = ? WHERE workflow_id = ?",
            (json.dumps(_snapshot("wf-tamper", 1, state="tampered")), "wf-tamper"),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    status = reopened.recovery_status("wf-tamper")
    assert status["verified"] is False
    with pytest.raises(JournalReplicationError):
        reopened.recovery_projection("wf-tamper")
    source.close()
    reopened.close()


def test_replicated_event_tamper_fails_closed(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-event-tamper", 1)
    replica_path = str(tmp_path / "replica.sqlite3")
    replica = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    replica.apply_checkpoint(
        build_checkpoint(
            source,
            source_node_id="node-a",
            stream_id="journal-1",
            workflow_id="wf-event-tamper",
        )
    )
    replica.close()

    import sqlite3

    connection = sqlite3.connect(replica_path)
    try:
        connection.execute(
            "UPDATE replicated_events SET event_digest = ? WHERE workflow_id = ?",
            ("f" * 64, "wf-event-tamper"),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    assert reopened.recovery_status("wf-event-tamper")["verified"] is False
    with pytest.raises(JournalReplicationError):
        reopened.recovery_projection("wf-event-tamper")
    source.close()
    reopened.close()


def test_checkpoint_wire_form_is_json_stable(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-5", 1)
    checkpoint = build_checkpoint(
        source, source_node_id="node-a", stream_id="journal-1", workflow_id="wf-5"
    )
    restored = JournalCheckpoint.from_dict(json.loads(json.dumps(checkpoint.to_dict())))
    assert restored == checkpoint
    source.close()


def test_tcp_transport_persists_checkpoint_across_replica_restart(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-tcp", 1)
    _append(source, "wf-tcp", 2)
    checkpoint = build_checkpoint(
        source,
        source_node_id="node-a",
        stream_id="journal-1",
        workflow_id="wf-tcp",
    )
    replica_path = str(tmp_path / "replica.sqlite3")
    secret = b"journal-replication-test-secret-32b"
    replica = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    server = JournalReplicationTcpServer(replica, secret=secret)
    host, port = server.start()
    try:
        result = send_checkpoint(host, port, checkpoint, secret=secret)
        assert result["durable_sequence"] == 2
        assert send_checkpoint(host, port, checkpoint, secret=secret)["idempotent"] is True
    finally:
        server.close()
        replica.close()

    reopened = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    assert reopened.recovery_status("wf-tcp")["verified"] is True
    assert reopened.decide_recovery("wf-tcp").action == "retry"
    reopened.close()
    source.close()


def test_tcp_transport_rejects_wrong_secret_without_replica_write(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-auth", 1)
    checkpoint = build_checkpoint(
        source,
        source_node_id="node-a",
        stream_id="journal-1",
        workflow_id="wf-auth",
    )
    replica = DurableJournalReplica(
        str(tmp_path / "replica.sqlite3"),
        source_node_id="node-a",
        stream_id="journal-1",
    )
    server = JournalReplicationTcpServer(
        replica, secret=b"journal-replication-test-secret-32b"
    )
    host, port = server.start()
    try:
        with pytest.raises(JournalReplicationTransportError):
            send_checkpoint(
                host,
                port,
                checkpoint,
                secret=b"different-replication-test-secret",
            )
        assert replica.recovery_status("wf-auth")["verified"] is False
    finally:
        server.close()
        replica.close()
        source.close()


def test_tcp_transport_rejects_short_secret(tmp_path: Path):
    replica = DurableJournalReplica(
        str(tmp_path / "replica.sqlite3"),
        source_node_id="node-a",
        stream_id="journal-1",
    )
    with pytest.raises(JournalReplicationTransportError):
        JournalReplicationTcpServer(replica, secret=b"short")
    replica.close()


def test_cross_process_tcp_replication_survives_receiver_process_death(tmp_path: Path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-process", 1)
    _append(source, "wf-process", 2)
    checkpoint = build_checkpoint(
        source,
        source_node_id="node-a",
        stream_id="journal-1",
        workflow_id="wf-process",
    )
    replica_path = str(tmp_path / "receiver.sqlite3")
    secret = b"process-replication-secret-material-32b"
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    stop_event = context.Event()
    process = context.Process(
        target=_serve_replica_child,
        args=(replica_path, ready_queue, stop_event, secret),
    )
    process.start()
    try:
        host, port = ready_queue.get(timeout=15)
        ack = send_checkpoint(host, port, checkpoint, secret=secret)
        assert ack["durable_sequence"] == 2
    finally:
        process.terminate()
        process.join(timeout=10)
        ready_queue.close()
        ready_queue.join_thread()
    assert process.exitcode is not None

    reopened = DurableJournalReplica(
        replica_path, source_node_id="node-a", stream_id="journal-1"
    )
    assert reopened.recovery_status("wf-process")["verified"] is True
    assert reopened.decide_recovery("wf-process").action == "retry"
    reopened.close()
    source.close()
