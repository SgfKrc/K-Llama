"""Durable, transport-neutral replication of task-journal checkpoints.

This module provides explicit checkpoint replication and recovery for the
statically configured cluster. It gives a configured donor a durable replica
store with a tamper-evident event chain, atomic batch application, idempotent
replay, and a recovery gate. Hashes detect inconsistency but do not authenticate
a source; the companion transport binds the source identity to a shared HMAC
secret. The replica never exposes an unverified or partial checkpoint as
recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from typing import Any, Mapping, Protocol


REPLICATION_SCHEMA_VERSION = "qlh.journal.replication.v1"
_GENESIS_DIGEST = "0" * 64


class JournalReplicationError(ValueError):
    """Base error for replication contract violations."""


class JournalReplicationConflict(JournalReplicationError):
    """Raised when a checkpoint forks or conflicts with durable state."""


class JournalReplicationGap(JournalReplicationConflict):
    """Raised when a checkpoint does not continue the durable event chain."""


class ReplicationJournal(Protocol):
    def get_snapshot(self, workflow_id: str) -> dict[str, Any] | None: ...

    def list_events(self, workflow_id: str) -> list[dict[str, Any]]: ...


def _canonical_json(value: Any, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise JournalReplicationError(f"{label} is not JSON serializable") from exc


def _event_digest(previous_digest: str, event: Mapping[str, Any]) -> str:
    encoded = _canonical_json(dict(event), "journal event")
    return hashlib.sha256(
        (previous_digest + "\n" + encoded).encode("utf-8")
    ).hexdigest()


def _checkpoint_digest(
    source_node_id: str,
    stream_id: str,
    workflow_id: str,
    event_chain_head: str,
    snapshot: Mapping[str, Any],
) -> str:
    encoded = _canonical_json(
        {
            "source_node_id": source_node_id,
            "stream_id": stream_id,
            "workflow_id": workflow_id,
            "event_chain_head": event_chain_head,
            "snapshot": dict(snapshot),
        },
        "journal checkpoint",
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_event_sequence(
    events: list[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], str]:
    if not events:
        raise JournalReplicationError("journal checkpoint must contain events")
    workflow_id = snapshot.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id:
        raise JournalReplicationError("snapshot workflow_id is required")
    raw_last_sequence = snapshot.get("last_sequence", 0)
    if isinstance(raw_last_sequence, bool) or not isinstance(raw_last_sequence, int):
        raise JournalReplicationError("snapshot last_sequence must be an integer")
    last_sequence = raw_last_sequence
    if last_sequence <= 0 or last_sequence != len(events):
        raise JournalReplicationGap(
            "checkpoint events must cover every sequence through snapshot last_sequence"
        )

    previous_digest = _GENESIS_DIGEST
    records: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for expected_sequence, raw_event in enumerate(events, start=1):
        if not isinstance(raw_event, Mapping):
            raise JournalReplicationError("checkpoint event must be an object")
        event = dict(raw_event)
        if event.get("workflow_id") != workflow_id:
            raise JournalReplicationConflict("event workflow_id does not match snapshot")
        raw_sequence = event.get("sequence", 0)
        if isinstance(raw_sequence, bool) or not isinstance(raw_sequence, int):
            raise JournalReplicationError("checkpoint event sequence must be an integer")
        sequence = raw_sequence
        if sequence != expected_sequence:
            raise JournalReplicationGap(
                f"checkpoint expected sequence {expected_sequence}, got {sequence}"
            )
        for field in ("event_id", "entity_type", "entity_id", "event_type"):
            if not isinstance(event.get(field), str) or not event[field]:
                raise JournalReplicationError(f"checkpoint event {field} is required")
        if event["event_id"] in event_ids:
            raise JournalReplicationConflict("checkpoint contains duplicate event_id")
        event_ids.add(event["event_id"])
        occurred_at = event.get("occurred_at")
        if (
            isinstance(occurred_at, bool)
            or not isinstance(occurred_at, (int, float))
            or not math.isfinite(float(occurred_at))
        ):
            raise JournalReplicationError("checkpoint event occurred_at must be finite")
        if not isinstance(event.get("payload"), Mapping):
            raise JournalReplicationError("checkpoint event payload must be an object")
        event_digest = _event_digest(previous_digest, event)
        records.append(
            {
                "event": event,
                "event_digest": event_digest,
                "previous_digest": previous_digest,
            }
        )
        previous_digest = event_digest
    return tuple(records), previous_digest


@dataclass(frozen=True)
class JournalCheckpoint:
    source_node_id: str
    stream_id: str
    workflow_id: str
    created_at: float
    snapshot: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    event_chain_head: str
    checkpoint_digest: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.source_node_id, self.stream_id, self.workflow_id)
        ):
            raise JournalReplicationError("checkpoint identity fields are required")
        if not math.isfinite(float(self.created_at)):
            raise JournalReplicationError("checkpoint created_at must be finite")
        if self.snapshot.get("workflow_id") != self.workflow_id:
            raise JournalReplicationConflict("checkpoint workflow_id does not match snapshot")
        try:
            events = [record["event"] for record in self.records]
        except (KeyError, TypeError) as exc:
            raise JournalReplicationError("checkpoint record event is required") from exc
        normalized_records, chain_head = _validate_event_sequence(events, self.snapshot)
        if normalized_records != self.records or chain_head != self.event_chain_head:
            raise JournalReplicationConflict("checkpoint event digest chain is invalid")
        expected_digest = _checkpoint_digest(
            self.source_node_id,
            self.stream_id,
            self.workflow_id,
            self.event_chain_head,
            self.snapshot,
        )
        if expected_digest != self.checkpoint_digest:
            raise JournalReplicationConflict("checkpoint digest is invalid")

    @property
    def last_sequence(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLICATION_SCHEMA_VERSION,
            "source_node_id": self.source_node_id,
            "stream_id": self.stream_id,
            "workflow_id": self.workflow_id,
            "created_at": self.created_at,
            "snapshot": self.snapshot,
            "records": list(self.records),
            "event_chain_head": self.event_chain_head,
            "checkpoint_digest": self.checkpoint_digest,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "JournalCheckpoint":
        if not isinstance(raw, Mapping) or raw.get("schema_version") != REPLICATION_SCHEMA_VERSION:
            raise JournalReplicationError("unsupported journal checkpoint schema")
        snapshot = raw.get("snapshot")
        raw_records = raw.get("records")
        if not isinstance(snapshot, Mapping) or not isinstance(raw_records, list):
            raise JournalReplicationError("checkpoint snapshot and records are required")
        identity = tuple(raw.get(key) for key in ("source_node_id", "stream_id", "workflow_id"))
        if any(not isinstance(value, str) or not value for value in identity):
            raise JournalReplicationError("checkpoint identity fields must be non-empty strings")
        raw_created_at = raw.get("created_at", 0.0)
        if isinstance(raw_created_at, bool) or not isinstance(raw_created_at, (int, float)):
            raise JournalReplicationError("checkpoint created_at must be numeric")
        records: list[dict[str, Any]] = []
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise JournalReplicationError("checkpoint record must be an object")
            event = raw_record.get("event")
            if not isinstance(event, Mapping):
                raise JournalReplicationError("checkpoint record event is required")
            event_digest = raw_record.get("event_digest")
            previous_digest = raw_record.get("previous_digest")
            if not isinstance(event_digest, str) or not isinstance(previous_digest, str):
                raise JournalReplicationError("checkpoint record digests must be strings")
            records.append(
                {
                    "event": dict(event),
                    "event_digest": event_digest,
                    "previous_digest": previous_digest,
                }
            )
        created_at = float(raw_created_at)
        chain_head = raw.get("event_chain_head")
        checkpoint_digest = raw.get("checkpoint_digest")
        if not isinstance(chain_head, str) or not isinstance(checkpoint_digest, str):
            raise JournalReplicationError("checkpoint digests must be strings")
        return cls(
            source_node_id=identity[0],
            stream_id=identity[1],
            workflow_id=identity[2],
            created_at=created_at,
            snapshot=dict(snapshot),
            records=tuple(records),
            event_chain_head=chain_head,
            checkpoint_digest=checkpoint_digest,
        )


def build_checkpoint(
    journal: ReplicationJournal,
    *,
    source_node_id: str,
    stream_id: str,
    workflow_id: str,
    created_at: float | None = None,
) -> JournalCheckpoint:
    """Take a consistent enough journal projection for transport.

    SQLiteTaskJournal returns the event list and snapshot from durable storage.
    The replica revalidates the relationship, so a concurrent source update is
    rejected rather than silently advertised as a complete checkpoint.
    """

    if any(
        not isinstance(value, str) or not value
        for value in (source_node_id, stream_id, workflow_id)
    ):
        raise JournalReplicationError("source, stream, and workflow identities are required")
    snapshot = journal.get_snapshot(workflow_id)
    if not isinstance(snapshot, Mapping):
        raise JournalReplicationError(f"workflow snapshot not found: {workflow_id}")
    raw_events = journal.list_events(workflow_id)
    records, chain_head = _validate_event_sequence(raw_events, snapshot)
    created = time.time() if created_at is None else float(created_at)
    snapshot_copy = dict(snapshot)
    checkpoint_digest = _checkpoint_digest(
        source_node_id, stream_id, workflow_id, chain_head, snapshot_copy
    )
    return JournalCheckpoint(
        source_node_id=source_node_id,
        stream_id=stream_id,
        workflow_id=workflow_id,
        created_at=created,
        snapshot=snapshot_copy,
        records=records,
        event_chain_head=chain_head,
        checkpoint_digest=checkpoint_digest,
    )


class DurableJournalReplica:
    """SQLite WAL follower for one source/stream journal replication contract."""

    def __init__(
        self,
        path: str,
        *,
        source_node_id: str,
        stream_id: str,
        busy_timeout_ms: int = 5000,
    ) -> None:
        if not source_node_id or not stream_id:
            raise JournalReplicationError("source_node_id and stream_id are required")
        self.path = os.path.abspath(os.path.expanduser(path))
        self.source_node_id = source_node_id
        self.stream_id = stream_id
        self._busy_timeout_ms = max(100, int(busy_timeout_ms))
        self._lock = threading.RLock()
        self._closed = False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise JournalReplicationError("journal replica is closed")
        connection = sqlite3.connect(
            self.path,
            timeout=self._busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
                if mode is None or str(mode[0]).lower() != "wal":
                    raise JournalReplicationError("journal replica requires SQLite WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS replication_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS replicated_events (
                        workflow_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL CHECK (sequence > 0),
                        event_json TEXT NOT NULL,
                        event_digest TEXT NOT NULL,
                        previous_digest TEXT NOT NULL,
                        received_at REAL NOT NULL,
                        PRIMARY KEY (workflow_id, sequence),
                        UNIQUE (workflow_id, event_digest)
                    );
                    CREATE TABLE IF NOT EXISTS replicated_snapshots (
                        workflow_id TEXT PRIMARY KEY,
                        last_sequence INTEGER NOT NULL CHECK (last_sequence > 0),
                        event_chain_head TEXT NOT NULL,
                        checkpoint_digest TEXT NOT NULL,
                        snapshot_json TEXT NOT NULL,
                        received_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_replicated_events_workflow
                    ON replicated_events(workflow_id, sequence);
                    """
                )
                metadata = {
                    "schema_version": REPLICATION_SCHEMA_VERSION,
                    "source_node_id": self.source_node_id,
                    "stream_id": self.stream_id,
                }
                for key, value in metadata.items():
                    row = connection.execute(
                        "SELECT value FROM replication_metadata WHERE key = ?", (key,)
                    ).fetchone()
                    if row is not None and str(row[0]) != value:
                        raise JournalReplicationConflict(
                            f"replica metadata conflict for {key}"
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO replication_metadata(key, value) VALUES (?, ?)",
                        (key, value),
                    )
            finally:
                connection.close()

    def apply_checkpoint(
        self,
        checkpoint: JournalCheckpoint | Mapping[str, Any],
        *,
        received_at: float | None = None,
        permission_check=None,
    ) -> dict[str, Any]:
        """Atomically apply a checkpoint and return its durable watermark."""

        if isinstance(checkpoint, JournalCheckpoint):
            checkpoint = JournalCheckpoint.from_dict(checkpoint.to_dict())
        else:
            checkpoint = JournalCheckpoint.from_dict(checkpoint)
        if (
            checkpoint.source_node_id != self.source_node_id
            or checkpoint.stream_id != self.stream_id
        ):
            raise JournalReplicationConflict("checkpoint source or stream does not match replica")
        arrival = time.time() if received_at is None else float(received_at)
        if not math.isfinite(arrival):
            raise JournalReplicationError("received_at must be finite")
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                if permission_check is not None:
                    permission_check()
                existing_rows = connection.execute(
                    """
                    SELECT sequence, event_json, event_digest, previous_digest
                    FROM replicated_events WHERE workflow_id = ? ORDER BY sequence
                    """,
                    (checkpoint.workflow_id,),
                ).fetchall()
                if len(existing_rows) > checkpoint.last_sequence:
                    raise JournalReplicationGap("source checkpoint moved backwards")
                for row, record in zip(existing_rows, checkpoint.records):
                    if (
                        int(row["sequence"]) != int(record["event"]["sequence"])
                        or row["event_json"] != _canonical_json(record["event"], "journal event")
                        or row["event_digest"] != record["event_digest"]
                        or row["previous_digest"] != record["previous_digest"]
                    ):
                        raise JournalReplicationConflict(
                            f"replicated event fork at {checkpoint.workflow_id}:{row['sequence']}"
                        )
                if existing_rows and len(existing_rows) < checkpoint.last_sequence:
                    expected_previous = str(existing_rows[-1]["event_digest"])
                    if checkpoint.records[len(existing_rows)]["previous_digest"] != expected_previous:
                        raise JournalReplicationGap("checkpoint does not continue replica chain")

                for record in checkpoint.records[len(existing_rows):]:
                    event = record["event"]
                    connection.execute(
                        """
                        INSERT INTO replicated_events(
                            workflow_id, sequence, event_json, event_digest,
                            previous_digest, received_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            checkpoint.workflow_id,
                            int(event["sequence"]),
                            _canonical_json(event, "journal event"),
                            record["event_digest"],
                            record["previous_digest"],
                            arrival,
                        ),
                    )

                current = connection.execute(
                    """
                    SELECT last_sequence, event_chain_head, checkpoint_digest,
                           snapshot_json
                    FROM replicated_snapshots WHERE workflow_id = ?
                    """,
                    (checkpoint.workflow_id,),
                ).fetchone()
                snapshot_json = _canonical_json(checkpoint.snapshot, "workflow snapshot")
                current_sequence = 0 if current is None else int(current["last_sequence"])
                if current is not None and len(existing_rows) != current_sequence:
                    raise JournalReplicationGap("replica event count is behind its snapshot")
                if current is not None and current_sequence > checkpoint.last_sequence:
                    raise JournalReplicationGap("source checkpoint moved backwards")
                if current is not None and current_sequence == checkpoint.last_sequence:
                    if (
                        current["event_chain_head"] != checkpoint.event_chain_head
                        or current["checkpoint_digest"] != checkpoint.checkpoint_digest
                    ):
                        raise JournalReplicationConflict(
                            f"replicated snapshot fork at {checkpoint.workflow_id}"
                        )
                    idempotent = current["snapshot_json"] == snapshot_json
                elif current is None or current_sequence < checkpoint.last_sequence:
                    idempotent = False
                    if current_sequence > 0 and (
                        checkpoint.records[current_sequence - 1]["event_digest"]
                        != current["event_chain_head"]
                    ):
                        raise JournalReplicationGap("checkpoint does not continue snapshot chain")
                    if current is None:
                        connection.execute(
                            """
                            INSERT INTO replicated_snapshots(
                                workflow_id, last_sequence, event_chain_head,
                                checkpoint_digest, snapshot_json, received_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                checkpoint.workflow_id,
                                checkpoint.last_sequence,
                                checkpoint.event_chain_head,
                                checkpoint.checkpoint_digest,
                                snapshot_json,
                                arrival,
                            ),
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE replicated_snapshots
                            SET last_sequence = ?, event_chain_head = ?,
                                checkpoint_digest = ?, snapshot_json = ?, received_at = ?
                            WHERE workflow_id = ?
                            """,
                            (
                                checkpoint.last_sequence,
                                checkpoint.event_chain_head,
                                checkpoint.checkpoint_digest,
                                snapshot_json,
                                arrival,
                                checkpoint.workflow_id,
                            ),
                        )
                if permission_check is not None:
                    permission_check()
                connection.execute("COMMIT")
                return {
                    "accepted": True,
                    "idempotent": idempotent,
                    "source_node_id": self.source_node_id,
                    "stream_id": self.stream_id,
                    "workflow_id": checkpoint.workflow_id,
                    "durable_sequence": checkpoint.last_sequence,
                    "event_chain_head": checkpoint.event_chain_head,
                    "checkpoint_digest": checkpoint.checkpoint_digest,
                }
            except (JournalReplicationError, sqlite3.IntegrityError):
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            except sqlite3.Error as exc:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise JournalReplicationError(f"failed to apply journal checkpoint: {exc}") from exc
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def truncate_to(
        self,
        workflow_id: str,
        *,
        last_sequence: int,
        event_chain_head: str | None = None,
        received_at: float | None = None,
        permission_check=None,
    ) -> dict[str, Any]:
        """Drop the **uncommitted tail**, leaving the durable prefix ending at the committed point.

        为什么需要它（#93）：旧主可能在「还没有拿到多数耐久确认」时就把事件写到本地
        （= 未提交尾部）。新主一旦接管，若照原样继续服务，就会把**从未提交**的事件
        当成既成事实 ⇒ 违反一致性承诺。

        安全约束（不满足即拒绝，绝不"尽力而为"）：
        - ``last_sequence == 0`` 合法（截成空链）；
        - 否则本地链上**必须存在**该 sequence（哈希链保证该点唯一）⇒ 不允许凭空造点；
          ``event_chain_head`` 若传入则必须与该点摘要一致（口径 = 事件链摘要）。
        - 只删 ``sequence > last_sequence`` 的事件。快照行**不删**：它的内容对应更高的点、
          已不可再当可信前缀，因此记入 ``truncated:<workflow_id>`` 元数据留作取证。
        """
        sequence = int(last_sequence)
        if sequence < 0:
            raise JournalReplicationError("last_sequence must be >= 0")
        arrival = time.time() if received_at is None else float(received_at)
        if not math.isfinite(arrival):
            raise JournalReplicationError("received_at must be finite")
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                if permission_check is not None:
                    permission_check()
                if sequence > 0:
                    row = connection.execute(
                        "SELECT event_digest FROM replicated_events "
                        "WHERE workflow_id = ? AND sequence = ?",
                        (workflow_id, sequence),
                    ).fetchone()
                    if row is None:
                        raise JournalReplicationGap(
                            "cannot truncate to a point the replica does not durably hold"
                        )
                    if event_chain_head is not None and str(row["event_digest"]) != str(event_chain_head):
                        raise JournalReplicationConflict(
                            "truncate target digest does not match the durable event chain"
                        )
                removed = connection.execute(
                    "DELETE FROM replicated_events WHERE workflow_id = ? AND sequence > ?",
                    (workflow_id, sequence),
                ).rowcount
                snapshot = connection.execute(
                    "SELECT last_sequence FROM replicated_snapshots WHERE workflow_id = ?",
                    (workflow_id,),
                ).fetchone()
                # 快照仍指向被截断掉的更高点 ⇒ 它是一个**过期的**前缀，必须留痕。
                stale = snapshot is not None and int(snapshot["last_sequence"]) > sequence
                if stale:
                    connection.execute(
                        "INSERT INTO replication_metadata(key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (f"truncated:{workflow_id}", f"{sequence}@{int(arrival)}"),
                    )
                if permission_check is not None:
                    permission_check()
                connection.execute("COMMIT")
                return {
                    "accepted": True,
                    "workflow_id": workflow_id,
                    "durable_sequence": sequence,
                    "removed_events": max(0, int(removed or 0)),
                    "snapshot_stale": stale,
                    "source_node_id": self.source_node_id,
                    "stream_id": self.stream_id,
                }
            except (JournalReplicationError, sqlite3.IntegrityError):
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            except sqlite3.Error as exc:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise JournalReplicationError(f"failed to truncate journal replica: {exc}") from exc
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def durable_prefix(self, workflow_id: str) -> dict[str, Any]:
        """只读诊断：本地**事件链**现状（**不**做快照一致性校验）。

        截断后副本必然处于「链不完整」状态（`recovery_status.verified` 为 False），
        此时仍要能看到自己手里到底有什么，才能决定追赶起点。
        """
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS n, MIN(sequence) AS first_sequence,
                           MAX(sequence) AS last_sequence
                    FROM replicated_events WHERE workflow_id = ?
                    """,
                    (workflow_id,),
                ).fetchone()
                head = connection.execute(
                    "SELECT event_digest FROM replicated_events "
                    "WHERE workflow_id = ? ORDER BY sequence DESC LIMIT 1",
                    (workflow_id,),
                ).fetchone()
            finally:
                connection.close()
        count = int(row["n"] or 0) if row is not None else 0
        return {
            "workflow_id": workflow_id,
            "event_count": count,
            "first_sequence": None if count == 0 else int(row["first_sequence"]),
            "last_sequence": 0 if count == 0 else int(row["last_sequence"]),
            "event_chain_head": "" if head is None else str(head["event_digest"]),
        }

    def truncated_watermark(self, workflow_id: str) -> str:
        """Return the recorded truncation marker for a workflow, if any (diagnostics)."""
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT value FROM replication_metadata WHERE key = ?",
                    (f"truncated:{workflow_id}",),
                ).fetchone()
            finally:
                connection.close()
        return "" if row is None else str(row["value"])

    def recovery_status(self, workflow_id: str) -> dict[str, Any]:
        with self._lock:
            connection = self._connect()
            try:
                snapshot = connection.execute(
                    """
                    SELECT last_sequence, event_chain_head, checkpoint_digest,
                           snapshot_json
                    FROM replicated_snapshots WHERE workflow_id = ?
                    """,
                    (workflow_id,),
                ).fetchone()
                count = connection.execute(
                    "SELECT COUNT(*) FROM replicated_events WHERE workflow_id = ?",
                    (workflow_id,),
                ).fetchone()
            finally:
                connection.close()
        if snapshot is None:
            return {
                "verified": False,
                "durable": False,
                "source_node_id": self.source_node_id,
                "stream_id": self.stream_id,
                "workflow_id": workflow_id,
                "reason": "workflow_not_replicated",
            }
        durable_sequence = int(snapshot["last_sequence"])
        verified = int(count[0]) == durable_sequence
        snapshot_value: dict[str, Any] | None = None
        if verified:
            try:
                parsed_snapshot = json.loads(snapshot["snapshot_json"])
            except (TypeError, json.JSONDecodeError):
                verified = False
            else:
                if not isinstance(parsed_snapshot, dict):
                    verified = False
                else:
                    snapshot_value = parsed_snapshot
                    raw_sequence = parsed_snapshot.get("last_sequence", 0)
                    verified = (
                        parsed_snapshot.get("workflow_id") == workflow_id
                        and not isinstance(raw_sequence, bool)
                        and isinstance(raw_sequence, int)
                        and raw_sequence == durable_sequence
                    )
        if verified:
            with self._lock:
                connection = self._connect()
                try:
                    rows = connection.execute(
                        """
                        SELECT sequence, event_json, event_digest, previous_digest
                        FROM replicated_events WHERE workflow_id = ? ORDER BY sequence
                        """,
                        (workflow_id,),
                    ).fetchall()
                finally:
                    connection.close()
            previous_digest = _GENESIS_DIGEST
            for expected_sequence, row in enumerate(rows, start=1):
                try:
                    event = json.loads(row["event_json"])
                except (TypeError, json.JSONDecodeError):
                    verified = False
                    break
                try:
                    raw_event_sequence = event.get("sequence", 0)
                    row_sequence = int(row["sequence"])
                except (TypeError, ValueError):
                    verified = False
                    break
                if (
                    isinstance(raw_event_sequence, bool)
                    or not isinstance(raw_event_sequence, int)
                    or event.get("workflow_id") != workflow_id
                    or raw_event_sequence != expected_sequence
                    or row_sequence != expected_sequence
                    or row["previous_digest"] != previous_digest
                    or row["event_digest"] != _event_digest(previous_digest, event)
                ):
                    verified = False
                    break
                previous_digest = row["event_digest"]
            verified = (
                verified
                and previous_digest == snapshot["event_chain_head"]
                and snapshot_value is not None
                and _checkpoint_digest(
                    self.source_node_id,
                    self.stream_id,
                    workflow_id,
                    snapshot["event_chain_head"],
                    snapshot_value,
                )
                == snapshot["checkpoint_digest"]
            )
        return {
            "verified": verified,
            "durable": verified,
            "source_node_id": self.source_node_id,
            "stream_id": self.stream_id,
            "workflow_id": workflow_id,
            "durable_sequence": durable_sequence,
            "event_chain_head": snapshot["event_chain_head"],
            "checkpoint_digest": snapshot["checkpoint_digest"],
            "reason": "verified" if verified else "replica_chain_verification_failed",
        }

    def export_checkpoint(
        self, workflow_id: str, *, created_at: float | None = None,
    ) -> JournalCheckpoint:
        """把一个**已自洽**的 workflow 导出成 checkpoint（供落后的对端追赶）。

        与 `recovery_status().verified` 同一判据：不自洽（例如刚截断过尾部、快照与事件链
        对不上）就**拒绝导出** —— 否则等于把一段没人担保的前缀当成「已提交」发出去。
        """
        status = self.recovery_status(workflow_id)
        if not status.get("verified"):
            raise JournalReplicationError(
                f"workflow is not durably replicated: {status.get('reason', 'unverified')}"
            )
        with self._lock:
            connection = self._connect()
            try:
                snapshot_row = connection.execute(
                    "SELECT last_sequence, event_chain_head, checkpoint_digest, snapshot_json "
                    "FROM replicated_snapshots WHERE workflow_id = ?",
                    (workflow_id,),
                ).fetchone()
                event_rows = connection.execute(
                    "SELECT sequence, event_json, event_digest, previous_digest "
                    "FROM replicated_events WHERE workflow_id = ? ORDER BY sequence",
                    (workflow_id,),
                ).fetchall()
            finally:
                connection.close()
        if snapshot_row is None:
            raise JournalReplicationError(f"workflow snapshot not found: {workflow_id}")
        records: list[dict[str, Any]] = []
        for row in event_rows:
            try:
                event = json.loads(row["event_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise JournalReplicationError("replicated event is not valid JSON") from exc
            records.append({
                "event": event,
                "event_digest": str(row["event_digest"]),
                "previous_digest": str(row["previous_digest"]),
            })
        try:
            snapshot = json.loads(snapshot_row["snapshot_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise JournalReplicationError("replicated snapshot is not valid JSON") from exc
        # `JournalCheckpoint` 的构造会**重新校验**整条摘要链与 checkpoint_digest：
        # 万一库被改坏，这里就会抛，而不是把坏前缀发出去。
        return JournalCheckpoint(
            source_node_id=self.source_node_id,
            stream_id=self.stream_id,
            workflow_id=workflow_id,
            created_at=time.time() if created_at is None else float(created_at),
            snapshot=snapshot,
            # ⚠️ 必须是 **tuple**：`JournalCheckpoint.__post_init__` 拿
            #    `_validate_event_sequence()`（返回 tuple）与传入值做**结构比较**，
            #    传 list 会永远不相等，报「checkpoint event digest chain is invalid」（踩过）。
            records=tuple(records),
            event_chain_head=str(snapshot_row["event_chain_head"]),
            checkpoint_digest=str(snapshot_row["checkpoint_digest"]),
        )

    def recovery_projection(self, workflow_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        status = self.recovery_status(workflow_id)
        if not status.get("verified"):
            raise JournalReplicationError(
                f"workflow is not durably replicated: {status.get('reason', 'unverified')}"
            )
        with self._lock:
            connection = self._connect()
            try:
                snapshot_row = connection.execute(
                    "SELECT snapshot_json FROM replicated_snapshots WHERE workflow_id = ?",
                    (workflow_id,),
                ).fetchone()
                event_rows = connection.execute(
                    """
                    SELECT event_json FROM replicated_events
                    WHERE workflow_id = ? ORDER BY sequence
                    """,
                    (workflow_id,),
                ).fetchall()
            finally:
                connection.close()
        if snapshot_row is None:
            raise JournalReplicationError("replicated snapshot disappeared")
        return (
            json.loads(snapshot_row["snapshot_json"]),
            [json.loads(row["event_json"]) for row in event_rows],
            status,
        )

    def decide_recovery(self, workflow_id: str):
        """Run the normal recovery policy only after durable replication is verified."""

        from cluster_recovery import decide_recovery

        snapshot, events, _status = self.recovery_projection(workflow_id)
        return decide_recovery(snapshot, events)

    def health(self) -> dict[str, Any]:
        status = self.recovery_status("__health_probe__")
        return {
            "enabled": True,
            "available": not self._closed,
            "backend": "sqlite",
            "path": self.path,
            "schema_version": REPLICATION_SCHEMA_VERSION,
            "source_node_id": self.source_node_id,
            "stream_id": self.stream_id,
            "probe_reason": status["reason"],
        }

    def close(self) -> None:
        with self._lock:
            self._closed = True


__all__ = [
    "DurableJournalReplica",
    "JournalCheckpoint",
    "JournalReplicationConflict",
    "JournalReplicationError",
    "JournalReplicationGap",
    "REPLICATION_SCHEMA_VERSION",
    "build_checkpoint",
]
