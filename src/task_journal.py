"""Durable append-only journal for task-graph workflow metadata."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol


SCHEMA_VERSION = 1
TERMINAL_WORKFLOW_STATES = ("completed", "failed", "cancelled")


class TaskJournalError(RuntimeError):
    """Base error for durable task-journal operations."""


class TaskJournalConflict(TaskJournalError):
    """Raised when an event violates idempotency or sequence ordering."""


class TaskJournalInUse(TaskJournalError):
    """Raised when another process owns the same task-journal instance."""


@dataclass(frozen=True)
class JournalEvent:
    event_id: str
    workflow_id: str
    sequence: int
    entity_type: str
    entity_id: str
    event_type: str
    occurred_at: float
    payload: dict[str, Any]


class TaskJournal(Protocol):
    def append_event(self, event: JournalEvent, snapshot: dict) -> bool: ...
    def get_snapshot(self, workflow_id: str) -> Optional[dict]: ...
    def list_snapshots(self, limit: int = 20) -> list[dict]: ...
    def list_events(self, workflow_id: str) -> list[dict]: ...
    def list_nonterminal_snapshots(self, limit: int = 100) -> list[dict]: ...
    def cleanup_terminal(
        self,
        *,
        max_age_seconds: float = 0.0,
        max_records: int = 0,
        now: Optional[float] = None,
    ) -> dict: ...
    def health(self) -> dict: ...
    def close(self) -> None: ...


class SQLiteTaskJournal:
    """SQLite event journal with an atomically updated snapshot projection.

    锁所有权合约：实例化即获取跨进程独占锁（``_acquire_instance_lock``），直到
    ``close()`` 才释放。因此只有**真正承担任务图写服务**的进程才允许在确定要
    成为 writer 的时机实例化它（首个真实任务链执行、工作流查询、启动恢复，
    或运行时切换启用时）；宿主模块的 **import 副作用**不得实例化——否则任何仅
    import 宿主模块的探针 / pytest 冷启动子进程 / Windows ``multiprocessing``
    spawn worker 都会与生产 writer 争锁，并被错误判定为
    「任务图 journal 初始化失败，任务图已禁用」。
    """

    def __init__(
        self,
        path: str,
        busy_timeout_ms: int = 5000,
        acquire_instance_lock: bool = True,
    ):
        resolved_path = os.path.abspath(os.path.expanduser(path))
        if not resolved_path:
            raise TaskJournalError("task journal path must not be empty")
        self.path = resolved_path
        self._busy_timeout_ms = max(100, int(busy_timeout_ms))
        self._lock = threading.RLock()
        self._lock_file = None
        self._closed = False
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            if acquire_instance_lock:
                self._acquire_instance_lock()
            self._initialize()
        except TaskJournalError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise TaskJournalError(
                f"failed to initialize task journal: {exc}"
            ) from exc

    def _acquire_instance_lock(self) -> None:
        lock_path = self.path + ".lock"
        lock_file = open(lock_path, "a+b")
        try:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except (OSError, IOError) as exc:
            lock_file.close()
            raise TaskJournalInUse(
                "task journal is already owned by another process"
            ) from exc
        self._lock_file = lock_file

    def _release_instance_lock(self) -> None:
        lock_file = self._lock_file
        self._lock_file = None
        if lock_file is None:
            return
        try:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError):
            pass
        finally:
            lock_file.close()

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise TaskJournalError("task journal is closed")
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self._busy_timeout_ms / 1000.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA synchronous = FULL")
            return connection
        except sqlite3.Error as exc:
            raise TaskJournalError(f"failed to open task journal: {exc}") from exc

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
                if mode is None or str(mode[0]).lower() != "wal":
                    raise TaskJournalError(
                        "task journal requires SQLite WAL mode on a local filesystem"
                    )
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS journal_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS workflow_events (
                        event_id TEXT PRIMARY KEY,
                        workflow_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL CHECK (sequence > 0),
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at REAL NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE (workflow_id, sequence)
                    );

                    CREATE INDEX IF NOT EXISTS idx_workflow_events_lookup
                    ON workflow_events (workflow_id, sequence);

                    CREATE TABLE IF NOT EXISTS workflow_snapshots (
                        workflow_id TEXT PRIMARY KEY,
                        last_sequence INTEGER NOT NULL CHECK (last_sequence > 0),
                        state TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        state_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_workflow_snapshots_updated
                    ON workflow_snapshots (updated_at DESC);
                    """
                )
                connection.execute(
                    """
                    INSERT INTO journal_metadata(key, value)
                    VALUES ('schema_version', ?)
                    ON CONFLICT(key) DO NOTHING
                    """,
                    (str(SCHEMA_VERSION),),
                )
                row = connection.execute(
                    "SELECT value FROM journal_metadata WHERE key = 'schema_version'"
                ).fetchone()
                if row is None or int(row["value"]) != SCHEMA_VERSION:
                    raise TaskJournalError(
                        "unsupported task journal schema version: "
                        f"{None if row is None else row['value']}"
                    )
            except sqlite3.Error as exc:
                raise TaskJournalError(
                    f"failed to create task journal schema: {exc}"
                ) from exc
            finally:
                connection.close()

    @staticmethod
    def _serialize(value: dict, label: str) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise TaskJournalError(f"{label} is not JSON serializable: {exc}") from exc

    @staticmethod
    def _event_row_matches(
        row: sqlite3.Row,
        event: JournalEvent,
        payload_json: str,
    ) -> bool:
        return (
            row["workflow_id"] == event.workflow_id
            and int(row["sequence"]) == event.sequence
            and row["entity_type"] == event.entity_type
            and row["entity_id"] == event.entity_id
            and row["event_type"] == event.event_type
            and float(row["occurred_at"]) == float(event.occurred_at)
            and row["payload_json"] == payload_json
        )

    def append_event(self, event: JournalEvent, snapshot: dict) -> bool:
        """Append one event and update its workflow snapshot in one transaction.

        Returns True for a new event and False for an exact idempotent replay.
        """
        if not event.event_id or not event.workflow_id:
            raise TaskJournalError("event_id and workflow_id must not be empty")
        if event.sequence <= 0:
            raise TaskJournalError("event sequence must be positive")
        if snapshot.get("workflow_id") != event.workflow_id:
            raise TaskJournalConflict("snapshot workflow_id does not match event")
        if int(snapshot.get("last_sequence", 0)) != event.sequence:
            raise TaskJournalConflict("snapshot sequence does not match event")

        payload_json = self._serialize(event.payload, "event payload")
        snapshot_json = self._serialize(snapshot, "workflow snapshot")
        state = str(snapshot.get("state", ""))
        created_at = float(snapshot.get("created_at", event.occurred_at))

        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    """
                    SELECT workflow_id, sequence, entity_type, entity_id,
                           event_type, occurred_at, payload_json
                    FROM workflow_events WHERE event_id = ?
                    """,
                    (event.event_id,),
                ).fetchone()
                if existing is not None:
                    if not self._event_row_matches(existing, event, payload_json):
                        raise TaskJournalConflict(
                            f"event_id has conflicting content: {event.event_id}"
                        )
                    existing_snapshot = connection.execute(
                        """
                        SELECT last_sequence, state_json
                        FROM workflow_snapshots WHERE workflow_id = ?
                        """,
                        (event.workflow_id,),
                    ).fetchone()
                    if (
                        existing_snapshot is None
                        or int(existing_snapshot["last_sequence"]) < event.sequence
                    ):
                        raise TaskJournalConflict(
                            f"event_id has conflicting snapshot: {event.event_id}"
                        )
                    if (
                        int(existing_snapshot["last_sequence"]) == event.sequence
                        and existing_snapshot["state_json"] != snapshot_json
                    ):
                        raise TaskJournalConflict(
                            f"event_id has conflicting snapshot: {event.event_id}"
                        )
                    connection.execute("COMMIT")
                    return False

                previous = connection.execute(
                    """
                    SELECT last_sequence FROM workflow_snapshots
                    WHERE workflow_id = ?
                    """,
                    (event.workflow_id,),
                ).fetchone()
                expected_sequence = (
                    int(previous["last_sequence"]) + 1
                    if previous is not None else 1
                )
                if event.sequence != expected_sequence:
                    raise TaskJournalConflict(
                        f"workflow {event.workflow_id} expected sequence "
                        f"{expected_sequence}, got {event.sequence}"
                    )

                connection.execute(
                    """
                    INSERT INTO workflow_events(
                        event_id, workflow_id, sequence, entity_type,
                        entity_id, event_type, occurred_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.workflow_id,
                        event.sequence,
                        event.entity_type,
                        event.entity_id,
                        event.event_type,
                        float(event.occurred_at),
                        payload_json,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO workflow_snapshots(
                        workflow_id, last_sequence, state, created_at,
                        updated_at, state_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workflow_id) DO UPDATE SET
                        last_sequence = excluded.last_sequence,
                        state = excluded.state,
                        updated_at = excluded.updated_at,
                        state_json = excluded.state_json
                    """,
                    (
                        event.workflow_id,
                        event.sequence,
                        state,
                        created_at,
                        float(event.occurred_at),
                        snapshot_json,
                    ),
                )
                connection.execute("COMMIT")
                return True
            except TaskJournalConflict:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            except sqlite3.IntegrityError as exc:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise TaskJournalConflict(
                    f"task journal constraint conflict: {exc}"
                ) from exc
            except sqlite3.Error as exc:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise TaskJournalError(
                    f"failed to append task journal event: {exc}"
                ) from exc
            finally:
                connection.close()

    def get_snapshot(self, workflow_id: str) -> Optional[dict]:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT state_json FROM workflow_snapshots
                    WHERE workflow_id = ?
                    """,
                    (workflow_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise TaskJournalError(
                    f"failed to read task journal snapshot: {exc}"
                ) from exc
            finally:
                connection.close()
        if row is None:
            return None
        try:
            return json.loads(row["state_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise TaskJournalError(
                f"invalid task journal snapshot for {workflow_id}: {exc}"
            ) from exc

    def list_snapshots(self, limit: int = 20) -> list[dict]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT state_json FROM workflow_snapshots
                    ORDER BY created_at DESC, workflow_id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise TaskJournalError(
                    f"failed to list task journal snapshots: {exc}"
                ) from exc
            finally:
                connection.close()
        try:
            return [json.loads(row["state_json"]) for row in rows]
        except (TypeError, json.JSONDecodeError) as exc:
            raise TaskJournalError(f"invalid task journal snapshot: {exc}") from exc

    def list_events(self, workflow_id: str) -> list[dict]:
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT event_id, workflow_id, sequence, entity_type,
                           entity_id, event_type, occurred_at, payload_json
                    FROM workflow_events
                    WHERE workflow_id = ?
                    ORDER BY sequence ASC
                    """,
                    (workflow_id,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise TaskJournalError(
                    f"failed to list task journal events: {exc}"
                ) from exc
            finally:
                connection.close()
        try:
            return [
                {
                    "event_id": row["event_id"],
                    "workflow_id": row["workflow_id"],
                    "sequence": int(row["sequence"]),
                    "entity_type": row["entity_type"],
                    "entity_id": row["entity_id"],
                    "event_type": row["event_type"],
                    "occurred_at": float(row["occurred_at"]),
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            ]
        except (TypeError, json.JSONDecodeError) as exc:
            raise TaskJournalError(f"invalid task journal event: {exc}") from exc

    def list_nonterminal_snapshots(self, limit: int = 100) -> list[dict]:
        safe_limit = max(1, min(int(limit), 1000))
        placeholders = ",".join("?" for _ in TERMINAL_WORKFLOW_STATES)
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    f"""
                    SELECT state_json FROM workflow_snapshots
                    WHERE state NOT IN ({placeholders})
                    ORDER BY updated_at ASC, workflow_id ASC
                    LIMIT ?
                    """,
                    (*TERMINAL_WORKFLOW_STATES, safe_limit),
                ).fetchall()
            except sqlite3.Error as exc:
                raise TaskJournalError(
                    f"failed to list recovery candidates: {exc}"
                ) from exc
            finally:
                connection.close()
        try:
            return [json.loads(row["state_json"]) for row in rows]
        except (TypeError, json.JSONDecodeError) as exc:
            raise TaskJournalError(
                f"invalid recovery candidate snapshot: {exc}"
            ) from exc

    def cleanup_terminal(
        self,
        *,
        max_age_seconds: float = 0.0,
        max_records: int = 0,
        now: Optional[float] = None,
    ) -> dict:
        """Delete terminal workflows selected by age or retained-record limit."""
        safe_age = max(0.0, float(max_age_seconds))
        safe_records = max(0, int(max_records))
        cleanup_time = time.time() if now is None else float(now)
        if safe_age <= 0 and safe_records <= 0:
            return {
                "deleted_workflows": 0,
                "deleted_events": 0,
                "deleted_by_age": 0,
                "deleted_by_limit": 0,
                "remaining_terminal": self._count_terminal_snapshots(),
            }

        placeholders = ",".join("?" for _ in TERMINAL_WORKFLOW_STATES)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    f"""
                    SELECT workflow_id, updated_at
                    FROM workflow_snapshots
                    WHERE state IN ({placeholders})
                    ORDER BY updated_at DESC, workflow_id DESC
                    """,
                    TERMINAL_WORKFLOW_STATES,
                ).fetchall()
                by_age: set[str] = set()
                if safe_age > 0:
                    cutoff = cleanup_time - safe_age
                    by_age = {
                        str(row["workflow_id"])
                        for row in rows
                        if float(row["updated_at"]) < cutoff
                    }
                by_limit: set[str] = set()
                if safe_records > 0 and len(rows) > safe_records:
                    by_limit = {
                        str(row["workflow_id"])
                        for row in rows[safe_records:]
                    }
                delete_ids = sorted(by_age | by_limit)
                deleted_events = 0
                for offset in range(0, len(delete_ids), 500):
                    batch = delete_ids[offset:offset + 500]
                    delete_placeholders = ",".join("?" for _ in batch)
                    cursor = connection.execute(
                        f"""
                        DELETE FROM workflow_events
                        WHERE workflow_id IN ({delete_placeholders})
                        """,
                        batch,
                    )
                    deleted_events += max(0, int(cursor.rowcount))
                    connection.execute(
                        f"""
                        DELETE FROM workflow_snapshots
                        WHERE workflow_id IN ({delete_placeholders})
                        """,
                        batch,
                    )
                connection.execute("COMMIT")
                return {
                    "deleted_workflows": len(delete_ids),
                    "deleted_events": deleted_events,
                    "deleted_by_age": len(by_age),
                    "deleted_by_limit": len(by_limit),
                    "remaining_terminal": len(rows) - len(delete_ids),
                }
            except sqlite3.Error as exc:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise TaskJournalError(
                    f"failed to clean task journal: {exc}"
                ) from exc
            finally:
                connection.close()

    def _count_terminal_snapshots(self) -> int:
        placeholders = ",".join("?" for _ in TERMINAL_WORKFLOW_STATES)
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    f"""
                    SELECT COUNT(*) AS count FROM workflow_snapshots
                    WHERE state IN ({placeholders})
                    """,
                    TERMINAL_WORKFLOW_STATES,
                ).fetchone()
                return 0 if row is None else int(row["count"])
            except sqlite3.Error as exc:
                raise TaskJournalError(
                    f"failed to count terminal task snapshots: {exc}"
                ) from exc
            finally:
                connection.close()

    def health(self) -> dict:
        started_at = time.perf_counter()
        try:
            with self._lock:
                connection = self._connect()
                try:
                    row = connection.execute(
                        """
                        SELECT value FROM journal_metadata
                        WHERE key = 'schema_version'
                        """
                    ).fetchone()
                    mode = connection.execute("PRAGMA journal_mode").fetchone()
                finally:
                    connection.close()
            ok = row is not None and int(row[0]) == SCHEMA_VERSION
            return {
                "enabled": True,
                "available": ok,
                "backend": "sqlite",
                "path": self.path,
                "journal_mode": "" if mode is None else str(mode[0]).lower(),
                "schema_version": SCHEMA_VERSION,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
                "error": "" if ok else "sqlite schema metadata unavailable",
            }
        except (TaskJournalError, TypeError, ValueError) as exc:
            return {
                "enabled": True,
                "available": False,
                "backend": "sqlite",
                "path": self.path,
                "schema_version": SCHEMA_VERSION,
                "error": str(exc),
            }

    def close(self) -> None:
        """Release the cross-process ownership lock."""
        with self._lock:
            self._closed = True
            self._release_instance_lock()
