"""SQLite model control transactions: state, outbox, debts and audit commit together.

The complete model-control aggregate has its own recovery domain. Local commits
are explicitly local_sqlite; this adapter never upgrades them to quorum commits.
"""
from __future__ import annotations

import atexit
from copy import deepcopy
import json
import os
import sqlite3
import threading
import time
from qlh_core.contracts.deployment import MODEL_CONTROL_DOMAIN, canonical, ModelControlError
from qlh_core.adapters.sqlite_journal import SnapshotJournal
from qlh_core.adapters.model_control_retention import (
    guard_retired_records, initialize_retention, retain_model_control,
)


_PROFILE_PATH = os.environ.get("QLH_PROFILE_CONTROL_READS", "")
_PROFILE_RECORDS: list[tuple[float, str, int, str]] = []
_PROFILE_LOCK = threading.Lock()
_PROFILE_BATCH = 256


def _flush_profile_control_reads() -> None:
    if not _PROFILE_PATH or not _PROFILE_RECORDS:
        return
    try:
        with open(_PROFILE_PATH, "a", encoding="utf-8") as handle:
            handle.writelines(f"{stamp:.6f}\t{kind}\t{size}\t{thread}\n"
                              for stamp, kind, size, thread in _PROFILE_RECORDS)
        _PROFILE_RECORDS.clear()
    except OSError:
        pass


def _profile_control_read(kind: str, size: int) -> None:
    if not _PROFILE_PATH:
        return
    thread = threading.current_thread().name
    with _PROFILE_LOCK:
        _PROFILE_RECORDS.append((time.monotonic(), kind, size, thread))
        if len(_PROFILE_RECORDS) >= _PROFILE_BATCH:
            _flush_profile_control_reads()


atexit.register(_flush_profile_control_reads)


_READ_TRIGGERS = {
    f"model_control_read_{action.lower()}": (
        f"CREATE TRIGGER model_control_read_{action.lower()} AFTER {action} ON model_control_state "
        "BEGIN UPDATE model_control_read_revision SET generation=generation+1 WHERE id=1; END"
    ) for action in ("INSERT", "UPDATE", "DELETE")
}
_SUPPLY_QUERY_FIELDS = (
    "operation_id", "intent", "status", "reason_code", "reason", "created_at", "updated_at",
    "nodes", "plan", "source_progress", "capacity_diagnostic",
)


def empty_state():
    return {"schema_version": 1, "revision": 0, "boot_id": "", "selected_operation": "",
            "active_deployment": "", "operations": {}, "supplies": {}, "deployments": {},
            "outbox": {}, "cleanup": {}, "artifact_cleanup": {}, "lifecycle": None, "commit_scope": "local_sqlite"}


class SqliteModelControlRepository:
    def __init__(self, *, connect, transaction, max_events=64, max_event_bytes=8 * 1024 * 1024,
                 read_mode=None):
        self._connect = connect
        self._transaction = transaction
        mode = read_mode if read_mode is not None else os.environ.get("QLH_CONTROL_READ_MODE", "combined")
        if mode not in {"legacy", "execution", "query", "combined"}:
            raise ValueError("model_control_read_mode_invalid")
        self.execution_reads_enabled = mode in {"execution", "combined"}
        self.query_reads_enabled = mode in {"query", "combined"}
        self._read_lock = threading.RLock()
        self._read_token = None
        self._read_state = None
        self._read_schema = None
        self._journal = SnapshotJournal("model_control_events", domain=MODEL_CONTROL_DOMAIN,
            error=lambda: ModelControlError("model_control_journal_corrupt"),
            max_events=max_events, max_bytes=max_event_bytes)

    @staticmethod
    def _initialize(connection):
        connection.execute("CREATE TABLE IF NOT EXISTS model_control_state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS model_control_events (sequence INTEGER PRIMARY KEY, domain_id TEXT NOT NULL, previous_digest TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS model_control_read_revision (id INTEGER PRIMARY KEY CHECK(id=1), incarnation TEXT NOT NULL, generation INTEGER NOT NULL)")
        connection.execute("INSERT OR IGNORE INTO model_control_read_revision VALUES (1,hex(randomblob(16)),0)")
        for definition in _READ_TRIGGERS.values():
            connection.execute(definition.replace("CREATE TRIGGER ", "CREATE TRIGGER IF NOT EXISTS ", 1))

    @staticmethod
    def _read(connection):
        row = connection.execute("SELECT payload FROM model_control_state WHERE id=1").fetchone()
        payload = row["payload"] if row else ""
        _profile_control_read("row", len(payload))
        value = json.loads(payload) if row else empty_state()
        if value.get("schema_version") != 1:
            raise ModelControlError("model_control_schema_invalid")
        return value

    def read(self):
        # No schema creation or state repair during queries.
        _profile_control_read("connect", 0)
        connection = self._connect()
        try:
            present = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_control_state'").fetchone()
            return self._read(connection) if present else empty_state()
        finally:
            connection.close()

    def _committed_read_token(self, connection):
        database = next(row for row in connection.execute("PRAGMA database_list") if row["name"] == "main")
        if not database["file"]:
            return None
        filename = os.path.normcase(os.path.realpath(database["file"]))
        info = os.stat(filename)
        database_identity = (filename, info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)
        schema = connection.execute("PRAGMA schema_version").fetchone()[0]
        try:
            row = connection.execute("SELECT incarnation,generation FROM model_control_read_revision WHERE id=1").fetchone()
        except sqlite3.OperationalError as error:
            if "no such table: model_control_read_revision" not in str(error):
                raise
            return None
        if row is None:
            return None
        schema_key = (*database_identity, schema, row["incarnation"])
        if schema_key != self._read_schema:
            definitions = {item["name"]: item["sql"] for item in connection.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND tbl_name='model_control_state'")}
            if any(" ".join(definitions.get(name, "").split()) != " ".join(definition.split())
                   for name, definition in _READ_TRIGGERS.items()):
                return None
            self._read_schema = schema_key
        return (*schema_key, row["generation"])

    @staticmethod
    def _execution_view(state):
        key = state["active_deployment"]
        lifecycle = state.get("lifecycle")
        return {**{field: state[field] for field in (
                    "schema_version", "revision", "boot_id", "selected_operation", "active_deployment", "commit_scope")},
                "deployments": {key: state["deployments"][key]} if key in state["deployments"] else {},
                "lifecycle": {field: lifecycle[field] for field in ("total_layers", "local_assignment")
                              if field in lifecycle} if lifecycle else None}

    def _read_view(self, kind, operation_id=None):
        _profile_control_read("connect", 0)
        with self._read_lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN")
                token = self._committed_read_token(connection)
                if token is None:
                    self._read_token, self._read_state = None, None
                if token is not None and token == self._read_token:
                    state = self._read_state
                else:
                    present = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_control_state'").fetchone()
                    state = self._read(connection) if present else empty_state()
                    if token is not None:
                        self._read_state, self._read_token = state, token
                if kind == "stamp":
                    value = {field: state[field] for field in ("revision", "boot_id")}
                else:
                    value = self._execution_view(state)
                    if kind == "query":
                        selected = operation_id if operation_id is not None else state["selected_operation"]
                        value["operations"] = {selected: state["operations"][selected]} if selected in state["operations"] else {}
                        supply = state["supplies"].get(selected)
                        value["supplies"] = {selected: {field: supply[field] for field in _SUPPLY_QUERY_FIELDS
                                                       if field in supply}} if supply is not None else {}
                _profile_control_read("view:" + kind, 0)
                return deepcopy(value)
            finally:
                connection.rollback()
                connection.close()

    def read_execution(self):
        return self._read_view("execution") if self.execution_reads_enabled else self.read()

    def read_query(self, operation_id=None):
        return self._read_view("query", operation_id) if self.query_reads_enabled else self.read()

    def read_stamp(self):
        return self._read_view("stamp") if self.query_reads_enabled else self.read()

    def transact(self, mutation, *, domain=MODEL_CONTROL_DOMAIN):
        with self._transaction() as connection:
            self._initialize(connection)
            self._journal.initialize(connection)
            self._journal.compact(connection, self._read)
            initialize_retention(connection)
            state = self._read(connection)
            self._journal.verify_head(connection, state)
            before_state = deepcopy(state)
            before = canonical(state)
            mutation(state)
            guard_retired_records(connection, before_state, state)
            retain_model_control(connection, state)
            if canonical(state) == before:
                return deepcopy(state)
            state["revision"] += 1
            self._journal.append(connection, state, domain=domain)
            connection.execute("INSERT INTO model_control_state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (canonical(state),))
            self._journal.compact(connection, self._read)
        return deepcopy(state)

    def prepare_recovery(self):
        with self._transaction() as connection:
            status = self._journal.verify(connection, self._read)
            self._initialize(connection)
            self._journal.initialize(connection)
            self._journal.compact(connection, self._read, verified=True)
            return status

    def recovery_status(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            return self._journal.verify(connection, self._read)
        finally:
            connection.rollback()
            connection.close()
