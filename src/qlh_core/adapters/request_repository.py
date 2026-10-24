"""SQLite request journal, hash chained in requests:v1 with local_sqlite commits."""
from __future__ import annotations

from copy import deepcopy
import json
from qlh_core.contracts.execution import REQUEST_DOMAIN, ExecutionError
from qlh_core.contracts.deployment import canonical
from qlh_core.adapters.sqlite_journal import SnapshotJournal


def empty_state():
    return {"schema_version": 1, "revision": 0, "boot_id": "", "requests": {}, "commit_scope": "local_sqlite"}


class SqliteRequestRepository:
    def __init__(self, *, connect, transaction, max_events=64, max_event_bytes=8 * 1024 * 1024):
        self._connect = connect
        self._transaction = transaction
        self._journal = SnapshotJournal("request_journal_events", domain=REQUEST_DOMAIN,
            error=lambda: ExecutionError("request_journal_corrupt"), max_events=max_events, max_bytes=max_event_bytes)

    @staticmethod
    def _initialize(connection):
        connection.execute("CREATE TABLE IF NOT EXISTS request_journal_state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS request_journal_events (sequence INTEGER PRIMARY KEY, domain_id TEXT NOT NULL, previous_digest TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL)")

    @staticmethod
    def _read(connection):
        row = connection.execute("SELECT payload FROM request_journal_state WHERE id=1").fetchone()
        value = json.loads(row["payload"]) if row else empty_state()
        if value.get("schema_version") != 1:
            raise ExecutionError("request_journal_schema_invalid")
        return value

    def read(self):
        # No schema creation or state repair during queries.
        connection = self._connect()
        try:
            present = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='request_journal_state'").fetchone()
            return self._read(connection) if present else empty_state()
        finally:
            connection.close()

    def transact(self, mutation, *, domain=REQUEST_DOMAIN):
        with self._transaction() as connection:
            self._initialize(connection)
            self._journal.initialize(connection)
            self._journal.compact(connection, self._read)
            state = self._read(connection)
            self._journal.verify_head(connection, state)
            before = canonical(state)
            mutation(state)
            if canonical(state) == before:
                return deepcopy(state)
            state["revision"] += 1
            self._journal.append(connection, state, domain=domain)
            connection.execute("INSERT INTO request_journal_state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (canonical(state),))
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
