"""SQLite request journal, hash chained in requests:v1 with local_sqlite commits."""
from __future__ import annotations

from copy import deepcopy
import json
from qlh_core.contracts.execution import REQUEST_DOMAIN, ExecutionError
from qlh_core.contracts.deployment import canonical, digest


def empty_state():
    return {"schema_version": 1, "revision": 0, "boot_id": "", "requests": {}, "commit_scope": "local_sqlite"}


class SqliteRequestRepository:
    def __init__(self, *, connect, transaction):
        self._connect = connect
        self._transaction = transaction

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
            state = self._read(connection)
            before = canonical(state)
            mutation(state)
            if canonical(state) == before:
                return deepcopy(state)
            state["revision"] += 1
            previous = connection.execute("SELECT sequence,digest FROM request_journal_events ORDER BY sequence DESC LIMIT 1").fetchone()
            sequence = int(previous["sequence"]) + 1 if previous else 1
            previous_digest = previous["digest"] if previous else "0" * 64
            event = {"sequence": sequence, "domain_id": domain, "previous_digest": previous_digest, "state": state}
            event_digest = digest(event)
            connection.execute("INSERT INTO request_journal_events VALUES (?,?,?,?,?)", (sequence, domain, previous_digest, event_digest, canonical(event)))
            connection.execute("INSERT INTO request_journal_state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (canonical(state),))
        return deepcopy(state)

    def recovery_status(self):
        connection = self._connect()
        try:
            present = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='request_journal_events'").fetchone()
            if not present:
                return {"verified": False, "commit_scope": "local_sqlite"}
            connection.execute("BEGIN")
            previous = "0" * 64
            sequence = 0
            last = None
            for row in connection.execute("SELECT * FROM request_journal_events ORDER BY sequence"):
                value = json.loads(row["payload"])
                if (row["sequence"] != sequence + 1 or row["domain_id"] != REQUEST_DOMAIN
                        or row["previous_digest"] != previous or value.get("previous_digest") != previous
                        or value.get("sequence") != row["sequence"] or value.get("domain_id") != REQUEST_DOMAIN
                        or digest(value) != row["digest"]):
                    raise ExecutionError("request_journal_corrupt")
                sequence, previous, last = row["sequence"], row["digest"], value["state"]
            if last is not None and last != self._read(connection):
                raise ExecutionError("request_journal_corrupt")
            return {"verified": sequence > 0, "durable_sequence": sequence,
                    "checkpoint_digest": previous, "commit_scope": "local_sqlite"}
        finally:
            connection.rollback()
            connection.close()
