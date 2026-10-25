"""Verified snapshot checkpoints with a bounded, contiguous SQLite journal tail."""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from qlh_core.contracts.deployment import canonical, digest


logger = logging.getLogger(__name__)


class SnapshotJournal:
    def __init__(self, table, *, domain, error, max_events=64, max_bytes=8 * 1024 * 1024):
        if max_events < 1 or max_bytes < 1:
            raise ValueError("journal retention limits must be positive")
        self.table = table
        self.checkpoint_table = table + "_checkpoint"
        self.domain = domain
        self.error = error
        self.max_events = max_events
        self.max_bytes = max_bytes

    def initialize(self, connection):
        connection.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (sequence INTEGER PRIMARY KEY, domain_id TEXT NOT NULL, previous_digest TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL)")
        connection.execute(f"CREATE TABLE IF NOT EXISTS {self.checkpoint_table} (id INTEGER PRIMARY KEY CHECK(id=1), sequence INTEGER NOT NULL, domain_id TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL)")

    @staticmethod
    def _exists(connection, table):
        return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None

    def _checkpoint(self, connection):
        if not self._exists(connection, self.checkpoint_table):
            return None
        return connection.execute(f"SELECT * FROM {self.checkpoint_table} WHERE id=1").fetchone()

    def _decode(self, row, *, checkpoint=False):
        try:
            value = json.loads(row["payload"])
            if (not isinstance(value, dict) or type(row["sequence"]) is not int or row["sequence"] < 1
                    or row["domain_id"] != self.domain or value.get("domain_id") != self.domain
                    or value.get("sequence") != row["sequence"] or digest(value) != row["digest"]
                    or not isinstance(value.get("state"), dict)
                    or value["state"].get("revision") != row["sequence"]
                    or not isinstance(value.get("previous_digest"), str)
                    or len(value["previous_digest"]) != 64
                    or (not checkpoint and value["previous_digest"] != row["previous_digest"])):
                raise self.error()
            return value
        except (ValueError, TypeError, KeyError) as exc:
            raise self.error() from exc

    def verify(self, connection, read_state):
        started = time.perf_counter()
        rows = payload_bytes = 0
        sequence, previous, last = 0, "0" * 64, None
        try:
            state_table = self.table.removesuffix("_events") + "_state"
            if not self._exists(connection, state_table):
                if self._exists(connection, self.table) or self._exists(connection, self.checkpoint_table):
                    raise self.error()
                return {"verified": False, "commit_scope": "local_sqlite"}
            if not self._exists(connection, self.table):
                raise self.error()
            checkpoint = self._checkpoint(connection)
            if checkpoint is not None:
                value = self._decode(checkpoint, checkpoint=True)
                sequence, previous, last = checkpoint["sequence"], checkpoint["digest"], value["state"]
                payload_bytes += len(checkpoint["payload"].encode("utf-8"))
            for row in connection.execute(f"SELECT * FROM {self.table} ORDER BY sequence"):
                value = self._decode(row)
                if row["sequence"] != sequence + 1 or row["previous_digest"] != previous:
                    raise self.error()
                sequence, previous, last = row["sequence"], row["digest"], value["state"]
                rows += 1
                payload_bytes += len(row["payload"].encode("utf-8"))
            state = read_state(connection)
            if (last is not None and last != state) or (last is None and state["revision"]):
                raise self.error()
            return {"verified": sequence > 0, "durable_sequence": sequence,
                    "checkpoint_digest": previous, "commit_scope": "local_sqlite"}
        except (sqlite3.DatabaseError, KeyError, TypeError, ValueError) as exc:
            raise self.error() from exc
        finally:
            logger.info("event=control_journal_replay domain=%s rows=%s bytes=%s sequence=%s elapsed_ms=%s",
                        self.domain, rows, payload_bytes, sequence, round((time.perf_counter() - started) * 1000))

    def append(self, connection, state, *, domain):
        previous = connection.execute(f"SELECT sequence,digest FROM {self.table} ORDER BY sequence DESC LIMIT 1").fetchone()
        if previous is None:
            previous = self._checkpoint(connection)
        sequence = previous["sequence"] + 1 if previous else 1
        if state["revision"] != sequence or domain != self.domain:
            raise self.error()
        previous_digest = previous["digest"] if previous else "0" * 64
        event = {"sequence": sequence, "domain_id": domain, "previous_digest": previous_digest, "state": state}
        payload = canonical(event)
        connection.execute(f"INSERT INTO {self.table} VALUES (?,?,?,?,?)",
                           (sequence, domain, previous_digest, digest(event), payload))

    def verify_head(self, connection, state):
        row = connection.execute(f"SELECT * FROM {self.table} ORDER BY sequence DESC LIMIT 1").fetchone()
        checkpoint = row is None
        if checkpoint:
            row = self._checkpoint(connection)
        if row is None:
            if state["revision"]:
                raise self.error()
        elif self._decode(row, checkpoint=checkpoint)["state"] != state:
            raise self.error()

    def _cutoff(self, connection):
        rows = connection.execute(
            f"SELECT sequence,length(CAST(payload AS BLOB)) AS bytes FROM {self.table} ORDER BY sequence DESC LIMIT ?",
            (self.max_events + 1,)).fetchall()
        total = 0
        exceeded = False
        for index, row in enumerate(rows):
            if index and (index >= self.max_events or total + row["bytes"] > self.max_bytes):
                exceeded = True
                break
            total += row["bytes"]
        if exceeded:
            total = 0
            for index, row in enumerate(rows):
                if index and (index >= max(1, self.max_events // 2)
                              or total + row["bytes"] > max(1, self.max_bytes // 2)):
                    return row["sequence"]
                total += row["bytes"]
        return None

    def compact(self, connection, read_state, *, verified=False):
        cutoff = self._cutoff(connection)
        if cutoff is None:
            return
        if not verified:
            self.verify(connection, read_state)
        anchor = connection.execute(f"SELECT * FROM {self.table} WHERE sequence=?", (cutoff,)).fetchone()
        self._decode(anchor)
        connection.execute(
            f"INSERT INTO {self.checkpoint_table} VALUES (1,?,?,?,?) ON CONFLICT(id) DO UPDATE SET sequence=excluded.sequence,domain_id=excluded.domain_id,digest=excluded.digest,payload=excluded.payload",
            (anchor["sequence"], anchor["domain_id"], anchor["digest"], anchor["payload"]))
        removed = connection.execute(f"DELETE FROM {self.table} WHERE sequence<=?", (cutoff,)).rowcount
        logger.info("event=control_journal_checkpoint domain=%s sequence=%s removed_events=%s",
                    self.domain, cutoff, removed)
