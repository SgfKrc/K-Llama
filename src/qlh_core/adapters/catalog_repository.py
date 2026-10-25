"""SQLite transport for the catalog owner; connections are supplied by bootstrap.

The legacy public model_registry payload stays compatible. Versions live in a
separate table and survive deletion, so same-name recreation cannot cause ABA.
"""
from __future__ import annotations

import json


class SqliteCatalogRepository:
    def __init__(self, *, connect, transaction, write_row, now):
        self._connect = connect
        self._transaction = transaction
        self._write_row = write_row
        self._now = now

    @staticmethod
    def _read(connection, model_id):
        from qlh_core.contracts.catalog import RegistrySnapshot
        row = connection.execute(
            "SELECT payload FROM model_registry WHERE model_id = ?", (model_id,),
        ).fetchone()
        version = connection.execute(
            "SELECT revision FROM model_registry_versions WHERE model_id = ?", (model_id,),
        ).fetchone()
        revision = int(version["revision"]) if version else (1 if row else 0)
        payload = None
        if row is not None:
            payload = json.loads(row["payload"])
            if not isinstance(payload, dict):
                raise ValueError("invalid_catalog_payload")
        return RegistrySnapshot(model_id, revision, payload)

    def snapshot(self, model_id):
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            return self._read(connection, model_id)
        finally:
            connection.rollback()
            connection.close()

    def compare_and_swap(self, model_id, payload, *, expected_revision, next_revision):
        with self._transaction() as connection:
            current = self._read(connection, model_id)
            if current.registry_revision != expected_revision:
                return False
            from qlh_core.adapters.publication_cleanup import require_adoptable
            require_adoptable(connection, payload)
            self._write_row(connection, model_id, payload, self._now())
            connection.execute(
                "INSERT INTO model_registry_versions(model_id, revision) VALUES (?, ?) "
                "ON CONFLICT(model_id) DO UPDATE SET revision=excluded.revision",
                (model_id, next_revision),
            )
        return True

    def remove(self, model_id, *, expected_revision, next_revision):
        with self._transaction() as connection:
            current = self._read(connection, model_id)
            if current.registry_revision != expected_revision or current.payload is None:
                return False
            connection.execute("DELETE FROM model_registry WHERE model_id = ?", (model_id,))
            connection.execute(
                "INSERT INTO model_registry_versions(model_id, revision) VALUES (?, ?) "
                "ON CONFLICT(model_id) DO UPDATE SET revision=excluded.revision",
                (model_id, next_revision),
            )
        return True
