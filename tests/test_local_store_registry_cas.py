"""Registry CAS tests against temporary SQLite stores and independent connections."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

import local_store


def _row(name="old"):
    return {"model_id": "fixture", "name": name, "model_path": "models/source",
            "gguf_path": "", "manifest_sha256": "a" * 64,
            "manifest_external_files": {"external/model.gguf": "models/original/model.gguf"}}


def test_registry_cas_updates_matching_payload_and_all_columns():
    old = _row()
    local_store.save_local_experimental_model("fixture", json.dumps(old, indent=2))
    snapshot = local_store.get_local_experimental_models()[0]
    updated = {**snapshot, "name": "converted", "gguf_path": "models/converted.gguf"}
    assert local_store.save_local_experimental_model_cas("fixture", updated, expected=snapshot)
    assert local_store.get_local_experimental_models() == [updated]
    with sqlite3.connect(local_store._sqlite_path()) as connection:
        name, gguf_path = connection.execute(
            "SELECT name, gguf_path FROM model_registry WHERE model_id = 'fixture'",
        ).fetchone()
    assert name == "converted" and gguf_path == updated["gguf_path"]


@pytest.mark.parametrize("change", ["name", "binding", "delete", "corrupt"])
def test_registry_cas_never_replaces_changed_deleted_or_invalid_payload(change):
    old = _row()
    local_store.save_local_experimental_model("fixture", old)
    snapshot = local_store.get_local_experimental_models()[0]
    current = deepcopy(old)
    if change == "delete":
        local_store.delete_local_experimental_model("fixture")
    elif change == "corrupt":
        with sqlite3.connect(local_store._sqlite_path()) as connection:
            connection.execute("UPDATE model_registry SET payload = 'invalid JSON'")
    else:
        if change == "name":
            current["name"] = "reregistered"
        else:
            current["manifest_external_files"]["external/model.gguf"] = "models/new/model.gguf"
        assert local_store.save_local_experimental_model("fixture", current, expected_revision=snapshot.registry_revision)
    with sqlite3.connect(local_store._sqlite_path()) as connection:
        before = connection.execute("SELECT * FROM model_registry").fetchall()
    assert not local_store.save_local_experimental_model_cas(
        "fixture", {**old, "name": "stale conversion"}, expected=snapshot,
    )
    with sqlite3.connect(local_store._sqlite_path()) as connection:
        assert connection.execute("SELECT * FROM model_registry").fetchall() == before


def test_registry_cas_absent_snapshot_inserts_only_once():
    first, second = _row("first"), _row("second")
    assert local_store.save_local_experimental_model_cas("fixture", first, expected=None)
    assert not local_store.save_local_experimental_model_cas("fixture", second, expected=None)
    assert local_store.get_local_experimental_models() == [first]


@pytest.mark.parametrize("existing", [False, True])
def test_registry_cas_concurrent_contenders_have_exactly_one_winner(existing):
    old = _row()
    if existing:
        local_store.save_local_experimental_model("fixture", old)
    else:
        local_store.initialize_local_store()
    barrier = threading.Barrier(2)

    def save(name):
        row = _row(name)
        barrier.wait(timeout=5)
        return row, local_store.save_local_experimental_model_cas(
            "fixture", row, expected=deepcopy(old) if existing else None,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save, ["one", "two"]))
    winners = [row for row, success in results if success]
    assert len(winners) == 1
    assert local_store.get_local_experimental_models() == winners


def test_registry_cas_comparison_holds_sqlite_write_lock(monkeypatch):
    old = _row()
    local_store.save_local_experimental_model("fixture", old)
    store_path = local_store._sqlite_path()
    connect = local_store._connect
    blocked = []

    class ConnectionProbe:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, *args):
            cursor = self.connection.execute(sql, *args)
            if sql.startswith("SELECT payload FROM model_registry WHERE"):
                row = cursor.fetchone()
                # Bypass the module's Python lock: SQLite itself must protect
                # the interval between reading this payload and publishing it.
                competitor = sqlite3.connect(store_path, timeout=0, isolation_level=None)
                try:
                    with pytest.raises(sqlite3.OperationalError, match="locked"):
                        competitor.execute(
                            "UPDATE model_registry SET payload = ? WHERE model_id = ?",
                            (json.dumps(_row("competitor")), "fixture"),
                        )
                    blocked.append(True)
                finally:
                    competitor.close()
                return SimpleNamespace(fetchone=lambda: row)
            return cursor

    monkeypatch.setattr(local_store, "_connect", lambda: ConnectionProbe(connect()))
    updated = _row("cas-winner")
    assert local_store.save_local_experimental_model_cas("fixture", updated, expected=old)
    assert blocked and all(blocked)
    assert local_store.get_local_experimental_models() == [updated]
