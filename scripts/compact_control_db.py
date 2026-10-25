"""Inspect or explicitly compact an offline control database, preserving a backup."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
from qlh_core.adapters.request_repository import SqliteRequestRepository
from qlh_core.adapters.model_control_retention import initialize_retention, retain_model_control


def inspect_database(path):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        connection.execute("BEGIN")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result = {"path": str(path), "file_bytes": path.stat().st_size,
                  "page_size": connection.execute("PRAGMA page_size").fetchone()[0],
                  "page_count": connection.execute("PRAGMA page_count").fetchone()[0],
                  "free_pages": connection.execute("PRAGMA freelist_count").fetchone()[0],
                  "user_version": connection.execute("PRAGMA user_version").fetchone()[0], "journals": {}}
        for table in ("model_control_events", "request_journal_events"):
            if table in tables:
                row = connection.execute(f"SELECT COUNT(*),MIN(sequence),MAX(sequence) FROM {table}").fetchone()
                result["journals"][table] = dict(zip(("rows", "first_sequence", "last_sequence"), row))
        if "model_control_state" in tables:
            row = connection.execute("SELECT payload FROM model_control_state WHERE id=1").fetchone()
            if row:
                state = json.loads(row[0])
                with sqlite3.connect(":memory:") as preview:
                    preview.execute("BEGIN IMMEDIATE")
                    initialize_retention(preview)
                    removed = retain_model_control(preview, state)
                    result["history_retention"] = {"would_archive": removed,
                        "remaining": {kind: len(state[kind]) for kind in removed},
                        "records": [list(record) for record in preview.execute(
                            "SELECT record_kind,record_id FROM model_control_tombstones ORDER BY record_kind,record_id")]}
        return result
    finally:
        connection.close()


def fingerprints(connection):
    values = {}
    for name, sql in connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
        if name.startswith(("model_control_", "request_journal_")):
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        digest = hashlib.sha256(sql.encode("utf-8"))
        count = 0
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({quoted})")]
        order = ",".join('"' + column.replace('"', '""') + '"' for column in columns)
        for row in connection.execute(f"SELECT * FROM {quoted} ORDER BY {order}"):
            encoded = json.dumps(tuple(value.hex() if isinstance(value, bytes) else value for value in row),
                                 ensure_ascii=True, separators=(",", ":"))
            digest.update(encoded.encode("utf-8"))
            count += 1
        values[name] = {"rows": count, "sha256": digest.hexdigest()}
    return values


def compact_database(path, backup):
    if path == backup or backup.exists():
        raise ValueError("backup must be a new file separate from the database")
    if not backup.parent.is_dir() or shutil.disk_usage(backup.parent).free < path.stat().st_size + 1024 * 1024:
        raise ValueError("backup directory is missing or lacks free space")
    if shutil.disk_usage(path.parent).free < 2 * path.stat().st_size + 1024 * 1024:
        raise ValueError("database directory lacks vacuum workspace")
    def connect():
        connection = sqlite3.connect(path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        return connection
    @contextmanager
    def transaction():
        connection = connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
    source = connect()
    try:
        if source.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("database quick_check failed")
        before_version = source.execute("PRAGMA user_version").fetchone()[0]
        before_tables = fingerprints(source)
        with closing(sqlite3.connect(backup)) as destination:
            source.backup(destination)
        tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        source.close()
    reports = {}
    for prefix, adapter in (("model_control", SqliteModelControlRepository), ("request_journal", SqliteRequestRepository)):
        table = prefix + "_state"
        if not tables.intersection({table, prefix + "_events", prefix + "_events_checkpoint"}):
            continue
        repository = adapter(connect=connect, transaction=transaction)
        started = time.perf_counter()
        before = repository.prepare_recovery()
        before_state = repository.read()
        if table == "model_control_state":
            repository.transact(lambda state: None)
        after_state = repository.read()
        started_replay = time.perf_counter()
        after = repository.recovery_status()
        reports[table] = {"before": before, "after": after, "maintenance_s": time.perf_counter() - started,
                          "bounded_replay_s": time.perf_counter() - started_replay,
                          "before_state_bytes": len(json.dumps(before_state)), "after_state_bytes": len(json.dumps(after_state))}
        for key in ("boot_id", "selected_operation", "active_deployment", "cleanup", "artifact_cleanup", "lifecycle"):
            if before_state.get(key) != after_state.get(key):
                raise ValueError("maintenance changed protected state: " + key)
    source = connect()
    try:
        before_vacuum = path.stat().st_size
        started = time.perf_counter()
        source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        source.execute("VACUUM")
        source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        vacuum_s = time.perf_counter() - started
        if (source.execute("PRAGMA quick_check").fetchone()[0] != "ok"
                or source.execute("PRAGMA user_version").fetchone()[0] != before_version
                or fingerprints(source) != before_tables):
            raise ValueError("maintenance integrity check failed")
    finally:
        source.close()
    return {"backup": str(backup), "backup_bytes": backup.stat().st_size, "journals": reports,
            "vacuum_s": vacuum_s, "before_vacuum_bytes": before_vacuum, "after": inspect_database(path),
            "preserved_tables": before_tables}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--offline", action="store_true", help="confirm all product processes using this database have exited")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    path = args.database.expanduser().resolve(strict=True)
    protected_paths = {path, *(Path(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal"))}
    if args.backup:
        protected_paths.add(args.backup.expanduser().resolve())
    if args.output and args.output.expanduser().resolve() in protected_paths:
        parser.error("--output must be separate from the database and backup")
    report = {"dry_run": not args.apply, "before": inspect_database(path)}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.apply:
        if not args.offline or args.backup is None:
            parser.error("--apply requires --offline and --backup")
        report.update(compact_database(path, args.backup.expanduser().resolve()))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
