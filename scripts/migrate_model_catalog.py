"""Review or import a legacy registry without copying model files.

Default is a read-only dry-run. --apply requires the exported dry-run report;
source, catalog revisions and byte inventory are rechecked before publication.
SQLite inputs must be offline snapshots (no pending WAL) or JSON exports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qlh_core.adapters.artifacts import ArtifactStore, LegacyRegistryImporter
from qlh_core.contracts.catalog import CatalogError, RegistrySnapshot


def readonly_connection(path):
    path = Path(path).resolve()
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise CatalogError("catalog_snapshot_has_wal", "use an offline SQLite snapshot or JSON export")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def read_source(path):
    path = Path(path)
    if path.suffix.lower() in (".sqlite", ".sqlite3", ".db"):
        connection = readonly_connection(path)
        try:
            return [dict(row) for row in connection.execute("SELECT model_id, payload FROM model_registry ORDER BY model_id")], 1
        finally:
            connection.close()
    if path.stat().st_size > 64 * 1024 * 1024:
        raise CatalogError("legacy_registry_too_large", "registry metadata exceeds size limit")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload, 1
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise CatalogError("legacy_registry_schema_unsupported", "expected models list and schema_version")
    return payload["models"], payload.get("schema_version", payload.get("schema", 1))


class ReadOnlyCatalog:
    def __init__(self, path):
        self.path = Path(path)

    def snapshot(self, model_id):
        if not self.path.exists():
            return RegistrySnapshot(model_id, 0, None)
        connection = readonly_connection(self.path)
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            row = (connection.execute("SELECT payload FROM model_registry WHERE model_id=?", (model_id,)).fetchone()
                   if "model_registry" in tables else None)
            version = (connection.execute("SELECT revision FROM model_registry_versions WHERE model_id=?", (model_id,)).fetchone()
                       if "model_registry_versions" in tables else None)
            return RegistrySnapshot(model_id, int(version[0]) if version else (1 if row else 0), json.loads(row[0]) if row else None)
        finally:
            connection.close()

    def publish(self, *args, **kwargs):
        raise RuntimeError("read-only dry-run catalog")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--sqlite-path", type=Path, required=True, help="destination catalog database")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--review-report", type=Path)
    args = parser.parse_args(argv)
    if args.apply and args.review_report is None:
        parser.error("--apply requires --review-report from a dry-run")
    try:
        rows, schema = read_source(args.source)
        artifacts = ArtifactStore()
        importer = LegacyRegistryImporter(ReadOnlyCatalog(args.sqlite_path), artifacts)
        preview = importer.dry_run(rows, source_schema=schema)
        if args.apply:
            from qlh_core.adapters.artifacts import verify_import_report
            expected = json.loads(args.review_report.read_text(encoding="utf-8-sig"))
            if not verify_import_report(expected) or expected["plan_sha256"] != preview.plan_sha256:
                raise CatalogError("legacy_import_plan_stale", "source or destination changed since dry-run")
            import local_store
            with local_store.bind_local_store(args.sqlite_path):
                local_store.initialize_local_store(str(args.sqlite_path), migrate_legacy=False)
                importer = LegacyRegistryImporter(local_store._catalog_service(), artifacts)
                report = importer.import_registry(rows, source_schema=schema, expected_report=expected)
        else:
            report = preview
        print(json.dumps(report.as_dict(), ensure_ascii=True, sort_keys=True))
        return 0 if all(entry.get("result") not in ("failed", "conflict") and entry["action"] != "reject" for entry in report.entries) else 1
    except (CatalogError, OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"reason_code": getattr(exc, "code", "catalog_migration_failed"), "message": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
