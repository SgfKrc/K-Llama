"""Explicit, bounded cleanup of unregistered publications in a managed cache.

Catalog adoption and cleanup claims serialize in SQLite. Hashing, walking,
renaming and deletion happen outside the transaction. Every target is resolved
inside the caller's managed root, and a dry-run receipt precedes reclamation.
"""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import stat

from qlh_core.contracts.deployment import ModelControlError, digest


def publication_evidence(location):
    path = Path(location).resolve(strict=True)
    if path.is_symlink() or not path.is_dir():
        raise ModelControlError("cleanup_location_invalid")
    files = []
    for entry in sorted(path.rglob("*")):
        info = entry.lstat()
        if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ModelControlError("cleanup_external_link")
        if entry.is_file():
            files.append([entry.relative_to(path).as_posix(), info.st_size, info.st_mtime_ns])
    info = path.stat()
    return {"device": info.st_dev, "inode": info.st_ino, "files_digest": digest(files),
            "file_count": len(files), "size_bytes": sum(value[1] for value in files)}


def payload_references(payload, location):
    # Catalog documents include legacy paths, typed locations, external files
    # and aliases. Check all string leaves; an unknown shape stays conservative.
    if isinstance(payload, dict):
        return any(payload_references(value, location) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(payload_references(value, location) for value in payload)
    if isinstance(payload, str) and payload:
        try:
            path = Path(payload).expanduser().resolve()
            return path == location or path.is_relative_to(location)
        except (OSError, ValueError):
            return False
    return False


class PublicationReclaimer:
    def __init__(self, repository):
        self.repository = repository

    def plan(self, cleanup_id, *, managed_root):
        state = self.repository.read()
        debt = state.get("artifact_cleanup", {}).get(cleanup_id)
        if debt is None:
            raise ModelControlError("publication_cleanup_not_found")
        root = Path(managed_root).resolve(strict=True)
        location = Path(debt["location"]).absolute()
        resolved = location.resolve(strict=location.exists())
        if resolved == root or not resolved.is_relative_to(root) or location != resolved:
            raise ModelControlError("cleanup_outside_managed_root")
        quarantine = root / (".qlh-control-reclaim-" + cleanup_id)
        if location.exists() and publication_evidence(location) != debt["evidence"]:
            raise ModelControlError("cleanup_publication_changed")
        return {"cleanup_id": cleanup_id, "location": str(location), "quarantine": str(quarantine),
                "managed_root": str(root), "file_count": debt["evidence"]["file_count"],
                "size_bytes": debt["evidence"]["size_bytes"], "dry_run": True}

    def reclaim(self, cleanup_id, *, managed_root, dry_run=True):
        import json
        report = self.plan(cleanup_id, managed_root=managed_root)
        if dry_run:
            return report
        location, quarantine = Path(report["location"]), Path(report["quarantine"])
        # The adapter's claim runs under the same transaction as catalog CAS.
        original_transaction = self.repository._transaction
        from contextlib import contextmanager
        @contextmanager
        def claimed_transaction():
            with original_transaction() as connection:
                self.repository._initialize(connection)
                state = self.repository._read(connection)
                if any(d.get("status") not in {"retired", "aborted"} for d in state["deployments"].values()) or state["cleanup"]:
                    raise ModelControlError("cleanup_runtime_pending")
                exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_registry'").fetchone()
                if exists:
                    for row in connection.execute("SELECT payload FROM model_registry"):
                        if payload_references(json.loads(row["payload"]), location):
                            raise ModelControlError("cleanup_catalog_referenced")
                yield connection
        # Supply a separate repository object, never temporarily mutate the live port.
        from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
        claimed = SqliteModelControlRepository(connect=self.repository._connect, transaction=claimed_transaction)
        def claim(state):
            debt = state["artifact_cleanup"][cleanup_id]
            if debt["status"] == "reclaimed":
                return
            debt.update(status="reclaiming", quarantine=str(quarantine))
        claimed.transact(claim)
        if location.exists():
            if quarantine.exists():
                raise ModelControlError("cleanup_quarantine_conflict")
            if publication_evidence(location) != self.repository.read()["artifact_cleanup"][cleanup_id]["evidence"]:
                raise ModelControlError("cleanup_publication_changed")
            os.rename(location, quarantine)
        if quarantine.exists():
            root = Path(report["managed_root"])
            if quarantine.resolve() != quarantine or not quarantine.is_relative_to(root) or quarantine == root:
                raise ModelControlError("cleanup_outside_managed_root")
            if publication_evidence(quarantine) != self.repository.read()["artifact_cleanup"][cleanup_id]["evidence"]:
                raise ModelControlError("cleanup_publication_changed")
            shutil.rmtree(quarantine)
        def complete(state):
            state["artifact_cleanup"][cleanup_id]["status"] = "reclaimed"
        self.repository.transact(complete)
        return {**report, "dry_run": False, "status": "reclaimed"}


def require_adoptable(connection, payload):
    """Catalog CAS calls this within its own write transaction."""
    import json
    row = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_control_state'").fetchone()
    if row is None:
        return
    row = connection.execute("SELECT payload FROM model_control_state WHERE id=1").fetchone()
    if row is None:
        return
    state = json.loads(row["payload"])
    for debt in state.get("artifact_cleanup", {}).values():
        if debt["status"] in {"reclaiming", "reclaimed"} and payload_references(payload, Path(debt["location"])):
            raise ModelControlError("catalog_publication_retired")
