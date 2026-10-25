"""Bound hot model-control history inside its owner's SQLite write transaction.

Initialize before mutation, guard against the pre-mutation snapshot, then retain
before incrementing the revision and writing the journal/state. These functions
never commit. Tombstones stay outside the aggregate and must live as long as the
database so that retired identities and idempotency keys cannot be reused.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from pathlib import PurePosixPath, PureWindowsPath

from qlh_core.contracts.deployment import (
    DEPLOYMENT_TERMINAL, OPERATION_TERMINAL, ModelControlError, digest,
)


HISTORY_KEEP = {"supplies": 16, "operations": 16, "deployments": 16, "outbox": 64}
_IDENTITY = {"supplies": "operation_id", "operations": "operation_id",
             "deployments": "deployment_id", "outbox": "command_id"}
_TERMINAL = {"supplies": frozenset({"available", "failed", "cancelled"}),
             "operations": OPERATION_TERMINAL, "deployments": DEPLOYMENT_TERMINAL,
             "outbox": frozenset({"acknowledged", "superseded"})}
_CONFLICT = {"supplies": "supply_terminal_conflict", "operations": "operation_terminal_conflict",
             "deployments": "deployment_retired", "outbox": "deployment_retired"}
_FIELDS = {
    "supplies": frozenset({
        "operation_id", "intent", "idempotency_key", "status", "reason_code", "reason", "created_at",
        "updated_at", "loaded_ready", "nodes", "plan", "capacity_diagnostic", "requires_prepare",
        "wait_for_nodes", "source_progress", "source_model_sha256", "tokenizer_sha256",
        "candidate_node_ids", "master_model_path", "retention",
    }),
    "operations": frozenset({
        "operation_id", "intent", "idempotency_key", "status", "phase", "created_at", "updated_at",
        "deadline", "reason_code", "reason", "cause", "config_id", "candidate_id", "generation",
        "durable", "commit_scope",
    }),
    "deployments": frozenset({
        "deployment_id", "boot_id", "authority", "connections", "created_at", "assignments",
        "input_revisions", "local_assignment", "local_executors", "status", "published",
        "validity", "observed_at",
        "reason_code", "receipts",
    }),
    "outbox": frozenset({
        "command_id", "node_id", "payload", "action", "boot_id", "authority", "connection_generation",
        "state", "attempts", "created_at", "last_delivery", "delivered",
    }),
}
_STATE_FIELDS = frozenset({
    "schema_version", "revision", "boot_id", "selected_operation", "active_deployment", "operations",
    "supplies", "deployments", "outbox", "cleanup", "artifact_cleanup", "lifecycle", "commit_scope",
})
_TIMES = ("created_at", "updated_at", "observed_at", "last_delivery")


def _require_transaction(connection):
    if not connection.in_transaction:
        raise ModelControlError("model_control_retention_transaction_required")


def initialize_retention(connection):
    """Create the archive and key index; call only inside the owner's write transaction."""
    _require_transaction(connection)
    connection.execute("""CREATE TABLE IF NOT EXISTS model_control_tombstones (
        record_kind TEXT NOT NULL, record_id TEXT NOT NULL, status TEXT NOT NULL,
        idempotency_key TEXT NOT NULL, intent_digest TEXT NOT NULL,
        PRIMARY KEY (record_kind, record_id)) WITHOUT ROWID""")
    connection.execute("""CREATE INDEX IF NOT EXISTS model_control_tombstones_idempotency
        ON model_control_tombstones(record_kind, idempotency_key) WHERE idempotency_key <> ''""")


def guard_retired_records(connection, before_state, state):
    """Reject newly introduced retired IDs or archived keys before any state/journal write.

    Same-intent key reuse raises operation_idempotency_reused, matching the live
    supply owner. It never grants permission to recreate the archived record.
    """
    _require_transaction(connection)
    for kind, identity_field in _IDENTITY.items():
        before = before_state.get(kind, {})
        for record_id, record in state.get(kind, {}).items():
            old = before.get(record_id)
            identities = {record_id} if old is None else set()
            if isinstance(record, dict):
                identity = record.get(identity_field)
                if (isinstance(identity, str) and identity
                        and (not isinstance(old, dict) or identity != old.get(identity_field))):
                    identities.add(identity)
            for identity in identities:
                retired = connection.execute(
                    "SELECT 1 FROM model_control_tombstones WHERE record_kind=? AND record_id=?",
                    (kind, identity),
                ).fetchone()
                if retired:
                    raise ModelControlError(_CONFLICT[kind])
            if not isinstance(record, dict):
                continue
            key = record.get("idempotency_key")
            if not key or (isinstance(old, dict) and old.get("idempotency_key") == key
                           and old.get("intent") == record.get("intent")):
                continue
            archived = connection.execute(
                "SELECT intent_digest FROM model_control_tombstones "
                "WHERE record_kind=? AND idempotency_key=? AND idempotency_key <> '' AND record_id<>?",
                (kind, key, record_id),
            ).fetchall()
            if archived:
                same_intent = all(row[0] == digest(record.get("intent")) for row in archived)
                raise ModelControlError("operation_idempotency_reused" if same_intent
                                        else "operation_idempotency_conflict")


def _strings(value):
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _safe_history(kind, record_id, record):
    if not isinstance(record, dict) or set(record) - _FIELDS[kind]:
        return False
    if not isinstance(record_id, str) or not record_id or record.get(_IDENTITY[kind]) != record_id:
        return False
    status = record.get("state" if kind == "outbox" else "status")
    if not isinstance(status, str) or status not in _TERMINAL[kind]:
        return False
    key = record.get("idempotency_key", "")
    if key is not None and not isinstance(key, str):
        return False
    if key and not isinstance(record.get("intent"), dict):
        return False
    if any(type(record[field]) not in (int, float) or not isfinite(record[field])
           for field in _TIMES if field in record):
        return False
    if kind == "outbox":
        return (isinstance(record.get("payload"), dict) and bool(record.get("node_id"))
                and record.get("action") in {"prepare", "commit", "abort", "finalize", "release", "cleanup"})
    return True


def _path(value):
    if not isinstance(value, str) or not value:
        return ""
    if "\\" in value or (len(value) > 1 and value[1] == ":"):
        path = PureWindowsPath(value)
        return path.as_posix().rstrip("/").casefold() if path.is_absolute() else ""
    path = PurePosixPath(value)
    return path.as_posix().rstrip("/") if path.is_absolute() else ""


def _artifact_locations(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"location", "quarantine"}:
                location = _path(item)
                if location:
                    yield location
            yield from _artifact_locations(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _artifact_locations(item)


def _closure(protected, links):
    pending = list(protected)
    while pending:
        for target in links[pending.pop()] - protected:
            protected.add(target)
            pending.append(target)


def _protected_records(state, records, candidates):
    identities = defaultdict(set)
    values = {}
    for node, record in records.items():
        kind, record_id = node
        identities[record_id].add(node)
        values[node] = set(_strings(record))
        if not isinstance(record, dict):
            continue
        candidate_id = record.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            identities[candidate_id].add(node)
        if kind == "outbox" and isinstance(record.get("payload"), dict):
            payload = record["payload"]
            cleanup_id = payload.get("cleanup_id") or (payload.get("config_id") if payload.get("cleanup_action") else "")
            if isinstance(cleanup_id, str) and cleanup_id:
                identities[cleanup_id].add(node)
    links = defaultdict(set)
    for node, strings in values.items():
        for value in strings:
            for target in identities.get(value, ()):
                links[node].add(target)
                links[target].add(node)
    protected = set(records) - candidates
    if set(state) - _STATE_FIELDS or state.get("schema_version", 1) != 1:
        protected.update(records)
    for key, value in state.items():
        if key not in _IDENTITY:
            for identity in _strings(value):
                protected.update(identities.get(identity, ()))
    locations = set(_artifact_locations(state.get("artifact_cleanup", {})))
    if locations:
        for node, strings in values.items():
            paths = {_path(value) for value in strings} - {""}
            if any(path == location or path.startswith(location + "/") or location.startswith(path + "/")
                   for path in paths for location in locations):
                protected.add(node)
    _closure(protected, links)
    return protected, links


def retain_model_control(connection, state, *, keep=None):
    """Archive safe history in place and return per-collection removal counts.

    Defaults keep the latest 16/16/16/64 unprotected terminal records. Protected
    records are additional: selected/active/lifecycle/debt references, unfinished
    records, unknown fields, and everything connected to them always remain hot.
    Optional ``keep`` overrides individual collection limits. No age-based rule
    or inferred cleanup completion is used, including for orphaned pending work.
    """
    _require_transaction(connection)
    limits = {**HISTORY_KEEP, **(keep or {})}
    if set(limits) != set(_IDENTITY) or any(type(value) is not int or value < 0 for value in limits.values()):
        raise ValueError("model_control_retention_keep_invalid")
    records = {(kind, record_id): record for kind in _IDENTITY
               for record_id, record in state.get(kind, {}).items()}
    candidates = {node for node, record in records.items() if _safe_history(*node, record)}
    protected, links = _protected_records(state, records, candidates)
    for kind, limit in limits.items():
        history = [node for node in candidates - protected if node[0] == kind]
        history.sort(key=lambda node: (max(records[node].get(field, 0) for field in _TIMES), node[1]), reverse=True)
        protected.update(history[:limit])
    _closure(protected, links)
    retiring = sorted(candidates - protected)
    removed = {kind: 0 for kind in _IDENTITY}
    for kind, record_id in retiring:
        record = records[(kind, record_id)]
        existing = connection.execute(
            "SELECT 1 FROM model_control_tombstones WHERE record_kind=? AND record_id=?", (kind, record_id),
        ).fetchone()
        if existing:
            raise ModelControlError(_CONFLICT[kind])
        connection.execute("INSERT INTO model_control_tombstones VALUES (?,?,?,?,?)", (
            kind, record_id, record["state" if kind == "outbox" else "status"],
            record.get("idempotency_key") or "", digest(record.get("intent")),
        ))
    for kind, record_id in retiring:
        del state[kind][record_id]
        removed[kind] += 1
    return removed
