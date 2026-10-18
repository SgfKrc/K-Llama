"""Model control contracts; the journal domain is independent of workflow events."""
from __future__ import annotations

from typing import Protocol
import hashlib
import json

MODEL_CONTROL_DOMAIN = "model-control:v1"
OPERATION_TERMINAL = frozenset({"ready", "failed", "cancelled", "superseded"})
DEPLOYMENT_TERMINAL = frozenset({"aborted", "retired"})
DEPLOYMENT_TRANSITIONS = {
    "candidate": {"preparing", "aborting"},
    "preparing": {"prepared", "aborting"},
    "prepared": {"committing", "aborting"},
    "committing": {"active", "aborting"},
    "active": {"invalidated", "retiring"},
    "invalidated": {"retiring"},
    "retiring": {"retired"},
    "aborting": {"aborted"},
    "retired": set(), "aborted": set(),
}


class ModelControlError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def command_key(node_id, payload):
    return digest({"node_id": node_id, "payload": payload})


class ModelControlRepository(Protocol):
    def read(self) -> dict: ...
    def transact(self, mutation, *, domain: str = MODEL_CONTROL_DOMAIN) -> dict: ...
    def recovery_status(self) -> dict: ...


def frozen_assignments(expected):
    return {node: {key: value for key, value in item.items() if key != "phase"}
            for node, item in expected.items()}


def require_runtime_receipt(assignment, receipt):
    """Validate runtime evidence against the complete frozen assignment."""
    from qlh_core.contracts.load import RuntimeLoadSpec
    identity = assignment.get("worker_prepare_identity", assignment.get("identity", {}))
    try:
        spec = RuntimeLoadSpec.from_dict(assignment["load_spec"])
        actual = RuntimeLoadSpec.from_dict(receipt["load_spec"])
        valid = (actual == spec and identity.get("load_spec_digest") == spec.digest
                 and receipt["layer_range"] == identity["layer_range"]
                 and receipt["artifact_sha256"] == identity["artifact_sha256"]
                 and receipt["memory_scope"] == "weight_tensors"
                 and type(receipt["materialized_bytes"]) is int and receipt["materialized_bytes"] > 0)
    except (ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise ModelControlError("deployment_runtime_receipt_mismatch")
    budget = assignment.get("required_bytes", assignment.get("capacity_bytes", 0))
    if budget and receipt["materialized_bytes"] + spec.kv_cache_bytes > budget:
        raise ModelControlError("deployment_materialization_budget_exceeded")
