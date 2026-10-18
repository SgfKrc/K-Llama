"""P4.5 state catalog: the explicit boundary for future HA recovery.

This is a manifest, not a replication implementation.  It records which
state a future leader handoff must carry, which state can be rebuilt, which
state is audit evidence only, and which state must never cross node
boundaries.  The manifest intentionally contains metadata only; it never
serializes secrets, model weights, sessions, or runtime handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CATALOG_SCHEMA_VERSION = "qlh.cluster.state_catalog.v1"
CATALOG_DOCUMENT_TYPE = "cluster_state_catalog"
STATE_CLASSES = frozenset(
    {"must_replicate", "rebuildable", "audit_only", "must_not_migrate"}
)
BOUNDARIES = frozenset(
    {
        "durable_local",
        "memory_only",
        "derived",
        "external_asset",
        "mixed",
        "not_yet_durable",
    }
)
RECOVERY_STRATEGIES = frozenset(
    {
        "journal_replay_before_write",
        "reissue_from_quorum",
        "rebuild_from_local_state",
        "rebuild_from_local_assets",
        "reconcile_then_retry",
        "preserve_append_only",
        "invalidate_and_reacquire",
        "never_copy",
        "user_owned_local_only",
    }
)
HANDOFF_POLICIES = frozenset(
    {
        "required_before_new_leader_write",
        "rebuild_after_new_leader",
        "retain_as_audit_evidence",
        "invalidate_on_term_change",
        "explicit_user_export_only",
    }
)


class StateCatalogError(ValueError):
    """Raised when the state catalog is malformed or internally ambiguous."""


@dataclass(frozen=True)
class StateCatalogEntry:
    state_id: str
    owner: str
    source: str
    state_class: str
    authority: str
    current_boundary: str
    recovery_strategy: str
    handoff_policy: str
    sensitive: bool
    notes: str

    def __post_init__(self) -> None:
        fields = {
            "state_id": self.state_id,
            "owner": self.owner,
            "source": self.source,
            "authority": self.authority,
            "notes": self.notes,
        }
        for field, value in fields.items():
            if not isinstance(value, str) or not value.strip():
                raise StateCatalogError(f"{field} must be non-empty text")
        if self.state_class not in STATE_CLASSES:
            raise StateCatalogError(f"unsupported state class: {self.state_class}")
        if self.current_boundary not in BOUNDARIES:
            raise StateCatalogError(
                f"unsupported current boundary: {self.current_boundary}"
            )
        if self.recovery_strategy not in RECOVERY_STRATEGIES:
            raise StateCatalogError(
                f"unsupported recovery strategy: {self.recovery_strategy}"
            )
        if self.handoff_policy not in HANDOFF_POLICIES:
            raise StateCatalogError(
                f"unsupported handoff policy: {self.handoff_policy}"
            )
        if not isinstance(self.sensitive, bool):
            raise StateCatalogError("sensitive must be boolean")
        if self.state_class == "must_not_migrate" and self.recovery_strategy != "never_copy":
            raise StateCatalogError(
                f"forbidden state {self.state_id} must use never_copy"
            )
        if self.state_class == "must_replicate" and self.handoff_policy != "required_before_new_leader_write":
            raise StateCatalogError(
                f"authoritative state {self.state_id} must gate new leader writes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "owner": self.owner,
            "source": self.source,
            "state_class": self.state_class,
            "authority": self.authority,
            "current_boundary": self.current_boundary,
            "recovery_strategy": self.recovery_strategy,
            "handoff_policy": self.handoff_policy,
            "sensitive": self.sensitive,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class StateCatalog:
    entries: tuple[StateCatalogEntry, ...]
    schema_version: str = CATALOG_SCHEMA_VERSION
    document_type: str = CATALOG_DOCUMENT_TYPE

    def __post_init__(self) -> None:
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            raise StateCatalogError("unsupported state catalog schema")
        if self.document_type != CATALOG_DOCUMENT_TYPE:
            raise StateCatalogError("unsupported state catalog document type")
        entries = tuple(self.entries)
        if not entries:
            raise StateCatalogError("state catalog cannot be empty")
        if any(not isinstance(entry, StateCatalogEntry) for entry in entries):
            raise StateCatalogError("state catalog entries must be StateCatalogEntry values")
        state_ids = [entry.state_id for entry in entries]
        if len(state_ids) != len(set(state_ids)):
            raise StateCatalogError("state catalog contains duplicate state_id values")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateCatalog":
        required = {"schema_version", "document_type", "entries"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise StateCatalogError("state catalog fields are not canonical")
        if not isinstance(value["entries"], list):
            raise StateCatalogError("state catalog entries must be a list")
        entry_fields = {
            "state_id", "owner", "source", "state_class", "authority",
            "current_boundary", "recovery_strategy", "handoff_policy",
            "sensitive", "notes",
        }
        entries: list[StateCatalogEntry] = []
        for raw in value["entries"]:
            if not isinstance(raw, Mapping) or set(raw) != entry_fields:
                raise StateCatalogError("state catalog entry fields are not canonical")
            entries.append(StateCatalogEntry(**dict(raw)))
        return cls(
            entries=tuple(entries),
            schema_version=value["schema_version"],
            document_type=value["document_type"],
        )


def build_state_catalog() -> StateCatalog:
    """Return the reviewed P4.5 whitelist in deterministic order."""
    entries: Sequence[StateCatalogEntry] = (
        StateCatalogEntry(
            "control.quorum_ledger", "control-plane", "src/cluster_quorum.py:SQLiteVoterLedger",
            "must_replicate", "cluster", "durable_local",
            "journal_replay_before_write", "required_before_new_leader_write", False,
            "Durable voter-set, term and one-vote records are required before quorum election.",
        ),
        StateCatalogEntry(
            "control.active_certificate", "control-plane", "src/cluster_control_contract.py:ControlPlaneAuthority",
            "rebuildable", "cluster", "memory_only", "reissue_from_quorum",
            "rebuild_after_new_leader", False,
            "Revalidate or reissue from the voter ledger; never trust a copied in-memory authority.",
        ),
        StateCatalogEntry(
            "task_graph.journal_events", "task-graph", "src/recovery_journal.py:workflow_events",
            "must_replicate", "cluster", "durable_local", "journal_replay_before_write",
            "required_before_new_leader_write", False,
            "Append-only events are the recoverable task authority; snapshots are only projections.",
        ),
        StateCatalogEntry(
            "task_graph.snapshots", "task-graph", "src/recovery_journal.py:workflow_snapshots",
            "rebuildable", "cluster", "durable_local", "rebuild_from_local_state",
            "rebuild_after_new_leader", False,
            "Rebuild from accepted journal events; a snapshot is not cross-node atomic state.",
        ),
        StateCatalogEntry(
            "cluster.membership_and_revocations", "cluster-control", "src/cluster_join.py + scheduler node registry",
            "must_replicate", "cluster", "mixed", "journal_replay_before_write",
            "required_before_new_leader_write", False,
            "Membership and revocation decisions need a durable signed ledger; live registry entries are observations.",
        ),
        StateCatalogEntry(
            "cluster.join_replay_nonces", "cluster-control", "src/cluster_join.py:JoinGrantLedger",
            "must_replicate", "cluster", "durable_local", "journal_replay_before_write",
            "required_before_new_leader_write", True,
            "Replay protection must survive leader change; only the nonce ledger metadata is covered.",
        ),
        StateCatalogEntry(
            "cluster.node_private_keys", "cluster-control", "src/cluster_join.py:cluster_join_keys",
            "must_not_migrate", "node", "durable_local", "never_copy",
            "invalidate_on_term_change", True,
            "Private issuer and node keys stay on their owner; public identity transfer uses a new grant.",
        ),
        StateCatalogEntry(
            "pipeline.active_layout", "pipeline-control", "src/pipeline_reshard.py:PipelineReshardCoordinator",
            "must_replicate", "cluster", "memory_only", "journal_replay_before_write",
            "required_before_new_leader_write", False,
            "The active manifest and its control epoch must be durable before a leader switch.",
        ),
        StateCatalogEntry(
            "pipeline.staged_reshard_plans", "pipeline-control", "src/pipeline_reshard.py:PipelineReshardCoordinator",
            "rebuildable", "cluster", "memory_only", "reconcile_then_retry",
            "rebuild_after_new_leader", False,
            "Uncommitted plans may be discarded and recomputed from current assets and node observations.",
        ),
        StateCatalogEntry(
            "pipeline.worker_leases", "pipeline-control", "src/llama_rpc_contract.py + scheduler",
            "must_not_migrate", "node", "memory_only", "never_copy",
            "invalidate_on_term_change", False,
            "Never transfer live reservations; old-term leases expire and are acquired again.",
        ),
        StateCatalogEntry(
            "cluster.presence_observations", "cluster-observation", "src/api_server.py:presence heartbeat",
            "rebuildable", "cluster", "memory_only", "rebuild_from_local_state",
            "rebuild_after_new_leader", False,
            "Presence is an observation source, not a grant of control authority.",
        ),
        StateCatalogEntry(
            "models.local_asset_catalog", "model-assets", "models/*.qlh-model-asset.json + src/model_config.py",
            "rebuildable", "node", "external_asset", "rebuild_from_local_assets",
            "rebuild_after_new_leader", False,
            "Re-scan local manifests; model weights and absolute paths never enter HA state.",
        ),
        StateCatalogEntry(
            "models.loaded_runtime_and_kv", "inference-runtime", "src/model_module.py + src/llama_engine.py",
            "must_not_migrate", "node", "memory_only", "never_copy",
            "invalidate_on_term_change", False,
            "Model objects, file handles, CUDA contexts and KV caches are node-local runtime state.",
        ),
        StateCatalogEntry(
            "model_download.jobs", "model-assets", "src/model_download_jobs.py + local_store model_download_jobs",
            "rebuildable", "node", "durable_local", "reconcile_then_retry",
            "rebuild_after_new_leader", False,
            "Interrupted downloads are reconciled against local files and manifests; they are not authority.",
        ),
        StateCatalogEntry(
            "auth.identities_and_totp", "security", "src/auth_store.py:auth_users/auth_totp",
            "must_replicate", "cluster", "durable_local", "journal_replay_before_write",
            "required_before_new_leader_write", True,
            "Replicate only through an encrypted protected state path; never expose raw secrets in a manifest.",
        ),
        StateCatalogEntry(
            "auth.sessions", "security", "src/auth_store.py:auth_sessions",
            "must_not_migrate", "node", "durable_local", "never_copy",
            "invalidate_on_term_change", True,
            "Invalidate sessions at leader change and require fresh authentication.",
        ),
        StateCatalogEntry(
            "governance.review_tickets", "governance", "src/local_store.py:review_tickets",
            "must_replicate", "cluster", "durable_local", "journal_replay_before_write",
            "required_before_new_leader_write", True,
            "Review decisions affect control actions and must remain auditable across a handoff.",
        ),
        StateCatalogEntry(
            "audit.control_events", "audit", "local audit/event logs",
            "audit_only", "cluster", "durable_local", "preserve_append_only",
            "retain_as_audit_evidence", True,
            "Evidence is retained and may be exported; it does not grant write authority.",
        ),
        StateCatalogEntry(
            "user.conversations", "user-data", "src/local_store.py:sessions/session_messages",
            "must_not_migrate", "user", "durable_local", "never_copy",
            "explicit_user_export_only", True,
            "User-owned conversations are outside HA control state and require explicit export.",
        ),
    )
    return StateCatalog(entries=tuple(entries))


def validate_state_catalog(value: StateCatalog | Mapping[str, Any]) -> StateCatalog:
    """Parse and validate a catalog before any future recovery code consumes it."""
    if isinstance(value, StateCatalog):
        return value
    return StateCatalog.from_dict(value)


__all__ = [
    "BOUNDARIES",
    "CATALOG_DOCUMENT_TYPE",
    "CATALOG_SCHEMA_VERSION",
    "HANDOFF_POLICIES",
    "RECOVERY_STRATEGIES",
    "STATE_CLASSES",
    "StateCatalog",
    "StateCatalogEntry",
    "StateCatalogError",
    "build_state_catalog",
    "validate_state_catalog",
]
