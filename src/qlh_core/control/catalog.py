"""Catalog is the only owner of registry publication and revision advancement."""
from __future__ import annotations

from collections.abc import Mapping
from qlh_core.contracts.catalog import (
    ArtifactLocation, ArtifactRef, CatalogError, CatalogRepository, ModelRevision,
    RegistrySnapshot, ResolvedArtifact, freeze, thaw, require_revision,
)


class CatalogService:
    def __init__(self, store: CatalogRepository) -> None:
        self._store = store

    def snapshot(self, model_id: str) -> RegistrySnapshot:
        result = self._store.snapshot(model_id)
        if not isinstance(result, RegistrySnapshot) or result.model_id != model_id:
            raise CatalogError("catalog_snapshot_invalid", "repository returned another identity")
        return result

    read = snapshot

    @staticmethod
    def validate_payload(model_id: str, payload: Mapping) -> dict:
        if not isinstance(model_id, str) or not model_id.strip():
            raise CatalogError("catalog_model_id_invalid", "model_id is required")
        if not isinstance(payload, Mapping):
            raise CatalogError("catalog_payload_invalid", "registry payload must be an object")
        normalized = thaw(freeze(payload))
        declared = normalized.get("model_id", model_id)
        if declared != model_id:
            raise CatalogError("catalog_model_id_mismatch", "payload belongs to another model")
        normalized["model_id"] = model_id
        resolved = normalized.get("resolved_artifact")
        if resolved is not None:
            if not isinstance(resolved, Mapping):
                raise CatalogError("catalog_payload_invalid", "resolved_artifact must be an object")
            try:
                revision = ModelRevision.from_dict(resolved["model_revision"])
                artifact = ArtifactRef.from_dict(resolved["artifact"])
                location = dict(resolved["location"])
                location["external_files"] = tuple(location.get("external_files", {}).items())
                ArtifactLocation(**location)
            except (KeyError, TypeError) as exc:
                raise CatalogError("catalog_payload_invalid", "invalid resolved_artifact") from exc
            if revision.model_id != model_id or not revision.admits(artifact):
                raise CatalogError("artifact_revision_mismatch", "artifact must bind the catalog revision")
        return normalized

    def publish(self, model_id: str, payload: Mapping, expected_revision: int) -> bool:
        require_revision(expected_revision)
        normalized = self.validate_payload(model_id, payload)
        return self._store.compare_and_swap(model_id, normalized,
                                            expected_revision=expected_revision,
                                            next_revision=expected_revision + 1)

    def remove(self, model_id: str, expected_revision: int) -> bool:
        require_revision(expected_revision)
        if not isinstance(model_id, str) or not model_id.strip():
            raise CatalogError("catalog_model_id_invalid", "model_id is required")
        return self._store.remove(model_id, expected_revision=expected_revision,
                                  next_revision=expected_revision + 1)


    def publish_resolved(self, model_id: str, resolved: ResolvedArtifact,
                         expected_revision: int, *, payload: Mapping | None = None) -> bool:
        if not isinstance(resolved, ResolvedArtifact) or resolved.revision.model_id != model_id:
            raise CatalogError("artifact_revision_mismatch", "resolved artifact belongs to another model")
        values = thaw(freeze(payload or {}))
        values["resolved_artifact"] = resolved.as_dict()
        return self.publish(model_id, values, expected_revision)

    def resolve(self, model_id: str, *, resolver) -> ResolvedArtifact:
        """Revalidate bytes through an injected inventory port before admitting use.

        A durable binding records registration, never native readiness or a live
        verification receipt. resolver is ArtifactStore.resolve (or a platform port).
        """
        payload = self.snapshot(model_id).as_payload() or {}
        binding = payload.get("resolved_artifact")
        if not isinstance(binding, dict):
            raise CatalogError("artifact_not_bound", "model has no frozen artifact binding")
        revision = ModelRevision.from_dict(binding["model_revision"])
        artifact = ArtifactRef.from_dict(binding["artifact"])
        location = dict(binding["location"])
        location["external_files"] = tuple(location.get("external_files", {}).items())
        result = resolver(revision, artifact, ArtifactLocation(**location))
        if (not isinstance(result, ResolvedArtifact) or result.revision != revision
                or result.artifact != artifact or result.location != ArtifactLocation(**location)):
            raise CatalogError("artifact_resolution_invalid", "resolver returned another artifact")
        return result
