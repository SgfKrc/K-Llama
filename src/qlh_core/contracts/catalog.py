"""Immutable catalog identities and ports. Importing this module performs no IO."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, BinaryIO, Protocol

CATALOG_SCHEMA_VERSION = 1
IMPORT_SCHEMA_VERSION = 1


class CatalogError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CatalogError("catalog_payload_invalid", "JSON object keys must be strings")
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise CatalogError("catalog_payload_invalid", "payload must contain finite JSON values")


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(thaw(freeze(value)), sort_keys=True,
                                    ensure_ascii=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def require_digest(value: str, *, optional: bool = False) -> None:
    if optional and value == "":
        return
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CatalogError("catalog_digest_invalid", "expected a lowercase SHA-256 digest")


def require_revision(value: int) -> None:
    if type(value) is not int or value < 0:
        raise CatalogError("catalog_revision_invalid", "revision must be a nonnegative integer")


@dataclass(frozen=True)
class RegistrySnapshot:
    model_id: str
    registry_revision: int
    payload: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        require_revision(self.registry_revision)
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise CatalogError("catalog_model_id_invalid", "model_id is required")
        if self.payload is not None:
            if not isinstance(self.payload, Mapping):
                raise CatalogError("catalog_payload_invalid", "payload must be an object")
            object.__setattr__(self, "payload", freeze(self.payload))

    def as_payload(self) -> dict | None:
        return thaw(self.payload)


class CatalogRepository(Protocol):
    def snapshot(self, model_id: str) -> RegistrySnapshot: ...
    def compare_and_swap(self, model_id: str, payload: dict, *,
                         expected_revision: int, next_revision: int) -> bool: ...
    def remove(self, model_id: str, *, expected_revision: int,
               next_revision: int) -> bool: ...


@dataclass(frozen=True)
class EquivalenceProof:
    """An explicit verified conversion statement, never a shared file digest.

    evidence_sha256 names the independently checked conversion/provenance record.
    Both endpoints keep their own byte digest even when this proof admits relay.
    """
    source_sha256: str
    target_sha256: str
    source_format: str
    target_format: str
    source_revision: str
    tokenizer_sha256: str
    template_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        for value in (self.source_sha256, self.target_sha256, self.tokenizer_sha256,
                      self.template_sha256, self.evidence_sha256):
            require_digest(value)
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.source_format, self.target_format, self.source_revision)):
            raise CatalogError("catalog_proof_invalid", "proof requires formats and source revision")
        if self.source_format == self.target_format:
            raise CatalogError("catalog_proof_invalid", "cross-format proof requires different formats")

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class ModelRevision:
    model_id: str
    lineage: str
    source_revision: str
    architecture: str
    total_layers: int
    hidden_size: int
    source_weights_sha256: str
    manifest_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    equivalence_proofs: tuple[EquivalenceProof, ...] = ()
    schema_version: int = CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != CATALOG_SCHEMA_VERSION:
            raise CatalogError("catalog_schema_unsupported", "unsupported model revision schema")
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.model_id, self.lineage, self.source_revision, self.architecture)):
            raise CatalogError("catalog_model_revision_invalid", "model lineage and revision are required")
        if any(type(value) is not int or value <= 0 for value in (self.total_layers, self.hidden_size)):
            raise CatalogError("catalog_model_revision_invalid", "model dimensions must be positive")
        for value in (self.source_weights_sha256, self.manifest_sha256,
                      self.tokenizer_sha256, self.template_sha256):
            require_digest(value)
        proofs = tuple(self.equivalence_proofs)
        for proof in proofs:
            if (not isinstance(proof, EquivalenceProof)
                    or proof.source_revision != self.source_revision
                    or proof.tokenizer_sha256 != self.tokenizer_sha256
                    or proof.template_sha256 != self.template_sha256):
                raise CatalogError("catalog_proof_mismatch", "proof must bind the frozen semantic revision")
        object.__setattr__(self, "equivalence_proofs", proofs)

    def as_dict(self) -> dict:
        return {**{key: getattr(self, key) for key in self.__dataclass_fields__
                   if key != "equivalence_proofs"},
                "equivalence_proofs": [proof.as_dict() for proof in self.equivalence_proofs]}

    @classmethod
    def from_dict(cls, payload: Mapping) -> ModelRevision:
        values = thaw(freeze(payload))
        values["equivalence_proofs"] = tuple(EquivalenceProof(**proof) for proof in
                                              values.get("equivalence_proofs", ()))
        return cls(**values)

    @property
    def revision_id(self) -> str:
        return canonical_digest(self.as_dict())

    def admits(self, artifact: ArtifactRef) -> bool:
        return (artifact.model_revision == self.revision_id
                and (artifact.layer_range is None or artifact.layer_range[1] <= self.total_layers))

    def proves_equivalent(self, left: ArtifactRef, right: ArtifactRef) -> bool:
        if not self.admits(left) or not self.admits(right):
            return False
        if left.layer_range != right.layer_range or left.components != right.components:
            return False
        if left.format == right.format:
            return left == right
        endpoints = {(left.content_sha256, left.format), (right.content_sha256, right.format)}
        return any(endpoints == {(proof.source_sha256, proof.source_format),
                                 (proof.target_sha256, proof.target_format)}
                   for proof in self.equivalence_proofs)


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        # Manifest names are portable relative POSIX paths. Windows drive/UNC
        # paths and traversal are rejected even on an Android/Linux host.
        if (not isinstance(self.path, str) or not self.path or self.path.startswith("/")
                or "\\" in self.path or ":" in self.path
                or any(part in ("", ".", "..") for part in self.path.split("/"))):
            raise CatalogError("artifact_locator_invalid", "manifest path must be a relative file name")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise CatalogError("artifact_size_invalid", "file size must be nonnegative")
        require_digest(self.sha256)

    def as_dict(self) -> dict:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


def bundle_sha256(entries: Sequence[ManifestEntry]) -> str:
    return hashlib.sha256("\n".join(
        f"{entry.path}\0{entry.size_bytes}\0{entry.sha256}"
        for entry in sorted(entries, key=lambda entry: entry.path)
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    content_sha256: str
    format: str
    model_revision: str
    size_bytes: int
    manifest_sha256: str = ""
    quantization: str = ""
    layer_range: tuple[int, int] | None = None
    components: tuple[str, ...] = ("weights",)
    files: tuple[ManifestEntry, ...] = ()

    def __post_init__(self) -> None:
        require_digest(self.content_sha256)
        require_digest(self.model_revision)
        require_digest(self.manifest_sha256, optional=True)
        if not isinstance(self.format, str) or not self.format.strip():
            raise CatalogError("artifact_format_invalid", "artifact format is required")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise CatalogError("artifact_size_invalid", "artifact size must be nonnegative")
        if self.layer_range is not None:
            pair = tuple(self.layer_range)
            if (len(pair) != 2 or any(type(n) is not int for n in pair)
                    or not 0 <= pair[0] < pair[1]):
                raise CatalogError("artifact_range_invalid", "layer range must be nonempty and half open")
            object.__setattr__(self, "layer_range", pair)
        if not isinstance(self.quantization, str):
            raise CatalogError("artifact_quantization_invalid", "quantization must be a string")
        if (not isinstance(self.components, (tuple, list)) or not self.components
                or any(not isinstance(item, str) or not item for item in self.components)):
            raise CatalogError("artifact_components_invalid", "components must be a sequence of names")
        components = tuple(sorted(set(self.components)))
        if not components or any(not isinstance(item, str) or not item for item in components):
            raise CatalogError("artifact_components_invalid", "artifact components are required")
        object.__setattr__(self, "components", components)
        if (not isinstance(self.files, (tuple, list))
                or any(not isinstance(item, ManifestEntry) for item in self.files)):
            raise CatalogError("artifact_manifest_invalid", "files must contain ManifestEntry values")
        files = tuple(sorted(self.files, key=lambda item: item.path))
        if len({entry.path for entry in files}) != len(files):
            raise CatalogError("artifact_manifest_invalid", "manifest contains duplicate paths")
        if files and (sum(entry.size_bytes for entry in files) != self.size_bytes
                      or bundle_sha256(files) != self.content_sha256):
            raise CatalogError("artifact_manifest_invalid", "bundle digest/size does not match its manifest")
        object.__setattr__(self, "files", files)

    def as_dict(self) -> dict:
        return {"content_sha256": self.content_sha256, "format": self.format,
                "model_revision": self.model_revision, "size_bytes": self.size_bytes,
                "manifest_sha256": self.manifest_sha256, "quantization": self.quantization,
                "layer_range": list(self.layer_range) if self.layer_range else None,
                "components": list(self.components), "files": [entry.as_dict() for entry in self.files]}

    @classmethod
    def from_dict(cls, payload: Mapping) -> ArtifactRef:
        values = thaw(freeze(payload))
        files = values.get("files", ())
        if (not isinstance(files, (list, tuple))
                or any(not isinstance(entry, Mapping) for entry in files)):
            raise CatalogError("artifact_manifest_invalid", "files must be an array of manifest entries")
        try:
            values["files"] = tuple(ManifestEntry(**entry) for entry in files)
            return cls(**values)
        except TypeError as exc:
            raise CatalogError("artifact_manifest_invalid", "invalid artifact fields") from exc

    @property
    def artifact_id(self) -> str:
        return canonical_digest(self.as_dict())


@dataclass(frozen=True)
class ArtifactLocation:
    locator: str
    kind: str = "disk"
    grant: str = ""
    external_files: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("disk", "saf") or not isinstance(self.locator, str) or not self.locator:
            raise CatalogError("artifact_locator_invalid", "disk path or SAF document locator required")
        bindings = tuple(sorted(tuple(pair) for pair in self.external_files))
        if len({pair[0] for pair in bindings}) != len(bindings):
            raise CatalogError("artifact_locator_invalid", "duplicate external bindings")
        for name, locator in bindings:
            ManifestEntry(name, 0, "0" * 64)
            if not name.startswith("external/") or not isinstance(locator, str) or not locator:
                raise CatalogError("artifact_locator_invalid", "external binding must be explicit")
        object.__setattr__(self, "external_files", bindings)


@dataclass(frozen=True)
class InventoryReceipt:
    artifact: ArtifactRef
    location: ArtifactLocation
    state: str
    reason_code: str = ""
    measured_bytes: int = 0
    measured_sha256: str = ""

    def __post_init__(self) -> None:
        if self.state not in ("registered", "verified", "unavailable"):
            raise CatalogError("artifact_inventory_invalid", "unknown inventory state")
        if self.state == "verified" and (self.measured_sha256 != self.artifact.content_sha256
                                         or self.measured_bytes != self.artifact.size_bytes):
            raise CatalogError("artifact_inventory_invalid", "verified receipt must match exact bytes")


@dataclass(frozen=True)
class ResolvedArtifact:
    revision: ModelRevision
    artifact: ArtifactRef
    location: ArtifactLocation
    receipt: InventoryReceipt

    def __post_init__(self) -> None:
        if (not self.revision.admits(self.artifact)
                or self.receipt.state != "verified"
                or self.receipt.artifact != self.artifact
                or self.receipt.location != self.location):
            raise CatalogError("artifact_resolution_invalid", "resolution requires matching revision and verified bytes")

    def as_dict(self) -> dict:
        return {"model_revision": self.revision.as_dict(), "artifact": self.artifact.as_dict(),
                "location": {"kind": self.location.kind, "locator": self.location.locator,
                             "grant": self.location.grant,
                             "external_files": dict(self.location.external_files)}}


class DocumentPort(Protocol):
    """Platform owns SAF grant and locator semantics; authorization is not a hash."""
    def validate_grant(self, grant: str, locator: str) -> bool: ...
    def validate_locator(self, locator: str) -> bool: ...
    def open_binary(self, locator: str) -> BinaryIO: ...


class RegistryPublisher(Protocol):
    def snapshot(self, model_id: str) -> RegistrySnapshot: ...
    def publish(self, model_id: str, payload: dict, expected_revision: int) -> bool: ...


def registry_load_identity(model_id: str, rows: Sequence[Mapping] | None) -> dict[str, str]:
    """Legacy load fingerprints use the same immutable catalog snapshot contract.

    Builtin names have no privilege: an explicit registry override is always bound.
    This fingerprint does not claim that locally registered bytes are verified.
    """
    for row in rows or ():
        if not isinstance(row, Mapping) or row.get("model_id") != model_id:
            continue
        snapshot = RegistrySnapshot(model_id, getattr(row, "registry_revision", 0), row)
        payload = snapshot.as_payload()
        identity = {"payload_sha256": canonical_digest(payload),
                    "registry_revision": str(snapshot.registry_revision)}
        for key in ("artifact_id", "sha256", "model_sha256", "artifact_sha256", "manifest_sha256",
                    "revision", "resolved_revision", "commit_hash", "quantization",
                    "tokenizer_sha256", "template_sha256", "chat_template_sha256",
                    "cross_format_provenance_sha256"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                identity[key] = value.strip()
        source = payload.get("source")
        if isinstance(source, dict):
            for key in ("provider", "repo_id", "requested_revision", "resolved_revision"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    identity[f"source_{key}"] = value.strip()
        resolved = payload.get("resolved_artifact")
        if isinstance(resolved, dict):
            revision = ModelRevision.from_dict(resolved["model_revision"])
            artifact = ArtifactRef.from_dict(resolved["artifact"])
            if not revision.admits(artifact):
                raise CatalogError("artifact_revision_mismatch", "registered artifact belongs to another revision")
            identity["model_revision_id"] = revision.revision_id
            identity["artifact_id"] = artifact.artifact_id
        return identity
    return {}
