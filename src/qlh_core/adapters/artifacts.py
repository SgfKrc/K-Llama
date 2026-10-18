"""Byte inventory and zero-copy legacy import through explicit catalog ports."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Protocol

from qlh_core.contracts.catalog import (
    IMPORT_SCHEMA_VERSION, ArtifactLocation, ArtifactRef, CatalogError, DocumentPort,
    InventoryReceipt, ManifestEntry, ModelRevision, RegistryPublisher,
    ResolvedArtifact, bundle_sha256, canonical_digest, freeze, thaw,
)

_MAX_METADATA_BYTES = 64 * 1024 * 1024
_METADATA_NAMES = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
    "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "vocab.json",
    "merges.txt", "chat_template.jinja", "chat_template.json", "model.manifest.json",
    ".qlh-model-asset.json", "cross_format.provenance.json", "model.safetensors.index.json",
)
_RUNTIME_FIELDS = frozenset((
    "selected", "selected_model", "selected_model_id", "ready", "native", "native_ready",
    "is_loaded", "model_loaded", "active", "active_plan", "active_deployment", "deployment",
    "connection", "kv", "kv_cache", "lease", "runtime", "runtime_receipt",
))


def _hash_stream(handle) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        if not isinstance(chunk, bytes):
            raise CatalogError("artifact_read_failed", "document port must return binary data")
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def _stat_token(result) -> tuple:
    # Windows fstat/path.stat can expose different ctime semantics. Device,
    # file identity, size and last-write time are the portable mutation token.
    return (result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns)


def _hash_disk(path: Path) -> tuple[int, str]:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        result = _hash_stream(handle)
        after = os.fstat(handle.fileno())
    if _stat_token(before) != _stat_token(after) or _stat_token(after) != _stat_token(path.stat()):
        raise CatalogError("artifact_changed_during_verification", "artifact changed while hashing")
    return result


def _json_object(path: Path) -> dict:
    if path.stat().st_size > _MAX_METADATA_BYTES:
        raise CatalogError("artifact_metadata_invalid", "metadata exceeds size limit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CatalogError("artifact_metadata_invalid", "metadata must be an object")
    return payload


class InventoryPort(Protocol):
    def measure(self, artifact: ArtifactRef, location: ArtifactLocation) -> tuple[int, str]: ...


class DiskInventory:
    def measure(self, artifact: ArtifactRef, location: ArtifactLocation) -> tuple[int, str]:
        root = Path(location.locator).expanduser().resolve(strict=True)
        if not artifact.files:
            if not root.is_file():
                raise CatalogError("artifact_locator_invalid", "single artifact requires a file")
            if artifact.manifest_sha256:
                manifests = (Path(str(root) + ".manifest.json"), root.with_suffix(".manifest.json"))
                manifest = next((path for path in manifests if path.is_file()), None)
                if manifest is None or _hash_disk(manifest)[1] != artifact.manifest_sha256:
                    raise CatalogError("artifact_manifest_mismatch", "artifact manifest changed or is missing")
            return _hash_disk(root)
        if not root.is_dir():
            raise CatalogError("artifact_locator_invalid", "bundle requires a directory")
        if artifact.manifest_sha256:
            manifest = next((root / name for name in ("model.manifest.json", ".qlh-model-asset.json")
                             if (root / name).is_file()), None)
            if manifest is None:
                raise CatalogError("artifact_manifest_mismatch", "bundle manifest is missing")
            payload = _json_object(manifest)
            canonical = {key: value for key, value in payload.items()
                         if key not in ("artifact_sha256", "manifest_sha256")}
            if canonical_digest(canonical) != artifact.manifest_sha256:
                raise CatalogError("artifact_manifest_mismatch", "bundle manifest no longer matches registration")
        bindings = dict(location.external_files)
        measured = []
        for entry in artifact.files:
            if entry.path.startswith("external/"):
                if entry.path not in bindings:
                    raise CatalogError("artifact_external_binding_missing", "external manifest entry has no binding")
                path = Path(bindings[entry.path]).expanduser().resolve(strict=True)
            else:
                path = (root / entry.path).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise CatalogError("artifact_locator_invalid", "manifest resolves outside artifact root")
            size, digest = _hash_disk(path)
            if size != entry.size_bytes or digest != entry.sha256:
                raise CatalogError("artifact_digest_mismatch", "manifest file differs from registered bytes")
            measured.append(ManifestEntry(entry.path, size, digest))
        return sum(entry.size_bytes for entry in measured), bundle_sha256(measured)

    def validate_revision(self, revision: ModelRevision, artifact: ArtifactRef,
                          location: ArtifactLocation) -> None:
        if not artifact.files:
            return
        root = Path(location.locator).expanduser().resolve(strict=True)
        names = {entry.path: entry for entry in artifact.files}
        manifest_names = {"model.manifest.json", ".qlh-model-asset.json", "cross_format.provenance.json"}
        for name in _METADATA_NAMES:
            if name not in manifest_names and (root / name).is_file() and name not in names:
                raise CatalogError("artifact_metadata_unbound", "loader metadata is absent from frozen manifest")
        tokenizer = names.get("tokenizer.json")
        if tokenizer is not None and tokenizer.sha256 != revision.tokenizer_sha256:
            raise CatalogError("artifact_tokenizer_mismatch", "tokenizer differs from the frozen revision")
        if "chat_template.jinja" in names:
            template_digest = names["chat_template.jinja"].sha256
        else:
            config = (_json_object(root / "tokenizer_config.json")
                      if "tokenizer_config.json" in names else {})
            template_digest = canonical_digest(config.get("chat_template"))
        if template_digest != revision.template_sha256:
            raise CatalogError("artifact_template_mismatch", "template differs from the frozen revision")


class SafInventory:
    def __init__(self, documents: DocumentPort) -> None:
        self._documents = documents

    def measure(self, artifact: ArtifactRef, location: ArtifactLocation) -> tuple[int, str]:
        if not self._documents.validate_grant(location.grant, location.locator):
            raise CatalogError("artifact_saf_grant_invalid", "SAF persisted grant is unavailable")
        if not self._documents.validate_locator(location.locator):
            raise CatalogError("artifact_saf_locator_invalid", "SAF document locator is unavailable")
        if artifact.files:
            raise CatalogError("artifact_saf_bundle_unsupported", "SAF bundles require explicit document bindings")
        # A grant proves access only. Always open and hash the current document.
        with self._documents.open_binary(location.locator) as handle:
            return _hash_stream(handle)


class ArtifactStore:
    def __init__(self, *, disk: InventoryPort | None = None,
                 documents: DocumentPort | None = None, saf: InventoryPort | None = None) -> None:
        self._ports = {"disk": disk or DiskInventory()}
        if saf is not None:
            self._ports["saf"] = saf
        elif documents is not None:
            self._ports["saf"] = SafInventory(documents)
        self._inventory: dict[tuple, InventoryReceipt] = {}
        self._lock = threading.RLock()

    def _remember(self, receipt: InventoryReceipt) -> InventoryReceipt:
        with self._lock:
            self._inventory[(receipt.artifact.artifact_id, receipt.location)] = receipt
        return receipt

    def register(self, artifact: ArtifactRef, location: ArtifactLocation) -> InventoryReceipt:
        return self._remember(InventoryReceipt(artifact, location, "registered"))

    def inspect(self, artifact: ArtifactRef, location: ArtifactLocation) -> InventoryReceipt:
        """Read-only observation; dry-run never changes even the inventory cache."""
        port = self._ports.get(location.kind)
        try:
            if port is None:
                raise CatalogError("artifact_inventory_port_missing", "platform inventory port was not injected")
            size, digest = port.measure(artifact, location)
            if size != artifact.size_bytes or digest != artifact.content_sha256:
                return InventoryReceipt(artifact, location, "unavailable", "artifact_digest_mismatch", size, digest)
            return InventoryReceipt(artifact, location, "verified", measured_bytes=size, measured_sha256=digest)
        except CatalogError as exc:
            return InventoryReceipt(artifact, location, "unavailable", exc.code)
        except PermissionError:
            return InventoryReceipt(artifact, location, "unavailable", "artifact_access_denied")
        except FileNotFoundError:
            return InventoryReceipt(artifact, location, "unavailable", "artifact_missing")
        except (OSError, ValueError, UnicodeError):
            return InventoryReceipt(artifact, location, "unavailable", "artifact_read_failed")

    def verify(self, artifact: ArtifactRef, location: ArtifactLocation) -> InventoryReceipt:
        return self._remember(self.inspect(artifact, location))

    def inventory(self) -> tuple[InventoryReceipt, ...]:
        with self._lock:
            return tuple(self._inventory.values())

    def inspect_resolved(self, revision: ModelRevision, artifact: ArtifactRef,
                         location: ArtifactLocation) -> ResolvedArtifact:
        return self._resolve(revision, artifact, location, remember=False)

    def resolve(self, revision: ModelRevision, artifact: ArtifactRef,
                location: ArtifactLocation) -> ResolvedArtifact:
        return self._resolve(revision, artifact, location, remember=True)

    def _resolve(self, revision: ModelRevision, artifact: ArtifactRef,
                 location: ArtifactLocation, *, remember: bool) -> ResolvedArtifact:
        if not revision.admits(artifact):
            raise CatalogError("artifact_revision_mismatch", "artifact belongs to another model revision")
        receipt = self.verify(artifact, location) if remember else self.inspect(artifact, location)
        if receipt.state != "verified":
            raise CatalogError(receipt.reason_code, "artifact bytes are not verified")
        validator = getattr(self._ports.get(location.kind), "validate_revision", None)
        if validator is not None:
            validator(revision, artifact, location)
        return ResolvedArtifact(revision, artifact, location, receipt)


def _location(payload: Mapping) -> ArtifactLocation:
    values = thaw(freeze(payload))
    values["external_files"] = tuple(values.get("external_files", {}).items())
    return ArtifactLocation(**values)


def binding_from_payload(payload: Mapping) -> tuple[ModelRevision, ArtifactRef, ArtifactLocation]:
    binding = payload["resolved_artifact"]
    revision = ModelRevision.from_dict(binding["model_revision"])
    artifact = ArtifactRef.from_dict(binding["artifact"])
    location = _location(binding["location"])
    if revision.model_id != payload.get("model_id") or not revision.admits(artifact):
        raise CatalogError("artifact_revision_mismatch", "binding belongs to another model revision")
    return revision, artifact, location


def artifact_load_identity(path: str, *, model_id: str,
                           rows: Iterable[Mapping] | None = None) -> dict:
    """Bound metadata fingerprint for legacy loaders, without loading weights.

    Typed registrations retain the exact model/artifact identity. Full byte
    verification belongs to ArtifactStore.resolve at import/prepare boundaries;
    repeated status and reconnect fingerprints never scan weight streams.
    """
    root = Path(path)
    metadata_root = root if root.is_dir() else root.parent
    paths = [metadata_root / name for name in _METADATA_NAMES]
    if not root.is_dir():
        paths.extend((Path(str(root) + ".manifest.json"), root.with_suffix(".manifest.json")))
    metadata = {}
    for candidate in dict.fromkeys(paths):
        if not candidate.exists():
            continue
        try:
            if candidate.stat().st_size > _MAX_METADATA_BYTES:
                metadata[str(candidate)] = {"oversized": _stat_token(candidate.stat())}
            else:
                metadata[str(candidate)] = _hash_disk(candidate)[1]
        except (OSError, CatalogError):
            metadata[str(candidate)] = "unreadable"
    identity: dict[str, Any] = {"metadata_sha256": canonical_digest(metadata)}
    for row in rows or ():
        if not isinstance(row, Mapping) or row.get("model_id") != model_id or not row.get("resolved_artifact"):
            continue
        revision, artifact, location = binding_from_payload(row)
        normalized = os.path.normcase(str(root.resolve()))
        if location.kind != "disk" or os.path.normcase(str(Path(location.locator).resolve())) != normalized:
            raise CatalogError("artifact_locator_mismatch", "loader path differs from registered artifact")
        # Weight stat tokens invalidate the load fingerprint without reading
        # weights. The prepare guard alone decides whether new bytes are valid.
        file_stats = {}
        for entry in artifact.files:
            if entry.path.startswith("external/"):
                bound = dict(location.external_files).get(entry.path)
                candidate = Path(bound) if bound else None
            else:
                candidate = root / entry.path
            try:
                file_stats[entry.path] = _stat_token(candidate.stat()) if candidate else "unbound"
            except OSError:
                file_stats[entry.path] = "missing"
        identity.update({"artifact_file_stats_sha256": canonical_digest(file_stats),
                         "model_revision_id": revision.revision_id,
                         "artifact_id": artifact.artifact_id,
                         "content_sha256": artifact.content_sha256,
                         "tokenizer_sha256": revision.tokenizer_sha256,
                         "template_sha256": revision.template_sha256})
        break
    return identity


@dataclass(frozen=True)
class ImportReport:
    plan: Mapping
    entries: tuple[Mapping, ...]
    dry_run: bool
    schema_version: int = IMPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan", freeze(self.plan))
        object.__setattr__(self, "entries", tuple(freeze(entry) for entry in self.entries))

    @property
    def plan_sha256(self) -> str:
        return canonical_digest(self.plan)

    def _body(self) -> dict:
        entries = thaw(self.entries)
        return {"schema_version": self.schema_version, "dry_run": self.dry_run,
                "plan": thaw(self.plan), "plan_sha256": self.plan_sha256, "entries": entries,
                "impact": {"rows": len(entries),
                           "publish_candidates": sum(e["action"] == "publish" for e in entries),
                           "imported": sum(e.get("result") == "imported" for e in entries),
                           "rejected": sum(e["action"] == "reject" for e in entries),
                           "runtime_state_restored": False},
                "space_estimate": {
                    "artifact_copy_bytes": 0,
                    "reused_artifact_bytes": sum(e.get("artifact_bytes", 0) for e in entries
                                                 if e["action"] == "publish"),
                    "catalog_payload_bytes": sum(e.get("metadata_bytes", 0) for e in entries
                                                 if e["action"] == "publish"),
                    "database_overhead_bytes": None,
                    "database_overhead_reason": "repository_and_page_size_dependent"}}

    @property
    def report_sha256(self) -> str:
        return canonical_digest(self._body())

    def as_dict(self) -> dict:
        return {**self._body(), "report_sha256": self.report_sha256}

    def verify(self) -> bool:
        return verify_import_report(self.as_dict())


def verify_import_report(payload: Mapping) -> bool:
    """Integrity check for exported reports; hashes do not imply a trusted signature."""
    try:
        body = thaw(freeze(payload))
        expected = body.pop("report_sha256")
        return (type(body["schema_version"]) is int
                and body["schema_version"] == IMPORT_SCHEMA_VERSION
                and body["plan_sha256"] == canonical_digest(body["plan"])
                and expected == canonical_digest(body))
    except (KeyError, TypeError, ValueError):
        return False


def _strip_runtime(payload: dict) -> tuple[dict, tuple[str, ...]]:
    removed = []

    def clean(value, prefix=""):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key in _RUNTIME_FIELDS:
                    removed.append(path)
                else:
                    result[key] = clean(item, path)
            return result
        if isinstance(value, list):
            return [clean(item, f"{prefix}[{index}]") for index, item in enumerate(value)]
        return value

    return clean(payload), tuple(sorted(removed))


def _legacy_disk_binding(payload: dict, artifacts: ArtifactStore) -> tuple[ModelRevision, ArtifactRef, ArtifactLocation] | None:
    """Bind a v1 legacy manifest using live metadata, without copying an artifact.

    Missing semantic metadata leaves a registration unverified. Invalid declared
    metadata rejects just this row; no inferred conversion proof is manufactured.
    """
    raw = payload.get("model_path") or payload.get("gguf_path")
    if not raw:
        return None
    root = Path(raw).expanduser().resolve(strict=True)
    if not root.is_dir():
        return None
    manifest_path = next((root / name for name in ("model.manifest.json", ".qlh-model-asset.json")
                          if (root / name).is_file()), None)
    if manifest_path is None:
        return None
    manifest = _json_object(manifest_path)
    manifest_schema = manifest.get("schema", manifest.get("schema_version"))
    if type(manifest_schema) is not int or manifest_schema != 1:
        raise CatalogError("legacy_manifest_schema_unsupported", "unsupported legacy manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CatalogError("artifact_manifest_invalid", "legacy manifest has no file inventory")
    entries = tuple(ManifestEntry(str(entry["path"]), entry.get("size_bytes", entry.get("size")),
                                  str(entry["sha256"])) for entry in files)
    expected_digest = bundle_sha256(entries)
    declared_digest = manifest.get("artifact_sha256")
    if declared_digest and declared_digest != expected_digest:
        raise CatalogError("artifact_manifest_mismatch", "legacy aggregate digest is invalid")
    manifest_digest = canonical_digest({key: value for key, value in manifest.items()
                                        if key not in ("artifact_sha256", "manifest_sha256")})
    if manifest.get("manifest_sha256") and manifest["manifest_sha256"] != manifest_digest:
        raise CatalogError("artifact_manifest_mismatch", "legacy manifest digest is invalid")
    # Never promote an entry whose registered hashes have already drifted.
    bindings = tuple((str(key), str(value)) for key, value in
                     (payload.get("manifest_external_files") or {}).items())
    location = ArtifactLocation(str(root), external_files=bindings)
    temporary = ArtifactRef(expected_digest, "safetensors", "0" * 64,
                            sum(e.size_bytes for e in entries), manifest_digest, files=entries)
    measured = artifacts.inspect(temporary, location)
    if measured.state != "verified":
        raise CatalogError(measured.reason_code, "legacy bundle bytes no longer match")
    names = {entry.path: entry for entry in entries}
    if "config.json" not in names or "tokenizer.json" not in names:
        return None
    config = _json_object(root / "config.json")
    config = config.get("text_config", config)
    layers = config.get("num_hidden_layers", config.get("n_layer"))
    width = config.get("hidden_size", config.get("n_embd"))
    architecture = config.get("model_type", "")
    if not architecture or type(layers) is not int or type(width) is not int:
        return None
    tokenizer_config = {}
    if "tokenizer_config.json" in names:
        tokenizer_config = _json_object(root / "tokenizer_config.json")
    elif (root / "tokenizer_config.json").exists():
        return None  # An unbound chat template cannot become a frozen identity.
    if "chat_template.jinja" in names:
        template_digest = names["chat_template.jinja"].sha256
    elif (root / "chat_template.jinja").exists():
        return None
    else:
        template_digest = canonical_digest(tokenizer_config.get("chat_template"))
    source = payload.get("source")
    source = source if isinstance(source, dict) else {}
    upstream_revision = (source.get("resolved_revision") or payload.get("resolved_revision")
                         or payload.get("revision") or manifest.get("revision") or expected_digest)
    weights = tuple(entry for entry in entries if entry.path.endswith((".safetensors", ".bin")))
    if not weights:
        return None
    revision = ModelRevision(payload["model_id"], str(source.get("repo_id") or payload["model_id"]),
                             str(upstream_revision), str(architecture), layers, width,
                             bundle_sha256(weights), manifest_digest, names["tokenizer.json"].sha256,
                             template_digest)
    artifact = ArtifactRef(expected_digest, "safetensors", revision.revision_id,
                           temporary.size_bytes, manifest_digest,
                           str(payload.get("quantization", "") or ""), files=entries)
    return revision, artifact, location


class LegacyRegistryImporter:
    def __init__(self, catalog: RegistryPublisher, artifacts: ArtifactStore, *, resolve_legacy=None) -> None:
        self._catalog = catalog
        self._artifacts = artifacts
        self._resolve_legacy = resolve_legacy

    def _plan(self, rows: Iterable[Mapping | str], source_schema: int) -> ImportReport:
        if type(source_schema) is not int or source_schema != 1:
            raise CatalogError("legacy_registry_schema_unsupported", "unsupported legacy registry schema; source preserved")
        raw_rows = list(rows)
        sources = []
        entries = []
        seen = set()
        for index, raw in enumerate(raw_rows):
            entry = {"index": index, "model_id": "", "action": "reject", "state": "unavailable",
                     "reason_code": "legacy_registry_row_invalid", "artifact_bytes": 0,
                     "metadata_bytes": 0, "ignored_runtime_fields": []}
            try:
                source = json.loads(raw) if isinstance(raw, str) else thaw(freeze(raw))
                sources.append(source)
                if not isinstance(source, dict):
                    raise CatalogError("legacy_registry_row_invalid", "legacy row must be an object")
                # SQLite reader ports may preserve a corrupt payload as a row.
                if "payload" in source and "model_id" in source:
                    declared_id = source["model_id"]
                    source = json.loads(source["payload"]) if isinstance(source["payload"], str) else source["payload"]
                    if not isinstance(source, dict) or source.get("model_id", declared_id) != declared_id:
                        raise CatalogError("legacy_registry_row_invalid", "legacy row identity mismatch")
                    source = {**source, "model_id": declared_id}
                model_id = source.get("model_id")
                if not isinstance(model_id, str) or not model_id.strip():
                    raise CatalogError("catalog_model_id_invalid", "legacy row has no model id")
                entry["model_id"] = model_id
                if model_id in seen:
                    raise CatalogError("legacy_registry_duplicate", "duplicate legacy model id")
                row_schema = source.get("schema", source.get("schema_version", 1))
                if type(row_schema) is not int or row_schema != 1:
                    raise CatalogError("legacy_registry_row_schema_unsupported", "unsupported row schema")
                payload, removed = _strip_runtime(source)
                entry["ignored_runtime_fields"] = list(removed)
                if payload.get("resolved_artifact"):
                    binding = binding_from_payload(payload)
                elif self._resolve_legacy is not None:
                    resolved = self._resolve_legacy(payload)
                    if resolved is not None and not isinstance(resolved, ResolvedArtifact):
                        raise CatalogError("artifact_resolution_invalid", "legacy resolver must return ResolvedArtifact")
                    binding = ((resolved.revision, resolved.artifact, resolved.location)
                               if resolved is not None else None)
                else:
                    binding = _legacy_disk_binding(payload, self._artifacts)
                if binding is not None:
                    revision, artifact, location = binding
                    if revision.model_id != model_id:
                        raise CatalogError("artifact_revision_mismatch", "legacy resolver returned another model")
                    resolved = self._artifacts.inspect_resolved(revision, artifact, location)
                    payload["resolved_artifact"] = resolved.as_dict()
                    entry.update(state="verified", artifact_bytes=artifact.size_bytes,
                                 reason_code="artifact_verified")
                else:
                    entry.update(state="registered", reason_code="artifact_semantic_identity_missing")
                    raw_path = payload.get("model_path") or payload.get("gguf_path")
                    if raw_path:
                        path = Path(raw_path).expanduser()
                        if not path.exists():
                            entry["reason_code"] = "artifact_missing"
                        elif path.is_file():
                            entry["artifact_bytes"] = path.stat().st_size
                current = self._catalog.snapshot(model_id)
                entry.update(payload=payload, expected_revision=current.registry_revision,
                             metadata_bytes=len(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")))
                if current.payload is not None:
                    entry.update(action="skip", reason_code=("catalog_already_imported"
                                 if current.as_payload() == payload else "catalog_destination_exists"))
                else:
                    entry["action"] = "publish"
                seen.add(model_id)
            except (CatalogError, KeyError, TypeError, ValueError, OSError) as exc:
                if len(sources) <= index:
                    sources.append({"invalid_raw": raw if isinstance(raw, str) else repr(raw)})
                entry["reason_code"] = getattr(exc, "code", "legacy_registry_row_invalid")
            entries.append(entry)
        plan = {"source_schema": source_schema, "target_schema": IMPORT_SCHEMA_VERSION,
                "source_sha256": canonical_digest(sources), "entries": entries,
                "zero_copy": True, "runtime_state_restored": False}
        return ImportReport(plan, tuple(entries), True)

    def dry_run(self, rows: Iterable[Mapping | str], *, source_schema: int = 1) -> ImportReport:
        return self._plan(rows, source_schema)

    def import_registry(self, rows: Iterable[Mapping | str], *, source_schema: int = 1,
                        expected_report: ImportReport | Mapping | None = None) -> ImportReport:
        report = self._plan(rows, source_schema)
        if expected_report is not None:
            exported = expected_report.as_dict() if isinstance(expected_report, ImportReport) else expected_report
            if not verify_import_report(exported) or exported["plan_sha256"] != report.plan_sha256:
                raise CatalogError("legacy_import_plan_stale", "dry-run report or source/target inventory changed")
        results = []
        for planned in report.entries:
            result = thaw(planned)
            if result["action"] == "publish":
                try:
                    if self._catalog.publish(result["model_id"], result["payload"], result["expected_revision"]):
                        result["result"] = "imported"
                        # Only a post-publication local inventory observation is cached.
                        if result["payload"].get("resolved_artifact"):
                            _, artifact, location = binding_from_payload(result["payload"])
                            self._artifacts.verify(artifact, location)
                    else:
                        result.update(result="conflict", reason_code="catalog_revision_conflict")
                except (CatalogError, OSError, ValueError) as exc:
                    result.update(result="failed", reason_code=getattr(exc, "code", "catalog_import_failed"))
            else:
                result["result"] = "skipped"
            results.append(result)
        return ImportReport(report.plan, tuple(results), False)


class CatalogLoadGuard:
    """Inject into native load/prepare boundaries; identity polling stays read-only.

    configure_catalog(CatalogLoadGuard(catalog, artifacts)) binds this port to a
    manager. GGUF factories additionally supply model_id before their first load.
    """
    def __init__(self, catalog: RegistryPublisher, artifacts: ArtifactStore) -> None:
        self._catalog = catalog
        self._artifacts = artifacts

    def __call__(self, model_id: str, model_path: str) -> ResolvedArtifact | None:
        snapshot = self._catalog.snapshot(model_id)
        payload = snapshot.as_payload()
        resolved = verify_registered_artifact(model_id, model_path, rows=[payload] if payload else (),
                                              artifacts=self._artifacts)
        if resolved is not None:
            current = self._catalog.snapshot(model_id)
            if current.registry_revision != snapshot.registry_revision or current.payload != snapshot.payload:
                raise CatalogError("catalog_revision_conflict", "registration changed while preparing artifact")
        return resolved


def verify_registered_artifact(model_id: str, model_path: str, *, rows=(),
                               artifacts: ArtifactStore | None = None) -> ResolvedArtifact | None:
    for row in rows or ():
        if not isinstance(row, Mapping) or row.get("model_id") != model_id:
            continue
        if row.get("resolved_artifact") is None:
            return None
        revision, artifact, location = binding_from_payload(row)
        if (location.kind != "disk" or os.path.normcase(os.path.realpath(location.locator))
                != os.path.normcase(os.path.realpath(model_path))):
            raise CatalogError("artifact_locator_mismatch", "prepare path differs from frozen registration")
        return (artifacts or ArtifactStore()).resolve(revision, artifact, location)
    return None
