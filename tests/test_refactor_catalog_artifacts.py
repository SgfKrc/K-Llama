"""R2 catalog identity, byte inventory and read-only import contracts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from qlh_core.adapters.artifacts import (
    ArtifactStore, LegacyRegistryImporter, artifact_load_identity, verify_import_report,
)
from qlh_core.contracts.catalog import (
    ArtifactLocation, ArtifactRef, CatalogError, EquivalenceProof, InventoryReceipt,
    ManifestEntry, ModelRevision, RegistrySnapshot, ResolvedArtifact,
    bundle_sha256, canonical_digest,
)
from qlh_core.control.catalog import CatalogService

pytestmark = pytest.mark.refactor_contract


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def revision(**changes) -> ModelRevision:
    values = dict(model_id="fixture", lineage="org/fixture", source_revision="commit-1",
                  architecture="qwen2", total_layers=4, hidden_size=8,
                  source_weights_sha256="a" * 64, manifest_sha256="b" * 64,
                  tokenizer_sha256="c" * 64, template_sha256="d" * 64)
    return ModelRevision(**{**values, **changes})


class MemoryRepository:
    def __init__(self):
        self.rows = {}
        self.versions = {}
        self.writes = []
        self.lock = threading.RLock()

    def snapshot(self, model_id):
        with self.lock:
            return RegistrySnapshot(model_id, self.versions.get(model_id, 0), self.rows.get(model_id))

    def compare_and_swap(self, model_id, payload, *, expected_revision, next_revision):
        with self.lock:
            if self.versions.get(model_id, 0) != expected_revision:
                return False
            assert next_revision == expected_revision + 1
            self.rows[model_id] = deepcopy(payload)
            self.versions[model_id] = next_revision
            self.writes.append((model_id, next_revision))
            return True

    def remove(self, model_id, *, expected_revision, next_revision):
        with self.lock:
            if self.versions.get(model_id, 0) != expected_revision or model_id not in self.rows:
                return False
            assert next_revision == expected_revision + 1
            del self.rows[model_id]
            self.versions[model_id] = next_revision
            self.writes.append((model_id, next_revision))
            return True


@pytest.fixture
def catalog():
    return CatalogService(MemoryRepository())


def bound_file(tmp_path, *, model_id="fixture", data=b"GGUF fixture"):
    path = tmp_path / (model_id + ".gguf")
    path.write_bytes(data)
    model = revision(model_id=model_id)
    artifact = ArtifactRef(sha(data), "gguf", model.revision_id, len(data), layer_range=(0, 2))
    location = ArtifactLocation(str(path))
    resolved = ArtifactStore().resolve(model, artifact, location)
    return path, resolved, {"model_id": model_id, "gguf_path": str(path),
                            "resolved_artifact": resolved.as_dict()}


def legacy_bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    values = {"config.json": json.dumps({"model_type": "qwen2", "num_hidden_layers": 4,
                                         "hidden_size": 8}).encode(),
              "tokenizer.json": b'{"version":1}',
              "tokenizer_config.json": b'{"chat_template":"hello {{ messages }}"}',
              "model.safetensors": b"weights"}
    entries = []
    for name, data in values.items():
        (root / name).write_bytes(data)
        entries.append(ManifestEntry(name, len(data), sha(data)))
    entries.sort(key=lambda e: e.path)
    manifest = {"schema": 1, "model_type": "safetensors", "revision": "commit-1",
                "source": "fixture", "files": [entry.as_dict() for entry in entries]}
    manifest["manifest_sha256"] = canonical_digest(manifest)
    manifest["artifact_sha256"] = bundle_sha256(entries)
    (root / "model.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, {"model_id": "fixture", "model_path": str(root), "model_type": "safetensors"}


def test_snapshot_is_deeply_frozen_and_caller_owned(catalog):
    payload = {"model_id": "fixture", "source": {"revision": "one"}, "tags": ["a"]}
    assert catalog.publish("fixture", payload, 0)
    payload["source"]["revision"] = "mutated"
    snapshot = catalog.snapshot("fixture")
    assert snapshot.registry_revision == 1
    assert snapshot.payload["source"]["revision"] == "one"
    with pytest.raises(TypeError):
        snapshot.payload["source"]["revision"] = "bad"
    with pytest.raises(FrozenInstanceError):
        snapshot.registry_revision = 5
    exported = snapshot.as_payload()
    exported["tags"].append("new")
    assert snapshot.as_payload()["tags"] == ["a"]


def test_owner_advances_versions_and_delete_recreate_cannot_aba(catalog):
    initial = catalog.snapshot("fixture")
    payload = {"model_id": "fixture", "name": "same bytes"}
    assert catalog.publish("fixture", payload, initial.registry_revision)
    old = catalog.snapshot("fixture")
    assert catalog.remove("fixture", old.registry_revision)
    tombstone = catalog.snapshot("fixture")
    assert tombstone.registry_revision == 2 and tombstone.payload is None
    assert catalog.publish("fixture", payload, tombstone.registry_revision)
    assert not catalog.publish("fixture", payload, old.registry_revision)
    assert not catalog.remove("fixture", old.registry_revision)
    assert catalog.snapshot("fixture").registry_revision == 3


def test_two_owners_share_cas_instead_of_process_local_version(catalog):
    other = CatalogService(catalog._store)
    barrier = threading.Barrier(2)

    def publish(service):
        expected = service.snapshot("fixture")
        barrier.wait(timeout=5)
        return service.publish("fixture", {"name": "candidate"}, expected.registry_revision)

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(publish, (catalog, other))) == [False, True]
    assert catalog.snapshot("fixture").registry_revision == 1


@pytest.mark.parametrize("payload,expected,code", [
    ({"model_id": "another"}, 0, "catalog_model_id_mismatch"),
    ({"value": float("nan")}, 0, "catalog_payload_invalid"),
    ({}, True, "catalog_revision_invalid"),
    ({}, -1, "catalog_revision_invalid"),
])
def test_invalid_publication_never_reaches_repository(catalog, payload, expected, code):
    with pytest.raises(CatalogError) as error:
        catalog.publish("fixture", payload, expected)
    assert error.value.code == code
    assert catalog._store.writes == []


@pytest.mark.parametrize("field,value", [
    ("source_revision", "commit-2"), ("manifest_sha256", "1" * 64),
    ("tokenizer_sha256", "2" * 64), ("template_sha256", "3" * 64),
    ("source_weights_sha256", "4" * 64), ("hidden_size", 16),
])
def test_every_semantic_revision_component_changes_identity(field, value):
    model = revision()
    assert replace(model, **{field: value}).revision_id != model.revision_id


def test_different_sha_cross_format_requires_exact_explicit_proof():
    model = revision()
    left = ArtifactRef("1" * 64, "safetensors", model.revision_id, 8)
    right = ArtifactRef("2" * 64, "gguf", model.revision_id, 4)
    assert not model.proves_equivalent(left, right)
    proof = EquivalenceProof(left.content_sha256, right.content_sha256, "safetensors", "gguf",
                             model.source_revision, model.tokenizer_sha256, model.template_sha256,
                             "9" * 64)
    proven = replace(model, equivalence_proofs=[proof])
    left = replace(left, model_revision=proven.revision_id)
    right = replace(right, model_revision=proven.revision_id)
    assert proven.proves_equivalent(left, right)
    assert left.content_sha256 != right.content_sha256
    assert not proven.proves_equivalent(left, replace(right, content_sha256="3" * 64))
    assert not proven.proves_equivalent(left, replace(right, layer_range=(1, 3)))
    with pytest.raises(CatalogError, match="semantic revision"):
        replace(proven, tokenizer_sha256="5" * 64)


@pytest.mark.parametrize("name", ["../weights", "/weights", "C:/weights", "a\\b", "a/./b", "a//b"])
def test_manifest_paths_cannot_escape_root_on_any_platform(name):
    with pytest.raises(CatalogError):
        ManifestEntry(name, 1, "a" * 64)


def test_registered_is_distinct_from_verified_and_changed_bytes_revoke_inventory(tmp_path):
    path, resolved, _ = bound_file(tmp_path)
    store = ArtifactStore()
    assert store.register(resolved.artifact, resolved.location).state == "registered"
    assert store.verify(resolved.artifact, resolved.location).state == "verified"
    path.write_bytes(b"corrupted")
    receipt = store.verify(resolved.artifact, resolved.location)
    assert receipt.state == "unavailable" and receipt.reason_code == "artifact_digest_mismatch"
    assert store.inventory() == (receipt,)
    with pytest.raises(CatalogError) as error:
        store.resolve(resolved.revision, resolved.artifact, resolved.location)
    assert error.value.code == "artifact_digest_mismatch"


def test_resolved_artifact_cannot_be_created_from_registered_receipt(tmp_path):
    _, resolved, _ = bound_file(tmp_path)
    registered = InventoryReceipt(resolved.artifact, resolved.location, "registered")
    with pytest.raises(CatalogError):
        replace(resolved, receipt=registered)
    with pytest.raises(CatalogError):
        replace(resolved, revision=replace(resolved.revision, source_revision="new"))


def test_catalog_publish_resolved_then_resolve_verifies_current_bytes(catalog, tmp_path):
    path, resolved, _ = bound_file(tmp_path)
    assert catalog.publish_resolved("fixture", resolved, 0, payload={"name": "display"})
    assert catalog.resolve("fixture", resolver=ArtifactStore().resolve) == resolved
    path.write_bytes(b"changed")
    with pytest.raises(CatalogError) as error:
        catalog.resolve("fixture", resolver=ArtifactStore().resolve)
    assert error.value.code == "artifact_digest_mismatch"


class Documents:
    def __init__(self, data=b"document", *, grant=True, locator=True):
        self.data, self.grant, self.locator = data, grant, locator
        self.opens = 0
        self.checks = []

    def validate_grant(self, grant, locator):
        self.checks.append((grant, locator))
        return self.grant

    def validate_locator(self, locator):
        return self.locator

    def open_binary(self, locator):
        self.opens += 1
        return io.BytesIO(self.data)


@pytest.mark.parametrize("grant,locator,code", [
    (False, True, "artifact_saf_grant_invalid"),
    (True, False, "artifact_saf_locator_invalid"),
    (True, True, "artifact_digest_mismatch"),
])
def test_saf_grant_and_locator_do_not_prove_document_bytes(grant, locator, code):
    documents = Documents(b"wrong", grant=grant, locator=locator)
    location = ArtifactLocation("content://documents/opaque", "saf", "persisted-grant")
    artifact = ArtifactRef(sha(b"expected"), "gguf", revision().revision_id, len(b"expected"))
    receipt = ArtifactStore(documents=documents).verify(artifact, location)
    assert receipt.state == "unavailable" and receipt.reason_code == code
    assert documents.opens == int(grant and locator)
    assert documents.checks == [("persisted-grant", location.locator)]


def test_saf_inventory_hashes_live_document_each_prepare():
    documents = Documents()
    artifact = ArtifactRef(sha(documents.data), "gguf", revision().revision_id, len(documents.data))
    location = ArtifactLocation("content://documents/opaque", "saf", "persisted")
    store = ArtifactStore(documents=documents)
    assert store.resolve(revision(), artifact, location).receipt.state == "verified"
    documents.data = b"tampered"
    assert store.verify(artifact, location).state == "unavailable"
    assert documents.opens == 2


def test_import_dry_run_has_no_writes_and_report_is_integrity_checkable(catalog, tmp_path):
    path, _, payload = bound_file(tmp_path)
    payload.update(selected=True, ready=True, native={"handle": 123}, runtime={"kv": "old"})
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    artifacts = ArtifactStore()
    importer = LegacyRegistryImporter(catalog, artifacts)
    report = importer.dry_run(["invalid JSON", payload, {"model_id": "other", "name": "legacy"}])
    assert catalog._store.writes == [] and artifacts.inventory() == ()
    assert report.entries[0]["action"] == "reject"
    assert report.entries[1]["state"] == "verified"
    assert report.entries[2]["state"] == "registered"
    assert set(report.entries[1]["ignored_runtime_fields"]) == {"selected", "ready", "native", "runtime"}
    assert report.verify() and verify_import_report(report.as_dict())
    exported = report.as_dict()
    exported["entries"][1]["payload"]["gguf_path"] = "tampered"
    assert not verify_import_report(exported)
    assert report.as_dict()["space_estimate"]["artifact_copy_bytes"] == 0
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    applied = importer.import_registry(["invalid JSON", payload, {"model_id": "other", "name": "legacy"}],
                                       expected_report=report)
    assert applied.verify() and len(catalog._store.writes) == 2
    persisted = catalog.snapshot("fixture").as_payload()
    assert not {"selected", "ready", "native", "runtime"}.intersection(persisted)
    assert persisted["resolved_artifact"]["location"]["locator"] == str(path)
    assert applied.as_dict()["impact"]["runtime_state_restored"] is False
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}


def test_v1_manifest_is_zero_copy_imported_as_frozen_revision(catalog, tmp_path):
    root, row = legacy_bundle(tmp_path)
    importer = LegacyRegistryImporter(catalog, ArtifactStore())
    preview = importer.dry_run([row])
    assert preview.entries[0]["state"] == "verified"
    report = importer.import_registry([row], expected_report=preview)
    assert report.entries[0]["result"] == "imported"
    resolved = catalog.resolve("fixture", resolver=ArtifactStore().resolve)
    assert resolved.location.locator == str(root.resolve())
    assert resolved.revision.source_revision == "commit-1"
    assert resolved.revision.tokenizer_sha256 == sha((root / "tokenizer.json").read_bytes())
    assert resolved.revision.template_sha256 == canonical_digest("hello {{ messages }}")


def test_bad_manifest_row_cannot_block_good_neighbor_or_replace_existing(catalog, tmp_path):
    root, bad = legacy_bundle(tmp_path)
    (root / "tokenizer.json").write_bytes(b"changed")
    assert catalog.publish("existing", {"name": "verified-new"}, 0)
    importer = LegacyRegistryImporter(catalog, ArtifactStore())
    report = importer.import_registry([bad, {"model_id": "good"}, {"model_id": "existing", "name": "stale"}])
    assert report.entries[0]["action"] == "reject"
    assert report.entries[0]["reason_code"] == "artifact_digest_mismatch"
    assert report.entries[1]["result"] == "imported"
    assert report.entries[2]["reason_code"] == "catalog_destination_exists"
    assert catalog.snapshot("existing").as_payload()["name"] == "verified-new"


@pytest.mark.parametrize("mutation", ["source", "target", "bytes"])
def test_import_rejects_drift_after_review(catalog, tmp_path, mutation):
    path, _, row = bound_file(tmp_path)
    importer = LegacyRegistryImporter(catalog, ArtifactStore())
    preview = importer.dry_run([row])
    if mutation == "source":
        row["name"] = "changed"
    elif mutation == "target":
        assert catalog.publish("fixture", {"name": "newer"}, 0)
    else:
        path.write_bytes(b"changed")
    with pytest.raises(CatalogError) as error:
        importer.import_registry([row], expected_report=preview)
    assert error.value.code == "legacy_import_plan_stale"


def test_unknown_schema_stops_before_destination_reads(catalog, monkeypatch):
    monkeypatch.setattr(catalog, "snapshot", lambda *_: pytest.fail("must reject schema before reads"))
    with pytest.raises(CatalogError) as error:
        LegacyRegistryImporter(catalog, ArtifactStore()).dry_run([], source_schema=2)
    assert error.value.code == "legacy_registry_schema_unsupported"


def test_sqlite_row_payload_corruption_is_quarantined(catalog):
    importer = LegacyRegistryImporter(catalog, ArtifactStore())
    report = importer.import_registry([
        {"model_id": "broken", "payload": "invalid JSON"},
        {"model_id": "good", "payload": json.dumps({"name": "source"})},
    ])
    assert report.entries[0]["action"] == "reject"
    assert report.entries[1]["result"] == "imported"
    assert catalog.snapshot("good").as_payload()["name"] == "source"


def test_registry_override_of_builtin_binds_all_revision_fields(monkeypatch):
    import model_config
    from model_module import ModelManager
    monkeypatch.setattr(model_config, "get_builtin_model", lambda *_: object())
    row = {"model_id": "builtin", "revision": "one", "manifest_sha256": "a" * 64,
           "tokenizer_sha256": "b" * 64, "template_sha256": "c" * 64}
    initial = ModelManager._registry_load_identity("builtin", [row])
    assert initial["revision"] == "one"
    for field in ("revision", "manifest_sha256", "tokenizer_sha256", "template_sha256"):
        changed = {**row, field: "other"}
        assert ModelManager._registry_load_identity("builtin", [changed]) != initial


def test_identity_observes_tokenizer_and_template_drift_without_directory_stat_change(tmp_path):
    from model_module import ModelManager
    root, row = legacy_bundle(tmp_path)
    before = root.stat().st_mtime_ns
    initial = ModelManager._artifact_load_identity(str(root), model_id="fixture", db_experimental_models=[row])
    (root / "tokenizer_config.json").write_bytes(b'{"chat_template":"changed"}')
    assert root.stat().st_mtime_ns == before
    changed = ModelManager._artifact_load_identity(str(root), model_id="fixture", db_experimental_models=[row])
    assert changed["metadata_sha256"] != initial["metadata_sha256"]


def test_repeated_typed_identity_does_not_read_weight_bytes(tmp_path, monkeypatch):
    import qlh_core.adapters.artifacts as artifacts
    from model_module import ModelManager
    path, resolved, row = bound_file(tmp_path)
    original_hash = artifacts._hash_disk

    def no_weight_hash(candidate):
        assert candidate != path, "load identity must not hash model weights"
        return original_hash(candidate)

    monkeypatch.setattr(artifacts, "_hash_disk", no_weight_hash)
    monkeypatch.setattr(ArtifactStore, "resolve", lambda *_: pytest.fail("prepare verification belongs outside identity"))
    identities = [ModelManager._artifact_load_identity(str(path), model_id="fixture", db_experimental_models=[row])
                  for _ in range(5)]
    assert all(value == identities[0] for value in identities)
    assert identities[0]["model_revision_id"] == resolved.revision.revision_id
    assert identities[0]["artifact_id"] == resolved.artifact.artifact_id


def test_typed_identity_never_falls_back_on_revision_or_locator_mismatch(tmp_path):
    path, _, row = bound_file(tmp_path)
    bad = deepcopy(row)
    bad["resolved_artifact"]["model_revision"]["source_revision"] = "another"
    with pytest.raises(CatalogError) as error:
        artifact_load_identity(str(path), model_id="fixture", rows=[bad])
    assert error.value.code == "artifact_revision_mismatch"
    other = tmp_path / "other.gguf"
    other.write_bytes(path.read_bytes())
    with pytest.raises(CatalogError) as error:
        artifact_load_identity(str(other), model_id="fixture", rows=[row])
    assert error.value.code == "artifact_locator_mismatch"


def test_gguf_descriptor_does_not_borrow_builtin_tokenizer(monkeypatch, tmp_path):
    from llama_engine import LlamaCppEngine
    path = tmp_path / "fixture.gguf"
    path.write_bytes(b"GGUF fixture")
    monkeypatch.setattr("relay_segment_info.read_gguf_layer_info", lambda _: {
        "architecture": "qwen2", "n_layer": 2, "n_embd": 8})
    monkeypatch.setattr("relay_segment_info.read_gguf_tensor_bytes", lambda _: {
        "token_embd.weight": 8, "blk.0.weight": 4, "blk.1.weight": 4, "output_norm.weight": 4})
    monkeypatch.setattr("relay_segment_info.read_gguf_tokenizer_sha256", lambda _: "")
    monkeypatch.setattr("model_config.get_builtin_model", lambda *_: pytest.fail("builtin name is not revision proof"))
    engine = LlamaCppEngine()
    engine._model_path = str(path)
    engine.active_model_id = "builtin"
    assert engine.get_pipeline_descriptor()["tokenizer_sha256"] == ""


@pytest.mark.parametrize("boundary", ["manager_load", "manager_prepare", "gguf_load", "gguf_prepare", "typed_rows_load"])
def test_tamper_after_import_is_rejected_before_production_load_boundary(catalog, tmp_path, monkeypatch, boundary):
    from model_module import ModelManager
    from llama_engine import LlamaCppEngine
    from qlh_core.adapters.artifacts import CatalogLoadGuard

    if boundary == "manager_prepare":
        root, row = legacy_bundle(tmp_path)
        path = root
        tamper = root / "model.safetensors"
    else:
        path, _, row = bound_file(tmp_path)
        tamper = path
    importer = LegacyRegistryImporter(catalog, ArtifactStore())
    assert importer.import_registry([row]).entries[0]["result"] == "imported"
    tamper.write_bytes(b"tampered since import")
    guard = CatalogLoadGuard(catalog, ArtifactStore())
    if boundary in ("manager_load", "typed_rows_load"):
        manager = ModelManager()
        if boundary == "manager_load":
            manager.configure_catalog(guard)
        monkeypatch.setattr(manager, "_resolve_model_load_request", lambda **_: {
            "path": str(path), "model_id": "fixture", "engine": "pytorch",
            "effective_quantization": "fp16"})
        monkeypatch.setattr(manager, "_load_pytorch", lambda *_: pytest.fail("native loader ran before verification"))
        invoke = lambda: manager.load_model(model_path=str(path), model_id="fixture",
                                            db_experimental_models=[catalog.snapshot("fixture").as_payload()])
    elif boundary == "manager_prepare":
        manager = ModelManager()
        manager.configure_catalog(guard)
        monkeypatch.setattr("pipeline_model_descriptor.inspect_pipeline_model",
                            lambda *_args, **_kwargs: pytest.fail("metadata prepare ran before verification"))
        invoke = lambda: manager.prepare_pipeline_model("fixture", str(path))
    else:
        engine = LlamaCppEngine()
        engine.configure_catalog(guard, model_id="fixture")
        monkeypatch.setattr(engine, "get_pipeline_descriptor", lambda: pytest.fail("GGUF inspection preceded guard"))
        if boundary == "gguf_load":
            invoke = lambda: engine.load_model(model_path=str(path))
        else:
            invoke = lambda: engine.prepare_pipeline_model("fixture", str(path))
    with pytest.raises(CatalogError) as error:
        invoke()
    assert error.value.code == "artifact_digest_mismatch"


def test_registry_tombstone_version_changes_identity_even_with_identical_payload(tmp_path):
    from model_module import ModelManager

    class Payload(dict):
        def __init__(self, version):
            super().__init__(model_id="fixture", revision="same", source={"resolved_revision": "same"})
            self.registry_revision = version

    old = ModelManager._registry_load_identity("fixture", [Payload(1)])
    recreated = ModelManager._registry_load_identity("fixture", [Payload(3)])
    assert old["payload_sha256"] == recreated["payload_sha256"]
    assert old["registry_revision"] == "1"
    assert recreated["registry_revision"] == "3"
    assert old != recreated
    assert ModelManager._registry_load_identity("fixture", [dict(Payload(1))])["registry_revision"] == "0"


def test_boolean_schema_versions_are_rejected_at_every_public_boundary(catalog):
    with pytest.raises(CatalogError) as error:
        revision(schema_version=True)
    assert error.value.code == "catalog_schema_unsupported"
    importer = LegacyRegistryImporter(catalog, ArtifactStore())
    report = importer.dry_run([{"model_id": "bad", "schema": True}, {"model_id": "good"}])
    assert report.entries[0]["reason_code"] == "legacy_registry_row_schema_unsupported"
    assert report.entries[1]["action"] == "publish"
    exported = report.as_dict()
    exported["schema_version"] = True
    exported["report_sha256"] = canonical_digest({key: value for key, value in exported.items()
                                                   if key != "report_sha256"})
    assert not verify_import_report(exported)


@pytest.mark.parametrize("changes,code", [
    ({"components": "weights"}, "artifact_components_invalid"),
    ({"components": [["weights"]]}, "artifact_components_invalid"),
    ({"files": "invalid"}, "artifact_manifest_invalid"),
    ({"files": ["invalid"]}, "artifact_manifest_invalid"),
    ({"quantization": []}, "artifact_quantization_invalid"),
])
def test_invalid_artifact_container_types_are_named_errors(changes, code):
    with pytest.raises(CatalogError) as error:
        ArtifactRef("a" * 64, "gguf", revision().revision_id, 1, **changes)
    assert error.value.code == code


def test_invalid_typed_manifest_is_quarantined_without_blocking_neighbor(catalog, tmp_path):
    _, _, row = bound_file(tmp_path)
    row["resolved_artifact"]["artifact"]["files"] = ["broken"]
    report = LegacyRegistryImporter(catalog, ArtifactStore()).import_registry([row, {"model_id": "good"}])
    assert report.entries[0]["reason_code"] == "artifact_manifest_invalid"
    assert report.entries[1]["result"] == "imported"


def test_guard_rejects_registry_replacement_during_byte_verification(catalog, tmp_path):
    from qlh_core.adapters.artifacts import CatalogLoadGuard, DiskInventory
    _, resolved, _ = bound_file(tmp_path)
    assert catalog.publish_resolved("fixture", resolved, 0)

    class ConcurrentInventory:
        def measure(self, artifact, location):
            measured = DiskInventory().measure(artifact, location)
            assert catalog.remove("fixture", 1)
            assert catalog.publish_resolved("fixture", resolved, 2)
            return measured

    with pytest.raises(CatalogError) as error:
        CatalogLoadGuard(catalog, ArtifactStore(disk=ConcurrentInventory()))("fixture", resolved.location.locator)
    assert error.value.code == "catalog_revision_conflict"


def test_invalid_same_id_row_does_not_reserve_good_candidate(catalog):
    report = LegacyRegistryImporter(catalog, ArtifactStore()).import_registry([
        {"model_id": "fixture", "schema": True}, {"model_id": "fixture", "name": "valid"},
    ])
    assert report.entries[0]["action"] == "reject"
    assert report.entries[1]["result"] == "imported"
    assert catalog.snapshot("fixture").as_payload()["name"] == "valid"


@pytest.mark.parametrize("filename,data", [
    ("chat_template.jinja", b"new template"), ("tokenizer.model", b"new tokenizer"),
])
def test_new_unbound_loader_metadata_is_rejected_after_import(catalog, tmp_path, filename, data):
    from qlh_core.adapters.artifacts import CatalogLoadGuard
    root, row = legacy_bundle(tmp_path)
    assert LegacyRegistryImporter(catalog, ArtifactStore()).import_registry([row]).entries[0]["result"] == "imported"
    (root / filename).write_bytes(data)
    with pytest.raises(CatalogError) as error:
        CatalogLoadGuard(catalog, ArtifactStore())("fixture", str(root))
    assert error.value.code == "artifact_metadata_unbound"


def test_bundle_weight_stat_changes_identity_without_hashing_weights(catalog, tmp_path, monkeypatch):
    import qlh_core.adapters.artifacts as artifacts
    from model_module import ModelManager
    root, row = legacy_bundle(tmp_path)
    assert LegacyRegistryImporter(catalog, ArtifactStore()).import_registry([row]).entries[0]["result"] == "imported"
    typed = catalog.snapshot("fixture").as_payload()
    original_hash = artifacts._hash_disk

    def no_weight_hash(candidate):
        assert candidate.suffix != ".safetensors"
        return original_hash(candidate)

    monkeypatch.setattr(artifacts, "_hash_disk", no_weight_hash)
    initial = ModelManager._artifact_load_identity(str(root), model_id="fixture", db_experimental_models=[typed])
    weight = root / "model.safetensors"
    weight.write_bytes(b"changed weight bytes")
    changed = ModelManager._artifact_load_identity(str(root), model_id="fixture", db_experimental_models=[typed])
    assert initial["artifact_file_stats_sha256"] != changed["artifact_file_stats_sha256"]
    assert initial["model_revision_id"] == changed["model_revision_id"]
