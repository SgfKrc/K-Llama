"""R2 production composition and the source-to-segment admission boundary."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_host import ModelHost, _LazyModelManager
from qlh_core.adapters.artifacts import ArtifactStore
from qlh_core.adapters.settings import SettingsStore
from qlh_core.bootstrap.runtime import AppRuntime
from qlh_core.contracts.catalog import ArtifactLocation, ArtifactRef, CatalogError
from tests.test_refactor_catalog_artifacts import bound_file, legacy_bundle, revision

pytestmark = pytest.mark.refactor_contract


def runtime_at(root, **environment):
    return AppRuntime(SettingsStore({
        "QLH_STATE_DIR": str(root / "state"),
        "QLH_NODE_CONFIG_PATH": str(root / "node.json"),
        "QLH_NODE_SECRET_STORE_PATH": str(root / "secret.json"),
        "QLH_NODE_ROLE": "client", "QLH_RUN_MODE": "single", **environment,
    }, dotenv_path=root / "absent.env").load())


def test_runtime_catalog_configuration_is_lazy_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(_LazyModelManager, "_get_instance", lambda *_: pytest.fail("catalog configuration warmed torch"))
    runtime = runtime_at(tmp_path)
    try:
        services = runtime.services
        assert services.host.peek_manager() is None
        assert services.stage_host()._host.peek_manager() is None
        assert vars(services.host)["_catalog_prepare"] is services.catalog_prepare
        assert vars(services.stage_host()._host)["_catalog_prepare"] is services.catalog_prepare
        assert not Path(services.store.path).exists()
    finally:
        runtime.close()


def test_two_runtime_catalogs_and_stage_hosts_share_no_alias_state(tmp_path):
    runtimes = [runtime_at(tmp_path / name) for name in ("one", "two")]
    try:
        services = [runtime.services for runtime in runtimes]
        paths = []
        for index, service in enumerate(services):
            service.store.initialize()
            root = tmp_path / str(index)
            root.mkdir()
            path, _, row = bound_file(root, data=f"GGUF-{index}".encode())
            assert service.catalog.publish("fixture", row, 0)
            paths.append(path)
        for index, service in enumerate(services):
            assert service.catalog_prepare("fixture", str(paths[index])).location.locator == str(paths[index])
            assert vars(service.stage_host()._host)["_catalog_prepare"] is service.catalog_prepare
            with pytest.raises(CatalogError, match="frozen registration"):
                service.catalog_prepare("fixture", str(paths[1-index]))
        assert services[0].stage_host() is not services[1].stage_host()
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.mark.parametrize("entry", ["candidate", "prepare", "load", "switch"])
def test_runtime_host_rejects_typed_tamper_before_native_entry(tmp_path, monkeypatch, entry):
    import llama_engine
    runtime = runtime_at(tmp_path)
    try:
        service = runtime.services
        service.store.initialize()
        path, _, row = bound_file(tmp_path)
        assert service.catalog.publish("fixture", row, 0)
        path.write_bytes(b"modified after registration")
        monkeypatch.setattr(llama_engine.LlamaCppEngine, "get_pipeline_descriptor", lambda *_: pytest.fail("unverified metadata"))
        monkeypatch.setattr(llama_engine.LlamaCppEngine, "_load_model_runtime", lambda *_a, **_k: pytest.fail("unverified native load"))
        if entry == "candidate":
            invoke = lambda: service.host.prepare_pipeline_candidate("txn", "fixture", str(path))
        elif entry == "prepare":
            invoke = lambda: service.host.prepare_pipeline_model("fixture", str(path))
        else:
            invoke = lambda: getattr(service.host, "load_model" if entry == "load" else "switch_model")(
                model_id="fixture", model_path=str(path), engine="llama_cpp")
        with pytest.raises(CatalogError) as error:
            invoke()
        assert error.value.code == "artifact_digest_mismatch"
        assert not service.host.has_loaded_model()
    finally:
        runtime.close()


def test_forwarded_lazy_layer_load_gets_the_runtime_catalog(tmp_path, monkeypatch):
    calls = []
    manager = SimpleNamespace(configure_catalog=lambda guard, **_: calls.append(guard),
                              load_layer_range=lambda *_: "segment")
    monkeypatch.setattr(_LazyModelManager, "_get_instance", lambda *_: manager)
    host = ModelHost()
    guard = object()
    host.configure_catalog(guard)
    assert host.load_layer_range(0, 2) == "segment"
    assert calls == [guard]


def test_torch_segment_revalidates_source_before_metadata_or_weight_loading(tmp_path, monkeypatch):
    from model_module import ModelManager
    from qlh_core.adapters.artifacts import CatalogLoadGuard, LegacyRegistryImporter
    from tests.test_refactor_catalog_artifacts import MemoryRepository
    from qlh_core.control.catalog import CatalogService
    root, row = legacy_bundle(tmp_path)
    catalog = CatalogService(MemoryRepository())
    assert LegacyRegistryImporter(catalog, ArtifactStore()).import_registry([row]).entries[0]["result"] == "imported"
    manager = ModelManager()
    manager.configure_catalog(CatalogLoadGuard(catalog, ArtifactStore()))
    manager._active_model_id = "fixture"
    manager._full_model_path = str(root)
    (root / "model.safetensors").write_bytes(b"changed")
    monkeypatch.setattr("model_module.AutoConfig.from_pretrained", lambda *_a, **_k: pytest.fail("metadata before verification"))
    with pytest.raises(CatalogError) as error:
        manager.load_layer_range(0, 2, has_lm_head=False)
    assert error.value.code == "artifact_digest_mismatch"


@pytest.fixture
def prepared_crop(tmp_path, monkeypatch):
    from llama_engine import LlamaCppEngine
    runtime = runtime_at(tmp_path, QLH_LAYER_ARTIFACT_DIR=str(tmp_path))
    service = runtime.services
    service.store.initialize()
    source = tmp_path / "source.gguf"
    source.write_bytes(b"whole source metadata and weights")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    model = revision()
    artifact = ArtifactRef(source_sha, "gguf", model.revision_id, source.stat().st_size)
    resolved = service.artifacts.resolve(model, artifact, ArtifactLocation(str(source)))
    assert service.catalog.publish_resolved("fixture", resolved, 0)
    crop = tmp_path / "segment.gguf"
    crop.write_bytes(b"only layers zero and one")
    manifest = {"source_layer_range": [0, 2], "n_layer": 4,
                "source_model_sha256": source_sha,
                "artifact_sha256": hashlib.sha256(crop.read_bytes()).hexdigest()}
    crop.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("model_supply_worker.selected_gguf_path", lambda: "")
    monkeypatch.setattr(LlamaCppEngine, "get_pipeline_descriptor", lambda *_: {
        "model_path": str(source), "model_sha256": source_sha, "total_layers": 4,
        "pipeline_runtime_supported": True})
    loaded = []
    monkeypatch.setattr(LlamaCppEngine, "_load_model_runtime", lambda self, path, **_: loaded.append(path))
    service.host.prepare_pipeline_model("fixture", str(source))
    try:
        yield service, source, crop, loaded
    finally:
        runtime.close()


def test_catalog_source_to_verified_crop_materializes_only_the_segment(prepared_crop):
    service, source, crop, loaded = prepared_crop
    result = service.host.load_layer_range(0, 2, has_embedding=True)
    assert result["artifact"] == str(crop)
    assert loaded == [str(crop)] and str(source) not in loaded
    engine = service.host.peek_manager()
    assert engine.is_pipeline_prepared
    assert engine._layer_load_proof.artifact_sha256 == hashlib.sha256(crop.read_bytes()).hexdigest()


def test_catalog_source_does_not_admit_corrupt_crop(prepared_crop):
    service, source, crop, loaded = prepared_crop
    crop.write_bytes(b"changed after crop publication")
    with pytest.raises(RuntimeError, match="裁层工件"):
        service.host.load_layer_range(0, 2)
    assert loaded == []


@pytest.mark.parametrize("change", ["replace", "remove"])
def test_catalog_source_change_remains_rejected_on_segment_retry(prepared_crop, change):
    service, source, crop, loaded = prepared_crop
    engine = service.host.peek_manager()
    prepared = engine._resolved_artifact
    snapshot = service.catalog.snapshot("fixture")
    if change == "remove":
        assert service.catalog.remove("fixture", snapshot.registry_revision)
    else:
        replacement = replace(revision(), source_revision="commit-2")
        artifact = ArtifactRef(hashlib.sha256(source.read_bytes()).hexdigest(), "gguf",
                               replacement.revision_id, source.stat().st_size)
        resolved = service.artifacts.resolve(replacement, artifact, ArtifactLocation(str(source)))
        assert service.catalog.publish_resolved("fixture", resolved, snapshot.registry_revision)
    for _ in range(2):
        with pytest.raises(CatalogError) as error:
            service.host.load_layer_range(0, 2)
        assert error.value.code == "artifact_revision_mismatch"
        assert engine._resolved_artifact == prepared
        assert loaded == []


@pytest.mark.parametrize("change", ["replace", "remove"])
def test_torch_source_change_remains_rejected_on_segment_retry(tmp_path, monkeypatch, change):
    from model_module import ModelManager
    from qlh_core.adapters.artifacts import CatalogLoadGuard, LegacyRegistryImporter
    from tests.test_refactor_catalog_artifacts import MemoryRepository
    from qlh_core.control.catalog import CatalogService

    root, row = legacy_bundle(tmp_path)
    artifacts = ArtifactStore()
    catalog = CatalogService(MemoryRepository())
    assert LegacyRegistryImporter(catalog, artifacts).import_registry([row]).entries[0]["result"] == "imported"
    guard = CatalogLoadGuard(catalog, artifacts)
    manager = ModelManager()
    manager.configure_catalog(guard)
    manager._active_model_id = "fixture"
    manager._full_model_path = str(root)
    prepared = manager._verify_catalog_artifact("fixture", str(root))
    snapshot = catalog.snapshot("fixture")
    if change == "remove":
        assert catalog.remove("fixture", snapshot.registry_revision)
    else:
        replacement = replace(prepared.revision, source_revision="commit-2")
        artifact = replace(prepared.artifact, model_revision=replacement.revision_id)
        resolved = artifacts.resolve(replacement, artifact, prepared.location)
        assert catalog.publish_resolved("fixture", resolved, snapshot.registry_revision)
    monkeypatch.setattr("model_module.AutoConfig.from_pretrained",
                        lambda *_a, **_k: pytest.fail("changed source reached metadata loading"))
    for _ in range(2):
        with pytest.raises(CatalogError) as error:
            manager.load_layer_range(0, 2, has_lm_head=False)
        assert error.value.code == "artifact_revision_mismatch"
        assert manager._resolved_artifact == prepared
