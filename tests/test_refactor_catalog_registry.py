"""Catalog ownership, durable revisions and concurrent production writers."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import asyncio
import threading

import pytest
import local_store

pytestmark = pytest.mark.refactor_contract


def seed(model_id="revision-fixture", **values):
    payload = {"model_id": model_id, "name": "seed", **values}
    assert local_store.save_local_experimental_model(model_id, payload)
    return local_store.get_local_model_registry_snapshot(model_id)


def test_compatibility_save_cannot_replace_an_existing_revision():
    old = seed()
    assert not local_store.save_local_experimental_model(old.model_id, {"name": "stale"})
    assert local_store.get_local_model_registry_snapshot(old.model_id) == old


def test_deleted_alias_keeps_revision_across_recreation_and_restart():
    old = seed()
    assert local_store.delete_local_experimental_model(old.model_id, expected_revision=old.registry_revision)
    deleted = local_store.get_local_model_registry_snapshot(old.model_id)
    assert deleted.payload is None and deleted.registry_revision == old.registry_revision + 1
    assert local_store.publish_local_model_registry(old.model_id, old.as_payload(), expected_revision=deleted.registry_revision)
    recreated = local_store.get_local_model_registry_snapshot(old.model_id)
    assert recreated.as_payload() == old.as_payload()
    assert recreated.registry_revision == old.registry_revision + 2
    local_store._initialized_paths.discard(local_store._sqlite_path())
    assert local_store.get_local_model_registry_snapshot(old.model_id) == recreated
    assert not local_store.publish_local_model_registry(old.model_id, {"name": "stale"}, expected_revision=old.registry_revision)
    assert not local_store.delete_local_experimental_model(old.model_id, expected_revision=old.registry_revision)


def test_concurrent_revision_writers_have_one_winner_and_no_partial_metadata():
    old = seed(manifest={"files": [{"sha256": "a" * 64}]})
    barrier = threading.Barrier(4)
    def publish(i):
        barrier.wait(timeout=5)
        return i, local_store.publish_local_model_registry(old.model_id, {"name": str(i), "source": str(i)}, expected_revision=old.registry_revision)
    with ThreadPoolExecutor(max_workers=4) as pool:
        winners = [i for i, won in pool.map(publish, range(4)) if won]
    assert len(winners) == 1
    snapshot = local_store.get_local_model_registry_snapshot(old.model_id)
    assert snapshot.registry_revision == old.registry_revision + 1
    assert snapshot.as_payload() == {"model_id": old.model_id, "name": str(winners[0]), "source": str(winners[0])}


def test_versioned_legacy_projection_rejects_payload_aba():
    old = seed()
    expected = deepcopy(local_store.get_local_experimental_models()[0])
    assert local_store.delete_local_experimental_model(old.model_id)
    assert local_store.save_local_experimental_model(old.model_id, old.as_payload())
    assert not local_store.save_local_experimental_model_cas(old.model_id, {"name": "stale"}, expected=expected)


def test_snapshot_is_deeply_immutable_and_detached_from_public_projection():
    old = seed(manifest={"files": [{"sha256": "a" * 64}]})
    with pytest.raises(TypeError):
        old.payload["manifest"]["files"][0]["sha256"] = "b" * 64
    public = local_store.get_local_experimental_models()[0]
    public["manifest"]["files"][0]["sha256"] = "b" * 64
    assert local_store.get_local_model_registry_snapshot(old.model_id) == old


@pytest.mark.parametrize("revision", [True, -1, 1.5, "1"])
def test_revision_must_be_a_nonnegative_integer(revision):
    with pytest.raises(ValueError):
        local_store.publish_local_model_registry("invalid", {}, expected_revision=revision)
    assert local_store.get_local_model_registry_snapshot("invalid").registry_revision == 0


def test_download_queued_before_supply_does_not_overwrite_new_registration(tmp_path, monkeypatch):
    import model_download_jobs as jobs
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.gguf").write_bytes(b"fixture")
    old = seed("queued-model")
    queued = []
    job = jobs.create_job(source=str(source), target=str(tmp_path / "downloaded"), model_id=old.model_id, executor=queued.append)
    assert local_store.publish_local_model_registry(old.model_id, {"name": "supply", "source": "supply"}, expected_revision=old.registry_revision)
    queued[0]()
    assert jobs.get_job(job["job_id"])["status"] == jobs.STATUS_FAILED
    assert local_store.get_local_model_registry_snapshot(old.model_id).payload["source"] == "supply"


def test_cleanup_preserves_bytes_adopted_by_another_alias(tmp_path):
    import model_download_jobs as jobs
    root = tmp_path / "downloaded"
    root.mkdir()
    weight = root / "model.gguf"
    weight.write_bytes(b"fixture")
    seed("adopter", gguf_path=str(weight))
    assert jobs.retain_unregistered_publication(root)["reason_code"] == "cleanup_pending"
    assert weight.read_bytes() == b"fixture"


def test_manual_registration_returns_conflict_when_publication_changes_during_hash(tmp_path, monkeypatch):
    import api_server
    old = seed("manual-model")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fixture")
    def build(*args, **kwargs):
        assert local_store.publish_local_model_registry(old.model_id, {"name": "supply", "source": "supply"}, expected_revision=old.registry_revision)
        return {"artifact_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    monkeypatch.setattr(api_server, "build_manifest", build)
    monkeypatch.setattr(api_server, "write_manifest", lambda *args: None)
    request = api_server.RegisterModelRequest(model_id=old.model_id, name="manual", model_type="gguf", gguf_path=str(gguf))
    with pytest.raises(api_server.HTTPException) as error:
        asyncio.run(api_server.register_model(request))
    assert error.value.status_code == 409
    assert error.value.detail["reason_code"] == "MODEL_REGISTRY_CHANGED"
    assert local_store.get_local_model_registry_snapshot(old.model_id).payload["source"] == "supply"


def test_two_runtime_catalogs_keep_aliases_and_versions_isolated(tmp_path):
    from types import SimpleNamespace
    from qlh_core.adapters.local_store import RuntimeLocalStore
    from qlh_core.control.catalog import CatalogService
    stores = []
    for name in ("a", "b"):
        cfg = SimpleNamespace(state_dir=str(tmp_path / name), environment_snapshot=lambda: {})
        store = RuntimeLocalStore(cfg)
        store.initialize()
        stores.append(CatalogService(store.catalog_repository()))
    for index, service in enumerate(stores):
        assert service.publish("same-alias", {"name": str(index)}, expected_revision=0)
    assert stores[0].snapshot("same-alias").payload["name"] == "0"
    assert stores[1].snapshot("same-alias").payload["name"] == "1"
    assert stores[0].remove("same-alias", expected_revision=1)
    assert stores[1].snapshot("same-alias").registry_revision == 1


def test_migration_cli_dry_run_leaves_destination_absent_then_imports_exact_report(tmp_path, capsys):
    import json
    from scripts.migrate_model_catalog import main
    from qlh_core.adapters.artifacts import verify_import_report
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps({"schema_version": 1, "models": [{"model_id": "legacy", "name": "fixture", "selected": True, "ready": True}]}), encoding="utf-8")
    target = tmp_path / "new" / "catalog.sqlite3"
    before = source.read_bytes()
    assert main([str(source), "--sqlite-path", str(target)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert verify_import_report(report)
    assert report["dry_run"] is True
    assert not target.parent.exists() and source.read_bytes() == before
    review = tmp_path / "review.json"
    review.write_text(json.dumps(report), encoding="utf-8")
    assert main([str(source), "--sqlite-path", str(target), "--apply", "--review-report", str(review)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["entries"][0]["result"] == "imported"
    assert verify_import_report(result)
    with local_store.bind_local_store(target):
        snapshot = local_store.get_local_model_registry_snapshot("legacy")
    assert snapshot.as_payload() == {"model_id": "legacy", "name": "fixture"}
    assert source.read_bytes() == before


def test_migration_cli_rejects_stale_report_before_creating_database(tmp_path, capsys):
    import json
    from scripts.migrate_model_catalog import main
    source, target = tmp_path / "legacy.json", tmp_path / "catalog.sqlite3"
    source.write_text('[{"model_id":"legacy","name":"one"}]', encoding="utf-8")
    assert main([str(source), "--sqlite-path", str(target)]) == 0
    report = tmp_path / "review.json"
    report.write_text(capsys.readouterr().out, encoding="utf-8")
    source.write_text('[{"model_id":"legacy","name":"two"}]', encoding="utf-8")
    assert main([str(source), "--sqlite-path", str(target), "--apply", "--review-report", str(report)]) == 2
    assert json.loads(capsys.readouterr().err)["reason_code"] == "legacy_import_plan_stale"
    assert not target.exists()


def test_migration_cli_reads_corrupt_sqlite_rows_without_mutating_source(tmp_path, capsys):
    import json
    import sqlite3
    from scripts.migrate_model_catalog import main
    source, target = tmp_path / "legacy.sqlite3", tmp_path / "target.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE model_registry(model_id TEXT PRIMARY KEY,payload TEXT)")
        connection.executemany("INSERT INTO model_registry VALUES (?, ?)", [("bad", "broken"), ("good", '{"model_id":"good"}')])
    before = source.read_bytes()
    assert main([str(source), "--sqlite-path", str(target)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert [entry["action"] for entry in report["entries"]] == ["reject", "publish"]
    assert source.read_bytes() == before and not target.exists()


def test_registration_manifest_is_create_only_and_same_revision_is_idempotent(tmp_path):
    from model_registry_validation import publish_registration_manifest
    manifest = {"artifact_sha256": "a" * 64, "files": []}
    path = publish_registration_manifest(tmp_path, manifest)
    before = path.read_bytes()
    assert publish_registration_manifest(tmp_path, manifest) == path
    with pytest.raises(ValueError, match="registration_manifest_conflict"):
        publish_registration_manifest(tmp_path, {"artifact_sha256": "b" * 64})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".qlh-register-*"))


@pytest.mark.parametrize("reference", ["external", "typed"])
def test_failed_publication_retains_bytes_without_racy_registry_scan(tmp_path, monkeypatch, reference):
    import model_download_jobs as jobs
    root = tmp_path / "published"
    root.mkdir()
    file = root / "model.gguf"
    file.write_bytes(b"fixture")
    if reference == "external":
        seed("external-owner", manifest_external_files={"external/model.gguf": str(file)})
    else:
        # A typed-only alias deliberately has no legacy model_path/gguf_path.
        from tests.test_refactor_catalog_artifacts import bound_file
        file, _, row = bound_file(root)
        assert local_store.save_local_experimental_model(row["model_id"], row)
    def forbidden_scan():
        raise AssertionError("registry scan cannot authorize artifact deletion")
    monkeypatch.setattr(local_store, "get_local_experimental_models", forbidden_scan)
    expected_bytes = file.read_bytes()
    receipt = jobs.retain_unregistered_publication(root)
    assert {key: receipt[key] for key in ("reason_code", "path", "retained")} == {
        "reason_code": "cleanup_pending", "path": str(root), "retained": True}
    assert len(receipt["cleanup_id"]) == 64
    from qlh_core.bootstrap.model_control import build_model_control
    assert build_model_control().snapshot()["artifact_cleanup"][receipt["cleanup_id"]]["location"] == str(root)
    assert file.read_bytes() == expected_bytes


def test_cli_failed_registration_reports_recoverable_publication(tmp_path, monkeypatch, capsys):
    import json
    from scripts.model_tools import import_model as wizard
    target = tmp_path / "remote"
    def download(_source, staging, **kwargs):
        file = staging / "model.gguf"
        file.write_bytes(b"fixture")
        return [file]
    monkeypatch.setattr(wizard, "download_model", download)
    monkeypatch.setattr(wizard, "register_model", lambda *args, **kwargs: False)
    assert wizard.main(["Org/fixture", "--target", str(target), "--register"]) == 2
    receipt = json.loads(capsys.readouterr().err)
    assert receipt["reason_code"] == "cleanup_pending" and receipt["path"] == str(target)
    assert (target / "model.gguf").read_bytes() == b"fixture"
    assert not list(tmp_path.glob(".remote.qlh-import-*"))


def test_cli_output_error_after_success_does_not_delete_registered_artifact(tmp_path, monkeypatch):
    import builtins
    from scripts.model_tools import import_model as wizard
    target = tmp_path / "remote"
    def download(_source, staging, **kwargs):
        file = staging / "model.gguf"
        file.write_bytes(b"fixture")
        return [file]
    original_print = builtins.print
    def fail_after_registered(*args, **kwargs):
        if args and str(args[0]).startswith("[4/4] registration succeeded"):
            raise OSError("output channel closed")
        return original_print(*args, **kwargs)
    monkeypatch.setattr(wizard, "download_model", download)
    monkeypatch.setattr(builtins, "print", fail_after_registered)
    assert wizard.main(["Org/fixture", "--target", str(target), "--register", "--model-id", "cli-published"]) == 2
    snapshot = local_store.get_local_model_registry_snapshot("cli-published")
    assert snapshot.payload is not None
    assert (target / "model.gguf").read_bytes() == b"fixture"


def test_cleanup_pending_job_persists_absolute_publication_location(tmp_path, monkeypatch):
    import model_download_jobs as jobs
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.gguf").write_bytes(b"fixture")
    models = tmp_path / "models"
    monkeypatch.setattr(jobs, "_register", lambda *args, **kwargs: False)
    job = jobs.create_job(source=str(source), target="relative/retained", model_id="relative-publication", models_root=str(models), executor=lambda run: run())
    assert job["status"] == jobs.STATUS_FAILED
    assert job["error"].startswith("cleanup_pending:")
    assert job["target"] == str(models / "relative" / "retained")
    assert (models / "relative" / "retained" / "model.gguf").is_file()
