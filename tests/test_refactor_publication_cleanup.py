"""Managed-cache reclamation is serialized with catalog adoption; fixture files only."""
import json
from pathlib import Path
import pytest
from qlh_core.adapters.publication_cleanup import PublicationReclaimer, publication_evidence, require_adoptable
from qlh_core.contracts.deployment import ModelControlError
from tests.test_refactor_deployment_owner import control

pytestmark = pytest.mark.refactor_contract


@pytest.fixture
def publication(control, tmp_path):
    owner, repository, connect = control
    root = tmp_path / "managed-cache"
    root.mkdir()
    location = root / "unregistered"
    location.mkdir()
    (location / "tiny.fixture").write_bytes(b"tiny local fixture")
    key = owner.retain_publication(str(location), evidence=publication_evidence(location))
    return owner, repository, connect, root, location, key


def test_retention_debt_survives_restart_and_reclaim_defaults_to_dry_run(publication):
    owner, repository, _, root, location, key = publication
    reclaimer = PublicationReclaimer(repository)
    report = reclaimer.reclaim(key, managed_root=root)
    assert report["dry_run"] and report["file_count"] == 1 and location.is_dir()
    assert owner.snapshot()["artifact_cleanup"][key]["status"] == "pending"
    report = reclaimer.reclaim(key, managed_root=root, dry_run=False)
    assert report["status"] == "reclaimed" and not location.exists()
    assert owner.snapshot()["artifact_cleanup"][key]["status"] == "reclaimed"


@pytest.mark.parametrize("typed", [False, True])
def test_any_catalog_alias_blocks_deletion(publication, typed):
    _, repository, connect, root, location, key = publication
    payload = ({"resolved_artifact": {"location": {"uri": str(location)}}} if typed else {"model_path": str(location)})
    with connect() as connection:
        connection.execute("CREATE TABLE model_registry(model_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("INSERT INTO model_registry VALUES (?,?)", ("concurrent-alias", json.dumps(payload)))
    with pytest.raises(ModelControlError, match="catalog_referenced"):
        PublicationReclaimer(repository).reclaim(key, managed_root=root, dry_run=False)
    assert location.exists()
    assert repository.read()["artifact_cleanup"][key]["status"] == "pending"


def test_claimed_or_reclaimed_location_cannot_be_readopted(publication):
    _, repository, connect, root, location, key = publication
    PublicationReclaimer(repository).reclaim(key, managed_root=root, dry_run=False)
    with connect() as connection, pytest.raises(ModelControlError, match="publication_retired"):
        require_adoptable(connection, {"model_path": str(location)})


def test_changed_bytes_or_outside_root_are_preserved(publication, tmp_path):
    _, repository, _, root, location, key = publication
    reclaimer = PublicationReclaimer(repository)
    with pytest.raises(ModelControlError, match="outside_managed_root"):
        reclaimer.reclaim(key, managed_root=location, dry_run=False)
    (location / "tiny.fixture").write_bytes(b"new generation must be retained")
    with pytest.raises(ModelControlError, match="publication_changed"):
        reclaimer.reclaim(key, managed_root=root, dry_run=False)
    assert location.is_dir()


def test_crash_after_quarantine_rename_resumes_exact_obligation(publication, monkeypatch):
    import qlh_core.adapters.publication_cleanup as adapter
    _, repository, _, root, location, key = publication
    remove = adapter.shutil.rmtree
    monkeypatch.setattr(adapter.shutil, "rmtree", lambda *_: (_ for _ in ()).throw(OSError("injected cleanup failure")))
    with pytest.raises(OSError):
        PublicationReclaimer(repository).reclaim(key, managed_root=root, dry_run=False)
    debt = repository.read()["artifact_cleanup"][key]
    assert debt["status"] == "reclaiming" and Path(debt["quarantine"]).is_dir()
    monkeypatch.setattr(adapter.shutil, "rmtree", remove)
    assert PublicationReclaimer(repository).reclaim(key, managed_root=root, dry_run=False)["status"] == "reclaimed"
