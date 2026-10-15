"""Retirement of canceled receiver checkpoints requires a complete live copy."""
import hashlib
import json
import threading
from pathlib import Path

import pytest
import model_supply_assets as assets
from tests.test_model_supply_assets import _kwargs, _make_gguf, _server, _receive, _interrupted_prefix
from tests.test_model_supply_worker import setup


def copies(tmp_path, *, changed=None, operation="op-old"):
    source = tmp_path / "full.gguf"
    if not source.exists():
        _make_gguf(source)
    root = tmp_path / "receiver"
    root.mkdir(exist_ok=True)
    kwargs = _kwargs(source)
    old = assets.build_gguf_segment(source, tmp_path / "builder", **kwargs)
    current = assets.build_gguf_segment(source, root, **{**kwargs, "operation_id": "op-complete"})
    manifest = assets.validate_supply_manifest(old)
    manifest = {**manifest, "operation_id": operation}
    if changed:
        manifest.update(changed)
    if changed and "layer_range" in changed:
        start, end = manifest["layer_range"]
        manifest["layer_weight_bytes"] = [32 if start <= layer < end else 0 for layer in range(manifest["total_layers"])]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = assets._digest(manifest)
    manifest = assets.validate_supply_manifest(manifest)
    stage = root / (".supply-receive-" + manifest["manifest_sha256"])
    stage.mkdir()
    weight = next(e for e in manifest["files"] if e["kind"] == "weights")
    data = Path(old["local_files"][weight["path"]]).read_bytes()
    length = len(data) // 2
    partial = stage / (weight["path"] + ".part")
    partial.write_bytes(data[:length])
    assets._ResumeJournal(stage, manifest).checkpoint(weight, length, hashlib.sha256(data[:length]).hexdigest())
    for entry in manifest["files"]:
        if entry["kind"] != "weights":
            (stage / entry["path"]).write_bytes(Path(old["local_files"][entry["path"]]).read_bytes())
    return root, stage, current, partial


def collect(root, current, **kw):
    directory = Path(next(iter(current["local_files"].values()))).parent
    return assets.collect_supply_segments(root, protected_paths=[directory], native_paths=[], keep=0,
                                         receive_idle=True, **kw)


def test_real_interruption_resume_complete_retires_only_canceled_checkpoint(tmp_path):
    source = tmp_path / "full.gguf"; _make_gguf(source)
    kwargs = _kwargs(source)
    first = assets.build_gguf_segment(source, tmp_path / "builder", **kwargs)
    second = assets.build_gguf_segment(source, tmp_path / "builder", **{**kwargs, "operation_id": "op-next"})
    root = tmp_path / "receiver"
    stage, record, partial = _interrupted_prefix(first, root)
    with _server(second) as (url, _):
        result = _receive(second, root, url)
    assert result["resumed_bytes"] == record["prefix_bytes"]
    assert partial.exists()  # Only the exclusive maintenance owner can retire it.
    report = assets.collect_supply_segments(root, protected_paths=[result["model_path"]],
        native_paths=[], keep=0, receive_idle=True)
    assert report["partial_removed"] == 1 and report["removed"] == 1, report
    assert report["reclaimed_bytes"] >= record["prefix_bytes"]
    assert not stage.exists()
    assert assets.sha256_file(result["model_path"]) == second["artifact_sha256"]


@pytest.mark.parametrize("change", ["node", "source", "tokenizer", "range", "role", "proof"])
def test_foreign_identity_does_not_authorize_checkpoint_retirement(change, tmp_path):
    changes = {"node": {"node_id": "different-node"}, "source": {"source_model_sha256": "a"*64},
               "tokenizer": {"tokenizer_sha256": "b"*64}, "range": {"layer_range": [0, 2], "segment_mode": "head"},
               "role": {"layer_range": [2, 4], "segment_mode": "tail"},
               "proof": {"conversion_proof": {"converter_revision": "unrelated"}}}
    root, stage, current, partial = copies(tmp_path, changed=changes[change])
    before = partial.read_bytes()
    report = collect(root, current)
    assert report["removed"] == 0 and partial.read_bytes() == before
    assert stage.is_dir()


@pytest.mark.parametrize("case", ["no_complete", "unknown_native", "receiver_not_excluded", "active_stage", "budget"])
def test_unproven_or_pinned_checkpoint_is_kept(case, tmp_path):
    root, stage, current, partial = copies(tmp_path)
    directory = Path(next(iter(current["local_files"].values()))).parent
    options = dict(protected_paths=[directory], native_paths=[], keep=0, receive_idle=True)
    if case == "no_complete":
        (directory / assets._MARKER).unlink()
    elif case == "unknown_native": options["native_paths"] = None
    elif case == "receiver_not_excluded": options["receive_idle"] = False
    elif case == "active_stage": options["protected_paths"].append(stage)
    else: options["max_hash_bytes"] = 1
    report = assets.collect_supply_segments(root, **options)
    assert report["removed"] == 0 and partial.is_file()


@pytest.mark.parametrize("damage", ["prefix", "unrecorded_suffix", "unsigned_journal", "unknown_file", "missing_record", "witness"])
def test_changed_or_unowned_bytes_are_retained(damage, tmp_path):
    root, stage, current, partial = copies(tmp_path)
    if damage == "prefix":
        data = bytearray(partial.read_bytes()); data[-1] ^= 1; partial.write_bytes(data)
    elif damage == "unrecorded_suffix":
        with partial.open("ab") as out: out.write(b"corrupt")
    elif damage == "unsigned_journal":
        marker = stage / assets._RESUME_MARKER
        journal = json.loads(marker.read_text()); journal["prefixes"][0]["prefix_sha256"] = "0"*64
        marker.write_text(json.dumps(journal))
    elif damage == "unknown_file": (stage / "user-note.txt").write_text("keep")
    elif damage == "missing_record": (stage / assets._RESUME_MARKER).unlink()
    else:
        path = Path(current["local_files"]["segment.gguf"])
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    report = collect(root, current)
    assert report["removed"] == 0 and stage.is_dir()


def test_retirement_shares_a_bounded_deletion_budget(tmp_path):
    root, first, current, _ = copies(tmp_path)
    _, second, _, _ = copies(tmp_path, operation="op-another")
    report = collect(root, current, max_remove=1)
    assert report["partial_removed"] == 1
    assert int(first.exists()) + int(second.exists()) == 1


def test_busy_receiver_excludes_gc_and_prefix_donor_retirement(setup):
    worker, _, _ = setup
    entered, release = threading.Event(), threading.Event()
    def receiver():
        with worker._execution_lock:
            entered.set()
            release.wait(5)
    thread = threading.Thread(target=receiver)
    thread.start()
    assert entered.wait(3)
    try:
        report = worker.collect_garbage()
        assert report["removed"] == 0 and report["skipped"] == {"receiver_busy": 1}
    finally:
        release.set(); thread.join(3)
    assert not thread.is_alive()


def test_linked_partial_is_not_retired(tmp_path):
    import os
    root, stage, current, partial = copies(tmp_path)
    outside = tmp_path / "outside-partial"
    os.link(partial, outside)
    before = outside.read_bytes()
    assert collect(root, current)["removed"] == 0
    assert stage.exists() and outside.read_bytes() == before


def test_pending_selection_invalidates_retirement_snapshot(setup, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    worker, _, _ = setup
    @contextmanager
    def native(_root):
        yield []
    monkeypatch.setattr(worker, "_native_gc_paths", native)
    observed = []
    def collect_owned(*args, **kw):
        assert kw["receive_idle"] is True
        assert not kw["cancelled"]()
        with worker._lock:
            worker._pending = SimpleNamespace(operation_id="new-owner")
        observed.append(kw["cancelled"]())
        return {"removed": 0, "reclaimed_bytes": 0, "examined": 0, "skipped": {"cancelled": 1}}
    monkeypatch.setattr(assets, "collect_supply_segments", collect_owned)
    report = worker.collect_garbage()
    assert observed == [True] and report["removed"] == 0


def test_partial_changed_during_hash_is_not_deleted(tmp_path, monkeypatch):
    root, stage, current, partial = copies(tmp_path)
    # Mutation occurs after the complete witness was hashed, while the GC is
    # about to compare the old bytes. Revalidate both files before deletion.
    original_hash = assets._hash_file
    changed = []
    def mutate(path, cancelled):
        result = original_hash(path, cancelled)
        if Path(path).name == "segment.gguf" and not changed:
            changed.append(True)
            with partial.open("ab") as out:
                out.write(b"unverified-suffix")
        return result
    monkeypatch.setattr(assets, "_hash_file", mutate)
    assert collect(root, current)["removed"] == 0
    assert changed and stage.is_dir()


def test_cancellation_preserves_canceled_checkpoint(tmp_path):
    root, stage, current, partial = copies(tmp_path)
    before = partial.read_bytes()
    with pytest.raises(assets.SupplyError) as error:
        collect(root, current, cancelled=lambda: True)
    assert error.value.code == "cancelled"
    assert partial.read_bytes() == before


def test_symlink_member_keeps_outside_data_and_checkpoint(tmp_path):
    root, stage, current, partial = copies(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir(); sentinel = outside / "sentinel"; sentinel.write_bytes(b"user")
    link = stage / "unowned"
    import sys
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(outside), str(link))
    else:
        link.symlink_to(outside, target_is_directory=True)
    assert collect(root, current)["removed"] == 0
    assert stage.is_dir() and sentinel.read_bytes() == b"user"
