"""Issue94: bounded, verified retention across model switches and node moves."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import model_supply_assets as assets
from tests.test_model_supply_assets import (
    _cross_format_fixture, _kwargs, _make_gguf, _receive,
    _safetensors_fixture, _server,
)


def _source(tmp_path, fmt, name="a"):
    source = tmp_path / (name + ".gguf" if fmt == "gguf" else name)
    (_make_gguf(source) if fmt == "gguf" else _safetensors_fixture(source))
    if name != "a":
        weight = source if fmt == "gguf" else source / "model.safetensors"
        data = bytearray(weight.read_bytes())
        data[-1] ^= 1
        weight.write_bytes(data)
    return source


def _builder(fmt):
    return assets.build_gguf_segment if fmt == "gguf" else assets.build_safetensors_segment


def _directory(manifest):
    return Path(next(iter(manifest["local_files"].values()))).parent


def _size(manifest):
    return sum(entry["size_bytes"] for entry in manifest["files"]) + (_directory(manifest) / assets._MARKER).stat().st_size


def _forbid_generation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("verified retained tensors must not be generated again")
    monkeypatch.setattr(assets, "_run_cut", forbidden)
    monkeypatch.setattr(assets, "_copy_selected", forbidden)


def _collect(root, **kwargs):
    return assets.collect_supply_segments(root, native_paths=[], keep=2,
        max_retained_bytes=1024**2, **kwargs)


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_a_b_a_reuses_retained_builder_and_receiver_without_cut_or_http(fmt, tmp_path, monkeypatch):
    source_a, source_b = _source(tmp_path, fmt), _source(tmp_path, fmt, "b")
    builder, master, worker = _builder(fmt), tmp_path / "master", tmp_path / "worker"
    first = builder(source_a, master, **_kwargs(source_a))
    with _server(first) as (url, _):
        received_a = _receive(first, worker, url)
    second = builder(source_b, master, **{**_kwargs(source_b), "operation_id": "op-b"})
    with _server(second) as (url, _):
        received_b = _receive(second, worker, url)
    master_report = _collect(master, protected_paths=[_directory(second)])
    worker_report = _collect(worker, protected_paths=[received_b["model_path"]])
    assert master_report["retained"] == worker_report["retained"] == 1
    assert master_report["retained_bytes"] == _size(first)
    _forbid_generation(monkeypatch)
    third = builder(source_a, master, **{**_kwargs(source_a), "operation_id": "op-a-again"})
    assert third["_reused_bytes"] == sum(entry["size_bytes"] for entry in first["files"])
    with _server(third) as (url, state):
        received_again = _receive(third, worker, url)
    assert not state["requests"]
    assert received_again["actual_downloaded_bytes"] == 0
    assert received_again["manifest"]["operation_id"] == "op-a-again"
    for entry in first["files"]:
        assert os.path.samefile(received_a["local_files"][entry["path"]],
                                received_again["local_files"][entry["path"]])


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_builder_moves_node_with_new_receipt_and_identical_verified_files(fmt, tmp_path, monkeypatch):
    source = _source(tmp_path, fmt)
    kwargs, root = _kwargs(source), tmp_path / "operations"
    builder = _builder(fmt)
    first = builder(source, root / "op-1" / "worker-1", **kwargs, reuse_root=root)
    original = (_directory(first) / assets._MARKER).read_bytes()
    changed = {**kwargs, "operation_id": "op-moved", "node_id": "worker-2"}
    assert assets.segment_cache_hint(root, {**assets.validate_supply_manifest(first), **changed,
        "operation_id": "op-moved"})
    _forbid_generation(monkeypatch)
    second = builder(source, root / "op-moved" / "worker-2", **changed, reuse_root=root)
    assert second["node_id"] == "worker-2" and second["operation_id"] == "op-moved"
    assert second["manifest_sha256"] != first["manifest_sha256"]
    assert second["files"] == first["files"]
    assert (_directory(first) / assets._MARKER).read_bytes() == original
    for entry in first["files"]:
        assert os.path.samefile(first["local_files"][entry["path"]], second["local_files"][entry["path"]])
    assert assets.reuse_segment(second, root)["manifest"]["node_id"] == "worker-2"


def test_retention_byte_limit_includes_marker_and_keeps_newest_fitting_segment(tmp_path):
    source_a, source_b = _source(tmp_path, "safetensors"), _source(tmp_path, "safetensors", "b")
    root = tmp_path / "cache"
    first = assets.build_safetensors_segment(source_a, root, **_kwargs(source_a))
    second = assets.build_safetensors_segment(source_b, root, **{**_kwargs(source_b), "operation_id": "op-b"})
    os.utime(_directory(first), ns=(1_000_000, 1_000_000))
    report = assets.collect_supply_segments(root, native_paths=[], keep=2, max_retained_bytes=_size(second))
    assert report["retained"] == 1 and report["retained_bytes"] == _size(second)
    assert report["removed"] == 1
    assert not _directory(first).exists() and _directory(second).is_dir()
    report = assets.collect_supply_segments(root, native_paths=[], keep=2, max_retained_bytes=_size(second) - 1)
    assert report["removed"] == 1 and report["retained_bytes"] == 0


def test_duplicate_operation_receipts_cannot_crowd_out_other_model(tmp_path):
    source_a, source_b = _source(tmp_path, "safetensors"), _source(tmp_path, "safetensors", "b")
    root = tmp_path / "cache"
    first = assets.build_safetensors_segment(source_a, root, **_kwargs(source_a))
    copies = [assets.build_safetensors_segment(source_b, root,
        **{**_kwargs(source_b), "operation_id": "op-b-" + str(index)}) for index in range(4)]
    for index, manifest in enumerate([first, *copies], 1):
        os.utime(_directory(manifest), ns=(index * 1_000_000, index * 1_000_000))
    report = _collect(root)
    assert report["retained"] == 2 and report["removed"] == 3
    assert _directory(first).is_dir() and _directory(copies[-1]).is_dir()
    assert all(not _directory(item).exists() for item in copies[:-1])


@pytest.mark.parametrize("protection", ["native", "lease", "unknown"])
def test_zero_cache_budget_preserves_native_and_owner_lease_references(protection, tmp_path):
    source = _source(tmp_path, "gguf")
    root = tmp_path / "cache"
    manifest = assets.build_gguf_segment(source, root, **_kwargs(source))
    kwargs = {"native_paths": [], "protected_paths": []}
    if protection == "unknown":
        kwargs["native_paths"] = None
    else:
        kwargs["native_paths" if protection == "native" else "protected_paths"] = [manifest["local_files"]["segment.gguf"]]
    report = assets.collect_supply_segments(root, keep=2, max_retained_bytes=0, **kwargs)
    assert report["removed"] == 0 and _directory(manifest).is_dir()
    assert report["retained_bytes"] == 0


@pytest.mark.parametrize("bound,value", [("keep", True), ("keep", -1), ("max_retained_bytes", -1),
    ("max_retained_bytes", True), ("max_retained_bytes", 1.5), ("max_hash_bytes", -1)])
def test_retention_rejects_invalid_bounds(bound, value, tmp_path):
    with pytest.raises(assets.SupplyError) as error:
        assets.collect_supply_segments(tmp_path, native_paths=[], **{bound: value})
    assert error.value.code == "invalid_retention"


def test_corrupt_recent_receipt_does_not_consume_the_retention_slot(tmp_path):
    source_a, source_b = _source(tmp_path, "safetensors"), _source(tmp_path, "safetensors", "b")
    root = tmp_path / "cache"
    first = assets.build_safetensors_segment(source_a, root, **_kwargs(source_a))
    corrupt = assets.build_safetensors_segment(source_b, root, **{**_kwargs(source_b), "operation_id": "op-b"})
    weight = Path(next(value for name, value in corrupt["local_files"].items() if name.endswith(".safetensors")))
    data = bytearray(weight.read_bytes()); data[-1] ^= 1; weight.write_bytes(data)
    os.utime(_directory(first), ns=(1_000_000, 1_000_000))
    report = assets.collect_supply_segments(root, native_paths=[], keep=1, max_retained_bytes=1024**2)
    assert report["retained"] == 1 and report["retained_bytes"] == _size(first)
    assert report["skipped"]["file_digest_mismatch"] == 1
    assert _directory(first).is_dir() and _directory(corrupt).is_dir()


def test_cross_node_cross_format_reuse_reverifies_source_provenance(tmp_path, monkeypatch):
    source, target = tmp_path / "hf", tmp_path / "converted.gguf"
    kwargs = _cross_format_fixture(source, target)
    root = tmp_path / "cache"
    first = assets.build_safetensors_segment(source, root, **kwargs)
    _forbid_generation(monkeypatch)
    moved = {**kwargs, "operation_id": "op-moved", "node_id": "worker-2"}
    second = assets.build_safetensors_segment(source, root, **moved)
    assert second["node_id"] == "worker-2" and second["conversion_proof"] == first["conversion_proof"]
    proof = json.loads((source / "cross_format.provenance.json").read_text())
    proof["conversion"]["converter_revision"] = "e" * 40
    (source / "cross_format.provenance.json").write_text(json.dumps(proof))
    with pytest.raises(assets.SupplyError) as error:
        assets.build_safetensors_segment(source, root, **{**moved, "operation_id": "op-tampered"})
    assert error.value.code == "conversion_proof_invalid"


def test_changed_conversion_proof_still_generates_new_tensors_on_node_move(tmp_path, monkeypatch):
    source = _source(tmp_path, "gguf")
    kwargs, root = _kwargs(source), tmp_path / "cache"
    first = assets.build_gguf_segment(source, root, **kwargs)
    calls, original = [], assets._run_cut
    def cut(*args, **kw):
        calls.append(True)
        return original(*args, **kw)
    monkeypatch.setattr(assets, "_run_cut", cut)
    second = assets.build_gguf_segment(source, root, **{**kwargs, "operation_id": "op-upgrade",
        "node_id": "worker-2", "conversion_proof": {"converter_revision": "next-version"}})
    assert calls == [True]
    assert not os.path.samefile(first["local_files"]["segment.gguf"], second["local_files"]["segment.gguf"])


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_cross_node_rejects_self_consistent_cache_with_wrong_source_tensor_bytes(fmt, tmp_path, monkeypatch):
    source, root = _source(tmp_path, fmt), tmp_path / "cache"
    kwargs, builder = _kwargs(source), _builder(fmt)
    first = builder(source, root, **kwargs)
    directory = _directory(first)
    payload = assets.validate_supply_manifest(first)
    weight = next(entry for entry in payload["files"] if entry["kind"] == "weights")
    path = directory / weight["path"]
    if fmt == "gguf":
        _, tensors = assets._gguf_table(path)
        offset = next(value[0] for name, value in tensors.items() if name.startswith("blk."))
    else:
        header, start = assets._safetensors_header(path)
        offset = start + next(iter(header.values()))["data_offsets"][0]
    data = bytearray(path.read_bytes()); data[offset] ^= 1; path.write_bytes(data)
    if fmt == "gguf":
        native_path = path.with_suffix(".manifest.json")
        native = json.loads(native_path.read_text())
        native["artifact_sha256"] = assets.sha256_file(path)
        assets._write_json(native_path, native)
    payload["files"] = assets._inventory(directory, {entry["path"]: entry["kind"] for entry in payload["files"]})
    payload["artifact_sha256"] = assets.sha256_file(path) if fmt == "gguf" else assets._weight_digest(payload["files"])
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = assets._digest(payload)
    assets._write_json(directory / assets._MARKER, payload)
    # Both inventory SHA and local component checks pass. A node move must
    # additionally verify that tensor contents derive from the claimed source.
    assets._verify_files(directory, payload)
    assets._verify_segment(directory, payload)
    original = assets._run_cut if fmt == "gguf" else assets._copy_selected
    calls = []
    def generate(*args, **kw):
        calls.append(True)
        return original(*args, **kw)
    monkeypatch.setattr(assets, "_run_cut" if fmt == "gguf" else "_copy_selected", generate)
    second = builder(source, root, **{**kwargs, "operation_id": "op-move", "node_id": "worker-2"})
    assert calls
    assert second["artifact_sha256"] == first["artifact_sha256"]
    assert second.get("_reused_bytes", 0) == 0


def test_valid_cross_format_converter_upgrade_invalidates_reuse(tmp_path, monkeypatch):
    source, target, root = tmp_path / "hf", tmp_path / "converted.gguf", tmp_path / "cache"
    kwargs = _cross_format_fixture(source, target)
    first = assets.build_safetensors_segment(source, root, **kwargs)
    proof = json.loads((source / "cross_format.provenance.json").read_text())
    proof["conversion"]["converter_revision"] = "e" * 40
    proof.pop("provenance_sha256")
    proof["provenance_sha256"] = assets._digest(proof)
    assets._write_json(source / "cross_format.provenance.json", proof)
    original, calls = assets._copy_selected, []
    def copy(*args, **kw):
        calls.append(True)
        return original(*args, **kw)
    monkeypatch.setattr(assets, "_copy_selected", copy)
    second = assets.build_safetensors_segment(source, root, **{**kwargs,
        "operation_id": "op-upgrade", "node_id": "worker-2", "conversion_proof": proof})
    assert calls and second["conversion_proof"] != first["conversion_proof"]
    assert second.get("_reused_bytes", 0) == 0


def test_cache_budget_eviction_causes_real_regeneration(tmp_path, monkeypatch):
    source, root = _source(tmp_path, "gguf"), tmp_path / "cache"
    kwargs = _kwargs(source)
    first = assets.build_gguf_segment(source, root, **kwargs)
    report = assets.collect_supply_segments(root, native_paths=[], keep=2, max_retained_bytes=0)
    assert report["removed"] == 1 and not _directory(first).exists()
    original, calls = assets._run_cut, []
    def cut(*args, **kw):
        calls.append(True)
        return original(*args, **kw)
    monkeypatch.setattr(assets, "_run_cut", cut)
    second = assets.build_gguf_segment(source, root, **{**kwargs, "operation_id": "op-after-eviction"})
    assert calls == [True] and second["artifact_sha256"] == first["artifact_sha256"]


def test_retention_cancellation_during_hash_preserves_generation(tmp_path, monkeypatch):
    source, root = _source(tmp_path, "gguf"), tmp_path / "cache"
    manifest = assets.build_gguf_segment(source, root, **_kwargs(source))
    cancelled, original = [], assets._hash_file
    def hash_then_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled.append(True)
        return result
    monkeypatch.setattr(assets, "_hash_file", hash_then_cancel)
    with pytest.raises(assets.SupplyError) as error:
        assets.collect_supply_segments(root, native_paths=[], keep=2, max_retained_bytes=0,
            cancelled=lambda: bool(cancelled))
    assert error.value.code == "cancelled" and _directory(manifest).is_dir()


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_warm_retention_does_not_rehash_weight_payloads(fmt, tmp_path, monkeypatch):
    source, root = _source(tmp_path, fmt), tmp_path / "cache"
    manifest = _builder(fmt)(source, root, **_kwargs(source))
    assets._verify_files(_directory(manifest), manifest)
    hashes, scans = [], []
    original_hash, original_candidates = assets._hash_file, assets._segment_candidates

    def hash_file(path, cancelled=None):
        hashes.append(Path(path))
        return original_hash(path, cancelled)

    def candidates(*args, **kwargs):
        scans.append(True)
        return original_candidates(*args, **kwargs)

    monkeypatch.setattr(assets, "_hash_file", hash_file)
    monkeypatch.setattr(assets, "_segment_candidates", candidates)
    for _ in range(2):
        report = _collect(root, receive_idle=True)
        assert report["retained"] == 1 and report["removed"] == 0
        assert report["retained_bytes"] == _size(manifest)
    weights = {Path(manifest["local_files"][entry["path"]])
               for entry in manifest["files"] if entry["kind"] == "weights"}
    assert weights.isdisjoint(hashes)
    assert len(scans) == 2


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_retention_revalidates_changed_cached_weights(fmt, tmp_path, monkeypatch):
    source, root = _source(tmp_path, fmt), tmp_path / "cache"
    manifest = _builder(fmt)(source, root, **_kwargs(source))
    directory = _directory(manifest)
    assets._verify_files(directory, manifest)
    weight = Path(next(manifest["local_files"][entry["path"]]
                       for entry in manifest["files"] if entry["kind"] == "weights"))
    before = weight.stat()
    content = bytearray(weight.read_bytes())
    content[-1] ^= 1
    weight.write_bytes(content)
    os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    hashes, original = [], assets._hash_file

    def hash_file(path, cancelled=None):
        hashes.append(Path(path))
        return original(path, cancelled)

    monkeypatch.setattr(assets, "_hash_file", hash_file)
    report = _collect(root)
    assert report["retained"] == report["removed"] == 0
    assert report["skipped"]["file_digest_mismatch"] == 1
    assert weight in hashes and directory.is_dir()


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_eviction_still_hashes_live_weights_after_cached_retention(fmt, tmp_path, monkeypatch):
    source, root = _source(tmp_path, fmt), tmp_path / "cache"
    manifest = _builder(fmt)(source, root, **_kwargs(source))
    assert _collect(root)["retained"] == 1
    hashes, original = [], assets._hash_file

    def hash_file(path, cancelled=None):
        hashes.append(Path(path))
        return original(path, cancelled)

    monkeypatch.setattr(assets, "_hash_file", hash_file)
    report = assets.collect_supply_segments(root, native_paths=[], keep=0)
    weights = {Path(manifest["local_files"][entry["path"]])
               for entry in manifest["files"] if entry["kind"] == "weights"}
    assert report["removed"] == 1 and not _directory(manifest).exists()
    assert weights.issubset(hashes)


def test_exhausted_removal_budget_does_not_hash_an_eviction_candidate(tmp_path, monkeypatch):
    source, root = _source(tmp_path, "gguf"), tmp_path / "cache"
    manifest = assets.build_gguf_segment(source, root, **_kwargs(source))

    def forbidden(*args, **kwargs):
        pytest.fail("an exhausted removal budget must not trigger payload verification")

    monkeypatch.setattr(assets, "_hash_file", forbidden)
    monkeypatch.setattr(assets, "_verify_segment", forbidden)
    report = assets.collect_supply_segments(root, native_paths=[], keep=0, max_remove=0)
    assert report["removed"] == 0 and _directory(manifest).is_dir()
    assert report["skipped"] == {"removal_budget": 1}


def _historical_operations(root, count):
    root.mkdir()
    for index in range(count):
        operation = root / f"supply_{index:04d}"
        (operation / "master").mkdir(parents=True)
        (operation / "source").mkdir()
        for name in ("intent.json", "operation.json"):
            (operation / name).write_text("{}", encoding="utf-8")
        os.utime(operation, ns=(index + 1, index + 1))


@pytest.mark.parametrize("history_count", [180, assets._CACHE_SCAN_LIMIT + 17])
@pytest.mark.parametrize("fmt,protection", [("gguf", "native"), ("safetensors", "lease")])
def test_a_b_a_many_operations_reuses_recent_verified_cache(
        history_count, fmt, protection, tmp_path, monkeypatch):
    source_a, source_b = _source(tmp_path, fmt), _source(tmp_path, fmt, "b")
    root, builder = tmp_path / "operations", _builder(fmt)
    _historical_operations(root, history_count)
    kwargs_a = {**_kwargs(source_a), "node_id": "master", "operation_id": "supply_zz-a"}
    kwargs_b = {**_kwargs(source_b), "node_id": "master", "operation_id": "supply_zz-b"}
    first = builder(source_a, root / kwargs_a["operation_id"] / "master", **kwargs_a, reuse_root=root)
    second = builder(source_b, root / kwargs_b["operation_id"] / "master", **kwargs_b, reuse_root=root)
    for ordinal, kwargs in enumerate((kwargs_a, kwargs_b), 1):
        os.utime(root / kwargs["operation_id"], ns=(ordinal * 10**9, ordinal * 10**9))
    original_marker = (_directory(first) / assets._MARKER).read_bytes()
    candidates = list(assets._segment_candidates(root))
    protected = [next(iter(second["local_files"].values()))]
    report = assets.collect_supply_segments(root, keep=2, max_retained_bytes=1024**2,
        native_paths=protected if protection == "native" else [],
        protected_paths=protected if protection == "lease" else [])
    changed = {**kwargs_a, "operation_id": "supply_zz-a-again"}
    hint = assets.segment_cache_hint(root, {**assets.validate_supply_manifest(first), **changed})
    _forbid_generation(monkeypatch)
    third = builder(source_a, root / changed["operation_id"] / "master", **changed, reuse_root=root)
    assert _directory(first) in candidates and hint
    assert report["examined"] == 2 and report["retained"] == 1
    assert report["retained_bytes"] == _size(first) and report["skipped"]["referenced"] == 1
    assert third["_reused_bytes"] == sum(entry["size_bytes"] for entry in first["files"])
    assert third["files"] == first["files"]
    assert third["operation_id"] == changed["operation_id"]
    assert (_directory(first) / assets._MARKER).read_bytes() == original_marker
    assert _directory(second).is_dir()
    for name in first["local_files"]:
        assert os.path.samefile(first["local_files"][name], third["local_files"][name])


def test_candidate_scan_has_separate_finite_root_and_descendant_budgets(tmp_path, monkeypatch):
    root = tmp_path / "operations"
    root.mkdir()
    # The newest operation is inside the bounded root prefix; its oversized
    # metadata directory then exhausts the independent descendant budget.
    for index in range(assets._CACHE_ROOT_SCAN_LIMIT + 3):
        operation = root / f"supply_{index:04d}"
        operation.mkdir()
        os.utime(operation, ns=(1, 1))
    latest = root / "supply_0000"
    node = latest / "master"
    node.mkdir()
    for index in range(assets._CACHE_SCAN_LIMIT + 17):
        (node / f"metadata-{index:04d}.json").touch()
    os.utime(latest, ns=(10**9, 10**9))
    counts = {"root": 0, "descendant": 0, "opened": 0}
    original = os.scandir
    class CountedScan:
        def __init__(self, directory):
            self.directory = Path(directory)
        def __enter__(self):
            counts["opened"] += 1
            self.entries = original(self.directory)
            return self
        def __iter__(self):
            return self
        def __next__(self):
            entry = next(self.entries)
            key = "root" if self.directory == root else "descendant"
            counts[key] += 1
            limit = assets._CACHE_ROOT_SCAN_LIMIT if key == "root" else assets._CACHE_SCAN_LIMIT
            assert counts[key] <= limit, "directory discovery exceeded its hard bound"
            return entry
        def __exit__(self, *args):
            self.entries.close()
    def forbidden(*args, **kwargs):
        pytest.fail("candidate discovery must not read metadata or hash payloads")
    monkeypatch.setattr(assets.os, "scandir", CountedScan)
    monkeypatch.setattr(assets, "_hash_file", forbidden)
    monkeypatch.setattr(assets, "_read_json", forbidden)
    assert list(assets._segment_candidates(root)) == []
    assert counts["root"] == assets._CACHE_ROOT_SCAN_LIMIT
    assert counts["descendant"] == assets._CACHE_SCAN_LIMIT
    assert counts["opened"] <= 1 + assets._CACHE_ROOT_SCAN_LIMIT + assets._CACHE_SCAN_LIMIT


def test_candidate_limit_prefers_recent_generations(tmp_path):
    generations = []
    for index in range(assets._CACHE_GENERATION_LIMIT + 17):
        directory = tmp_path / f"segment-{index:064x}"
        directory.mkdir()
        os.utime(directory, ns=(index + 1, index + 1))
        generations.append(directory)
    candidates = list(assets._segment_candidates(tmp_path))
    assert candidates == list(reversed(generations))[:assets._CACHE_GENERATION_LIMIT]


@pytest.mark.parametrize("phase", ["root", "descendant"])
def test_candidate_scan_cancellation_propagates_during_discovery(phase, tmp_path, monkeypatch):
    root = tmp_path / "operations"
    _historical_operations(root, assets._CACHE_SCAN_LIMIT + 17)
    latest = root / "supply_zz-latest"
    generation = latest / "master" / ("segment-" + "a" * 64)
    generation.mkdir(parents=True)
    state = {"checks": 0, "descendant": False}
    original = os.scandir
    def scan(directory):
        if Path(directory) == latest:
            state["descendant"] = True
        return original(directory)
    def cancelled():
        state["checks"] += 1
        return state["checks"] >= 64 if phase == "root" else state["descendant"]
    monkeypatch.setattr(assets.os, "scandir", scan)
    with pytest.raises(assets.SupplyError) as error:
        list(assets._segment_candidates(root, cancelled))
    assert error.value.code == "cancelled" and generation.is_dir()
