"""Supply inventory and registration reuse must keep live-version fences."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

import config
import model_supply_assets as assets
import model_supply_digest_cache as digests
import model_supply_worker as module
import relay_segment_info
from tests.test_model_supply_worker import Scheduler, command, finish, gguf, segment, setup


def master(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    scheduler = Scheduler("master")
    worker = module.ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    scheduler.worker = worker
    return worker, scheduler


def no_read(*args, **kwargs):
    pytest.fail("unchanged cached file was read again")


def mutate(path):
    before = path.stat()
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_shared_digest_survives_a_fresh_process(tmp_path):
    source = gguf(tmp_path / "source.gguf", 4)
    expected = digests.cached_sha256(source)
    code = """
import sys
from pathlib import Path
import config
config.STATE_DIR = sys.argv[1]
from model_supply_digest_cache import cached_sha256
def reject(*args):
    raise AssertionError("cold process reread unchanged payload")
assert cached_sha256(Path(sys.argv[2]), hash_file=reject) == sys.argv[3]
"""
    env = dict(os.environ, PYTHONPATH=str(Path(module.__file__).parent))
    result = subprocess.run([sys.executable, "-c", code, str(config.STATE_DIR), str(source), expected],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_shared_digest_uses_one_store_across_worker_roots(tmp_path, monkeypatch):
    source = gguf(tmp_path / "source.gguf")
    expected = module._file_sha_cached(tmp_path / "first", source)
    monkeypatch.setattr(module, "_file_sha", no_read)
    assert module._file_sha_cached(tmp_path / "second", source) == expected
    cache = digests._store_path()
    assert cache.is_file()
    assert not (tmp_path / "first" / "source-digests.json").exists()
    assert list(json.loads(cache.read_text())["entries"]) == [str(source.resolve())]


def test_digest_rejects_same_size_mtime_restored_change(tmp_path):
    source = gguf(tmp_path / "source.gguf")
    old = digests.cached_sha256(source)
    mutate(source)
    calls = []
    def live(path, cancelled):
        calls.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assert digests.cached_sha256(source, hash_file=live) != old
    assert calls == [source]


def test_digest_cache_hit_observes_cancellation(tmp_path):
    source = gguf(tmp_path / "source.gguf")
    digests.cached_sha256(source)
    with pytest.raises(digests.DigestCacheError, match="cancelled"):
        digests.cached_sha256(source, lambda: True, hash_file=no_read)


def test_digest_change_during_hash_is_not_persisted(tmp_path):
    source = gguf(tmp_path / "source.gguf")
    def changing(path, cancelled):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        mutate(path)
        return digest
    with pytest.raises(digests.DigestCacheError, match="file_digest_mismatch"):
        digests.cached_sha256(source, hash_file=changing)
    assert str(source) not in digests._entries(digests._store_path())


def test_concurrent_digest_misses_hash_once(tmp_path):
    source = gguf(tmp_path / "source.gguf")
    entered, release = threading.Event(), threading.Event()
    calls = []
    def live(path, cancelled):
        calls.append(path)
        entered.set()
        assert release.wait(5)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(digests.cached_sha256, source, hash_file=live)
        assert entered.wait(2)
        second = pool.submit(digests.cached_sha256, source, hash_file=live)
        release.set()
        assert first.result(timeout=5) == second.result(timeout=5)
    assert calls == [source]


def test_metadata_items_snapshots_entries_under_lock(tmp_path, monkeypatch):
    source = gguf(tmp_path / "source.gguf")
    other = gguf(tmp_path / "other.gguf")
    namespace = "ranges-v1"
    digests.cached_metadata_items(source, namespace, ("first",),
                                  lambda path, missing: {"first": "cached"})
    store = digests._store_path()
    scanning, writer_checked, writer_saved = threading.Event(), threading.Event(), threading.Event()

    class RacingEntries(dict):
        def values(self):
            iterator = iter(super().values())
            yield next(iterator)
            scanning.set()
            assert writer_checked.wait(5)
            yield from iterator

    monkeypatch.setitem(digests._STORES, str(store), RacingEntries(digests._entries(store)))

    def save_other():
        assert scanning.wait(5)
        acquired = digests._LOCK.acquire(blocking=False)
        if acquired:
            try:
                digests._save(store, other, {"stamp": list(digests.file_stamp(other))})
            finally:
                digests._LOCK.release()
            writer_checked.set()
        else:
            writer_checked.set()
            digests._save(store, other, {"stamp": list(digests.file_stamp(other))})
        writer_saved.set()

    def compute(path, missing):
        assert writer_saved.wait(5), "metadata computation held the store lock"
        assert path == source
        assert missing == ("second",)
        return {"second": "computed"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(save_other)
        reader = pool.submit(digests.cached_metadata_items, source, namespace,
                             ("first", "second"), compute)
        assert reader.result(timeout=10) == {"first": "cached", "second": "computed"}
        writer.result(timeout=10)
    assert str(other) in json.loads(store.read_text())["entries"]


@pytest.mark.parametrize("reload_store", [False, True])
def test_metadata_items_full_hit_does_not_save(tmp_path, monkeypatch, reload_store):
    source = gguf(tmp_path / "source.gguf")
    expected = {"first": {"sha256": "a" * 64}}
    assert digests.cached_metadata_items(source, "ranges-v1", expected,
                                         lambda path, missing: deepcopy(expected)) == expected
    store = digests._store_path()
    before = store.read_bytes(), digests.file_stamp(store)
    if reload_store:
        monkeypatch.delitem(digests._STORES, str(store))

    def reject_save(*args):
        pytest.fail("metadata cache hit rewrote digest store")

    monkeypatch.setattr(digests, "_save", reject_save)
    result = digests.cached_metadata_items(source, "ranges-v1", expected, no_read)
    assert result == expected
    result["first"]["sha256"] = "b" * 64
    assert digests.cached_metadata_items(source, "ranges-v1", expected, no_read) == expected
    assert (store.read_bytes(), digests.file_stamp(store)) == before


@pytest.mark.parametrize("reuse", [False, True])
def test_metadata_items_persists_only_added_values(tmp_path, monkeypatch, reuse):
    source = gguf(tmp_path / "source.gguf")
    digests.cached_metadata_items(source, "ranges-v1", ("first",),
                                  lambda path, missing: {"first": "cached"})
    target = gguf(tmp_path / "other.gguf") if reuse else source
    keys = ("first",) if reuse else ("first", "second")
    expected = {"first": "cached"} if reuse else {"first": "cached", "second": "computed"}
    calls = []
    original_save = digests._save

    def save(store, path, entry):
        calls.append(path)
        original_save(store, path, entry)

    def compute(path, missing):
        assert not reuse
        assert path == target
        assert missing == ("second",)
        return {"second": "computed"}

    monkeypatch.setattr(digests, "_save", save)
    assert digests.cached_metadata_items(target, "ranges-v1", keys, compute) == expected
    assert calls == [target]
    store = digests._store_path()
    payload = json.loads(store.read_text())
    assert payload["schema_version"] == 1
    assert payload["entries"][str(target)]["metadata"]["ranges-v1"] == expected
    monkeypatch.delitem(digests._STORES, str(store))
    assert digests.cached_metadata_items(target, "ranges-v1", keys, no_read) == expected
    assert calls == [target]


@pytest.mark.parametrize("state", ["hit", "reuse", "miss"])
@pytest.mark.parametrize("error", ["cancelled", "file_digest_mismatch"])
def test_metadata_items_keeps_final_fences(tmp_path, monkeypatch, state, error):
    source = gguf(tmp_path / "source.gguf")
    digests.cached_metadata_items(source, "ranges-v1", ("first",),
                                  lambda path, missing: {"first": "cached"})
    target = gguf(tmp_path / "other.gguf") if state == "reuse" else source
    keys = ("first", "second") if state == "miss" else ("first",)
    store = digests._store_path()
    before = store.read_bytes(), deepcopy(digests._entries(store))
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        if checks == (4 if state == "miss" else 3):
            if error == "cancelled":
                return True
            mutate(target)
        return False

    def compute(path, missing):
        assert state == "miss"
        assert missing == ("second",)
        return {"second": "computed"}

    monkeypatch.setattr(digests, "_save", no_read)
    with pytest.raises(digests.DigestCacheError, match=error):
        digests.cached_metadata_items(target, "ranges-v1", keys, compute, cancelled)
    assert (store.read_bytes(), digests._entries(store)) == before


def test_metadata_items_full_hit_still_rejects_conflicts(tmp_path, monkeypatch):
    source = gguf(tmp_path / "source.gguf")
    other = gguf(tmp_path / "other.gguf")
    digests.cached_metadata_items(source, "ranges-v1", ("first",),
                                  lambda path, missing: {"first": "cached"})
    store = digests._store_path()
    digests._save(store, other, {"stamp": list(digests.file_stamp(other)),
                                "metadata": {"ranges-v1": {"first": "conflicting"}}})
    monkeypatch.setattr(digests, "_save", no_read)
    with pytest.raises(digests.DigestCacheError, match="metadata_conflict"):
        digests.cached_metadata_items(source, "ranges-v1", ("first",), no_read)


def test_inventory_reuses_metadata_for_aliases_rounds_and_restart(setup, tmp_path, monkeypatch):
    worker, _, _ = setup
    source = gguf(tmp_path / "source.gguf", 4)
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter([
        ("source-one", source), ("source-one-gguf", source)]))
    calls = {"layers": 0, "tokenizer": 0, "hash": 0}
    def recording(name, original):
        def invoke(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)
        return invoke
    monkeypatch.setattr(module, "_file_sha", recording("hash", module._file_sha))
    monkeypatch.setattr(relay_segment_info, "read_gguf_layer_info",
                        recording("layers", relay_segment_info.read_gguf_layer_info))
    monkeypatch.setattr(relay_segment_info, "read_gguf_tokenizer_sha256",
                        recording("tokenizer", relay_segment_info.read_gguf_tokenizer_sha256))
    first = worker.inventory()["sources"]
    assert len(first) == 2
    assert worker.inventory()["sources"] == first
    digests._STORES.clear()
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path != source, "warm inventory reopened GGUF payload"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    assert worker.inventory()["sources"] == first
    assert calls == {"layers": 1, "tokenizer": 1, "hash": 1}


def test_inventory_refreshes_changed_source_metadata(setup, tmp_path, monkeypatch):
    worker, _, _ = setup
    source = gguf(tmp_path / "source.gguf", 4)
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter([("source-one", source)]))
    old = worker.inventory()["sources"][0]["source_model_sha256"]
    before = source.stat()
    source.write_bytes(source.read_bytes().replace(b"gpt2", b"test"))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    current = worker.inventory()["sources"][0]
    assert current["source_model_sha256"] != old
    assert current["tokenizer_sha256"] == relay_segment_info.read_gguf_tokenizer_sha256(source)
    assert list(worker._sources) == [("source-one", current["source_model_sha256"], current["tokenizer_sha256"])]


@pytest.mark.parametrize("fmt", ["gguf", "safetensors"])
def test_warm_registration_skips_payload_and_metadata(tmp_path, monkeypatch, fmt):
    worker, scheduler = master(tmp_path, monkeypatch)
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master", fmt=fmt)
    assert worker.activate_local(manifest, directory)["status"] == "available"
    monkeypatch.setattr(module, "_verify_files", no_read)
    monkeypatch.setattr(assets, "_verify_segment", no_read)
    monkeypatch.setattr(assets, "_hash_file", no_read)
    monkeypatch.setattr(module, "_file_sha", no_read)
    counts = []
    original = module._selection_key
    def fingerprint(*args, **kwargs):
        counts.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_selection_key", fingerprint)
    assert worker.activate_local(deepcopy(manifest), directory)["status"] == "available"
    assert len(counts) == 2
    assert scheduler.refreshed == 2
    assert module.get_selected_model(worker.root)["manifest"] == manifest


def test_registration_does_not_reuse_changed_native_manifest(tmp_path, monkeypatch):
    worker, _ = master(tmp_path, monkeypatch)
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    assert worker.activate_local(manifest, directory)["status"] == "available"
    native = directory / "segment.manifest.json"
    before = native.stat()
    native.write_bytes(native.read_bytes().replace(b'"n_layer": 4', b'"n_layer": 9'))
    os.utime(native, ns=(before.st_atime_ns, before.st_mtime_ns))
    result = worker.activate_local(manifest, directory)
    assert result["reason_code"] == "artifact_sha256_mismatch"


def test_local_cut_reuses_inventory_source_checks(setup, tmp_path, monkeypatch):
    worker, _, client = setup
    source = gguf(tmp_path / "source.gguf", 4)
    directory = worker.root / "segment-built"
    manifest = segment(directory, source_sha=assets.sha256_file(source))
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter([("source-one", source)]))
    worker.inventory()
    monkeypatch.setattr(assets, "reuse_segment", lambda *args, **kwargs: None)
    original = module._file_sha
    def no_source_read(path, *args, **kwargs):
        assert path != source
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module, "_file_sha", no_source_read)
    monkeypatch.setattr(assets, "build_gguf_segment", lambda *args, **kwargs: {
        **manifest, "local_files": {entry["path"]: str(directory / entry["path"])
                                    for entry in manifest["files"]}})
    worker.handle(command(manifest, strategy="local_cut"))
    assert client.wait()["status"] == "available"
    finish(worker)
