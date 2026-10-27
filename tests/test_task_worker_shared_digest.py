"""Worker registration, hello and prepare share version-fenced artifact digests."""
import builtins
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import config
import model_supply_assets as assets
import model_supply_digest_cache as digests
import model_supply_worker as supply
import scheduler_task_worker as task_worker
from task_worker_capabilities import build_engine_capabilities
from tests.test_model_supply_worker import Scheduler as SupplyScheduler, segment
from worker_prepare import resolve_prepare_selection


@pytest.fixture(autouse=True)
def isolated_digest_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(digests, "_STORES", {})


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "middle.gguf"
    path.write_bytes(b"original payload")
    return path


def track_payload_reads(monkeypatch, artifact, *, payload_offset=0):
    calls = []
    original = io.open

    class CountedReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            position = self.handle.tell()
            data = self.handle.read(size)
            if data and position + len(data) > payload_offset:
                calls.append((position, len(data)))
            return data

    def counted_open(file, mode="r", *args, **kwargs):
        handle = original(file, mode, *args, **kwargs)
        if (isinstance(file, (str, bytes, os.PathLike)) and Path(file).resolve() == artifact
                and ("r" in mode or "+" in mode)):
            return CountedReader(handle)
        return handle

    monkeypatch.setattr(builtins, "open", counted_open)
    monkeypatch.setattr(io, "open", counted_open)
    return calls


@pytest.mark.parametrize("disk_reload", [False, True])
def test_registration_warms_hello_prepare_and_recheck_without_payload_reads(
    tmp_path, monkeypatch, disk_reload,
):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    monkeypatch.setattr(supply, "_SELECTION_CACHE", {})
    monkeypatch.setenv("QLH_LAYER_GGUF", "")
    owner = SupplyScheduler("master")
    worker = supply.ModelSupplyWorker(owner, tmp_path / "supply" / "received")
    owner.worker = worker
    directory = worker.root.parent / "built"
    manifest = segment(directory, node="master")
    assert worker.activate_local(manifest, directory)["status"] == "available"
    if disk_reload:
        digests._STORES.clear()
    artifact = directory / "segment.gguf"
    _, tensors = assets._gguf_table(artifact)
    reads = track_payload_reads(
        monkeypatch, artifact, payload_offset=min(entry[0] for entry in tensors.values()),
    )
    scheduler = task_worker.SchedulerTaskWorkerMixin()
    scheduler._model_supply_worker = worker
    scheduler._host = SimpleNamespace(model_loaded=False, is_loaded=False)
    scheduler._active_layer_config = {}
    scheduler._model_control = SimpleNamespace(boot_id="fixture-boot")
    scheduler._task_worker_engine_schema = lambda: build_engine_capabilities([
        ("llama_cpp", task_worker.LLAMA_CPP_KEEP_HEAD),
    ])
    scheduler._runtime_profile_for_capabilities = lambda: task_worker.RUNTIME_PROFILE_UNSPECIFIED
    scheduler._task_worker_multi_position_argmax_available = lambda: False
    capabilities = scheduler._task_worker_capabilities()
    assert capabilities["layer_ranges"] == [[1, 3]]
    assert capabilities["layer_artifacts"] == [{
        "layer_range": [1, 3], "segment_mode": "middle", "model_id": "segment.gguf",
        "artifact_sha256": manifest["artifact_sha256"],
        "source_model_sha256": manifest["source_model_sha256"],
        "layer_weight_bytes": [8, 8], "source_model_id": manifest["source_model_id"],
        "hidden_size": 2, "tokenizer_sha256": manifest["tokenizer_sha256"],
    }]
    identity = {**manifest, "stage_model_identity": capabilities["models"][0]}
    prepared = resolve_prepare_selection(scheduler, identity)
    assert prepared["manifest"] == manifest
    assert resolve_prepare_selection(scheduler, identity) == prepared
    assert reads == []


@pytest.mark.parametrize("first_owner", ["worker", "registration"])
def test_worker_and_registration_read_payload_only_once(artifact, monkeypatch, first_owner):
    reads = track_payload_reads(monkeypatch, artifact)
    owners = {
        "worker": task_worker._verified_layer_artifact_sha256,
        "registration": assets._hash_file_cached,
    }
    expected = hashlib.sha256(b"original payload").hexdigest()
    assert owners[first_owner](artifact) == expected
    for _ in range(2):
        assert task_worker._verified_layer_artifact_sha256(artifact) == expected
        assert assets._hash_file_cached(artifact) == expected
    assert len(reads) == 1


def test_worker_digest_survives_fresh_process_without_payload_read(artifact):
    expected = task_worker._verified_layer_artifact_sha256(artifact)
    assert expected == hashlib.sha256(b"original payload").hexdigest()
    code = """
import builtins
import io
import os
from pathlib import Path
import sys
import config
config.STATE_DIR = sys.argv[1]
from scheduler_task_worker import _verified_layer_artifact_sha256
from model_supply_assets import _hash_file_cached
artifact = Path(sys.argv[2]).resolve()
original = io.open
def guarded_open(file, mode="r", *args, **kwargs):
    if (isinstance(file, (str, bytes, os.PathLike)) and Path(file).resolve() == artifact
            and ("r" in mode or "+" in mode)):
        raise AssertionError("fresh worker process reread cached artifact")
    return original(file, mode, *args, **kwargs)
builtins.open = guarded_open
io.open = guarded_open
assert _verified_layer_artifact_sha256(artifact) == sys.argv[3]
assert _hash_file_cached(artifact) == sys.argv[3]
assert _verified_layer_artifact_sha256(artifact) == sys.argv[3]
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(task_worker.__file__).parent))
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(config.STATE_DIR), str(artifact), expected],
        env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("disk_reload", [False, True])
@pytest.mark.parametrize("change", ["rewrite", "replace"])
def test_worker_digest_invalidates_same_size_restored_mtime(
    artifact, monkeypatch, disk_reload, change,
):
    reads = track_payload_reads(monkeypatch, artifact)
    previous = task_worker._verified_layer_artifact_sha256(artifact)
    before = artifact.stat()
    stamp = digests.file_stamp(artifact)
    changed = artifact.with_suffix(".new") if change == "replace" else artifact
    changed.write_bytes(b"modified payload")
    os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))
    if change == "replace":
        os.replace(changed, artifact)
    assert artifact.stat().st_size == before.st_size
    assert artifact.stat().st_mtime_ns == before.st_mtime_ns
    assert digests.file_stamp(artifact) != stamp
    if disk_reload:
        digests._STORES.clear()
    expected = hashlib.sha256(b"modified payload").hexdigest()
    assert expected != previous
    assert task_worker._verified_layer_artifact_sha256(artifact) == expected
    assert assets._hash_file_cached(artifact) == expected
    assert task_worker._verified_layer_artifact_sha256(artifact) == expected
    assert len(reads) == 2


def test_worker_digest_observes_change_time_only(artifact, monkeypatch):
    stamp = list(digests.file_stamp(artifact))
    monkeypatch.setattr(digests, "file_stamp", lambda path: tuple(stamp))
    reads = track_payload_reads(monkeypatch, artifact)
    assert task_worker._verified_layer_artifact_sha256(artifact)
    artifact.write_bytes(b"modified payload")
    stamp[-1] += 1
    expected = hashlib.sha256(b"modified payload").hexdigest()
    assert task_worker._verified_layer_artifact_sha256(artifact) == expected
    assert len(reads) == 2


@pytest.mark.parametrize("unavailable", ["missing", "directory"])
def test_worker_digest_returns_empty_for_unavailable_file(artifact, unavailable):
    assert task_worker._verified_layer_artifact_sha256(artifact)
    artifact.unlink()
    if unavailable == "directory":
        artifact.mkdir()
    assert task_worker._verified_layer_artifact_sha256(artifact) == ""


@pytest.mark.parametrize("error", [
    OSError("unavailable"), digests.DigestCacheError("file_digest_mismatch"),
    digests.DigestCacheError("cancelled"),
])
def test_worker_digest_cache_failure_returns_empty(artifact, monkeypatch, error):
    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(digests, "cached_sha256", failed)
    assert task_worker._verified_layer_artifact_sha256(artifact) == ""


def test_worker_digest_rejects_change_during_hash(artifact, monkeypatch):
    original = digests._hash_file

    def changing_hash(path, cancelled=None):
        digest = original(path, cancelled)
        # ★ 2026-10-27：**必须先跨出当前时间戳 tick 再写。**
        #
        #   这个 fixture 的文件只有 16 字节（`b"original payload"`，改后
        #   `b"modified payload"` 也是 16 字节 ⇒ **size 完全相同**），所以"文件
        #   变了"只能由时间戳体现。而读 16 字节几乎瞬时，写紧随其后 ⇒ 很容易落在
        #   **同一个时间戳 tick** 里，mtime 与 ChangeTime 都不推进 ⇒ `file_stamp`
        #   完全相等 ⇒ 被测的"hash 期间发生了变更"永远检测不到。
        #
        #   症状是随机失败（Heisenbug）：加一行 print 就过，因为打印把两次系统调用
        #   推到了不同 tick。
        time.sleep(0.005)
        path.write_bytes(b"modified payload")
        return digest

    monkeypatch.setattr(digests, "_hash_file", changing_hash)
    assert task_worker._verified_layer_artifact_sha256(artifact) == ""
    assert str(artifact) not in digests._entries(digests._store_path())


def test_worker_digest_rechecks_live_version_on_cache_hit(artifact, monkeypatch):
    assert task_worker._verified_layer_artifact_sha256(artifact)
    stamp = digests.file_stamp(artifact)
    reads = track_payload_reads(monkeypatch, artifact)
    versions = iter([stamp, (*stamp[:-1], stamp[-1] + 1)])
    monkeypatch.setattr(digests, "file_stamp", lambda path: next(versions))
    assert task_worker._verified_layer_artifact_sha256(artifact) == ""
    assert reads == []


def test_worker_warm_digest_keeps_shared_cancellation_contract(artifact, monkeypatch):
    expected = task_worker._verified_layer_artifact_sha256(artifact)
    reads = track_payload_reads(monkeypatch, artifact)
    with pytest.raises(assets.SupplyError) as failure:
        assets._hash_file_cached(artifact, lambda: True)
    assert failure.value.code == "cancelled"
    assert task_worker._verified_layer_artifact_sha256(artifact) == expected
    assert reads == []


def test_worker_digest_does_not_serialize_unrelated_payload_reads(artifact, tmp_path, monkeypatch):
    other = tmp_path / "other.gguf"
    other.write_bytes(b"another payload")
    entered, release = threading.Event(), threading.Event()
    original = digests._hash_file

    def blocked_hash(path, cancelled=None):
        if path == artifact:
            entered.set()
            assert release.wait(10)
        return original(path, cancelled)

    monkeypatch.setattr(digests, "_hash_file", blocked_hash)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(task_worker._verified_layer_artifact_sha256, artifact)
        try:
            assert entered.wait(5)
            second = executor.submit(task_worker._verified_layer_artifact_sha256, other)
            assert second.result(timeout=5) == hashlib.sha256(b"another payload").hexdigest()
        finally:
            release.set()
        assert first.result(timeout=5) == hashlib.sha256(b"original payload").hexdigest()
