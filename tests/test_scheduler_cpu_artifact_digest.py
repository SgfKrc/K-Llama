"""Local CPU planning and execution share one versioned artifact digest store."""
import builtins
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

import config
import model_supply_digest_cache as digests
import scheduler


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    path = tmp_path / "middle.gguf"
    path.write_bytes(b"cpu middle 8-24")
    monkeypatch.setenv("QLH_LAYER_GGUF", str(path))
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(digests, "_STORES", {})
    return path


@pytest.fixture
def payload_reads(artifact, monkeypatch):
    calls = []
    original = builtins.open

    def counted_open(file, mode="r", *args, **kwargs):
        if (isinstance(file, (str, bytes, os.PathLike)) and Path(file).resolve() == artifact
                and ("r" in mode or "+" in mode)):
            calls.append(str(file))
        return original(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counted_open)
    monkeypatch.setattr(io, "open", counted_open)
    return calls


@pytest.mark.parametrize("first_owner", ["scheduler", "shared"])
def test_cpu_and_shared_digest_read_payload_only_once(artifact, payload_reads, first_owner):
    expected = hashlib.sha256(b"cpu middle 8-24").hexdigest()
    owners = {
        "scheduler": scheduler._local_cpu_artifact_sha256,
        "shared": lambda: digests.cached_sha256(artifact),
    }
    assert owners[first_owner]() == expected
    for _ in range(2):
        assert scheduler._local_cpu_artifact_sha256() == expected
        assert digests.cached_sha256(artifact) == expected
    assert len(payload_reads) == 1


def test_cpu_digest_survives_fresh_process_without_payload_read(artifact):
    expected = scheduler._local_cpu_artifact_sha256()
    assert expected == hashlib.sha256(b"cpu middle 8-24").hexdigest()
    assert digests._store_path().is_file()
    code = """
import builtins
import io
import os
from pathlib import Path
import sys
import config
config.STATE_DIR = sys.argv[1]
import scheduler
import model_supply_digest_cache as digests
artifact = Path(sys.argv[2]).resolve()
original = builtins.open
def guarded_open(file, mode="r", *args, **kwargs):
    if (isinstance(file, (str, bytes, os.PathLike)) and Path(file).resolve() == artifact
            and ("r" in mode or "+" in mode)):
        raise AssertionError("fresh process reread cached CPU artifact")
    return original(file, mode, *args, **kwargs)
builtins.open = guarded_open
io.open = guarded_open
assert scheduler._local_cpu_artifact_sha256() == sys.argv[3]
assert scheduler._local_cpu_artifact_sha256() == sys.argv[3]
assert digests.cached_sha256(artifact) == sys.argv[3]
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(scheduler.__file__).parent))
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(config.STATE_DIR), str(artifact), expected],
        env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("disk_reload", [False, True])
@pytest.mark.parametrize("change", ["rewrite", "replace"])
def test_cpu_digest_invalidates_same_size_restored_mtime(
    artifact, payload_reads, disk_reload, change,
):
    previous = scheduler._local_cpu_artifact_sha256()
    before = artifact.stat()
    stamp = digests.file_stamp(artifact)
    if change == "replace":
        replacement = artifact.with_suffix(".new")
        replacement.write_bytes(b"bad middle 8-24")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, artifact)
    else:
        artifact.write_bytes(b"bad middle 8-24")
        os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert artifact.stat().st_size == before.st_size
    assert artifact.stat().st_mtime_ns == before.st_mtime_ns
    assert digests.file_stamp(artifact) != stamp
    if disk_reload:
        digests._STORES.clear()
    expected = hashlib.sha256(b"bad middle 8-24").hexdigest()
    assert expected != previous
    assert scheduler._local_cpu_artifact_sha256() == expected
    assert digests.cached_sha256(artifact) == expected
    assert scheduler._local_cpu_artifact_sha256() == expected
    assert len(payload_reads) == 2


@pytest.mark.parametrize("missing", ["unset", "deleted"])
def test_cpu_digest_returns_empty_for_unavailable_artifact(artifact, monkeypatch, missing):
    assert scheduler._local_cpu_artifact_sha256()
    if missing == "unset":
        monkeypatch.delenv("QLH_LAYER_GGUF")
    else:
        artifact.unlink()
    assert scheduler._local_cpu_artifact_sha256() == ""


def test_cpu_digest_rejects_change_during_hash(artifact, monkeypatch):
    original = digests._hash_file

    def changing_hash(path, cancelled=None):
        digest = original(path, cancelled)
        path.write_bytes(b"bad middle 8-24")
        return digest

    monkeypatch.setattr(digests, "_hash_file", changing_hash)
    assert scheduler._local_cpu_artifact_sha256() == ""
    assert str(artifact) not in digests._entries(digests._store_path())
