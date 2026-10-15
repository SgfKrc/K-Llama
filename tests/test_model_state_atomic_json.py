"""Real Windows sharing locks and ordered supply/intent state publication."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import errno
import json
import os
from pathlib import Path
import stat
import sys
import threading
from types import SimpleNamespace

import pytest

import atomic_json
from atomic_json import atomic_write_json
from model_intent import ModelIntentCoordinator
from model_supply import ModelSupplyCoordinator, ModelSupplyError


@pytest.fixture
def state(tmp_path):
    scheduler = SimpleNamespace(
        _model_supply_worker=SimpleNamespace(cancel=lambda operation_id: None),
        _tcp_server=None, _layer_config_lock=threading.RLock(),
        _pipeline_load_transaction=None, _active_pipeline_capacity_plan=None,
        get_distributed_inference_enabled=lambda: True,
    )
    supply = ModelSupplyCoordinator(scheduler, tmp_path)
    scheduler._model_supply = supply
    intent = ModelIntentCoordinator(scheduler, lambda operation_id: None, root=tmp_path)
    operation_id = "supply_dbac01f9fd054f86a72df7ca724517df"
    supply._active = operation_id
    supply._operations[operation_id] = {
        "operation_id": operation_id, "intent": {}, "status": "supplying", "nodes": {},
    }
    supply._events[operation_id] = threading.Event()
    supply._leases[(operation_id, "worker")] = {"token": "secret", "files": {}}
    intent._current = operation_id
    intent._records[operation_id] = {"operation_id": operation_id, "status": "queued"}
    supply._persist(supply._operations[operation_id])
    intent._persist(intent._records[operation_id])
    return SimpleNamespace(root=tmp_path, supply=supply, intent=intent, operation_id=operation_id)


def _publication(state, filename):
    if filename == "operation.json":
        return (state.root / state.operation_id / filename,
                lambda: state.supply._update(state.operation_id, status="building"))
    path = state.root / state.operation_id / filename if filename == "intent.json" else state.root / filename
    return path, lambda: state.intent._update(state.operation_id, status="preparing")


@contextmanager
def _windows_read_handle_without_delete_share(path):
    """Use the actual kernel sharing contract, with a thread-safe early release."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE (no FILE_SHARE_DELETE),
    # OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL.
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    guard = threading.Lock()

    def release():
        nonlocal handle
        with guard:
            if handle is not None:
                if not kernel.CloseHandle(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                handle = None

    try:
        yield release
    finally:
        release()


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows kernel sharing locks")
@pytest.mark.parametrize("filename", ["operation.json", "intent.json", "current-intent.json"])
def test_windows_reader_release_allows_state_replace(state, monkeypatch, filename):
    path, publish = _publication(state, filename)
    previous = path.read_bytes()
    blocked = threading.Event()
    errors = []
    replace = os.replace

    def observed_replace(source, target):
        try:
            return replace(source, target)
        except OSError as exc:
            if Path(target) == path:
                errors.append(exc.winerror)
                assert path.read_bytes() == previous
                blocked.set()
            raise

    monkeypatch.setattr(atomic_json.os, "replace", observed_replace)
    with _windows_read_handle_without_delete_share(path) as release:
        def release_from_another_thread():
            assert blocked.wait(5), "no real sharing failure was observed"
            threading.Event().wait(0.04)
            release()

        with ThreadPoolExecutor(max_workers=1) as pool:
            released = pool.submit(release_from_another_thread)
            try:
                publish()
            finally:
                released.result(timeout=5)
    assert errors and set(errors) <= {5, 32, 33}
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["operation_id"] == state.operation_id
    if filename != "current-intent.json":
        assert saved["status"] == ("building" if filename == "operation.json" else "preparing")
    assert not list(state.root.rglob("*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows kernel sharing locks")
@pytest.mark.parametrize("filename", ["operation.json", "intent.json", "current-intent.json"])
def test_windows_unreleased_reader_fails_with_bounded_retries(state, monkeypatch, filename):
    path, publish = _publication(state, filename)
    previous = path.read_bytes()
    sleeps, attempts = [], []
    replace = os.replace

    def observed_replace(source, target):
        if Path(target) == path:
            attempts.append(Path(source))
        return replace(source, target)

    monkeypatch.setattr(atomic_json.os, "replace", observed_replace)
    monkeypatch.setattr(atomic_json, "time", SimpleNamespace(sleep=sleeps.append))
    with _windows_read_handle_without_delete_share(path):
        with pytest.raises((OSError, ModelSupplyError)) as failed:
            publish()
    cause = failed.value.__cause__ if isinstance(failed.value, ModelSupplyError) else failed.value
    assert isinstance(cause, PermissionError) and cause.winerror in {5, 32, 33}
    assert len(attempts) == 8 and len(set(attempts)) == 1
    assert sleeps == [0.01 * 2 ** attempt for attempt in range(7)]
    assert path.read_bytes() == previous
    assert not list(state.root.rglob("*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows read-only permission semantics")
def test_windows_permanent_readonly_denial_is_not_swallowed(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")
    previous = path.read_bytes()
    sleeps = []
    monkeypatch.setattr(atomic_json, "time", SimpleNamespace(sleep=sleeps.append))
    path.chmod(stat.S_IREAD)
    try:
        with pytest.raises(PermissionError) as failed:
            atomic_write_json(path, {"new": True})
        assert failed.value.winerror == 5
        assert len(sleeps) == 7
        assert path.read_bytes() == previous
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        path.chmod(stat.S_IWRITE)


@pytest.mark.parametrize("winerror", [5, 32, 33])
@pytest.mark.parametrize("permanent", [False, True])
def test_only_windows_sharing_errors_are_retried_to_a_fixed_bound(tmp_path, monkeypatch, winerror, permanent):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")
    previous = path.read_bytes()
    error = PermissionError(errno.EACCES, "injected sharing/access error")
    error.winerror = winerror
    attempts, sleeps = [], []
    replace = os.replace

    def fail_then_replace(source, target):
        attempts.append(Path(source))
        assert Path(source).read_text(encoding="utf-8") == '{"new": true}'
        assert path.read_bytes() == previous
        if permanent or len(attempts) < 3:
            raise error
        return replace(source, target)

    monkeypatch.setattr(atomic_json, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(atomic_json, "time", SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(atomic_json.os, "replace", fail_then_replace)
    if permanent:
        with pytest.raises(PermissionError) as failed:
            atomic_write_json(path, {"new": True})
        assert failed.value is error
        assert len(attempts) == 8 and path.read_bytes() == previous
    else:
        atomic_write_json(path, {"new": True})
        assert len(attempts) == 3 and json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert sleeps == [0.01 * 2 ** attempt for attempt in range(len(attempts) - 1)]
    assert sum(sleeps) <= 1.27
    assert len(set(attempts)) == 1 and not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("platform,winerror", [("linux", 5), ("win32", None), ("win32", 2), ("win32", 112)])
def test_other_errors_are_raised_without_retry(tmp_path, monkeypatch, platform, winerror):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")
    previous = path.read_bytes()
    error = OSError(errno.EIO, "injected non-transient error")
    if winerror is not None:
        error.winerror = winerror
    attempts, sleeps = [], []

    def fail(source, target):
        attempts.append(source)
        raise error

    monkeypatch.setattr(atomic_json, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(atomic_json, "time", SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(atomic_json.os, "replace", fail)
    with pytest.raises(OSError) as failed:
        atomic_write_json(path, {"new": True})
    assert failed.value is error and len(attempts) == 1 and not sleeps
    assert path.read_bytes() == previous and not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt(), OSError(errno.ENOSPC, "disk full")])
def test_prepublication_failure_and_cleanup_preserve_original_error(tmp_path, monkeypatch, error):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")
    previous = path.read_bytes()
    temporary = []
    unlink = Path.unlink

    def fail_sync(fd):
        raise error

    def fail_cleanup(self, *args, **kwargs):
        temporary.append(self)
        raise PermissionError(errno.EACCES, "cleanup also denied")

    monkeypatch.setattr(atomic_json.os, "fsync", fail_sync)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    try:
        with pytest.raises(type(error)) as failed:
            atomic_write_json(path, {"new": True})
        assert failed.value is error and path.read_bytes() == previous
        assert len(temporary) == 1
    finally:
        for candidate in temporary:
            unlink(candidate, missing_ok=True)


def test_serialization_failure_cleans_temp_and_leaves_previous_json(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")
    with pytest.raises(TypeError):
        atomic_write_json(path, {"not_json": object()})
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_overlapping_writers_use_distinct_complete_sibling_files(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")
    barrier = threading.Barrier(2)
    staged = {}
    replace = os.replace

    def overlapping_replace(source, target):
        source = Path(source)
        assert source.parent == path.parent and source.name != "state.json.tmp"
        # Windows can also briefly deny overlapping renames. Synchronize
        # staging once per writer; its retry must not wait on an exited peer.
        if source not in staged:
            staged[source] = json.loads(source.read_text(encoding="utf-8"))
            assert b"\xe4\xb8\xad\xe6\x96\x87" in source.read_bytes()
            barrier.wait(timeout=5)
        assert len(staged) == 2
        return replace(source, target)

    monkeypatch.setattr(atomic_json.os, "replace", overlapping_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(atomic_write_json, path, {"writer": "first", "text": "\u4e2d\u6587"}, ensure_ascii=False)
        second = pool.submit(atomic_write_json, path, {"writer": "second", "text": "\u4e2d\u6587"}, ensure_ascii=False)
        first.result(timeout=5)
        second.result(timeout=5)
    assert {value["writer"] for value in staged.values()} == {"first", "second"}
    assert json.loads(path.read_text(encoding="utf-8")) in staged.values()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("filename", ["operation.json", "intent.json"])
def test_retry_keeps_owner_lock_and_cancel_cannot_be_overwritten(state, monkeypatch, filename):
    path, publish = _publication(state, filename)
    owner = state.supply if filename == "operation.json" else state.intent
    lock = owner._condition if filename == "operation.json" else owner._lock
    retrying, resume, cancelling = threading.Event(), threading.Event(), threading.Event()
    attempts = []
    error = PermissionError(errno.EACCES, "injected Windows read lock")
    error.winerror = 5
    replace = os.replace

    def fail_first_replace(source, target):
        if Path(target) == path:
            assert lock._is_owned(), "owner lock was released before replacement"
            attempts.append(json.loads(Path(source).read_text(encoding="utf-8"))["status"])
            if len(attempts) == 1:
                raise error
        return replace(source, target)

    def pause_retry(delay):
        assert lock._is_owned(), "backoff released the state owner lock"
        retrying.set()
        assert resume.wait(5)

    def cancel():
        cancelling.set()
        return owner.cancel(state.operation_id)

    monkeypatch.setattr(atomic_json, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(atomic_json, "time", SimpleNamespace(sleep=pause_retry))
    monkeypatch.setattr(atomic_json.os, "replace", fail_first_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        older = pool.submit(publish)
        try:
            assert retrying.wait(5)
            acquired = lock.acquire(blocking=False)
            if acquired:
                lock.release()
            assert not acquired, "another writer could enter while the older state was retrying"
            newer = pool.submit(cancel)
            assert cancelling.wait(5) and not newer.done()
        finally:
            resume.set()
        older.result(timeout=5)
        assert newer.result(timeout=5)["status"] == "cancelled"
    assert attempts == [attempts[0], attempts[0], "cancelled"]
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "cancelled"
    assert state.supply._events[state.operation_id].is_set() and not state.supply._leases
    owner._update(state.operation_id, status="preparing")
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "cancelled"
    assert not list(state.root.rglob("*.tmp"))


def test_current_pointer_retry_cannot_overwrite_new_selection(state, monkeypatch):
    path = state.root / "current-intent.json"
    newer_id = "supply_" + "b" * 32
    retrying, resume, newer_started = threading.Event(), threading.Event(), threading.Event()
    attempts = []
    error = PermissionError(errno.EACCES, "injected Windows read lock")
    error.winerror = 5
    replace = os.replace

    def fail_first_replace(source, target):
        if Path(target) == path:
            assert state.intent._lock._is_owned()
            attempts.append(json.loads(Path(source).read_text(encoding="utf-8"))["operation_id"])
            if len(attempts) == 1:
                raise error
        return replace(source, target)

    def pause_retry(delay):
        assert state.intent._lock._is_owned()
        retrying.set()
        assert resume.wait(5)

    def select_newer():
        newer_started.set()
        with state.intent._lock:
            state.intent._current = newer_id
            state.intent._records[newer_id] = {"operation_id": newer_id, "status": "queued"}
            state.intent._persist(state.intent._records[newer_id])

    monkeypatch.setattr(atomic_json, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(atomic_json, "time", SimpleNamespace(sleep=pause_retry))
    monkeypatch.setattr(atomic_json.os, "replace", fail_first_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        older = pool.submit(state.intent._persist, state.intent._records[state.operation_id])
        try:
            assert retrying.wait(5)
            newer = pool.submit(select_newer)
            assert newer_started.wait(5) and not newer.done()
        finally:
            resume.set()
        older.result(timeout=5)
        newer.result(timeout=5)
    assert attempts == [state.operation_id, state.operation_id, newer_id]
    assert json.loads(path.read_text(encoding="utf-8")) == {"operation_id": newer_id}
    assert json.loads((state.root / newer_id / "intent.json").read_text(encoding="utf-8"))["operation_id"] == newer_id
    assert not list(state.root.rglob("*.tmp"))


def test_coordinators_preserve_existing_json_format(state):
    record = state.supply._operations[state.operation_id]
    record["reason"] = "\u4e2d\u6587"
    state.supply._persist(record)
    operation = (state.root / state.operation_id / "operation.json").read_text(encoding="utf-8")
    assert operation == json.dumps(record, ensure_ascii=False, indent=2)
    assert b"\xe4\xb8\xad\xe6\x96\x87" in (state.root / state.operation_id / "operation.json").read_bytes()
    record = state.intent._records[state.operation_id]
    record["reason"] = "\u4e2d\u6587"
    state.intent._persist(record)
    assert (state.root / state.operation_id / "intent.json").read_text(encoding="utf-8") == json.dumps(record)
    assert (state.root / "current-intent.json").read_text(encoding="utf-8") == json.dumps({"operation_id": state.operation_id})
