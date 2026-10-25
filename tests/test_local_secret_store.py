from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "src")

from local_secret_store import LocalSecretStore, LocalSecretStoreError


def test_local_secret_store_encrypts_and_authenticates_records(tmp_path):
    path = tmp_path / "node_secrets.json"
    store = LocalSecretStore(path)
    store.set("cluster_secret", "cluster-secret-value")

    assert store.get("cluster_secret") == "cluster-secret-value"
    assert "cluster-secret-value" not in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "qlh.local-secret-store.v1"

    payload = json.loads(path.read_text(encoding="utf-8"))
    ciphertext = payload["records"]["cluster_secret"]["ciphertext"]
    payload["records"]["cluster_secret"]["ciphertext"] = ciphertext[:-1] + (
        "A" if ciphertext[-1] != "A" else "B"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LocalSecretStoreError, match="authentication failed"):
        store.get("cluster_secret")


def test_local_secret_store_delete_is_idempotent(tmp_path):
    store = LocalSecretStore(tmp_path / "node_secrets.json")
    store.set("cluster_secret", "cluster-secret-value")
    store.delete("cluster_secret")
    store.delete("cluster_secret")
    assert store.get("cluster_secret") == ""


def test_local_secret_store_instances_share_one_path_lock(tmp_path):
    path = tmp_path / "node_secrets.json"
    first = LocalSecretStore(path)
    second = LocalSecretStore(path)

    assert first._lock is second._lock


def test_existing_store_without_wrapping_key_fails_closed(tmp_path):
    path = tmp_path / "node_secrets.json"
    store = LocalSecretStore(path)
    store.set("cluster_secret", "cluster-secret-value")
    original = path.read_bytes()
    store.key_path.unlink()

    with pytest.raises(LocalSecretStoreError, match="wrapping key is missing"):
        store.get("cluster_secret")
    with pytest.raises(LocalSecretStoreError, match="wrapping key is missing"):
        store.set("cluster_secret", "replacement-secret")

    assert path.read_bytes() == original


def test_concurrent_instances_preserve_records_without_replacement_retries(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import local_secret_store as module

    path = tmp_path / "shared_secrets.json"
    store = LocalSecretStore(path)
    store.set("anchor", "unchanged")
    replace = module.os.replace
    calls = []
    gate = threading.Barrier(8)

    def record_replace(source, target):
        calls.append((source, target))
        return replace(source, target)

    monkeypatch.setattr(module.os, "replace", record_replace)

    def write_and_read(index):
        own_store = LocalSecretStore(path.parent / "." / path.name)
        gate.wait(timeout=5)
        for revision in range(3):
            own_store.set(f"worker-{index}", f"value-{index}-{revision}")
            assert own_store.get("anchor") == "unchanged"
        return own_store.get(f"worker-{index}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write_and_read, range(8)))
    assert results == [f"value-{index}-2" for index in range(8)]
    assert all(store.get(f"worker-{index}") == value for index, value in enumerate(results))
    assert len(calls) == 24, "one atomic replace per write, without retrying"


@pytest.mark.skipif(sys.platform != "win32", reason="requires a real Windows sharing violation")
def test_windows_deny_delete_handle_preserves_committed_secret_and_does_not_retry(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes
    import local_secret_store as module

    store = LocalSecretStore(tmp_path / "held_secrets.json")
    store.set("pending", "committed")
    before = store.path.read_bytes()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # A valid reader permits READ/WRITE but deliberately omits SHARE_DELETE.
    # The Windows filesystem must reject atomic replacement of this target.
    handle = kernel.CreateFileW(str(store.path), 0x80000000, 0x1 | 0x2,
                                None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    replace = module.os.replace
    attempts = []

    def record_replace(source, target):
        attempts.append((source, target))
        return replace(source, target)

    monkeypatch.setattr(module.os, "replace", record_replace)
    try:
        with pytest.raises(PermissionError) as caught:
            store.delete("pending")
        assert caught.value.winerror == 5
        assert len(attempts) == 1
        assert store.path.read_bytes() == before
        assert store.get("pending") == "committed"
    finally:
        assert kernel.CloseHandle(handle)
    # A distinct operation after its real owner releases the handle succeeds.
    store.delete("pending")
    assert store.get("pending") == ""
    assert len(attempts) == 2
