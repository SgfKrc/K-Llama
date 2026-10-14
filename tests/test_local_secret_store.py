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
