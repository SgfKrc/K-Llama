"""Manifest publication preserves its wire bytes while tolerating Windows read locks."""
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import atomic_json
import model_registry_validation


def test_manifest_read_lock_retry_preserves_previous_document(tmp_path, monkeypatch):
    path = tmp_path / "model.manifest.json"
    previous = b'{"previous": true}\n'
    path.write_bytes(previous)
    manifest = {"z": {"b": 2, "a": 1}, "a": "manifest"}
    expected = (json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n").replace("\n", os.linesep).encode()
    real_replace = atomic_json.os.replace
    attempts = []
    def locked_once(source, target):
        attempts.append(Path(source))
        assert Path(source).read_bytes() == expected
        if len(attempts) == 1:
            assert path.read_bytes() == previous
            error = PermissionError(errno.EACCES, "injected scanner sharing lock")
            error.winerror = 5
            raise error
        assert path.read_bytes() == previous
        return real_replace(source, target)
    monkeypatch.setattr(atomic_json, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(atomic_json.os, "replace", locked_once)
    monkeypatch.setattr(atomic_json, "time", SimpleNamespace(sleep=lambda _: None))
    assert model_registry_validation.write_manifest(tmp_path, manifest) == path
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert path.read_bytes() == expected
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.part"))
