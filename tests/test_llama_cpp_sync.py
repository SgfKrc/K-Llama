from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.model_tools import sync_llama_cpp as sync


def _write_lock(tmp_path: Path, patch_path: str, digest: str) -> Path:
    lock = {
        "upstream": {"revision": "a" * 40},
        "patches": [{
            "path": patch_path,
            "target": "conversion/base.py",
            "marker": "QLH patch marker",
            "sha256": digest,
        }],
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def test_contract_rejects_malformed_patch_digest(tmp_path: Path):
    lock = _write_lock(tmp_path, "patch.diff", "g" * 64)

    with pytest.raises(sync.SyncError, match="SHA-256"):
        sync._contract(lock)


def test_contract_requires_a_single_patch(tmp_path: Path):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"upstream": {"revision": "a" * 40}, "patches": []}), encoding="utf-8")

    with pytest.raises(sync.SyncError, match="one patch"):
        sync._contract(lock)


@pytest.mark.skipif(sys.platform != "win32", reason="基准路径用 Windows 盘符（G:/repo），Linux 上会退化成相对路径")
def test_safe_relative_rejects_escape():
    with pytest.raises(sync.SyncError, match="escapes"):
        sync._safe_relative(Path("G:/repo"), "../outside.patch")
