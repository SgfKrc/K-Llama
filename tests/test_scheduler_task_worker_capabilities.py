"""`_task_worker_capabilities` 的 llama.cpp 判据必须同时认 keep-head shim。

背景（2026-10-11 三机实测）：ROG 上**没有** pip 的 `llama_cpp`，但它配了
`QLH_KEEP_HEAD_SHIM` + 私有 libllama，能正常执行层段 —— 而 HELLO 被协议判为
`model engine was not declared by the worker`。根因是判据只查
`importlib.util.find_spec("llama_cpp")`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from keep_head_shim import default_shim_path  # noqa: E402
from scheduler_task_worker import keep_head_shim_available  # noqa: E402


def test_env_pointing_at_existing_shim_is_available(monkeypatch, tmp_path: Path):
    shim = tmp_path / "qlh_keep_head.dll"
    shim.write_bytes(b"stub")
    monkeypatch.setenv("QLH_KEEP_HEAD_SHIM", str(shim))
    assert keep_head_shim_available() is True


def test_env_pointing_at_missing_shim_is_not_available(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("QLH_KEEP_HEAD_SHIM", str(tmp_path / "nope.dll"))
    assert keep_head_shim_available() is False


def test_falls_back_to_repo_default_when_env_absent(monkeypatch):
    monkeypatch.delenv("QLH_KEEP_HEAD_SHIM", raising=False)
    # 与 keep_head_shim.resolve_keep_head_shim 同一事实源：仓库默认产物在则可用。
    assert keep_head_shim_available() is default_shim_path().is_file()


def test_never_raises(monkeypatch):
    """判据失败必须降级为 False，绝不抛出（否则 HELLO 会整条失败）。"""
    monkeypatch.setenv("QLH_KEEP_HEAD_SHIM", "")
    assert isinstance(keep_head_shim_available(), bool)
