"""单元测试 — PyTorch 侧 flash-attn 开关与设备映射同源（票 20 `DEVICE-AWARE-INFER-01`）。

只验证开关解析与默认值，不加载模型：默认必须等于本行接线前的硬编码 `False`，
否则就是"有开关却隐式打开"。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import inference_switches
import model_module


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        inference_switches.ENV_FLASH_ATTN,
        inference_switches.ENV_KV_CACHE,
        inference_switches.ENV_THREADS,
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_matches_the_previous_hardcoded_value():
    assert model_module._resolve_pytorch_flash_attn() is False


def test_explicit_override_opens_it(monkeypatch):
    monkeypatch.setenv(inference_switches.ENV_FLASH_ATTN, "1")
    assert model_module._resolve_pytorch_flash_attn() is True


def test_explicit_disable_keeps_it_off(monkeypatch):
    monkeypatch.setenv(inference_switches.ENV_FLASH_ATTN, "0")
    assert model_module._resolve_pytorch_flash_attn() is False


def test_same_source_as_the_llamacpp_paths(monkeypatch):
    """同一个覆盖在 PyTorch 侧与 llama.cpp 侧必须给出一致的值。"""
    monkeypatch.setenv(inference_switches.ENV_FLASH_ATTN, "1")
    assert inference_switches.resolve_from_device_profile().flash_attn is True
    assert model_module._resolve_pytorch_flash_attn() is True


def test_mapping_failure_falls_back_to_off(monkeypatch):
    """映射不可用时保持关闭并降级，不把加载路径一起打断。"""

    def _boom():
        raise RuntimeError("mapping unavailable")

    monkeypatch.setattr(inference_switches, "resolve_from_device_profile", _boom)
    assert model_module._resolve_pytorch_flash_attn() is False
