"""票 20 完成定义第三条：`n_ctx` 由「每段 KV 预算」推导，而不是全局常量。

只验证**接线与回退**：预算推导不可用时必须逐字回到既有默认值。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import inference_switches
import llama_engine


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for name in (inference_switches.ENV_KV_CACHE, inference_switches.ENV_FLASH_ATTN,
                 inference_switches.ENV_THREADS):
        monkeypatch.delenv(name, raising=False)


def test_smaller_budget_narrows_the_default(monkeypatch):
    """预算算出的槽位更小时采用它（这是本机制存在的意义）。"""
    from config import MAX_SEQ_LEN

    baseline = MAX_SEQ_LEN if MAX_SEQ_LEN > 0 else 4096
    narrowed = baseline - 1024 if baseline > 1024 else 512
    monkeypatch.setattr(inference_switches, "suggest_n_ctx_from_gguf",
                        lambda model_path, **kwargs: narrowed)
    assert llama_engine._default_n_ctx("dummy.gguf") == narrowed


def test_unavailable_budget_keeps_the_previous_default(monkeypatch):
    """推导不可用（None）⇒ 回到既有默认，不改变行为。"""
    from config import MAX_SEQ_LEN

    baseline = MAX_SEQ_LEN if MAX_SEQ_LEN > 0 else 4096
    monkeypatch.setattr(inference_switches, "suggest_n_ctx_from_gguf",
                        lambda model_path, **kwargs: None)
    assert llama_engine._default_n_ctx("dummy.gguf") == baseline


def test_exception_in_budget_keeps_the_previous_default(monkeypatch):
    """推导抛异常也必须回退，不能把加载路径一起打断。"""

    def _boom(model_path, **kwargs):
        raise RuntimeError("gguf header unreadable")

    monkeypatch.setattr(inference_switches, "suggest_n_ctx_from_gguf", _boom)
    from config import MAX_SEQ_LEN

    assert llama_engine._default_n_ctx("dummy.gguf") == (MAX_SEQ_LEN if MAX_SEQ_LEN > 0 else 4096)


def test_never_widens_beyond_the_baseline(monkeypatch):
    """即使预算算得很大也不放大：调用点传的 max_ctx 是既有默认值。"""
    seen = {}

    def _capture(model_path, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(inference_switches, "suggest_n_ctx_from_gguf", _capture)
    llama_engine._default_n_ctx("dummy.gguf")
    from config import MAX_SEQ_LEN

    assert seen["max_ctx"] == (MAX_SEQ_LEN if MAX_SEQ_LEN > 0 else 4096)
