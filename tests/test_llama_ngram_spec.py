"""n-gram 自投机在「绑定路径」上的注入契约（票 15c 的落地部分）。

依据实测（2026-10-13）：
* `draft_model=LlamaPromptLookupDecoding` 在绑定路径上有效（重复文本 1.30-1.40×，
  自由文本不劣化）；
* `llama_engine.py` 的 `load_kwargs.update(kwargs)` 无白名单过滤 ⇒ 可注入；
* `.venv-test` 的绑定是 **CPU-only** ⇒ 这正是裁决里「无 CUDA 边缘设备走绑定」的路径。

副作用必须记录：`llama_cpp/llama.py` 一旦收到 `draft_model` 会把 `logits_all`
强制为 True（内存/显存开销上升），因此**默认关闭**。
"""

import pytest

from llama_engine import _apply_ngram_draft_model


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr("config.NGRAM_SPEC_ENABLED", False, raising=True)
    kwargs: dict = {}
    _apply_ngram_draft_model(kwargs)
    assert "draft_model" not in kwargs


def test_injects_when_enabled_with_configured_params(monkeypatch):
    monkeypatch.setattr("config.NGRAM_SPEC_ENABLED", True, raising=True)
    monkeypatch.setattr("config.NGRAM_SPEC_MAX_NGRAM", 3, raising=True)
    monkeypatch.setattr("config.NGRAM_SPEC_NUM_PRED", 12, raising=True)
    kwargs: dict = {}
    _apply_ngram_draft_model(kwargs)
    draft = kwargs.get("draft_model")
    assert draft is not None, "启用时必须注入 draft_model"
    assert draft.max_ngram_size == 3
    assert draft.num_pred_tokens == 12


def test_caller_supplied_draft_model_is_not_overwritten(monkeypatch):
    monkeypatch.setattr("config.NGRAM_SPEC_ENABLED", True, raising=True)
    sentinel = object()
    kwargs = {"draft_model": sentinel}
    _apply_ngram_draft_model(kwargs)
    assert kwargs["draft_model"] is sentinel


def test_missing_binding_class_does_not_raise(monkeypatch):
    """旧绑定可能没有 LlamaPromptLookupDecoding —— 必须降级而不是崩。"""
    monkeypatch.setattr("config.NGRAM_SPEC_ENABLED", True, raising=True)
    import llama_cpp.llama_speculative as spec_mod

    monkeypatch.delattr(spec_mod, "LlamaPromptLookupDecoding", raising=False)
    kwargs: dict = {}
    _apply_ngram_draft_model(kwargs)
    assert "draft_model" not in kwargs


def test_disabled_does_not_touch_existing_kwargs(monkeypatch):
    monkeypatch.setattr("config.NGRAM_SPEC_ENABLED", False, raising=True)
    kwargs = {"n_ctx": 4096, "draft_model": "kept"}
    _apply_ngram_draft_model(kwargs)
    assert kwargs == {"n_ctx": 4096, "draft_model": "kept"}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
