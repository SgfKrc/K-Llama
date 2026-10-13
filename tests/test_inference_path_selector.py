"""整模推理路径选择契约（票 15a `INFER-PATH-SELECT-01`）。

裁决（2026-10-12）：**有 CUDA 走 server 子进程，无 CUDA（边缘）老实走绑定**。

本文件锁住四条判据：
1. `auto` + 有 CUDA ⇒ server；`auto` + 无 CUDA ⇒ binding；
2. **层段接力强制走绑定**（即使有 CUDA）—— 这是守卫，防止"为了统一把层段塞进 server"；
3. 显式请求 server 但**无 CUDA** ⇒ **具名拒绝，绝不静默降级**；
4. 层段接力却显式请求 server ⇒ 同样具名拒绝。

另有一条能力广告契约：路径差异必须能投影成能力声明，且 **server 路径不得宣称
承担层段**（否则容量求解与路由会误判这台机器"能做什么"）。
"""

from __future__ import annotations

import pytest

from inference_path_selector import (
    BINDING,
    SERVER,
    InferencePathError,
    select_inference_path,
)


def test_auto_with_cuda_selects_server():
    decision = select_inference_path(cuda_available=True)
    assert decision.path == SERVER
    assert decision.cuda_available is True
    assert "CUDA" in decision.reason


def test_auto_without_cuda_selects_binding():
    decision = select_inference_path(cuda_available=False)
    assert decision.path == BINDING
    assert decision.cuda_available is False


def test_layer_segment_always_uses_binding_even_with_cuda():
    decision = select_inference_path(cuda_available=True, layer_segment_required=True)
    assert decision.path == BINDING
    assert "层段" in decision.reason


def test_requesting_server_for_layer_segment_is_rejected():
    with pytest.raises(InferencePathError) as excinfo:
        select_inference_path(
            requested=SERVER, cuda_available=True, layer_segment_required=True,
        )
    assert "层段" in str(excinfo.value)


def test_explicit_server_without_cuda_is_rejected_not_downgraded():
    """不许"没 CUDA 就悄悄退回绑定"—— 必须具名失败。"""
    with pytest.raises(InferencePathError) as excinfo:
        select_inference_path(requested=SERVER, cuda_available=False)
    message = str(excinfo.value)
    assert "CUDA" in message


def test_explicit_binding_with_cuda_is_allowed():
    """用户主动选择绑定（例如做数值对照）时必须允许，不强行改走 server。"""
    decision = select_inference_path(requested=BINDING, cuda_available=True)
    assert decision.path == BINDING
    assert decision.requested == BINDING


def test_unknown_requested_path_is_rejected():
    with pytest.raises(InferencePathError):
        select_inference_path(requested="quantum", cuda_available=True)


def test_capability_advertisement_never_claims_layer_segments_for_server():
    caps = select_inference_path(cuda_available=True).to_capability()
    assert caps["inference_path"] == SERVER
    assert caps["serves_layer_segments"] is False, "server 路径不得宣称承担层段"

    caps_binding = select_inference_path(
        cuda_available=False, layer_segment_required=True,
    ).to_capability()
    assert caps_binding["inference_path"] == BINDING
    assert caps_binding["serves_layer_segments"] is True


def test_decision_is_serialisable():
    payload = select_inference_path(cuda_available=False).to_capability()
    assert set(payload) == {
        "inference_path", "cuda_available", "serves_layer_segments", "reason",
    }
    assert all(isinstance(value, (str, bool)) for value in payload.values())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
