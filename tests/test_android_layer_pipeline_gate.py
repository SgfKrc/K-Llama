"""★ Android 参与层流水线的门禁（2026-09-19，D29 待办 C）。

## 背景
此前有两处「按平台一刀切」的判定，理由都写作「Android 无 **PyTorch** 推理能力」：

* `scheduler.py` 的 `validate_layer_override`（Phase 4.1）—— 拒绝给 Android 分配层段；
* `api_server.py` 的 `first_connect_bootstrap` —— `pipeline_worker = node_type == "pc"`（且 android 段
  硬编码 `False`）。

该判据**过宽**：**Android 不能跑 PyTorch，但能跑 llama.cpp/GGUF 引擎**，而 llama.cpp 现在
**也能做层前向**（`forward_layers_from_hidden` 当下游、`forward_layers_to_hidden` 当上游，
且 `BackendId.LLAMA_CPP` 已声明 `Capability.FORWARD_LAYERS`）⇒ 有 GGUF 引擎的 Android
可以参与层流水线。

## 本测试锁定的契约（保守、向后兼容）
判定必须**按能力**，且**未自报能力的 Android 仍被拒绝**。
"""

from __future__ import annotations

import pytest

from api_server import (
    _client_supports_forward_layers,
    _validate_pipeline_role_switches,
)
from koakuma_engine import Capability
from tests.test_scheduler import _engine_contract
from scheduler import NodeInfo, _node_supports_forward_layers


def _node(node_type: str, device_info: dict | None = None) -> NodeInfo:
    return NodeInfo(
        node_id=f"{node_type}-1",
        role="client",
        node_type=node_type,
        device_info=device_info or {},
    )


class TestNodeSupportsForwardLayers:
    """`scheduler._node_supports_forward_layers`。"""

    def test_pc_without_reporting_is_rejected(self):
        """平台名不构成执行能力证据。"""
        assert _node_supports_forward_layers(_node("pc")) is False

    def test_android_without_reporting_is_still_rejected(self):
        """★ 未自报能力的 android ⇒ **仍拒绝**（本改动不得无意放行）。"""
        assert _node_supports_forward_layers(_node("android")) is False

    def test_android_reporting_capability_is_allowed(self):
        """★ 自报 FORWARD_LAYERS 的 Android ⇒ **允许**（本改动的核心目标）。"""
        node = _node("android", {**_engine_contract("llama_cpp", "llama_cpp_android"), "capabilities": [Capability.FORWARD_LAYERS]})
        assert _node_supports_forward_layers(node) is True

    def test_android_reporting_capability_as_string(self):
        node = _node("android", {"capabilities": Capability.FORWARD_LAYERS})
        assert _node_supports_forward_layers(node) is False

    def test_android_reporting_llama_cpp_backend_is_allowed(self):
        """自报 backend_id=llama_cpp 的 Android ⇒ 由公开能力表判定 ⇒ 允许。"""
        node = _node("android", {**_engine_contract("llama_cpp", "llama_cpp_android"), "backend_id": "llama_cpp"})
        assert _node_supports_forward_layers(node) is True

    def test_android_reporting_island_backend_is_rejected(self):
        """自报一个**不**支持层前向的 backend ⇒ 仍然拒绝。"""
        node = _node("android", {"backend_id": "island"})
        assert _node_supports_forward_layers(node) is False

    def test_irrelevant_capability_does_not_pass(self):
        node = _node("android", {"capabilities": [Capability.CHAT]})
        assert _node_supports_forward_layers(node) is False

    def test_malformed_device_info_is_safe(self):
        node = NodeInfo(node_id="android-x", role="client", node_type="android")
        node.device_info = None  # type: ignore[assignment]
        assert _node_supports_forward_layers(node) is False


class TestClientSupportsForwardLayers:
    """`api_server._client_supports_forward_layers`（bootstrap 的 pipeline_worker 判定）。"""

    @pytest.mark.parametrize("node_type", ["pc", None, ""])
    def test_pc_defaults_true(self, node_type):
        """`pc` 与「未给出」都按旧行为放行（注意：**大小写敏感**，与原实现保持一致）。"""
        assert _client_supports_forward_layers(node_type, None) is True

    def test_uppercase_pc_stays_rejected(self):
        """⚠️ 原实现 `node_type == "pc"` 是**大小写敏感**的；本改动不引入规范化，避免行为漂移。"""
        assert _client_supports_forward_layers("PC", None) is False

    def test_android_defaults_false(self):
        assert _client_supports_forward_layers("android", None) is False
        assert _client_supports_forward_layers("android", {}) is False

    def test_android_with_reported_capability(self):
        assert _client_supports_forward_layers(
            "android", {"capabilities": [Capability.FORWARD_LAYERS]}
        ) is True

    def test_android_with_forward_layers_flag(self):
        assert _client_supports_forward_layers("android", {"forward_layers": True}) is True
        assert _client_supports_forward_layers("android", {"forward_layers": False}) is False

    def test_android_with_llama_cpp_backend(self):
        assert _client_supports_forward_layers("android", {"backend_id": "llama_cpp"}) is True
        assert _client_supports_forward_layers("android", {"engine": "llama_cpp"}) is True

    def test_android_with_torch_backend(self):
        """Android 不可能跑 torch，但能力表一致 ⇒ 按 backend 判定也应放行（判据只看能力）。"""
        assert _client_supports_forward_layers("android", {"backend_id": "pytorch"}) is True

    def test_android_with_unsupported_backend(self):
        assert _client_supports_forward_layers("android", {"backend_id": "island"}) is False

    def test_non_dict_capabilities_are_safe(self):
        assert _client_supports_forward_layers("android", "nonsense") is False  # type: ignore[arg-type]
        assert _client_supports_forward_layers("pc", 123) is True  # type: ignore[arg-type]


class TestPipelineRoleSwitches:
    def test_probe_only_relay_does_not_block_route_a(self):
        _validate_pipeline_role_switches(
            relay_enabled=True,
            relay_probe_only=True,
            task_worker_enabled=True,
        )

    def test_active_relay_still_conflicts_with_route_a(self):
        with pytest.raises(RuntimeError, match="不能同时开启"):
            _validate_pipeline_role_switches(
                relay_enabled=True,
                relay_probe_only=False,
                task_worker_enabled=True,
            )
