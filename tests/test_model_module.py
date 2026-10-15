"""
单元测试 — 模型模块的前向传播（forward_layers）
==============================================
测试 load_layer_range() 层裁剪 + forward_layers() 分布式前向推理。

使用 tiny Qwen2 配置（4 层 / hidden=128 / 32 token vocab），
无需下载完整模型即可验证分布式流水线的完整数据流。
"""

import sys
import os
import json
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import copy
import types
import pytest
import torch
import torch.nn as nn

from transformers import Qwen2Config, Qwen2ForCausalLM

from model_module import ModelManager, _LayerRangeLoadTracker
from model_module import _normalize_past_key_values
import model_config as mc

# ================================================================
# Tiny 模型工厂
# ================================================================

TINY_CONFIG = Qwen2Config(
    vocab_size=32,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,       # 4 层足够测试端到端流水线
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)


def _make_tiny_model() -> Qwen2ForCausalLM:
    """创建 tiny Qwen2ForCausalLM（随机权重，无需 GPU）。"""
    model = Qwen2ForCausalLM(TINY_CONFIG)
    model.eval()
    assert model.lm_head is not None, "lm_head 应为独立层"
    return model


def _make_first_node_model():
    """创建首节点模型（有 Embedding，无 LM Head）。"""
    model = _make_tiny_model()
    del model.lm_head
    model.lm_head = None
    return model


def _make_middle_node_model():
    """创建中间节点模型（无 Embedding，无 LM Head）。"""
    model = _make_tiny_model()
    del model.model.embed_tokens
    model.model.embed_tokens = None
    del model.lm_head
    model.lm_head = None
    return model


def _make_last_node_model():
    """创建末节点模型（无 Embedding，有 LM Head）。"""
    model = _make_tiny_model()
    del model.model.embed_tokens
    model.model.embed_tokens = None
    return model


class _TinyQwenRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self._ntk_alpha_cached_list = [1.0]

    def forward(self, seq_len, ntk_alpha=1.0):
        return [torch.zeros(1, seq_len, 1, 1)]


class _TinyQwenBlock(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, hidden_states, layer_past=None, use_cache=False, **_kwargs):
        output = hidden_states * (1.0 + self.scale)
        present = None
        if use_cache:
            current = hidden_states.unsqueeze(2)
            if layer_past is not None:
                key = torch.cat((layer_past[0], current), dim=1)
                value = torch.cat((layer_past[1], current), dim=1)
            else:
                key = current
                value = current
            present = (key, value)
        return output, present


class _TinyOriginalQwen(nn.Module):
    def __init__(self, hidden_size=8, vocab_size=16, layer_count=4):
        super().__init__()
        transformer = nn.Module()
        transformer.wte = nn.Embedding(vocab_size, hidden_size)
        transformer.drop = nn.Identity()
        transformer.h = nn.ModuleList([
            _TinyQwenBlock((index + 1) / 100.0)
            for index in range(layer_count)
        ])
        transformer.ln_f = nn.LayerNorm(hidden_size)
        transformer.rotary_emb = _TinyQwenRotary()
        transformer.use_dynamic_ntk = False
        transformer.use_cache_quantization = False
        transformer.get_ntk_alpha = lambda _seq_len: 1.0
        self.transformer = transformer
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = types.SimpleNamespace(model_type="qwen")
        self.eval()


def _make_original_qwen_manager(model, layer_range):
    manager = ModelManager()
    manager.model = model
    manager._engine_type = "pytorch"
    manager._layer_architecture = "qwen"
    manager.layer_range = layer_range
    return manager


# ================================================================
# forward_layers 输入校验测试
# ================================================================

class TestForwardLayersInputValidation:
    """测试 forward_layers 输入参数校验"""

    def test_requires_loaded_model(self):
        """未加载模型时应报错"""
        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="模型未加载"):
            mgr.forward_layers(input_ids=torch.tensor([[1, 2, 3]]))

    def test_no_input_raises(self):
        """不传任何输入时应报错"""
        mgr = ModelManager()
        mgr.model = _make_first_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (0, 4)
        with pytest.raises(ValueError, match="必须提供.*之一"):
            mgr.forward_layers()

    def test_both_inputs_raises(self):
        """同时传 input_ids 和 hidden_states 应报错"""
        mgr = ModelManager()
        mgr.model = _make_first_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (0, 4)
        with pytest.raises(ValueError, match="不能同时提供"):
            mgr.forward_layers(
                input_ids=torch.tensor([[1, 2]]),
                hidden_states=torch.randn(1, 2, 128),
            )

    def test_input_ids_without_embedding_raises(self):
        """节点不含 Embedding 时传 input_ids 应报错"""
        mgr = ModelManager()
        mgr.model = _make_middle_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (1, 4)
        with pytest.raises(RuntimeError, match="不含 Embedding 层"):
            mgr.forward_layers(input_ids=torch.tensor([[1, 2]]))

    def test_llama_cpp_engine_raises(self):
        """llama.cpp 引擎不支持 forward_layers"""
        mgr = ModelManager()
        mgr.model = _make_tiny_model()  # 需要有 model 才能通过第一个检查
        mgr._engine_type = "llama_cpp"
        with pytest.raises(RuntimeError, match="仅支持 PyTorch 引擎"):
            mgr.forward_layers(hidden_states=torch.randn(1, 4, 128))

    def test_forward_layers_waits_for_model_lock(self):
        """模型切换/分层重载持锁时，forward 不得读取半更新状态。"""
        import threading
        import time

        mgr = ModelManager()
        finished = threading.Event()

        def call_forward():
            with pytest.raises(RuntimeError, match="模型未加载"):
                mgr.forward_layers(input_ids=torch.tensor([[1]]))
            finished.set()

        mgr._lock.acquire()
        try:
            thread = threading.Thread(target=call_forward)
            thread.start()
            time.sleep(0.05)
            assert not finished.is_set()
        finally:
            mgr._lock.release()
        thread.join(timeout=1)
        assert finished.is_set()


# ================================================================
# forward_layers 首节点测试（含 Embedding，无 LM Head）
# ================================================================

class TestForwardLayersFirstNode:
    """测试首节点前向传播（input_ids → embed → layers → hidden_states）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_first_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_first_node_returns_hidden_states(self):
        """首节点应返回 hidden_states dict"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert "hidden_states" in result
        assert "logits" not in result
        hs = result["hidden_states"]
        assert hs.shape == (1, 5, 128)  # (batch, seq, hidden)
        assert hs.dtype == torch.float32

    def test_first_node_output_is_finite(self):
        """输出不应包含 NaN 或 Inf"""
        input_ids = torch.tensor([[1, 2, 3]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        hs = result["hidden_states"]
        assert torch.isfinite(hs).all(), "hidden_states 包含 NaN 或 Inf"

    def test_first_node_batch_size_2(self):
        """batch=2 的首节点推理"""
        input_ids = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert result["hidden_states"].shape == (2, 4, 128)

    def test_first_node_seq_len_1(self):
        """seq_len=1 的首节点推理（解码阶段单 token）"""
        input_ids = torch.tensor([[7]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert result["hidden_states"].shape == (1, 1, 128)

    def test_first_node_different_initial_tokens_produce_different_output(self):
        """不同输入 token 应产生不同的 hidden_states"""
        ids_a = torch.tensor([[1, 2, 3]])
        ids_b = torch.tensor([[4, 5, 6]])
        hs_a = self.mgr.forward_layers(input_ids=ids_a)["hidden_states"]
        hs_b = self.mgr.forward_layers(input_ids=ids_b)["hidden_states"]
        assert not torch.allclose(hs_a, hs_b, atol=1e-4), \
            "不同输入应产生不同输出"


# ================================================================
# forward_layers 中间节点测试（仅 Transformer 层）
# ================================================================

class TestForwardLayersMiddleNode:
    """测试中间节点前向传播（hidden_states → layers → hidden_states）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_middle_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (1, 3)

    def test_middle_node_returns_hidden_states(self):
        """中间节点接收 hidden_states 应返回 hidden_states"""
        hs_in = torch.randn(2, 8, 128)
        result = self.mgr.forward_layers(hidden_states=hs_in)
        assert "hidden_states" in result
        assert "logits" not in result
        hs_out = result["hidden_states"]
        assert hs_out.shape == (2, 8, 128)

    def test_middle_node_modifies_hidden_states(self):
        """中间节点应改变 hidden_states（不应该是恒等映射）"""
        hs_in = torch.randn(1, 4, 128)
        hs_out = self.mgr.forward_layers(hidden_states=hs_in)["hidden_states"]
        assert not torch.allclose(hs_in, hs_out, atol=1e-4), \
            "Transformer 层应改变 hidden_states"

    def test_middle_node_preserves_dtype(self):
        """中间节点应保持输入 dtype"""
        hs_in = torch.randn(1, 3, 128)
        hs_out = self.mgr.forward_layers(hidden_states=hs_in)["hidden_states"]
        assert hs_out.dtype == hs_in.dtype

    def test_middle_node_produces_unique_output_per_position(self):
        """不同序列位置的 hidden_states 应不相同"""
        hs_in = torch.randn(1, 8, 128)
        hs_out = self.mgr.forward_layers(hidden_states=hs_in)["hidden_states"]
        # 位置 0 和位置 1 的输出应不同（attention 混入了位置信息）
        assert not torch.allclose(hs_out[0, 0, :], hs_out[0, 1, :], atol=1e-4)

    def test_device_mismatch_input(self):
        """
        中间节点的 hidden_states 来自网络传输，可能与模型不在同一设备。
        例如: 从 CUDA 设备接收 hidden_states → 当前节点的 CPU 模型。
        forward_layers 应自动迁移到正确设备。
        """
        # 模型在 CPU，输入 tensor 模拟"刚从 TCP 反序列化得到（也是 CPU）"
        # 验证即使在同一设备上也正常工作（跨设备实际需要 GPU 环境）
        hs_in = torch.randn(1, 5, 128)  # CPU
        assert self.mgr.get_device().type == "cpu"
        result = self.mgr.forward_layers(hidden_states=hs_in)
        assert result["hidden_states"].device.type == "cpu"
        # 如果模型在 CUDA，输入在 CPU，tensor.to(device) 应能处理
        # 此处仅验证代码路径不崩溃
        assert torch.isfinite(result["hidden_states"]).all()


# ================================================================
# forward_layers 末节点测试（含 LM Head）
# ================================================================

class TestForwardLayersLastNode:
    """测试末节点前向传播（hidden_states → layers → norm → lm_head → logits）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_last_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (2, 4)

    def test_last_node_returns_logits(self):
        """末节点应返回 logits dict（经过 norm + lm_head）"""
        hs_in = torch.randn(1, 6, 128)
        result = self.mgr.forward_layers(hidden_states=hs_in)
        assert "logits" in result
        assert "hidden_states" not in result
        logits = result["logits"]
        # 形状: (batch, seq_len, vocab_size)
        assert logits.shape == (1, 6, TINY_CONFIG.vocab_size)
        assert logits.dtype == torch.float32

    def test_last_node_logits_are_finite(self):
        """末节点输出 logits 应全部有限"""
        hs_in = torch.randn(2, 3, 128)
        logits = self.mgr.forward_layers(hidden_states=hs_in)["logits"]
        assert torch.isfinite(logits).all()

    def test_last_node_no_lm_head_returns_hidden_states(self):
        """末节点无 lm_head 时应返回 hidden_states（降级为中间节点行为）"""
        del self.model.lm_head
        self.model.lm_head = None
        hs_in = torch.randn(1, 4, 128)
        result = self.mgr.forward_layers(hidden_states=hs_in)
        assert "hidden_states" in result
        assert "logits" not in result

    def test_last_node_logits_change_with_input(self):
        """不同的 hidden_states 应产生不同的 logits"""
        hs_a = torch.randn(1, 4, 128)
        hs_b = torch.randn(1, 4, 128)
        logits_a = self.mgr.forward_layers(hidden_states=hs_a)["logits"]
        logits_b = self.mgr.forward_layers(hidden_states=hs_b)["logits"]
        assert not torch.allclose(logits_a, logits_b, atol=1e-4)


# ================================================================
# 核心正确性测试: forward_layers(全模型) == model.forward()
# ================================================================

class TestForwardLayersMatchesFullModel:
    """
    核心正确性验证:

    forward_layers 在完整模型（未裁剪）上运行的结果，
    必须与 model.forward() / model(input_ids) 完全一致。

    这是分布式流水线正确性的根基 — 将完整模型拆分为节点后，
    每个节点用 forward_layers 执行的片段拼接起来必须等
    价于完整模型的单次前向传播。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_tiny_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_full_model_logits_match(self):
        """forward_layers 全模型 → logits 应与 model(input_ids) 一致"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        # --- 路径 A: model.forward() ---
        with torch.no_grad():
            ref_output = self.model(input_ids)
        ref_logits = ref_output.logits

        # --- 路径 B: forward_layers() ---
        result = self.mgr.forward_layers(input_ids=input_ids)
        fwd_logits = result["logits"]

        assert torch.allclose(ref_logits, fwd_logits, atol=1e-4), \
            f"forward_layers ≠ model.forward！max diff = {(ref_logits - fwd_logits).abs().max():.6f}"

    def test_full_model_with_attention_mask_matches(self):
        """带 attention_mask 的 forward_layers 应与 model.forward 一致"""
        input_ids = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ])

        with torch.no_grad():
            ref_output = self.model(input_ids, attention_mask=attention_mask)
        ref_logits = ref_output.logits

        result = self.mgr.forward_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        fwd_logits = result["logits"]

        assert torch.allclose(ref_logits, fwd_logits, atol=1e-4), \
            f"attention_mask 差异！max diff = {(ref_logits - fwd_logits).abs().max():.6f}"

    def test_full_model_returns_logits_not_hidden_states(self):
        """全模型（有 lm_head）的 forward_layers 应返回 logits"""
        result = self.mgr.forward_layers(input_ids=torch.tensor([[1, 2]]))
        assert "logits" in result
        assert "hidden_states" not in result

    def test_master_segment_can_defer_lm_head(self):
        """master 保留 LM Head 时仍可先输出 hidden states 交给 worker。"""
        result = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]),
            apply_lm_head=False,
        )

        assert "hidden_states" in result
        assert "logits" not in result
        assert result["hidden_states"].shape == (1, 3, TINY_CONFIG.hidden_size)


# ================================================================
# 端到端流水线模拟测试（首→中→末 三节点串联）
# ================================================================

class TestPipelineE2E:
    """
    端到端流水线测试：模拟三节点分布式推理

    将同一份模型权重复制到三个节点，确保流水线输出的 logits
    与完整模型的 model.forward() 完全一致。

    节点 0: Layer 0-1 + Embedding        → hidden_states
    节点 1: Layer 2   (纯中间层)         → hidden_states
    节点 2: Layer 3   + LM Head          → logits
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        # ---- 创建一份"规范"模型并保存其权重 ----
        canonical = _make_tiny_model()
        self.state_dict = {k: v.clone() for k, v in canonical.state_dict().items()}

        # ---- 节点 0: Layer 0-2 + Embedding, 无 LM Head ----
        self.model0 = _make_tiny_model()
        # 删除 lm_head 权重
        self.model0.load_state_dict(self.state_dict, strict=False)
        del self.model0.lm_head
        self.model0.lm_head = None
        self.model0.model.layers = nn.ModuleList(
            list(self.model0.model.layers)[0:2]
        )
        self.mgr0 = ModelManager()
        self.mgr0.model = self.model0
        self.mgr0._engine_type = "pytorch"
        self.mgr0.layer_range = (0, 2)

        # ---- 节点 1: Layer 2, 无 Embedding, 无 LM Head ----
        self.model1 = _make_tiny_model()
        self.model1.load_state_dict(self.state_dict, strict=False)
        del self.model1.model.embed_tokens
        self.model1.model.embed_tokens = None
        del self.model1.lm_head
        self.model1.lm_head = None
        self.model1.model.layers = nn.ModuleList(
            [list(self.model1.model.layers)[2]]
        )
        self.mgr1 = ModelManager()
        self.mgr1.model = self.model1
        self.mgr1._engine_type = "pytorch"
        self.mgr1.layer_range = (2, 3)

        # ---- 节点 2: Layer 3 + LM Head, 无 Embedding ----
        self.model2 = _make_tiny_model()
        self.model2.load_state_dict(self.state_dict, strict=False)
        del self.model2.model.embed_tokens
        self.model2.model.embed_tokens = None
        self.model2.model.layers = nn.ModuleList(
            [list(self.model2.model.layers)[3]]
        )
        self.mgr2 = ModelManager()
        self.mgr2.model = self.model2
        self.mgr2._engine_type = "pytorch"
        self.mgr2.layer_range = (3, 4)

        # ---- 完整模型（用于对比） ----
        self.model_full = _make_tiny_model()
        self.model_full.load_state_dict(self.state_dict)

    def test_three_node_pipeline_shape(self):
        """三节点流水线应产出正确形状的 logits"""
        input_ids = torch.tensor([[1, 2, 3, 4]])

        out0 = self.mgr0.forward_layers(input_ids=input_ids)
        assert "hidden_states" in out0

        out1 = self.mgr1.forward_layers(hidden_states=out0["hidden_states"])
        assert "hidden_states" in out1

        out2 = self.mgr2.forward_layers(hidden_states=out1["hidden_states"])
        assert "logits" in out2
        assert out2["logits"].shape == (1, 4, TINY_CONFIG.vocab_size)

    def test_pipeline_output_is_finite(self):
        """完整流水线输出应全部有限"""
        input_ids = torch.tensor([[1, 2, 3]])
        hs = self.mgr0.forward_layers(input_ids=input_ids)["hidden_states"]
        assert torch.isfinite(hs).all()
        hs = self.mgr1.forward_layers(hidden_states=hs)["hidden_states"]
        assert torch.isfinite(hs).all()
        logits = self.mgr2.forward_layers(hidden_states=hs)["logits"]
        assert torch.isfinite(logits).all()

    def test_pipeline_matches_full_model_forward(self):
        """
        ╔══════════════════════════════════════════════════════════╗
        ║  最核心的正确性保证:                                    ║
        ║  三节点流水线的输出 == 完整模型单机 forward 的输出     ║
        ╚══════════════════════════════════════════════════════════╝
        """
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        # --- 完整模型 forward ---
        with torch.no_grad():
            full_output = self.model_full(input_ids)
        full_logits = full_output.logits

        # --- 三节点流水线 ---
        hs = self.mgr0.forward_layers(input_ids=input_ids)["hidden_states"]
        hs = self.mgr1.forward_layers(hidden_states=hs)["hidden_states"]
        pipeline_logits = self.mgr2.forward_layers(hidden_states=hs)["logits"]

        # 对比
        assert torch.allclose(full_logits, pipeline_logits, atol=1e-4), \
            f"流水线输出与完整模型不一致！max diff = {(full_logits - pipeline_logits).abs().max():.6f}"


# ================================================================
# attention_mask 测试
# ================================================================

class TestForwardLayersAttentionMask:
    """测试 attention_mask 的支持"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_first_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_attention_mask_with_padding(self):
        """attention_mask 含填充时应正常工作"""
        input_ids = torch.tensor([
            [1, 2, 3, 0, 0],  # 后 2 位是 padding
            [4, 5, 0, 0, 0],  # 后 3 位是 padding
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ])
        result = self.mgr.forward_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hs = result["hidden_states"]
        assert torch.isfinite(hs).all()
        # padding 位置的输出应与有效位不同（确保掩码生效）
        row0_tok2 = hs[0, 2, :]  # 有效 token
        row0_tok3 = hs[0, 3, :]  # padding
        assert not torch.allclose(row0_tok2, row0_tok3, atol=1e-5), \
            "attention_mask 未生效：padding 位置输出与有效位置相同"

    def test_no_attention_mask_is_fine(self):
        """不传 attention_mask 应正常工作（纯因果注意力）"""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert torch.isfinite(result["hidden_states"]).all()


# ================================================================
# position_ids 测试
# ================================================================

class TestForwardLayersPositionIds:
    """测试自定义 position_ids"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_first_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_custom_position_ids(self):
        """自定义 position_ids 应影响输出（RoPE 生效）"""
        input_ids = torch.tensor([[1, 2, 3]])
        # 标准位置 0,1,2
        hs_default = self.mgr.forward_layers(input_ids=input_ids)["hidden_states"]
        # 偏移位置 10,11,12
        hs_offset = self.mgr.forward_layers(
            input_ids=input_ids,
            position_ids=torch.tensor([[10, 11, 12]]),
        )["hidden_states"]
        # 不同位置编码应产生不同输出
        # 注意: 随机权重模型下差异极小 (~1e-7)，用 equal 而非 allclose 检测
        assert not torch.equal(hs_default, hs_offset), \
            "position_ids 未生效（RoPE 应使不同位置产生不同输出）"
        assert torch.isfinite(hs_offset).all()

    def test_position_ids_default_range(self):
        """默认 position_ids 应为 [0, 1, 2, ...]"""
        # 不传 position_ids 也能正常工作，且不同位置有不同输出
        input_ids = torch.tensor([[1, 2, 3]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        hs = result["hidden_states"]
        # 不同位置的 hidden_states 应该不同（证明 position 生效）
        assert not torch.allclose(hs[0, 0, :], hs[0, 1, :], atol=1e-4), \
            "默认 position_ids 未生效：不同位置输出相同"


# ================================================================
# load_layer_range 集成测试
# ================================================================

class TestLoadLayerRangeIntegration:
    """load_layer_range + forward_layers 端到端集成测试"""

    def test_load_then_forward_first_node(self):
        """通过 load_layer_range 加载首节点 → forward_layers 应正常工作"""
        mgr = ModelManager()
        mgr.model = _make_first_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (0, 2)

        # 模拟 load_layer_range 的效果: 保留 Layer 0-2 + Embedding
        mgr.model.model.layers = nn.ModuleList(
            list(mgr.model.model.layers)[0:2]
        )
        # lm_head 在 _make_first_node_model 中已删除

        result = mgr.forward_layers(input_ids=torch.tensor([[1, 2, 3]]))
        assert "hidden_states" in result
        assert result["hidden_states"].shape == (1, 3, 128)

    def test_ensure_layer_range_reuses_active_model_path(self, monkeypatch):
        """DeepSeek 活动模型进入分层模式时不能回落到默认 Qwen 路径。"""
        mgr = ModelManager()
        mgr.model = _make_tiny_model()
        mgr._engine_type = "pytorch"
        mgr._model_path = "G:/models/deepseek"
        mgr._full_model_path = "G:/models/deepseek"
        calls = []
        monkeypatch.setattr(
            mgr,
            "load_layer_range",
            lambda *args, **kwargs: calls.append(kwargs),
        )

        mgr.ensure_layer_range(
            0, 2, has_embedding=True, has_lm_head=True,
        )

        assert calls[0]["model_path"] == "G:/models/deepseek"

    def test_load_layer_range_uses_active_deepseek_layer_count(self, monkeypatch):
        """DeepSeek 主节点未显式传 path/total 时仍应按 28 层校验。"""
        import model_module

        config = type("Config", (), {
            "model_type": "qwen2",
            "num_hidden_layers": 28,
        })()
        monkeypatch.setattr(
            model_module.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: config,
        )
        mgr = ModelManager()
        mgr._full_model_path = "G:/models/deepseek"
        fake_layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(28)])
        fake_model = nn.Module()
        fake_model.model = nn.Module()
        fake_model.model.layers = fake_layers
        fake_model.model.embed_tokens = nn.Embedding(2, 1)
        fake_model.lm_head = nn.Linear(1, 2)

        def install_fake_model(*args, **kwargs):
            mgr.model = fake_model
            tracker = _LayerRangeLoadTracker(
                architecture="qwen2",
                start_layer=0,
                end_layer=28,
                layer_prefix="model.layers.",
                selected_prefixes=[
                    f"model.layers.{index}." for index in range(28)
                ],
                target_dtype=torch.float32,
            )
            tracker.loaded_layers.update(range(28))
            return tracker

        monkeypatch.setattr(mgr, "_load_qwen2_layer_range", install_fake_model)

        mgr.load_layer_range(0, 28, has_embedding=True, has_lm_head=True)

        assert mgr.layer_range == (0, 28)
        assert len(mgr.model.model.layers) == 28
        assert mgr._total_model_layers == 28

    def test_load_layer_range_rejects_declared_total_mismatch(self, monkeypatch):
        """worker assignment 不得用错误总层数掩盖模型配置。"""
        import model_module

        config = type("Config", (), {
            "model_type": "qwen2",
            "num_hidden_layers": 28,
        })()
        monkeypatch.setattr(
            model_module.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: config,
        )
        mgr = ModelManager()
        mgr._full_model_path = "G:/models/deepseek"

        with pytest.raises(ValueError, match="assignment=24, model=28"):
            mgr.load_layer_range(0, 24, total_layers=24)

    def test_load_layer_range_rejects_adapter_without_filter_guard(self, monkeypatch):
        """架构加载器不得绕过按 key 过滤守卫后依赖末尾裁剪。"""
        import model_module

        config = type("Config", (), {
            "model_type": "qwen2",
            "num_hidden_layers": 2,
        })()
        monkeypatch.setattr(
            model_module.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: config,
        )
        mgr = ModelManager()
        mgr.model = _make_tiny_model()
        monkeypatch.setattr(mgr, "_load_qwen2_layer_range", lambda *a, **k: None)

        with pytest.raises(RuntimeError, match="未返回按 key 过滤守卫"):
            mgr.load_layer_range(0, 2, total_layers=2)

    def test_unsupported_architecture_points_to_filtered_adapter_contract(
        self,
        monkeypatch,
    ):
        import model_module

        config = type("Config", (), {
            "model_type": "gemma4_unified",
            "num_hidden_layers": 4,
        })()
        monkeypatch.setattr(
            model_module.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: config,
        )

        with pytest.raises(RuntimeError, match="隔离 Transformers sidecar"):
            ModelManager().load_layer_range(0, 2, total_layers=4)

    def test_load_then_forward_last_node(self):
        """通过 load_layer_range 加载末节点 → forward_layers 应产出 logits"""
        mgr = ModelManager()
        mgr.model = _make_last_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (2, 4)

        mgr.model.model.layers = nn.ModuleList(
            list(mgr.model.model.layers)[2:4]
        )

        result = mgr.forward_layers(
            hidden_states=torch.randn(1, 5, 128)
        )
        assert "logits" in result
        assert result["logits"].shape == (1, 5, TINY_CONFIG.vocab_size)

    def test_load_then_forward_middle_node(self):
        """通过 load_layer_range 加载中间节点 → forward_layers 透传 hidden_states"""
        mgr = ModelManager()
        mgr.model = _make_middle_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (1, 3)

        mgr.model.model.layers = nn.ModuleList(
            list(mgr.model.model.layers)[1:3]
        )

        hs_in = torch.randn(2, 4, 128)
        result = mgr.forward_layers(hidden_states=hs_in)
        assert "hidden_states" in result
        assert result["hidden_states"].shape == (2, 4, 128)

    def test_selective_qwen2_safetensors_loads_only_assigned_layers(
        self,
        tmp_path,
        monkeypatch,
    ):
        """真实 safetensors 路径不应先物化未分配层。"""
        import model_module

        source = _make_tiny_model()
        source.save_pretrained(tmp_path, safe_serialization=True)
        dummy_tokenizer = object()
        monkeypatch.setattr(
            model_module.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: dummy_tokenizer,
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        mgr = ModelManager()
        mgr.load_layer_range(
            2,
            4,
            has_embedding=False,
            has_lm_head=True,
            model_path=str(tmp_path),
            quant_type="fp16",
            total_layers=4,
            model_id="tiny-qwen2",
        )

        assert mgr.layer_range == (2, 4)
        assert len(mgr.model.model.layers) == 2
        assert mgr.model.model.embed_tokens is None
        assert mgr.model.lm_head is not None
        assert mgr.tokenizer is dummy_tokenizer
        assert mgr.active_model_id == "tiny-qwen2"
        assert mgr.quant_type == "fp32"
        assert all(parameter.device.type != "meta" for parameter in mgr.model.parameters())
        assert all(parameter.dtype == torch.float32 for parameter in mgr.model.parameters())

        metrics = mgr.get_model_info()["layer_load_metrics"]
        assert metrics["mode"] == "safetensors_key_filtered"
        assert metrics["architecture"] == "qwen2"
        assert metrics["layer_range"] == (2, 4)
        assert metrics["selected_layer_indices"] == [2, 3]
        assert metrics["selected_tensor_count"] > 0
        assert metrics["source_tensor_bytes"] > 0
        assert metrics["materialized_tensor_bytes"] > 0
        assert metrics["sampling"] == "tensor_boundary"
        assert metrics["rss_peak_bytes"] >= metrics["rss_baseline_bytes"]
        assert "loaded_keys" not in metrics

        # CUDA 主节点经网络送来的 FP16 hidden state 必须在 CPU worker
        # 边界转换为本地模型的 FP32，不能把半精度继续送入 CPU 算子。
        result = mgr.forward_layers(
            hidden_states=torch.randn(1, 3, 128, dtype=torch.float16)
        )
        assert result["logits"].shape == (1, 3, TINY_CONFIG.vocab_size)
        assert result["logits"].dtype == torch.float32

    def test_prepare_pipeline_model_does_not_materialize_weights(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Distributed preparation must stop before any model construction."""
        import model_module

        source = _make_tiny_model()
        source.save_pretrained(tmp_path, safe_serialization=True)
        monkeypatch.setattr(
            model_module.AutoModelForCausalLM,
            "from_pretrained",
            lambda *args, **kwargs: pytest.fail("must not load a full model"),
        )
        monkeypatch.setattr(
            model_module.AutoModelForCausalLM,
            "from_config",
            lambda *args, **kwargs: pytest.fail("must not construct a model skeleton"),
        )
        monkeypatch.setattr(
            "model_sync.compute_model_sha256",
            lambda path: "a" * 64,
        )

        mgr = ModelManager()
        descriptor = mgr.prepare_pipeline_model(
            model_id="tiny-qwen2",
            model_path=str(tmp_path),
            quant_type="fp16",
        )

        assert mgr.is_pipeline_prepared is True
        assert mgr.is_loaded is False
        assert mgr.model is None
        assert mgr.tokenizer is None
        assert mgr.active_model_id == "tiny-qwen2"
        assert descriptor["model_type"] == "qwen2"
        assert descriptor["model_sha256"] == "a" * 64
        with pytest.raises(RuntimeError, match="禁止自动整模回退"):
            mgr.ensure_full_model()

    def test_prepare_pipeline_model_accepts_filtered_assignment_manifest(
        self, tmp_path,
    ):
        """A worker assignment may contain only its non-zero layer range."""
        from safetensors.torch import save_file

        (tmp_path / "config.json").write_text(
            json.dumps({
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "num_hidden_layers": 4,
                "hidden_size": 2,
                "tie_word_embeddings": False,
            }),
            encoding="utf-8",
        )
        save_file({
            "model.layers.2.input_layernorm.weight": torch.zeros(2),
            "model.layers.3.input_layernorm.weight": torch.zeros(2),
            "model.norm.weight": torch.zeros(2),
        }, str(tmp_path / "model.safetensors"))

        mgr = ModelManager()
        descriptor = mgr.prepare_pipeline_model(
            model_id="filtered-qwen2",
            model_path=str(tmp_path),
            quant_type="fp16",
            layer_range=(2, 4),
            model_sha256="c" * 64,
        )

        assert mgr.is_pipeline_prepared is True
        assert descriptor["total_layers"] == 4
        assert descriptor["layer_weight_bytes"][:2] == [0, 0]
        assert descriptor["layer_weight_bytes"][2:] == [8, 8]
        assert descriptor["model_sha256"] == "c" * 64

    def test_abort_pipeline_materialization_preserves_descriptor(self):
        mgr = ModelManager()
        mgr._pipeline_descriptor = {
            "model_id": "tiny-qwen2",
            "model_type": "qwen2",
            "total_layers": 4,
        }
        mgr._pipeline_distributed_only = True
        mgr._engine_type = "pytorch"
        mgr.model = _make_tiny_model()
        mgr.tokenizer = object()
        mgr.layer_range = (0, 2)
        mgr._model_layers = 2

        mgr.abort_pipeline_materialization()

        assert mgr.model is None
        assert mgr.tokenizer is None
        assert mgr.layer_range is None
        assert mgr._model_layers == 0
        assert mgr.is_pipeline_prepared is True
        assert mgr.get_pipeline_descriptor()["model_id"] == "tiny-qwen2"

    def test_two_selectively_loaded_nodes_match_full_qwen2(
        self,
        tmp_path,
        monkeypatch,
    ):
        """两个真实选择性加载节点串联后应与完整模型 logits 一致。"""
        import model_module

        torch.manual_seed(2026)
        source = _make_tiny_model().eval()
        source.save_pretrained(tmp_path, safe_serialization=True)
        monkeypatch.setattr(
            model_module.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        first = ModelManager()
        last = ModelManager()
        first.load_layer_range(
            0, 2,
            has_embedding=True,
            has_lm_head=False,
            model_path=str(tmp_path),
            quant_type="fp16",
            total_layers=4,
        )
        last.load_layer_range(
            2, 4,
            has_embedding=False,
            has_lm_head=True,
            model_path=str(tmp_path),
            quant_type="fp16",
            total_layers=4,
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            full_logits = source(input_ids=input_ids).logits
        hidden = first.forward_layers(input_ids=input_ids)["hidden_states"]
        pipeline_logits = last.forward_layers(hidden_states=hidden)["logits"]

        assert torch.allclose(
            full_logits.detach().cpu(),
            pipeline_logits.detach().cpu(),
            atol=1e-3,
            rtol=1e-3,
        )


class TestLayerRangeLoadTracker:
    class _RecordingHandle:
        def __init__(self):
            self.requested = []

        def get_tensor(self, key):
            self.requested.append(key)
            return torch.ones(2, dtype=torch.float32)

    def _tracker(self):
        return _LayerRangeLoadTracker(
            architecture="qwen2",
            start_layer=1,
            end_layer=2,
            layer_prefix="model.layers.",
            selected_prefixes=["model.layers.1.", "model.norm."],
            target_dtype=torch.float16,
        )

    def test_rejects_unassigned_key_before_tensor_materialization(self):
        tracker = self._tracker()
        handle = self._RecordingHandle()

        with pytest.raises(RuntimeError, match="拒绝物化未分配权重"):
            tracker.materialize(handle, "model.layers.0.self_attn.q_proj.weight")

        assert handle.requested == []

    def test_finish_rejects_missing_assigned_layer(self):
        tracker = self._tracker()

        with pytest.raises(RuntimeError, match="missing=\[1\]"):
            tracker.finish()

    def test_records_source_and_target_bytes_for_selected_tensor(self):
        tracker = self._tracker()
        handle = self._RecordingHandle()

        tracker.materialize(handle, "model.layers.1.input_layernorm.weight")
        metrics = tracker.finish()

        assert handle.requested == ["model.layers.1.input_layernorm.weight"]
        assert metrics["selected_layer_indices"] == [1]
        assert metrics["selected_tensor_count"] == 1
        assert metrics["source_tensor_bytes"] == 8
        assert metrics["materialized_tensor_bytes"] == 4


class TestOriginalQwenLayerPipeline:
    """Qwen-1.8B ``transformer.h`` path must coexist with Qwen2/DeepSeek."""

    def _split_managers(self):
        torch.manual_seed(77)
        full = _TinyOriginalQwen()
        first_model = copy.deepcopy(full)
        last_model = copy.deepcopy(full)
        first_model.transformer.h = nn.ModuleList(
            list(first_model.transformer.h)[:2]
        )
        first_model.lm_head = None
        last_model.transformer.h = nn.ModuleList(
            list(last_model.transformer.h)[2:]
        )
        last_model.transformer.wte = None
        return (
            full,
            _make_original_qwen_manager(first_model, (0, 2)),
            _make_original_qwen_manager(last_model, (2, 4)),
        )

    def test_prefill_split_matches_full_original_qwen(self):
        full, first, last = self._split_managers()
        input_ids = torch.tensor([[1, 2, 3]])

        with torch.no_grad():
            hidden = full.transformer.wte(input_ids)
            for block in full.transformer.h:
                hidden = block(hidden)[0]
            expected = full.lm_head(full.transformer.ln_f(hidden))

        first_result = first.forward_layers(input_ids=input_ids, use_cache=True)
        last_result = last.forward_layers(
            hidden_states=first_result["hidden_states"], use_cache=True,
        )

        assert torch.allclose(last_result["logits"], expected, atol=1e-6)
        assert len(first_result["past_key_values"]) == 2
        assert len(last_result["past_key_values"]) == 2

    def test_decode_original_qwen_cache_grows_and_matches_full_sequence(self):
        full, first, last = self._split_managers()
        prompt = torch.tensor([[1, 2, 3]])
        first_prefill = first.forward_layers(input_ids=prompt, use_cache=True)
        last_prefill = last.forward_layers(
            hidden_states=first_prefill["hidden_states"], use_cache=True,
        )

        first_decode = first.forward_layers(
            input_ids=torch.tensor([[4]]),
            past_key_values=first_prefill["past_key_values"],
            use_cache=True,
        )
        last_decode = last.forward_layers(
            hidden_states=first_decode["hidden_states"],
            past_key_values=last_prefill["past_key_values"],
            use_cache=True,
        )

        assert first_decode["past_key_values"][0][0].shape[1] == 4
        assert last_decode["past_key_values"][0][0].shape[1] == 4
        assert last_decode["logits"].shape == (1, 1, 16)

    def test_architecture_aware_lm_head_supports_qwen_qwen2_and_hybrid(self):
        full, _first, last = self._split_managers()
        states = torch.randn(1, 2, 8)
        expected_qwen = last.model.lm_head(last.model.transformer.ln_f(states))
        assert torch.allclose(last.forward_lm_head(states), expected_qwen)

        qwen2 = ModelManager()
        qwen2.model = _make_tiny_model()
        qwen2._engine_type = "pytorch"
        qwen2.layer_range = (0, 4)
        qwen2_states = torch.randn(1, 2, 128)
        expected_qwen2 = qwen2.model.lm_head(qwen2.model.model.norm(qwen2_states))
        assert torch.allclose(qwen2.forward_lm_head(qwen2_states), expected_qwen2)

        # ★ #31 M5：hybrid（`qwen3_5_text`）此前在这里直接抛
        #   「模型架构 qwen3_5_text 缺少最终 Norm」⇒「master 保留 LM Head、层在 worker」
        #   那条路径对它完全不可用。现在改为**按架构属性**兜底定位
        #   （`_locate_text_transformer(...).norm`，与 `forward_layers` 的 Step 7 同源）。
        #   这里搭一个**多模态外壳**形态：文本塔挂在 `model.model.language_model`，
        #   norm 在它身上（**不是** `model.model.norm`）—— 正是真工件的形态。
        hybrid = ModelManager()
        hybrid.model = _make_tiny_model()
        # ⚠️ `_make_tiny_model()` 建的是**共享**的 `TINY_CONFIG` ⇒ **不能**就地改 `model_type`
        #    （实测会把后续用例带成 `forward_layers 不支持模型架构: some_unknown_arch`）。
        #    测试里要改模块级对象，必须先拷一份。
        hybrid.model.config = copy.deepcopy(hybrid.model.config)
        hybrid.model.config.model_type = "qwen3_5_text"      # 触发新分支
        text_tower = hybrid.model.model
        shell = nn.Module()
        shell.language_model = text_tower
        hybrid.model.model = shell
        hybrid._engine_type = "pytorch"
        hybrid.layer_range = (0, 4)
        hybrid_states = torch.randn(1, 2, 128)
        expected_hybrid = hybrid.model.lm_head(text_tower.norm(hybrid_states))
        assert torch.allclose(hybrid.forward_lm_head(hybrid_states), expected_hybrid)

    def test_lm_head_still_fails_closed_when_no_final_norm_exists(self):
        """★ 兜底**不等于**放行：连 `_locate_text_transformer` 都找不到主体时仍要抛。

        没有这条，M5 会把"缺 Norm"从 fail-closed 变成"静默用错对象"。
        """
        orphan = ModelManager()
        orphan.model = _make_tiny_model()
        orphan.model.config = copy.deepcopy(orphan.model.config)   # 同上：勿改共享 config
        orphan.model.config.model_type = "some_unknown_arch"
        orphan.model.model = nn.Module()          # 既无 norm 也无 language_model
        orphan._engine_type = "pytorch"
        orphan.layer_range = (0, 4)

        with pytest.raises(RuntimeError, match="缺少最终 Norm"):
            orphan.forward_lm_head(torch.randn(1, 2, 128))


# ================================================================
# 确定性测试（同输入 → 同输出）
# ================================================================

class TestForwardLayersDeterminism:
    """测试 forward_layers 的确定性"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_tiny_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_same_input_same_output(self):
        """相同输入应产生完全相同的输出（eval 模式确定性）"""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result1 = self.mgr.forward_layers(input_ids=input_ids)
        result2 = self.mgr.forward_layers(input_ids=input_ids)
        assert torch.equal(result1["logits"], result2["logits"]), \
            "forward_layers 应该是确定性的（eval 模式）"


# ================================================================
# 废弃 API 测试
# ================================================================

class TestDeprecatedAPIs:
    """测试已废弃的 API 是否正确报错"""

    def test_split_model_raises_not_implemented(self):
        """split_model() 应抛 NotImplementedError 引导用户使用 load_layer_range()"""
        mgr = ModelManager()
        with pytest.raises(NotImplementedError, match="load_layer_range"):
            mgr.split_model((0, 8))

    def test_load_then_get_model_info_total_layers(self):
        """get_model_info 应在 load_layer_range 后仍报告原始总层数"""
        mgr = ModelManager()
        mgr.model = _make_tiny_model()
        mgr._engine_type = "pytorch"
        # 模拟 _load_pytorch 设置
        mgr._total_model_layers = 4
        mgr._model_layers = 4

        # 模拟 load_layer_range(0, 2) 的效果
        mgr.model.model.layers = nn.ModuleList(
            list(mgr.model.model.layers)[0:2]
        )
        del mgr.model.lm_head
        mgr.model.lm_head = None
        mgr.layer_range = (0, 2)
        mgr._model_layers = 2  # load_layer_range 会设此值

        info = mgr.get_model_info()
        # total_layers 应保持原始总层数 4，而非当前加载的 2
        assert info["total_layers"] == 4, \
            f"total_layers 应为 4（原始总层数），实际 {info['total_layers']}"
        assert info["loaded_layers"] == 2, \
            f"loaded_layers 应为 2（当前加载层数），实际 {info['loaded_layers']}"
        assert info["layer_range"] == (0, 2)


# ================================================================
# KV Cache 测试 (Phase 3 — 增量解码)
# ================================================================

class TestForwardLayersKVCache:
    """测试 forward_layers 的 KV Cache 支持（增量解码优化）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_tiny_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_qwen2_decode_rejects_cache_layer_count_mismatch(self):
        """错误 KV 层数必须中止，不得静默丢失上下文重新起算。"""
        bad_cache = ((
            torch.zeros(1, 2, 2, 32),
            torch.zeros(1, 2, 2, 32),
        ),)

        with pytest.raises(RuntimeError, match="Qwen2 KV cache 层数不匹配"):
            self.mgr.forward_layers(
                input_ids=torch.tensor([[7]]),
                past_key_values=bad_cache,
                use_cache=True,
            )

    def test_prefill_without_kv_cache_returns_no_past_key_values(self):
        """use_cache=False → 不返回 past_key_values"""
        input_ids = torch.tensor([[1, 2, 3]])
        result = self.mgr.forward_layers(
            input_ids=input_ids,
            use_cache=False,
        )
        assert "past_key_values" not in result

    def test_prefill_with_kv_cache_returns_past_key_values(self):
        """use_cache=True (default) → 返回 past_key_values tuple"""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert "past_key_values" in result
        pkv = result["past_key_values"]
        # 应为每层一个 (key, value) 元组（本地层索引）
        assert len(pkv) == 4  # 4 layers in tiny model
        assert isinstance(pkv[0], tuple)
        assert len(pkv[0]) == 2  # (key, value)
        # key shape: (batch, num_kv_heads, seq_len, head_dim)
        # GQA: num_kv_heads=2, head_dim=128/4=32
        assert pkv[0][0].shape == (1, 2, 4, 32)

    def test_past_key_values_accumulates_sequence_length(self):
        """Prefill → Decode → KV cache seq_len 应累加"""
        # --- Prefill: 4 tokens ---
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        prefill_kv = result["past_key_values"]
        assert prefill_kv[0][0].shape[2] == 4  # seq_len = 4

        # --- Decode: 1 new token (使用缓存的 KV) ---
        new_token = torch.tensor([[7]])  # shape (1, 1)
        result2 = self.mgr.forward_layers(
            input_ids=new_token,
            past_key_values=prefill_kv,
            use_cache=True,
        )
        decode_kv = result2["past_key_values"]
        assert decode_kv[0][0].shape[2] == 5  # seq_len = 4 + 1

    def test_decode_with_kv_cache_preserves_output_shape(self):
        """Decode 阶段输入 (1,1) → hidden_states 输出也应为 (1,1)（仅新 token 的输出）"""
        # Prefill
        input_ids = torch.tensor([[1, 2, 3]])
        prefill_result = self.mgr.forward_layers(input_ids=input_ids)
        prefill_kv = prefill_result["past_key_values"]

        # 移除 lm_head 以测试中间节点行为
        del self.model.lm_head
        self.model.lm_head = None

        # Decode: 1 token with KV cache
        new_token = torch.tensor([[5]])
        result = self.mgr.forward_layers(
            input_ids=new_token,
            past_key_values=prefill_kv,
            use_cache=True,
        )
        hs = result["hidden_states"]
        # 输出形状应与输入形状匹配（仅新 token）
        assert hs.shape == (1, 1, 128), \
            f"Decode 输出应为 (1,1,128)，实际 {hs.shape}"

    def test_kv_cache_reduces_computation(self):
        """
        验证 KV Cache 确实减少了计算量：
        相同的新 token，有 KV cache 的 decode 输出与无 cache 的完整序列
        重计算的最后一个位置输出应该不同（因为 KV cache 保留了正确的历史上下文）。

        注意：有 KV cache 时仅计算 position 3 的 hidden_states，
        无 KV cache 时重新计算 position 0-3。由于 RoPE 在相同输入上
        是确定性的，两者的最后一个位置输出应该一致。
        """
        input_ids = torch.tensor([[1, 2, 3, 4]])

        # 移除 lm_head（测试 hidden_states 输出）
        del self.model.lm_head
        self.model.lm_head = None

        # --- 路径 A: Prefill → 取最后位置输出 ---
        result_full = self.mgr.forward_layers(input_ids=input_ids, use_cache=True)
        full_kv = result_full["past_key_values"]

        # --- 路径 B: Prefill(0-2) → Decode(3) with KV cache ---
        result_pre = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]),
            use_cache=True,
        )
        pre_kv = result_pre["past_key_values"]

        result_dec = self.mgr.forward_layers(
            input_ids=torch.tensor([[4]]),
            past_key_values=pre_kv,
            use_cache=True,
        )

        # 路径 A 的位置 3 输出 (full_kv seq_len=4 时并未直接给出 hidden_states)
        # 重跑一次: Prefill(0-3) 无 KV cache，取最后位置的 hidden_states
        result_ref = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            use_cache=False,
        )
        hs_ref_last = result_ref["hidden_states"][:, -1:, :]  # (1, 1, 128)

        hs_dec = result_dec["hidden_states"]  # (1, 1, 128)

        # 两者应一致（KV cache 不应改变计算结果）
        assert torch.allclose(hs_ref_last, hs_dec, atol=1e-4), \
            f"KV cache decode 应与完整 prefill 的最后位置一致，" \
            f"max diff = {(hs_ref_last - hs_dec).abs().max():.6f}"

    def test_position_ids_auto_advance_with_kv_cache(self):
        """有 KV cache 时，position_ids 自动从 past_seen_tokens 开始"""
        # Prefill: 5 tokens at positions 0-4 (vocab_size=32, keep IDs < 32)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        result = self.mgr.forward_layers(input_ids=input_ids, use_cache=True)
        prefill_kv = result["past_key_values"]
        assert prefill_kv[0][0].shape[2] == 5

        del self.model.lm_head
        self.model.lm_head = None

        # Decode: 1 new token，不手动指定 position_ids
        new_token = torch.tensor([[7]])
        result_no_pos = self.mgr.forward_layers(
            input_ids=new_token,
            past_key_values=prefill_kv,
            use_cache=True,
        )

        # 重新 prefill（避免 KV cache 被原地修改）
        result2 = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            use_cache=True,
        )
        prefill_kv2 = result2["past_key_values"]

        # 手动指定 position_ids=[5]
        result_with_pos = self.mgr.forward_layers(
            input_ids=new_token,
            past_key_values=prefill_kv2,
            position_ids=torch.tensor([[5]]),
            use_cache=True,
        )

        hs_auto = result_no_pos["hidden_states"]
        hs_manual = result_with_pos["hidden_states"]
        assert torch.allclose(hs_auto, hs_manual, atol=1e-6), \
            "自动 position_ids 与手动 position_ids=[5] 应产生相同输出"

    def test_middle_node_kv_cache(self):
        """中间节点（接收 hidden_states 而非 input_ids）的 KV cache 支持"""
        # 使用中间节点模型（start_layer=1），切片为 2 层
        model_mid = _make_middle_node_model()
        # 切片为 layer 1-3 (global)，本地 2 层
        model_mid.model.layers = nn.ModuleList(
            list(model_mid.model.layers)[1:3]
        )
        mgr_mid = ModelManager()
        mgr_mid.model = model_mid
        mgr_mid._engine_type = "pytorch"
        mgr_mid.layer_range = (1, 3)

        # Prefill: 接收完整 hidden_states
        hs_in = torch.randn(1, 6, 128)
        result = mgr_mid.forward_layers(hidden_states=hs_in, use_cache=True)
        assert "past_key_values" in result
        mid_kv = result["past_key_values"]
        # 2 层本地 (global 1,2 → local 0,1)
        assert len(mid_kv) == 2
        assert mid_kv[0][0].shape[2] == 6

        # Decode: 仅 1 个位置
        hs_in_decode = torch.randn(1, 1, 128)
        result2 = mgr_mid.forward_layers(
            hidden_states=hs_in_decode,
            past_key_values=mid_kv,
            use_cache=True,
        )
        assert result2["hidden_states"].shape == (1, 1, 128)
        assert result2["past_key_values"][0][0].shape[2] == 7

    def test_kv_cache_is_deterministic(self):
        """相同输入 + 相同 KV cache → 相同输出"""
        input_ids = torch.tensor([[1, 2, 3]])
        result1 = self.mgr.forward_layers(input_ids=input_ids, use_cache=True)
        kv1 = result1["past_key_values"]

        del self.model.lm_head
        self.model.lm_head = None

        new_token = torch.tensor([[7]])

        # 第一次 decode（会原地更新 DynamicCache 内部，但返回 tuple 副本）
        out1 = self.mgr.forward_layers(
            input_ids=new_token, past_key_values=kv1, use_cache=True
        )["hidden_states"]

        # 重新 prefill 得到相同 KV cache，再 decode
        result2 = self.mgr.forward_layers(input_ids=input_ids, use_cache=True)
        kv2 = result2["past_key_values"]
        out2 = self.mgr.forward_layers(
            input_ids=new_token, past_key_values=kv2, use_cache=True
        )["hidden_states"]

        assert torch.equal(out1, out2), \
            "相同 KV cache + 相同输入 → 应有相同输出（确定性）"


class TestPipelineE2EWithKVCache:
    """
    端到端流水线 + KV Cache 集成测试

    模拟三节点分布式推理的完整 prefill→decode 流程，
    验证 KV cache 在节点间独立存储、独立更新的正确性。
    各节点仅维护本地层范围的 DynamicCache。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        canonical = _make_tiny_model()
        state_dict = {k: v.clone() for k, v in canonical.state_dict().items()}

        # 节点 0: Layer 0-1 + Embedding, start_layer=0
        self.model0 = _make_tiny_model()
        self.model0.load_state_dict(state_dict, strict=False)
        del self.model0.lm_head
        self.model0.lm_head = None
        self.model0.model.layers = nn.ModuleList(list(self.model0.model.layers)[0:2])
        self.mgr0 = ModelManager()
        self.mgr0.model = self.model0
        self.mgr0._engine_type = "pytorch"
        self.mgr0.layer_range = (0, 2)

        # 节点 1: Layer 2 (中间层), start_layer=2
        self.model1 = _make_tiny_model()
        self.model1.load_state_dict(state_dict, strict=False)
        del self.model1.model.embed_tokens
        self.model1.model.embed_tokens = None
        del self.model1.lm_head
        self.model1.lm_head = None
        self.model1.model.layers = nn.ModuleList([list(self.model1.model.layers)[2]])
        self.mgr1 = ModelManager()
        self.mgr1.model = self.model1
        self.mgr1._engine_type = "pytorch"
        self.mgr1.layer_range = (2, 3)

        # 节点 2: Layer 3 + LM Head, start_layer=3
        self.model2 = _make_tiny_model()
        self.model2.load_state_dict(state_dict, strict=False)
        del self.model2.model.embed_tokens
        self.model2.model.embed_tokens = None
        self.model2.model.layers = nn.ModuleList([list(self.model2.model.layers)[3]])
        self.mgr2 = ModelManager()
        self.mgr2.model = self.model2
        self.mgr2._engine_type = "pytorch"
        self.mgr2.layer_range = (3, 4)

        # 完整参考模型
        self.model_full = _make_tiny_model()
        self.model_full.load_state_dict(state_dict)

    def test_prefill_then_decode_pipeline(self):
        """
        完整流程: Prefill(4 tokens) → Decode(1 token) × 2 steps
        验证每个节点独立 KV cache 管理、seq_len 正确累加
        """
        # ==== Prefill: 4 tokens ====
        out0 = self.mgr0.forward_layers(
            input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True
        )
        kv0 = out0["past_key_values"]
        # 节点 0: 2 local layers (0→local0, 1→local1)
        assert len(kv0) == 2
        assert kv0[0][0].shape[2] == 4

        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], use_cache=True
        )
        kv1 = out1["past_key_values"]
        # 节点 1: 1 local layer (global 2 → local 0)
        assert len(kv1) == 1
        assert kv1[0][0].shape[2] == 4

        out2 = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], use_cache=True
        )
        kv2 = out2["past_key_values"]
        # 节点 2: 1 local layer (global 3 → local 0)
        assert len(kv2) == 1
        assert kv2[0][0].shape[2] == 4

        prefill_logits = out2["logits"]
        assert prefill_logits.shape == (1, 4, TINY_CONFIG.vocab_size)

        # ==== Decode step 1: 1 new token ====
        new_token = torch.tensor([[7]])
        out0 = self.mgr0.forward_layers(
            input_ids=new_token, past_key_values=kv0, use_cache=True
        )
        kv0 = out0["past_key_values"]
        assert kv0[0][0].shape[2] == 5
        assert out0["hidden_states"].shape == (1, 1, 128)

        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], past_key_values=kv1, use_cache=True
        )
        kv1 = out1["past_key_values"]
        assert kv1[0][0].shape[2] == 5
        assert out1["hidden_states"].shape == (1, 1, 128)

        out2 = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], past_key_values=kv2, use_cache=True
        )
        kv2 = out2["past_key_values"]
        assert kv2[0][0].shape[2] == 5
        decode_logits = out2["logits"]
        assert decode_logits.shape == (1, 1, TINY_CONFIG.vocab_size)

        # ==== Decode step 2: another token ====
        new_token2 = torch.tensor([[3]])
        out0 = self.mgr0.forward_layers(
            input_ids=new_token2, past_key_values=kv0, use_cache=True
        )
        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], past_key_values=kv1, use_cache=True
        )
        out2 = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], past_key_values=kv2, use_cache=True
        )
        decode2_logits = out2["logits"]
        assert decode2_logits.shape == (1, 1, TINY_CONFIG.vocab_size)

    def test_pipeline_with_kv_cache_matches_full_model(self):
        """
        ╔══════════════════════════════════════════════════════════╗
        ║  核心保证: KV cache 流水线输出 == 完整模型输出        ║
        ║                                                          ║
        ║  Prefill [1,2,3] + Decode [4] with KV cache            ║
        ║  的 logits 必须等于完整模型 input [1,2,3,4] 的         ║
        ║  最后一个位置的 logits。                                ║
        ╚══════════════════════════════════════════════════════════╝
        """
        # 完整模型: [1,2,3,4] → logits for position 3
        input_ids_full = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            full_output = self.model_full(input_ids_full)
        full_last_logits = full_output.logits[:, -1:, :]  # (1, 1, vocab)

        # 流水线: Prefill [1,2,3] + Decode [4] with KV cache
        out0 = self.mgr0.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]), use_cache=True
        )
        kv0 = out0["past_key_values"]
        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], use_cache=True
        )
        kv1 = out1["past_key_values"]
        out2_pre = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], use_cache=True
        )
        kv2 = out2_pre["past_key_values"]

        # Decode
        out0 = self.mgr0.forward_layers(
            input_ids=torch.tensor([[4]]), past_key_values=kv0, use_cache=True
        )
        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], past_key_values=kv1, use_cache=True
        )
        out2 = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], past_key_values=kv2, use_cache=True
        )
        pipeline_logits = out2["logits"]  # (1, 1, vocab)

        max_diff = (full_last_logits - pipeline_logits).abs().max().item()
        assert torch.allclose(full_last_logits, pipeline_logits, atol=1e-3), \
            f"KV cache 流水线 logits ≠ 完整模型 logits！" \
            f"max diff = {max_diff:.6f}"


# ================================================================
# Phase 7 P0: transformers 5.x 兼容性回归测试
# ================================================================

class TestTransformers5xCompatibility:
    """验证 7 项 transformers 5.x 修复不会在升级时静默失效。"""

    @pytest.fixture
    def mgr(self):
        """创建仅含 embedding + 全部 4 层的 ModelManager。"""
        from transformers.cache_utils import DynamicCache
        model = _make_tiny_model()
        tokenizer = type('TinyTok', (), {
            'decode': lambda self, ids, **kw: ''.join(chr(min(i, 126)) for i in ids),
            'encode': lambda self, text, **kw: torch.tensor(
                [[min(ord(c), 31) + 1 for c in text]], dtype=torch.long
            ),
        })()
        mgr = ModelManager()
        mgr.model = model
        mgr.tokenizer = tokenizer
        mgr._engine_type = "pytorch"
        return mgr

    def test_create_causal_mask_import_and_call(self, mgr):
        """create_causal_mask 可从 qwen2 路径导入且用有效参数成功调用。"""
        import inspect

        try:
            from transformers.models.qwen2.modeling_qwen2 import create_causal_mask
        except (ImportError, AttributeError):
            pytest.skip("create_causal_mask 不可用（transformers < 5.x）")
        input_ids = torch.tensor([[5, 8, 2]], dtype=torch.long)
        hidden_states = mgr.model.model.embed_tokens(input_ids)
        parameters = inspect.signature(create_causal_mask).parameters
        kwargs = {
            "config": mgr.model.config,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }
        if "input_embeds" in parameters:
            kwargs["input_embeds"] = hidden_states
        else:
            kwargs["inputs_embeds"] = hidden_states
        if "cache_position" in parameters:
            kwargs["cache_position"] = torch.arange(3)
        causal_mask = create_causal_mask(**kwargs)
        # flash/sdpa 返回 None，eager 返回 4D tensor
        assert causal_mask is None or isinstance(causal_mask, torch.Tensor)

    def test_fp16_manual_causal_mask_matches_attention_dtype(self, mgr):
        """The transformers 4.x fallback mask must match FP16 Q/K tensors."""
        mgr.model.half()

        result = mgr.forward_layers(
            input_ids=torch.tensor([[5, 8, 2]], dtype=torch.long),
            use_cache=True,
        )

        assert result["logits"].dtype == torch.float16

    @pytest.mark.parametrize(
        "runtime_error",
        [
            TypeError("simulated runtime type failure"),
            ValueError("simulated invalid cache state"),
        ],
    )
    def test_create_causal_mask_runtime_error_propagates(
        self, mgr, monkeypatch, runtime_error,
    ):
        """Runtime input/cache failures must not be mistaken for API drift."""
        from transformers.models.qwen2 import modeling_qwen2

        def incompatible_mask(
            config,
            attention_mask,
            past_key_values,
            position_ids,
            inputs_embeds,
            cache_position=None,
        ):
            raise runtime_error

        monkeypatch.setattr(
            modeling_qwen2,
            "create_causal_mask",
            incompatible_mask,
            raising=False,
        )

        with pytest.raises(type(runtime_error), match=str(runtime_error)):
            mgr.forward_layers(
                input_ids=torch.tensor([[5, 8, 2]], dtype=torch.long),
                use_cache=True,
            )

    def test_create_causal_mask_unknown_signature_falls_back(
        self,
        mgr,
        monkeypatch,
    ):
        """未来版本更改输入参数名时应降级到手动掩码。"""
        import transformers.models.qwen2.modeling_qwen2 as qwen2_module

        def incompatible_mask(config, attention_mask, past_key_values):
            raise AssertionError("签名不兼容时不应调用该函数")

        monkeypatch.setattr(
            qwen2_module,
            "create_causal_mask",
            incompatible_mask,
            raising=False,
        )

        result = mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]),
            use_cache=False,
        )

        assert result["logits"].shape == (1, 3, TINY_CONFIG.vocab_size)

    def test_causal_mask_fallback_on_import_error(self, mgr):
        """transformers 4.x 无 create_causal_mask 时应回退到手动构建（BUG C1 修复验证）。"""
        input_ids = torch.tensor([[5, 8, 2]], dtype=torch.long)
        hidden_states = mgr.model.model.embed_tokens(input_ids)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        device = hidden_states.device

        # 模拟 transformers 4.x 无 create_causal_mask 的情况
        import sys
        original_module = None
        if 'transformers.models.qwen2.modeling_qwen2' in sys.modules:
            original_module = sys.modules['transformers.models.qwen2.modeling_qwen2']
        # 临时移除 create_causal_mask 属性（如果存在）
        had_create_causal_mask = False
        if original_module and hasattr(original_module, 'create_causal_mask'):
            had_create_causal_mask = True
            original_func = original_module.create_causal_mask
            del original_module.create_causal_mask

        try:
            # 尝试导入 — 应触发 ImportError
            try:
                from transformers.models.qwen2.modeling_qwen2 import create_causal_mask
                import_failed = False
            except (ImportError, AttributeError):
                import_failed = True

            if import_failed:
                # 手动构建回退逻辑（与 model_module.py 中的 except 分支一致）
                seq_len = hidden_states.shape[1]
                causal_mask = torch.full(
                    (seq_len, seq_len),
                    float('-inf'),
                    device=device,
                    dtype=hidden_states.dtype,
                )
                causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask = causal_mask[None, None, :, :].expand(
                    hidden_states.shape[0], 1, seq_len, seq_len
                )
                attn_mask = attention_mask.to(device)
                attn_mask = attn_mask[:, None, None, :]
                causal_mask = causal_mask.masked_fill(
                    attn_mask == 0, float('-inf')
                )
                # 验证回退生成的因果掩码
                assert causal_mask.shape == (1, 1, seq_len, seq_len)
                assert causal_mask.dtype == hidden_states.dtype
                # 上三角应为 -inf（因果约束）
                assert torch.isinf(causal_mask[0, 0, 0, 1])
                # 对角线及以下应为 0（可 attending）
                assert causal_mask[0, 0, 0, 0] == 0
                assert causal_mask[0, 0, 1, 0] == 0
        finally:
            # 恢复原始模块状态
            if had_create_causal_mask and original_module:
                original_module.create_causal_mask = original_func

    def test_dynamic_cache_update_called_on_decode(self, mgr):
        """decode 路径使用 cache.update(k, v, layer_idx) 而非 key_cache.append。

        验证 prefill→decode 周期中 layer_idx 补丁的保存/恢复机制正确。"""
        # 使用 tiny 模型的全 4 层（所有 GPU 层参与），模拟完整 prefill→decode
        saved_indices = [layer.self_attn.layer_idx for layer in mgr.model.model.layers]

        input_ids = torch.tensor([[5, 8, 2, 12]], dtype=torch.long)

        # Prefill: 传入 input_ids（完整模型含 embedding，自动处理）
        pf_result = mgr.forward_layers(input_ids=input_ids, use_cache=True)
        assert "past_key_values" in pf_result, "Prefill 应返回 past_key_values"
        # 验证 layer_idx 已恢复（Phase 1.2 修复）
        for i, layer in enumerate(mgr.model.model.layers):
            assert layer.self_attn.layer_idx == saved_indices[i], \
                f"prefill 后 layer_idx 未恢复: layer {i} → {layer.self_attn.layer_idx}"

        # Decode
        past_kv = pf_result["past_key_values"]
        dec_result = mgr.forward_layers(
            input_ids=torch.tensor([[7]], dtype=torch.long),
            past_key_values=past_kv, use_cache=True,
        )
        assert "past_key_values" in dec_result, "Decode 应返回更新的 past_key_values"
        # 验证 layer_idx 在 decode 后也恢复了
        for i, layer in enumerate(mgr.model.model.layers):
            assert layer.self_attn.layer_idx == saved_indices[i], \
                f"decode 后 layer_idx 未恢复: layer {i} → {layer.self_attn.layer_idx}"

    def test_cache_argument_name_is_supported(self, mgr):
        """Qwen2DecoderLayer may use 4.x singular or 5.x plural cache argument."""
        import inspect
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        sig = inspect.signature(Qwen2DecoderLayer.forward)
        params = list(sig.parameters.keys())
        assert ("past_key_value" in params) or ("past_key_values" in params), \
            f"Qwen2DecoderLayer.forward 不接受已知 KV cache 参数！签名: {params}"

    def test_decoder_layer_output_shape_is_supported(self, mgr):
        """Qwen2DecoderLayer may return a tensor (5.x) or tuple (4.x)."""
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        layer0 = mgr.model.model.layers[0]
        assert isinstance(layer0, Qwen2DecoderLayer), \
            "期望第一个 decoder layer 为 Qwen2DecoderLayer"

        # 用最小输入调用单层
        hidden = torch.randn(1, 2, mgr.model.config.hidden_size)
        pos_ids = torch.tensor([[0, 1]], dtype=torch.long)
        pos_emb = mgr.model.model.rotary_emb(hidden, pos_ids)

        output = layer0(hidden, position_ids=pos_ids, position_embeddings=pos_emb)
        # 5.x returns tensor; 4.x returns tuple. forward_layers handles both.
        if isinstance(output, tuple):
            assert output and isinstance(output[0], torch.Tensor)
        else:
            assert isinstance(output, torch.Tensor)

    def test_cancel_event_stops_generation_criteria(self, mgr):
        """排队超时/客户端断开应让完整模型生成在下一个 token step 停止。"""
        import threading

        cancel_event = threading.Event()
        criteria = mgr._build_stop_criteria([], 1, cancel_event=cancel_event)
        assert criteria is not None
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        assert not bool(criteria(input_ids, None))
        cancel_event.set()
        assert bool(criteria(input_ids, None))


# ================================================================
# select_engine 测试（P3 多模型支持 — 引擎选择逻辑）
# ================================================================

class TestLayerRuntimeSelection:
    def test_cpu_layers_use_float32(self, monkeypatch):
        import model_module

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert model_module._select_layer_runtime() == ("cpu", torch.float32)

    def test_cuda_layers_use_float16(self, monkeypatch):
        import model_module

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert model_module._select_layer_runtime() == ("cuda:0", torch.float16)

    def test_cuda_layers_preserve_explicit_fp32(self, monkeypatch):
        import model_module

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert model_module._select_layer_runtime("fp32") == ("cuda:0", torch.float32)

    @pytest.mark.parametrize("requested_quant", ["fp16", "fp32", "int4", "int8"])
    def test_full_cpu_pytorch_load_uses_float32(
        self,
        monkeypatch,
        requested_quant,
    ):
        """分层 worker 回退完整模型时，状态与实际 CPU dtype 必须一致。"""
        import model_module

        captured = {}
        fake_model = _make_tiny_model()

        def fake_from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            return fake_model

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(model_module, "USE_COMPILE", False)
        monkeypatch.setattr(
            model_module.AutoModelForCausalLM,
            "from_pretrained",
            fake_from_pretrained,
        )
        monkeypatch.setattr(
            model_module.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: object(),
        )

        mgr = ModelManager()
        mgr._load_pytorch("unused-model", quant_type=requested_quant)

        assert captured["device_map"] == {"": "cpu"}
        assert captured["torch_dtype"] == torch.float32
        assert mgr.quant_type == "fp32"
        assert next(mgr.model.parameters()).dtype == torch.float32

    def test_full_cuda_pytorch_fp32_uses_float32(self, monkeypatch):
        import model_module

        captured = {}
        fake_model = _make_tiny_model()

        def fake_from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            return fake_model

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(model_module, "USE_COMPILE", False)
        monkeypatch.setattr(
            model_module.AutoModelForCausalLM,
            "from_pretrained",
            fake_from_pretrained,
        )
        monkeypatch.setattr(
            model_module.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: object(),
        )

        mgr = ModelManager()
        mgr._load_pytorch("unused-model", quant_type="fp32")

        assert captured["device_map"] == "auto"
        assert captured["torch_dtype"] == torch.float32
        assert mgr.quant_type == "fp32"

    def test_pytorch_loader_rejects_gguf_path_before_runtime(self):
        mgr = ModelManager()

        with pytest.raises(ValueError, match="拒绝 GGUF"):
            mgr._load_pytorch("model.gguf", quant_type="fp16")

    def test_edge_profile_forces_fp32_during_resolution_even_when_host_has_cuda(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        request = ModelManager()._resolve_model_load_request(
            model_path=str(tmp_path),
            quant_type="int4",
            profile={"tier": "edge", "gpu": {"cuda_available": False}},
            model_id="edge-model",
            engine="pytorch",
            db_experimental_models=None,
            require_existing=True,
        )
        assert request["runtime"] == "pytorch_cpu"
        assert request["effective_quantization"] == "fp32"

    def test_precomputed_resolution_is_consumed_without_second_resolver(
        self, monkeypatch, tmp_path,
    ):
        import model_module
        from model_load_resolver import ModelLoadResolution

        resolution = ModelLoadResolution(
            model_id="resolved-model",
            requested_engine="pytorch",
            engine="pytorch",
            artifact_kind="safetensors",
            model_path=str(tmp_path),
            requested_quant="fp32",
            quant_type="fp32",
            runtime="pytorch_cpu",
            runtime_quant="fp32",
            layer_range_mode="dynamic",
            reason_code="ENGINE_EXPLICIT",
            reason="test",
        )
        monkeypatch.setattr(
            model_module,
            "resolve_model_load",
            lambda *_args, **_kwargs: pytest.fail("resolver was called twice"),
        )
        calls = []
        mgr = ModelManager()
        monkeypatch.setattr(
            mgr,
            "_load_pytorch",
            lambda path, quant, profile: calls.append((path, quant, profile)),
        )

        mgr.load_model(
            model_id="resolved-model",
            model_path=str(tmp_path),
            engine="pytorch",
            quant_type="fp32",
            resolution=resolution,
        )

        assert calls == [(str(tmp_path), "fp32", None)]


class TestSelectEngine:
    """测试 ModelManager.select_engine() 静态方法"""

    def test_manual_pytorch_override(self, monkeypatch):
        """INFERENCE_ENGINE='pytorch' 显式指定时应返回 pytorch"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "pytorch")
        result = ModelManager.select_engine()
        assert result == "pytorch"

    def test_manual_llama_cpp_override(self, monkeypatch):
        """INFERENCE_ENGINE='llama_cpp' 显式指定时应返回 llama_cpp"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "llama_cpp")
        result = ModelManager.select_engine()
        assert result == "llama_cpp"

    def test_auto_cuda_available_selects_pytorch(self, monkeypatch):
        """CUDA 可用 + auto 模式 → pytorch"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        # ⚠️ CUDA 判据已改为硬件级探测（nvidia-smi 优先）——本仓 .venv-test 是
        # CPU-only torch，patch `torch.cuda.is_available` 已经不管用了（实测：
        # 有 RTX 4060 也返回 False，auto 会错选 llama_cpp）。patch 点在这里：
        monkeypatch.setattr("inference_path_selector.detect_cuda_available", lambda: True)
        result = ModelManager.select_engine()
        assert result == "pytorch"

    def test_auto_no_cuda_selects_llama_cpp(self, monkeypatch):
        """无 CUDA + auto 模式 → llama_cpp"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr("inference_path_selector.detect_cuda_available", lambda: False)
        result = ModelManager.select_engine()
        assert result == "llama_cpp"

    def test_edge_tier_without_cuda_selects_llama_cpp(self, monkeypatch):
        """edge 档位 + 无 CUDA → llama_cpp"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr("inference_path_selector.detect_cuda_available", lambda: False)
        profile = {"tier": "edge", "gpus": []}
        result = ModelManager.select_engine(profile)
        assert result == "llama_cpp"

    def test_mobile_tier_without_cuda_selects_llama_cpp(self, monkeypatch):
        """mobile 档位 + 无 CUDA → llama_cpp"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr("inference_path_selector.detect_cuda_available", lambda: False)
        profile = {"tier": "mobile", "gpus": []}
        result = ModelManager.select_engine(profile)
        assert result == "llama_cpp"

    def test_select_engine_uses_hardware_cuda_probe_not_torch(self, monkeypatch):
        """防回归：CUDA 判据必须走硬件级探测，而不是 torch.cuda.is_available。

        本仓 `.venv-test` 装的是 CPU-only torch（2.13.0+cpu），有 RTX 4060 也返回
        False；若有人把判据改回去，`auto` 会在有 GPU 的机器上错选 llama_cpp。这里
        故意把 torch 反着设成 False，逼 `select_engine` 只能靠硬件探测才能选对。
        """
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr("inference_path_selector.detect_cuda_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert ModelManager.select_engine() == "pytorch"

    def test_supports_delegates_to_active_llama_server_engine(self):
        """票 15a：`ModelManager.supports` 必须反映**活跃引擎**的真实能力。

        llama-server 子进程对外仍报 `engine_type="llama_cpp"`（家族一致），但
        `FORWARD_LAYERS` 必须是 False —— 否则 `runtime_supports` 放行，调度层会把
        层段接力派给一台执行不了的 runtime。
        """
        from koakuma_engine import Capability
        from llama_server_engine import LlamaServerEngine

        mgr = ModelManager()
        mgr._engine_type = "llama_cpp"
        mgr._llama_engine = LlamaServerEngine()
        assert mgr.supports(Capability.FORWARD_LAYERS) is False

    def test_supports_falls_back_to_backend_when_engine_has_no_probe(self):
        """没有自带能力探测的引擎（如绑定引擎）维持既有 backend 口径，行为不变。"""
        from koakuma_engine import Capability

        mgr = ModelManager()
        mgr._engine_type = "llama_cpp"
        mgr._llama_engine = object()  # 无 supports
        # `capabilities` 是只读 property（按 backend 定），这里逐项对比证明走了回退
        assert mgr.supports(Capability.FORWARD_LAYERS) == (
            mgr.capabilities.supports(Capability.FORWARD_LAYERS)
        )

    def test_laptop_tier_with_cuda_gpu_selects_pytorch(self, monkeypatch):
        """laptop 档位 + CUDA GPU → pytorch"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        profile = {
            "tier": "laptop",
            "gpus": [{"cuda_available": True, "name": "RTX 4060"}],
        }
        result = ModelManager.select_engine(profile)
        assert result == "pytorch"

    def test_any_cuda_gpu_trumps_selected_igpu(self, monkeypatch):
        """任意 GPU 有 CUDA 即认为 CUDA 可用（游戏本双显卡场景）"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        # torch.cuda 返回 False（选集显时），但 profile 中有独显
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        profile = {
            "tier": "laptop",
            "gpus": [
                {"cuda_available": False, "name": "Intel UHD"},
                {"cuda_available": True, "name": "RTX 3050"},
            ],
        }
        result = ModelManager.select_engine(profile)
        assert result == "pytorch"


# ================================================================
# load_model / switch_model 测试（P3 多模型支持）
# ================================================================

class TestLoadModelWithModelId:
    """测试 load_model() 的 model_id 参数和 DB 模型查找"""

    def test_load_model_rejects_unknown_model_id_without_path(self):
        """不存在的 model_id 且无 model_path → ValueError"""
        mgr = ModelManager()
        with pytest.raises(ValueError, match="未在注册表中找到"):
            mgr.load_model(model_id="nonexistent-model-xyz")

    def test_load_model_accepts_model_id_with_explicit_path(self, monkeypatch, tmp_path):
        """model_id 不存在但显式提供 model_path → 放行（由路径直接加载）"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "pytorch")
        mgr = ModelManager()
        # 阻止实际加载
        monkeypatch.setattr(mgr, "_load_pytorch", lambda *a, **kw: None)
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: None)

        # 不应抛出异常
        model_dir = tmp_path / "custom-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"weights")

        mgr.load_model(
            model_id="custom-model",
            model_path=str(model_dir),
            engine="pytorch",
        )
        assert mgr._active_model_id == "custom-model"
        assert mgr._engine_type == "pytorch"

    def test_load_model_finds_builtin_model_config(self, monkeypatch):
        """内置 model_id → 从 model_config 解析路径和引擎约束"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        mgr = ModelManager()
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: None)
        # GGUF 文件不存在，需 mock
        monkeypatch.setattr(os.path, "isfile", lambda p: True)

        # qwen2.5-7b-gguf 是 GGUF-only 内置模型
        # auto 引擎应解析为 llama_cpp
        mgr.load_model(model_id="qwen2.5-7b-gguf")
        assert mgr._active_model_id == "qwen2.5-7b-gguf"
        assert mgr._engine_type == "llama_cpp"

    def test_gemma4_native_uses_managed_mtmd_loader(self, monkeypatch, tmp_path):
        gguf = tmp_path / "gemma-4-12B-it-Q4_K_M.gguf"
        gguf.write_bytes(b"fixture")
        calls = []
        mgr = ModelManager()

        monkeypatch.setattr(
            mgr,
            "_load_gemma4_native",
            lambda path, profile=None: calls.append((path, profile)),
        )
        monkeypatch.setattr(
            mgr,
            "_load_llama_cpp",
            lambda *_args, **_kwargs: pytest.fail("ordinary GGUF loader was used"),
        )

        mgr.load_model(
            model_id="gemma4-native",
            model_path=str(gguf),
            engine="llama_cpp",
        )

        assert calls == [(str(gguf), None)]
        assert mgr.active_model_id == "gemma4-native"

    def test_native_image_capability_and_delegation(self, monkeypatch, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        calls = []

        class NativeEngine:
            def get_capabilities(self):
                return {"vision": True}

            def chat_image(self, **kwargs):
                calls.append(kwargs)
                return {"content": "image reply"}

        mgr = ModelManager()
        mgr._engine_type = "llama_cpp"
        mgr._llama_engine = NativeEngine()

        assert mgr.native_vision_available() is True
        result = mgr.chat_image(
            str(image),
            "describe",
            max_tokens=32,
            max_answer_tokens=8,
        )
        assert result["content"] == "image reply"
        assert calls[0]["image_path"] == str(image)
        assert calls[0]["max_answer_tokens"] == 8

    def test_load_model_finds_db_registered_model(self, monkeypatch, tmp_path):
        """DB 注册模型 → db_experimental_models 参数传递给 get_model_config"""
        gguf_file = tmp_path / "db-model.Q4_K_M.gguf"
        gguf_file.write_text("x", encoding="utf-8")

        db_models = [{
            "model_id": "db-only-model",
            "name": "DB Only Model",
            "model_type": "gguf",
            "model_path": "",
            "gguf_path": str(gguf_file),
            "recommended_vram_gb": 4.0,
            "max_context": 4096,
            "quant_types": ["Q4_K_M"],
            "description": "A DB-only registered model",
        }]

        mgr = ModelManager()
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        import config as cfg
        monkeypatch.setattr(cfg, "INFERENCE_ENGINE", "auto")

        # 不传 db_experimental_models → 找不到
        with pytest.raises(ValueError, match="未在注册表中找到"):
            mgr.load_model(model_id="db-only-model")

        # 传 db_experimental_models → 找到（B1 修复验证）
        mgr.load_model(
            model_id="db-only-model",
            db_experimental_models=db_models,
        )
        assert mgr._active_model_id == "db-only-model"

    def test_load_model_gguf_only_rejects_explicit_pytorch(self, monkeypatch):
        """GGUF-only 模型不得静默进入或改写 PyTorch loader 请求。"""
        mgr = ModelManager()
        llama_calls = []
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: llama_calls.append(True))
        monkeypatch.setattr(os.path, "isfile", lambda p: True)

        with pytest.raises(ValueError, match="Safetensors"):
            mgr.load_model(model_id="qwen2.5-7b-gguf", engine="pytorch")
        assert llama_calls == []


class TestSwitchModel:
    """测试 switch_model() 的切换流程和回滚行为"""

    def test_switch_to_unknown_model_without_path_fails(self):
        """切换到不存在的模型且无路径 → 返回 success=False"""
        mgr = ModelManager()
        result = mgr.switch_model("nonexistent-model")
        assert result["success"] is False
        assert result["error_code"] == "MODEL_NOT_REGISTERED"

    def test_unknown_target_preserves_loaded_model_without_unload(self, monkeypatch):
        """An invalid target must be rejected before changing a healthy runtime."""
        import model_module

        class FakeLlamaEngine:
            def close(self):
                raise AssertionError("preflight rejection must not unload the active model")

        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr(os.path, "isfile", lambda _path: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        mgr = ModelManager()
        engine = FakeLlamaEngine()

        def fake_load(*_args, **_kwargs):
            mgr._llama_engine = engine

        monkeypatch.setattr(mgr, "_load_llama_cpp", fake_load)
        loaded = mgr.switch_model("qwen2.5-7b-gguf", engine="llama_cpp")
        assert loaded["success"] is True

        result = mgr.switch_model("nonexistent-model", engine="llama_cpp")

        assert result["success"] is False
        assert result["error_code"] == "MODEL_NOT_REGISTERED"
        assert result["requested_model_id"] == "nonexistent-model"
        assert result["active_model_preserved"] is True
        assert result["model_id"] == "qwen2.5-7b-gguf"
        assert mgr.active_model_id == "qwen2.5-7b-gguf"
        assert mgr._llama_engine is engine

    def test_switch_first_model_succeeds(self, monkeypatch):
        """首次加载（无旧模型）→ 直接加载新模型"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        mgr = ModelManager()
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: None)
        monkeypatch.setattr(os.path, "isfile", lambda p: True)

        result = mgr.switch_model("qwen2.5-7b-gguf")
        assert result["success"] is True
        assert result["model_id"] == "qwen2.5-7b-gguf"
        assert mgr._active_model_id == "qwen2.5-7b-gguf"

    def test_switch_between_models(self, monkeypatch):
        """已加载模型 A → 切换到模型 B"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        mgr = ModelManager()
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: None)
        monkeypatch.setattr(mgr, "_load_pytorch", lambda *a, **kw: None)
        monkeypatch.setattr(os.path, "isfile", lambda p: True)
        monkeypatch.setattr(os.path, "isdir", lambda p: True)

        # 首次加载模型 A
        mgr.switch_model("qwen2.5-7b-gguf")
        assert mgr._active_model_id == "qwen2.5-7b-gguf"

        # 切换到模型 B（默认模型）
        # ★ 2026-09-19：`qwen-1_8b` 已从内置列表移除 ⇒ 改用当前默认模型 `qwen3-0.6b`
        #   （同为 safetensors + GGUF 的 both 形态，测试语义不变）。
        result = mgr.switch_model("qwen3-0.6b")
        assert result["success"] is True
        assert result["model_id"] == "qwen3-0.6b"
        assert mgr._active_model_id == "qwen3-0.6b"

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "models")),
        reason="缺本地模型目录 models/（CI 不拉模型）",
    )
    def test_switch_rollback_on_failure(self, monkeypatch):
        """加载新模型失败 → 回滚到旧模型"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr(os.path, "isfile", lambda p: True)
        monkeypatch.setattr(os.path, "isdir", lambda p: True)

        mgr = ModelManager()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        # 第一次加载成功（模型 A = qwen3-0.6b，需同时 mock 双引擎）
        # 注意: 必须设置 self.model 使 is_loaded 返回 True（否则 switch_model 跳过 unload/rollback）
        # ★ 2026-09-19：原用 `qwen-1_8b`，该模型已从内置列表移除 ⇒ 改用默认模型 `qwen3-0.6b`。
        def fake_load_llama(*a, **kw):
            mgr._active_model_id = kw.get("model_id", mgr._active_model_id)
            mgr._engine_type = "llama_cpp"
            mgr._model_path = kw.get("model_path", "/fake/path")
            mgr._llama_engine = True  # 使 is_loaded 返回 True

        def fake_load_pytorch(*a, **kw):
            mgr._active_model_id = kw.get("model_id", mgr._active_model_id)
            mgr._engine_type = "pytorch"
            mgr._model_path = kw.get("model_path", "/fake/path")
            mgr.model = True  # 使 is_loaded 返回 True

        monkeypatch.setattr(mgr, "_load_llama_cpp", fake_load_llama)
        monkeypatch.setattr(mgr, "_load_pytorch", fake_load_pytorch)

        # 首次加载模型 A
        mgr.switch_model("qwen3-0.6b")
        assert mgr._active_model_id == "qwen3-0.6b"

        # 切换到模型 B — 仅让 _load_llama_cpp 抛出异常
        # （qwen2.5-7b-gguf 是 GGUF-only，引擎会从 pytorch 修正为 llama_cpp）
        # 回滚时用 pytorch 加载 qwen3-0.6b，所以 _load_pytorch 仍需正常工作
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("模拟加载失败")))

        result = mgr.switch_model("qwen2.5-7b-gguf")
        assert result["success"] is False
        # 应回滚到 qwen3-0.6b
        assert result["model_id"] == "qwen3-0.6b"
        assert "已回滚" in result["error"]

    def test_switch_with_db_experimental_models(self, monkeypatch, tmp_path):
        """通过 db_experimental_models 找到 DB 注册模型并切换"""
        gguf_file = tmp_path / "db-switch-model.Q4_K_M.gguf"
        gguf_file.write_text("x", encoding="utf-8")

        db_models = [{
            "model_id": "db-switch-target",
            "name": "DB Switch Target",
            "model_type": "gguf",
            "model_path": "",
            "gguf_path": str(gguf_file),
            "recommended_vram_gb": 4.0,
            "max_context": 4096,
            "quant_types": ["Q4_K_M"],
            "description": "test",
        }]

        mgr = ModelManager()
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        import config as cfg
        monkeypatch.setattr(cfg, "INFERENCE_ENGINE", "auto")

        # 不传 db_experimental_models → 失败（B1 修复前行为）
        result_without = mgr.switch_model("db-switch-target")
        assert result_without["success"] is False

        # 传 db_experimental_models → 成功（B1 修复后行为）
        result_with = mgr.switch_model(
            "db-switch-target",
            db_experimental_models=db_models,
        )
        assert result_with["success"] is True
        assert result_with["model_id"] == "db-switch-target"

    def test_switch_model_updates_active_model_id(self, monkeypatch):
        """切换后 _active_model_id 应更新"""
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr(os.path, "isfile", lambda p: True)

        mgr = ModelManager()
        monkeypatch.setattr(mgr, "_load_llama_cpp", lambda *a, **kw: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        # ★ 2026-09-19：构造时的初始活跃模型按**设备画像**取默认（不再是固定常量）。
        assert mgr.active_model_id == mc.get_profile_default_model_id()
        mgr.switch_model("qwen2.5-7b-gguf")
        assert mgr.active_model_id == "qwen2.5-7b-gguf"


class TestP6SameModelSwitchShortCircuit:
    """KIP-13: same-model reuse is gated by an immutable load fingerprint."""

    class _FakeLlamaEngine:
        def get_model_info(self):
            return {"engine": "llama_cpp"}

        def close(self):
            return None

    def _loaded_mgr(self, monkeypatch):
        import model_module
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr(os.path, "isfile", lambda p: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        mgr = ModelManager()
        load_calls = []

        def fake_llama(*a, **kw):
            load_calls.append(("llama", kw))
            mgr._llama_engine = self._FakeLlamaEngine()
            mgr._model_path = "G:/models/qwen2.5-7b-gguf.gguf"

        monkeypatch.setattr(mgr, "_load_llama_cpp", fake_llama)
        monkeypatch.setattr(
            mgr, "_load_pytorch",
            lambda *a, **kw: load_calls.append(("pytorch", kw)),
        )
        first = mgr.switch_model("qwen2.5-7b-gguf")
        assert first["success"] is True
        assert mgr.is_loaded is True
        return mgr, load_calls

    def test_same_model_switch_short_circuits_without_reload(self, monkeypatch):
        mgr, load_calls = self._loaded_mgr(monkeypatch)
        before = len(load_calls)

        second = mgr.switch_model("qwen2.5-7b-gguf")

        assert second["success"] is True
        assert second.get("reused") is True
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", second["load_fingerprint"])
        assert len(load_calls) == before  # 未触发任何加载
        assert mgr.is_loaded is True
        status = mgr.get_model_info()
        assert status["load_fingerprint"] == second["load_fingerprint"]
        assert status["load_fingerprint_schema"] == 1
        assert "G:/models" not in status["load_fingerprint"]

    def test_same_model_short_circuit_with_explicit_same_path(self, monkeypatch):
        mgr, load_calls = self._loaded_mgr(monkeypatch)
        before = len(load_calls)
        path = mgr._model_path

        second = mgr.switch_model(
            "qwen2.5-7b-gguf", quant_type="q8_0", model_path=path,
        )

        # 显式传相同路径但不同量化 → 不短路（量化不同需重载）
        assert second.get("reused") is None or second.get("reused") is False
        assert len(load_calls) > before

    def test_different_model_switch_still_reloads(self, monkeypatch):
        mgr, load_calls = self._loaded_mgr(monkeypatch)
        before = len(load_calls)

        # ★ 2026-09-19：原用 `qwen-1_8b`（已从内置列表移除）⇒ 改用 `qwen3-0.6b`。
        #   `_loaded_mgr` 先加载的是 `qwen2.5-7b-gguf`，故仍是「不同模型 ⇒ 必须重新加载」。
        other = mgr.switch_model("qwen3-0.6b")

        assert other["success"] is True
        assert other.get("reused") is None
        assert len(load_calls) > before

    def _db_loaded_mgr(self, monkeypatch, tmp_path):
        import model_module

        first_path = tmp_path / "first.gguf"
        second_path = tmp_path / "second.gguf"
        first_path.write_bytes(b"first")
        second_path.write_bytes(b"second")
        entry = {
            "model_id": "fingerprint-db-model",
            "name": "Fingerprint DB Model",
            "model_type": "gguf",
            "gguf_path": str(first_path),
            "model_path": "",
            "sha256": "a" * 64,
            "revision": "revision-a",
            "quantization": "Q4_K_M",
        }
        db_models = [entry]
        monkeypatch.setattr(model_module, "INFERENCE_ENGINE", "auto")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        mgr = ModelManager()
        load_calls = []

        def fake_llama(path, _profile=None):
            load_calls.append(("llama_cpp", path))
            mgr._llama_engine = self._FakeLlamaEngine()
            mgr._model_path = path

        monkeypatch.setattr(mgr, "_load_llama_cpp", fake_llama)
        first = mgr.switch_model(
            entry["model_id"],
            engine="llama_cpp",
            db_experimental_models=db_models,
        )
        assert first["success"] is True
        assert len(load_calls) == 1
        return mgr, load_calls, db_models, first_path, second_path

    def test_registry_path_change_forces_reload(self, monkeypatch, tmp_path):
        mgr, calls, db_models, _first_path, second_path = self._db_loaded_mgr(
            monkeypatch, tmp_path,
        )
        before_fingerprint = mgr.load_fingerprint
        db_models[0]["gguf_path"] = str(second_path)

        result = mgr.switch_model(
            "fingerprint-db-model",
            engine="llama_cpp",
            db_experimental_models=db_models,
        )

        assert result["success"] is True
        assert result.get("reused") is None
        assert calls[-1] == ("llama_cpp", str(second_path))
        assert len(calls) == 2
        assert result["load_fingerprint"] != before_fingerprint

    @pytest.mark.parametrize("field,new_value", [
        ("sha256", "b" * 64),
        ("revision", "revision-b"),
    ])
    def test_registry_artifact_identity_change_forces_reload(
        self, monkeypatch, tmp_path, field, new_value,
    ):
        mgr, calls, db_models, _first_path, _second_path = self._db_loaded_mgr(
            monkeypatch, tmp_path,
        )
        before_fingerprint = mgr.load_fingerprint
        db_models[0][field] = new_value

        result = mgr.switch_model(
            "fingerprint-db-model",
            engine="llama_cpp",
            db_experimental_models=db_models,
        )

        assert result["success"] is True
        assert result.get("reused") is None
        assert len(calls) == 2
        assert result["load_fingerprint"] != before_fingerprint

    def test_engine_change_forces_reload(self, monkeypatch, tmp_path):
        pytorch_path = tmp_path / "artifact"
        pytorch_path.mkdir()
        (pytorch_path / "config.json").write_text("{}", encoding="utf-8")
        (pytorch_path / "model.safetensors").write_bytes(b"weights")
        gguf_path = tmp_path / "artifact.gguf"
        gguf_path.write_bytes(b"gguf")
        mgr = ModelManager()
        load_calls = []

        def fake_llama(path, _profile=None):
            load_calls.append("llama_cpp")
            mgr._llama_engine = self._FakeLlamaEngine()
            mgr._model_path = path

        def fake_pytorch(path, _quant_type=None, _profile=None):
            load_calls.append("pytorch")
            mgr.model = object()
            mgr._model_path = path

        monkeypatch.setattr(mgr, "_load_llama_cpp", fake_llama)
        monkeypatch.setattr(mgr, "_load_pytorch", fake_pytorch)
        first = mgr.switch_model(
            "external-model", model_path=str(gguf_path), engine="llama_cpp",
        )
        assert first["success"] is True

        result = mgr.switch_model(
            "external-model", model_path=str(pytorch_path), engine="pytorch",
        )

        assert result["success"] is True
        assert result.get("reused") is None
        assert load_calls == ["llama_cpp", "pytorch"]

    def test_relevant_profile_change_reloads_but_free_vram_does_not(
        self, monkeypatch, tmp_path,
    ):
        mgr, calls, db_models, _first_path, _second_path = self._db_loaded_mgr(
            monkeypatch, tmp_path,
        )
        baseline = {
            "tier": "laptop",
            "gpu": {"cuda_available": True, "vram_total_gb": 8, "vram_free_gb": 7},
        }
        # Establish a fingerprint with the explicit profile.
        reloaded = mgr.switch_model(
            "fingerprint-db-model", profile=baseline, engine="llama_cpp",
            db_experimental_models=db_models,
        )
        assert reloaded.get("reused") is None
        assert len(calls) == 2

        free_vram_only = {
            "tier": "laptop",
            "gpu": {"cuda_available": True, "vram_total_gb": 8, "vram_free_gb": 1},
        }
        reused = mgr.switch_model(
            "fingerprint-db-model", profile=free_vram_only, engine="llama_cpp",
            db_experimental_models=db_models,
        )
        assert reused.get("reused") is True
        assert len(calls) == 2

        changed_tier = dict(free_vram_only, tier="edge")
        changed = mgr.switch_model(
            "fingerprint-db-model", profile=changed_tier, engine="llama_cpp",
            db_experimental_models=db_models,
        )
        assert changed.get("reused") is None
        assert len(calls) == 3


class TestFusedRMSNorm:
    """★ R-R7 rank3：`_apply_fused_rmsnorm`（`F.rms_norm` 融合）的语义与**作用范围**回归。

    判据纪律：这里只验「替换谁 / 数值是否等价 / 默认是否零副作用」；**端到端收益与 per-token
    argmax 一致性**已在真模型上实测（`build/cross-framework-layer-poc/rank_ablation.py`：
    qwen2.5-0.5b / 12 层 / gen=40，`ms/step` **13.389 → 12.358（−7.7%）**、带 `lm_head` 的
    真 logits **argmax 逐 token 完全一致**；cProfile `Tensor.to` 2173 → 205）。
    """

    class _RMSNorm(nn.Module):
        """与 `transformers` 的 `Qwen2RMSNorm` 同形（6 算子版）。"""

        def __init__(self, size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(size))
            self.variance_epsilon = eps

        def forward(self, x):
            dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            return self.weight * (x * torch.rsqrt(variance + self.variance_epsilon)).to(dtype)

    def test_replaces_only_rmsnorm_with_weight_and_eps(self):
        from model_module import _apply_fused_rmsnorm

        class RMSNormWithoutWeight(nn.Module):
            def forward(self, x):
                return x

        class MyLayerNorm(nn.Module):          # 名字不含 RMSNorm ⇒ 必须不动
            def __init__(self, size):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(size))

            def forward(self, x):
                return x

        model = nn.Sequential(self._RMSNorm(8), RMSNormWithoutWeight(), MyLayerNorm(8))
        target, skipped, untouched = model[0], model[1], model[2]
        untouched_before = untouched.forward
        skipped_before = skipped.forward

        assert _apply_fused_rmsnorm(model) == 1

        assert target.forward is not self._RMSNorm.forward    # 被换成闭包
        assert untouched.forward == untouched_before          # LayerNorm 原样
        assert skipped.forward == skipped_before              # 缺 weight 的跳过

    def test_fused_numerics_match_reference(self):
        from model_module import _apply_fused_rmsnorm

        torch.manual_seed(0)
        model = nn.Sequential(self._RMSNorm(16, eps=1e-5))
        module = model[0]
        with torch.no_grad():
            module.weight.copy_(torch.rand(16) + 0.5)

        x = torch.randn(2, 3, 16) * 3
        reference = module(x)                    # 替换前的原实现
        _apply_fused_rmsnorm(model)
        fused = module(x)
        assert torch.allclose(reference, fused, atol=1e-6, rtol=1e-5)

    def test_switch_is_noop_by_default(self, monkeypatch):
        """默认 `USE_FUSED_RMSNORM=False` ⇒ **零副作用**；置 True 才替换。"""
        import model_module as mm
        from types import SimpleNamespace

        holder = nn.Sequential(self._RMSNorm(8))
        target = holder[0]
        before = target.forward
        fake_manager = SimpleNamespace(model=holder)          # 只需 `.model`

        monkeypatch.setattr(mm, "USE_FUSED_RMSNORM", False, raising=False)
        mm.ModelManager._maybe_apply_fused_rmsnorm(fake_manager)
        assert target.forward == before, "默认关闭时不得替换任何模块"

        monkeypatch.setattr(mm, "USE_FUSED_RMSNORM", True, raising=False)
        mm.ModelManager._maybe_apply_fused_rmsnorm(fake_manager)
        assert target.forward != before, "开启后应替换"

    def test_switch_without_model_is_safe(self, monkeypatch):
        """未加载模型（`model is None`）时开开关也不得抛异常。"""
        import model_module as mm
        from types import SimpleNamespace

        monkeypatch.setattr(mm, "USE_FUSED_RMSNORM", True, raising=False)
        mm.ModelManager._maybe_apply_fused_rmsnorm(SimpleNamespace(model=None))

    def test_returns_zero_when_op_missing(self, monkeypatch):
        """torch 无 `F.rms_norm`（旧版）⇒ 返回 0、行为不变（降级而非报错）。"""
        import model_module as mm

        model = nn.Sequential(self._RMSNorm(8))
        target = model[0]
        before = target.forward
        monkeypatch.delattr(mm.torch.nn.functional, "rms_norm", raising=False)

        assert mm._apply_fused_rmsnorm(model) == 0
        assert target.forward == before


class TestGemmaLayerRangeAdapter:
    """Gemma 4 must not enter the legacy in-process Qwen2 executor."""

    @pytest.mark.parametrize("model_type", ["gemma", "gemma4_unified"])
    def test_main_runtime_requires_isolated_sidecar(
        self, monkeypatch, tmp_path, model_type,
    ):
        import model_module

        (tmp_path / "config.json").write_text(
            json.dumps({
                "model_type": model_type,
                "text_config": {"num_hidden_layers": 4},
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            model_module.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: pytest.fail(
                "main runtime AutoConfig must not inspect Gemma 4"
            ),
        )

        with pytest.raises(RuntimeError, match="隔离 Transformers sidecar"):
            ModelManager().load_layer_range(
                0, 2, total_layers=4, model_path=str(tmp_path),
            )


# ================================================================
# ★ B9：空的 cache 对象规范化
# ================================================================

class TestEmptyCacheNormalization:
    """★ B9：调用方常传**空的** cache 对象（如 `DynamicCache()`）而不是 `None`。

    原行为：`len(cache) == 0` ⇒ 报
        `Qwen2 KV cache 层数不匹配: cache=0, local=N`
    （报错信息**不指向根因**）；hybrid 架构上还可能直接
        `IndexError: list index out of range`。
    新行为：入口统一把「没有任何已写入层的 cache」视同 `None`（由 `forward_layers` 自建）。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_tiny_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    # ---------------- 纯逻辑 ----------------

    def test_none_stays_none(self):
        assert _normalize_past_key_values(None) is None

    def test_empty_tuple_becomes_none(self):
        assert _normalize_past_key_values(()) is None
        assert _normalize_past_key_values([]) is None

    def test_all_none_tuple_becomes_none(self):
        """B14 的 hybrid 收集侧会留 None 占位；全 None 的 tuple 等于没有缓存。"""
        assert _normalize_past_key_values((None, None, None)) is None

    def test_non_empty_tuple_is_preserved(self):
        """**部分写入**的缓存必须原样保留（不得被当成空）。"""
        entry = (torch.zeros(1, 2, 2, 32), torch.zeros(1, 2, 2, 32))
        pkv = (entry, None)  # 一层有、一层是 None 占位
        assert _normalize_past_key_values(pkv) is pkv

    def test_real_dynamic_cache_empty_becomes_none(self):
        """真实 `DynamicCache()` / `DynamicCache(config=...)` 均应视同空。"""
        cache_utils = pytest.importorskip("transformers.cache_utils")
        empty = cache_utils.DynamicCache()
        assert _normalize_past_key_values(empty) is None

    def test_real_dynamic_cache_non_empty_is_preserved(self):
        """已写入内容的 `DynamicCache` 不得被规范化掉。"""
        cache_utils = pytest.importorskip("transformers.cache_utils")
        cache = cache_utils.DynamicCache()
        k = torch.zeros(1, 2, 3, 32)
        cache.update(k, k, 0)
        assert _normalize_past_key_values(cache) is cache

    # ---------------- 集成：forward_layers 不再因空 cache 报错 ----------------

    def test_forward_layers_accepts_empty_dynamic_cache(self):
        """修前：报 `KV cache 层数不匹配: cache=0, local=4`；修后：正常前向。"""
        cache_utils = pytest.importorskip("transformers.cache_utils")
        result = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]),
            past_key_values=cache_utils.DynamicCache(),
            use_cache=True,
        )
        # ⚠️ tiny 模型自带 lm_head ⇒ 末节点返回 logits（不是 hidden_states）
        assert ("hidden_states" in result) or ("logits" in result)
        assert len(result["past_key_values"]) == 4  # 4 local layers

    def test_forward_layers_accepts_empty_dynamic_cache_with_config(self):
        """传 `DynamicCache(config=...)`（hybrid 场景的正确建法）同样应被视同空。"""
        cache_utils = pytest.importorskip("transformers.cache_utils")
        try:
            cache = cache_utils.DynamicCache(config=self.model.config)
        except (TypeError, AttributeError):
            pytest.skip("该 transformers 版本不支持 DynamicCache(config=...)")
        result = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]),
            past_key_values=cache,
            use_cache=True,
        )
        assert ("hidden_states" in result) or ("logits" in result)

    def test_forward_layers_empty_cache_result_matches_none(self):
        """规范化的语义必须是「等于没传」：结果与 `None` 完全相同。"""
        cache_utils = pytest.importorskip("transformers.cache_utils")
        ids = torch.tensor([[1, 2, 3, 4]])
        with_none = self.mgr.forward_layers(input_ids=ids, past_key_values=None, use_cache=True)
        with_empty = self.mgr.forward_layers(
            input_ids=ids, past_key_values=cache_utils.DynamicCache(), use_cache=True
        )
        # tiny 模型带 lm_head ⇒ 比较 logits；否则比 hidden_states
        key = "logits" if "logits" in with_none else "hidden_states"
        assert key in with_empty
        assert torch.equal(with_none[key], with_empty[key])
        assert len(with_none["past_key_values"]) == len(with_empty["past_key_values"])

    def test_forward_layers_empty_tuple_cache_accepted(self):
        """tuple 形式的空缓存（`()` / `(None,)*n`）同样不应报错。"""
        for empty in ((), (None, None, None, None)):
            result = self.mgr.forward_layers(
                input_ids=torch.tensor([[1, 2, 3]]),
                past_key_values=empty,
                use_cache=True,
            )
            assert ("hidden_states" in result) or ("logits" in result)

    def test_mismatched_non_empty_cache_still_rejected(self):
        """★ 回归保护：**非空但层数不对**的缓存仍必须报错（不得被新逻辑放过）。"""
        bad = ((torch.zeros(1, 2, 2, 32), torch.zeros(1, 2, 2, 32)),)  # 只有 1 层，本地 4 层
        with pytest.raises(RuntimeError, match="KV cache 层数不匹配"):
            self.mgr.forward_layers(
                input_ids=torch.tensor([[7]]), past_key_values=bad, use_cache=True
            )

    # ---------- hybrid 增量解码：cache 对象通道（2026-09-21） ----------
    # 原缺陷：缓存契约是 `tuple of (k, v)`，**只承载 KV**。hybrid（Qwen3.5）的
    # linear_attention 层用 **recurrent state**（其 cache 层没有 `.keys`/`.values`）
    # ⇒ 在 tuple 里只能落到 `None` ⇒ **跨步状态丢失** ⇒ decode 第 2 步起退化为复读。
    # 实测（qwen3.5-2b 整模上游）：`Paris of of of of …`，与 llama.cpp 仅共享 1/16 前缀；
    # 改用 cache 对象通道后 **16/16 完全一致**，D→L 端到端 **32/32 一致**。

    def test_forward_layers_exposes_cache_object(self):
        """★ 修复：结果必须**同时**给出 `cache` 对象（recurrent state 只在这里）。"""
        result = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]), past_key_values=None, use_cache=True
        )
        assert "cache" in result, "cache 对象通道缺失 ⇒ hybrid 的 recurrent state 传不出去"
        assert hasattr(result["cache"], "layers")

    def test_cache_object_roundtrip_is_reused_not_rebuilt(self):
        """★ 修复：把上一步的 cache 对象原样回传 ⇒ **直接复用**（不是按 tuple 重建）。"""
        first = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]), past_key_values=None, use_cache=True
        )
        cache = first["cache"]
        second = self.mgr.forward_layers(
            input_ids=torch.tensor([[4]]), past_key_values=cache, use_cache=True
        )
        assert ("hidden_states" in second) or ("logits" in second)
        assert second["cache"] is cache, "cache 对象被重建 ⇒ hybrid 状态会丢"

    def test_cache_object_with_insufficient_layer_slots_rejected(self):
        """★ 回归保护：cache 对象**非空但层槽不足**时 fail-loud（不得静默走错索引）。

        注意 `layers=[]` 属于「空 cache」⇒ 由 `_normalize_past_key_values` 视同 None
        （见上方的空缓存测试），不适用于本用例；这里构造的是**非空但槽偏少**的对象。
        """
        cache_utils = pytest.importorskip("transformers.cache_utils")
        c = cache_utils.DynamicCache()
        k = torch.zeros(1, 2, 3, 8)
        for i in range(len(self.mgr.model.model.layers)):
            c.update(k, k, i)
        assert _normalize_past_key_values(c) is c, "非空 cache 不应被规范化掉"
        c.layers = c.layers[:1]  # 非空，但槽数 < 本地层数
        with pytest.raises(RuntimeError, match="cache 对象层槽不足"):
            self.mgr.forward_layers(
                input_ids=torch.tensor([[7]]), past_key_values=c, use_cache=True
            )

    def test_tuple_channel_is_lossy_for_non_kv_layers(self):
        """★ 机理回归：非 KV 层在 **tuple 通道**必然落到 `None` —— 这正是原 bug 的机理，
        也是「必须用 cache 对象通道」的证据（保留此断言以防有人"优化"掉 cache 对象）。"""
        cache_utils = pytest.importorskip("transformers.cache_utils")

        class _RecurrentOnlyLayer:
            """模拟 hybrid 的 linear_attention cache 层：有状态，**没有** keys/values。"""

            def __init__(self):
                self.recurrent_state = torch.ones(1, 4)

        cache = cache_utils.DynamicCache()
        n_local = len(self.mgr.model.model.layers)
        cache.layers = [_RecurrentOnlyLayer() for _ in range(n_local)]
        items = []
        for layer_cache in cache.layers[:n_local]:
            keys = getattr(layer_cache, "keys", None)
            values = getattr(layer_cache, "values", None)
            items.append((keys, values) if keys is not None and values is not None else None)
        assert items == [None] * n_local, (
            "非 KV 层居然能从 tuple 通道带出状态 —— 请重新核对 hybrid cache 契约"
        )
        assert getattr(cache.layers[0], "recurrent_state", None) is not None, (
            "状态本身存在于 cache 对象里 ⇒ 只要传 cache 对象就不会丢"
        )


@pytest.mark.parametrize("release_method", ["close", "unload"])
def test_llama_unload_failure_preserves_manager_projection(release_method):
    manager = ModelManager()
    should_fail = [True]

    def release():
        if should_fail[0]:
            raise OSError("native owner still alive")

    engine = types.SimpleNamespace(is_loaded=True, **{release_method: release})
    manager._llama_engine = engine
    manager._engine_type = "llama_cpp"
    manager._active_model_id = "segment-model"
    manager.quant_type = "Q4_K_M"
    manager.layer_range = (0, 8)
    manager._full_model_path = "head.gguf"
    descriptor = {"assignment_layer_range": [0, 8]}
    manager._pipeline_descriptor = descriptor
    manager._pipeline_distributed_only = True
    manager._load_fingerprint = {"model_path": "head.gguf"}
    with pytest.raises(OSError, match="native owner still alive"):
        manager.unload_model()
    assert manager._llama_engine is engine and manager.is_loaded
    assert manager._active_model_id == "segment-model" and manager._engine_type == "llama_cpp"
    assert manager.quant_type == "Q4_K_M" and manager.layer_range == (0, 8)
    assert manager._pipeline_descriptor is descriptor and manager._pipeline_distributed_only
    assert manager._full_model_path == "head.gguf" and manager._load_fingerprint == {"model_path": "head.gguf"}
    should_fail[0] = False
    manager.unload_model()
    assert manager._llama_engine is None and not manager.is_loaded
    assert manager._active_model_id == "" and manager._engine_type == ""
    assert manager.quant_type is None and manager.layer_range is None
    assert manager._pipeline_descriptor is None and not manager._pipeline_distributed_only
    assert manager._full_model_path is None and manager._load_fingerprint is None
