import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import model_config as mc

# 默认模型的本地工件（本机 models 是 Junction；CI 不拉模型）。
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def test_deepseek_r1_distill_qwen_slots_are_registered():
    expected = {
        "deepseek-r1-distill-qwen-1.5b": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "models/deepseek-r1-distill-qwen-1.5b",
            4.0,
            "",
            "safetensors",
        ),
        "deepseek-r1-distill-qwen-7b": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "models/deepseek-r1-distill-qwen-7b",
            8.0,
            "models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
            "both",
        ),
        "deepseek-r1-distill-qwen-14b": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "models/deepseek-r1-distill-qwen-14b",
            16.0,
            "",
            "safetensors",
        ),
        "deepseek-r1-distill-qwen-32b": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "models/deepseek-r1-distill-qwen-32b",
            24.0,
            "",
            "safetensors",
        ),
    }

    for model_id, (hf_id, local_suffix, min_vram, gguf_suffix, model_type) in expected.items():
        cfg = mc.get_builtin_model(model_id)

        assert cfg is not None
        assert cfg.name.startswith("DeepSeek-R1-Distill-Qwen")
        assert cfg.model_type == model_type
        assert cfg.is_experimental is True
        assert cfg.location == "external"
        assert cfg.huggingface_id == hf_id
        assert os.path.normpath(cfg.model_path).endswith(os.path.normpath(local_suffix))
        if gguf_suffix:
            assert os.path.normpath(cfg.gguf_path).endswith(os.path.normpath(gguf_suffix))
        else:
            assert cfg.gguf_path == ""
        assert cfg.recommended_vram_gb == min_vram
        assert cfg.max_context == 32768
        assert "int4" in cfg.quant_types


def test_new_small_and_distilqwen_slots_are_registered():
    # ★ 2026-09-19：`qwen-1_8b` 退役后 `qwen3-0.6b` 成为**默认**模型 ⇒
    #   它不再是 experimental（否则 `test_experimental_models_are_hidden_without_cuda`
    #   会在无 CUDA 时把它隐藏，导致「默认模型不可见」）。这里把它从「全是实验」的
    #   集合中移出，单独断言。
    expected = {
        "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "models/qwen2.5-0.5b-instruct", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", 1.5, 32768),
        "minicpm4-0.5b": ("openbmb/MiniCPM4-0.5B", "models/minicpm4-0.5b", "models/minicpm4-0.5b-q4_k_m.gguf", 1.5, 32768),
        "distilqwen25-ds3-0324-7b": ("alibaba-pai/DistilQwen2.5-DS3-0324-7B", "models/distilqwen25-ds3-0324-7b", "models/distilqwen25-ds3-0324-7b-q4_k_m.gguf", 8.0, 32768),
    }
    for model_id, (hf_id, model_suffix, gguf_suffix, vram, context) in expected.items():
        cfg = mc.get_builtin_model(model_id)
        assert cfg is not None
        assert cfg.model_type == "both"
        assert cfg.huggingface_id == hf_id
        assert cfg.recommended_vram_gb == vram
        assert cfg.max_context == context
        assert cfg.is_experimental is True
        assert os.path.normpath(cfg.model_path).endswith(os.path.normpath(model_suffix))
        assert os.path.normpath(cfg.gguf_path).endswith(os.path.normpath(gguf_suffix))

    # ★ qwen3-0.6b：现在是**默认**模型 ⇒ 必须**不是** experimental（见函数头注释）
    default_cfg = mc.get_builtin_model("qwen3-0.6b")
    assert default_cfg is not None
    assert default_cfg.model_type == "both"
    assert default_cfg.huggingface_id == "Qwen/Qwen3-0.6B"
    assert default_cfg.recommended_vram_gb == 2.0
    assert default_cfg.max_context == 40960
    assert default_cfg.is_experimental is False, "它已是默认门面模型，不应被标为实验"
    assert mc.DEFAULT_MODEL_ID == "qwen3-0.6b"


def test_experimental_models_are_hidden_without_cuda(monkeypatch):
    monkeypatch.setattr(mc, "is_cuda_available", lambda: False)

    visible_ids = {model.model_id for model in mc.get_visible_models()}

    assert mc.DEFAULT_MODEL_ID in visible_ids
    assert "deepseek-r1-distill-qwen-1.5b" not in visible_ids
    assert "deepseek-r1-distill-qwen-7b" not in visible_ids
    assert "deepseek-r1-distill-qwen-14b" not in visible_ids
    assert "deepseek-r1-distill-qwen-32b" not in visible_ids


def test_deepseek_r1_distill_slots_are_visible_with_cuda(monkeypatch):
    monkeypatch.setattr(mc, "is_cuda_available", lambda: True)

    visible_ids = {model.model_id for model in mc.get_visible_models()}

    assert "deepseek-r1-distill-qwen-1.5b" in visible_ids
    assert "deepseek-r1-distill-qwen-7b" in visible_ids
    assert "deepseek-r1-distill-qwen-14b" in visible_ids
    assert "deepseek-r1-distill-qwen-32b" in visible_ids


def test_model_file_status_marks_available_formats(tmp_path):
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("x", encoding="utf-8")
    gguf_file = tmp_path / "model.Q4_K_M.gguf"
    gguf_file.write_text("x", encoding="utf-8")

    cfg = mc.ModelConfig(
        model_id="test-model",
        name="Test Model",
        model_type="both",
        model_path=str(model_dir),
        gguf_path=str(gguf_file),
    )

    status = mc.get_model_file_status(cfg)

    assert status["is_available"] is True
    assert status["has_safetensors"] is True
    assert status["has_gguf"] is True
    assert status["available_formats"] == ["safetensors", "gguf"]


# ================================================================
# 默认模型按设备画像选择（2026-09-19 用户裁定）
# ================================================================

def test_default_model_is_chosen_by_device_tier():
    """★ 默认模型按设备画像分档：**边缘（含轻薄本/集显）<1B，PC ~2B**。

    选型依据（实测，见 `mc.DEFAULT_MODEL_BY_TIER` 注释）：2B int4 的 VRAM 峰值为
    **2.24 GB** ⇒ `ULTRABOOK`（≤2 GB 共享显存）放不下，必须退到 <1B —— 这是硬约束。
    """
    pc_tiers = ("workstation", "laptop")
    small_tiers = ("ultrabook", "edge", "mobile")

    for tier in pc_tiers:
        assert mc.get_default_model_id(tier) == "qwen3-5-2b", f"{tier} 应取 2B 档"

    for tier in small_tiers:
        model_id = mc.get_default_model_id(tier)
        model = mc.get_builtin_model(model_id)
        assert model is not None, f"{tier} 的默认模型 {model_id} 必须在内置注册表中"
        # <1B：以参数规模语义衡量 —— 用推荐显存要求佐证（0.5B/0.6B 均 ≤2.0 GB）
        assert model.recommended_vram_gb <= 2.0, f"{tier} 应取 <1B 小模型，实得 {model_id}"


def test_default_model_falls_back_for_unknown_tier():
    """未知 / 缺失画像不得抛异常，一律退回兜底常量（否则会卡死无人值守自动加载）。"""
    for bogus in (None, "", "bogus-tier", 123, object()):
        assert mc.get_default_model_id(bogus) == mc.DEFAULT_MODEL_ID


def test_default_model_id_accepts_enum_and_str():
    """既接受 `DeviceTier` 枚举，也接受其字符串值（避免调用方强耦合）。"""
    import device_profiler

    assert mc.get_default_model_id(device_profiler.DeviceTier.WORKSTATION) == "qwen3-5-2b"
    assert mc.get_default_model_id("workstation") == "qwen3-5-2b"


@pytest.mark.skipif(not _MODELS_DIR.is_dir(), reason="缺本地模型目录 models/（CI 不拉模型）")
def test_profile_default_paths_resolve_to_existing_files():
    """`get_profile_default_model_paths()` 解析出的路径必须真实存在（否则自动加载必失败）。"""
    paths = mc.get_profile_default_model_paths()
    assert paths["model_id"], "必须给出 model_id"
    assert mc.get_builtin_model(paths["model_id"]) is not None
    # 至少一种形态的文件存在
    has_dir = bool(paths["model_path"]) and os.path.isdir(paths["model_path"])
    has_gguf = bool(paths["gguf_path"]) and os.path.isfile(paths["gguf_path"])
    assert has_dir or has_gguf, f"默认模型的路径都不存在：{paths}"


def test_registered_builtin_override_is_shared_by_catalog_and_execution(monkeypatch, tmp_path):
    builtin = mc.get_builtin_model("qwen3-0.6b")
    row = {"model_id": builtin.model_id, "model_type": "gguf", "model_path": "",
           "gguf_path": str(tmp_path / "updated.gguf"), "name": "User verified source"}
    monkeypatch.setattr(mc, "is_cuda_available", lambda: True)
    resolved = mc.get_model_config(builtin.model_id, [row])
    catalog = next(m for m in mc.get_all_model_configs([row]) if m.model_id == builtin.model_id)
    visible = next(m for m in mc.get_visible_models([row]) if m.model_id == builtin.model_id)
    assert resolved == catalog == visible
    assert resolved.name == row["name"]
    assert resolved.gguf_path == row["gguf_path"] and resolved.model_path == ""
    assert resolved.max_context == builtin.max_context
    assert resolved.is_experimental == builtin.is_experimental


def test_invalid_registered_builtin_never_silently_uses_old_artifact():
    mid = "qwen3-0.6b"
    row = {"model_id": mid, "model_path": ["invalid"]}
    assert mc.get_model_config(mid, [row]) is None
    assert mid not in {m.model_id for m in mc.get_all_model_configs([row])}
