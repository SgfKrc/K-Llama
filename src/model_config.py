"""
多模型配置管理 -- ModelConfig dataclass + 内置模型注册表
========================================================

P3: 多模型实验支持。提供:
  - ModelConfig dataclass: 模型元数据（路径/显存/量化/上下文等）
  - 内置模型列表（默认 Qwen-1.8B + 已知实验模型）
  - 与 DB cluster_config 中用户注册模型合并的统一查询接口

门控: 实验模型仅在 CUDA 可用时暴露（server-side torch.cuda.is_available()）。
      前端通过 hasDedicatedGpu (device profile) 门控。

依赖: config.py (_get_app_root, MODEL_NAME, MODEL_PATH, GGUF_MODEL_PATH)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---- 复用 config.py 的路径工具（避免循环导入） ----


def _get_app_root() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_APP_ROOT = _get_app_root()


def _get_bundled_asset_root() -> str:
    """Return PyInstaller's data root while preserving user-model paths."""
    if getattr(sys, "frozen", False):
        return str(getattr(sys, "_MEIPASS", _APP_ROOT))
    return _APP_ROOT


_BUNDLED_ASSET_ROOT = _get_bundled_asset_root()


# Shared, path-free model-fleet metadata.  Artifact digests stay unset until
# the downloaded files are verified by the manifest pipeline.
MODEL_PROFILE_METADATA: dict[str, dict] = {
    "qwen2.5-0.5b": {
        "revision": "builtin-qwen25-0.5b-v2",
        "template": "qwen_chat_v1",
        "thinking": "unknown",
        "vision": "unknown",
        "roles": ("answer", "summarizer"),
        "resources": {"min_ram_gb": 3.0, "min_vram_gb": 1.5, "min_disk_gb": 2.0, "edge_compatible": True},
        "evidence": {"probe_ticket": "M-SM-B1", "artifact_digest_mode": "missing"},
    },
    "qwen25-05b-f16": {
        "revision": "contrast-qwen25-0.5b-f16-v1",
        "template": "qwen_chat_v1",
        "thinking": "unknown",
        "vision": "unknown",
        "roles": ("answer", "summarizer"),
        "resources": {"min_ram_gb": 3.0, "min_vram_gb": 1.5, "min_disk_gb": 2.0, "edge_compatible": True},
        "evidence": {"artifact_digest_mode": "missing"},
    },
    "qwen3-0.6b": {
        "revision": "builtin-qwen3-0.6b-v2",
        "template": "qwen3_chat_v1",
        "thinking": "declared",
        "vision": "unknown",
        "roles": ("tool_router", "summarizer"),
        "resources": {"min_ram_gb": 4.0, "min_vram_gb": 2.0, "min_disk_gb": 3.0, "edge_compatible": True},
        "evidence": {"probe_ticket": "M-SM-B1", "artifact_digest_mode": "missing"},
    },
    "minicpm4-0.5b": {
        "revision": "builtin-minicpm4-0.5b-v2",
        "template": "minicpm4_chat_v1",
        "thinking": "unknown",
        "vision": "unknown",
        "roles": ("answer", "summarizer"),
        "resources": {"min_ram_gb": 3.0, "min_vram_gb": 1.5, "min_disk_gb": 3.0, "edge_compatible": True},
        "evidence": {"probe_ticket": "M-SM-B1", "artifact_digest_mode": "missing"},
    },
    "distilqwen25-ds3-0324-7b": {
        "revision": "builtin-distilqwen-ds3-0324-v2",
        "template": "qwen_chat_v1",
        "thinking": "unknown",
        "vision": "unknown",
        "roles": ("answer", "summarizer"),
        "resources": {"min_ram_gb": 12.0, "min_vram_gb": 8.0, "min_disk_gb": 20.0, "edge_compatible": False},
        "evidence": {"probe_ticket": "DSW-D1", "artifact_digest_mode": "missing"},
    },
}


def get_model_profile_metadata(model_id: str) -> dict:
    """Return a defensive copy of path-free fleet metadata for one model."""
    value = MODEL_PROFILE_METADATA.get(model_id, {})
    return {
        **value,
        "roles": tuple(value.get("roles", ("answer",))),
        "resources": dict(value.get("resources", {})),
        "evidence": dict(value.get("evidence", {})),
    }


def resolve_model_path(path: str) -> str:
    """Return an absolute path for a model file or directory."""
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(_APP_ROOT, path)


# ================================================================
# ModelConfig
# ================================================================

@dataclass
class ModelConfig:
    """单个模型的完整配置元数据。

    字段:
        model_id: 唯一标识符，如 "qwen-1_8b"。用于 API/DB 查找。
        name: 用户可见的显示名称，如 "Qwen-1.8B-Chat"。
        model_path: safetensors 格式模型目录的本地路径（相对 APP_ROOT 或绝对路径）。
        gguf_path: GGUF 文件路径（可选，仅 GGUF/双格式模型需要）。
        model_type: "safetensors" | "gguf" | "both"。
        recommended_vram_gb: 推荐的最低显存（GB），用于 UI 展示。
        max_context: 该模型支持的最大上下文长度。
        is_experimental: True = 用户下载的重模型，False = 内置默认模型。
        huggingface_id: HuggingFace 仓库 ID（下载参考）。
        quant_types: 支持的量化类型列表，如 ["fp16","int8","int4"]。
        description: 单行中文说明，用于 UI tooltip。
        location: "bundled"（内置）| "external"（用户下载）。
    """
    model_id: str
    name: str
    model_type: str = "safetensors"          # "safetensors" | "gguf" | "both"
    model_path: str = ""
    gguf_path: str = ""
    recommended_vram_gb: float = 3.5
    max_context: int = 4096
    is_experimental: bool = False
    huggingface_id: str = ""
    quant_types: list = field(default_factory=lambda: ["fp16", "int8", "int4"])
    description: str = ""
    location: str = "bundled"                # "bundled" | "external"


# ================================================================
# 内置模型注册表
# ================================================================

# ★ 2026-09-19：默认模型由 `qwen-1_8b` 改为按**设备画像**选择。
#   原因：Qwen-1.8B 的 remote code（2023 年老架构）与 transformers 5.x **链式不兼容**
#   （uint8 权重被 `_init_weights` 覆盖 ⇒ 崩溃；缺 `generation_config`；`generate` 依赖
#   已被移除的 `super().generate`；5.x 生成栈又依赖更多新属性）—— 逐个补属性是无底洞。
#   同体积新模型已有替代（`qwen3-0.6b` / `qwen2.5-0.5b`）。
#   ⚠️ 2026-09-19 用户裁定：**从内置列表中移除该条目**（此前只标「已退役」，
#   但仍出现在列表第一位，导致 UI 误选/误加载成 1.8B）。模型文件保留在 `models/`
#   与其 GGUF 路径不变，只是不再作为可选项暴露。
#
# ★ 2026-09-19（用户新裁定）：默认模型**按设备画像分档** ——
#   **边缘设备（含轻薄本/集显）用 <1B 小模型，PC 用 ~2B（贴合原 1.8B 体量）**。
#   实现见 `DEFAULT_MODEL_BY_TIER` + `get_default_model_id()`。
DEFAULT_MODEL_ID = "qwen3-0.6b"          # 兜底（无画像信息时使用；= MOBILE/EDGE 档）

#: 按设备画像分档的默认模型。键取 `device_profiler.DeviceTier` 的**字符串值**
#: （刻意不 import `DeviceTier`，避免 `model_config` ↔ `device_profiler` 反向依赖）。
#:
#: ## 分档依据（2026-09-19 实测，主仓引擎 + int4 + 3 轮重复）
#: 2B 档实测 **VRAM 峰值 2.24 GB**（`qwen3-5-2b` 与 `qwenseek-2b` 同为 2.24 GB）⇒
#: `ULTRABOOK`（≤2 GB **共享**显存）**放不下**，必须退到 <1B ⇒ 这是**硬约束**而非偏好。
#:
#: ## PC 档为何选 `qwen3-5-2b`
#: 与 `qwenseek-2b` 的单变量对照（同 prompt / 同贪心 / 同 int4，唯一变量=模型）：
#:
#: ======================  ================  ==================
#: 指标                    qwen3-5-2b        qwenseek-2b
#: ======================  ================  ==================
#: 加载                    9.36 s            12.38 s
#: 稳态吞吐                18.19 ± 1.49      17.57 ± 1.93 tok/s
#: 轮内波动                ±0.41 ~ 0.72      ±1.74 ~ 2.39
#: VRAM 峰值               2.24 GB           2.24 GB
#: 完整性元数据            `.qlh-model-asset.json` 有    无
#: ======================  ================  ==================
#:
#: ① **速度差仅 3.5%，落在 std 内 ⇒ 无显著差异**（不构成选型依据）；
#: ② `qwen3-5-2b` **加载快 24%** 且**轮内波动小 2~3×**（更稳定）；
#: ③ 质量（人工对照）：`qwen3-5-2b` 输出更完整（改写题给多方案 + Markdown，复杂度题带公式），
#:    `qwenseek-2b` 更简洁（改写题只给「这个方案尚可。」）—— **各有取向**，非一边倒；
#: ④ `qwenseek-2b` **缺 `.qlh-model-asset.json`**（来源/SHA 完整性未受管）⇒ 更适合实验而非默认。
#: ⇒ 综合取 `qwen3-5-2b`。**若后续要更短的输出风格，可换 `qwenseek-2b`**（此处留可追溯的对照数据）。
DEFAULT_MODEL_BY_TIER: dict = {
    "workstation": "qwen3-5-2b",   # 桌面工作站：2B 档，显存充裕
    "laptop": "qwen3-5-2b",        # 游戏本 / 独显本：2B 档（实测 2.24 GB，8 GB 卡余量充足）
    "ultrabook": "qwen3-0.6b",     # 轻薄本 / 集显：≤2 GB 共享显存 ⇒ 2B 放不下 ⇒ <1B
    "edge": "qwen3-0.6b",          # 边缘设备（树莓派 / Jetson / 旧笔记本）：<1B
    "mobile": "qwen2.5-0.5b",      # 移动端：最小可用
}


def get_default_model_id(tier: str = None) -> str:
    """按设备画像档位返回默认模型 id。

    Args:
        tier: `DeviceTier` 的字符串值（``workstation`` / ``laptop`` / ``ultrabook`` /
            ``edge`` / ``mobile``），或 `DeviceTier` 枚举本身；`None` / 未知值 ⇒ 返回兜底
            `DEFAULT_MODEL_ID`。

    ⚠️ 分档表里的 id 若在内置注册表中缺失，会**回退到兜底值**（不让画像配置把系统卡死）。
    """
    key = getattr(tier, "value", tier)
    if not key:
        return DEFAULT_MODEL_ID
    candidate = DEFAULT_MODEL_BY_TIER.get(str(key))
    if not candidate or get_builtin_model(candidate) is None:
        return DEFAULT_MODEL_ID
    return candidate

BUILTIN_MODELS: list[ModelConfig] = [
    ModelConfig(
        model_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        model_type="safetensors",
        model_path=os.path.join(_APP_ROOT, "models", "qwen2.5-7b-instruct"),
        gguf_path="",
        recommended_vram_gb=16.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        quant_types=["fp16", "int8", "int4"],
        description="7B 参数指令模型。INT4 量化约需 8GB VRAM，FP16 需 16GB+。需手动下载。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen2.5-14b",
        name="Qwen2.5-14B-Instruct",
        model_type="safetensors",
        model_path=os.path.join(_APP_ROOT, "models", "qwen2.5-14b-instruct"),
        gguf_path="",
        recommended_vram_gb=32.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="Qwen/Qwen2.5-14B-Instruct",
        quant_types=["int8", "int4"],
        description="14B 参数指令模型。INT4 量化约需 12GB VRAM。仅限高端 GPU。需手动下载。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen2.5-7b-gguf",
        name="Qwen2.5-7B-Instruct (GGUF)",
        model_type="gguf",
        model_path="",
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen2.5-7b-instruct-Q4_K_M.gguf"),
        recommended_vram_gb=6.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        quant_types=["Q4_K_M", "Q5_K_M", "Q8_0"],
        description="7B 参数 GGUF 格式。Q4_K_M ~4.7GB，CPU 可运行（速度较慢）。需手动下载。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen2.5-0.5b",
        name="Qwen2.5-0.5B-Instruct",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "qwen2.5-0.5b-instruct"),
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        recommended_vram_gb=1.5,
        max_context=32768,
        is_experimental=True,
        huggingface_id="Qwen/Qwen2.5-0.5B-Instruct",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="亚 1B 轻量 Qwen2.5 基线。用于链路、模板和上下文策略对照，不承诺回答正确率。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen25-05b-f16",
        name="Qwen2.5-0.5B-Instruct (F16 GGUF)",
        model_type="gguf",
        model_path="",
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen25-05b-f16.gguf"),
        recommended_vram_gb=1.5,
        max_context=32768,
        is_experimental=True,
        huggingface_id="Qwen/Qwen2.5-0.5B-Instruct",
        quant_types=["fp16"],
        description=(
            "Qwen2.5-0.5B 的 F16 GGUF 整模。用于跨框架分段执行的对照基准 —— "
            "与设备侧（Y700）持有的同一份工件，便于 L→L 与 D→L 两条路径对拍。"
        ),
        location="external",
    ),
    ModelConfig(
        model_id="qwen3-0.6b",
        name="Qwen3-0.6B",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "qwen3-0.6b"),
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen3-0.6b-q8_0.gguf"),
        recommended_vram_gb=2.0,
        max_context=40960,
        # ★ 2026-09-19：`qwen-1_8b` 退役后本模型成为**默认**，故不再是 experimental
        #   （`test_experimental_models_are_hidden_without_cuda` 要求默认模型在无 CUDA 时也可见）。
        is_experimental=False,
        huggingface_id="Qwen/Qwen3-0.6B",
        quant_types=["fp16", "int8", "int4", "Q8_0"],
        description="亚 1B Qwen3 对照模型。模板声明 enable_thinking 开关，需经 sidecar 和真实推理门。",
        location="external",
    ),
    ModelConfig(
        model_id="minicpm4-0.5b",
        name="MiniCPM4-0.5B",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "minicpm4-0.5b"),
        gguf_path=os.path.join(_APP_ROOT, "models", "minicpm4-0.5b-q4_k_m.gguf"),
        recommended_vram_gb=1.5,
        max_context=32768,
        is_experimental=True,
        huggingface_id="openbmb/MiniCPM4-0.5B",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="亚 1B MiniCPM4 新架构探针。优先验证 llama.cpp GGUF 架构、模板和停止词兼容性。",
        location="external",
    ),
    ModelConfig(
        model_id="distilqwen25-ds3-0324-7b",
        name="DistilQwen2.5-DS3-0324-7B",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "distilqwen25-ds3-0324-7b"),
        gguf_path=os.path.join(_APP_ROOT, "models", "distilqwen25-ds3-0324-7b-q4_k_m.gguf"),
        recommended_vram_gb=8.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="alibaba-pai/DistilQwen2.5-DS3-0324-7B",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="Qwen2.5 架构快思考替代候选。先完成 GGUF、模板和预算内收敛验证，保持 candidate。",
        location="external",
    ),
    ModelConfig(
        model_id="deepseek-r1-distill-qwen-1.5b",
        name="DeepSeek-R1-Distill-Qwen-1.5B",
        model_type="safetensors",
        model_path=os.path.join(_APP_ROOT, "models", "deepseek-r1-distill-qwen-1.5b"),
        gguf_path="",
        recommended_vram_gb=4.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        quant_types=["fp16", "int8", "int4"],
        description="R1 蒸馏推理模型，基于 Qwen2.5 Dense 架构。低门槛兼容性验证槽位，需手动下载 Safetensors。",
        location="external",
    ),
    ModelConfig(
        model_id="deepseek-r1-distill-qwen-7b",
        name="DeepSeek-R1-Distill-Qwen-7B",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "deepseek-r1-distill-qwen-7b"),
        gguf_path=os.path.join(_APP_ROOT, "models", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"),
        recommended_vram_gb=8.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="R1 蒸馏推理模型，基于 Qwen2.5 Dense 架构。支持 Safetensors 与 Q4_K_M GGUF 试水。",
        location="external",
    ),
    ModelConfig(
        model_id="deepseek-r1-distill-qwen-14b",
        name="DeepSeek-R1-Distill-Qwen-14B",
        model_type="safetensors",
        model_path=os.path.join(_APP_ROOT, "models", "deepseek-r1-distill-qwen-14b"),
        gguf_path="",
        recommended_vram_gb=16.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        quant_types=["fp16", "int8", "int4"],
        description="R1 蒸馏推理模型，基于 Qwen2.5 Dense 架构。INT4 建议 16GB+ VRAM，需手动下载 Safetensors。",
        location="external",
    ),
    ModelConfig(
        model_id="deepseek-r1-distill-qwen-32b",
        name="DeepSeek-R1-Distill-Qwen-32B",
        model_type="safetensors",
        model_path=os.path.join(_APP_ROOT, "models", "deepseek-r1-distill-qwen-32b"),
        gguf_path="",
        recommended_vram_gb=24.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        quant_types=["fp16", "int8", "int4"],
        description="R1 蒸馏推理模型，基于 Qwen2.5 Dense 架构。INT4 建议 24GB+ VRAM 或分布式验证，需手动下载 Safetensors。",
        location="external",
    ),
    ModelConfig(
        model_id="gemma4-native",
        name="Gemma 4 12B Native (GGUF + MTMD)",
        model_type="gguf",
        model_path="",
        gguf_path=os.path.join(_BUNDLED_ASSET_ROOT, "models", "gemma4-native", "gemma-4-12B-it-Q4_K_M.gguf"),
        recommended_vram_gb=7.5,
        max_context=4096,
        is_experimental=True,
        huggingface_id="bartowski/gemma-4-12B-it-GGUF",
        quant_types=["Q4_K_M"],
        description="受管 HF bartowski GGUF + mmproj，原生 llama.cpp MTMD 图像理解；需先通过冻结记录校验。",
        location="bundled",
    ),
    # ============================================================
    # ★ 2026-09-19 补注册：此前 `models/` 里已存在但**不在内置清单**的模型。
    #   用户报障「有些模型不在列表里，比如有些 2B 模型」⇒ 逐一按磁盘实际资产登记。
    #   路径/量化名均以磁盘为准（见 `models/*/.qlh-model-asset.json` 与 GGUF header）。
    # ============================================================
    ModelConfig(
        model_id="qwen3-4b",
        name="Qwen3-4B",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "qwen3-4b"),
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen3-4b-gguf", "Qwen3-4B-Q4_K_M.gguf"),
        recommended_vram_gb=4.0,
        max_context=40960,
        is_experimental=True,
        huggingface_id="Qwen/Qwen3-4B",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="Qwen3 4B Dense（36 层 / hidden 2560 / vocab 151936，tie embeddings）。磁盘 8.04 GB safetensors + Q4_K_M GGUF。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen3-5-2b",
        name="Qwen3.5-2B",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "qwen3-5-2b"),
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen3-5-2b-gguf", "qwen35-2b-Q4_K_M.gguf"),
        recommended_vram_gb=3.0,
        max_context=262144,
        is_experimental=True,
        huggingface_id="Qwen/Qwen3.5-2B",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="Qwen3.5 2B（架构 qwen3_5，24 层 / hidden 2048 / vocab 248320，tie embeddings）。磁盘 4.55 GB safetensors + Q4_K_M GGUF。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen3-5-9b",
        name="Qwen3.5-9B",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "qwen3-5-9b"),
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen3-5-9b-gguf", "Qwen3.5-9B-Q4_K_M.gguf"),
        recommended_vram_gb=8.0,
        max_context=262144,
        is_experimental=True,
        huggingface_id="Qwen/Qwen3.5-9B",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="Qwen3.5 9B（架构 qwen3_5，32 层 / hidden 4096 / 不共享词表）。磁盘 19.31 GB safetensors + Q4_K_M GGUF（INT4 需 8GB+ VRAM）。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen3-vl-4b-instruct",
        name="Qwen3-VL-4B-Instruct（多模态）",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "qwen3-vl-4b-instruct"),
        gguf_path=os.path.join(
            _APP_ROOT, "models", "qwen3-vl-4b-instruct-gguf", "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
        ),
        recommended_vram_gb=4.0,
        max_context=262144,
        is_experimental=True,
        huggingface_id="Qwen/Qwen3-VL-4B-Instruct",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="⚠️ **多模态（视觉）**：与 `gemma4-native` 同定位。磁盘含 `mmproj-Qwen3VL-4B-Instruct-F16.gguf` 视觉投影件，需 llama.cpp MTMD 路径。架构 qwen3_vl / 36 层 / hidden 2560。",
        location="external",
    ),
    ModelConfig(
        model_id="qwenseek-2b",
        name="QwenSeek-2B",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "qwenseek-2b"),
        gguf_path=os.path.join(_APP_ROOT, "models", "qwenseek-2b-q4_k_m.gguf"),
        recommended_vram_gb=3.0,
        max_context=262144,
        is_experimental=True,
        huggingface_id="",
        quant_types=["fp16", "int8", "int4", "Q4_K_M"],
        description="QwenSeek 2B（架构 qwen3_5_text，24 层 / hidden 2048 / vocab 248320，tie embeddings）。磁盘 3.76 GB safetensors + Q4_K_M GGUF（1.27 GB）。",
        location="external",
    ),
    ModelConfig(
        model_id="gemma4-12b-safetensors",
        name="Gemma 4 12B（Safetensors 多模态）",
        model_type="safetensors",
        model_path=os.path.join(_APP_ROOT, "models", "gemma4-12b-safetensors"),
        gguf_path="",
        recommended_vram_gb=12.0,
        max_context=262144,
        is_experimental=True,
        huggingface_id="google/gemma-4-12b-it",
        quant_types=["fp16", "int8", "int4"],
        description="⚠️ **多模态**：与 `gemma4-native`（GGUF+MTMD）同定位的 safetensors 版本。架构 gemma4_unified / 48 层 / hidden 3840 / vocab 262144。磁盘 23.92 GB ⇒ INT4 建议 12GB+ VRAM。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen35-9b-dsv4flash",
        name="Qwen3.5-9B-DS v4 Flash（GGUF）",
        model_type="gguf",
        model_path="",
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen35-9b-dsv4flash-q4_k_m.gguf"),
        recommended_vram_gb=7.5,
        max_context=32768,
        is_experimental=True,
        huggingface_id="",
        quant_types=["Q4_K_M"],
        description="Qwen3.5 9B 的 DeepSeek-V4-Flash 蒸馏版（GGUF header：arch=qwen35, size_label=9.0B）。仅 Q4_K_M GGUF 形态，5.63 GB。",
        location="external",
    ),
]


# ================================================================
# 查询函数
# ================================================================

def get_builtin_model(model_id: str) -> Optional[ModelConfig]:
    """从内置注册表中按 model_id 查找。"""
    for m in BUILTIN_MODELS:
        if m.model_id == model_id:
            return m
    return None


def get_profile_default_model_id() -> str:
    """按**当前设备画像**返回默认模型 id（惰性探测；探测失败 ⇒ 兜底 `DEFAULT_MODEL_ID`）。

    与 `get_default_model_id(tier)` 的区别：本函数自己去找当前设备的 tier，
    调用方不必持有画像对象。`device_profiler.get_profile()` 是**全局单例缓存**，
    因此重复调用几乎无成本；这里用**惰性 import**（`device_profiler` 会 import `config`，
    顶层 import 会形成循环）。
    """
    tier = None
    try:
        from device_profiler import get_profile

        tier = get_profile().tier
    except Exception as exc:  # noqa: BLE001 —— 画像不可用时退到兜底，不能因此让加载失败
        logger.debug("取设备画像失败，默认模型退到 %s（%s）", DEFAULT_MODEL_ID, exc)
    return get_default_model_id(tier)


def get_profile_default_model_paths() -> dict:
    """按**当前设备画像**解析默认模型的路径信息（供 `config.MODEL_PATH` 等消费方使用）。

    Returns:
        dict：``{"model_id", "name", "model_path", "gguf_path", "model_type", "preferred_engine"}``。
        任何异常都退到兜底模型（不抛），保证「无人值守自动加载」不会因画像问题中断。
    """
    model = get_default_model()
    try:
        # ⚠️ 必须在此**函数内**导入：`device_profiler` 会 import `config`，顶层导入形成循环。
        from device_profiler import get_profile

        model = get_default_model(get_profile().tier)
    except Exception:  # noqa: BLE001 —— 画像不可用 ⇒ 用兜底模型
        pass
    return {
        "model_id": model.model_id,
        "name": model.name,
        "model_path": resolve_model_path(model.model_path) if model.model_path else "",
        "gguf_path": resolve_model_path(model.gguf_path) if model.gguf_path else "",
        "model_type": model.model_type,
    }


def get_builtin_models() -> list[ModelConfig]:
    """返回所有内置模型（含实验模型）。"""
    return list(BUILTIN_MODELS)


def get_default_model(tier: str = None) -> ModelConfig:
    """返回默认模型配置（可按设备画像档位选择）。

    Args:
        tier: `DeviceTier` 的字符串值或枚举；`None` ⇒ 用兜底 `DEFAULT_MODEL_ID`。
              分档规则与实测依据见 `DEFAULT_MODEL_BY_TIER` 的注释。
    """
    model_id = get_default_model_id(tier)
    model = get_builtin_model(model_id)
    if model is None:
        raise RuntimeError(f"默认模型 '{model_id}' 在内置注册表中未找到")
    return model


def has_safetensors_files(model: ModelConfig) -> bool:
    """Return True when the model's safetensors/bin directory exists on disk."""
    model_dir = resolve_model_path(model.model_path)
    if not model_dir or not os.path.isdir(model_dir):
        return False
    try:
        names = os.listdir(model_dir)
    except OSError:
        return False
    has_config = os.path.isfile(os.path.join(model_dir, "config.json"))
    has_weights = any(
        name.endswith(".safetensors") or name.endswith(".bin")
        for name in names
    )
    return has_config and has_weights


def has_gguf_file(model: ModelConfig) -> bool:
    """Return True when the model's configured GGUF file exists on disk."""
    gguf_path = resolve_model_path(model.gguf_path)
    return bool(gguf_path and os.path.isfile(gguf_path))


def get_model_file_status(model: ModelConfig) -> dict:
    """Summarize local file availability for a model configuration."""
    has_safetensors = has_safetensors_files(model)
    has_gguf = has_gguf_file(model)
    formats: list[str] = []
    if has_safetensors:
        formats.append("safetensors")
    if has_gguf:
        formats.append("gguf")

    expected: list[str] = []
    if model.model_type in ("safetensors", "both"):
        expected.append(f"Safetensors: {model.model_path or '(未配置)'}")
    if model.model_type in ("gguf", "both"):
        expected.append(f"GGUF: {model.gguf_path or '(未配置)'}")

    is_available = bool(formats)
    return {
        "is_available": is_available,
        "available_formats": formats,
        "has_safetensors": has_safetensors,
        "has_gguf": has_gguf,
        "unavailable_reason": "" if is_available else "模型文件未落盘；请先下载到配置路径。",
        "expected_paths": expected,
    }


def is_cuda_available() -> bool:
    """检测 CUDA 是否可用（server-side 门控）。"""
    try:
        from config import INFERENCE_ENGINE
        from torch_runtime import cuda_available

        if str(INFERENCE_ENGINE or "llama_cpp").lower() in {"llama_cpp", "gguf", "island"}:
            return False
        return cuda_available(load=True)
    except ImportError:
        return False


def model_config_from_registry(entry: dict, builtin: Optional[ModelConfig] = None) -> Optional[ModelConfig]:
    """Apply one authoritative registry row, retaining only omitted defaults.

    Empty artifact paths are intentional overrides. Invalid rows never fall
    back to a different builtin artifact under the same model identifier.
    """
    if not isinstance(entry, dict):
        return None
    mid = entry.get("model_id")
    if not isinstance(mid, str) or not mid:
        return None
    base = builtin or ModelConfig(model_id=mid, name=mid, is_experimental=True, location="external")
    fields = {name: entry.get(name, getattr(base, name)) for name in (
        "name", "model_type", "model_path", "gguf_path", "huggingface_id", "description",
    )}
    if any(not isinstance(value, str) for value in fields.values()):
        return None
    if fields["model_type"] not in {"safetensors", "gguf", "both"}:
        return None
    quants = entry.get("quant_types", base.quant_types)
    if not isinstance(quants, list) or any(not isinstance(value, str) or not value for value in quants):
        return None
    try:
        vram = float(entry.get("recommended_vram_gb", base.recommended_vram_gb))
        context = int(entry.get("max_context", base.max_context))
        import math
        if not math.isfinite(vram) or vram < 0 or context <= 0:
            return None
        return ModelConfig(model_id=mid, **fields, recommended_vram_gb=vram,
                           max_context=context, quant_types=list(quants),
                           is_experimental=base.is_experimental,
                           location="external")
    except (TypeError, ValueError, OverflowError):
        return None


def get_all_model_configs(db_experimental_models: Optional[list[dict]] = None) -> list[ModelConfig]:
    """Resolve the catalog using the same registry precedence as execution."""
    rows = {entry.get("model_id"): entry for entry in (db_experimental_models or [])
            if isinstance(entry, dict) and isinstance(entry.get("model_id"), str)}
    result = []
    for builtin in get_builtin_models():
        entry = rows.pop(builtin.model_id, None)
        resolved = model_config_from_registry(entry, builtin) if entry is not None else builtin
        if resolved is not None:
            result.append(resolved)
    for entry in rows.values():
        resolved = model_config_from_registry(entry)
        if resolved is not None:
            result.append(resolved)
    return result


def get_visible_models(db_experimental_models: Optional[list[dict]] = None) -> list[ModelConfig]:
    """Resolve registered overrides before applying the legacy CUDA visibility gate."""
    cuda_ok = is_cuda_available()
    return [model for model in get_all_model_configs(db_experimental_models)
            if not model.is_experimental or cuda_ok]


def get_model_config(model_id: str, db_experimental_models: Optional[list[dict]] = None) -> Optional[ModelConfig]:
    """Use a registered row before the builtin default for every execution path."""
    builtin = get_builtin_model(model_id)
    for entry in db_experimental_models or []:
        if isinstance(entry, dict) and entry.get("model_id") == model_id:
            return model_config_from_registry(entry, builtin)
    return builtin
