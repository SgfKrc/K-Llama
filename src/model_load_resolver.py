"""Single-source model load resolution.

This module is deliberately dependency-light.  Callers discover registry and
filesystem facts, then this module atomically chooses the backend, artifact,
runtime precision and layer-range implementation.  It does not import FastAPI,
PyTorch, model managers, schedulers or service hosts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from koakuma_engine import BackendId, normalize_backend_request


LAYER_RANGE_DYNAMIC: Final = "dynamic"
LAYER_RANGE_PRECUT_ARTIFACT: Final = "precut_artifact"
LAYER_RANGE_NONE: Final = "none"

OP_FULL_MODEL: Final = "full_model"
OP_LAYER_RANGE: Final = "layer_range"
OP_DYNAMIC_LAYER_RANGE: Final = "dynamic_layer_range"
OP_DISTRIBUTED_LOAD: Final = "distributed_load"

RUNTIME_PYTORCH_CUDA: Final = "pytorch_cuda"
RUNTIME_PYTORCH_CPU: Final = "pytorch_cpu"
RUNTIME_LLAMA_CPP: Final = "llama_cpp_native"
RUNTIME_ISLAND: Final = "island_remote"


@dataclass(frozen=True)
class EngineCapability:
    artifact_kind: str | None
    layer_range_mode: str


ENGINE_CAPABILITIES: Final = {
    BackendId.LLAMA_CPP: EngineCapability(
        artifact_kind="gguf",
        layer_range_mode=LAYER_RANGE_PRECUT_ARTIFACT,
    ),
    BackendId.PYTORCH: EngineCapability(
        artifact_kind="safetensors",
        layer_range_mode=LAYER_RANGE_DYNAMIC,
    ),
    BackendId.ISLAND: EngineCapability(
        artifact_kind=None,
        layer_range_mode=LAYER_RANGE_NONE,
    ),
}


class ModelLoadResolutionError(ValueError):
    """Stable, transport-independent resolution failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ModelLoadFacts:
    model_id: str | None
    model_name: str = ""
    registered: bool = True
    has_safetensors: bool = False
    has_gguf: bool = False
    safetensors_path: str | None = None
    gguf_path: str | None = None
    preferred_engine: str = "auto"
    cuda_available: bool = False
    island_enabled: bool = False
    island_base_url: str = ""


@dataclass(frozen=True)
class ModelLoadResolution:
    model_id: str | None
    requested_engine: str
    engine: str
    artifact_kind: str | None
    model_path: str | None
    requested_quant: str
    quant_type: str
    runtime: str
    runtime_quant: str
    layer_range_mode: str
    reason_code: str
    reason: str


def preferred_engine_for_artifacts(
    *,
    has_safetensors: bool,
    has_gguf: bool,
    cuda_available: bool,
) -> str:
    """Return the common ``auto`` preference for one node and artifact set."""

    if has_safetensors and (cuda_available or not has_gguf):
        return BackendId.PYTORCH
    if has_gguf:
        return BackendId.LLAMA_CPP
    if has_safetensors:
        return BackendId.PYTORCH
    return BackendId.PYTORCH if cuda_available else BackendId.LLAMA_CPP


def effective_pytorch_cuda_available(
    *,
    system_cuda_available: bool,
    profile: dict | None = None,
) -> bool:
    """Return the CUDA decision actually used by the PyTorch loader.

    Edge/mobile profiles without an assigned CUDA GPU intentionally force the
    loader onto CPU even when the host runtime can see another CUDA device.
    Resolution must make the same decision before quant selection.
    """

    if not system_cuda_available:
        return False
    if not isinstance(profile, dict):
        return True
    tier = str(profile.get("tier", "laptop") or "laptop").strip().lower()
    gpu = profile.get("gpu") if isinstance(profile.get("gpu"), dict) else {}
    if tier in {"ultrabook", "edge", "mobile"} and not bool(
        gpu.get("cuda_available", False)
    ):
        return False
    return True


def _artifact_available(facts: ModelLoadFacts, engine: str) -> bool:
    if engine == BackendId.PYTORCH:
        return bool(facts.has_safetensors and facts.safetensors_path)
    if engine == BackendId.LLAMA_CPP:
        return bool(facts.has_gguf and facts.gguf_path)
    return engine == BackendId.ISLAND


def _select_engine(
    facts: ModelLoadFacts,
    requested: str,
    *,
    operation: str,
) -> tuple[str, str, str]:
    if operation in {OP_LAYER_RANGE, OP_DYNAMIC_LAYER_RANGE} and requested == BackendId.ISLAND:
        raise ModelLoadResolutionError(
            "LAYER_RANGE_MODE_UNSUPPORTED",
            "island 引擎不提供本地层段物化。",
        )
    if requested == BackendId.ISLAND:
        if not facts.island_enabled or not facts.island_base_url:
            raise ModelLoadResolutionError(
                "ISLAND_ENDPOINT_UNCONFIGURED",
                "孤岛引擎未配置端点：请设置 QLH_ISLAND_BASE_URL 后重试。",
            )
        return BackendId.ISLAND, "ENGINE_EXPLICIT", "显式选择 island"

    if (
        operation == OP_DISTRIBUTED_LOAD
        and facts.has_safetensors
        and facts.safetensors_path
        and requested == "auto"
    ):
        return (
            BackendId.PYTORCH,
            "DISTRIBUTED_LOAD_PREFERS_DYNAMIC_LAYERS",
            "分布式加载优先选择可按 key 动态物化层段的 pytorch/Safetensors；"
            "llama.cpp 的预切 GGUF 层段能力保持独立可用。",
        )

    if operation == OP_DYNAMIC_LAYER_RANGE:
        # ★ 2026-10-11（三机实测 / total_layers 缺口）：**仅有 GGUF 裁层工件**时，
        #   允许走 llama.cpp 分支 —— `llama_engine.LlamaCppEngine.prepare_pipeline_model`
        #   的契约原文就是 Prepare GGUF pipeline metadata without constructing llama.cpp，
        #   段区间由工件 manifest 的 `source_layer_range` 声明。
        #   此前这里一律要求 Safetensors，后果是 GGUF 主节点**无法只声明** descriptor ⇒
        #   master 的 `total_layers` 只能靠**加载**模型获得，而加载的是裁层工件 ⇒
        #   视图被钉成该段层数 ⇒ 远端区间恒被拒
        #   （实测 node[android-8e8240bf].layer_ranges[0] exceeds total_layers=16）。
        #   ⚠️ 既有契约**不变**：双格式或显式要求 precut engine 时仍按原样判决
        #   （`LAYER_RANGE_MODE_MISMATCH` / 优先 pytorch），见
        #   tests/test_model_load_resolver.py 的 test_dynamic_layer_range_rejects_explicit_precut_engine。
        has_gguf = bool(facts.has_gguf and facts.gguf_path)
        has_safetensors = bool(facts.has_safetensors and facts.safetensors_path)
        if not has_safetensors and has_gguf:
            return (
                BackendId.LLAMA_CPP,
                "DYNAMIC_LAYER_RANGE_REQUIRES_LLAMA_CPP",
                "仅落了 GGUF 裁层工件：用 llama.cpp 只声明层元数据（不物化权重）。",
            )
        if not has_safetensors or not facts.safetensors_path:
            raise ModelLoadResolutionError(
                "DYNAMIC_LAYER_ARTIFACT_UNAVAILABLE",
                "动态层段物化需要已落盘的 Safetensors 模型目录。",
            )
        if requested == BackendId.PYTORCH:
            return BackendId.PYTORCH, "ENGINE_EXPLICIT", "显式选择 pytorch 动态层段"
        if requested == BackendId.LLAMA_CPP:
            raise ModelLoadResolutionError(
                "LAYER_RANGE_MODE_MISMATCH",
                "请求的是 llama.cpp 预切 GGUF 层段，不能作为 PyTorch 动态层段物化。",
            )
        return (
            BackendId.PYTORCH,
            "DYNAMIC_LAYER_RANGE_REQUIRES_PYTORCH",
            "当前操作需要按 key 动态物化层段，因此使用 pytorch/Safetensors；"
            "llama.cpp 的层段模式仍是 manifest 声明的预切 GGUF 工件。",
        )

    if requested != "auto":
        return requested, "ENGINE_EXPLICIT", f"显式选择 {requested}"

    try:
        preferred = normalize_backend_request(facts.preferred_engine)
    except ValueError:
        preferred = "auto"
    if preferred == "auto":
        preferred = preferred_engine_for_artifacts(
            has_safetensors=facts.has_safetensors,
            has_gguf=facts.has_gguf,
            cuda_available=facts.cuda_available,
        )
    if _artifact_available(facts, preferred):
        return preferred, "ENGINE_AUTO_PREFERRED", f"auto 选择 {preferred}"
    fallback = (
        BackendId.PYTORCH
        if _artifact_available(facts, BackendId.PYTORCH)
        else BackendId.LLAMA_CPP
    )
    if _artifact_available(facts, fallback):
        return fallback, "ENGINE_AUTO_ARTIFACT_FALLBACK", f"auto 按已落盘工件选择 {fallback}"
    return preferred, "ENGINE_AUTO_PREFERRED", f"auto 选择 {preferred}"


def _normalize_pytorch_quant(raw: str) -> str:
    normalized = raw.strip().lower()
    aliases = {
        "f32": "fp32",
        "float32": "fp32",
        "f16": "fp16",
        "float16": "fp16",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"fp32", "fp16", "int8", "int4"}:
        raise ModelLoadResolutionError(
            "MODEL_QUANT_UNSUPPORTED",
            f"PyTorch 不支持量化类型 '{raw}'；可选: fp32, fp16, int8, int4。",
        )
    return normalized


def _resolve_quant(
    engine: str,
    requested_quant: str,
    *,
    cuda_available: bool,
    operation: str,
) -> tuple[str, str, str]:
    raw = requested_quant.strip()
    if engine == BackendId.ISLAND:
        return "island", "island", "孤岛精度由远端运行时决定"
    if engine == BackendId.LLAMA_CPP:
        return "gguf", "gguf", "GGUF 精度由所选工件声明，忽略请求精度标签"

    quant = _normalize_pytorch_quant(raw or "int4")
    if not cuda_available:
        return "fp32", "fp32", f"CPU-only PyTorch 将请求精度 {quant} 显式解析为 fp32"
    if operation in {OP_LAYER_RANGE, OP_DYNAMIC_LAYER_RANGE} and quant in {"int4", "int8"}:
        return "fp16", "fp16", f"动态层段当前以 fp16 物化；请求精度 {quant} 已显式转换"
    return quant, quant, f"CUDA PyTorch 使用 {quant}"


def resolve_model_load(
    facts: ModelLoadFacts,
    *,
    requested_engine: str | None,
    requested_quant: str | None,
    operation: str = OP_FULL_MODEL,
) -> ModelLoadResolution:
    """Resolve one complete, immutable model-load decision."""

    if operation not in {
        OP_FULL_MODEL,
        OP_LAYER_RANGE,
        OP_DYNAMIC_LAYER_RANGE,
        OP_DISTRIBUTED_LOAD,
    }:
        raise ModelLoadResolutionError(
            "MODEL_LOAD_OPERATION_UNSUPPORTED",
            f"不支持的模型加载操作: {operation}",
        )
    try:
        normalized_request = normalize_backend_request(requested_engine)
    except ValueError as exc:
        raise ModelLoadResolutionError(
            "MODEL_ENGINE_UNSUPPORTED",
            f"不支持的引擎: {requested_engine}",
        ) from exc

    if normalized_request != BackendId.ISLAND and not facts.registered:
        raise ModelLoadResolutionError(
            "MODEL_NOT_REGISTERED",
            f"模型 '{facts.model_id}' 未在注册表中找到。",
        )

    engine, reason_code, engine_reason = _select_engine(
        facts,
        normalized_request,
        operation=operation,
    )
    capability = ENGINE_CAPABILITIES[engine]

    if engine == BackendId.PYTORCH:
        path = facts.safetensors_path
        if not facts.has_safetensors or not path:
            raise ModelLoadResolutionError(
                "SAFETENSORS_ARTIFACT_UNAVAILABLE",
                f"模型 '{facts.model_name or facts.model_id}' 未配置或未下载 Safetensors 文件。",
            )
        if str(path).lower().endswith(".gguf"):
            raise ModelLoadResolutionError(
                "MODEL_PATH_ENGINE_MISMATCH",
                "PyTorch loader 拒绝 GGUF 路径。",
            )
        runtime = RUNTIME_PYTORCH_CUDA if facts.cuda_available else RUNTIME_PYTORCH_CPU
    elif engine == BackendId.LLAMA_CPP:
        path = facts.gguf_path
        if not facts.has_gguf or not path:
            raise ModelLoadResolutionError(
                "GGUF_ARTIFACT_UNAVAILABLE",
                f"模型 '{facts.model_name or facts.model_id}' 未配置或未下载 GGUF 文件。",
            )
        if not str(path).lower().endswith(".gguf"):
            raise ModelLoadResolutionError(
                "MODEL_PATH_ENGINE_MISMATCH",
                "llama.cpp loader 只接受 GGUF 工件路径。",
            )
        runtime = RUNTIME_LLAMA_CPP
    else:
        path = None
        runtime = RUNTIME_ISLAND

    requested_quant_text = str(requested_quant or "")
    quant, runtime_quant, quant_reason = _resolve_quant(
        engine,
        requested_quant_text,
        cuda_available=facts.cuda_available,
        operation=operation,
    )
    quant_converted = bool(
        engine == BackendId.PYTORCH
        and _normalize_pytorch_quant(requested_quant_text or "int4") != quant
    )
    if quant_converted:
        reason_code = "RUNTIME_QUANT_CONVERTED"

    return ModelLoadResolution(
        model_id=facts.model_id,
        requested_engine=normalized_request,
        engine=engine,
        artifact_kind=capability.artifact_kind,
        model_path=path,
        requested_quant=requested_quant_text,
        quant_type=quant,
        runtime=runtime,
        runtime_quant=runtime_quant,
        layer_range_mode=capability.layer_range_mode,
        reason_code=reason_code,
        reason=f"{engine_reason}；{quant_reason}",
    )


__all__ = [
    "ENGINE_CAPABILITIES",
    "LAYER_RANGE_DYNAMIC",
    "LAYER_RANGE_NONE",
    "LAYER_RANGE_PRECUT_ARTIFACT",
    "ModelLoadFacts",
    "ModelLoadResolution",
    "ModelLoadResolutionError",
    "OP_LAYER_RANGE",
    "OP_DYNAMIC_LAYER_RANGE",
    "OP_DISTRIBUTED_LOAD",
    "OP_FULL_MODEL",
    "effective_pytorch_cuda_available",
    "preferred_engine_for_artifacts",
    "resolve_model_load",
]
