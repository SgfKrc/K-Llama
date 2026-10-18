"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['get_presets', 'get_status', 'health', 'readiness']}

async def health():
    """健康检查"""
    return {"status": "ok", "timestamp": _api_module.time.time()}


async def readiness():
    """Report runtime readiness without making model loading mandatory."""
    return _api_module._runtime_readiness_snapshot()


async def get_presets():
    """
    返回预设问题列表，包含预估 Token 消耗和显存占用。

    类似豆包/千问 APP 的建议提问功能。
    Token 估算基于 Qwen-1.8B 的经验数据：
      - 中文约 1.5-2 tokens/字
      - 英文约 1-1.3 tokens/字
      - 回复通常为问题的 1-3 倍长度
    """
    # 根据当前加载的量化类型估算速度
    speed_map = {"fp16": 53, "int8": 10, "int4": 29}
    tok_s = speed_map.get(_api_module.model_host.current_quant if _api_module.model_host.model_loaded else "int4", 29)

    # 从设备画像获取档位，调整预估
    max_tokens = _api_module.model_host.generation_config.get("max_new_tokens", 512)

    presets = [
        {
            "id": "intro",
            "icon": "👋",
            "label": "自我介绍",
            "question": "请简单介绍一下你自己，你能做什么？",
            "estimated_prompt_tokens": 25,
            "estimated_response_tokens": 120,
            "estimated_memory_mb": round(145 * 96 / 1024, 1),  # ~13.6 MB KV cache
            "estimated_seconds": round(120 / tok_s, 1),
        },
        {
            "id": "edge_computing",
            "icon": "🌐",
            "label": "边缘计算科普",
            "question": "什么是边缘计算？它和云计算有什么区别？",
            "estimated_prompt_tokens": 35,
            "estimated_response_tokens": 200,
            "estimated_memory_mb": round(235 * 96 / 1024, 1),  # ~22.0 MB
            "estimated_seconds": round(200 / tok_s, 1),
        },
        {
            "id": "model_quantization",
            "icon": "⚡",
            "label": "模型量化原理",
            "question": "大模型的INT4量化是怎么做到的？精度损失大吗？",
            "estimated_prompt_tokens": 40,
            "estimated_response_tokens": 250,
            "estimated_memory_mb": round(290 * 96 / 1024, 1),  # ~27.2 MB
            "estimated_seconds": round(250 / tok_s, 1),
        },
        {
            "id": "code_assist",
            "icon": "💻",
            "label": "Python 代码助手",
            "question": "用Python写一个函数，计算两个大文件的MD5哈希并比较是否相同",
            "estimated_prompt_tokens": 45,
            "estimated_response_tokens": 300,
            "estimated_memory_mb": round(345 * 96 / 1024, 1),  # ~32.3 MB
            "estimated_seconds": round(300 / tok_s, 1),
        },
        {
            "id": "creative",
            "icon": "✨",
            "label": "创意写作",
            "question": "以「边缘设备上的AI觉醒」为题，写一个300字的科幻微小说",
            "estimated_prompt_tokens": 50,
            "estimated_response_tokens": 400,
            "estimated_memory_mb": round(450 * 96 / 1024, 1),  # ~42.2 MB
            "estimated_seconds": round(400 / tok_s, 1),
        },
        {
            "id": "reasoning",
            "icon": "🧩",
            "label": "逻辑推理",
            "question": "A说B撒谎，B说C撒谎，C说A和B都在撒谎。请问谁说的是真话？",
            "estimated_prompt_tokens": 55,
            "estimated_response_tokens": 350,
            "estimated_memory_mb": round(405 * 96 / 1024, 1),  # ~38.0 MB
            "estimated_seconds": round(350 / tok_s, 1),
        },
    ]

    return {
        "presets": presets,
        "current_speed_tok_s": tok_s,
        "current_quant": _api_module.model_host.current_quant if _api_module.model_host.model_loaded else None,
        "max_new_tokens": max_tokens,
    }


async def get_status():
    """获取系统完整状态（含设备档位）"""
    gpu_info = {}
    torch_module = _api_module.loaded_torch()
    if torch_module is not None and _api_module._torch_cuda_available():
        gpu_info = {
            "name": torch_module.cuda.get_device_name(0),
            "total_mb": round(torch_module.cuda.get_device_properties(0).total_memory / (1024**2)),
            "allocated_mb": round(torch_module.cuda.memory_allocated() / (1024**2), 1),
            "reserved_mb": round(torch_module.cuda.memory_reserved() / (1024**2), 1),
            "utilization": round(
                torch_module.cuda.memory_allocated()
                / torch_module.cuda.get_device_properties(0).total_memory
                * 100,
                1,
            ),
        }

    # ---- KV 缓存统计（基于实际对话 token 消耗估算） ----
    # 注：单机模式下 model.generate() 使用内置 KV 缓存，PagedKVCache 未接入。
    # 这里根据实际对话 token 数估算 KV 缓存显存占用。
    num_heads = 16
    head_dim = 64
    num_layers = 24   # Qwen-1.8B
    dtype_bytes = 2   # fp16/bf16
    total_tokens = _api_module.conversation_stats["total_prompt_tokens"] + _api_module.conversation_stats["total_generated_tokens"]
    # KV cache per token = num_layers × 2(K+V) × num_heads × head_dim × dtype_bytes
    kv_bytes_per_token = num_layers * 2 * num_heads * head_dim * dtype_bytes
    kv_memory_mb = round(total_tokens * kv_bytes_per_token / (1024 ** 2), 2)

    # 已分配页估算（以当前 PAGE_SIZE 为基准）
    page_size = _api_module.PAGE_SIZE
    estimated_pages = (total_tokens + page_size - 1) // page_size if total_tokens > 0 else 0
    max_pages = _api_module.MAX_PAGE_NUM
    utilization = estimated_pages / max_pages if max_pages > 0 else 0.0

    kv_stats = {
        "total_tokens": total_tokens,
        "max_tokens": page_size * max_pages,
        "allocated_pages": estimated_pages,
        "free_pages": max_pages - estimated_pages,
        "max_pages": max_pages,
        "page_size": page_size,
        "utilization": round(utilization, 4),
        "estimated_memory_mb": kv_memory_mb,
        "rounds": _api_module.conversation_stats["rounds"],
        "total_time_s": round(_api_module.conversation_stats["total_time_seconds"], 1),
    }

    # 设备画像摘要
    device_summary = None
    if _api_module.device_profile:
        device_summary = {
            "tier": _api_module.device_profile.get("tier"),
            "tier_label": _api_module.device_profile.get("tier_label"),
            "tier_icon": _api_module.device_profile.get("tier_icon"),
            "score": _api_module.device_profile.get("score_total"),
            "gpus": _api_module.device_profile.get("gpus", []),
            "selected_gpu_index": _api_module.device_profile.get("selected_gpu_index", 0),
            "recommendations": _api_module.device_profile.get("recommendations", [])[:3],
            "warnings": _api_module.device_profile.get("warnings", []),
        }

    active_info = {}
    if _api_module.model_host.model_loaded and _api_module.model_manager.is_loaded:
        try:
            active_info = _api_module.model_manager.get_model_info()
        except Exception:
            active_info = {}
    pipeline_prepared = _api_module._pipeline_model_is_prepared()
    pipeline_descriptor = {}
    if pipeline_prepared:
        try:
            from pipeline_model_descriptor import public_pipeline_descriptor
            pipeline_descriptor = public_pipeline_descriptor(
                _api_module.model_manager.get_pipeline_descriptor()
            )
        except Exception:
            _api_module.logger.debug("读取流水线准备状态失败", exc_info=True)

    # ---- TP 孤岛状态（启用时上报，端点已脱敏）----
    import config as _cfg
    island_status = None
    if getattr(_cfg, "ISLAND_ENABLED", False):
        from island_engine import mask_island_url
        island_status = {
            "enabled": True,
            "backend": getattr(_cfg, "ISLAND_BACKEND", ""),
            "base_url": mask_island_url(getattr(_cfg, "ISLAND_BASE_URL", "")),
            "model": active_info.get("model", "") if active_info.get("engine") == "island" else getattr(_cfg, "ISLAND_MODEL", ""),
            "tp_size": getattr(_cfg, "ISLAND_TP_SIZE", 1),
            "gpu_count": getattr(_cfg, "ISLAND_GPU_COUNT", 1),
            "vram_gb": getattr(_cfg, "ISLAND_VRAM_GB", 0.0),
        }

    # ---- 路线 B：外部推理服务状态（启用时上报，端点已脱敏）----
    external_status = None
    if getattr(_cfg, "EXTERNAL_ENABLED", False):
        from external_provider import (
            check_external_reachable,
            get_external_chat_client,
            mask_external_url,
        )
        _ext_client = get_external_chat_client()
        external_status = {
            "enabled": True,
            "label": getattr(_cfg, "EXTERNAL_LABEL", ""),
            "base_url": mask_external_url(
                getattr(_cfg, "EXTERNAL_BASE_URL", "")
            ),
            "model": (
                _ext_client.model_name
                or getattr(_cfg, "EXTERNAL_MODEL", "")
            ),
            "data_scope": getattr(_cfg, "EXTERNAL_DATA_SCOPE", "opt_in"),
            "reasoning_effort": getattr(
                _cfg, "EXTERNAL_REASONING_EFFORT", "",
            ),
            "min_prompt_chars": getattr(_cfg, "EXTERNAL_MIN_PROMPT_CHARS", 0),
            # 轻量健康检查结果，external_provider 内部缓存 ~30s，
            # 不会在每次 /api/status 调用时都探活外部端点。
            # 探活是同步阻塞 IO（外部端点黑洞时最长 connect_timeout 秒），
            # 必须放到线程池，否则整个事件循环停摆、所有 SSE 流一起卡住。
            "reachable": await _api_module.run_in_threadpool(check_external_reachable),
        }

    return {
        "model_loaded": _api_module.model_host.model_loaded,
        "pipeline_prepared": pipeline_prepared,
        "pipeline_descriptor": pipeline_descriptor,
        "current_quant": _api_module.model_host.current_quant,
        "use_compile": _api_module.USE_COMPILE if _api_module.model_host.model_loaded else False,
        # ★ 2026-09-24：**不再**兜底到静态 `MODEL_NAME`（那是默认模型，会张冠李戴 —— 见
        #   `docs/未完成工作备忘-2026-09-23.md` §4.8 B2）。改为按活跃模型 id 兜底，
        #   与同仓既有写法 `inference_service/engine_host.py:664` 同口径。
        "model_name": active_info.get("model_name") or active_info.get("model_id") or (
            _api_module.model_manager.active_model_id
            if _api_module.model_host.model_loaded else ""
        ),
        "model_path": active_info.get("model_path", _api_module.MODEL_PATH),
        "active_model_id": active_info.get("model_id", _api_module.model_manager.active_model_id if _api_module.model_host.model_loaded else None),
        "engine": active_info.get("engine", "") if _api_module.model_host.model_loaded else "",
        "island": island_status,
        "external": external_status,

        "run_mode": _api_module.RUN_MODE,
        "node_role": _api_module.scheduler._effective_role(),
        "node_id": _api_module.scheduler.get_effective_node_id(),
        "max_nodes": _api_module.scheduler._max_nodes,
        "gpu": gpu_info,
        "kv_cache": kv_stats,
        "conversation_turns": len(_api_module._get_active_history()),
        "generation_config": _api_module.model_host.generation_config,
        "device": device_summary,
    }


def register_routes() -> None:
    router.add_api_route("/api/health", health, methods=["GET"])
    router.add_api_route("/api/ready", readiness, methods=["GET"])
    router.add_api_route("/api/presets", get_presets, methods=["GET"])
    router.add_api_route("/api/status", get_status, methods=["GET"])
