"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = (
    "InferenceModeRequest",
    "CreateModelDownloadRequest",
    "LoadModelRequest",
    "PreparePipelineModelRequest",
    "RegisterModelRequest",
    "Request",
    "SwitchModelRequest",
)

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['get_current_model', 'unload_model', 'load_model', 'prepare_pipeline_model', 'list_available_models', 'list_models', 'list_local_model_assets', 'preflight_local_model_asset', 'switch_model', 'list_model_registry', 'register_model', 'unregister_model', 'list_model_presets', 'search_model_repositories', 'create_model_download', 'list_model_downloads', 'get_model_download', 'cancel_model_download', 'downloadable_pytorch_model', 'download_pytorch_model_file', 'downloadable_pipeline_assignment', 'list_gguf_models', 'download_model_file']}

async def get_current_model(request: Request = None):
    """当前模型信息"""
    _api_module.require_model_api_source(request)
    if _api_module._pipeline_model_is_prepared():
        from pipeline_model_descriptor import public_pipeline_descriptor
        descriptor = public_pipeline_descriptor(
            _api_module.model_manager.get_pipeline_descriptor()
        )
        return {
            "inference_mode": _api_module._node_inference_mode(),
            "loaded": False,
            "pipeline_prepared": True,
            "segment_materialized": bool(
                getattr(_api_module.model_manager, "is_loaded", False)
            ),
            "quant_type": _api_module.model_host.current_quant,
            "model_id": descriptor.get("model_id"),
            "descriptor": descriptor,
        }
    if not _api_module.model_host.model_loaded:
        return {
            "inference_mode": _api_module._node_inference_mode(),
            "loaded": False,
            "pipeline_prepared": False,
            "quant_type": None,
            "model_id": None,
        }

    info = _api_module.model_manager.get_model_info()
    mem = _api_module.model_manager.get_memory_usage()
    # ★ 2026-10-11（ROG+Y700+本机三机实测）：整模加载态此前**不返回** `descriptor`，
    #   只有 `pipeline_prepared` 态才返回。后果：Android worker 的
    #   `MainViewModel` 要求 `descriptor.model_sha256` 为 64-hex 才认定
    #   `hasCoordinatorIdentity` ⇒ 恒为 false ⇒ 不扫描/不广告层段工件 ⇒ 容量求解器
    #   看不到该节点（实测 `candidate_node_count=1`）⇒ Route A 无法 prepare，
    #   而 prepare 又正是 descriptor 出现的条件 ⇒ 死锁。
    #   这里补**完整**身份：优先复用**公开流水线描述符**（`public_pipeline_descriptor`，
    #   白名单含 `total_layers` / `hidden_size` / `tokenizer_sha256` / `model_sha256`）——
    #   前三项正是 `model_preflight.descriptor_missing` 的判据，缺一即
    #   `model_preflight_descriptor_incomplete`（实测三机接力就卡在这一层）。
    #   拿不到描述符时退回 `_active_task_graph_model_identity()`（pytorch 整模走
    #   `compute_model_sha256`，带 `model.sha256` + 指纹缓存）。投影失败只降级，
    #   **不让端点 500**（替身 manager 缺属性时仍返回原结构）。
    descriptor = {}
    get_pipeline_descriptor = getattr(
        _api_module.model_manager, "get_pipeline_descriptor", None
    )
    if callable(get_pipeline_descriptor):
        try:
            from pipeline_model_descriptor import public_pipeline_descriptor

            descriptor = public_pipeline_descriptor(get_pipeline_descriptor() or {})
        except Exception:
            _api_module.logger.warning("整模流水线描述符投影失败", exc_info=True)
            descriptor = {}
    try:
        identity = _api_module._active_task_graph_model_identity()
    except Exception:
        _api_module.logger.warning("整模身份投影失败", exc_info=True)
        identity = None
    if identity is not None:
        descriptor = {
            **descriptor,
            "model_id": descriptor.get("model_id") or identity.model_id,
            "model_sha256": descriptor.get("model_sha256") or identity.sha256,
        }
    return {
        "inference_mode": _api_module._node_inference_mode(),
        "loaded": True,
        "model_id": _api_module.model_manager.active_model_id,
        "descriptor": descriptor,
        "quant_type": _api_module.model_host.current_quant,
        # ★ 2026-09-24：**不再**兜底到静态 `MODEL_NAME`（默认模型 ⇒ 会张冠李戴，见
        #   `docs/未完成工作备忘-2026-09-23.md` §4.8 B2）。按活跃模型 id 兜底。
        "model_name": (info.get("model_name") or info.get("model_id")
                       or _api_module.model_manager.active_model_id),
        "model_path": info.get("model_path", ""),
        "engine": info.get("engine", ""),
        "total_params": info.get("total_params", "N/A"),
        "device": info.get("device", "N/A"),
        "gpu_allocated_gb": mem.get("gpu_allocated_gb", 0),
        "gpu_reserved_gb": mem.get("gpu_reserved_gb", 0),
    }

async def set_inference_mode(req: InferenceModeRequest, request: Request = None):
    _api_module.require_model_api_source(request)
    return await _api_module.run_in_threadpool(_api_module._set_node_inference_mode, req.mode)


async def unload_model(request: Request = None):
    """Explicitly release the local LLM before loading another engine."""
    _api_module.require_model_api_source(request)
    try:
        return await _api_module.run_in_threadpool(_api_module._unload_model_under_model_lock)
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        _api_module.logger.error("模型卸载失败: %s", exc, exc_info=True)
        raise _api_module.HTTPException(status_code=500, detail=f"模型卸载失败: {exc}") from exc

async def load_model(req: LoadModelRequest, request: Request = None):
    """
    加载/切换模型。

    耗时约 5-20 秒（取决于量化类型），期间会先卸载旧模型。
    使用 switch_model 获得失败时自动回滚到上一个模型的保护。
    """
    _api_module.require_model_api_source(request)
    global kv_cache, conversation_stats

    operation = (
        _api_module.OP_DISTRIBUTED_LOAD
        if _api_module._node_inference_mode() == "distributed"
        else _api_module.OP_FULL_MODEL
    )
    resolution = _api_module._resolve_model_load_plan(
        req.model_id,
        req.engine,
        req.quant_type,
        operation=operation,
    )
    effective_engine = resolution.engine
    resolved_model_path = resolution.model_path
    quant = resolution.quant_type
    _api_module.logger.info(
        "模型加载裁决: model=%s engine=%s path_kind=%s quant=%s runtime=%s "
        "layer_range_mode=%s reason_code=%s",
        resolution.model_id,
        effective_engine,
        resolution.artifact_kind,
        quant,
        resolution.runtime,
        resolution.layer_range_mode,
        resolution.reason_code,
    )

    # Distributed participation prepares segments; explicit local mode loads a whole model.
    load_mode, load_reason = _api_module._decide_pipeline_load_mode(
        resolution.model_id, effective_engine, quant
    )
    _api_module.logger.info(f"加载模式决策: mode={load_mode}（{load_reason}）")
    if load_mode == "pipeline":
        return await _prepare_pipeline_model_resolution(resolution)

    try:
        t0 = _api_module.time.time()

        import config as cfg

        def _prepare_model_load() -> None:
            # 新模型不能复用旧模型的上下文；必须和推理处于同一互斥边界。
            _api_module._reset_runtime_conversation_state(clear_histories=True)
            cfg.INFERENCE_ENGINE = effective_engine
            cfg.QUANT_TYPE = quant
            cfg.USE_COMPILE = req.use_compile
            _api_module.model_host.model_loaded = True
            _api_module.model_host.current_quant = getattr(_api_module.model_manager, "quant_type", None) or quant
            _api_module.model_host.generation_config["use_compile"] = req.use_compile
            _api_module._init_kv_cache()
            _api_module.scheduler.refresh_task_worker_capabilities()

        # P3修复: 使用 switch_model 获得失败时自动回滚保护
        _api_module.logger.info(f"加载模型: engine={effective_engine}, quant={quant}, compile={req.use_compile}")

        # ★ 2026-09-19：模型加载耗时 5-20 s，**必须**挪到线程池执行。
        #   否则会占住事件循环 ⇒ 期间 `/api/health` 等端点全部无响应
        #   （用户实测：「加载过程中健康不可达，加载完成后恢复」）。
        def _change_model():
            _api_module._require_local_model_mode()
            return _api_module.model_manager.switch_model(
                # ★ 2026-09-19：未指定模型时按**设备画像**取默认（边缘 <1B / PC ~2B）。
                model_id=resolution.model_id,
                quant_type=quant,
                profile=_api_module.device_profile,
                engine=effective_engine if effective_engine != "auto" else None,
                model_path=resolved_model_path,
                db_experimental_models=_api_module._get_registered_experimental_models(),
                resolution=resolution,
            )

        def _do_switch():
            return _api_module._run_exclusive_model_change(
                _change_model,
                prepare=_prepare_model_load,
                release_worker_reservation=True,
            )

        result = await _api_module.run_in_threadpool(_do_switch)

        if result["success"]:
            elapsed = _api_module.time.time() - t0
            status = await _api_module.get_status()
            status["loaded"] = True
            status["pipeline_prepared"] = False
            status["inference_mode"] = _api_module._node_inference_mode()
            status["load_time_seconds"] = round(elapsed, 1)
            status["model_name"] = result.get("model_name", "")

            _api_module.logger.info(f"模型加载完成 ({elapsed:.1f}s): {quant}")
            return status
        else:
            # 切换失败 — 检查是否回滚成功
            _api_module._sync_local_model_projection()
            raise _api_module.coded_http_error(
                500,
                str(result.get("error_code") or "MODEL_LOAD_FAILED"),
                result["error"],
            )

    except _api_module.HTTPException:
        raise
    except Exception as e:
        _api_module._sync_local_model_projection()
        _api_module.logger.error(f"模型加载失败: {e}", exc_info=True)
        raise _api_module.HTTPException(500, f"模型加载失败: {str(e)}")

async def _prepare_pipeline_model_resolution(resolution, *, operation_id=""):
    resolved_model_path = resolution.model_path
    quant = resolution.quant_type
    if not operation_id:
        # Direct prepare of a supplied path keeps the receipt's operation ID.
        # This also prevents equal-SHA bytes from a previous operation granting
        # a new generation without a matching selected manifest.
        worker = getattr(_api_module.scheduler, "_model_supply_worker", None)
        if worker is not None:
            from pathlib import Path
            from model_supply_worker import get_selected_model
            selected = get_selected_model(worker.root)
            selected_path = worker.selected_model_path()
            if selected is not None and selected_path is not None and Path(resolved_model_path).resolve() == Path(selected_path).resolve():
                operation_id = selected["manifest"]["operation_id"]
    candidate_id = f"modeltxn_{_api_module.uuid.uuid4().hex}"
    candidate_prepared = False
    transaction_started = False

    def _after_commit() -> None:
        # The process-wide runtime selection is part of the published model
        # identity.  Keep it behind the same commit fence as the manager and
        # active capacity plan; prepare/commit failures must leave the previous
        # selection untouched.
        import config as cfg

        cfg.INFERENCE_ENGINE = (
            _api_module.backend_id_for(
                _api_module.model_host, default=resolution.engine,
            )
            or resolution.engine
        )
        cfg.QUANT_TYPE = _api_module.model_host.current_quant or quant
        _api_module._reset_runtime_conversation_state(clear_histories=True)
        try:
            _api_module._init_kv_cache()
        except Exception:
            _api_module.logger.warning(
                "流水线模型提交后初始化 KV 缓存失败",
                exc_info=True,
            )
        try:
            _api_module.scheduler.refresh_task_worker_capabilities()
        except Exception:
            _api_module.logger.debug(
                "流水线模型提交后刷新 Worker 能力失败",
                exc_info=True,
            )

    try:
        result = await _api_module.run_in_threadpool(
            lambda: _api_module.model_host.prepare_pipeline_candidate(
                candidate_id,
                model_id=resolution.model_id,
                model_path=resolved_model_path,
                quant_type=quant,
            )
        )
        candidate_prepared = True
        capacity_plan = await _api_module.run_in_threadpool(
            lambda: _api_module.scheduler.start_pipeline_model_transaction(
                candidate_id=candidate_id,
                descriptor=result,
                model_path=resolved_model_path,
                quant_type=quant,
                require_distributed=True,
                on_commit=_after_commit,
                operation_id=operation_id,
            )
        )
        if not capacity_plan.get("admitted") or capacity_plan.get(
            "transaction_phase"
        ) in {"aborted", "invalidated", "rejected"}:
            _api_module.model_host.abort_pipeline_candidate(candidate_id)
            candidate_prepared = False
            raise _api_module.coded_http_error(
                409,
                str(
                    capacity_plan.get("reason_code", "model_preflight_rejected")
                    or "model_preflight_rejected"
                ),
                str(capacity_plan.get("reason", "") or "模型与 Worker 工件事前裁决未通过"),
            )
        transaction_started = True
        from pipeline_model_descriptor import public_pipeline_descriptor
        return {
            "success": True,
            "loaded": False,
            "pipeline_prepared": True,
            "inference_mode": _api_module._node_inference_mode(),
            "model_id": resolution.model_id,
            "engine": resolution.engine,
            "quant_type": quant,
            "transaction_id": candidate_id,
            "transaction_phase": capacity_plan.get(
                "transaction_phase", "preparing"
            ),
            "descriptor": public_pipeline_descriptor(result),
            "model_preflight": capacity_plan.get("model_preflight", {}),
            "capacity_plan_id": capacity_plan.get("plan_id", ""),
        }
    except _api_module.HTTPException:
        raise
    except Exception as exc:
        if candidate_prepared and not transaction_started:
            try:
                _api_module.model_host.abort_pipeline_candidate(candidate_id)
            except Exception:
                _api_module.logger.warning(
                    "流水线候选准备失败后的清理失败: candidate=%s",
                    candidate_id,
                    exc_info=True,
                )
        _api_module.logger.error("准备流水线模型失败: %s", exc, exc_info=True)
        error_code = (
            "SUPPLY_PREPARE_CANCELLED"
            if getattr(exc, "code", "") == "SUPPLY_PREPARE_CANCELLED"
            else "MODEL_TXN_CONFLICT"
            if "MODEL_TXN_CONFLICT" in str(exc)
            else "MODEL_TXN_PREPARE_FAILED"
        )
        status_code = 409 if error_code in {"MODEL_TXN_CONFLICT", "SUPPLY_PREPARE_CANCELLED"} else 400
        raise _api_module.coded_http_error(
            status_code, error_code, f"准备流水线模型失败: {exc}"
        ) from exc


async def prepare_pipeline_model(req: PreparePipelineModelRequest):
    """Prepare a model's proper segments before publishing pipeline execution.

    This endpoint never sets ``model_loaded`` or constructs a complete model.
    The transaction loads the assigned local and remote segments and requires
    matching native receipts before publication. Chat executes committed
    runtimes without filling a missing prepare through lazy loading.

    ★ 2026-10-11：引擎改为 **auto** —— 此前硬编码 pytorch，导致 GGUF 裁层工件
    （`llama_engine.LlamaCppEngine.prepare_pipeline_model`，契约同样是
    "without constructing llama.cpp"）无法只声明层元数据 ⇒ GGUF 主节点的
    `total_layers` 只能靠加载模型获得 ⇒ 与远端区间冲突。改由
    `_resolve_model_load_plan` 按落盘格式选引擎：目录模型走 pytorch 动态层段，
    GGUF 工件走 llama.cpp 只声明元数据。
    """
    if _api_module._node_inference_mode() != "distributed":
        raise _api_module.coded_http_error(409, "LOCAL_INFERENCE_MODE", "Switch to distributed inference before preparing a pipeline.")
    resolution = _api_module._resolve_model_load_plan(
        req.model_id,
        "auto",
        req.quant_type,
        operation=_api_module.OP_DYNAMIC_LAYER_RANGE,
    )
    return await _prepare_pipeline_model_resolution(resolution)

async def list_available_models():
    """列出可选模型配置 + 可用引擎"""
    # 检测所有已注册模型（内置 + DB 注册）的实际落盘格式。
    # 旧逻辑只检查默认 Qwen 文件，DeepSeek/用户注册模型已下载时会漏报引擎。
    model_payloads = [_api_module._model_api_payload(m) for m in _api_module._get_all_model_configs()]
    engine_ids = {
        engine
        for payload in model_payloads
        for engine in payload.get("supported_engines", [])
    }
    available_engines = []

    if "llama_cpp" in engine_ids:
        available_engines.append({
            "id": "llama_cpp",
            "name": "llama.cpp + GGUF",
            "description": "GGUF 量化模型，适合 CPU/集显或轻量试水",
            "model_size_gb": None,
            "requires_cuda": False,
        })

    if "pytorch" in engine_ids:
        has_cuda = _api_module._torch_cuda_available()
        available_engines.append({
            "id": "pytorch",
            "name": "PyTorch + Safetensors" + (" (CUDA)" if has_cuda else " (CPU)"),
            "description": "Safetensors 格式，支持 INT4/INT8/FP16 量化" + ("，GPU 加速" if has_cuda else "，CPU 模式较慢"),
            "model_size_gb": None,
            "requires_cuda": has_cuda,
        })
    # P3修复: 量化选项动态化 — 仅返回当前环境实际可用的量化精度
    pytorch_quants = []
    if "pytorch" in engine_ids:
        has_cuda = _api_module._torch_cuda_available()
        pytorch_quants = [
            {
                "id": "int4",
                "name": "INT4 量化 ⭐",
                "description": "4-bit 量化，显存 ~1.8 GB，速度 ~29 tok/s（推荐边缘设备）",
                "memory_gb": 1.8,
                "speed_tok_s": 29,
                "compile_support": False,
                "engine": "pytorch",
                "is_available": True,
            },
            {
                "id": "int8",
                "name": "INT8 量化",
                "description": "8-bit 量化，显存 ~2.3 GB，速度 ~10 tok/s",
                "memory_gb": 2.3,
                "speed_tok_s": 10,
                "compile_support": False,
                "engine": "pytorch",
                "is_available": True,
            },
        ]
        if has_cuda:
            pytorch_quants.insert(0, {
                "id": "fp16",
                "name": "FP16 原版",
                "description": "原始精度，显存 ~3.5 GB，速度最快 (~53 tok/s)",
                "memory_gb": 3.5,
                "speed_tok_s": 53,
                "compile_support": True,
                "engine": "pytorch",
                "is_available": True,
            })

    gguf_quants = []
    if "llama_cpp" in engine_ids:
        gguf_quants = [
            {
                "id": "gguf",
                "name": "GGUF 量化",
                "description": "量化精度由 GGUF 文件决定（Q4_K_M / Q5_K_M 等），适合 CPU/集显",
                "memory_gb": None,
                "speed_tok_s": None,
                "compile_support": False,
                "engine": "llama_cpp",
                "is_available": True,
            },
        ]

    # ---- TP 孤岛引擎（无本地模型文件，启用即可选）----
    island_quants = []
    import config as _cfg
    if getattr(_cfg, "ISLAND_ENABLED", False) and getattr(_cfg, "ISLAND_BASE_URL", ""):
        from island_engine import mask_island_url
        _island_url_masked = mask_island_url(getattr(_cfg, "ISLAND_BASE_URL", ""))
        available_engines.append({
            "id": "island",
            "name": f"TP 孤岛 ({getattr(_cfg, 'ISLAND_BACKEND', 'openai-compatible')})",
            "description": f"整请求转发到孤岛端点 {_island_url_masked}，量化/并行由孤岛后端决定",
            "model_size_gb": None,
            "requires_cuda": False,
        })
        island_quants = [
            {
                "id": "island",
                "name": "孤岛后端",
                "description": f"由孤岛后端决定（{getattr(_cfg, 'ISLAND_BACKEND', '')}，TP={getattr(_cfg, 'ISLAND_TP_SIZE', 1)}）",
                "memory_gb": None,
                "speed_tok_s": None,
                "compile_support": False,
                "engine": "island",
                "is_available": True,
            },
        ]

    # ---- 外部推理服务（路线 B，按请求路由，不占用本地显存）----
    external_quants = []
    if getattr(_cfg, "EXTERNAL_ENABLED", False) and getattr(_cfg, "EXTERNAL_BASE_URL", ""):
        from external_provider import mask_external_url
        _external_url_masked = mask_external_url(getattr(_cfg, "EXTERNAL_BASE_URL", ""))
        _external_scope = getattr(_cfg, "EXTERNAL_DATA_SCOPE", "opt_in")
        available_engines.append({
            "id": "external_api",
            "name": f"外部推理服务 ({getattr(_cfg, 'EXTERNAL_LABEL', '')})",
            "description": (
                f"整请求路由到外部端点 {_external_url_masked}"
                f"（数据作用域: {_external_scope}，按请求 allow_external 授权，"
                f"非本地引擎，无需加载）"
            ),
            "model_size_gb": None,
            "requires_cuda": False,
        })
        external_quants = [
            {
                "id": "external_api",
                "name": "外部服务后端",
                "description": (
                    f"由外部端点决定（{getattr(_cfg, 'EXTERNAL_LABEL', '')}，"
                    f"作用域 {_external_scope}）"
                ),
                "memory_gb": None,
                "speed_tok_s": None,
                "compile_support": False,
                "engine": "external_api",
                "is_available": True,
            },
        ]

    return {
        "models": pytorch_quants + gguf_quants + island_quants + external_quants,
        "current": _api_module.model_host.current_quant if _api_module.model_host.model_loaded else None,
        "current_engine": (
            _api_module.backend_id_for(_api_module.model_manager)
            if _api_module.model_host.model_loaded and _api_module.model_manager.is_loaded
            else None
        ),
        "available_engines": available_engines,
    }

async def list_models():
    """列出所有可用模型配置（内置 + 用户注册），含 active_model_id。

    - 实验模型仅在 CUDA 可用时返回
    - 始终包含默认 Qwen-1.8B 模型
    """
    models_data = [_api_module._model_api_payload(m) for m in _api_module._get_all_model_configs()]

    return {
        "models": models_data,
        "active_model_id": _api_module.model_manager.active_model_id if _api_module.model_host.model_loaded else None,
    }

async def list_local_model_assets(request: Request = None):
    """List detected local sidecar/task-route assets without registering loaders."""
    _api_module.require_model_api_source(request)
    from local_model_assets import discover_local_model_assets

    # Inventory may hash multi-GB manifests.  Never run that filesystem work
    # on FastAPI's event loop: while it runs, even /health and /device/profile
    # would appear to time out to every client.
    # Inventory is intentionally metadata-only.  Full SHA-256 verification is
    # still performed by the explicit preflight endpoint before runtime use;
    # doing it here would read multi-GB weights on every screen refresh.
    return await _api_module.run_in_threadpool(
        lambda: discover_local_model_assets(verify_hashes=False)
    )

async def preflight_local_model_asset(model_id: str, request: Request = None):
    """Run a supported read-only Sidecar preflight; never load model weights."""
    _api_module.require_model_api_source(request)
    from local_model_assets import preflight_local_model_asset as run_preflight

    return await _api_module.run_in_threadpool(run_preflight, model_id)

async def switch_model(req: SwitchModelRequest, request: Request = None):
    """
    切换到另一个模型（P3 多模型支持）。

    会卸载当前模型，然后加载新模型。
    仅 CUDA 环境可用（非 CUDA 返回 403）。
    """
    _api_module.require_model_api_source(request)
    global kv_cache, conversation_stats

    operation = (
        _api_module.OP_DISTRIBUTED_LOAD
        if _api_module._node_inference_mode() == "distributed"
        else _api_module.OP_FULL_MODEL
    )
    resolution = _api_module._resolve_model_load_plan(
        req.model_id,
        req.engine,
        req.quant_type,
        operation=operation,
    )
    effective_engine = resolution.engine
    resolved_model_path = resolution.model_path
    quant = resolution.quant_type
    _api_module.logger.info(
        "模型切换裁决: model=%s engine=%s path_kind=%s quant=%s runtime=%s "
        "layer_range_mode=%s reason_code=%s",
        resolution.model_id,
        effective_engine,
        resolution.artifact_kind,
        quant,
        resolution.runtime,
        resolution.layer_range_mode,
        resolution.reason_code,
    )

    # Apply the same explicit mode contract as /api/models/load.
    load_mode, load_reason = _api_module._decide_pipeline_load_mode(
        resolution.model_id, effective_engine, quant
    )
    _api_module.logger.info(f"切换模式决策: mode={load_mode}（{load_reason}）")
    if load_mode == "pipeline":
        return await _prepare_pipeline_model_resolution(resolution)

    try:
        import config as cfg

        def _prepare_model_switch() -> None:
            # 新模型不能复用旧模型的上下文；必须和推理处于同一互斥边界。
            _api_module._reset_runtime_conversation_state(clear_histories=True)
            if effective_engine != "auto":
                cfg.INFERENCE_ENGINE = effective_engine
            cfg.QUANT_TYPE = quant
            _api_module.model_host.model_loaded = True
            _api_module.model_host.current_quant = getattr(_api_module.model_manager, "quant_type", None) or quant
            _api_module._init_kv_cache()
            _api_module.scheduler.refresh_task_worker_capabilities()

        def _change_model():
            _api_module._require_local_model_mode()
            return _api_module.model_manager.switch_model(
                model_id=resolution.model_id,
                quant_type=quant,
                profile=_api_module.device_profile,
                engine=effective_engine if effective_engine != "auto" else None,
                model_path=resolved_model_path,
                db_experimental_models=_api_module._get_registered_experimental_models(),
                resolution=resolution,
            )

        result = await _api_module.run_in_threadpool(
            lambda: _api_module._run_exclusive_model_change(
                _change_model,
                prepare=_prepare_model_switch,
                release_worker_reservation=True,
            )
        )

        if result["success"]:
            result.update(loaded=True, pipeline_prepared=False, inference_mode=_api_module._node_inference_mode())
            return result
        else:
            # 切换失败 — 检查是否回滚成功
            _api_module._sync_local_model_projection()
            raise _api_module.coded_http_error(
                500,
                str(result.get("error_code") or "MODEL_SWITCH_FAILED"),
                result["error"],
            )

    except _api_module.HTTPException:
        raise
    except Exception as e:
        _api_module.logger.error(f"模型切换异常: {e}", exc_info=True)
        raise _api_module.HTTPException(status_code=500, detail=f"模型切换失败: {e}")

async def list_model_registry(request: Request = None):
    """列出用户注册的实验模型配置。"""
    _api_module.require_model_api_source(request)
    db_models = _api_module._get_registered_experimental_models()
    return {"models": [_api_module._public_registry_payload(entry) for entry in db_models]}

async def register_model(req: RegisterModelRequest, request: Request = None):
    """注册一个新的实验模型配置。

    模型文件需用户自行下载到指定路径。
    """
    _api_module.require_model_api_source(request)
    if req.model_type not in {"safetensors", "gguf", "both"}:
        raise _api_module.HTTPException(status_code=400, detail="model_type 必须是 safetensors | gguf | both")
    resolved_model_path = _api_module.mc.resolve_model_path(req.model_path) if req.model_path else ""
    resolved_gguf_path = _api_module.mc.resolve_model_path(req.gguf_path) if req.gguf_path else ""
    try:
        artifact = _api_module.validate_model_artifact(
            req.model_type,
            resolved_model_path,
            resolved_gguf_path,
        )
        manifest_root = _api_module.Path(artifact["model_path"] or _api_module.Path(artifact["gguf_path"]).parent)
        manifest = _api_module.build_manifest(
            manifest_root,
            artifact["files"],
            model_type=req.model_type,
            source=req.huggingface_id,
        )
        _api_module.write_manifest(manifest_root, manifest)
    except ValueError as exc:
        raise _api_module.HTTPException(status_code=400, detail=f"model preflight failed: {exc}") from exc
    config = {
        "model_id": req.model_id,
        "name": req.name,
        "model_type": req.model_type,
        "model_path": req.model_path,
        "gguf_path": req.gguf_path,
        "recommended_vram_gb": req.recommended_vram_gb,
        "max_context": req.max_context,
        "huggingface_id": req.huggingface_id,
        "description": req.description,
        "quant_types": ["fp16", "int8", "int4"] if req.model_type != "gguf" else ["Q4_K_M"],
        "sha256": manifest["artifact_sha256"],
        "artifact_sha256": manifest["artifact_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest": manifest,
    }

    try:
        ok = _api_module._local_store.save_local_experimental_model(
            req.model_id, _api_module.json.dumps(config, ensure_ascii=False),
        )
        if not ok:
            raise _api_module.HTTPException(status_code=500, detail="注册模型配置失败")
        return {"status": "registered", "model_id": req.model_id}
    except _api_module.HTTPException:
        raise
    except Exception as e:
        raise _api_module.HTTPException(status_code=500, detail=f"注册模型失败: {e}")

async def unregister_model(model_id: str, request: Request = None):
    """删除一个用户注册的实验模型配置。

    不会删除磁盘上的模型文件，仅取消注册。
    """
    _api_module.require_model_api_source(request)
    # 不允许删除内置模型
    if _api_module.mc.get_builtin_model(model_id):
        raise _api_module.HTTPException(status_code=400, detail=f"内置模型 '{model_id}' 不允许删除。")

    try:
        deleted = _api_module._local_store.delete_local_experimental_model(model_id)
        if not deleted:
            raise _api_module.HTTPException(status_code=404, detail=f"模型 '{model_id}' 未注册")
        return {"status": "deleted", "model_id": model_id}
    except _api_module.HTTPException:
        raise
    except Exception as e:
        raise _api_module.HTTPException(status_code=500, detail=f"取消注册失败: {e}")

async def list_model_presets(request: Request = None):
    """预设列表，含本机可装性评估（installable / blocked_reasons）。"""
    _api_module.require_model_api_source(request)
    return {"presets": _api_module.model_download_jobs.list_presets()}

async def search_model_repositories(
    q: str = "", source: str = "all", page: int = 1, limit: int = 20,
    proxy: str = "",
    request: Request = None,
):
    """搜索公开模型仓库，HF 直连→代理→ModelScope 兜底。"""
    _api_module.require_model_api_source(request)
    try:
        return await _api_module.run_in_threadpool(
            lambda: _api_module.model_search.search_models(
                q, source=source, page=page, limit=limit, proxy=proxy or None,
            )
        )
    except _api_module.model_search.ModelSearchError as exc:
        status = 400 if exc.code in {
            "QUERY_REQUIRED", "QUERY_TOO_LONG", "SOURCE_INVALID", "PAGINATION_INVALID",
        } else 502
        detail: dict[str, _api_module.Any] = {"message": exc.message}
        if exc.attempts:
            detail["attempts"] = exc.attempts
        raise _api_module.coded_http_error(status, exc.code, detail) from exc

async def create_model_download(req: CreateModelDownloadRequest, request: Request = None):
    """排队一个下载 job（异步，轮询 /api/models/downloads/{id} 查进度）。"""
    _api_module.require_model_api_source(request)
    try:
        job = _api_module.model_download_jobs.create_job(
            source=req.source, target=req.target, model_id=req.model_id,
            preset_id=req.preset_id, engine=req.engine or "auto", quant=req.quant,
            use_modelscope=req.use_modelscope, proxy=req.proxy,
            expected_sha256=req.expected_sha256, gguf_path=req.gguf_path,
            allow_cpu=req.allow_cpu,
            executor=lambda fn: _api_module._download_executor.submit(fn),
        )
        return {"job": job, "status": "queued"}
    except _api_module.model_download_jobs.JobError as exc:
        raise _api_module.coded_http_error(400, exc.code, exc.message)
    except _api_module.HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _api_module.HTTPException(status_code=500, detail=f"创建下载任务失败: {exc}")

async def list_model_downloads(limit: int = 100, request: Request = None):
    """列出下载 job（新到旧）。"""
    _api_module.require_model_api_source(request)
    return {"jobs": _api_module.model_download_jobs.list_jobs(limit=limit)}

async def get_model_download(job_id: str, request: Request = None):
    """查询单个下载 job 进度。"""
    _api_module.require_model_api_source(request)
    job = _api_module.model_download_jobs.get_job(job_id)
    if not job:
        raise _api_module.HTTPException(status_code=404, detail=f"下载任务 '{job_id}' 不存在")
    return {"job": job}

async def cancel_model_download(job_id: str, request: Request = None):
    """取消排队中的下载 job；执行中的由后台线程控制，返回当前状态。"""
    _api_module.require_model_api_source(request)
    cancelled = _api_module.model_download_jobs.cancel_job(job_id)
    if not cancelled:
        job = _api_module.model_download_jobs.get_job(job_id)
        if not job:
            raise _api_module.HTTPException(status_code=404, detail=f"下载任务 '{job_id}' 不存在")
    return {"cancelled": cancelled, "job": _api_module.model_download_jobs.get_job(job_id)}

async def downloadable_pytorch_model(request: Request, model_id: str = ""):
    """Return the exact active PyTorch model manifest to Tailnet workers."""
    _api_module._require_trusted_model_peer(request)
    info = _api_module._active_pytorch_model()
    if model_id and model_id != info["model_id"]:
        raise _api_module.HTTPException(409, "请求模型已不是主节点当前流水线模型")

    root = _api_module.os.path.realpath(info["model_path"])
    files = []
    for directory, dirnames, filenames in _api_module.os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for filename in sorted(filenames):
            if filename in {"model.sha256", "model.sha256.meta.json"} or filename.endswith(".part"):
                continue
            if not filename.lower().endswith((
                ".safetensors", ".bin", ".json", ".py", ".tiktoken",
                ".model", ".txt", ".jinja", ".spm", ".vocab",
            )):
                continue
            path = _api_module.os.path.realpath(_api_module.os.path.join(directory, filename))
            try:
                if _api_module.os.path.commonpath([root, path]) != root or not _api_module.os.path.isfile(path):
                    continue
            except ValueError:
                continue
            relative_path = _api_module.os.path.relpath(path, root).replace(_api_module.os.sep, "/")
            files.append({
                "path": relative_path,
                "size_bytes": _api_module.os.path.getsize(path),
                "sha256": _api_module._model_file_sha256(path),
            })
    return {
        "model_id": info["model_id"],
        "sha256": info["model_sha256"],
        "total_layers": info["total_layers"],
        "files": files,
        "count": len(files),
    }

async def download_pytorch_model_file(
    model_id: str,
    relative_path: str,
    request: Request,
):
    """Stream one active-model file to an admitted Tailnet pipeline worker."""
    _api_module._require_trusted_model_peer(request)
    info = _api_module._active_pytorch_model()
    if model_id != info["model_id"]:
        raise _api_module.HTTPException(409, "请求模型已不是主节点当前流水线模型")
    root = _api_module.os.path.realpath(info["model_path"])
    path = _api_module.os.path.realpath(_api_module.os.path.join(root, relative_path.replace("/", _api_module.os.sep)))
    try:
        inside_root = _api_module.os.path.commonpath([root, path]) == root
    except ValueError:
        inside_root = False
    if not inside_root or not _api_module.os.path.isfile(path):
        raise _api_module.HTTPException(404, "模型文件不存在")
    if _api_module.os.path.basename(path) in {"model.sha256", "model.sha256.meta.json"} or path.endswith(".part"):
        raise _api_module.HTTPException(404, "模型文件不存在")
    file_size = _api_module.os.path.getsize(path)
    range_header = request.headers.get("range", "").strip()
    start, end, status_code = 0, file_size - 1, 200
    if range_header:
        if not range_header.lower().startswith("bytes=") or "," in range_header:
            return _api_module.Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
        value = range_header[6:].strip()
        try:
            raw_start, raw_end = value.split("-", 1)
            if raw_start:
                start = int(raw_start)
                end = int(raw_end) if raw_end else file_size - 1
            else:
                suffix = int(raw_end)
                if suffix <= 0:
                    raise ValueError
                start = max(0, file_size - suffix)
                end = file_size - 1
            if start < 0 or start >= file_size or end < start:
                raise ValueError
            end = min(end, file_size - 1)
            status_code = 206
        except (TypeError, ValueError):
            return _api_module.Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    length = end - start + 1

    def _iter_file():
        with open(path, "rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Disposition": f'attachment; filename="{_api_module.os.path.basename(path)}"',
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    return _api_module.StreamingResponse(
        _iter_file(), media_type="application/octet-stream",
        status_code=status_code, headers=headers,
    )

async def downloadable_pipeline_assignment(
    request: Request,
    model_id: str,
    config_id: str = "",
    plan_id: str = "",
    node_id: str = "",
    start_layer: int = 0,
    end_layer: int = 0,
    total_layers: int = 0,
    has_embedding: int = 0,
    has_lm_head: int = 0,
):
    """Return a filtered, generation-bound assignment manifest."""
    _api_module._require_trusted_model_peer(request)
    info = _api_module._active_pytorch_model()
    if model_id != info["model_id"]:
        raise _api_module.HTTPException(409, "请求模型已不是主节点当前流水线模型")
    with _api_module.scheduler._layer_config_lock:
        transaction = _api_module.scheduler._pipeline_load_transaction
        transaction_plan = dict(transaction.get("plan", {})) if transaction else {}
        transaction_config_id = str(transaction.get("config_id", "")) if transaction else ""
        transaction_phase = str(transaction.get("phase", "")) if transaction else ""
        qwen3_transaction = _api_module.scheduler._qwen3_pipeline_dry_run
        qwen3_contract = (
            dict(qwen3_transaction.contract)
            if qwen3_transaction is not None
            and qwen3_transaction.network_dispatch
            else {}
        )
        qwen3_phase = (
            qwen3_transaction.phase if qwen3_transaction is not None else ""
        )
    qwen3_active = bool(
        qwen3_contract
        and qwen3_phase == "preparing"
        and qwen3_contract.get("config_id") == config_id
        and qwen3_contract.get("plan_id") == plan_id
        and qwen3_contract.get("model_id") == model_id
    )
    production_active = bool(
        transaction
        and transaction_phase == "preparing"
        and transaction_config_id == config_id
        and str(transaction_plan.get("plan_id", "")) == plan_id
    )
    if not production_active and not qwen3_active:
        raise _api_module.HTTPException(409, "pipeline assignment is no longer an active prepare generation")
    source_assignments = (
        qwen3_contract.get("segments", [])
        if qwen3_active
        else transaction_plan.get("assignments", [])
    )
    def _assignment_range(item: dict) -> tuple[int, int] | None:
        layer_range = item.get("layer_range")
        if isinstance(layer_range, (list, tuple)) and len(layer_range) == 2:
            values = layer_range
        else:
            values = (item.get("start_layer"), item.get("end_layer"))
        try:
            return int(values[0]), int(values[1])
        except (TypeError, ValueError):
            return None

    matching = [
        item for item in source_assignments
        if str(item.get("node_id", "")) == node_id
        and _assignment_range(item) == (int(start_layer), int(end_layer))
    ]
    if not matching:
        raise _api_module.HTTPException(409, "pipeline assignment does not match the active plan")
    expected = matching[0]
    if bool(expected.get("has_embedding", False)) != bool(has_embedding) or bool(
        expected.get("has_lm_head", False)
    ) != bool(has_lm_head):
        raise _api_module.HTTPException(409, "pipeline assignment component contract changed")
    from pipeline_assignment_manifest import build_assignment_manifest

    try:
        manifest = build_assignment_manifest(
            info["model_path"], model_id=info["model_id"],
            model_sha256=info["model_sha256"], config_id=config_id,
            plan_id=plan_id, node_id=node_id, start_layer=start_layer,
            end_layer=end_layer, total_layers=total_layers,
            has_embedding=bool(has_embedding), has_lm_head=bool(has_lm_head),
        )
    except Exception as exc:
        raise _api_module.HTTPException(409, str(exc)) from exc
    if qwen3_active and manifest.get("manifest_sha256") != expected.get(
        "assignment_manifest_sha256"
    ):
        raise _api_module.HTTPException(409, "Qwen3 assignment manifest digest changed")
    return manifest

# Catalog hashing performs blocking disk I/O; FastAPI runs this handler in its worker pool.
def list_gguf_models(request: Request = None):
    """
    列出可下载的 GGUF 模型文件及其 SHA256 校验值。

    Android 全有模式调用此接口获取可下载的模型列表和下载 URL。
    """
    _api_module.require_model_api_source(request)
    models = []
    if not _api_module.os.path.isdir(_api_module._MODELS_DIR):
        return {"models": models, "exists": False, "count": 0}

    for fname in sorted(_api_module.os.listdir(_api_module._MODELS_DIR)):
        if not fname.lower().endswith(".gguf"):
            continue
        fpath = _api_module.os.path.join(_api_module._MODELS_DIR, fname)
        if not _api_module.os.path.isfile(fpath):
            continue
        size = _api_module.os.path.getsize(fpath)

        # 读取或计算 SHA256（优先读已有 .sha256 文件）
        sha256 = ""
        sha256_file = fpath + ".sha256"
        if _api_module.os.path.isfile(sha256_file):
            try:
                with open(sha256_file, "r") as checksum_file:
                    sha256 = checksum_file.read().strip().split()[0]
            except Exception:
                pass
        if not sha256:
            try:
                hasher = _api_module.hashlib.sha256()
                with open(fpath, "rb") as model_file:
                    for chunk in iter(lambda: model_file.read(8192), b""):
                        hasher.update(chunk)
                sha256 = hasher.hexdigest()
                # 缓存到 .sha256 文件
                with open(sha256_file, "w") as checksum_file:
                    checksum_file.write(f"{sha256}  {fname}\n")
            except Exception:
                sha256 = ""

        models.append({
            "filename": fname,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 1),
            "sha256": sha256,
            "download_url": f"/api/models/download/{fname}",
        })

    return {
        "models": models,
        "exists": True,
        "count": len(models),
    }

async def download_model_file(filename: str, request: Request = None):
    """
    下载 GGUF 模型文件（支持 Range 断点续传）。

    Android ModelManager 调用此接口下载模型，支持分段下载和断点续传。
    """
    _api_module.require_model_api_source(request)
    # 安全检查：防止路径穿越
    safe_name = _api_module.os.path.basename(filename)
    if safe_name != filename or ".." in filename:
        raise _api_module.HTTPException(400, "无效的文件名")

    if not safe_name.lower().endswith(".gguf"):
        raise _api_module.HTTPException(400, "仅支持 .gguf 模型文件下载")

    file_path = _api_module.os.path.join(_api_module._MODELS_DIR, safe_name)
    if not _api_module.os.path.isfile(file_path):
        raise _api_module.HTTPException(404, f"模型文件不存在: {safe_name}")

    return _api_module.FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=safe_name,
    )


def register_routes() -> None:
    router.add_api_route('/api/models/inference-mode', set_inference_mode, methods=['PUT'])
    router.add_api_route('/api/models/current', get_current_model, methods=['GET'])
    router.add_api_route('/api/models/unload', unload_model, methods=['POST'])
    router.add_api_route('/api/models/load', load_model, methods=['POST'])
    router.add_api_route('/api/models/prepare-pipeline', prepare_pipeline_model, methods=['POST'])
    router.add_api_route('/api/models/available', list_available_models, methods=['GET'])
    router.add_api_route('/api/models', list_models, methods=['GET'])
    router.add_api_route('/api/models/local-assets', list_local_model_assets, methods=['GET'])
    router.add_api_route('/api/models/local-assets/{model_id}/preflight', preflight_local_model_asset, methods=['POST'])
    router.add_api_route('/api/models/switch', switch_model, methods=['POST'])
    router.add_api_route('/api/models/registry', list_model_registry, methods=['GET'])
    router.add_api_route('/api/models/registry', register_model, methods=['POST'])
    router.add_api_route('/api/models/registry/{model_id}', unregister_model, methods=['DELETE'])
    router.add_api_route('/api/models/presets', list_model_presets, methods=['GET'])
    router.add_api_route('/api/models/search', search_model_repositories, methods=['GET'])
    router.add_api_route('/api/models/downloads', create_model_download, methods=['POST'])
    router.add_api_route('/api/models/downloads', list_model_downloads, methods=['GET'])
    router.add_api_route('/api/models/downloads/{job_id}', get_model_download, methods=['GET'])
    router.add_api_route('/api/models/downloads/{job_id}', cancel_model_download, methods=['DELETE'])
    router.add_api_route('/api/models/downloadable', downloadable_pytorch_model, methods=['GET'])
    router.add_api_route('/api/models/files/{model_id}/{relative_path:path}', download_pytorch_model_file, methods=['GET'])
    router.add_api_route('/api/models/pipeline-assignment/{model_id}', downloadable_pipeline_assignment, methods=['GET'])
    router.add_api_route('/api/models/gguf', list_gguf_models, methods=['GET'])
    router.add_api_route('/api/models/download/{filename}', download_model_file, methods=['GET'])
