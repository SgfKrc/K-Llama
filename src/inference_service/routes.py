"""inference-svc HTTP 路由（微服务架构改造计划 §4.1 契约，前缀 /v1）。

依赖注入：request.app.state.engine_host / request.app.state.kv_host
（生产由 inference_svc_main 组装；契约测试注入 Fake，见
tests/test_inference_service_protocol.py）。

SSE 事件格式对齐 api_server /api/chat/stream（2026-08-03 基线）：
  data: {"token": "..."}                                        # fast 模式逐 token
  data: {"done": true, "response": "...", "followups": [...],
         "metrics": {...}, "request_id": "..."}                 # 结束事件
  data: {"done": true, "error": "...", "request_id": "..."}     # 错误事件
无 event: 行，纯 data 事件（前端 EventSource 依赖逐字节保真）。
"""
import asyncio
import base64
import concurrent.futures
import json
import logging
import re
import threading
from typing import Any, Dict, Iterator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from api_errors import coded_http_error
import config as _cfg
from model_api_access import require_model_api_source
from request_deadline import (
    REQUEST_DEADLINE_EXCEEDED,
    RequestDeadline,
    RequestDeadlineExceeded,
)
from model_load_resolver import (
    ModelLoadFacts,
    ModelLoadResolutionError,
    OP_DISTRIBUTED_LOAD,
    OP_FULL_MODEL,
    OP_LAYER_RANGE,
    effective_pytorch_cuda_available,
    preferred_engine_for_artifacts,
    resolve_model_load,
)
from inference_service.kv_host import KVCapacityError, KVCleanupError
from . import __contract_version__, __version__
from .protocol import (
    ChatCancelRequest,
    ChatRequest,
    EmbeddingRequest,
    KVFreeRequest,
    KVInitRequest,
    LayerForwardRequest,
    LayerLoadRequest,
    LayerUnloadRequest,
    LMHeadRequest,
    LoadModelRequest,
    SpeculativeRunRequest,
    SwitchModelRequest,
    UnloadModelRequest,
    WorkerStageRequest,
)
from .tensor_transport import deserialize_tensor, serialize_tensor

router = APIRouter(prefix="/v1")

logger = logging.getLogger("inference_service.routes")


def _normalize_operation_id(value: Optional[str]) -> str:
    """Normalize the HTTP request identity to the durable receipt contract."""

    cleaned = re.sub(r"[^A-Za-z0-9._:-]", "", str(value or "").strip())
    return cleaned[:256] or "-"


async def _iterate_sync_generator(
    iterable,
    cancel_event=None,
    request_deadline: RequestDeadline | None = None,
):
    """桥接阻塞式生成器而不阻塞 ASGI 事件循环（复制自
    api_server.py:4132 的等价实现；生成期间 /v1/chat/cancel、/v1/health
    仍可被处理——取消功能依赖此桥接）。"""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    done = object()

    def _put_from_thread(payload) -> bool:
        future = asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
        while True:
            try:
                future.result(timeout=0.1)
                return True
            except concurrent.futures.TimeoutError:
                if cancel_event is not None and cancel_event.is_set():
                    future.cancel()
                    return False
            except Exception:
                return False

    def _pump():
        try:
            for item in iterable:
                if not _put_from_thread((item, None)):
                    return
        except Exception as exc:
            _put_from_thread((None, exc))
        finally:
            _put_from_thread((done, None))

    loop = asyncio.get_running_loop()
    threading.Thread(target=_pump, name="inference-svc-stream-bridge", daemon=True).start()
    completed_normally = False
    try:
        while True:
            wait_timeout = 0.1
            if request_deadline is not None:
                remaining = request_deadline.remaining()
                if remaining <= 0:
                    if cancel_event is not None:
                        cancel_event.set()
                    raise RequestDeadlineExceeded(
                        "request deadline exceeded while streaming"
                    )
                wait_timeout = min(wait_timeout, remaining)
            try:
                item, error = await asyncio.wait_for(
                    queue.get(), timeout=wait_timeout,
                )
            except asyncio.TimeoutError:
                continue
            if item is done:
                completed_normally = True
                break
            if error is not None:
                raise error
            yield item
    finally:
        if not completed_normally and cancel_event is not None:
            cancel_event.set()
        close = getattr(iterable, "close", None)
        if callable(close):
            try:
                close()
            except ValueError:
                pass


# ----------------------------------------------------------------------
# 依赖获取
# ----------------------------------------------------------------------
def _engine_host(request: Request):
    host = getattr(request.app.state, "engine_host", None)
    if host is None:
        raise HTTPException(status_code=503, detail="engine host 未初始化")
    return host


def _kv_host(request: Request):
    host = getattr(request.app.state, "kv_host", None)
    if host is None:
        raise HTTPException(status_code=503, detail="kv host 未初始化")
    return host


def _require_master_role(request: Request) -> None:
    """1.3 角色感知：client 角色（从节点）无完整模型，chat 端点 404。"""
    if getattr(request.app.state, "node_role", "master") == "client":
        raise HTTPException(
            status_code=404,
            detail="client 角色不提供 chat 接口（从节点无完整模型）",
        )


# SSE 序列化（对齐 api_server 事件格式）
# ----------------------------------------------------------------------
def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(
    message: Any,
    request_id: str = "-",
    reason_code: str = "",
) -> str:
    if isinstance(message, dict):
        reason_code = reason_code or str(
            message.get("reason_code")
            or message.get("error_code")
            or message.get("code")
            or ""
        )
        message = message.get("message") or json.dumps(message, ensure_ascii=False)
    payload = {"done": True, "error": message, "request_id": request_id}
    if reason_code:
        payload["reason_code"] = reason_code
    return _sse_event(payload)


def _encode_tensor(tensor) -> str:
    return base64.b64encode(serialize_tensor(tensor)).decode("ascii")


def _decode_tensor(tensor_ref: str):
    try:
        data = base64.b64decode(tensor_ref.encode("ascii"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"tensor_ref base64 解码失败: {e}")
    return deserialize_tensor(data)


# ----------------------------------------------------------------------
# 存活/就绪/状态
# ----------------------------------------------------------------------
@router.get("/health", response_model=None)
async def health():
    return {
        "status": "ok",
        "service": "inference-svc",
        "version": __version__,
        "contract_version": __contract_version__,
    }


@router.get("/ready")
async def ready(request: Request):
    return _engine_host(request).ready()


@router.get("/status")
async def status(request: Request):
    host = _engine_host(request)
    kv = _kv_host(request)
    result = host.status()
    result["kv_cache"] = kv.status()
    return result


# ----------------------------------------------------------------------
# 模型生命周期
# ----------------------------------------------------------------------
_GENERATION_ID_PATTERN = re.compile(r"^gen_[A-Za-z0-9_-]{8,96}$")


def _resolve_service_model_load(
    *,
    host,
    model_id: Optional[str],
    engine: Optional[str],
    quant_type: Optional[str],
    operation: str,
):
    import config as cfg
    import model_config as mc
    from torch_runtime import cuda_available

    profile_loader = getattr(host, "_ensure_device_profile", None)
    profile = profile_loader() if callable(profile_loader) else getattr(
        host, "_device_profile", None
    )
    cuda_ok = effective_pytorch_cuda_available(
        system_cuda_available=cuda_available(load=True),
        profile=profile,
    )

    resolved_id = model_id or mc.get_profile_default_model_id()
    model = mc.get_model_config(resolved_id, {})
    if model is None:
        facts = ModelLoadFacts(
            model_id=resolved_id,
            registered=False,
            cuda_available=cuda_ok,
            island_enabled=bool(getattr(cfg, "ISLAND_ENABLED", False)),
            island_base_url=str(getattr(cfg, "ISLAND_BASE_URL", "") or ""),
        )
    else:
        status = mc.get_model_file_status(model)
        facts = ModelLoadFacts(
            model_id=resolved_id,
            model_name=model.name,
            registered=True,
            has_safetensors=bool(status["has_safetensors"]),
            has_gguf=bool(status["has_gguf"]),
            safetensors_path=(
                mc.resolve_model_path(model.model_path)
                if status["has_safetensors"] else None
            ),
            gguf_path=(
                mc.resolve_model_path(model.gguf_path)
                if status["has_gguf"] else None
            ),
            preferred_engine=preferred_engine_for_artifacts(
                has_safetensors=bool(status["has_safetensors"]),
                has_gguf=bool(status["has_gguf"]),
                cuda_available=cuda_ok,
            ),
            cuda_available=cuda_ok,
            island_enabled=bool(getattr(cfg, "ISLAND_ENABLED", False)),
            island_base_url=str(getattr(cfg, "ISLAND_BASE_URL", "") or ""),
        )
    try:
        return resolve_model_load(
            facts,
            requested_engine=engine or "auto",
            requested_quant=quant_type,
            operation=operation,
        )
    except ModelLoadResolutionError as exc:
        status_code = 404 if exc.code == "MODEL_NOT_REGISTERED" else 400
        raise coded_http_error(status_code, exc.code, exc.message) from exc


@router.post("/models/load")
async def models_load(req: LoadModelRequest, request: Request):
    require_model_api_source(request)
    host = _engine_host(request)
    operation = (
        OP_LAYER_RANGE
        if req.layer_range
        else OP_DISTRIBUTED_LOAD
        if getattr(host, "_run_mode", "standalone") == "distributed"
        else OP_FULL_MODEL
    )
    resolution = _resolve_service_model_load(
        host=host,
        model_id=req.model_id,
        engine=req.engine,
        quant_type=req.quant_type,
        operation=operation,
    )
    if req.layer_range:
        return host.load_layer_range(
            layer_range=req.layer_range,
            model_id=resolution.model_id,
            model_path=resolution.model_path,
            quant_type=resolution.quant_type,
            engine=resolution.engine,
            layer_range_mode=resolution.layer_range_mode,
            resolution=resolution,
        )
    return host.load_model(
        engine=resolution.engine,
        quant_type=resolution.quant_type,
        use_compile=req.use_compile,
        model_id=resolution.model_id,
        model_path=resolution.model_path,
        resolution=resolution,
    )


@router.post("/models/unload")
async def models_unload(req: UnloadModelRequest, request: Request):
    require_model_api_source(request)
    return _engine_host(request).unload_model()


@router.post("/models/switch")
async def models_switch(req: SwitchModelRequest, request: Request):
    require_model_api_source(request)
    host = _engine_host(request)
    operation = (
        OP_DISTRIBUTED_LOAD
        if getattr(host, "_run_mode", "standalone") == "distributed"
        else OP_FULL_MODEL
    )
    resolution = _resolve_service_model_load(
        host=host,
        model_id=req.model_id,
        engine=req.engine,
        quant_type=None,
        operation=operation,
    )
    return host.switch_model(
        model_id=resolution.model_id,
        engine=resolution.engine,
        quant_type=resolution.quant_type,
        model_path=resolution.model_path,
        resolution=resolution,
    )


@router.get("/models/current")
async def models_current(request: Request):
    require_model_api_source(request)
    return _engine_host(request).current_model()


@router.get("/models")
async def models_list(request: Request):
    """模型注册表 + 文件状态（对齐 api_server /api/models；DB 实验模型
    由 control-svc /models/registry 承载，此处仅内置模型）。"""
    require_model_api_source(request)
    return _engine_host(request).list_models()


@router.get("/models/local-assets")
async def models_local_assets(request: Request):
    """Read-only inventory of locally present sidecar/task-route assets."""
    require_model_api_source(request)
    return _engine_host(request).list_local_model_assets()


@router.post("/models/local-assets/{model_id}/preflight")
async def models_local_asset_preflight(request: Request, model_id: str):
    """Run a supported read-only Sidecar preflight; never loads model weights."""
    require_model_api_source(request)
    return await run_in_threadpool(_engine_host(request).preflight_local_model_asset, model_id)


@router.get("/models/available")
async def models_available(request: Request):
    """可选模型配置 + 可用引擎（对齐 api_server /api/models/available）。"""
    require_model_api_source(request)
    return _engine_host(request).available_models()


# ----------------------------------------------------------------------
# 对话
# ----------------------------------------------------------------------
@router.post("/chat")
def chat(req: ChatRequest, request: Request):
    """完整对话（JSON；1.2c 已接入 _execute_chat_full 完整复制）。
    同步 def：chat_full 是 CPU/推理阻塞调用，交给 FastAPI 线程池执行，
    不阻塞事件循环（对齐 api_server run_in_threadpool 语义）。"""
    _require_master_role(request)
    host = _engine_host(request)
    request_id = _normalize_operation_id(
        request.headers.get("X-QLH-Request-ID"),
    )
    request_deadline = RequestDeadline.start(_cfg.PIPELINE_TIMEOUT)
    generation_id, cancel_event = host.register_generation(req.generation_id)
    try:
        result = host.chat_full_with_operation_id(
            req, cancel_event, request_id, request_deadline,
        )
    except RequestDeadlineExceeded as exc:
        raise coded_http_error(
            504, REQUEST_DEADLINE_EXCEEDED, str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        host.unregister_generation(generation_id)
    result["request_id"] = request_id
    result["generation_id"] = generation_id
    return result


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE 流式（事件格式与 api_server /api/chat/stream 一致）。"""
    _require_master_role(request)
    host = _engine_host(request)
    request_id = _normalize_operation_id(
        request.headers.get("X-QLH-Request-ID"),
    )
    request_deadline = RequestDeadline.start(_cfg.PIPELINE_TIMEOUT)
    generation_id, cancel_event = host.register_generation(req.generation_id)

    # T9.5：distributed_required 无分布式路径时明确失败（所有模式）
    routing_gate = host._routing_gate_error(req)
    if routing_gate:
        host.unregister_generation(generation_id)
        return StreamingResponse(
            iter([_sse_error(routing_gate, request_id)]),
            media_type="text/event-stream",
        )

    if req.streaming_mode == "full":
        # full：完整功能，推理完成后一次性返回单个 done 事件（SSE 格式）；
        # chat_full 阻塞调用放线程池（api_server run_in_threadpool 语义）
        try:
            result = await run_in_threadpool(
                host.chat_full_with_operation_id,
                req,
                cancel_event,
                request_id,
                request_deadline,
            )
        except RequestDeadlineExceeded as e:
            return StreamingResponse(
                iter([_sse_error(
                    str(e), request_id, REQUEST_DEADLINE_EXCEEDED,
                )]),
                media_type="text/event-stream",
            )
        except HTTPException as e:
            return StreamingResponse(
                iter([_sse_error(e.detail, request_id)]),
                media_type="text/event-stream",
            )
        except Exception as e:
            return StreamingResponse(
                iter([_sse_error(str(e), request_id)]),
                media_type="text/event-stream",
            )
        finally:
            host.unregister_generation(generation_id)
        payload = {
            "done": True,
            "response": result.get("content", ""),
            "thinking_content": result.get("thinking_content"),
            "followups": result.get("followups", []),
            "metrics": result.get("metrics", {}),
            "generation_id": generation_id,
            "request_id": request_id,
        }
        return StreamingResponse(
            iter([_sse_event(payload)]), media_type="text/event-stream"
        )

    # interactive：start → token* → done | error | cancelled（T9 契约 §9.4.1）；
    # EngineHost 与 full/fast 共用历史读取、裁剪和完成时提交。
    if req.streaming_mode == "interactive":
        async def _generate_interactive():
            yield _sse_event({
                "start": True,
                "generation_id": generation_id,
                "request_id": request_id,
                "session_id": req.session_id,
                "routing_preference": req.routing_preference,
            })
            completed_normally = False
            try:
                async for event in _iterate_sync_generator(
                    host.chat_stream_events_with_operation_id(
                        req, cancel_event, request_id, request_deadline,
                    ),
                    cancel_event,
                    request_deadline,
                ):
                    if event.get("done"):
                        event["request_id"] = request_id
                        event["generation_id"] = generation_id
                        event.setdefault("session_id", req.session_id)
                        metrics = event.setdefault("metrics", {})
                        metrics["routing_preference"] = req.routing_preference
                        metrics.setdefault("distributed_used", False)
                        try:
                            host._enforce_distributed_required(
                                req,
                                metrics,
                                detail="interactive 终态未完成允许的分布式执行",
                            )
                        except HTTPException as exc:
                            yield _sse_error(str(exc.detail), request_id)
                            return
                        event["history_committed"] = host.commit_stream_event(
                            req, event, cancel_event, request_id,
                        )
                    yield _sse_event(event)
                completed_normally = True
            except RequestDeadlineExceeded as e:
                yield _sse_error(
                    str(e), request_id, REQUEST_DEADLINE_EXCEEDED,
                )
            except Exception as e:
                yield _sse_error(str(e), request_id)
            finally:
                if not completed_normally:
                    cancel_event.set()
                host.unregister_generation(generation_id)

        return StreamingResponse(
            _generate_interactive(), media_type="text/event-stream"
        )

    # fast：真流式逐 token（同步生成器经线程桥接，不阻塞事件循环）
    async def _generate():
        completed_normally = False
        try:
            async for event in _iterate_sync_generator(
                host.chat_stream_events_with_operation_id(
                    req, cancel_event, request_id, request_deadline,
                ),
                cancel_event,
                request_deadline,
            ):
                if event.get("done"):
                    # engine_host 薄实现的 done 事件带 "-" 占位，此处覆盖为真实值
                    event["request_id"] = request_id
                    event["generation_id"] = generation_id
                    event["history_committed"] = host.commit_stream_event(
                        req, event, cancel_event, request_id,
                    )
                yield _sse_event(event)
            completed_normally = True
        except RequestDeadlineExceeded as e:
            yield _sse_error(
                str(e), request_id, REQUEST_DEADLINE_EXCEEDED,
            )
        except Exception as e:
            yield _sse_error(str(e), request_id)
        finally:
            if not completed_normally:
                cancel_event.set()
            host.unregister_generation(generation_id)

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/chat/cancel")
async def chat_cancel(req: ChatCancelRequest, request: Request):
    _require_master_role(request)
    # 格式校验对齐 api_server.py:441-442（400 语义）
    if not _GENERATION_ID_PATTERN.fullmatch(req.generation_id or ""):
        raise HTTPException(400, "generation_id 格式无效")
    host = _engine_host(request)
    if not host.cancel_generation(req.generation_id):
        # 未注册的合法 id → cancel_pending（对齐 api_server.py:449-456，
        # 不返回 404；contract_diff 2026-08-05 复测暴露 404 语义偏差）
        return {
            "status": "cancel_pending",
            "generation_id": req.generation_id,
        }
    return {
        "status": "cancel_requested",
        "generation_id": req.generation_id,
    }


# ----------------------------------------------------------------------
# 实验端点
# ----------------------------------------------------------------------
@router.post("/speculative/run")
def speculative_run(req: SpeculativeRunRequest, request: Request):
    """投机解码 draft-verify 实验端点（1.2b 接入真实实现；
    门控/异常映射复制自 api_server.experimental_speculative_chat）。
    同步 def：run_speculative_chat 阻塞，交线程池执行。"""
    _require_master_role(request)
    import config as _cfg

    host = _engine_host(request)
    if not getattr(_cfg, "SPEC_ENABLED", False):
        raise HTTPException(
            404,
            "投机解码实验未启用。请设置 QLH_SPEC_ENABLED=true 并配置 "
            "QLH_SPEC_VERIFY_BASE_URL（或复用 QLH_EXTERNAL_BASE_URL）后重启。",
        )
    from external_provider import ExternalScopeDeniedError
    from speculative import (
        SpeculativeCapabilityError,
        SpeculativeConfigError,
        SpeculativeError,
    )

    try:
        result = host.speculative_run(req)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="speculative 实验端点未接入")
    except ExternalScopeDeniedError as exc:
        logger.info(
            "数据作用域拒绝投机解码外部校验: scope=%s（消息正文未发送）",
            getattr(_cfg, "EXTERNAL_DATA_SCOPE", ""),
        )
        raise HTTPException(403, str(exc)) from None
    except SpeculativeConfigError as exc:
        raise HTTPException(409, str(exc)) from None
    except SpeculativeCapabilityError as exc:
        raise HTTPException(502, str(exc)) from None
    except SpeculativeError as exc:
        raise HTTPException(502, str(exc)) from None

    return {
        "content": result["content"],
        "finish_reason": result["finish_reason"],
        "metrics": result["metrics"],
        "rounds": result["rounds"],
        "request_id": request.headers.get("X-QLH-Request-ID", "-"),
    }


# ----------------------------------------------------------------------
# 层段（client 角色；master 角色本地层段同用）
# ----------------------------------------------------------------------
@router.post("/layers/load")
async def layers_load(req: LayerLoadRequest, request: Request):
    return _engine_host(request).load_layer_range(
        layer_range=req.layer_range, embed=req.embed, lm_head=req.lm_head
    )


@router.post("/layers/unload")
async def layers_unload(req: LayerUnloadRequest, request: Request):
    return _engine_host(request).unload_layer_range(layer_range=req.layer_range)


@router.post("/layers/forward")
async def layers_forward(req: LayerForwardRequest, request: Request):
    host = _engine_host(request)
    # ★ #31 M6：hybrid（如 `qwen3_5`）的 recurrent state **不在** KV 分页里 ⇒ `KVHost` 的
    #   `PagedKVCache` 对它是**错误的载体**：拿它当 `past_key_values` 同一轮不报错、数值却错
    #   （静默丢状态，事后极难归因）。而且本端点的响应**只回张量、不回 cache**
    #   ⇒ prefill 产出的 recurrent state 也传不出去、下一轮 decode 必失败。
    #   所以这里**一律明确 fail-closed**（501），而不是"看起来能跑"。
    #   真支持它需要把载体换成 transformers `Cache`（连带 `KVHost` 的分页/cold-tier 设计），
    #   属独立工程 —— 见 `已知问题记录.md` #31 §31.5 M6。
    kind = getattr(host, "kv_state_kind", None)
    if callable(kind) and kind() == "hybrid":
        raise HTTPException(
            status_code=501,
            detail=(
                "该模型是混合层型架构（含 linear_attention 层，recurrent state 不在 KV 分页中）："
                "/layers/forward 的 PagedKVCache 载体不适用（且本端点不回传 cache）"
                "⇒ 请改用进程内层流水线（scheduler_pipeline）路径"
            ),
        )
    hidden = _decode_tensor(req.tensor_ref)
    past_key_values = None
    if req.past_key_values_ref:
        kv_host = _kv_host(request)
        past_key_values = kv_host.get(req.past_key_values_ref)
        if past_key_values is None:
            raise HTTPException(
                status_code=404,
                detail=f"未知 KV 任务引用: {req.past_key_values_ref}",
            )
    try:
        result = host.forward_layers(
            layer_range=req.layer_range,
            hidden=hidden,
            past_key_values=past_key_values,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # 结果张量（hidden states 或 logits）回传
    if hasattr(result, "detach"):  # torch.Tensor
        return {"output_ref": _encode_tensor(result), "task_id": req.task_id}
    return {"output": result, "task_id": req.task_id}


@router.post("/layers/embedding")
async def layers_embedding(req: EmbeddingRequest, request: Request):
    host = _engine_host(request)
    input_ids = _decode_tensor(req.tensor_ref)
    try:
        result = host.embedding(input_ids)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="embedding 段未接入（1.2）")
    return {"output_ref": _encode_tensor(result)}


@router.post("/layers/lm_head")
async def layers_lm_head(req: LMHeadRequest, request: Request):
    host = _engine_host(request)
    hidden = _decode_tensor(req.tensor_ref)
    try:
        result = host.lm_head(hidden)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="lm_head 段未接入（1.2）")
    return {"output_ref": _encode_tensor(result)}


# ----------------------------------------------------------------------
# task-worker Stage 执行（1.4：scheduler-svc 注入 InferenceClient 使用；
# 1.2d 随 task_graph 执行段复制完成后为完整实现）
# ----------------------------------------------------------------------
@router.post("/worker/stage")
def worker_stage(req: WorkerStageRequest, request: Request):
    """远程 Stage 执行（SchedulerCallbackSet.execute_task_worker_stage 的 HTTP 化）。

    body 为 ProviderStageRequest 的 JSON 序列化（dataclasses.asdict 兼容）；
    cancel 通过 request_id 关联的 generation 取消事件实现。
    同步 def：execute_task_worker_stage 阻塞，交线程池执行。
    """
    host = _engine_host(request)
    request_id = request.headers.get("X-QLH-Request-ID", "-")
    # 用 register 返回的 gid 做注销：req.request_id 可能为空（默认 ""），
    # 若用其注销会导致注册表条目泄漏且该 generation 永远无法 cancel
    generation_id, cancel_event = host.register_generation(req.request_id or None)

    from dataclasses import fields
    from task_provider import StageRequest as ProviderStageRequest

    kwargs = {f.name: getattr(req, f.name) for f in fields(ProviderStageRequest)
              if hasattr(req, f.name)}
    if kwargs.get("model_identity") is None:
        kwargs["model_identity"] = None
    stage_request = ProviderStageRequest(**kwargs)

    try:
        result = host.execute_task_worker_stage(stage_request, cancel_event)
    except Exception as e:
        from task_graph import TaskGraphError
        if isinstance(e, TaskGraphError):
            raise HTTPException(status_code=422, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        host.unregister_generation(generation_id)
    return result


# ----------------------------------------------------------------------
# KV 缓存生命周期
# ----------------------------------------------------------------------
@router.post("/kv/init")
async def kv_init(req: KVInitRequest, request: Request):
    require_model_api_source(request)
    kv_host = _kv_host(request)
    try:
        return kv_host.init(
            task_id=req.task_id,
            device=req.device,
            page_size=req.page_size,
            max_pages=req.max_pages,
            cold_cache_dir=req.cold_cache_dir,
            cold_max_pages=req.cold_max_pages,
            cache_unit_size=req.cache_unit_size,
        )
    except KVCapacityError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/kv/free")
async def kv_free(req: KVFreeRequest, request: Request):
    require_model_api_source(request)
    kv_host = _kv_host(request)
    try:
        return kv_host.free(task_id=req.task_id)
    except KVCleanupError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "KV_CLEANUP_FAILED",
                "message": str(exc),
                "task_id": exc.task_id,
            },
        ) from exc
    except KeyError:
        raise HTTPException(status_code=404, detail=f"未知任务: {req.task_id}")
