"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

import asyncio

from fastapi import APIRouter, File
from api._routing import configure_route_module
from api_errors import error_code_from_exception

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = (
    "ChatRequest",
    "ChatResponse",
    "Request",
    "SpeculativeExperimentRequest",
    "UploadFile",
)

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['upload_file', 'chat', 'chat_stream', 'cancel_chat_generation', 'clear_chat']}

def _uses_linear_executor(req):
    scheduler = _api_module.scheduler
    application = getattr(scheduler, "_execution_app", None)
    return bool(application and req.routing_preference != "local_only"
                and _api_module.RUN_MODE == "distributed"
                and scheduler.get_distributed_inference_enabled()
                and scheduler._effective_role() == "master"
                and application.handles())


async def upload_file(file: UploadFile = File(...)):
    """
    上传文本文件，返回解析后的内容。

    支持 txt / md / csv / py / json / log 等纯文本格式。
    限制 5 MB，超过 5000 行自动截断（保留前 5000 行）。
    """
    import os as _os

    # 1. 校验扩展名
    filename = file.filename or "untitled"
    ext = _os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_TEXT_EXTENSIONS:
        raise _api_module.HTTPException(
            400,
            f"不支持的文件类型: {ext}。"
            f"支持的格式: {', '.join(sorted(ALLOWED_TEXT_EXTENSIONS))}",
        )

    # 2. 读取内容
    try:
        raw = await file.read()
    except Exception as e:
        raise _api_module.HTTPException(400, f"文件读取失败: {e}")

    if len(raw) > MAX_UPLOAD_BYTES:
        raise _api_module.HTTPException(
            413,
            f"文件过大 ({len(raw) / 1024 / 1024:.1f} MB)，"
            f"限制 {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB",
        )

    # 3. 解码（尝试 UTF-8 → GBK → latin-1）
    content = None
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            content = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        raise _api_module.HTTPException(400, "无法解码文件内容，请确认文件编码为 UTF-8 或 GBK")

    # 4. 统计 + 截断
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines > MAX_UPLOAD_LINES:
        content = "\n".join(lines[:MAX_UPLOAD_LINES])
        truncated = True
    else:
        truncated = False

    # 统计字符数和词数近似值
    char_count = len(content)
    word_count = len(content.split())

    # 检测语言类型（用于前端代码高亮）
    lang_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".html": "html", ".css": "css",
        ".json": "json", ".md": "markdown", ".csv": "csv",
        ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
        ".sh": "bash", ".bash": "bash", ".ps1": "powershell",
        ".cpp": "cpp", ".c": "c", ".h": "c", ".java": "java",
        ".go": "go", ".rs": "rust", ".rb": "ruby",
        ".sql": "sql", ".r": "r", ".swift": "swift", ".kt": "kotlin",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    }
    language = lang_map.get(ext, "plaintext")

    _api_module.logger.info(
        f"文件上传: {filename} ({ext}) {char_count} 字符 "
        f"{total_lines} 行{' (已截断)' if truncated else ''}"
    )

    return {
        "filename": filename,
        "extension": ext,
        "language": language,
        "char_count": char_count,
        "word_count": word_count,
        "line_count": total_lines if not truncated else MAX_UPLOAD_LINES,
        "total_lines": total_lines,
        "truncated": truncated,
        "truncated_lines": total_lines - MAX_UPLOAD_LINES if truncated else 0,
        "size_bytes": len(raw),
        "content": content,
    }



async def _watch_client_disconnect(request: Request, cancel_event,
                                   generation_id: str = "") -> None:
    """★ 2026-09-30：客户端断开 ⇒ 复用既有取消链路（与 `cancel_chat_generation` 同语义）。

    此前 `/api/chat` 既不接收 `Request` 也不检测断开：实测 `curl -m 3` 切断后服务端
    仍把 `max_new_tokens` 跑满（step 一路到 299/300）—— 纯浪费，分布式下还持续占着
    远端 relay 段。`chat_stream` 靠 `yield` 抛错自然终止，这个同步端点需要显式轮询。

    ⚠️ **不用 `Request.is_disconnected()`**：starlette 1.3 里它用「立即取消的
    CancelScope」做**非阻塞**探测（`message = await self._receive()` 被当场取消），
    只在该 receive 队列里**已经**躺着 `http.disconnect` 时才会返回 True。实测本服务
    （uvicorn 0.49 + starlette 1.3）在 `/api/chat` 上恒返回 False —— watcher 明明在
    按 0.25s 轮询（日志 `断开检测轮询中 polls=20/40`）却从不命中。
    改成**带超时地真等** `receive()`：断开事件到达即返回，未到达则超时继续。
    """
    poll_timeout = 0.05
    idle_sleep = 0.15
    try:
        while not cancel_event.is_set():
            try:
                message = await asyncio.wait_for(
                    request._receive(), timeout=poll_timeout)
            except asyncio.TimeoutError:
                await asyncio.sleep(idle_sleep)
                continue
            if message.get("type") == "http.disconnect":
                cancel_event.set()
                _api_module.logger.info(
                    "客户端已断开，已请求取消生成: generation=%s", generation_id or "-",
                )
                return
    except asyncio.CancelledError:
        raise
    except Exception:                       # 断开检测本身失败不得影响请求
        _api_module.logger.warning("客户端断开检测失败", exc_info=True)

def _check_registered_chat_source(request):
    if _api_module.is_registered_chat_request(request):
        _api_module.require_model_intent_source(request, _api_module.scheduler)


async def chat(req: ChatRequest, request: Request = None):
    """
    发送消息并获取模型回复（多轮对话）。

    自动维护对话历史 + KV 缓存。
    若模型未加载，自动尝试加载默认模型。

    FastAPI 注入 Request；直接协程调用可省略断开检测。
    """
    _check_registered_chat_source(request)
    req.session_id = _api_module._chat_context.resolve_session(req.session_id, client_id=req.client_id)
    generation_id, cancel_event = _api_module._register_generation(req.generation_id)
    req.generation_id = generation_id
    request_deadline = _api_module.RequestDeadline.start(
        _api_module.PIPELINE_TIMEOUT,
    )
    # 路线 B：请求带外部 flag 但被数据作用域拒绝时记一条 INFO（每请求一次）
    _api_module._maybe_log_external_scope_denial(req, _api_module._external_route_decision(req))

    def _run_chat_request_unlocked():
        _check_registered_chat_source(request)
        _api_module._require_speculative_route(req)
        # A required worker request waits on the master, which can dispatch a
        # layer stage back to this same worker. Holding its model lock across
        # that wait would prevent the stage from executing. Required routing
        # already fails closed instead of running a local fallback.
        if (req.routing_preference == "distributed_required"
                and _api_module._should_forward_chat_to_master()):
            return _api_module._execute_requested_chat(
                req, cancel_event, request_deadline, forward_only=True,
            )
        if (_uses_linear_executor(req)
                or (req.speculative_gamma is not None and req.speculative_gamma > 1)):
            return _api_module._execute_requested_chat(req, cancel_event, request_deadline)
        with _api_module.hold_lock_until_deadline(
            _api_module.model_host.full_chat_execution_lock,
            request_deadline,
            operation="full chat execution lock",
        ):
            if (
                not _api_module.model_host.model_loaded
                or not _api_module.model_manager.is_loaded
                or _api_module._pipeline_model_is_prepared()
            ):
                try:
                    _api_module._ensure_chat_model_or_forwarding(req)
                except _api_module.HTTPException:
                    raise
                except FileNotFoundError:
                    raise _api_module.HTTPException(
                        400,
                        "模型未加载且未找到可自动加载的模型文件。"
                        "请先在控制面板中加载模型。",
                    )
                except Exception as exc:
                    raise _api_module.HTTPException(
                        500,
                        f"自动加载模型失败: {exc}。请手动在控制面板中加载模型。",
                    ) from exc
            return _api_module._execute_requested_chat(
                req, cancel_event, request_deadline,
            )

    def _run_chat_request():
        with _api_module._chat_context.transaction(
            serialize=not _uses_linear_executor(req),
            deadline=request_deadline,
        ):
            token = _api_module._request_id_ctx.set(
                _api_module._request_id_ctx.get("-"),
            )
            try:
                return _run_chat_request_unlocked()
            finally:
                _api_module._request_id_ctx.reset(token)

    watcher = None
    if request is None:
        _api_module.logger.debug(
            "chat: 未注入 Request（直接调用），跳过客户端断开检测")
    else:
        watcher = asyncio.create_task(
            _watch_client_disconnect(request, cancel_event, generation_id))
    try:
        result = await _api_module.run_in_threadpool(_run_chat_request)
    except _api_module.RequestDeadlineExceeded as exc:
        raise _api_module.coded_http_error(
            504,
            _api_module.REQUEST_DEADLINE_EXCEEDED,
            str(exc),
        ) from exc
    except _api_module.ChatGenerationCancelled as exc:
        # ★ 2026-10-07（DIST-NEXT-8）：非流式取消同样带独立终态与稳定 reason code，
        #   与其它 409/拒绝响应的风格一致（此前只有一个中文 message）。
        raise _api_module.HTTPException(
            409,
            {
                "message": "生成已取消",
                "generation_id": exc.generation_id,
                **_api_module.request_outcome_metrics(
                    started=True,
                    cancelled=True,
                    reason_code=_api_module.REASON_GENERATION_CANCELLED,
                ),
            },
        ) from exc
    finally:
        if watcher is not None:
            watcher.cancel()
        _api_module._unregister_generation(generation_id, cancel_event)
    return _api_module.ChatResponse(
        content=result["content"],
        thinking_content=result.get("thinking_content"),
        metrics=result["metrics"],
        followups=result["followups"],
    )

async def chat_stream(req: ChatRequest, request: Request):
    """
    流式聊天端点 (Server-Sent Events)。

    支持两种模式（通过 streaming_mode 参数切换）:

    fast — 真流式，逐 token 推送:
      - 路径1: 分布式流水线 → 逐 token SSE
      - 路径2: 单机 PyTorch → 逐 token SSE（TextIteratorStreamer）
      - 路径3: llama.cpp / 其他 → 假流式回退（单次 done 事件）
      - 与 full/interactive 共用会话历史、上下文窗口和完成时提交

    full — 假流式，完整功能:
      - 走 /api/chat 全流程：会话管理、对话历史、追问生成、DB 持久化
      - 推理完成后一次性返回单个 done 事件（SSE 格式）
      - 功能与 /api/chat 完全一致，仅响应格式不同

    事件格式:
        data: {"token": "你"}
        data: {"done": true, "response": "...", "followups": [...], "metrics": {...}}
    """
    import asyncio as _asyncio
    import json as _json
    request_id = _api_module._request_id_ctx.get("-")
    _check_registered_chat_source(request)
    req.session_id = _api_module._chat_context.resolve_session(req.session_id, client_id=req.client_id)
    req.session_id = _api_module._chat_context.resolve_session(req.session_id, client_id=req.client_id)
    generation_id, cancel_event = _api_module._register_generation(req.generation_id)
    req.generation_id = generation_id
    request_deadline = _api_module.RequestDeadline.start(
        _api_module.PIPELINE_TIMEOUT,
    )

    async def _generate():
        previous_request_id = _api_module._request_id_ctx.get("-")
        _api_module._request_id_ctx.set(request_id)
        completed_normally = False
        transaction_acquired = False
        try:
            if req.streaming_mode == "full":
                # Flush headers through proxies before context/model locks or
                # generation can consume the client's connection timeout.
                # SSE comments carry no token or completion semantics.
                yield ": processing\n\n"
            loop = _asyncio.get_running_loop()
            if not _uses_linear_executor(req):
                acquire_future = loop.run_in_executor(
                    None,
                    lambda: _api_module._chat_context.acquire_transaction(
                        deadline=request_deadline,
                    ),
                )
                try:
                    await _asyncio.shield(acquire_future)
                    transaction_acquired = True
                except _asyncio.CancelledError:
                    async def _release_abandoned_acquire():
                        try:
                            await acquire_future
                        finally:
                            _api_module._chat_context.release_transaction()

                    _asyncio.create_task(_release_abandoned_acquire())
                    raise
            async for chunk in _generate_events():
                yield chunk
            completed_normally = True
        except _api_module.HTTPException as exc:
            yield _error_event(
                exc.detail, refused=exc.status_code == 400,
                reason_code=error_code_from_exception(exc),
            )
            completed_normally = True
        except _api_module.RequestDeadlineExceeded as exc:
            yield _error_event(
                str(exc),
                reason_code=_api_module.REQUEST_DEADLINE_EXCEEDED,
            )
            completed_normally = True
        finally:
            if not completed_normally:
                cancel_event.set()
            _api_module._unregister_generation(generation_id, cancel_event)
            if transaction_acquired:
                _api_module._chat_context.release_transaction()
            # StreamingResponse may close an async generator from a different
            # task context. ContextVar tokens cannot be reset across contexts.
            _api_module._request_id_ctx.set(previous_request_id)

    async def _run_with_request_id(loop, func):
        def _runner():
            token = _api_module._request_id_ctx.set(request_id)
            try:
                return func()
            finally:
                _api_module._request_id_ctx.reset(token)

        return await loop.run_in_executor(None, _runner)

    def _error_event(message, *, refused: bool = False, reason_code: str = "") -> str:
        from qlh_core.contracts.snapshots import reason
        cause = message.get("reason_detail") if isinstance(message, dict) else None
        """★ DIST-NEXT-8：错误事件同样带相位与互斥终态。

        `refused=True` 用于「不可恢复、dispatch 前的具名拒绝」（例如路由门）；
        其余为链路/执行失败。两者以前只有一段 `error` 文本，聚合时分不开。
        """
        if isinstance(message, dict):
            reason_code = reason_code or str(
                message.get("reason_code")
                or message.get("error_code")
                or message.get("code")
                or ""
            )
            message = message.get("message") or _api_module.json.dumps(
                message, ensure_ascii=False,
            )
        phase = _api_module.request_outcome_metrics(
            started=True,
            refused=bool(refused),
            failed=not refused,
            reason_code=reason_code or (
                _api_module.REASON_REQUEST_REFUSED if refused
                else _api_module.REASON_REQUEST_FAILED
            ),
        )
        return f"data: {_json.dumps({'done': True, 'error': message, 'request_id': request_id, 'reason_detail': reason('request', phase.get('reason_code') or reason_code or 'request_failed', 'chat_sse', cause=cause), **phase}, ensure_ascii=False)}\n\n"

    async def _iterate_sync_generator(iterable):
        """Bridge a blocking generator without blocking the ASGI event loop."""
        import concurrent.futures as _concurrent_futures

        queue = _asyncio.Queue(maxsize=1)
        done = object()

        def _put_from_thread(payload) -> bool:
            future = _asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
            while True:
                try:
                    future.result(timeout=0.1)
                    return True
                except _concurrent_futures.TimeoutError:
                    if cancel_event.is_set():
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

        loop = _asyncio.get_running_loop()
        _api_module.threading.Thread(
            target=_pump,
            name=f"chat-stream-bridge-{generation_id[-8:]}",
            daemon=True,
        ).start()
        completed_normally = False
        try:
            while True:
                item, error = await queue.get()
                if item is done:
                    completed_normally = True
                    break
                if error is not None:
                    raise error
                yield item
        finally:
            if not completed_normally:
                cancel_event.set()
            close = getattr(iterable, "close", None)
            if callable(close):
                try:
                    close()
                except ValueError:
                    pass

    async def _generate_events():
        _api_module._require_speculative_route(req)
        # 路线 B：请求带外部 flag 但被数据作用域拒绝时记一条 INFO（每请求一次）
        _api_module._maybe_log_external_scope_denial(req, _api_module._external_route_decision(req))
        # T9.5：distributed_required 无分布式路径时明确失败（interactive/fast 共用）
        routing_gate = _api_module._routing_gate_error(req)
        if routing_gate:
            yield _error_event(routing_gate, refused=True)
            return
        # ================================================================
        # ★ interactive 模式（T9 聊天页契约）：真流式逐 token +
        #    完成时会话事务提交（user + assistant 一次写入）
        # ================================================================
        if req.streaming_mode == "interactive":
            try:
                prepared_context = _api_module._prepare_chat_context(
                    req.session_id, req.message, client_id=req.client_id, runtime_kv=not _uses_linear_executor(req),
                )
            except Exception as exc:
                yield _error_event(f"会话上下文加载失败: {exc}")
                return
            target_session_id = prepared_context.session_id
            history = prepared_context.history
            request_messages = [
                *history,
                {"role": "user", "content": req.message},
            ]

            yield f"data: {_json.dumps({'start': True, 'generation_id': generation_id, 'request_id': request_id, 'session_id': target_session_id, 'routing_preference': req.routing_preference}, ensure_ascii=False)}\n\n"

            response_parts: list[str] = []
            thinking_parts: list[str] = []
            metrics: dict = {}
            error: _api_module.Optional[str] = None
            terminal_reason_code = ""
            cancelled = False
            distributed_used = False

            def _append_event(event: dict) -> None:
                nonlocal metrics, error, terminal_reason_code
                if event.get("token"):
                    response_parts.append(str(event["token"]))
                if event.get("thinking"):
                    thinking_parts.append(str(event["thinking"]))
                if isinstance(event.get("metrics"), dict):
                    metrics.update(event["metrics"])
                if event.get("error"):
                    error = str(event["error"])
                    terminal_reason_code = str(
                        event.get("reason_code") or terminal_reason_code
                    )

            loop = _asyncio.get_running_loop()

            def _token_frame(event: dict) -> Optional[str]:
                """token 事件 → SSE 帧；非 token 事件返回 None。"""
                if event.get("token") is not None:
                    return f"data: {_json.dumps({'token': event['token']}, ensure_ascii=False)}\n\n"
                return None

            try:
                # ---- 外部路由（数据作用域门控，与 fast/full 同语义）----
                _ext_decision = _api_module._external_route_decision(req)
                if _ext_decision.use_external:
                    async for event in _iterate_sync_generator(
                        _api_module._external_stream_events(
                            req, cancel_event, history=history,
                        ),
                    ):
                        _append_event(event)
                        frame = _token_frame(event)
                        if frame:
                            yield frame
                        if event.get("done"):
                            break
                elif req.image_data_urls:
                    error = (
                        "本地原生图像请求仅支持 full 响应模式；"
                        "请改用 streaming_mode=full 和 local_only"
                    )
                elif (req.routing_preference != "local_only"
                      and _api_module._should_forward_chat_to_master()):
                    # 计划 §9.5：聊天请求不绕过网关直打本地模型；从节点
                    # interactive 明确失败并引导连接主节点，不静默降级。
                    error = (
                        "interactive 模式要求连接主节点；当前节点是从节点，"
                        "请用 --host 指定主节点 endpoint 后重试"
                    )
                elif _api_module._pipeline_worker_is_reserved():
                    error = (
                        "本设备正作为 PyTorch 分层从节点，interactive 模式不可用；"
                        "请连接主节点"
                    )
                else:
                    if (
                        req.routing_preference == "local_only"
                        and _api_module._pipeline_model_is_prepared()
                    ):
                        error = (
                            "当前模型仅以分布式流水线模式准备；"
                            "local_only 请求需要先显式加载完整模型。"
                        )
                    if (
                        (not _api_module.model_host.model_loaded or not _api_module.model_manager.is_loaded)
                        and not _api_module._pipeline_model_is_prepared()
                        and req.routing_preference != 'distributed_required'
                        and (req.speculative_gamma is None or req.speculative_gamma == 1)
                    ):
                        try:
                            await _run_with_request_id(loop, _api_module._auto_load_default_model)
                        except Exception as exc:
                            error = f"本地回退模型加载失败: {exc}"
                    if error is None:
                        distributed_used = False
                        # 路径 1: 分布式流水线流式（master；local_only 强制跳过）
                        if (req.routing_preference != "local_only"
                                and _api_module.scheduler.get_distributed_inference_enabled()
                                and _api_module.RUN_MODE == "distributed"
                                and _api_module.scheduler._effective_role() == "master"
                                and _api_module.runtime_supports(
                                    _api_module.model_manager, _api_module.Capability.FORWARD_LAYERS,
                                )):
                            distributed_used = True
                            async for event in _iterate_sync_generator(_api_module.scheduler.run_pipeline_stream(
                                req.message,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.pipeline_sampling()[0],
                                top_p=req.pipeline_sampling()[1],
                                **req.pipeline_speculation(),
                                _context_revision=prepared_context.revision,
                                routing_preference=req.routing_preference,
                                session_id=target_session_id,
                                messages=request_messages,
                                show_thinking=req.show_thinking,
                                enable_thinking=req.enable_thinking,
                                _require_distributed=(req.routing_preference == "distributed_required"),
                                _force_distributed_assignment=True,
                                _cancel_event=cancel_event,
                                _request_deadline=request_deadline,
                            )):
                                _append_event(event)
                                frame = _token_frame(event)
                                if frame:
                                    yield frame
                        # 路径 2: 单机 PyTorch 流式
                        elif (_api_module.backend_id_for(_api_module.model_manager) == "llama_cpp"
                              and _api_module.model_manager.is_loaded
                              and callable(getattr(_api_module.model_manager, "chat_stream", None))
                              and callable(getattr(
                                  _api_module.scheduler, "_run_full_model_inference_stream", None,
                              ))):
                            _api_module._require_speculative_route(req, local_fallback=True)
                            async for event in _iterate_sync_generator(
                                _api_module.scheduler._run_full_model_inference_stream(
                                    req.message,
                                    max_new_tokens=req.max_new_tokens,
                                    temperature=req.temperature,
                                    top_p=req.top_p,
                                    messages=request_messages,
                                    show_thinking=req.show_thinking,
                                    enable_thinking=req.enable_thinking,
                                    _cancel_event=cancel_event,
                                    _request_deadline=request_deadline,
                                ),
                            ):
                                _append_event(event)
                                frame = _token_frame(event)
                                if frame:
                                    yield frame
                        elif (_api_module.backend_id_for(_api_module.model_manager) == "pytorch"
                                and _api_module.model_manager.is_loaded):
                            _api_module._require_speculative_route(req, local_fallback=True)
                            async for event in _iterate_sync_generator(_api_module.scheduler._run_full_model_inference_stream(
                                req.message,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                top_p=req.top_p,
                                session_id=target_session_id,
                                messages=request_messages,
                                show_thinking=req.show_thinking,
                                enable_thinking=req.enable_thinking,
                                _cancel_event=cancel_event,
                                _request_deadline=request_deadline,
                            )):
                                _append_event(event)
                                frame = _token_frame(event)
                                if frame:
                                    yield frame
                        # 路径 3: llama.cpp / 其他 → 假流式回退（整段作为单 token 事件）
                        else:
                            result = await _run_with_request_id(
                                loop,
                                lambda: _api_module.scheduler.run_pipeline_safe(
                                    req.message,
                                    max_new_tokens=req.max_new_tokens,
                                    temperature=req.pipeline_sampling()[0],
                                    top_p=req.pipeline_sampling()[1],
                                    **req.pipeline_speculation(),
                                    _context_revision=prepared_context.revision,
                                    routing_preference=req.routing_preference,
                                    session_id=target_session_id,
                                    messages=request_messages,
                                    _require_distributed=(req.routing_preference == "distributed_required"),
                                    _force_distributed_assignment=(req.routing_preference != "local_only"),
                                    _cancel_event=cancel_event,
                                    _request_deadline=request_deadline,
                                ),
                            )
                            if isinstance(result.get("metrics"), dict):
                                metrics.update(result["metrics"])
                            response = result.get("response", "")
                            if response:
                                response_parts.append(str(response))
                                yield f"data: {_json.dumps({'token': str(response)}, ensure_ascii=False)}\n\n"
                            if result.get("error"):
                                error = str(result["error"])
                                terminal_reason_code = str(result.get("reason_code") or "")
            except _api_module.ChatGenerationCancelled:
                cancelled = True
            except _api_module.HTTPException as exc:
                error = exc.detail
                terminal_reason_code = error_code_from_exception(exc)
            except Exception as exc:
                _api_module.logger.error(f"interactive 模式推理失败: {exc}", exc_info=True)
                error = str(exc)

            if cancelled:
                partial = "".join(response_parts)
                # ★ 2026-10-07（DIST-NEXT-8）：取消是**独立终态**，带相位与稳定 reason ——
                #   此前这条事件只有一个 `cancelled` 布尔，聚合时会与「链路错误」「回退」
                #   混在一起，真机排障看不出「用户取消」还是「执行失败」。
                payload = {
                    "cancelled": True,
                    "generation_id": generation_id,
                    "request_id": request_id,
                    "session_id": target_session_id,
                    "partial": partial,
                    **_api_module.request_outcome_metrics(
                        started=True,
                        cancelled=True,
                        reason_code=_api_module.REASON_GENERATION_CANCELLED,
                    ),
                }
                yield f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            if error:
                yield _error_event(
                    error, reason_code=terminal_reason_code,
                )
                return

            response_text = "".join(response_parts)
            metrics.setdefault("request_id", request_id)
            metrics["generation_id"] = generation_id
            metrics["routing_preference"] = req.routing_preference
            metrics["distributed_requested"] = req.routing_preference in (
                "distributed_preferred", "distributed_required",
            )
            # ★ 2026-10-05（DIST-4）：不再用局部标志**无条件覆盖**。
            #   局部 `distributed_used` 在**进入流水线分支时**就置 `True`（`:526`），
            #   而 pipeline 内部仍可能整模回退（事件 metrics 会是
            #   `distributed_used=False` + `execution_mode=fallback_full_model_streaming`）
            #   ⇒ 无条件覆盖会产出 `distributed_used=True` 与
            #   `execution_mode=fallback_full_model_streaming` **并存**的自相矛盾字段，
            #   把「回退成功」伪装成「分布式成功」。
            #   改为：上游（事件 metrics / 响应 metrics）已给出判定时以它为准，
            #   只有在完全没有判定时才回退到局部标志。
            metrics["distributed_used"] = bool(
                metrics.get("distributed_used", distributed_used)
            )
            effective_distributed_used = metrics["distributed_used"]
            if (req.routing_preference in ("distributed_preferred",
                                           "distributed_required")
                    and not effective_distributed_used):
                # 请求分布式但实际本地：必须展示回退原因（计划 §9.5）
                metrics.setdefault("fallback", True)
                metrics.setdefault(
                    "fallback_reason",
                    "distributed_unavailable_fallback_to_local",
                )
            try:
                _api_module._enforce_distributed_required(
                    req,
                    metrics,
                    detail="interactive 终态未完成允许的分布式执行",
                )
            except _api_module.HTTPException as exc:
                yield _error_event(exc.detail)
                return
            # ★ 2026-10-07（DIST-NEXT-8）：统一相位 + 互斥终态。
            #   `admitted` = 本次请求真的拿到了分布式 assignment（config_id /
            #   workers_used / layer_assignments 任一存在）；`fallback=True` 只描述
            #   「回退后仍完成」，不再用于表达取消或拒绝。
            metrics = _api_module.merge_request_outcome(
                metrics,
                admitted=bool(
                    metrics.get("config_id")
                    or metrics.get("workers_used")
                    or metrics.get("layer_assignments")
                ),
                started=True,
                completed=True,
                fallback=bool(metrics.get("fallback")),
            )
            committed = _api_module._commit_chat_context_turn(
                target_session_id, req.message, response_text, metrics,
                expected_revision=prepared_context.revision,
            )
            done_payload = {
                "done": True,
                "response": response_text,
                "thinking_content": "".join(thinking_parts) or None,
                "followups": [],
                "metrics": metrics,
                "generation_id": generation_id,
                "request_id": request_id,
                "session_id": target_session_id,
                "history_committed": committed,
            }
            yield f"data: {_json.dumps(done_payload, ensure_ascii=False)}\n\n"
            return

        # ================================================================
        # ★ full 模式：完整 chat 流程，假流式（SSE 单事件）
        # ================================================================
        if req.streaming_mode == "full":
            loop = _asyncio.get_running_loop()

            def _execute_full_stream_request():
                _check_registered_chat_source(request)
                # Keep this worker available for the master's returning stage.
                if (req.routing_preference == "distributed_required"
                        and _api_module._should_forward_chat_to_master()):
                    return _api_module._execute_requested_chat(
                        req, cancel_event, request_deadline, forward_only=True,
                    )
                if (_uses_linear_executor(req)
                        or (req.speculative_gamma is not None and req.speculative_gamma > 1)):
                    return _api_module._execute_requested_chat(req, cancel_event, request_deadline)
                with _api_module.hold_lock_until_deadline(
                    _api_module.model_host.full_chat_execution_lock,
                    request_deadline,
                    operation="full chat execution lock",
                ):
                    if (
                        not _api_module.model_host.model_loaded
                        or not _api_module.model_manager.is_loaded
                        or _api_module._pipeline_model_is_prepared()
                    ):
                        _api_module._ensure_chat_model_or_forwarding(req)
                    return _api_module._execute_requested_chat(
                        req, cancel_event, request_deadline,
                    )

            try:
                result = await _run_with_request_id(
                    loop, _execute_full_stream_request,
                )
                _check_registered_chat_source(request)
                _done_payload = _json.dumps({
                    'done': True,
                    'response': result['content'],
                    'thinking_content': result.get('thinking_content'),
                    'followups': result['followups'],
                    'metrics': result['metrics'],
                    'request_id': request_id,
                }, ensure_ascii=False)
                yield f"data: {_done_payload}\n\n"
            except _api_module.RequestDeadlineExceeded as e:
                yield _error_event(
                    str(e),
                    reason_code=_api_module.REQUEST_DEADLINE_EXCEEDED,
                )
            except _api_module.HTTPException as e:
                yield _error_event(
                    e.detail,
                    reason_code=error_code_from_exception(e),
                )
            except Exception as e:
                _api_module.logger.error(f"full 模式推理失败: {e}", exc_info=True)
                yield _error_event(str(e))
            return

        # ================================================================
        # fast 模式：真流式；上下文与完成时提交和其他模式完全一致。
        # ================================================================
        try:
            prepared_context = _api_module._prepare_chat_context(
                req.session_id, req.message, client_id=req.client_id, runtime_kv=not _uses_linear_executor(req),
            )
        except Exception as exc:
            yield _error_event(f"会话上下文加载失败: {exc}")
            return
        target_session_id = prepared_context.session_id
        history = prepared_context.history
        request_messages = [
            *history,
            {"role": "user", "content": req.message},
        ]
        fast_response_parts: list[str] = []
        fast_metrics: dict = {}
        fast_committed = False

        def _capture_fast_event(event: dict) -> dict:
            nonlocal fast_committed
            if event.get("token") is not None:
                fast_response_parts.append(str(event["token"]))
            if isinstance(event.get("metrics"), dict):
                fast_metrics.update(event["metrics"])
            if not event.get("done") or fast_committed:
                return event
            if (
                event.get("error")
                or event.get("cancelled")
                or cancel_event.is_set()
                or fast_metrics.get("cancelled")
            ):
                return event
            response_text = str(
                event.get("response") or "".join(fast_response_parts)
            )
            if not response_text:
                return event
            event_metrics = dict(fast_metrics)
            event_metrics.setdefault("request_id", request_id)
            try:
                _api_module._enforce_distributed_required(
                    req,
                    event_metrics,
                    detail="fast 终态未完成允许的分布式执行",
                )
            except _api_module.HTTPException as exc:
                return {
                    "done": True,
                    "error": str(exc.detail),
                    "metrics": event_metrics,
                    "request_id": request_id,
                    "session_id": target_session_id,
                    "history_committed": False,
                }
            event["metrics"] = event_metrics
            event["response"] = response_text
            event["session_id"] = target_session_id
            event["history_committed"] = _api_module._commit_chat_context_turn(
                target_session_id, req.message, response_text, event_metrics,
                expected_revision=prepared_context.revision,
            )
            fast_committed = True
            return event

        # ---- 路线 B: 外部推理服务真流式（数据作用域门控，默认不出集群）----
        external_fallback_reason = ""
        _ext_decision = _api_module._external_route_decision(req)
        if req.image_data_urls and not _ext_decision.use_external:
            yield _error_event(
                "多模态请求必须使用已启用且获数据作用域授权的 external_api："
                f"{_ext_decision.reason}"
            )
            return
        if _ext_decision.use_external:
            loop = _asyncio.get_running_loop()
            ext_events = _api_module._external_stream_events(
                req, cancel_event, history=history,
            )
            first_event = None
            external_error = None
            try:
                # 首个事件同步拉取：失败发生在任何 SSE 数据发出之前，
                # 可以干净地回退到下方既有本地路径
                first_event = await _run_with_request_id(
                    loop, lambda: next(ext_events, None),
                )
            except Exception as exc:
                external_error = exc
            if external_error is None:
                if first_event is not None:
                    first_event = _capture_fast_event(first_event)
                    yield f"data: {_json.dumps(first_event, ensure_ascii=False)}\n\n"
                    if not first_event.get("done"):
                        try:
                            async for event in _iterate_sync_generator(ext_events):
                                # Once any external SSE event has been emitted,
                                # a stream failure is terminal and must cancel.
                                event = _capture_fast_event(event)
                                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            _api_module.logger.error(f"外部推理服务流式失败: {e}")
                            yield _error_event(str(e))
                return
            # 外部失败 → 与 /api/chat 同语义：优先回退本地路径
            if req.image_data_urls:
                yield _error_event(
                    "多模态外部推理服务调用失败，禁止丢弃图片后回退："
                    f"{external_error}"
                )
                return
            if req.prefer_external and not (
                (_api_module.model_host.model_loaded and _api_module.model_manager.is_loaded)
                or _api_module._pipeline_model_is_prepared()
            ):
                yield _error_event(
                    f"外部推理服务调用失败，且本地无可用推理引擎：{external_error}"
                )
                return
            external_fallback_reason = f"external_api_failed: {external_error}"
            _api_module.logger.warning(
                f"外部推理服务流式调用失败: {external_error}，回退到本地路径"
            )

        # ---- 路径 0: 分布式 client 优先转发主节点 ----
        # A pipeline worker may already hold a valid PyTorch segment. It must
        # never enter the local PyTorch streaming branch, which restores the
        # full model and invalidates the master's ready ACK.
        # T9.5: local_only 强制不走转发，仅本地执行
        if (req.routing_preference != "local_only"
                and _api_module._should_forward_chat_to_master()):
            loop = _asyncio.get_running_loop()
            result = await _run_with_request_id(
                loop,
                lambda: _api_module.scheduler.forward_inference_to_master(
                    message=req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.pipeline_sampling()[0],
                    top_p=req.pipeline_sampling()[1],
                    **req.pipeline_speculation(),
                    show_thinking=req.show_thinking,
                    enable_thinking=req.enable_thinking,
                    routing_preference=req.routing_preference,
                    session_id=target_session_id,
                    messages=request_messages,
                    request_id=request_id,
                    _cancel_event=cancel_event,
                    _request_deadline=request_deadline,
                ),
            )
            if result.get("status") != "ok" and req.speculative_gamma is not None and req.speculative_gamma > 1:
                code = (result.get("reason_code") or (result.get("metrics") or {}).get("reason_code")
                        or "speculative_forward_failed")
                yield _error_event(result.get("error") or code, reason_code=code)
                return
            if result.get("status") == "ok":
                metrics = result.get("metrics", {}) or {}
                metrics.setdefault("request_id", request_id)
                done_event = _capture_fast_event({
                    'done': True,
                    'response': result.get('content', ''),
                    'thinking_content': result.get('thinking_content'),
                    'metrics': metrics,
                    'request_id': request_id,
                })
                _done_payload = _json.dumps(done_event, ensure_ascii=False)
                yield f"data: {_done_payload}\n\n"
                return
            if _api_module._pipeline_worker_is_reserved():
                yield _error_event(
                    result.get("error")
                    or "本设备正作为 PyTorch 分层从节点，无法转发到主节点。"
                )
                return

        if _api_module._pipeline_worker_is_reserved():
            yield _error_event(
                "本设备正作为 PyTorch 分层从节点，"
                "请先断开主节点或明确切换本地模型。"
            )
            return

        if (
            req.routing_preference == "local_only"
            and _api_module._pipeline_model_is_prepared()
        ):
            yield _error_event(
                "当前模型仅以分布式流水线模式准备；"
                "local_only 请求需要先显式加载完整模型。"
            )
            return

        if (
            (not _api_module.model_host.model_loaded or not _api_module.model_manager.is_loaded)
            and not _api_module._pipeline_model_is_prepared()
            and req.routing_preference != 'distributed_required'
            and (req.speculative_gamma is None or req.speculative_gamma == 1)
        ):
            loop = _asyncio.get_running_loop()
            try:
                await _run_with_request_id(loop, _api_module._auto_load_default_model)
            except Exception as exc:
                yield _error_event(f"本地回退模型加载失败: {exc}")
                return

        # ---- 路径 1: 分布式流水线流式（local_only 强制跳过，走单机）----
        if (req.routing_preference != "local_only"
                and _api_module.scheduler.get_distributed_inference_enabled()
                and _api_module.RUN_MODE == "distributed"
                and _api_module.scheduler._effective_role() == "master"
                and _api_module.runtime_supports(_api_module.model_manager, _api_module.Capability.FORWARD_LAYERS)):
            try:
                async for event in _iterate_sync_generator(_api_module.scheduler.run_pipeline_stream(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.pipeline_sampling()[0],
                    top_p=req.pipeline_sampling()[1],
                    **req.pipeline_speculation(),
                    _context_revision=prepared_context.revision,
                    routing_preference=req.routing_preference,
                    session_id=target_session_id,
                    messages=request_messages,
                    show_thinking=req.show_thinking,
                    enable_thinking=req.enable_thinking,
                    _require_distributed=(req.routing_preference == "distributed_required"),
                    _force_distributed_assignment=True,
                    _cancel_event=cancel_event,
                    _request_deadline=request_deadline,
                )):
                    event = _capture_fast_event(event)
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                _api_module.logger.error(f"流式推理失败: {e}", exc_info=True)
                yield _error_event(str(e), reason_code=error_code_from_exception(e))

        # ---- 路径 2: 单机 PyTorch 流式 ----
        elif (_api_module.backend_id_for(_api_module.model_manager) == "llama_cpp"
                and _api_module.model_manager.is_loaded
                and callable(getattr(_api_module.model_manager, "chat_stream", None))
                and callable(getattr(
                    _api_module.scheduler, "_run_full_model_inference_stream", None,
                ))):
            try:
                _api_module._require_speculative_route(req, local_fallback=True)
                async for event in _iterate_sync_generator(
                    _api_module.scheduler._run_full_model_inference_stream(
                        req.message,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        messages=request_messages,
                        show_thinking=req.show_thinking,
                        enable_thinking=req.enable_thinking,
                        _cancel_event=cancel_event,
                        _request_deadline=request_deadline,
                    ),
                ):
                    event = _capture_fast_event(event)
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                _api_module.logger.error(f"llama.cpp 流式推理失败: {e}", exc_info=True)
                yield _error_event(str(e), reason_code=error_code_from_exception(e))

        elif (_api_module.backend_id_for(_api_module.model_manager) == "pytorch"
                and _api_module.model_manager.is_loaded):
            try:
                _api_module._require_speculative_route(req, local_fallback=True)
                async for event in _iterate_sync_generator(_api_module.scheduler._run_full_model_inference_stream(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=target_session_id,
                    messages=request_messages,
                    show_thinking=req.show_thinking,
                    enable_thinking=req.enable_thinking,
                    _cancel_event=cancel_event,
                    _request_deadline=request_deadline,
                )):
                    event = _capture_fast_event(event)
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                _api_module.logger.error(f"单机流式推理失败: {e}", exc_info=True)
                yield _error_event(str(e), reason_code=error_code_from_exception(e))

        # ---- 路径 3: llama.cpp / 从节点 / 模型未加载 → 假流式回退 ----
        else:
            loop = _asyncio.get_running_loop()
            result = await _run_with_request_id(
                loop,
                lambda: _api_module.scheduler.run_pipeline_safe(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.pipeline_sampling()[0],
                    top_p=req.pipeline_sampling()[1],
                    **req.pipeline_speculation(),
                    _context_revision=prepared_context.revision,
                    routing_preference=req.routing_preference,
                    session_id=target_session_id,
                    messages=request_messages,
                    _require_distributed=(req.routing_preference == "distributed_required"),
                    _force_distributed_assignment=(req.routing_preference != "local_only"),
                    _cancel_event=cancel_event,
                    _request_deadline=request_deadline,
                )
            )
            # 一次性返回完整结果（SSE 格式，单事件）
            metrics = result.get('metrics', {}) or {}
            metrics.setdefault("request_id", request_id)
            if external_fallback_reason and not metrics.get("fallback_reason"):
                metrics["fallback"] = True
                metrics["fallback_reason"] = external_fallback_reason
            done_event = _capture_fast_event({
                'done': True,
                'response': result.get('response', ''),
                'error': result.get('error'),
                'reason_code': result.get('reason_code'),
                'metrics': metrics,
                'request_id': request_id,
            })
            yield f"data: {_json.dumps(done_event, ensure_ascii=False)}\n\n"

    return _api_module.StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
            "X-Generation-ID": generation_id,
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )

async def cancel_chat_generation(generation_id: str):
    return {
        "status": _api_module._request_generation_cancel(generation_id),
        "generation_id": generation_id,
    }

def clear_chat(session_id: str = "default"):
    """清空当前活跃会话的对话历史与 KV 缓存"""
    global kv_cache, conversation_stats
    result = _api_module.delete_conversations(session_id)
    _api_module.conversation_stats = {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }
    if _api_module.kv_cache:
        _api_module.kv_cache.clear()
    _api_module._init_kv_cache()
    _api_module.logger.info("对话历史已清空: session=%s", result["session_id"])
    return {
        "status": "cleared",
        "session_id": result["session_id"],
        "deleted_count": result.get("deleted_count", 0),
        "conversation_turns": 0,
    }


def register_routes() -> None:
    for name in ('clear_chat',):
        handler = _api_module._serialized_conversation_mutation(globals()[name])
        globals()[name] = handler
    router.add_api_route('/api/chat/upload', upload_file, methods=['POST'])
    router.add_api_route('/api/chat', chat, methods=['POST'], response_model=ChatResponse)
    router.add_api_route('/api/chat/stream', chat_stream, methods=['POST'])
    router.add_api_route('/api/chat/generations/{generation_id}/cancel', cancel_chat_generation, methods=['POST'])
    router.add_api_route('/api/chat/clear', clear_chat, methods=['POST'])
