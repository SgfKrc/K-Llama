"""
FastAPI 后端服务 — 模型管理 + 对话接口 + 性能监控 + 设备检测
===============================================================
启动: python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
      或在项目根目录: uvicorn src.api_server:app --host 0.0.0.0 --port 8000

功能:
- POST /api/models/load      — 加载/切换模型 (fp16 / int4 / int8)
- POST /api/chat             — 对话（多轮会话，自动维护 KV 缓存）
- POST /api/chat/clear       — 清空对话历史 + KV 缓存
- GET  /api/status           — 系统状态（模型信息、GPU、KV缓存、设备档位）
- GET  /api/models/current   — 当前模型信息
- GET  /api/device/profile   — 完整设备画像（CPU/RAM/GPU/Disk/OS）
- POST /api/device/auto-configure — 应用设备自适应配置
- POST /api/chat/upload       — 上传文本文件（txt/md/csv/py/json/log）
- GET  /api/presets           — 预设问题列表
"""

import hashlib
import json
import logging
import re
import time
import sys
import os
import threading
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, cast

# 确保 src 目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from release_contract import (
    FIXED_RELEASE_ENV,
    PRODUCT_VERSION,
    RELEASE_PROFILE_NAME,
    release_profile_enforced,
)
from distributed_completion import (
    DistributedCompletionError,
    validate_distributed_completion,
)
from torch_runtime import LazyTorch, cuda_available, loaded_torch

torch = LazyTorch()


def _torch_cuda_available() -> bool:
    """Whether CUDA is usable, *without* requiring PyTorch to be installed.

    Status and model-listing endpoints must still answer in the L-tier environment
    (GGUF/llama.cpp, no torch), reporting "no CUDA" rather than raising there.
    """

    should_load = False
    try:
        active_engine = str(getattr(model_host, "engine_type", "") or "").lower()
        if active_engine == "pytorch":
            should_load = True
        else:
            import config as _cfg

            should_load = str(
                getattr(_cfg, "INFERENCE_ENGINE", "llama_cpp") or "llama_cpp",
            ).lower() in {"pytorch", "auto"}
    except Exception:
        pass
    return cuda_available(load=should_load)

from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException, Request,
                    UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from api_errors import coded_http_error, error_response_content
from model_api_access import require_model_api_source

if TYPE_CHECKING:  # torch-backed (D-tier); imported lazily so the L-tier can start.
    from paged_kv_cache import PagedKVCache
from multimodal import (
    build_openai_user_content,
    materialize_image_data_url,
    validate_image_data_urls,
)
from device_profiler import DeviceProfiler, get_profile
from scheduler import Scheduler as ClusterScheduler
from task_graph import (
    DEPENDENCY_FAILURES_KEY,
    StageSpec,
    TaskGraphCoordinator,
    TaskGraphError,
    TaskGraphUnavailable,
    WorkflowCancelled,
    WorkflowDeadlineExceeded,
    WorkflowExecutionError,
    WorkflowNotFound,
    dual_candidate_template,
)
from request_deadline import (
    REQUEST_DEADLINE_EXCEEDED,
    RequestDeadline,
    RequestDeadlineExceeded,
    coerce_request_deadline,
    hold_lock_until_deadline,
)
from task_journal import SQLiteTaskJournal, TaskJournalError
from task_provider import (
    LocalFullModelProvider,
    ModelIdentity,
    ProviderError,
    ProviderExecutionError,
    ProviderExecutor,
    StageRequest as ProviderStageRequest,
)
import model_config as mc
# ★ 2026-09-19：认证改为 monolith 内实现（抛弃 control-svc 反代）
import auth_app
import auth_service
from model_registry_validation import build_manifest, validate_model_artifact, write_manifest
from config import (
    MODEL_NAME, MODEL_PATH, QUANT_TYPE, USE_COMPILE,
    DEVICE, PAGE_SIZE, MAX_PAGE_NUM, MAX_SEQ_LEN, RUN_MODE,
    NODE_ROLE, NODE_ID, MAX_NODES, SERVER_IP, SERVER_PORT, API_PORT,
    TASK_GRAPH_ENABLED, TASK_GRAPH_MAX_RECORDS,
    TASK_GRAPH_MAX_PARALLEL_STAGES, TASK_GRAPH_JOURNAL_PATH,
    TASK_GRAPH_RETENTION_DAYS, TASK_GRAPH_RETENTION_MAX_RECORDS,
    TASK_WORKER_EXPERIMENTAL_ENABLED,
    PIPELINE_TIMEOUT,
    PIPELINE_RELAY_ENABLED,
    PIPELINE_RELAY_PROBE_ONLY,
)

# ★ 2026-10-04 裁定「一个节点只能是一种角色」：这两个开关此前从未出现在同一
#   条件里，且 `TASK_WORKER_EXPERIMENTAL_ENABLED` 还能运行时热切换。两者同时
#   开启会让集群同时存在 relay 宿主与 v3 stage worker ⇒ `_run_pipeline` 对混合
#   拓扑整体拒绝（`stage_offer_nodes != pipeline_nodes`），而调度侧仍把 relay
#   宿主的容量计为可用（`capacity_source="relay_exempt"`）⇒ 看得见、拿不到。
#   启动期直接 fail-closed，而不是留到运行时由「谁先生效」决定行为。
def _validate_pipeline_role_switches(
    *,
    relay_enabled: bool,
    relay_probe_only: bool,
    task_worker_enabled: bool,
) -> None:
    """Reject only an active A1/data-plane collision with Route A."""
    if relay_enabled and not relay_probe_only and task_worker_enabled:
        raise RuntimeError(
            "QLH_RELAY_ENABLED 与 QLH_TASK_WORKER_EXPERIMENTAL_ENABLED 不能同时开启："
            "一个节点只能是一种角色（relay 段宿主 或 v3 stage worker）。"
            "两者并存会让混合拓扑被整体拒绝。请只保留其中一个。"
        )


_validate_pipeline_role_switches(
    relay_enabled=PIPELINE_RELAY_ENABLED,
    relay_probe_only=PIPELINE_RELAY_PROBE_ONLY,
    task_worker_enabled=TASK_WORKER_EXPERIMENTAL_ENABLED,
)

# 主节点用户自持 SQLite；生产运行时不再加载远端 PostgreSQL 驱动。
import local_store as _local_store
from chat_context import (
    ConversationContextConflict,
    ConversationContextService,
    PreparedChatContext,
    TurnPersistenceResult,
)
# ★ 2026-10-07（DIST-NEXT-8）：请求相位/终态的单一来源 —— 取消、拒绝与链路错误
#   不再被 `fallback` 或 error 语义吞掉（详见 `src/request_outcome.py`）。
from request_outcome import (
    REASON_GENERATION_CANCELLED,
    REASON_REQUEST_FAILED,
    REASON_REQUEST_REFUSED,
    merge_phase_metrics as merge_request_outcome,
    request_phase_metrics as request_outcome_metrics,
)
import model_download_jobs
import model_search
from cluster_join import (
    JoinContractError,
    JoinGrantLedger,
    build_join_request,
    decode_join_request,
    decode_join_grant,
    encode_join_grant,
    encode_join_request,
    generate_join_keypair,
    issue_join_grant,
    load_join_private_key,
    verify_join_grant,
    verify_and_consume_join_grant,
)
from cluster_fence import ControlFence, ControlFenceError
from cluster_score import build_management_score_snapshot
from node_config import load_node_config, write_node_config

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_conversation_transaction_depth: ContextVar[int] = ContextVar(
    "conversation_transaction_depth", default=0,
)
_LOG_BUFFER_MAXLEN = 5000
_log_buffer: deque[dict] = deque(maxlen=_LOG_BUFFER_MAXLEN)
# P7: the aggregate endpoint must have a bounded fan-out and one total wait
# budget, independent of the number of online workers.
LOG_AGGREGATE_DEADLINE_SECONDS = 3.0
LOG_AGGREGATE_MAX_CONCURRENCY = 8
_log_buffer_lock = threading.RLock()
_log_buffer_total_seen = 0
_join_ledger_lock = threading.RLock()
_join_ledger_instance: JoinGrantLedger | None = None

# P4.5 fencing is opt-in until a voter set/certificate distribution path is
# configured.  When enabled, the middleware below covers control-plane writes;
# read/status routes remain available in read-only mode.
control_fence = ControlFence.from_environment()


def _get_join_ledger() -> JoinGrantLedger:
    """Use the same user-owned local SQLite file as the control plane."""
    global _join_ledger_instance
    with _join_ledger_lock:
        if _join_ledger_instance is None:
            _join_ledger_instance = JoinGrantLedger(_local_store.initialize_local_store())
        return _join_ledger_instance


def _join_endpoint_parts(endpoint: str) -> tuple[str, int]:
    """Split host:port while preserving IPv6 bracket notation."""
    from urllib.parse import urlsplit

    parsed = urlsplit(f"//{endpoint}")
    if not parsed.hostname or parsed.port is None:
        raise JoinContractError("master_endpoint is invalid", code="invalid_endpoint")
    return parsed.hostname, int(parsed.port)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get("-")
        return True


_request_id_filter = RequestIdFilter()


def _current_node_id_safe() -> str:
    try:
        return scheduler.get_effective_node_id()
    except Exception:
        try:
            from node_runtime import node_runtime
            return node_runtime.get_node_id()
        except Exception:
            return NODE_ID


def _current_device_ip_safe() -> str:
    try:
        return getattr(scheduler, "_lan_ip", "") or SERVER_IP
    except Exception:
        return SERVER_IP


class MemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _log_buffer_total_seen
        try:
            entry = {
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(record.created),
                ),
                "level": record.levelname,
                "levelno": record.levelno,
                "name": record.name,
                "message": record.getMessage(),
                "filename": record.filename,
                "lineno": record.lineno,
                "funcName": record.funcName,
                "request_id": getattr(record, "request_id", _request_id_ctx.get("-")),
                "node_id": _current_node_id_safe(),
                "device_ip": _current_device_ip_safe(),
                "thread": record.threadName,
            }
            if record.exc_info:
                entry["exc_text"] = self.format(record)
            with _log_buffer_lock:
                _log_buffer_total_seen += 1
                entry["seq"] = _log_buffer_total_seen
                _log_buffer.append(entry)
        except Exception:
            self.handleError(record)


def _close_logging_handlers(keep_memory: bool = False):
    """
    关闭并移除 root logger 上的 handlers，避免 Windows 下日志文件被占用。

    Args:
        keep_memory: 若为 True，保留 MemoryLogHandler 和 StreamHandler，
                     仅关闭文件类 handler（RotatingFileHandler）。
                     用于日志文件删除操作期间维持内存缓冲和终端输出。
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        is_file_handler = isinstance(handler, logging.FileHandler)
        if keep_memory and not is_file_handler and isinstance(
            handler, (logging.StreamHandler, MemoryLogHandler)
        ):
            continue  # 保留终端和内存 handler，确保删除期间日志不丢失
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def setup_logging():
    """配置日志：控制台输出 + RotatingFileHandler（5MB×5 滚动）。"""
    import logging.handlers
    from datetime import datetime
    from config import LOG_DIR, LOG_LEVEL

    os.makedirs(LOG_DIR, exist_ok=True)
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    _close_logging_handlers()

    # 控制台 handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)
    ch.addFilter(_request_id_filter)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] request_id=%(request_id)s %(message)s"))
    root.addHandler(ch)

    # 文件 handler（按日期 + 大小滚动）
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, f"qlh-{datetime.now():%Y-%m-%d}.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.addFilter(_request_id_filter)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] request_id=%(request_id)s %(name)s: %(message)s"))
    root.addHandler(fh)

    # 内存环形缓冲 handler（用于 /api/logs/recent，不持有文件句柄）
    mh = MemoryLogHandler()
    mh.setLevel(level)
    mh.addFilter(_request_id_filter)
    mh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] request_id=%(request_id)s %(name)s: %(message)s"))
    root.addHandler(mh)

    # uvicorn 日志也走文件
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).propagate = True

    root.setLevel(level)


setup_logging()
logger = logging.getLogger("api_server")

# ============================================================
# FastAPI 应用初始化
# ============================================================

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan 上下文管理器（替代废弃的 @app.on_event）

    ★ 2026-10-05：runtime 启动/关闭改为**进程级只执行一次**。
    `run_api_servers()` 在通配地址下为 `0.0.0.0` 与 `::` 各起一个
    `uvicorn.Server`（共享同一 `app`），uvicorn 会对每个 server 各跑一次
    lifespan。此前每次都重启 `_run_runtime_startup` 线程、每次 shutdown 都调
    `_shutdown_resources()`，导致进程内两套 TCPServer，且任一 server 关闭会
    连带停掉整个 runtime。现在用进程级引用计数共享一次启动，并只在最后一个
    lifespan 退出时收尾；后续 lifespan 不会重置 readiness。
    """
    global _runtime_startup_thread, _runtime_lifecycle_users
    with _runtime_lifecycle_lock:
        first = _runtime_lifecycle_users == 0
        _runtime_lifecycle_users += 1
        if first:
            _reset_runtime_readiness()
            _mark_process_ready()
            _runtime_startup_done.clear()
            _runtime_startup_thread = threading.Thread(
                target=_run_runtime_startup,
                name="runtime-startup",
                daemon=True,
            )
            _runtime_startup_thread.start()
    try:
        # The HTTP process can serve liveness/readiness while the slower
        # runtime components finish initializing in the background.
        yield
    finally:
        # ---- shutdown ----
        # 只有最后一个共享 lifespan 负责收尾。
        with _runtime_lifecycle_lock:
            _runtime_lifecycle_users = max(0, _runtime_lifecycle_users - 1)
            last = _runtime_lifecycle_users == 0
        if last:
            _runtime_startup_done.wait(timeout=30.0)
            await _shutdown_resources()


app = FastAPI(
    title="轻量化大模型分布式边缘推理优化系统",
    version=PRODUCT_VERSION,
    description="北京交通大学 · 大学生创新创业训练计划",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000",
                   "http://127.0.0.1:5173", "http://127.0.0.1:5174",
                   "http://127.0.0.1:8000", "http://localhost:8000",
                   "http://[::1]:5173", "http://[::1]:5174", "http://[::1]:3000",
                   "http://[::1]:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_API_SOURCE_EXEMPT_PATHS = frozenset({
    "/api/health",
    "/api/ready",
    "/api/bootstrap/info",
    "/api/bootstrap/first-connect",
})
_API_AUTH_EXEMPT_PATHS = frozenset({
    "/api/health",
    "/api/ready",
    "/api/bootstrap/info",
    "/api/bootstrap/first-connect",
    "/api/auth/capability",
    "/api/auth/login",
    # These are node-to-node contracts. They remain source-gated, but do not
    # require a human Bearer session when auth is enabled.
    "/api/cluster/join/request",
    "/api/cluster/join/consume",
    "/api/cluster/android/register",
    "/api/cluster/android/heartbeat",
    "/api/cluster/quorum/voter",
})


def _boundary_response(status_code: int, detail: object) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def _is_control_write_path(request: Request) -> bool:
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return False
    path = request.url.path
    if path.startswith("/api/cluster/queue"):
        return True
    if path.startswith("/api/cluster/config") or path.startswith("/api/cluster/layers"):
        return True
    if path.startswith("/api/cluster/model-runtime") or path.startswith("/api/cluster/qwen3"):
        return True
    if path.startswith("/api/cluster/transfer-master") or path.startswith("/api/cluster/spare-master"):
        return True
    if path.startswith("/api/cluster/reset-identity"):
        return True
    if "/api/cluster/nodes/" in path and path.endswith("/deregister"):
        return True
    return False


@app.middleware("http")
async def api_boundary_middleware(request: Request, call_next):
    """Apply the common source/auth boundary before route validation or work.

    Individual handlers still keep their role and node-state checks. This
    layer prevents a newly added API route from accidentally bypassing the
    common network boundary, while the explicit bootstrap exceptions preserve
    first-connect and node heartbeat protocols.
    """
    path = request.url.path
    if request.method == "OPTIONS" or not path.startswith("/api/"):
        return await call_next(request)

    if path not in _API_SOURCE_EXEMPT_PATHS:
        try:
            require_model_api_source(request)
        except HTTPException as exc:
            return _boundary_response(exc.status_code, exc.detail)

    if (
        auth_service.auth_required()
        and path not in _API_AUTH_EXEMPT_PATHS
        and not (path == "/api/users" and request.method == "POST"
                 and auth_service.is_bootstrap_open())
    ):
        authorization = request.headers.get("Authorization")
        if auth_service.resolve_bearer(authorization) is None:
            return _boundary_response(
                401,
                {"code": "auth_required", "message": "需要登录（Bearer token）"},
            )

    if _is_control_write_path(request) and control_fence.enabled:
        try:
            control_fence.admit_http(
                request.headers,
                action=f"http.{request.method.lower()}.{request.url.path}",
                request_id=_request_id_ctx.get("-"),
            )
        except ControlFenceError as exc:
            control_fence.clear_context()
            status = 503 if exc.code == "control_fence_unavailable" else 409
            return _boundary_response(
                status,
                {"code": exc.code, "message": str(exc)},
            )

    try:
        return await call_next(request)
    except ControlFenceError as exc:
        status = 503 if exc.code == "control_fence_unavailable" else 409
        return _boundary_response(status, {"code": exc.code, "message": str(exc)})
    finally:
        if control_fence.enabled:
            control_fence.clear_context()

def _normalize_request_id(value: str | None) -> str:
    if not value:
        return uuid.uuid4().hex
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "", value.strip())
    if not cleaned:
        return uuid.uuid4().hex
    return cleaned[:64]


@app.middleware("http")
async def request_id_logging_middleware(request: Request, call_next):
    request_id = _normalize_request_id(request.headers.get("X-Request-ID"))
    token = _request_id_ctx.set(request_id)
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.error(
            "event=http_request_error request_id=%s method=%s path=%s status=%s",
            request_id, request.method, request.url.path, status_code,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误，请查看后端日志", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id, request.method, request.url.path, status_code, duration_ms,
        )
        _request_id_ctx.reset(token)


@app.exception_handler(HTTPException)
async def http_exception_with_request_id(request: Request, exc: HTTPException):
    request_id = _request_id_ctx.get("-")
    headers = dict(exc.headers or {})
    headers["X-Request-ID"] = request_id
    if exc.status_code >= 500:
        logger.error(
            "event=http_exception request_id=%s method=%s path=%s status=%s detail=%s",
            request_id, request.method, request.url.path, exc.status_code, exc.detail,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response_content(exc, request_id=request_id),
        headers=headers,
    )

# ============================================================
# 全局状态
# ============================================================

# 推理宿主单例（阶段 0.2/0.4：api_server 与 scheduler 共享；model_manager 为
# 兼容名，属性读写代理到内部 ModelManager）
from model_host import SchedulerCallbackSet, model_host
from koakuma_engine import (
    Capability,
    backend_capabilities,
    backend_id_for,
    registered_backends,
    runtime_supports,
)
from model_load_resolver import (
    ModelLoadFacts,
    ModelLoadResolution,
    ModelLoadResolutionError,
    OP_DISTRIBUTED_LOAD,
    OP_DYNAMIC_LAYER_RANGE,
    OP_FULL_MODEL,
    effective_pytorch_cuda_available,
    preferred_engine_for_artifacts,
    resolve_model_load,
)

model_manager = model_host
kv_cache: "Optional[PagedKVCache]" = None  # PagedKVCache is D-tier (torch) only
_chat_context = ConversationContextService(
    load_history=lambda session_id: _load_conversation_history(session_id),
    persist_turn=lambda session_id, user, assistant, metrics: _persist_conversation_turn(
        session_id, user, assistant, metrics,
    ),
)
active_session_id: Optional[str] = None           # compatibility facade
session_histories: dict[str, list[dict]] = _chat_context.histories
conversation_stats: dict = {                    # 累计对话统计（实际消耗追踪）
    "total_prompt_tokens": 0,
    "total_generated_tokens": 0,
    "total_time_seconds": 0.0,
    "rounds": 0,
}
device_profile: Optional[dict] = None           # 设备画像缓存
_device_profile_ready = threading.Event()
_device_profile_started = False

# Process liveness and inference-runtime readiness are intentionally separate.
# A model is not part of this gate: the application must remain usable so the
# user can inspect/download/select a model after entering the TUI.
_RUNTIME_COMPONENTS = ("local_store", "scheduler", "device_profile")
_runtime_readiness_lock = threading.RLock()
_runtime_readiness: dict[str, Any] = {
    "process_ready": False,
    "ready": False,
    "status": "starting",
    "components": {name: False for name in _RUNTIME_COMPONENTS},
    "error": None,
    "started_at": None,
    "ready_at": None,
}
_runtime_startup_done = threading.Event()
_runtime_startup_thread: Optional[threading.Thread] = None
# ★ 2026-10-05：runtime 启动/关闭的**进程级**引用计数与锁。
#   `run_api_servers()` 在通配地址下会为 `0.0.0.0` 与 `::` 各起一个
#   `uvicorn.Server`（共享同一 `app`），uvicorn 因此对每个 server 各跑一次
#   lifespan。此前 `_lifespan` 每次都重启 `_run_runtime_startup` 线程、且每次
#   shutdown 都调 `_shutdown_resources()` ⇒ 进程内出现两套 TCPServer（已在
#   `scheduler.start()` 侧止血），且任一 server 关闭会连带停掉整个 runtime。
#   现在只让首个 lifespan 启动、最后一个 lifespan 关闭，后续 lifespan 复用运行时。
_runtime_lifecycle_lock = threading.Lock()
_runtime_lifecycle_users = 0


def _reset_runtime_readiness() -> None:
    with _runtime_readiness_lock:
        _runtime_readiness.update(
            process_ready=False,
            ready=False,
            status="starting",
            components={name: False for name in _RUNTIME_COMPONENTS},
            error=None,
            started_at=time.time(),
            ready_at=None,
        )


def _mark_process_ready() -> None:
    with _runtime_readiness_lock:
        _runtime_readiness["process_ready"] = True


def _mark_runtime_component(
    name: str, ready: bool, *, error: Optional[str] = None,
) -> None:
    if name not in _RUNTIME_COMPONENTS:
        return
    with _runtime_readiness_lock:
        components = _runtime_readiness["components"]
        components[name] = bool(ready)
        if error:
            _runtime_readiness["error"] = str(error)
        all_ready = all(components.values())
        _runtime_readiness["ready"] = all_ready
        if all_ready:
            _runtime_readiness["status"] = "ready"
            _runtime_readiness["error"] = None
            _runtime_readiness["ready_at"] = time.time()
        elif error:
            _runtime_readiness["status"] = "degraded"


def _mark_runtime_startup_failure(error: BaseException) -> None:
    with _runtime_readiness_lock:
        _runtime_readiness["ready"] = False
        _runtime_readiness["status"] = "failed"
        _runtime_readiness["error"] = f"{type(error).__name__}: {error}"


def _runtime_readiness_snapshot() -> dict[str, Any]:
    with _runtime_readiness_lock:
        return {
            **_runtime_readiness,
            "components": dict(_runtime_readiness["components"]),
        }

# 调度器（单机 / 分布式模式共用）
scheduler: ClusterScheduler = ClusterScheduler()
scheduler.set_control_fence(control_fence)
# HA-ROLE-AUTO-01: explicit ``NODE_ROLE=auto`` must construct the same
# controller used by scheduler startup. Configuration failures stay read-only;
# they are surfaced in ``/api/cluster/my-role`` rather than becoming a static
# master by accident.
auto_role_runtime = None
if NODE_ROLE == "auto":
    try:
        from cluster_auto_role_runtime import install_auto_role_controller

        auto_role_runtime = install_auto_role_controller(scheduler, control_fence)
    except Exception as exc:
        logger.error(
            "auto role wiring unavailable; scheduler will remain read-only: %s",
            getattr(exc, "code", type(exc).__name__),
        )
    # ``install_auto_role_controller`` may strengthen the initially optional
    # fence with the auto-role authority. Keep middleware and route modules on
    # that same object, otherwise API writes would bypass the runtime fence.
    if scheduler._control_fence is not control_fence:
        control_fence = scheduler._control_fence
_task_graph_runtime_lock = threading.RLock()
_RUNTIME_FEATURE_ENV_NAMES = frozenset(FIXED_RELEASE_ENV)


def _apply_runtime_feature_settings(
    requested: Mapping[str, bool],
) -> dict[str, bool]:
    '''Validate hot feature changes against the immutable release profile.'''
    normalized: dict[str, bool] = {}
    for env_name, value in requested.items():
        if env_name not in _RUNTIME_FEATURE_ENV_NAMES:
            raise ValueError(f'unknown runtime feature setting: {env_name}')
        if not isinstance(value, bool):
            raise TypeError(f'runtime feature setting must be bool: {env_name}')
        normalized[env_name] = value

    if not release_profile_enforced():
        return normalized

    for env_name, value in normalized.items():
        requested_value = '1' if value else '0'
        required_value = FIXED_RELEASE_ENV[env_name]
        if requested_value == required_value:
            continue
        raise coded_http_error(
            409,
            'RELEASE_PROFILE_FEATURE_LOCKED',
            {
                'code': 'RELEASE_PROFILE_FEATURE_LOCKED',
                'message': (
                    f'release profile {RELEASE_PROFILE_NAME!r} fixes '
                    f'{env_name}={required_value}; requested {requested_value}'
                ),
                'feature': env_name,
                'requested': requested_value,
                'required': required_value,
                'release_profile': RELEASE_PROFILE_NAME,
            },
        )
    return normalized


def _sync_task_worker_runtime_module(enabled: bool) -> None:
    """Synchronize the flag with the module owning the live scheduler."""
    import scheduler as scheduler_module

    scheduler_modules = {
        scheduler_module,
        sys.modules.get(type(scheduler).__module__),
    }
    for module in scheduler_modules:
        if module is not None:
            module.TASK_WORKER_EXPERIMENTAL_ENABLED = bool(enabled)


# Keep a persisted feature flag effective even when package and legacy module
# imports resolve to different scheduler module names during server startup.
_sync_task_worker_runtime_module(TASK_WORKER_EXPERIMENTAL_ENABLED)


def _task_graph_feature_settings() -> dict[str, Any]:
    """Return the user-owned task-graph switches and their control state."""
    role = scheduler._effective_role()
    return {
        "task_graph_enabled": bool(TASK_GRAPH_ENABLED),
        "task_worker_experimental_enabled": bool(
            TASK_WORKER_EXPERIMENTAL_ENABLED
        ),
        "can_toggle": role == "master",
        "physical_validation_pending": bool(
            TASK_WORKER_EXPERIMENTAL_ENABLED
            and scheduler.get_task_worker_protocol_status().get(
                "admission_state"
            ) == "n2_4_experimental_physical_validation_pending"
        ),
    }


def _persist_task_graph_feature_settings(
    *, task_graph_enabled: bool, task_worker_experimental_enabled: bool,
) -> None:
    data = load_node_config()
    features = data.get("features") if isinstance(data.get("features"), dict) else {}
    data["features"] = {
        **features,
        "task_graph_enabled": bool(task_graph_enabled),
        "task_worker_experimental_enabled": bool(
            task_worker_experimental_enabled
        ),
    }
    write_node_config(data)


def _set_task_graph_runtime_settings(
    *, task_graph_enabled: bool | None = None,
    task_worker_experimental_enabled: bool | None = None,
) -> dict[str, Any]:
    """Apply task-graph switches to already imported modules.

    Worker dispatch remains independently gated by readiness/admission; this
    setter only opens the experimental control-plane switch.
    """
    global TASK_GRAPH_ENABLED, TASK_WORKER_EXPERIMENTAL_ENABLED
    with _task_graph_runtime_lock:
        import config as cfg

        next_graph = bool(TASK_GRAPH_ENABLED if task_graph_enabled is None else task_graph_enabled)
        next_worker = bool(
            TASK_WORKER_EXPERIMENTAL_ENABLED
            if task_worker_experimental_enabled is None
            else task_worker_experimental_enabled
        )
        validated = _apply_runtime_feature_settings({
            'QLH_TASK_GRAPH_ENABLED': next_graph,
            'QLH_TASK_WORKER_EXPERIMENTAL_ENABLED': next_worker,
        })
        next_graph = validated['QLH_TASK_GRAPH_ENABLED']
        next_worker = validated['QLH_TASK_WORKER_EXPERIMENTAL_ENABLED']
        graph_changed = next_graph != bool(TASK_GRAPH_ENABLED)
        TASK_GRAPH_ENABLED = next_graph
        TASK_WORKER_EXPERIMENTAL_ENABLED = next_worker
        cfg.TASK_GRAPH_ENABLED = next_graph
        cfg.TASK_WORKER_EXPERIMENTAL_ENABLED = next_worker
        _sync_task_worker_runtime_module(next_worker)

        if graph_changed:
            old_coordinator = globals().get("task_graph_coordinator")
            new_coordinator = _create_task_graph_coordinator()
            globals()["task_graph_coordinator"] = new_coordinator
            if old_coordinator is not None and old_coordinator is not new_coordinator:
                try:
                    old_coordinator.close()
                except Exception:
                    logger.warning("旧任务图协调器关闭失败", exc_info=True)
            if next_graph and scheduler._effective_role() == "master":
                _ensure_local_task_provider()

        _persist_task_graph_feature_settings(
            task_graph_enabled=next_graph,
            task_worker_experimental_enabled=next_worker,
        )
        return _task_graph_feature_settings()


def _create_task_graph_coordinator() -> TaskGraphCoordinator:
    if not TASK_GRAPH_ENABLED:
        return TaskGraphCoordinator(
            max_records=TASK_GRAPH_MAX_RECORDS,
            max_parallel_stages=TASK_GRAPH_MAX_PARALLEL_STAGES,
        )
    journal = None
    try:
        journal = SQLiteTaskJournal(TASK_GRAPH_JOURNAL_PATH)
        coordinator = TaskGraphCoordinator(
            max_records=TASK_GRAPH_MAX_RECORDS,
            journal=journal,
            max_parallel_stages=TASK_GRAPH_MAX_PARALLEL_STAGES,
        )
        recovery = coordinator.recover_persisted_workflows()
        cleanup = coordinator.cleanup_journal(
            max_age_days=TASK_GRAPH_RETENTION_DAYS,
            max_records=TASK_GRAPH_RETENTION_MAX_RECORDS,
        )
        if recovery.get("recovered_workflows", 0):
            logger.warning("任务图启动恢复完成: %s", recovery)
        if cleanup.get("deleted_workflows", 0):
            logger.info("任务图 journal 保留清理完成: %s", cleanup)
        return coordinator
    except (TaskJournalError, TaskGraphUnavailable) as exc:
        if journal is not None:
            try:
                journal.close()
            except Exception:
                pass
        coordinator = TaskGraphCoordinator(
            max_records=TASK_GRAPH_MAX_RECORDS,
            max_parallel_stages=TASK_GRAPH_MAX_PARALLEL_STAGES,
            availability_error=f"task journal unavailable: {exc}",
        )
        logger.error(
            "任务图 journal 初始化失败，任务图已禁用: %s",
            exc,
            exc_info=True,
        )
        return coordinator


class _LazyTaskGraphCoordinator:
    """惰性提供 TaskGraph writer —— import 副作用不得争用生产 journal 锁。

    单写者独占锁必须只由「真正承担任务图写服务」的进程持有。探针、
    pytest 冷启动子进程以及 Windows ``multiprocessing`` spawn worker 往往只是
    **import** 本模块（拿一些常量/工具函数），并不需要执行任务链；它们若在
    import 阶段就去获取 journal 锁，就会和已经在跑的生产服务争锁，进而被错误
    记录为「任务图 journal 初始化失败，任务图已禁用」。

    因此把 writer 的物化（获取独占锁 + 启动恢复 + 保留清理）推迟到**首次真实
    使用**：真正执行任务链、查询任务工作流状态或运行时热切换时才初始化。
    失败关闭（fail-closed）语义保持不变：若此刻锁已被另一个进程持有，物化出的
    coordinator 会携带 availability_error，后续任务链执行全部经
    ``journal_status()['available'] == False`` 被拒绝，绝不出现双写者。
    """

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self._lock = threading.RLock()
        self._coordinator: Optional[TaskGraphCoordinator] = None

    def _ensure(self) -> TaskGraphCoordinator:
        with self._lock:
            if self._coordinator is None:
                self._coordinator = self._factory()
            return self._coordinator

    def _is_materialized(self) -> bool:
        """True 表示 writer 已被真实构造（即 journal 锁已被本进程持有）。"""
        with self._lock:
            return self._coordinator is not None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ensure(), name)

    def close(self) -> None:
        coordinator = None
        with self._lock:
            if self._coordinator is not None:
                coordinator, self._coordinator = self._coordinator, None
        if coordinator is not None:
            coordinator.close()

    def __repr__(self) -> str:
        return f"<_LazyTaskGraphCoordinator materialized={self._is_materialized()}>"


# 模块 import 阶段只创建惰性代理；journal 锁的获取推迟到首次真实使用，
# 避免 Windows/pytest 冷启动子进程在 import api_server 时与生产 writer 争锁。
task_graph_coordinator: TaskGraphCoordinator = cast(
    TaskGraphCoordinator,
    _LazyTaskGraphCoordinator(_create_task_graph_coordinator),
)
_task_graph_execution_slot = threading.BoundedSemaphore(1)
_generation_registry_lock = threading.RLock()
_generation_cancel_events: dict[str, threading.Event] = {}
_generation_pending_cancellations: dict[str, float] = {}
_GENERATION_ID_PATTERN = re.compile(r"^gen_[A-Za-z0-9_-]{8,96}$")
_GENERATION_REGISTRY_LIMIT = 512


class ChatGenerationCancelled(RuntimeError):
    def __init__(self, generation_id: str):
        self.generation_id = generation_id
        super().__init__(f"generation {generation_id} cancelled")


def _prune_pending_generations_locked() -> None:
    cutoff = time.time() - 300.0
    for generation_id, created_at in list(_generation_pending_cancellations.items()):
        if created_at < cutoff:
            _generation_pending_cancellations.pop(generation_id, None)


def _register_generation(generation_id: Optional[str]) -> tuple[str, threading.Event]:
    resolved_id = generation_id or f"gen_{uuid.uuid4().hex}"
    if not _GENERATION_ID_PATTERN.fullmatch(resolved_id):
        raise HTTPException(400, "generation_id 格式无效")
    event = threading.Event()
    with _generation_registry_lock:
        _prune_pending_generations_locked()
        if resolved_id in _generation_cancel_events:
            raise HTTPException(409, f"generation_id 已在执行: {resolved_id}")
        if _generation_pending_cancellations.pop(resolved_id, None) is not None:
            event.set()
        _generation_cancel_events[resolved_id] = event
    return resolved_id, event


def _unregister_generation(generation_id: str, event: threading.Event) -> None:
    with _generation_registry_lock:
        if _generation_cancel_events.get(generation_id) is event:
            _generation_cancel_events.pop(generation_id, None)


def _request_generation_cancel(generation_id: str) -> str:
    if not _GENERATION_ID_PATTERN.fullmatch(generation_id):
        raise HTTPException(400, "generation_id 格式无效")
    with _generation_registry_lock:
        event = _generation_cancel_events.get(generation_id)
        if event is not None:
            event.set()
            return "cancel_requested"
        _prune_pending_generations_locked()
        if len(_generation_pending_cancellations) >= _GENERATION_REGISTRY_LIMIT:
            oldest = min(
                _generation_pending_cancellations,
                key=lambda item: _generation_pending_cancellations[item],
            )
            _generation_pending_cancellations.pop(oldest, None)
        _generation_pending_cancellations[generation_id] = time.time()
        return "cancel_pending"


def _raise_if_generation_cancelled(
    cancel_event: Optional[threading.Event], generation_id: Optional[str],
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ChatGenerationCancelled(generation_id or "gen_unknown")


def _serialized_conversation_mutation(func):
    """Run synchronous conversation mutations under the full-chat lock."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        depth = _conversation_transaction_depth.get(0)
        if depth > 0:
            with model_host.full_chat_execution_lock:
                return func(*args, **kwargs)
        with _chat_context.transaction():
            token = _conversation_transaction_depth.set(depth + 1)
            try:
                with model_host.full_chat_execution_lock:
                    return func(*args, **kwargs)
            finally:
                _conversation_transaction_depth.reset(token)
    return wrapper


def _refresh_pipeline_layer_config() -> None:
    """主节点模型变化后重新下发层配置，并使旧 ACK 失效。"""
    try:
        if scheduler._effective_role() == "master":
            request_sync = getattr(
                scheduler, "request_authoritative_layer_sync", None,
            )
            if callable(request_sync):
                request_sync()
            else:
                scheduler.push_layer_config_to_clients()
    except Exception as e:
        # 模型本身已经加载成功；同步失败时保持流水线 not-ready，后续请求
        # 会安全回退到主节点本地推理。
        logger.warning(f"模型加载后刷新流水线层配置失败: {e}", exc_info=True)


def _run_exclusive_model_change(
    change, prepare=None, *, release_worker_reservation: bool = False,
):
    """Publish control-plane invalidation only after the model change succeeds."""
    with model_host.full_chat_execution_lock:
        with scheduler._inference_lock:
            with scheduler._layer_execution_lock:
                result = change()
                if isinstance(result, dict) and result.get("success") is False:
                    return result

                begin_transition = getattr(
                    scheduler, "_begin_layer_config_model_change", None,
                )
                transition_started = callable(begin_transition)
                if transition_started:
                    begin_transition()
                try:
                    if prepare is not None:
                        prepare()
                    invalidate_transaction = getattr(
                        scheduler, "_invalidate_pipeline_load_transaction", None,
                    )
                    if callable(invalidate_transaction):
                        invalidate_transaction(
                            reason_code="pipeline_model_changed",
                            reason="local model replacement committed",
                        )
                    with scheduler._layer_config_lock:
                        scheduler._worker_assignments.clear()
                        scheduler._layer_config_retry_state.clear()
                        scheduler._pipeline_load_transaction = None
                        scheduler._active_pipeline_capacity_plan = None
                        scheduler._active_layer_config = None
                        scheduler._last_layer_config_ack_payload = None
                        scheduler._local_pipeline_steps.clear()
                    if release_worker_reservation:
                        release = getattr(
                            scheduler,
                            "release_pipeline_worker_for_local_model",
                            None,
                        )
                        if callable(release):
                            release()
                        else:
                            scheduler._pipeline_worker_reserved = False
                    _refresh_pipeline_layer_config()
                finally:
                    if transition_started:
                        end_transition = getattr(
                            scheduler, "_end_layer_config_model_change", None,
                        )
                        if callable(end_transition):
                            end_transition()
                return result


# ============================================================
# SIGTERM 优雅关闭（Linux systemd / 容器环境）
# 注意：uvicorn 内置了自己的 SIGTERM/SIGINT 处理器，在 uvicorn.run() 中
# 自动处理优雅关闭。不在此处注册模块级信号处理器，因为会被 uvicorn 覆盖。


# ============================================================
# 启动事件 — 设备检测（通过 lifespan 调用）
# ============================================================

def _run_runtime_startup() -> None:
    """Run the slower runtime initialization outside the HTTP event loop."""
    try:
        _startup_device_detection()
    except BaseException as exc:  # keep liveness available and report /ready
        logger.critical("runtime startup failed: %s", exc, exc_info=True)
        _mark_runtime_startup_failure(exc)
    finally:
        _runtime_startup_done.set()


def _startup_device_detection():
    """Initialize local runtime components and detect slower hardware in background."""
    global device_profile, _device_profile_started
    active_scheduler: ClusterScheduler = globals()["scheduler"]

    _mark_runtime_component("local_store", False)
    try:
        sqlite_path = _local_store.initialize_local_store()
        logger.info("主节点 SQLite 已就绪: %s", sqlite_path)
        _mark_runtime_component("local_store", True)
    except Exception as exc:
        logger.critical("主节点 SQLite 不可写，拒绝启动: %s", exc)
        _mark_runtime_component("local_store", False, error=str(exc))
        raise RuntimeError("主节点 SQLite 初始化失败") from exc

    def _detect_device_profile() -> None:
        global device_profile
        try:
            profiler = get_profile()
            device_profile = profiler.to_dict()
            active_scheduler.update_local_device_profile(device_profile)
            logger.info(
                f"🚀 设备检测完成: tier={profiler.tier.value} "
                f"score={profiler.score:.1f}/100 | "
                f"CPU={profiler.cpu.physical_cores}核 RAM={profiler.ram.total_gb}GB "
                f"GPU={profiler.gpu.name}"
            )
            logger.info(f"   推荐配置: {profiler.recommend_config()['description']}")
            for warning in device_profile.get("warnings", []):
                logger.warning(f"   {warning}")
            _mark_runtime_component("device_profile", True)
        except Exception as e:
            logger.error(f"设备检测失败: {e}")
            device_profile = None
            _mark_runtime_component("device_profile", False, error=str(e))
        finally:
            _device_profile_ready.set()

    if not _device_profile_started:
        _device_profile_started = True
        threading.Thread(
            target=_detect_device_profile,
            name="device-profile",
            daemon=True,
        ).start()
    elif device_profile is not None:
        _mark_runtime_component("device_profile", True)
    elif _device_profile_ready.is_set():
        _mark_runtime_component("device_profile", False, error="device profile unavailable")

    # 初始化调度器（单机模式下不启动 TCP 监听）
    try:
        active_scheduler.start()
        logger.info(f"调度器已初始化: mode={RUN_MODE}")
        _mark_runtime_component("scheduler", True)
    except Exception as e:
        logger.error(f"调度器初始化失败: {e}")
        _mark_runtime_component("scheduler", False, error=str(e))

    # L5: 启动日志保留策略清理线程
    try:
        _start_log_retention_thread()
    except Exception as e:
        logger.warning(f"日志保留线程启动失败: {e}")

    logger.info("主节点 SQLite 已就绪")

    # P3: 启动审查工单过期检查后台线程（仅 master，每 5 分钟）
    try:
        if active_scheduler._effective_role() == "master":
            import threading as _th2
            def _review_expire_loop():
                import time as _time
                from review import ReviewManager
                _time.sleep(120)  # 启动后等 2 分钟再开始（避免空跑）
                while getattr(active_scheduler, '_running', True):
                    try:
                        ReviewManager().resolve_expired()
                    except Exception:
                        pass
                    _time.sleep(300)  # 每 5 分钟检查一次
            _t = _th2.Thread(target=_review_expire_loop, daemon=True, name="review-expire")
            _t.start()
            logger.info("⏳ 审查工单过期检查线程已启动 (master)")
    except Exception as e:
        logger.warning(f"审查工单过期检查启动失败: {e}")


# ============================================================
# 关闭事件 — 资源清理（通过 lifespan 调用）
# ============================================================

async def _shutdown_resources():
    """应用关闭时清理调度器、推理引擎与 TCP 服务。"""
    active_scheduler: ClusterScheduler = globals()["scheduler"]
    try:
        with model_host.full_chat_execution_lock:
            task_graph_coordinator.close()
    except Exception as e:
        logger.warning(f"任务图 journal 关闭异常: {e}")
    # 1. 停止调度器（关闭 TCP 连接，注销从节点）
    try:
        active_scheduler.stop()
        logger.info("调度器已停止")
    except Exception as e:
        logger.warning(f"调度器停止异常: {e}")

# ============================================================
# 优雅退出（TUI / 外部命令触发）
#
# 优先级：
#   1. 通过 `python src/api_server.py` 启动时注册的 uvicorn.Server 实例
#      → 设置 should_exit，触发 uvicorn 内置优雅关闭（lifespan shutdown
#        会执行 _shutdown_resources，跨平台可靠）
#   2. POSIX：os.kill(SIGTERM) → uvicorn 信号处理器触发同样的优雅关闭
#   3. Windows 兜底（`python -m uvicorn` 直启、无 server 引用）：
#      直接执行资源清理后退出进程
# ============================================================

_uvicorn_servers: list = []           # 通过 register_uvicorn_server() 注册（双栈时多个）
_SHUTDOWN_TOKEN = os.environ.get("QLH_SHUTDOWN_TOKEN", "") or ""


def register_uvicorn_server(server) -> None:
    """供 `python src/api_server.py` 入口注册 uvicorn.Server 实例（支持双栈多实例）。"""
    global _uvicorn_servers
    if server is not None and server not in _uvicorn_servers:
        _uvicorn_servers.append(server)


def _graceful_exit() -> None:
    """在后台线程中触发后端优雅退出（先等响应返回）。"""
    time.sleep(0.5)
    servers = list(_uvicorn_servers)
    if servers:
        for server in servers:
            server.should_exit = True
        logger.info("event=system_shutdown trigger=uvicorn_should_exit")
        return
    if os.name != "nt":
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
        logger.info("event=system_shutdown trigger=signal_sigterm")
        return
    # Windows 兜底：直接清理资源（uvicorn 主循环不感知，需自行收尾）
    import asyncio
    logger.warning("event=system_shutdown trigger=direct_cleanup (未注册 uvicorn server)")
    try:
        asyncio.run(_shutdown_resources())
    except Exception as e:
        logger.warning(f"优雅退出清理异常: {e}")
    os._exit(0)


class SystemShutdownRequest(BaseModel):
    reason: str = Field(default="", description="退出原因说明（写入日志）")




# ============================================================
# Pydantic 模型
# ============================================================

class LoadModelRequest(BaseModel):
    engine: str = Field(
        default="llama_cpp",
        description="推理引擎: llama_cpp (GGUF, 推荐) | pytorch (Safetensors) | island (TP 孤岛) | auto",
    )
    quant_type: str = Field(
        default="int4",
        description="PyTorch 量化精度: fp16 | int8 | int4（llama_cpp 引擎忽略此参数）",
    )
    use_compile: bool = Field(
        default=False,
        description="是否开启 torch.compile 算子融合（仅 PyTorch FP16 有效）",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="模型唯一标识（P3多模型支持）。不传则使用默认 Qwen-1.8B。",
    )


class PreparePipelineModelRequest(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=128)
    quant_type: str = Field(
        default="fp16",
        description="层段运行精度请求；第一期执行器仍以实际设备 dtype 为准",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息", min_length=1)
    image_data_urls: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="PNG/JPEG/WebP base64 data URL；external_api 最多四张，本地 MTMD 仅一张",
    )
    session_id: Optional[str] = Field(default=None, description="会话ID，为空时使用当前活跃会话")
    max_new_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    show_thinking: bool = Field(default=False, description="启用深度思考展示")
    enable_thinking: Optional[bool] = Field(
        default=None,
        description=(
            "★ 深度思考「开关」（区别于 show_thinking 的「展示」）："
            "True=强制开启、False=强制关闭、None=沿用模型模板默认。"
            "对支持关闭思考的模型（如 Qwen3，模板 qwen3_chat_v1）传 False 可真正阻止其生成 "
            "thinking 内容（省算力），而不是事后把已生成的内容丢掉。"
            "注意：仅对声明了 enable_thinking 模板的模型生效，其余模型忽略。"
        ),
    )
    streaming_mode: str = Field(
        default="full",
        description="流式模式（仅 /api/chat/stream 生效）: full=假流式完整功能（默认） | fast=真流式逐token | interactive=真流式逐token + T9 事件契约；三者共用历史、裁剪和完成时会话提交",
    )
    routing_preference: Literal[
        "auto", "local_only", "distributed_preferred", "distributed_required"
    ] = Field(
        default="auto",
        description="请求级路由偏好（T9 契约）: auto=沿用集群配置与 scheduler 决策 | local_only=仅主节点本地执行 | distributed_preferred=优先分布式，不可用允许本地回退 | distributed_required=无合格分布式路径时明确失败",
    )
    client_node_id: Optional[str] = Field(default=None, description="请求来源节点 ID（Android/PC 客户端上报）")
    client_node_type: Optional[str] = Field(default=None, description="请求来源节点类型: pc | android")
    client_mode: Optional[str] = Field(default=None, description="请求来源模式: thin | full")
    client_app_variant: Optional[str] = Field(default=None, description="请求来源 App variant: full | lite")
    execution_mode: Literal["auto", "task_graph"] = Field(
        default="auto",
        description="执行模式: auto=现有聊天路由 | task_graph=固定任务链实验",
    )
    task_graph_template: Literal["dual_candidate"] = Field(
        default="dual_candidate",
        description="任务链模板；首期仅支持双候选校验",
    )
    task_graph_remote_stage: Literal[
        "", "candidate_a", "candidate_b", "aggregate"
    ] = Field(
        default="",
        description="N2.1 手动指定唯一远端 Stage；为空时全部本地执行",
    )
    task_graph_remote_provider_id: str = Field(
        default="",
        max_length=64,
        description="N2.1 显式远端 Provider ID；不参与自动选择",
    )
    task_graph_auto_remote: bool = Field(
        default=False,
        description="N2.3 自动为无副作用候选 Stage 选择 PC Full Worker，并允许受控回退本地",
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="客户端预生成的 wf_ 工作流 ID，用于执行期间查询和取消",
    )
    generation_id: Optional[str] = Field(
        default=None,
        description="客户端预生成的 gen_ 执行 ID，用于所有聊天模式协作取消",
    )
    task_graph_inferred: bool = Field(
        default=False,
        description=(
            "`#29` 诊断位：`execution_mode` 是**由 N2.1 显式字段推断**而来，"
            "不是用户直接指定的 `task_graph`。"
        ),
    )
    allow_external: bool = Field(
        default=False,
        description=(
            "路线 B 数据作用域按请求授权：允许本请求路由到外部推理服务"
            "（QLH_EXTERNAL_*）。缺省 False——旧客户端行为不变，数据不出集群。"
        ),
    )
    prefer_external: bool = Field(
        default=False,
        description=(
            "路线 B：优先使用外部推理服务（仍受 QLH_EXTERNAL_DATA_SCOPE "
            "作用域门控约束；deny 档位下即使置 true 也不外发）。"
        ),
    )

    @model_validator(mode="after")
    def resolve_task_graph_mode(self):
        """`#29`：`execution_mode=auto` 时按 **N2.1 显式字段**推断是否走任务图。

        为什么要按「显式」判：`task_graph_template` 的默认值就是 `"dual_candidate"`
        （`src/inference_service/protocol.py` 与这里同款），**光看它有值无法区分**用户
        是否真的指定过 —— 得用 pydantic 的 `model_fields_set`。

        这里只做**推断**（显式给了 N2.1 字段 ⇒ 意图明确是任务图，把 `auto` 升级为
        `task_graph`）。这些字段此前只在 `execution_mode == "task_graph"` 分支内才被
        读到，等于用户写了也不生效。

        「只给模板、没有任何 N2.1 字段」那一档**不在这里**拒绝 —— 它需要带
        `reason_code` 的 400 响应，而 validator 抛不出 `coded_http_error`；
        该判定在路由层（`routes_chat`）做。
        """
        explicit_n21 = (
            self.task_graph_auto_remote
            or bool(str(self.task_graph_remote_stage or "").strip())
            or bool(str(self.task_graph_remote_provider_id or "").strip())
        )
        if self.execution_mode == "auto" and explicit_n21:
            self.execution_mode = "task_graph"
            self.task_graph_inferred = True
        return self

    @model_validator(mode="after")
    def validate_multimodal_route(self):
        self.image_data_urls = validate_image_data_urls(self.image_data_urls)
        if not self.image_data_urls:
            return self
        if self.execution_mode != "auto":
            raise ValueError("图像请求暂不支持 task_graph 执行模式")
        if self.routing_preference == "local_only":
            if len(self.image_data_urls) != 1:
                raise ValueError("本地 MTMD 图像请求仅支持一张图片")
            if self.streaming_mode != "full":
                raise ValueError("本地 MTMD 图像请求仅支持 full 响应模式")
            if self.allow_external or self.prefer_external:
                raise ValueError("local_only 图像请求不能同时授权外部路由")
            return self
        if not self.allow_external or not self.prefer_external:
            raise ValueError("图像请求必须显式设置 allow_external 和 prefer_external")
        return self


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    thinking_content: Optional[str] = None
    metrics: dict = {}
    followups: list[str] = []


class NodeDetail(BaseModel):
    node_id: str
    role: str
    node_type: str = "pc"
    state: str
    address: str = ""
    hostname: str = ""
    device_info: dict = {}
    network_type: str = "unknown"
    connected_at: float = 0.0
    last_heartbeat: float = 0.0
    avg_rtt_ms: float = 0.0
    last_rtt_ms: float = 0.0
    task_count: int = 0
    error_count: int = 0
    is_available: bool = False
    network_path: Optional[dict] = None
    # 「在线但不干活」可见化：state=online 只代表心跳还在，这两个字段说明该节点
    # 是否真的落在流水线容量计划里，以及没落进去时的原因码。
    pipeline_participating: bool = False
    pipeline_exclusion_reason: str = ""


class ClusterStatus(BaseModel):
    run_mode: str
    nodes_ready: bool
    nodes: dict[str, NodeDetail] = {}
    current_task: Optional[dict] = None
    tcp_server: Optional[dict] = None
    pipeline: Optional[dict] = None
    # 只读投影：容量决策（admitted / reason_code / participating / control_only / worker_count）。
    pipeline_capacity: Optional[dict] = None
    pipeline_queue: Optional[dict] = None
    network_path: Optional[dict] = None


class UpdateMaxNodesRequest(BaseModel):
    max_nodes: int = Field(..., ge=1, le=64, description="新的最大节点数（包含 master）")


class ConnectToMasterRequest(BaseModel):
    master_host: str = Field(..., description="主节点 IP 地址", min_length=1)
    master_port: int = Field(8888, ge=1, le=65535, description="主节点端口")
    switch_to_client: bool = Field(
        False,
        description="待配置节点显式切换为从节点后加入现有集群",
    )


class ClusterJoinRequestCreate(BaseModel):
    master_endpoint: str = Field(..., min_length=3, max_length=256)
    target_node_id: Optional[str] = Field(default=None, max_length=128)
    cluster_id: str = Field(default="", max_length=128)
    capabilities: list[str] = Field(default_factory=lambda: ["presence", "task"], max_length=16)
    request_ttl_seconds: int = Field(default=600, ge=60, le=3600)


class ClusterJoinGrantIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_code: Optional[str] = Field(default=None, max_length=16 * 1024)
    request: Optional[dict[str, Any]] = None
    otp_code: Optional[str] = Field(
        default=None,
        max_length=16,
        description="当前已登录管理员的 Auth App/TOTP 一次性确认码",
    )
    ttl_seconds: int = Field(default=300, ge=60, le=900)


class ClusterSecretRotateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    otp_code: Optional[str] = Field(default=None, max_length=16)
    tailnet_revocation_completed: bool = Field(
        default=False,
        description=(
            "Operator attestation that the revoked node was removed from the Tailnet; "
            "the service does not query or prove Tailscale control-plane state"
        ),
    )


class ClusterJoinConsume(BaseModel):
    grant_code: str = Field(..., min_length=12, max_length=16 * 1024)


class FirstConnectBootstrapRequest(BaseModel):
    node_id: Optional[str] = Field(default=None, max_length=64, description="客户端稳定节点 ID")
    node_type: str = Field(default="pc", description="节点类型: pc | android")
    hostname: str = Field(default="", max_length=128, description="设备名")
    platform: str = Field(default="", max_length=64, description="平台: windows | linux | android")
    app_variant: str = Field(default="", max_length=32, description="Android full | lite")
    app_version: str = Field(default="", max_length=64, description="客户端版本")
    capabilities: dict = Field(default_factory=dict, description="设备画像/能力")
    credential_public_key: str = Field(..., min_length=128, max_length=2048)
    credential_request_nonce: str = Field(..., min_length=16, max_length=128)
    credential_requested_at: int = Field(..., ge=0)


# ============================================================
# 辅助函数
# ============================================================

def _peek_model_manager():
    """Return an existing manager instance without warming a lazy proxy."""
    manager = model_manager
    if manager is model_host:
        peek = getattr(model_host, "peek_manager", None)
        return peek() if callable(peek) else None
    if type(manager).__name__ == "_LazyModelManager":
        try:
            return object.__getattribute__(manager, "_instance")
        except AttributeError:
            return None
    return manager


def _local_llm_is_loaded() -> bool:
    """Inspect LLM ownership without materializing the lazy manager."""

    checker = getattr(model_host, "has_loaded_model", None)
    if model_manager is model_host and callable(checker):
        if bool(checker()):
            return True
        manager = _peek_model_manager()
        return bool(
            manager is not None
            and getattr(manager, "is_pipeline_prepared", False)
        )
    manager = _peek_model_manager()
    return bool(model_host.model_loaded) or bool(
        manager is not None
        and (
            getattr(manager, "is_loaded", False)
            or getattr(manager, "is_pipeline_prepared", False)
        )
    )

def _build_chat_prompt(messages: list[dict], system_prompt: Optional[str] = None,
                       assistant_prefill: Optional[str] = None) -> str:
    """
    使用 Qwen 的 chat template 构建对话 prompt。
    Qwen-1.8B-Chat 使用 <|im_start|>/<|im_end|> 格式。

    Args:
        messages: 对话历史列表
        system_prompt: 可选的系统提示，会插入在对话历史之前
        assistant_prefill: 可选的助手预填文本（强制模型从此处续写），
                           用于引导结构化输出，如深度思考的「【思考】\n」
    """
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    if assistant_prefill:
        parts.append(assistant_prefill)
    return "\n".join(parts)


def _build_model_chat_prompt(tokenizer, messages: list[dict],
                             system_prompt: Optional[str] = None,
                             assistant_prefill: Optional[str] = None) -> str:
    """Build a prompt with the active tokenizer's native chat template."""
    chat_messages = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    chat_messages.extend(messages)

    try:
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # DeepSeek-R1-Distill templates already end with "<think>\n" when
        # add_generation_prompt=True. Appending the legacy Chinese prefill here
        # mixes two incompatible thinking protocols.
        native_thinking_prompt = "<think>" in prompt[-64:].lower()
        if assistant_prefill and not native_thinking_prompt:
            prompt += assistant_prefill
        return prompt
    except Exception:
        return _build_chat_prompt(
            messages,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
        )


# ================================================================
# 深度思考展示
# ================================================================

THINKING_START = "【思考】"
THINKING_END   = "【思考结束】"

THINKING_SYSTEM_PROMPT = (
    "你是一个善于深度思考的AI助手。回答前先进行推理分析，再给出答案。\n\n"
    "严格按以下格式输出：\n"
    "【思考】\n"
    "（你的推理过程，2-3句话即可）\n"
    "【思考结束】\n"
    "（你的最终回答）\n\n"
    "注意：\n"
    "- 必须在【思考结束】之后写回答内容\n"
    "- 回答部分不要写标记符号\n"
    "- 不要重复输出【思考】或【思考结束】"
)


def _strip_native_thinking_tags(text: str) -> str:
    """Remove native thinking/answer tags and leaked ChatML sentinels."""
    import re as _re

    if not text:
        return text

    result = _re.sub(
        r'<\s*think\s*>.*?<\s*/\s*think\s*>',
        '',
        text,
        flags=_re.DOTALL | _re.IGNORECASE,
    )
    # DeepSeek templates put "<think>\n" in the prompt. The generated completion
    # can therefore start with "reasoning...</think>\nanswer" and contain only
    # the closing tag. In that case, drop everything up to the closing tag.
    result = _re.sub(
        r'^.*?<\s*/\s*think\s*>',
        '',
        result,
        count=1,
        flags=_re.DOTALL | _re.IGNORECASE,
    )

    response_match = _re.search(
        r'<\s*(?:answer|response)\s*>(.*?)(?:<\s*/\s*(?:answer|response)\s*>|$)',
        result,
        flags=_re.DOTALL | _re.IGNORECASE,
    )
    if response_match:
        result = response_match.group(1)

    result = _re.sub(r'<\s*/?\s*(?:think|answer|response)\s*>', '', result, flags=_re.IGNORECASE)
    result = result.replace('<|im_end|>', '').replace('<|im_start|>', '')
    result = _re.sub(r'<\s*\|im_(?:start|end)\|\s*>', '', result)
    result = _re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def _parse_thinking_response(text: str) -> tuple:
    """
    解析模型输出，分离思考内容和最终答案。

    当 show_thinking 启用时，模型应输出：

        【思考】
        (推理过程)
        【思考结束】
        (最终答案)

    本函数对各种格式错误具有容错能力：
    - 缺少结束标记 → 尝试智能分割
    - 答案为空 → 从思考中提取最后一段作为答案
    - 重复标记 → 使用第一次出现的有效标记对

    Args:
        text: 模型原始输出文本（已包含预填的【思考】前缀）

    Returns:
        (answer_content, thinking_content)
        - answer_content: 最终答案文本（绝不包含思考标记）
        - thinking_content: 思考过程文本，格式不匹配时为 None
    """
    import re as _re

    if not text:
        return "", None

    # ---- 查找标记位置 ----
    start_idx = text.find(THINKING_START)
    end_idx = text.find(THINKING_END)

    # ---- 情况1：标记成对且顺序正确 ----
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        thinking = text[start_idx + len(THINKING_START):end_idx].strip()
        answer = text[end_idx + len(THINKING_END):].strip()

        # 清理思考中的标题前缀
        thinking = _re.sub(r'^分析思路[：:]\s*', '', thinking)

        # 清理答案开头的标题前缀
        answer = _re.sub(r'^【最终答案】[：:]?\s*', '', answer)
        answer = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', answer, flags=_re.IGNORECASE)
        for _pat in [r'^\[你的最终回答[^\]]*\]\s*', r'^\[你的推理过程[^\]]*\]\s*',
                     r'^（推理内容）\s*', r'^（答案内容）\s*',
                     r'^（给用户的答案[^）]*）\s*']:
            answer = _re.sub(_pat, '', answer)

        # 清理答案中残留的思考标记（模型可能在答案里又输出了标记）
        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "").strip()

        # 开始标记之前的内容拼入答案
        prefix = text[:start_idx].strip()
        if prefix:
            answer = prefix + ("\n" + answer if answer else "")

        # 思考内容为空 → 格式未遵循，fallthrough 到情况2
        if thinking:
            # 如果答案为空但思考非空 → 尝试从思考中提取最后一段作为答案
            # 1.8B 模型常见失败模式：把所有内容都放在思考里，答案留空
            if not answer and thinking:
                paragraphs = thinking.split("\n")
                # 取最后一段非空内容作为答案
                for p in reversed(paragraphs):
                    p = p.strip()
                    if p and len(p) > 10:
                        answer = p
                        break
                # 如果还是空，用整个思考作为答案
                if not answer:
                    answer = thinking
            return answer, thinking

    # ---- 情况2：DeepSeek-R1 / Qwen3 本地  格式 ----
    # 这些模型通过 ChatML 原生输出  ...  包裹思考，
    # 不依赖 THINKING_SYSTEM_PROMPT 注入的【思考】标记。
    import re as _re2
    native_match = _re2.search(
        r'<\s*think\s*>(.*?)<\s*/\s*think\s*>',
        text,
        flags=_re2.DOTALL | _re2.IGNORECASE,
    )
    if native_match:
        thinking = native_match.group(1).strip()
        # 取  之后、</think> 之前的内容作为思考
        answer = text[:native_match.start()].strip()
        after_think = text[native_match.end():].strip()
        # 去除  标记
        after_think = _re2.sub(r'<\s*/?\s*(?:response|answer)\s*>', '', after_think, flags=_re2.IGNORECASE)
        if after_think:
            answer = (answer + '\n' + after_think).strip() if answer else after_think
        # 也尝试从  标记中提取回答
        response_match = _re2.search(
            r'<\s*(?:response|answer)\s*>(.*)',
            answer if answer else '',
            flags=_re2.DOTALL | _re2.IGNORECASE,
        )
        if response_match:
            answer = response_match.group(1).strip()
        # 清理残余标签
        answer = _re2.sub(r'<\s*/?\s*(?:think|response|answer)\s*>', '', answer, flags=_re2.IGNORECASE)
        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "")
        answer = answer.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()
        if thinking:
            return answer, thinking

    closing_only_match = _re2.search(
        r'^(.*?)<\s*/\s*think\s*>(.*)$',
        text,
        flags=_re2.DOTALL | _re2.IGNORECASE,
    )
    if closing_only_match:
        thinking = closing_only_match.group(1).strip()
        thinking = thinking.replace(THINKING_START, "").replace(THINKING_END, "").strip()
        answer = closing_only_match.group(2).strip()
        answer = _re2.sub(r'<\s*/?\s*(?:response|answer)\s*>', '', answer, flags=_re2.IGNORECASE)
        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "")
        answer = answer.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()
        if answer or thinking:
            return answer, thinking or None

    # ---- 情况3：格式未遵循（缺少标记或标记顺序错误） ----
    # 清理所有思考标记，返回干净的文本作为答案
    cleaned = text.replace(THINKING_START, "").replace(THINKING_END, "").strip()
    # 也清理本地格式标记
    cleaned = _strip_native_thinking_tags(cleaned)
    # 清理常见的标题前缀
    cleaned = _re.sub(r'^分析思路[：:]\s*', '', cleaned)
    cleaned = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', cleaned, flags=_re.IGNORECASE)
    return cleaned, None


def _format_model_response(text: str, show_thinking: bool,
                           native_thinking_prompt: bool = False) -> tuple[str, Optional[str]]:
    """Format generated text without exposing unfinished native reasoning."""
    if show_thinking:
        return _parse_thinking_response(text)
    if native_thinking_prompt and "</think>" not in (text or "").lower():
        return "", None
    return _strip_native_thinking_tags(text), None


# ================================================================
# 多会话管理
# ================================================================

def _sync_chat_context_facade() -> None:
    _chat_context.adopt_facade_state(active_session_id, session_histories)


def _publish_chat_context_facade() -> None:
    global active_session_id, session_histories
    active_session_id = _chat_context.active_session_id
    session_histories = _chat_context.histories


def _load_conversation_history(session_id: str) -> list[dict]:
    """Load one canonical context window; storage failures are not empty chats."""

    try:
        rows = _local_store.load_local_conversation(session_id)
    except Exception:
        logger.error(
            "SQLite 会话历史加载失败: session=%s", session_id, exc_info=True,
        )
        raise
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def _get_active_history() -> list[dict]:
    """Return the active canonical history, reloading stale cache if needed."""

    _sync_chat_context_facade()
    history = _chat_context.active_history()
    _publish_chat_context_facade()
    return history


def _prepare_chat_context(
    requested_session_id: Optional[str],
    first_message: Optional[str] = None,
    *,
    force_reload: bool = False,
) -> PreparedChatContext:
    """Resolve one request to the shared session snapshot used by every mode."""

    global kv_cache
    _sync_chat_context_facade()
    prepared = _chat_context.prepare(
        requested_session_id, force_reload=force_reload,
    )
    _publish_chat_context_facade()

    if prepared.session_changed or prepared.reloaded:
        if kv_cache:
            kv_cache.clear()
        _init_kv_cache()
    if first_message and not prepared.history:
        _auto_title_session(prepared.session_id, first_message)
    return prepared


def _switch_session(target_id: str, *, force_reload: bool = False) -> None:
    """Activate a session, reloading it even when the ID did not change."""

    prepared = _prepare_chat_context(target_id, force_reload=force_reload)
    logger.info(
        "已切换到会话: %s (reloaded=%s)",
        prepared.session_id,
        prepared.reloaded,
    )


def _reset_runtime_conversation_state(clear_histories: bool = True) -> None:
    """Clear in-memory conversation/KV state after a model change."""
    global kv_cache, conversation_stats

    if kv_cache:
        kv_cache.clear()
    kv_cache = None
    _sync_chat_context_facade()
    _chat_context.invalidate(clear_histories=clear_histories)
    _publish_chat_context_facade()
    conversation_stats = {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }


def _auto_title_session(session_id: str, first_message: str) -> None:
    """用首条用户消息自动生成会话标题（截取前30字）"""
    title = first_message.strip()[:30]
    if len(first_message.strip()) > 30:
        title += "..."
    try:
        _local_store.update_local_session_title(session_id, title)
    except Exception as exc:
        logger.warning("SQLite 自动标题更新失败: session=%s: %s", session_id, exc)


def _persist_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metrics: Optional[dict] = None,
) -> TurnPersistenceResult:
    """Commit a completed turn to local SQLite, then optionally export it."""
    try:
        if not _local_store.get_local_save_history():
            return TurnPersistenceResult.SKIPPED
        request_id = _request_id_ctx.get("")
        committed = _local_store.save_local_conversation_turn(
            session_id,
            user_message,
            assistant_message,
            metrics,
            operation_id=request_id if request_id and request_id != "-" else None,
        )
    except ValueError:
        # Durable operation-id conflicts are contract violations, not a
        # best-effort persistence failure. Keep them fail-closed.
        raise
    except Exception as exc:
        logger.error("SQLite 对话提交失败: session=%s: %s", session_id, exc)
        raise RuntimeError("SQLite 对话提交失败") from exc

    return (
        TurnPersistenceResult.COMMITTED
        if committed
        else TurnPersistenceResult.REPLAY
    )


def _commit_chat_context_turn(
    session_id: Optional[str],
    user_message: str,
    assistant_message: str,
    metrics: Optional[dict] = None,
    *,
    expected_revision: Optional[int] = None,
    operation_id: Optional[str] = None,
) -> bool:
    """Publish one successful turn to memory and SQLite through one service."""

    _sync_chat_context_facade()
    resolved_operation_id = operation_id or _request_id_ctx.get("")
    try:
        persisted = _chat_context.commit_turn(
            session_id,
            user_message,
            assistant_message,
            metrics,
            operation_id=(
                resolved_operation_id
                if resolved_operation_id and resolved_operation_id != "-"
                else None
            ),
            expected_revision=expected_revision,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ConversationContextConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    _publish_chat_context_facade()
    return persisted


def _is_question(text: str) -> bool:
    """
    判断文本是否为真正的疑问句，而非陈述句。

    Qwen-1.8B 小模型容易输出陈述句（如"机器学习有以下特点："），
    此函数用于过滤这类不合格输出。
    """
    text = text.strip()
    if not text:
        return False

    # 必须以问号结尾
    if not (text.endswith('？') or text.endswith('?')):
        return False

    # 必须包含疑问指示词
    question_indicators = [
        '吗', '呢',
        '什么', '怎么', '如何', '为何',
        '哪些', '哪个', '哪种', '哪位',
        '有没有', '能否', '是否', '可否',
        '能不能', '会不会', '可不可以',
        '多少', '几',
        '谁', '哪', '何时', '怎样',
        '可以', '能帮', '推荐', '介绍',
    ]
    has_indicator = any(ind in text for ind in question_indicators)
    if not has_indicator:
        return False

    # 拒绝陈述句式关键词
    statement_patterns = [
        '有以下', '包括以下', '如下',
        '例如', '比如',
        '这是', '以下是', '下面是',
        '区别在于', '不同之处', '特点有',
        '首先', '其次', '然后', '最后',
        '第一', '第二', '第三',
        '步骤', '流程', '方法有',
    ]
    if any(p in text for p in statement_patterns):
        return False

    # 拒绝看起来像列举的开头
    if re.match(r'^[\d]+[\.\、\)）]', text):
        return False

    return True


def _generate_followups(
    history: list[dict], tokenizer, model, device,
    cancel_event: Optional[threading.Event] = None,
) -> list[str]:
    """
    根据对话上下文，让模型生成 2-3 个追问建议。

    类似豆包/千问 App 的追问推荐功能。
    使用 few-shot prompt + 问句质量验证 + 模板兜底，适配 1.8B 小模型。
    """
    if not history or len(history) < 2:
        return []
    if cancel_event is not None and cancel_event.is_set():
        return []

    # ---- Few-shot prompt：强调只输出疑问句，给出正确和错误示例 ----
    system_prompt = (
        "根据对话历史，生成3个用户可能追问的疑问句。\n"
        "严格规则：\n"
        "1. 每个输出必须以 Q: 开头，单独一行\n"
        "2. 每个输出必须是疑问句（以？结尾），严禁输出陈述句\n"
        "3. 不要输出解释、列举、定义等陈述性内容\n"
        "正确示例:\n"
        "Q: 深度学习与机器学习有什么区别？\n"
        "Q: 能推荐一些入门学习资源吗？\n"
        "Q: 这个概念在实际中有哪些应用？\n"
        "错误示例（严禁输出）:\n"
        "Q: 机器学习和深度学习有以下几点区别：\n"
        "Q: 深度学习是机器学习的一个分支\n"
        "Q: 1. 监督学习 2. 无监督学习"
    )
    followup_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    # 只取最近 3 轮对话
    recent = history[-6:]
    for msg in recent:
        followup_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    followup_prompt += "<|im_start|>assistant\n"

    questions = []

    try:
        inputs = tokenizer(followup_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        stop_criteria_kwargs = (
            {"cancel_event": cancel_event} if cancel_event is not None else {}
        )
        stop_criteria = model_manager._build_stop_criteria(
            [], input_ids.shape[1], **stop_criteria_kwargs,
        )
        generation_kwargs = {}
        if stop_criteria is not None:
            generation_kwargs["stopping_criteria"] = stop_criteria

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )

        if cancel_event is not None and cancel_event.is_set():
            return []

        generated = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # 解析 Q: 前缀的行，也兼容编号格式
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 匹配 Q: 前缀
            if line.upper().startswith("Q:") or line.upper().startswith("Q：") or line.startswith("问："):
                # 取第一个冒号后的内容
                q = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            else:
                # 兼容编号格式: 1. xxx, 1、xxx, 1) xxx
                q = re.sub(r'^[\d]+[\.\、\)）\s\-]+', '', line).strip()
            # 长度过滤 + 问句验证：必须通过 _is_question() 检查
            if q and len(q) >= 5 and len(q) <= 80 and _is_question(q):
                questions.append(q)

        # ---- 质量过滤 ----
        # 过滤包含幻觉模型名称的追问（通义千问、ChatGPT、Claude 等）
        hallucination_patterns = [
            "通义千问", "千问", "ChatGPT", "Claude", "GPT-", "文心一言",
            "讯飞星火", "豆包", "Kimi", "Copilot", "Bard", "Gemini",
            "百川", "智谱", "ChatGLM", "混元",
        ]
        questions = [
            q for q in questions
            if not any(p in q for p in hallucination_patterns)
        ]

        # 过滤高度重复的追问（如 "通义千问，通义千问，通义千问"）
        filtered = []
        seen_words = set()
        for q in questions:
            # 提取核心关键词
            words = frozenset(q[:10])  # 前 10 个字符作为特征
            if words not in seen_words:
                seen_words.add(words)
                filtered.append(q)
        questions = filtered

        logger.info(f"模型追问生成: {len(questions)} 条 → {questions}")

    except Exception as e:
        logger.warning(f"追问生成失败（非致命）: {e}")
        questions = []

    # ---- 模板兜底：如果模型输出不足 2 条，用规则补足 ----
    if len(questions) < 2:
        fallback = _fallback_followups(history, questions)
        questions = fallback

    return questions[:3]


def _generate_followups_llama(
    history: list[dict], cancel_event: Optional[threading.Event] = None,
) -> list[str]:
    """
    使用 llama.cpp 引擎生成追问建议。

    通过 model_manager.chat() 调用（llama.cpp 路径），
    使用简化的 few-shot prompt 适配小模型能力。
    失败时回退到关键词模板兜底。
    """
    if not history or len(history) < 2:
        return []
    if cancel_event is not None and cancel_event.is_set():
        return []

    # 简化版 prompt：直接要求输出问题，不需要 Q: 前缀格式
    system_prompt = (
        "根据对话内容，生成2-3个你会追问的问题。每个问题一行，以？结尾。"
    )
    followup_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"根据以下对话，生成我想追问的问题：\n"
         f"用户：{history[-2]['content'][:200]}\n"
         f"助手：{history[-1]['content'][:300]}"},
    ]

    questions = []
    try:
        result = model_manager.chat(
            messages=followup_messages,
            max_tokens=128,
            temperature=0.8,
            top_p=0.9,
            _cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            return []
        text = result.get("content", "").strip()

        # 解析：每行一个追问
        for line in text.split("\n"):
            line = line.strip()
            # 清理编号前缀
            line = re.sub(r'^[\d]+[\.\、\)）\s\-]+', '', line).strip()
            # 清理 Q: 前缀
            if line.upper().startswith("Q:") or line.upper().startswith("Q："):
                line = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            if line and len(line) >= 5 and len(line) <= 80 and _is_question(line):
                questions.append(line)

        # 质量过滤（同 _generate_followups）
        hallucination_patterns = [
            "通义千问", "千问", "ChatGPT", "Claude", "GPT-", "文心一言",
            "讯飞星火", "豆包", "Kimi", "Copilot", "Bard", "Gemini",
            "百川", "智谱", "ChatGLM", "混元",
        ]
        questions = [q for q in questions if not any(p in q for p in hallucination_patterns)]

        # 去重
        filtered = []
        seen = set()
        for q in questions:
            key = q[:15]
            if key not in seen:
                seen.add(key)
                filtered.append(q)
        questions = filtered

        logger.info(f"llama.cpp 追问生成: {len(questions)} 条 → {questions}")

    except Exception as e:
        logger.warning(f"llama.cpp 追问生成失败（非致命）: {e}")
        questions = []

    # 模板兜底
    if len(questions) < 2:
        fallback = _fallback_followups(history, questions)
        questions = fallback

    return questions[:3]


def _fallback_followups(history: list[dict], existing: list[str]) -> list[str]:
    """
    基于对话关键词匹配的追问模板兜底。

    当 1.8B 小模型无法生成合格追问时启用。
    """
    # 提取最后一轮问答的关键词
    last_assistant = ""
    last_user = ""
    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg["content"]
        if msg["role"] == "user" and not last_user:
            last_user = msg["content"]

    combined = (last_user + " " + last_assistant).lower()

    # 关键词 → 追问模板映射（按优先级排序，更具体的匹配在前）
    templates = []

    if any(kw in combined for kw in ["量化", "quant", "int4", "int8", "fp16", "精度"]):
        templates.extend([
            "INT4和INT8量化在实际应用中如何选择？",
            "量化会对模型推理能力造成多大影响？",
            "除了量化还有哪些模型压缩方法？",
        ])

    if any(kw in combined for kw in ["边缘计算", "边缘", "edge", "分布式", "推理"]):
        templates.extend([
            "边缘推理和云端推理各有什么优缺点？",
            "分布式推理中的通信开销如何优化？",
            "边缘设备的算力瓶颈通常在哪里？",
        ])

    if any(kw in combined for kw in ["python", "代码", "编程", "写一个", "函数", "算法"]):
        templates.extend([
            "这段代码的时间复杂度是多少？",
            "有没有更高效的实现方式？",
            "能解释一下这段代码的核心逻辑吗？",
        ])

    if any(kw in combined for kw in ["模型", "训练", "微调", "lora", "参数"]):
        templates.extend([
            "这个模型的训练数据来源是什么？",
            "如何在特定领域数据上微调模型？",
            "LoRA微调相比全参数微调有哪些优势？",
        ])

    if any(kw in combined for kw in ["transformer", "注意力", "attention", "架构"]):
        templates.extend([
            "Transformer相比RNN有哪些优势？",
            "自注意力机制的计算复杂度如何？",
            "多头注意力的作用是什么？",
        ])

    if any(kw in combined for kw in ["token", "tokenizer", "分词", "词表"]):
        templates.extend([
            "不同的分词方法对模型性能有影响吗？",
            "中文分词和英文分词的主要区别是什么？",
            "BPE分词算法的原理是什么？",
        ])

    if any(kw in combined for kw in ["显存", "gpu", "内存", "oom", "优化", "加速"]):
        templates.extend([
            "还有哪些降低推理显存占用的方法？",
            "CPU推理在什么场景下比GPU更合适？",
            "KV Cache的显存占用如何估算？",
        ])

    if any(kw in combined for kw in ["应用", "场景", "实际", "落地", "工业"]):
        templates.extend([
            "当前这个技术还有哪些落地挑战？",
            "业界有哪些成功的应用案例可以参考？",
            "这项技术的商业化前景如何？",
        ])

    if any(kw in combined for kw in ["hello", "你好", "介绍", "你是谁", "能做什么"]):
        templates.extend([
            "你能帮我写代码吗？",
            "你的知识截止到什么时候？",
            "你擅长哪些类型的任务？",
        ])

    if any(kw in combined for kw in ["学习", "入门", "新手", "教程", "怎么学"]):
        templates.extend([
            "有哪些推荐的学习资源或课程？",
            "学习这个需要什么前置知识？",
            "从入门到精通大概需要多久？",
        ])

    if any(kw in combined for kw in ["区别", "对比", "比较", "不同", "差异", "选择"]):
        templates.extend([
            "在选择时应该考虑哪些关键因素？",
            "有没有具体的场景举例说明？",
            "未来哪个方向更有发展前景？",
        ])

    if any(kw in combined for kw in ["安全", "隐私", "加密", "攻击", "漏洞"]):
        templates.extend([
            "这种攻击的防御措施有哪些？",
            "业界有哪些典型的安全事件？",
            "如何在性能和安全性之间平衡？",
        ])

    if any(kw in combined for kw in ["数据", "dataset", "数据集", "预处理", "清洗"]):
        templates.extend([
            "数据质量对模型效果的影响有多大？",
            "有哪些常用的数据增强方法？",
            "如何处理数据中的类别不平衡问题？",
        ])

    # 默认通用追问（更智能的追问）
    default_templates = [
        "能再详细解释一下吗？",
        "这个结论有什么前提条件或局限性？",
        "有没有相关的参考资料或论文推荐？",
        "实际应用中需要注意哪些细节？",
        "能举一个具体的例子说明吗？",
    ]

    # 选择不重复的追问
    result = list(existing)
    candidate_pool = templates + default_templates
    for q in candidate_pool:
        if q not in result and len(result) < 3:
            result.append(q)

    if len(result) < 2:
        # 不可能到这一步，但也处理一下
        for q in default_templates:
            if q not in result and len(result) < 3:
                result.append(q)

    logger.info(f"追问兜底: 模型生成了 {len(existing)} 条，模板补充至 {len(result)} 条")
    return result


def _safe_torch_model():
    """★ 2026-09-19：安全获取底层 **PyTorch 模型**，非 PyTorch 引擎返回 None。

    为什么需要：`model_manager`（`ModelHost`）会把未知属性转发给底层引擎；
    **llama_cpp 引擎（`LlamaCppEngine`）只有 `_model`，没有 `.model`**
    ⇒ 直接 `model_manager.model` 会抛
    `AttributeError: 'LlamaCppEngine' object has no attribute 'model'`
    （用户实测：加载完成后 `_init_kv_cache()` 即崩，表现为 HTTP 500）。

    判定标准：**只看「取属性是否成功」**——
      * 取属性抛 `AttributeError`（`ModelHost` 把未知属性转发给 `LlamaCppEngine`）⇒ 返回 None；
      * 取到 `None`（未加载 / llama.cpp 引擎下 `self.model` 为 None）⇒ 返回 None；
      * 其余**原样返回**（PyTorch 真模型、乃至测试替身都照旧）。

    ⚠️ 早先版本用 `isinstance(candidate, torch.nn.Module)` 判定，**过严**：会把测试里的
    **替身模型**也挡掉（`test_local_pytorch_chat_restores_full_model_before_generate` 于是失败）。
    现在只做「属性可访问性」判断，**PyTorch 路径与替身路径行为完全不变**。
    """
    try:
        candidate = getattr(model_manager, "model", None)
    except Exception:  # noqa: BLE001 —— 属性转发本身可能抛（AttributeError 等）
        return None
    return candidate


def _init_kv_cache():
    """初始化分页 KV 缓存（根据设备画像自适应大小）"""
    global kv_cache
    try:
        from paged_kv_cache import PagedKVCache  # torch-backed (D-tier only)
    except ImportError:
        # L-tier (no torch): the paged KV cache is a PyTorch feature and is not wired
        # into the single-machine decode loop anyway (see the note further down).
        kv_cache = None
        return

    # ★ 2026-09-19：**非 PyTorch 引擎直接跳过**。
    #   本函数是 PyTorch 特性（见上方注释：L-tier 未接入单机解码循环）。
    #   而 `model_manager` 会把未知属性转发给底层引擎，`LlamaCppEngine` 既无 `.model`
    #   也无 `.get_device` ⇒ 继续往下会连抛 AttributeError（用户实测的 HTTP 500）。
    _torch_model = _safe_torch_model()
    if _torch_model is None:
        kv_cache = None
        logger.debug("非 PyTorch 引擎（或无 torch 模型）⇒ 跳过 paged KV 初始化")
        return

    num_heads = 16      # Qwen-1.8B: 16 attention heads
    head_dim = 64       # 隐藏维度 2048 / 16 heads = 128, 但实际是 64 per head for K/V
    # 从模型获取实际的 head_dim
    if _torch_model is not None:
        try:
            cfg = _torch_model.config
            num_heads = cfg.num_attention_heads
            head_dim = cfg.hidden_size // num_heads
        except Exception:
            pass

    # 优先使用设备画像自适应大小
    if device_profile:
        kv_cache = PagedKVCache.from_profile(
            profile=device_profile,
            device=str(model_manager.get_device()),
            dtype=torch.float16,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        logger.info(
            f"🧠 KV 缓存已初始化 (profile): num_heads={num_heads}, "
            f"head_dim={head_dim}, device={model_manager.get_device()}"
        )
    else:
        kv_cache = PagedKVCache(
            page_size=PAGE_SIZE,
            max_pages=MAX_PAGE_NUM,
            device=str(model_manager.get_device()),
            dtype=torch.float16,
        )
        logger.info(
            f"🧠 KV 缓存已初始化 (default): page_size={PAGE_SIZE}, "
            f"max_pages={MAX_PAGE_NUM}, device={model_manager.get_device()}"
        )
    return kv_cache


# ============================================================
# API 路由
# ============================================================













class SelectGpuRequest(BaseModel):
    gpu_index: int = Field(..., ge=0, description="GPU 列表中要切换到的序号")








def _unload_model_under_model_lock() -> dict:
    """Unload the local LLM without materializing the lazy manager when idle.

    Model and SD lifecycles share ``full_chat_execution_lock``. Keeping the
    reset here makes the explicit UI unload path equivalent to a model switch
    and invalidates stale worker state before the next engine is loaded.
    """
    loaded = _local_llm_is_loaded()

    def _change() -> bool:
        if loaded:
            unload = getattr(model_manager, "unload_model", None)
            if not callable(unload):
                raise HTTPException(status_code=500, detail="当前推理引擎不支持卸载模型")
            unload()
        _reset_runtime_conversation_state(clear_histories=True)
        model_host.model_loaded = False
        model_host.current_quant = None
        return loaded

    unloaded = _run_exclusive_model_change(
        _change,
        release_worker_reservation=True,
    )
    try:
        scheduler.refresh_task_worker_capabilities()
    except Exception as exc:
        logger.warning("卸载模型后刷新 Worker 能力失败: %s", exc)
    return {
        "success": True,
        "loaded": False,
        "unloaded": unloaded,
        "message": "模型已卸载" if unloaded else "当前没有已加载的模型",
    }








def _chat_origin(req: ChatRequest) -> str:
    """根据请求上报信息推断请求来源，用于 metrics 展示。"""
    if req.client_node_type == "android":
        return "android_http"
    if req.client_node_type == "pc":
        return "pc_http"
    return "web_http"


def should_prefer_pytorch(prefer_flag: object, safetensors_path: object) -> bool:
    """`QLH_PREFER_PYTORCH` 让路的判据（`#54`，纯函数，可单测）。

    为真需同时满足三项：配置位为真、画像模型有 safetensors 目录、且该目录**存在**。
    动机见 `config.PREFER_PYTORCH`：qwen3.5 的 keep-head 上游挂点在 `output_norm` 之后，
    llama.cpp 上游被 fail-closed 拒绝（`#35`），而 GGUF 分支的优先级在 PyTorch 之前
    ⇒ 必须显式让路，否则本机会落到 llama.cpp 而拿不到该模型。
    """
    return bool(prefer_flag) and bool(safetensors_path) and os.path.isdir(safetensors_path)


def parse_pipeline_readiness(reason_text: str) -> dict:
    """从失败原因文本里解析**结构化**的 pipeline readiness（DIST-4 第二句要求）。

    文本形如：`pipeline_failed_then_local_pytorch: … | readiness=<code>: <reason>`
    ⇒ 返回 `{"reason_code": …, "reason": …}`；无该标记时返回 `{}`。

    抽成模块级纯函数，使该判据可脱离请求上下文单测（与 DIST-NEXT-8 的
    `request_outcome.request_phase_metrics` 同一思路）。
    """
    text = str(reason_text or "")
    marker = "readiness="
    if marker not in text:
        return {}
    tail = text.split(marker, 1)[1].strip()
    code, _, reason = tail.partition(":")
    return {"reason_code": code.strip(), "reason": reason.strip()}


def _augment_chat_metrics(metrics: dict | None, req: ChatRequest, **defaults) -> dict:
    """补齐统一聊天 metrics 字段，不覆盖调度器已给出的真实执行信息。"""
    result = dict(metrics or {})
    for key, value in defaults.items():
        result.setdefault(key, value)
    origin = _chat_origin(req)
    result.setdefault("request_origin", origin)
    result.setdefault("request_origin_node_id", req.client_node_id or "")
    result.setdefault("request_origin_node_type", req.client_node_type or "")
    result.setdefault("client_mode", req.client_mode or "")
    result.setdefault("client_app_variant", req.client_app_variant or "")
    result.setdefault("serving_node_id", scheduler.get_effective_node_id())
    result.setdefault("distributed_requested", scheduler.get_distributed_inference_enabled())
    # ★ 2026-10-08：把**请求侧意图**也放进 metrics。此前 `routing_preference` 只存在于
    #   响应顶层（`result`），TUI/安卓端拿到的 `metrics` 里没有它 ⇒ 界面**无法判断
    #   "用户是否要求了分布式"**，只能看全局开关 `distributed_requested`。后果：用户用
    #   `/route required`（或 preferred）要求分布式、而全局开关恰好关着时，`distributed_requested`
    #   为 False ⇒ "已请求分布式，实际本地"这条提示**不会出现**，界面完全静默
    #   （与"核心功能保证分布式一致性、交互时却违反且不提示"一致）。
    result.setdefault("routing_preference", getattr(req, "routing_preference", "") or "")
    result.setdefault("distributed_used", False)
    result.setdefault("fallback", False)
    result.setdefault("fallback_reason", "")
    result.setdefault("workers_used", [])
    result.setdefault("layer_assignments", [])
    result.setdefault("request_id", _request_id_ctx.get("-"))
    result.setdefault("generation_id", req.generation_id or "")
    # ★ 2026-10-05（DIST-4）：单条结构化摘要，让**一条日志**即可重建整条链路
    #   （验收条文：「从单条请求日志可重建完整链路」）。此前**没有任何日志打印最终
    #   metrics** —— 层区间（`scheduler_pipeline` 的「Route-A 分配原始」）与 worker
    #   名单（「Route-A stage handoff」）分属两条**不含 request_id** 的日志，单请求
    #   串行时可靠时间顺序人工拼，并发时会串。
    #   字段与 DIST-4 的六项要求一一对应：route / assignment（workers_used +
    #   layer_assignments）/ generation（generation_id + config_id）/ worker /
    #   layer range（claimed_layers + layer_segments）/ fallback reason。
    #
    #   ★ 另加三个「线路/能力状态」字段（DIST-4 第二句要求的显式指标，按裁定走
    #   可 grep 的日志字段而非新增 metrics，避免同一事实两处维护）。这三类是最容易
    #   让「为什么没走分布式」被误读的来源：
    #     - `a1_relay_probe_only` / `relay_enabled`：A1 是否已处于探针专用
    #       （= 已从产品入口剔除）。全局环境级，打进摘要才能让单条日志自证环境。
    #     - `legacy_bridge_rejected`：本次是否撞上 legacy/A3 混链拒绝。
    #     - `capability_missing_reason`：失败链里带出的具名就绪原因（若有）。
    _failure_text = str(result.get("fallback_reason", "") or "")
    _capability_reason = ""
    _marker = "readiness="
    if _marker in _failure_text:
        _capability_reason = _failure_text.split(_marker, 1)[1].strip()
    # ★ 2026-10-08（DIST-4 第二句要求）：把**结构化**的 pipeline readiness 写回**响应 metrics**。
    #   此前只把 `reason_code` 拼进 `fallback_reason` 字符串（见 `_execute_chat_full` 的失败分支）
    #   ⇒ 响应体里没有结构化字段，调用方无法从单条响应判断「为什么没走分布式」。
    #   现在补 `pipeline_readiness`（`reason_code` + `reason`）与 `pipeline_failure_reason`，
    #   与 DIST-NEXT-8 的 `outcome` 并列 —— 单条响应即可重建失败归因。
    _readiness = parse_pipeline_readiness(_failure_text)
    if _readiness:
        result.setdefault("pipeline_readiness", _readiness)
        result.setdefault("pipeline_failure_reason", _failure_text)
    # ★ 2026-10-07（DIST-NEXT-8）：把互斥终态打进摘要 —— 一行内即可区分
    #   completed / fallback_completed / cancelled / refused / failed，
    #   不再靠 `fallback` 布尔去猜。
    _metrics = result.get("metrics")
    _outcome = dict(_metrics or {}).get("outcome", "")
    logger.info(
        "event=chat_route_summary request_id=%s generation_id=%s "
        "routing_preference=%s distributed_requested=%s distributed_used=%s "
        "fallback=%s fallback_reason=%s execution_mode=%s route=%s "
        "workers_used=%s claimed_layers=%s layer_segments=%s config_id=%s "
        "outcome=%s "
        "a1_relay_probe_only=%s relay_enabled=%s legacy_bridge_rejected=%s "
        "capability_missing_reason=%s",
        result.get("request_id", ""),
        result.get("generation_id", ""),
        result.get("routing_preference", ""),
        result.get("distributed_requested", False),
        result.get("distributed_used", False),
        result.get("fallback", False),
        result.get("fallback_reason", ""),
        result.get("execution_mode", ""),
        result.get("route", ""),
        result.get("workers_used", []),
        result.get("claimed_layers", []),
        result.get("layer_segments", []),
        result.get("config_id", ""),
        _outcome,
        PIPELINE_RELAY_PROBE_ONLY,
        PIPELINE_RELAY_ENABLED,
        "route_a_mixed_legacy_execution_bridge_not_ready" in _failure_text,
        _capability_reason,
    )
    return result


# ================================================================
# 路线 B：外部推理服务整请求路由（数据作用域门控，默认不出集群）
# ================================================================

def _distributed_path_available() -> bool:
    """当前是否有可用的分布式执行路径（T9.5 路由偏好判断）。

    判定口径：分布式全局启用时，主节点可协调流水线/Worker；从节点可
    转发主节点（由主节点协调执行）。两者都计为“合格分布式路径”。
    """
    try:
        if not (scheduler.get_distributed_inference_enabled()
                and RUN_MODE == "distributed"):
            return False
        role = scheduler._effective_role()
        return role in ("master", "client")
    except Exception:
        return False


def _routing_gate_error(req: ChatRequest) -> Optional[str]:
    """路由偏好预检（T9.5）：返回拒绝原因，None 表示允许继续。

    - distributed_required：没有合格分布式路径时明确失败，不静默回退；
    - local_only / auto / distributed_preferred：始终允许（回退语义由执行路径处理）。
    """
    if req.routing_preference == "distributed_required":
        if not _distributed_path_available():
            return (
                "distributed_required 但当前没有可用的分布式路径"
                "（分布式未启用或本节点不是主节点）"
            )
    return None


def _enforce_distributed_required(
    req: ChatRequest,
    metrics: dict | None = None,
    *,
    detail: str = "分布式执行未完成",
) -> None:
    """Reject silent local fallback for an explicitly required route."""
    try:
        validate_distributed_completion(
            req.routing_preference, metrics, detail=detail,
        )
    except DistributedCompletionError as exc:
        raise HTTPException(
            503, f"distributed_required 执行失败：{exc}",
        ) from exc


def _external_route_decision(req: ChatRequest):
    """按当前配置 + 请求 flag 计算外部路由决策（纯函数包装，读实时配置）。"""
    if req.routing_preference in {"local_only", "distributed_required"}:
        # T9.5：local_only 覆盖一切外部路由意图，数据不出集群
        from types import SimpleNamespace as _SN
        return _SN(
            use_external=False,
            eligible=False,
            reason=f"{req.routing_preference}_override",
        )
    import config as _cfg
    from external_provider import decide_external_route

    return decide_external_route(
        enabled=bool(getattr(_cfg, "EXTERNAL_ENABLED", False)),
        base_url=str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or ""),
        data_scope=str(getattr(_cfg, "EXTERNAL_DATA_SCOPE", "opt_in")),
        allow_external=bool(req.allow_external),
        prefer_external=bool(req.prefer_external),
        prompt_chars=len(req.message or ""),
        min_prompt_chars=int(getattr(_cfg, "EXTERNAL_MIN_PROMPT_CHARS", 0) or 0),
    )


def _local_native_vision_available(manager=None) -> bool:
    """Check the active local engine without importing or loading a model."""
    target = manager or model_manager
    check = getattr(target, "native_vision_available", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except Exception:
        return False


def _maybe_log_external_scope_denial(req: ChatRequest, decision) -> None:
    """请求带外部 flag 但被数据作用域拒绝时记一条 INFO（每请求一次，不记正文）。"""
    if not (req.allow_external or req.prefer_external):
        return
    if decision.eligible or not str(decision.reason).startswith("scope"):
        return
    import config as _cfg

    logger.info(
        "数据作用域拒绝外部路由: reason=%s, scope=%s, generation_id=%s"
        "（消息正文未发送、不落日志）",
        decision.reason,
        getattr(_cfg, "EXTERNAL_DATA_SCOPE", ""),
        req.generation_id or "-",
    )


def _execute_external_chat(
    req: ChatRequest,
    history: list,
    target_session_id: Optional[str],
    context_revision: int,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """整请求路由到外部推理服务（与 llama.cpp/孤岛整请求路径同构）。"""
    global conversation_stats
    import config as _cfg
    from external_provider import get_external_chat_client

    client = get_external_chat_client()
    client.ensure_connected()
    request_history = [
        *history,
        {
            "role": "user",
            "content": build_openai_user_content(
                req.message, req.image_data_urls,
            ),
        },
    ]
    # 作用域门控在 client.chat 内部强制执行（外发前最后一道关口）
    result = client.chat(
        request_history,
        max_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        allow_external=req.allow_external,
        cancel_event=cancel_event,
    )
    _raise_if_generation_cancelled(cancel_event, req.generation_id)
    response_text = result.get("content", "")
    if not req.show_thinking:
        response_text = _strip_native_thinking_tags(response_text)
    completed_history = [
        *history,
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": response_text},
    ]
    tokens_per_sec = result.get("tokens_per_second", 0)
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    metrics = _augment_chat_metrics(
        {
            "engine": "external_api",
            "execution_mode": "external_api",
            "route": f"{_chat_origin(req)}_to_external_api",
            "provider": "external_openai",
            "external_label": getattr(_cfg, "EXTERNAL_LABEL", ""),
            "external_base_url": client.masked_base_url,
            "data_scope": getattr(_cfg, "EXTERNAL_DATA_SCOPE", ""),
            "model": result.get("model", "") or client.model_name,
            "tokens_per_second": round(tokens_per_sec, 1) if tokens_per_sec else 0,
            "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else 0,
            "generated_tokens": completion_tokens,
            "completion_tokens": completion_tokens,
            "usage": usage,
            "usage_estimated": bool(result.get("usage_estimated", False)),
            "fallback": False,
            "fallback_reason": "",
        },
        req,
    )

    # 追问生成不再二次外发（少一次数据出集群 + 少一次计费），用模板兜底
    followups = _fallback_followups(completed_history, [])

    save_metrics = dict(metrics)
    save_metrics["followups"] = followups
    _commit_chat_context_turn(
        target_session_id, req.message, response_text, save_metrics,
        expected_revision=context_revision,
    )

    conversation_stats["total_generated_tokens"] += completion_tokens
    conversation_stats["rounds"] += 1
    try:
        scheduler.record_task_complete(success=True)
    except Exception:
        pass

    logger.info(
        f"外部推理完成: {completion_tokens} tokens, "
        f"endpoint={client.masked_base_url}"
    )
    return {
        "content": response_text,
        "thinking_content": None,
        "metrics": metrics,
        "followups": followups,
    }


def _external_stream_events(
    req: ChatRequest,
    cancel_event: Optional[threading.Event] = None,
    history: Optional[list[dict]] = None,
):
    """
    外部推理服务真流式事件生成器（fast 模式）。

    逐 chunk 产出 {"token": ...}，结束时产出 done 事件；
    取消 = chunk 边界断流 + 关闭连接（best-effort，外部端可能继续算）。
    """
    import config as _cfg
    from external_provider import get_external_chat_client

    client = get_external_chat_client()
    client.ensure_connected()
    parts: list[str] = []
    t0 = time.time()
    request_messages = [
        *(history or []),
        {
            "role": "user",
            "content": build_openai_user_content(
                req.message, req.image_data_urls,
            ),
        },
    ]
    for chunk in client.chat_stream(
        request_messages,
        max_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        allow_external=req.allow_external,
        cancel_event=cancel_event,
    ):
        parts.append(chunk)
        yield {"token": chunk}
    elapsed = time.time() - t0
    cancelled = bool(cancel_event is not None and cancel_event.is_set())
    metrics = _augment_chat_metrics(
        {
            "engine": "external_api",
            "execution_mode": "external_api",
            "route": f"{_chat_origin(req)}_to_external_api",
            "provider": "external_openai",
            "external_label": getattr(_cfg, "EXTERNAL_LABEL", ""),
            "external_base_url": client.masked_base_url,
            "data_scope": getattr(_cfg, "EXTERNAL_DATA_SCOPE", ""),
            "model": client.model_name,
            # fast 模式无 usage 事件透传：以 chunk 数估算生成 token 数
            "generated_tokens": len(parts),
            "completion_tokens": len(parts),
            "usage_estimated": True,
            "elapsed_seconds": round(elapsed, 3),
            "cancelled": cancelled,
            "fallback": False,
            "fallback_reason": "",
        },
        req,
    )
    yield {
        "done": True,
        "response": "".join(parts),
        "metrics": metrics,
        "request_id": _request_id_ctx.get("-"),
    }


# ================================================================
# 路线 C-1（实验）：投机解码 draft-verify —— 独立实验端点，默认关闭
# ================================================================
# 接入方式的选择（调研方案 §2.3 要求"必须以独立 execution_mode 门控，默认关闭"）:
#
#   本 PoC **不**把投机解码接进 /api/chat 主聊天路径，而是单开一个实验端点。
#   理由:
#     1. §2.3 的接入点是**本地解码循环**（model_module 的 generate/decode），
#        主聊天路径只是它的调用方。在聊天路径里插分支既到不了真正的接入点，
#        又要把 execution_mode 的 Literal、回退链、指标、取消、持久化
#        全部改一遍——那是"改动量中～大"的部分，风险落在最核心路径上。
#     2. 主路径必须"QLH_SPEC_ENABLED=false 时逐字节不变"。独立端点天然满足：
#        本段代码在开关关闭时只会返回 404，不参与任何既有请求的处理。
#     3. 真实 draft 模型需要 PyTorch 解码运行时；PoC 环境没有。实验端点可以
#        显式声明 draft 来源（stub / 注入），聊天路径不能。
#   代价：本端点不写会话历史、不做追问、不入 conversation_stats——它是
#   实验测量入口，不是产品路径。接产品前的剩余工作见实施说明文档。

class SpeculativeExperimentRequest(BaseModel):
    """投机解码实验请求（仅 /api/experimental/speculative 使用）。"""
    message: str = Field(default="", max_length=8192)
    execution_mode: Literal["speculative_assisted"] = Field(
        default="speculative_assisted",
        description="仅接受 speculative_assisted —— 本端点就是该模式的实验入口",
    )
    allow_external: bool = Field(
        default=False,
        description=(
            "数据作用域按请求授权：草稿 token 由用户内容派生，本路径确实"
            "把数据送出集群，与路线 B 共用 QLH_EXTERNAL_DATA_SCOPE 门控。"
        ),
    )
    max_new_tokens: int = Field(default=64, ge=1, le=1024)
    gamma: int = Field(default=0, ge=0, le=16, description="每轮草稿数；0=用 QLH_SPEC_GAMMA")
    max_rounds: int = Field(default=0, ge=0, le=1024, description="0=用 QLH_SPEC_MAX_ROUNDS")
    temperature: float = Field(
        default=-1.0, ge=-1.0, le=2.0,
        description="<0 = 用 QLH_SPEC_TEMPERATURE；0 = 贪心模式",
    )
    seed: int = Field(default=0, ge=0, le=2147483647, description="RNG 种子，保证可复现")
    draft_hint: str = Field(
        default="", max_length=2048,
        description="PoC 假 draft 模型的提示序列：命中则接受率高，用于演示接受率对比",
    )


def _run_speculative_experiment(req: SpeculativeExperimentRequest) -> dict:
    """同步执行一次投机解码会话（阻塞 HTTP，调用方须放线程池）。"""
    from speculative import run_speculative_chat

    return run_speculative_chat(
        req.message,
        allow_external=bool(req.allow_external),
        max_new_tokens=int(req.max_new_tokens),
        gamma=(int(req.gamma) if req.gamma > 0 else None),
        max_rounds=(int(req.max_rounds) if req.max_rounds > 0 else None),
        temperature=(float(req.temperature) if req.temperature >= 0 else None),
        seed=int(req.seed),
        draft_hint=req.draft_hint or "",
    )






def _execute_task_graph_chat(
    req: ChatRequest,
    cancel_event: Optional[threading.Event] = None,
    request_deadline: Optional[RequestDeadline] = None,
) -> dict:
    """Run the fixed local task graph without claiming multi-device execution."""
    global conversation_stats

    if not TASK_GRAPH_ENABLED:
        raise coded_http_error(
            409,
            "TASK_GRAPH_DISABLED",
            "任务链实验未启用。请设置 QLH_TASK_GRAPH_ENABLED=true 后重启。",
        )
    journal = task_graph_coordinator.journal_status()
    if not journal.get("available", False):
        raise HTTPException(
            503,
            {
                "message": "任务链 journal 不可用，已拒绝不可恢复执行。",
                "reason": journal.get("error", "journal health check failed"),
            },
        )
    if scheduler._effective_role() != "master":
        raise HTTPException(409, "任务链协调器当前只允许在主节点运行。")
    if NODE_ROLE == "auto":
        raise coded_http_error(
            503, "ha_commit_unavailable", "自动选主尚无 journal 多数提交水位，拒绝任务链写入。",
        )

    if not _task_graph_execution_slot.acquire(blocking=False):
        raise HTTPException(429, "已有任务链正在执行，请稍后重试。")
    try:
        with hold_lock_until_deadline(
            model_host.full_chat_execution_lock,
            request_deadline,
            operation="full chat execution lock",
        ):
            return _execute_task_graph_chat_with_slot(
                req, cancel_event, request_deadline,
            )
    finally:
        _task_graph_execution_slot.release()


def _dispatch_local_task_provider(
    request: ProviderStageRequest,
    cancel_event: threading.Event,
) -> dict:
    executor = request.runtime_context.get("local_provider_executor")
    if not callable(executor):
        raise TaskGraphError("本地任务 Provider 缺少请求执行上下文")
    result = cast(ProviderExecutor, executor)(request, cancel_event)
    if not isinstance(result, dict):
        raise TaskGraphError("本地任务 Provider 返回值必须是 dict")
    return result


def _ensure_local_task_provider() -> None:
    if task_graph_coordinator.has_provider("local_full_model"):
        return
    try:
        task_graph_coordinator.register_provider(LocalFullModelProvider(
            _dispatch_local_task_provider,
            provider_id="local_full_model",
            node_id=scheduler.get_effective_node_id(),
            max_concurrency=1,
        ))
    except ProviderError:
        if not task_graph_coordinator.has_provider("local_full_model"):
            raise


def _prepared_pipeline_model_identity() -> Optional[ModelIdentity]:
    """Identity of an explicit distributed-only artifact that is not materialized.

    `POST /api/models/prepare-pipeline` 刻意不物化权重，但它把 inspect 出的描述符
    （`model_sha256` / 模型 id / 路径 / 引擎）留在了 model_manager 上 ⇒ 层流水线需要的
    "当前模型身份"从那里取。否则 Route A 会陷入死锁：它要求活动模型身份，而活动身份
    要求 `model_loaded`，distributed-only 路径按设计又不加载权重。
    """
    if not getattr(model_manager, "is_pipeline_prepared", False):
        return None
    descriptor = model_manager.get_pipeline_descriptor() or {}
    model_id = str(
        getattr(model_manager, "active_model_id", "") or descriptor.get("model_id", "")
    )
    sha256 = str(descriptor.get("model_sha256", "") or "").lower()
    if not model_id or not sha256:
        return None
    try:
        return ModelIdentity(
            model_id=model_id,
            engine="pytorch",
            format="safetensors",
            revision="prepared",
            sha256=sha256,
        )
    except ValueError:
        logger.warning("已准备的流水线模型身份非法", exc_info=True)
        return None


def _active_task_graph_model_identity() -> Optional[ModelIdentity]:
    if not model_host.model_loaded or not model_manager.is_loaded:
        return _prepared_pipeline_model_identity()
    engine = backend_id_for(model_manager)
    model_path = str(getattr(model_manager, "_model_path", "") or "")
    model_id = str(getattr(model_manager, "active_model_id", "") or "")
    if engine not in registered_backends() or not model_id or not model_path:
        return None
    if engine == "island":
        # 孤岛模型无本地 artifact：以"端点指纹 + 后端模型名"替代文件摘要，
        # 统计中如实标注为外部端点（不伪装成本地文件，见调研方案 §2.2）。
        island_engine = getattr(model_manager, "_island_engine", None)
        backend_model = str(getattr(island_engine, "model_name", "") or "")
        masked_url = str(getattr(island_engine, "masked_base_url", "") or model_path)
        if not backend_model:
            return None
        digest = hashlib.sha256(
            f"{masked_url}::{backend_model}".encode("utf-8")
        ).hexdigest()
        return ModelIdentity(
            model_id=model_id,
            engine="island",
            format="openai_api",
            revision=f"island-{digest[:12]}",
            sha256=digest,
        )
    try:
        from model_sync import compute_file_sha256, compute_model_sha256

        if engine == "pytorch":
            digest = compute_model_sha256(model_path)
            model_format = "safetensors"
        else:
            digest = compute_file_sha256(model_path)
            model_format = "gguf"
    except Exception:
        logger.warning("无法计算任务链当前模型摘要", exc_info=True)
        return None
    if len(digest) != 64:
        return None
    return ModelIdentity(
        model_id=model_id,
        engine=engine,
        format=model_format,
        revision=f"local-{digest[:12]}",
        sha256=digest,
    )


def _sync_remote_task_worker_providers() -> list[str]:
    """Register stable remote Provider objects without changing request policy."""
    registered = []
    for provider in scheduler.remote_task_worker_providers():
        if not task_graph_coordinator.has_provider(provider.provider_id):
            try:
                task_graph_coordinator.register_provider(provider)
            except ProviderError:
                if not task_graph_coordinator.has_provider(provider.provider_id):
                    raise
        registered.append(provider.provider_id)
    return sorted(registered)


def _eligible_remote_task_worker_provider_ids(
    model_identity: ModelIdentity,
    stage_type: str,
    *,
    limit: int = 4,
) -> list[str]:
    """Return healthy exact-model Workers in deterministic least-loaded order."""
    if not TASK_WORKER_EXPERIMENTAL_ENABLED:
        return []
    providers = {
        provider.provider_id: provider
        for provider in scheduler.remote_task_worker_providers()
    }
    _sync_remote_task_worker_providers()
    statuses = {
        str(item.get("provider_id", "")): item
        for item in task_graph_coordinator.provider_status()
    }
    eligible = []
    for provider_id, provider in providers.items():
        status = statuses.get(provider_id, {})
        if (
            status.get("provider_kind") != "remote_full_worker"
            or not status.get("healthy")
            or not status.get("available")
            or stage_type not in status.get("supported_stage_types", [])
            or not provider.supports_model_identity(model_identity, stage_type)
        ):
            continue
        max_concurrency = max(1, int(status.get("max_concurrency", 1) or 1))
        active = max(0, int(status.get("active_reservations", 0) or 0))
        eligible.append((
            active / max_concurrency,
            active,
            provider_id,
        ))
    eligible.sort()
    return [item[2] for item in eligible[:max(0, int(limit))]]


def _execute_task_worker_stage(
    stage_request: ProviderStageRequest,
    provider_cancel_event: threading.Event,
) -> dict:
    """Execute the shared local/remote Stage contract on a full model."""
    root_input = stage_request.root_input
    options = root_input.get("task_options", {})
    if not isinstance(options, dict):
        raise TaskGraphError("任务 Stage 缺少有效执行参数")
    try:
        candidate_budget = max(
            1, min(int(options.get("candidate_max_tokens", 512)), 512)
        )
        final_budget = max(
            1, min(int(options.get("final_max_tokens", 1024)), 1024)
        )
        temperature = max(
            0.0, min(float(options.get("temperature", 0.7)), 2.0)
        )
        top_p = max(0.0, min(float(options.get("top_p", 0.9)), 1.0))
    except (TypeError, ValueError) as exc:
        raise TaskGraphError("任务 Stage 执行参数无效") from exc
    show_thinking = bool(options.get("show_thinking", False))

    def run_model(
        messages: list[dict],
        max_tokens: int,
        *,
        retry_empty_on_same_provider: bool = False,
    ) -> dict:
        if provider_cancel_event.is_set():
            return {
                "content": "",
                "usage": {},
                "tokens_per_second": 0,
                "model": model_manager.active_model_id,
            }
        result = model_manager.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            _cancel_event=provider_cancel_event,
        )
        raw_content = str(result.get("content", "") or "").strip()
        thinking_content = None
        if show_thinking:
            content, thinking_content = _format_model_response(
                raw_content, show_thinking=True,
            )
        else:
            content = _strip_native_thinking_tags(raw_content)
        if not content and not provider_cancel_event.is_set():
            if retry_empty_on_same_provider:
                raise ProviderExecutionError(
                    "complete model returned an empty aggregate result",
                    code="empty_provider_output",
                    provider_id=stage_request.provider_id,
                    same_provider_retryable=True,
                )
            raise TaskGraphError("完整模型返回空 Stage 结果")
        return {
            "content": content,
            "thinking_content": thinking_content,
            "usage": dict(result.get("usage", {}) or {}),
            "tokens_per_second": result.get("tokens_per_second", 0),
            "model": result.get("model", model_manager.active_model_id),
            "usage_estimated": bool(result.get("usage_estimated", False)),
        }

    if stage_request.stage_type == "full_inference":
        candidate_instructions = {
            "candidate_a": (
                "独立分析用户问题，给出准确、可验证且简洁的候选答案。"
                "不要提及其他候选或任务链。"
            ),
            "candidate_b": (
                "从不同角度独立解决用户问题，重点检查遗漏、反例和不确定性。"
                "输出可直接供后续汇总的候选答案。"
            ),
        }
        instruction = candidate_instructions.get(stage_request.stage_id)
        messages = root_input.get("messages")
        if instruction is None or not isinstance(messages, list):
            raise TaskGraphError("完整推理 Stage 输入无效")
        if show_thinking:
            instruction = f"{instruction}\n\n{THINKING_SYSTEM_PROMPT}"
        return run_model(
            [{"role": "system", "content": instruction}, *messages],
            candidate_budget,
        )
    if stage_request.stage_type == "aggregate":
        message = str(root_input.get("message", "") or "")
        if not message:
            raise TaskGraphError("聚合 Stage 缺少原始问题")
        candidate_payload = {
            stage_id: value.get("content", "")
            for stage_id, value in stage_request.dependencies.items()
            if stage_id != DEPENDENCY_FAILURES_KEY
            and isinstance(value, dict)
            and str(value.get("content", "") or "").strip()
        }
        if not candidate_payload:
            raise TaskGraphError("聚合 Stage 没有可用候选")
        failure_payload = stage_request.dependencies.get(
            DEPENDENCY_FAILURES_KEY, {},
        )
        aggregation_prompt = (
            "请根据原始问题和可用的独立候选，输出一个最终答案。"
            "纠正冲突和明显错误；没有证据时明确不确定性。"
            "只输出最终答案，不描述内部任务链。\n\n"
            f"原始问题：{message}\n\n候选：\n"
            + json.dumps(candidate_payload, ensure_ascii=False)
            + (
                "\n\n未完成候选摘要：\n"
                + json.dumps(failure_payload, ensure_ascii=False)
                if isinstance(failure_payload, dict) and failure_payload
                else ""
            )
        )
        try:
            return run_model(
                ([{"role": "system", "content": THINKING_SYSTEM_PROMPT}]
                 if show_thinking else [])
                + [{"role": "user", "content": aggregation_prompt}],
                final_budget,
                retry_empty_on_same_provider=True,
            )
        except ProviderError:
            raise
        except (TimeoutError, ConnectionError) as exc:
            raise ProviderExecutionError(
                "transient aggregate model execution failed",
                code="provider_execution_failed",
                provider_id=stage_request.provider_id,
                same_provider_retryable=True,
            ) from exc
    if stage_request.stage_type == "layer_forward":
        # ★ 2026-10-03：本函数是 `EngineHost.execute_task_worker_stage` 的**同源副本**
        #   （见 `inference_service/engine_host.py` 顶部注释），PC 侧在 api_server 进程里
        #   走的是**这一份** ⇒ 只改 `engine_host.py` 不会让本机以层段 worker 身份工作
        #   （实测：Surface 连上后每个 offer 都回
        #   `TaskGraphError: 不支持的 Stage 类型: layer_forward`）。
        #   委托同一份实现；但本函数是**模块级**（没有 `self`），所以用一个模块级缓存实例
        #   承载 keep-head 上游（它本来就按 env 创建一次后复用，重载 500MB GGUF 代价高）。
        global _LAYER_STAGE_HOST
        if _LAYER_STAGE_HOST is None:
            from inference_service.engine_host import EngineHost as _EngineHost

            _LAYER_STAGE_HOST = _EngineHost()
        return _LAYER_STAGE_HOST._execute_layer_forward_stage(
            stage_request, provider_cancel_event,
        )
    raise TaskGraphError(f"不支持的 Stage 类型: {stage_request.stage_type}")


#: 模块级缓存：`_execute_task_worker_stage` 是模块级函数（无 `self`），层段 Stage 需要
#: 一个持有 keep-head 上游的载体。上游按 env 创建一次后复用（重载 500MB GGUF 代价高）。
_LAYER_STAGE_HOST = None


def _execute_task_graph_chat_with_slot(
    req: ChatRequest,
    cancel_event: Optional[threading.Event] = None,
    request_deadline: Optional[RequestDeadline] = None,
) -> dict:
    """Execute one workflow while the process-wide task-graph slot is held."""

    request_deadline = coerce_request_deadline(
        request_deadline,
        timeout_seconds=PIPELINE_TIMEOUT,
    )
    request_deadline.require_remaining()

    remote_stage_id = str(req.task_graph_remote_stage or "")
    remote_provider_id = str(req.task_graph_remote_provider_id or "")
    distributed_task_required = (
        req.routing_preference == "distributed_required"
    )
    auto_remote = bool(
        req.task_graph_auto_remote
        or (
            not remote_stage_id
            and req.routing_preference in {
                "distributed_preferred", "distributed_required",
            }
        )
    )
    if bool(remote_stage_id) != bool(remote_provider_id):
        raise coded_http_error(
            400,
            "TASK_GRAPH_MANUAL_REMOTE_FIELDS_INCOMPLETE",
            "N2.1 手动远端执行必须同时指定 Stage 和 Provider ID。",
        )
    if auto_remote and remote_stage_id:
        raise coded_http_error(
            400,
            "TASK_GRAPH_REMOTE_POLICY_CONFLICT",
            "N2.3 自动 Worker 选择不能与 N2.1 手动远端 Stage 同时启用。",
        )
    if remote_stage_id and not TASK_WORKER_EXPERIMENTAL_ENABLED:
        raise coded_http_error(
            409,
            "TASK_WORKER_EXPERIMENT_DISABLED",
            "PC Full Worker 实验调度未启用。请设置 "
            "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED=true 后重启。",
        )
    if auto_remote and not TASK_WORKER_EXPERIMENTAL_ENABLED:
        if distributed_task_required:
            raise coded_http_error(
                503,
                "TASK_WORKER_EXPERIMENT_DISABLED",
                "distributed_required 需要已启用的 PC Full Worker 实验调度。",
            )

    prepared_context = _prepare_chat_context(req.session_id, req.message)
    request_deadline.require_remaining()
    target_session_id = prepared_context.session_id
    history = prepared_context.history

    base_messages = list(history) + [{"role": "user", "content": req.message}]
    root_input = {
        "message": req.message,
        "messages": base_messages,
        "task_options": {
            "candidate_max_tokens": max(1, min(req.max_new_tokens, 512)),
            "final_max_tokens": max(1, min(req.max_new_tokens, 1024)),
            "temperature": req.temperature,
            "top_p": req.top_p,
            "show_thinking": req.show_thinking,
            # ★ 2026-09-19：深度思考**开关**（与「展示」区分），None ⇒ 沿用模板默认。
            "enable_thinking": req.enable_thinking,
        },
    }

    try:
        _ensure_local_task_provider()
    except ProviderError as exc:
        raise HTTPException(
            503,
            {
                "message": "本地任务 Provider 注册失败。",
                "reason": f"{exc.code}: {exc}",
            },
        ) from exc

    model_identity = None
    stages: Optional[list[StageSpec]] = None
    final_stage_id = ""
    auto_provider_ids: list[str] = []
    auto_fallback_reason = ""
    if remote_stage_id:
        remote_providers = {
            provider.provider_id: provider
            for provider in scheduler.remote_task_worker_providers()
        }
        _sync_remote_task_worker_providers()
        remote_provider = remote_providers.get(remote_provider_id)
        if remote_provider is None:
            raise HTTPException(
                404, "The selected remote PC Full Worker does not exist."
            )
        remote_status = next((
            item for item in task_graph_coordinator.provider_status()
            if item.get("provider_id") == remote_provider_id
            and item.get("provider_kind") == "remote_full_worker"
        ), None)
        if remote_status is None:
            raise HTTPException(404, "指定的远端 PC Full Worker Provider 不存在。")
        if not remote_status.get("healthy") or not remote_status.get("available"):
            raise HTTPException(503, "指定的远端 PC Full Worker 当前不可用或正忙。")
        model_identity = _active_task_graph_model_identity()
        if model_identity is None:
            raise HTTPException(409, "手动远端 Stage 要求主节点先加载完整模型并生成精确身份。")
        template_stages, final_stage_id = dual_candidate_template()
        selected_stage = next((
            stage for stage in template_stages
            if stage.stage_id == remote_stage_id
        ), None)
        if (
            selected_stage is None
            or not remote_provider.supports_model_identity(
                model_identity, selected_stage.stage_type,
            )
        ):
            raise HTTPException(
                409,
                {
                    "message": (
                        "The selected remote PC Full Worker does not have "
                        "the exact active model required by this Stage."
                    ),
                    "reason_code": "model_identity_mismatch",
                    "provider_id": remote_provider_id,
                    "stage_id": remote_stage_id,
                },
            )
        stages = [
            replace(
                stage,
                provider=remote_provider_id,
                fallback_providers=(),
                pure=False,
                max_same_provider_retries=0,
            )
            if stage.stage_id == remote_stage_id else stage
            for stage in template_stages
        ]
    elif auto_remote:
        if not TASK_WORKER_EXPERIMENTAL_ENABLED:
            auto_fallback_reason = "task_worker_experiment_disabled"
        else:
            model_identity = _active_task_graph_model_identity()
        if TASK_WORKER_EXPERIMENTAL_ENABLED and model_identity is None:
            auto_fallback_reason = "model_identity_unavailable"
            if distributed_task_required:
                raise coded_http_error(
                    409,
                    "TASK_WORKER_MODEL_IDENTITY_UNAVAILABLE",
                    "distributed_required 需要主节点已加载完整模型并生成精确身份。",
                )
        elif TASK_WORKER_EXPERIMENTAL_ENABLED:
            template_stages, final_stage_id = dual_candidate_template()
            eligible_by_stage: dict[str, list[str]] = {}
            for stage in template_stages:
                if stage.stage_type != "full_inference":
                    continue
                stage_model_identity = (
                    stage.model_identity or model_identity
                )
                eligible = _eligible_remote_task_worker_provider_ids(
                    stage_model_identity,
                    stage.stage_type,
                )
                eligible_by_stage[stage.stage_id] = eligible
                for provider_id in eligible:
                    if provider_id not in auto_provider_ids:
                        auto_provider_ids.append(provider_id)
            if not auto_provider_ids:
                auto_fallback_reason = "no_eligible_remote_provider"
                if distributed_task_required:
                    raise coded_http_error(
                        503,
                        "TASK_WORKER_NO_ELIGIBLE_REMOTE_PROVIDER",
                        "distributed_required 没有可用且模型身份精确匹配的 PC Full Worker。",
                    )
            else:
                selected_by_model: dict[ModelIdentity, int] = {}
                planned_stages = []
                for stage in template_stages:
                    if stage.stage_type != "full_inference":
                        planned_stages.append(stage)
                        continue
                    stage_model_identity = (
                        stage.model_identity or model_identity
                    )
                    eligible = eligible_by_stage.get(stage.stage_id, [])
                    candidate_index = selected_by_model.get(
                        stage_model_identity, 0,
                    )
                    selected_by_model[stage_model_identity] = candidate_index + 1
                    if candidate_index >= len(eligible):
                        planned_stages.append(replace(
                            stage,
                            provider="local_full_model",
                            fallback_providers=(),
                            pure=True,
                        ))
                        continue
                    primary = eligible[candidate_index]
                    other_remotes = [
                        provider_id for provider_id in eligible
                        if provider_id != primary
                    ][:3]
                    planned_stages.append(replace(
                        stage,
                        provider=primary,
                        fallback_providers=tuple(
                            [
                                *other_remotes,
                                *(
                                    [] if distributed_task_required
                                    else ["local_full_model"]
                                ),
                            ]
                        ),
                        pure=True,
                    ))
                stages = planned_stages

    request_id = str(_request_id_ctx.get("-") or "-")
    runtime_context = {
        "local_provider_executor": _execute_task_worker_stage,
        "task_graph_remote_policy": (
            "manual" if remote_stage_id else "auto" if auto_remote else "local"
        ),
    }
    try:
        if stages is None:
            final_output, workflow = task_graph_coordinator.run_template(
                template=req.task_graph_template,
                root_input=root_input,
                request_id=request_id,
                session_id=target_session_id or "default",
                model_identity=model_identity,
                runtime_context=runtime_context,
                workflow_id=req.workflow_id,
                cancel_event=cancel_event,
                request_deadline=request_deadline,
            )
        else:
            final_output, workflow = task_graph_coordinator.run(
                stages=stages,
                final_stage_id=final_stage_id,
                template=req.task_graph_template,
                root_input=root_input,
                request_id=request_id,
                session_id=target_session_id or "default",
                model_identity=model_identity,
                runtime_context=runtime_context,
                workflow_id=req.workflow_id,
                cancel_event=cancel_event,
                request_deadline=request_deadline,
            )
    except WorkflowCancelled as exc:
        raise HTTPException(
            409,
            {"message": "任务链已取消", "workflow_id": exc.workflow_id},
        ) from exc
    except WorkflowExecutionError as exc:
        status_code = 504 if exc.code == "request_deadline_exceeded" else 500
        raise coded_http_error(
            status_code,
            exc.code,
            {
                "message": str(exc),
                "workflow_id": exc.workflow_id,
                "stage_id": exc.stage_id,
            },
        ) from exc
    except WorkflowDeadlineExceeded as exc:
        raise coded_http_error(
            504,
            exc.code,
            {
                "message": str(exc),
                "workflow_id": exc.workflow_id,
            },
        ) from exc
    except TaskGraphUnavailable as exc:
        raise HTTPException(
            503,
            {
                "message": "任务链 journal 写入失败，执行已停止。",
                "reason": str(exc),
            },
        ) from exc
    except TaskGraphError as exc:
        raise HTTPException(400, f"任务链请求无效: {exc}") from exc

    response_text = str(final_output.get("content", "") or "").strip()
    thinking_content = final_output.get("thinking_content")
    if not response_text:
        raise HTTPException(500, "任务链最终聚合结果为空。")

    if distributed_task_required:
        precommit_attempts = [
            attempt
            for stage in workflow.get("stages", [])
            for attempt in stage.get("attempts", [])
            if attempt.get("state") == "completed"
        ]
        precommit_remote_attempts = [
            attempt
            for attempt in precommit_attempts
            if attempt.get("provider_kind") == "remote_full_worker"
        ]
        if not precommit_remote_attempts:
            try:
                task_graph_coordinator.discard_result(
                    workflow["workflow_id"],
                    terminal_reason_code=REASON_REQUEST_REFUSED,
                )
            except TaskGraphUnavailable as exc:
                raise HTTPException(
                    503,
                    {
                        "message": "distributed_required 校验失败且结果无法丢弃。",
                        "reason": str(exc),
                    },
                ) from exc
            raise coded_http_error(
                503,
                "TASK_WORKER_REMOTE_STAGE_NOT_EXECUTED",
                "distributed_required 未执行任何远端 Full Worker Stage。",
            )
        try:
            validate_distributed_completion(
                req.routing_preference,
                {
                    "distributed_used": True,
                    "execution_mode": "task_graph",
                    "workers_used": [
                        str(attempt.get("provider_node_id", "") or "")
                        for attempt in precommit_remote_attempts
                        if attempt.get("provider_node_id")
                    ],
                },
            )
        except DistributedCompletionError as exc:
            try:
                task_graph_coordinator.discard_result(
                    workflow["workflow_id"],
                    terminal_reason_code=REASON_REQUEST_REFUSED,
                )
            except TaskGraphUnavailable as discard_exc:
                raise HTTPException(
                    503,
                    {
                        "message": "发行 profile 拒绝任务图结果且结果无法丢弃。",
                        "reason": str(discard_exc),
                    },
                ) from discard_exc
            raise coded_http_error(
                503,
                "DISTRIBUTED_REQUIRED_RELEASE_ROUTE_FORBIDDEN",
                f"distributed_required 发行路线校验失败：{exc}",
            ) from exc

    try:
        with hold_lock_until_deadline(
            _generation_registry_lock,
            request_deadline,
            operation="generation commit lock",
        ):
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            if cancel_event is not None and cancel_event.is_set():
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
            try:
                workflow = task_graph_coordinator.commit_result(
                    workflow["workflow_id"],
                )
            except WorkflowCancelled as exc:
                raise ChatGenerationCancelled(
                    req.generation_id or "gen_unknown",
                ) from exc
            except WorkflowDeadlineExceeded as exc:
                raise coded_http_error(
                    504, REQUEST_DEADLINE_EXCEEDED, str(exc),
                ) from exc
            except TaskGraphUnavailable:
                raise
    except ChatGenerationCancelled:
        try:
            task_graph_coordinator.discard_result(workflow["workflow_id"])
        except TaskGraphUnavailable as exc:
            raise HTTPException(
                503,
                {
                    "message": "任务链取消无法写入 journal。",
                    "reason": str(exc),
                },
            ) from exc
        raise
    except TaskGraphUnavailable as exc:
        raise HTTPException(
            503,
            {
                "message": "任务链 journal 写入失败，结果未提交。",
                "reason": str(exc),
            },
        ) from exc

    attempts = [
        attempt
        for stage in workflow.get("stages", [])
        for attempt in stage.get("attempts", [])
        if attempt.get("state") == "completed"
    ]
    usages = [
        dict(attempt.get("result_metadata", {}).get("usage", {}) or {})
        for attempt in attempts
    ]
    prompt_tokens = sum(int(usage.get("prompt_tokens", 0) or 0) for usage in usages)
    completion_tokens = sum(
        int(usage.get("completion_tokens", 0) or 0) for usage in usages
    )
    providers = sorted({
        str(attempt.get("provider", "") or "")
        for attempt in attempts
        if attempt.get("provider")
    })
    serving_node_id = scheduler.get_effective_node_id()
    participating_nodes = sorted({
        str(attempt.get("provider_node_id", "") or serving_node_id)
        for attempt in attempts
    }) or [serving_node_id]
    remote_attempts = [
        attempt for attempt in attempts
        if attempt.get("provider_kind") == "remote_full_worker"
    ]
    remote_nodes = sorted({
        str(attempt.get("provider_node_id", "") or "")
        for attempt in remote_attempts
        if attempt.get("provider_node_id")
    })
    remote_used = bool(remote_attempts)
    if distributed_task_required and not remote_used:
        raise coded_http_error(
            503,
            "TASK_WORKER_REMOTE_STAGE_NOT_EXECUTED",
            "distributed_required 未执行任何远端 Full Worker Stage。",
        )
    provider_status_by_id = {
        str(item.get("provider_id", "")): item
        for item in task_graph_coordinator.provider_status()
    }
    planned_remote_nodes = sorted({
        str(provider_status_by_id.get(provider_id, {}).get("node_id", "") or "")
        for provider_id in auto_provider_ids
        if provider_status_by_id.get(provider_id, {}).get("node_id")
    })
    retried_stages = [
        stage for stage in workflow.get("stages", [])
        if int(stage.get("retry_count", 0) or 0) > 0
    ]
    retry_error_codes = [
        str(stage.get("last_retry_error_code", "") or "")
        for stage in retried_stages
        if stage.get("last_retry_error_code")
    ]
    same_provider_retry_count = sum(
        int(stage.get("same_provider_retry_count", 0) or 0)
        for stage in retried_stages
    )
    total_retry_count = sum(
        int(stage.get("retry_count", 0) or 0)
        for stage in retried_stages
    )
    reassignment_count = max(
        0, total_retry_count - same_provider_retry_count,
    )
    fallback_used = reassignment_count > 0 or bool(
        auto_remote and not auto_provider_ids
    )
    retry_reason = retry_error_codes[0] if retry_error_codes else ""
    fallback_reason = (
        retry_reason if reassignment_count > 0 else auto_fallback_reason
    )
    metrics = _augment_chat_metrics(
        {
            "engine": backend_id_for(model_manager),
            "execution_mode": "task_graph",
            "provider": (
                providers[0] if len(providers) == 1 else "task_graph"
            ),
            "orchestrator": "task_graph",
            "subproviders": providers,
            "workflow_id": workflow["workflow_id"],
            "workflow_template": workflow["template"],
            "workflow_state": workflow["state"],
            "partial_result": bool(workflow.get("partial_result", False)),
            "stage_retry_count": total_retry_count,
            "same_provider_retry_count": same_provider_retry_count,
            "reassignment_count": reassignment_count,
            "retry_reason": retry_reason,
            "stage_count": workflow["stage_count"],
            "stage_attempt_count": workflow["attempt_count"],
            "nodes_planned": (
                1 + len(planned_remote_nodes)
                if auto_remote else len(participating_nodes)
            ),
            "nodes_participated": len(participating_nodes),
            "participating_nodes": participating_nodes,
            "distributed_requested": bool(remote_stage_id or auto_remote),
            "distributed_used": remote_used,
            "distributed_kind": (
                "task_graph_remote_manual"
                if remote_used and remote_stage_id
                else "task_graph_remote_auto"
                if remote_used and auto_remote
                else "task_graph_local_fallback"
                if auto_remote
                else "task_graph_local_poc"
            ),
            "workers_used": remote_nodes,
            "manual_remote_stage": remote_stage_id,
            "manual_remote_provider": remote_provider_id,
            "auto_remote_enabled": auto_remote,
            "auto_remote_providers": auto_provider_ids,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "usage_estimated": any(
                bool(attempt.get("result_metadata", {}).get(
                    "usage_estimated", False,
                ))
                for attempt in attempts
            ),
            "elapsed_seconds": workflow["duration_seconds"],
            "fallback": fallback_used,
            "fallback_reason": fallback_reason,
        },
        req,
        route=(
            f"{_chat_origin(req)}_to_task_graph_manual_remote"
            if remote_stage_id
            else f"{_chat_origin(req)}_to_task_graph_auto_remote"
            if auto_remote
            else f"{_chat_origin(req)}_to_local_task_graph"
        ),
    )
    completed_history = [
        *history,
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": response_text},
    ]
    followups = _fallback_followups(completed_history, [])
    save_metrics = dict(metrics)
    save_metrics["followups"] = followups
    _commit_chat_context_turn(
        target_session_id, req.message, response_text, save_metrics,
        expected_revision=prepared_context.revision,
    )

    conversation_stats["total_prompt_tokens"] += prompt_tokens
    conversation_stats["total_generated_tokens"] += completion_tokens
    conversation_stats["total_time_seconds"] += workflow["duration_seconds"]
    conversation_stats["rounds"] += 1
    try:
        scheduler.record_task_complete(success=True)
    except Exception:
        pass

    return {
        "content": response_text,
        "thinking_content": thinking_content,
        "metrics": metrics,
        "followups": followups,
    }


def _execute_chat_full(
    req: ChatRequest,
    cancel_event: Optional[threading.Event] = None,
    request_deadline: Optional[RequestDeadline] = None,
) -> dict:
    """
    执行完整聊天流程 — 从 /api/chat 提取的共用核心逻辑。

    处理: 会话切换、自动标题、客户端转发、流水线推理、
          llama.cpp、PyTorch、历史维护、DB 持久化、追问生成。

    Returns:
        {"content": str, "thinking_content": str|None,
         "metrics": dict, "followups": list[str]}

    Raises:
        HTTPException: 模型未加载、OOM、推理失败
    """
    global kv_cache, conversation_stats
    request_deadline = coerce_request_deadline(
        request_deadline,
        timeout_seconds=PIPELINE_TIMEOUT,
    )
    request_deadline.require_remaining()
    # T9.5：distributed_required 无分布式路径时明确失败（full 模式）
    routing_gate = _routing_gate_error(req)
    if routing_gate:
        raise HTTPException(400, routing_gate)
    _raise_if_generation_cancelled(cancel_event, req.generation_id)

    # ---- 多会话支持 ----
    prepared_context = _prepare_chat_context(req.session_id, req.message)
    request_deadline.require_remaining()
    target_session_id = prepared_context.session_id
    history = prepared_context.history

    # ---- 路线 B：外部推理服务整请求路由（数据作用域门控，默认不出集群）----
    # 决策为纯函数（external_provider.decide_external_route）；不满足条件时
    # use_external=False，直接落回下方既有本地/流水线逻辑，行为完全不变。
    # （作用域拒绝的 INFO 日志由端点入口统一记录，每请求一次）
    external_fallback_reason = ""
    _ext_decision = _external_route_decision(req)
    local_native_image = bool(
        req.image_data_urls and _local_native_vision_available()
    )
    if (
        local_native_image
        and not _ext_decision.use_external
        and len(req.image_data_urls) != 1
    ):
        raise HTTPException(
            400,
            "本地 MTMD 图像回退仅支持一张图片；多图请求不能静默丢弃其余图片",
        )
    if req.image_data_urls and not _ext_decision.use_external and not local_native_image:
        raise HTTPException(
            400,
            "当前本地模型没有原生视觉能力，且 external_api 不可用或未获授权："
            f"{_ext_decision.reason}",
        )
    if _ext_decision.use_external:
        try:
            return _execute_external_chat(
                req,
                history,
                target_session_id,
                prepared_context.revision,
                cancel_event,
            )
        except ChatGenerationCancelled:
            raise
        except HTTPException:
            raise
        except Exception as exc:
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            if req.image_data_urls and not local_native_image:
                raise HTTPException(
                    502,
                    f"多模态外部推理服务调用失败，禁止丢弃图片后回退：{exc}",
                ) from exc
            if not model_host.model_loaded or not model_manager.is_loaded:
                if req.prefer_external:
                    raise HTTPException(
                        502,
                        f"外部推理服务调用失败，且本地无可用推理引擎：{exc}",
                    ) from exc
                try:
                    # 回退必须复用统一约束：转发型从节点不加载本地模型（落到
                    # 下方转发分支），被预约的 PyTorch 流水线从节点原样上抛 503。
                    # 直接调 _auto_load_default_model 会绕过这两条不变量。
                    _ensure_chat_model_or_forwarding()
                except HTTPException:
                    raise
                except Exception as load_exc:
                    raise HTTPException(
                        502,
                        f"外部推理服务调用失败（{exc}），"
                        f"且本地模型加载失败：{load_exc}",
                    ) from exc
            external_fallback_reason = f"external_api_failed: {exc}"
            logger.warning(f"外部推理服务调用失败: {exc}，回退到本地推理路径")

    # ---- 分布式推理路由：从节点转发给主节点（local_only 强制本地）----
    if (req.routing_preference != "local_only"
            and scheduler.get_distributed_inference_enabled()
            and RUN_MODE == "distributed"
            and scheduler._effective_role() == "client"):
        try:
            result = scheduler.forward_inference_to_master(
                message=req.message,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                show_thinking=req.show_thinking,
                enable_thinking=req.enable_thinking,
                routing_preference=req.routing_preference,
                session_id=req.session_id,
                messages=list(history) + [{"role": "user", "content": req.message}],
                request_id=_request_id_ctx.get("-"),   # L5: 链路追踪
                _cancel_event=cancel_event,
                _request_deadline=request_deadline,
            )
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            if result.get("reason_code") == REQUEST_DEADLINE_EXCEEDED:
                raise coded_http_error(
                    504,
                    REQUEST_DEADLINE_EXCEEDED,
                    result.get("error") or "request deadline exceeded",
                )
            if result.get("status") == "ok":
                response_text = result.get("content", "")
                forward_metrics = _augment_chat_metrics(
                    result.get("metrics", {}),
                    req,
                    engine="distributed_forward",
                    execution_mode="forwarded_to_master",
                    route="pc_client_forward_to_master",
                )
                if external_fallback_reason and not forward_metrics.get(
                    "fallback_reason",
                ):
                    forward_metrics["fallback"] = True
                    forward_metrics["fallback_reason"] = external_fallback_reason

                _enforce_distributed_required(
                    req,
                    forward_metrics,
                    detail="从节点转发结果未标记为分布式执行",
                )
                _commit_chat_context_turn(
                    target_session_id, req.message, response_text, forward_metrics,
                    expected_revision=prepared_context.revision,
                )

                conversation_stats["rounds"] += 1
                try:
                    scheduler.record_task_complete(success=True)
                except Exception:
                    pass

                master_followups = result.get("followups", [])
                if master_followups:
                    followups = master_followups[:3]
                else:
                    followups = _fallback_followups(history, [])

                return {
                    "content": response_text,
                    "thinking_content": result.get("thinking_content"),
                    "metrics": forward_metrics,
                    "followups": followups,
                }
            elif result.get("status") == "disconnected":
                logger.warning("分布式推理转发失败（未连接主节点），回退到本地推理")
                _enforce_distributed_required(req, detail="从节点未连接主节点")
            elif result.get("status") == "timeout":
                logger.warning("分布式推理转发超时，回退到本地推理")
                _enforce_distributed_required(req, detail="分布式转发超时")
            else:
                logger.warning(f"分布式推理转发失败: {result.get('error', 'unknown')}，回退到本地推理")
                _enforce_distributed_required(
                    req,
                    detail=str(result.get("error", "分布式转发失败")),
                )
        except ChatGenerationCancelled:
            raise
        except HTTPException:
            raise
        except Exception as e:
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            logger.warning(f"分布式推理转发异常: {e}，回退到本地推理")
            _enforce_distributed_required(req, detail=str(e))

        if _pipeline_worker_is_reserved():
            raise HTTPException(
                503,
                "本设备正作为 PyTorch 分层从节点，"
                "当前无法转发到主节点，已拒绝覆盖分层模型。",
            )
        if not model_host.model_loaded or not model_manager.is_loaded:
            if req.routing_preference != 'distributed_required':
                _auto_load_default_model()

    if _pipeline_worker_is_reserved():
        raise HTTPException(
            503,
            "本设备正作为 PyTorch 分层从节点，"
            "请先断开主节点或明确切换本地模型。",
        )

    # ---- 分布式流水线推理路径（主节点 + PyTorch 引擎 + 从节点可用；local_only 跳过）----
    pipeline_attempted = False
    pipeline_failure_reason = ""
    if (req.routing_preference != "local_only"
            and scheduler.get_distributed_inference_enabled()
            and RUN_MODE == "distributed"
            and scheduler._effective_role() == "master"
            and runtime_supports(model_manager, Capability.FORWARD_LAYERS)):
        pipeline_attempted = True
        try:
            pipeline_result = scheduler.run_pipeline_safe(
                req.message,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                session_id=req.session_id,
                messages=list(history) + [{"role": "user", "content": req.message}],
                show_thinking=req.show_thinking,
                enable_thinking=req.enable_thinking,
                # ★ A1 / X 档：请求级路由偏好要贯通到流水线内部（`local_only` ⇒ 本段不委派给
                #   远端 relay 段）。此前只在流水线**入口**用它做"是否走分布式路径"的判断，
                #   没往 `_run_pipeline` 里传，于是 relay 委派完全不区分请求。
                routing_preference=req.routing_preference,
                _require_distributed=(req.routing_preference == "distributed_required"),
                _force_distributed_assignment=True,
                _cancel_event=cancel_event,
                _request_deadline=request_deadline,
            )
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            if pipeline_result.get("error"):
                if (
                    pipeline_result.get("reason_code")
                    == REQUEST_DEADLINE_EXCEEDED
                ):
                    raise coded_http_error(
                        504,
                        REQUEST_DEADLINE_EXCEEDED,
                        pipeline_result.get("error")
                        or "request deadline exceeded",
                    )
                pipeline_failure_reason = str(pipeline_result["error"])
                # ★ 2026-10-05（DIST-4）：pipeline 失败时除了 `error` 字符串，还要
                #   保住 `metrics.pipeline_readiness` —— 那里带**具名 reason_code**
                #   （如 `route_a_mixed_legacy_execution_bridge_not_ready` /
                #   `pipeline_recovery_pending` / `worker_stage_offer_not_ready`），
                #   而 `error` 文案可能已被折叠成更通用的表述。此前整份 metrics 被
                #   丢弃 ⇒ 「为什么失败」在请求级不可追溯。
                _pipeline_readiness = None
                _raw_metrics = pipeline_result.get("metrics")
                if isinstance(_raw_metrics, dict):
                    _pipeline_readiness = _raw_metrics.get("pipeline_readiness")
                if isinstance(_pipeline_readiness, dict) and _pipeline_readiness:
                    pipeline_failure_reason = (
                        f"{pipeline_failure_reason} | readiness="
                        f"{_pipeline_readiness.get('reason_code', '')}"
                        f": {_pipeline_readiness.get('reason', '')}"
                    )
                logger.warning(f"流水线推理失败: {pipeline_result['error']}，回退到本地推理")
                _enforce_distributed_required(
                    req,
                    detail=str(pipeline_result["error"]),
                )
            else:
                response_text = pipeline_result.get("response", "")
                if not response_text:
                    logger.warning("流水线返回空响应，回退到本地推理")
                    _enforce_distributed_required(req, detail="流水线返回空响应")
                else:
                    pipeline_metrics = _augment_chat_metrics(
                        pipeline_result.get("metrics", {}),
                        req,
                        engine="distributed_pipeline",
                        execution_mode="distributed_pipeline",
                        route="master_pipeline",
                    )
                    _enforce_distributed_required(
                        req,
                        pipeline_metrics,
                        detail="流水线结果未标记为分布式执行",
                    )
                    if external_fallback_reason and not pipeline_metrics.get(
                        "fallback_reason",
                    ):
                        pipeline_metrics["fallback"] = True
                        pipeline_metrics["fallback_reason"] = (
                            external_fallback_reason
                        )
                    _commit_chat_context_turn(
                        target_session_id, req.message, response_text, pipeline_metrics,
                        expected_revision=prepared_context.revision,
                    )

                    conversation_stats["rounds"] += 1
                    if not pipeline_metrics.get("distributed_used"):
                        try:
                            scheduler.record_task_complete(success=True)
                        except Exception:
                            pass

                    if _pipeline_model_is_prepared():
                        followups = _fallback_followups(history, [])
                    elif backend_id_for(model_manager) in ("llama_cpp", "island"):
                        followups = _generate_followups_llama(history)
                    elif backend_id_for(model_manager) == "pytorch":
                        # 流水线成功后主节点仍保留首段裁剪模型，不能拿它生成追问。
                        followups = _fallback_followups(history, [])
                    else:
                        followups = _fallback_followups(history, [])

                    return {
                        "content": response_text,
                        "thinking_content": pipeline_result.get("thinking"),
                        "metrics": pipeline_metrics,
                        "followups": followups,
                    }
        except ChatGenerationCancelled:
            raise
        except RequestDeadlineExceeded as exc:
            raise coded_http_error(
                504, REQUEST_DEADLINE_EXCEEDED, str(exc),
            ) from exc
        except HTTPException:
            raise
        except Exception as e:
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            pipeline_failure_reason = str(e)
            logger.warning(f"流水线推理异常: {e}，回退到本地推理")
            _enforce_distributed_required(req, detail=str(e))

    if req.routing_preference == "distributed_required":
        raise HTTPException(503, "distributed_required 执行失败：当前引擎未完成分布式流水线")

    if _pipeline_model_is_prepared() and not pipeline_attempted:
        raise HTTPException(
            409 if req.routing_preference == "local_only" else 503,
            "当前模型仅以分布式流水线模式准备，禁止作为本地完整模型执行；"
            "请等待流水线就绪或显式加载完整模型。",
        )

    # ---- 流水线失败 ⇒ 整模回退守卫（★ 2026-10-09：**引擎无关**）----
    #   ★ 原先只覆盖 llama.cpp / 孤岛（下面那个分支），PyTorch 路径漏了 ⇒ pipeline 尝试
    #   失败后会**静默坠入整模**：`ensure_full_model()` 对 distributed-only 模型抛
    #   RuntimeError，随后被函数尾部的 `except Exception` 泛化成
    #   `500 推理失败: 当前模型以分布式专用模式准备…`，真实原因（`pipeline_failure_reason`）
    #   在响应里彻底丢失。实测代价：64-token 请求 duration 仅 2779ms、没有任何 Route-A
    #   prefill 却报 500，排查时把"为什么没走分布式"误导成了"路由判据问题"（实际与
    #   max_new_tokens 无关）。这里把它提到引擎分支之前，失败即**具名 503**。
    if pipeline_attempted:
        ensure_full_guard = getattr(model_manager, "ensure_full_model", None)
        if not callable(ensure_full_guard):
            raise HTTPException(
                503,
                "分布式流水线失败且当前引擎不提供整模回退校验",
            )
        try:
            ensure_full_guard()
        except Exception as exc:
            raise HTTPException(
                503,
                "分布式流水线失败（原因: "
                f"{pipeline_failure_reason or '未记录'}），整模回退已拒绝: {exc}",
            ) from exc

    # ---- llama.cpp / 孤岛引擎路径（整请求推理，不参与层拆分）----
    if backend_id_for(model_manager) in ("llama_cpp", "island"):
        if pipeline_attempted:
            # Mixed A1 relay + A3 Route-A failures must reuse the full-model
            # guard before this direct chat fallback.  Otherwise a partial
            # GGUF can bypass the scheduler check and return HTTP 200 garbage.
            ensure_full = getattr(model_manager, "ensure_full_model", None)
            if not callable(ensure_full):
                raise HTTPException(
                    503,
                    "分布式流水线失败且当前引擎不提供整模回退校验",
                )
            try:
                ensure_full()
            except Exception as exc:
                raise HTTPException(
                    503,
                    f"分布式流水线失败，整模回退已拒绝: {exc}",
                ) from exc
        try:
            engine_name = backend_id_for(model_manager)
            request_history = [
                *history,
                {"role": "user", "content": req.message},
            ]
            if local_native_image:
                if engine_name != "llama_cpp":
                    raise RuntimeError("本地原生图像请求要求 llama.cpp MTMD 引擎")
                with materialize_image_data_url(req.image_data_urls[0]) as image_path:
                    result = model_manager.chat_image(
                        image_path=image_path,
                        prompt=req.message,
                        max_tokens=min(req.max_new_tokens + 96, 512),
                        max_answer_tokens=min(req.max_new_tokens, 256),
                        temperature=req.temperature,
                        top_p=req.top_p,
                        _cancel_event=cancel_event,
                    )
            else:
                result = model_manager.chat(
                    messages=request_history,
                    max_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    show_thinking=req.show_thinking,
                    enable_thinking=req.enable_thinking,
                    _cancel_event=cancel_event,
                )
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            response_text, thinking_content = _format_model_response(
                result.get("content", ""),
                req.show_thinking,
            )
            completed_history = [
                *request_history,
                {"role": "assistant", "content": response_text},
            ]
            tokens_per_sec = result.get("tokens_per_second", 0)
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            route_suffix = f"{engine_name}_mtmd" if local_native_image else engine_name
            local_route = f"{_chat_origin(req)}_to_master_local_{route_suffix}"
            fallback_reason = ""
            if external_fallback_reason:
                # 路线 B 外部路由失败后的本地回退（原因优先展示外部失败）
                fallback_reason = external_fallback_reason
            elif pipeline_failure_reason:
                fallback_reason = pipeline_failure_reason
            elif scheduler.get_distributed_inference_enabled() and RUN_MODE == "distributed":
                if engine_name == "island":
                    fallback_reason = "island engine delegates whole-request inference to the TP island"
                else:
                    fallback_reason = "llama.cpp engine does not support layer-split pipeline"
            metrics = _augment_chat_metrics(
                {
                    "engine": engine_name,
                    "execution_mode": f"local_{route_suffix}",
                    "route": local_route,
                    "native_mtmd": local_native_image,
                    "tokens_per_second": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                    "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                    "generated_tokens": completion_tokens,
                    "completion_tokens": completion_tokens,
                    "usage": usage,
                    "fallback": bool(fallback_reason),
                    "fallback_reason": fallback_reason,
                },
                req,
            )

            followups = _generate_followups_llama(
                completed_history, cancel_event,
            )
            _raise_if_generation_cancelled(cancel_event, req.generation_id)

            save_metrics = dict(metrics)
            save_metrics["followups"] = followups
            _commit_chat_context_turn(
                target_session_id, req.message, response_text, save_metrics,
                expected_revision=prepared_context.revision,
            )

            conversation_stats["total_generated_tokens"] += completion_tokens
            conversation_stats["rounds"] += 1
            try:
                scheduler.record_task_complete(success=True)
            except Exception:
                pass

            return {
                "content": response_text,
                "thinking_content": thinking_content,
                "metrics": metrics,
                "followups": followups,
            }
        except ChatGenerationCancelled:
            raise
        except HTTPException:
            raise
        except Exception as e:
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            try:
                scheduler.record_task_error()
            except Exception:
                pass
            # 孤岛路径的失败不应记成 llama.cpp（llama_cpp 路径日志保持原样）
            _engine_label = (
                "孤岛引擎"
                if backend_id_for(model_manager) == "island"
                else "llama.cpp"
            )
            logger.error(f"{_engine_label} 推理失败: {e}", exc_info=True)
            raise HTTPException(500, f"推理失败: {str(e)}")

    # ---- PyTorch 引擎路径（CUDA/独显）----
    try:
        model_manager.ensure_full_model()
        tier_max = model_host.generation_config.get("tier_max_new_tokens", model_host.generation_config["max_new_tokens"])
        thinking_budget = 384 if req.show_thinking else 0
        effective_max = min(req.max_new_tokens + thinking_budget,
                            tier_max + thinking_budget,
                            4096)
        model_host.generation_config["max_new_tokens"] = effective_max
        model_host.generation_config["temperature"] = req.temperature
        model_host.generation_config["top_p"] = req.top_p

        request_history = [
            *history,
            {"role": "user", "content": req.message},
        ]

        tokenizer = model_manager.tokenizer
        thinking_prompt = THINKING_SYSTEM_PROMPT if req.show_thinking else None
        thinking_prefill = "【思考】\n" if req.show_thinking else None
        prompt = _build_model_chat_prompt(
            tokenizer,
            request_history,
            system_prompt=thinking_prompt,
            assistant_prefill=thinking_prefill,
        )
        # ★ 数值对照诊断：与 `scheduler_pipeline._run_pipeline` 的同一行日志配对，
        #   用于确认两条路喂给模型的 prompt 是否逐字一致 —— 这是跨机 relay
        #   数值（per-token argmax）对照成立的前提。
        logger.info(
            "本地 prompt: chars=%d head=%r tail=%r",
            len(prompt), prompt[:70], prompt[-50:],
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_manager.get_device())
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_manager.get_device())
        prompt_len = input_ids.shape[1]
        stop_sequences = model_manager._merge_stop_sequences(None)
        generation_kwargs = {}
        eos_token_ids = model_manager._get_generation_eos_token_ids(stop_sequences)
        if eos_token_ids is not None:
            generation_kwargs["eos_token_id"] = eos_token_ids
        stop_criteria_kwargs = (
            {"cancel_event": cancel_event} if cancel_event is not None else {}
        )
        stop_criteria = model_manager._build_stop_criteria(
            stop_sequences, prompt_len, **stop_criteria_kwargs,
        )
        if stop_criteria is not None:
            generation_kwargs["stopping_criteria"] = stop_criteria

        t0 = time.time()
        with torch.no_grad():
            outputs = _safe_torch_model().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=effective_max,
                temperature=req.temperature if req.temperature > 0 else 1.0,
                top_p=req.top_p,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
        _raise_if_generation_cancelled(cancel_event, req.generation_id)
        elapsed = time.time() - t0

        generated_ids = outputs[0][prompt_len:]
        raw_text = model_manager._decode_generated_ids(generated_ids, stop_sequences).strip()

        # ★ 2026-10-07（与 Route-A 同源的真机根因）：判据必须是「模板注入的思考块**尚未闭合**」，
        #   而不是「prompt 里出现过 `<think`」。qwen3-5-2b 模板在 `enable_thinking` 非 true 时
        #   注入的是已闭合的 `'<think>\n\n</think>\n\n'` ⇒ 生成段只含正文，永远等不到
        #   `</think>`，旧判据会把正文整段丢掉（判成空响应）。
        _prompt_tail = prompt[-128:].lower()
        native_thinking_prompt = bool(
            not req.show_thinking
            and "<think" in _prompt_tail
            and "</think>" not in _prompt_tail
        )
        parsed_text = raw_text
        if req.show_thinking and not native_thinking_prompt and "<think" not in raw_text.lower():
            parsed_text = "【思考】\n" + raw_text
        response_text, thinking_content = _format_model_response(
            parsed_text,
            req.show_thinking,
            native_thinking_prompt=native_thinking_prompt,
        )

        completed_history = [
            *request_history,
            {"role": "assistant", "content": response_text},
        ]

        new_tokens = len(generated_ids)
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        metrics = _augment_chat_metrics(
            {
                "engine": "pytorch",
                "execution_mode": "local_pytorch",
                "route": f"{_chat_origin(req)}_to_master_local_pytorch",
                "prompt_tokens": prompt_len,
                "new_tokens": new_tokens,
                "generated_tokens": new_tokens,
                "total_tokens": prompt_len + new_tokens,
                "elapsed_seconds": round(elapsed, 3),
                "tokens_per_second": round(tokens_per_sec, 1),
                "gpu_memory_mb": round(torch.cuda.memory_allocated() / (1024**2), 1)
                if torch.cuda.is_available()
                else 0,
            },
            req,
            # ★ 2026-10-05（DIST-4）：此前这里**只用** `external_fallback_reason`，
            #   把同函数内 `:3783` / `:3849` 写入的 `pipeline_failure_reason` 丢掉了
            #   ⇒ master 为 PyTorch 引擎时，「pipeline 失败 ⇒ 本地整模回退」的 metrics
            #   是 `distributed_used=False` + `fallback=False` + `fallback_reason=""`。
            #   这与「请求从来没走分布式」完全无法区分 —— 正是 DIST-4 要打击的
            #   「整模回退伪装成普通本地推理」。现在两者取或，并在来自 pipeline 失败
            #   时保留来源标识，使单条响应即可回溯到真实失败原因。
            fallback=bool(external_fallback_reason or pipeline_failure_reason),
            fallback_reason=(
                external_fallback_reason
                or (
                    f"pipeline_failed_then_local_pytorch: {pipeline_failure_reason}"
                    if pipeline_failure_reason else ""
                )
            ),
        )

        followups = _generate_followups(
            completed_history,
            tokenizer,
            _safe_torch_model(),
            model_manager.get_device(),
            cancel_event,
        )
        _raise_if_generation_cancelled(cancel_event, req.generation_id)

        save_metrics = dict(metrics)
        save_metrics["followups"] = followups
        _commit_chat_context_turn(
            target_session_id, req.message, response_text, save_metrics,
            expected_revision=prepared_context.revision,
        )

        conversation_stats["total_prompt_tokens"] += prompt_len
        conversation_stats["total_generated_tokens"] += new_tokens
        conversation_stats["total_time_seconds"] += elapsed
        conversation_stats["rounds"] += 1
        try:
            scheduler.record_task_complete(success=True)
        except Exception:
            pass

        logger.info(
            f"推理完成: {new_tokens} tokens / {elapsed:.2f}s = {tokens_per_sec:.1f} tok/s"
        )

        return {
            "content": response_text,
            "thinking_content": thinking_content,
            "metrics": metrics,
            "followups": followups,
        }

    except ChatGenerationCancelled:
        raise
    except RequestDeadlineExceeded as exc:
        raise coded_http_error(
            504, REQUEST_DEADLINE_EXCEEDED, str(exc),
        ) from exc
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        try:
            scheduler.record_task_error()
        except Exception:
            pass
        if kv_cache:
            kv_cache.clear()
        _get_active_history().clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(507, "GPU 显存不足（OOM），已自动清空对话历史。请缩短消息后重试。")

    except Exception as e:
        try:
            scheduler.record_task_error()
        except Exception:
            pass
        logger.error(f"推理异常: {e}", exc_info=True)
        raise HTTPException(500, f"推理失败: {str(e)}")


def _execute_requested_chat(
    req: ChatRequest,
    cancel_event: Optional[threading.Event] = None,
    request_deadline: Optional[RequestDeadline] = None,
) -> dict:
    if req.execution_mode == "task_graph":
        return _execute_task_graph_chat(
            req, cancel_event, request_deadline,
        )
    return _execute_chat_full(req, cancel_event, request_deadline)


def _commit_interactive_history(
    session_id: Optional[str],
    user_message: str,
    response_text: str,
    metrics: dict,
) -> bool:
    """Compatibility wrapper for the shared completed-turn transaction."""
    return _commit_chat_context_turn(
        session_id, user_message, response_text, metrics,
    )


def _should_forward_chat_to_master() -> bool:
    return bool(
        scheduler.get_distributed_inference_enabled()
        and RUN_MODE == "distributed"
        and scheduler._effective_role() == "client"
    )


def _pipeline_worker_is_reserved() -> bool:
    check = getattr(scheduler, "has_pipeline_worker_reservation", None)
    return bool(callable(check) and check())


def _pipeline_model_is_prepared() -> bool:
    """Return whether the master owns metadata for a distributed-only model."""
    manager = _peek_model_manager()
    return bool(
        manager is not None
        and getattr(manager, "is_pipeline_prepared", False)
    )


def _ensure_chat_model_or_forwarding(req: Optional[ChatRequest] = None) -> None:
    """Load a local model only when this request cannot be master-forwarded."""
    if req is not None and _external_route_decision(req).use_external:
        # 路线 B：本请求将整体路由到外部推理服务，无需本地模型。
        # 外部失败时的本地回退在 _execute_chat_full 内按需加载模型。
        return
    if req is not None and req.routing_preference == "local_only":
        # T9.5：local_only 强制本地执行，从节点也不转发（能力不足时由下方
        # 检查给出明确错误，而不是绕过网关转发）。
        pass
    elif _should_forward_chat_to_master():
        return
    if _pipeline_worker_is_reserved():
        raise HTTPException(
            503,
            "本设备正作为 PyTorch 分层从节点，不能加载本地完整模型。",
        )
    if _pipeline_model_is_prepared():
        if req is not None and req.routing_preference == "local_only":
            raise HTTPException(
                409,
                "当前模型仅以分布式流水线模式准备；local_only 请求需要先显式加载完整模型。",
            )
        return
    if (
        model_host.model_loaded and model_manager.is_loaded
    ):
        return
    if req is not None and req.routing_preference == 'distributed_required':
        raise coded_http_error(
            503,
            'DISTRIBUTED_PIPELINE_NOT_READY',
            'distributed_required_pipeline_not_ready',
        )
    _auto_load_default_model()


def _auto_load_default_model():
    """自动加载默认模型（用于 thin client / Android 首次请求时服务端无模型的情况）。"""
    global kv_cache, conversation_stats

    import config as cfg
    import glob

    # 0. TP 孤岛引擎优先（启用即为孤岛网关节点，无本地文件依赖）
    if getattr(cfg, "ISLAND_ENABLED", False) and getattr(cfg, "ISLAND_BASE_URL", ""):
        from island_engine import mask_island_url

        resolution = resolve_model_load(
            ModelLoadFacts(
                model_id=None,
                island_enabled=True,
                island_base_url=str(cfg.ISLAND_BASE_URL),
            ),
            requested_engine="island",
            requested_quant="island",
        )

        logger.info(
            f"自动加载孤岛引擎: endpoint={mask_island_url(cfg.ISLAND_BASE_URL)}"
        )
        t0 = time.time()
        cfg.INFERENCE_ENGINE = resolution.engine
        cfg.QUANT_TYPE = resolution.quant_type
        cfg.USE_COMPILE = False
        _run_exclusive_model_change(
            lambda: model_manager.load_model(
                profile=device_profile,
                engine=resolution.engine,
                quant_type=resolution.quant_type,
                resolution=resolution,
            )
        )
        _init_kv_cache()
        conversation_stats = {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        }
        model_host.model_loaded = True
        model_host.current_quant = resolution.quant_type
        scheduler.refresh_task_worker_capabilities()
        logger.info(f"✅ 孤岛引擎自动连接完成 ({time.time() - t0:.1f}s)")
        return

    # ★ 2026-09-19：**按设备画像**选择默认模型（用户裁定：边缘/轻薄本 <1B，PC ~2B）。
    #   此前这里直接读 `cfg.GGUF_MODEL_PATH` / `cfg.MODEL_PATH` —— 它们是**静态常量**
    #   （且原指向已退役的 `qwen-1_8b`）⇒ 无论什么设备都加载同一个模型。
    #   现在优先用画像模型自己的路径，取不到才回退到旧的「扫 models 目录」逻辑。
    _active = cfg.get_active_model_paths() if hasattr(cfg, "get_active_model_paths") else {}
    _active_id = str(_active.get("model_id") or "")
    _active_gguf = str(_active.get("gguf_path") or "")
    _active_safetensors = str(_active.get("model_path") or "")

    # 1. 优先查找 GGUF 文件（llama.cpp 引擎，不依赖 transformers/bitsandbytes）
    gguf_candidates = []
    gguf_configured = (
        _active_gguf
        if _active_id
        else cfg.GGUF_MODEL_PATH
    )
    if os.path.isfile(gguf_configured):
        gguf_candidates.append(gguf_configured)
    # 只有旧配置没有画像 model_id 时才允许目录扫描；已有 model_id 时若偷换成
    # 目录里的另一个 GGUF，会造成 active identity 与真实工件不一致。
    models_dir = os.path.dirname(gguf_configured)
    if not _active_id and os.path.isdir(models_dir):
        for f in sorted(glob.glob(os.path.join(models_dir, "*.gguf"))):
            if f not in gguf_candidates:
                gguf_candidates.append(f)

    if _active_id:
        logger.info(f"默认模型按设备画像选择: {_active_id}")

    # ★ 2026-10-07：`QLH_PREFER_PYTORCH=1` 且画像模型带 safetensors 目录时，
    #   跳过下面的 GGUF 分支，让本机走 PyTorch（`engine=pytorch`）。
    #   动机见 `config.PREFER_PYTORCH` 的说明：qwen3.5 的 keep-head 上游挂点在
    #   `output_norm` 之后，llama.cpp 上游被 fail-closed 拒绝（#35），
    #   而 GGUF 分支的优先级在 PyTorch 之前 ⇒ 必须显式让路。
    prefer_pytorch = should_prefer_pytorch(
        getattr(cfg, "PREFER_PYTORCH", False), _active_safetensors,
    )
    if prefer_pytorch:
        logger.info(
            f"QLH_PREFER_PYTORCH=1 ⇒ 本机优先走 PyTorch: {_active_safetensors}"
        )

    safetensors_path = (
        _active_safetensors
        if _active_safetensors and os.path.isdir(_active_safetensors)
        else "" if _active_id
        else cfg.MODEL_PATH if os.path.isdir(cfg.MODEL_PATH) else ""
    )
    gguf_path = gguf_candidates[0] if gguf_candidates else ""
    cuda_ok = _resolution_cuda_available()
    auto_facts = ModelLoadFacts(
        model_id=_active_id or None,
        model_name=_active_id or "默认模型",
        registered=True,
        has_safetensors=bool(safetensors_path),
        has_gguf=bool(gguf_path),
        safetensors_path=safetensors_path or None,
        gguf_path=gguf_path or None,
        preferred_engine="pytorch" if prefer_pytorch else preferred_engine_for_artifacts(
            has_safetensors=bool(safetensors_path),
            has_gguf=bool(gguf_path),
            cuda_available=cuda_ok,
        ),
        cuda_available=cuda_ok,
    )
    try:
        resolution = resolve_model_load(
            auto_facts,
            requested_engine="pytorch" if prefer_pytorch else "auto",
            requested_quant=getattr(cfg, "QUANT_TYPE", "int4"),
        )
    except ModelLoadResolutionError as exc:
        raise FileNotFoundError(
            f"未找到可自动加载的模型文件（{exc.code}: {exc.message}）。已检查:\n"
            f"  GGUF 配置路径: {gguf_configured}\n"
            f"  Safetensors 路径: {cfg.MODEL_PATH}\n"
            f"  models 目录: {models_dir}"
        ) from exc

    engine = resolution.engine
    model_path = resolution.model_path
    quant = resolution.quant_type

    logger.info(f"自动加载默认模型: path={model_path}, engine={engine}")

    t0 = time.time()
    cfg.INFERENCE_ENGINE = engine
    cfg.QUANT_TYPE = quant
    cfg.USE_COMPILE = False

    _run_exclusive_model_change(
        lambda: model_manager.load_model(
            model_path=model_path,
            quant_type=quant,
            profile=device_profile,
            model_id=resolution.model_id,
            engine=engine,
            resolution=resolution,
        )
    )

    _init_kv_cache()
    conversation_stats = {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }
    model_host.model_loaded = True
    model_host.current_quant = getattr(model_manager, "quant_type", None) or quant
    scheduler.refresh_task_worker_capabilities()
    elapsed = time.time() - t0
    logger.info(f"默认模型自动加载完成 ({elapsed:.1f}s)")




# ================================================================
# SSE 流式输出
# ================================================================



def _public_task_journal(status: dict) -> dict:
    return {key: value for key, value in status.items() if key != "path"}


def _workflow_safe_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _workflow_safe_timestamp(value: Any) -> float:
    try:
        timestamp = float(value or 0.0)
        return timestamp if timestamp == timestamp and 0 <= timestamp <= 1e15 else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _workflow_safe_duration(value: Any) -> float:
    try:
        duration = float(value or 0.0)
        return duration if duration == duration and 0 <= duration <= 1e12 else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _workflow_safe_code(value: Any) -> str:
    code = str(value or "").strip()
    return code[:64] if code and all(
        char.isalnum() or char in "_.:-" for char in code
    ) else ""


def _workflow_observability(snapshot: dict) -> dict:
    stages = snapshot.get("stages", [])
    if not isinstance(stages, list):
        stages = []
    attempts = [
        attempt
        for stage in stages
        if isinstance(stage, dict)
        for attempt in stage.get("attempts", [])
        if isinstance(attempt, dict)
    ]
    retry_count = _workflow_safe_count(snapshot.get("retry_count", 0))
    if not retry_count:
        retry_count = sum(
            _workflow_safe_count(stage.get("retry_count", 0))
            for stage in stages if isinstance(stage, dict)
        )
    same_provider_retry_count = _workflow_safe_count(
        snapshot.get("same_provider_retry_count", 0)
    )
    if not same_provider_retry_count:
        same_provider_retry_count = sum(
            _workflow_safe_count(stage.get("same_provider_retry_count", 0))
            for stage in stages if isinstance(stage, dict)
        )
    rejection_count = _workflow_safe_count(
        snapshot.get("result_rejection_count", 0)
    )
    if not rejection_count:
        rejection_count = sum(
            _workflow_safe_count(stage.get("result_rejection_count", 0))
            for stage in stages if isinstance(stage, dict)
        )
    rejected_stages = sorted(
        (
            stage for stage in stages
            if isinstance(stage, dict)
            and stage.get("last_result_rejection_reason")
        ),
        key=lambda stage: _workflow_safe_timestamp(
            stage.get("last_result_rejected_at", 0.0)
        ),
    )
    last_rejection = rejected_stages[-1] if rejected_stages else {}
    actual_providers = sorted({
        str(attempt.get("provider", "") or "")
        for attempt in attempts if attempt.get("provider")
    })
    actual_nodes = sorted({
        str(attempt.get("provider_node_id", "") or "")
        for attempt in attempts if attempt.get("provider_node_id")
    })
    state = str(snapshot.get("state", "unknown") or "unknown")
    recovered_after_restart = bool(
        snapshot.get("recovered_after_restart", False)
    )
    recovery_reason = str((
        snapshot.get("recovery_reason", "")
        or snapshot.get("error_code", "")
    ) if recovered_after_restart else "")
    return {
        "state": state,
        "result_ready": state == "result_ready",
        "terminal": state in {"completed", "failed", "cancelled"},
        "partial_result": bool(snapshot.get("partial_result", False)),
        "recovered_after_restart": recovered_after_restart,
        "recovery_reason": recovery_reason,
        "retry_count": retry_count,
        "same_provider_retry_count": same_provider_retry_count,
        "reassignment_count": max(
            0, retry_count - same_provider_retry_count,
        ),
        "retrying": retry_count > 0 and state in {"running", "created"},
        "result_rejection_count": rejection_count,
        "last_result_rejection_reason": str(
            last_rejection.get("last_result_rejection_reason", "") or ""
        ),
        "last_result_rejected_at": last_rejection.get(
            "last_result_rejected_at"
        ),
        "winner_count": sum(
            bool(stage.get("winner_attempt_id"))
            for stage in stages if isinstance(stage, dict)
        ),
        "actual_providers": actual_providers,
        "actual_nodes": actual_nodes,
    }


def _public_workflow(snapshot: dict, journal: dict) -> dict:
    public = dict(snapshot)
    public["observability"] = _workflow_observability(public)
    public["journal"] = _public_task_journal(journal)
    return public


def _public_workflow_summary(snapshot: dict, journal: dict) -> dict:
    """Return the bounded, content-free projection used by mobile audit views.

    The normal workflow endpoint intentionally remains detailed for the desktop
    control plane.  Mobile clients must never receive prompts, input bindings,
    raw provider errors, model identities, lease material, paths, or output
    metadata merely to render activity status.
    """
    stages = snapshot.get("stages", [])
    if not isinstance(stages, list):
        stages = []
    safe_stages = []
    for stage in stages[:8]:
        if not isinstance(stage, dict):
            continue
        attempts = stage.get("attempts", [])
        if not isinstance(attempts, list):
            attempts = []
        safe_attempts = []
        for attempt in attempts[:4]:
            if not isinstance(attempt, dict):
                continue
            safe_attempts.append({
                "attempt_id": str(attempt.get("attempt_id", "") or ""),
                "provider_kind": str(attempt.get("provider_kind", "") or ""),
                "provider_node_id": str(attempt.get("provider_node_id", "") or ""),
                "state": str(attempt.get("state", "unknown") or "unknown"),
                "error_code": _workflow_safe_code(attempt.get("error_code")),
                "started_at": _workflow_safe_timestamp(attempt.get("started_at")),
                "finished_at": _workflow_safe_timestamp(attempt.get("finished_at")),
                "duration_seconds": _workflow_safe_duration(attempt.get("duration_seconds")),
            })
        safe_stages.append({
            "stage_id": str(stage.get("stage_id", "") or ""),
            "stage_type": str(stage.get("stage_type", "") or ""),
            "state": str(stage.get("state", "unknown") or "unknown"),
            "started_at": _workflow_safe_timestamp(stage.get("started_at")),
            "finished_at": _workflow_safe_timestamp(stage.get("finished_at")),
            "duration_seconds": _workflow_safe_duration(stage.get("duration_seconds")),
            "retry_count": _workflow_safe_count(stage.get("retry_count")),
            "result_rejection_count": _workflow_safe_count(stage.get("result_rejection_count")),
            "error_code": _workflow_safe_code(stage.get("last_retry_error_code")),
            "attempt_count": len(safe_attempts),
            "attempts": safe_attempts,
        })
    observability = _workflow_observability(snapshot)
    safe_observability = {
        key: observability.get(key)
        for key in (
            "state", "result_ready", "terminal", "partial_result",
            "recovered_after_restart", "retry_count", "same_provider_retry_count",
            "reassignment_count", "retrying", "result_rejection_count",
            "winner_count", "actual_providers", "actual_nodes",
        )
    }
    return {
        "workflow_id": str(snapshot.get("workflow_id", "") or ""),
        "template": str(snapshot.get("template", "") or ""),
        "state": str(snapshot.get("state", "unknown") or "unknown"),
        "created_at": _workflow_safe_timestamp(snapshot.get("created_at")),
        "started_at": _workflow_safe_timestamp(snapshot.get("started_at")),
        "result_ready_at": _workflow_safe_timestamp(snapshot.get("result_ready_at")),
        "finished_at": _workflow_safe_timestamp(snapshot.get("finished_at")),
        "duration_seconds": _workflow_safe_duration(snapshot.get("duration_seconds")),
        "stage_count": _workflow_safe_count(snapshot.get("stage_count")),
        "completed_stage_count": _workflow_safe_count(snapshot.get("completed_stage_count")),
        "failed_stage_count": _workflow_safe_count(snapshot.get("failed_stage_count")),
        "attempt_count": _workflow_safe_count(snapshot.get("attempt_count")),
        "retry_count": _workflow_safe_count(snapshot.get("retry_count")),
        "same_provider_retry_count": _workflow_safe_count(snapshot.get("same_provider_retry_count")),
        "result_rejection_count": _workflow_safe_count(snapshot.get("result_rejection_count")),
        "cancel_requested": bool(snapshot.get("cancel_requested", False)),
        "observability": safe_observability,
        "stages": safe_stages,
        "journal": {
            key: journal.get(key)
            for key in ("available", "record_count", "retention_days")
            if key in journal
        },
    }
















# ============================================================
# P3: 多模型实验支持 API
# ============================================================

class SwitchModelRequest(BaseModel):
    model_id: str = Field(..., description="目标模型唯一标识")
    quant_type: str = Field(default="int4", description="量化精度")
    engine: str = Field(default="auto", description="推理引擎")


class RegisterModelRequest(BaseModel):
    model_id: str = Field(..., description="模型唯一标识")
    name: str = Field(..., description="显示名称")
    model_type: str = Field(default="safetensors", description="safetensors | gguf | both")
    model_path: str = Field(default="", description="safetensors 目录路径")
    gguf_path: str = Field(default="", description="GGUF 文件路径")
    recommended_vram_gb: float = Field(default=8.0, description="推荐显存 (GB)")
    max_context: int = Field(default=4096, description="最大上下文长度")
    huggingface_id: str = Field(default="", description="HuggingFace 仓库 ID")
    description: str = Field(default="", description="简短说明")


def _get_registered_experimental_models() -> list[dict]:
    """从主节点 SQLite 读取用户注册的实验模型。"""
    try:
        return _local_store.get_local_experimental_models()
    except Exception:
        pass
    return []


def _cuda_gate():
    """CUDA 门控：非 CUDA 环境拒绝请求。"""
    if not mc.is_cuda_available():
        raise HTTPException(
            status_code=403,
            detail="实验模型功能仅限 PC 独显版 (CUDA) 使用。当前环境未检测到 CUDA GPU。",
        )


def _db_entry_to_model_config(entry: dict) -> Optional[mc.ModelConfig]:
    """Convert a DB model entry into ModelConfig; invalid rows are ignored."""
    mid = entry.get("model_id", "")
    if not mid:
        return None
    try:
        return mc.ModelConfig(
            model_id=mid,
            name=entry.get("name", mid),
            model_type=entry.get("model_type", "safetensors"),
            model_path=entry.get("model_path", ""),
            gguf_path=entry.get("gguf_path", ""),
            recommended_vram_gb=float(entry.get("recommended_vram_gb", 8.0)),
            max_context=int(entry.get("max_context", 4096)),
            is_experimental=True,
            huggingface_id=entry.get("huggingface_id", ""),
            quant_types=entry.get("quant_types", ["int4"]),
            description=entry.get("description", ""),
            location="external",
        )
    except (TypeError, ValueError):
        return None


def _get_all_model_configs() -> list[mc.ModelConfig]:
    """Return builtin + DB-registered models without hiding unavailable entries."""
    models = mc.get_builtin_models()
    seen = {m.model_id for m in models}
    for entry in _get_registered_experimental_models():
        model = _db_entry_to_model_config(entry)
        if model and model.model_id not in seen:
            models.append(model)
            seen.add(model.model_id)
    return models


def _public_model_path(value: str) -> str:
    """Expose an asset-relative path, never the server's filesystem root."""
    if not value:
        return ""
    try:
        path = Path(value).expanduser().resolve(strict=False)
        root = Path(mc._APP_ROOT).resolve()
        return path.relative_to(root).as_posix()
    except (OSError, ValueError):
        return Path(str(value)).name


def _public_expected_paths(values: list[str] | tuple[str, ...]) -> list[str]:
    """Redact paths embedded in the human-readable availability hints."""
    public: list[str] = []
    for value in values:
        label, separator, raw_path = str(value).partition(": ")
        if separator:
            public.append(f"{label}: {_public_model_path(raw_path)}")
        else:
            public.append(_public_model_path(str(value)))
    return public


def _model_manifest_metadata(model: mc.ModelConfig) -> dict[str, Any]:
    """Read verified digest fields without exposing manifest paths."""
    model_dir = Path(mc.resolve_model_path(model.model_path))
    if not model_dir.is_dir():
        return {}
    for filename in (".qlh-model-asset.json", "model.manifest.json"):
        manifest_path = model_dir / filename
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("schema") != 1:
            continue
        artifact_sha256 = str(manifest.get("artifact_sha256") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
            continue
        digests: dict[str, str] = {}
        for entry in manifest.get("files", []):
            if not isinstance(entry, dict):
                continue
            digest = str(entry.get("sha256") or "").lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                digests[Path(str(entry.get("path") or "")).name] = digest
        return {
            "artifact_sha256": artifact_sha256,
            "tokenizer_digest": digests.get("tokenizer.json"),
            "chat_template_digest": digests.get("tokenizer_config.json"),
            "evidence": {
                "artifact_digest_mode": "manifest",
                "manifest_sha256": str(manifest.get("manifest_sha256") or ""),
            },
        }
    return {}


def _model_profile_payload(
    model: mc.ModelConfig,
    *,
    preferred_engine: str,
) -> dict[str, Any]:
    """Build the path-free model-fleet contract consumed by Harness."""
    metadata = mc.get_model_profile_metadata(model.model_id)
    manifest = _model_manifest_metadata(model)
    thinking = str(metadata.get("thinking", "unknown"))
    vision = str(metadata.get("vision", "unknown"))
    roles = list(metadata.get("roles", ("answer",)))
    resources = {
        "recommended_vram_gb": model.recommended_vram_gb,
        "max_context": model.max_context,
        **dict(metadata.get("resources", {})),
    }
    profile: dict[str, Any] = {
        "profile_schema": "qlh.harness.model_profile.v1",
        "model_id": model.model_id,
        "revision": str(metadata.get("revision") or f"catalog-{model.model_id}-v1"),
        "backend": preferred_engine,
        "format": model.model_type,
        "artifact_sha256": manifest.get("artifact_sha256"),
        "tokenizer_digest": manifest.get("tokenizer_digest"),
        "chat_template_digest": manifest.get("chat_template_digest"),
        "context": {
            "n_ctx": model.max_context,
            "input_budget": min(8192, model.max_context),
            "max_new_tokens": 1024 if model.recommended_vram_gb >= 8 else 512,
        },
        "generation": {"thinking": thinking},
        "adaptation": {
            "prompt_family": str(metadata.get("template") or "unknown"),
            "tool_mode": "host_router",
        },
        "roles": roles,
        "aliases": [model.model_id],
        "resources": resources,
        "capabilities": {
            "json_output": {"status": "unknown", "evidence": []},
            "tool_call_generation": {"status": "unknown", "evidence": []},
            "tool_result_reinjection": {"status": "unknown", "evidence": []},
            "multimodal": {"status": vision, "evidence": []},
            "thinking_control": {
                "status": "declared" if thinking == "declared" else "unknown",
                "evidence": [],
            },
        },
        "status": "candidate",
        "production_eligible": False,
        "evidence": {
            "source": "qlh.main_model_catalog",
            "weights_loaded": False,
            "network_used": False,
            **dict(metadata.get("evidence", {})),
            **dict(manifest.get("evidence", {})),
        },
    }
    canonical = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    profile["profile_digest"] = hashlib.sha256(canonical).hexdigest()
    return profile


def _public_registry_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Project registry rows without exposing persisted server paths."""
    payload = dict(entry)
    for field in ("model_path", "gguf_path"):
        if field in payload:
            payload[field] = _public_model_path(str(payload[field] or ""))
    if isinstance(payload.get("expected_paths"), (list, tuple)):
        payload["expected_paths"] = _public_expected_paths(payload["expected_paths"])
    return payload


def _model_api_payload(model: mc.ModelConfig) -> dict:
    """Serialize a model config with local availability and loadability metadata."""
    file_status = mc.get_model_file_status(model)
    supported_engines: list[str] = []

    if file_status["has_gguf"]:
        supported_engines.append("llama_cpp")
    if file_status["has_safetensors"]:
        supported_engines.append("pytorch")

    is_available = bool(supported_engines)
    unavailable_reason = file_status["unavailable_reason"]
    if file_status["is_available"] and not supported_engines:
        unavailable_reason = "模型文件已存在，但当前设备缺少可用推理后端。"

    # ★ 必须用 `_torch_cuda_available()`（L-tier 档不加载 torch），
    #   不能用 `_resolution_cuda_available()`（它走 `cuda_available(load=True)`，会强制拉起 torch）。
    #   列表接口在 GGUF/llama.cpp-only 环境也必须答得出来，不能"为了列模型"而加载 torch。
    #   回归: tests/test_api_cold_start.py::test_default_gguf_api_import_and_bootstrap_queries_do_not_load_torch
    preferred_engine = preferred_engine_for_artifacts(
        has_safetensors=file_status["has_safetensors"],
        has_gguf=file_status["has_gguf"],
        cuda_available=_torch_cuda_available(),
    )
    default_quant = "Q4_K_M" if preferred_engine == "llama_cpp" else "int4"
    profile = _model_profile_payload(model, preferred_engine=preferred_engine)

    return {
        "model_id": model.model_id,
        "name": model.name,
        "is_builtin": mc.get_builtin_model(model.model_id) is not None,
        "model_type": model.model_type,
        "is_experimental": model.is_experimental,
        "recommended_vram_gb": model.recommended_vram_gb,
        "max_context": model.max_context,
        "quant_types": model.quant_types,
        "description": model.description,
        "huggingface_id": model.huggingface_id,
        "location": model.location,
        "model_path": _public_model_path(model.model_path),
        "gguf_path": _public_model_path(model.gguf_path),
        "is_available": is_available,
        "unavailable_reason": unavailable_reason,
        "available_formats": file_status["available_formats"],
        "has_safetensors": file_status["has_safetensors"],
        "has_gguf": file_status["has_gguf"],
        "expected_paths": _public_expected_paths(file_status["expected_paths"]),
        "supported_engines": supported_engines,
        "preferred_engine": preferred_engine,
        "default_quant_type": default_quant,
        "profile": profile,
        "requires_cuda": bool(
            model.is_experimental
            and file_status["has_safetensors"]
            and "pytorch" not in supported_engines
        ),
    }


def _model_resolution_http_error(exc: ModelLoadResolutionError) -> HTTPException:
    status = 404 if exc.code == "MODEL_NOT_REGISTERED" else 400
    return coded_http_error(status, exc.code, exc.message)


def _resolution_cuda_available() -> bool:
    """Probe CUDA for an actual load decision, preserving injected probes."""

    system_cuda = False
    try:
        system_cuda = bool(mc.is_cuda_available())
    except Exception:
        pass
    if not system_cuda:
        system_cuda = cuda_available(load=True)
    return effective_pytorch_cuda_available(
        system_cuda_available=system_cuda,
        profile=device_profile,
    )


def _model_load_facts(
    model_id: Optional[str],
    *,
    preferred_engine: Optional[str] = None,
) -> ModelLoadFacts:
    """Collect registry/filesystem facts without making a backend decision."""

    import config as cfg

    resolved_id = model_id or mc.get_profile_default_model_id()
    model = mc.get_model_config(resolved_id, _get_registered_experimental_models())
    if model is None:
        return ModelLoadFacts(
            model_id=resolved_id,
            registered=False,
            cuda_available=_resolution_cuda_available(),
            island_enabled=bool(getattr(cfg, "ISLAND_ENABLED", False)),
            island_base_url=str(getattr(cfg, "ISLAND_BASE_URL", "") or ""),
        )

    status = mc.get_model_file_status(model)
    cuda_ok = _resolution_cuda_available()
    preferred = preferred_engine or preferred_engine_for_artifacts(
        has_safetensors=bool(status["has_safetensors"]),
        has_gguf=bool(status["has_gguf"]),
        cuda_available=cuda_ok,
    )
    return ModelLoadFacts(
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
        preferred_engine=preferred,
        cuda_available=cuda_ok,
        island_enabled=bool(getattr(cfg, "ISLAND_ENABLED", False)),
        island_base_url=str(getattr(cfg, "ISLAND_BASE_URL", "") or ""),
    )


def _resolve_model_load_plan(
    model_id: Optional[str],
    engine: Optional[str],
    quant_type: Optional[str],
    *,
    operation: str = OP_FULL_MODEL,
    preferred_engine: Optional[str] = None,
) -> ModelLoadResolution:
    try:
        return resolve_model_load(
            _model_load_facts(model_id, preferred_engine=preferred_engine),
            requested_engine=engine,
            requested_quant=quant_type,
            operation=operation,
        )
    except ModelLoadResolutionError as exc:
        raise _model_resolution_http_error(exc) from exc


def _normalize_quant_for_engine(quant_type: str, engine: str) -> str:
    """Compatibility wrapper; production routes consume one full resolution."""

    canonical = str(engine or "").lower()
    facts = ModelLoadFacts(
        model_id=None,
        has_safetensors=canonical in {"pytorch", "torch"},
        has_gguf=canonical in {"llama_cpp", "llama.cpp", "llama-cpp", "gguf"},
        safetensors_path="model-dir" if canonical in {"pytorch", "torch"} else None,
        gguf_path="model.gguf" if canonical in {"llama_cpp", "llama.cpp", "llama-cpp", "gguf"} else None,
        cuda_available=_resolution_cuda_available(),
        island_enabled=canonical == "island",
        island_base_url="configured" if canonical == "island" else "",
    )
    try:
        return resolve_model_load(
            facts,
            requested_engine=engine,
            requested_quant=quant_type,
        ).quant_type
    except ModelLoadResolutionError as exc:
        raise _model_resolution_http_error(exc) from exc


def _estimate_full_model_load_bytes(model_id: Optional[str], engine: str, quant: str) -> int:
    """估计**整模**加载所需的显存/内存字节数；拿不到信息时返回 0（表示"未知"）。

    GGUF 走 llama.cpp：内存需求≈工件文件大小（mmap 时更省，但按最坏情况估）。
    PyTorch 走 safetensors：参数字节按 dtype 估算（fp32=4 / fp16=2 / int8=1 / int4≈0.5 字节/参数）。
    """
    if not model_id:
        return 0
    try:
        model = mc.get_model_config(model_id, _get_registered_experimental_models())
    except Exception:  # pragma: no cover - 诊断/决策路径不抛
        return 0
    if model is None:
        return 0
    try:
        if str(engine).lower() in ("llama_cpp", "gguf"):
            path = mc.resolve_model_path(getattr(model, "gguf_path", "") or "")
            if path and os.path.isfile(path):
                return int(os.path.getsize(path))
            return 0
        # PyTorch：优先用 safetensors 实际文件大小（最准），否则按参数估算。
        path = mc.resolve_model_path(getattr(model, "model_path", "") or "")
        total = 0
        if path and os.path.isdir(path):
            for entry in os.scandir(path):
                if entry.is_file() and entry.name.endswith(".safetensors"):
                    total += int(entry.stat().st_size)
        if total > 0:
            return total
        params = float(getattr(model, "parameter_count", 0) or 0)
        if params <= 0:
            # 退而求其次：用推荐显存反推（recommended_vram_gb 已含 KV/激活余量）。
            rec = float(getattr(model, "recommended_vram_gb", 0) or 0)
            return int(rec * 1024 ** 3 * 0.7) if rec > 0 else 0
        per_param = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}.get(
            str(quant).lower(), 2.0
        )
        return int(params * per_param)
    except Exception:  # pragma: no cover
        return 0


def _decide_pipeline_load_mode(model_id: Optional[str], engine: str, quant: str) -> tuple:
    """★ 2026-10-09（自适应加载，用户裁定）决定分布式下的加载形态。

    返回 `(mode, reason)`，`mode ∈ {"full", "pipeline"}`：
    * `"full"` —— 可用显存/内存装得下**整模** ⇒ 直接整模加载（简单、启动快）；
    * `"pipeline"` —— 装不下整模 ⇒ 走 `prepare_pipeline_model`，**各节点只在自己的分配段物化权重**
      （这才是为边缘/小显存设计的路径；`#82` 的容量恒拒就发生在这里走错时）。

    判据只做"整模装不装得下"这一层，**不替求解器决定怎么切分** ——
    切分仍由 `pipeline_capacity` 按真实节点能力求解。
    """
    if RUN_MODE != "distributed":
        return "full", "非分布式模式"
    need = _estimate_full_model_load_bytes(model_id, engine, quant)
    if need <= 0:
        return "full", "无法估计整模需求（保持默认整模加载）"
    profile = device_profile or {}
    # ★ 2026-10-09：优先用**实时**探测，而不是注册时的画像快照 ——
    #   `device_profile` 是启动时 `profiler.to_dict()` 的结果（且顶层未必有 `ram`），
    #   实测在 CUDA 机器上拿不到可用值 ⇒ 决策退化成"未知 ⇒ 保持整模"。
    #   这里直接问运行时：CUDA ⇒ `torch.cuda.mem_get_info`；CPU ⇒ `psutil.virtual_memory`。
    available = 0.0
    source = ""
    try:
        import torch_runtime as _tr

        if _tr.cuda_available(load=False):
            import torch as _torch

            _free, _total = _torch.cuda.mem_get_info(0)
            available = float(_free) / 1024 ** 3
            source = "torch.cuda.mem_get_info"
    except Exception:  # pragma: no cover - 探测失败则退化
        available = 0.0
    if available <= 0:
        try:
            import psutil as _psutil

            available = float(_psutil.virtual_memory().available) / 1024 ** 3
            source = "psutil.virtual_memory"
        except Exception:  # pragma: no cover
            available = 0.0
    if available <= 0:
        # 最后退化到注册时的画像（字段名兼容 PC 的 `ram.available_gb` 与 `gpu.vram_free_gb`）。
        gpu_p = profile.get("gpu") or {}
        ram_p = profile.get("ram") or {}
        available = float(
            (gpu_p.get("vram_free_gb") or ram_p.get("available_gb") or 0) or 0
        )
        source = "device_profile" if available > 0 else ""
    if available <= 0:
        return "full", "可用显存/内存未知（保持默认整模加载）"
    # 留 15% 余量给 KV 缓存/激活/分配器开销。
    fits = available * 1024 ** 3 * 0.85 >= need
    if fits:
        return "full", f"整模 {need / 1024**3:.2f}GB 可装入 {source}={available:.2f}GB"
    return (
        "pipeline",
        f"整模 {need / 1024**3:.2f}GB 装不下 {source}={available:.2f}GB ⇒ 改为按段物化",
    )


def _validate_model_load_request(model_id: Optional[str], engine: str) -> None:
    """Compatibility wrapper; production routes consume one full resolution."""

    _resolve_model_load_plan(model_id, engine, "int4")


def _resolve_model_path_for_engine(model_id: Optional[str], engine: str) -> Optional[str]:
    """Compatibility wrapper; production routes consume one full resolution."""

    return _resolve_model_load_plan(model_id, engine, "int4").model_path


def _effective_engine_for_model(model_id: Optional[str], engine: str) -> str:
    """Compatibility wrapper preserving both layer-range implementation modes."""

    operation = OP_DISTRIBUTED_LOAD if RUN_MODE == "distributed" else OP_FULL_MODEL
    return _resolve_model_load_plan(model_id, engine, "int4", operation=operation).engine
















# ============================================================
# 模型一键下载（P0A）API
# ============================================================

class CreateModelDownloadRequest(BaseModel):
    preset_id: str = Field(default="", description="预设 ID；提供则按预设解析 source/model_id/量化")
    source: str = Field(default="", description="HF repo id / ModelScope 路径 / 本地目录")
    target: str = Field(default="", description="目标目录（默认 models/<name>）")
    model_id: str = Field(default="", description="注册的 model_id（默认取自 target 名）")
    engine: str = Field(default="auto", description="default_engine 覆盖（auto 用预设）")
    quant: str = Field(default="", description="量化精度")
    use_modelscope: bool = Field(default=False, description="走 ModelScope 下载")
    proxy: str = Field(default="", description="代理，覆盖环境设置")
    expected_sha256: str = Field(default="", description="严格 SHA-256 校验（可选）")
    gguf_path: str = Field(default="", description="显式 GGUF 路径")
    allow_cpu: bool = Field(default=True, description="允许 CPU 运行（safetensors 默认否）")


_download_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qlh-model-dl")














# ============================================================
# 集群管理 API
# ============================================================

























def _client_supports_forward_layers(node_type: str, capabilities: dict | None) -> bool:
    """客户端能否承担**层前向传播**（层流水线的一段）。

    ★ 2026-09-19：判据从「**平台**」改为「**能力**」。原实现是 `node_type == "pc"`，
    理由写作「Android 无 PyTorch 推理能力」—— 该判据**过宽**：**Android 不能跑 PyTorch，
    但能跑 llama.cpp/GGUF 引擎**，而 llama.cpp 现在**也能做层前向**
    （`LlamaCppEngine.forward_layers_from_hidden()` 当下游、`forward_layers_to_hidden()` 当上游）⇒
    有 GGUF 引擎的 Android 可以参与层流水线。

    判定顺序（**保守、向后兼容**）：

    1. `capabilities` 明确给出 `FORWARD_LAYERS`（或 `forward_layers=True`）⇒ **True**（自报最权威）；
    2. `capabilities["backend_id"]`（或 `"engine"`）给出已知 backend ⇒ 按公开的
       `backend_capabilities(...)` 判定；
    3. **未自报** ⇒ 退回旧行为：**仅 `node_type == "pc"`** ⇒ 未声明能力的 Android 仍被拒绝。
    """
    info = capabilities if isinstance(capabilities, dict) else {}

    reported = info.get("capabilities")
    if isinstance(reported, (list, tuple, set, frozenset)):
        if Capability.FORWARD_LAYERS in set(reported):
            return True
    elif isinstance(reported, str) and reported == Capability.FORWARD_LAYERS:
        return True

    if info.get("forward_layers") is True:
        return True

    backend_id = info.get("backend_id") or info.get("engine")
    if isinstance(backend_id, str) and backend_id:
        if backend_capabilities(backend_id).supports(Capability.FORWARD_LAYERS):
            return True

    return (node_type or "pc") == "pc"






# ============================================================
# 认证与账户（2026-09-19：**monolith 内实现**，抛弃 control-svc 反代）
# ------------------------------------------------------------
# 背景：原设计把账户/登录态/TOTP 放在独立的 control-svc（127.0.0.1:8030），
# 本进程只做反代。微服务改造叫停后该服务已不存在，反代恒 503 ⇒ 现改为在
# **本进程内**实现，并复用同一个用户级 SQLite（auth_store）。
# 差异：无独立进程/端口、无网络跳、无「control service unavailable」。
# ============================================================

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=512)
    totp_code: Optional[str] = Field(default=None, max_length=16,
                                     description="已绑定 Auth App 时必填的 6 位码")


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=8, max_length=512)
    role: Literal["admin", "operator", "viewer"] = "viewer"


class PatchUserRequest(BaseModel):
    role: Optional[Literal["admin", "operator", "viewer"]] = None
    disabled: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=512)


class TotpVerifyRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=16)


def _principal_payload(p) -> dict:
    return {"username": p.username, "role": p.role, "is_admin": bool(getattr(p, "is_admin", False))}
























class ManualRegisterRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64, description="节点标识")
    hostname: str = Field(default="", description="主机名")
    address: str = Field(default="", description="预留 IP:Port")
    network_type: str = Field(default="unknown", description="网络类型: wifi | ethernet | unknown")
    node_type: str = Field(default="pc", description="设备平台: pc | android")


class AndroidPresenceRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64, description="Android 稳定节点标识")
    hostname: str = Field(default="", description="Android 设备名")
    address: str = Field(default="", description="HTTP 客户端地址（可选，仅展示）")
    network_type: str = Field(default="unknown", description="网络类型: wifi | mobile | ethernet | vpn | other | unknown")
    device_info: dict = Field(default_factory=dict, description="Android 设备画像/运行状态")
    client_mode: str = Field(default="thin", description="客户端模式: thin | full")
    app_variant: str = Field(default="full", description="Android flavor: full | lite")
    app_version: str = Field(default="", description="App 版本")
    presence_generation: int = Field(default=0, ge=0, description="当前 presence lease 代次")
    presence_lease_id: str = Field(default="", max_length=128, description="当前 presence lease 标识")












class ResetIdentityRequest(BaseModel):
    confirm: str = Field(default="", description="输入 'reset' 确认重置")




# ============================================================
# 推理调度队列 API (Phase 3 — MLFQ 三级队列可视化与管理)
# ============================================================

class SetQueueStrategyRequest(BaseModel):
    strategy: str = Field(..., pattern="^(fifo|mlfq)$", description="调度策略: fifo | mlfq")


class ControlCertificateRequest(BaseModel):
    certificate: dict = Field(..., description="已由 quorum voter set 签发的控制证书")








class CancelTaskResponse(BaseModel):
    success: bool
    task_id: str
    message: str = ""














# ============================================================
# 分布式推理开关 API
# ============================================================

class TaskGraphConfigRequest(BaseModel):
    enabled: Optional[bool] = Field(default=None, description="是否启用任务链本地实验")
    worker_experimental_enabled: Optional[bool] = Field(
        default=None, description="是否打开 PC Full Worker 实验控制面"
    )







class DistributedInferenceRequest(BaseModel):
    enabled: bool = Field(..., description="是否启用分布式推理")




# ============================================================
# 动态模型分层 API
# ============================================================







class LayerOverrideItem(BaseModel):
    node_id: str = Field(..., description="节点标识")
    start_layer: int = Field(..., ge=0, description="起始层（含）")
    end_layer: int = Field(..., ge=1, description="结束层（不含）")


class LayerOverrideRequest(BaseModel):
    assignments: list[LayerOverrideItem] = Field(..., min_length=1, description="分层覆盖列表")






class Qwen3LocalChainBeginRequest(BaseModel):
    contract: dict


class Qwen3LocalChainExecuteRequest(BaseModel):
    input_ref: str = Field(..., min_length=1, max_length=2048)
    batch_size: int = Field(..., ge=1, le=64)
    sequence_length: int = Field(..., ge=1, le=1048576)


class Qwen3LocalChainParityRequest(BaseModel):
    reference_prefill: str = Field(..., min_length=1, max_length=2048)
    reference_decode: str = Field(..., min_length=1, max_length=2048)
    rtol: float = Field(default=1e-4, ge=0.0, le=1.0)
    atol: float = Field(default=1e-5, ge=0.0, le=1.0)


class ModelRuntimeSidecarBeginRequest(BaseModel):
    profile: Literal["qwen3_sidecar", "gemma4_pipeline"]
    contract: Optional[dict] = None
    contract_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ModelRuntimeSidecarActionRequest(BaseModel):
    profile: Literal["qwen3_sidecar", "gemma4_pipeline"]


class ModelRuntimeContractBindRequest(BaseModel):
    profile: Literal["qwen3_sidecar", "gemma4_pipeline"]
    model_id: str = Field(..., min_length=1, max_length=128)


def _require_qwen3_local_master():
    if scheduler._effective_role() != "master":
        raise HTTPException(403, "Qwen3 local chain is available only on the master node")
    return scheduler


def _raise_qwen3_local_http(exc: Exception) -> None:
    code = str(getattr(exc, "reason_code", "qwen3_local_chain_rejected"))
    message = str(getattr(exc, "reason", str(exc)))[:2048]
    status = 403 if "master" in code.lower() or "master" in message.lower() else 409 if any(token in code or token in message.lower() for token in (
        "phase", "stale", "active", "fenced", "duplicate",
    )) else 400
    raise HTTPException(status, {"code": code, "message": message}) from exc




























# ============================================================
# 角色转让 API
# ============================================================

class TransferMasterRequest(BaseModel):
    target_node_id: str = Field(..., min_length=1, max_length=64,
                                 description="目标从节点 ID（将升级为新主节点）")






# ============================================================
# 备用主节点管理 API
# ============================================================

class SpareMasterRequest(BaseModel):
    target_node_id: str










# ============================================================
# P3: 主节点转让审查 API
# ============================================================

class CreateReviewRequest(BaseModel):
    target_node_id: str = Field(..., description="拟转让的目标从节点 ID")
    reason: str = Field(default="", description="转让原因")
    timeout_hours: float = Field(default=48.0, description="超时时间（小时）")


class CastVoteRequest(BaseModel):
    ticket_id: str = Field(..., description="工单 ID")
    vote: int = Field(..., description="-1（阻止）、0（弃权）、+1（赞同）")
    comment: str = Field(default="", description="投票附言")


















# ============================================================
# 用户偏好设置 API（本地 SQLite 为事实源）
# ============================================================



class UserSettingsRequest(BaseModel):
    settings: dict = Field(default={}, description="完整的用户设置 JSON")




# ============================================================
# 对话持久化状态 API
# ============================================================



# ============================================================
# 对话历史 API（数据库持久化）
# ============================================================





# ============================================================
# 会话管理 API（多会话支持）
# ============================================================

class CreateSessionRequest(BaseModel):
    title: str = Field(default="新对话", description="会话标题")
    first_message: Optional[str] = Field(default=None, description="可选的首条消息用于自动生成标题")


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256, description="新标题")
















# ============================================================
# 数据库健康检查
# ============================================================





# ================================================================
# 模型文件下载（供 Android 等远程节点下载 GGUF 模型）
# ================================================================

# GGUF 模型存放目录
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
if not os.path.isdir(_MODELS_DIR):
    _MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _require_trusted_model_peer(request: Request) -> None:
    from bootstrap import is_trusted_bootstrap_source

    peer_host = request.client.host if request.client else ""
    if not is_trusted_bootstrap_source(peer_host):
        raise HTTPException(403, "source network is not trusted")


def _active_pytorch_model() -> dict:
    info = scheduler._get_active_pipeline_model_info()
    if not info:
        raise HTTPException(409, "主节点当前未加载或准备可分层的 PyTorch 模型")
    return info


def _model_file_sha256(path: str) -> str:
    from model_sync import compute_file_sha256

    return compute_file_sha256(path)












# ============================================================
# 日志管理 API
# ============================================================


_LOG_FILE_RE = re.compile(r"^[^/\\]+\.log(?:\.\d+)?$")
_LOG_FILE_LOCK = threading.RLock()
_LOCAL_LOG_CLIENTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost", "testclient"}


def _get_request_client(request: Request) -> str:
    client = getattr(request, "client", None)
    host = getattr(client, "host", "") if client else ""
    return host or "unknown"


def _get_effective_role_safe() -> str:
    try:
        return scheduler._effective_role()
    except Exception:
        return "unknown"


def _require_log_api_access(request: Request) -> str:
    """
    L0 安全边界：日志可能包含隐私与调试细节，默认只允许本机访问。

    远程管理员访问需要显式配置 QLH_LOG_ADMIN_TOKEN，并在请求头中传入
    X-QLH-Log-Token。当前项目还没有 Web 登录态，所以不能仅凭 master
    进程角色放行 LAN 浏览器。
    """
    client_host = _get_request_client(request)
    if client_host in _LOCAL_LOG_CLIENTS:
        return client_host

    admin_token = os.environ.get("QLH_LOG_ADMIN_TOKEN", "").strip()
    request_token = request.headers.get("X-QLH-Log-Token", "").strip()
    if admin_token and request_token and request_token == admin_token:
        return f"{client_host}:admin-token"

    role = _get_effective_role_safe()
    logger.warning(
        "拒绝日志接口访问: client=%s role=%s path=%s",
        client_host, role, request.url.path,
    )
    raise HTTPException(403, "日志接口仅允许本机访问；远程访问需管理员授权")


def _log_admin_action(action: str, requester: str, target: str, status: str,
                      error: str = "") -> None:
    role = _get_effective_role_safe()
    if error:
        logger.warning(
            "日志管理操作: action=%s requester=%s role=%s target=%s status=%s error=%s",
            action, requester, role, target, status, error,
        )
    else:
        logger.info(
            "日志管理操作: action=%s requester=%s role=%s target=%s status=%s",
            action, requester, role, target, status,
        )


def _snapshot_recent_logs() -> tuple[list[dict], int]:
    with _log_buffer_lock:
        return [dict(item) for item in _log_buffer], _log_buffer_total_seen


def _normalize_log_limit(limit: int) -> int:
    return max(1, min(int(limit or 200), 1000))


def _filter_recent_logs(entries: list[dict], level: str = "", name: str = "",
                        node_id: str = "", request_id: str = "") -> list[dict]:
    level = (level or "").strip().upper()
    name = (name or "").strip()
    node_id = (node_id or "").strip()
    request_id = (request_id or "").strip()

    if level:
        levelno = logging._nameToLevel.get(level)
        if isinstance(levelno, int):
            entries = [item for item in entries if item.get("levelno", 0) >= levelno]
        else:
            entries = [item for item in entries if item.get("level", "").upper() == level]
    if name:
        entries = [item for item in entries if name in item.get("name", "")]
    if node_id:
        entries = [item for item in entries if item.get("node_id") == node_id]
    if request_id:
        entries = [item for item in entries if item.get("request_id") == request_id]
    return entries


def _is_log_filename(filename: str) -> bool:
    """允许普通 .log 和 RotatingFileHandler 生成的 .log.N 备份。"""
    return (
        filename == os.path.basename(filename)
        and ".." not in filename
        and _LOG_FILE_RE.fullmatch(filename) is not None
    )


def _validate_log_filename(filename: str) -> str:
    """只允许访问 LOG_DIR 下的单个日志文件。"""
    if not _is_log_filename(filename):
        raise HTTPException(400, "无效的日志文件名")
    return filename










# ★ 通配路由 /api/logs/{filename:path} 移到最后，避免抢占 /api/logs/export、/api/logs/node/* 等特定路由




# ============================================================
# L5: 日志压缩包导出
# ============================================================



# ============================================================
# L5: 前端错误上报
# ============================================================

class ClientErrorReport(BaseModel):
    message: str = ""
    source: str = ""        # 错误来源: "window.onerror" | "unhandledrejection" | "manual"
    stack: str = ""          # 堆栈跟踪
    url: str = ""            # 发生错误的页面 URL
    line: int = 0            # 行号
    col: int = 0             # 列号
    user_agent: str = ""     # 浏览器 UA
    extra: dict = Field(default_factory=dict)  # 附加上下文（如 session_id、当前操作）


def _truncate_log_field(value, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"




# ============================================================
# L5: 日志保留策略清理
# ============================================================

_log_retention_thread_started = False


def _run_log_retention_cleanup() -> None:
    """
    按 config.LOG_MAX_AGE_DAYS 和 LOG_MAX_TOTAL_SIZE_MB 清理旧日志。

    策略：
    1. 先按天数删除过期文件（mtime < now - LOG_MAX_AGE_DAYS）
    2. 再按总空间删除最旧文件（total > LOG_MAX_TOTAL_SIZE_MB）
    3. 不会删除当天日志文件
    """
    from config import LOG_DIR, LOG_MAX_AGE_DAYS, LOG_MAX_TOTAL_SIZE_MB
    from datetime import datetime, timedelta

    if LOG_MAX_AGE_DAYS <= 0 and LOG_MAX_TOTAL_SIZE_MB <= 0:
        return

    with _LOG_FILE_LOCK:
        if not os.path.isdir(LOG_DIR):
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now()
        cutoff_time = now - timedelta(days=LOG_MAX_AGE_DAYS) if LOG_MAX_AGE_DAYS > 0 else None
        max_bytes = LOG_MAX_TOTAL_SIZE_MB * 1024 * 1024 if LOG_MAX_TOTAL_SIZE_MB > 0 else 0

        # 收集所有日志文件信息
        files_info = []
        for fname in os.listdir(LOG_DIR):
            if not _is_log_filename(fname):
                continue
            fpath = os.path.join(LOG_DIR, fname)
            try:
                st = os.stat(fpath)
                files_info.append({
                    "name": fname,
                    "path": fpath,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                })
            except OSError:
                continue

        deleted_age = 0
        deleted_size = 0
        kept = []

        # 第一阶段：按天数清理
        for fi in files_info:
            mtime_dt = datetime.fromtimestamp(fi["mtime"])
            # 不删除当天日志
            if fi["name"].startswith(f"qlh-{today_str}"):
                kept.append(fi)
                continue
            if cutoff_time and mtime_dt < cutoff_time:
                try:
                    os.remove(fi["path"])
                    deleted_age += 1
                except OSError:
                    kept.append(fi)
            else:
                kept.append(fi)

        total_size = sum(f["size"] for f in kept)

        # 第二阶段：按总空间清理（最旧优先，但跳过当天日志）
        if max_bytes > 0:
            if total_size > max_bytes:
                # 按修改时间升序（最旧的在前）
                kept.sort(key=lambda f: f["mtime"])
                for fi in kept:
                    if total_size <= max_bytes:
                        break
                    if fi["name"].startswith(f"qlh-{today_str}"):
                        continue
                    try:
                        os.remove(fi["path"])
                        total_size -= fi["size"]
                        deleted_size += 1
                    except OSError:
                        pass

        if deleted_age > 0 or deleted_size > 0:
            remaining = len(kept) - deleted_size
            remaining_bytes = total_size if max_bytes > 0 else sum(f["size"] for f in kept)
            logger.info(
                "event=log_retention_cleanup deleted_age=%d deleted_size=%d "
                "remaining_files=%d total_size_mb=%.1f",
                deleted_age, deleted_size,
                remaining,
                remaining_bytes / (1024 * 1024),
            )


def _start_log_retention_thread() -> None:
    """启动日志保留策略后台线程（仅启动一次）。"""
    global _log_retention_thread_started
    if _log_retention_thread_started:
        return

    from config import (
        LOG_RETENTION_CHECK_INTERVAL, LOG_MAX_AGE_DAYS, LOG_MAX_TOTAL_SIZE_MB,
    )

    # 双维度禁用时跳过启动（无清理任务可执行）
    if LOG_MAX_AGE_DAYS <= 0 and LOG_MAX_TOTAL_SIZE_MB <= 0:
        return

    _log_retention_thread_started = True

    def _retention_loop() -> None:
        # 启动后等待 5 分钟再首次清理（避免干扰初始化）
        time.sleep(300)
        while True:
            try:
                _run_log_retention_cleanup()
            except Exception:
                logger.warning("日志保留清理异常", exc_info=True)
            time.sleep(LOG_RETENTION_CHECK_INTERVAL)

    t = threading.Thread(target=_retention_loop, daemon=True, name="log-retention")
    t.start()
    logger.info(
        "日志保留策略已启动: max_age_days=%d max_total_mb=%d interval_s=%d",
        LOG_MAX_AGE_DAYS, LOG_MAX_TOTAL_SIZE_MB, LOG_RETENTION_CHECK_INTERVAL,
    )


# ============================================================
# L5: 多节点日志聚合 API
# ============================================================







# ★ 通配路由必须放在所有特定 /api/logs/* 路由之后，避免抢占






# The core API intentionally does not mount a product shell.  Web/Android
# clients live in sibling repositories and consume the versioned API/contracts.

# ============================================================
# 启动入口
# ============================================================

from api import routes_device as _device_routes
from api import routes_health as _health_routes
from api import routes_logs as _logs_routes
from api import routes_cluster as _cluster_routes
from api import routes_models as _models_routes
from api import routes_auth as _auth_routes
from api import routes_sessions as _sessions_routes
from api import routes_tasks as _tasks_routes
from api import routes_chat as _chat_routes
from api import routes_system as _system_routes

_api_route_modules = (
    _health_routes, _device_routes, _cluster_routes, _models_routes,
    _auth_routes, _sessions_routes, _tasks_routes, _chat_routes,
    _system_routes, _logs_routes,
)
for _route_module in _api_route_modules:
    _route_module.configure_api_module(sys.modules[__name__])
    _route_module.register_routes()
    app.include_router(_route_module.router)
    globals().update(_route_module.exported_handlers())


# API composition root: construct the scheduler's complete callback contract
# in one step. Scheduler never imports this module or reads host private attrs.
# ============================================================
_scheduler_callbacks = SchedulerCallbackSet(
    active_task_graph_model_identity=_active_task_graph_model_identity,
    execute_task_worker_stage=_execute_task_worker_stage,
    build_model_chat_prompt=_build_model_chat_prompt,
    thinking_system_prompt=THINKING_SYSTEM_PROMPT,
    snapshot_recent_logs=_snapshot_recent_logs,
    filter_recent_logs=_filter_recent_logs,
    format_model_response=_format_model_response,
)
model_host.configure_scheduler_callbacks(_scheduler_callbacks)
scheduler.configure_callbacks(_scheduler_callbacks)


def _api_bind_hosts(host: str) -> list[str]:
    """API 监听地址展开：通配 → 双栈（0.0.0.0 + [::]）。"""
    return ["0.0.0.0", "::"] if (host or "").strip() in ("", "0.0.0.0", "::") else [(host or "").strip()]


def run_api_servers(host: str = "0.0.0.0", port: int = None) -> None:
    """
    启动 API 服务器（IPv4/IPv6 双栈）。

    通配地址同时监听 0.0.0.0 与 [::]；IPv6 socket 显式 IPV6_V6ONLY=1，
    保证 v4 流量仍从 v4 socket 进入（request.client.host 保持纯 v4，
    不影响 is_trusted_bootstrap_source 等基于地址的信任判断），
    也避免 Linux 上 v6 socket 与 v4 socket 端口冲突。
    所有实例共享同一 app；任一实例退出则整体退出。
    """
    import uvicorn
    from network_address import create_listen_sockets

    port = port or API_PORT
    hosts = _api_bind_hosts(host)

    sockets = create_listen_sockets(
        hosts,
        port,
        backlog=2048,
        allow_partial=len(hosts) > 1,
    )
    actual_port = int(sockets[0].getsockname()[1])

    servers = []
    for sock in sockets:
        bind_host = str(sock.getsockname()[0])
        config = uvicorn.Config(
            app,
            host=bind_host,
            port=actual_port,
            log_level="info",
            log_config=None,
            timeout_graceful_shutdown=10,
        )
        server = uvicorn.Server(config)
        register_uvicorn_server(server)
        servers.append(server)

    try:
        if len(servers) == 1:
            servers[0].run(sockets=[sockets[0]])
            return

        threads = [
            threading.Thread(
                target=server.run,
                kwargs={"sockets": [sock]},
                daemon=True,
            )
            for server, sock in zip(servers, sockets)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass


if __name__ == "__main__":
    # 启动前检查模型文件（静默模式，仅日志提示）
    from model_downloader import ensure_model_or_warn
    ensure_model_or_warn()

    # 自建 uvicorn.Server 并注册，使 POST /api/system/shutdown 能触发
    # 跨平台优雅关闭（should_exit → lifespan shutdown → _shutdown_resources）。
    # 支持 start_tui 脚本的 QLH_BACKEND_HOST / QLH_BACKEND_PORT 覆盖；
    # 默认双栈监听（IPv4 + IPv6）。
    logger.info("启动 API 服务器...")
    _backend_host = os.environ.get("QLH_BACKEND_HOST", "0.0.0.0")
    _backend_port = int(os.environ.get("QLH_BACKEND_PORT") or API_PORT)
    run_api_servers(_backend_host, _backend_port)
