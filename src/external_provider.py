"""
外部推理服务 Provider — OpenAI 兼容外部端点的整请求路由（路线 B PoC）
====================================================================
职责:
1. 把一个集群信任域之外的 OpenAI 兼容端点（租用 GPU 盒子上的 vLLM/SGLang、
   实验室服务器、云 API）接入 QLH：整条聊天请求作为**路由决策**交给外部端点，
   不要求对方运行任何 QLH 代码，也无专职网关节点
2. 数据作用域门控（安全边界）：默认 opt_in——只有显式携带 allow_external
   的请求可以离开集群；作用域检查内嵌在外部调用的最后一道关口，
   后续重构无法绕过
3. best-effort 取消：OpenAI 端点无服务端取消原语，取消 = chunk 边界断流 +
   关闭连接，外部端可能继续算完（与调研方案 §2.2 的 cancel 缺口结论一致）
4. ExternalOpenAIProvider 实现 execution_provider.ExecutionProvider 契约，
   供任务链系统显式选择（PoC 不自动注册进聊天任务链，见接入指南）

与路线 A（TP 孤岛）的关系:
  - 传输层**组合复用** island_engine.IslandEngine（OpenAI 兼容 HTTP 客户端，
    含凭据脱敏 / URL 内嵌账号 → BasicAuth / SSE 解析 / chunk 边界取消）。
    不抽公共基类：孤岛引擎语义与 17 个测试保持逐字节不变。
  - 错误在本模块边界重新分类为 External* 系列（中文文案面向"外部推理服务"），
    孤岛错误文案不外泄到路线 B 的调用方。

配置（config.py / 环境变量，QLH_EXTERNAL_*，与 QLH_ISLAND_* 同风格）:
  QLH_EXTERNAL_ENABLED / BASE_URL / API_KEY / MODEL / TIMEOUT / CONNECT_TIMEOUT
  QLH_EXTERNAL_DATA_SCOPE        deny | opt_in(默认) | allow_all
  QLH_EXTERNAL_MIN_PROMPT_CHARS  长上下文卸载阈值（0=关）
  QLH_EXTERNAL_LABEL             展示名
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from island_engine import (
    IslandEngine,
    IslandEngineError,
    IslandHTTPError,
    IslandStreamInterruptedError,
    IslandTimeoutError,
    IslandUnreachableError,
    mask_island_url,
)
from execution_provider import (
    LocalFullModelProvider,
    ModelIdentity,
    ProviderExecutionError,
    StageRequest,
)

logger = logging.getLogger(__name__)

# 数据作用域档位（安全边界，见调研方案 §2.2"数据作用域硬约束"）
DATA_SCOPE_DENY = "deny"
DATA_SCOPE_OPT_IN = "opt_in"
DATA_SCOPE_ALLOW_ALL = "allow_all"

# 端点脱敏与孤岛同一实现：去内嵌账号密码 + 查询串
mask_external_url = mask_island_url

_HTTP_STATUS_PATTERN = re.compile(r"状态码 (\d{3})")


def _configured_reasoning_effort() -> str:
    import config as _cfg

    return str(getattr(_cfg, "EXTERNAL_REASONING_EFFORT", "") or "")


# ================================================================
# 错误分类 — 与孤岛错误一一对应，但文案面向"外部推理服务"
# ================================================================

class ExternalServiceError(RuntimeError):
    """外部推理服务统一错误基类。"""


class ExternalScopeDeniedError(ExternalServiceError):
    """数据作用域拒绝：请求未获得离开集群的授权（消息内容未发送）。"""


class ExternalUnreachableError(ExternalServiceError):
    """外部端点不可达。"""


class ExternalTimeoutError(ExternalServiceError):
    """外部端点响应超时。"""


class ExternalHTTPError(ExternalServiceError):
    """外部端点返回 HTTP 错误状态码。"""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = int(status_code or 0)


class ExternalStreamInterruptedError(ExternalServiceError):
    """外部端点流式响应中断。"""


def _map_transport_error(exc: Exception, masked: str) -> ExternalServiceError:
    """把孤岛传输层错误映射为外部服务错误（文案不提"孤岛"，不泄露凭据）。"""
    if isinstance(exc, IslandTimeoutError):
        return ExternalTimeoutError(
            f"外部推理服务超时：{masked} 未在限定时间内响应。"
            f"请确认外部端点负载正常，或调大 QLH_EXTERNAL_TIMEOUT。"
        )
    if isinstance(exc, IslandStreamInterruptedError):
        return ExternalStreamInterruptedError(
            f"外部推理服务流式响应中断：与 {masked} 的连接在传输中途断开"
            f"或未正常结束。"
        )
    if isinstance(exc, IslandUnreachableError):
        return ExternalUnreachableError(
            f"外部推理服务不可达：无法连接 {masked}。"
            f"请确认外部端点已启动且网络可达（QLH_EXTERNAL_BASE_URL）。"
        )
    if isinstance(exc, IslandHTTPError):
        match = _HTTP_STATUS_PATTERN.search(str(exc))
        status = int(match.group(1)) if match else 0
        return ExternalHTTPError(
            f"外部推理服务 HTTP 错误：{masked} 返回状态码 "
            f"{status if status else '未知'}。",
            status_code=status,
        )
    return ExternalServiceError(
        f"外部推理服务错误：{masked}（{type(exc).__name__}）。"
    )


def _error_retryable(exc: Exception) -> bool:
    """按调研方案 §2.2 错误映射：429/5xx/超时/断连可重试，其余 4xx 不可。"""
    if isinstance(exc, (
        ExternalUnreachableError,
        ExternalTimeoutError,
        ExternalStreamInterruptedError,
    )):
        return True
    if isinstance(exc, ExternalHTTPError):
        return exc.status_code == 429 or exc.status_code >= 500 or exc.status_code == 0
    return False


# ================================================================
# 数据作用域门控 — 外部调用前的最后一道关口（不可绕过）
# ================================================================

def ensure_external_scope_allowed(
    allow_external: bool,
    data_scope: Optional[str] = None,
) -> None:
    """
    数据作用域检查：在**每次**外部调用发出前强制执行。

    此函数被 ExternalChatClient.chat / chat_stream 与
    ExternalOpenAIProvider 的执行器内联调用——即路由层之外的第二道防线，
    未来任何重构都无法在不删除本调用的情况下把未授权请求发到外部端点。

    Raises:
        ExternalScopeDeniedError: 作用域不放行（消息内容未发送）
    """
    if data_scope is None:
        import config as _cfg
        data_scope = str(getattr(_cfg, "EXTERNAL_DATA_SCOPE", DATA_SCOPE_OPT_IN))
    scope = (data_scope or "").strip().lower()
    if scope == DATA_SCOPE_ALLOW_ALL:
        return
    if scope == DATA_SCOPE_OPT_IN and allow_external:
        return
    # deny、opt_in 未带 flag、以及一切未知档位一律拒绝（默认 DENY 姿态）
    raise ExternalScopeDeniedError(
        f"数据作用域禁止外部路由：QLH_EXTERNAL_DATA_SCOPE={scope or 'opt_in'}，"
        f"本请求未获得离开集群的授权，消息内容未发送。"
    )


# ================================================================
# 路由决策 — 纯函数，可独立单测
# ================================================================

@dataclass(frozen=True)
class ExternalRouteDecision:
    """外部路由决策结果。reason 取值：
    disabled | no_base_url | scope_deny | scope_opt_in_missing_flag |
    no_trigger | prefer_external | long_prompt
    """
    use_external: bool
    eligible: bool
    reason: str


def decide_external_route(
    *,
    enabled: bool,
    base_url: str,
    data_scope: str,
    allow_external: bool,
    prefer_external: bool,
    prompt_chars: int,
    min_prompt_chars: int,
) -> ExternalRouteDecision:
    """
    外部路由决策（纯函数）:

      eligible = 已启用 且 配置了端点 且 作用域放行
                 （allow_all，或 opt_in 且请求携带 allow_external；deny 一票否决）
      use_external = eligible 且 (prefer_external 或 提示词达到长上下文阈值)

    不满足时 use_external=False，调用方回落到既有本地/流水线逻辑，行为不变。
    """
    if not enabled:
        return ExternalRouteDecision(False, False, "disabled")
    if not str(base_url or "").strip():
        return ExternalRouteDecision(False, False, "no_base_url")
    scope = (data_scope or "").strip().lower()
    if scope == DATA_SCOPE_ALLOW_ALL:
        pass
    elif scope == DATA_SCOPE_OPT_IN:
        if not allow_external:
            return ExternalRouteDecision(False, False, "scope_opt_in_missing_flag")
    else:
        # deny 与一切未知档位：硬禁用，即使携带 flag（默认 DENY 姿态）
        return ExternalRouteDecision(False, False, "scope_deny")
    if prefer_external:
        return ExternalRouteDecision(True, True, "prefer_external")
    threshold = max(0, int(min_prompt_chars or 0))
    if threshold > 0 and int(prompt_chars or 0) >= threshold:
        return ExternalRouteDecision(True, True, "long_prompt")
    return ExternalRouteDecision(False, True, "no_trigger")


# ================================================================
# 外部对话客户端 — 组合复用 IslandEngine 作为传输层
# ================================================================

class ExternalChatClient:
    """
    面向外部 OpenAI 兼容端点的对话客户端。

    传输层组合复用 IslandEngine（凭据脱敏 / BasicAuth / SSE / chunk 边界取消），
    对外只暴露 External* 错误分类；作用域门控内嵌在 chat / chat_stream 中。
    """

    def __init__(self):
        self._engine: Optional[IslandEngine] = None
        self._configured_url: str = ""
        self._lock = threading.RLock()

    # ---- 连接管理 ----

    def connect(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
    ) -> None:
        """建立与外部端点的连接（= GET /v1/models 健康检查 + 模型名解析）。"""
        import config as _cfg

        resolved_url = (
            base_url if base_url is not None
            else str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or "")
        ).rstrip("/")
        if not resolved_url:
            raise ExternalServiceError(
                "外部推理服务未配置：请设置 QLH_EXTERNAL_BASE_URL "
                "（如 https://gpu-box.example.com:8000）后重试。"
            )
        masked = mask_external_url(resolved_url)
        logger.info(f"连接外部推理服务: {masked}")
        engine = IslandEngine()
        try:
            engine.load_model(
                base_url=resolved_url,
                api_key=(
                    api_key if api_key is not None
                    else getattr(_cfg, "EXTERNAL_API_KEY", "")
                ),
                model=(
                    model if model is not None
                    else getattr(_cfg, "EXTERNAL_MODEL", "")
                ),
                timeout=(
                    timeout if timeout is not None
                    else getattr(_cfg, "EXTERNAL_TIMEOUT", 120)
                ),
                connect_timeout=(
                    connect_timeout if connect_timeout is not None
                    else getattr(_cfg, "EXTERNAL_CONNECT_TIMEOUT", 5)
                ),
            )
        except IslandEngineError as exc:
            raise _map_transport_error(exc, masked) from None
        with self._lock:
            old_engine = self._engine
            self._engine = engine
            self._configured_url = resolved_url
        if old_engine is not None:
            try:
                old_engine.unload()
            except Exception:
                pass
        logger.info(
            f"外部推理服务就绪: endpoint={masked}, model={engine.model_name}"
        )

    def ensure_connected(self) -> None:
        """按当前配置确保已连接；配置变更（BASE_URL）时自动重连。"""
        import config as _cfg

        configured = str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or "").rstrip("/")
        with self._lock:
            engine = self._engine
            if (
                engine is not None
                and engine.is_loaded
                and self._configured_url == configured
            ):
                return
        self.connect()

    def close(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
            self._configured_url = ""
        if engine is not None:
            try:
                engine.unload()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._engine is not None and self._engine.is_loaded

    @property
    def masked_base_url(self) -> str:
        with self._lock:
            if self._engine is not None:
                return self._engine.masked_base_url
        import config as _cfg
        return mask_external_url(str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or ""))

    @property
    def model_name(self) -> str:
        with self._lock:
            if self._engine is not None:
                return self._engine.model_name
        import config as _cfg
        return str(getattr(_cfg, "EXTERNAL_MODEL", "") or "")

    def _require_engine(self) -> IslandEngine:
        with self._lock:
            engine = self._engine
        if engine is None or not engine.is_loaded:
            raise ExternalServiceError(
                "外部推理服务未连接，请先调用 ensure_connected()/connect()。"
            )
        return engine

    # ---- 对话补全 ----

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        *,
        allow_external: bool,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        非流式对话补全（携带 cancel_event 时内部走流式以支持 chunk 边界取消）。

        作用域门控在此处强制执行——这是消息离开集群前的最后一行代码。
        """
        # ★ 数据作用域最后关口：任何路径都必须经过这里才能外发
        ensure_external_scope_allowed(allow_external)
        engine = self._require_engine()
        masked = engine.masked_base_url
        try:
            return engine.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=_configured_reasoning_effort(),
                _cancel_event=cancel_event,
            )
        except IslandEngineError as exc:
            raise _map_transport_error(exc, masked) from None

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        *,
        allow_external: bool,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """
        流式对话补全，逐 chunk 产出增量文本。

        取消语义（best-effort）：cancel_event 置位后在 chunk 边界停止迭代，
        生成器关闭时底层 SSE 连接随之关闭；外部端可能继续算完当前请求。
        """
        # ★ 数据作用域最后关口（生成器首次迭代时执行，仍先于任何网络请求）
        ensure_external_scope_allowed(allow_external)
        engine = self._require_engine()
        masked = engine.masked_base_url
        stream = engine.chat_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=_configured_reasoning_effort(),
        )
        try:
            for chunk in stream:
                if cancel_event is not None and cancel_event.is_set():
                    break  # chunk 边界断流；close() 触发底层连接关闭
                yield chunk
                if cancel_event is not None and cancel_event.is_set():
                    break
        except IslandEngineError as exc:
            if cancel_event is not None and cancel_event.is_set():
                return  # 取消导致的断流不作为错误上抛
            raise _map_transport_error(exc, masked) from None
        finally:
            stream.close()


# 进程级共享客户端（聊天路径使用；配置变更时 ensure_connected 自动重连）
_shared_client: Optional[ExternalChatClient] = None
_shared_client_lock = threading.Lock()


def get_external_chat_client() -> ExternalChatClient:
    global _shared_client
    with _shared_client_lock:
        if _shared_client is None:
            _shared_client = ExternalChatClient()
        return _shared_client


# ================================================================
# 可用性 / 健康检查（带 30s 缓存，供 /api/status 使用）
# ================================================================

def check_external_available() -> bool:
    """外部推理服务是否已启用且配置了端点（不发起网络请求）。"""
    try:
        import config as _cfg
        return bool(
            getattr(_cfg, "EXTERNAL_ENABLED", False)
            and getattr(_cfg, "EXTERNAL_BASE_URL", "")
        )
    except Exception:
        return False


_reachable_cache_lock = threading.Lock()
_reachable_cache: Dict[str, Any] = {
    "checked_at": 0.0,
    "reachable": None,
    "base_url": "",
    "probing": False,   # 单飞标志：同一时刻只允许一个探活在途
    # 冷缓存并发时，未抢到单飞的调用方在此事件上短暂等待探活结果，
    # 避免"端点健康却返回 None（未配置）"的误报
    "done_event": None,
}


def _split_url_credentials(url: str):
    """剥离 URL 内嵌账号密码，返回 (干净 URL, BasicAuth 元组或 None)。"""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url, None
    if "@" not in parts.netloc:
        return url, None
    userinfo, host = parts.netloc.rsplit("@", 1)
    user, _, password = userinfo.partition(":")
    clean = urlunsplit(
        (parts.scheme, host, parts.path, parts.query, parts.fragment)
    )
    return clean, (unquote(user), unquote(password))


def check_external_reachable(
    cache_seconds: float = 30.0, force: bool = False,
) -> Optional[bool]:
    """
    轻量健康检查（GET /v1/models），结果缓存 cache_seconds 秒。

    /api/status 每次调用都会走到这里，缓存保证不会对外部端点形成探活风暴。
    未启用/未配置时返回 None。
    """
    import config as _cfg

    if not check_external_available():
        return None
    base_url = str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or "").rstrip("/")
    now = time.time()
    with _reachable_cache_lock:
        if (
            not force
            and _reachable_cache["reachable"] is not None
            and _reachable_cache["base_url"] == base_url
            and now - _reachable_cache["checked_at"] < max(1.0, cache_seconds)
        ):
            return bool(_reachable_cache["reachable"])
        if not force and _reachable_cache["probing"]:
            # 已有探活在途：不再重复发起，避免 N 个并发 /api/status 形成探活风暴
            cached = _reachable_cache["reachable"]
            if cached is not None:
                return bool(cached)   # 有历史结果：直接复用（可能略陈旧）
            # 冷缓存：等在途探活出结果，否则会把健康端点误报成"未配置"(None)
            waiter = _reachable_cache["done_event"]
        else:
            waiter = None
            _reachable_cache["probing"] = True
            _reachable_cache["done_event"] = threading.Event()

    if waiter is not None:
        connect_timeout = float(getattr(_cfg, "EXTERNAL_CONNECT_TIMEOUT", 5))
        waiter.wait(timeout=max(connect_timeout, 2.0) + 1.0)
        with _reachable_cache_lock:
            cached = _reachable_cache["reachable"]
        return None if cached is None else bool(cached)

    reachable = False
    try:
        import httpx

        clean_url, basic = _split_url_credentials(base_url)
        headers = {}
        api_key = str(getattr(_cfg, "EXTERNAL_API_KEY", "") or "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        connect_timeout = float(getattr(_cfg, "EXTERNAL_CONNECT_TIMEOUT", 5))
        with httpx.Client(
            timeout=httpx.Timeout(
                max(connect_timeout, 2.0), connect=connect_timeout,
            ),
            auth=(httpx.BasicAuth(*basic) if basic is not None else None),
        ) as client:
            response = client.get(
                f"{clean_url.rstrip('/')}/v1/models", headers=headers,
            )
            reachable = response.status_code == 200
    except Exception:
        reachable = False
    finally:
        # finally 保证单飞标志一定归零、等待方一定被唤醒，
        # 否则一次异常会让探活永久停摆、并发调用方空等到超时
        with _reachable_cache_lock:
            _reachable_cache["checked_at"] = time.time()
            _reachable_cache["reachable"] = reachable
            _reachable_cache["base_url"] = base_url
            _reachable_cache["probing"] = False
            done_event = _reachable_cache["done_event"]
            _reachable_cache["done_event"] = None
        if done_event is not None:
            done_event.set()
    return reachable


def reset_reachable_cache() -> None:
    """清空健康检查缓存（测试与配置热更新用）。"""
    with _reachable_cache_lock:
        _reachable_cache["checked_at"] = 0.0
        _reachable_cache["reachable"] = None
        _reachable_cache["base_url"] = ""
        _reachable_cache["probing"] = False
        pending = _reachable_cache["done_event"]
        _reachable_cache["done_event"] = None
    if pending is not None:
        pending.set()   # 唤醒可能仍在等待的调用方，避免测试/热更新时空等


# ================================================================
# 模型身份 / 端点指纹
# ================================================================

def external_endpoint_fingerprint(
    base_url: Optional[str] = None, model: Optional[str] = None,
) -> str:
    """端点指纹 = sha256(脱敏端点 :: 服务端模型名)，入 journal 与统计。"""
    import config as _cfg

    masked = mask_external_url(
        base_url if base_url is not None
        else str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or "")
    )
    served = (
        model if model is not None
        else str(getattr(_cfg, "EXTERNAL_MODEL", "") or "")
    )
    return hashlib.sha256(f"{masked}::{served}".encode("utf-8")).hexdigest()


def external_model_identity(
    model_id: str = "external-api",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> ModelIdentity:
    """
    外部模型身份：无本地 artifact，以端点指纹替代文件摘要，
    统计中如实标注 engine="external_api"/format="openai_api"（不伪装成本地文件）。
    """
    digest = external_endpoint_fingerprint(base_url, model)
    return ModelIdentity(
        model_id=model_id,
        engine="external_api",
        format="openai_api",
        revision=f"external-{digest[:12]}",
        sha256=digest,
    )


# ================================================================
# ExecutionProvider 实现 — 供任务链系统显式选择
# ================================================================

class ExternalOpenAIProvider(LocalFullModelProvider):
    """
    外部 OpenAI 兼容端点的任务链 Provider（ExecutionProvider 契约）。

    - supported_stage_types: full_inference / aggregate（现行任务链的全部
      Stage 类型；调研方案的 "verify" 类型尚未在任务系统中定义，暂不声明）
    - 作用域门控：Stage root_input 必须携带 "allow_external": true 才可外发
      （opt_in 档位下），deny 档位一律拒绝——与聊天路径同一 ensure 函数
    - 取消：cancel(attempt_id) 置位后在 chunk 边界断流并关闭连接（best-effort）
    - 错误映射：429/5xx/超时/断连 → retryable=True；其余 4xx → retryable=False

    注意：PoC 不把本 Provider 自动注册进聊天任务链协调器——注册后
    ProviderRegistry 的"首个兼容者"按 provider_id 字典序选择，
    "external_openai" 会排在 "local_full_model" 之前，导致未 opt_in 的
    默认本地工作流产生无谓的失败 attempt。由部署者/实验代码显式注册。
    """

    def __init__(
        self,
        client: Optional[ExternalChatClient] = None,
        *,
        provider_id: str = "external_openai",
        node_id: str = "",
        max_concurrency: int = 2,
    ):
        self._client = client or ExternalChatClient()
        super().__init__(
            self._execute_stage,
            provider_id=provider_id,
            node_id=node_id,
            supported_stage_types=("full_inference", "aggregate"),
            max_concurrency=max_concurrency,
            provider_kind="external_openai",
        )

    @property
    def client(self) -> ExternalChatClient:
        return self._client

    def endpoint_fingerprint(self) -> str:
        return external_endpoint_fingerprint()

    def model_identity(self) -> ModelIdentity:
        return external_model_identity()

    # ---- Stage 执行器 ----

    def _execute_stage(
        self,
        request: StageRequest,
        cancel_event: threading.Event,
    ) -> dict:
        root_input = request.root_input or {}
        allow_external = bool(root_input.get("allow_external", False))
        options = root_input.get("task_options", {})
        if not isinstance(options, dict):
            options = {}
        try:
            max_tokens = max(1, min(int(options.get(
                "final_max_tokens"
                if request.stage_type == "aggregate"
                else "candidate_max_tokens",
                512,
            )), 2048))
            temperature = max(0.0, min(float(options.get("temperature", 0.7)), 2.0))
            top_p = max(0.0, min(float(options.get("top_p", 0.9)), 1.0))
        except (TypeError, ValueError):
            max_tokens, temperature, top_p = 512, 0.7, 0.9

        messages = self._build_stage_messages(request)
        try:
            # 作用域先行：被拒绝的 Stage 连健康检查请求都不对外发出；
            # client.chat 内部还会再执行一次同一门控（最后关口，不可绕过）
            ensure_external_scope_allowed(allow_external)
            self._client.ensure_connected()
            result = self._client.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                allow_external=allow_external,
                cancel_event=cancel_event,
            )
        except ExternalScopeDeniedError as exc:
            # 作用域拒绝：允许协调器在 attempt 边界改派其他 Provider（如本地）
            raise ProviderExecutionError(
                str(exc),
                code="external_scope_denied",
                provider_id=self.provider_id,
                retryable=True,
            ) from exc
        except ExternalServiceError as exc:
            raise ProviderExecutionError(
                str(exc),
                code="external_backend_failed",
                provider_id=self.provider_id,
                retryable=_error_retryable(exc),
            ) from exc

        return {
            "content": str(result.get("content", "") or ""),
            "usage": dict(result.get("usage", {}) or {}),
            "tokens_per_second": result.get("tokens_per_second", 0),
            "model": str(result.get("model", "") or self._client.model_name),
            "usage_estimated": bool(result.get("usage_estimated", False)),
            "finish_reason": str(result.get("finish_reason", "") or ""),
            "endpoint_fingerprint": self.endpoint_fingerprint(),
        }

    @staticmethod
    def _build_stage_messages(request: StageRequest) -> List[Dict[str, Any]]:
        """由 Stage 输入构造 OpenAI messages（full_inference / aggregate）。"""
        import json as _json

        from multimodal import normalize_openai_message_content
        from execution_provider import DEPENDENCY_FAILURES_KEY

        root_input = request.root_input or {}
        if request.stage_type == "aggregate":
            message = str(root_input.get("message", "") or "")
            candidates = {
                stage_id: value.get("content", "")
                for stage_id, value in (request.dependencies or {}).items()
                if stage_id != DEPENDENCY_FAILURES_KEY
                and isinstance(value, dict)
                and str(value.get("content", "") or "").strip()
            }
            prompt = (
                "请根据原始问题和可用的独立候选，输出一个最终答案。"
                "纠正冲突和明显错误；没有证据时明确不确定性。"
                "只输出最终答案，不描述内部任务链。\n\n"
                f"原始问题：{message}\n\n候选：\n"
                + _json.dumps(candidates, ensure_ascii=False)
            )
            return [{"role": "user", "content": prompt}]
        messages = root_input.get("messages")
        if isinstance(messages, list) and messages:
            return [
                {
                    "role": str(item.get("role", "user")),
                    "content": normalize_openai_message_content(
                        item.get("content", ""),
                    ),
                }
                for item in messages
                if isinstance(item, dict)
            ]
        message = str(root_input.get("message", "") or "")
        return [{"role": "user", "content": message}]
