"""QLH 后端的最小 REST / SSE 客户端（**纯标准库**）。

从旧的 ``tui_admin.py`` 抽出的 ``ApiClient``，用途：

* 供 Textual 外壳（``src/tui_textual.py``）与后续的单命令薄层共用一份协议实现；
* 只依赖标准库（``urllib``/``json``），SSE 解析复用 ``src/tui_sse.py``（同样零依赖），
  因此**无 UI 依赖的环境**（Edge、CI）也能 import 本模块。

历史：``ApiClient`` 原在 ``tui_admin.py`` 内，随该文件的界面代码一起被视为可归档对象；
协议层不应与界面同生共死，故独立成模块。
"""

from __future__ import annotations

import json
import uuid
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:  # 同目录导入（python src/xxx.py）
    from tui_sse import SSEDecoder, decode_json_event
    from tui_shared import API_PATHS, build_interactive_request
except ImportError:  # pragma: no cover - 包导入路径
    from .tui_sse import SSEDecoder, decode_json_event  # type: ignore
    from .tui_shared import API_PATHS, build_interactive_request  # type: ignore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
BACKEND_HINT = "请先启动本机后端（python src/api_server.py），或用 qlh 自动启动。"


class ApiError(Exception):
    """后端返回的错误（含 HTTP 状态码，便于界面区分 4xx/5xx）。"""

    def __init__(self, message: str, status: Optional[int] = None,
                 reason_code: str = "", reason_detail=None) -> None:
        super().__init__(message)
        self.status = status
        self.reason_code = reason_code
        self.reason_detail = reason_detail


class ApiClient:
    """与 FastAPI 后端通信的极简 REST 客户端（纯标准库）。"""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 5.0, log_token: str = "",
                 auth_token: str = "", client_id: str = "") -> None:
        self.client_id = client_id or "tui:" + uuid.uuid4().hex
        self._snapshot_cache = {}
        self.host = host
        self.port = port
        self.timeout = timeout
        self.log_token = log_token
        #: ★ 2026-09-19（G-⑤）：登录态 Bearer token。由 TUI 的 /login 写入，
        #   之后所有请求自动带 `Authorization: Bearer <token>`。
        self.auth_token = auth_token

    def accept_snapshot(self, value, *, stream="model_control"):
        if value is None:
            return None
        identity = value.get("identity") or {}
        revision, observed = value.get("revision"), value.get("observed_at")
        if (value.get("schema_version") != 2 or value.get("validity") != "valid"
                or not identity.get("boot_id") or type(revision) is not int
                or type(observed) not in (int, float) or not math.isfinite(observed) or observed <= 0 or revision < 0):
            raise ApiError("状态快照无效", reason_code="snapshot_invalid")
        prior = self._snapshot_cache.get(stream)
        if prior and (observed < prior[2] or (identity["boot_id"] == prior[0] and revision < prior[1])):
            raise ApiError("状态快照已过期", reason_code="snapshot_stale")
        self._snapshot_cache[stream] = (identity["boot_id"], revision, observed)
        return value

    @property
    def base_url(self) -> str:
        from network_address import build_url

        return build_url("http", self.host, self.port)

    def _request_headers(self, *, accept: str = "application/json",
                         content_type: bool = False,
                         with_log_token: bool = False) -> Dict[str, str]:
        """Build headers shared by REST, downloads, and streaming requests.

        The API source boundary is the direct TCP peer observed by the
        backend; it is intentionally not represented by a caller-controlled
        source header.  Authentication and optional log credentials are the
        only client headers that need to be propagated here.
        """
        headers = {"Accept": accept, "X-QLH-Client-ID": self.client_id}
        if content_type:
            headers["Content-Type"] = "application/json"
        if with_log_token and self.log_token:
            headers["X-QLH-Log-Token"] = self.log_token
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    # ------------------------------------------------------------ 底层请求

    def _request_url(self, method: str, url: str, body=None, params=None,
                     with_log_token: bool = False, timeout: Optional[float] = None):
        """执行一个已拼好的 URL 请求，供 API 面和根路径文档共用。"""
        if params:
            qs = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
            if qs:
                url = url + ("&" if "?" in url else "?") + qs
        data = None
        headers = self._request_headers(
            content_type=body is not None,
            with_log_token=with_log_token,
        )
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        effective_timeout = self.timeout if timeout is None else float(timeout)
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            reason_code = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw) if raw else {}
                reason_detail = parsed.get("reason_detail") if isinstance(parsed, dict) else None
                reason_code = str((reason_detail or {}).get("code") or parsed.get("error_code", "")) if isinstance(parsed, dict) else ""
                detail = parsed.get("detail", raw) if isinstance(parsed, dict) else raw
                if isinstance(detail, dict):
                    reason_code = str(detail.get("code") or detail.get("reason_code") or "")
                    detail = detail.get("message") or detail.get("reason") or json.dumps(detail, ensure_ascii=False)
            except Exception:  # noqa: BLE001 - 响应体不是 JSON 时保留状态码
                detail = ""
            label = f"{reason_code}: " if reason_code else ""
            raise ApiError("HTTP %d: %s%s" % (e.code, label, detail or e.reason),
                           status=e.code, reason_code=reason_code, reason_detail=locals().get("reason_detail"))
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, (TimeoutError, OSError)) and "timed out" in str(reason):
                raise ApiError("请求超时（>%.0fs）: %s" % (effective_timeout, url))
            raise ApiError("无法连接后端 %s（%s）。%s" % (self.base_url, reason, BACKEND_HINT))
        except TimeoutError:
            raise ApiError("请求超时（>%.0fs）: %s" % (effective_timeout, url))
        except OSError as e:
            raise ApiError("网络错误 %s: %s。%s" % (self.base_url, e, BACKEND_HINT))
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError:
            return {"detail": text}

    def request(self, method: str, path: str, body=None, params=None,
                with_log_token: bool = False, timeout: Optional[float] = None):
        """``timeout`` 为 None 时用实例超时；模型加载等长操作按需放宽。"""
        normalized = path if path.startswith("/") else "/" + path
        if normalized.startswith("/api/"):
            normalized = normalized[4:]
        return self._request_url(
            method, self.base_url + "/api" + normalized, body=body, params=params,
            with_log_token=with_log_token, timeout=timeout)

    def request_root(self, method: str, path: str, body=None, params=None,
                     timeout: Optional[float] = None):
        """访问后端根路径，主要用于 ``/openapi.json`` 等非业务 API。"""
        normalized = path if path.startswith("/") else "/" + path
        return self._request_url(
            method, self.base_url + normalized, body=body, params=params,
            timeout=timeout)

    def get_openapi(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """读取运行中后端的实际 OpenAPI 路由表。"""
        value = self.request_root("GET", "/openapi.json", timeout=timeout)
        return value if isinstance(value, dict) else {}

    def download(self, path: str, target: Path, *, with_log_token: bool = False,
                 timeout: Optional[float] = 60.0) -> Path:
        """下载二进制端点响应到本地文件（日志导出/模型文件等）。"""
        normalized = path if path.startswith("/") else "/" + path
        if normalized.startswith("/api/"):
            normalized = normalized[4:]
        url = self.base_url + "/api" + normalized
        headers = self._request_headers(
            accept="*/*", with_log_token=with_log_token,
        )
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            raise ApiError("HTTP %d: %s" % (exc.code, exc.reason), status=exc.code) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ApiError("无法下载 %s：%s" % (url, exc)) from exc
        target = Path(target)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            raise ApiError("无法保存下载文件 %s：%s" % (target, exc)) from exc
        return target

    def get(self, path, params=None, with_log_token: bool = False,
            timeout: Optional[float] = None):
        return self.request("GET", path, params=params,
                           with_log_token=with_log_token, timeout=timeout)

    def post(self, path, body=None, params=None):
        return self.request("POST", path, body=body, params=params)

    def put(self, path, body=None):
        return self.request("PUT", path, body=body)

    def delete(self, path, body=None, params=None, timeout: Optional[float] = None):
        return self.request("DELETE", path, body=body, params=params, timeout=timeout)


def cancel_generation(api: ApiClient, generation_id: str) -> None:
    """取消某次生成；``generation_id`` 含 ``/`` 时必须 safe="" 编码。"""
    path = API_PATHS["chat_cancel"].format(
        generation_id=urllib.parse.quote(generation_id, safe=""))
    api.post(path)


# ============================================================
# 写操作：模型控制 / 会话管理 / 队列控制
# ============================================================
# 权限边界：后端 ``model_api_access.require_model_api_source()`` 对 loopback 默认放行
# （``is_model_api_source_trusted`` → 127.0.0.1 直接 True），因此**本机 TUI 可直接调用**；
# 从节点远程控制主节点需要主节点显式配置 ``QLH_MODEL_API_TRUSTED_CIDRS``。

#: 模型加载/卸载耗时约 5-20 秒（后端 /models/load 注释），故放宽超时
MODEL_CONTROL_TIMEOUT = 180.0




def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}


def _quoted(**values: Any) -> Dict[str, str]:
    """路径参数必须 safe="" 编码（session_id 可能含 "/"）。"""
    return {key: urllib.parse.quote(str(value), safe="") for key, value in values.items()}


def load_model(api: ApiClient, model_id: Optional[str] = None, *, engine: str = "llama_cpp",
               quant_type: str = "int4", use_compile: bool = False) -> Dict[str, Any]:
    """加载/切换模型（POST ``/api/models/load``）。

    后端内部走 ``switch_model``，因此**失败会自动回滚**到上一个模型；耗时 5-20 秒，
    期间先卸载旧模型。注意 ``/api/models/switch`` 仅 CUDA 可用（非 CUDA 返回 403），
    故统一走本接口。
    """
    body: Dict[str, Any] = {
        "engine": engine,
        "quant_type": quant_type,
        "use_compile": bool(use_compile),
    }
    if model_id:
        body["model_id"] = model_id
    return _as_dict(api.request("POST", API_PATHS["models_load"], body=body,
                               timeout=MODEL_CONTROL_TIMEOUT))


def unload_model(api: ApiClient) -> Dict[str, Any]:
    """显式释放本地 LLM（POST ``/api/models/unload``）。"""
    return _as_dict(api.request("POST", API_PATHS["models_unload"],
                               timeout=MODEL_CONTROL_TIMEOUT))



def _successful_control(value: Dict[str, Any]) -> Dict[str, Any]:
    if value.get("success") is False or value.get("status") in {"error", "failed"}:
        raise ApiError(str(value.get("reason") or value.get("message") or "模型控制失败"),
                       reason_code=str(value.get("reason_code") or ""))
    return value


def set_model_mode(api: ApiClient, mode: str) -> Dict[str, Any]:
    if mode not in {"local", "distributed"}:
        raise ValueError("模型模式必须是 local 或 distributed")
    return _successful_control(_as_dict(api.request(
        "PUT", API_PATHS["distributed_config"], body={"enabled": mode == "distributed"},
        timeout=MODEL_CONTROL_TIMEOUT)))


def load_local_model(api: ApiClient, model_id: str, *, engine: str = "llama_cpp",
                     quant_type: str = "int4") -> Dict[str, Any]:
    """Leave cluster participation and release segments before loading a whole model."""
    set_model_mode(api, "local")
    _successful_control(unload_model(api))
    return _successful_control(load_model(api, model_id, engine=engine, quant_type=quant_type))


def create_model_intent(api: ApiClient, model_id: str, *, engine: str = "auto",
                        quant_type: str = "") -> Dict[str, Any]:
    return api.accept_snapshot(_as_dict(api.request("POST", API_PATHS["models_intent"], body={
        "model_id": model_id, "engine": engine, "quant_type": quant_type,
        "strategy": "auto", "timeout_seconds": 3600,
    }, timeout=MODEL_CONTROL_TIMEOUT)))


def get_model_intent_catalog(api: ApiClient) -> Dict[str, Any]:
    return _as_dict(api.get(API_PATHS["models_intent_catalog"]))


def current_model_intent(api: ApiClient) -> Optional[Dict[str, Any]]:
    value = api.get(API_PATHS["models_intent_current"])
    return api.accept_snapshot(value) if isinstance(value, dict) and value.get("operation_id") else None


def _model_intent_path(operation_id: str) -> str:
    if not re.fullmatch(r"supply_[0-9a-f]{32}", operation_id):
        raise ValueError("非法模型 operation_id（需要 supply_32hex）")
    return API_PATHS["models_intent_operation"].format(operation_id=operation_id)


def get_model_intent(api: ApiClient, operation_id: str) -> Dict[str, Any]:
    path = _model_intent_path(operation_id)
    return api.accept_snapshot(_as_dict(api.get(path)))


def cancel_model_intent(api: ApiClient, operation_id: str) -> Dict[str, Any]:
    path = _model_intent_path(operation_id)
    return api.accept_snapshot(_as_dict(api.delete(path, timeout=MODEL_CONTROL_TIMEOUT)))


def connect_model_intent(api: ApiClient) -> Dict[str, Any]:
    return _as_dict(api.request("POST", API_PATHS["models_intent_connect"], body={},
                               timeout=MODEL_CONTROL_TIMEOUT))


def clear_backend_history(api: ApiClient) -> Dict[str, Any]:
    """清空后端当前会话的历史与 KV 缓存（POST ``/api/chat/clear``）。"""
    return _as_dict(api.post(API_PATHS["chat_clear"]))


# ------------------------------------------------------------ 会话管理

def list_sessions(api: ApiClient) -> List[Dict[str, Any]]:
    """``GET /api/sessions`` → 会话 dict 列表（兼容裸列表与 ``{sessions: [...]}``）。"""
    value = api.get(API_PATHS["sessions"])
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = value.get("sessions") or value.get("items") or []
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def create_session(api: ApiClient, title: Optional[str] = None) -> Dict[str, Any]:
    """新建并激活会话（POST ``/api/sessions``）。"""
    body = {"title": title} if title else {}
    return _as_dict(api.post(API_PATHS["sessions"], body=body))


def activate_session(api: ApiClient, session_id: str) -> Dict[str, Any]:
    """切换到指定会话并取回历史（POST ``/sessions/{id}/activate``）。"""
    path = API_PATHS["session_activate"].format(**_quoted(session_id=session_id))
    return _as_dict(api.post(path))


def rename_session(api: ApiClient, session_id: str, title: str) -> Dict[str, Any]:
    """重命名会话（PUT ``/sessions/{id}``）。"""
    path = API_PATHS["session_detail"].format(**_quoted(session_id=session_id))
    return _as_dict(api.put(path, {"title": title}))


def delete_session(api: ApiClient, session_id: str) -> Dict[str, Any]:
    """删除会话及其全部对话消息（DELETE ``/sessions/{id}``）——破坏性，调用方须先确认。"""
    path = API_PATHS["session_detail"].format(**_quoted(session_id=session_id))
    return _as_dict(api.delete(path))


# ------------------------------------------------------------ 队列控制

def pause_queue(api: ApiClient) -> Dict[str, Any]:
    """暂停接受新请求（POST ``/cluster/queue/pause``，仅主节点）。"""
    return _as_dict(api.post(API_PATHS["cluster_queue_pause"]))


def resume_queue(api: ApiClient) -> Dict[str, Any]:
    """恢复接受新请求（POST ``/cluster/queue/resume``，仅主节点）。"""
    return _as_dict(api.post(API_PATHS["cluster_queue_resume"]))




def clear_queue(api: ApiClient) -> Dict[str, Any]:
    """清空排队任务（POST ``/cluster/queue/clear``）——不影响执行中的任务，仍须确认。"""
    return _as_dict(api.post(API_PATHS["cluster_queue_clear"]))


def cancel_queue_task(api: ApiClient, task_id: str) -> Dict[str, Any]:
    """取消**单个**排队任务（DELETE ``/cluster/queue/task/{task_id}``，仅主节点）。

    ★ 2026-09-19 补缺口 A：此前 TUI 只能整体 ``clear``，卡住的单个任务无法取消。
    执行中的流水线任务会在当前 token step 完成后经 ``PIPELINE_ABORT`` 中止。
    后端返回 ``{success, task_id, message}``；任务不存在/已完成时 ``success=False``。
    """
    path = API_PATHS["cluster_queue_task_cancel"].format(**_quoted(task_id=task_id))
    return _as_dict(api.request("DELETE", path))


def list_log_files(api: ApiClient) -> Dict[str, Any]:
    """列出后端日志文件（GET ``/logs``）。★ 补缺口 B。"""
    return _as_dict(api.get(API_PATHS["logs_list"], with_log_token=True))


def download_log_file(api: ApiClient, filename: str, target: Path) -> Path:
    """下载单个日志文件（GET ``/logs/download?filename=...``）。★ 补缺口 B。"""
    query = urllib.parse.urlencode({"filename": filename})
    return api.download(f"{API_PATHS['logs_download']}?{query}", target, with_log_token=True)


def read_log_file(api: ApiClient, filename: str) -> Dict[str, Any]:
    """读单个日志文件（GET ``/logs/{filename}``）。★ 补缺口 B。"""
    path = API_PATHS["logs_file"].format(**_quoted(filename=filename))
    return _as_dict(api.get(path, with_log_token=True))


def delete_log_file(api: ApiClient, filename: str) -> Dict[str, Any]:
    """删单个日志文件（DELETE ``/logs/{filename}``）——不可撤销，调用方须先确认。★ 补缺口 B。"""
    path = API_PATHS["logs_file"].format(**_quoted(filename=filename))
    return _as_dict(api.request("DELETE", path))


def logs_nodes_summary(api: ApiClient) -> Dict[str, Any]:
    """各节点日志汇总（GET ``/logs/nodes-summary``）。★ 补缺口 B。"""
    return _as_dict(api.get(API_PATHS["logs_nodes_summary"], with_log_token=True))

# ---------------------------------------------------------------- 集群高可用(E)
def master_health(api: ApiClient) -> Dict[str, Any]:
    """主节点健康（GET ``/cluster/master-health``）。★ 补缺口 E。"""
    return _as_dict(api.get(API_PATHS["cluster_master_health"]))


def transfer_logs(api: ApiClient) -> Dict[str, Any]:
    """角色转让日志（GET ``/cluster/transfer-logs``）。★ 补缺口 E。"""
    return _as_dict(api.get(API_PATHS["cluster_transfer_logs"]))


def get_spare_master(api: ApiClient) -> Dict[str, Any]:
    """查询备用主节点（GET ``/cluster/spare-master``）。★ 补缺口 E。"""
    return _as_dict(api.get(API_PATHS["cluster_spare_master"]))


def spare_master_logs(api: ApiClient) -> Dict[str, Any]:
    """备用主节点操作日志（GET ``/cluster/spare-master/logs``）。★ 补缺口 E。"""
    return _as_dict(api.get(API_PATHS["cluster_spare_master_logs"]))


def designate_spare_master(api: ApiClient, target_node_id: str) -> Dict[str, Any]:
    """指定备用主节点（POST ``/cluster/spare-master``，仅主节点）。

    ⚠️ 变更集群角色配置：集群节点数需 >= 2，目标须在线且为 client。★ 补缺口 E。
    """
    return _as_dict(api.post(API_PATHS["cluster_spare_master"],
                             {"target_node_id": target_node_id}))


def clear_spare_master(api: ApiClient) -> Dict[str, Any]:
    """清除备用主节点指定（DELETE ``/cluster/spare-master``，仅主节点）。★ 补缺口 E。"""
    return _as_dict(api.request("DELETE", API_PATHS["cluster_spare_master"]))


def transfer_master(api: ApiClient, target_node_id: str) -> Dict[str, Any]:
    """把主节点身份转让给指定从节点（POST ``/cluster/transfer-master``，仅主节点）。

    ⚠️⚠️ **高危**：转让后**双方需重启**才生效（原主转从、新主转主）。★ 补缺口 E。
    """
    return _as_dict(api.post(API_PATHS["cluster_transfer_master"],
                             {"target_node_id": target_node_id}))


def reset_master_identity(api: ApiClient) -> Dict[str, Any]:
    """重置主节点身份标识（POST ``/cluster/reset-identity``，仅主节点）。

    ⚠️⚠️ **高危**：替换主节点 SQLite 里的 MAC 记录（更换机器/网卡后用），绑定当前物理 MAC。
    后端要求请求体 ``confirm == "reset"``；本函数已固定填入。★ 补缺口 E。
    """
    return _as_dict(api.post(API_PATHS["cluster_reset_identity"], {"confirm": "reset"}))


def cancel_queue_task(api: ApiClient, task_id: str) -> Dict[str, Any]:
    """取消**单个**排队任务（DELETE ``/cluster/queue/task/{task_id}``，仅主节点）。

    ★ 2026-09-19 补缺口 A：此前 TUI 只能整体 ``clear``，卡住的单个任务无法取消。
    执行中的流水线任务会在当前 token step 完成后经 ``PIPELINE_ABORT`` 中止。
    后端返回 ``{success, task_id, message}``；任务不存在/已完成时 ``success=False``。
    """
    path = API_PATHS["cluster_queue_task_cancel"].format(**_quoted(task_id=task_id))
    return _as_dict(api.request("DELETE", path))


def list_log_files(api: ApiClient) -> Dict[str, Any]:
    """列出后端日志文件（GET ``/logs``）。★ 补缺口 B。"""
    return _as_dict(api.get(API_PATHS["logs_list"], with_log_token=True))


def download_log_file(api: ApiClient, filename: str, target: Path) -> Path:
    """下载单个日志文件（GET ``/logs/download?filename=...``）。★ 补缺口 B。"""
    query = urllib.parse.urlencode({"filename": filename})
    return api.download(f"{API_PATHS['logs_download']}?{query}", target, with_log_token=True)


def read_log_file(api: ApiClient, filename: str) -> Dict[str, Any]:
    """读单个日志文件（GET ``/logs/{filename}``）。★ 补缺口 B。"""
    path = API_PATHS["logs_file"].format(**_quoted(filename=filename))
    return _as_dict(api.get(path, with_log_token=True))


def iter_chat_payloads(
    api: ApiClient,
    message: str,
    *,
    session_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    routing_preference: str = "auto",
    execution_mode: str = "auto",
    show_thinking: bool = False,
    enable_thinking: Optional[bool] = None,
    max_new_tokens: int = 1024,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    read_timeout: Optional[float] = None,
    streaming_mode: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """POST ``/api/chat/stream`` 并逐事件产出 JSON payload（供界面 worker 消费）。

    同步生成器：调用方负责放进线程/worker，避免阻塞 UI 事件循环。
    """
    body = build_interactive_request(
        message,
        session_id=session_id,
        generation_id=generation_id,
        routing_preference=routing_preference,
        execution_mode=execution_mode,
        show_thinking=show_thinking,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        streaming_mode=streaming_mode,
    )
    url = api.base_url + "/api" + API_PATHS["chat_stream"]
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=api._request_headers(
            accept="text/event-stream", content_type=True,
        ),
        method="POST",
    )
    # Full responses have no token activity during distributed prefill.
    # Match the Android finite read budget; preserve explicit caller overrides.
    effective_read_timeout = (120.0 if body['streaming_mode'] == 'full' else 60.0) if read_timeout is None else read_timeout
    decoder = SSEDecoder()
    try:
        with urllib.request.urlopen(req, timeout=effective_read_timeout) as resp:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                for event in decoder.feed(chunk):
                    payload = decode_json_event(event)
                    if payload:
                        yield payload
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw else {}
            detail = parsed.get("detail", raw) if isinstance(parsed, dict) else raw
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
        except Exception:  # noqa: BLE001 - preserve the HTTP status on bad bodies
            detail = ""
        raise ApiError(
            "HTTP %d: %s" % (exc.code, detail or exc.reason), status=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise ApiError("无法连接后端（SSE）：%s" % getattr(exc, "reason", exc)) from exc
    except OSError as exc:
        raise ApiError("SSE 网络错误: %s" % exc) from exc


# ---------------------------------------------------------- 模型资产(C) / 存储(D)
def list_models_available(api: ApiClient) -> Dict[str, Any]:
    """可选模型配置 + 可用引擎（GET ``/models/available``）。★ 补缺口 C。"""
    return _as_dict(api.get(API_PATHS["models_available"]))


def list_model_registry(api: ApiClient) -> Dict[str, Any]:
    """用户注册的实验模型（GET ``/models/registry``）。★ 补缺口 C。"""
    return _as_dict(api.get(API_PATHS["models_registry"]))


def list_models_downloadable(api: ApiClient) -> Dict[str, Any]:
    """可下载模型清单（GET ``/models/downloadable``）。★ 补缺口 C。"""
    return _as_dict(api.get(API_PATHS["models_downloadable"]))


def list_local_gguf(api: ApiClient) -> Dict[str, Any]:
    """本地 GGUF 文件（GET ``/models/gguf``）。★ 补缺口 C。"""
    return _as_dict(api.get(API_PATHS["models_gguf"]))


def db_health(api: ApiClient) -> Dict[str, Any]:
    """数据库健康（GET ``/db/health``）。★ 补缺口 D。"""
    return _as_dict(api.get(API_PATHS["db_health"]))


def storage_health(api: ApiClient) -> Dict[str, Any]:
    """存储健康（GET ``/storage/health``）。★ 补缺口 D。"""
    return _as_dict(api.get(API_PATHS["storage_health"]))


# ---------------------------------------------------------------- 会话细粒度(F)
def get_conversation(api: ApiClient, session_id: str = "default",
                     limit: int = 200) -> Dict[str, Any]:
    """读对话历史（GET ``/conversations?session_id=&limit=``）。★ 补缺口 F。"""
    query = urllib.parse.urlencode({"session_id": session_id, "limit": int(limit)})
    return _as_dict(api.get(f"{API_PATHS['conversations']}?{query}"))


def conversation_sync_status(api: ApiClient) -> Dict[str, Any]:
    """本地持久化状态（GET ``/conversations/sync-status``）。★ 补缺口 F。"""
    return _as_dict(api.get(API_PATHS["conversation_sync_status"]))


def session_info(api: ApiClient, session_id: str) -> Dict[str, Any]:
    """单个会话元数据（GET ``/sessions/{session_id}``）。★ 补缺口 F。"""
    path = API_PATHS["session_info"].format(**_quoted(session_id=session_id))
    return _as_dict(api.get(path))


def delete_turn(api: ApiClient, session_id: str, turn_index: int) -> Dict[str, Any]:
    """删单轮（DELETE ``/sessions/{session_id}/turns/{turn_index}``）。

    ⚠️ 同时删除该轮的 user + assistant 两条消息；`turn_index` 为 0-based。★ 补缺口 F。
    """
    path = API_PATHS["session_turn"].format(
        **_quoted(session_id=session_id), turn_index=int(turn_index))
    return _as_dict(api.request("DELETE", path))


# ---------------------------------------------------- 认证与账户(G-⑤, 2026-09-19)
def auth_capability(api: ApiClient) -> Dict[str, Any]:
    """认证能力（GET ``/auth/capability``）：是否强制登录 / 是否处于首次引导。"""
    return _as_dict(api.get(API_PATHS["auth_capability"]))


def auth_login(api: ApiClient, username: str, password: str,
               totp_code: Optional[str] = None) -> Dict[str, Any]:
    """登录（POST ``/auth/login``）。返回含明文 ``token``；**调用方须写入 api.auth_token**。"""
    body: Dict[str, Any] = {"username": username, "password": password}
    if totp_code:
        body["totp_code"] = totp_code
    return _as_dict(api.post(API_PATHS["auth_login"], body))


def auth_logout(api: ApiClient) -> Dict[str, Any]:
    """注销（POST ``/auth/logout``）——服务端吊销当前 Bearer。"""
    return _as_dict(api.post(API_PATHS["auth_logout"]))


def auth_me(api: ApiClient) -> Dict[str, Any]:
    """当前主体（GET ``/auth/me``）。"""
    return _as_dict(api.get(API_PATHS["auth_me"]))


def auth_totp_provision(api: ApiClient) -> Dict[str, Any]:
    """生成并绑定 TOTP 密钥（POST ``/auth/totp/provision``）⇒ 返回 ``otpauth_uri``。"""
    return _as_dict(api.post(API_PATHS["auth_totp_provision"]))


def auth_totp_verify(api: ApiClient, code: str) -> Dict[str, Any]:
    """校验 TOTP（POST ``/auth/totp/verify``）。"""
    return _as_dict(api.post(API_PATHS["auth_totp_verify"], {"code": code}))


def list_auth_users(api: ApiClient) -> Dict[str, Any]:
    """账户列表（GET ``/users``，需 admin）。"""
    return _as_dict(api.get(API_PATHS["auth_users"]))


def create_auth_user(api: ApiClient, username: str, password: str,
                     role: str = "viewer") -> Dict[str, Any]:
    """创建账户（POST ``/users``）。**首次引导**时可无登录调用（仅允许 admin）。"""
    return _as_dict(api.post(API_PATHS["auth_users"],
                             {"username": username, "password": password, "role": role}))


def patch_auth_user(api: ApiClient, username: str, *, role: Optional[str] = None,
                    disabled: Optional[bool] = None,
                    password: Optional[str] = None) -> Dict[str, Any]:
    """修改账户（PATCH ``/users/{username}``）：角色 / 禁用 / 重置口令。"""
    path = API_PATHS["auth_user"].format(**_quoted(username=username))
    body: Dict[str, Any] = {}
    if role is not None:
        body["role"] = role
    if disabled is not None:
        body["disabled"] = bool(disabled)
    if password is not None:
        body["password"] = password
    return _as_dict(api.request("PATCH", path, body))


def delete_auth_user(api: ApiClient, username: str) -> Dict[str, Any]:
    """删除账户（DELETE ``/users/{username}``）。"""
    path = API_PATHS["auth_user"].format(**_quoted(username=username))
    return _as_dict(api.request("DELETE", path))
