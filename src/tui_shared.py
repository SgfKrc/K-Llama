"""T9 共享层 — 聊天页与管理 TUI 共用的路径、格式化与命令注册。

设计（TUI 适配实施计划 §9.3）：
- 只含纯函数与常量，无第三方依赖（tui_admin.py 的标准库环境也可安全导入）；
- 端点字符串集中在 API_PATHS，聊天页与管理 TUI 不各自散落复制；
- metrics 展示只依据 done 事件实际字段，不推断分布式参与。
"""

from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from multimodal import (
        MAX_CHAT_IMAGE_BYTES,
        MAX_CHAT_IMAGES,
        MAX_CHAT_IMAGE_TOTAL_BYTES,
        validate_image_data_urls,
    )
except ImportError:  # pragma: no cover - package import path
    from .multimodal import (  # type: ignore
        MAX_CHAT_IMAGE_BYTES,
        MAX_CHAT_IMAGES,
        MAX_CHAT_IMAGE_TOTAL_BYTES,
        validate_image_data_urls,
    )

# ============================================================
# API 端点路径（相对 /api 前缀；host 由调用方拼接）
# ============================================================

API_PATHS = {
    "chat_stream": "/chat/stream",
    "chat_cancel": "/chat/generations/{generation_id}/cancel",
    "sessions": "/sessions",
    "sessions_detail": "/sessions/{session_id}",
    "sessions_activate": "/sessions/{session_id}/activate",
    "conversations": "/conversations",
    "models_current": "/models/current",
    "models_intent": "/models/intent",
    "models_intent_catalog": "/models/intent/catalog",
    "models_intent_current": "/models/intent/current",
    "models_intent_operation": "/models/intent/{operation_id}",
    "models_intent_connect": "/models/intent/connect",
    "distributed_config": "/cluster/config/distributed-inference",
    "cluster_resources": "/cluster/resources",
    "cluster_control_plane": "/cluster/control-plane",
    "cluster_layers": "/cluster/layers",
    "cluster_pipeline_capacity": "/cluster/pipeline-capacity",
    "cluster_pipeline_reshard": "/cluster/pipeline-reshard",
    # 只读运维面（Textual 外壳的节点/队列/日志屏）
    "cluster_nodes": "/cluster/nodes",
    "cluster_queue": "/cluster/queue",
    "cluster_log_aggregate": "/cluster/nodes/log-aggregate",
    # 设备画像（后端所在机器）
    "device_profile": "/device/profile",
    # 2026-09-17 按后端真实返回对齐（此前沿用旧 TUI 假设，导致多屏显示"无数据"）
    "health": "/health",
    "readiness": "/ready",
    "system_status": "/status",
    "models_list": "/models",

    # ---- 写操作面（用户 2026-09-17 裁决：模型控制 + 会话管理 + 队列控制）----
    # 权限：后端 model_api_access.require_model_api_source() 对 **loopback 默认放行**，
    # 故本机 TUI 可直接调用；从节点远程控制主节点需显式 QLH_MODEL_API_TRUSTED_CIDRS。
    "models_load": "/models/load",          # POST {engine, quant_type, use_compile, model_id}
    "models_unload": "/models/unload",      # POST（无请求体）
    "chat_clear": "/chat/clear",            # POST 清空后端当前会话历史 + KV 缓存
    "session_detail": "/sessions/{session_id}",            # PUT 重命名 / DELETE 删除
    "session_activate": "/sessions/{session_id}/activate",  # POST 恢复会话
    "cluster_queue_pause": "/cluster/queue/pause",          # POST（仅主节点）
    "cluster_queue_resume": "/cluster/queue/resume",        # POST（仅主节点）
    "cluster_queue_clear": "/cluster/queue/clear",          # POST（仅主节点）
    # ---- 2026-09-19 补缺口 A：队列「单任务」取消（此前只能整体 clear）----
    "cluster_queue_task_cancel": "/cluster/queue/task/{task_id}",  # DELETE（仅主节点）
    # ---- 2026-09-19 补缺口 B：日志细粒度（此前只有 recent/stats/export/整体清理）----
    "logs_list": "/logs",                         # GET 日志文件列表
    "logs_download": "/logs/download",            # GET 下载
    "logs_file": "/logs/{filename}",              # GET 读单文件 / DELETE 删单文件
    "logs_nodes_summary": "/logs/nodes-summary",  # GET 各节点日志汇总
    # ---- Static cluster role maintenance ----
    "cluster_master_health": "/cluster/master-health",       # GET configured master health
    "cluster_reset_identity": "/cluster/reset-identity",     # POST ⚠️ 高危（需 confirm="reset"）
    # ---- 2026-09-19 补缺口 C：模型资产浏览（只读）----
    "models_available": "/models/available",       # GET 可选模型配置 + 可用引擎
    "models_registry": "/models/registry",         # GET 用户注册的实验模型
    "models_downloadable": "/models/downloadable",  # GET 可下载清单
    "models_gguf": "/models/gguf",                 # GET 本地 GGUF 文件
    # ---- 2026-09-19 补缺口 D：存储健康（只读）----
    "db_health": "/db/health",                     # GET SQLite 健康
    "storage_health": "/storage/health",           # GET 存储健康
    # ---- 2026-09-19 补缺口 F：会话细粒度（读为主）----
    "session_info": "/sessions/{session_id}",              # GET 会话元数据
    "session_turn": "/sessions/{session_id}/turns/{turn_index}",  # DELETE 删单轮
    "conversation_sync_status": "/conversations/sync-status",  # GET 持久化状态
}

# ============================================================
# 路由偏好
# ============================================================

MODEL_INTENT_STATUSES = (
    "pending", "queued", "waiting_capacity", "waiting_nodes", "inspecting", "inventory", "planning", "building",
    "supplying", "verifying", "preparing", "ready", "failed", "cancelled", "superseded",
)
MODEL_INTENT_TERMINAL = frozenset({"ready", "failed", "cancelled", "superseded"})
MODEL_INTENT_LABELS = dict(zip(MODEL_INTENT_STATUSES, (
    "待处理", "排队中", "等待容量", "等待节点", "检查模型", "盘点资产", "规划层段", "构建层段",
    "供给中", "校验中", "准备中", "就绪", "失败", "已取消", "已被替代",
)))


def model_intent_ready(operation: Dict[str, Any]) -> bool:
    revision, observed = operation.get("revision"), operation.get("observed_at")
    return (operation.get("schema_version") == 2 and operation.get("validity") == "valid"
            and type(revision) is int and revision >= 0
            and isinstance(operation.get("identity"), dict) and bool(operation["identity"].get("boot_id"))
            and type(observed) in (int, float) and math.isfinite(observed) and observed > 0
            and operation.get("status") == "ready" and operation.get("runnable") is True
            and operation.get("loaded_ready") is True)


# These are capacity rejections, rather than worker discovery/registration states.
_MODEL_INTENT_CAPACITY_REASONS = frozenset({
    "pipeline_distributed_capacity_insufficient",
    "pipeline_cluster_capacity_insufficient",
    "pipeline_capacity_single_node_insufficient",
    "pipeline_capacity_manual_insufficient",
    "pipeline_reshard_capacity_insufficient",
    "node_capacity_unavailable",
    "SUPPLY_CAPACITY_INSUFFICIENT",
    "SUPPLY_MASTER_CAPACITY_INSUFFICIENT",
    "SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT",
})


def _format_capacity_bytes(value: Any) -> Optional[str]:
    # Missing/legacy fields are not zero budgets. Never derive a deficit here.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    for unit, divisor in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value >= divisor:
            return f"{value / divisor:.2f} {unit} ({value} bytes)"
    return f"{value} bytes"


def _capacity_diagnostic_has_positive_deficit(diagnostic: Dict[str, Any]) -> bool:
    nodes = diagnostic.get("nodes")
    if not isinstance(nodes, (list, tuple)):
        return False
    for node in nodes:
        if not isinstance(node, dict):
            continue
        deficit = node.get("deficit_bytes")
        if isinstance(deficit, int) and not isinstance(deficit, bool) and deficit > 0:
            return True
    return False


def _format_model_intent_capacity(
    diagnostic: Dict[str, Any], *, displayed_reason: Optional[str] = None,
) -> List[str]:
    scope_value = diagnostic.get("scope")
    scope = {"candidate_plan": "候选方案", "minimum_segment": "最小层段"}.get(
        scope_value if isinstance(scope_value, str) else "", "范围未提供"
    )
    lines = [f"容量不足（{scope}；释放所示缺口不保证可执行）"]
    reason = diagnostic.get("reason")
    if isinstance(reason, str) and reason.strip() and reason != displayed_reason:
        lines.append(reason)
    nodes = diagnostic.get("nodes")
    if isinstance(nodes, (list, tuple)):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("node_id")
            parts = [node_id if isinstance(node_id, str) and node_id else "未标识节点"]
            for key, labels in (
                ("engine", {"llama_cpp": "llama.cpp", "pytorch": "PyTorch"}),
                ("execution_device", {"cpu": "CPU", "cuda": "CUDA"}),
                ("resource", {"ram": "RAM", "vram": "显存（VRAM）", "host_ram": "主机暂存 RAM"}),
            ):
                value = node.get(key)
                if isinstance(value, str) and value:
                    parts.append(labels.get(value, value))
            layer_range = node.get("layer_range")
            if (isinstance(layer_range, (list, tuple)) and len(layer_range) == 2
                    and all(isinstance(bound, int) and not isinstance(bound, bool) for bound in layer_range)
                    and 0 <= layer_range[0] < layer_range[1]):
                parts.append(f"层段 [{layer_range[0]}, {layer_range[1]})")
            reserve_label = "含边界的预留" if node.get("resource") in ("ram", "vram") else "预留"
            for key, label in (("available_bytes", "可用"), ("required_bytes", "所需"),
                               ("deficit_bytes", "缺口"), ("reserve_bytes", reserve_label)):
                value = _format_capacity_bytes(node.get(key))
                if value is not None:
                    parts.append(f"{label} {value}")
            lines.append(" · ".join(parts))
    pending = diagnostic.get("pending_node_ids")
    if isinstance(pending, list):
        pending_ids = [node_id for node_id in pending if isinstance(node_id, str) and node_id.strip()]
        if pending_ids:
            lines.append(f"仍在加入的节点：{', '.join(pending_ids)}")
    action = diagnostic.get("suggested_action")
    if isinstance(action, str) and action.strip():
        lines.append(f"建议：{action}")
    return lines


def format_model_intent(operation: Dict[str, Any]) -> str:
    """Display only the intent owner's facts, including node receipts and reasons."""
    if not operation:
        return "尚未选择分布式模型"
    status = str(operation.get("status") or "unknown")
    intent = operation.get("intent") or {}
    model_id = intent.get("model_id", "") if isinstance(intent, dict) else ""
    label = MODEL_INTENT_LABELS.get(status, status)
    reason_code = operation.get("reason_code")
    diagnostic = operation.get("capacity_diagnostic")
    diagnostic_matches = (isinstance(diagnostic, dict) and isinstance(reason_code, str)
                          and bool(reason_code) and diagnostic.get("reason_code") == reason_code)
    # Solver exclusions may share a code with worker/segment availability failures.
    # Only the owner's explicit positive deficit disambiguates an unfamiliar code.
    capacity_cause = (
        status in {"waiting_nodes", "failed"} and isinstance(reason_code, str)
        and reason_code != "SUPPLY_WAITING_NODES"
        and (reason_code in _MODEL_INTENT_CAPACITY_REASONS
             or (diagnostic_matches and _capacity_diagnostic_has_positive_deficit(diagnostic)))
    )
    if status == "waiting_nodes" and capacity_cause:
        label = "等待容量"
    if status == "ready" and not model_intent_ready(operation):
        label = "等待加载确认"
    lines = [f"{model_id} · {label} ({status})",
             f"operation_id: {operation.get('operation_id') or '—'}",
             f"config_id: {operation.get('config_id') or '—'} · generation: {operation.get('generation', 0)}"
             f" · loaded_ready: {str(operation.get('loaded_ready') is True).lower()}"]
    if operation.get("reason_code") or operation.get("reason"):
        lines.append(f"{operation.get('reason_code') or ''}: {operation.get('reason') or ''}")
    # A matching failed cause remains useful after a deadline or host RAM rejection.
    # Other lifecycle states and reasons must not display a previous plan's diagnosis.
    if capacity_cause and diagnostic_matches:
        lines.extend(_format_model_intent_capacity(
            diagnostic, displayed_reason=operation.get("reason"),
        ))
    nodes = operation.get("nodes") or {}
    if isinstance(nodes, dict):
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            line = (f"{node_id}: {node.get('status') or '—'} · "
                    f"{node.get('bytes_received', 0)}/{node.get('total_bytes', 0)} bytes")
            if node.get("layer_range") is not None:
                line += f" · layer_range: {node['layer_range']}"
            if node.get("reason_code") or node.get("reason"):
                line += f" · {node.get('reason_code') or ''}: {node.get('reason') or ''}"
            lines.append(line)
    return "\n".join(lines)


ROUTING_PREFERENCES = (
    "auto",
    "local_only",
    "distributed_preferred",
    "distributed_required",
)

ROUTE_LABELS = {
    "auto": "route:auto",
    "local_only": "route:local",
    "distributed_preferred": "route:distributed",
    "distributed_required": "route:required",
}

ROUTE_SHORT_ARGS = {
    "auto": "auto",
    "local": "local_only",
    "distributed": "distributed_preferred",
    "required": "distributed_required",
}

#: 请求级执行模式由版本化客户端合同限定。
EXECUTION_MODES = (
    "auto",
)

EXECUTION_MODE_LABELS = {
    "auto": "execute:auto",
}


# ============================================================
# interactive 请求体构造
# ============================================================

def build_interactive_request(
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
    image_data_urls: Optional[List[str]] = None,
    streaming_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """构造聊天流请求；绑定从节点可显式使用 full，默认保留 T9 interactive 契约。"""
    if streaming_mode not in (None, "interactive", "full"):
        raise ValueError("streaming_mode must be 'interactive' or 'full'")
    if routing_preference not in ROUTING_PREFERENCES:
        routing_preference = "auto"
    if execution_mode not in EXECUTION_MODES:
        execution_mode = "auto"
    images = validate_image_data_urls(image_data_urls or [])
    local_image = bool(images and routing_preference == "local_only")
    if local_image and len(images) != 1:
        raise ValueError("本地 MTMD 图像请求仅支持一张图片")
    if local_image and streaming_mode == "interactive":
        raise ValueError("本地 MTMD 图像请求要求 streaming_mode='full'")
    body = {
        "message": message,
        "streaming_mode": "full" if local_image else streaming_mode or "interactive",
        "generation_id": generation_id,
        "session_id": session_id,
        "routing_preference": routing_preference,
        "execution_mode": execution_mode,
        "show_thinking": show_thinking,
        "enable_thinking": enable_thinking,
        "max_new_tokens": max_new_tokens,
    }
    # Omitted sampling uses the API's route-specific default. Explicit caller
    # choices stay explicit and receive the normal capability admission check.
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if images:
        # 图像请求使用统一自动路由。
        body.update({
            "image_data_urls": images,
            "execution_mode": "auto",
            "allow_external": not local_image,
            "prefer_external": not local_image,
        })
    return body


def load_local_chat_image(path_value: str) -> Dict[str, Any]:
    """读取 TUI 用户明确指定的本地图片并构造一次性 data URL。"""
    candidate = (path_value or "").strip().strip('"')
    if not candidate:
        raise ValueError("用法: /image <PNG/JPEG/WebP 本地路径>")
    path = Path(candidate).expanduser()
    try:
        if not path.is_file():
            raise ValueError("图片路径不存在或不是普通文件")
        size = path.stat().st_size
        if size > MAX_CHAT_IMAGE_BYTES:
            raise ValueError("图片超过单张 8 MiB 上限")
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"无法读取图片: {exc}") from exc

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("只支持内容正确的 PNG、JPEG 或 WebP 图片")

    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    validate_image_data_urls([data_url])
    return {
        "name": path.name,
        "size": size,
        "data_url": data_url,
    }


# ============================================================
# metrics 格式化（done 事件）
# ============================================================

#: 请求侧要求分布式的 `routing_preference` 取值（见 `ChatRequest.routing_preference`）。
_DISTRIBUTED_REQUEST_PREFS = frozenset({"distributed_preferred", "distributed_required"})


def _distributed_was_requested(metrics: Dict[str, Any]) -> bool:
    """本次请求**是否要求过分布式**（请求侧意图 或 全局开关，任一为真）。

    ★ 2026-10-08：此前只看 `distributed_requested`（= 全局开关 `distributed_inference`）。
    用户用 `/route required` 要求分布式、而全局开关恰好关着时它是 `False` ⇒
    「已请求分布式，实际本地」这条提示**不出现**、界面完全静默。

    请求侧意图由 `metrics["routing_preference"]` 承载（`api_server._augment_chat_metrics`
    写入；此前它只在响应顶层、进不了界面）。
    """
    if metrics.get("distributed_requested"):
        return True
    return str(metrics.get("routing_preference") or "").strip().lower() in _DISTRIBUTED_REQUEST_PREFS


def _distributed_evidence(metrics: Dict[str, Any]) -> str:
    """远端**实际承了哪段层**的短证据串（无证据则空串）。

    为什么需要（2026-10-08）：`execution_mode=route_a_stage_offer_v3` 只说明**选了哪条
    路由协议**，不证明远端真的执行了 —— 两者在旧 footer 里长得一模一样，于是用户只能
    靠「吞吐变慢」反推是不是分布式。这里把 metrics 里已有的承层事实（`layer_segments`
    / `claimed_layers` / `workers_used`，见 `scheduler_pipeline` 的 route A metrics）
    压成一段短串，让「分布式」可核验而不是仅声明。
    """
    segments = metrics.get("layer_segments")
    if isinstance(segments, (list, tuple)) and segments:
        try:
            return f"（{len(segments)} 段·层 {int(segments[0][0])}-{int(segments[-1][1])}）"
        except (TypeError, ValueError, IndexError):
            return f"（{len(segments)} 段）"
    claimed = metrics.get("claimed_layers")
    if isinstance(claimed, (list, tuple)) and len(claimed) == 2:
        try:
            return f"（层 {int(claimed[0])}-{int(claimed[1])}）"
        except (TypeError, ValueError):
            pass
    workers = metrics.get("workers_used")
    if isinstance(workers, (list, tuple)) and workers:
        return f"（{len(workers)} 个 worker）"
    return ""


def format_metrics(
    metrics: Optional[Dict[str, Any]] = None,
    *,
    history_committed: Optional[bool] = None,
) -> str:
    """把 done 事件 metrics 格式化为状态行文本。

    展示规则（§9.5）：只读取完成事件中的实际字段；fallback 必须展示原因；
    history_committed=false 必须提示。

    ★ 2026-10-08（分布式可核验）：`distributed_used` **无论真假都要显式呈现**。
    旧规则只在「请求了分布式但没用」时才提示，于是下面三种情况在界面上**完全一样**：
    ① 真的分布式；② 请求了但静默退回本地；③ 请求根本没带分布式意图 —— 用户只能靠
    吞吐速度猜（实测反馈：跑通一次却无法确认是不是真分布式）。现在：

    * `distributed_used=True` ⇒ `分布式 ✓` ＋ 承层证据（远端实际承了哪段层）；
    * 请求了但没用 ⇒ `⚠️ 已请求分布式，实际本地`（保留原措辞）；
    * 后端**明确**给了 `distributed_used=False` ⇒ `⚠️ 本地执行（未用分布式）`
      —— 这一条正是「整模回退伪装成普通本地推理」在 UI 上的对策。
    """
    metrics = metrics or {}
    parts: List[str] = []
    engine = metrics.get("engine") or metrics.get("execution_mode") or "unknown"
    route = metrics.get("execution_mode") or "local"
    parts.append(f"{engine} · {route}")
    tokens = metrics.get("tokens_generated") or metrics.get("generated_tokens")
    if tokens is not None:
        parts.append(f"{tokens} tokens")
    tok_s = metrics.get("tok_per_sec") or metrics.get("tokens_per_second")
    if tok_s is not None:
        try:
            parts.append(f"{float(tok_s):.1f} tok/s")
        except (TypeError, ValueError):
            parts.append(f"{tok_s} tok/s")
    if metrics.get("fallback"):
        parts.append(f"⚠️ 回退: {metrics.get('fallback_reason', '未知')}")
    if metrics.get("distributed_used"):
        parts.append(f"分布式 ✓{_distributed_evidence(metrics)}")
    elif _distributed_was_requested(metrics):
        parts.append("⚠️ 已请求分布式，实际本地")
    elif "distributed_used" in metrics:
        parts.append("⚠️ 本地执行（未用分布式）")
    # 模型准备 identity 只取本轮 done.metrics，兼容未返回这些字段的旧请求。
    for key, label in (("model_operation_id", "模型准备"), ("model_config_id", "模型配置")):
        value = metrics.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}：{value}")
    generation = metrics.get("model_generation")
    if isinstance(generation, int) and not isinstance(generation, bool):
        parts.append(f"模型代次：{generation}")
    if history_committed is False:
        parts.append("历史未提交")
    return " · ".join(parts)


def parse_session_line(session: Dict[str, Any]) -> str:
    """会话 dict → 单行显示文本（兼容后端 id / session_id 两种字段）。"""
    session_id = session.get("session_id") or session.get("id") or ""
    title = session.get("title") or "(未命名)"
    count = session.get("message_count")
    suffix = f" · {count} 条" if count is not None else ""
    return f"{session_id}  {title}{suffix}"


# ============================================================
# T9 命令注册表（/help 与校验共用）
# ============================================================

#: 命令注册表 —— **只登记已实现的命令**（ChatPane.on_input_submitted 分支与之一一对应）；
#: 未实现的旧规范条目（/image、/images、/image-clear）已移除，避免 /help "写着却不能用"。
COMMAND_SPECS: List[Dict[str, str]] = [
    {"name": "/new", "args": "[title]", "desc": "新建并切换会话（POST /sessions）"},
    {"name": "/resume", "args": "<session_id>", "desc": "恢复历史会话并渲染其消息"},
    {"name": "/rename", "args": "<title>", "desc": "重命名当前会话"},
    {"name": "/sessions", "args": "", "desc": "列出最近会话"},
    {"name": "/delete-session", "args": "", "desc": "删除当前会话及其全部消息（需确认）"},
    {"name": "/reset", "args": "", "desc": "清空后端会话历史与 KV 缓存（需确认）"},
    {"name": "/model", "args": "load <id> [engine] [quant] | unload | cancel | status | connect | mode local|distributed",
     "desc": "选模与准备状态；local 退出集群并卸载后加载整模，distributed 自动供给并准备"},
    {"name": "/queue",
     "args": "pause | resume | clear | cancel <task_id>",
     "desc": "队列控制（clear 需确认）"},
    {"name": "/route", "args": "auto|local|distributed|required",
     "desc": "设置请求级路由偏好"},
    {"name": "/thinking", "args": "on|off", "desc": "思考内容**展示**（仅 UI 显隐，不改变模型行为）"},
    {"name": "/reasoning", "args": "on|off|auto",
     "desc": "深度思考**开关**（改变模型行为）：on=强制思考 / off=强制不思考（省算力，"
             "可避免 Qwen3 等模型输出超长 `<think>`）/ auto=沿用模型模板默认"},
    {"name": "/cancel", "args": "", "desc": "取消当前生成"},
    {"name": "/logs", "args": "list | download <file> | read <file> | delete <file> | nodes",
     "desc": "日志细粒度：文件列表 / 下载 / 查看 / 删除（需确认）/ 各节点汇总"},
    {"name": "/history", "args": "[<session_id>] [limit] | sync-status | info <session_id> "
                              "| drop-turn <session_id> <turn_index>",
     "desc": "会话历史：查看对话（默认当前会话）/ 本地持久化状态 / 会话详情 / "
             "删单轮（需确认，删 user+assistant 两条）"},
    {"name": "/assets", "args": "available | registry | downloadable | gguf",
     "desc": "模型资产浏览（只读）：可选模型与引擎 / 已注册实验模型 / 可下载清单 / 本地 GGUF"},
    {"name": "/storage", "args": "", "desc": "存储与数据库健康（只读）"},
    {"name": "/ha", "args": "health | reset-identity",
     "desc": "静态集群主节点：健康检查 / 重置本机身份"},
    {"name": "/clear", "args": "", "desc": "清空本地显示（不动后端；清后端用 /reset）"},
    {"name": "/help", "args": "", "desc": "显示本帮助"},
    {"name": "/quit", "args": "", "desc": "退出聊天页"},
]


def help_text() -> str:
    """生成 /help 的一行文本。"""
    items = []
    for spec in COMMAND_SPECS:
        parts = [spec["name"]]
        if spec["args"]:
            parts.append(spec["args"])
        items.append(" ".join(parts))
    return "可用命令：" + " ".join(items) + "；Enter 发送 · Alt+Enter 换行 · Ctrl+C 停止"


def resolve_route_arg(arg: str) -> Optional[str]:
    """把 /route 参数解析为完整 routing_preference；非法返回 None。"""
    normalized = (arg or "").strip().lower()
    if normalized in ROUTING_PREFERENCES:
        return normalized
    return ROUTE_SHORT_ARGS.get(normalized)
