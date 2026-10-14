"""QLH 单命令薄层（**只读**）—— 取代已归档的旧交互 TUI 命令注册表。

背景（2026-09-17）：旧标准库 TUI（`src/tui_admin.py` 3247 行 + `tui_chat_screen.py` +
`tui_splash.py`）整体归档到 `_to_delete/`（自绘 ANSI 在真实 conhost 下不可见、观感与
维护成本都不划算，交互面改由 Textual 外壳 `src/tui_textual.py` 承担）。

因此 `qlh status` 这类**单命令模式**不能继续依赖旧 TUI，改由本模块实现：

* 只用**纯标准库**协议层（`tui_api.ApiClient` + `tui_sse`/`tui_shared`），不 import 任何 UI；
* 只做**只读查询**（status/models/nodes/queue/device/logs/help），不写后端、不启动后端；
* 命中写操作/旧交互命令时给出明确提示并返回 rc=2（**不静默失效**）；
* `--fixture PATH` 离线回放能力保留（原 `tui_chat_screen.smoke_from_fixture` 迁到这里）。

用法::

    python src/tui_commands.py status --port 8000
    python src/tui_commands.py models --json
    python src/tui_commands.py --fixture tests/fixtures/chat_stream.sse
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from typing import Any, Callable, Dict, List, Optional

try:  # 直接运行（python src/tui_commands.py）
    from tui_api import (
        BACKEND_HINT,
        DEFAULT_HOST,
        DEFAULT_PORT,
        ApiClient,
        ApiError,
    )
    from tui_shared import API_PATHS, format_metrics, format_model_intent, parse_session_line
    from tui_sse import SSEDecoder, decode_json_event
except ImportError:  # pragma: no cover - 包导入路径
    from .tui_api import (  # type: ignore
        BACKEND_HINT,
        DEFAULT_HOST,
        DEFAULT_PORT,
        ApiClient,
        ApiError,
    )
    from .tui_shared import API_PATHS, format_metrics, format_model_intent, parse_session_line  # type: ignore
    from .tui_sse import SSEDecoder, decode_json_event  # type: ignore

# ---------------------------------------------------------------- 只读命令表

READ_ONLY_COMMANDS: Dict[str, str] = {
    "status": "系统状态与当前模型（/health、/status、/models/current）",
    "models": "模型目录与当前状态（分布式读取 intent catalog/current）",
    "nodes": "集群节点列表（/cluster/nodes）",
    "queue": "请求队列与调度策略（/cluster/queue）",
    "device": "本机设备画像（/device/profile）",
    "logs": "聚合日志（/cluster/nodes/log-aggregate）",
    "help": "显示本帮助",
}

#: 旧交互 TUI 的命令（归档后不再支持）——给出明确指引，而不是"未知命令"
LEGACY_COMMANDS = {
    "quit", "q", "exit", "shutdown", "halt", "screen", "goto", "refresh", "r",
    "model", "switch", "load", "quant", "engine", "presets", "gpu", "connect",
    "join", "dist", "host", "interval", "timeout", "token", "chat", "new",
    "sessions", "resume", "rename", "delete-session", "route", "thinking", "cancel",
    "st", "log", "help ", "nodes?",
}

# ---------------------------------------------------------------- 文本工具


def disp_width(text: str) -> int:
    """显示宽度：东亚宽/全角字符按 2 列（原在已归档的 tui_admin 中）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize(text: str) -> str:
    """过滤 ANSI 转义序列与危险控制字符（保留 \\n 与 \\t）。"""
    if not text:
        return ""
    return _CTRL_RE.sub("", _ANSI_RE.sub("", text))


def wrap_display(text: str, width: int) -> List[str]:
    """按显示宽度折行（中文按 2 列）。"""
    if width <= 0:
        return [text]
    out: List[str] = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        cur = ""
        for ch in para:
            if cur and disp_width(cur + ch) > width:
                out.append(cur)
                cur = ch
            else:
                cur += ch
        out.append(cur)
    return out


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _line(label: str, value: Any) -> str:
    return "  %-14s: %s" % (label, value if value not in (None, "") else "—")


# ---------------------------------------------------------------- 只读命令


def cmd_status(api: ApiClient, args: argparse.Namespace) -> int:
    health = api.get("/health")
    status = _safe(api, "/status")
    distributed = _distributed_mode(api, status)
    current = _safe(api, API_PATHS["models_intent_current"] if distributed else API_PATHS["models_current"])
    _emit({"health": health, "status": status, "current_model": current}, args.json)
    if args.json:
        return 0
    print("◆ 系统状态")
    print(_line("后端", api.base_url))
    print(_line("健康", (health or {}).get("status", "ok")))
    if isinstance(status, dict):
        print(_line("节点角色", status.get("node_role")))
        print(_line("节点 ID", status.get("node_id")))
        print(_line("最大节点数", status.get("max_nodes")))
    if distributed and isinstance(current, dict):
        print(format_model_intent(current))
    elif isinstance(current, dict):
        print(_line("当前模型", current.get("model_id") or current.get("name")))
        print(_line("引擎", current.get("engine")))
        print(_line("格式", current.get("format")))
    return 0


def cmd_models(api: ApiClient, args: argparse.Namespace) -> int:
    status = _safe(api, API_PATHS["system_status"])
    distributed = _distributed_mode(api, status)
    registry = _safe(api, API_PATHS["models_intent_catalog"] if distributed else API_PATHS["models_list"])
    current = _safe(api, API_PATHS["models_intent_current"] if distributed else API_PATHS["models_current"])
    _emit({"models": registry, "current": current}, args.json)
    if args.json:
        return 0
    current_id = ((current.get("intent") or {}).get("model_id") if distributed
                  else current.get("model_id")) if isinstance(current, dict) else None
    if distributed:
        binding = (registry.get("binding") or {}) if isinstance(registry, dict) else {}
        print(_line("主节点连接", binding.get("status")))
        if binding.get("reason_code") or binding.get("reason"):
            print(_line("连接原因", f"{binding.get('reason_code', '')}: {binding.get('reason', '')} · TUI J 发现/重试连接"))
        if isinstance(current, dict):
            print(format_model_intent(current))
    items = registry.get("models") if isinstance(registry, dict) else registry
    if isinstance(items, dict):
        items = [{"model_id": key, **(value if isinstance(value, dict) else {})}
                 for key, value in items.items()]
    print("◆ 模型（* = 当前）")
    if not items:
        print("  （后端未返回模型列表）")
    for item in (items or [])[:64]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id") or item.get("name") or "—"
        mark = "*" if model_id == current_id else " "
        print("  %s %-32s %-9s %s" % (
            mark, model_id, item.get("format") or ", ".join(item.get("available_formats") or []) or "—",
            item.get("engine") or item.get("preferred_engine") or "—"))
    print(_line("当前模型", current_id))
    return 0


def cmd_nodes(api: ApiClient, args: argparse.Namespace) -> int:
    nodes = api.get(API_PATHS["cluster_nodes"])
    _emit(nodes, args.json)
    if args.json:
        return 0
    items = nodes.get("nodes") if isinstance(nodes, dict) else nodes
    if isinstance(items, dict):
        items = [{"node_id": key, **(value if isinstance(value, dict) else {})}
                 for key, value in items.items()]
    print("◆ 集群节点")
    if not items:
        print("  （无节点数据）")
    for node in (items or [])[:64]:
        if not isinstance(node, dict):
            continue
        print("  %-20s %-10s %-18s %s" % (
            node.get("node_id") or node.get("id") or "—",
            node.get("role") or "—",
            node.get("hostname") or "—",
            node.get("address") or node.get("host") or "—",
        ))
    return 0


def cmd_queue(api: ApiClient, args: argparse.Namespace) -> int:
    queue = api.get(API_PATHS["cluster_queue"])
    _emit(queue, args.json)
    if args.json:
        return 0
    if not isinstance(queue, dict):
        print(queue)
        return 0
    print("◆ 请求队列")
    print(_line("队列", "%s / %s" % (queue.get("queue_size", "—"), queue.get("max_size", "—"))))
    print(_line("调度策略", queue.get("strategy")))
    items = queue.get("tasks") or queue.get("queue") or queue.get("items") or []
    if not items:
        print("  （空闲）")
    for task in items[:64]:
        if not isinstance(task, dict):
            continue
        print("  %-24s %-14s %-12s %s" % (
            task.get("task_id") or task.get("id") or "—",
            task.get("type") or task.get("stage") or "—",
            task.get("status") or task.get("state") or "—",
            task.get("node_id") or task.get("worker") or "—",
        ))
    return 0


def cmd_device(api: ApiClient, args: argparse.Namespace) -> int:
    profile = api.get(API_PATHS["device_profile"])
    _emit(profile, args.json)
    if args.json:
        return 0
    if not isinstance(profile, dict):
        print(profile)
        return 0
    os_info = profile.get("os") or {}
    os_text = (f"{os_info.get('system', '')} {os_info.get('release', '')}".strip()
               if isinstance(os_info, dict) else str(os_info))
    cpu = profile.get("cpu") or {}
    ram = profile.get("ram") or profile.get("memory") or {}
    disk = profile.get("disk") or {}
    print("◆ 本机设备画像（后端所在机器）")
    print(_line("操作系统", os_text))
    print(_line("主机名", profile.get("hostname")))
    print(_line("CPU", cpu.get("model") or cpu.get("brand")))
    print(_line("核心", "物理 %s / 逻辑 %s" % (
        cpu.get("physical_cores", "—"), cpu.get("logical_cores", "—"))))
    print(_line("内存", "总量 %s GB / 可用 %s GB" % (
        ram.get("total_gb", "—"), ram.get("available_gb", "—"))))
    if disk:
        print(_line("磁盘", "剩余 %s GB / 总 %s GB" % (
            disk.get("free_gb", "—"), disk.get("total_gb", "—"))))
    print(_line("档位", "%s (%s) 评分 %s" % (
        profile.get("tier_label", "—"), profile.get("tier", "—"),
        profile.get("score_total", "—"))))
    for gpu in (profile.get("gpus") or []):
        if isinstance(gpu, dict):
            print("  GPU: %s  %s  CUDA=%s  显存=%s GB" % (
                gpu.get("name") or "—", gpu.get("gpu_type") or "—",
                "支持" if gpu.get("cuda_available") else "不支持",
                gpu.get("vram_total_gb", "—")))
    for item in (profile.get("recommendations") or [])[:5]:
        print("  建议: %s" % item)
    for item in (profile.get("warnings") or [])[:5]:
        print("  警告: %s" % item)
    return 0


def cmd_logs(api: ApiClient, args: argparse.Namespace) -> int:
    payload = api.get(API_PATHS["cluster_log_aggregate"], with_log_token=True)
    _emit(payload, args.json)
    if args.json:
        return 0
    lines = (payload or {}).get("lines") if isinstance(payload, dict) else payload
    if isinstance(lines, dict):
        flat: List[str] = []
        for name, value in lines.items():
            if isinstance(value, list):
                flat.extend(f"{name}: {item}" for item in value)
            else:
                flat.append(f"{name}: {value}")
        lines = flat
    if isinstance(lines, str):
        lines = lines.splitlines()
    print("◆ 聚合日志（末尾 %d 行）" % min(len(lines or []), 120))
    for line in (lines or [])[-120:]:
        print(sanitize(str(line)))
    return 0


def cmd_help(api: Optional[ApiClient], args: argparse.Namespace) -> int:
    print("QLH 单命令模式（只读）——交互界面请用 qlh / koakuma")
    print()
    for name, desc in READ_ONLY_COMMANDS.items():
        print("  qlh %-8s %s" % (name, desc))
    print()
    print("  选项: --host/--port/--timeout/--log-token/--json/--fixture PATH")
    print("  说明: 单命令模式**不启动后端**；写操作请在交互界面或 API 侧完成。")
    return 0


COMMAND_HANDLERS: Dict[str, Callable[[ApiClient, argparse.Namespace], int]] = {
    "status": cmd_status,
    "models": cmd_models,
    "nodes": cmd_nodes,
    "queue": cmd_queue,
    "device": cmd_device,
    "logs": cmd_logs,
}


def _distributed_mode(api: ApiClient, status: Any) -> bool:
    config = _safe(api, API_PATHS["distributed_config"])
    if isinstance(config, dict) and isinstance(config.get("enabled"), bool):
        return config["enabled"]
    return isinstance(status, dict) and status.get("run_mode") == "distributed"


def _safe(api: ApiClient, path: str) -> Any:
    """容错读取：失败返回 {_error: ...}，让主命令仍能打印其余部分。"""
    try:
        return api.get(path)
    except ApiError as exc:
        return {"_error": str(exc)}


# ---------------------------------------------------------------- fixture 回放


def smoke_from_fixture(path: str) -> int:
    """离线回放 SSE fixture（原 tui_chat_screen.smoke_from_fixture，已随归档迁入）。"""
    raw = open(path, "rb").read()
    events = SSEDecoder().feed(raw)
    payloads = [p for p in (decode_json_event(e) for e in events) if p]
    print("  事件数: %d   有效 payload: %d" % (len(events), len(payloads)))

    tokens = [p["token"] for p in payloads if isinstance(p.get("token"), str)]
    text = sanitize("".join(tokens))
    print("  拼接文本长度: %d" % len(text))

    done = next((p for p in payloads if p.get("done")), None)
    if done:
        print("  format_metrics : %s" % format_metrics(
            done.get("metrics"), history_committed=done.get("history_committed")))
        resp = done.get("response") or ""
        print("  done.response 长度: %d" % len(resp))

    cancelled = next((p for p in payloads if p.get("cancelled")), None)
    if cancelled:
        print("  cancelled.partial 长度: %d" % len(cancelled.get("partial") or ""))

    print("  ANSI 残留: %s" % ("有" if _ANSI_RE.search(text) else "无"))
    print("  折行首行: %s" % (wrap_display(text, 40)[:1] or ["(空)"]))
    print("  parse_session_line 可用: %s" % callable(parse_session_line))
    return 0


# ---------------------------------------------------------------- 入口


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlh <command>",
        description="QLH 单命令模式（只读；交互界面请用 qlh / koakuma）",
        add_help=False,
    )
    parser.add_argument("command", nargs="?", default="help")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--log-token", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fixture", default="")
    parser.add_argument("-h", "--help", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.fixture:
        return smoke_from_fixture(args.fixture)
    command = (args.command or "help").strip().lstrip("/").lower()
    if args.help and command == "help":
        return cmd_help(None, args)

    if command in LEGACY_COMMANDS and command not in READ_ONLY_COMMANDS:
        print("[提示] `%s` 属旧交互 TUI 的命令，已随归档移除（_to_delete/）。" % command)
        print("       交互请用 `qlh` / `koakuma`；只读查询见 `qlh help`。")
        return 2
    if command not in READ_ONLY_COMMANDS:
        print("[错误] 未知命令: %s" % command)
        print("       可用只读命令：%s" % " ".join(sorted(READ_ONLY_COMMANDS)))
        return 2
    if command == "help":
        return cmd_help(None, args)

    api = ApiClient(host=args.host, port=args.port, timeout=args.timeout,
                    log_token=args.log_token)
    try:
        api.get("/health")
    except ApiError as exc:
        print("后端未在运行（%s）。" % api.base_url)
        print("  单命令模式不自动启动后端；请先运行 `qlh`（交互外壳会拉起后端）。")
        print("  详情: %s" % exc)
        return 1
    handler = COMMAND_HANDLERS[command]
    try:
        return handler(api, args)
    except ApiError as exc:
        print("[错误] %s" % exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - 单命令不应崩栈
        print("[错误] 命令失败: %s" % exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
