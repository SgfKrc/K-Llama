"""Koakuma TUI —— **Textual 外壳**（主仓统一交互入口）。

为什么换掉自绘标准库 TUI（2026-09-17 实测结论）：

* 自绘 ANSI splash 在真实 conhost 下**不可见**——像素字依赖 ``▀``(U+2580) 半块字符 +
  24 位真彩色，默认字体/色深下整块 logo 渲染为空白；且帧序列用 ``\\n`` 换行（VT 模式下
  不回第 0 列）导致逐行错位。这不是一处笔误，而是跨终端自绘的结构性风险。
* Textual 自己处理终端适配（VT 检测、字体无关布局、Rich 渲染、鼠标/滚动/焦点），
  并在 Windows Terminal 与传统 conhost 上都可用。

启动体验（2026-09-17 用户要求）：

* **标题页与启动条合体**：LOGO 下方依次显示快速流式副标题、跑马灯启动条和状态行，
  副标题为 ``Lightweight Edge Distributed Inference System``，不再"先纯文本等待、再进 TUI"两段式；
* 状态文案统一以 **「少女祈祷中：」** 开头（避免与标题 ``Koakuma`` 重复）；
* 后端冷启动（``BackendSupervisor.ensure_ready``）在启动屏的 worker 线程里执行，
  阶段文本实时反映到启动条下方。

主界面版式（2026-09-17 用户反馈后重做）：

* **导航移到左侧竖栏**（原先顶部 Tab）：形成"左窄右宽"版式，分栏按**黄金比例**
  0.382 : 0.618 分割（``#nav`` = 38%，``#content`` = 62%，即 ``1fr``）；
* 侧栏带 ``min-width``/``max-width`` 兜底，窄终端（80 列）与宽终端都不会失衡；
* **每页统一排版规范**：``.page-title``（页名）→ ``.page-hint``（一句话说明 + 数据源）
  → 内容面板（表格/日志/键值），错误与空态样式一致，不再"裸放一个表"。

边界：

* Textual 是**主仓 TUI 的依赖**（``requirements-tui.txt``；Edge 同样安装，见
  ``requirements-edge.txt``）；协议层 ``src/tui_api.py`` 仍是纯标准库，
  因此**非 UI 路径**（单命令、CI）不需要装 Textual。

用法（由 ``qlh`` 调用）::

    python src/tui_textual.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Button,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tui_api import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    ApiClient,
    ApiError,
    activate_session,
    cancel_generation,
    clear_backend_history,
    clear_queue,
    create_session,
    delete_session,
    cancel_queue_task,
    delete_log_file,
    download_log_file,
    iter_chat_payloads,
    list_log_files,
    list_sessions,
    load_local_model,
    create_model_intent,
    cancel_model_intent,
    connect_model_intent,
    set_model_mode,
    clear_spare_master,
    designate_spare_master,
    get_spare_master,
    logs_nodes_summary,
    master_health,
    read_log_file,
    reset_master_identity,
    spare_master_logs,
    conversation_sync_status,
    create_auth_user,
    db_health,
    delete_auth_user,
    delete_turn,
    get_conversation,
    list_local_gguf,
    list_model_registry,
    list_models_available,
    list_auth_users,
    list_models_downloadable,
    patch_auth_user,
    session_info,
    storage_health,
    transfer_logs,
    transfer_master,
    pause_queue,
    rename_session,
    resume_queue,
    unload_model,
    auth_capability,
    auth_login,
    auth_logout,
    auth_me,
    auth_totp_provision,
    auth_totp_verify,
)
from tui_shared import (
    API_PATHS, COMMAND_SPECS, MODEL_INTENT_TERMINAL, format_metrics,
    format_model_intent, model_intent_ready,
)  # noqa: E402

LOGO = (
    "  ██╗  ██╗ ██████╗  █████╗ ██╗  ██╗██╗   ██╗███╗   ███╗ █████╗ \n"
    "  ██║ ██╔╝██╔═══██╗██╔══██╗██║ ██╔╝██║   ██║████╗ ████║██╔══██╗\n"
    "  █████╔╝ ██║   ██║███████║█████╔╝ ██║   ██║██╔████╔██║███████║\n"
    "  ██╔═██╗ ██║   ██║██╔══██║██╔═██╗ ██║   ██║██║╚██╔╝██║██╔══██║\n"
    "  ██║  ██╗╚██████╔╝██║  ██║██║  ██╗╚██████╔╝██║ ╚═╝ ██║██║  ██║\n"
    "  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝\n"
)

#: 启动行前缀（用户 2026-09-17 指定：不用 "Koakuma:"，避免与标题重复）
SPLASH_PREFIX = "少女祈祷中："
SPLASH_SUBTITLE = "Lightweight Edge Distributed Inference System"
BAR_WIDTH = 30
BAR_MARQUEE = 9
BAR_BLOCK = "█"
BAR_EMPTY = "░"
SUBTITLE_TICK_SECONDS = 0.025
SUBTITLE_CHARS_PER_TICK = 3

# The backend's distributed read projections can legitimately wait for a
# remote node.  Keep the fast default for health/core reads, but don't turn a
# slow cluster projection into a false "unreachable" state after five seconds.
PAGE_READ_TIMEOUT = 15.0

# ★ 2026-10-23（`docs/已知问题记录.md` #113-④）：``/api/models/intent/catalog`` 是
#   唯一会触发主节点 GGUF 解析的重接口（冷启动 1.9–4.2 s，缓存未命中历史值
#   12.8–13.8 s），而 ``PAGE_READ_TIMEOUT`` 只有 15 s。它不能再跟着 5 秒核心刷新
#   走：一旦超时，服务端任务仍在跑而 TUI 已经重发 ⇒ 请求堆积、抢 GIL，把单请求
#   从小几秒拖回 12 s（正反馈）。现在 catalog 只在「首次刷新 / 显式刷新（``r``、
#   模型写操作收尾）/ 进入「模型」屏且数据已陈旧 / 距上次满 ``CATALOG_REFRESH_SECONDS``
#   秒的兜底」时拉取，其余轮次复用上一次结果，语义与字段不变。
CATALOG_REFRESH_SECONDS = 60.0

#: 侧栏导航页表：(key, 页名, 一句话说明)
PAGES: List[Tuple[str, str, str]] = [
    ("chat", "聊天", "SSE 流式对话 · /help 查看命令"),
    ("status", "状态", "运行概览 · /health /status /models"),
    ("models", "模型", "选模 · L 自动供给并准备 · U 卸载 · C 取消选模"),
    ("cluster", "分布式", "配置 · 容量 · 层段 · /cluster/*"),
    ("nodes", "节点", "成员 · 入群 · 邀请 · 连接 · /cluster/nodes"),
    ("queue", "队列", "FIFO · P 暂停/恢复 · C 清空排队"),
    ("logs", "日志", "筛选 · 统计 · 导出 · /logs/*"),
    ("device", "设备", "画像 · 自动配置 · GPU · /device/*"),
    ("settings", "设置", "会话参数与依赖边界"),
    ("api", "调试", "未接线功能的 API 兜底 · OpenAPI"),
]

CSS = """
Screen { background: $surface; }

/* ---------------------------------------------------------------- 启动屏 */
#splash-logo { color: #8fa8c4; text-align: center; padding: 1 0 0 0; }
#splash-subtitle { color: $text-muted; text-align: center; padding: 0 0 0 0; }
#splash-bar { color: #6b8aa8; text-align: center; padding: 1 0 0 0; }
#splash-status { color: $text; text-align: center; padding: 0 0 1 0; }
#splash-hint { text-align: center; color: $text-disabled; }

/* ------------------------------------------------- 主界面骨架（左窄右宽） */
#topbar { height: 1; padding: 0 1; color: $text-muted; background: $panel; }
#body { height: 1fr; }

/* 黄金分割：38% : 62%（≈0.382 : 0.618），并给窄/宽终端兜底 */
#sidebar {
    width: 38%;          /* 黄金分割窄侧（0.382 : 0.618） */
    min-width: 18;       /* 窄终端兜底：侧栏不被压到不可读 */
    height: 1fr;
    background: $panel;
    border-right: solid $primary 25%;
}
#nav { height: 1fr; padding: 1 0; }
#nav-summary {
    height: auto;
    padding: 0 1 1 1;
    border-top: solid $primary 20%;
    color: $text-muted;
}
#nav ListItem { padding: 0 1; }
#nav ListItem Label { width: 1fr; }
#nav > ListItem.--highlight { background: $primary 35%; text-style: bold; }

#content { width: 1fr; height: 1fr; }

/* ------------------------------------------------- 每页统一排版规范 */
.page { height: 1fr; padding: 1 2; }
.page-title { height: 1; color: $accent; text-style: bold; }
.page-hint { height: 1; color: $text-muted; margin-bottom: 1; }
.panel { height: 1fr; }
.scroll-panel { height: 1fr; }
.empty { color: $text-disabled; }
.error { color: $error; }

/* ---------------------------------------------------------------- 内容 */
#models-table, #resources-table, #nodes-table, #queue-table,
#status-table, #logs-log { height: 1fr; }
#model-controls { height: 3; }
#model-controls Button { min-width: 12; width: 1fr; }
#intent-pane { height: auto; max-height: 12; overflow-y: auto; }
#models-pane { height: auto; }
#status-pane, #queue-pane, #logs-pane { height: auto; color: $text-muted; }
#ha-pane { height: auto; min-height: 7; border: round $primary 20%; padding: 0 1; margin-bottom: 1; }
#device-pane, #settings-pane { height: auto; }
#gpu-table { height: auto; max-height: 14; }

/* ---------------------------------------------------------- 调试兜底 */
#api-table { height: 1fr; }
#api-detail { height: auto; min-height: 3; color: $text-muted; padding: 0 1; }
#api-path, #api-params, #api-body { height: 3; margin: 0 0 1 0; }
#api-result { height: 10; min-height: 5; border: round $primary 20%; padding: 0 1; overflow-y: auto; }

/* ---------------------------------------------------------------- 聊天 */
/* 对话文本用 Static + 缓冲渲染：Textual 8 的 RichLog.write() 不支持 end=，
   逐 token 流式写入会抛 TypeError —— 表现为"聊天得不到回复"。 */
#chat-scroll { height: 1fr; border: round $primary 20%; padding: 0 1; }
#chat-log { height: auto; }
#chat-status { padding: 0 1; height: auto; min-height: 1; color: $text-muted; }
#chat-input { dock: bottom; }
"""


def _fmt_bytes(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def kv(label: str, value: Any) -> str:
    """键值行：标签定宽 12 列，值缺省显示 em dash（各页排版统一）。"""
    text = "—" if value in (None, "") else str(value)
    return f"  {label:<12} {text}"


def _join(value: Any, sep: str = ", ") -> str:
    if isinstance(value, (list, tuple)):
        return sep.join(str(item) for item in value) or "—"
    return "—" if value in (None, "") else str(value)


def _log_lines(source: Any, node_id: str) -> List[str]:
    """把 ``{node_id, logs: [...]}`` 形状的日志段摊平成 ``[node] 行``。"""
    if not isinstance(source, dict):
        return []
    name = source.get("node_id") or node_id
    lines = source.get("logs") or source.get("lines") or []
    if isinstance(lines, str):
        lines = lines.splitlines()
    return [f"[{name}] {line}" for line in lines]


class SplashScreen(Screen):
    """启动屏：LOGO + **启动条** + 状态行（三段一体，见模块头注释）。

    ``wait_for_backend=True`` 时**不自动进入**主界面——由 ``KoakumaApp`` 在后端就绪后
    调用 ``show_main()``；否则短暂展示后自动进入。
    """

    BINDINGS = [Binding("escape,enter,space", "finish", "进入", show=False)]

    def __init__(self, *, status: str = "准备启动", wait_for_backend: bool = False) -> None:
        super().__init__()
        self.splash_status = status
        self.wait_for_backend = bool(wait_for_backend)
        self.bar_pos = 0
        self.subtitle_pos = 0

    # ------------------------------------------------------------ 渲染

    def compose(self) -> ComposeResult:
        yield Static(LOGO, id="splash-logo")
        yield Static(self.subtitle_text(), id="splash-subtitle")
        yield Static(self.bar_text(), id="splash-bar")
        yield Static(self.status_line(), id="splash-status")
        yield Static("q / ctrl+c 退出 · 任意键进入", id="splash-hint")

    def on_mount(self) -> None:
        self.set_interval(0.09, self.tick_bar)
        self.set_interval(SUBTITLE_TICK_SECONDS, self.tick_subtitle)
        if not self.wait_for_backend:
            self.set_timer(0.6, self.action_finish)

    def subtitle_text(self) -> str:
        return SPLASH_SUBTITLE[:self.subtitle_pos]

    def tick_subtitle(self) -> None:
        if self.subtitle_pos >= len(SPLASH_SUBTITLE):
            return
        self.subtitle_pos = min(
            len(SPLASH_SUBTITLE),
            self.subtitle_pos + SUBTITLE_CHARS_PER_TICK,
        )
        try:
            self.query_one("#splash-subtitle", Static).update(self.subtitle_text())
        except Exception:  # noqa: BLE001 - 启动屏可能尚未挂载
            pass

    def bar_text(self) -> str:
        cells = [BAR_EMPTY] * BAR_WIDTH
        for offset in range(BAR_MARQUEE):
            cells[(self.bar_pos + offset) % BAR_WIDTH] = BAR_BLOCK
        return "".join(cells)

    def status_line(self) -> str:
        return f"{SPLASH_PREFIX}{self.splash_status}"

    def tick_bar(self) -> None:
        self.bar_pos = (self.bar_pos + 1) % BAR_WIDTH
        try:
            self.query_one("#splash-bar", Static).update(self.bar_text())
        except Exception:  # noqa: BLE001 - 启动屏可能已被替换
            pass

    def set_status(self, text: str) -> None:
        """更新启动条下方的阶段文本（由后端启动 worker 的进度驱动）。"""
        self.splash_status = text
        try:
            self.query_one("#splash-status", Static).update(self.status_line())
        except Exception:  # noqa: BLE001
            pass

    def action_finish(self) -> None:
        self.app.show_main()


class ConfirmScreen(ModalScreen[bool]):
    """写操作前的模态确认。

    项目偏好：**涉及删除/破坏的操作必须先列出将影响的内容**（dry-run 精神），
    所以 ``body`` 必须写清"将要删/改什么"，不能只问一句"确定吗"。
    """

    BINDINGS = [
        Binding("y,enter", "confirm", "确认", show=False),
        Binding("n,escape", "cancel", "取消", show=False),
    ]

    CSS = """
    ConfirmScreen { align: center middle; }
    #confirm-box {
        width: 70; max-width: 92%; height: auto;
        border: thick $warning; background: $surface; padding: 1 2;
    }
    #confirm-title { height: auto; text-style: bold; color: $warning; }
    #confirm-body { height: auto; margin: 1 0; }
    #confirm-keys { height: auto; color: $text-muted; }
    """

    def __init__(self, title: str, body: str, *, confirm_label: str = "确认") -> None:
        super().__init__()
        self.confirm_title = title
        self.confirm_body = body
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(self.confirm_title, id="confirm-title")
            yield Static(self.confirm_body, id="confirm-body")
            yield Static(f"[b]y[/] / [b]{self.confirm_label}[/] 执行    [b]n[/] / Esc 取消",
                         id="confirm-keys")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ActionFormScreen(ModalScreen[Dict[str, str] | None]):
    """小型表单弹窗：把高频 API 操作变成可发现的终端交互。"""

    BINDINGS = [
        Binding("escape", "cancel", "取消", show=False),
    ]

    CSS = """
    ActionFormScreen { align: center middle; }
    #action-form-box {
        width: 82; max-width: 94%; height: auto; max-height: 90%;
        border: thick $primary; background: $surface; padding: 1 2;
    }
    #action-form-title { height: auto; text-style: bold; color: $accent; }
    #action-form-body { height: auto; margin: 1 0; color: $text-muted; }
    .form-label { height: 1; color: $text-muted; }
    .form-input { margin-bottom: 1; }
    #action-form-buttons { height: 3; align: right middle; }
    #action-form-buttons Button { margin-left: 1; }
    """

    def __init__(self, title: str, body: str,
                 fields: List[Tuple[str, str, str]],
                 *, confirm_label: str = "执行") -> None:
        super().__init__()
        self.form_title = title
        self.form_body = body
        self.fields = fields
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="action-form-box"):
            yield Static(self.form_title, id="action-form-title")
            yield Static(self.form_body, id="action-form-body")
            for field_id, label, default in self.fields:
                yield Label(label, classes="form-label")
                yield Input(
                    value=default,
                    id=f"form-{field_id}",
                    classes="form-input",
                    password=field_id in {"otp_code", "password"},
                )
            with Horizontal(id="action-form-buttons"):
                yield Button(self.confirm_label, variant="primary", id="form-submit")
                yield Button("取消", variant="default", id="form-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form-submit":
            self.action_submit()
        elif event.button.id == "form-cancel":
            self.action_cancel()

    def action_submit(self) -> None:
        values = {
            field_id: self.query_one(f"#form-{field_id}", Input).value
            for field_id, _label, _default in self.fields
        }
        self.dismiss(values)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChatPane(Vertical):
    """聊天页：SSE 流式输出（等价旧 ChatScreen 的事件处理）。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chat_buffer = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-scroll"):
            yield Static("", id="chat-log", markup=True)
        yield Static("就绪。输入消息并回车发送；/help 查看命令", id="chat-status")
        yield Input(placeholder="输入消息…（/help 查看命令）", id="chat-input")

    # ------------------------------------------------------------ 缓冲与渲染

    def write_line(self, text: str) -> None:
        """追加一整行（命令输出 / 用户输入）。"""
        self.chat_buffer += ("\n" if self.chat_buffer else "") + text
        self.render_chat()

    def append_chunk(self, piece: str) -> None:
        """流式追加 token（不换行）；整体重渲染由 Static 承担。"""
        self.chat_buffer += piece
        self.render_chat()

    def replace_transcript(self, text: str) -> None:
        self.chat_buffer = text
        self.render_chat()

    def clear_chat(self) -> None:
        self.chat_buffer = ""
        self.render_chat()

    def on_mount(self) -> None:
        # ★ 挂载完成前到达的流式片段可能没渲染成（compose 尚未 yield 出 #chat-log，
        # 例如真后端的首包比首帧快）⇒ 这里补画一次，保证早期内容不丢。
        self.render_chat()

    def render_chat(self) -> None:
        try:
            self.query_one("#chat-log", Static).update(self.chat_buffer)
        except Exception:  # noqa: BLE001 - compose 尚未 yield 出 #chat-log（同 #chat-scroll 的道理）
            return
        try:
            self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)
        except Exception:  # noqa: BLE001 - 挂载早期可能尚无滚动容器
            pass

    def set_status(self, text: str) -> None:
        self.query_one("#chat-status", Static).update(text or "完成")

    # ------------------------------------------------------------ 发送

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        if not text:
            return
        event.input.value = ""
        if text in {"/help", "/?"}:
            self.write_line(self.help_markup())
            return
        if text == "/quit":
            self.app.exit()
            return
        if text == "/clear":
            self.clear_chat()
            return
        if text.startswith("/route "):
            value = text.split(" ", 1)[1].strip()
            mapping = {"auto": "auto", "local": "local_only",
                       "distributed": "distributed_preferred", "required": "distributed_required"}
            if value in mapping:
                self.app.routing_preference = mapping[value]
                self.query_one("#chat-status", Static).update(f"路由偏好 → {mapping[value]}")
            else:
                self.query_one("#chat-status", Static).update(
                    "[red]用法: /route auto|local|distributed|required")
            return
        if text.startswith("/thinking "):
            value = text.split(" ", 1)[1].strip().lower()
            self.app.show_thinking = value in {"on", "1", "true", "yes"}
            self.query_one("#chat-status", Static).update(
                f"thinking 展示 → {'on' if self.app.show_thinking else 'off'}")
            return
        if text.startswith("/reasoning "):
            value = text.split(" ", 1)[1].strip().lower()
            if value in {"auto", "default", "none"}:
                self.app.enable_thinking = None
            elif value in {"on", "1", "true", "yes"}:
                self.app.enable_thinking = True
            elif value in {"off", "0", "false", "no"}:
                self.app.enable_thinking = False
            else:
                self.query_one("#chat-status", Static).update(
                    "[red]用法: /reasoning on|off|auto")
                return
            state = {None: "auto", True: "on", False: "off"}[self.app.enable_thinking]
            self.query_one("#chat-status", Static).update(f"reasoning → {state}")
            return
        if text == "/cancel":
            generation = getattr(self.app, "generation_id", None)
            if generation:
                try:
                    cancel_generation(self.app.api, generation)
                    self.query_one("#chat-status", Static).update("已请求取消")
                except Exception as exc:  # noqa: BLE001
                    self.query_one("#chat-status", Static).update(f"[red]取消失败: {exc}")
            else:
                self.query_one("#chat-status", Static).update("当前没有正在生成的请求")
            return
        if text.startswith("/model"):
            self.cmd_model(text)
            return
        if text.startswith("/queue"):
            self.cmd_queue(text)
            return
        if text.startswith("/logs"):
            self.cmd_logs(text)
            return
        if text.startswith("/ha"):
            self.cmd_ha(text)
            return
        if text.startswith("/assets"):
            self.cmd_assets(text)
            return
        if text.startswith("/history"):
            self.cmd_history(text)
            return
        if text.startswith("/login"):
            self.cmd_login(text)
            return
        if text.strip() == "/logout":
            self.cmd_logout()
            return
        if text.strip() == "/whoami":
            self.cmd_whoami()
            return
        if text.startswith("/totp"):
            self.cmd_totp(text)
            return
        if text.startswith("/users"):
            self.cmd_users(text)
            return
        if text.startswith("/storage"):
            self.cmd_storage(text)
            return
        if text == "/sessions":
            self.cmd_sessions()
            return
        if text.startswith("/new"):
            self.cmd_new(text)
            return
        if text.startswith("/resume"):
            self.cmd_resume(text)
            return
        if text.startswith("/rename"):
            self.cmd_rename(text)
            return
        if text == "/delete-session":
            self.cmd_delete_session()
            return
        if text == "/reset":
            self.cmd_reset()
            return
        if text.startswith("/"):
            self.query_one("#chat-status", Static).update(f"[red]未知命令: {text}")
            return

        gate = self.app.main_screen.chat_model_gate() if self.app.main_screen else ""
        if gate:
            event.input.value = event.value
            self.set_status(escape(gate))
            return
        self.write_line(f"[b $accent]你[/] {text}")
        self.write_line("[dim]assistant[/] ")
        self.stream_reply(text)

    # ------------------------------------------------------------ 命令（模型/队列/会话）

    def help_markup(self) -> str:
        """由 ``COMMAND_SPECS`` 生成帮助（与实现同源，不会"写着却不能用"）。"""
        lines = ["[b]可用命令[/]"]
        for spec in COMMAND_SPECS:
            usage = spec["name"] + (f" {spec['args']}" if spec["args"] else "")
            lines.append(f"  [b]{usage}[/]  {spec['desc']}")
        lines.append("  [dim]写操作都会先弹确认框；模型/队列也可在对应屏用按键操作[/]")
        return "\n".join(lines)

    def status_line(self, text: str) -> None:
        self.query_one("#chat-status", Static).update(text)

    def cmd_model(self, text: str) -> None:
        parts = text.split()
        screen = self.app.main_screen
        if screen is None:
            return
        if len(parts) == 2 and parts[1] == "status":
            self.write_line(escape(screen.model_intent_text()))
        elif len(parts) == 2 and parts[1] == "cancel":
            screen.action_cancel_model_intent()
        elif len(parts) == 2 and parts[1] == "connect":
            screen.connect_intent()
        elif len(parts) == 3 and parts[1] == "mode" and parts[2] in {"local", "distributed"}:
            screen.confirm_model_mode(parts[2])
        elif len(parts) == 2 and parts[1] == "unload":
            screen.confirm_model_unload()
        elif len(parts) >= 3 and parts[1] == "load":
            info = screen.model_rows.get(parts[2]) or {}
            engine, quant = screen.model_load_args(info)
            screen.confirm_model_load(parts[2], parts[3] if len(parts) > 3 else engine,
                                      parts[4] if len(parts) > 4 else quant)
        else:
            self.status_line("用法: /model load <id> [engine] [quant] | unload | cancel | status | connect | mode local|distributed")

    def model_call(self, kind: str, model_id: str, engine: str, quant: str) -> None:
        if self.app.main_screen:
            self.app.main_screen.run_model_action(kind, model_id, engine, quant)

    def cmd_login(self, text: str) -> None:
        """★ 2026-09-19（G-⑤）：登录。`/login <username> <password> [totp_code]`。

        ⚠️ **凭据只在本进程内存中短暂持有**（密码用完即弃）；登录态 token 写入
        `app.api.auth_token`，**不落盘**。已绑定 Auth App 的账户必须附 6 位验证码。
        """
        parts = text.split(maxsplit=3)
        if len(parts) < 3:
            self.status_line("用法: /login <username> <password> [totp_code]")
            return
        username, password = parts[1], parts[2]
        totp_code = parts[3] if len(parts) > 3 else None
        self.auth_call("login", username, password, totp_code)

    def cmd_logout(self) -> None:
        """注销（服务端吊销登录态，并清空本地 token）。"""
        self.auth_call("logout")

    def cmd_whoami(self) -> None:
        """显示当前主体与认证能力（是否强制登录 / 是否首次引导）。"""
        self.auth_call("whoami")

    def cmd_totp(self, text: str) -> None:
        """Auth App：`provision` 生成密钥与 otpauth URI；`verify <code>` 校验一次。"""
        parts = text.split()
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub == "provision":
            self.auth_call("totp-provision")
            return
        if sub == "verify" and len(parts) == 3:
            self.auth_call("totp-verify", parts[2])
            return
        self.status_line("用法: /totp provision | verify <code>")

    def cmd_users(self, text: str) -> None:
        """账户管理（需 admin）：list / add / role / disable / enable / passwd / del。"""
        parts = text.split()
        sub = parts[1].lower() if len(parts) > 1 else "list"
        if sub == "list":
            self.auth_call("users-list")
            return
        if sub == "add" and len(parts) >= 4:
            role = parts[4] if len(parts) > 4 else "viewer"
            self.auth_call("user-add", parts[2], parts[3], role)
            return
        if sub == "role" and len(parts) == 4:
            self.auth_call("user-role", parts[2], parts[3])
            return
        if sub in {"disable", "enable"} and len(parts) == 3:
            self.auth_call("user-disable" if sub == "disable" else "user-enable", parts[2])
            return
        if sub == "passwd" and len(parts) == 4:
            self.auth_call("user-passwd", parts[2], parts[3])
            return
        if sub == "del" and len(parts) == 3:
            target = parts[2]
            self.app.confirm(
                f"删除账户 {target}",
                "将删除该账户及其 Auth App 绑定与全部登录态，**不可撤销**。",
                lambda: self.auth_call("user-delete", target),
                confirm_label="删除")
            return
        self.status_line(
            "用法: /users list | add <name> <pass> [role] | role <name> <role>"
            " | disable|enable <name> | passwd <name> <pass> | del <name>")

    @work(thread=True, exclusive=True, group="authctl")
    def auth_call(self, action: str, a: str = "", b: str = "", c: str = "") -> None:
        app = self.app
        self.app.call_from_thread(self.status_line, f"认证操作 {action} …")
        try:
            if action == "login":
                result = auth_login(app.api, a, b, totp_code=c or None)
                app.api.auth_token = str(result.get("token") or "")
                text = (f"[green]已登录[/] {result.get('username')}"
                        f"（role={result.get('role')}）")
            elif action == "logout":
                result = auth_logout(app.api)
                app.api.auth_token = ""
                text = f"[green]已注销[/]（revoked={result.get('revoked')}）"
            elif action == "whoami":
                me = auth_me(app.api)
                cap = auth_capability(app.api)
                text = (f"[green]当前主体[/] {me.get('username')}"
                        f"（role={me.get('role')}）\n"
                        f"[dim]认证：required={cap.get('required')} "
                        f"available={cap.get('available')} "
                        f"bootstrap_open={cap.get('bootstrap_open')} "
                        f"users={cap.get('user_count')}[/]")
            elif action == "totp-provision":
                result = auth_totp_provision(app.api)
                text = ("[green]Auth App 已绑定[/]\n"
                        f"密钥: {result.get('secret')}\n"
                        f"URI : {result.get('otpauth_uri')}\n"
                        f"[dim]算法 {result.get('algorithm')} / {result.get('digits')} 位 / "
                        f"{result.get('period')}s。⚠️ 重新 provision 会使旧条目失效。[/]")
            elif action == "totp-verify":
                result = auth_totp_verify(app.api, a)
                ok = bool(result.get("verified"))
                text = (f"[{'green' if ok else 'red'}]TOTP 校验 "
                        f"{'通过' if ok else '未通过'}[/]")
            elif action == "users-list":
                result = list_auth_users(app.api)
                rows = result.get("users") or []
                lines = [f"  {u.get('username'):<16} {u.get('role'):<9} "
                         f"{'disabled' if u.get('disabled') else 'active':<9} "
                         f"totp={'yes' if u.get('totp_bound') else 'no'}"
                         for u in rows]
                text = "[green]账户[/]\n" + ("\n".join(lines) if lines else "  （无）")
            elif action == "user-add":
                result = create_auth_user(app.api, a, b, role=c or "viewer")
                text = (f"[green]已创建[/] {result.get('username')}"
                        f"（role={result.get('role')}）")
            elif action == "user-role":
                patch_auth_user(app.api, a, role=b)
                text = f"[green]已改角色[/] {a} → {b}"
            elif action in {"user-disable", "user-enable"}:
                disabled = action == "user-disable"
                patch_auth_user(app.api, a, disabled=disabled)
                text = f"[green]{'已禁用' if disabled else '已启用'}[/] {a}"
            elif action == "user-passwd":
                patch_auth_user(app.api, a, password=b)
                text = f"[green]已重置口令[/] {a}"
            elif action == "user-delete":
                delete_auth_user(app.api, a)
                text = f"[green]已删除账户[/] {a}"
            else:
                raise ValueError(f"未知认证动作: {action}")
        except (ApiError, ValueError) as exc:
            text = f"[red]认证 {action} 失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text.splitlines()[0])

    def cmd_history(self, text: str) -> None:
        """★ 2026-09-19 补缺口 F：会话细粒度。

        ``/history [<session_id>] [limit]`` 查看对话历史（默认当前会话）/
        ``sync-status`` 本地持久化状态 / ``info <session_id>`` 会话元数据 /
        ``drop-turn <session_id> <turn_index>`` 删单轮（**需确认**，删 user+assistant 两条）。
        """
        parts = text.split()
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub == "sync-status":
            self.history_call("sync-status")
            return
        if sub == "info" and len(parts) == 3:
            self.history_call("info", parts[2])
            return
        if sub == "drop-turn" and len(parts) == 4:
            sid, turn = parts[2], parts[3]
            self.app.confirm(
                f"删除第 {turn} 轮对话",
                "将同时删除该轮的 **user + assistant 两条消息**，不可撤销。",
                lambda: self.history_call("drop-turn", sid, turn),
                confirm_label="删除")
            return
        if sub in {"info", "drop-turn"}:
            self.status_line(
                "用法: /history [<session_id>] [limit] | sync-status"
                " | info <session_id> | drop-turn <session_id> <turn_index>")
            return
        # 默认：查看历史；可带 session_id 与 limit
        sid = parts[1] if len(parts) > 1 else ""
        limit = parts[2] if len(parts) > 2 else ""
        self.history_call("show", sid, limit)

    @work(thread=True, exclusive=True, group="history")
    def history_call(self, action: str, value: str = "", extra: str = "") -> None:
        app = self.app
        self.app.call_from_thread(self.status_line, f"会话历史 {action} …")
        try:
            if action == "sync-status":
                result = conversation_sync_status(app.api)
            elif action == "info":
                result = session_info(app.api, value)
            elif action == "drop-turn":
                result = delete_turn(app.api, value, int(extra))
            elif action == "show":
                sid = value or (app.session_id or "default")
                limit = int(extra) if extra.isdigit() else 200
                result = get_conversation(app.api, sid, limit)
            else:
                raise ValueError(f"未知会话动作: {action}")
            body = json.dumps(result, ensure_ascii=False, indent=2)[:4000]
            label = f"{action} {value}".strip()
            text = f"[green]会话历史 {label}[/]\n{body}"
        except (ApiError, ValueError) as exc:
            text = f"[red]会话历史 {action} 失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text.splitlines()[0])

    def cmd_assets(self, text: str) -> None:
        """★ 2026-09-19 补缺口 C：模型资产浏览（只读）。

        ``available`` 可选模型配置 + 可用引擎 / ``registry`` 已注册实验模型 /
        ``downloadable`` 可下载清单 / ``gguf`` 本地 GGUF 文件。
        """
        parts = text.split()
        sub = parts[1].lower() if len(parts) > 1 else "available"
        if sub not in {"available", "registry", "downloadable", "gguf"}:
            self.status_line(
                "用法: /assets available | registry | downloadable | gguf")
            return
        self.assets_call(sub)

    @work(thread=True, exclusive=True, group="assets")
    def assets_call(self, which: str) -> None:
        app = self.app
        self.app.call_from_thread(self.status_line, f"资产查询 {which} …")
        try:
            if which == "available":
                result = list_models_available(app.api)
            elif which == "registry":
                result = list_model_registry(app.api)
            elif which == "downloadable":
                result = list_models_downloadable(app.api)
            elif which == "gguf":
                result = list_local_gguf(app.api)
            else:
                raise ValueError(f"未知资产查询: {which}")
            body = json.dumps(result, ensure_ascii=False, indent=2)[:4000]
            text = f"[green]资产 {which}[/]\n{body}"
        except (ApiError, ValueError) as exc:
            text = f"[red]资产 {which} 失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text.splitlines()[0])

    def cmd_storage(self, text: str) -> None:
        """★ 2026-09-19 补缺口 D：存储与数据库健康（只读）。"""
        self.storage_call()

    @work(thread=True, exclusive=True, group="storage")
    def storage_call(self) -> None:
        app = self.app
        self.app.call_from_thread(self.status_line, "存储健康查询 …")
        try:
            result = {
                "db": db_health(app.api),
                "storage": storage_health(app.api),
            }
            body = json.dumps(result, ensure_ascii=False, indent=2)[:4000]
            text = f"[green]存储与数据库健康[/]\n{body}"
        except (ApiError, ValueError) as exc:
            text = f"[red]存储健康查询失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text.splitlines()[0])

    def cmd_ha(self, text: str) -> None:
        """★ 2026-09-19 补缺口 E：集群高可用（备用主节点 / 主节点转让 / 身份重置）。

        ⚠️ **分级**：``health`` / ``transfer-logs`` / ``spare`` / ``spare-logs`` 为**只读**；
        ``designate`` / ``clear-spare`` / ``transfer`` / ``reset-identity`` 为**写操作**，
        一律先经 ``self.app.confirm`` 二次确认，且**文案点明后果**。
        """
        parts = text.split()
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub in {"health", "transfer-logs", "spare", "spare-logs"}:
            self.ha_call(sub)
            return
        if sub == "designate" and len(parts) == 3:
            node = parts[2]
            self.app.confirm(
                f"指定备用主节点 {node}",
                "变更集群高可用配置：该节点将成为主节点宕机时的接管候选。"
                "要求集群节点数 >= 2 且目标在线。",
                lambda: self.ha_call("designate", node),
                confirm_label="指定")
            return
        if sub == "clear-spare":
            self.app.confirm(
                "清除备用主节点指定",
                "将取消当前的备用主节点配置。",
                lambda: self.ha_call("clear-spare"),
                confirm_label="清除")
            return
        if sub == "transfer" and len(parts) == 3:
            node = parts[2]
            self.app.confirm(
                f"⚠️ 转让主节点身份给 {node}",
                "**高危操作**：主节点身份将转让给该从节点，"
                "**双方都需要重启服务**才能生效（原主转从、新主转主）。",
                lambda: self.ha_call("transfer", node),
                confirm_label="转让")
            return
        if sub == "reset-identity":
            self.app.confirm(
                "⚠️ 重置主节点身份",
                "**高危且不可撤销**：将替换主节点数据库中的 MAC 记录并绑定当前物理 MAC。"
                "仅用于更换机器/网卡后。请确认你确实要这样做。",
                lambda: self.ha_call("reset-identity"),
                confirm_label="重置")
            return
        self.status_line(
            "用法: /ha health | transfer-logs | spare | spare-logs | designate <node>"
            " | clear-spare | transfer <node> | reset-identity")

    @work(thread=True, exclusive=True, group="hactl")
    def ha_call(self, action: str, value: str = "") -> None:
        app = self.app
        self.app.call_from_thread(self.status_line, f"高可用操作 {action} …")
        try:
            if action == "health":
                result = master_health(app.api)
            elif action == "transfer-logs":
                result = transfer_logs(app.api)
            elif action == "spare":
                result = get_spare_master(app.api)
            elif action == "spare-logs":
                result = spare_master_logs(app.api)
            elif action == "designate":
                result = designate_spare_master(app.api, value)
            elif action == "clear-spare":
                result = clear_spare_master(app.api)
            elif action == "transfer":
                result = transfer_master(app.api, value)
            elif action == "reset-identity":
                result = reset_master_identity(app.api)
            else:
                raise ValueError(f"未知高可用动作: {action}")
            body = json.dumps(result, ensure_ascii=False, indent=2)[:4000]
            text = f"[green]高可用 {action}[/] → {value}\n{body}" if value else \
                f"[green]高可用 {action}[/]\n{body}"
            if action == "transfer":
                text += "\n[yellow]提示：转让后需重启双方服务才生效[/]"
        except (ApiError, ValueError) as exc:
            text = f"[red]高可用 {action} 失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text.splitlines()[0])

    def cmd_logs(self, text: str) -> None:
        """★ 2026-09-19 补缺口 B：日志细粒度（文件列表 / 下载 / 查看 / 删除 / 节点汇总）。

        此前 TUI 只有 ``/logs/recent``（筛选）、``/logs/stats``、``/logs/export``（打包导出）
        与整体 ``DELETE /logs``；**单个文件的浏览/下载/删除**缺失。
        """
        parts = text.split()
        usage = ("用法: /logs list | download <file> | read <file>"
                 " | delete <file> | nodes")
        if len(parts) == 2 and parts[1].lower() in {"list", "nodes"}:
            self.logs_call(parts[1].lower())
            return
        if len(parts) == 3 and parts[1].lower() in {"download", "read"}:
            self.logs_call(parts[1].lower(), parts[2])
            return
        if len(parts) == 3 and parts[1].lower() == "delete":
            filename = parts[2]
            self.app.confirm(
                f"删除日志文件 {filename}",
                "将删除后端该日志文件，**不可撤销**（整体清理请用日志页的 X）。",
                lambda: self.logs_call("delete", filename),
                confirm_label="删除")
            return
        self.status_line(usage)

    @work(thread=True, exclusive=True, group="logfiles")
    def logs_call(self, action: str, value: str = "") -> None:
        app = self.app
        self.app.call_from_thread(self.status_line, f"日志操作 {action} …")
        try:
            if action == "list":
                result = list_log_files(app.api)
            elif action == "nodes":
                result = logs_nodes_summary(app.api)
            elif action == "read":
                result = read_log_file(app.api, value)
            elif action == "download":
                target = Path("logs") / Path(value).name
                saved = download_log_file(app.api, value, target)
                self.app.call_from_thread(self.write_line, f"[green]已下载[/] → {saved}")
                self.app.call_from_thread(self.status_line, f"日志 {value} 已下载")
                return
            elif action == "delete":
                result = delete_log_file(app.api, value)
            else:
                raise ValueError(f"未知日志动作: {action}")
            body = json.dumps(result, ensure_ascii=False, indent=2)[:4000]
            text = f"[green]日志 {action}[/] → {value}\n{body}" if value else \
                f"[green]日志 {action}[/]\n{body}"
        except (ApiError, ValueError) as exc:
            text = f"[red]日志 {action} 失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text.splitlines()[0])

    def cmd_queue(self, text: str) -> None:
        parts = text.split()
        usage = ("用法: /queue pause | resume | clear"
                 " | cancel <task_id>")
        if len(parts) == 2 and parts[1].lower() in {"pause", "resume"}:
            self.queue_call(parts[1].lower())
            return
        if len(parts) == 3 and parts[1].lower() == "cancel":
            self.queue_call("cancel", parts[2])
            return
        if len(parts) == 2 and parts[1].lower() == "clear":
            self.app.confirm(
                "清空排队任务",
                "将清空后端**排队中**的任务（执行中的不受影响）。此操作不可撤销。",
                lambda: self.queue_call("clear"),
                confirm_label="清空")
            return
        self.status_line(usage)

    @work(thread=True, exclusive=True, group="queuectl")
    def queue_call(self, action: str, value: str = "") -> None:
        app = self.app
        self.app.call_from_thread(self.status_line, f"队列操作 {action} …")
        try:
            if action == "pause":
                pause_queue(app.api)
            elif action == "resume":
                resume_queue(app.api)
            elif action == "clear":
                clear_queue(app.api)
            elif action == "cancel":
                result = cancel_queue_task(app.api, value)
                if not result.get("success"):
                    raise ValueError(str(result.get("message") or "任务不存在或已完成"))
            else:
                raise ValueError(f"未知队列动作: {action}")
            text = f"[green]队列 {action} 完成[/]" + (f" → {value}" if value else "")
        except (ApiError, ValueError) as exc:
            text = f"[red]队列 {action} 失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text)

    def cmd_sessions(self) -> None:
        self.session_call("list", "")

    def cmd_new(self, text: str) -> None:
        self.session_call("new", text[len("/new"):].strip())

    def cmd_resume(self, text: str) -> None:
        session_id = text[len("/resume"):].strip()
        if not session_id:
            self.status_line("用法: /resume <session_id>（先 /sessions 查看）")
            return
        self.session_call("resume", session_id)

    def cmd_rename(self, text: str) -> None:
        title = text[len("/rename"):].strip()
        if not title:
            self.status_line("用法: /rename <新标题>")
            return
        self.session_call("rename", title)

    def cmd_delete_session(self) -> None:
        session_id = getattr(self.app, "session_id", None)
        if not session_id:
            self.status_line("当前没有会话（先 /new 或 /resume）")
            return
        self.app.confirm(
            "删除会话",
            f"将删除会话 [b]{session_id}[/] **及其全部对话消息**（数据库 + 内存）。\n"
            "此操作不可撤销。",
            lambda: self.session_call("delete", session_id),
            confirm_label="删除")

    def cmd_reset(self) -> None:
        self.app.confirm(
            "清空后端会话历史",
            "将清空后端当前会话的对话历史与 KV 缓存（本地显示一并清空）。\n"
            "会话本身保留；此操作不可撤销。",
            lambda: self.session_call("reset", ""),
            confirm_label="清空")

    def render_history(self, messages: Any) -> str:
        """把后端会话历史渲染为对话文本（兼容 role/sender 与 content/message 两种命名）。"""
        if not isinstance(messages, list):
            return "[dim]（该会话没有历史消息）[/]"
        lines = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("sender") or "?")
            label = {"user": "[b $accent]你[/]",
                     "assistant": "[dim]assistant[/]"}.get(role, role)
            content = message.get("content") or message.get("message") or ""
            lines.append(f"{label} {content}")
        return "\n".join(lines) or "[dim]（该会话没有历史消息）[/]"

    @work(thread=True, exclusive=True, group="sessions")
    def session_call(self, action: str, value: str) -> None:
        """会话管理（新建/恢复/重命名/删除/清空后端历史）——全部在 worker 线程里跑。"""
        app = self.app
        clear_local = False
        self.app.call_from_thread(self.status_line, f"会话操作 {action} …")
        try:
            if action == "list":
                items = list_sessions(app.api)
                if not items:
                    self.app.call_from_thread(
                        self.status_line, "（后端没有会话；用 /new 创建一个）")
                    return
                lines = ["[b]最近会话[/]（用 /resume <id> 恢复）"]
                for item in items[:10]:
                    sid = item.get("session_id") or item.get("id") or "—"
                    title = item.get("title") or "(未命名)"
                    count = item.get("message_count")
                    lines.append(f"  [b]{sid}[/]  {title}"
                                 + (f" · {count} 条" if count is not None else ""))
                self.app.call_from_thread(self.write_line, "\n".join(lines))
                self.app.call_from_thread(self.status_line, f"共 {len(items)} 个会话")
                return
            if action == "new":
                result = create_session(app.api, value or None)
                session_id = str(result.get("session_id") or result.get("id") or "") or None
                app.session_id = session_id
                text = f"[green]已新建会话[/] {session_id or ''}".strip()
                clear_local = True
            elif action == "resume":
                result = activate_session(app.api, value)
                app.session_id = value
                messages = result.get("messages") or result.get("history") or []
                self.app.call_from_thread(self.replace_transcript, self.render_history(messages))
                text = f"[green]已恢复会话[/] {value}（{len(messages)} 条消息）"
            elif action == "rename":
                if not app.session_id:
                    raise ValueError("当前没有会话")
                rename_session(app.api, app.session_id, value)
                text = f"[green]已重命名[/] → {value}"
            elif action == "delete":
                delete_session(app.api, value)
                if app.session_id == value:
                    app.session_id = None
                text = f"[green]已删除会话[/] {value}"
                clear_local = True
            elif action == "reset":
                clear_backend_history(app.api)
                text = "[green]后端会话历史已清空[/]"
                clear_local = True
            else:
                raise ValueError(f"未知会话动作: {action}")
        except (ApiError, ValueError) as exc:
            text = f"[red]会话 {action} 失败[/]：{exc}"
        if clear_local:
            self.app.call_from_thread(self.clear_chat)
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text)

    def stream_reply(self, message: str) -> Any:
        """在发送事件中捕获角色与端点，线程启动后不重新解释本次请求。"""
        app = self.app
        screen = getattr(app, "main_screen", None)
        api = app.api
        context = (api, str(getattr(api, "base_url", "")), screen,
                   screen.chat_model_context() if screen else None)
        options = {
            "session_id": getattr(app, "session_id", None),
            "generation_id": getattr(app, "generation_id", None),
            "routing_preference": screen.chat_routing_preference() if screen else app.routing_preference,
            "execution_mode": getattr(app, "execution_mode", "auto"),
            "show_thinking": app.show_thinking,
            "enable_thinking": app.enable_thinking,
            "streaming_mode": screen.chat_streaming_mode() if screen else None,
        }
        return self._stream_reply(message, context, options)

    def chat_context_current(self, context: Tuple[Any, ...]) -> bool:
        """仅在原端点、模型与角色上下文仍有效时接受回包；在 UI 线程调用。"""
        api, endpoint, screen, model_context = context
        if (not self.is_attached or self.app.api is not api
                or str(getattr(api, "base_url", "")) != endpoint
                or getattr(self.app, "main_screen", None) is not screen):
            return False
        return screen is None or (screen.chat_model_context() == model_context and not screen.chat_model_gate())

    @work(thread=True, exclusive=True, group="chat")
    def _stream_reply(self, message: str, context: Tuple[Any, ...], options: Dict[str, Any]) -> None:
        """在后台消费 SSE；回包和收尾先在主线程校验原请求上下文。"""
        app = self.app
        acc: List[str] = []
        status = "…"
        error_text = ""

        def apply_payload(payload: Dict[str, Any]) -> bool:
            nonlocal status, error_text
            if not self.chat_context_current(context):
                return False
            if payload.get("start"):
                if payload.get("generation_id"):
                    app.generation_id = payload["generation_id"]
                if payload.get("session_id"):
                    app.session_id = payload["session_id"]
            elif isinstance(payload.get("token"), str) and payload["token"]:
                acc.append(payload["token"])
                self.append_chunk(payload["token"])
            elif isinstance(payload.get("thinking"), str) and payload["thinking"]:
                if options["show_thinking"]:
                    self.append_chunk(f"[dim italic]{payload['thinking']}[/]")
            elif payload.get("done"):
                if payload.get("error"):
                    error_text = str(payload["error"])
                final = payload.get("response")
                if isinstance(final, str) and final and "".join(acc) != final:
                    if options["streaming_mode"] == "full" and not acc:
                        self.append_chunk(final)
                    else:
                        self.replace_transcript(final)
                if payload.get("session_id"):
                    app.session_id = payload["session_id"]
                if not error_text:
                    status = format_metrics(
                        payload.get("metrics") or None,
                        history_committed=payload.get("history_committed"),
                    )
            elif payload.get("cancelled"):
                status = "已被取消"
            elif payload.get("error"):
                error_text = str(payload["error"])
            return True

        def finish() -> None:
            if not self.chat_context_current(context):
                return
            app.generation_id = None
            if error_text:
                first = error_text.strip().splitlines()[0]
                self.append_chunk(f"[red]✗ {error_text}[/]")
                self.set_status(f"[red]后端错误: {first}[/]")
            else:
                self.set_status(status)

        try:
            if not app.call_from_thread(self.chat_context_current, context):
                return
            for payload in iter_chat_payloads(context[0], message, **options):
                if not app.call_from_thread(apply_payload, payload):
                    return
        except Exception as exc:  # noqa: BLE001 - 网络异常不应让界面崩溃
            error_text = f"{type(exc).__name__}: {exc}"
        finally:
            app.call_from_thread(finish)

    # ------------------------------------------------------------ UI 更新（主线程）
    # write_line / append_chunk / replace_transcript / set_status 见上方"缓冲与渲染"段。


class MainScreen(Screen):
    """主界面：左侧导航（黄金比例分栏）+ 右侧内容区（9 个功能屏 + 调试兜底）。"""

    BINDINGS = [
        Binding("r", "refresh", "刷新"),
        Binding("l", "load_model", "加载模型"),
        Binding("u", "unload_model", "卸载模型"),
        Binding("d", "model_download", "下载模型"),
        Binding("f", "model_preflight", "模型预检"),
        Binding("i", "model_register", "登记模型"),
        Binding("v", "model_search", "搜索模型"),
        Binding("p", "queue_toggle_pause", "暂停/恢复"),
        Binding("c", "queue_clear", "清空排队"),
        Binding("t", "cluster_toggle", "分布式开关"),
        Binding("m", "cluster_max_nodes", "最大节点"),
        Binding("j", "cluster_connect", "连接主节点"),
        Binding("b", "cluster_join_request", "生成入群请求"),
        Binding("k", "cluster_join_consume", "消费入群授权"),
        Binding("o", "cluster_join_grant", "签发入群授权"),
        Binding("g", "device_auto_config", "设备配置"),
        Binding("h", "device_select_gpu", "选择 GPU"),
        Binding("e", "logs_export", "导出日志"),
        Binding("w", "settings_write", "写入设置"),
        Binding("a", "api_refresh", "更新端点"),
        Binding("x", "api_execute", "执行端点"),
        Binding("]", "next_page", "下一屏"),
        Binding("[", "prev_page", "上一屏"),
        Binding("q", "quit_app", "退出"),
        Binding("ctrl+c", "quit_app", "退出", show=False),
    ]

    #: 每屏可用键提示（让"这屏能做什么"可见；与 BINDINGS 保持一致）
    PAGE_KEYS = {
        "chat": "/help 看命令",
        "models": "L 选模 · U 卸载 · C 取消选模/下载 · J 发现/重试连接",
        "cluster": "T 分布式 · M 最大节点 · R 刷新容量",
        "nodes": "I 邀请 · J 连接 · B 请求码 · O 签发授权 · K 消费授权 · X 注销",
        "queue": "P 暂停/恢复 · S 策略 · C 清空排队",
        "logs": "F 筛选 · S 统计 · E 导出 · X 清理",
        "device": "G 自动配置 · H 选择 GPU",
        "settings": "W 写入用户设置",
        "api": "A 更新端点 · X 执行选中端点",
    }

    def __init__(self) -> None:
        super().__init__()
        self.page_index = 0
        self.health_text = "…"
        self.model_text = "…"
        # ★ 2026-10-09（稳定性 #73-②）：模型加载中的**权威标记**。此前只有状态栏
        #   （`set_busy`）知道"正在加载"，而「模型」面板仍按 `/api/status` 渲染（后者
        #   要等加载完成才更新）⇒ 加载期间同时出现「正在加载 qwen3-5-2b（5-20 秒）…」
        #   与「模型 未加载」两个互相矛盾的状态（用户截图实证）。
        self._model_loading_id = ""
        self.model_mode: Optional[str] = None
        self.model_intent: Dict[str, Any] = {}
        self.model_binding: Dict[str, Any] = {}
        self.model_selection_id = ""
        self._selection_epoch = 0
        self._model_action_inflight = False
        self._model_error = ""
        self._model_error_kind = ""
        self._intent_poll_error = ""
        self._intent_restore_enabled = True
        self._intent_unloaded_operation_id = ""
        self._local_model_ready: Optional[bool] = None
        self._chat_model_gated = False
        self._last_core: Optional[Tuple[Any, ...]] = None
        self.log_line_count = 0
        #: ``/models`` 原始项（模型屏写操作需要 engine/quant/可用性）
        self.model_rows: Dict[str, Dict[str, Any]] = {}
        #: ``/cluster/queue`` 最近一次结果（队列屏写操作需要 paused/strategy/深度）
        self.queue_state: Dict[str, Any] = {}
        #: 最近一次写操作的结果文本（操作没有回显窗口，故常驻侧栏摘要）
        self.op_status = ""
        self.cuda_available: Optional[bool] = None
        self.model_aux: Dict[str, Any] = {}
        self.cluster_aux: Dict[str, Any] = {}
        self.node_aux: Dict[str, Any] = {}
        self.log_filters: Dict[str, Any] = {}
        self.log_stats: Dict[str, Any] = {}
        #: OpenAPI 动态调试兜底；功能屏不把稳定业务流程退化成裸端点。
        self.api_operations: Dict[str, Dict[str, Any]] = {}
        self.api_selected_key = ""
        #: Refresh state is deliberately kept in the screen controller.  A
        #: screen switch must never fan out into a new refresh storm.
        self.backend_available: Optional[bool] = None
        self.runtime_ready: Optional[bool] = None
        self._core_inflight = False
        self._core_refresh_pending = False
        #: 分布式 catalog 的按需刷新状态（#113-④）：最后一次拉取时刻（单调时钟，
        #: ``None`` = 从未拉过）与结果缓存。核心刷新只在到点时拉它。
        self._catalog_fetched_at: Optional[float] = None
        self._catalog_cache: Dict[str, Any] = {}
        #: 缓存所属的后端（换端点即作废，不能让旧主节点的注册表留在屏上）。
        self._catalog_endpoint: Optional[Tuple[str, int]] = None
        self._initial_page_data_started = False
        self._pages_inflight = False
        self._device_inflight = False
        self._model_aux_inflight = False
        self._cluster_aux_inflight = False
        self._api_catalog_inflight = False
        self._refresh_state_lock = threading.Lock()

    # ------------------------------------------------------------ 版式

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="topbar")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                with ListView(id="nav"):
                    for key, label, _hint in PAGES:
                        yield ListItem(Label(f" {label}"), id=f"nav-{key}")
                yield Static(id="nav-summary")
            with ContentSwitcher(initial=f"page-{PAGES[0][0]}", id="content"):
                with Vertical(id="page-chat", classes="page"):
                    yield Static("聊天", classes="page-title")
                    yield Static(PAGES[0][2], classes="page-hint")
                    yield ChatPane(id="chat-pane")
                with Vertical(id="page-status", classes="page"):
                    yield Static("状态 · 运行概览", classes="page-title")
                    yield Static(PAGES[1][2], classes="page-hint")
                    yield Static("加载中…", id="status-pane")
                    yield DataTable(id="status-table")
                with Vertical(id="page-models", classes="page"):
                    yield Static("模型 · 注册表与当前加载", classes="page-title")
                    yield Static(PAGES[2][2], classes="page-hint")
                    with Horizontal(id="model-controls"):
                        yield Button("Local 整模", id="model-mode-local")
                        yield Button("分布式", id="model-mode-distributed")
                        yield Button("发现/重试连接", id="model-intent-connect")
                        yield Button("取消选模", id="model-intent-cancel")
                    yield Static("加载中…", id="intent-pane", markup=False)
                    yield Static("加载中…", id="models-pane")
                    yield DataTable(id="models-table")
                with Vertical(id="page-cluster", classes="page"):
                    yield Static("分布式 · 集群资源", classes="page-title")
                    yield Static(PAGES[3][2], classes="page-hint")
                    yield Static("加载中…", id="cluster-pane")
                    yield DataTable(id="resources-table")
                with Vertical(id="page-nodes", classes="page"):
                    yield Static("节点 · 成员与角色", classes="page-title")
                    yield Static(PAGES[4][2], classes="page-hint")
                    yield Static("加载中…", id="ha-pane")
                    yield Static("选择节点后可执行邀请、连接和注销。", id="nodes-pane")
                    yield DataTable(id="nodes-table")
                with Vertical(id="page-queue", classes="page"):
                    yield Static("队列 · 有界 FIFO", classes="page-title")
                    yield Static(PAGES[5][2], classes="page-hint")
                    yield Static("加载中…", id="queue-pane")
                    yield DataTable(id="queue-table")
                with Vertical(id="page-logs", classes="page"):
                    yield Static("日志 · 聚合视图", classes="page-title")
                    yield Static(PAGES[6][2], classes="page-hint")
                    yield Static("加载中…", id="logs-pane")
                    yield RichLog(id="logs-log", markup=False, wrap=True)
                with Vertical(id="page-device", classes="page"):
                    yield Static("设备 · 本机画像", classes="page-title")
                    yield Static(PAGES[7][2], classes="page-hint")
                    with VerticalScroll(classes="scroll-panel"):
                        yield Static("加载中…", id="device-pane")
                        yield DataTable(id="gpu-table")
                with Vertical(id="page-settings", classes="page"):
                    yield Static("设置 · 会话与边界", classes="page-title")
                    yield Static(PAGES[8][2], classes="page-hint")
                    with VerticalScroll(classes="scroll-panel"):
                        yield Static("", id="settings-pane")
                with Vertical(id="page-api", classes="page"):
                    yield Static("调试 · 未接线端点", classes="page-title")
                    yield Static(PAGES[9][2], classes="page-hint")
                    yield DataTable(id="api-table")
                    yield Static("选择端点后可编辑路径、查询参数和 JSON 请求体。GET 直接执行；写操作会确认。", id="api-detail")
                    yield Input(placeholder="路径，例如 /cluster/config", id="api-path")
                    yield Input(placeholder='查询参数 JSON，例如 {"limit": 20}', id="api-params")
                    yield Input(placeholder='请求体 JSON，例如 {"enabled": true}', id="api-body")
                    yield RichLog(id="api-result", markup=False, wrap=True)
        yield Footer()

    # ------------------------------------------------------------ 切屏

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """侧栏高亮即切屏（↑↓ 浏览，无需回车）。"""
        if event.item is not None and event.item.id:
            self.switch_page(event.item.id.removeprefix("nav-"))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """鼠标点击/回车同样切屏。"""
        if event.item is not None and event.item.id:
            self.switch_page(event.item.id.removeprefix("nav-"))

    def switch_page(self, key: str) -> None:
        keys = [item[0] for item in PAGES]
        if key not in keys:
            return
        current_key = PAGES[self.page_index][0]
        self.page_index = keys.index(key)
        try:
            self.query_one("#content", ContentSwitcher).current = f"page-{key}"
        except Exception:  # noqa: BLE001 - 切屏期间节点可能未挂载
            pass
        if key != current_key:
            self.refresh_page_data(key)

    def action_next_page(self) -> None:
        self.set_page((self.page_index + 1) % len(PAGES))

    def action_prev_page(self) -> None:
        self.set_page((self.page_index - 1) % len(PAGES))

    def set_page(self, index: int) -> None:
        """按序号切屏，并同步侧栏高亮（键位与鼠标两条路径保持一致）。"""
        key = PAGES[index % len(PAGES)][0]
        self.switch_page(key)
        try:
            self.query_one("#nav", ListView).index = index
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ 文本

    def about_text(self) -> str:
        return (
            "[b $accent]Koakuma[/] · QLH 分布式边缘推理 · Textual 外壳\n"
            + kv("后端地址", self.app.api.base_url)
            + "\n"
            + kv("界面", "9 个功能屏 + 调试兜底（左栏切换 / [ ] 上下屏 / r 刷新 / q 退出）")
            + "\n"
            + kv("依赖边界", "外壳需 Textual（requirements-tui.txt，Edge 同装）；")
            + "\n"
            + kv("", "协议层 src/tui_api.py 纯标准库，单命令/CI 不依赖 UI")
        )

    def settings_text(self) -> str:
        """设置页：本地会话参数（旧管理 TUI 的「设置」语义，多为启动参数固化）。"""
        api = self.app.api
        return (
            "[b $accent]会话设置[/][dim]（当前会话生效；可用启动参数固化）[/]\n"
            + kv("host", f"{api.host}（支持 Tailscale IP，如 100.x.x.x）") + "\n"
            + kv("port", api.port) + "\n"
            + kv("完整地址", api.base_url) + "\n"
            + kv("请求超时", f"{api.timeout:.0f} 秒") + "\n"
            + kv("自动刷新", f"{self.app.interval:.0f} 秒（核心状态/资源；分布式模型注册表"
                            f"按需拉取，最多 {CATALOG_REFRESH_SECONDS:.0f} 秒一次）") + "\n"
            + kv("日志 Token", "已设置" if api.log_token else "未设置（聚合日志可能需要 --log-token）") + "\n"
            + kv("路由偏好", self.app.routing_preference) + "\n"
            + kv("thinking", "on" if self.app.show_thinking else "off") + "\n"
            + kv("reasoning", {None: "auto", True: "on", False: "off"}[self.app.enable_thinking]) + "\n"
            + "\n" + self.about_text()
        )

    # ------------------------------------------------------------ 挂载

    def on_mount(self) -> None:
        self.query_one("#status-table", DataTable).add_columns("项目", "值")
        self.query_one("#models-table", DataTable).add_columns(
            "", "模型 ID", "名称", "格式", "引擎", "状态")
        self.query_one("#resources-table", DataTable).add_columns(
            "节点", "角色 / 状态", "CPU 物理/逻辑", "内存 总/可用", "GPU 数/CUDA", "显存 总/可用")
        self.query_one("#nodes-table", DataTable).add_columns(
            "节点", "角色", "类型", "状态", "地址", "RTT")
        self.query_one("#queue-table", DataTable).add_columns("位置", "请求", "执行资源", "终局")
        self.query_one("#gpu-table", DataTable).add_columns(
            "", "名称", "类型", "CUDA", "显存GB", "驱动")
        self.query_one("#api-table", DataTable).add_columns(
            "方法", "路径", "领域", "模式", "说明")
        self.query_one("#models-pane", Static).update("加载中…")
        self.query_one("#ha-pane", Static).update("加载中…")
        self.query_one("#cluster-pane", Static).update("加载中…")
        self.query_one("#settings-pane", Static).update(self.settings_text())
        self.refresh_topbar()
        # Only core state is refreshed periodically.  Expensive/optional
        # projections are loaded after the health check and then on demand.
        self.app.main_screen = self
        self.action_reload()
        self.set_interval(self.app.interval, self.action_reload)
        self.set_interval(self.app.interval, self.refresh_topbar)

    def refresh_topbar(self) -> None:
        page = PAGES[self.page_index]
        keys = self.PAGE_KEYS.get(page[0])
        hint = f"   [dim]·[/]   [dim]{keys}[/]" if keys else ""
        self.query_one("#topbar", Static).update(
            f"[dim]后端[/] {self.app.api.base_url}"
            f"   [dim]·[/]   [dim]当前[/] {page[1]}{hint}"
            f"   [dim]·[/]   [dim]r 刷新 · [ ] 上下屏 · q 退出[/]")
        self.update_sidebar(page)

    def update_sidebar(self, page: Tuple[str, str, str]) -> None:
        """左栏底部摘要：让黄金分割的窄侧不只是空导航（宽终端尤其明显）。"""
        api = self.app.api
        try:
            self.query_one("#nav-summary", Static).update(
                "[b $accent]后端[/]\n"
                f"{api.host}:{api.port}\n"
                f"[dim]健康[/] {self.health_text}\n"
                f"[dim]模型[/] {escape(self.model_text) if self.model_mode == 'distributed' else self.model_text}\n"
                f"[dim]刷新[/] {self.app.interval:.0f}s   [dim]当前[/] {page[1]}"
                + (f"\n[dim]最近[/] {self.op_status}" if self.op_status else ""))
        except Exception:  # noqa: BLE001 - 挂载期间可能尚未就绪
            pass

    # ------------------------------------------------------------ 只读数据

    def action_refresh(self) -> None:
        """刷新核心状态，并刷新当前屏的按需数据。"""
        self.action_reload(refresh_page=True)

    @work(thread=True, exclusive=True, group="main")
    def action_reload(self, refresh_page: bool = False, refresh_catalog: bool = False) -> None:
        """核心状态刷新。

        ``refresh_catalog``（#113-④）只影响分布式模式：为 ``True`` 时本轮**强制**
        重新拉 ``/models/intent/catalog``（显式刷新、模型写操作收尾），否则该重接口
        按 ``CATALOG_REFRESH_SECONDS`` 节流，未到点复用上一次结果。
        """
        with self._refresh_state_lock:
            if self._core_inflight:
                self._core_refresh_pending = True
                return
            self._core_inflight = True
        epoch = self._selection_epoch
        mode = self.model_mode
        api = self.app.api
        endpoint = (api, str(api.base_url))
        # 端点/字段以 2026-09-17 实测的后端真实结构为准：
        #   /status → model_name / model_loaded / active_model_id / engine / run_mode / node_*
        #   /models → {models: [...], active_model_id}（19 个内置模型）
        try:
            health = self.fetch_json(API_PATHS["health"], api=api)
            if "_error" in health:
                # A failed health probe is authoritative for this refresh cycle.
                # Do not spend another 15 seconds failing every optional endpoint.
                error = {"_error": health["_error"]}
                self.app.call_from_thread(
                    self.apply_data, health, error, error, error, refresh_page, None, epoch,
                    None, None, endpoint)
                return
            readiness = self.fetch_json(API_PATHS["readiness"], api=api)
            # Older remote nodes do not expose /ready. Preserve their previous
            # behavior while treating errors from the current endpoint as a
            # real runtime gate.
            if readiness.get("_status") == 404:
                readiness = {"ready": True, "status": "legacy"}
            elif "_error" in readiness:
                self.app.call_from_thread(
                    self.apply_data, health, {}, {}, {}, refresh_page, readiness, epoch,
                    None, None, endpoint)
                return
            if readiness.get("ready") is False:
                self.app.call_from_thread(
                    self.apply_data, health, {}, {}, {}, refresh_page, readiness, epoch,
                    None, None, endpoint)
                return
            status = self.fetch_json(API_PATHS["system_status"], api=api)
            if mode is None:
                config = self.fetch_json(API_PATHS["distributed_config"], api=api)
                if isinstance(config.get("enabled"), bool):
                    mode = "distributed" if config["enabled"] else "local"
                elif status.get("run_mode") in {"local", "distributed"}:
                    mode = status["run_mode"]
            intent = None
            if mode == "distributed":
                # #113-④：catalog 不再每 5 秒拉一次（见 distributed_catalog）。
                registry = self.distributed_catalog(api, force=refresh_page or refresh_catalog)
                # Current belongs to the master and can be replaced by another device,
                # including after our previously observed operation became terminal.
                intent = self.fetch_json(API_PATHS["models_intent_current"], timeout=PAGE_READ_TIMEOUT, api=api)
            elif mode == "local":
                registry = self.fetch_json(API_PATHS["models_list"], api=api)
            else:
                registry = {"_error": "无法确认模型模式，请刷新或显式选择 Local/分布式"}
            resources = self.fetch_json(API_PATHS["cluster_resources"], api=api)
            self.app.call_from_thread(
                self.apply_data, health, status, registry, resources, refresh_page,
                readiness, epoch, mode, intent, endpoint)
        finally:
            with self._refresh_state_lock:
                self._core_inflight = False
                pending = self._core_refresh_pending
                self._core_refresh_pending = False
            if pending and self.is_attached:
                self.app.call_from_thread(self.action_reload, refresh_page, refresh_catalog)

    def distributed_catalog(self, api: Any, *, force: bool) -> Dict[str, Any]:
        """分布式注册表：按需拉 ``/models/intent/catalog``，未到点复用上一次结果。

        ``#113-④``：这个接口是唯一会触发主节点 GGUF 解析的重接口，而核心刷新只有
        5 秒、``PAGE_READ_TIMEOUT`` 15 秒——超时后服务端任务仍在跑而 TUI 已重发，
        并发堆积抢 GIL，会把单请求从小几秒拖回 12 秒。因此它被移出周期性刷新：

        * ``force=True``：显式刷新（``r``）或模型写操作收尾，立刻拉；
        * 其它轮次：距上次拉取不足 ``CATALOG_REFRESH_SECONDS`` 秒就用缓存。

        成功与失败（含 15 秒超时）**都**记时，否则失败路径仍会 5 秒重发一次，
        并发堆积照旧。返回的仍是联机拉到的同一份 dict（语义与字段不变）。
        """
        now = time.monotonic()
        endpoint = (str(getattr(api, "base_url", "")), id(api))
        if endpoint != self._catalog_endpoint:
            force = True  # 首次或换了后端（换端点）：缓存作废，必须重新拉
        if (not force and self._catalog_fetched_at is not None
                and now - self._catalog_fetched_at < CATALOG_REFRESH_SECONDS):
            return self._catalog_cache or {
                "_error": "模型注册表尚未刷新，请按 r 或进入「模型」屏",
                "_code": "INTENT_CATALOG_DEFERRED"}
        registry = self.fetch_json(API_PATHS["models_intent_catalog"],
                                   timeout=PAGE_READ_TIMEOUT, api=api)
        self._catalog_fetched_at = now
        self._catalog_cache = registry
        self._catalog_endpoint = endpoint
        return registry

    def request_catalog_refresh(self) -> None:
        """进入「模型」屏时的按需补拉（#113-④）：数据仍新鲜就不重复打主节点。"""
        if self._catalog_fetched_at is None:
            return  # 首轮核心刷新正在拉
        if time.monotonic() - self._catalog_fetched_at < CATALOG_REFRESH_SECONDS:
            return
        self.action_reload(refresh_catalog=True)

    def fetch_json(self, path: str, *, timeout: Optional[float] = None,
                   api: Any = None) -> Dict[str, Any]:
        api = self.app.api if api is None else api
        try:
            normalized = path if path.startswith("/") else "/" + path
            if timeout is not None and isinstance(api, ApiClient):
                value = api.get(normalized, timeout=timeout)
            else:
                value = api.get(normalized)
            return value if isinstance(value, dict) else {"value": value}
        except ApiError as exc:
            return {"_error": str(exc), "_status": exc.status, "_code": exc.reason_code}

    def apply_data(self, health: Dict[str, Any], status: Dict[str, Any],
                   registry: Dict[str, Any], resources: Dict[str, Any],
                   refresh_page: bool = False,
                   readiness: Optional[Dict[str, Any]] = None,
                   model_epoch: Optional[int] = None,
                   model_mode: Optional[str] = None,
                   intent: Optional[Dict[str, Any]] = None,
                   model_endpoint: Optional[Tuple[Any, str]] = None) -> None:
        if not self.is_attached or not self.query("#status-pane"):
            return
        if ((model_epoch is not None and model_epoch != self._selection_epoch)
                or (model_endpoint is not None and
                    (model_endpoint[0] is not self.app.api
                     or model_endpoint[1] != str(self.app.api.base_url)))):
            self._core_refresh_pending = True
            return
        if self.model_mode is None and model_mode is not None:
            self.model_mode = model_mode
        if self.model_mode == "distributed" and not self._model_action_inflight:
            if "_error" in registry:
                self.model_binding = {"status": "disconnected", "reason": registry["_error"],
                                      "reason_code": registry.get("_code") or "INTENT_CATALOG_UNAVAILABLE"}
            elif isinstance(registry.get("binding"), dict):
                self.model_binding = registry["binding"]
                if self.model_binding.get("status") == "connected" and self._model_error_kind == "connect":
                    self._model_error = ""
            if intent is not None:
                self.apply_model_intent(intent, authoritative_current=True)
        self._last_core = (health, status, registry, readiness)
        self.backend_available = "_error" not in health
        self.runtime_ready = (
            False
            if isinstance(readiness, dict) and "_error" in readiness
            else (
                bool(readiness["ready"])
                if isinstance(readiness, dict) and "ready" in readiness
                else self.backend_available
            )
        )
        self.render_model_intent()
        if self.runtime_ready is False:
            self.refresh_topbar()
            return
        self.fill_models(registry, status)
        self.fill_resources(resources)
        self.refresh_topbar()
        if not self.backend_available:
            return
        if not self._initial_page_data_started:
            self._initial_page_data_started = True
            self.load_pages()
            self.load_device()
        if refresh_page:
            self.refresh_page_data(PAGES[self.page_index][0], force=True)

    def refresh_page_data(self, key: str, *, force: bool = False) -> None:
        """Load only the data owned by the selected screen.

        This is intentionally a dispatcher rather than a timer per screen.
        Highlight/selection events can arrive twice during a terminal redraw,
        so each worker also has an in-flight guard.
        """
        if self.backend_available is not True or self.runtime_ready is not True:
            return
        if key in {"nodes", "queue", "logs"}:
            if force or not self._pages_inflight:
                self.load_pages()
        elif key == "models":
            if force or not self.model_aux:
                self.load_model_aux()
            if self.model_mode == "distributed":
                # #113-④：注册表只在进入该屏（且已陈旧）时补拉，不再跟着核心刷新。
                self.request_catalog_refresh()
        elif key == "cluster":
            if force or not self.cluster_aux:
                self.load_cluster_aux()
        elif key == "device":
            if force or not self._device_inflight:
                self.load_device()
        elif key == "api":
            if force or not self.api_operations:
                self.load_api_catalog()

    # ------------------------------------------------------------ 状态页

    def fill_status(self, health: Dict[str, Any], status: Dict[str, Any],
                    registry: Dict[str, Any],
                    readiness: Optional[Dict[str, Any]] = None) -> None:
        pane = self.query_one("#status-pane", Static)
        table = self.query_one("#status-table", DataTable)
        table.clear()
        if "_error" in health:
            self.health_text = "[red]不可达[/]"
            self.model_text = self.model_intent_summary() if self.model_mode == "distributed" else "—"
            pane.update(f"[red]后端不可达[/]  ·  {health['_error']}")
            table.add_row("后端地址", self.app.api.base_url)
            table.add_row("连接状态", "[red]不可达[/]")
            table.add_row("提示", "确认后端在运行（qlh 会自动拉起本机后端）")
            return

        if isinstance(readiness, dict) and (
            readiness.get("ready") is False or "_error" in readiness
        ):
            self.health_text = "[green]API ok[/]"
            self.model_text = "[yellow]初始化中[/]"
            status_text = str(readiness.get("status") or "probe-unavailable")
            pane.update(f"[green]API 已响应[/]  ·  [yellow]运行时初始化中[/]  ·  {status_text}")
            table.add_row("后端地址", self.app.api.base_url)
            table.add_row("健康", str(health.get("status") or "ok"))
            table.add_row("运行时", f"[yellow]{status_text}[/]")
            components = readiness.get("components") or {}
            for name in ("local_store", "scheduler", "device_profile"):
                state = "ready" if components.get(name) else "starting"
                table.add_row(name, state)
            if readiness.get("error"):
                table.add_row("错误", str(readiness["error"]))
            return

        self.health_text = "[green]ok[/]"
        if self.model_mode == "distributed":
            self.model_text = self.model_intent_summary()
            pane.update(f"[green]后端可用[/] · 分布式 · {escape(self.model_text)}")
            table.add_row("后端地址", self.app.api.base_url)
            table.add_row("运行模式", "distributed")
            table.add_row("节点角色", str(status.get("node_role") or "—"))
            table.add_row("模型", escape(self.model_text))
            table.add_row("operation_id", str(self.model_intent.get("operation_id") or "—"))
            table.add_row("config_id", str(self.model_intent.get("config_id") or "—"))
            table.add_row("generation", str(self.model_intent.get("generation", 0)))
            table.add_row("loaded_ready", str(model_intent_ready(self.model_intent)).lower())
            table.add_row("流水线", "已准备" if model_intent_ready(self.model_intent) else "未准备")
            reason_source = (self.model_binding if self.model_binding.get("status") != "connected"
                             else self.model_intent)
            table.add_row("原因", escape(self._intent_poll_error or self._model_error or
                          f"{reason_source.get('reason_code', '')}: {reason_source.get('reason', '')}"))
            return
        loaded = (bool(status.get("model_loaded")) if self._local_model_ready is None
                  else self._local_model_ready)
        model_name = (status.get("model_name") or "—" if self._local_model_ready is None
                      else self.model_selection_id or "—")
        active = ((status.get("active_model_id") or registry.get("active_model_id"))
                  if self._local_model_ready is None else self.model_selection_id if loaded else None)
        # ★ 2026-10-09（#73-②）：`正在加载` 优先于 `/api/status`（后者要到加载完成才更新），
        #   否则加载中会同时出现「正在加载…」与「未加载」。
        if self._model_loading_id:
            self.model_text = f"[yellow]加载中[/] {self._model_loading_id}"
        else:
            self.model_text = str(active) if active else ("[yellow]未加载[/]" if not loaded else "—")
        # ★ 2026-10-09（#73-②）：这三处必须与 `model_text` / 表格**同源**。
        #   此前只改了 `model_text` 与表格，漏了这一条摘要 ⇒ 加载期间顶部仍显示
        #   「模型未加载」，与下面「加载中…」**同屏矛盾**（正是用户截图里的 ②）。
        if self._model_loading_id:
            pane.update(
                f"[green]后端可用[/]  ·  [yellow]加载中[/] {self._model_loading_id}"
                f"  ·  {self.app.api.base_url}"
            )
        elif loaded:
            pane.update(f"[green]后端可用[/]  ·  模型已加载  ·  {self.app.api.base_url}")
        else:
            pane.update(f"[green]后端可用[/]  ·  [yellow]模型未加载[/]（此时聊天会返回后端错误）"
                        f"  ·  {self.app.api.base_url}")
        table.add_row("后端地址", self.app.api.base_url)
        table.add_row("健康", str(health.get("status") or health.get("ok") or "ok"))
        table.add_row("运行模式", str(self.model_mode or status.get("run_mode") or "—"))
        table.add_row("节点角色", f"{status.get('node_role') or '—'} · {status.get('node_id') or '—'}")
        table.add_row("最大节点数", str(status.get("max_nodes", "—")))
        if self._model_loading_id:
            table.add_row("模型", f"[yellow]加载中…[/] {self._model_loading_id}")
            table.add_row("当前模型 ID", f"[yellow]加载中[/] {self._model_loading_id}")
        else:
            table.add_row("模型", f"{model_name}（{'已加载' if loaded else '未加载'}）")
            table.add_row("当前模型 ID", str(active or "—（未加载）"))
        table.add_row("引擎", str(status.get("engine") or "—（未加载）"))
        table.add_row("量化", str(status.get("current_quant") or "—"))
        prepared = status.get("pipeline_prepared") if self._local_model_ready is None else False
        table.add_row("流水线", "已准备" if prepared else "未准备")
        table.add_row("对话轮次", str(status.get("conversation_turns", "—")))
        table.add_row("可用模型数", str(len(registry.get("models") or []))
                      if "_error" not in registry else "—")
        table.add_row("路由偏好", self.app.routing_preference)
        table.add_row("刷新间隔", f"{self.app.interval:.0f} 秒")

    # ------------------------------------------------------------ 模型页

    def _models_cursor_key(self, table: DataTable) -> str:
        """记住模型表当前光标行的 row key（空表 / 坐标越界时返回空串）。"""
        try:
            if not table.row_count:
                return ""
            row_key, _column_key = table.coordinate_to_cell_key(table.cursor_coordinate)
            return str(row_key.value or "")
        except Exception:  # noqa: BLE001 - 空表或坐标越界
            return ""

    def _restore_models_cursor(self, table: DataTable, key: str) -> None:
        """把光标恢复到 row key 命中的行（**不依赖行序**）；找不到就留在原位。"""
        if not key or not table.row_count:
            return
        try:
            for index, row_key in enumerate(table.rows.keys()):
                if str(row_key.value) == key:
                    table.move_cursor(row=index)
                    return
        except Exception:  # noqa: BLE001
            pass

    def fill_models(self, registry: Dict[str, Any], status: Dict[str, Any]) -> None:
        """``/models`` → ``{models: [...], active_model_id}``（19 个内置模型）。

        此前误用 ``/models/current``——它只返回 ``{loaded, quant_type, model_id}``
        且未加载时 ``model_id`` 为 null，于是页面显示"后端未返回模型列表"。

        ★ 2026-09-19 BUG 修复：**保留光标位置**。此前 ``table.clear()`` 后重建会让光标
        跳回第 0 行，而本页会被后台刷新反复重建；用户按视觉记忆选好行再按 ``L`` 加载，
        实际加载的却是**列表第一项**（历史上第一位正是 ``qwen-1_8b``）——
        对应报障「选择了其他模型光标还在第一位，然后加载成 1.8B」。
        现在按 **row key**（而非行号）记住并恢复，行序变化也不受影响。
        """
        table = self.query_one("#models-table", DataTable)
        previous_key = self._models_cursor_key(table)
        try:
            table.clear()
            self.model_rows = {}
            if "_error" in registry:
                label = "主节点目录不可用" if self.model_mode == "distributed" else "模型目录不可用"
                table.add_row("", f"[red]{label}[/]", "", "", "", registry["_error"])
                return
            rows = registry.get("models")
            if not isinstance(rows, list) or not rows:
                table.add_row("", "[dim]（后端未返回模型列表）[/]", "", "", "", "")
                return
            if self.model_mode == "distributed":
                active = (self.model_intent.get("intent") or {}).get("model_id")
            elif self._local_model_ready is not None:
                active = self.model_selection_id if self._local_model_ready else None
            else:
                active = registry.get("active_model_id") or status.get("active_model_id")
            for item in rows[:64]:
                if not isinstance(item, dict):
                    continue
                model_id = str(item.get("model_id") or "—")
                if item.get("is_available") is False:
                    state = f"[yellow]不可用[/] {item.get('unavailable_reason') or ''}".strip()
                else:
                    state = "[green]可用[/]"
                self.model_rows[model_id] = item
                table.add_row(
                    "[green]◆[/]" if model_id == active else "",
                    model_id,
                    str(item.get("name") or "—"),
                    _join(item.get("available_formats")),
                    str(item.get("preferred_engine") or "—"),
                    state,
                    key=model_id,  # 写操作按 row key 取模型，不依赖行序
                )
        finally:
            self._restore_models_cursor(table, previous_key)

    @work(thread=True, exclusive=True, group="modelaux")
    def load_model_aux(self) -> None:
        """补齐模型屏的资产、预设和异步下载任务，不把它们退化为 JSON 调试。"""
        with self._refresh_state_lock:
            if (self._model_aux_inflight
                    or self.backend_available is not True
                    or self.runtime_ready is not True):
                return
            self._model_aux_inflight = True
        payload: Dict[str, Any] = {}
        try:
            if self.model_mode == "distributed":
                return
            for name, path in (
                ("assets", "/models/local-assets"),
                ("presets", "/models/presets"),
                ("downloads", "/models/downloads"),
            ):
                payload[name] = self.fetch_json(path, timeout=PAGE_READ_TIMEOUT)
            self.app.call_from_thread(self.fill_model_aux, payload)
        finally:
            with self._refresh_state_lock:
                self._model_aux_inflight = False

    def fill_model_aux(self, payload: Dict[str, Any]) -> None:
        if not self.is_attached or not self.query("#models-pane"):
            return
        self.model_aux = payload
        pane = self.query_one("#models-pane", Static)
        if self.model_mode == "distributed":
            pane.update("主节点注册表 · 光标仅浏览，L 确认选模 · ◆ 当前 operation 的模型\n"
                        "自动供给 → prepare · 就绪前可取消 · 层段由后端规划")
            return
        assets = payload.get("assets") or {}
        presets = payload.get("presets") or {}
        downloads = payload.get("downloads") or {}
        asset_items = []
        if isinstance(assets, dict):
            asset_items = assets.get("models") or assets.get("assets") or []
        preset_items = presets.get("presets") if isinstance(presets, dict) else []
        download_items = downloads.get("jobs") if isinstance(downloads, dict) else []
        if not isinstance(asset_items, list):
            asset_items = []
        if not isinstance(preset_items, list):
            preset_items = []
        if not isinstance(download_items, list):
            download_items = []
        errors = [str(value.get("_error")) for value in payload.values()
                  if isinstance(value, dict) and value.get("_error")]
        jobs = []
        for job in download_items[:5]:
            if isinstance(job, dict):
                jobs.append(f"{job.get('job_id') or job.get('id') or 'job'}:{job.get('status', '—')}")
        text = (
            f"资产 {len(asset_items)} · 下载预设 {len(preset_items)} · 下载任务 {len(download_items)}"
            f"  · L 加载 U 卸载 D 下载 F 预检 I 登记"
        )
        if jobs:
            text += "\n最近任务：" + "  ".join(jobs)
        if errors:
            text += "\n[yellow]部分模型辅助接口不可用：[/]" + "；".join(errors)
        pane.update(text)

    # ------------------------------------------------------------ 模型资产操作

    def open_form(self, title: str, body: str,
                  fields: List[Tuple[str, str, str]],
                  callback: Callable[[Dict[str, str]], None],
                  *, confirm_label: str = "执行") -> None:
        def _done(values: Optional[Dict[str, str]]) -> None:
            if values is not None:
                callback(values)
        self.app.push_screen(ActionFormScreen(title, body, fields,
                                              confirm_label=confirm_label), _done)

    def action_model_download(self) -> None:
        if PAGES[self.page_index][0] != "models":
            return
        self.open_form(
            "下载模型",
            "优先填写预设 ID；也可直接填写来源和目标。下载任务会进入后台队列。",
            [
                ("preset_id", "预设 ID（可选）", ""),
                ("source", "HF/ModelScope 来源或本地目录", ""),
                ("target", "目标目录（可选）", ""),
                ("model_id", "模型 ID（可选）", ""),
                ("quant", "量化（可选，如 Q4_K_M）", ""),
                ("gguf_path", "显式 GGUF 路径（可选）", ""),
                ("allow_cpu", "允许 CPU：true/false", "true"),
            ],
            self.submit_model_download,
        )

    def action_model_search(self) -> None:
        if PAGES[self.page_index][0] != "models":
            return
        self.open_form(
            "搜索模型仓库",
            "搜索结果只展示仓库元数据；下载前仍需检查格式、摘要、架构和设备预算。",
            [
                ("q", "关键词", ""),
                ("source", "来源：all/huggingface/modelscope", "all"),
                ("page", "页码", "1"),
                ("limit", "条数", "20"),
            ],
            self.submit_model_search,
            confirm_label="搜索",
        )

    def submit_model_search(self, values: Dict[str, str]) -> None:
        query = values.get("q", "").strip()
        if not query:
            self.write_status("[red]请输入搜索关键词[/]")
            return
        try:
            page = max(1, int(values.get("page", "1")))
            limit = max(1, min(50, int(values.get("limit", "20"))))
        except ValueError:
            self.write_status("[red]页码和条数必须是整数[/]")
            return
        self.run_model_search(query, values.get("source", "all").strip() or "all", page, limit)

    @work(thread=True, exclusive=True, group="modelaux")
    def run_model_search(self, query: str, source: str, page: int, limit: int) -> None:
        try:
            result = self.app.api.get(
                "/models/search", params={"q": query, "source": source, "page": page, "limit": limit})
            self.app.call_from_thread(self.show_model_search, result)
        except ApiError as exc:
            self.app.call_from_thread(self.write_status, f"[red]模型搜索失败[/]：{exc}")

    def show_model_search(self, result: Any) -> None:
        pane = self.query_one("#models-pane", Static)
        if not isinstance(result, dict):
            pane.update(str(result))
            return
        items = result.get("models") or result.get("results") or result.get("items") or []
        lines = [f"搜索结果 {len(items)} 条（V 重新搜索）"]
        for item in items[:8]:
            if isinstance(item, dict):
                lines.append(f"{item.get('id') or item.get('model_id') or item.get('name', '—')} · "
                             f"{item.get('source') or item.get('downloads') or '—'}")
        pane.update("\n".join(lines))

    def action_model_cancel_download(self) -> None:
        if PAGES[self.page_index][0] != "models":
            return
        downloads = self.model_aux.get("downloads") or {}
        items = downloads.get("jobs") if isinstance(downloads, dict) else []
        first_job = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else {}
        default_id = str(first_job.get("job_id") or first_job.get("id") or "")
        self.open_form(
            "取消模型下载",
            "只能取消仍在排队的任务；执行中的任务由后端返回当前状态。",
            [("job_id", "下载任务 ID", default_id)],
            self.submit_model_cancel_download,
            confirm_label="取消下载",
        )

    def submit_model_cancel_download(self, values: Dict[str, str]) -> None:
        job_id = values.get("job_id", "").strip()
        if not job_id:
            self.write_status("[red]必须填写下载任务 ID[/]")
            return
        path = "/models/downloads/%s" % urllib.parse.quote(job_id, safe="")
        self.app.confirm(
            "取消模型下载",
            f"将请求取消下载任务 {job_id}。已开始执行的任务可能只能返回当前状态。",
            lambda: self.run_json_operation(
                "DELETE", path, None, success="下载任务取消请求已发送",
                refresh=(self.load_model_aux,)),
            confirm_label="取消",
        )

    def submit_model_download(self, values: Dict[str, str]) -> None:
        body = {key: value.strip() for key, value in values.items() if value.strip()}
        if "allow_cpu" in body:
            body["allow_cpu"] = body["allow_cpu"].lower() not in {"0", "false", "no", "否"}
        self.app.confirm(
            "创建模型下载任务",
            json.dumps(body, ensure_ascii=False),
            lambda: self.run_json_operation(
                "POST", "/models/downloads", body,
                success="下载任务已创建", refresh=(self.load_model_aux,)),
            confirm_label="下载",
        )

    def action_model_preflight(self) -> None:
        if PAGES[self.page_index][0] == "logs":
            self.action_logs_filter()
            return
        if PAGES[self.page_index][0] != "models":
            return
        model_id = self.selected_model_id(self.query_one("#models-table", DataTable))
        if not model_id:
            self.write_status("[yellow]请先选择模型[/]")
            return
        path = "/models/local-assets/%s/preflight" % urllib.parse.quote(model_id, safe="")
        self.run_json_operation("POST", path, None, success="模型预检完成",
                                refresh=(self.load_model_aux,))

    def action_model_unregister(self) -> None:
        if PAGES[self.page_index][0] != "models":
            return
        model_id = self.selected_model_id(self.query_one("#models-table", DataTable))
        if not model_id:
            self.write_status("[yellow]请先选择模型[/]")
            return
        path = "/models/registry/%s" % urllib.parse.quote(model_id, safe="")
        self.app.confirm(
            "取消登记模型",
            f"只删除注册表项，不删除本地模型文件：{model_id}\n此操作不可撤销。",
            lambda: self.run_json_operation(
                "DELETE", path, None, success="模型注册表项已删除",
                refresh=(self.action_reload, self.load_model_aux)),
            confirm_label="删除",
        )

    def action_model_register(self) -> None:
        if PAGES[self.page_index][0] == "nodes":
            self.action_node_invite()
            return
        if PAGES[self.page_index][0] != "models":
            return
        self.open_form(
            "登记模型资产",
            "登记会写入主仓模型注册表；模型文件仍需由本地资产或下载任务提供。",
            [
                ("model_id", "模型 ID", ""),
                ("name", "显示名称", ""),
                ("model_type", "类型：gguf/safetensors/both", "gguf"),
                ("gguf_path", "GGUF 路径", ""),
                ("huggingface_id", "HuggingFace ID（可选）", ""),
                ("description", "说明（可选）", ""),
            ],
            self.submit_model_register,
        )

    def submit_model_register(self, values: Dict[str, str]) -> None:
        body = {key: value.strip() for key, value in values.items() if value.strip()}
        self.app.confirm(
            "登记模型资产",
            json.dumps(body, ensure_ascii=False),
            lambda: self.run_json_operation(
                "POST", "/models/registry", body,
                success="模型已登记", refresh=(self.action_reload, self.load_model_aux)),
            confirm_label="登记",
        )

    @work(thread=True, exclusive=True, group="modelctl")
    def run_json_operation(self, method: str, path: str, body: Any = None, *,
                           success: str = "操作完成", refresh: Tuple[Callable[[], Any], ...] = (),
                           with_log_token: bool = False) -> None:
        try:
            result = self.app.api.request(
                method, path, body=body, with_log_token=with_log_token, timeout=180.0)
            self.app.call_from_thread(self.finish_json_operation, success, result)
            for fn in refresh:
                self.app.call_from_thread(fn)
        except ApiError as exc:
            self.app.call_from_thread(self.finish_json_operation, "操作失败：" + str(exc), {})

    def finish_json_operation(self, text: str, result: Any) -> None:
        self.write_status(f"[green]{text}[/]") if not text.startswith("操作失败") else self.write_status(f"[red]{text}[/]")
        if isinstance(result, dict) and result:
            self.op_status = text

    # ------------------------------------------------------------ 分布式页

    def fill_resources(self, resources: Dict[str, Any]) -> None:
        """``/cluster/resources`` → ``{available: {local, remote}, totals}``。

        此前按顶层 ``nodes`` 解析——真实返回里没有该键，于是页面显示"无在线节点"。
        """
        table = self.query_one("#resources-table", DataTable)
        table.clear()
        if "_error" in resources:
            table.add_row("[red]不可用[/]", resources["_error"], "", "", "", "")
            return
        available = resources.get("available") or {}
        nodes: List[Dict[str, Any]] = []
        if isinstance(available.get("local"), dict):
            nodes.append(available["local"])
        if isinstance(available.get("remote"), list):
            nodes.extend(item for item in available["remote"] if isinstance(item, dict))
        if not nodes:
            table.add_row("[dim]（无在线节点）[/]", str(resources.get("scope") or "—"),
                          "", "", "", "")
            return
        totals = resources.get("totals") or {}
        physical_total = totals.get("physical_cores")
        physical_total = physical_total if physical_total is not None else "—"
        for node in nodes[:64]:
            cpu = node.get("cpu") or {}
            ram = node.get("ram") or {}
            gpu = node.get("gpu") or {}
            physical_cores = cpu.get("physical_cores")
            table.add_row(
                str(node.get("node_id") or "—"),
                f"{node.get('role') or '—'} / "
                f"{'[green]在线[/]' if node.get('available') else '[yellow]离线[/]'}",
                f"{physical_cores if physical_cores is not None else '—'} / "
                f"{cpu.get('logical_cores', '—')}",
                f"{ram.get('total_gb', '—')} / {ram.get('available_gb', '—')} GB",
                f"{gpu.get('count', '—')} / {gpu.get('cuda_count', '—')}",
                f"{gpu.get('vram_total_gb', '—')} / {gpu.get('vram_free_gb', '—')} GB",
            )
        if totals:
            table.add_row(
                "[b]合计[/]",
                f"{resources.get('available_node_count', '—')}/"
                f"{resources.get('node_count', '—')} 可用",
                f"{physical_total} / {totals.get('logical_cores', '—')}",
                f"{totals.get('ram_total_gb', '—')} / {totals.get('ram_available_gb', '—')} GB",
                f"{totals.get('gpu_count', '—')} / {totals.get('cuda_gpu_count', '—')}",
                f"{totals.get('vram_total_gb', '—')} / {totals.get('vram_free_gb', '—')} GB",
            )

    @work(thread=True, exclusive=True, group="clusteraux")
    def load_cluster_aux(self) -> None:
        with self._refresh_state_lock:
            if (self._cluster_aux_inflight
                    or self.backend_available is not True
                    or self.runtime_ready is not True):
                return
            self._cluster_aux_inflight = True
        payload: Dict[str, Any] = {}
        try:
            for name, path in (
                ("config", "/cluster/config"),
                ("role", "/cluster/my-role"),
                ("fence", API_PATHS["cluster_control_plane"]),
                ("score", API_PATHS["cluster_management_score"]),
                ("transfer_logs", API_PATHS["cluster_transfer_logs"]),
                ("distributed", "/cluster/config/distributed-inference"),
                ("capacity", "/cluster/pipeline-capacity"),
                ("reshard", "/cluster/pipeline-reshard"),
                ("layers", "/cluster/layers"),
            ):
                payload[name] = self.fetch_json(path, timeout=PAGE_READ_TIMEOUT)
            self.app.call_from_thread(self.fill_cluster_aux, payload)
        finally:
            with self._refresh_state_lock:
                self._cluster_aux_inflight = False

    def render_ha_status(self, payload: Dict[str, Any]) -> str:
        """Render the read-only HA projection without inventing missing state."""

        fence = payload.get("fence") if isinstance(payload.get("fence"), dict) else {}
        role = payload.get("role") if isinstance(payload.get("role"), dict) else {}
        auto = role.get("auto_role") if isinstance(role.get("auto_role"), dict) else {}
        role_name = str(role.get("node_role") or role.get("runtime_node_role") or "").lower()
        leader = (
            fence.get("leader_id")
            or role.get("master_node_id")
            or role.get("leader_id")
            or (role.get("node_id") if role_name == "master" else None)
            or "unknown"
        )
        term = fence.get("committed_term")
        if term is None:
            term = auto.get("term")
        epoch = fence.get("voter_set_epoch")
        certificate = str(fence.get("certificate_digest") or "none")
        if len(certificate) > 16:
            certificate = certificate[:16] + "..."

        outcome = auto.get("last_outcome") if isinstance(auto.get("last_outcome"), dict) else {}
        if fence.get("enabled") is False:
            majority = "not enabled"
        elif outcome.get("accepted") is True:
            majority = "available"
        elif auto.get("state") == "read_only" or str(outcome.get("reason")) in {
            "quorum_unavailable", "quorum_requires_witness",
        }:
            majority = "unavailable"
        elif fence.get("available") is False:
            majority = "unavailable"
        elif fence.get("read_only_reason"):
            majority = "unavailable"
        else:
            majority = "unknown (not exposed)"

        score_payload = payload.get("score") if isinstance(payload.get("score"), dict) else {}
        score_version = str(score_payload.get("algorithm_version") or "").strip()
        if not score_version:
            score_version = "not reported"

        transfer = payload.get("transfer_logs")
        logs = transfer.get("logs") if isinstance(transfer, dict) else []
        latest = logs[-1] if isinstance(logs, list) and logs and isinstance(logs[-1], dict) else {}
        handoff_reason = (
            latest.get("reason") or latest.get("result") or latest.get("event_type") or "none"
        )
        read_only = (
            fence.get("read_only_reason")
            or auto.get("reason")
            or "none"
        )
        if fence.get("_error"):
            read_only = str(fence["_error"])
            majority = "unavailable"

        return "\n".join((
            "[b $accent]HA control (read-only)[/]",
            f"leader: {leader}    term/epoch: {term if term is not None else 'unknown'}/{epoch if epoch is not None else 'unknown'}",
            f"certificate: {certificate}    majority: {majority}",
            f"score version: {score_version}",
            f"last handoff: {handoff_reason}",
            f"read-only reason: {read_only}",
        ))

    def fill_cluster_aux(self, payload: Dict[str, Any]) -> None:
        self.cluster_aux = payload
        self.query_one("#ha-pane", Static).update(self.render_ha_status(payload))
        pane = self.query_one("#cluster-pane", Static)
        config = payload.get("config") or {}
        distributed = payload.get("distributed") or {}
        role = payload.get("role") or {}
        capacity = payload.get("capacity") or {}
        if not isinstance(capacity, dict):
            capacity = {}
        enabled = distributed.get("enabled") if isinstance(distributed, dict) else None
        if enabled is None and isinstance(config, dict):
            enabled = config.get("distributed_inference_enabled")
        if self.model_mode is not None:
            enabled = self.model_mode == "distributed"
        parts = [
            f"角色 {role.get('role') or role.get('node_role') or '—'}",
            f"分布式 {'开启' if enabled else '关闭' if enabled is not None else '未知'}",
            f"容量 {capacity.get('status') or capacity.get('total_layers') or '已刷新'}",
            "T 开关 · M 最大节点 · J 连接主节点 · R 刷新",
        ]
        if isinstance(payload.get("reshard"), dict) and payload["reshard"].get("_error"):
            parts.append("[yellow]重分片数据不可用[/]")
        pane.update(" · ".join(str(part) for part in parts))

    def action_cluster_toggle(self) -> None:
        if PAGES[self.page_index][0] != "cluster":
            return
        current = (self.cluster_aux.get("distributed") or {}).get("enabled")
        if current is None and self.model_mode is None:
            self.write_status("[yellow]分布式配置尚未加载，请先刷新[/]")
            return
        mode = self.model_mode or ("distributed" if current else "local")
        self.confirm_model_mode("local" if mode == "distributed" else "distributed")

    def action_cluster_max_nodes(self) -> None:
        if PAGES[self.page_index][0] != "cluster":
            return
        current = ((self.cluster_aux.get("config") or {}).get("max_nodes")
                   if isinstance(self.cluster_aux.get("config"), dict) else "")
        self.open_form(
            "调整最大节点数",
            "只修改集群容量上限，不会预创建节点。",
            [("max_nodes", "最大节点数（1-64）", str(current or "3"))],
            self.submit_max_nodes,
        )

    def submit_max_nodes(self, values: Dict[str, str]) -> None:
        try:
            number = max(1, min(64, int(values.get("max_nodes", ""))))
        except ValueError:
            self.write_status("[red]最大节点数必须是 1-64 的整数[/]")
            return
        self.app.confirm(
            "更新最大节点数", f"将集群最大节点数设为 {number}。",
            lambda: self.run_json_operation(
                "PUT", "/cluster/config/max-nodes", {"max_nodes": number},
                success="最大节点数已更新", refresh=(self.load_cluster_aux, self.load_pages)),
            confirm_label="更新",
        )

    def action_cluster_connect(self) -> None:
        if PAGES[self.page_index][0] in {"models", "cluster", "nodes"}:
            self.connect_intent()

    def submit_cluster_connect(self, values: Dict[str, str]) -> None:
        # Kept for callers of the former form; discovery owns the destination.
        self.connect_intent()

    def action_cluster_join_request(self) -> None:
        if PAGES[self.page_index][0] != "nodes":
            return
        self.open_form(
            "生成入群请求码",
            "在待加入节点生成一次性请求码；主节点仍需通过 Auth App/TOTP 审批后签发授权。",
            [
                ("master_endpoint", "主节点地址（host:port）", ""),
                ("target_node_id", "目标节点 ID（可选）", ""),
                ("cluster_id", "集群 ID（可选）", "qlh-default"),
                ("capabilities", "能力标签（逗号分隔）", "presence,task"),
                ("request_ttl_seconds", "请求有效期（60-3600 秒）", "600"),
            ],
            self.submit_cluster_join_request,
            confirm_label="生成",
        )

    def submit_cluster_join_request(self, values: Dict[str, str]) -> None:
        endpoint = values.get("master_endpoint", "").strip()
        if not endpoint:
            self.write_status("[red]必须填写主节点地址[/]")
            return
        try:
            ttl = max(60, min(3600, int(values.get("request_ttl_seconds", "600"))))
        except ValueError:
            self.write_status("[red]请求有效期必须是整数[/]")
            return
        capabilities = [item.strip() for item in values.get("capabilities", "").split(",") if item.strip()]
        body: Dict[str, Any] = {
            "master_endpoint": endpoint,
            "cluster_id": values.get("cluster_id", "").strip(),
            "capabilities": capabilities or ["presence", "task"],
            "request_ttl_seconds": ttl,
        }
        target_node_id = values.get("target_node_id", "").strip()
        if target_node_id:
            body["target_node_id"] = target_node_id
        self.run_cluster_join_request(body)

    @work(thread=True, exclusive=True, group="nodectl")
    def run_cluster_join_request(self, body: Dict[str, Any]) -> None:
        try:
            result = self.app.api.request("POST", "/cluster/join/request", body=body)
            self.app.call_from_thread(self.show_cluster_join_request, result)
        except ApiError as exc:
            self.app.call_from_thread(self.write_status, f"[red]生成入群请求失败[/]：{exc}")

    def show_cluster_join_request(self, result: Any) -> None:
        self.node_aux["join_request"] = result
        code = result.get("request_code") if isinstance(result, dict) else None
        pane = self.query_one("#nodes-pane", Static)
        if code:
            pane.update(
                "入群请求码（交给主节点 Auth App/TOTP 审批流程）：\n"
                + str(code)
                + "\n\nK 消费主节点签发的 grant_code。")
            self.write_status("[green]入群请求码已生成[/]")
        else:
            pane.update("入群请求响应：" + json.dumps(result, ensure_ascii=False, separators=(", ", ": ")))
            self.write_status("[yellow]后端未返回 request_code[/]")

    def action_cluster_join_consume(self) -> None:
        if PAGES[self.page_index][0] != "nodes":
            return
        self.open_form(
            "消费入群授权",
            "消费一次性 grant_code；成功后本节点会切换为从节点并连接主节点。",
            [("grant_code", "一次性授权码", "")],
            self.submit_cluster_join_consume,
            confirm_label="消费授权",
        )

    def action_cluster_join_grant(self) -> None:
        if PAGES[self.page_index][0] != "nodes":
            return
        self.open_form(
            "签发入群授权",
            "主节点管理员必须输入当前 Auth App/TOTP 一次性验证码；旧布尔审批不会被接受。",
            [
                ("request_code", "目标节点请求码", ""),
                ("otp_code", "Auth App 一次性验证码", ""),
                ("ttl_seconds", "授权有效期（60-900 秒）", "300"),
            ],
            self.submit_cluster_join_grant,
            confirm_label="签发",
        )

    def submit_cluster_join_grant(self, values: Dict[str, str]) -> None:
        request_code = values.get("request_code", "").strip()
        otp_code = values.get("otp_code", "").strip()
        if not request_code:
            self.write_status("[red]必须填写 request_code[/]")
            return
        if not otp_code:
            self.write_status("[red]必须填写 Auth App 一次性验证码[/]")
            return
        try:
            ttl = max(60, min(900, int(values.get("ttl_seconds", "300"))))
        except ValueError:
            self.write_status("[red]授权有效期必须是整数[/]")
            return
        body = {
            "request_code": request_code,
            "otp_code": otp_code,
            "ttl_seconds": ttl,
        }
        self.app.confirm(
            "签发入群授权",
            f"将为请求 {request_code[:32]}... 签发一次性客户端授权；验证码不会显示在确认文本中。",
            lambda: self.run_json_operation(
                "POST", "/cluster/join/grant", body,
                success="入群授权签发请求已发送",
                refresh=(self.load_pages, self.load_cluster_aux),
            ),
            confirm_label="签发",
        )

    def submit_cluster_join_consume(self, values: Dict[str, str]) -> None:
        grant_code = values.get("grant_code", "").strip()
        if not grant_code:
            self.write_status("[red]必须填写 grant_code[/]")
            return
        self.app.confirm(
            "消费入群授权",
            "成功后当前节点将切换为从节点并连接授权中的主节点；一次性授权不可重复使用。",
            lambda: self.run_json_operation(
                "POST", "/cluster/join/consume", {"grant_code": grant_code},
                success="入群授权消费请求已发送", refresh=(self.load_pages, self.load_cluster_aux)),
            confirm_label="消费",
        )

    # ------------------------------------------------------------ 运维面（节点/队列/日志）

    @work(thread=True, exclusive=True, group="pages")
    def load_pages(self) -> None:
        """拉取节点、队列、聚合日志和日志筛选数据。"""
        with self._refresh_state_lock:
            if (self._pages_inflight
                    or self.backend_available is not True
                    or self.runtime_ready is not True):
                return
            self._pages_inflight = True
        try:
            nodes = self.fetch_json(API_PATHS["cluster_nodes"], timeout=PAGE_READ_TIMEOUT)
            queue = self.fetch_json(API_PATHS["cluster_queue"], timeout=PAGE_READ_TIMEOUT)
            logs = self.fetch_json(API_PATHS["cluster_log_aggregate"], timeout=PAGE_READ_TIMEOUT)
            recent = self.fetch_json_params(
                "/logs/recent", self.log_filters, with_log_token=True,
                timeout=PAGE_READ_TIMEOUT)
            stats = self.fetch_json_params(
                "/logs/stats", {}, with_log_token=True, timeout=PAGE_READ_TIMEOUT)
            logs["_recent"] = recent
            logs["_stats"] = stats
            self.app.call_from_thread(self.apply_pages, nodes, queue, logs)
        finally:
            with self._refresh_state_lock:
                self._pages_inflight = False

    def fetch_json_params(self, path: str, params: Dict[str, Any], *,
                          with_log_token: bool = False,
                          timeout: Optional[float] = None) -> Dict[str, Any]:
        try:
            if timeout is not None and isinstance(self.app.api, ApiClient):
                value = self.app.api.get(
                    path, params=params, with_log_token=with_log_token, timeout=timeout)
            else:
                value = self.app.api.get(path, params=params, with_log_token=with_log_token)
            return value if isinstance(value, dict) else {"value": value}
        except ApiError as exc:
            return {"_error": str(exc)}

    def apply_pages(self, nodes: Dict[str, Any], queue: Dict[str, Any],
                    logs: Dict[str, Any]) -> None:
        self.fill_nodes(nodes)
        self.fill_queue(queue)
        self.fill_logs(logs)

    def fill_nodes(self, nodes: Dict[str, Any]) -> None:
        self.node_aux = nodes
        if self.cluster_aux:
            self.query_one("#ha-pane", Static).update(self.render_ha_status(self.cluster_aux))
        table = self.query_one("#nodes-table", DataTable)
        table.clear()
        if "_error" in nodes:
            table.add_row("[red]不可用[/]", nodes["_error"], "", "", "", "")
            return
        items = nodes.get("nodes")
        if isinstance(items, dict):  # {node_id: {...}} 形状
            items = [{"node_id": key, **(value if isinstance(value, dict) else {})}
                     for key, value in items.items()]
        if not items:
            table.add_row("[dim]（无节点数据）[/]", "", "", "", "", "")
            return
        for node in items[:64]:
            if not isinstance(node, dict):
                continue
            state = str(node.get("state") or ("online" if node.get("is_available") else "—"))
            colored = {"online": "[green]在线[/]", "offline": "[red]离线[/]"}.get(state, state)
            rtt = node.get("avg_rtt_ms")
            node_id = str(node.get("node_id") or "—")
            table.add_row(
                node_id,
                str(node.get("role") or "—"),
                str(node.get("node_type") or "—"),
                colored,
                str(node.get("address") or "—"),
                f"{rtt:.0f} ms" if isinstance(rtt, (int, float)) else "—",
                key=node_id,
            )
        pane = self.query_one("#nodes-pane", Static)
        pane.update(
            f"节点 {len(items)} · 在线 {sum(1 for item in items if isinstance(item, dict) and str(item.get('state', '')).lower() == 'online')}"
            " · I 邀请 · J 连接 · B 请求码 · K 消费授权 · X 注销")

    def selected_node_id(self) -> str:
        table = self.query_one("#nodes-table", DataTable)
        if table.row_count == 0:
            return ""
        try:
            row_key, _column_key = table.coordinate_to_cell_key(table.cursor_coordinate)
            return str(row_key.value or "")
        except Exception:  # noqa: BLE001
            return ""

    def action_node_invite(self) -> None:
        if PAGES[self.page_index][0] != "nodes":
            return
        self.run_node_invite()

    @work(thread=True, exclusive=True, group="nodectl")
    def run_node_invite(self) -> None:
        try:
            invite = self.app.api.get("/cluster/invite")
        except ApiError as exc:
            self.app.call_from_thread(self.write_status, f"[red]获取邀请信息失败[/]：{exc}")
            return
        self.app.call_from_thread(self.show_node_invite, invite)

    def show_node_invite(self, invite: Any) -> None:
        self.node_aux["invite"] = invite
        self.query_one("#nodes-pane", Static).update(
            "邀请信息：" + json.dumps(invite, ensure_ascii=False, separators=(", ", ": ")))
        self.write_status("[green]邀请信息已显示[/]")

    def action_node_deregister(self) -> None:
        if PAGES[self.page_index][0] != "nodes":
            return
        node_id = self.selected_node_id()
        if not node_id or node_id == "—":
            self.write_status("[yellow]请先选择节点[/]")
            return
        path = "/cluster/nodes/%s/deregister" % urllib.parse.quote(node_id, safe="")
        self.app.confirm(
            "注销节点",
            f"将请求注销节点 {node_id}。在线节点可能被后端拒绝，操作不可撤销。",
            lambda: self.run_json_operation(
                "POST", path, None, success="节点注销请求已发送", refresh=(self.load_pages, self.load_cluster_aux)),
            confirm_label="注销",
        )

    def fill_queue(self, queue: Dict[str, Any]) -> None:
        pane = self.query_one("#queue-pane", Static)
        table = self.query_one("#queue-table", DataTable)
        table.clear()
        self.queue_state = queue if "_error" not in queue else {}
        if "_error" in queue:
            pane.update(f"[red]队列不可用[/] · {queue['_error']}")
            return
        state = "已暂停接收" if queue.get("paused") else "接受请求"
        pane.update(f"{state} · FIFO · {queue.get('queue_size', 0)}/{queue.get('capacity', '—')} · revision {queue.get('revision', '—')}")
        active = queue.get("active")
        if active:
            table.add_row("执行中", str(active.get("request_id", "")), str(active.get("execution_state", "unknown")), str(active.get("terminal") or "进行中"))
        for item in queue.get("tasks", []):
            table.add_row(str(item.get("position", "—")), str(item.get("request_id", "")), f"等待 {item.get('wait_seconds', 0):.1f}s", str(item.get("outcome") or "排队中"))

    def fill_logs(self, logs: Dict[str, Any]) -> None:
        view = self.query_one("#logs-log", RichLog)
        pane = self.query_one("#logs-pane", Static)
        view.clear()
        if "_error" in logs:
            pane.update(f"[red]后端日志不可用[/]  ·  {logs['_error']}"
                        "（聚合日志可能需要 X-QLH-Log-Token，见 qlh --log-token）")
            self.log_line_count = 0
            return
        # 真实结构：{local: {node_id, logs: [...]}, workers: [...], limit, total_workers}
        lines = _log_lines(logs.get("local"), "local")
        workers = logs.get("workers") or []
        if isinstance(workers, dict):
            worker_items = [{"node_id": key, **(value if isinstance(value, dict) else {})}
                            for key, value in workers.items()]
        else:
            worker_items = [item for item in workers if isinstance(item, dict)]
        for item in worker_items:
            lines.extend(_log_lines(item, "worker"))
        shown = lines[-200:]
        self.log_line_count = len(shown)
        recent = logs.get("_recent") or {}
        recent_count = recent.get("count") if isinstance(recent, dict) else None
        stats = logs.get("_stats") or {}
        stats_note = ""
        if isinstance(stats, dict) and not stats.get("_error"):
            stats_note = f"  · 文件 {stats.get('files_count', '—')} 个"
        filter_note = f"  · 筛选 {self.log_filters}" if self.log_filters else ""
        pane.update(f"共 {len(lines)} 行（显示末尾 {len(shown)}）"
                    f"  ·  local + {len(worker_items)} worker"
                    f"  ·  total_workers={logs.get('total_workers', 0)}"
                    f"  ·  limit={logs.get('limit', '—')}"
                    f"  · recent={recent_count if recent_count is not None else '—'}"
                    f"{stats_note}{filter_note}"
                    "\nF 筛选 · S 统计 · E 导出 · X 清理日志")
        if not shown:
            view.write("（后端未返回日志行）")
            return
        for line in shown:
            view.write(line)

    def action_logs_filter(self) -> None:
        if PAGES[self.page_index][0] != "logs":
            return
        self.open_form(
            "筛选最近日志",
            "空字段表示不筛选；筛选只影响 /logs/recent，不隐藏聚合日志。",
            [
                ("level", "最低级别（DEBUG/INFO/WARNING/ERROR）", str(self.log_filters.get("level", ""))),
                ("name", "logger 名称包含", str(self.log_filters.get("name", ""))),
                ("node_id", "节点 ID", str(self.log_filters.get("node_id", ""))),
                ("request_id", "请求 ID", str(self.log_filters.get("request_id", ""))),
                ("limit", "返回条数（1-1000）", str(self.log_filters.get("limit", "200"))),
            ],
            self.submit_log_filter,
        )

    def submit_log_filter(self, values: Dict[str, str]) -> None:
        filters = {key: value.strip() for key, value in values.items() if value.strip()}
        if "limit" in filters:
            try:
                filters["limit"] = max(1, min(1000, int(filters["limit"])))
            except ValueError:
                self.write_status("[red]日志条数必须是整数[/]")
                return
        self.log_filters = filters
        self.load_pages()
        self.write_status("[green]日志筛选已更新[/]")

    def action_logs_stats(self) -> None:
        if PAGES[self.page_index][0] != "logs":
            return
        self.run_logs_stats()

    @work(thread=True, exclusive=True, group="logctl")
    def run_logs_stats(self) -> None:
        try:
            stats = self.app.api.get("/logs/stats", with_log_token=True)
        except ApiError as exc:
            self.app.call_from_thread(self.write_status, f"[red]读取日志统计失败[/]：{exc}")
            return
        self.app.call_from_thread(self.show_logs_stats, stats)

    def show_logs_stats(self, stats: Any) -> None:
        self.log_stats = stats if isinstance(stats, dict) else {}
        self.query_one("#logs-pane", Static).update(
            "日志统计：" + json.dumps(self.log_stats, ensure_ascii=False, separators=(", ", ": "))
            + "\nF 筛选 · S 统计 · E 导出 · X 清理日志")
        self.write_status("[green]日志统计已刷新[/]")

    @work(thread=True, exclusive=True, group="logctl")
    def run_logs_export(self) -> None:
        target = Path("logs") / f"qlh-logs-export-{int(time.time())}.zip"
        try:
            saved = self.app.api.download("/logs/export", target, with_log_token=True)
            self.app.call_from_thread(self.write_status, f"[green]日志已导出[/]：{saved}")
        except ApiError as exc:
            self.app.call_from_thread(self.write_status, f"[red]日志导出失败[/]：{exc}")

    def action_logs_export(self) -> None:
        if PAGES[self.page_index][0] != "logs":
            return
        self.run_logs_export()

    def action_logs_clear(self) -> None:
        if PAGES[self.page_index][0] != "logs":
            return
        self.app.confirm(
            "清理日志文件",
            "将删除后端日志目录中的 .log 文件，内存日志缓冲不受影响。此操作不可撤销。",
            lambda: self.run_json_operation(
                "DELETE", "/logs", None, success="日志文件已清理", refresh=(self.load_pages,),
                with_log_token=True),
            confirm_label="删除",
        )

    # ------------------------------------------------------------ 设备页

    @work(thread=True, exclusive=True, group="device")
    def load_device(self) -> None:
        with self._refresh_state_lock:
            if (self._device_inflight
                    or self.backend_available is not True
                    or self.runtime_ready is not True):
                return
            self._device_inflight = True
        try:
            profile = self.fetch_json(API_PATHS["device_profile"], timeout=PAGE_READ_TIMEOUT)
            self.app.call_from_thread(self.fill_device, profile)
        finally:
            with self._refresh_state_lock:
                self._device_inflight = False

    def fill_device(self, profile: Dict[str, Any]) -> None:
        pane = self.query_one("#device-pane", Static)
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        if "_error" in profile:
            pane.update(f"[red]设备画像不可用[/]\n{profile['_error']}")
            table.add_row("", "[dim]（画像不可用，无 GPU 数据）[/]", "", "", "", "")
            return
        # 真实结构：platform.{os,os_version,hostname,machine}、cpu.model_name、disk
        platform = profile.get("platform") or {}
        os_text = " ".join(
            str(platform.get(key) or "") for key in ("os", "os_version")).strip()
        cpu = profile.get("cpu") or {}
        ram = profile.get("ram") or profile.get("memory") or {}
        disk = profile.get("disk") or {}
        lines = [
            kv("操作系统", os_text or "—"),
            kv("主机名", platform.get("hostname")),
            kv("架构", f"{platform.get('machine', '—')} · {platform.get('architecture', '—')}"
                       f" · Python {platform.get('python_version', '—')}"),
            kv("CPU", cpu.get("model_name")),
            kv("核心", f"物理 {cpu.get('physical_cores', '—')} / 逻辑 {cpu.get('logical_cores', '—')}"
                       f" · 使用率 {cpu.get('usage_percent', '—')}%"),
            kv("内存", f"总量 {ram.get('total_gb', '—')} GB / 可用 {ram.get('available_gb', '—')} GB"
                       f" · 已用 {ram.get('percent_used', '—')}%"),
        ]
        if disk:
            lines.append(kv("磁盘", f"剩余 {disk.get('free_gb', '—')} GB / "
                                    f"总 {disk.get('total_gb', '—')} GB（{disk.get('path', '—')}）"))
        lines.append(kv("档位评估", f"{profile.get('tier_label', '—')} "
                                    f"({profile.get('tier', '—')}) · 评分 {profile.get('score_total', '—')}"))
        for item in (profile.get("recommendations") or [])[:5]:
            lines.append(f"  [green]建议[/]         {item}")
        for item in (profile.get("warnings") or [])[:5]:
            lines.append(f"  [yellow]警告[/]         {item}")
        pane.update("\n".join(lines))

        gpus = profile.get("gpus") or []
        selected = profile.get("selected_gpu_index", 0)
        self.cuda_available = any(
            bool(gpu.get("cuda_available")) for gpu in gpus if isinstance(gpu, dict))
        if not gpus:
            table.add_row("", "[dim]（未检测到 GPU）[/]", "", "", "", "")
            return
        for index, gpu in enumerate(gpus):
            if not isinstance(gpu, dict):
                continue
            table.add_row(
                "[green]◆[/]" if index == selected else "",
                str(gpu.get("name") or "—"),
                str(gpu.get("gpu_type") or "—"),
                "[green]支持[/]" if gpu.get("cuda_available") else "[yellow]不支持[/]",
                str(gpu.get("vram_total_gb", "—")),
                str(gpu.get("driver_version") or "—"),
            )

    def action_device_auto_config(self) -> None:
        if PAGES[self.page_index][0] != "device":
            return
        self.app.confirm(
            "应用设备自动配置",
            "后端将根据设备画像和评分选择推理档位，并可能重建 KV 缓存。",
            lambda: self.run_json_operation(
                "POST", "/device/auto-configure", None, success="设备自动配置已应用",
                refresh=(self.load_device, self.action_reload)),
            confirm_label="应用",
        )

    def action_device_select_gpu(self) -> None:
        if PAGES[self.page_index][0] != "device":
            return
        self.open_form(
            "选择 GPU",
            "选择后需要重新加载模型才会对推理生效。",
            [("gpu_index", "GPU 序号", "0")],
            self.submit_gpu_selection,
        )

    def submit_gpu_selection(self, values: Dict[str, str]) -> None:
        try:
            index = int(values.get("gpu_index", ""))
        except ValueError:
            self.write_status("[red]GPU 序号必须是整数[/]")
            return
        self.app.confirm(
            "切换 GPU", f"将选择 GPU #{index}，当前模型若已加载需要重新加载。",
            lambda: self.run_json_operation(
                "POST", "/device/select-gpu", {"gpu_index": index}, success="GPU 已切换",
                refresh=(self.load_device, self.action_reload)),
            confirm_label="切换",
        )

    def action_settings_write(self) -> None:
        if PAGES[self.page_index][0] != "settings":
            return
        self.run_settings_read()

    @work(thread=True, exclusive=True, group="settings")
    def run_settings_read(self) -> None:
        try:
            current = self.app.api.get("/user/settings")
        except ApiError as exc:
            self.app.call_from_thread(self.write_status, f"[red]读取用户设置失败[/]：{exc}")
            current = {}
        settings = current.get("settings", {}) if isinstance(current, dict) else {}
        self.app.call_from_thread(self.open_settings_form, settings)

    def open_settings_form(self, settings: Any) -> None:
        if not isinstance(settings, dict):
            settings = {}
        self.open_form(
            "写入用户设置",
            "请输入完整 JSON 对象；空对象会清空用户自定义设置。",
            [("settings", "设置 JSON", json.dumps(settings, ensure_ascii=False))],
            self.submit_settings,
        )

    def submit_settings(self, values: Dict[str, str]) -> None:
        try:
            settings = json.loads(values.get("settings", "{}") or "{}")
        except json.JSONDecodeError as exc:
            self.write_status(f"[red]设置 JSON 无效[/]：{exc.msg}")
            return
        if not isinstance(settings, dict):
            self.write_status("[red]设置必须是 JSON 对象[/]")
            return
        self.app.confirm(
            "写入用户设置", json.dumps(settings, ensure_ascii=False),
            lambda: self.run_json_operation(
                "PUT", "/user/settings", {"settings": settings}, success="用户设置已保存",
                refresh=()),
            confirm_label="保存",
        )

    # ------------------------------------------------------------ 写操作

    def write_status(self, text: str) -> None:
        """写操作没有独立回显窗口，结果常驻侧栏摘要（避免"点了没反应"）。"""
        self.op_status = text
        self.refresh_topbar()

    def finish_write_action(self, text: str) -> None:
        """写操作收尾：回显 + 重新拉取受影响的数据。"""
        self.write_status(text)
        self.action_reload()
        self.load_pages()

    def finish_light_action(self, text: str) -> None:
        self.write_status(text)
        self.load_pages()

    def selected_model_id(self, table: DataTable) -> str:
        """取光标行的 row key（``fill_models`` 以 model_id 作 key，不依赖行序）。"""
        if table.row_count == 0:
            return ""
        try:
            row_key, _column_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:  # noqa: BLE001 - 空表或坐标越界
            return ""
        return str(row_key.value or "")

    def model_load_args(self, info: Dict[str, Any]) -> Tuple[str, str]:
        """由注册表项推出 (engine, quant_type)——不猜参数，缺省即后端默认。

        量化沿用目录的 default_quant_type；旧目录缺此字段时优先 int4，
        再退到可用列表首项。
        """
        engine = str(info.get("preferred_engine") or "llama_cpp")
        quant_types = info.get("quant_types") or info.get("quantizations") or []
        if isinstance(quant_types, dict):
            quant_types = list(quant_types.keys())
        elif isinstance(quant_types, str):
            quant_types = [quant_types]
        choices = [str(item) for item in quant_types]
        if info.get("default_quant_type"):
            quant = str(info["default_quant_type"])
        elif "int4" in choices:
            quant = "int4"
        else:
            quant = choices[0] if choices else "int4"
        return engine, quant

    def action_load_model(self) -> None:
        if PAGES[self.page_index][0] != "models":
            self.write_status("[yellow]请先切到「模型」屏（侧栏或 [ ] 键）[/]")
            return
        model_id = self.selected_model_id(self.query_one("#models-table", DataTable))
        if not model_id or model_id not in self.model_rows:
            self.write_status("[yellow]模型屏没有可加载的行[/]")
            return
        info = self.model_rows[model_id]
        if info.get("is_available") is False:
            self.write_status(f"[yellow]{model_id} 不可用[/]：{escape(str(info.get('unavailable_reason') or '后端未说明原因'))}")
            return
        engine, quant = self.model_load_args(info)
        self.confirm_model_load(model_id, engine, quant)

    def confirm_model_load(self, model_id: str, engine: str, quant: str) -> None:
        if self.model_mode is None:
            self.write_status("[yellow]请先选择 Local 或分布式模式[/]")
            return
        if self._model_action_inflight:
            self.write_status("[yellow]模型操作进行中，请等待[/]")
            return
        if self.model_mode == "distributed":
            note = "自动规划与供给层段，然后准备推理；无需手选层段。就绪前聊天等待。"
            if self.model_intent and self.model_intent.get("status") not in MODEL_INTENT_TERMINAL:
                note += "\n若更换选择，将先取消尚未完成的选模 operation。"
        else:
            note = "显式退出集群，真实卸载当前模型/层段，然后加载本地整模。"
        self.app.confirm(
            "选择模型", f"{model_id}\n引擎 {engine} · 量化 {quant}\n{note}",
            lambda: self.run_model_action("load", model_id, engine, quant), confirm_label="加载")

    def action_unload_model(self) -> None:
        if PAGES[self.page_index][0] != "models":
            self.write_status("[yellow]请先切到「模型」屏（侧栏或 [ ] 键）[/]")
            return
        self.confirm_model_unload()

    def confirm_model_unload(self) -> None:
        if self.model_mode == "distributed" and self.model_intent and self.model_intent.get("status") not in MODEL_INTENT_TERMINAL:
            self.action_cancel_model_intent()
            return
        self.app.confirm(
            "卸载模型", "释放当前模型与推理计划；正在生成/排队的请求会失败。",
            lambda: self.run_model_action("unload", "", "", ""), confirm_label="卸载")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "model-mode-local": lambda: self.confirm_model_mode("local"),
            "model-mode-distributed": lambda: self.confirm_model_mode("distributed"),
            "model-intent-connect": self.connect_intent,
            "model-intent-cancel": self.action_cancel_model_intent,
        }
        if event.button.id in actions:
            event.stop()
            actions[event.button.id]()

    def confirm_model_mode(self, mode: str) -> None:
        if self._model_action_inflight:
            self.write_status("[yellow]模型操作进行中，请等待[/]")
            return
        text = ("退出集群，取消未完成的选模并真实卸载模型/层段；随后可选择本地整模。"
                if mode == "local" else "启用集群参与；从主节点注册表选模并自动供给、准备。")
        self.app.confirm("切换模型模式", text,
                         lambda: self.run_model_action(mode, "", "", ""), confirm_label="切换")

    def action_cancel_model_intent(self) -> None:
        if self.model_mode != "distributed" or not self.model_intent:
            self.write_status("[yellow]没有可取消的选模 operation[/]")
            return
        if self.model_intent.get("status") in MODEL_INTENT_TERMINAL:
            self.write_status("[yellow]选模已结束；ready 模型请用 U 卸载[/]")
            return
        self.app.confirm("取消选模", str(self.model_intent["operation_id"]),
                         lambda: self.run_model_action("cancel", "", "", ""), confirm_label="取消选模")

    def connect_intent(self) -> None:
        if self.model_mode != "distributed":
            self.write_status("[yellow]请先显式切换到分布式模式，再发现主节点[/]")
            return
        self.run_model_action("connect", "", "", "")

    def run_model_action(self, kind: str, model_id: str, engine: str, quant: str) -> None:
        # Capture the user's choice synchronously, before starting a worker or poll.
        if self._model_action_inflight:
            self.write_status("[yellow]模型操作进行中，请等待[/]")
            return
        previous = dict(self.model_intent)
        if kind in {"load", "local", "distributed"}:
            self._intent_restore_enabled = True
        elif kind == "unload":
            self._intent_restore_enabled = False
            self._intent_unloaded_operation_id = str(previous.get("operation_id") or "")
        if kind in {"local", "distributed"}:
            self.model_mode = kind
            self.model_intent = {}
            self.model_selection_id = ""
            self.model_rows = {}
            self.query_one("#models-table", DataTable).clear()
            self.model_aux = {}
            self.model_binding = {}
        if kind == "load":
            self.model_selection_id = model_id
            self.model_intent = {}
            self._begin_model_load(model_id)
        self._selection_epoch += 1
        self._model_action_inflight = True
        self._model_error = ""
        self._intent_poll_error = ""
        if self.model_mode == "local" or kind == "unload":
            self._local_model_ready = False
        self.render_model_intent()
        self.set_busy("发现/连接主节点…" if kind == "connect" else "模型操作进行中…")
        self.model_action_worker(kind, model_id, engine, quant, self.model_mode,
                                 self._selection_epoch, previous)

    @work(thread=True, group="model-intent-control")
    def model_action_worker(self, kind: str, model_id: str, engine: str, quant: str,
                            mode: Optional[str], epoch: int, previous: Dict[str, Any]) -> None:
        result: Dict[str, Any] = {}
        error = ""
        try:
            if kind == "load" and mode == "distributed":
                old_intent = previous.get("intent") or {}
                same_selection = (old_intent.get("model_id") == model_id
                                  and old_intent.get("engine", engine) == engine
                                  and old_intent.get("quant_type", quant) == quant)
                if previous.get("operation_id") and previous.get("status") not in MODEL_INTENT_TERMINAL and not same_selection:
                    cancel_model_intent(self.app.api, str(previous["operation_id"]))
                result = create_model_intent(self.app.api, model_id, engine=engine, quant_type=quant)
                if not result.get("operation_id"):
                    raise ApiError("INTENT_RESPONSE_INVALID: 主节点未返回 operation_id")
            elif kind == "cancel":
                result = cancel_model_intent(self.app.api, str(previous["operation_id"]))
            elif kind == "connect":
                result = connect_model_intent(self.app.api)
            else:
                if previous.get("operation_id") and previous.get("status") not in MODEL_INTENT_TERMINAL:
                    cancel_model_intent(self.app.api, str(previous["operation_id"]))
                if kind == "load":
                    result = load_local_model(self.app.api, model_id, engine=engine, quant_type=quant)
                elif kind in {"local", "distributed"}:
                    result = set_model_mode(self.app.api, kind)
                    if kind == "local":
                        result = unload_model(self.app.api)
                else:
                    result = unload_model(self.app.api)
                if result.get("success") is False or result.get("status") in {"error", "failed"}:
                    raise ApiError(str(result.get("reason") or result.get("message") or "模型操作失败"))
        except Exception as exc:  # network/runtime failures must release the UI gate
            error = str(exc)
        self.app.call_from_thread(self.finish_model_action, kind, model_id, epoch, result, error)

    def finish_model_action(self, kind: str, model_id: str, epoch: int,
                            result: Dict[str, Any], error: str) -> None:
        if not self.is_attached or not self.query("#intent-pane") or epoch != self._selection_epoch:
            return
        # Polls begun while this write was in flight must also be retired.
        self._selection_epoch += 1
        self._model_action_inflight = False
        self._end_model_load()
        self._model_error = error
        self._model_error_kind = kind if error else ""
        if error:
            text = f"[red]模型操作失败[/]：{escape(error)} · 可重试"
            if kind == "connect":
                self.model_binding = {"status": "disconnected", "reason": error,
                                      "reason_code": "INTENT_CONNECT_FAILED"}
        elif kind in {"load", "cancel"} and self.model_mode == "distributed":
            self.apply_model_intent(result)
            text = escape(self.model_intent_summary())
        elif kind == "connect":
            self.model_binding = result.get("binding", result)
            text = ("主节点发现/连接已完成" if self.model_binding.get("status") == "connected"
                    else "主节点连接尚未完成，请重试：" + escape(str(self.model_binding.get("reason_code") or "INTENT_MASTER_DISCONNECTED")))
        elif kind == "load":
            self._local_model_ready = True
            text = f"[green]本地整模已加载[/] {model_id}"
        else:
            if kind == "unload":
                self.model_intent = {}
                self.model_selection_id = ""
            text = "本地模式已启用，模型已卸载" if kind == "local" else "分布式模式已启用" if kind == "distributed" else "模型已卸载"
        self.render_model_intent()
        self.query_one("#chat-pane", ChatPane).write_line(text)
        self.write_status(text)
        # 写操作收尾必须立刻看到最新注册表/绑定，故本轮强制拉一次（#113-④）。
        self.action_reload(refresh_catalog=True)

    def apply_model_intent(self, value: Dict[str, Any], *, authoritative_current: bool = False) -> None:
        # Core current reads have already passed the epoch and endpoint checks.
        # Writes and operation-specific results must still match our selection.
        if authoritative_current and self._model_action_inflight:
            return
        if "_error" in value:
            self._intent_poll_error = str(value["_error"])
            return
        incoming_id = value.get("operation_id")
        if not incoming_id:
            if authoritative_current and (not value or value == {"value": None}):
                self._intent_poll_error = ""
                self.model_intent = {}
            return
        current_id = self.model_intent.get("operation_id")
        if authoritative_current:
            # An explicit unload suppresses only that old operation. A later
            # selection on another device must still become visible here.
            if not self._intent_restore_enabled and incoming_id == self._intent_unloaded_operation_id:
                self._intent_poll_error = ""
                return
            self._intent_restore_enabled = True
            if incoming_id != current_id and self._model_error_kind in {"load", "cancel"}:
                self._model_error = ""
                self._model_error_kind = ""
        else:
            if current_id and incoming_id != current_id:
                return
            returned_model = (value.get("intent") or {}).get("model_id")
            if not current_id and self.model_selection_id and returned_model and returned_model != self.model_selection_id:
                return
        self._intent_poll_error = ""
        self.model_intent = value
        if model_intent_ready(value) and self._model_error_kind == "cancel":
            self._model_error = ""

    def model_intent_summary(self) -> str:
        if self._model_action_inflight:
            return f"{self.model_selection_id} · 操作进行中"
        if self._model_error:
            return f"{self.model_selection_id} · 失败: {self._model_error}"
        summary = format_model_intent(self.model_intent).splitlines()[0]
        if self._intent_poll_error or self.model_binding.get("status") != "connected":
            return f"主节点状态待确认 · 上次: {summary}"
        return summary

    def model_intent_text(self) -> str:
        text = f"模式: {self.model_mode or '待确认'}"
        if self.model_mode == "distributed":
            binding = self.model_binding
            master = binding.get("master_node_id") or binding.get("master_host") or ("本机主节点" if binding.get("is_master") else "—")
            text += f" · 主节点: {master} · {binding.get('status') or '待连接'}"
            if binding.get("reason_code") or binding.get("reason"):
                text += f"\n{binding.get('reason_code') or ''}: {binding.get('reason') or ''} · 发现/重试连接 (J)"
            text += "\n" + format_model_intent(self.model_intent)
        if self._model_action_inflight:
            text += f"\n{self.model_selection_id} · 模型操作进行中"
        if self._model_error or self._intent_poll_error:
            text += f"\n{self._model_error or self._intent_poll_error} · 请刷新/重试"
        return text

    def chat_routing_preference(self) -> str:
        """发送路由由 owner 模式决定；未知模式沿用 app 的命令偏好。"""
        return {"distributed": "distributed_required", "local": "local_only"}.get(
            self.model_mode, self.app.routing_preference)

    def chat_streaming_mode(self) -> str:
        """从节点用既有 full 转发入口；主节点和本地文本保留 interactive。"""
        if (self.model_mode == "distributed" and self.model_binding.get("status") == "connected"
                and self.model_binding.get("is_master") is False):
            return "full"
        return "interactive"

    def chat_model_context(self) -> Tuple[Any, ...]:
        """复用模型控制 epoch 和 owner 投影，隔离角色/选模变化后的迟到回包。"""
        return (self._selection_epoch, self.model_mode,
                self.model_binding.get("status"), self.model_binding.get("is_master"),
                self.model_binding.get("master_node_id"), self.model_binding.get("master_host"),
                self.model_intent.get("operation_id"), self.model_intent.get("config_id"),
                self.model_intent.get("generation"), model_intent_ready(self.model_intent))

    def chat_model_gate(self) -> str:
        prefix = ""
        if self._model_action_inflight or self._model_loading_id:
            prefix = "模型操作进行中，请等待。"
        elif self._model_error:
            prefix = "模型操作失败，请重试。"
        elif self.model_mode == "distributed":
            if self._intent_poll_error or self.model_binding.get("status") != "connected":
                prefix = "主节点失连，请使用发现/重试连接。"
            elif not model_intent_ready(self.model_intent):
                prefix = "模型尚未就绪，请等待或重新选模。"
        elif self.model_mode == "local" and self._local_model_ready is False:
            prefix = "本地模型未就绪，请加载整模后发送。"
        if not prefix:
            return ""
        # Keep input visible even for a large cluster; full node receipts live on the model page.
        details = self.model_intent_summary() if self.model_mode == "distributed" else self.model_selection_id
        operation_id = self.model_intent.get("operation_id")
        if operation_id:
            details += f"\noperation_id: {operation_id}"
        reason = self._model_error or self._intent_poll_error
        if not reason:
            source = self.model_binding if self.model_binding.get("status") != "connected" else self.model_intent
            reason = f"{source.get('reason_code') or ''}: {source.get('reason') or ''}".strip(": ")
        if reason:
            details += f"\n{reason}"
        return prefix + "\n" + details

    def render_model_intent(self) -> None:
        self.query_one("#intent-pane", Static).update(self.model_intent_text())
        if self.model_mode == "distributed":
            self.fill_model_aux({})
        self.query_one("#model-mode-local", Button).variant = "primary" if self.model_mode == "local" else "default"
        self.query_one("#model-mode-distributed", Button).variant = "primary" if self.model_mode == "distributed" else "default"
        self.query_one("#model-intent-cancel", Button).disabled = (
            self._model_action_inflight or not self.model_intent or
            self.model_intent.get("status") in MODEL_INTENT_TERMINAL)
        if self.model_mode == "distributed":
            self.model_text = self.model_intent_summary()
        if self._last_core:
            self.fill_status(*self._last_core)
        gate = self.chat_model_gate()
        if gate or self._chat_model_gated:
            self.query_one("#chat-pane", ChatPane).set_status(escape(gate) if gate else "模型已就绪，可以发送消息。")
        self._chat_model_gated = bool(gate)
        self.refresh_topbar()

    def set_busy(self, text: str) -> None:
        self.write_status(f"[yellow]…[/] {text}")

    # ★ 2026-10-09（稳定性 #73-②）：模型加载的**单一权威状态**，供「状态」屏与状态栏
    #   共用 —— 消除"未加载"与"正在加载…"并存的自相矛盾（用户截图实证）。
    #   经 `call_from_thread` 调用：它们写 UI 状态，必须在主线程执行。
    def _begin_model_load(self, model_id: str) -> None:
        self._model_loading_id = str(model_id or "")
        self.model_text = f"[yellow]加载中[/] {self._model_loading_id}"

    def _end_model_load(self) -> None:
        self._model_loading_id = ""

    def action_queue_toggle_pause(self) -> None:
        """队列屏 P：暂停/恢复接受新请求（可逆，无需确认）。"""
        if PAGES[self.page_index][0] != "queue":
            self.write_status("[yellow]请先切到「队列」屏[/]")
            return
        if not self.queue_state:
            self.write_status("[yellow]队列数据不可用，无法操作[/]")
            return
        self.run_queue_action("resume" if self.queue_state.get("paused") else "pause")


    def action_queue_clear(self) -> None:
        """队列屏 C：清空排队任务（列出将清空的内容后再确认）。"""
        if PAGES[self.page_index][0] == "models":
            if self.model_intent and self.model_intent.get("status") not in MODEL_INTENT_TERMINAL:
                self.action_cancel_model_intent()
            else:
                self.action_model_cancel_download()
            return
        if PAGES[self.page_index][0] != "queue":
            self.write_status("[yellow]请先切到「队列」屏[/]")
            return
        if not self.queue_state:
            self.write_status("[yellow]队列数据不可用，无法操作[/]")
            return
        pending = self.queue_state.get("queue_size", "—")
        listing = []
        for level in ("tasks",):
            tasks = self.queue_state.get(level) or []
            if not tasks:
                continue
            names = ", ".join(
                str(task.get("task_id") or task.get("id") or "task")
                for task in tasks[:5] if isinstance(task, dict))
            listing.append(f"  {level.upper()}（{len(tasks)}）：{names}")
        self.app.confirm(
            "清空排队任务",
            f"将清空**排队中**的任务（queue_size={pending}），执行中的任务**不受影响**：\n"
            + ("\n".join(listing) if listing else "  （FIFO 当前为空）")
            + "\n此操作不可撤销。",
            lambda: self.run_queue_action("clear"),
            confirm_label="清空",
        )

    @work(thread=True, exclusive=True, group="queuectl")
    def run_queue_action(self, action: str, value: str = "") -> None:
        app = self.app
        self.app.call_from_thread(self.set_busy, f"队列操作 {action} …")
        try:
            if action == "pause":
                pause_queue(app.api)
            elif action == "resume":
                resume_queue(app.api)
            elif action == "clear":
                clear_queue(app.api)
            elif action == "cancel":
                result = cancel_queue_task(app.api, value)
                if not result.get("success"):
                    raise ValueError(str(result.get("message") or "任务不存在或已完成"))
            else:
                raise ValueError(f"未知队列动作: {action}")
            text = f"[green]队列 {action} 完成[/]" + (f" → {value}" if value else "")
        except (ApiError, ValueError) as exc:
            text = f"[red]队列 {action} 失败[/]：{exc}"
        self.app.call_from_thread(self.finish_light_action, text)

    # ------------------------------------------------------------ OpenAPI 调试兜底

    @work(thread=True, exclusive=True, group="api")
    def load_api_catalog(self) -> None:
        """从运行中后端读取 OpenAPI，避免 TUI 与路由表再次漂移。"""
        with self._refresh_state_lock:
            if (self._api_catalog_inflight
                    or self.backend_available is not True
                    or self.runtime_ready is not True):
                return
            self._api_catalog_inflight = True
        try:
            if isinstance(self.app.api, ApiClient):
                schema = self.app.api.get_openapi(timeout=PAGE_READ_TIMEOUT)
            else:
                schema = self.app.api.get_openapi()
            operations: Dict[str, Dict[str, Any]] = {}
            for openapi_path, methods in (schema.get("paths") or {}).items():
                if not isinstance(methods, dict):
                    continue
                for method, operation in methods.items():
                    method = str(method).upper()
                    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
                        continue
                    if not isinstance(operation, dict):
                        operation = {}
                    path = str(openapi_path)
                    if path.startswith("/api"):
                        path = path[4:] or "/"
                    key = f"{method} {openapi_path}"
                    parts = [part for part in path.split("/") if part]
                    domain = parts[0] if parts else "root"
                    if path.startswith("/chat/stream") or path.startswith("/chat/upload"):
                        mode = "专用界面"
                    elif method == "GET" or method == "HEAD":
                        mode = "查询"
                    else:
                        mode = "JSON 操作"
                    operations[key] = {
                        "key": key,
                        "method": method,
                        "openapi_path": str(openapi_path),
                        "path": path,
                        "domain": domain,
                        "mode": mode,
                        "summary": str(operation.get("summary") or operation.get("description") or ""),
                        "operation_id": str(operation.get("operationId") or ""),
                        "operation": operation,
                    }
            self.app.call_from_thread(self.fill_api_catalog, operations, "")
        except Exception as exc:  # noqa: BLE001 - 端点清单不可用时仍保留主界面
            self.app.call_from_thread(self.fill_api_catalog, {}, str(exc))
        finally:
            with self._refresh_state_lock:
                self._api_catalog_inflight = False

    def fill_api_catalog(self, operations: Dict[str, Dict[str, Any]], error: str = "") -> None:
        self.api_operations = operations
        table = self.query_one("#api-table", DataTable)
        table.clear()
        self.api_selected_key = ""
        if error:
            table.add_row("—", "[red]OpenAPI 不可用[/]", "", "", error)
            self.query_one("#api-detail", Static).update(
                "[red]无法读取端点清单[/]：" + error + "\n确认后端支持 /openapi.json，或先刷新。")
            return
        for key, item in sorted(operations.items(), key=lambda pair: pair[0]):
            summary = item["summary"].replace("\n", " ").strip()
            if len(summary) > 48:
                summary = summary[:45] + "..."
            table.add_row(
                item["method"], item["path"], item["domain"], item["mode"], summary,
                key=key,
            )
        if operations:
            first = next(iter(sorted(operations)))
            self.select_api_operation(first)
            self.query_one("#api-result", RichLog).write(
                f"已发现 {len(operations)} 个后端端点。GET 可查询，JSON 操作执行前需确认。")
            self.op_status = f"端点 {len(operations)}"
        else:
            self.query_one("#api-detail", Static).update("[yellow]后端未返回业务端点。[/]")
        self.refresh_topbar()

    def _api_row_key(self, event: Any) -> str:
        value = getattr(getattr(event, "row_key", None), "value", None)
        return str(value or "")

    def on_data_table_row_highlighted(self, event: Any) -> None:
        table = getattr(event, "data_table", None)
        if getattr(table, "id", None) != "api-table":
            return
        key = self._api_row_key(event)
        if key in self.api_operations:
            self.select_api_operation(key)

    def on_data_table_row_selected(self, event: Any) -> None:
        self.on_data_table_row_highlighted(event)

    def select_api_operation(self, key: str) -> None:
        item = self.api_operations.get(key)
        if not item:
            return
        self.api_selected_key = key
        operation = item.get("operation") or {}
        params = operation.get("parameters") or []
        query_names = [str(param.get("name")) for param in params
                       if isinstance(param, dict) and param.get("in") == "query"]
        body = ""
        if item["method"] not in {"GET", "HEAD"}:
            body = "{}"
        detail = (
            f"[b]{item['method']}[/] {item['path']}  ·  {item['domain']}  ·  {item['mode']}\n"
            f"operationId={item['operation_id'] or '—'}"
            + (f"  ·  query: {', '.join(query_names)}" if query_names else "")
            + ("\n该端点由聊天页专用流式处理，请返回聊天页。"
               if item["mode"] == "专用界面" else "")
        )
        self.query_one("#api-detail", Static).update(detail)
        self.query_one("#api-path", Input).value = item["path"]
        self.query_one("#api-params", Input).value = ""
        self.query_one("#api-body", Input).value = body

    def action_api_refresh(self) -> None:
        if PAGES[self.page_index][0] != "api":
            self.write_status("[yellow]请先切到「端点」屏[/]")
            return
        self.load_api_catalog()

    def action_api_execute(self) -> None:
        if PAGES[self.page_index][0] == "models":
            self.action_model_unregister()
            return
        if PAGES[self.page_index][0] == "nodes":
            self.action_node_deregister()
            return
        if PAGES[self.page_index][0] == "logs":
            self.action_logs_clear()
            return
        if PAGES[self.page_index][0] != "api":
            self.write_status("[yellow]请先切到「端点」屏[/]")
            return
        item = self.api_operations.get(self.api_selected_key)
        if not item:
            self.write_status("[yellow]请先选择端点[/]")
            return
        if item["mode"] == "专用界面":
            self.write_status("[yellow]该端点由聊天页专用协议处理，请从聊天页操作[/]")
            return
        path = self.query_one("#api-path", Input).value.strip()
        if not path or "{" in path or "}" in path:
            self.write_status("[yellow]请先把路径中的 {参数} 替换为实际值[/]")
            return
        try:
            params = self._parse_api_json(self.query_one("#api-params", Input).value, "查询参数")
            body = self._parse_api_json(self.query_one("#api-body", Input).value, "请求体")
        except ValueError as exc:
            self.write_status(f"[red]{exc}[/]")
            return
        method = item["method"]
        action = lambda: self.run_api_operation(method, path, params, body)
        if method not in {"GET", "HEAD"}:
            self.app.confirm(
                "执行端点",
                f"{method} {path}\n请求体：{json.dumps(body, ensure_ascii=False)}\n\n"
                "这是后端写操作，确认后才会发送。",
                action,
                confirm_label="执行",
            )
        else:
            action()

    @staticmethod
    def _parse_api_json(value: str, label: str) -> Any:
        text = (value or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}不是有效 JSON：{exc.msg}") from exc
        if label == "查询参数" and not isinstance(parsed, dict):
            raise ValueError("查询参数必须是 JSON 对象")
        return parsed

    @work(thread=True, exclusive=True, group="api")
    def run_api_operation(self, method: str, path: str, params: Any, body: Any) -> None:
        try:
            result = self.app.api.request(
                method, path, body=None if method in {"GET", "HEAD"} else body,
                params=params, with_log_token=True, timeout=180.0 if method not in {"GET", "HEAD"} else None)
            self.app.call_from_thread(self.show_api_result, method, path, result, "")
        except ApiError as exc:
            self.app.call_from_thread(self.show_api_result, method, path, {}, str(exc))

    def show_api_result(self, method: str, path: str, result: Any, error: str) -> None:
        view = self.query_one("#api-result", RichLog)
        view.clear()
        if error:
            view.write(f"{method} {path}\n错误：{error}")
            self.op_status = "端点失败"
        else:
            view.write(f"{method} {path}\n" + json.dumps(result, ensure_ascii=False, indent=2, default=str))
            self.op_status = "端点完成"
        self.update_sidebar(PAGES[self.page_index])

    # ------------------------------------------------------------ 动作

    def action_quit_app(self) -> None:
        self.app.exit()


class KoakumaApp(App):
    """统一交互入口的 Textual 实现。"""

    TITLE = "Koakuma"
    SUB_TITLE = SPLASH_SUBTITLE
    CSS = CSS

    def __init__(self, api: ApiClient, *, interval: float = 5.0,
                 routing_preference: str = "auto", show_thinking: bool = False,
                 enable_thinking: Optional[bool] = None,
                 supervisor: Any = None) -> None:
        super().__init__()
        self.api = api
        self.interval = float(interval)
        self.routing_preference = routing_preference
        # ★ 2026-10-07（#29 发起侧缺口）：请求级执行模式（`/mode` 命令可改）。
        #   之前 TUI 从不传该字段 ⇒ 任务图模式在 UI 里没有入口。
        self.execution_mode = "auto"
        self.show_thinking = show_thinking
        #: ★ 深度思考**开关**（None=auto 沿用模板默认 / True=强制思考 / False=强制不思考）。
        #:  与 `show_thinking`（仅控制 UI 是否显示）语义不同：本项**改变模型行为** ——
        #:  False 时引擎会经 chat template 的 `enable_thinking=False` **真正阻止**生成 `<think>`
        #:  （省算力），而不是靠事后剥离（后者依赖模板含 `<think>` 且能找到 `</think>`）。
        self.enable_thinking: Optional[bool] = enable_thinking
        #: 传入 BackendSupervisor 即在启动屏内完成冷启动（LOGO + 启动条同屏反馈）
        self.supervisor = supervisor
        self.session_id: Optional[str] = None
        self.generation_id: Optional[str] = None
        self.main_screen: Optional[MainScreen] = None

    # ------------------------------------------------------------ 启动流程

    def on_mount(self) -> None:
        waiting = self.supervisor is not None
        self.push_screen(SplashScreen(
            status="检查本地后端" if waiting else "连接后端",
            wait_for_backend=waiting,
        ))
        if waiting:
            self.set_interval(0.2, self.refresh_splash_status)
            self.start_backend()

    @work(thread=True, exclusive=True, group="backend")
    def start_backend(self) -> None:
        """在启动屏展示期间把本机后端拉起来（阶段文本实时写回启动屏）。"""
        try:
            self.supervisor.ensure_ready()
        except BaseException as exc:  # noqa: BLE001 - 交回主线程展示
            self.call_from_thread(self.backend_failed, str(exc))
            return
        self.call_from_thread(self.show_main)

    def refresh_splash_status(self) -> None:
        if self.supervisor is None:
            return
        screen = self.screen
        if isinstance(screen, SplashScreen):
            screen.set_status(self.supervisor.status_message)

    def backend_failed(self, message: str) -> None:
        screen = self.screen
        if isinstance(screen, SplashScreen):
            screen.wait_for_backend = False
            screen.set_status(f"[red]后端启动失败[/]：{message}（按任意键进入界面查看状态）")

    def confirm(self, title: str, body: str, on_confirm, *,
                confirm_label: str = "确认") -> None:
        """弹出模态确认；用户确认后才执行 ``on_confirm()``（写操作的统一闸门）。"""
        def _done(approved: Optional[bool]) -> None:
            if approved:
                on_confirm()

        self.push_screen(ConfirmScreen(title, body, confirm_label=confirm_label), _done)

    def show_main(self) -> None:
        """从启动屏切到主界面（重复调用安全）。"""
        if isinstance(self.screen, MainScreen):
            return
        self.switch_screen(MainScreen())


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, interval: float = 5.0,
        routing_preference: str = "auto", show_thinking: bool = False,
        supervisor: Any = None, log_token: str = "",
        enable_thinking: Optional[bool] = None) -> int:
    """启动 Textual 外壳（供 qlh.py 调用）。

    ``supervisor`` 非空时，本机后端冷启动在**启动屏内**完成（LOGO + 启动条同屏）。

    ★ 2026-10-03：补 `enable_thinking` 形参。它是**改变模型行为**的开关（经 chat
    template 传 `enable_thinking`），与只控制 UI 展示的 `show_thinking` 语义不同
    （见 `KoakumaApp.__init__` 的说明）。`main()` 一直从 `--reasoning` 解析并把它
    传进来，但 `run()` 没有这个形参 ⇒ `python -m tui_textual` 直接
    `TypeError: run() got an unexpected keyword argument 'enable_thinking'`（实测），
    而启动器正是这样调它的。
    """
    api = ApiClient(host=host, port=port, log_token=log_token)
    try:
        KoakumaApp(api, interval=interval, routing_preference=routing_preference,
                   show_thinking=show_thinking, enable_thinking=enable_thinking,
                   supervisor=supervisor).run()
    finally:
        if supervisor is not None:
            supervisor.stop()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Koakuma TUI (Textual shell)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--route", default="auto")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--reasoning", choices=["on", "off", "auto"], default="auto",
        help="深度思考开关（改变模型行为）：on/off/auto（默认沿用模型模板）",
    )
    parser.add_argument("--log-token", default="", help="remote log API token")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args.host, args.port, interval=args.interval,
               routing_preference=args.route, show_thinking=args.thinking,
               enable_thinking={"on": True, "off": False, "auto": None}.get(args.reasoning),
               log_token=args.log_token)


if __name__ == "__main__":
    raise SystemExit(main())
