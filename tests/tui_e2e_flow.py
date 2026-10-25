"""TUI 端到端 flow 的 **step 引擎**（`--mode assert` / `--mode demo` 共用同一份剧本）。

计划来源：`docs/archive/tui/TUI端到端flow测试与答辩演示复用计划-2026-09-24.md`（具名票 `CORE-TUI-E2E-01`）。
剧本单一来源：`fixtures/tui_e2e_flow.json`（`actions` / `expect` / `caption` 分离）。

为什么引擎放在 `tests/` 而不是 `scripts/`
----------------------------------------
`scripts/*` 被 `.gitignore` **整体忽略 + 逐条白名单**（`.gitignore:169-222`），且该文件属并行组
在制品。新增 `scripts/tui_e2e_flow.py` 在补白名单行**之前不会入库** ⇒ 若把引擎放那里，
核心逻辑就等于没交付。用户 2026-09-24 裁定（`dec-fc91fb5c966c7786`）：**引擎放这里（可入库）**，
`scripts/tui_e2e_flow.py` 只放一层转发壳。两者命令行等价。

纪律（计划 §3.3 / §6.2）
------------------------
* 缺 `textual` ⇒ **不改判为失败**：`run_flow()` 返回 `environment_missing=True`，退出码 `2`
  （等价 pytest 的 `skip`；**不得**把 skip 改写成 passed）。
* `--mode demo` 只要出现 DEGRADED/FAIL **绝不返回 0**；演示模式**不进门禁、不进 CI**。
* 不做"失败自动重试"：`--mode assert` 停在**首个**不满足的 expect（不掩盖失败）。
* 不碰生产代码路径：全部通过公共 API 驱动（`Pilot` + `App.notify` / `App.export_screenshot`）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

DEFAULT_FLOW = ROOT / "fixtures" / "tui_e2e_flow.json"
DEFAULT_FRAMES_DIR = ROOT / "build" / "tui-e2e" / "frames"

#: 每步状态
PASS = "PASS"
DEGRADED = "DEGRADED"
FAIL = "FAIL"
#: ★ D-a 档：缺工件 ⇒ **SKIP**（等价 pytest 的 skip，既不计 PASS 也不计 FAIL）
SKIP = "SKIP"

#: 退出码（计划 §3.3 / §4.1，与剧本里的 `exit_codes` 对齐）
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ENV_MISSING = 2
EXIT_DEGRADED = 3

#: 负向 expect：`{"kind": "raised", "exception": "<短名>"} => 期望的异常类名
RAISED_KINDS = {"no_matches": "NoMatches"}


# ------------------------------------------------------------------ 结果模型


@dataclass
class StepResult:
    step_id: str
    caption: str = ""
    status: str = PASS
    reason: str = ""
    frame: str = ""

    def to_dict(self) -> dict:
        return {"id": self.step_id, "caption": self.caption, "status": self.status,
                "reason": self.reason, "frame": self.frame}


@dataclass
class FlowResult:
    mode: str = "assert"
    topology: str = ""
    physical_dual_host: bool = False
    real_model_loaded: bool = False
    environment_missing: bool = False
    env_reason: str = ""
    steps: list = field(default_factory=list)

    @property
    def failed(self) -> list:
        return [s for s in self.steps if s.status == FAIL]

    @property
    def degraded(self) -> list:
        return [s for s in self.steps if s.status == DEGRADED]

    @property
    def skipped(self) -> list:
        """因环境缺失跳过的步骤（D-a 档缺工件）—— **不计失败、也不计通过**。"""
        return [s for s in self.steps if s.status == SKIP]

    def exit_code(self) -> int:
        if self.environment_missing:
            return EXIT_ENV_MISSING
        if self.failed:
            return EXIT_FAIL
        if self.mode == "demo" and self.degraded:
            return EXIT_DEGRADED
        return EXIT_OK

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "topology": self.topology,
            "physical_dual_host": self.physical_dual_host,
            "real_model_loaded": self.real_model_loaded,
            "environment_missing": self.environment_missing,
            "env_reason": self.env_reason,
            "exit_code": self.exit_code(),
            "counts": {
                "total": len(self.steps),
                "pass": len([s for s in self.steps if s.status == PASS]),
                "degraded": len(self.degraded),
                "fail": len(self.failed),
                "skip": len(self.skipped),
            },
            "steps": [s.to_dict() for s in self.steps],
        }


def load_flow(path: Path | str | None = None) -> dict:
    """读剧本（默认 `fixtures/tui_e2e_flow.json`）。"""
    target = Path(path) if path else DEFAULT_FLOW
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ------------------------------------------------------------------ 假后端


class FakeApi:
    """最小假后端：记录调用、按前缀返回预设结果（写操作不发真实请求）。

    表面与 `ApiClient` 对齐，照 `tests/test_tui_write_ops.py:33-77` 的 `RecordingApi`。
    """

    base_url = "http://record:8000"
    host = "127.0.0.1"
    port = 8000
    timeout = 5.0
    log_token = ""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = dict(responses or {})

    def _record(self, method, path, body=None, timeout=None):
        self.calls.append({"method": method, "path": path, "body": body, "timeout": timeout})
        for prefix, value in self.responses.items():
            if path.startswith(prefix):
                return value
        return {}

    def request(self, method, path, body=None, params=None, with_log_token=False, timeout=None):
        return self._record(method, path, body, timeout)

    def get(self, path, params=None, with_log_token=False):
        return self._record("GET", path)

    def post(self, path, body=None, params=None):
        return self._record("POST", path, body)

    def put(self, path, body=None):
        return self._record("PUT", path, body)

    def delete(self, path, body=None, params=None, timeout=None):
        return self._record("DELETE", path, body)

    def find(self, method=None, path=None):
        for call in self.calls:
            if (path is None or call["path"] == path) and (method is None or call["method"] == method):
                return call
        return None


# ------------------------------------------------------------------ 等待原语


async def wait_for(pilot, predicate, timeout: float = 6.0) -> bool:
    """轮询等待；切屏/挂载竞态期间查询可能抛 `NoMatches`，按「未就绪」处理。

    照抄既有约定 `tests/test_tui_write_ops.py:187-200`（不另发明节奏）。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:  # noqa: BLE001 - ContentSwitcher 未挂载的页面查不到
            pass
        await pilot.pause(0.05)
    try:
        return bool(predicate())
    except Exception:  # noqa: BLE001
        return False


class Session:
    """一次 `run_test()` 会话（一个 app 实例）。"""

    def __init__(self, app, api, pilot):
        self.app = app
        self.api = api
        self.pilot = pilot


def _build_app(api_kind: str):
    """按剧本声明构造 app。

    - `fake`（默认）= 可用的假后端（F1 档，零模型）；
    - `unreachable` = 不可达端点（用于"后端不可达时外壳照常可用"的降级步，照
      `tests/test_tui_textual.py:114-135` 的既有写法）；
    - **`real`**（★ 2026-10-09 稳定性 #73 新增）= **真后端**，这是"真端到端"档：
      真模型 +（若集群有 worker）真分布式层段。此前框架只支持 fake/unreachable，
      ⇒ **任何剧本都打不出真分布式**（`topology` 永远只能是 `single_host_loopback`）。
      主机/端口/超时可用 `QLH_TUI_E2E_REAL_HOST|PORT|TIMEOUT` 覆盖。
    """
    from tui_api import ApiClient
    from tui_textual import KoakumaApp

    app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5), interval=30)
    if api_kind == "unreachable":
        app.api = ApiClient(host="127.0.0.1", port=1, timeout=0.3)
    elif api_kind == "real":
        import os

        app.api = ApiClient(
            host=os.environ.get("QLH_TUI_E2E_REAL_HOST", "127.0.0.1"),
            port=int(os.environ.get("QLH_TUI_E2E_REAL_PORT", "8000")),
            timeout=float(os.environ.get("QLH_TUI_E2E_REAL_TIMEOUT", "180")),
        )
    else:
        app.api = FakeApi()
    return app, app.api


# ------------------------------------------------------------------ action 处理器


async def _action_enter_main(session, spec, flow, sink):
    from tui_textual import MainScreen

    session.app.show_main()
    await wait_for(session.pilot, lambda: isinstance(session.app.screen, MainScreen))
    nav_ids = flow["nav_ids"]
    await wait_for(session.pilot,
                   lambda: len(list(session.app.screen.query("#nav ListItem"))) == len(nav_ids))


async def _action_click_nav(session, spec, flow, sink):
    """鼠标点击侧栏项；点击**自带断言**（切屏 + 高亮同步）—— `Pilot.click` 返回值即断言。

    语义依据（2026-09-24 探针实测）：命中返回 `True`；落点在屏幕内但不在目标 widget 上返回
    `False`；落点超出屏幕抛 `OutOfBounds`；selector 不存在抛 `NoMatches`。
    """
    nav_ids = flow["nav_ids"]
    page = spec["page"]
    item_id = f"nav-{page}"
    if item_id not in nav_ids:
        raise AssertionError(f"剧本里的 page={page!r} 不在 nav_ids 中")
    index = nav_ids.index(item_id)

    landed = await session.pilot.click(f"#nav ListItem#{item_id}")
    ok = await wait_for(
        session.pilot,
        lambda: session.app.screen.query_one("#content").current == f"page-{page}")
    if not ok or landed is False:
        raise AssertionError(f"点击 {item_id} 未切屏（click 返回 {landed!r}）")
    nav = session.app.screen.query_one("#nav")
    if nav.index != index:
        raise AssertionError(f"点 {item_id} 后侧栏高亮应为 {index}，实际 {nav.index}")


async def _action_click_nav_unknown(session, spec, flow, sink):
    """负向：点不存在的 selector。异常由上层捕获并交给 `raised` expect 判定。"""
    await session.pilot.click(f"#nav ListItem#nav-{spec['page']}")


async def _action_press(session, spec, flow, sink):
    await session.pilot.press(spec["key"])
    await session.pilot.pause()


async def _action_chat_command(session, spec, flow, sink):
    from textual.widgets import Input

    pane = session.app.screen.query_one("#chat-pane")
    box = pane.query_one(Input)
    box.value = spec["text"]
    box.focus()
    await session.pilot.press("enter")
    await session.pilot.pause()


def _chat_text(app) -> str:
    """取聊天区当前内容。

    ★ 2026-10-09 修正：`chat_buffer` 是 **`ChatPane`（`#chat-pane`）的属性**，不是 `app`
    的属性 —— 初版写成 `getattr(app, "chat_buffer")` 恒为空串，于是"回复已到达"也被判成
    "后端无响应"（假失败）。这里按组件层级取，并在缓冲为空时回退到 `#chat-log` 的渲染内容。
    """
    try:
        pane = app.screen.query_one("#chat-pane")
    except Exception:  # noqa: BLE001 - 页面未挂载
        pane = None
    buf = str(getattr(pane, "chat_buffer", "") or "") if pane is not None else ""
    if buf:
        return buf
    try:
        rendered = app.screen.query_one("#chat-log").renderable
        return str(rendered or "")
    except Exception:  # noqa: BLE001
        return ""


def _assistant_text(app) -> str:
    """取**最后一条 assistant 回复**之后的文本（用于等"回复真的开始产出"）。

    ★ 2026-10-09：`send_chat` 原先的等待条件是「聊天区总长度增长」—— 这在「先 /clear 再发」
    的步骤里会**立刻满足**（用户消息一渲染长度就变了）⇒ 没等模型输出就返回，把"回复还没来"
    误判成"回复为空/被截断"。这里改为盯**最后一条 assistant 标记之后**的内容。
    """
    buf = _chat_text(app)
    idx = max(buf.rfind("assistant"), buf.rfind("[dim]assistant[/]"))
    if idx < 0:
        return ""
    return buf[idx + len("assistant"):].replace("[/]", "")


# ★ 2026-10-09：**连接/后端类错误的兜底判据**（本轮实测教训）。
#   后端没在运行时，`real_chat_not_rejected` 原先只判 `not_contains` 的几条业务文案 ⇒
#   把「assistant 的回复其实是 ApiError: 无法连接后端」**假判为 PASS**；
#   只有新加的 `min_chars` 判据抓到了它。这里统一把"连不上/超时/5xx"视为失败 ——
#   任何 chat 步骤都不许把"错误"当成"回复"。
_BACKEND_UNREACHABLE_MARKERS = (
    "无法连接后端",
    "ApiError",
    "WinError 10061",
    "WinError 10054",
    "目标计算机积极拒绝",
    "连接被拒绝",
    "请求超时",
    "模型未加载",
    "HTTP 500",
    "HTTP 502",
    "HTTP 503",
)


def _assert_no_backend_error(text: str, where: str) -> None:
    """`text` 里若含连接/后端错误标记 ⇒ 立刻失败（防「错误被当回复」的假 PASS）。"""
    for marker in _BACKEND_UNREACHABLE_MARKERS:
        if marker in text:
            raise AssertionError(
                f"{where}：assistant 回复里含后端/连接错误 {marker!r}"
                f"（后端未运行、已重启或拒绝服务）。回复前 300 字：{text[:300]!r}"
            )


async def _action_send_chat(session, spec, flow, sink):
    """在聊天屏发**真消息**并等回复落地（真后端 / `api: "real"` 档专用）。

    与 `chat_command` 的关键区别：后者只把文本敲进去就返回（用于验证外壳的输入与
    命令分发，后端回不回都不关心）；本 action **必须等 assistant 真的产出内容** ——
    否则「发送成功但后端 500 / 回了错误 / 还没来得及回」都可能被误判成通过或"被截断"。
    """
    from textual.widgets import Input

    pane = session.app.screen.query_one("#chat-pane")
    box = pane.query_one(Input)
    box.value = spec["text"]
    box.focus()
    await session.pilot.press("enter")
    timeout = float(spec.get("timeout", 300.0))
    min_reply = int(spec.get("min_reply_chars", 2))
    ok = await wait_for(
        session.pilot,
        lambda: len(_assistant_text(session.app)) > min_reply,
        timeout=timeout,
    )
    if not ok:
        raise AssertionError(
            f"send_chat 后 {timeout:.0f}s 内 assistant 未产出内容（后端无响应？）"
            f" 当前内容前 300 字：{_chat_text(session.app)[:300]!r}"
        )
    # 流式可能还在继续：再给一点时间让可判定的文本到位。
    await session.pilot.pause(spec.get("settle", 2.0))
    # ★ 2026-10-09：等到了"内容"不等于"回复成功" —— 若那是连接/后端错误，必须当场失败。
    _assert_no_backend_error(_assistant_text(session.app), "send_chat")


async def _action_set_cluster_aux(session, spec, flow, sink):
    session.app.screen.cluster_aux = dict(spec["data"])


async def _action_cluster_toggle(session, spec, flow, sink):
    session.app.screen.action_cluster_toggle()
    from tui_textual import ConfirmScreen

    await wait_for(session.pilot, lambda: isinstance(session.app.screen, ConfirmScreen))


async def _action_confirm(session, spec, flow, sink):
    await session.pilot.press(spec["key"])
    await session.pilot.pause(0.2)


# ---- D-a 档：本机 loopback 多段（演示降级）----
#
# 只**调用**既有脚本（`scripts/relay_mid_service.py` / `scripts/relay_health.py`），不改它们；
# 段服务强制 loopback（脚本自身也强制）。用途与边界见
# `docs/archive/tui/TUI端到端flow测试与答辩演示复用计划-2026-09-24.md` §4.2 D-a / §6.1 风险 6。


def _free_loopback_port() -> int:
    """取一个当前空闲的 loopback 端口，避免与开发中的后端或残留服务抢端口。"""
    import socket as _socket

    probe = _socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _read_ready_json(stream, timeout: float) -> dict:
    """从子进程 stdout 读**一行** JSON（`relay_mid_service` 起服务后打印 ready 行）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = stream.readline() if stream is not None else ""
        if not line:
            time.sleep(0.05)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except ValueError:
            continue        # 服务可能先打人类可读行，跳过
    return {}


async def _action_start_relay_service(session, spec, flow, sink):
    """起一个本机 loopback relay 段服务；就绪后把句柄放进 `sink["state"]`。

    ⚠️ 缺 shim / 段工件时**不建服务**并置 `sink["skip"]` ⇒ 整档记为 SKIP（**不判失败**，
    与 `tests/test_real_model_smoke.py` 的开关式约定一致）。
    """
    import subprocess

    shim = ROOT / str(spec["shim"])
    model = ROOT / str(spec["model"])
    missing = [str(path.relative_to(ROOT)) for path in (shim, model) if not path.is_file()]
    if missing:
        sink["skip"] = (f"缺 D-a 档工件 {missing}"
                        "（段工件由 scripts/cut_layers.py --keep-head/--k/--end 生成）")
        return

    port = _free_loopback_port()
    command = [
        sys.executable, str(ROOT / "scripts" / "relay_mid_service.py"),
        "--role", str(spec.get("role", "middle")),
        "--listen", f"127.0.0.1:{port}",
        "--keep-head-shim", str(shim),
        "--model", str(model),
    ]
    if spec.get("n_embd"):
        command += ["--n-embd", str(spec["n_embd"])]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    ready = _read_ready_json(process.stdout, timeout=180.0)
    sink.setdefault("process", process)
    sink["port"] = port
    sink["ready"] = ready
    sink["endpoint"] = f"127.0.0.1:{port}"


async def _reap_relay_service(state: dict) -> None:
    """兜底回收本档起的段服务（幂等）。

    ⚠️ 必要性：`--mode assert` 停在**首个**失败步 ⇒ 若失败发生在 `da_service_down` 之前，
    那个步骤不会执行 ⇒ 段服务可能变孤儿。这里放在 `finally` 里无条件收。
    """
    process = state.pop("process", None)
    if process is None:
        return
    try:
        process.terminate()
        process.wait(timeout=15)
    except Exception:  # noqa: BLE001 - 停不掉就强杀
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            pass


async def _action_stop_relay_service(session, spec, flow, sink):
    """停掉本档起的段服务（幂等）。"""
    await _reap_relay_service(sink)


async def _action_relay_health_check(session, spec, flow, sink):
    """用 `scripts/relay_health.py --check` 探该段 —— **真链路**证据（不是"进程起来了"）。"""
    import subprocess

    endpoint = sink.get("endpoint", "")
    if not endpoint:
        sink["skip"] = "D-a 档未起服务（见上一步）"
        return
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "relay_health.py"),
         "--check", f"middle=tcp:{endpoint}", "--json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
    )
    sink["health"] = {"returncode": proc.returncode, "stdout": proc.stdout[-2000:]}


async def _action_relay_hidden_roundtrip(session, spec, flow, sink):
    """与该段做一次 hidden 往返，并**逐字节**校验。

    要求"回来了但**不是原样**"——用于排除"段把输入原样回显"这类假通过。
    宽度取自服务端 ready 行的 `n_embd`（或剧本 `n_embd`）。
    """
    from relay_transport import RelayTcpClient, expected_hidden_bytes

    state = sink
    endpoint = state.get("endpoint", "")
    if not endpoint:
        sink["skip"] = "D-a 档未起服务（见上一步）"
        return
    ready = state.get("ready") or {}
    n_embd = int(spec.get("n_embd") or ready.get("n_embd") or 0)
    if n_embd < 1:
        sink["error"] = f"拿不到段宽度 n_embd（ready={ready!r}）"
        return

    n_tokens = 2
    payload = bytes((index * 7 + 3) % 256
                    for index in range(expected_hidden_bytes(n_tokens, n_embd)))
    host, _, port_text = endpoint.partition(":")
    client = RelayTcpClient(host, int(port_text), n_embd=n_embd, timeout=180.0)
    try:
        returned = client.request_hidden(payload, n_tokens=n_tokens)
    finally:
        client.close()
    sink["roundtrip"] = {"sent": bytes(payload), "got": bytes(returned)}


ACTIONS = {
    "enter_main": _action_enter_main,
    "click_nav": _action_click_nav,
    "click_nav_unknown": _action_click_nav_unknown,
    "press": _action_press,
    "chat_command": _action_chat_command,
    "send_chat": _action_send_chat,
    "set_cluster_aux": _action_set_cluster_aux,
    "cluster_toggle": _action_cluster_toggle,
    "confirm": _action_confirm,
    # D-a 档（本机 loopback 多段）
    "start_relay_service": _action_start_relay_service,
    "stop_relay_service": _action_stop_relay_service,
    "relay_health_check": _action_relay_health_check,
    "relay_hidden_roundtrip": _action_relay_hidden_roundtrip,
}

#: 特殊 action：不经过 handler，由执行器用来切分「会话」
SESSION_ACTION = "new_app"


# ------------------------------------------------------------------ expect 处理器


async def _expect_screen_is_main(session, spec, flow):
    from tui_textual import MainScreen

    if not isinstance(session.app.screen, MainScreen):
        raise AssertionError(f"应在 MainScreen，实际 {type(session.app.screen).__name__}")


async def _expect_nav_count(session, spec, flow):
    items = list(session.app.screen.query("#nav ListItem"))
    if len(items) != int(spec["value"]):
        raise AssertionError(f"侧栏应有 {spec['value']} 项，实际 {len(items)}")
    ids = [item.id for item in items]
    if ids != list(flow["nav_ids"]):
        raise AssertionError(f"侧栏 id 顺序不符: {ids}")


async def _expect_page(session, spec, flow):
    current = session.app.screen.query_one("#content").current
    if current != spec["value"]:
        raise AssertionError(f"当前屏应为 {spec['value']}，实际 {current}")


async def _expect_nav_index(session, spec, flow):
    index = session.app.screen.query_one("#nav").index
    if index != int(spec["value"]):
        raise AssertionError(f"侧栏高亮应为 {spec['value']}，实际 {index}")


async def _expect_app_attr(session, spec, flow):
    actual = getattr(session.app, spec["name"], None)
    if actual != spec["value"]:
        raise AssertionError(f"app.{spec['name']} 应为 {spec['value']!r}，实际 {actual!r}")


async def _expect_api_called(session, spec, flow):
    await wait_for(session.pilot,
                   lambda: session.api.find(spec["method"], spec["path"]) is not None)
    call = session.api.find(spec["method"], spec["path"])
    if call is None:
        raise AssertionError(
            f"未调用 {spec['method']} {spec['path']}；已调用: {[c['path'] for c in session.api.calls]}")
    if "body" in spec and call["body"] != spec["body"]:
        raise AssertionError(f"{spec['path']} 请求体应为 {spec['body']}，实际 {call['body']}")


async def _expect_status_pane_contains_any(session, spec, flow):
    from textual.widgets import Static

    def _render() -> str:
        return str(session.app.screen.query_one("#status-pane", Static).render())

    def _hit() -> bool:
        text = _render()
        return any(needle in text for needle in spec["values"])

    if not await wait_for(session.pilot, _hit, timeout=8.0):
        raise AssertionError(f"状态屏未出现 {spec['values']}；实际: {_render()[:160]!r}")


async def _expect_app_exited(session, spec, flow):
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not session.app.is_running:
            return
        try:
            await session.pilot.pause(0.05)
        except Exception:  # noqa: BLE001 - 退出过程中会话可能已结束
            if not session.app.is_running:
                return
            raise
    raise AssertionError("按 q 后应用仍在运行")


# ---- D-a 档的 expect ----
#
# 持久状态挂在 `flow["_state"]`（由 `_run_steps` 注入），所以这里能跨步读到段服务句柄与往返结果。

async def _expect_service_ready(session, spec, flow):
    state = flow.get("_state") or {}
    if state.get("skip"):
        raise _StepSkipped(str(state["skip"]))
    if not state.get("endpoint"):
        raise AssertionError(f"relay 段未就绪：ready={state.get('ready')!r}")
    ready = state.get("ready") or {}
    if not ready:
        raise AssertionError(f"未收到段服务的 ready 行（端口 {state.get('port')}）")


async def _expect_health_ok(session, spec, flow):
    state = flow.get("_state") or {}
    if state.get("skip"):
        raise _StepSkipped(str(state["skip"]))
    health = state.get("health") or {}
    if not health:
        raise AssertionError("未执行健康检查")
    if health.get("returncode") != 0:
        raise AssertionError(
            f"健康检查判定异常（rc={health.get('returncode')}）：{health.get('stdout', '')[:300]}")


async def _expect_relay_roundtrip_ok(session, spec, flow):
    state = flow.get("_state") or {}
    if state.get("skip"):
        raise _StepSkipped(str(state["skip"]))
    result = state.get("roundtrip")
    if result is None:
        raise AssertionError(f"未完成 hidden 往返{state.get('error') or ''}")
    sent, got = result["sent"], result["got"]
    if len(got) != len(sent):
        raise AssertionError(f"往返长度不符: 送 {len(sent)} 字节、回 {len(got)} 字节")
    if got == sent:
        raise AssertionError("段把 hidden **原样返回**了 —— 这不是有效往返（假通过）")


async def _expect_service_stopped(session, spec, flow):
    state = flow.get("_state") or {}
    process = state.get("process")
    if process is not None and process.poll() is None:
        raise AssertionError(f"段服务仍在运行（pid={process.pid}）")


async def _expect_chat_reply(session, spec, flow):
    """断言聊天缓冲内容（真后端档）。`contains` / `not_contains` / `min_chars` 可同时给。

    这是「回复真的回来了、且不是报错」的判据：`not_contains` 用来卡住后端错误文案
    （如"后端错误"/"禁止整模回退"/"Traceback"），**避免把失败当成功** ——
    本 goal 的起点就是"看着像走了分布式、实际被拒且报错被藏起来"。
    `min_chars` 用于 #73-③「生成不完整」：长回答 prompt 必须真的产出足够长的文本，
    否则"截断"会伪装成 PASS。
    """
    buf = _chat_text(session.app)
    if not buf.strip():
        raise AssertionError("聊天区为空：没有产生任何对话内容")
    min_chars = int(spec.get("min_chars") or 0)
    # ★ 只量**最后一条 assistant 回复**，不含用户消息（否则"用户 prompt 很长"会伪装成"回复成型"）
    reply = _assistant_text(session.app)
    # ★ 2026-10-09：先拦"回复其实是错误"的情况，再谈长度/内容（否则错误会被当成合格回复）。
    _assert_no_backend_error(reply, "chat_reply")
    if min_chars and len(reply) < min_chars:
        raise AssertionError(
            f"assistant 回复过短：{len(reply)} < 要求 {min_chars}（可能被截断）。"
            f"回复前 400 字：{reply[:400]!r}"
        )
    for needle in spec.get("contains", []) or []:
        if needle not in buf:
            raise AssertionError(f"chat_buffer 缺少 {needle!r}（前 400 字：{buf[:400]!r}）")
    for needle in spec.get("not_contains", []) or []:
        if needle in buf:
            raise AssertionError(f"chat_buffer 不应包含 {needle!r}（前 400 字：{buf[:400]!r}）")


EXPECTS = {
    "screen_is_main": _expect_screen_is_main,
    "chat_reply": _expect_chat_reply,
    "nav_count": _expect_nav_count,
    "page": _expect_page,
    "nav_index": _expect_nav_index,
    "app_attr": _expect_app_attr,
    "api_called": _expect_api_called,
    "status_pane_contains_any": _expect_status_pane_contains_any,
    "app_exited": _expect_app_exited,
    # D-a 档（本机 loopback 多段）
    "service_ready": _expect_service_ready,
    "health_ok": _expect_health_ok,
    "relay_roundtrip_ok": _expect_relay_roundtrip_ok,
    "service_stopped": _expect_service_stopped,
}


# ------------------------------------------------------------------ 执行器


class _StepSkipped(Exception):
    """该步因**环境缺失**跳过（例如 D-a 档缺 shim / 段工件）。

    照 `tests/test_real_model_smoke.py` 的约定：**不判为失败**；报告里记 `SKIP`，
    既不计入 PASS 也不计入 FAIL ⇒ 退出码不受影响。
    """


class _StepFailure(Exception):
    """一步未通过。assert ⇒ 记 FAIL 并停止；demo ⇒ 记 DEGRADED 并继续。"""


def _frame_path(frames_dir, step_id: str) -> Path | None:
    if not frames_dir:
        return None
    return Path(frames_dir) / f"{step_id}.svg"


def _select_replay(steps: list, only_step: str | None):
    """逐条模式：截取「本步所属会话」到目标步的片段，**完整执行**（actions + expect）。

    ⚠️ 不能"只重放 actions、跳过前置步的 expect"：负向步（`unknown_selector_fails_loud`）
    的 expect 恰恰是「必须抛异常」，跳过判定会把它的正常行为误判成失败。完整执行也让
    前置步真失败时能如实报出来（前置失败 ⇒ 目标步本就无法判定）。
    """
    if not only_step:
        return steps
    target = next((i for i, s in enumerate(steps) if s["id"] == only_step), None)
    if target is None:
        raise KeyError(f"剧本中没有 step id={only_step!r}")
    start = 0
    for index in range(target, -1, -1):
        if any(a.get("kind") == SESSION_ACTION for a in steps[index].get("actions", [])):
            start = index
            break
    return steps[start:target + 1]


async def _run_steps(flow, *, mode, pace, frames_dir, only_step=None) -> FlowResult:
    result = FlowResult(mode=mode, topology=flow.get("topology", ""),
                        physical_dual_host=bool(flow.get("physical_dual_host", False)),
                        real_model_loaded=bool(flow.get("real_model_loaded", False)))

    replay = _select_replay(list(flow["steps"]), only_step)
    #: ★ 跨步持久状态（段服务句柄 / 健康检查结果 / 往返结果）—— expect handler 也从这里读
    flow.setdefault("_state", {})

    session: Session | None = None
    stack: AsyncExitStack | None = None
    try:
        for step in replay:
            step_id = step["id"]
            caption = step.get("caption", "")
            try:
                session, stack = await _run_one_step(
                    session, stack, step, flow, mode=mode, pace=pace,
                    frames_dir=frames_dir,
                )
                record = StepResult(step_id, caption, PASS)
                # 会话可能在这一步里被重建 ⇒ frame 只记路径
                frame = _frame_path(frames_dir, step_id)
                if frame is not None:
                    record.frame = str(frame)
            except _StepSkipped as exc:
                record = StepResult(step_id, caption, SKIP, str(exc))
                result.steps.append(record)
                # 缺工件 ⇒ 后续同族步骤也会跳过：继续跑（它们是幂等的空操作），但**不**计失败
                continue
            except _StepFailure as exc:
                record = StepResult(step_id, caption,
                                    FAIL if mode == "assert" else DEGRADED, str(exc))
                frame = _frame_path(frames_dir, step_id)
                if frame is not None:
                    record.frame = str(frame)
                result.steps.append(record)
                if mode == "assert":
                    return result
                continue
            except Exception as exc:  # noqa: BLE001 - 任何异常都算该步失败（含缺失依赖）
                record = StepResult(step_id, caption, FAIL, f"{type(exc).__name__}: {exc}")
                result.steps.append(record)
                if mode == "assert":
                    return result
                continue
            result.steps.append(record)
    finally:
        # ★ 兜底回收：D-a 档若中途失败（assert 模式停在首个失败步），`da_service_down` 不会执行
        await _reap_relay_service(flow.get("_state") or {})
        if stack is not None:
            await stack.aclose()
    return result


async def _run_one_step(session, stack, step, flow, *, mode, pace, frames_dir):
    """执行一步的 actions 并判定其 expect，必要时重建会话。

    返回 `(session, stack)`：会话被重建时 `stack` 也随之更换，因此**必须**把它回传给调用方，
    否则外层 `finally` 关的是旧 stack ⇒ 新会话不会被回收（真 bug，已实测修掉）。

    `step["needs_app"] = False` 的步骤（D-a 档的服务启停 / 健康检查 / 协议往返）**不建 app** ——
    它们只驱动外部进程与协议，既拿不到也不需要 TUI 会话。
    """
    actions = list(step.get("actions", []))
    new_app_spec = next((a for a in actions if a.get("kind") == SESSION_ACTION), None)
    needs_app = bool(step.get("needs_app", True))

    if new_app_spec is not None or (session is None and needs_app):
        # 结束旧会话（`AsyncExitStack.aclose()` 才会真正退出 `run_test()` 上下文）
        if stack is not None:
            await stack.aclose()
        stack = AsyncExitStack()
        await stack.__aenter__()
        api_kind = str(new_app_spec.get("api", "fake")) if new_app_spec else "fake"
        app, api = _build_app(api_kind)
        pilot = await stack.enter_async_context(app.run_test(size=(120, 40)))
        session = Session(app, api, pilot)

    raised: Exception | None = None
    for action in actions:
        kind = action.get("kind")
        if kind == SESSION_ACTION:
            continue
        handler = ACTIONS.get(kind)
        if handler is None:
            raise _StepFailure(f"未知 action kind={kind!r}")
        if mode == "demo":
            if session is not None:      # D-a 档的无 app 步骤没有字幕可打、也没有 pilot
                _demo_caption(session, step.get("caption", ""))
                await session.pilot.pause(pace)
        try:
            # ★ 第 4 个参数是**跨步持久状态**（`flow["_state"]`）：D-a 档用它传段服务句柄、
            #   健康检查结果与往返结果，expect handler 也从同一处读。
            await handler(session, action, flow, flow["_state"])
        except Exception as exc:  # noqa: BLE001 - 负向 action 的异常留给 expect 判定
            raised = exc
            break

    if mode == "demo" and frames_dir:
        _save_frame(session, Path(frames_dir) / f"{step['id']}.svg")

    failures = []
    for spec in step.get("expect", []):
        kind = spec.get("kind")
        if kind == "raised":
            want = RAISED_KINDS.get(str(spec.get("exception", "")), str(spec.get("exception", "")))
            if raised is None:
                failures.append(f"期望抛 {want}，但没有异常")
            elif type(raised).__name__ != want:
                failures.append(f"期望抛 {want}，实际 {type(raised).__name__}: {raised}")
            continue
        if raised is not None:
            failures.append(f"action 阶段已抛 {type(raised).__name__}: {raised}")
            break
        handler = EXPECTS.get(kind)
        if handler is None:
            failures.append(f"未知 expect kind={kind!r}")
            continue
        try:
            await handler(session, spec, flow)
        except AssertionError as exc:
            failures.append(str(exc))
            if mode == "assert":
                break

    if failures:
        raise _StepFailure("; ".join(failures))
    return session, stack


def _demo_caption(session, caption: str) -> None:
    if not caption:
        return
    try:
        session.app.notify(caption, title="TUI E2E", timeout=3)
    except Exception:  # noqa: BLE001 - 字幕失败不影响流程
        pass


def _save_frame(session, target: Path) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(session.app.export_screenshot(), encoding="utf-8")
    except Exception:  # noqa: BLE001 - 留证失败不影响流程
        pass


def run_flow(*, mode: str = "assert", pace: float = 1.2, frames_dir=None,
             flow_path=None, only_step: str | None = None) -> FlowResult:
    """跑一遍剧本。缺 `textual` ⇒ `environment_missing=True`（**不判为失败**）。"""
    flow = load_flow(flow_path)
    try:
        import textual  # noqa: F401
    except ImportError as exc:
        result = FlowResult(mode=mode, topology=flow.get("topology", ""),
                            physical_dual_host=bool(flow.get("physical_dual_host", False)),
                            real_model_loaded=bool(flow.get("real_model_loaded", False)))
        result.environment_missing = True
        result.env_reason = f"缺少 UI 依赖 textual: {exc}"
        return result

    return asyncio.run(_run_steps(flow, mode=mode, pace=pace, frames_dir=frames_dir,
                                  only_step=only_step))


def main(argv=None) -> int:
    """CLI 入口（`scripts/tui_e2e_flow.py` 与 `python tests/tui_e2e_flow.py` 共用）。"""
    # ★ 2026-09-24 修复：Windows 控制台默认 GBK，报告里的 `⇒` 等符号会让 `print(text)` 抛
    #   `UnicodeEncodeError` ⇒ **整个报告打不出来**（实测：缺工件时 SKIP 原因含 `⇒` 即触发）。
    #   这里把标准输出/错误改成 UTF-8 + 容错替换，报告永远打得出来。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 老环境没有 reconfigure 就照旧
            pass

    parser = argparse.ArgumentParser(description="QLH TUI 端到端 flow（F1 档）")
    parser.add_argument("--mode", choices=("assert", "demo"), default="assert")
    parser.add_argument("--pace", type=float, default=1.2, help="demo 模式每步停顿秒数")
    parser.add_argument("--frames", default=None, help="demo 模式截图目录")
    parser.add_argument("--scenario", default=None, help="剧本 JSON 路径")
    parser.add_argument("--json-out", default=None, help="结果 JSON 落盘路径")
    parser.add_argument("--step", default=None, help="只跑某一个 step id")
    args = parser.parse_args(argv)

    frames = args.frames or (str(DEFAULT_FRAMES_DIR) if args.mode == "demo" else None)
    result = run_flow(mode=args.mode, pace=args.pace, frames_dir=frames,
                      flow_path=args.scenario, only_step=args.step)
    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")

    print(text)
    print(f"[tui-e2e] mode={result.mode} topology={result.topology} "
          f"physical_dual_host={result.physical_dual_host} "
          f"real_model_loaded={result.real_model_loaded} exit={result.exit_code()}")
    return result.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
