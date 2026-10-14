"""TUI 写操作面测试：协议层契约 + 确认闸门 + 模型屏/队列屏/聊天屏三个入口。

用户 2026-09-17 裁决的范围：**模型控制 + 会话管理 + 队列控制**，
入口为「屏内按键 + 聊天屏命令」混合。本文件锁定三件事：

1. 协议层每个写操作打到**后端真实端点**上（方法/路径/请求体/超时），
   路径参数必须 ``safe=""`` 编码（session_id 含 ``/`` 时不会多切一段 URL）；
2. **确认闸门**：破坏性/长耗时操作在确认前**不得**发出任何请求（dry-run 精神）；
3. 三个入口（模型屏 L/U、队列屏 P/S/C、聊天屏 /model /queue /new …）都接上了。

后端契约来源：api_server.py 实测（2026-09-17）
  POST /api/models/load     {engine, quant_type, use_compile, model_id}（耗时 5-20 秒）
  POST /api/models/unload
  POST /api/chat/clear
  POST /api/sessions / POST /sessions/{id}/activate / PUT /sessions/{id} / DELETE /sessions/{id}
  POST /cluster/queue/pause | resume | strategy{strategy} | clear
权限：model_api_access.require_model_api_source() 对 loopback 默认放行（本机 TUI 可直接调用）。
"""

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tui_api  # noqa: E402 - 纯标准库，始终可导入
from tui_api import ApiClient  # noqa: E402


class RecordingApi:
    """记录全部调用并返回预设结果的假后端（写操作不发真实请求）。"""

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

    def paths(self, method=None):
        return [call["path"] for call in self.calls
                if method is None or call["method"] == method]

    def find(self, path, method=None):
        for call in self.calls:
            if call["path"] == path and (method is None or call["method"] == method):
                return call
        return None

    # 与 ApiClient 一致的表面
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


# ------------------------------------------------------------------ 协议层契约

def test_model_control_calls_match_backend_contract():
    """模型加载/卸载：路径、请求体、放宽后的超时都必须对得上。"""
    api = RecordingApi()
    tui_api.load_model(api, "qwen-1_8b", engine="llama_cpp", quant_type="int4")
    tui_api.unload_model(api)

    load = api.find("/models/load", "POST")
    assert load is not None, api.calls
    assert load["body"] == {"engine": "llama_cpp", "quant_type": "int4",
                            "use_compile": False, "model_id": "qwen-1_8b"}
    assert load["timeout"] == tui_api.MODEL_CONTROL_TIMEOUT >= 120, "加载 5-20 秒，必须放宽超时"
    assert api.find("/models/unload", "POST") is not None


def test_headerless_load_omits_model_id():
    """不指定 model_id 时必须**不发**该键（后端按默认模型处理）。"""
    api = RecordingApi()
    tui_api.load_model(api)
    assert "model_id" not in api.find("/models/load", "POST")["body"]


def test_session_calls_quote_path_parameters():
    """session_id 含 "/" 时必须 %2F，否则会多切一段 URL。"""
    api = RecordingApi()
    tui_api.activate_session(api, "sess/1")
    tui_api.rename_session(api, "sess/1", "新标题")
    tui_api.delete_session(api, "sess/1")

    assert api.find("/sessions/sess%2F1/activate", "POST") is not None
    rename = api.find("/sessions/sess%2F1", "PUT")
    assert rename is not None and rename["body"] == {"title": "新标题"}
    assert api.find("/sessions/sess%2F1", "DELETE") is not None


def test_queue_calls_and_strategy_validation():
    """队列控制端点 + 非法策略必须被本地拒绝（不发请求）。"""
    api = RecordingApi()
    tui_api.pause_queue(api)
    tui_api.resume_queue(api)
    tui_api.set_queue_strategy(api, "fifo")
    tui_api.clear_queue(api)

    assert api.find("/cluster/queue/pause", "POST") is not None
    assert api.find("/cluster/queue/resume", "POST") is not None
    strategy = api.find("/cluster/queue/strategy", "POST")
    assert strategy is not None and strategy["body"] == {"strategy": "fifo"}
    assert api.find("/cluster/queue/clear", "POST") is not None

    before = len(api.calls)
    with pytest.raises(ValueError):
        tui_api.set_queue_strategy(api, "bogus")
    assert len(api.calls) == before, "非法策略不得发出请求"


def test_list_sessions_accepts_bare_list_and_wrapped():
    class Bare(RecordingApi):
        def get(self, path, params=None, with_log_token=False):
            return [{"session_id": "a"}, "junk", {"session_id": "b"}]

    class Wrapped(RecordingApi):
        def get(self, path, params=None, with_log_token=False):
            return {"sessions": [{"session_id": "c"}]}

    assert [item["session_id"] for item in tui_api.list_sessions(Bare())] == ["a", "b"]
    assert [item["session_id"] for item in tui_api.list_sessions(Wrapped())] == ["c"]


# ------------------------------------------------------------------ UI 写操作面

pytest.importorskip("textual")

from textual.widgets import DataTable, Input  # noqa: E402

from tui_textual import ConfirmScreen, KoakumaApp, MainScreen  # noqa: E402

#: 让"当前屏"与写操作数据都真实存在的最小后端形状
RESPONSES = {
    "/health": {"status": "ok"},
    "/status": {"model_loaded": False, "model_name": "Qwen/Qwen-1.8B-Chat",
                "active_model_id": None, "run_mode": "local", "node_role": "master",
                "node_id": "master", "max_nodes": 3},
    "/models": {"models": [{"model_id": "qwen-1_8b", "name": "Qwen-1.8B-Chat",
                            "available_formats": ["safetensors", "gguf"],
                            "preferred_engine": "pytorch", "is_available": True}],
                "active_model_id": None},
    "/cluster/queue": {"running": True, "paused": False, "strategy": "mlfq", "queue_size": 0,
                       "max_size": 100, "q0_depth": 0, "q1_depth": 0, "q2_depth": 0,
                       "q0": [], "q1": [], "q2": [], "completed_count": 0,
                       "aging_params": {"q0_max_tokens": 128, "q1_max_tokens": 512}},
    "/cluster/nodes/log-aggregate": {"local": {"node_id": "master", "logs": []},
                                     "workers": [], "total_workers": 0},
    "/sessions": {"session_id": "s-new", "title": "t"},
    "/cluster/queue/pause": {"paused": True},
    "/cluster/queue/resume": {"paused": False},
    "/cluster/queue/strategy": {"strategy": "fifo"},
    "/cluster/queue/clear": {"cleared": 0},
    "/models/load": {"loaded": True, "model_id": "qwen-1_8b"},
    "/models/unload": {"loaded": False},
}


def _run(coro):
    return asyncio.run(coro)


async def _wait_for(pilot, predicate, timeout=6.0):
    """轮询等待；切屏/挂载竞态期间查询可能抛 NoMatches，按"未就绪"处理。"""
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


async def _enter_main(pilot, app):
    """等主界面就绪。

    注意 ``ContentSwitcher`` **只挂载当前屏**，所以不能用 ``#models-table`` 之类
    的控件判断就绪（未显示时根本查不到）——改用不依赖挂载的控制器状态。
    """
    app.show_main()
    await _wait_for(pilot, lambda: isinstance(app.screen, MainScreen))
    await _wait_for(pilot, lambda: bool(app.screen.model_rows))
    return app.screen


def _make_app():
    app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5), interval=30)
    api = RecordingApi(RESPONSES)
    app.api = api
    return app, api


def test_confirm_screen_gates_both_answers():
    """确认框本身：y→True、n→False，且回调只在确认后触发。"""
    async def _main():
        app, _api = _make_app()
        answers = []
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(ConfirmScreen("标题", "正文"), answers.append)
            # ★ 用轮询等待替代单次 `pilot.pause()`：`push_screen` 的屏幕切换是
            #   异步的，全量跑时消息泵被拖慢，单次 pause 未必已经切过去 ⇒ 假红
            #   （实测单跑/整文件 6 次全绿，仅全量下 `app.screen` 仍是 MainScreen）。
            assert await _wait_for(
                pilot, lambda: isinstance(app.screen, ConfirmScreen)
            ), "确认屏未在限定时间内成为当前屏幕"
            await pilot.press("n")
            assert await _wait_for(pilot, lambda: answers), "取消也应回调（值 False）"
            assert answers == [False]

            app.push_screen(ConfirmScreen("标题", "正文"), answers.append)
            assert await _wait_for(
                pilot, lambda: isinstance(app.screen, ConfirmScreen)
            ), "第二个确认屏未在限定时间内成为当前屏幕"
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: len(answers) == 2)
            assert answers == [False, True]

    _run(_main())


def test_models_screen_load_cancelled_sends_nothing_then_confirmed_loads():
    """模型屏 L：取消 → 零请求；确认 → 按光标行的模型发 POST /models/load。"""
    async def _main():
        app, api = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _enter_main(pilot, app)
            screen.switch_page("models")
            await pilot.pause()
            table = screen.query_one("#models-table", DataTable)
            table.move_cursor(row=0, column=0)
            await pilot.pause()

            await pilot.press("l")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen), "长耗时+破坏性操作必须先确认"
            await pilot.press("n")
            await pilot.pause(0.3)
            assert "请先" not in screen.op_status
            assert api.find("/models/load") is None, "取消后不得发起加载"

            await pilot.press("l")
            await pilot.pause()
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/models/load") is not None)
            writes = api.paths("POST")
            assert writes.index("/models/unload") < writes.index("/models/load")
            assert api.find("/cluster/config/distributed-inference", "PUT")["body"] == {"enabled": False}
            body = api.find("/models/load")["body"]
            assert body["model_id"] == "qwen-1_8b"
            assert body["engine"] == "pytorch", "引擎应取注册表 preferred_engine"
            assert body["quant_type"] == "int4"

    _run(_main())


def test_models_screen_unload_confirms_and_posts():
    async def _main():
        app, api = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _enter_main(pilot, app)
            screen.switch_page("models")
            await pilot.pause()
            await pilot.press("u")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/models/unload") is not None)

    _run(_main())


def test_domain_screens_route_functional_operations_to_real_paths():
    """领域屏操作走专用合同；调试页不是模型/集群/日志功能的替代品。"""
    async def _main():
        app, api = _make_app()
        async with app.run_test(size=(130, 42)) as pilot:
            screen = await _enter_main(pilot, app)

            screen.switch_page("models")
            await pilot.pause()
            screen.submit_model_download({
                "preset_id": "qwen-0.5b",
                "source": "huggingface",
                "allow_cpu": "true",
            })
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/models/downloads", "POST") is not None)
            assert api.find("/models/downloads", "POST")["body"] == {
                "preset_id": "qwen-0.5b", "source": "huggingface", "allow_cpu": True,
            }

            screen.model_aux = {"downloads": {"jobs": [{"job_id": "job/1"}]}}
            screen.submit_model_cancel_download({"job_id": "job/1"})
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(
                pilot, lambda: api.find("/models/downloads/job%2F1", "DELETE") is not None)

            screen.switch_page("cluster")
            screen.cluster_aux = {"distributed": {"enabled": False}}
            screen.action_cluster_toggle()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(
                pilot, lambda: api.find("/cluster/config/distributed-inference", "PUT") is not None)
            assert api.find("/cluster/config/distributed-inference", "PUT")["body"] == {"enabled": True}

            screen.switch_page("nodes")
            api.responses["/cluster/join/request"] = {"request_code": "qlhjoinreq1.fixture"}
            screen.submit_cluster_join_request({
                "master_endpoint": "100.100.52.106:8888",
                "cluster_id": "qlh-default",
                "capabilities": "presence,task",
                "request_ttl_seconds": "600",
            })
            assert await _wait_for(pilot, lambda: api.find("/cluster/join/request", "POST") is not None)
            assert api.find("/cluster/join/request", "POST")["body"]["master_endpoint"] == "100.100.52.106:8888"

            screen.submit_cluster_join_grant({
                "request_code": "qlhjoinreq1.fixture",
                "otp_code": "123456",
                "ttl_seconds": "300",
            })
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/cluster/join/grant", "POST") is not None)
            assert api.find("/cluster/join/grant", "POST")["body"] == {
                "request_code": "qlhjoinreq1.fixture",
                "otp_code": "123456",
                "ttl_seconds": 300,
            }

            screen.submit_cluster_join_consume({"grant_code": "qlhjoin1.fixture"})
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/cluster/join/consume", "POST") is not None)

            screen.switch_page("logs")
            screen.action_logs_stats()
            assert await _wait_for(pilot, lambda: api.find("/logs/stats", "GET") is not None)

    _run(_main())


def test_quant_prefers_int4_over_fp16():
    """量化优先级：注册表给了 int4 就用 int4（后端默认量化，显存最省）。

    实测 2026-09-17：注册表项 quant_types 以 fp16 开头，若盲取首项会加载 fp16，
    与后端默认（int4）不一致且更吃显存。
    """
    from tui_textual import MainScreen

    screen = object.__new__(MainScreen)  # 纯逻辑方法，无需挂载
    args = MainScreen.model_load_args
    assert args(screen, {"preferred_engine": "pytorch",
                         "quant_types": ["fp16", "int8", "int4"]}) == ("pytorch", "int4")
    assert args(screen, {"preferred_engine": "llama_cpp",
                         "quant_types": ["fp16"]}) == ("llama_cpp", "fp16")
    assert args(screen, {}) == ("llama_cpp", "int4")
    assert args(screen, {"quant_types": {"int8": {}, "int4": {}}}) == ("llama_cpp", "int4")


def test_write_keys_are_inert_outside_their_page():
    """非目标屏按写操作键不得发起请求（避免"在别处按了 p 就暂停了队列"）。"""
    async def _main():
        app, api = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _enter_main(pilot, app)
            screen.switch_page("chat")
            await pilot.pause()
            await pilot.pause()
            for key in ("l", "u", "p", "c"):
                await pilot.press(key)
                await pilot.pause(0.1)
            assert not isinstance(app.screen, ConfirmScreen)
            assert api.find("/models/load") is None
            assert api.find("/cluster/queue/pause") is None
            assert api.find("/cluster/queue/clear") is None

    _run(_main())


def test_queue_screen_pause_strategy_and_clear():
    """队列屏：P 直接暂停（可逆），S 需确认，C 需确认且列出将清空内容。"""
    async def _main():
        app, api = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _enter_main(pilot, app)
            await _wait_for(pilot, lambda: bool(screen.queue_state))
            screen.switch_page("queue")
            await pilot.pause()

            await pilot.press("p")
            assert await _wait_for(pilot, lambda: api.find("/cluster/queue/pause") is not None)

            await pilot.press("s")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(
                pilot, lambda: api.find("/cluster/queue/strategy") is not None)
            assert api.find("/cluster/queue/strategy")["body"] == {"strategy": "fifo"}

            await pilot.press("c")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")
            await pilot.pause(0.3)
            assert api.find("/cluster/queue/clear") is None, "取消后不得清空"
            await pilot.press("c")
            await pilot.pause()
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/cluster/queue/clear") is not None)

    _run(_main())


def test_chat_commands_drive_model_queue_and_sessions():
    """/model、/queue、/new、/resume、/delete-session 都要真的打后端。"""
    async def _main():
        app, api = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_main(pilot, app)
            pane = app.screen.query_one("#chat-pane")
            box = pane.query_one("#chat-input", Input)

            async def submit(value):
                box.focus()
                box.value = value
                await pilot.press("enter")
                await pilot.pause()

            await submit("/queue pause")
            assert await _wait_for(pilot, lambda: api.find("/cluster/queue/pause") is not None)

            await submit("/queue strategy bogus")
            await pilot.pause(0.2)
            assert api.find("/cluster/queue/strategy") is None, "非法参数只提示，不发请求"

            await submit("/model load qwen-1_8b llama_cpp int4")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/models/load") is not None)

            await submit("/new 测试会话")
            assert await _wait_for(pilot, lambda: api.find("/sessions", "POST") is not None)
            assert api.find("/sessions", "POST")["body"] == {"title": "测试会话"}
            assert await _wait_for(pilot, lambda: app.session_id == "s-new")

            await submit("/delete-session")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/sessions/s-new", "DELETE") is not None)
            assert await _wait_for(pilot, lambda: app.session_id is None)

            await submit("/reset")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            assert await _wait_for(pilot, lambda: api.find("/chat/clear") is not None)

    _run(_main())


def test_help_lists_only_implemented_commands():
    """帮助由 COMMAND_SPECS 生成：登记的每条命令都必须有实现分支（无 /image 幽灵）。"""
    from tui_shared import COMMAND_SPECS

    names = {spec["name"] for spec in COMMAND_SPECS}
    for required in ("/model", "/queue", "/new", "/resume", "/rename", "/sessions",
                     "/delete-session", "/reset", "/clear", "/cancel", "/route",
                     "/thinking", "/help", "/quit"):
        assert required in names, required
    assert not {"image", "/image", "/images", "/image-clear"} & names, (
        "未实现的 /image* 不应登记（曾经 /help 写着却不能用）")

    async def _main():
        app, _api = _make_app()
        async with app.run_test(size=(120, 40)) as pilot:
            await _enter_main(pilot, app)
            pane = app.screen.query_one("#chat-pane")
            box = pane.query_one("#chat-input", Input)
            box.focus()
            box.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            for name in ("/model", "/queue", "/reset", "/delete-session"):
                assert name in pane.chat_buffer, name

    _run(_main())
