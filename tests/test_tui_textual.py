"""Textual 外壳 + 协议层的 headless 测试。

要点：
* ``tui_api``（协议层）是纯标准库，**没有** textual 也必须能 import 并正确报错；
* Textual 外壳在**后端不可用**时也必须能安全启动（只显示错误，不崩）；
* 十屏（聊天/状态/模型/分布式/节点/队列/日志/设备/设置/端点）侧栏导航与外层 Splash→Main 切换必须存在；
* 各屏按后端**真实字段**渲染（回归 2026-09-17 的接线错位），聊天后端错误必须可见。

缺 textual 时整个模块跳过（Edge 之外的极简环境仍可跑其余测试）。
"""

import asyncio
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tui_api import ApiClient, ApiError  # noqa: E402  - 纯标准库，始终可导入


# ------------------------------------------------------------------ 协议层

def test_client_reports_connection_failure_as_api_error():
    """无后端时必须抛 ApiError（界面据此显示错误，而不是崩栈）。"""
    api = ApiClient(host="127.0.0.1", port=1, timeout=0.5)
    with pytest.raises(ApiError):
        api.get("/health")


def test_chat_cancel_path_escapes_slash():
    """generation_id 含 "/" 时必须被编码，否则会多切一段 URL。"""
    from tui_api import API_PATHS
    import urllib.parse

    quoted = urllib.parse.quote("sess/1", safe="")
    assert quoted == "sess%2F1"
    assert "{generation_id}" in API_PATHS["chat_cancel"]


# ------------------------------------------------------------------ Textual 外壳

pytest.importorskip("textual")


def _run(coro):
    return asyncio.run(coro)


def test_shell_boots_then_switches_to_main():
    from textual.widgets import ContentSwitcher, ListView

    from tui_textual import KoakumaApp, MainScreen, SPLASH_SUBTITLE, SplashScreen

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        assert app.SUB_TITLE == SPLASH_SUBTITLE
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, SplashScreen), "应先显示启动屏"
            splash = app.screen
            from textual.widgets import Static

            status = str(splash.query_one("#splash-status", Static).render())
            assert "少女祈祷中" in status, "启动行应以「少女祈祷中：」开头"
            subtitle = splash.query_one("#splash-subtitle", Static)
            # 打字机是「前缀逐步变长」的动画：并行/慢 runner 上第一帧就可能已经
            # 打完，断言「此刻还没打完」必然偶发失败（2026-10-16 CI 实测）。
            # 不变量：任意时刻都是完整文案的前缀，且最终等于完整文案。
            # 这里只轮询到打完为止 —— 等久了启动条会先走完并把画面切到主界面，
            # 后面的 #splash-bar 断言就会 NoMatches。
            partial = str(subtitle.render())
            assert SPLASH_SUBTITLE.startswith(partial), (
                f"副标题必须是完整文案的前缀，实际 {partial!r}")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if str(subtitle.render()) == SPLASH_SUBTITLE:
                    break
                await pilot.pause(0.05)
            assert str(subtitle.render()) == SPLASH_SUBTITLE
            bar = str(splash.query_one("#splash-bar", Static).render())
            assert "█" in bar and "░" in bar, "标题下方应有跑马灯启动条"
            app.show_main()
            await pilot.pause()
            assert isinstance(app.screen, MainScreen), "应切换到主界面"
            nav = app.screen.query_one("#nav", ListView)
            sidebar = app.screen.query_one("#sidebar")
            content = app.screen.query_one("#content", ContentSwitcher)
            items = list(app.screen.query("#nav ListItem"))
            assert len(items) == 10, (
                "侧栏应有 聊天/状态/模型/分布式/节点/队列/日志/设备/设置/端点 十项")
            assert [item.id for item in items] == [
                "nav-chat", "nav-status", "nav-models", "nav-cluster", "nav-nodes",
                "nav-queue", "nav-logs", "nav-device", "nav-settings", "nav-api"]
            for widget_id in ("#nodes-table", "#queue-table", "#logs-log",
                              "#device-pane", "#gpu-table", "#settings-pane",
                              "#status-table", "#api-table", "#api-result", "#topbar"):
                assert app.screen.query_one(widget_id) is not None, widget_id
            settings = str(app.screen.query_one("#settings-pane", Static).render())
            assert "会话设置" in settings and "依赖边界" in settings, "设置屏应含会话参数与关于信息"

            # 版式：左窄右宽，分栏按黄金比例 ≈ 0.382 : 0.618
            await pilot.pause()
            total = sidebar.size.width + content.size.width
            assert 0 < sidebar.size.width < content.size.width, "导航栏应在左且窄于内容区"
            assert abs(sidebar.size.width / total - 0.382) < 0.03, (
                f"黄金比例分栏偏离过多: {sidebar.size.width}/{total}")

            # 键位切屏：[ ] 上下屏，侧栏高亮同步
            assert content.current == "page-chat"
            await pilot.press("]")
            await pilot.pause()
            assert content.current == "page-status", "应切到第二屏"
            assert nav.index == 1, "侧栏高亮应与内容同步"
            await pilot.press("[")
            await pilot.pause()
            assert content.current == "page-chat", "应切回第一屏"

    _run(_main())


def test_shell_survives_backend_absent():
    """后端不可达时：只读屏显示错误文本，界面照常可用。"""
    from textual.widgets import Static

    from tui_textual import KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        async with app.run_test() as pilot:
            app.show_main()
            await pilot.pause()
            pane = app.screen.query_one("#status-pane", Static)
            rendered = ""
            deadline = time.time() + 6
            while time.time() < deadline:
                await pilot.pause(0.1)
                rendered = str(pane.render())
                if "无法连接" in rendered or "后端不可达" in rendered:
                    break
            assert "后端不可达" in rendered or "无法连接" in rendered

    _run(_main())


#: 后端**真实**返回形状（2026-09-17 实测，保留关键字段名）——用于回归"接线是否对上"。
#: 早期实现沿用旧 TUI 假设（/models/current、resources.nodes、logs.lines、profile.os、
#: cpu.model、queue.tasks），真机下表现为"只有一行/不显示"。
LIVE_SHAPES = {
    "/health": {"status": "ok"},
    "/status": {
        "model_loaded": False, "model_name": "Qwen/Qwen-1.8B-Chat", "active_model_id": None,
        "engine": "", "run_mode": "distributed", "node_role": "master", "node_id": "master",
        "max_nodes": 3, "current_quant": "int4", "conversation_turns": 0,
        "pipeline_prepared": False,
    },
    "/models": {
        "models": [{"model_id": "qwen-1_8b", "name": "Qwen-1.8B-Chat",
                    "available_formats": ["safetensors", "gguf"],
                    "preferred_engine": "pytorch", "is_available": True,
                    "unavailable_reason": ""}],
        "active_model_id": None,
    },
    "/cluster/resources": {
        "scope": "cluster", "node_count": 1, "available_node_count": 1,
        "available": {
            "local": {"node_id": "master", "role": "NodeRole.MASTER", "state": "online",
                      "available": True,
                      "cpu": {"physical_cores": 14, "logical_cores": 20},
                      "ram": {"total_gb": 15.6, "available_gb": 6.5},
                      "gpu": {"count": 2, "cuda_count": 0, "vram_total_gb": 4.0,
                              "vram_free_gb": 0.0}},
            "remote": [],
        },
        "totals": {"physical_cores": 14, "logical_cores": 20, "ram_total_gb": 15.6,
                   "ram_available_gb": 6.5, "gpu_count": 2, "cuda_gpu_count": 0,
                   "vram_total_gb": 4.0, "vram_free_gb": 0.0},
    },
    "/cluster/nodes": {"nodes": [{"node_id": "master", "role": "master", "node_type": "pc",
                                  "state": "online", "address": "100.90.76.108:8888",
                                  "hostname": "localhost", "avg_rtt_ms": 0}],
                       "count": 1, "online_count": 1},
    "/cluster/control-plane": {
        "enabled": True, "available": True, "cluster_id": "qlh-test",
        "voter_set_epoch": 4, "committed_term": 12, "leader_id": "master",
        "certificate_digest": "a" * 64, "read_only_reason": "",
    },
    "/cluster/management-score": {
        "schema_version": "qlh.ha.management-score.v1",
        "algorithm_version": "management-v1",
        "nodes": [{"node_id": "master", "score": 70.0, "rank": 1}],
        "signature_status": "unsigned_no_cluster_secret",
    },
    "/cluster/transfer-logs": {
        "logs": [{"event_type": "handoff_committed", "reason": "maintenance"}],
        "count": 1,
    },
    "/cluster/queue": {"running":True,"paused":False,"strategy":"fifo","queue_size":1,
        "capacity":100,"active":{"request_id":"active","execution_state":"running","terminal":None},
        "tasks":[{"position":1,"request_id":"waiting","wait_seconds":0,"outcome":None}]},
    "/cluster/nodes/log-aggregate": {"local": {"node_id": "master",
                                               "logs": ["line-1", "line-2"]},
                                     "workers": [], "total_workers": 0, "limit": 50},
    "/device/profile": {
        "tier": "laptop", "tier_label": "游戏本 / 独显本", "score_total": 23.8,
        "platform": {"os": "Windows", "os_version": "10.0.19045", "hostname": "DESKTOP-X",
                     "machine": "AMD64", "architecture": "64bit", "python_version": "3.12.10"},
        "cpu": {"model_name": "13th Gen Intel(R) Core(TM) i9-13900H",
                "physical_cores": 14, "logical_cores": 20, "usage_percent": 10.3},
        "ram": {"total_gb": 15.6, "available_gb": 6.5, "percent_used": 58.7},
        "disk": {"free_gb": 121.1, "total_gb": 503.1, "path": "G:\\models"},
        "gpus": [{"name": "RTX 4060", "gpu_type": "discrete", "cuda_available": False,
                  "vram_total_gb": 4.0, "driver_version": ""}],
        "warnings": [], "recommendations": [],
    },
}


# Distributed selectors read the master's path-free catalog.
LIVE_SHAPES["/models/intent/catalog"] = {
    **LIVE_SHAPES["/models"], "binding": {"status": "connected", "master_node_id": "master"},
}
LIVE_SHAPES["/models/intent/current"] = None


def _rows(screen, selector):
    from textual.widgets import DataTable

    table = screen.query_one(selector, DataTable)
    return [[str(cell) for cell in table.get_row(key)] for key in table.rows]


class _StubApi:
    """按真实形状应答的假后端（暴露给多个用例复用）。"""

    base_url = "http://stub:8000"
    host = "127.0.0.1"
    port = 8000
    timeout = 5.0
    log_token = ""
    def __init__(self):
        self.calls = []

    def get(self, path, **kwargs):
        return LIVE_SHAPES.get(path, {})

    def get_openapi(self):
        return {
            "paths": {
                "/api/health": {"get": {"summary": "健康检查", "operationId": "health"}},
                "/api/models/load": {"post": {"summary": "加载模型", "operationId": "load_model"}},
                "/api/chat/stream": {"post": {"summary": "流式聊天", "operationId": "chat_stream"}},
                "/api/sessions/{session_id}": {
                    "get": {"summary": "会话详情", "operationId": "session_detail"},
                },
            }
        }

    def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return {"ok": True, "method": method, "path": path}


def test_live_backend_shapes_render_every_page():
    """回归：各页必须按后端**真实**字段渲染（2026-09-17 实测形状）。"""
    from textual.widgets import Static

    from tui_textual import KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5), interval=3)
        app.api = _StubApi()
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause(2.0)
            screen = app.screen

            models = _rows(screen, "#models-table")
            assert models and models[0][1] == "qwen-1_8b", models
            assert models[0][3] == "safetensors, gguf", models

            resources = _rows(screen, "#resources-table")
            assert resources[0][0] == "master" and "14 / 20" in resources[0][2], resources
            assert "合计" in resources[-1][0], resources

            nodes = _rows(screen, "#nodes-table")
            assert nodes[0][0] == "master" and "在线" in nodes[0][3], nodes

            queue = _rows(screen, "#queue-table")
            assert [row[0] for row in queue] == ["执行中", "1"], queue

            device = str(screen.query_one("#device-pane", Static).render())
            assert "Windows 10.0.19045" in device and "i9-13900H" in device, device
            assert screen.log_line_count == 2, screen.log_line_count

            status = " ".join(f"{row[0]}={row[1]}" for row in _rows(screen, "#status-table"))
            assert "运行模式=distributed" in status, status
            assert "尚未选择分布式模型" in status, status

    _run(_main())


def test_android_resource_projection_displays_unknown_physical_cores_as_dash():
    from tui_textual import KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5), interval=30)
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause()
            app.screen.fill_resources({
                "scope": "cluster",
                "node_count": 1,
                "available_node_count": 1,
                "available": {
                    "local": {
                        "node_id": "y700", "role": "client", "available": True,
                        "cpu": {"physical_cores": None, "logical_cores": 8},
                        "ram": {"total_gb": 12.0, "available_gb": 5.0},
                        "gpu": {"count": 0, "cuda_count": 0},
                    },
                    "remote": [],
                },
                "totals": {
                    "physical_cores": None, "logical_cores": 8,
                    "ram_total_gb": 12.0, "ram_available_gb": 5.0,
                },
            })
            rows = _rows(app.screen, "#resources-table")
            assert rows[0][2] == "— / 8", rows
            assert rows[0][3] == "12.0 / 5.0 GB", rows
            assert rows[-1][2] == "— / 8", rows

    _run(_main())


def test_ha_projection_is_read_only_and_fail_closed_for_missing_quorum():
    from tui_textual import KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5), interval=30)
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause()
            screen = app.screen
            screen.node_aux = {
                "nodes": [{"node_id": "master", "device_info": {"master_score_version": "v1"}}]
            }
            rendered = screen.render_ha_status({
                "role": {"node_role": "master", "auto_role": {
                    "state": "read_only", "reason": "quorum_unavailable",
                }},
                "fence": {
                    "enabled": True, "available": True, "committed_term": 12,
                    "voter_set_epoch": 4, "leader_id": "master",
                    "read_only_reason": "control_certificate_missing",
                },
                "transfer_logs": {"logs": []},
                "score": {"algorithm_version": "management-v1", "nodes": []},
            })
            assert "leader: master" in rendered
            assert "term/epoch: 12/4" in rendered
            assert "majority: unavailable" in rendered
            assert "read-only reason: control_certificate_missing" in rendered
            assert "score version: management-v1" in rendered
            assert "last handoff: none" in rendered

    _run(_main())


def test_ha_projection_uses_versioned_score_snapshot():
    from tui_textual import KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5), interval=30)
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause()
            screen = app.screen
            rendered = screen.render_ha_status({
                "score": {
                    "algorithm_version": "management-v1",
                    "nodes": [{"node_id": "master", "score": 70.0}],
                },
            })
            assert "score version: management-v1" in rendered

    _run(_main())


def test_health_failure_short_circuits_optional_refreshes():
    """健康探测失败时不能继续堆叠状态、节点、日志和设备请求。"""
    from tui_api import ApiError
    from tui_textual import KoakumaApp

    class _UnavailableApi(_StubApi):
        def get(self, path, **kwargs):
            self.calls.append(path)
            if path == "/health":
                raise ApiError("fixture backend unavailable")
            return super().get(path, **kwargs)

    async def _main():
        api = _UnavailableApi()
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.1), interval=30)
        app.api = api
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause(0.5)
            assert api.calls == ["/health"], api.calls
            assert app.screen.health_text == "[red]不可达[/]"

    _run(_main())


def test_pages_refresh_lock_allows_only_one_inflight_worker(monkeypatch):
    """Two redraw-triggered refreshes must collapse into one page fetch."""
    from tui_textual import KoakumaApp, MainScreen

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.1))
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause()
            screen = app.screen
            screen.backend_available = True
            screen.runtime_ready = True

            first_fetch = threading.Event()
            release = threading.Event()
            calls = []
            calls_lock = threading.Lock()

            def fetch_json(path, **_kwargs):
                with calls_lock:
                    calls.append(path)
                    is_first = len(calls) == 1
                if is_first:
                    # 让第一轮刷新一直停在"在途"，好让第二次调用**确定**撞上
                    # `_pages_inflight` 守卫。
                    first_fetch.set()
                    assert release.wait(5)
                return {}

            monkeypatch.setattr(screen, "fetch_json", fetch_json)
            monkeypatch.setattr(screen, "fetch_json_params", lambda *args, **kwargs: {})
            monkeypatch.setattr(app, "call_from_thread", lambda *args, **kwargs: None)

            # ★ 不要用"两个线程靠 Barrier 同时撞锁"来验证：线程调度顺序没有保证，
            #   若第二个线程在第一个释放锁**之后**才检查 `_pages_inflight`，它就会
            #   再抓一整轮 —— 实测该写法单跑 8 次即 1 次 flaky（calls 变成 6 项）。
            #   改为确定性构造：先启动一次刷新并等它真的进入在途，再在当前线程发
            #   第二次刷新；此时 `_pages_inflight` 必为 True ⇒ 必被折叠。
            loader = threading.Thread(
                target=MainScreen.load_pages.__wrapped__, args=(screen,),
            )
            loader.start()
            for _ in range(500):
                if first_fetch.is_set():
                    break
                await asyncio.sleep(0.01)
            assert first_fetch.is_set(), "首次页面刷新未进入在途状态"
            assert screen._pages_inflight is True

            MainScreen.load_pages.__wrapped__(screen)   # 第二次：应被折叠为 no-op

            release.set()
            loader.join(timeout=5)
            assert not loader.is_alive()

            assert calls == [
                "/cluster/nodes",
                "/cluster/queue",
                "/cluster/nodes/log-aggregate",
            ]
            assert screen._pages_inflight is False

    _run(_main())


def test_runtime_readiness_failure_short_circuits_optional_refreshes():
    """A live API must not fan out while its runtime components are starting."""
    from tui_textual import KoakumaApp

    class _StartingApi(_StubApi):
        def get(self, path, **kwargs):
            self.calls.append(path)
            if path == "/ready":
                return {
                    "process_ready": True,
                    "ready": False,
                    "status": "starting",
                    "components": {
                        "local_store": True,
                        "scheduler": False,
                        "device_profile": False,
                    },
                }
            return super().get(path, **kwargs)

    async def _main():
        api = _StartingApi()
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.1), interval=30)
        app.api = api
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause(0.5)
            assert api.calls == ["/health", "/ready"], api.calls
            assert "初始化中" in app.screen.model_text

    _run(_main())


def _pane_text(widget) -> str:
    """取 `Static` 当前呈现的文本（Textual 8 用 `content`；旧版用 `renderable`/`_renderable`）。"""
    for attr in ("content", "renderable", "_renderable"):
        value = getattr(widget, attr, None)
        if value is not None:
            return str(value)
    return ""


def test_status_table_renders_model_and_pipeline_independently():
    from tui_textual import KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.1), interval=30)
        app.api = _StubApi()
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause(0.4)
            screen = app.screen
            screen.model_mode = "local"
            for loaded, prepared in ((False, False), (False, True), (True, False), (True, True)):
                screen.fill_status(
                    {"status": "ok"},
                    {
                        "model_loaded": loaded,
                        "pipeline_prepared": prepared,
                        "model_name": "Qwen",
                        "active_model_id": "qwen" if loaded else None,
                    },
                    {},
                    {},
                )
                rows = {name: value for name, value in _rows(screen, "#status-table")}
                assert rows["模型"] == f"Qwen（{'已加载' if loaded else '未加载'}）"
                assert rows["流水线"] == ("已准备" if prepared else "未准备")

    _run(_main())


def test_model_loading_summary_is_consistent_with_table():
    """★ 2026-10-09（#73-②，用户截图实证）：加载期间**摘要 pane** 必须与 `model_text` 同源。

    此前只改了 `model_text` 与状态表，漏了 `#status-pane` 的摘要 ⇒ 加载中顶部仍显示
    「模型未加载」，与下面「加载中…」**同屏矛盾**（这正是用户截图里的 ②）。
    """
    from textual.widgets import Static

    from tui_textual import KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.1), interval=30)
        app.api = _StubApi()
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause(0.4)
            screen = app.screen
            # 模拟"加载已开始、但 /api/status 还没更新"的窗口 —— 矛盾就出现在这个时机。
            screen.model_mode = "local"
            screen._model_loading_id = "qwen3-5-2b"
            screen.fill_status(
                {"status": "ok"},
                {"model_loaded": False, "model_name": "Qwen3.5-2B", "active_model_id": None},
                {},
                {},
            )
            await pilot.pause(0.2)
            assert "加载中" in screen.model_text, screen.model_text
            summary = _pane_text(screen.query_one("#status-pane", Static))
            assert "加载中" in summary, summary
            assert "模型未加载" not in summary, (
                f"加载期间摘要不得显示『模型未加载』（与『加载中』矛盾）：{summary}"
            )
            # 加载结束后必须回到常规呈现（不能卡在"加载中"）。
            screen._model_loading_id = ""
            screen.fill_status(
                {"status": "ok"},
                {"model_loaded": True, "model_name": "Qwen3.5-2B", "active_model_id": "qwen3-5-2b"},
                {},
                {},
            )
            await pilot.pause(0.2)
            done = _pane_text(screen.query_one("#status-pane", Static))
            assert "加载中" not in done, done
            assert "模型已加载" in done, done

    _run(_main())


def test_endpoint_workbench_discovers_and_executes_get():
    """端点页跟随 OpenAPI，GET 可执行，路径模板不能被误发。"""
    from textual.widgets import Input

    from tui_textual import KoakumaApp

    async def _main():
        stub = _StubApi()
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        app.api = stub
        async with app.run_test(size=(140, 48)) as pilot:
            app.show_main()
            await pilot.pause(1.0)
            screen = app.screen
            screen.switch_page("api")
            await pilot.pause(0.5)
            assert len(screen.api_operations) == 4
            assert "GET /api/health" in screen.api_operations
            assert screen.query_one("#api-path", Input).value == "/health"

            screen.api_selected_key = "GET /api/health"
            screen.query_one("#api-path", Input).value = "/health"
            screen.action_api_execute()
            deadline = time.time() + 3
            while time.time() < deadline and not stub.calls:
                await pilot.pause(0.1)
            assert stub.calls and stub.calls[-1][0:2] == ("GET", "/health")

            screen.api_selected_key = "POST /api/models/load"
            screen.select_api_operation(screen.api_selected_key)
            screen.action_api_execute()
            await pilot.pause(0.2)
            assert app.screen.__class__.__name__ == "ConfirmScreen"
            await pilot.press("n")
            await pilot.pause(0.1)
            assert len(stub.calls) == 1, "写操作必须先经过确认屏"

            screen.api_selected_key = "GET /api/sessions/{session_id}"
            screen.select_api_operation(screen.api_selected_key)
            screen.action_api_execute()
            await pilot.pause(0.2)
            assert len(stub.calls) == 1, "未替换的路径参数不应发送请求"

    _run(_main())


def test_chat_done_error_is_surfaced(monkeypatch):
    """回归：``/chat/stream`` 的 ``done.error``（如"模型加载失败"）必须显示出来。

    早先只看 done.response/metrics，会把错误吞掉 → 用户看到"没有回复也没有原因"。
    """
    from textual.widgets import Input, Static

    import tui_textual as tt

    def fake_stream(api, message, **kwargs):
        yield {"start": True, "generation_id": "gen-1"}
        yield {"done": True, "error": "本地回退模型加载失败: llama-cpp-python 未安装"}

    monkeypatch.setattr(tt, "iter_chat_payloads", fake_stream)

    async def _main():
        app = tt.KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        async with app.run_test() as pilot:
            app.show_main()
            await pilot.pause()
            pane = app.screen.query_one("#chat-pane")
            box = pane.query_one("#chat-input", Input)
            box.focus()
            box.value = "你好"
            await pilot.press("enter")
            rendered = ""
            deadline = time.time() + 6
            while time.time() < deadline:
                await pilot.pause(0.1)
                rendered = str(pane.query_one("#chat-status", Static).render())
                if "错误" in rendered:
                    break
            assert "后端错误" in rendered, rendered
            assert "llama-cpp-python" in rendered or "模型加载失败" in rendered, rendered

    _run(_main())


def test_chat_pane_help_and_unknown_command():
    from textual.widgets import Input, Static

    from tui_textual import ChatPane, KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        async with app.run_test() as pilot:
            app.show_main()
            await pilot.pause()
            pane = app.screen.query_one("#chat-pane", ChatPane)
            box = pane.query_one(Input)
            box.value = "/help"
            box.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert "可用命令" in pane.chat_buffer
            assert pane.query_one("#chat-log", Static) is not None
            # 未识别命令不应触发网络请求，只提示
            box.value = "/nonexistent"
            box.focus()
            await pilot.press("enter")
            await pilot.pause()

    _run(_main())


def test_route_command_updates_preference():
    from textual.widgets import Input

    from tui_textual import ChatPane, KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        async with app.run_test() as pilot:
            app.show_main()
            await pilot.pause()
            pane = app.screen.query_one("#chat-pane", ChatPane)
            box = pane.query_one(Input)
            box.value = "/route distributed"
            box.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.routing_preference == "distributed_preferred"

    _run(_main())


def test_models_table_cursor_is_preserved_across_refresh():
    """★ 2026-09-19 回归：模型表刷新后光标必须**停在同一个 row key**。

    用户报障：「选择了其他模型光标还在第一位，然后加载成 1.8B」。
    成因：`fill_models` 每轮后台刷新都 `clear()` 后重建，光标跳回第 0 行；用户按**视觉记忆**
    选好行再按 `L`，`action_load_model` 读的却是光标行 ⇒ 加载了**列表第一项**
    （历史上第一位正是 `qwen-1_8b`）。修法：按 row key 记住并恢复（不依赖行序）。
    """
    from textual.widgets import DataTable

    from tui_textual import KoakumaApp

    registry = {
        "models": [
            {"model_id": "qwen3-0.6b", "name": "Qwen3-0.6B", "is_available": True},
            {"model_id": "qwen3-5-2b", "name": "Qwen3.5-2B", "is_available": True},
            {"model_id": "qwenseek-2b", "name": "QwenSeek-2B", "is_available": True},
        ],
        "active_model_id": "qwen3-0.6b",
    }

    async def _main():
        # 周期性后台刷新（action_reload）会在慢 runner 上抢在断言前用 stub 的
        # /models 冷数据覆盖下面注入的 registry（2026-10-16 CI 实测：表里出现
        # 只有一项的 qwen-1_8b）。本用例只关心「刷新后光标保留」，所以关掉定时器，
        # 由显式的 fill_models 触发刷新。
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5), interval=3600)
        app.api = _StubApi()
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_main()
            await pilot.pause(2.0)
            screen = app.screen
            table = screen.query_one("#models-table", DataTable)

            screen.fill_models(registry, {})
            await pilot.pause()
            assert table.row_count == 3

            # 用户选中第 3 行（qwenseek-2b）
            table.move_cursor(row=2)
            await pilot.pause()
            assert screen.selected_model_id(table) == "qwenseek-2b"

            # 触发一次「后台刷新」（同一份数据重建表格）
            screen.fill_models(registry, {})
            await pilot.pause()

            assert table.cursor_coordinate.row == 2, "刷新后光标不应跳回第一行"
            assert screen.selected_model_id(table) == "qwenseek-2b", (
                "刷新后仍应指向用户选择的模型（否则会误加载列表第一项）"
            )

    _run(_main())
