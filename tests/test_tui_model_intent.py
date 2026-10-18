"""Acceptance through the main Textual entry: selection, recovery, races and chat."""
import asyncio
from copy import deepcopy
import threading
import time

import pytest

pytest.importorskip("textual")
from textual.widgets import Button, DataTable, Input, Static
import tui_textual as tt
from tui_api import ApiError, ApiClient

OP = "supply_" + "a" * 32
NEXT_OP = "supply_" + "b" * 32


def operation(status="supplying", model_id="master-big", operation_id=OP):
    return {"schema_version": 2, "revision": 7, "identity": {"boot_id": "tui-test"}, "observed_at": 100, "validity": "valid", "runnable": status == "ready", "operation_id": operation_id, "intent": {"model_id": model_id}, "status": status,
            "reason_code": "", "reason": "", "config_id": "config-8", "generation": 8,
            "loaded_ready": status == "ready",
            "nodes": {"worker-1": {"status": "receiving", "bytes_received": 13,
                                     "total_bytes": 100, "layer_range": [5, 12],
                                     "reason_code": "", "reason": ""}}}


class IntentApi:
    accept_snapshot = ApiClient.accept_snapshot
    base_url = "http://local-worker:8000"
    host = "127.0.0.1"
    port = 8000
    timeout = 5.0
    log_token = ""

    def __init__(self, current=None):
        self._snapshot_cache = {}
        self.current = current
        self.calls = []
        self.mode = "distributed"
        self.binding_error = False
        self.connect_error = False
        self.block_path = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.catalog = {
            "binding": {"status": "connected", "master_node_id": "master-registered"},
            "models": [{"model_id": "master-big", "name": "Master Big", "is_available": True,
                        "preferred_engine": "llama_cpp", "quant_types": ["int4", "Q4_K_M"],
                        "default_quant_type": "Q4_K_M", "available_formats": ["gguf"]},
                       {"model_id": "master-small", "is_available": True,
                        "preferred_engine": "llama_cpp", "default_quant_type": "Q4_K_M"}],
        }

    def get(self, path, **kwargs):
        self.calls.append(("GET", path, None))
        if path in {"/models/intent/current", f"/models/intent/{OP}", f"/models/intent/{NEXT_OP}"}:
            value = deepcopy(self.current)
            if path == self.block_path:
                self.block_path = None
                self.entered.set()
                assert self.release.wait(8), "blocked poll was not released"
            return value
        if path == "/models/intent/catalog":
            if self.binding_error:
                raise ApiError("INTENT_BINDING_MISSING: registered master disconnected", 409,
                               reason_code="INTENT_BINDING_MISSING")
            return deepcopy(self.catalog)
        return deepcopy({
            "/health": {"status": "ok"}, "/ready": {"ready": True},
            "/cluster/config/distributed-inference": {"enabled": self.mode == "distributed"},
            # An old local/current poll reports an unrelated whole model as loaded.
            "/status": {"run_mode": self.mode, "model_loaded": True, "active_model_id": "old-local",
                         "model_name": "Old Whole Model", "node_role": "client", "pipeline_prepared": True},
            "/models": {"models": [{"model_id": "worker-only", "is_available": True,
                                      "preferred_engine": "llama_cpp"}]},
        }.get(path, {}))

    def request(self, method, path, body=None, **kwargs):
        self.calls.append((method, path, body))
        if method == "PUT" and path == "/cluster/config/distributed-inference":
            self.mode = "distributed" if body["enabled"] else "local"
        if method == "POST" and path == "/models/intent":
            self.current = operation("queued", body["model_id"], NEXT_OP)
            return deepcopy(self.current)
        if method == "POST" and path == "/models/intent/connect":
            if self.connect_error:
                raise ApiError("INTENT_MASTER_NOT_FOUND: discovery failed", 409,
                               reason_code="INTENT_MASTER_NOT_FOUND")
            self.binding_error = False
            return deepcopy(self.catalog["binding"])
        if method == "DELETE" and path.startswith("/models/intent/"):
            self.current["status"] = "cancelled"
            self.current["reason_code"] = "INTENT_CANCELLED"
            return deepcopy(self.current)
        return {"success": True, "loaded": path == "/models/load"}

    def delete(self, path, **kwargs):
        return self.request("DELETE", path, **kwargs)


async def wait(pilot, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    assert predicate(), "main product event did not finish"


async def enter(app, pilot):
    app.show_main()
    await wait(pilot, lambda: isinstance(app.screen, tt.MainScreen) and bool(app.screen.model_rows))
    return app.screen


def content(screen, selector):
    return str(screen.query_one(selector, Static).content)


def test_main_model_selection_uses_master_intent_and_holds_chat_until_ready(monkeypatch):
    streamed = []
    def stream(api, message, **kwargs):
        streamed.append((message, kwargs))
        yield {"start": True, "generation_id": "gen-1"}
        yield {"token": "answer"}
        yield {"done": True, "response": "answer", "metrics": {"distributed_used": True}}
    monkeypatch.setattr(tt, "iter_chat_payloads", stream)

    async def run():
        api = IntentApi()
        app = tt.KoakumaApp(api, interval=30)
        app.routing_preference = "distributed_required"
        app.execution_mode = "auto"
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            assert set(screen.model_rows) == {"master-big", "master-small"}
            screen.switch_page("models")
            table = screen.query_one("#models-table", DataTable)
            table.focus()
            await pilot.press("l")
            await wait(pilot, lambda: isinstance(app.screen, tt.ConfirmScreen))
            await pilot.press("n")
            assert not any(method == "POST" for method, _, _ in api.calls)
            await pilot.press("l", "y")
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP)
            writes = [(method, path, body) for method, path, body in api.calls if method != "GET"]
            assert writes == [("POST", "/models/intent", {
                "model_id": "master-big", "engine": "llama_cpp", "quant_type": "Q4_K_M",
                "strategy": "auto", "timeout_seconds": 3600})]
            assert "queued" in content(screen, "#intent-pane")
            screen.switch_page("chat")
            box = screen.query_one("#chat-input", Input)
            box.focus()
            box.value = "hello"
            await pilot.press("enter")
            assert not streamed
            assert box.value == "hello", "a gated message must remain editable"
            assert "queued" in content(screen, "#chat-status")
            api.current = operation("preparing", operation_id=NEXT_OP)
            screen.action_refresh()
            await wait(pilot, lambda: screen.model_intent.get("status") == "preparing")
            assert "13/100 bytes" in content(screen, "#intent-pane")
            api.current = operation("ready", operation_id=NEXT_OP)
            screen.action_refresh()
            await wait(pilot, lambda: not screen.chat_model_gate())
            assert "就绪" in screen.model_text
            assert "模型未加载" not in content(screen, "#status-pane")
            await pilot.press("enter")
            await wait(pilot, lambda: bool(streamed))
            assert streamed[0][1]["routing_preference"] == "distributed_required"
            assert streamed[0][1]["execution_mode"] == "auto"
            assert not any(path in {"/models", "/models/load", "/models/current"} or path.endswith("/prepare")
                           for _, path, _ in api.calls)
    asyncio.run(run())


def test_restart_restores_same_operation_cancel_and_ready_cannot_delete():
    async def run():
        api = IntentApi(operation())
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == OP)
            screen.switch_page("models")
            count = sum(path == "/models/intent/current" for _, path, _ in api.calls)
            screen.action_refresh()
            await wait(pilot, lambda: sum(path == "/models/intent/current" for _, path, _ in api.calls) > count)
            assert not any(method == "POST" and path == "/models/intent" for method, path, _ in api.calls)
            await pilot.click("#model-intent-cancel")
            await wait(pilot, lambda: isinstance(app.screen, tt.ConfirmScreen))
            await pilot.press("y")
            await wait(pilot, lambda: screen.model_intent.get("status") == "cancelled")
            assert ("DELETE", f"/models/intent/{OP}", None) in api.calls
            assert "INTENT_CANCELLED" in content(screen, "#intent-pane")
            api.current = operation("ready")
            screen.action_refresh()
            await wait(pilot, lambda: screen.model_intent.get("status") == "ready")
            assert screen.query_one("#model-intent-cancel", Button).disabled
            count = sum(method == "DELETE" for method, _, _ in api.calls)
            screen.action_cancel_model_intent()
            await pilot.pause()
            assert sum(method == "DELETE" for method, _, _ in api.calls) == count
    asyncio.run(run())


@pytest.mark.parametrize("choice", ["local", "master-small"])
def test_late_old_current_poll_cannot_overwrite_user_mode_or_selection(choice):
    async def run():
        api = IntentApi(operation("ready"))
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == OP and not screen._core_inflight)
            screen.switch_page("models")
            api.block_path = "/models/intent/current"
            screen.action_refresh()
            await wait(pilot, api.entered.is_set)
            try:
                if choice == "local":
                    await pilot.click("#model-mode-local")
                else:
                    table = screen.query_one("#models-table", DataTable)
                    table.move_cursor(row=1)
                    table.focus()
                    await pilot.press("l")
                await wait(pilot, lambda: isinstance(app.screen, tt.ConfirmScreen))
                await pilot.press("y")
                await wait(pilot, lambda: not screen._model_action_inflight)
            finally:
                api.release.set()
            if choice == "local":
                await wait(pilot, lambda: "worker-only" in screen.model_rows)
                assert screen.model_mode == "local"
                assert not screen.model_intent
                status_table = screen.query_one("#status-table", DataTable)
                status_rows = {status_table.get_row(key)[0]: status_table.get_row(key)[1] for key in status_table.rows}
                assert status_rows["运行模式"] == "local"
                assert status_rows["流水线"] == "未准备"
                writes = [(method, path, body) for method, path, body in api.calls if method != "GET"]
                assert writes[:2] == [("PUT", "/cluster/config/distributed-inference", {"enabled": False}),
                                      ("POST", "/models/unload", None)]
                table = screen.query_one("#models-table", DataTable)
                table.focus()
                await pilot.press("l", "y")
                await wait(pilot, lambda: any(path == "/models/load" for _, path, _ in api.calls))
                assert screen.model_selection_id == "worker-only"
            else:
                await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP)
                assert screen.model_mode == "distributed"
                assert screen.model_selection_id == "master-small"
                assert screen.model_intent["intent"]["model_id"] == "master-small"
                assert screen.selected_model_id(screen.query_one("#models-table", DataTable)) == "master-small"
            # Exclusive workers can retire before their blocked thread drains.
            # Let its queued replacement start, then drain the core read before unmount.
            await wait(pilot, lambda: not screen._core_inflight)
            await pilot.pause()
            await wait(pilot, lambda: not screen._core_inflight and not screen._core_refresh_pending)
            await app.workers.wait_for_complete()
    asyncio.run(run())


def test_disconnected_master_named_retry_posts_empty_body_without_form():
    async def run():
        api = IntentApi(operation("ready"))
        api.binding_error = True
        api.connect_error = True
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            app.show_main()
            await wait(pilot, lambda: bool(app.screen.model_binding))
            screen = app.screen
            screen.switch_page("models")
            await pilot.pause()
            assert "INTENT_BINDING_MISSING" in content(screen, "#intent-pane")
            await pilot.click("#model-intent-connect")
            await wait(pilot, lambda: bool(screen._model_error))
            assert "INTENT_MASTER_NOT_FOUND" in content(screen, "#intent-pane")
            assert isinstance(app.screen, tt.MainScreen), "no host form should open"
            api.connect_error = False
            await wait(pilot, lambda: not screen.query_one("#model-intent-connect", Button).has_class("-active"))
            await pilot.click("#model-intent-connect")
            await wait(pilot, lambda: screen.model_binding.get("status") == "connected" and not screen._model_error)
            assert all(body == {} for method, path, body in api.calls if path == "/models/intent/connect")
            assert not any(path == "/cluster/connect" for _, path, _ in api.calls)
            assert app.api.host == "127.0.0.1"
    asyncio.run(run())


def test_chat_model_command_uses_same_intent_controller():
    async def run():
        api = IntentApi()
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            box = screen.query_one("#chat-input", Input)
            box.focus()
            box.value = "/model load master-small llama_cpp Q4_K_M"
            await pilot.press("enter")
            await wait(pilot, lambda: isinstance(app.screen, tt.ConfirmScreen))
            await pilot.press("y")
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP)
            assert screen.model_selection_id == "master-small"
            box.focus()
            box.value = "/model cancel"
            await pilot.press("enter")
            await wait(pilot, lambda: isinstance(app.screen, tt.ConfirmScreen))
            await pilot.press("y")
            await wait(pilot, lambda: screen.model_intent.get("status") == "cancelled")
            assert not any(path == "/models/load" for _, path, _ in api.calls)
            await wait(pilot, lambda: not screen._core_inflight)
    asyncio.run(run())


@pytest.mark.parametrize("status", ["failed", "cancelled", "superseded"])
def test_terminal_operation_reason_remains_visible_and_chat_gated(status):
    async def run():
        current = operation(status)
        current.update(reason_code="INTENT_PLAN_RETIRED", reason="worker left cluster")
        current["nodes"]["worker-1"].update(status="failed", reason_code="WORKER_PREPARE_FAILED", reason="load failed")
        api = IntentApi(current)
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            text = content(screen, "#intent-pane")
            for fragment in (status, "INTENT_PLAN_RETIRED", "worker left cluster", "WORKER_PREPARE_FAILED", "load failed"):
                assert fragment in text
            assert screen.chat_model_gate()
            assert status in content(screen, "#chat-status")
    asyncio.run(run())


def test_unload_does_not_restore_an_old_ready_selection():
    async def run():
        api = IntentApi(operation("ready"))
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            screen.switch_page("models")
            screen.query_one("#models-table", DataTable).focus()
            await pilot.press("u", "y")
            await wait(pilot, lambda: not screen._model_action_inflight and any(path == "/models/unload" for _, path, _ in api.calls))
            screen.action_refresh()
            await wait(pilot, lambda: not screen._core_inflight)
            assert not screen.model_intent
            assert screen.chat_model_gate()
            assert not any(method == "DELETE" for method, _, _ in api.calls)
            # Suppressing that unloaded operation must not hide another device's new selection.
            writes = [call for call in api.calls if call[0] != "GET"]
            api.current = operation("queued", "master-small", NEXT_OP)
            screen.action_refresh()
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP)
            assert screen.model_intent["intent"]["model_id"] == "master-small"
            assert screen._intent_restore_enabled
            assert screen.chat_model_gate()
            assert [call for call in api.calls if call[0] != "GET"] == writes
    asyncio.run(run())


def test_selecting_another_model_cancels_pending_operation_before_new_intent():
    async def run():
        api = IntentApi(operation("preparing"))
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            screen.switch_page("models")
            table = screen.query_one("#models-table", DataTable)
            table.move_cursor(row=1)
            table.focus()
            await pilot.press("l", "y")
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP)
            writes = [(method, path) for method, path, _ in api.calls if method != "GET"]
            assert writes == [("DELETE", f"/models/intent/{OP}"), ("POST", "/models/intent")]
            assert screen.model_selection_id == "master-small"
    asyncio.run(run())


@pytest.mark.parametrize("binding_status", ["not_found", "local"])
def test_binding_state_blocks_even_a_previously_ready_operation(binding_status):
    async def run():
        api = IntentApi(operation("ready"))
        api.catalog["binding"] = {"status": binding_status, "reason_code": "INTENT_MASTER_DISCONNECTED",
                                  "reason": "bootstrap connection retired", "node_id": "this-worker"}
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            assert screen.chat_model_gate()
            assert "INTENT_MASTER_DISCONNECTED" in content(screen, "#chat-status")
            assert "主节点: this-worker" not in content(screen, "#intent-pane")
    asyncio.run(run())



def test_poll_started_during_cancel_cannot_restore_preparing_after_delete():
    class DelayedDelete(IntentApi):
        def __init__(self):
            super().__init__(operation("preparing"))
            self.delete_entered = threading.Event()
            self.delete_release = threading.Event()
        def request(self, method, path, **kwargs):
            if method == "DELETE":
                self.delete_entered.set()
                assert self.delete_release.wait(8)
            return super().request(method, path, **kwargs)

    async def run():
        api = DelayedDelete()
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            await wait(pilot, lambda: not screen._core_inflight)
            screen.switch_page("models")
            screen.query_one("#models-table", DataTable).focus()
            await pilot.press("c", "y")
            await wait(pilot, api.delete_entered.is_set)
            api.block_path = "/models/intent/current"
            screen.action_refresh()
            try:
                await wait(pilot, api.entered.is_set)
                api.delete_release.set()
                await wait(pilot, lambda: screen.model_intent.get("status") == "cancelled")
            finally:
                api.delete_release.set()
                api.release.set()
            await wait(pilot, lambda: not screen._core_inflight)
            assert screen.model_intent["status"] == "cancelled"
            assert screen.chat_model_gate()
    asyncio.run(run())


@pytest.mark.parametrize("old_status, external_model", [("failed", "master-small"), ("ready", "master-big")])
def test_external_current_replaces_terminal_operation_without_moving_catalog_or_writing(old_status, external_model):
    async def run():
        api = IntentApi()
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            screen.switch_page("models")
            table = screen.query_one("#models-table", DataTable)
            table.focus()
            await pilot.press("l", "y")
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP and not screen._core_inflight)
            assert screen.model_selection_id == "master-big"
            api.current = operation(old_status, operation_id=NEXT_OP)
            api.current.update(reason_code="SUPPLY_WORKER_DISCONNECTED", reason="old connection retired")
            screen.action_refresh()
            await wait(pilot, lambda: screen.model_intent.get("status") == old_status and not screen._core_inflight)
            # The catalog cursor is deliberately different from the externally selected model.
            sticky_model = "master-big" if external_model == "master-small" else "master-small"
            table.move_cursor(row=0 if sticky_model == "master-big" else 1)
            writes = [call for call in api.calls if call[0] != "GET"]
            epoch = screen._selection_epoch
            # Reorder the catalog as well: restoration must use the key, not the row number.
            api.catalog["models"].reverse()
            for status in ("queued", "preparing", "ready"):
                api.current = operation(status, external_model, OP)
                screen.action_refresh()
                await wait(pilot, lambda: screen.model_intent.get("operation_id") == OP
                           and screen.model_intent.get("status") == status and not screen._core_inflight)
                assert screen.selected_model_id(table) == sticky_model
                assert screen.model_selection_id == "master-big", "current must not rewrite the local requested model"
                assert screen._selection_epoch == epoch, "a refresh is not a selection action"
                assert screen.model_intent["intent"]["model_id"] == external_model
                assert external_model in screen.model_text
                assert OP in content(screen, "#intent-pane")
                assert "old connection retired" not in content(screen, "#intent-pane")
                assert bool(screen.chat_model_gate()) == (status != "ready")
                assert "◆" in str(table.get_row(external_model)[0])
                assert "◆" not in str(table.get_row(sticky_model)[0])
                assert [call for call in api.calls if call[0] != "GET"] == writes
            assert "光标仅浏览" in content(screen, "#models-pane")
            assert "当前 operation" in content(screen, "#models-pane")
            assert not any(method == "GET" and path in {f"/models/intent/{OP}", f"/models/intent/{NEXT_OP}"}
                           for method, path, _ in api.calls)
    asyncio.run(run())


def test_authoritative_empty_current_clears_stale_ready_and_keeps_catalog_cursor():
    async def run():
        api = IntentApi(operation("ready"))
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            screen.switch_page("models")
            table = screen.query_one("#models-table", DataTable)
            table.move_cursor(row=1)
            assert not screen.chat_model_gate()
            api.current = None
            screen.action_refresh()
            await wait(pilot, lambda: not screen.model_intent and not screen._core_inflight)
            assert screen.selected_model_id(table) == "master-small"
            assert screen.chat_model_gate()
            assert "尚未选择分布式模型" in content(screen, "#intent-pane")
            assert all("◆" not in str(table.get_row(key)[0]) for key in table.rows)
            assert all(method == "GET" for method, _, _ in api.calls)
    asyncio.run(run())


@pytest.mark.parametrize("release_order", ["read_first", "write_first"])
def test_current_read_during_submit_cannot_override_local_write(release_order, monkeypatch):
    class DelayedPost(IntentApi):
        def __init__(self):
            super().__init__(operation("ready"))
            self.post_entered = threading.Event()
            self.post_release = threading.Event()
        def request(self, method, path, **kwargs):
            if method == "POST" and path == "/models/intent":
                self.post_entered.set()
                assert self.post_release.wait(8)
            return super().request(method, path, **kwargs)

    async def run():
        api = DelayedPost()
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            await wait(pilot, lambda: not screen._core_inflight)
            screen.switch_page("models")
            table = screen.query_one("#models-table", DataTable)
            table.move_cursor(row=1)
            table.focus()
            await pilot.press("l", "y")
            await wait(pilot, api.post_entered.is_set)
            rendered_ids = []
            render = screen.render_model_intent
            def record():
                rendered_ids.append(screen.model_intent.get("operation_id"))
                render()
            monkeypatch.setattr(screen, "render_model_intent", record)
            api.block_path = "/models/intent/current"
            screen.action_refresh()
            try:
                await wait(pilot, api.entered.is_set)
                if release_order == "read_first":
                    api.release.set()
                    await wait(pilot, lambda: not screen._core_inflight)
                    assert screen._model_action_inflight
                    assert not screen.model_intent, "an in-flight POST owns the local selection"
                    api.post_release.set()
                else:
                    api.post_release.set()
                    await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP)
                    api.release.set()
                await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP
                           and not screen._model_action_inflight and not screen._core_inflight)
                assert screen.model_intent["intent"]["model_id"] == "master-small"
                assert screen.model_selection_id == "master-small"
                assert OP not in rendered_ids, "a stale result must never appear, even transiently"
                assert [call for call in api.calls if call[0] != "GET"] == [
                    ("POST", "/models/intent", {"model_id": "master-small", "engine": "llama_cpp",
                                               "quant_type": "Q4_K_M", "strategy": "auto", "timeout_seconds": 3600})]
            finally:
                api.release.set()
                api.post_release.set()
    asyncio.run(run())


@pytest.mark.parametrize("change", ["replace_client", "change_url"])
def test_delayed_current_from_old_endpoint_is_discarded(change, monkeypatch):
    async def run():
        api = IntentApi(operation("ready"))
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            await wait(pilot, lambda: not screen._core_inflight)
            api.block_path = "/models/intent/current"
            screen.action_refresh()
            await wait(pilot, api.entered.is_set)
            next_api = IntentApi(operation("ready", "master-small", NEXT_OP))
            next_api.base_url = "http://other-worker:8000"
            if change == "replace_client":
                app.api = next_api
            else:
                api.base_url = next_api.base_url
                api.current = deepcopy(next_api.current)
                next_api = api
            rendered_ids = []
            render = screen.render_model_intent
            def record():
                rendered_ids.append(screen.model_intent.get("operation_id"))
                render()
            monkeypatch.setattr(screen, "render_model_intent", record)
            try:
                api.release.set()
                await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP and not screen._core_inflight)
                assert rendered_ids and OP not in rendered_ids
                assert not screen.chat_model_gate()
                assert screen.model_intent["intent"]["model_id"] == "master-small"
                assert all(method == "GET" for method, _, _ in api.calls + next_api.calls)
                if change == "replace_client":
                    assert ("GET", "/cluster/resources", None) in api.calls, "one refresh must keep its captured client"
                    assert ("GET", "/health", None) in next_api.calls, "the discarded read must schedule a fresh endpoint read"
            finally:
                api.release.set()
    asyncio.run(run())


def test_external_model_current_can_be_cancelled_without_reusing_local_selection():
    async def run():
        api = IntentApi()
        app = tt.KoakumaApp(api, interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            screen.switch_page("models")
            table = screen.query_one("#models-table", DataTable)
            table.focus()
            await pilot.press("l", "y")
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == NEXT_OP and not screen._core_inflight)
            api.current = operation("preparing", "master-small", OP)
            screen.action_refresh()
            await wait(pilot, lambda: screen.model_intent.get("operation_id") == OP and not screen._core_inflight)
            assert screen.model_selection_id == "master-big"
            writes = [call for call in api.calls if call[0] != "GET"]
            await pilot.press("c", "y")
            await wait(pilot, lambda: screen.model_intent.get("status") == "cancelled" and not screen._model_action_inflight)
            assert screen.model_intent["operation_id"] == OP
            assert screen.model_intent["intent"]["model_id"] == "master-small"
            assert screen.model_selection_id == "master-big"
            assert [call for call in api.calls if call[0] != "GET"] == writes + [("DELETE", f"/models/intent/{OP}", None)]
    asyncio.run(run())


@pytest.mark.parametrize("mode, is_master, route, expected, expected_route", [
    pytest.param("distributed", False, None, "full", "distributed_required", id="default-worker"),
    pytest.param("distributed", True, None, "interactive", "distributed_required", id="default-master"),
    pytest.param("local", False, None, "interactive", "local_only", id="default-local"),
    pytest.param("distributed", False, "local_only", "full", "distributed_required", id="worker-overrides-local-preference"),
    pytest.param("distributed", True, "auto", "interactive", "distributed_required", id="master-overrides-auto-preference"),
    pytest.param("distributed", None, "distributed_preferred", "interactive", "distributed_required", id="missing-role-still-required"),
    pytest.param("local", False, "distributed_required", "interactive", "local_only", id="local-overrides-required-preference"),
])
def test_main_chat_request_uses_owner_mode_and_binding_role(monkeypatch, mode, is_master, route, expected, expected_route):
    import io
    import json
    import tui_api
    requests = []
    def urlopen(request, timeout):
        requests.append((request.method, request.full_url, json.loads(request.data), dict(request.header_items())))
        return io.BytesIO(b'data: {"done":true,"response":"role-answer","metrics":{"distributed_used":true}}\n\n')
    monkeypatch.setattr(tui_api.urllib.request, "urlopen", urlopen)

    async def run():
        api = IntentApi(operation("ready"))
        api.mode = mode
        if is_master is not None:
            api.catalog["binding"]["is_master"] = is_master
        api._request_headers = tui_api.ApiClient(auth_token="chat-fixture")._request_headers
        if route is None:
            app = tt.KoakumaApp(api, interval=2, enable_thinking=False)
        else:
            app = tt.KoakumaApp(api, interval=2, enable_thinking=False, routing_preference=route)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            assert screen.model_mode == mode
            pane = screen.query_one("#chat-pane", tt.ChatPane)
            pane.write_line("previous exchange")
            box = screen.query_one("#chat-input", Input)
            box.focus()
            box.value = "hello from the bound endpoint"
            await pilot.press("enter")
            await wait(pilot, lambda: "role-answer" in content(screen, "#chat-log"))
            assert len(requests) == 1
            method, url, body, headers = requests[0]
            assert (method, url) == ("POST", api.base_url + "/api/chat/stream")
            assert body["streaming_mode"] == expected
            if expected == "full":
                assert "previous exchange" in content(screen, "#chat-log")
                assert "hello from the bound endpoint" in content(screen, "#chat-log")
            assert body["routing_preference"] == expected_route
            assert body["enable_thinking"] is False
            assert app.routing_preference == ("auto" if route is None else route)
            assert headers["Authorization"] == "Bearer chat-fixture"
            assert "caller_host" not in body
            assert all(method == "GET" for method, _, _ in api.calls), "chat must not write mode or select a model"
    asyncio.run(run())


@pytest.mark.parametrize("route", ["auto", "local_only", "distributed_preferred", "distributed_required"])
def test_unknown_owner_mode_keeps_app_route_on_input_submitted(monkeypatch, route):
    streamed = []
    def stream(api, message, **kwargs):
        streamed.append(kwargs)
        yield {"done": True, "response": "unknown-mode-answer"}
    monkeypatch.setattr(tt, "iter_chat_payloads", stream)

    class UnknownModeApi(IntentApi):
        def get(self, path, **kwargs):
            if path == "/cluster/config/distributed-inference":
                self.calls.append(("GET", path, None))
                return {}
            return super().get(path, **kwargs)

    async def run():
        api = UnknownModeApi()
        api.mode = None
        app = tt.KoakumaApp(api, interval=30, routing_preference=route)
        async with app.run_test(size=(140, 50)) as pilot:
            app.show_main()
            await wait(pilot, lambda: isinstance(app.screen, tt.MainScreen)
                       and app.screen.runtime_ready is True and not app.screen._core_inflight)
            screen = app.screen
            assert screen.model_mode is None
            box = screen.query_one("#chat-input", Input)
            box.focus()
            box.value = "unknown owner mode"
            await pilot.press("enter")
            await wait(pilot, lambda: "unknown-mode-answer" in content(screen, "#chat-log"))
            assert len(streamed) == 1
            assert streamed[0]["routing_preference"] == route
            assert streamed[0]["streaming_mode"] == "interactive"
            assert app.routing_preference == route
            assert all(method == "GET" for method, _, _ in api.calls)
    asyncio.run(run())


@pytest.mark.parametrize("is_master", [False, True])
def test_role_specific_chat_still_requires_owner_native_ready(monkeypatch, is_master):
    streamed = []
    def stream(api, message, **kwargs):
        streamed.append(kwargs)
        yield {"done": True, "response": "native-ready-answer"}
    monkeypatch.setattr(tt, "iter_chat_payloads", stream)

    async def run():
        api = IntentApi(operation("preparing"))
        api.catalog["binding"]["is_master"] = is_master
        app = tt.KoakumaApp(api, interval=2, enable_thinking=False)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            box = screen.query_one("#chat-input", Input)
            box.focus()
            box.value = "hold until native ready"
            await pilot.press("enter")
            assert not streamed and box.value == "hold until native ready"
            assert "preparing" in content(screen, "#chat-status")
            api.current = operation("ready")
            api.current["loaded_ready"] = False
            screen.action_refresh()
            await wait(pilot, lambda: screen.model_intent.get("status") == "ready" and not screen._core_inflight)
            await pilot.press("enter")
            assert not streamed and box.value == "hold until native ready"
            assert "等待加载确认" in content(screen, "#chat-status")
            api.current["loaded_ready"] = True
            screen.action_refresh()
            await wait(pilot, lambda: not screen.chat_model_gate())
            await pilot.press("enter")
            await wait(pilot, lambda: "native-ready-answer" in content(screen, "#chat-log"))
            assert len(streamed) == 1
            assert streamed[0]["streaming_mode"] == ("interactive" if is_master else "full")
            assert streamed[0]["routing_preference"] == "distributed_required"
            assert all(method == "GET" for method, _, _ in api.calls)
    asyncio.run(run())


@pytest.mark.parametrize("change", ["selection", "mode", "binding", "endpoint"])
def test_late_chat_payload_cannot_modify_changed_model_or_endpoint_context(monkeypatch, change):
    entered = threading.Event()
    release = threading.Event()
    sent = []
    def stream(api, message, **kwargs):
        sent.append((api, kwargs))
        entered.set()
        assert release.wait(8)
        yield {"start": True, "generation_id": "old-generation", "session_id": "old-session"}
        yield {"token": "old-token"}
        yield {"done": True, "response": "old-response", "session_id": "old-session"}
    monkeypatch.setattr(tt, "iter_chat_payloads", stream)

    async def run():
        api = IntentApi(operation("ready"))
        api.catalog["binding"]["is_master"] = False
        app = tt.KoakumaApp(api, interval=2, enable_thinking=False)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            await wait(pilot, lambda: not screen._core_inflight)
            pane = screen.query_one("#chat-pane", tt.ChatPane)
            workers = []
            send = pane.stream_reply
            def record(message):
                worker = send(message)
                workers.append(worker)
                return worker
            monkeypatch.setattr(pane, "stream_reply", record)
            box = pane.query_one("#chat-input", Input)
            box.focus()
            box.value = "old-context-question"
            await pilot.press("enter")
            await wait(pilot, entered.is_set)
            try:
                if change == "mode":
                    box.value = "/model mode local"
                    await pilot.press("enter")
                    await wait(pilot, lambda: isinstance(app.screen, tt.ConfirmScreen))
                    await pilot.press("y")
                    await wait(pilot, lambda: screen.model_mode == "local" and not screen._model_action_inflight
                               and not screen._core_inflight)
                else:
                    if change == "selection":
                        api.current = operation("ready", "master-small", NEXT_OP)
                    elif change == "binding":
                        api.catalog["binding"]["is_master"] = True
                    else:
                        replacement = IntentApi(operation("ready", "master-small", NEXT_OP))
                        replacement.base_url = "http://new-worker:8000"
                        replacement.catalog["binding"]["is_master"] = True
                        app.api = replacement
                    screen.action_refresh()
                    await wait(pilot, lambda: not screen._core_inflight and
                               (screen.model_binding.get("is_master") is True if change == "binding"
                                else screen.model_intent.get("operation_id") == NEXT_OP))
                app.generation_id = "new-generation"
                app.session_id = "new-session"
                transcript = pane.chat_buffer
                status = content(screen, "#chat-status")
                release.set()
                await workers[0].wait()
                await pilot.pause()
                assert pane.chat_buffer == transcript
                assert content(screen, "#chat-status") == status
                assert app.generation_id == "new-generation"
                assert app.session_id == "new-session"
                assert len(sent) == 1 and sent[0][0] is api
                assert sent[0][1]["streaming_mode"] == "full"
                assert sent[0][1]["routing_preference"] == "distributed_required"
            finally:
                release.set()
    asyncio.run(run())


@pytest.mark.parametrize("change", ["binding", "mode"])
def test_chat_context_changed_before_worker_dispatch_does_not_send(monkeypatch, change):
    def no_stream(*args, **kwargs):
        pytest.fail("a queued chat with stale binding reached transport")
    monkeypatch.setattr(tt, "iter_chat_payloads", no_stream)

    async def run():
        api = IntentApi(operation("ready"))
        api.catalog["binding"]["is_master"] = False
        app = tt.KoakumaApp(api, interval=2, enable_thinking=False)
        async with app.run_test(size=(140, 50)) as pilot:
            screen = await enter(app, pilot)
            await wait(pilot, lambda: not screen._core_inflight)
            pane = screen.query_one("#chat-pane", tt.ChatPane)
            deferred = []
            dispatch = pane._stream_reply
            monkeypatch.setattr(pane, "_stream_reply", lambda *args: deferred.append(args))
            box = pane.query_one("#chat-input", Input)
            box.focus()
            box.value = "captured on worker"
            await pilot.press("enter")
            assert len(deferred) == 1 and deferred[0][2]["streaming_mode"] == "full"
            assert deferred[0][2]["routing_preference"] == "distributed_required"
            assert app.routing_preference == "auto"
            if change == "mode":
                box.value = "/model mode local"
                await pilot.press("enter")
                await wait(pilot, lambda: isinstance(app.screen, tt.ConfirmScreen))
                await pilot.press("y")
                await wait(pilot, lambda: screen.model_mode == "local" and not screen._model_action_inflight
                           and not screen._core_inflight)
            else:
                api.catalog["binding"]["is_master"] = True
                screen.action_refresh()
                await wait(pilot, lambda: screen.model_binding.get("is_master") is True and not screen._core_inflight)
            writes = [call for call in api.calls if call[0] != "GET"]
            assert writes == ([("PUT", "/cluster/config/distributed-inference", {"enabled": False}),
                               ("POST", "/models/unload", None)] if change == "mode" else [])
            await dispatch(*deferred[0]).wait()
            assert [call for call in api.calls if call[0] != "GET"] == writes
    asyncio.run(run())
