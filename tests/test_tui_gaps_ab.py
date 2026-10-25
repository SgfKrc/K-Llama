"""★ TUI 缺口补充测试（2026-09-19，用户裁定 dec-b218add896a8f0db）。

补两项：

* **A. 队列「单任务」取消** —— `/queue cancel <task_id>` → `DELETE /api/cluster/queue/task/{task_id}`。
  此前 TUI 只能整体 `clear`（`/queue clear`），卡住的单个任务无法取消。
* **B. 日志细粒度** —— `/logs list | download <file> | read <file> | delete <file> | nodes`。
  此前只有 `recent`（筛选）/`stats`/`export`（打包）与整体 `DELETE /logs`。

本测试锁定：路径常量、客户端函数的 HTTP 方法/路径/参数编码、命令表条目。
（不依赖真实后端 —— 用替身 ApiClient 断言调用形态。）
"""

from __future__ import annotations

from typing import Any

import pytest

import tui_api
from tui_shared import API_PATHS, COMMAND_SPECS


class FakeApi:
    """记录调用的替身，避免依赖真实后端。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def request(self, method: str, path: str, body: Any = None, **kw: Any) -> dict:
        self.calls.append(("request", method, path, body, kw))
        return {"success": True, "task_id": "t1"}

    def get(self, path: str, **kw: Any) -> dict:
        self.calls.append(("get", path, kw))
        return {"value": 1}

    def post(self, path: str, body: Any = None, **kw: Any) -> dict:
        self.calls.append(("post", path, body, kw))
        return {}

    def download(self, path: str, target: Any, **kw: Any) -> Any:
        self.calls.append(("download", path, str(target), kw))
        return target


class TestApiPaths:
    """路径常量必须存在且形状正确。"""

    def test_a_paths(self):
        assert API_PATHS["cluster_queue_task_cancel"] == "/cluster/queue/task/{task_id}"

    def test_b_paths(self):
        assert API_PATHS["logs_list"] == "/logs"
        assert API_PATHS["logs_download"] == "/logs/download"
        assert API_PATHS["logs_file"] == "/logs/{filename}"
        assert API_PATHS["logs_nodes_summary"] == "/logs/nodes-summary"


class TestA_CancelQueueTask:
    """A：单任务取消。"""

    def test_uses_delete_and_quotes_task_id(self):
        api = FakeApi()
        out = tui_api.cancel_queue_task(api, "task/with slash")
        assert api.calls[0][0] == "request"
        assert api.calls[0][1] == "DELETE"
        # 路径参数须 safe="" 编码（task_id 可能含 "/"）
        assert api.calls[0][2] == "/cluster/queue/task/task%2Fwith%20slash"
        assert out.get("success") is True

    def test_command_spec_mentions_cancel(self):
        queue = next(c for c in COMMAND_SPECS if c["name"] == "/queue")
        assert "cancel" in queue["args"], "命令表应暴露 cancel 用法"


class TestB_LogFiles:
    """B：日志细粒度。"""

    def test_list_files(self):
        api = FakeApi()
        tui_api.list_log_files(api)
        assert api.calls[0][0] == "get"
        assert api.calls[0][1] == "/logs"
        assert api.calls[0][2].get("with_log_token") is True, "日志接口须带 log token"

    def test_read_file_encodes_filename(self):
        api = FakeApi()
        tui_api.read_log_file(api, "a/b.log")
        assert api.calls[0][1] == "/logs/a%2Fb.log"
        assert api.calls[0][2].get("with_log_token") is True

    def test_delete_file_uses_delete(self):
        api = FakeApi()
        tui_api.delete_log_file(api, "x.log")
        assert api.calls[0][1] == "DELETE"
        assert api.calls[0][2] == "/logs/x.log"

    def test_download_passes_query(self):
        api = FakeApi()
        tui_api.download_log_file(api, "x.log", "logs/x.log")
        assert api.calls[0][0] == "download"
        assert api.calls[0][1] == "/logs/download?filename=x.log"
        assert api.calls[0][3].get("with_log_token") is True

    def test_nodes_summary(self):
        api = FakeApi()
        tui_api.logs_nodes_summary(api)
        assert api.calls[0][1] == "/logs/nodes-summary"

    def test_command_spec_exists(self):
        logs = next((c for c in COMMAND_SPECS if c["name"] == "/logs"), None)
        assert logs is not None, "命令表应包含 /logs"
        for token in ("list", "download", "read", "delete", "nodes"):
            assert token in logs["args"], f"/logs 用法应含 {token}"


class TestNoRegression:
    """既有队列/日志能力仍可用。"""

    def test_existing_queue_calls_intact(self):
        api = FakeApi()
        tui_api.clear_queue(api)
        assert api.calls[0][0] == "post"
        assert api.calls[0][1] == "/cluster/queue/clear"

    def test_existing_gaps_still_declared(self):
        for key in ("cluster_queue_pause", "cluster_queue_resume",
                     "cluster_queue_clear"):
            assert key in API_PATHS
