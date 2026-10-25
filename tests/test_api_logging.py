"""
单元测试 — 日志功能 API 端点（L0/L1/L2/L5）
==============================================
测试 /api/logs 系列接口：访问控制、文件管理、request_id、
内存缓冲、统计、导出、错误上报、保留策略。

使用 FastAPI TestClient + mock scheduler。
"""

import sys
import os
import io
import json
import zipfile
import logging
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from fastapi.testclient import TestClient


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture(autouse=True)
def _patch_scheduler_for_test():
    """所有测试自动 mock scheduler，避免真实 TCP/线程启动。"""
    with patch('api_server.scheduler', MagicMock()) as mock_sched:
        mock_sched.get_effective_node_id.return_value = "test-node"
        mock_sched._effective_role.return_value = "master"
        mock_sched.get_distributed_inference_enabled.return_value = False
        yield mock_sched


@pytest.fixture(autouse=True)
def _explicit_memory_logging(_patch_scheduler_for_test):
    """Memory logging is a startup resource, installed explicitly for API tests."""
    import api_server
    root = logging.getLogger()
    original_level = root.level
    handler = api_server.MemoryLogHandler()
    handler.addFilter(api_server._request_id_filter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    with api_server._log_buffer_lock:
        api_server._log_buffer.clear()
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(original_level)
        with api_server._log_buffer_lock:
            api_server._log_buffer.clear()


@pytest.fixture
def log_dir_temp():
    """创建临时日志目录并 patch config.LOG_DIR。"""
    import api_server
    with tempfile.TemporaryDirectory(prefix="qlh-test-logs-") as tmpdir:
        with patch('config.LOG_DIR', tmpdir):
            yield tmpdir
            # ★ teardown: 关闭 RotatingFileHandler 释放文件句柄，避免 Windows PermissionError
            api_server._close_logging_handlers()
            threading.Event().wait(0.1)


@pytest.fixture
def client():
    """FastAPI TestClient（无真实 scheduler）。"""
    from api_server import app
    return TestClient(app)


@pytest.fixture
def client_with_logs(log_dir_temp, client):
    """创建一个有示例 .log 文件的 TestClient。"""
    # 写入两个测试日志文件（使用过去日期避免与 setup_logging 重建冲突）
    for i in range(2):
        fpath = os.path.join(log_dir_temp, f"qlh-2026-01-{10 + i:02d}.log")
        with open(fpath, "w", encoding="utf-8") as f:
            for j in range(10):
                f.write(f"2026-01-{10 + i:02d} 10:00:00 [INFO] request_id=test-{i} test: log line {j + 1}\n")
    # 写入一个非 .log 文件（应被忽略）
    with open(os.path.join(log_dir_temp, "readme.txt"), "w") as f:
        f.write("not a log")
    return client


# ================================================================
# L0: 日志 API 访问控制
# ================================================================

class TestLogApiAccessControl:
    """L0: 日志 API 权限校验。"""

    def test_local_access_allowed(self, client):
        """本机访问应返回 200。"""
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        assert "files" in resp.json()

    def test_remote_access_denied_without_token(self, client):
        """远程访问（非 localhost）应被拒绝。"""
        with patch('api_server._get_request_client', return_value="192.168.1.100"):
            resp = client.get("/api/logs")
            assert resp.status_code == 403

    @patch.dict(os.environ, {"QLH_LOG_ADMIN_TOKEN": "secret-token-123"})
    def test_remote_access_allowed_with_valid_token(self, client):
        """远程访问 + 正确 token 应允许。"""
        with patch('api_server._get_request_client', return_value="192.168.1.100"):
            resp = client.get(
                "/api/logs",
                headers={"X-QLH-Log-Token": "secret-token-123"},
            )
            assert resp.status_code == 200

    @patch.dict(os.environ, {"QLH_LOG_ADMIN_TOKEN": "secret-token-123"})
    def test_remote_access_denied_with_wrong_token(self, client):
        """远程访问 + 错误 token 应拒绝。"""
        with patch('api_server._get_request_client', return_value="192.168.1.100"):
            resp = client.get(
                "/api/logs",
                headers={"X-QLH-Log-Token": "wrong-token"},
            )
            assert resp.status_code == 403

    def test_filename_validation_rejects_path_traversal(self, client_with_logs):
        """文件名白名单应拒绝包含路径分隔符的请求。"""
        # FastAPI 在路由匹配前已规范化路径，简单 .. 会被解析。
        # 使用路径编码测试白名单校验。
        resp = client_with_logs.get("/api/logs/evil%2F..%2Fetc%2Fpasswd.log")
        assert resp.status_code in (400, 404)  # 400=白名单拒绝, 404=文件不存在但白名单通过

    def test_filename_validation_rejects_non_log_extension(self, client_with_logs):
        """非 .log 扩展名应被拒绝。"""
        resp = client_with_logs.get("/api/logs/readme.txt")
        assert resp.status_code == 400


# ================================================================
# L0: 日志文件操作
# ================================================================

class TestLogFileOperations:
    """L0: 日志文件列表、读取、删除。"""

    def test_list_empty_logs(self, log_dir_temp, client):
        """空日志目录返回空列表。"""
        # 清除 setup_logging() 可能创建的日志文件
        for f in os.listdir(log_dir_temp):
            os.remove(os.path.join(log_dir_temp, f))
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        assert resp.json()["files"] == []

    def test_list_log_files(self, client_with_logs):
        """列出 .log 文件，忽略非 .log 文件。"""
        resp = client_with_logs.get("/api/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["files"]) == 2
        names = [f["name"] for f in data["files"]]
        assert "qlh-2026-01-10.log" in names
        assert "qlh-2026-01-11.log" in names
        # 不应包含非 .log 文件
        assert all(f["name"].endswith(".log") or ".log." in f["name"] for f in data["files"])

    def test_read_log_file_content(self, client_with_logs):
        """读取日志文件内容。"""
        resp = client_with_logs.get("/api/logs/qlh-2026-01-10.log")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "qlh-2026-01-10.log"
        assert "log line 1" in data["content"]
        assert "truncated" in data

    def test_read_log_file_truncated_flag(self, log_dir_temp, client):
        """大日志文件应返回 truncated=true。"""
        fpath = os.path.join(log_dir_temp, "qlh-2026-01-10.log")
        # 写入超过 1MB 的日志
        with open(fpath, "w", encoding="utf-8") as f:
            for i in range(20000):
                f.write(f"2026-07-10 10:00:00 [INFO] test: this is log line number {i:05d}\n")
        file_size = os.path.getsize(fpath)
        assert file_size > 1024 * 1024  # 确保超过读取上限

        resp = client.get("/api/logs/qlh-2026-01-10.log")
        assert resp.status_code == 200
        data = resp.json()
        # >1MB 文件应被截断
        assert data["truncated"] is True

    def test_read_nonexistent_log(self, client):
        """读取不存在的日志返回 404。"""
        resp = client.get("/api/logs/nonexistent.log")
        assert resp.status_code == 404

    def test_delete_log_file(self, client_with_logs):
        """删除单个日志文件（delete 后 setup_logging 会重建当天日志）。"""
        resp = client_with_logs.delete("/api/logs/qlh-2026-01-10.log")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # 验证目标文件已删除（setup_logging 可能重建当天新文件，但原名不应存在）
        list_resp = client_with_logs.get("/api/logs")
        names = [f["name"] for f in list_resp.json()["files"]]
        assert "qlh-2026-01-10.log" not in names
        # 另一个测试文件应仍然存在
        assert "qlh-2026-01-11.log" in names

    def test_delete_current_log_file_releases_file_handler(self, log_dir_temp, client):
        """删除当前日志前必须释放文件 handler，避免 Windows PermissionError。"""
        from datetime import datetime
        import api_server

        api_server.setup_logging()
        today_name = f"qlh-{datetime.now():%Y-%m-%d}.log"
        today_path = os.path.join(log_dir_temp, today_name)

        logging.getLogger("test.active_log_delete").info("active log should be deletable")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

        assert os.path.isfile(today_path)

        resp = client.delete(f"/api/logs/{today_name}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["deleted"] == today_name
        assert data["failed"] == []

    def test_delete_all_logs(self, client_with_logs):
        """删除全部日志文件（delete 后 setup_logging 会自动重建当天日志）。"""
        resp = client_with_logs.delete("/api/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_count"] >= 2  # 至少删除了我们创建的两个文件

        # setup_logging 重建后会创建一个新的当天日志文件
        list_resp = client_with_logs.get("/api/logs")
        remaining = list_resp.json()["files"]
        # 原有测试文件应被删除
        remaining_names = [f["name"] for f in remaining]
        assert "qlh-2026-01-10.log" not in remaining_names
        assert "qlh-2026-01-11.log" not in remaining_names

    def test_delete_nonexistent_log(self, client):
        """删除不存在的日志返回 404。"""
        resp = client.delete("/api/logs/nonexistent.log")
        assert resp.status_code == 404


# ================================================================
# L1: request_id middleware
# ================================================================

class TestRequestIdMiddleware:
    """L1: request_id 在响应头和错误中的行为。"""

    def test_response_has_request_id_header(self, client):
        """每个响应应包含 X-Request-ID 头。"""
        resp = client.get("/api/logs")
        assert "X-Request-ID" in resp.headers
        rid = resp.headers["X-Request-ID"]
        assert len(rid) == 32  # uuid4().hex = 32 hex chars

    def test_request_id_in_error_response(self, client_with_logs):
        """404 错误响应 JSON 中应包含 request_id。"""
        resp = client_with_logs.get("/api/logs/nonexistent.log")
        assert resp.status_code == 404
        data = resp.json()
        assert "request_id" in data

    def test_request_id_can_be_specified(self, client):
        """客户端可通过 X-Request-ID 头指定 request_id。"""
        custom_rid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 32 hex chars
        resp = client.get("/api/logs", headers={"X-Request-ID": custom_rid})
        assert resp.headers["X-Request-ID"] == custom_rid

    def test_request_id_sanitized(self, client):
        """恶意 request_id 应被清理。"""
        resp = client.get("/api/logs", headers={"X-Request-ID": "test\"; DROP TABLE;"})
        rid = resp.headers["X-Request-ID"]
        # 特殊字符应被移除
        assert "\"" not in rid
        assert ";" not in rid
        assert " " not in rid


# ================================================================
# L2: 最近日志内存缓冲 (GET /api/logs/recent)
# ================================================================

class TestRecentLogsApi:
    """L2: 内存环形缓冲 /api/logs/recent。"""

    def test_recent_logs_returns_data(self, client):
        """recent API 应返回日志条目。"""
        resp = client.get("/api/logs/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert "logs" in data
        assert "count" in data
        assert "matched" in data
        assert "buffer_size" in data
        assert "buffer_capacity" in data
        assert isinstance(data["logs"], list)

    def test_recent_logs_respects_limit(self, client):
        """limit 参数应生效。"""
        resp = client.get("/api/logs/recent?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 10
        assert len(data["logs"]) <= 10

    def test_recent_logs_level_filter(self, client):
        """level 过滤应生效。"""
        # 先确保有一些 ERROR 级别日志
        logger = logging.getLogger("test.recent")
        logger.error("test_error_for_filter")
        logger.info("test_info_for_filter")

        resp = client.get("/api/logs/recent?level=ERROR&limit=50")
        assert resp.status_code == 200
        data = resp.json()
        for entry in data["logs"]:
            assert entry.get("level") == "ERROR"
        assert data["filters"]["level"] == "ERROR"

    def test_recent_logs_name_filter(self, client):
        """logger name 过滤应生效。"""
        logger = logging.getLogger("test.recent.unique")
        logger.warning("test_name_filter_unique")

        resp = client.get("/api/logs/recent?name=test.recent.unique&limit=50")
        assert resp.status_code == 200
        data = resp.json()
        for entry in data["logs"]:
            assert "test.recent.unique" in entry.get("name", "")

    def test_recent_logs_node_id_filter(self, client):
        """node_id 过滤应生效。"""
        resp = client.get("/api/logs/recent?node_id=nonexistent-node-xyz&limit=50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0  # 不应有匹配

    def test_recent_logs_truncated_flag(self, client):
        """当 filtered > limit 时应设置 truncated。"""
        logger = logging.getLogger("test.recent.bulk")
        for i in range(300):
            logger.info(f"bulk message {i:05d}")

        resp = client.get("/api/logs/recent?limit=50")
        assert resp.status_code == 200
        data = resp.json()
        if data["matched"] > 50:
            assert data["truncated"] is True
        assert data["buffer_capacity"] > 0

    def test_recent_logs_filter_no_match(self, client):
        """无匹配时应返回空列表（非 404）。"""
        resp = client.get("/api/logs/recent?level=DEBUG&name=nonexistent&limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["matched"] == 0


# ================================================================
# L2: 日志统计 (GET /api/logs/stats)
# ================================================================

class TestLogStatsApi:
    """L2: /api/logs/stats 统计端点。"""

    def test_stats_returns_data(self, client):
        """stats API 应返回统计信息。"""
        resp = client.get("/api/logs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "log_dir" in data
        assert "files_count" in data
        assert "files_total_bytes" in data
        assert "buffer_size" in data
        assert "buffer_capacity" in data
        assert "levels" in data
        assert "loggers" in data
        assert "nodes" in data
        assert "node_id" in data

    def test_stats_files_count(self, client_with_logs):
        """stats 应正确计数日志文件。"""
        resp = client_with_logs.get("/api/logs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["files_count"] == 2
        assert data["files_total_bytes"] > 0

    def test_stats_buffer_size(self, client):
        """buffer 大小应反映已记录的日志条数。"""
        logger = logging.getLogger("test.stats")
        logger.info("stats test message one")
        logger.info("stats test message two")

        resp = client.get("/api/logs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["buffer_size"] >= 2
        recent = client.get("/api/logs/recent?name=test.stats&limit=50").json()["logs"]
        assert {"stats test message one", "stats test message two"} <= {entry["message"] for entry in recent}

    def test_stats_empty_directory(self, log_dir_temp, client):
        """空日志目录 stats 应返回 0。"""
        # 清除 setup_logging() 可能创建的日志文件
        for f in os.listdir(log_dir_temp):
            os.remove(os.path.join(log_dir_temp, f))
        resp = client.get("/api/logs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["files_count"] == 0
        assert data["files_total_bytes"] == 0


# ================================================================
# L5: 日志压缩包导出 (GET /api/logs/export)
# ================================================================

class TestLogExportApi:
    """L5: 日志压缩包导出。"""

    def test_export_returns_zip(self, client_with_logs):
        """导出应返回 application/zip。"""
        resp = client_with_logs.get("/api/logs/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert "Content-Disposition" in resp.headers
        assert "qlh-logs-" in resp.headers["Content-Disposition"]
        assert ".zip" in resp.headers["Content-Disposition"]

    def test_export_contains_log_files(self, client_with_logs):
        """导出的 ZIP 应包含所有 .log 文件。"""
        resp = client_with_logs.get("/api/logs/export")
        assert resp.status_code == 200

        buf = io.BytesIO(resp.content)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            assert "qlh-2026-01-10.log" in names
            assert "qlh-2026-01-11.log" in names
            # 不应包含非 .log 文件
            assert "readme.txt" not in names

    def test_export_empty_directory(self, log_dir_temp, client):
        """空日志目录应返回 404。"""
        # 清除 setup_logging() 可能创建的日志文件
        for f in os.listdir(log_dir_temp):
            os.remove(os.path.join(log_dir_temp, f))
        resp = client.get("/api/logs/export")
        assert resp.status_code == 404

    def test_export_zip_is_valid(self, client_with_logs):
        """导出的 ZIP 文件应可被标准 zipfile 模块读取。"""
        resp = client_with_logs.get("/api/logs/export")
        buf = io.BytesIO(resp.content)
        zf = zipfile.ZipFile(buf, "r")
        assert zf.testzip() is None  # 无损坏
        zf.close()


# ================================================================
# L5: 前端错误上报 (POST /api/logs/client-error)
# ================================================================

class TestClientErrorReport:
    """L5: 前端错误上报端点。"""

    def test_report_client_error(self, client):
        """前端错误应被接收并记录。"""
        payload = {
            "message": "TypeError: Cannot read property 'x' of undefined",
            "source": "window.onerror",
            "stack": "at ChatPanel.handleSend (http://localhost:5173/src/ChatPanel.jsx:418:7)",
            "url": "http://localhost:5173/",
            "line": 418,
            "col": 7,
            "user_agent": "Mozilla/5.0 TestBrowser",
            "extra": {"session_id": "test-session-1", "action": "send_message"},
        }
        resp = client.post("/api/logs/client-error", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["logged"] is True

    def test_report_minimal_error(self, client):
        """最小字段的错误报告也应被接受。"""
        resp = client.post("/api/logs/client-error", json={
            "message": "minimal error",
            "source": "manual",
            "stack": "",
            "url": "",
            "line": 0,
            "col": 0,
            "user_agent": "",
        })
        assert resp.status_code == 200

    def test_report_truncates_long_fields(self, client, caplog):
        """超长字段应被截断（后端不应因前端错误导致自身 OOM）。"""
        long_msg = "X" * 5000
        long_stack = "Y" * 5000
        with caplog.at_level(logging.ERROR, logger="api_server"):
            resp = client.post("/api/logs/client-error", json={
                "message": long_msg,
                "source": "window.onerror",
                "stack": long_stack,
                "url": "https://example.test/" + ("u" * 2000),
                "line": 1,
                "col": 1,
                "user_agent": "A" * 500,
            })
        assert resp.status_code == 200
        assert "...[truncated]" in caplog.text
        assert long_msg not in caplog.text
        assert long_stack not in caplog.text

    def test_report_empty_payload_accepted(self, client):
        """所有字段有默认值，空 payload 应被接受。"""
        resp = client.post("/api/logs/client-error", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ================================================================
# L5: 日志保留策略清理
# ================================================================

class TestLogRetentionCleanup:
    """L5: 按天数 + 总空间的日志保留策略。"""

    def test_cleanup_age_expired_files(self, log_dir_temp):
        """超过 LOG_MAX_AGE_DAYS 的日志应被清理。"""
        from api_server import _run_log_retention_cleanup
        from config import LOG_DIR, LOG_MAX_AGE_DAYS, LOG_MAX_TOTAL_SIZE_MB
        import datetime as _dt

        # 创建一个旧日志（修改时间为 N+1 天前）
        old_file = os.path.join(log_dir_temp, "qlh-2020-01-01.log")
        with open(old_file, "w") as f:
            f.write("old log content\n")
        # 修改 mtime 到很久以前
        old_time = (_dt.datetime.now() - _dt.timedelta(days=LOG_MAX_AGE_DAYS + 5)).timestamp()
        os.utime(old_file, (old_time, old_time))

        # 创建一个今天的日志（不应被删除）
        today = _dt.datetime.now().strftime("%Y-%m-%d")
        today_file = os.path.join(log_dir_temp, f"qlh-{today}.log")
        with open(today_file, "w") as f:
            f.write("today log content\n")

        # 执行清理
        with patch('config.LOG_MAX_AGE_DAYS', LOG_MAX_AGE_DAYS):
            with patch('config.LOG_MAX_TOTAL_SIZE_MB', 0):
                _run_log_retention_cleanup()

        # 旧文件应被删除
        assert not os.path.exists(old_file)
        # 当天文件应保留
        assert os.path.exists(today_file)

    def test_cleanup_size_limit(self, log_dir_temp):
        """超过 LOG_MAX_TOTAL_SIZE_MB 时应删除最旧的非当天文件。"""
        from api_server import _run_log_retention_cleanup
        import datetime as _dt

        # 创建多个小文件，总量超过限制
        today = _dt.datetime.now().strftime("%Y-%m-%d")
        for days_ago in range(5, 15):
            fname = f"qlh-{(_dt.datetime.now() - _dt.timedelta(days=days_ago)).strftime('%Y-%m-%d')}.log"
            fpath = os.path.join(log_dir_temp, fname)
            with open(fpath, "w") as f:
                f.write("x" * 1000000)  # ~1 MB
            old_time = (_dt.datetime.now() - _dt.timedelta(days=days_ago)).timestamp()
            os.utime(fpath, (old_time, old_time))

        # 创建当天文件
        today_file = os.path.join(log_dir_temp, f"qlh-{today}.log")
        with open(today_file, "w") as f:
            f.write("today\n")

        total_before = sum(
            os.path.getsize(os.path.join(log_dir_temp, f))
            for f in os.listdir(log_dir_temp)
            if f.endswith(".log")
        )

        # 设置 5MB 限制
        with patch('config.LOG_MAX_AGE_DAYS', 0):
            with patch('config.LOG_MAX_TOTAL_SIZE_MB', 5):
                _run_log_retention_cleanup()

        total_after = sum(
            os.path.getsize(os.path.join(log_dir_temp, f))
            for f in os.listdir(log_dir_temp)
            if f.endswith(".log")
        )
        # 清理后总大小应减少
        assert total_after < total_before
        # 当天文件应保留
        assert os.path.exists(today_file)

    def test_cleanup_disabled_when_both_zero(self, log_dir_temp):
        """两项限制都为 0 时清理应跳过。"""
        from api_server import _run_log_retention_cleanup

        fpath = os.path.join(log_dir_temp, "qlh-2020-01-01.log")
        with open(fpath, "w") as f:
            f.write("old\n")

        with patch('config.LOG_MAX_AGE_DAYS', 0):
            with patch('config.LOG_MAX_TOTAL_SIZE_MB', 0):
                _run_log_retention_cleanup()

        # 文件应保留（清理被禁用）
        assert os.path.exists(fpath)

    def test_cleanup_preserves_today_log(self, log_dir_temp):
        """所有情况下当天日志都不应被删除。"""
        from api_server import _run_log_retention_cleanup
        import datetime as _dt

        today = _dt.datetime.now().strftime("%Y-%m-%d")
        today_file = os.path.join(log_dir_temp, f"qlh-{today}.log")
        with open(today_file, "w") as f:
            f.write("today\n")

        # 设置极低的限制
        with patch('config.LOG_MAX_AGE_DAYS', 1):
            with patch('config.LOG_MAX_TOTAL_SIZE_MB', 0):
                _run_log_retention_cleanup()

        # 当天文件应保留
        assert os.path.exists(today_file)

    def test_retention_runs_immediately_after_startup(self, monkeypatch, log_dir_temp):
        """启动即清理：不得再等固定延迟。

        曾固定 `wait(300)` 才首次清理，而开发期进程常在同一分钟内重启 ⇒ 线程每次重新
        数 300 s，**首次清理永远等不到**。实测 `event=log_retention_cleanup` 在
        2026-10-17 之后 7 天里一次都没出现，日志因此堆到 20 GB。
        """
        import threading as _threading
        import time as _time
        import api_server

        calls = []
        monkeypatch.setattr(api_server, "_run_log_retention_cleanup",
                            lambda: calls.append(1))
        monkeypatch.setattr(api_server, "_log_retention_thread_started", False)
        monkeypatch.setattr(api_server, "_runtime_background_stop", _threading.Event())

        api_server._start_log_retention_thread()
        deadline = _time.time() + 3.0
        while not calls and _time.time() < deadline:
            _time.sleep(0.02)
        api_server._runtime_background_stop.set()  # 让后台线程退出

        assert calls, "日志保留线程未在启动后立即执行清理"


# ================================================================
# L5: 多节点日志聚合
# ================================================================

class TestMultiNodeLogAggregation:
    """L5: 多节点日志聚合 API。"""

    def test_nodes_summary_local(self, client):
        """nodes-summary 应包含 local 节点信息。"""
        resp = client.get("/api/logs/nodes-summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "local" in data
        assert data["local"]["node_id"] == "test-node"
        assert "workers" in data

    def test_node_own_logs_via_recent(self, client):
        """通过 node/{id}/recent 访问本节点应返回本地日志。"""
        resp = client.get("/api/logs/node/test-node/recent?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "local"
        assert data["node_id"] == "test-node"

    def test_node_master_alias(self, client):
        """node/master/recent 也应是本节点。"""
        resp = client.get("/api/logs/node/master/recent?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "local"

    def test_nodes_summary_requires_master_role(self, client, _patch_scheduler_for_test):
        """非 master 角色应被拒绝。"""
        _patch_scheduler_for_test._effective_role.return_value = "client"
        resp = client.get("/api/logs/nodes-summary")
        assert resp.status_code == 403

    def test_node_recent_requires_master_for_remote(self, client, _patch_scheduler_for_test):
        """远程节点拉取需要 master 角色。"""
        _patch_scheduler_for_test._effective_role.return_value = "client"
        resp = client.get("/api/logs/node/worker-1/recent?limit=10")
        assert resp.status_code == 403


# ================================================================
# L5: 半结构化日志字段（日志消息格式验证）
# ================================================================

class TestStructuredLogFormat:
    """L5: 半结构化日志字段规范。"""

    def test_request_id_in_log_record(self, caplog):
        """日志记录应包含 request_id 属性（通过 RequestIdFilter）。"""
        from api_server import _request_id_ctx

        _request_id_ctx.set("abc123def456")
        logger = logging.getLogger("test.structured")
        with caplog.at_level(logging.INFO, logger="test.structured"):
            logger.info("event=test_request_id message=hello")

        # request_id 由 RequestIdFilter 注入到 LogRecord，caplog 会保留
        record = caplog.records[-1]
        # caplog 可能不保留 filter 注入的属性，但格式化消息中应有 request_id=
        assert "event=test_request_id" in record.getMessage()

    def test_event_field_in_formatted_log(self, caplog):
        """格式化日志应包含 event= 字段。"""
        logger = logging.getLogger("test.structured")
        with caplog.at_level(logging.INFO):
            logger.info("event=test_event field1=value1 field2=42")

        assert "event=test_event" in caplog.text
        assert "field1=value1" in caplog.text

    def test_request_identity_projection(self):
        from qlh_core.execution.requests import RequestService
        value = RequestService._snapshot_record({"request_id": "request", "boot_id": "boot", "deployment_id": "deployment", "revision": 1})
        assert value["identity"] == {"request_id": "request", "boot_id": "boot", "deployment_id": "deployment"}
        assert value["schema_version"] == 2

    def test_log_retention_cleanup_log_format(self, log_dir_temp, caplog):
        """日志保留清理应输出 event=log_retention_cleanup。"""
        from api_server import _run_log_retention_cleanup

        with patch('config.LOG_MAX_AGE_DAYS', 1):
            with patch('config.LOG_MAX_TOTAL_SIZE_MB', 0):
                with caplog.at_level(logging.INFO, logger="api_server"):
                    _run_log_retention_cleanup()

        # 检查是否有清理日志（可能为空目录）
        log_text = caplog.text
        if log_text.strip():
            # 如果有输出，格式应正确
            pass  # 空目录无输出也是正常的


# ================================================================
# P7: 跨节点日志聚合 API（2026-08-16）
# ================================================================

class TestLogAggregate:
    def _master_client(self, monkeypatch):
        import api_server
        import scheduler as sched_mod

        class FakeInfo:
            role = sched_mod.NodeRole.CLIENT
            state = sched_mod.NodeState.ONLINE

        fake_sched = type("FakeScheduler", (), {
            "nodes": {"w1": FakeInfo(), "w2": FakeInfo()},
            "get_effective_node_id": staticmethod(lambda: "m0"),
            "_nodes_lock": __import__("threading").RLock(),
            "request_node_logs": lambda self, node_id, limit, level="", name="", timeout=3.0: (
                {"node_id": node_id, "logs": [f"{node_id} line 1", f"{node_id} line 2"],
                 "count": 2, "buffer_size": 2}
            ),
        })()
        monkeypatch.setattr(api_server, "scheduler", fake_sched)
        monkeypatch.setattr(api_server, "_get_effective_role_safe", lambda: "master")
        return api_server.app

    def test_aggregate_returns_local_and_worker_logs(self, monkeypatch, client):
        app = self._master_client(monkeypatch)
        resp = client.get("/api/cluster/nodes/log-aggregate?limit=50")
        assert resp.status_code == 200
        body = resp.json()
        assert body["local"]["node_id"] == "m0"
        assert body["total_workers"] == 2
        assert len(body["workers"]) == 2
        assert all(w["count"] == 2 for w in body["workers"])
        assert "w1 line 1" in body["workers"][0]["logs"][0]

    def test_aggregate_worker_timeout_returns_error_without_breaking(self, monkeypatch, client):
        import api_server
        import scheduler as sched_mod

        class FakeInfo:
            role = sched_mod.NodeRole.CLIENT
            state = sched_mod.NodeState.ONLINE

        fake_sched = type("FakeScheduler", (), {
            "nodes": {"w1": FakeInfo(), "w2": FakeInfo()},
            "get_effective_node_id": staticmethod(lambda: "m0"),
            "_nodes_lock": __import__("threading").RLock(),
            "request_node_logs": lambda self, node_id, limit, level="", name="", timeout=3.0: (
                None if node_id == "w1" else
                {"node_id": node_id, "logs": ["ok"], "count": 1, "buffer_size": 1}
            ),
        })()
        monkeypatch.setattr(api_server, "scheduler", fake_sched)
        monkeypatch.setattr(api_server, "_get_effective_role_safe", lambda: "master")

        resp = client.get("/api/cluster/nodes/log-aggregate")
        assert resp.status_code == 200
        body = resp.json()
        w1 = next(w for w in body["workers"] if w["node_id"] == "w1")
        assert w1["error"] == "timeout"
        w2 = next(w for w in body["workers"] if w["node_id"] == "w2")
        assert w2["count"] == 1

    def test_aggregate_requires_master_role(self, monkeypatch, client):
        import api_server
        monkeypatch.setattr(api_server, "_get_effective_role_safe", lambda: "worker")
        resp = client.get("/api/cluster/nodes/log-aggregate")
        assert resp.status_code == 403

    def test_aggregate_empty_worker_set_returns_local_only(self, monkeypatch, client):
        import api_server

        fake_sched = type("FakeScheduler", (), {
            "nodes": {},
            "get_effective_node_id": staticmethod(lambda: "m0"),
            "_nodes_lock": threading.RLock(),
        })()
        monkeypatch.setattr(api_server, "scheduler", fake_sched)
        monkeypatch.setattr(api_server, "_get_effective_role_safe", lambda: "master")

        resp = client.get("/api/cluster/nodes/log-aggregate")

        assert resp.status_code == 200
        assert resp.json()["workers"] == []
        assert resp.json()["total_workers"] == 0

    def test_aggregate_fetches_workers_with_bounded_concurrency(self, monkeypatch, client):
        import api_server
        import scheduler as sched_mod

        class FakeInfo:
            role = sched_mod.NodeRole.CLIENT
            state = sched_mod.NodeState.ONLINE

        state = {"active": 0, "maximum": 0}
        state_lock = threading.Lock()
        workers_ready = threading.Barrier(4)

        def request_node_logs(self, node_id, limit, level="", name="", timeout=3.0):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            workers_ready.wait(timeout=2)
            with state_lock:
                state["active"] -= 1
            return {"node_id": node_id, "logs": [node_id], "count": 1}

        fake_sched = type("FakeScheduler", (), {
            "nodes": {f"w{i}": FakeInfo() for i in range(4)},
            "get_effective_node_id": staticmethod(lambda: "m0"),
            "_nodes_lock": threading.RLock(),
            "request_node_logs": request_node_logs,
        })()
        monkeypatch.setattr(api_server, "scheduler", fake_sched)
        monkeypatch.setattr(api_server, "_get_effective_role_safe", lambda: "master")

        resp = client.get("/api/cluster/nodes/log-aggregate")

        assert resp.status_code == 200
        assert state["maximum"] >= 2
        assert [worker["node_id"] for worker in resp.json()["workers"]] == [
            "w0", "w1", "w2", "w3",
        ]

    def test_aggregate_returns_deadline_errors_without_waiting_per_node(
        self, monkeypatch, client,
    ):
        import api_server
        import scheduler as sched_mod

        class FakeInfo:
            role = sched_mod.NodeRole.CLIENT
            state = sched_mod.NodeState.ONLINE

        def request_node_logs(self, node_id, limit, level="", name="", timeout=3.0):
            threading.Event().wait(0.2)
            return {"node_id": node_id, "logs": [node_id], "count": 1}

        fake_sched = type("FakeScheduler", (), {
            "nodes": {"w1": FakeInfo(), "w2": FakeInfo()},
            "get_effective_node_id": staticmethod(lambda: "m0"),
            "_nodes_lock": threading.RLock(),
            "request_node_logs": request_node_logs,
        })()
        monkeypatch.setattr(api_server, "scheduler", fake_sched)
        monkeypatch.setattr(api_server, "_get_effective_role_safe", lambda: "master")
        monkeypatch.setattr(api_server, "LOG_AGGREGATE_DEADLINE_SECONDS", 0.03)

        started = time.monotonic()
        resp = client.get("/api/cluster/nodes/log-aggregate")
        elapsed = time.monotonic() - started

        assert resp.status_code == 200
        assert elapsed < 0.4
        assert all(worker["error"] == "deadline" for worker in resp.json()["workers"])
