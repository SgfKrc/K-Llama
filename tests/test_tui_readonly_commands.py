"""单命令只读薄层（``src/tui_commands.py``）单元测试。

背景：旧标准库 TUI（``tui_admin.py`` 等）已归档到 ``_to_delete/``；``qlh status`` 这类
单命令模式改由薄层实现。本测试守住三条纪律：

1. **只读**：命令只命中登记的只读端点，不写后端、不启动后端；
2. **不静默失效**：旧交互命令 / 未知命令都以 rc=2 + 明确指引返回；
3. **零 UI 依赖**：薄层只用标准库（无 torch / 无 textual 也应可用）。
"""

import contextlib
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tui_commands as tc  # noqa: E402
from tui_shared import API_PATHS  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "chat_interactive_fixture.sse")


def _capture(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tc.main(argv)
    return rc, buf.getvalue()


def test_help_lists_readonly_commands_only():
    rc, out = _capture(["help"])
    assert rc == 0
    for name in ("status", "models", "nodes", "queue", "device", "logs"):
        assert name in out, name
    assert "shutdown" not in out, "写操作不应出现在只读帮助里"


def test_unknown_command_exits_2():
    rc, out = _capture(["frobnicate"])
    assert rc == 2
    assert "未知命令" in out


@pytest.mark.parametrize("legacy", ["shutdown", "switch", "connect", "route", "presets"])
def test_legacy_commands_fail_loudly(legacy):
    """归档后旧命令必须给出指引，而不是静默成功或“未知命令”。"""
    rc, out = _capture([legacy])
    assert rc == 2
    assert "归档" in out and "qlh" in out


def test_readonly_command_without_backend_is_graceful():
    rc, out = _capture(["status", "--host", "127.0.0.1", "--port", "1", "--timeout", "0.3"])
    assert rc == 1, "后端不可达应返回 1（而不是抛异常）"
    assert "后端未在运行" in out


def test_fixture_replay_is_offline_and_nonzero_safe():
    assert os.path.exists(FIXTURE), "fixture 缺失：%s" % FIXTURE
    rc, out = _capture(["--fixture", FIXTURE])
    assert rc == 0
    assert "事件数" in out and "ANSI 残留" in out


def test_readonly_commands_hit_only_registered_readonly_paths(monkeypatch):
    """所有只读命令必须命中登记端点；调用集合 ⊆ 只读集合。"""
    seen = []

    class StubApi:
        base_url = "http://stub:8000"
        log_token = ""

        def __init__(self, **kwargs):
            pass

        def get(self, path, **kwargs):
            seen.append(path)
            return {"status": "ok", "nodes": [], "tasks": [], "lines": [],
                    "queue_size": 0, "max_size": 8}

    monkeypatch.setattr(tc, "ApiClient", StubApi)
    for name in ("status", "models", "nodes", "queue", "device", "logs"):
        rc, _ = _capture([name])
        assert rc == 0, name

    readonly = {
        "/health", "/status", API_PATHS["models_current"], "/models",
        API_PATHS["models_intent_catalog"], API_PATHS["models_intent_current"],
        API_PATHS["distributed_config"],
        API_PATHS["cluster_nodes"], API_PATHS["cluster_queue"],
        API_PATHS["device_profile"], API_PATHS["cluster_log_aggregate"],
    }
    assert set(seen) <= readonly, "薄层碰到了非只读端点: %s" % (set(seen) - readonly)


def test_migrated_text_helpers():
    """sanitize / wrap_display / disp_width 随归档迁入薄层，行为保持不变。"""
    assert tc.sanitize("\x1b[31m红\x1b[0m 字") == "红 字"
    assert tc.disp_width("中文ab") == 6
    assert tc.wrap_display("中文ab", 4) == ["中文", "ab"]


def test_cli_entrypoint_is_importable_without_ui():
    """薄层不 import 任何 UI（textual / 已归档模块）。"""
    source = open(tc.__file__, encoding="utf-8").read()
    for banned in ("import textual", "from textual", "import tui_admin", "import tui_textual"):
        assert banned not in source, "薄层不应依赖 %s" % banned
