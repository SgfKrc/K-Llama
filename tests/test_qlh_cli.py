import os
import sys
from types import SimpleNamespace

import pytest

import qlh


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch):
    """让 `--auto-start` 路径**读不到仓库根的 `.env`**。

    ★ 2026-10-27：这是本文件在开发机上必红、在 CI 上却绿的原因。

    `qlh._shell_options` 在 `--auto-start` 时调
    `SettingsStore(...).load_listener()`，而它按
    ``[get_app_root()/".env", src/".env", cwd/".env"]`` **直接扫磁盘**
    （见 `qlh_core/adapters/settings.py:_read_inputs`）—— `monkeypatch.delenv`
    对它**完全无效**。

    开发机的 `.env` 里有 `QLH_BACKEND_HOST=0.0.0.0`（为了局域网多机联调），于是
    "环境里没配就该 loopback" 这条断言在本机恒挂。把它隔离掉，让这些用例只反映
    **进程环境 + CLI 参数**这两层优先级，不被工作区的 `.env` 内容绑定。

    只覆盖 `load_listener`：它唯一的生产消费者就是这里的 shell 启动路径。
    返回**显式环境快照**（`dict(self._environment)`）而不是 `{}` —— 因为真实实现
    在读完 `.env` 后有一句 `environment.update(self._environment)`，**显式环境永远
    压过 `.env`**。这里等价于"只有显式环境这一层"，于是
    `monkeypatch.setenv("QLH_BACKEND_HOST", "0.0.0.0")` 的用例仍然成立。
    """
    from qlh_core.adapters.settings import SettingsStore

    monkeypatch.setattr(
        SettingsStore, "load_listener", lambda self: dict(self._environment))


def test_help_is_local_and_does_not_spawn(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(qlh.subprocess, "run", lambda *args, **kwargs: called.append(args))

    assert qlh.main(["--help"]) == 0
    assert "qlh chat" in capsys.readouterr().out
    assert called == []


def test_chat_fixture_uses_zero_dependency_smoke(monkeypatch):
    """fixture 回放走薄层（纯标准库），不经 UI、不经后端。"""
    seen = []

    def fake_run(module_script, args):
        seen.append((module_script, args))
        return 0

    monkeypatch.setattr(qlh, "_run", fake_run)
    assert qlh.main(["chat", "--fixture", "fixtures/chat.sse"]) == 0
    assert seen == [("src/tui_commands.py", ["--fixture", "fixtures/chat.sse"])]


def test_models_is_a_readonly_commands_command(monkeypatch):
    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return type("Result", (), {"returncode": 3})()

    monkeypatch.setattr(qlh.subprocess, "run", fake_run)
    assert qlh.main(["models"]) == 3
    assert os.path.normpath(seen[0][1]).endswith(os.path.normpath("src/tui_commands.py"))
    assert seen[0][2] == "models"


def test_builtin_engine_reports_archived(capsys):
    """`--tui-engine builtin` 保留参数但明确报已归档（不静默失效）。"""
    assert qlh.main(["--tui-engine", "builtin"]) == 2
    out = capsys.readouterr().out
    assert "归档" in out and "koakuma" in out


def test_admin_command_reports_archived(capsys):
    """`qlh admin` 指向的旧管理 TUI 已归档。"""
    assert qlh.main(["admin"]) == 2
    assert "归档" in capsys.readouterr().out


def test_readonly_subcommands_route_to_thin_layer(monkeypatch):
    """所有只读子命令统一转发到薄层（且带原样参数）。"""
    seen = []
    monkeypatch.setattr(qlh, "_run",
                        lambda script, args: (seen.append((script, list(args))), 0)[1])
    for name in ("status", "models", "nodes", "queue", "device", "logs", "help"):
        assert qlh.main([name]) == 0
    assert [s for s, _ in seen] == ["src/tui_commands.py"] * 7
    assert [a[0] for _, a in seen] == ["status", "models", "nodes", "queue",
                                       "device", "logs", "help"]


def test_no_args_enters_unified_tui_and_auto_starts(monkeypatch):
    """无参 `qlh` = 交互入口（Textual 外壳）+ 自动启动后端。"""
    seen = {}

    def fake_shell(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(qlh, "_run_textual_shell", fake_shell)
    assert qlh.main([]) == 0
    assert seen["args"] == ["--auto-start", "--screen", "chat"]


def test_chat_url_enters_unified_tui_without_remote_autostart(monkeypatch):
    seen = {}

    def fake_shell(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(qlh, "_run_textual_shell", fake_shell)
    assert qlh.main(["chat", "--host", "http://100.100.52.106:8000",
                     "--route", "distributed_preferred"]) == 0
    assert seen["args"] == [
        "--screen", "chat", "--host", "100.100.52.106", "--port", "8000",
        "--route", "distributed_preferred",
    ]


def test_chat_log_token_reaches_unified_tui(monkeypatch):
    seen = {}

    def fake_shell(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(qlh, "_run_textual_shell", fake_shell)
    assert qlh.main(["chat", "--log-token", "secret"]) == 0
    assert seen["args"] == ["--auto-start", "--screen", "chat", "--log-token", "secret"]


def test_chat_help_is_local(capsys):
    assert qlh.main(["chat", "--help"]) == 0
    output = capsys.readouterr().out
    assert "qlh chat" in output
    assert "--fixture" in output


def test_koakuma_is_an_alias_of_qlh():
    """`koakuma` 与 `qlh` 等价：两个薄壳必须转发到同一个入口（qlh.py）。"""
    from pathlib import Path

    root = Path(qlh.__file__).resolve().parent
    bat = (root / "koakuma.bat").read_text(encoding="utf-8")
    sh = (root / "koakuma.sh").read_text(encoding="utf-8")
    assert "qlh.py" in bat, "koakuma.bat 必须调用 qlh.py"
    assert "qlh.py" in sh, "koakuma.sh 必须调用 qlh.py"
    assert "%*" in bat, "koakuma.bat 必须原样透传参数"
    assert '"$@"' in sh, "koakuma.sh 必须原样透传参数"


def test_windows_launchers_are_ascii_only():
    """cmd 层不应依赖当前代码页；中文文案留给 Python/Textual。"""
    from pathlib import Path

    root = Path(qlh.__file__).resolve().parent
    for name in ("qlh.bat", "koakuma.bat", "start_tui.bat", "bjtu.bat", "start_backend.bat"):
        content = (root / name).read_text(encoding="utf-8")
        assert all(ord(char) < 128 for char in content), name


@pytest.mark.parametrize("command", [[], ["chat"], ["tui"], ["ui"]])
@pytest.mark.parametrize(
    "environment_bind, cli_bind, expected_bind",
    [
        (None, None, "127.0.0.1"),
        ("0.0.0.0", None, "0.0.0.0"),
        (None, "0.0.0.0", "0.0.0.0"),
        ("0.0.0.0", "127.0.0.1", "127.0.0.1"),
        (None, "::", "::"),
    ],
)
def test_shell_keeps_client_destination_separate_from_backend_bind(
    monkeypatch, command, environment_bind, cli_bind, expected_bind,
):
    seen = {}
    if environment_bind is None:
        monkeypatch.delenv("QLH_BACKEND_HOST", raising=False)
    else:
        monkeypatch.setenv("QLH_BACKEND_HOST", environment_bind)
    monkeypatch.setattr(qlh, "_textual_available", lambda: True)

    def fake_textual(host, port, **options):
        seen.update(host=host, port=port, **options)
        return 0

    monkeypatch.setitem(sys.modules, "tui_textual", SimpleNamespace(run=fake_textual))
    arguments = [*command, "--host", "http://127.0.0.1:18000"]
    if cli_bind is not None:
        arguments.extend(["--backend-host", cli_bind])
    assert qlh.main(arguments) == 0
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 18000)
    supervisor = seen["supervisor"]
    assert supervisor.host == "127.0.0.1"
    assert supervisor.bind_host == expected_bind
    assert supervisor.health_url == "http://127.0.0.1:18000/api/health"


@pytest.mark.parametrize("command", [[], ["chat"], ["tui"], ["ui"]])
def test_remote_shell_does_not_autostart_even_with_explicit_bind(monkeypatch, command):
    seen = {}
    monkeypatch.setattr(qlh, "_textual_available", lambda: True)
    monkeypatch.setenv("QLH_BACKEND_HOST", "0.0.0.0")
    monkeypatch.setitem(
        sys.modules, "tui_textual",
        SimpleNamespace(run=lambda host, port, **options: (seen.update(options), 0)[1]),
    )
    assert qlh.main([
        *command, "--host", "http://192.0.2.10:18000",
        "--backend-host", "0.0.0.0", "--auto-start", "--auto-start",
    ]) == 0
    assert seen["supervisor"] is None


def test_missing_backend_host_is_reported_before_start(monkeypatch, capsys):
    def unexpected_start(*args):
        pytest.fail("invalid bind arguments must not start the shell")

    monkeypatch.setattr(qlh, "_run_unified", unexpected_start)
    assert qlh.main(["chat", "--backend-host"]) == 2
    assert "--backend-host 缺少参数" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [["--help"], ["chat", "--help"]])
def test_help_explains_backend_bind_configuration(arguments, capsys):
    assert qlh.main(arguments) == 0
    output = capsys.readouterr().out
    assert "--backend-host" in output
    assert "QLH_BACKEND_HOST" in output
    assert "客户端" in output
