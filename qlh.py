"""Cross-platform command entry for the unified QLH TUI.

2026-09-17 起：

* 交互入口 = **Textual 外壳**（``src/tui_textual.py``）——自绘 ANSI 的标准库 TUI
  （``tui_admin.py`` / ``tui_chat_screen.py`` / ``tui_splash.py``）在真实 conhost 下实测
  不可见，已整体归档到 ``_to_delete/``（见 ``docs/TUI重写方案``）；
* 单命令模式 = **只读薄层** ``src/tui_commands.py``（status/models/nodes/queue/device/logs/help），
  不再依赖旧 TUI 的命令注册表；写操作请在交互界面或 API 侧完成；
* ``--tui-engine builtin`` 保留参数但**明确报已归档**（rc=2），不静默失效。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parent

#: 单命令模式支持的只读子命令（实现在 src/tui_commands.py）
READ_ONLY_SUBCOMMANDS = ("status", "models", "nodes", "queue", "device", "logs", "help")


def _usage() -> str:
    return (
        "QLH application\n"
        "  qlh maintenance <command>  应用验签、版本切换、回退、修复与数据保留\n"
        "  qlh migrate-state --source PATH --target PATH --backup PATH [--dry-run]\n"
        "  qlh                         进入交互外壳（Textual；本机后端自动启动）\n"
        "  qlh [options]               同上；选项直通（--port/--route/--host/--thinking/--log-token…）\n"
        "  qlh chat [--host URL] [--backend-host ADDRESS] [--route auto|local_only|distributed_preferred|distributed_required]\n"
        "    --host                   客户端连接目标；远程目标不自动启动后端\n"
        "    --backend-host           本机后端监听地址（默认 loopback；也可用 QLH_BACKEND_HOST）\n"
        "  qlh chat --fixture PATH     离线回放 SSE fixture（不联网、不依赖 UI）\n"
        "  qlh status|models|nodes|queue|device|logs|help\n"
        "                              单命令模式（只读、不启动后端、无 UI 依赖）\n"
    )


def _python() -> str:
    return sys.executable or os.environ.get("PYTHON", "python")


def _ensure_src_on_path() -> str:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def _textual_available() -> bool:
    try:
        return importlib.util.find_spec("textual") is not None
    except Exception:  # noqa: BLE001 - 探测失败按不可用处理
        return False


def _option_value(args: list[str], name: str, default: str) -> str:
    if name in args:
        index = args.index(name) + 1
        if index < len(args):
            return args[index]
    return default


def _run(module_script: str, args: list[str]) -> int:
    """单命令/fixture 的子进程语义（薄层与 fixture 都走这里，保持 CLI 隔离）。"""
    completed = subprocess.run(
        [_python(), str(ROOT / module_script), *args],
        cwd=str(ROOT),
    )
    return int(completed.returncode)


def _chat_args(args: list[str]) -> list[str]:
    """把 URL 形状的 chat CLI 翻译成 ``--host/--port`` 形式，并补默认值。"""
    translated = ["--auto-start", "--screen", "chat"]
    i = 0
    while i < len(args):
        value = args[i]
        if value == "--fixture":
            # Fixture mode is intentionally independent of the backend.
            return ["--fixture", args[i + 1]]
        if value in {"--host", "--backend-host", "--route", "--timeout", "--port", "--interval", "--tui-engine", "--log-token"}:
            if i + 1 >= len(args):
                raise ValueError("%s 缺少参数" % value)
            item = args[i + 1]
            if value == "--host":
                # Internal dispatch parses the translated hostname again;
                # bracket a bare IPv6 address before treating it as a URL.
                if "://" not in item and item.count(":") > 1 and not item.startswith("["):
                    item = "[" + item + "]"
                parsed = urllib.parse.urlsplit(item if "://" in item else "http://" + item)
                if parsed.hostname:
                    translated.extend(["--host", parsed.hostname])
                if parsed.port:
                    translated.extend(["--port", str(parsed.port)])
            elif value == "--route":
                translated.extend(["--route", item])
            else:
                translated.extend([value, item])
            i += 2
            continue
        if value in {"--thinking", "--plain", "--no-splash", "--no-color", "--auto-start"}:
            translated.append(value)
            i += 1
            continue
        if value == "--screen":
            # 内部调用（_run_unified(["--auto-start", "--screen", "chat"])）也会经过这里
            if i + 1 >= len(args):
                raise ValueError("--screen 缺少参数")
            translated.extend(["--screen", args[i + 1]])
            i += 2
            continue
        raise ValueError("chat 不支持参数: %s" % value)
    if "--host" in translated:
        host = translated[translated.index("--host") + 1].lower()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            translated = [item for item in translated if item != "--auto-start"]
    return translated


def _shell_options(translated: list[str]) -> dict:
    def value_of(name: str, default):
        return translated[translated.index(name) + 1] if name in translated else default

    environment = {}
    if "--auto-start" in translated:
        _ensure_src_on_path()
        from qlh_core.adapters.settings import SettingsStore
        from release_contract import DEFAULT_PACKAGED_NODE_ROLE
        environment = SettingsStore(default_node_role=DEFAULT_PACKAGED_NODE_ROLE).load_listener()
    default_port = environment.get("QLH_BACKEND_PORT") or environment.get("QLH_API_PORT") or 8000
    return {
        "host": value_of("--host", "127.0.0.1"),
        "bind_host": value_of("--backend-host", None),
        "port": int(value_of("--port", default_port)),
        "route": value_of("--route", "auto"),
        "interval": float(value_of("--interval", 5.0)),
        "thinking": "--thinking" in translated,
        "log_token": value_of("--log-token", ""),
        "auto_start": "--auto-start" in translated,
        "environment": environment,
    }


def _run_textual_shell(args: list[str]) -> int:
    """统一入口：进入 Textual 外壳。

    本机后端冷启动**不在这里等**——把 ``BackendSupervisor`` 交给外壳，由启动屏
    （LOGO + 启动条 + 「少女祈祷中：…」状态行）承载全过程。
    """
    _ensure_src_on_path()
    translated = _chat_args(args)
    options = _shell_options(translated)

    supervisor = None
    if options["auto_start"]:
        from tui_backend import BackendSupervisor

        supervisor = BackendSupervisor(
            host=options["host"], port=options["port"], bind_host=options["bind_host"],
            environment=options["environment"],
        )

    from tui_textual import run as run_textual

    return int(run_textual(
        options["host"],
        options["port"],
        interval=options["interval"],
        routing_preference=options["route"],
        show_thinking=options["thinking"],
        log_token=options["log_token"],
        supervisor=supervisor,
    ))


def _run_unified(args: list[str]) -> int:
    """统一交互入口：Textual 外壳（标准库 TUI 已归档，不再回退）。"""
    if _option_value(args, "--tui-engine", "auto") == "builtin":
        print("[提示] 标准库 TUI 已归档（_to_delete/）：自绘 ANSI 在真实 conhost 下不可见。")
        print("       交互请直接用 `qlh` / `koakuma`；只读查询见 `qlh help`。")
        return 2
    if not _textual_available():
        print("[错误] 未安装 textual —— 交互外壳需要它：")
        print("       python -m pip install -r requirements-tui.txt")
        print("       （只读查询仍可用：qlh status / models / nodes / queue / device / logs / help）")
        return 1
    return _run_textual_shell(args)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return _run_unified(["--auto-start", "--screen", "chat"])
    if args[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if args[0].startswith("-"):
        # 裸选项（`qlh --no-splash` / `qlh --port 9000`）等价 `qlh chat <options>`：
        # 让启动脚本可直接透传参数（start_tui.* 走的就是这条路）。
        # 未识别选项由 _chat_args 抛 ValueError -> 友好报错，不会误启动后端。
        try:
            return _run_unified(_chat_args(args))
        except ValueError as exc:
            print("[错误] %s" % exc)
            return 2

    command = args.pop(0).lower()
    if command == "migrate-state":
        _ensure_src_on_path()
        from qlh_core.adapters.state_migration import main as migrate_state
        return migrate_state(args)
    if command == "maintenance":
        sys.path.insert(0, str(ROOT / "packaging" / "packaging"))
        from app_maintenance import main as maintenance
        return maintenance(args)
    if command == "chat":
        if args and args[0] in {"-h", "--help"}:
            print(
                "qlh chat [--host URL] [--backend-host ADDRESS] [--port PORT] [--interval SECONDS] "
                "[--route auto|local_only|distributed_preferred|distributed_required] "
                "[--thinking] [--log-token TOKEN]\n"
                "qlh chat --fixture PATH\n"
                "--host 是客户端目标；--backend-host / QLH_BACKEND_HOST 配置本机监听（默认 loopback）"
            )
            return 0
        try:
            if args and args[0] == "--fixture":
                if len(args) != 2:
                    raise ValueError("--fixture 需要一个路径")
                return _run("src/tui_commands.py", ["--fixture", args[1]])
            return _run_unified(_chat_args(args))
        except ValueError as exc:
            print("[错误] %s" % exc)
            return 2
    if command == "admin":
        print("[提示] `qlh admin` 指向的标准库管理 TUI 已归档（_to_delete/）。")
        print("       交互请直接运行 `qlh` / `koakuma`；只读查询见 `qlh help`。")
        return 2
    if command in READ_ONLY_SUBCOMMANDS:
        return _run("src/tui_commands.py", [command, *args])
    if command in {"tui", "ui"}:
        return _run_unified(["--auto-start", "--screen", "chat", *args])
    print("unknown qlh command: %s\n\n%s" % (command, _usage()))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
