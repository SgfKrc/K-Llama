"""进程内 TUI 后端监督器。

仅在统一 ``qlh`` 交互入口显式启用时使用。管理 TUI 直接连接已有后端时不
导入 FastAPI/uvicorn，因此 Edge 的标准库 TUI 依赖边界保持不变。
"""

from __future__ import annotations

import os
import threading
import time
import logging
import urllib.error
import urllib.request
from typing import Mapping, Optional


class BackendStartupError(RuntimeError):
    """后端未能在规定时间内就绪。"""


class BackendSupervisor:
    """探活已有后端，否则在当前 Python 进程的 daemon 线程中启动它。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        bind_host: Optional[str] = None,
        environment: Optional[Mapping[str, str]] = None,
        probe_timeout: float = 1.0,
        startup_timeout: float = 30.0,
    ) -> None:
        self.host = str(host or "127.0.0.1")
        # host is the client/probe destination; only bind_host configures the
        # listener. Preserve the local IPv6 loopback default without exposing
        # any non-loopback interface unless explicitly configured.
        default_bind = "::1" if self.host == "::1" else "127.0.0.1"
        self.environment = None if environment is None else dict(environment)
        listener_environment = os.environ if self.environment is None else self.environment
        configured_bind = (
            listener_environment.get("QLH_BACKEND_HOST") if bind_host is None else bind_host
        )
        self.bind_host = str(configured_bind or "").strip() or default_bind
        self.port = int(port)
        self.probe_timeout = max(0.2, float(probe_timeout))
        # A refused local connection should not hold the splash screen for the
        # full readiness timeout before the supervisor starts the backend.
        self.initial_probe_timeout = min(self.probe_timeout, 0.2)
        self.startup_timeout = max(1.0, float(startup_timeout))
        self.server = None
        self.thread: Optional[threading.Thread] = None
        self.started_here = False
        self.error: Optional[BaseException] = None
        self.status_message = "检查本地后端"

    def _set_status(self, message: str) -> None:
        self.status_message = str(message)

    @property
    def health_url(self) -> str:
        from network_address import build_url

        return build_url("http", self.host, self.port) + "/api/health"

    def probe(self, *, timeout: Optional[float] = None) -> bool:
        effective_timeout = self.probe_timeout if timeout is None else max(
            0.05, float(timeout),
        )
        try:
            with urllib.request.urlopen(
                self.health_url, timeout=effective_timeout,
            ) as response:
                return 200 <= int(getattr(response, "status", 200)) < 300
        except (OSError, urllib.error.URLError, ValueError):
            return False

    def ensure_ready(self) -> bool:
        """返回是否由本监督器启动了后端。"""
        self._set_status("检查本地后端")
        if self.probe(timeout=self.initial_probe_timeout):
            self._set_status("后端已在运行")
            return False
        if self.host.lower() not in {"127.0.0.1", "localhost", "::1"}:
            self._set_status("等待远程后端")
            raise BackendStartupError(
                "远程后端未就绪，统一入口不会在远程节点启动后端: %s" % self.health_url,
            )

        try:
            self._set_status("加载后端组件")
            import api_server
            import uvicorn
        except Exception as exc:  # pragma: no cover - environment-specific
            self._set_status("后端组件加载失败")
            raise BackendStartupError("无法加载进程内后端: %s" % exc) from exc

        # api_server 的日志仍保留在文件和内存 handler；TUI 独占终端，
        # 不允许后端启动/访问日志穿透到同一个 stdout/stderr。
        if hasattr(api_server, "_get_app_runtime"):
            api_server._get_app_runtime().configure_listener(self.port, environment=self.environment)
        api_server._console_logging_enabled = False
        self._silence_console_logging()
        self._set_status("启动 API 服务")

        config = uvicorn.Config(
            api_server.app,
            host=self.bind_host,
            port=self.port,
            log_level="warning",
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=10,
        )
        self.server = uvicorn.Server(config)
        register = getattr(api_server, "register_uvicorn_server", None)
        if callable(register):
            register(self.server)
        self.thread = threading.Thread(
            target=self._run_server,
            name="qlh-tui-backend",
            daemon=True,
        )
        self.started_here = True
        self.thread.start()

        self._set_status("等待 API 健康检查")
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.probe():
                self._set_status("后端已就绪")
                return True
            if self.error is not None:
                break
            if self.thread is not None and not self.thread.is_alive():
                break
            time.sleep(0.1)
        detail = str(self.error or "后端线程已退出或健康检查超时")
        self._set_status("后端启动失败")
        self.stop()
        raise BackendStartupError(
            "进程内后端未在 %.0f 秒内就绪: %s" % (self.startup_timeout, detail),
        )

    def _run_server(self) -> None:
        try:
            self.server.run()
        except BaseException as exc:  # surface the failure to ensure_ready
            self.error = exc

    @staticmethod
    def _silence_console_logging() -> None:
        """静默控制台 handler，保留文件与内存日志供日志屏查看。"""
        root = logging.getLogger()
        for handler in root.handlers:
            if type(handler) is logging.StreamHandler and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.CRITICAL + 1)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if type(handler) is logging.StreamHandler and not isinstance(
                    handler, logging.FileHandler
                ):
                    handler.setLevel(logging.CRITICAL + 1)

    def stop(self) -> float | None:
        """请求本监督器创建的 server 停止，并短暂等待线程结束。

        返回 thread 存活时**已等待的秒数**（`None` 表示无需等待或已正常退出）。

        ★ 2026-10-27（#128）：**放弃等待时必须说出来**。

        此前这里是裸 `join(timeout=3.0)`：超时后线程仍然活着，但函数静默返回，
        调用方以为后端已停。后果是**旧 TUI 的 Python 进程继续存活**，而新进程
        恢复旧部署时 `_dispatch_pipeline_model_cleanup` 会依据旧进程 PID 判定
        "本机清债"并继续等待 ⇒ `GET` 正常但**选模 `POST` 长时间没有新 operation**。

        这里不改变"有界等待"这个决定（进程内硬杀另一个线程本来就不安全），
        只是把"没等到"变成**可观测的返回值 + 警告日志**，让调用方能提示用户
        "旧进程仍在，请确认后再启动"。后端侧 `_graceful_exit` 另有
        `QLH_SHUTDOWN_GRACE_SECONDS` 的兜底强退，两处合起来才能保证旧 PID 消失。
        """
        if not self.started_here or self.server is None:
            return None
        self.server.should_exit = True
        if self.thread is not None and self.thread.is_alive():
            started_at = time.monotonic()
            self.thread.join(timeout=3.0)
            waited = time.monotonic() - started_at
            if self.thread.is_alive():
                module_logger = logging.getLogger(__name__)
                module_logger.warning(
                    "event=backend_stop_thread_alive waited_s=%.1f "
                    "hint=后端未在等待窗口内退出；旧进程仍在，选模可能被旧部署清理阻塞",
                    waited,
                )
                return waited
        return None
