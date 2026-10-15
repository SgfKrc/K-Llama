"""llama-server 子进程生命周期（票 15b `LLAMA-SERVER-LIFECYCLE-01`）。

**为什么需要这个模块**

绑定路径（`llama-cpp-python`）实测是 **CPU-only**（`llama_cpp/lib/` 下没有
`ggml-cuda.dll`），而且拿不到 server 独有的能力：`--spec-type` 投机族、
`--cpu-moe`（实测 35B-A3B 上 `-ngl 99 --cpu-moe` = 15.94 tok/s，是纯 CPU 的
2.66×，且只吃 2863 MiB 显存）、`--cache-reuse`。据裁决（2026-10-12）：

    有 CUDA 走 server 子进程；无 CUDA（边缘）老实走绑定。

本模块负责前者的**进程生命周期**，模式与 `llama_rpc_device.LocalRpcWorker`
同构（Popen + 就绪探测 + 回收 + 上下文管理器），差别只有一处：
**就绪判据用 HTTP `/health` 而不是 TCP connect** —— TCP 通只说明端口起来了，
不代表模型加载完成（实测 4B 数秒、35B 数十秒的差距）。

**边界**：本路径只承担「整模推理」。层段/keep-head 前向依赖低层 C API 与本仓
llama.cpp 补丁（`t_layer_inp` 槽位），server 进程给不了 => 层段 worker 永远走绑定。
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger("llama_server_process")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_STARTUP_TIMEOUT = 180.0


@dataclass
class ServerSpec:
    """启动一个 llama-server 所需的参数（与 QLH 的加载语义对接）。

    `argv_prefix` 是给测试/包装脚本用的：它插在可执行文件之后、server 参数之前
    （例如用 `python <fake_server.py>` 顶替真 server 时）。
    """

    model_path: str
    exe: str | os.PathLike[str]
    argv_prefix: list[str] = field(default_factory=list)
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    cpu_moe: bool = False
    spec_type: str = ""
    chat_template_kwargs: dict | None = None
    mmproj_path: str = ""
    #: 推理开关片段，由 `inference_switches.server_argv_args()` 生成
    #: （`--threads` / `--cache-type-k` / `--cache-type-v` / `--flash-attn`）。
    #: 默认空 ⇒ 命令行在本字段引入前后逐字一致。
    inference_args: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)

    def to_argv(self, *, host: str, port: int) -> list[str]:
        argv: list[str] = [str(self.exe), *[str(a) for a in self.argv_prefix]]
        argv += [
            "-m", str(self.model_path),
            "-c", str(int(self.n_ctx)),
            "-ngl", str(int(self.n_gpu_layers)),
            "--host", str(host),
            "--port", str(int(port)),
        ]
        argv += [str(a) for a in self.inference_args]
        if self.mmproj_path:
            argv += ["--mmproj", str(self.mmproj_path)]
        if self.cpu_moe:
            argv.append("--cpu-moe")
        if self.spec_type:
            argv += ["--spec-type", str(self.spec_type)]
        if self.chat_template_kwargs is not None:
            argv += [
                "--chat-template-kwargs",
                json.dumps(self.chat_template_kwargs, separators=(",", ":")),
            ]
        argv += [str(a) for a in self.extra_args]
        return argv


class LlamaServerProcess:
    """管理一个本机 llama-server：启动、探活、回收。"""

    def __init__(
        self,
        spec: ServerSpec,
        *,
        host: str = DEFAULT_HOST,
        port: int = 0,
        log_path: str | os.PathLike[str] | None = None,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        health_poll_interval: float = 0.25,
    ) -> None:
        self.spec = spec
        self.host = host
        self._requested_port = int(port)
        self.port = int(port)
        self.log_path = pathlib.Path(log_path) if log_path else None
        self.startup_timeout = float(startup_timeout)
        self.health_poll_interval = float(health_poll_interval)
        self.proc: subprocess.Popen | None = None

    # ---------------------------------------------------------------- state
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---------------------------------------------------------------- start
    def _pick_port(self) -> int:
        """`port=0` 时由内核分配一个空闲端口（避免与既有实例冲突）。"""
        if self._requested_port:
            return self._requested_port
        with socket.socket() as probe:
            probe.bind((self.host, 0))
            return int(probe.getsockname()[1])

    def start(self) -> "LlamaServerProcess":
        exe = pathlib.Path(str(self.spec.exe))
        if not exe.exists():
            raise RuntimeError(f"llama-server 可执行文件不存在：{exe}")

        self.port = self._pick_port()
        argv = self.spec.to_argv(host=self.host, port=self.port)

        popen_kwargs: dict = {"cwd": str(exe.parent)}
        handle = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.log_path.open("wb")
            popen_kwargs.update(stdout=handle, stderr=subprocess.STDOUT)
        else:
            popen_kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logger.info("启动 llama-server：%s", " ".join(argv[:6]) + " ...")
        self.proc = subprocess.Popen(argv, **popen_kwargs)
        try:
            self._wait_ready()
        except BaseException:
            self.stop()
            raise
        finally:
            if handle is not None:
                handle.close()
        logger.info("llama-server 就绪：%s (pid=%s)", self.base_url, self.proc.pid)
        return self

    def _wait_ready(self) -> None:
        """就绪判据 = `GET /health` 返回 `{"status":"ok"}`（不是 TCP connect）。"""
        deadline = time.time() + self.startup_timeout
        health_url = f"{self.base_url}/health"
        while time.time() < deadline:
            proc = self.proc
            if proc is None:
                raise RuntimeError("llama-server 句柄丢失")
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server 启动即退出（rc={proc.returncode}）",
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        payload = json.loads(response.read() or b"{}")
                        if isinstance(payload, dict) and payload.get("status") == "ok":
                            return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(self.health_poll_interval)
        raise RuntimeError(
            f"llama-server 未在 {self.startup_timeout:.0f}s 内就绪（{health_url}）",
        )

    # ----------------------------------------------------------------- stop
    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - 极端情况
                logger.warning("llama-server 未能回收（pid=%s）", proc.pid)

    # ------------------------------------------------------------ context
    def __enter__(self) -> "LlamaServerProcess":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


__all__ = ["DEFAULT_HOST", "DEFAULT_STARTUP_TIMEOUT", "LlamaServerProcess", "ServerSpec"]
