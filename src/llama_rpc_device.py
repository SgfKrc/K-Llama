"""RPC device 接入层：把 ggml-rpc-server 的算力 device 交给本进程的 llama.cpp 使用。

机理与实测证据见
``local_docs/evidence/llama-pc-rpc/CORE-LLAMA-PC-RPC-01-rpc-backend-注册与-ctypes-装配-2026-09-17.md``：

1. RPC backend **故意不参与** ``ggml_backend_dev_count()`` 枚举
   （``ggml-rpc.cpp:1958``：其自带 reg 的 ``context == NULL``，``get_device`` 直接
   ``GGML_ABORT``）；
2. device 只能由 ``ggml_backend_rpc_add_server(endpoint)`` 造出（返回**带 devices 的
   新 reg**），并**必须再** ``ggml_backend_register(reg)`` 进全局注册表 —— 这是
   ``common/arg.cpp:1174-1181`` 的官方流程，漏掉最后一步则枚举里永远只有 CPU；
3. 之后把 device 写成 **NULL 结尾数组** 塞进 ``llama_model_params.devices``，
   模型即经 RPC 设备加载/计算。

设计约束（fail-closed）：任何一步失败都抛 ``RpcInjectionError``，**绝不静默退回纯 CPU**
——否则调用方会以为在跑 RPC 而实际没有。``llama_cpp.Llama`` 的 ``__init__`` 不接受
``devices``（绑定里标为 unused），因此注入点在 ``llama_model_load_from_file`` 调用处。
"""

from __future__ import annotations

import ctypes
import logging
import os
import pathlib
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: 与当前 llama.dll 协议匹配的 worker（上游同源码树编译产物，runtime/ 不入库）
DEFAULT_WORKER_EXE = _REPO_ROOT / "runtime" / "llama-cpp" / "b_rpc" / "ggml-rpc-server.exe"
DEFAULT_WORKER_PORT = 50163


class RpcInjectionError(RuntimeError):
    """RPC device 注入失败（fail-closed）。"""


@dataclass(frozen=True)
class RpcDevice:
    """注入后可用的一个 RPC device。"""

    endpoint: str
    ptr: int
    name: str
    description: str


def default_lib_dir() -> pathlib.Path:
    """当前 llama-cpp-python 的 DLL 目录（含 llama.dll / ggml-rpc.dll）。"""
    import llama_cpp

    return pathlib.Path(llama_cpp.__file__).resolve().parent / "lib"


def ensure_dll_search_path(lib_dir: pathlib.Path) -> None:
    """让 ``ctypes.CDLL`` 与 python 都能找到同目录依赖 DLL。

    否则 ``ctypes.CDLL('ggml.dll')`` 会因依赖解析失败报 ``exit 127``（不是逻辑错误）。
    """
    lib_dir = pathlib.Path(lib_dir)
    if not lib_dir.is_dir():
        raise RpcInjectionError(f"llama_cpp lib 目录不存在：{lib_dir}")
    try:
        os.add_dll_directory(str(lib_dir))  # 必须在 import llama_cpp 之前调用
    except (AttributeError, OSError):  # 非 Windows / 已加入
        pass
    os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")


class _GgmlApi:
    """按签名绑定注入所需的 ggml 符号（分布在三个 DLL 里）。"""

    def __init__(self, lib_dir: pathlib.Path) -> None:
        self.base = ctypes.CDLL(str(lib_dir / "ggml-base.dll"))
        self.ggml = ctypes.CDLL(str(lib_dir / "ggml.dll"))
        self.rpc = ctypes.CDLL(str(lib_dir / "ggml-rpc.dll"))

        self.ggml.ggml_backend_load_all.argtypes = []
        self.ggml.ggml_backend_load_all.restype = None
        self.ggml.ggml_backend_reg_by_name.argtypes = [ctypes.c_char_p]
        self.ggml.ggml_backend_reg_by_name.restype = ctypes.c_void_p
        self.ggml.ggml_backend_register.argtypes = [ctypes.c_void_p]
        self.ggml.ggml_backend_register.restype = None
        self.ggml.ggml_backend_dev_count.argtypes = []
        self.ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
        self.ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
        self.ggml.ggml_backend_dev_get.restype = ctypes.c_void_p

        self.base.ggml_backend_reg_get_proc_address.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.base.ggml_backend_reg_get_proc_address.restype = ctypes.c_void_p
        self.base.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
        self.base.ggml_backend_dev_name.restype = ctypes.c_char_p
        self.base.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
        self.base.ggml_backend_dev_description.restype = ctypes.c_char_p

        self.rpc.ggml_backend_rpc_reg.argtypes = []
        self.rpc.ggml_backend_rpc_reg.restype = ctypes.c_void_p

    def device_count(self) -> int:
        return int(self.ggml.ggml_backend_dev_count())

    def device_ptr(self, index: int) -> int:
        return int(self.ggml.ggml_backend_dev_get(index))

    def device_name(self, ptr: int) -> str:
        return (self.base.ggml_backend_dev_name(ptr) or b"").decode("utf-8", "replace")

    def device_description(self, ptr: int) -> str:
        return (self.base.ggml_backend_dev_description(ptr) or b"").decode("utf-8", "replace")


class LocalRpcWorker:
    """本机 ``ggml-rpc-server`` 子进程（仅用于单机实验/离线验证）。"""

    def __init__(
        self,
        exe: str | os.PathLike[str] = DEFAULT_WORKER_EXE,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_WORKER_PORT,
        threads: int = 4,
        log_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.exe = pathlib.Path(exe)
        self.host = host
        self.port = int(port)
        self.threads = int(threads)
        self.log_path = pathlib.Path(log_path) if log_path else None
        self.proc: subprocess.Popen | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def start(self, timeout: float = 30.0) -> "LocalRpcWorker":
        if not self.exe.exists():
            raise RpcInjectionError(f"ggml-rpc-server 不存在：{self.exe}")
        kwargs: dict = {"cwd": str(self.exe.parent)}
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.log_path.open("wb")
            kwargs.update(stdout=handle, stderr=subprocess.STDOUT)
        else:
            kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.proc = subprocess.Popen(
            [str(self.exe), "--host", self.host, "--port", str(self.port),
             "--device", "CPU", "--threads", str(self.threads)],
            **kwargs,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RpcInjectionError(f"worker 启动即退出（rc={self.proc.returncode}）：{self.exe}")
            with socket.socket() as s:
                s.settimeout(0.5)
                if s.connect_ex((self.host, self.port)) == 0:
                    logger.info("RPC worker 就绪：%s (pid=%s)", self.endpoint, self.proc.pid)
                    return self
            time.sleep(0.3)
        self.stop()
        raise RpcInjectionError(f"worker 未在 {timeout:.0f}s 内监听 {self.endpoint}")

    def stop(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is not None:
            self.proc = None
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        self.proc = None

    def __enter__(self) -> "LocalRpcWorker":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class RpcBackendInjector:
    """把 ``endpoints`` 上的 RPC device 注册进本进程的 ggml 全局注册表。"""

    def __init__(self, lib_dir: str | os.PathLike[str] | None = None) -> None:
        self.lib_dir = pathlib.Path(lib_dir) if lib_dir else default_lib_dir()
        self.api: _GgmlApi | None = None
        self.devices: list[RpcDevice] = []
        self._keepalive: list[object] = []

    # ------------------------------------------------------------ attach

    def attach(self, endpoints: Sequence[str]) -> list[RpcDevice]:
        """按 ``common/arg.cpp:add_rpc_devices()`` 的流程注入，失败即抛。"""
        if not endpoints:
            raise RpcInjectionError("endpoints 为空")
        ensure_dll_search_path(self.lib_dir)
        self.api = _GgmlApi(self.lib_dir)

        import llama_cpp.llama_cpp as lc

        lc.llama_backend_init()
        try:
            self.api.ggml.ggml_backend_load_all()
        except AttributeError:  # 已静态注册的构建
            pass

        rpc_reg = int(self.api.ggml.ggml_backend_reg_by_name(b"RPC") or 0)
        if not rpc_reg:  # 兜底：直接注册 backend 自带（无 device）的 reg
            logger.warning("注册表里没有 RPC backend，改用 ggml_backend_rpc_reg() 兜底")
            rpc_reg = int(self.api.rpc.ggml_backend_rpc_reg())
            self.api.ggml.ggml_backend_register(ctypes.c_void_p(rpc_reg))

        add_server_addr = int(
            self.api.base.ggml_backend_reg_get_proc_address(
                ctypes.c_void_p(rpc_reg), b"ggml_backend_rpc_add_server"
            ) or 0
        )
        if not add_server_addr:
            raise RpcInjectionError("RPC backend 未导出 ggml_backend_rpc_add_server")
        add_server = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(add_server_addr)
        self._keepalive.append(add_server)

        for endpoint in endpoints:
            srv_reg = int(add_server(str(endpoint).encode()) or 0)
            if not srv_reg:
                raise RpcInjectionError(
                    f"add_server({endpoint}) 返回 NULL：worker 未就绪，或 RPC 协议与 llama.dll 不匹配"
                )
            # ★ 官方流程的最后一步；漏掉则 device 永远枚举不到
            self.api.ggml.ggml_backend_register(ctypes.c_void_p(srv_reg))

        self.devices = self._collect(endpoints)
        if not self.devices:
            raise RpcInjectionError(f"注入后仍未枚举到 RPC device（endpoints={list(endpoints)}）")
        return self.devices

    def _collect(self, endpoints: Sequence[str]) -> list[RpcDevice]:
        assert self.api is not None
        wanted = {str(e) for e in endpoints}
        found: list[RpcDevice] = []
        for index in range(self.api.device_count()):
            ptr = self.api.device_ptr(index)
            name = self.api.device_name(ptr)
            desc = self.api.device_description(ptr)
            if name.startswith("RPC") and (desc in wanted or name != "RPC"):
                found.append(RpcDevice(endpoint=desc, ptr=ptr, name=name, description=desc))
        return found

    def cpu_device_ptr(self) -> int | None:
        """本机 CPU device 指针（分片时把它与 RPC device 一起放进 devices）。"""
        assert self.api is not None, "attach() 之后才能查询 device"
        for index in range(self.api.device_count()):
            ptr = self.api.device_ptr(index)
            if self.api.device_name(ptr) == "CPU":
                return ptr
        return None

    # ------------------------------------------------------------ llama 加载注入

    def device_array(self, devices: Sequence[RpcDevice] | None = None, *,
                     include_cpu: bool = False) -> ctypes.Array:
        """构造 NULL 结尾的 device 数组并保活（llama.cpp 加载期读它）。

        ``include_cpu=True`` 时把本机 CPU device 放在**最前**，顺序即 tensor_split 的顺序
        —— 于是层会在「本地 CPU + 远端 RPC」之间按比例分配（分片/部分驻留）。
        """
        entries = list(devices if devices is not None else self.devices)
        ptrs: list[int] = []
        if include_cpu:
            cpu = self.cpu_device_ptr()
            if cpu is None:
                raise RpcInjectionError("--include_cpu 要求注册表里有 CPU device")
            ptrs.append(cpu)
        ptrs.extend(dev.ptr for dev in entries)
        if not ptrs:
            raise RpcInjectionError("没有可注入的 device")
        array = (ctypes.c_void_p * (len(ptrs) + 1))()
        for i, ptr in enumerate(ptrs):
            array[i] = ctypes.c_void_p(ptr)
        array[len(ptrs)] = None
        self._keepalive.append(array)
        return array

    @contextmanager
    def patched_llama_loader(self, device_array: ctypes.Array | None = None, *,
                             tensor_split: Sequence[float] | None = None,
                             n_gpu_layers: int | None = None,
                             use_extra_bufts: bool | None = None) -> Iterator[ctypes.Array]:
        """在 ``Llama(...)`` 构造期间把 devices（+可选 tensor_split / use_extra_bufts）注入加载调用。

        ``llama-cpp-python`` 的 ``Llama.__init__`` 既不接受 ``devices`` 也不接受
        ``use_extra_bufts``（后者是 llama.cpp 的 ``--repack/-nr`` 开关），但
        ``_internals.LlamaModel.__init__`` 经模块属性调用该函数，故 patch 模块属性有效。

        ``use_extra_bufts=False`` 用于**跨路径数值对齐**：本机 CPU 路径默认启用 weight
        repacking（CPU_REPACK），而 RPC 路径的权重落在远端 buffer 里无法 repack，两条路径
        的浮点结果不同（max|Δ|≈0.98，见 ``local_docs`` §11/§13）；关掉后二者**逐比特一致**。
        """
        import llama_cpp.llama_cpp as lc

        array = device_array if device_array is not None else self.device_array()
        address = ctypes.cast(array, ctypes.c_void_p).value
        split = None
        if tensor_split is not None:
            split = (ctypes.c_float * len(tensor_split))(*[float(x) for x in tensor_split])
            self._keepalive.append(split)      # 加载期 llm 会读它，必须保活
        original = lc.llama_model_load_from_file

        def patched(path_model, params, *args, **kwargs):
            params.devices = address
            if split is not None:
                params.tensor_split = ctypes.cast(split, ctypes.POINTER(ctypes.c_float))
            if n_gpu_layers is not None:
                params.n_gpu_layers = int(n_gpu_layers)
            if use_extra_bufts is not None:
                params.use_extra_bufts = bool(use_extra_bufts)
            return original(path_model, params, *args, **kwargs)

        lc.llama_model_load_from_file = patched
        try:
            yield array
        finally:
            lc.llama_model_load_from_file = original

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self.devices = []
        self._keepalive.clear()


@contextmanager
def patched_model_params(**overrides) -> Iterator[None]:
    """临时覆盖 ``llama_model_load_from_file`` 的 ``params`` 字段（不涉及 devices）。

    纯本机路径也需要它：``use_extra_bufts`` 未被 llama-cpp-python 暴露，而"跨路径数值
    对齐"要求**两侧都关** CPU_REPACK（RPC 侧天然关闭），否则本机仍会 repack。
    """
    import llama_cpp.llama_cpp as lc

    original = lc.llama_model_load_from_file

    def patched(path_model, params, *args, **kwargs):
        for key, value in overrides.items():
            setattr(params, key, value)
        return original(path_model, params, *args, **kwargs)

    lc.llama_model_load_from_file = patched
    try:
        yield
    finally:
        lc.llama_model_load_from_file = original


class RpcCleanupError(RpcInjectionError):
    """A failed open still owns a worker and requires an explicit retry."""

    def __init__(self, owner, init_error, cleanup_error):
        super().__init__(f"RPC open failed: {init_error}; cleanup failed: {cleanup_error}")
        self.cleanup_owner = owner


@dataclass
class RpcSession:
    """一次性接入会话：可含本地 worker（可选）+ device 注入。"""

    endpoints: list[str]
    worker: LocalRpcWorker | None = None
    injector: RpcBackendInjector | None = None

    @classmethod
    def open(
        cls,
        endpoints: Sequence[str] | None = None,
        *,
        autostart_worker: str | os.PathLike[str] | None = None,
        worker_port: int = DEFAULT_WORKER_PORT,
        worker_threads: int = 4,
        worker_log: str | os.PathLike[str] | None = None,
        lib_dir: str | os.PathLike[str] | None = None,
    ) -> "RpcSession":
        session = cls(endpoints=[str(e) for e in (endpoints or [])])
        try:
            if autostart_worker is not None:
                session.worker = LocalRpcWorker(autostart_worker, port=worker_port,
                                                threads=worker_threads, log_path=worker_log)
                session.worker.start()
                session.endpoints.append(session.worker.endpoint)
            if not session.endpoints:
                raise RpcInjectionError("既没有 endpoints，也没有 autostart_worker")
            session.injector = RpcBackendInjector(lib_dir=lib_dir)
            session.injector.attach(session.endpoints)
            return session
        except Exception as init_error:
            try:
                session.close()
            except Exception as cleanup_error:
                raise RpcCleanupError(session, init_error, cleanup_error) from cleanup_error
            raise

    def close(self) -> None:
        if self.injector is not None:
            self.injector.close()
            self.injector = None
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def __enter__(self) -> "RpcSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
