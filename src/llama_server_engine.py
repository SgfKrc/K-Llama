"""本机 llama-server 引擎（票 15b 接线 + 15c server 半边）。

**组合，而不是重写**

* 进程生命周期来自 `llama_server_process.LlamaServerProcess`（票 15b）；
* HTTP 交互**复用既有的** OpenAI 兼容客户端 `island_engine.IslandEngine`
  （票面要求的"复用"——不另写传输层）。

**为什么对外仍报 `engine_type = "llama_cpp"`**

server 本来就是 llama.cpp 家族。把它做成独立的 `BackendId` 会波及枚举、归一化与所有
`backend_id_for` 分支，还会与 `pytorch`(D 档) 的默认判据纠缠。这里只在**「整模 GGUF」
这个子域内**分流，差异由 `inference_path` / `capability()` 单独暴露（票 15a 的能力声明）。

**边界**：只承担整模推理。层段接力必须走绑定路径（`serves_layer_segments=False`），
由 `inference_path_selector` 的守卫负责拒绝误用。
"""

from __future__ import annotations

import pathlib
from typing import Any, Iterator

import logging

from inference_path_selector import SERVER

logger = logging.getLogger("llama_server_engine")

FAMILY_ENGINE_TYPE = "llama_cpp"


def default_server_exe() -> str:
    """解析 `llama-server` 可执行文件：先看配置，再扫 `runtime/llama-cpp/*/`。

    实测（2026-10-13）本机可用的构建在 `runtime/llama-cpp/b10964/`（含 CUDA）。
    """
    import config as _cfg

    configured = str(getattr(_cfg, "LLAMA_SERVER_EXE", "") or "").strip()
    if configured:
        return configured
    root = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "llama-cpp"
    if not root.is_dir():
        return ""
    candidates = sorted(root.glob("*/llama-server.exe")) + sorted(
        root.glob("*/llama-server"),
    )
    return str(candidates[-1]) if candidates else ""


class LlamaServerEngine:
    """以本机 `llama-server` 子进程承担整模推理的引擎实现。"""

    engine_type = FAMILY_ENGINE_TYPE
    inference_path = SERVER

    def __init__(
        self,
        *,
        exe: str | None = None,
        argv_prefix: list[str] | None = None,
        host: str = "127.0.0.1",
        cpu_moe: bool = True,
        spec_type: str = "ngram-mod",
        chat_template_kwargs: dict | None = None,
        cuda_available: bool = True,
    ) -> None:
        self._exe = exe
        self._argv_prefix = list(argv_prefix or [])
        self._host = host
        # 默认值取实测最优组合（35B-A3B：-ngl 99 --cpu-moe --spec-type ngram-mod，
        # 15.94 tok/s、显存 2863 MiB；thinking 默认关闭以免 content 为空）。
        self._cpu_moe = bool(cpu_moe)
        self._spec_type = str(spec_type or "")
        self._chat_template_kwargs = (
            {"enable_thinking": False}
            if chat_template_kwargs is None
            else dict(chat_template_kwargs)
        )
        self._cuda_available = bool(cuda_available)
        self.server: Any = None          # LlamaServerProcess（启动后才有）
        self._client: Any = None         # IslandEngine
        self._model_path = ""

    # ------------------------------------------------------------- loading
    def load_model(
        self,
        model_path: str = "",
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        cpu_moe: bool | None = None,
        spec_type: str | None = None,
        mmproj_path: str = "",
        startup_timeout: float = 180.0,
        port: int = 0,
        log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """起 server 子进程，并把既有 HTTP 客户端指向它。"""
        from llama_server_process import LlamaServerProcess, ServerSpec

        exe = self._exe or default_server_exe()
        if not exe:
            raise RuntimeError(
                "找不到 llama-server 可执行文件：请设置 QLH_LLAMA_SERVER_EXE，"
                "或把带 CUDA 的构建放到 runtime/llama-cpp/<build>/ 下",
            )
        if not model_path:
            raise RuntimeError("LlamaServerEngine 需要显式 model_path（整模 GGUF）")

        spec = ServerSpec(
            model_path=str(model_path),
            exe=str(exe),
            argv_prefix=self._argv_prefix,
            n_ctx=int(n_ctx),
            n_gpu_layers=int(n_gpu_layers),
            cpu_moe=self._cpu_moe if cpu_moe is None else bool(cpu_moe),
            spec_type=self._spec_type if spec_type is None else str(spec_type),
            chat_template_kwargs=self._chat_template_kwargs,
            mmproj_path=str(mmproj_path or ""),
        )
        self.server = LlamaServerProcess(
            spec, host=self._host, port=int(port), log_path=log_path,
            startup_timeout=float(startup_timeout),
        )
        self.server.start()

        from island_engine import IslandEngine

        client = IslandEngine()
        client.load_model(base_url=self.server.base_url)
        self._client = client
        self._model_path = str(model_path)
        logger.info(
            "本机 llama-server 引擎就绪：%s（cpu_moe=%s, spec_type=%s）",
            self.server.base_url, spec.cpu_moe, spec.spec_type or "无",
        )

    # ------------------------------------------------------------------ state
    @property
    def is_loaded(self) -> bool:
        client = self._client
        if client is None:
            return False
        flag = getattr(client, "is_loaded", None)
        try:
            return bool(flag() if callable(flag) else flag)
        except Exception:  # noqa: BLE001 - 客户端探测异常按未加载处理
            return False

    @property
    def model_path(self) -> str:
        return self._model_path

    def unload(self) -> None:
        """先撤客户端，再回收子进程；可重复调用。"""
        client, self._client = self._client, None
        if client is not None:
            try:
                client.unload()
            except Exception:  # noqa: BLE001 - 卸载失败不阻断进程回收
                logger.warning("llama-server 客户端卸载失败", exc_info=True)
        server, self.server = self.server, None
        if server is not None:
            server.stop()

    # --------------------------------------------------------------- 委托面
    def chat(self, *args: Any, **kwargs: Any) -> dict:
        if self._client is None:
            raise RuntimeError("llama-server 未加载")
        return self._client.chat(*args, **kwargs)

    def chat_stream(self, *args: Any, **kwargs: Any) -> Iterator[dict]:
        if self._client is None:
            raise RuntimeError("llama-server 未加载")
        yield from self._client.chat_stream(*args, **kwargs)

    def get_model_info(self) -> dict:
        if self._client is None:
            return {"model_path": self._model_path, "engine": FAMILY_ENGINE_TYPE}
        return self._client.get_model_info()

    def get_memory_usage(self) -> dict:
        if self._client is None:
            return {}
        return self._client.get_memory_usage()

    def reset_kv_cache(self) -> None:
        if self._client is not None:
            self._client.reset_kv_cache()

    def tokenize(self, text: str) -> list[int]:
        if self._client is None:
            raise RuntimeError("llama-server 未加载")
        return self._client.tokenize(text)

    def detokenize(self, tokens: list[int]) -> str:
        if self._client is None:
            raise RuntimeError("llama-server 未加载")
        return self._client.detokenize(tokens)

    # ------------------------------------------------------------ capability
    def capability(self) -> dict:
        """能力声明，键与 `inference_path_selector.InferencePathDecision` 对齐。"""
        return {
            "inference_path": SERVER,
            "cuda_available": bool(self._cuda_available),
            "serves_layer_segments": False,
            "reason": "本机 llama-server 子进程承担整模推理；层段接力必须走绑定路径",
        }


__all__ = ["FAMILY_ENGINE_TYPE", "LlamaServerEngine", "default_server_exe"]
