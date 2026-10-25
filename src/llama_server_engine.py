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

    _environment = None
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
        self._inference_switches = None

    # ------------------------------------------------------------- loading
    def load_model(
        self,
        model_path: str = "",
        *,
        n_ctx: int | None = None,
        context_limit: int | None = None,
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

        from inference_switches import resolve_for_load, server_argv_args
        explicit_context = n_ctx is not None
        context_observation = {"reason": "explicit_context", "n_ctx": n_ctx}
        if n_ctx is None:
            from llama_engine import _default_n_ctx
            from inference_switches import resolve_from_device_profile, preference_environment
            preferences = resolve_from_device_profile(
                env=preference_environment(self._environment, set(kwargs)), execution_path="llama_server")
            context_observation = {}
            n_ctx = _default_n_ctx(model_path, ceiling=context_limit,
                execution_device=f"cuda:{int(kwargs.get('main_gpu', 0))}" if n_gpu_layers else "cpu",
                cache_type_k=kwargs.get("type_k", preferences.kv_cache_type_k),
                cache_type_v=kwargs.get("type_v", preferences.kv_cache_type_v),
                n_seq_max=kwargs.get("n_seq_max", 1), observation=context_observation)
        cpu_moe = self._cpu_moe if cpu_moe is None else bool(cpu_moe)
        spec_type = self._spec_type if spec_type is None else str(spec_type)
        baseline = dict(n_ctx=int(n_ctx), n_gpu_layers=int(n_gpu_layers), n_threads=None,
                        cpu_moe=cpu_moe, spec_type=spec_type,
                        chat_template_kwargs=self._chat_template_kwargs,
                        mmproj_path=str(mmproj_path or ""), argv_prefix=self._argv_prefix)
        baseline.update(kwargs)
        applied_keys = {"n_threads", "flash_attn", "type_k", "type_v", "n_threads_batch",
                        "n_batch", "n_ubatch", "n_seq_max", "main_gpu"}
        calibration_path = "llama_server" if not (set(kwargs) - applied_keys) else "llama_server_special"
        switches = resolve_for_load(model_path, baseline, execution_path=calibration_path,
                                    explicit_keys=set(kwargs), env=self._environment, engine_exe=exe)
        for key in ("n_threads", "flash_attn"):
            if key in kwargs:
                value = kwargs[key]
                if (key == "n_threads" and (type(value) is not int or value <= 0)
                        or key == "flash_attn" and type(value) is not bool):
                    raise ValueError("inference_server_option_invalid")
                setattr(switches, key, value)
                switches.sources[key] = "call_argument"
        for key, field in (("type_k", "kv_cache_type_k"), ("type_v", "kv_cache_type_v")):
            if key in kwargs:
                if kwargs[key] not in ("f16", "q8_0", "q4_0"):
                    raise ValueError("inference_kv_type_unsupported")
                setattr(switches, field, kwargs[key])
                switches.sources[field] = "call_argument"
        if any(kind in ("q8_0", "q4_0") for kind in (
                switches.kv_cache_type_k, switches.kv_cache_type_v)) and not switches.flash_attn:
            if kwargs.get("flash_attn") is False or switches.overrides.get("flash_attn") == "off":
                raise ValueError("inference_quantized_kv_requires_flash_attn")
            switches.flash_attn = True
            switches.sources["flash_attn"] = "kv_dependency"
        extra_args = []
        for name, flag in (("n_threads_batch", "--threads-batch"), ("n_batch", "--batch-size"),
                           ("n_ubatch", "--ubatch-size"), ("n_seq_max", "--parallel"),
                           ("main_gpu", "--main-gpu")):
            if name in kwargs:
                value = kwargs[name]
                if type(value) is not int or value < (0 if name == "main_gpu" else 1):
                    raise ValueError("inference_server_option_invalid")
                extra_args += [flag, str(value)]
        spec = ServerSpec(
            model_path=str(model_path), exe=str(exe), argv_prefix=self._argv_prefix,
            n_ctx=int(n_ctx), n_gpu_layers=int(n_gpu_layers), cpu_moe=cpu_moe,
            spec_type=spec_type, chat_template_kwargs=self._chat_template_kwargs,
            mmproj_path=str(mmproj_path or ""), inference_args=server_argv_args(switches),
            extra_args=extra_args,
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
        self._inference_switches = {
            "schema_version": 2, "scope": "loaded_model", "execution_path": "llama_server",
            "requested": switches.to_dict(),
            "effective": {"n_ctx": spec.n_ctx, "n_gpu_layers": spec.n_gpu_layers,
                          "n_threads": switches.n_threads or "server_auto",
                          "flash_attn": switches.flash_attn if "--flash-attn" in spec.inference_args else "server_auto",
                          "type_k": switches.kv_cache_type_k or "f16",
                          "type_v": switches.kv_cache_type_v or "f16",
                          "cpu_moe": spec.cpu_moe},
            "sources": dict(switches.sources), "argv": list(spec.inference_args) + list(extra_args),
            "observation": "loader_parameters", "context_budget": context_observation,
        }
        state = self._inference_switches
        state["effective"].update({key: kwargs[key] for key in (
            "n_threads_batch", "n_batch", "n_ubatch", "n_seq_max", "main_gpu") if key in kwargs})
        cache_sources = {"type_k": "kv_cache_type_k", "type_v": "kv_cache_type_v"}
        for key in state["effective"]:
            source_key = cache_sources.get(key, key)
            state["sources"][key] = ("call_argument" if key in kwargs else
                switches.sources.get(source_key, "server_baseline"))
        state["sources"]["n_ctx"] = "call_argument" if explicit_context else "context_budget"
        logger.info(
            "本机 llama-server 引擎就绪：%s（cpu_moe=%s, spec_type=%s）",
            self.server.base_url, spec.cpu_moe, spec.spec_type or "无",
        )
        logger.info("推理开关: %s", switches.describe())

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
        self._inference_switches = None

    def close(self) -> None:
        """与 `LlamaCppEngine.close()` 对齐的别名（ModelManager 会调用它）。"""
        self.unload()

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
        info = (dict(self._client.get_model_info()) if self._client is not None
                else {"model_path": self._model_path, "engine": FAMILY_ENGINE_TYPE})
        info["inference_switches"] = self._inference_switches
        return info

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
    def supports(self, capability: str) -> bool:
        """能力探测 —— `runtime_supports()` **优先**走这里，而不是 backend 白名单。

        ⚠️ 本引擎对外报 `engine_type="llama_cpp"`（家族一致），但 **server 子进程
        不承担层段接力**：层段/keep-head 前向依赖低层 C API 与本仓 llama.cpp 补丁。
        若不做这个复写，`runtime_supports(engine, FORWARD_LAYERS)` 会按 llama_cpp
        白名单放行（实测该家族 `FORWARD_LAYERS` 为 True），调度层于是把本引擎当成
        可承接层段的 runtime —— 票 15a 明令禁止这种"能起 server 就算能接力"的推断。
        """
        from koakuma_engine import (
            Capability,
            backend_capabilities,
            backend_id_for,
        )

        if capability == Capability.FORWARD_LAYERS:
            return False
        return backend_capabilities(backend_id_for(self)).supports(capability)

    def capability(self) -> dict:
        """能力声明，键与 `inference_path_selector.InferencePathDecision` 对齐。"""
        return {
            "inference_path": SERVER,
            "cuda_available": bool(self._cuda_available),
            "serves_layer_segments": False,
            "reason": "本机 llama-server 子进程承担整模推理；层段接力必须走绑定路径",
        }


__all__ = ["FAMILY_ENGINE_TYPE", "LlamaServerEngine", "default_server_exe"]
