"""
llama.cpp 推理引擎 — CPU / 集显设备的轻量级推理后端
====================================================
职责:
1. 加载 GGUF 量化模型（Q4_K_M / Q5_K_M / Q8_0 等）
2. ChatML 格式对话补全（兼容 Qwen 家族）
3. 流式输出 + KV 缓存管理
4. 与 PyTorch ModelManager 接口对齐，支持无缝切换

依赖: llama-cpp-python (pip install llama-cpp-python)

Qwen 家族的 ChatML 格式:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    你好<|im_end|>
    <|im_start|>assistant
    你好！有什么可以帮助你的？<|im_end|>

llama-cpp-python 通过 GGUF 元数据中的 tokenizer.chat_template
自动识别 Qwen 的对话格式，通常无需手动指定 chat_format。
若自动检测失败，可手动设置 chat_format="chatml"。

设计原则:
  - CUDA 设备 → PyTorch + bitsandbytes（保留不变）
  - CPU / 集显设备 → llama.cpp + GGUF（本模块）
  - 接口与 ModelManager 对齐，上游调用者无需修改
"""

from __future__ import annotations

import logging
import hashlib
import json

# ★ 2026-10-08（#53）：keep-head shim 的定位与 worker 侧共用同一事实源。
from keep_head_shim import missing_shim_hint, resolve_keep_head_shim
import math
import os
import re
import sys
import time
from ctypes import byref
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, List

# ⚠️ **必须平铺导入**（与 `keep_head_shim` 同规）：`relay_precision` 带**进程级状态**
#    （两侧档位登记 + 判据缓存）。若这里写 `from src.relay_precision import ...`，
#    而别处用 `from relay_precision import ...`，Python 会把它们当成**两个模块**，
#    状态各存一份 ⇒ 登记互不可见、检查永不触发（实测踩过，静默失效）。
from relay_precision import (
    RelayPrecisionMismatch as _RelayPrecisionMismatch,
    check_relay_precision as _check_relay_precision,
    note_downstream_precision as _note_downstream_precision,
)


def _note_relay_downstream_precision(engine) -> None:
    """登记**下游裁层工件**的量化档并检查两侧精度对齐（2026-10-08）。

    档位取 GGUF 的 `general.file_type`（**权威**，实测 `'1'`=f16、`'15'`=Q4_K_M），
    退化时才看路径标记/`quant_type`。

    ★ 只**记录**：默认 WARN 一次（按档位对去重），`QLH_RELAY_PRECISION_STRICT=1` 才 fail-loud。
    **不改变任何路由或准入语义**；判据来源与数值见 `src/relay_precision.py`。
    ⚠️ 除 `RelayPrecisionMismatch`（严格模式的**故意** fail-loud）外，一切异常都吞掉 ——
    精度诊断绝不能影响推理本身。
    """
    try:
        metadata = {}
        model = getattr(engine, "_model", None)
        if model is not None:
            metadata = getattr(model, "metadata", None) or {}
        _note_downstream_precision(
            metadata=metadata,
            model_path=getattr(engine, "_model_path", None),
            quant_type=getattr(engine, "_quant_type", None),
        )
        _check_relay_precision(logger=logging.getLogger(__name__))
    except _RelayPrecisionMismatch:
        raise
    except Exception:  # noqa: BLE001 - 诊断路径绝不影响推理
        pass

logger = logging.getLogger(__name__)

# 默认推荐量化类型 → GGUF 文件名
QUANT_FILES = {
    "Q4_K_M": "Qwen-1_8B-Chat-Q4_K_M.gguf",    # 推荐：速度/质量 最佳平衡 (~1.16 GB)
    "Q4_K_S": "Qwen-1_8B-Chat-Q4_K_S.gguf",     # 稍小 (~1.04 GB)
    "Q5_K_M": "Qwen-1_8B-Chat-Q5_K_M.gguf",     # 更高质量 (~1.31 GB)
    "Q8_0":   "Qwen-1_8B-Chat-Q8_0.gguf",        # 近无损 (~1.82 GB)
    "Q3_K_M": "Qwen-1_8B-Chat-Q3_K_M.gguf",      # 更小 (~0.94 GB)
}

CHATML_STOP_SEQUENCES = [
    "<|im_end|>",
    "<|im_start|>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|endoftext|>",
    "</s>",
]


def _merge_stop_sequences(stop: List[str] = None) -> List[str]:
    if isinstance(stop, str):
        stop = [stop]
    merged: List[str] = []
    for value in list(stop or []) + CHATML_STOP_SEQUENCES:
        if not isinstance(value, str):
            raise ValueError("stop sequences must be strings")
        if value and value not in merged:
            merged.append(value)
    return merged


def _apply_ngram_draft_model(load_kwargs: Dict[str, Any]) -> None:
    """按配置给绑定路径挂上 n-gram 自投机（就地修改 `load_kwargs`）。

    为什么选 n-gram：它只在 token 历史里做匹配，**不需要 draft 模型**、不占额外
    带宽与显存 —— 正合「边缘设备不与主模型争带宽」的约束。实测（2026-10-13，
    Qwen3.6-35B-A3B-Q4，CPU 绑定）：重复文本 1.30～1.40×，自由文本不劣化。

    契约：
    * 未启用或调用方已显式给出 `draft_model` ⇒ 原样不动；
    * 绑定缺 `LlamaPromptLookupDecoding`（旧版）⇒ 记 warning 后降级，不抛异常；
    * ⚠️ 一旦注入，`llama_cpp/llama.py` 会把 `logits_all` 强制为 True（开销上升）。
    """
    import config as _cfg

    if not _cfg.NGRAM_SPEC_ENABLED or "draft_model" in load_kwargs:
        return
    try:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
    except Exception:  # pragma: no cover - 旧绑定没有该类
        logger.warning(
            "n-gram 自投机不可用：当前 llama-cpp-python 未提供 "
            "LlamaPromptLookupDecoding；已按未启用处理",
        )
        return
    load_kwargs["draft_model"] = LlamaPromptLookupDecoding(
        max_ngram_size=int(_cfg.NGRAM_SPEC_MAX_NGRAM),
        num_pred_tokens=int(_cfg.NGRAM_SPEC_NUM_PRED),
    )
    logger.info(
        "n-gram 自投机已启用（max_ngram=%s, num_pred=%s）——"
        "注意绑定会强制 logits_all=True，仅在文本重复场景有收益",
        _cfg.NGRAM_SPEC_MAX_NGRAM,
        _cfg.NGRAM_SPEC_NUM_PRED,
    )


def _new_llama_with_seq_max(load_kwargs: Dict[str, Any], n_seq_max: Optional[int] = None):
    """构造 `llama_cpp.Llama`，可选地把 context 的 `n_seq_max` 抬到指定值。

    为什么需要：`llama-cpp-python` 的 `Llama.__init__` **只在 `embedding=True` 时**才设置
    `context_params.n_seq_max`（源码 :341-346）⇒ 非 embedding 时它保持 llama.cpp 的默认值 1 ⇒
    **任何 `seq_id > 0` 的 `llama_batch` 都会 `llama_decode rc=-1`**，多序列（批量交叠）无法使用。

    绕法：`Llama.__init__:253` 经**模块级** `llama_cpp.llama_cpp.llama_context_default_params()` 取默认
    params ⇒ 在建 context **之前**临时替换该函数即可。这里用 `try/finally` 严格恢复，
    并保持其余 `load_kwargs` 语义完全不变（`n_seq_max=None` 时行为与直接 `Llama(**load_kwargs)` 一致）。

    ⚠️ 若同时走 RPC 路径（`patched_llama_loader`），本函数只包住 `Llama(...)` 构造，
    与注入器的 patch 是**并列**关系，不改变其顺序语义。
    """
    from llama_cpp import Llama

    if n_seq_max is None:
        return Llama(**load_kwargs)

    import llama_cpp.llama_cpp as _lc

    original_default_params = _lc.llama_context_default_params

    def _patched_default_params():
        params = original_default_params()
        params.n_seq_max = max(int(n_seq_max), int(params.n_seq_max))
        return params

    _lc.llama_context_default_params = _patched_default_params
    try:
        return Llama(**load_kwargs)
    finally:
        _lc.llama_context_default_params = original_default_params


def _model_n_pos_per_embd(native_model, llama_cpp_module) -> int:
    """每个 token 的**位置分量数**（M-RoPE 模型为 4，其余为 1）。

    与 llama.cpp 的 `llama_hparams::n_pos_per_embd()` 对齐：`MROPE` / `IMROPE` ⇒ 4。
    为什么需要它：`embd` 注入通道下 llama.cpp 按 **planar** 布局取 `pos`
    （`llama-batch.cpp::llama_batch_allocr::ubatch_add`：
    `src_off = batch.token ? 0 : j*batch.n_tokens`）⇒ 调用方必须提供
    `n_tokens * n_pos_per_embd` 个位置；只给 n_tokens 个会让 M-RoPE 模型读到越界位置。
    见 `forward_layers_from_hidden()`（层段接力的下游入口）。
    """
    try:
        rope = int(llama_cpp_module.llama_model_rope_type(native_model))
    except AttributeError:   # 老绑定没有该 API ⇒ 只能按单分量处理（等价于修复前的行为）
        return 1
    mrope = int(getattr(llama_cpp_module, "LLAMA_ROPE_TYPE_MROPE", 8))
    imrope = int(getattr(llama_cpp_module, "LLAMA_ROPE_TYPE_IMROPE", 40))
    return 4 if rope in (mrope, imrope) else 1


class LlamaCppEngine:
    """
    llama.cpp 推理引擎 — 面向 CPU / 集显环境优化。

    使用方式:

        engine = LlamaCppEngine()
        engine.load_model("models/qwen-1_8b-chat-Q4_K_M.gguf")

        # 对话补全
        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好"},
        ]
        result = engine.chat(messages, max_tokens=512, temperature=0.7)
        print(result["content"])

        # 流式输出
        for chunk in engine.chat_stream(messages):
            print(chunk, end="", flush=True)
    """

    def __init__(self):
        self._model = None           # llama_cpp.Llama 实例
        self._model_path: str = ""
        self._mmproj_path: str = ""
        self._mtmd_context = None
        self._mtmd_module = None
        self._mtmd_capabilities: Dict[str, bool] = {"vision": False, "audio": False}
        self._quant_type: str = ""
        self._n_ctx: int = 4096      # 上下文窗口大小
        self._n_threads: int = 4     # CPU 线程数
        self._loaded: bool = False
        self._pipeline_descriptor: Dict[str, Any] | None = None
        self._pipeline_sha_cache: Dict[str, Any] = {}
        self._pipeline_distributed_only: bool = False
        # RAG embedding is opt-in. A normal text-generation load is never
        # silently reused as an embedding model.
        self._embedding_enabled: bool = False
        self._embedding_dimension: int | None = None
        self._embedding_model_id: str = ""
        self._chat_template: str = ""
        self._thinking_mode: str = "unknown"
        self._thinking_enabled: bool | None = None
        self._thinking_controlled: bool = False
        self._chat_template_kwargs: Dict[str, Any] = {}
        # RPC 接入（可选）：session 持有注入的 device 与（可选的）本地 worker，
        # 必须存活到 unload —— worker 一停，RPC device 上的计算即失败。
        self._rpc_session = None

    @property
    def engine_type(self) -> str:
        return "llama_cpp"

    # ================================================================
    # RPC device 接入（llama.cpp RPC backend）
    # ================================================================

    @staticmethod
    def _normalize_rpc_servers(rpc_servers) -> List[str]:
        """把 ``"host:port,host:port"`` 或列表归一为 endpoint 列表。"""
        if rpc_servers is None:
            return []
        if isinstance(rpc_servers, str):
            return [part.strip() for part in rpc_servers.split(",") if part.strip()]
        if isinstance(rpc_servers, (list, tuple)):
            return [str(part).strip() for part in rpc_servers if str(part).strip()]
        raise ValueError("rpc_servers must be a 'host:port' string or a list of endpoints")

    @staticmethod
    def _resolve_rpc_split(rpc_split, n_rpc_devices: int) -> List[float] | None:
        """把分片比例归一为 ``[本机 CPU, RPC0, RPC1, ...]`` 顺序的层占比列表。

        * ``None`` —— 不分片（全部层交给 RPC device，默认行为）；
        * 数字 ``x``（或等价的字符串 ``"0.5"``）—— 单 RPC device 时展开为 ``[1-x, x]``；
        * 字符串 ``"0.5,0.5"`` / 列表 —— 长度必须等于 ``1 + n_rpc_devices``。
        """
        if rpc_split is None:
            return None
        total = n_rpc_devices + 1
        if isinstance(rpc_split, str):
            parts = [part.strip() for part in rpc_split.split(",") if part.strip()]
            if not parts:
                raise ValueError("rpc_split 字符串为空")
            if len(parts) == 1:
                rpc_split = float(parts[0])      # "0.5" 等价于数字 0.5
            else:
                rpc_split = [float(part) for part in parts]
        if isinstance(rpc_split, (list, tuple)):
            values = [float(v) for v in rpc_split]
        elif isinstance(rpc_split, (int, float)) and not isinstance(rpc_split, bool):
            if n_rpc_devices != 1:
                raise ValueError("单个比例只适用于恰好 1 个 RPC device")
            values = [1.0 - float(rpc_split), float(rpc_split)]
        else:
            raise ValueError("rpc_split must be a number, a 'a,b' string, or a list")
        if len(values) != total:
            raise ValueError(
                f"rpc_split 需要 {total} 个比例（1 个本机 CPU + {n_rpc_devices} 个 RPC），实得 {len(values)}"
            )
        if any(v < 0 for v in values) or sum(values) <= 0:
            raise ValueError("rpc_split 比例必须非负且总和 > 0")
        return values

    def _open_rpc_session(self, *, endpoints, worker_exe, worker_port, worker_threads, worker_log):
        """注册 RPC device（必要时先起本机 worker）；失败即抛，不降级为纯 CPU。"""
        try:
            from llama_rpc_device import RpcCleanupError, RpcSession
        except ImportError:  # 以包形式导入本模块时（src.llama_engine）
            from .llama_rpc_device import RpcCleanupError, RpcSession

        try:
            session = RpcSession.open(
                endpoints=endpoints,
                autostart_worker=worker_exe,
                worker_port=worker_port,
                worker_threads=worker_threads,
                worker_log=worker_log,
            )
        except RpcCleanupError as exc:
            self._rpc_session = exc.cleanup_owner
            raise
        injected = [f"{dev.name}@{dev.endpoint}" for dev in session.injector.devices]
        logger.info(f"RPC device 已注入: {injected}")
        return session

    def _close_rpc_session(self) -> None:
        session = self._rpc_session
        if session is not None:
            session.close()
            self._rpc_session = None

    # ================================================================
    # 流水线模型描述器
    # ================================================================

    def _pipeline_file_sha256(self, path: str, *, use_cache: bool = True) -> str:
        """整文件 sha256，按 `(path, mtime, size)` 记忆 —— 大 GGUF 只读一次。

        与 PyTorch 侧 `model_sync.compute_model_sha256`（目录内 artifact 的稳定序
        联合哈希）**语义不同**：那里是「一组文件」，这里是「单个 GGUF」。两者都只用于
        「各节点是否握着同一份权重」的自证，由**主节点的描述器单独定口径**，因此
        混合部署时以主节点为准，不需要跨引擎可比。
        """
        try:
            stat = os.stat(path)
        except OSError:
            return ""
        key = (
            path,
            int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            int(stat.st_size),
        )
        cache = getattr(self, "_pipeline_sha_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._pipeline_sha_cache = cache
        if use_cache and cache.get("key") == key:
            return str(cache.get("value", ""))
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return ""
        value = digest.hexdigest()
        cache["key"] = key
        cache["value"] = value
        return value

    def get_pipeline_descriptor(self) -> dict:
        """流水线描述器：**零依赖**读 GGUF 头，不加载模型、不需要 torch。

        与 `ModelManager.get_pipeline_descriptor()`（PyTorch 侧）同一契约。主节点的
        层配置推送只认这个描述器（`scheduler._get_active_pipeline_model_info` ⇒
        `scheduler_pipeline._get_master_model_sha256`），所以 llama.cpp 侧必须能给出
        同样的 `model_type` / `total_layers` / `model_sha256`。

        为什么需要它：上述两条路径此前硬要求主节点是 PyTorch 模型 —— 那是
        「llama.cpp 不支持层拆分」时代的遗留判据，已被 `koakuma_engine` 的能力表推翻
        （`BackendId.LLAMA_CPP` 自 2026-09-19 起声明 `FORWARD_LAYERS`，`llama_engine`
        也有 `forward_layers_to_hidden` / `forward_layers_from_hidden`）。去 torch 化的
        边缘设备只得一份裁层 GGUF，正是靠这个描述器才当得上流水线主节点。

        `model_type` 取自 GGUF 的 `general.architecture`，先在共享描述器边界归一化
        llama.cpp 架构别名，再与 `PIPELINE_RUNTIME_MODEL_TYPES` 同源判定 —— 判据只有
        一处，不在这里另立白名单。
        """
        cached = getattr(self, "_pipeline_descriptor", None)
        if isinstance(cached, dict) and cached:
            return dict(cached)
        path = str(getattr(self, "_model_path", "") or "")
        if not path or not os.path.isfile(path):
            return {}
        try:
            from relay_segment_info import (
                read_artifact_manifest, read_gguf_layer_info,
                read_gguf_tokenizer_sha256,
            )
        except ImportError:  # 以包形式导入本模块时（src.llama_engine）
            try:
                from .relay_segment_info import (
                    read_artifact_manifest, read_gguf_layer_info,
                    read_gguf_tokenizer_sha256,
                )
            except ImportError:
                return {}
        info = read_gguf_layer_info(path)
        if not info:
            return {}
        from pipeline_model_descriptor import (
            PIPELINE_RUNTIME_MODEL_TYPES,
            canonical_pipeline_model_type,
            tokenizer_sha256_from_asset,
        )

        model_type = canonical_pipeline_model_type(info.get("architecture", ""))
        if model_type not in PIPELINE_RUNTIME_MODEL_TYPES:
            return {}
        local_total_layers = int(info.get("n_layer", 0) or 0)
        if local_total_layers <= 0:
            return {}
        # ★ 2026-10-11（三机实测 / total_layers 缺口，用户裁定「不能加载整模，这是接力的初衷」）：
        #   裁层工件的 GGUF 头只含**本段**层（head=16），而容量求解器的区间判据要的是
        #   **源模型**总层数（`pipeline_capacity.py:490` 取 `total_layers = len(layer_bytes)`，
        #   `:74` 用它校验各节点 `layer_ranges` 上界）。此前 `total_layers` 直接取工件头部的
        #   `n_layer` ⇒ 视图被钉成 16 ⇒ 远端 `[16,24)` / `[24,32)` 恒被拒
        #   ⇒ 只能靠"加载整模"绕过，而那**违背接力初衷** ⇒ 这里按工件 manifest 的
        #   `n_layer`/`source_layer_range` 把视图还原成源模型坐标系。
        artifact_info = {}
        for manifest_path in (
            Path(path).with_suffix(".manifest.json"),
            Path(str(path) + ".manifest.json"),
        ):
            manifest_info = read_artifact_manifest(manifest_path)
            if manifest_info:
                artifact_info = manifest_info
                break
        declared_range = artifact_info.get("source_layer_range")
        segment_start = 0
        if isinstance(declared_range, (list, tuple)) and len(declared_range) == 2:
            try:
                segment_start = int(declared_range[0])
            except (TypeError, ValueError):
                segment_start = 0
        try:
            declared_total = int(artifact_info.get("n_layer", 0) or 0)
        except (TypeError, ValueError):
            declared_total = 0
        total_layers = declared_total if declared_total > 0 else local_total_layers
        if total_layers < local_total_layers + segment_start:
            total_layers = local_total_layers + segment_start
        # `pipeline_capacity` 的容量账要**逐层字节**与**分量字节**，两者都必须 exact
        # （它按此判定哪个 worker 装得下哪几层）。GGUF 里只有张量表，所以现场按
        # ggml 类型算一遍 —— 这也是「不用 offset 差值」的原因，差值含 32 字节对齐填充。
        from relay_segment_info import read_gguf_tensor_bytes

        tensor_bytes = read_gguf_tensor_bytes(path)
        if not tensor_bytes:
            return {}
        layer_weight_bytes = [0] * total_layers
        component_weight_bytes = {
            "embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0,
        }
        block_prefix = re.compile(r"^blk\.(\d+)\.")
        for tensor_name, size in tensor_bytes.items():
            block = block_prefix.match(tensor_name)
            if block:
                # 工件内的 blk 索引是**本段局部**的（0..local_total_layers-1），
                # 要按 manifest 的 source_layer_range 平移到源模型坐标系。
                index = segment_start + int(block.group(1))
                if index < total_layers:
                    layer_weight_bytes[index] += size
                    continue
            if tensor_name.startswith("token_embd."):
                component_weight_bytes["embedding"] += size
            elif tensor_name.startswith("output_norm."):
                component_weight_bytes["final_norm"] += size
            elif tensor_name.startswith("output."):
                component_weight_bytes["lm_head"] += size
            else:
                component_weight_bytes["other"] += size
        # 只要求**本段**覆盖到字节：其它层属于别的段（远端执行），不是"缺失"。
        segment_span = (segment_start, segment_start + local_total_layers)
        if any(
            size <= 0
            for index, size in enumerate(layer_weight_bytes)
            if segment_span[0] <= index < segment_span[1]
        ):
            return {}
        try:
            weight_bytes = int(os.path.getsize(path))
        except OSError:
            weight_bytes = 0
        model_sha256 = self._pipeline_file_sha256(path)
        artifact_manifest_info = None
        tokenizer_sha256 = read_gguf_tokenizer_sha256(path) or ""
        for manifest_path in (
            Path(path).with_suffix(".manifest.json"),
            Path(str(path) + ".manifest.json"),
        ):
            manifest_info = read_artifact_manifest(manifest_path)
            if manifest_info:
                artifact_manifest_info = manifest_info
                declared_tokenizer = str(manifest_info.get("tokenizer_sha256", "") or "")
                if declared_tokenizer and declared_tokenizer != tokenizer_sha256:
                    return {}
                break
        if not tokenizer_sha256:
            tokenizer_sha256 = tokenizer_sha256_from_asset(Path(path).parent)
        if not tokenizer_sha256:
            try:
                from model_config import get_builtin_model, resolve_model_path

                registered = get_builtin_model(
                    str(getattr(self, "active_model_id", "") or "")
                )
                registered_path = (
                    resolve_model_path(str(getattr(registered, "model_path", "") or ""))
                    if registered is not None else ""
                )
                if registered_path:
                    tokenizer_sha256 = tokenizer_sha256_from_asset(
                        Path(registered_path)
                    )
            except Exception:
                logger.debug(
                    "GGUF pipeline descriptor could not resolve tokenizer asset",
                    exc_info=True,
                )
        descriptor = {
            "model_id": (
                str(getattr(self, "active_model_id", "") or "")
                or os.path.splitext(os.path.basename(path))[0]
            ),
            # A precut artifact may be registered under a deployment alias
            # (for example ``q35-artifacts``), while its manifest names the
            # logical source model shared by every relay segment. Keep both
            # identities: ``model_id`` selects the local registry entry;
            # ``source_model_id`` is the cross-node preflight contract.
            "source_model_id": str(
                (artifact_manifest_info or {}).get("source_model_id", "")
                or ""
            ),
            "model_path": path,
            "model_sha256": model_sha256,
            "source_model_sha256": str(
                (artifact_manifest_info or {}).get("source_model_sha256", "")
                or model_sha256
            ),
            "model_type": model_type,
            "total_layers": total_layers,
            "hidden_size": int(info.get("n_embd", 0) or 0),
            "tokenizer_sha256": tokenizer_sha256,
            "layer_weight_bytes": layer_weight_bytes,
            "component_weight_bytes": component_weight_bytes,
            "quant_type": str(getattr(self, "_quant_type", "") or ""),
            "inspection_mode": "gguf_header",
            "weight_bytes": weight_bytes,
            "pipeline_runtime_supported": True,
        }
        self._pipeline_descriptor = dict(descriptor)
        return dict(descriptor)

    def prepare_pipeline_model(
        self,
        model_id: str,
        model_path: str,
        quant_type: str = None,
        layer_range: tuple[int, int] | None = None,
        model_sha256: str | None = None,
    ) -> dict:
        """Prepare GGUF pipeline metadata without constructing llama.cpp."""
        resolved_path = os.path.abspath(model_path or "")
        if not resolved_path or not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"GGUF 模型文件未找到: {resolved_path or '(未提供路径)'}")
        if layer_range is not None:
            # ★ 2026-10-11（ROG+Y700+本机三机验收实测）：llama.cpp 的段由**裁层工件**
            #   表达（`model_load_resolver.ENGINE_CAPABILITIES[LLAMA_CPP].layer_range_mode
            #   == "precut_artifact"`）。但 Route A 的本地 commit 仍会带上本机段区间 ⇒
            #   此前一律拒绝，表现为 `pipeline_local_commit_failed`，GGUF 主节点**无法**
            #   参与层流水线。现在只在**与所加载工件 manifest 声明的区间一致**时接受
            #   （即"加载的正是这一段工件"），其余情况保持 fail-closed（整模仍拒绝）。
            try:
                from relay_segment_info import read_artifact_manifest
            except ImportError:  # 以包形式导入本模块时（src.llama_engine）
                from .relay_segment_info import read_artifact_manifest

            # 区间声明来源优先级：**先 `QLH_LAYER_GGUF`**（主节点以"整模 + 本地裁层
            # 工件"表达自己的段，从而对外仍提供**整模身份** `model_sha256`，
            # Android 侧据此匹配工件的 `source_model_sha256`），再退回所加载文件自身
            # 的 manifest（"直接加载该段工件"的用法）。
            declared: list[int] | None = None
            _bases = []
            from model_supply_worker import selected_gguf_path
            _env_gguf = selected_gguf_path() or os.environ.get("QLH_LAYER_GGUF", "").strip()
            if _env_gguf:
                _bases.append(Path(_env_gguf))
            _bases.append(Path(resolved_path))
            for _base in _bases:
                for manifest_path in (
                    _base.with_suffix(".manifest.json"),
                    Path(str(_base) + ".manifest.json"),
                ):
                    _info = read_artifact_manifest(manifest_path) or {}
                    _span = _info.get("source_layer_range")
                    if isinstance(_span, (list, tuple)) and len(_span) == 2:
                        declared = [int(_span[0]), int(_span[1])]
                        break
                if declared is not None:
                    break
            if declared != [int(layer_range[0]), int(layer_range[1])]:
                raise ValueError(
                    "GGUF 整模恢复不接受 layer_range；裁层范围由工件 manifest 声明"
                )
        if (self._model is not None or self._mtmd_context is not None
                or self._rpc_session is not None or getattr(self, "_keep_head", None) is not None):
            self.unload()
        self._model_path = resolved_path
        self.active_model_id = str(model_id or "")
        self._quant_type = str(quant_type or "GGUF")
        self._pipeline_descriptor = None
        self._pipeline_sha_cache = {}
        descriptor = self.get_pipeline_descriptor()
        if not descriptor.get("pipeline_runtime_supported", False):
            raise RuntimeError("GGUF 工件缺少可执行的流水线描述元数据")
        actual_sha256 = str(descriptor.get("model_sha256", "") or "").lower()
        expected_sha256 = str(model_sha256 or "").lower()
        if expected_sha256 and actual_sha256 != expected_sha256:
            self.unload()
            raise RuntimeError("GGUF 流水线模型摘要与恢复记录不一致")
        self._pipeline_descriptor = dict(descriptor)
        self._pipeline_distributed_only = True
        return dict(descriptor)

    @property
    def is_pipeline_prepared(self) -> bool:
        return bool(self._pipeline_distributed_only and self._pipeline_descriptor)

    def abort_pipeline_materialization(self) -> None:
        """Release an aborted GGUF segment but keep whole-model metadata."""
        descriptor = dict(self._pipeline_descriptor or {})
        if not self._pipeline_distributed_only or not descriptor:
            return
        full_model_path = str(descriptor.get("model_path", "") or "")
        model_id = str(descriptor.get("model_id", "") or "")
        quant_type = str(descriptor.get("quant_type", "") or self._quant_type)
        for key in (
            "assignment_layer_range", "partial_assignment", "loaded_artifact",
        ):
            descriptor.pop(key, None)
        self.unload()
        self._model_path = full_model_path
        self.active_model_id = model_id
        self._quant_type = quant_type
        self._pipeline_descriptor = descriptor
        self._pipeline_distributed_only = True

    def _find_layer_artifact(self, start: int, end: int) -> str:
        """按 `[start, end)` 在工件目录里找裁层 GGUF —— **靠 manifest 自证，不猜文件名**。

        工件由 `scripts/cut_layers.py` 产出，旁边必有 `<artifact>.gguf.manifest.json`，
        里面的 `source_layer_range`、`source_model_sha256`、`artifact_sha256`
        共同证明区间、源整模和段工件身份。命名并不统一
        （`cut-16` / `cut-16-20` / `head8` / `mid8-16` / `cut-k16` …），按文件名猜会
        在下一个命名上翻车，所以只认 manifest。

        搜索目录取 env `QLH_LAYER_ARTIFACT_DIR`；未设时用当前模型文件所在目录。
        """
        from pathlib import Path as _Path
        from relay_segment_info import read_artifact_manifest

        configured = os.environ.get("QLH_LAYER_ARTIFACT_DIR", "").strip()
        directory = (
            _Path(configured)
            if configured
            else _Path(str(getattr(self, "_model_path", "") or "")).parent
        )
        if not directory.is_dir():
            return ""
        descriptor = dict(getattr(self, "_pipeline_descriptor", None) or {})
        expected_source_sha256 = str(
            descriptor.get("source_model_sha256")
            or descriptor.get("model_sha256")
            or ""
        ).strip().lower()
        if not expected_source_sha256:
            source_path = str(
                descriptor.get("model_path", "")
                or getattr(self, "_model_path", "")
                or ""
            )
            expected_source_sha256 = self._pipeline_file_sha256(
                source_path,
                use_cache=False,
            ).lower()
        if not expected_source_sha256:
            raise RuntimeError("无法确定预切 GGUF 的源整模摘要，拒绝按层区间猜测工件")

        wanted = [int(start), int(end)]
        from model_supply_worker import selected_gguf_path
        supplied = selected_gguf_path()
        candidates = sorted(directory.glob("*.manifest.json"))
        if supplied:
            supplied_path = _Path(supplied)
            candidates = [supplied_path.with_suffix(".manifest.json")]
            directory = supplied_path.parent
        matches: list[str] = []
        for manifest in candidates:
            info = read_artifact_manifest(manifest)
            if not info:
                continue
            if list(info.get("source_layer_range") or []) != wanted:
                continue
            if str(info.get("source_model_sha256", "") or "").lower() != expected_source_sha256:
                continue
            expected_artifact_sha256 = str(
                info.get("artifact_sha256", "") or ""
            ).lower()
            if not expected_artifact_sha256:
                continue
            artifact = manifest.name[: -len(".manifest.json")]
            if not artifact.lower().endswith(".gguf"):
                artifact += ".gguf"
            candidate = directory / artifact
            if not candidate.is_file():
                continue
            if self._pipeline_file_sha256(
                str(candidate),
                use_cache=False,
            ).lower() != expected_artifact_sha256:
                continue
            matches.append(str(candidate))
        if len(matches) > 1:
            raise RuntimeError(
                f"层区间[{int(start)},{int(end)})存在多个同源且摘要有效的裁层工件，"
                "拒绝按文件名顺序静默选择"
            )
        if matches:
            return matches[0]
        return ""

    def load_layer_range(
        self,
        start_layer: int,
        end_layer: int,
        has_embedding: bool = False,
        has_lm_head: bool = False,
        **kwargs: Any,
    ) -> dict:
        """llama.cpp 的「加载层段」= **换一份裁层 GGUF**。

        PyTorch 侧按 key 物化层段（`LOAD_LAYER_RANGE` 语义）；llama.cpp 没有按层按需
        加载的能力，同一意图由工件表达 —— `koakuma_engine` 的能力表刻意**不**给
        `BackendId.LLAMA_CPP` 这个能力位，只给 `FORWARD_LAYERS`。

        所以本方法做的是：按 `[start_layer, end_layer)` 定位工件并加载。找不到工件即
        **失败**（`RuntimeError`），绝不静默退回整模 —— 那会让主节点跑着自己没被分配的
        层，下游段收到的 hidden 就从错误的层深取来，而链路看上去"成功"。
        """
        path = self._find_layer_artifact(int(start_layer), int(end_layer))
        if not path:
            raise RuntimeError(
                f"没有覆盖层区间[{int(start_layer)},{int(end_layer)})的裁层工件"
                f"（查了 QLH_LAYER_ARTIFACT_DIR 与模型目录）；"
                "llama.cpp 的层段由裁层 GGUF 表达，不是按 key 物化"
            )
        # 段视图 vs 整模视图：`pipeline_capacity` 与 worker 的层配置契约都按**整模**的
        # 总层数与摘要算账（PyTorch 侧 `inspect_pipeline_model` 的 `partial_assignment`
        # 就是这个语义）。换裁层 GGUF 之后 `_model_path` 指向段文件，若让描述器跟着变小，
        # 主节点会被判「模型已变化，请等待层配置重新同步」——实测卡在这里。
        # 所以：先留下整模描述器，加载段之后再装回去，只补层段标记。
        whole = dict(getattr(self, "_pipeline_descriptor", None) or {})
        was_pipeline_prepared = self.is_pipeline_prepared
        self.load_model(path)
        # A stage worker may be created directly and load its first artifact
        # without ever having cached a full-model descriptor.  The artifact
        # itself is still partial; leaving the descriptor empty would let
        # ModelHost.ensure_full_model() mistake head8/mid8 for a full model.
        whole.setdefault("model_path", path)
        whole["assignment_layer_range"] = [int(start_layer), int(end_layer)]
        whole["partial_assignment"] = True
        whole["loaded_artifact"] = path
        self._pipeline_descriptor = whole
        self._pipeline_distributed_only = was_pipeline_prepared
        logger.info(
            "llama.cpp 层段已加载: [%d,%d) embed=%s lm_head=%s -> %s",
            int(start_layer), int(end_layer), has_embedding, has_lm_head, path,
        )
        return {
            "success": True,
            "engine": "llama_cpp",
            "layer_range": f"{int(start_layer)}-{int(end_layer)}",
            "layer_start": int(start_layer),
            "layer_end": int(end_layer),
            "artifact": path,
        }

    # ================================================================
    # 模型加载
    # ================================================================

    def load_model(
        self,
        model_path: str = None,
        n_ctx: int = None,
        n_threads: int = None,
        chat_format: str = None,
        model_profile: dict = None,
        mmproj_path: str = None,
        mtmd_use_gpu: bool = False,
        embedding: bool = False,
        embedding_dimension: int | None = None,
        embedding_model_id: str | None = None,
        rpc_servers: str | List[str] | None = None,
        rpc_worker_exe: str | None = None,
        rpc_worker_port: int = 50163,
        rpc_worker_threads: int = 4,
        rpc_worker_log: str | None = None,
        rpc_split: float | str | List[float] | None = None,
        align_numerics: bool = False,
        n_seq_max: int | None = None,
        **kwargs,
    ) -> None:
        """
        加载 GGUF 量化模型。

        Args:
            model_path: GGUF 文件路径
            n_ctx: 上下文窗口大小（默认 4096，边缘设备建议 2048）
            n_threads: CPU 推理线程数（默认自动检测：物理核心数）
            chat_format: 对话格式（默认自动检测，Qwen 用 "chatml"）
                align_numerics: 跨路径数值对齐（关闭 CPU_REPACK，使 RPC 与本机 no-repack 逐比特一致）。
            n_seq_max: ★ 抬升 context 的并行序列上限（**多序列 / 批量交叠**所必需）。
                默认 None ⇒ 行为与旧版完全一致（llama.cpp 默认 1）。**为什么需要**：
                `llama-cpp-python` 的 `Llama.__init__` 只在 `embedding=True` 时设置它
                （源码 :341-346）⇒ 非 embedding 时任何 `seq_id > 0` 的批次都会
                `llama_decode rc=-1`。本实现经模块级 `llama_context_default_params()`
                在建 context 前注入，并在 `finally` 恢复（见 `_new_llama_with_seq_max`）。
        **kwargs: 透传给 `llama_cpp.Llama` 的额外参数。经实测（llama-cpp-python 0.3.35）
                可用且与本项目的容量/放置相关者：

                * `tensor_split` / `main_gpu` / `split_mode` —— 多设备切分；
                * `use_mmap` / `use_mlock` —— 页映射与常驻（**注意**：llama.cpp CLI 的
                  `--load-mode {mmap|none|dio|mlock|mmap+mlock}` 与 `--fit-target`
                  在 Python 绑定中**不存在**，无法透传；只有 `use_mmap` 这个布尔开关）；
                * `kv_overrides` —— **运行时覆盖 GGUF 元数据**（如需改 `*.block_count` /
                  `*.nextn_predict_layers` 等架构 KV，这是唯一通道）；
                * `numa` / `op_offload` / `flash_attn` / `swa_full` / `type_k` / `type_v`。

                **⚠️ 注意**：llama.cpp CLI 的 `--override-tensor`（按张量指定 buffer type）
                同样**未在 Python 绑定中暴露**，无法经此通道使用。

            rpc_servers: 已有的 ggml-rpc-server endpoint（`"host:port"` 或列表）。给了即走
                RPC device，**注入失败会抛错，不会静默退回纯 CPU**。
            rpc_worker_exe: 本机 `ggml-rpc-server` 路径；给了就由引擎启动/回收该 worker
                （单机实验用；远端 worker 场景只给 `rpc_servers`）。
            rpc_worker_port / rpc_worker_threads / rpc_worker_log: 本地 worker 的端口、
                CPU 线程数与日志路径。
            rpc_split: **分片（部分层驻留）**。`None`=全部层交给 RPC device（默认）；
                数字 `x`（单 RPC device）= 展开为 `[1-x, x]`；也可给 `"0.5,0.5"` 或列表，
                顺序为 `[本机 CPU, RPC0, RPC1, ...]`，长度须等于 `1 + RPC device 数`。
                例：`rpc_split=0.5` ⇒ 一半层留本机 CPU、一半放远端 RPC。
            align_numerics: **跨路径数值对齐**。本机 CPU 默认启用 CPU_REPACK（`--repack`），
                而 RPC 路径的权重落在远端 buffer 里**无法 repack**，于是两条路径的 logits 不同
                （实测 max|Δ|≈0.98、零均值、不随位置累积）⇒ 长序列会分叉。置 True 会把
                `use_extra_bufts=False` 注入 `llama_model_params`（llama-cpp-python 未暴露该
                开关，只能这样设），两条路径随即**逐比特一致**；代价实测为 0（见文档 §15）。
        """
        from config import MAX_SEQ_LEN

        if not isinstance(embedding, bool):
            raise ValueError("embedding must be a boolean")
        if embedding:
            if isinstance(embedding_dimension, bool) or not isinstance(embedding_dimension, int) or not 1 <= embedding_dimension <= 32_768:
                raise ValueError("embedding_dimension must be declared for an embedding model")
            if not isinstance(embedding_model_id, str) or not embedding_model_id.strip() or len(embedding_model_id) > 128:
                raise ValueError("embedding_model_id must be declared for an embedding model")
        elif embedding_dimension is not None or embedding_model_id is not None:
            raise ValueError("embedding identity requires embedding=True")

        if (self._model is not None or self._mtmd_context is not None
                or self._rpc_session is not None or getattr(self, "_keep_head", None) is not None):
            self.unload()
        self._pipeline_distributed_only = False
        self._model_path = model_path
        # 换模型即失效：描述器与文件摘要都跟着 `_model_path` 走。不重置会拿旧模型的
        # 层数/摘要去推层配置（`get_pipeline_descriptor` 缓存的就是它们）。
        self._pipeline_descriptor = None
        self._pipeline_sha_cache = {}
        # KV 是新 context 的，占用计数跟着归零（见 `_kv_used_cells`）。
        self._kv_used = 0
        model_profile = model_profile if isinstance(model_profile, dict) else {}
        self._chat_template = str(model_profile.get("template") or "")
        self._thinking_mode = str(model_profile.get("thinking") or "unknown")
        self._thinking_enabled = None
        self._thinking_controlled = False
        self._chat_template_kwargs = {}

        # 自动检测量化类型（从文件名提取）
        for quant_name, fname in QUANT_FILES.items():
            if fname in (model_path or ""):
                self._quant_type = quant_name
                break
        if not self._quant_type:
            self._quant_type = "GGUF"

        if n_ctx is None:
            n_ctx = MAX_SEQ_LEN if MAX_SEQ_LEN > 0 else 4096
        if n_threads is None:
            n_threads = self._auto_threads()

        self._n_ctx = n_ctx
        self._n_threads = n_threads

        logger.info(f"加载 GGUF 模型: {model_path}")
        logger.info(f"  量化类型: {self._quant_type}")
        logger.info(f"  上下文窗口: {n_ctx} tokens")
        logger.info(f"  CPU 线程数: {n_threads}")

        t0 = time.time()

        try:
            from llama_cpp import Llama

            load_kwargs: Dict[str, Any] = dict(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )
            if embedding:
                load_kwargs["embedding"] = True

            if chat_format:
                load_kwargs["chat_format"] = chat_format
            elif self._chat_template == "qwen_chat_v1":
                load_kwargs["chat_format"] = "chatml"
            elif model_path and "qwen-1_8b" in os.path.basename(model_path).lower():
                # Qwen-1.8B GGUF metadata is not always detected reliably.
                load_kwargs["chat_format"] = "chatml"

            load_kwargs.update(kwargs)
            # n-gram 自投机（绑定路径 / 无 CUDA 边缘）：默认关闭；
            # 调用方显式传了 draft_model 时不覆盖。
            _apply_ngram_draft_model(load_kwargs)

            rpc_endpoints = self._normalize_rpc_servers(rpc_servers)
            if rpc_endpoints or rpc_worker_exe:
                # RPC device 决定权重放在哪；显式 devices 时默认全部层交给它们（-1），
                # 调用方仍可经 n_gpu_layers 覆盖。
                self._rpc_session = self._open_rpc_session(
                    endpoints=rpc_endpoints,
                    worker_exe=rpc_worker_exe,
                    worker_port=rpc_worker_port,
                    worker_threads=rpc_worker_threads,
                    worker_log=rpc_worker_log,
                )
                injector = self._rpc_session.injector
                split = self._resolve_rpc_split(rpc_split, len(injector.devices))
                load_kwargs.setdefault("n_gpu_layers", -1)
                # 分片时把本机 CPU 也放进 devices，层按 tensor_split 分配
                array = injector.device_array(include_cpu=split is not None)
                # 跨路径数值对齐：本机 CPU 默认启用 CPU_REPACK，而 RPC 路径的权重在远端
                # buffer 里无法 repack → 两条路径浮点结果不同（§11/§13）。关掉 repack 后
                # 二者逐比特一致；llama-cpp-python 不暴露该开关，只能在注入点改 params。
                # （Llama.__init__ 既不接受 devices 也不接受 use_extra_bufts）
                with injector.patched_llama_loader(
                    device_array=array,
                    tensor_split=split,
                    use_extra_bufts=False if align_numerics else None,
                ):
                    self._model = _new_llama_with_seq_max(load_kwargs, n_seq_max)
                engine_label = "llama.cpp (RPC)"
                engine_detail = "devices=" + ",".join(
                    (["CPU"] if split is not None else [])
                    + [f"{dev.name}@{dev.endpoint}" for dev in injector.devices]
                ) + f" n_gpu_layers={load_kwargs.get('n_gpu_layers')}"
                if split is not None:
                    engine_detail += f" tensor_split={split}"
                if align_numerics:
                    engine_detail += " use_extra_bufts=False(repack off)"
                    logger.info("  数值对齐: use_extra_bufts=False —— 关闭 CPU_REPACK，与 RPC 路径逐比特一致")
                else:
                    logger.warning(
                        "  ⚠ 本机 CPU 默认启用 CPU_REPACK，而 RPC 路径无法 repack，两者浮点结果不同"
                        "（长序列可能分叉）；需与本机逐字一致请传 align_numerics=True"
                    )
            else:
                # 纯本机路径：跨路径对齐同样要求关闭 CPU_REPACK（RPC 侧天然关闭）
                if align_numerics:
                    from llama_rpc_device import patched_model_params
                    with patched_model_params(use_extra_bufts=False):
                        self._model = _new_llama_with_seq_max(load_kwargs, n_seq_max)
                    engine_detail = "use_extra_bufts=False(repack off)"
                    logger.info("  数值对齐: use_extra_bufts=False —— 关闭 CPU_REPACK")
                else:
                    self._model = _new_llama_with_seq_max(load_kwargs, n_seq_max)
                    engine_detail = ""
                engine_label = "llama.cpp (CPU)"

            self._loaded = True
            self._install_chat_template_controls()
            self._embedding_enabled = embedding
            self._embedding_dimension = embedding_dimension if embedding else None
            self._embedding_model_id = embedding_model_id.strip() if embedding else ""
            if mmproj_path:
                self.load_mmproj(mmproj_path, use_gpu=mtmd_use_gpu)

            load_time = time.time() - t0
            logger.info(f"GGUF 模型加载完成 ({load_time:.1f}s)")
            logger.info(f"  引擎: {engine_label}" + (f" {engine_detail}" if engine_detail else ""))

        except ImportError:
            raise ImportError(
                "llama-cpp-python 未安装。请执行:\n"
                "  pip install llama-cpp-python\n"
                "或从预编译 wheel 安装:\n"
                "  pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            )
        except Exception as e:
            logger.error(f"GGUF 模型加载失败: {e}")
            # Use the same ownership-preserving path as explicit unload:
            # tensors must close before their RPC device/process retires.
            self.unload()
            raise

    @staticmethod
    def _vram_free_bytes() -> Optional[int]:
        """查询当前 VRAM 空闲（nvidia-smi）；不可用时返回 None。"""
        import shutil
        import subprocess as _sp
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            out = _sp.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                return None
            return int(out.stdout.strip().splitlines()[0]) * 1024**2
        except Exception:
            return None

    @staticmethod
    def estimate_gpu_layers(
        total_layers: int,
        gguf_size_bytes: int,
        vram_free_bytes: int,
        *,
        kv_reserve_bytes: int = 512 * 1024**2,
        projector_reserve_bytes: int = 0,
        safety_margin: float = 1.15,
    ) -> int:
        """G4.5 部分 offload 层数估算（8GB 显存预算门）。

        全量 offload 在本机（RTX 4060 8GB）超预算（Q4 权重 7.1GB + mmproj +
        KV > 可用 ~7.4GB），必须部分 offload：按每层权重字节数计算可容纳层数，
        预留 KV 与安全余量（§5.7 决策：不承诺全量 offload）。
        """
        if total_layers <= 0 or gguf_size_bytes < 0 or vram_free_bytes < 0:
            raise ValueError("GPU layer budget inputs must be non-negative")
        if kv_reserve_bytes < 0 or projector_reserve_bytes < 0 or safety_margin <= 0:
            raise ValueError("GPU layer budget reserves are invalid")
        per_layer = max(1, gguf_size_bytes // total_layers)
        # safety_margin is headroom: required bytes * margin must fit in free VRAM.
        budget = int(
            vram_free_bytes / safety_margin
            - kv_reserve_bytes
            - projector_reserve_bytes
        )
        if budget <= 0:
            return 0
        return max(0, min(total_layers, budget // per_layer))

    def _install_chat_template_controls(self) -> None:
        """Expose thinking kwargs through the selected GGUF or registered handler."""
        chat_format = getattr(self._model, "chat_format", None)
        metadata = getattr(self._model, "metadata", None) or {}
        template_key = (
            "tokenizer.chat_template" if chat_format == "chat_template.default"
            else f"tokenizer.{chat_format}" if str(chat_format).startswith("chat_template.")
            else ""
        )
        native_template = metadata.get(template_key, "")
        native_thinking = bool(
            not getattr(self._model, "chat_handler", None)
            and isinstance(native_template, str)
            and re.search(r"\benable_thinking\b", native_template)
        )
        if self._chat_template != "qwen3_chat_v1" and not native_thinking:
            return
        try:
            from llama_cpp import llama_chat_format

            base_handler = (
                getattr(self._model, "chat_handler", None)
                or getattr(self._model, "_chat_handlers", {}).get(
                    getattr(self._model, "chat_format", None),
                )
                or llama_chat_format.get_chat_completion_handler(
                    getattr(self._model, "chat_format", None),
                )
            )
            if not callable(base_handler):
                return

            def chat_handler(*args, **handler_kwargs):
                template_kwargs = dict(self._chat_template_kwargs)
                if not template_kwargs:
                    return base_handler(*args, **handler_kwargs)
                try:
                    return base_handler(
                        *args, **{**template_kwargs, **handler_kwargs},
                    )
                except TypeError as exc:
                    if "unexpected keyword" not in str(exc):
                        raise
                    self._thinking_controlled = False
                    return base_handler(*args, **handler_kwargs)

            self._model.chat_handler = chat_handler
            if native_thinking:
                # Registry aliases need no hard-coded model-family mapping: the
                # selected GGUF Jinja template declares its own control variable.
                self._chat_template = chat_format
                self._thinking_mode = "declared"
            self._thinking_controlled = True
            logger.info(
                "llama.cpp thinking control installed: template=%s format=%s source=%s",
                self._chat_template, chat_format,
                template_key if native_thinking else "model_profile",
            )
        except Exception:
            logger.debug("Qwen3 chat-template control unavailable", exc_info=True)

    def _set_thinking_mode(self, enabled: bool | None) -> None:
        if (
            self._chat_template != "qwen3_chat_v1" and not self._thinking_controlled
        ) or enabled is None:
            return
        self._thinking_enabled = bool(enabled)
        self._chat_template_kwargs = {"enable_thinking": bool(enabled)}

    def _create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: List[str] = None,
        stream: bool,
        **kwargs,
    ):
        show_thinking = kwargs.pop("show_thinking", None)
        enable_thinking = kwargs.pop("enable_thinking", None)
        # None means use the template default for this request, including after
        # an earlier request explicitly disabled or enabled thinking.
        self._thinking_enabled = None
        self._chat_template_kwargs = {}
        self._set_thinking_mode(
            enable_thinking if enable_thinking is not None else show_thinking,
        )
        return self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=_merge_stop_sequences(stop),
            stream=stream,
            **kwargs,
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _verify_gemma4_native_assets(
        cls,
        lock_path: Path,
        gguf_path: Path,
        mmproj_path: Path,
    ) -> None:
        """Verify the exact frozen Gemma pair before any model initialization."""
        try:
            record = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("gemma4-native lock file is invalid") from exc
        if record.get("schema_version") != 1 or not isinstance(record.get("artifacts"), dict):
            raise RuntimeError("gemma4-native lock schema is invalid")
        artifacts = record["artifacts"]
        for key, path in (("main_gguf", gguf_path), ("mmproj", mmproj_path)):
            expected = artifacts.get(key)
            if not isinstance(expected, dict):
                raise RuntimeError(f"gemma4-native lock is missing {key}")
            filename = expected.get("filename")
            digest = expected.get("sha256")
            size_bytes = expected.get("size_bytes")
            if (
                not isinstance(filename, str)
                or path.name != filename
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 1
            ):
                raise RuntimeError(f"gemma4-native lock entry {key} is invalid")
            try:
                actual_size = path.stat().st_size
            except OSError as exc:
                raise FileNotFoundError(f"gemma4-native artifact is unavailable: {key}") from exc
            if actual_size != size_bytes:
                raise RuntimeError(f"gemma4-native {key} size does not match the lock")
            if cls._sha256_file(path) != digest.lower():
                raise RuntimeError(f"gemma4-native {key} SHA-256 does not match the lock")

    def load_gemma4_native(
        self,
        *,
        gguf_path: str = None,
        mmproj_path: str = None,
        n_ctx: int = 768,
        gpu_layers: int = -1,
        require_gpu_layers: int = 0,
        mtmd_use_gpu: bool = True,
        **llama_kwargs,
    ) -> Dict[str, Any]:
        """G4.5 便捷加载：gemma4 原生工件（受管目录）+ GPU 预算门 + 互斥规则。

        - 缺省工件路径从 models/gemma4-native/gemma4-native.lock.json 读取；
        - gpu_layers=-1 时按显存预算自动估算（部分 offload）；
        - 显存不足（低于 require_gpu_layers 对应预算）时 fail-closed。

        Args:
            **llama_kwargs: 透传给底层 `llama_cpp.Llama` 的额外加载参数
                （与 `load_model(**kwargs)` 同一通道）。详见 `load_model` 的说明。
        """
        # 显式拒绝会与本方法的预算门冲突的参数，避免静默覆盖语义。
        _conflicting = {"n_gpu_layers"}
        _bad = sorted(_conflicting & set(llama_kwargs))
        if _bad:
            raise ValueError(
                f"{', '.join(_bad)} 由本方法的 gpu_layers/require_gpu_layers 预算门管理，"
                "请改用 gpu_layers 形参（不通过 **llama_kwargs 传入）"
            )
        if isinstance(gpu_layers, bool) or not isinstance(gpu_layers, int):
            raise ValueError("gpu_layers must be an integer")
        if gpu_layers < -1 or gpu_layers > 36:
            raise ValueError("gpu_layers must be -1 or in the range 0..36")
        if isinstance(require_gpu_layers, bool) or not isinstance(require_gpu_layers, int):
            raise ValueError("require_gpu_layers must be an integer")
        if require_gpu_layers < 0 or require_gpu_layers > 36:
            raise ValueError("require_gpu_layers must be in the range 0..36")
        if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx < 1:
            raise ValueError("n_ctx must be a positive integer")
        if not isinstance(mtmd_use_gpu, bool):
            raise ValueError("mtmd_use_gpu must be boolean")

        if getattr(sys, "frozen", False):
            asset_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        else:
            asset_root = Path(__file__).resolve().parents[1]
        lock = asset_root / "models" / "gemma4-native" / "gemma4-native.lock.json"
        if not lock.is_file():
            raise RuntimeError("gemma4-native 工件未冻结：先运行 gemma4_native_freeze.py --hash")
        try:
            record = json.loads(lock.read_text(encoding="utf-8"))
            artifacts = record["artifacts"]
            main_record = artifacts["main_gguf"]
            mmproj_record = artifacts["mmproj"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("gemma4-native lock file is invalid") from exc
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or not isinstance(artifacts, dict)
            or not isinstance(main_record, dict)
            or not isinstance(mmproj_record, dict)
            or not isinstance(main_record.get("filename"), str)
            or not isinstance(mmproj_record.get("filename"), str)
        ):
            raise RuntimeError("gemma4-native lock schema is invalid")
        if gguf_path is None:
            gguf_path = str(lock.parent / main_record["filename"])
        if mmproj_path is None:
            mmproj_path = str(lock.parent / mmproj_record["filename"])
        gguf_file = Path(gguf_path).expanduser().absolute().resolve(strict=False)
        mmproj_file = Path(mmproj_path).expanduser().absolute().resolve(strict=False)
        if not gguf_file.is_file() or not mmproj_file.is_file():
            raise FileNotFoundError("gemma4-native 工件缺失：检查 models/gemma4-native/ 与冻结记录")

        # 额外加载参数（tensor_split / use_mmap / use_mlock / kv_overrides /
        # split_mode / main_gpu / numa / op_offload 等）先落入，再由预算门决定 n_gpu_layers。
        load_kwargs: Dict[str, Any] = dict(llama_kwargs)
        if gpu_layers > 0:
            load_kwargs["n_gpu_layers"] = int(gpu_layers)
        elif gpu_layers == -1:
            free = self._vram_free_bytes()
            if free is None:
                raise RuntimeError(
                    "无法查询显存（nvidia-smi 不可用）：请显式传 gpu_layers 或使用 CPU",
                )
            auto = self.estimate_gpu_layers(
                36,
                gguf_file.stat().st_size,
                free,
                projector_reserve_bytes=(mmproj_file.stat().st_size if mtmd_use_gpu else 0),
            )
            load_kwargs["n_gpu_layers"] = auto
            logger.info(
                "gemma4-native 自动 offload 层数: %d（VRAM %.1f GiB 预算门）",
                auto, free / 2**30,
            )
        if require_gpu_layers > 0 and load_kwargs.get("n_gpu_layers", 0) < require_gpu_layers:
            raise RuntimeError(
                f"显存预算不足：需要 ≥{require_gpu_layers} 层 offload，"
                f"实际 {load_kwargs.get('n_gpu_layers', 0)}（fail-closed）"
            )
        self._verify_gemma4_native_assets(lock, gguf_file, mmproj_file)
        self.load_model(
            model_path=str(gguf_file),
            n_ctx=n_ctx,
            mmproj_path=str(mmproj_file),
            mtmd_use_gpu=mtmd_use_gpu,
            **load_kwargs,
        )
        return self.get_capabilities()

    def unload(self) -> None:
        """卸载模型，释放内存。"""
        upstream = getattr(self, "_keep_head", None)
        if upstream is not None:
            # Keep ownership until physical retirement succeeds. This runner
            # holds segment weights independently of the in-process binding.
            upstream.close()
            self._keep_head = None
        self._keep_head_key = None
        self._free_mtmd_context()
        # 顺序：先释放模型（其张量还在 RPC device 上），再撤 device/worker
        model = self._model
        close = getattr(model, "close", None)
        if callable(close):
            close()
        self._model = None
        self._close_rpc_session()
        self._loaded = False
        self._embedding_enabled = False
        self._embedding_dimension = None
        self._embedding_model_id = ""
        self._chat_template = ""
        self._thinking_mode = "unknown"
        self._thinking_enabled = None
        self._thinking_controlled = False
        self._chat_template_kwargs = {}
        self._pipeline_descriptor = None
        self._pipeline_sha_cache = {}
        self._pipeline_distributed_only = False
        self._model_path = ""
        logger.info("GGUF 模型已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    def get_device(self) -> str:
        """流水线/缓存路径要的 device —— llama.cpp 恒为 CPU。

        PyTorch 侧返回 `torch.device`，调用方随后 `.to(...)` 搬运张量。llama.cpp 的权重
        就在本进程内存里，没有设备搬运这回事，所以返回 `"cpu"`；配套地，hidden 与 token
        id 一律用 numpy 表达（见 `_hidden_to_raw_f32` 与 `tokenizer` 适配器）。
        """
        return "cpu"

    # ================================================================
    # Native MTMD multimodal pipeline (G4.3.2B)
    # ================================================================

    def _load_mtmd_module(self):
        if self._mtmd_module is not None:
            return self._mtmd_module
        try:
            import llama_cpp.mtmd_cpp as mtmd
        except ImportError as exc:
            raise RuntimeError("llama.cpp MTMD binding is unavailable") from exc
        required = (
            "mtmd_context_params_default", "mtmd_init_from_file", "mtmd_free",
            "mtmd_support_vision", "mtmd_support_audio", "mtmd_default_marker",
            "mtmd_helper_bitmap_init_from_file", "mtmd_bitmap_free",
            "mtmd_input_chunks_init", "mtmd_input_chunks_free",
            "mtmd_input_chunks_size", "mtmd_input_chunks_get",
            "mtmd_input_chunk_get_type", "mtmd_input_text", "mtmd_bitmap_p_ctypes",
            "mtmd_tokenize",
            "mtmd_helper_eval_chunk_single", "mtmd_batch_init", "mtmd_batch_free",
            "mtmd_batch_add_chunk", "mtmd_batch_encode",
            "mtmd_batch_get_output_embd", "mtmd_helper_post_decode_callback",
            "mtmd_helper_decode_image_chunk",
        )
        missing = [name for name in required if not callable(getattr(mtmd, name, None))]
        if missing:
            raise RuntimeError("llama.cpp MTMD ABI missing symbols: " + ", ".join(missing))
        self._mtmd_module = mtmd
        return mtmd

    def _free_mtmd_context(self) -> None:
        context = self._mtmd_context
        self._mtmd_context = None
        self._mmproj_path = ""
        self._mtmd_capabilities = {"vision": False, "audio": False}
        if context is not None and self._mtmd_module is not None:
            try:
                self._mtmd_module.mtmd_free(context)
            except Exception:
                logger.debug("MTMD context release failed", exc_info=True)

    def load_mmproj(
        self,
        mmproj_path: str,
        *,
        use_gpu: bool = False,
        n_threads: int = None,
    ) -> Dict[str, Any]:
        """Load an explicit local projector and register native capabilities."""
        if not self.is_loaded:
            raise RuntimeError("load_model() must be called before load_mmproj()")
        if not mmproj_path or not os.path.isfile(mmproj_path):
            raise FileNotFoundError("MTMD projector file does not exist")
        mtmd = self._load_mtmd_module()
        self._free_mtmd_context()
        params = mtmd.mtmd_context_params_default()
        if hasattr(params, "use_gpu"):
            params.use_gpu = bool(use_gpu)
        if n_threads is not None and hasattr(params, "n_threads"):
            params.n_threads = int(n_threads)
        context = mtmd.mtmd_init_from_file(
            os.fspath(mmproj_path).encode("utf-8"), self._model.model, params,
        )
        if not context:
            raise RuntimeError("MTMD projector initialization failed")
        try:
            capabilities = {
                "vision": bool(mtmd.mtmd_support_vision(context)),
                "audio": bool(mtmd.mtmd_support_audio(context)),
            }
            if not capabilities["vision"]:
                raise RuntimeError("MTMD projector does not declare vision support")
        except Exception:
            try:
                mtmd.mtmd_free(context)
            except Exception:
                pass
            raise
        self._mtmd_context = context
        self._mmproj_path = os.path.abspath(mmproj_path)
        self._mtmd_capabilities = capabilities
        logger.info(
            "MTMD native capabilities registered: vision=%s audio=%s",
            capabilities["vision"], capabilities["audio"],
        )
        return self.get_capabilities()

    def get_capabilities(self) -> Dict[str, Any]:
        """Return a side-effect-free snapshot of registered capabilities."""
        return {
            "engine": "llama.cpp",
            "text": self.is_loaded,
            "native_mtmd": self._mtmd_context is not None,
            "vision": bool(self._mtmd_capabilities.get("vision")),
            "audio": bool(self._mtmd_capabilities.get("audio")),
            "mmproj_loaded": bool(self._mmproj_path),
            "embedding": self._embedding_enabled and self.is_loaded,
            "embedding_dimension": self._embedding_dimension if self._embedding_enabled else None,
            "embedding_model_id": self._embedding_model_id if self._embedding_enabled else None,
            "chat_template": self._chat_template or None,
            "thinking": self._thinking_mode,
            "thinking_controlled": self._thinking_controlled,
            "thinking_enabled": self._thinking_enabled,
        }

    @staticmethod
    def _mtmd_bitmap_pointer(wrapper):
        return getattr(wrapper, "bitmap", wrapper) if wrapper is not None else None

    def chat_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in one or two sentences. <__media__>",
        max_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 0.9,
        stop: List[str] = None,
        reasoning_channel_token_id: Optional[int] = 101,
        think_budget: int = 96,
        max_answer_tokens: int = 48,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the native MTMD image path for a single local image.

        Higher layers remain responsible for authorization, uploads and
        external-provider routing; this method only consumes a local image.
        """
        if not self.is_loaded:
            raise RuntimeError("model must be loaded before chat_image()")
        if not self._mtmd_context or not self._mtmd_capabilities.get("vision"):
            raise RuntimeError("native MTMD vision capability is not registered")
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError("image file does not exist")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if isinstance(max_answer_tokens, bool) or not isinstance(max_answer_tokens, int) or max_answer_tokens < 0:
            raise ValueError("max_answer_tokens must be non-negative")
        if isinstance(think_budget, bool) or not isinstance(think_budget, int) or think_budget < 0:
            raise ValueError("think_budget must be non-negative")
        if reasoning_channel_token_id is not None and (
            isinstance(reasoning_channel_token_id, bool)
            or not isinstance(reasoning_channel_token_id, int)
            or reasoning_channel_token_id < 0
        ):
            raise ValueError("reasoning_channel_token_id must be a non-negative integer or None")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        cancel_event = kwargs.pop("_cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            return {
                "content": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": os.path.basename(self._model_path),
                "finish_reason": "cancelled",
                "tokens_per_second": 0,
                "native_mtmd": True,
            }

        import llama_cpp
        mtmd = self._mtmd_module or self._load_mtmd_module()
        image_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
        if os.path.splitext(image_path)[1].lower() not in image_extensions:
            raise ValueError("chat_image() accepts image files only")

        marker = mtmd.mtmd_default_marker()
        marker_text = marker.decode("utf-8", "replace") if isinstance(marker, bytes) else str(marker)
        prompt = prompt.replace("<__media__>", marker_text)
        if marker_text not in prompt:
            prompt = marker_text + "\n" + prompt
        if prompt.count(marker_text) != 1:
            raise ValueError("chat_image() requires exactly one MTMD media marker")

        wrapper = None
        bitmap = None
        chunks = None
        media_batch = None
        decode_batch = None
        sampler = None
        t0 = time.time()
        finish_reason = "length"
        channel_found = reasoning_channel_token_id is None
        generated_tokens: List[int] = []
        pre_channel_tokens: List[int] = []
        sampled_token_count = 0
        # G4.5：思考预算（等效 llama.cpp b10434 reasoning-budget sampler）——
        # 思考段超预算强制结束（注入 101），HF 独立工件思考段失控时正文仍稳定
        CHANNEL_START_TOKEN_ID = 100  # "<|channel>"
        THOUGHT_TEXT_TOKEN_ID = 45518  # "thought" 字面（HF 工件无 100 时的思考开始）
        in_thinking = False
        think_tokens = 0

        try:
            wrapper = mtmd.mtmd_helper_bitmap_init_from_file(
                self._mtmd_context, os.fspath(image_path).encode("utf-8"), False,
            )
            bitmap = self._mtmd_bitmap_pointer(wrapper)
            if not bitmap:
                raise RuntimeError("MTMD bitmap initialization failed")
            if getattr(wrapper, "video_ctx", None):
                raise ValueError("video inputs are not supported by chat_image()")

            chunks = mtmd.mtmd_input_chunks_init()
            text = mtmd.mtmd_input_text(prompt.encode("utf-8"), True, True)
            bitmaps = (mtmd.mtmd_bitmap_p_ctypes * 1)(bitmap)
            rc = mtmd.mtmd_tokenize(
                self._mtmd_context, chunks, byref(text), bitmaps, 1,
            )
            if rc != 0:
                raise RuntimeError(f"MTMD tokenize failed (rc={rc})")

            model_ctx = self._model._ctx
            clear_kv = getattr(model_ctx, "kv_cache_clear", None)
            if callable(clear_kv):
                clear_kv()
            if hasattr(self._model, "n_tokens"):
                self._model.n_tokens = 0
            n_past = llama_cpp.llama_pos(0)
            n_batch = kwargs.pop("n_batch", 64)
            if isinstance(n_batch, bool) or not isinstance(n_batch, int) or n_batch < 1:
                raise ValueError("n_batch must be positive")

            @mtmd.mtmd_helper_post_decode_callback
            def _post_decode_callback(_batch, _user_data):
                return 0

            for index in range(mtmd.mtmd_input_chunks_size(chunks)):
                if cancel_event is not None and cancel_event.is_set():
                    finish_reason = "cancelled"
                    break
                chunk = mtmd.mtmd_input_chunks_get(chunks, index)
                chunk_type = mtmd.mtmd_input_chunk_get_type(chunk)
                new_n_past = llama_cpp.llama_pos(n_past.value)
                if chunk_type == mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    rc = mtmd.mtmd_helper_eval_chunk_single(
                        self._mtmd_context, model_ctx.ctx, chunk, n_past, 0,
                        n_batch, False, byref(new_n_past),
                    )
                elif chunk_type == mtmd.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                    if media_batch is None:
                        media_batch = mtmd.mtmd_batch_init(self._mtmd_context)
                    if not media_batch:
                        raise RuntimeError("MTMD batch initialization failed")
                    rc = mtmd.mtmd_batch_add_chunk(media_batch, chunk)
                    if rc == 0:
                        rc = mtmd.mtmd_batch_encode(media_batch)
                    if rc == 0:
                        embd = mtmd.mtmd_batch_get_output_embd(media_batch, chunk)
                        if not embd:
                            raise RuntimeError("MTMD image embedding is unavailable")
                        rc = mtmd.mtmd_helper_decode_image_chunk(
                            self._mtmd_context, model_ctx.ctx, chunk, embd, n_past, 0,
                            n_batch, byref(new_n_past), _post_decode_callback, None,
                        )
                else:
                    raise ValueError(f"unsupported MTMD input chunk type: {chunk_type}")
                if rc != 0:
                    raise RuntimeError(f"MTMD chunk evaluation failed (index={index}, rc={rc})")
                n_past = new_n_past

            if finish_reason == "cancelled":
                return {
                    "content": "",
                    "usage": {"prompt_tokens": int(n_past.value), "completion_tokens": 0,
                              "total_tokens": int(n_past.value)},
                    "model": os.path.basename(self._model_path),
                    "finish_reason": "cancelled",
                    "tokens_per_second": 0,
                    "native_mtmd": True,
                }

            from llama_cpp._internals import LlamaBatch, LlamaSampler

            decode_batch = LlamaBatch(n_tokens=1, embd=0, n_seq_max=1, verbose=False)
            sampler = LlamaSampler()
            if temperature <= 0:
                sampler.add_greedy()
            else:
                sampler.add_top_k(int(kwargs.pop("top_k", 40)))
                sampler.add_top_p(max(0.0, min(float(top_p), 1.0)))
                sampler.add_temp(float(temperature))
                sampler.add_dist(int(kwargs.pop("seed", 0)))

            n_past_value = int(n_past.value)
            last_token = llama_cpp.llama_token_bos(model_ctx.ctx)
            eos_token = llama_cpp.llama_token_eos(model_ctx.ctx)
            for _ in range(int(max_tokens)):
                if cancel_event is not None and cancel_event.is_set():
                    finish_reason = "cancelled"
                    break
                decode_batch.set_batch([last_token], n_past_value, False)
                model_ctx.decode(decode_batch)
                n_past_value += 1
                token = sampler.sample(model_ctx, -1)
                sampler.accept(token)
                if token == eos_token:
                    finish_reason = "stop"
                    break
                sampled_token_count += 1
                if reasoning_channel_token_id is not None:
                    if (
                        token == CHANNEL_START_TOKEN_ID
                        or (
                            not channel_found and not in_thinking
                            and not generated_tokens and not pre_channel_tokens
                            and token == THOUGHT_TEXT_TOKEN_ID
                        )
                    ):
                        # 思考段开始（通道标记或生成开头 "thought" 字面）
                        in_thinking = True
                        think_tokens = 0
                        channel_found = False
                        pre_channel_tokens.clear()
                        last_token = token
                        continue
                    if in_thinking:
                        think_tokens += 1
                        if token == reasoning_channel_token_id:
                            # 思考段自然结束
                            in_thinking = False
                            channel_found = True
                            last_token = token
                            continue
                        if think_budget > 0 and think_tokens > think_budget:
                            # 思考超预算：注入结束 tag（等效 reasoning-budget FORCING）
                            in_thinking = False
                            channel_found = True
                            last_token = reasoning_channel_token_id
                            continue
                        # 思考内容：丢弃
                        last_token = token
                        continue
                    if token == reasoning_channel_token_id:
                        if generated_tokens:
                            # 正文已结束（模型以 101 开新段即循环）：停止
                            break
                        # 思考段自然结束（无 100/45518 触发路径）
                        channel_found = True
                        last_token = token
                        continue
                    if not channel_found:
                        pre_channel_tokens.append(token)
                        last_token = token
                        continue
                    generated_tokens.append(token)
                    if max_answer_tokens > 0 and len(generated_tokens) >= max_answer_tokens:
                        break
                    last_token = token
                    continue
                generated_tokens.append(token)
                last_token = token

            output_tokens = generated_tokens if channel_found else pre_channel_tokens
            # G4.5 裁剪：头尾思考段残余（101/thought/空行）
            while output_tokens and output_tokens[-1] in (101, 106, 107, 45518):
                output_tokens.pop()
            while output_tokens and output_tokens[0] in (101, 106, 107):
                output_tokens.pop(0)
            content = self._model.detokenize(output_tokens).decode("utf-8", "replace")
            for stop_value in _merge_stop_sequences(stop):
                if stop_value and stop_value in content:
                    content = content.split(stop_value, 1)[0]
                    finish_reason = "stop"
                    break
            elapsed = time.time() - t0
            completion_tokens = sampled_token_count
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": int(n_past.value),
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(n_past.value) + completion_tokens,
                },
                "model": os.path.basename(self._model_path),
                "finish_reason": finish_reason,
                "tokens_per_second": round(completion_tokens / elapsed if elapsed > 0 else 0, 1),
                "native_mtmd": True,
                "reasoning_channel_found": channel_found,
            }
        finally:
            if sampler is not None:
                try:
                    sampler.close()
                except Exception:
                    logger.debug("MTMD sampler release failed", exc_info=True)
            if decode_batch is not None:
                try:
                    decode_batch.close()
                except Exception:
                    logger.debug("MTMD decode batch release failed", exc_info=True)
            if media_batch is not None:
                try:
                    mtmd.mtmd_batch_free(media_batch)
                except Exception:
                    logger.debug("MTMD media batch release failed", exc_info=True)
            if chunks is not None:
                try:
                    mtmd.mtmd_input_chunks_free(chunks)
                except Exception:
                    logger.debug("MTMD input chunks release failed", exc_info=True)
            if bitmap is not None:
                try:
                    mtmd.mtmd_bitmap_free(bitmap)
                except Exception:
                    logger.debug("MTMD bitmap release failed", exc_info=True)

    # ================================================================
    # 对话补全（对齐 ModelManager 接口）
    # ================================================================

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return native embeddings only for an explicitly declared model.

        ``llama-cpp-python`` exposes embedding extraction for a loaded GGUF,
        but enabling it for arbitrary chat models is a semantic error.  The
        caller must have loaded this engine with ``embedding=True`` and a
        declared identity/dimension.
        """
        if not self.is_loaded or not self._embedding_enabled or self._embedding_dimension is None:
            raise RuntimeError("native embedding capability is not registered")
        if not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("embedding input must be a non-empty list of text strings")
        create_embedding = getattr(self._model, "create_embedding", None)
        if not callable(create_embedding):
            raise RuntimeError("native llama.cpp binding has no embedding API")

        vectors: List[List[float]] = []
        for text in texts:
            try:
                response = create_embedding(text)
                data = response.get("data") if isinstance(response, dict) else None
                vector = data[0].get("embedding") if isinstance(data, list) and data and isinstance(data[0], dict) else None
            except Exception as exc:
                raise RuntimeError("native embedding inference failed") from exc
            if not isinstance(vector, list) or len(vector) != self._embedding_dimension:
                raise RuntimeError("native embedding dimension mismatch")
            try:
                normalized = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise RuntimeError("native embedding vector is invalid") from exc
            if any(not math.isfinite(value) for value in normalized):
                raise RuntimeError("native embedding vector is invalid")
            vectors.append(normalized)
        return vectors

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        对话补全，返回完整结果。

        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 温度 (0-2)
            top_p: nucleus sampling
            stop: 停止词列表

        Returns:
            {
                "content": "模型回复文本",
                "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": T},
                "model": "qwen-1_8b-chat-Q4_K_M",
                "finish_reason": "stop" | "length",
                "tokens_per_second": float,
            }
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        t0 = time.time()
        cancel_event = kwargs.pop("_cancel_event", None)

        if cancel_event is not None and cancel_event.is_set():
            return {
                "content": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "model": os.path.basename(self._model_path),
                "finish_reason": "cancelled",
                "tokens_per_second": 0,
                "usage_estimated": True,
            }

        # llama-cpp-python's chat API has no stopping_criteria argument. For
        # cooperative cancellation, consume its stream and close at a token
        # boundary; the task coordinator then discards this partial output.
        if cancel_event is not None:
            stream = self._create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=_merge_stop_sequences(stop),
                stream=True,
                **kwargs,
            )
            content_parts: List[str] = []
            finish_reason = "stop"
            try:
                for chunk in stream:
                    choices = chunk.get("choices", [])
                    if choices:
                        choice = choices[0]
                        text = choice.get("delta", {}).get("content", "")
                        if text:
                            content_parts.append(text)
                        finish_reason = choice.get("finish_reason") or finish_reason
                    if cancel_event.is_set():
                        break
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

            content = "".join(content_parts)
            elapsed = time.time() - t0
            completion_tokens = len(
                self._model.tokenize(
                    content.encode("utf-8"), add_bos=False, special=True,
                )
            ) if content else 0
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens,
                },
                "model": os.path.basename(self._model_path),
                "finish_reason": "cancelled" if cancel_event.is_set() else finish_reason,
                "tokens_per_second": round(
                    completion_tokens / elapsed if elapsed > 0 else 0, 1,
                ),
                "usage_estimated": True,
            }

        response = self._create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=_merge_stop_sequences(stop),
            stream=False,
            **kwargs,
        )

        elapsed = time.time() - t0
        choice = response["choices"][0]
        usage = response.get("usage", {})
        content = choice["message"].get("content", "")

        # 计算 tokens/s
        completion_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            f"推理完成: {completion_tokens} tokens / {elapsed:.1f}s "
            f"= {tok_per_sec:.1f} tok/s"
        )

        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("total_tokens", 0),
            },
            "model": os.path.basename(self._model_path),
            "finish_reason": choice.get("finish_reason", "stop"),
            "tokens_per_second": round(tok_per_sec, 1),
        }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        流式对话补全，逐步 yield token 文本。

        Args:
            同 chat()

        Yields:
            str: 增量文本 chunk
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        cancel_event = kwargs.pop("_cancel_event", None)
        stream = self._create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=_merge_stop_sequences(stop),
            stream=True,
            **kwargs,
        )

        try:
            for chunk in stream:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                if cancel_event is not None and cancel_event.is_set():
                    break
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    # ================================================================
    # 工具方法
    # ================================================================

    def _auto_threads(self) -> int:
        """自动检测 CPU 核心数，返回推荐线程数（物理核心数）。"""
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
            if physical and physical > 0:
                return min(physical, 8)  # 最多 8 线程，避免资源争抢
        except Exception:
            pass

        # fallback: os.cpu_count() 返回逻辑核心数
        logical = os.cpu_count() or 4
        return max(2, min(logical // 2, 8))

    def get_memory_usage(self) -> dict:
        """获取当前内存占用估算。"""
        import psutil
        mem = psutil.virtual_memory()
        process = psutil.Process()
        proc_mem = process.memory_info().rss / (1024 ** 3)
        return {
            "process_gb": round(proc_mem, 2),
            "system_available_gb": round(mem.available / (1024 ** 3), 1),
            "system_percent": mem.percent,
        }

    @property
    def model_name(self) -> str:
        """★ 2026-09-24：本引擎**实际加载的**模型名（供 `/status` 等消费者使用）。

        取值顺序：GGUF 自带的 `general.name`（最真实；实测 `qwen35-2b-Q4_K_M.gguf`
        ⇒ `"Qwen3 5 2b"`）→ GGUF **文件名**（去扩展名）→ 父目录名 → 空串。

        ⚠️ **绝不**返回与"实际加载了什么"无关的静态默认值。此前的缺陷正是：
        `get_model_info()` 不报 `model_name` ⇒ 消费方各自兜底到静态 `config.MODEL_NAME`
        （= 默认 0.6B）⇒ 加载 2B 却对外报 `Qwen/Qwen3-0.6B`。
        根因与实测证据见 `docs/未完成工作备忘-2026-09-23.md` §4.8 B2。
        """
        if not self._model_path:
            return ""
        model = getattr(self, "_model", None)
        if model is not None:
            try:
                metadata = getattr(model, "metadata", None) or {}
                name = str(metadata.get("general.name", "") or "").strip()
                if name:
                    return name
            except Exception:  # noqa: BLE001 - 元数据读不到就回退，绝不因它报错
                pass
        try:
            path = Path(self._model_path)
            return path.stem or path.parent.name or ""
        except Exception:  # noqa: BLE001
            return ""

    def get_model_info(self) -> dict:
        """获取模型基本信息。"""
        info = {
            "engine": "llama.cpp",
            "model_path": self._model_path,
            "model_name": self.model_name,
            "quant_type": self._quant_type,
            "n_ctx": self._n_ctx,
            "n_threads": self._n_threads,
            "loaded": self._loaded,
            "capabilities": self.get_capabilities(),
        }
        if self._loaded:
            info["memory"] = self.get_memory_usage()
        return info

    def _kv_used_cells(self) -> int:
        """当前 context 里 KV 已占用的格数（**自行跟踪**）。

        `llama-cpp-python` 没有暴露 `llama_get_kv_cache_used_cells`，而 `Llama.n_tokens`
        只在它自己的 `generate`/`create_completion` 路径里维护 —— 本引擎的层段 forward
        直接调原生 `llama_decode` ⇒ 两者都拿不到真实占用（实测：拿 `n_tokens` 恒为 0，
        LM Head 每次从位置 0 重跑 ⇒ `llama_decode rc=-1`）。

        所以由两个 `forward_layers_*` 入口累加、`load_model` 归零。它只服务于「给辅助
        decode 找位置」（如 LM Head），**不**参与主流水线的 `n_past` —— 那条由调度层的
        整数句柄驱动，两者互不影响。
        """
        return max(0, int(getattr(self, "_kv_used", 0) or 0))

    def forward_lm_head(self, hidden_states):
        """主节点的「架构感知 LM Head」：把末段回来的 hidden 投影成 logits。

        llama.cpp 的 LM Head 就在同一份 GGUF 里，所以「算 LM Head」= 用 hidden 跑一次
        `forward_layers_from_hidden` 并取 logits。**tied embeddings 的模型（Qwen2.5 就是）
        根本不存 `output.weight`** —— llama.cpp 会让 `token_embd.weight` 兼任，
        因此主节点只需拿到含 `token_embd` 的前段工件就能当流水线的头。这不是兜底，
        是 GGUF 的常规形态（`TENSOR` 表里查不到 `output.weight` 属正常）。

        返回形状 `[1, 1, n_vocab]`：调用方的 `_sample_pipeline_token_id` 契约要求 3 维且
        首维为 1（与 PyTorch 侧 `forward_lm_head` 的输出同形）。
        """
        if not self.is_loaded:
            raise RuntimeError("llama.cpp 模型未加载，无法执行 LM Head")
        # `llama_decode` 会**占** KV 位置。主节点首段已经占了若干格，若这里从
        # `n_past=0` 重跑就会以 `rc=-1` 撞车（`forward_layers_from_hidden` 的文档里记过
        # 这个坑）。LM Head 只把 hidden 投影成 logits、不关心位置语义，所以续在**实际
        # 已用格数**之后即可。
        #
        # ⚠️ 不能用 `llama_cpp.Llama.n_tokens`：那个计数只在它自己的
        #    `generate`/`create_completion` 路径里维护，而本引擎的层段 forward 直接走
        #    原生 `llama_decode` ⇒ 它恒为 0，等于又从 0 开始。实测症状正是 `rc=-1`。
        n_past = self._kv_used_cells()
        try:
            logits = self.forward_layers_from_hidden(
                hidden_states, n_past=n_past, all_logits=False,
            )
        finally:
            # ★ C 案：LM Head 对主链路的 KV **完全透明**。`llama_decode` 必然占位置，
            #   算完就把自己占的那一格退掉，否则流水线下一步 decode 的 `n_past` 会撞上它。
            self._drop_kv_from(n_past)
        if logits is None:
            raise RuntimeError("llama.cpp LM Head 未返回 logits")
        import numpy as _np

        array = _np.asarray(logits, dtype=_np.float32)
        if array.ndim == 1:
            array = array[None, None, :]
        elif array.ndim == 2:
            array = array[None, :, :]
        return array

    def _drop_kv_from(self, pos: int) -> None:
        """把 KV 里位置 `>= pos` 的格退掉，并同步自行跟踪的 `_kv_used`。

        `llama_decode` 只要调用就**必然占** KV 位置 —— 没有「只投影不占位」的 API ⇒
        辅助 decode（如 LM Head）算完后必须把自己占的删掉，否则流水线下一步 decode 的
        `n_past` 会撞上它（实测：relay 链首段 prefill + relay 段都正常，到 LM Head 之后
        的 decode 步就 `llama_decode rc=-1`）。

        走 `llama_memory_seq_rm(memory, seq_id, p0, p1)`：删掉序列 0 中位置落在
        `[pos, ∞)` 的 token（其 docstring 明写 "Removes all tokens that belong to the
        specified sequence and have positions in [p0, p1)"，`p1 = -1` 表示到末尾）。
        失败只记 debug —— 这只是 KV 记账，不该让一次推理失败。
        """
        try:
            import llama_cpp.llama_cpp as M

            native_ctx = getattr(getattr(self._model, "_ctx", None), "ctx", None)
            if native_ctx is None:
                return
            memory = M.llama_get_memory(native_ctx)
            if not memory:
                return
            M.llama_memory_seq_rm(memory, 0, int(pos), -1)
        except Exception:
            logger.debug("退掉辅助 decode 的 KV 占用失败（忽略）", exc_info=True)
        finally:
            self._kv_used = max(0, int(pos))

    def forward_layers(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache: bool = True,
        apply_lm_head: bool = False,
        hidden_states=None,
        **kwargs: Any,
    ) -> dict:
        """流水线执行循环要的层段 forward —— 按输入分派到两个既有入口。

        PyTorch 侧一个 `forward_layers` 同时表达「吃 token」与「吃 hidden」两条输入；
        llama.cpp 侧是 `forward_layers_to_hidden`（吃 token）与
        `forward_layers_from_hidden`（吃 hidden）两个方法。这里按实参分派，并把返回
        归一到调用方要的 `{"hidden_states": ..., "cache": ...}` 形状。

        **KV 用 `n_past` 计数表达**（llama.cpp 的 KV 活在 context 内，没有可跨调用传递
        的 cache 对象）：`past_key_values` 给整数就当作已有位置数，返回的 `cache` 是
        推进后的位置数。调用方的 `_prefer_cache_state` 会把 `cache` 原样存下、下一轮再
        当 `past_key_values` 传回来 —— 因此这个 int 必须能自洽往返。给不出整数（None /
        torch 风格的层 tuple）时从 0 起算，这与接力首步的语义一致。
        """
        if hidden_states is not None:
            return self.forward_layers_from_hidden(
                hidden_states, n_past=_kv_position(past_key_values),
            )
        if input_ids is None:
            raise ValueError("forward_layers 需要 input_ids 或 hidden_states")

        import numpy as _np

        ids = _np.asarray(input_ids, dtype=_np.int64).reshape(-1).tolist()
        n_past = _kv_position(past_key_values)
        # 排障：relay 链的 decode 步若每步都拿到 `n_past == 0`，就会每步清 KV、每步从
        # 位置 0 重算 ⇒ 输出恒定不变（实测症状：尾部段永远吐同样两个 token）。
        # 这条日志一次就能区分「KV 句柄没传到」与「数值本身错」。
        logger.info(
            "层段 forward: n_past=%s past_kv=%r(%s) tokens=%d",
            n_past, past_key_values, type(past_key_values).__name__, len(ids),
        )
        if n_past == 0:
            # 从位置 0 起 ⇒ 先清掉 ctx 里的旧 KV。`llama_decode` 在 `[0, n)` 上重跑会直接
            # `rc=-1`（`forward_layers_to_hidden` 的文档记过这个坑）。任务级的 `_kv_cache`
            # 只是 KV **句柄**表，清它不清 ctx ⇒ 第二个请求的 prefill 必撞车，实测症状
            # 正是 relay 链首段 `llama_decode 失败 rc=-1`。
            self._clear_context_kv()
        hidden = self.forward_layers_to_hidden(
            ids, n_past=n_past, all_positions=True,
        )
        if hidden is None:
            raise RuntimeError("llama.cpp 层段 forward 未返回 hidden_states")
        return {
            "hidden_states": hidden,
            "cache": n_past + len(ids),
        }

    def _clear_context_kv(self) -> None:
        """清掉原生 context 里的 KV（新序列从位置 0 起时必须做）。

        `reset_kv_cache()` 是既有的 **stateless no-op**（注释写明 llama.cpp 绑定没暴露
        等价 API），真正能用的是 `Llama._ctx.kv_cache_clear()`。顺带把自行跟踪的
        `_kv_used` 归零，否则后续辅助 decode（如 LM Head）会续在已经不存在的 KV 之后。
        """
        try:
            self._model._ctx.kv_cache_clear()
        except Exception:
            logger.debug("清 context KV 失败（忽略）", exc_info=True)
        self._kv_used = 0

    def _keep_head_upstream(self):
        """惰性构造 keep-head 上游（带补丁 shim）；不可用时返回 `None`。

        为什么需要它：见 `docs/已知问题记录.md` #35 —— pip 绑定的
        `llama_get_embeddings_ith` 返回的是 `output_norm(H)`，而层接力要求的是
        **末层输出、`output_norm` 之前**的残差流。shim 的
        `llama_set_embeddings_layer_inp(lid == n_layer)` 给出的正是后者。

        按「工件路径 + 层数」缓存：`load_layer_range` 换了工件就自动重建。
        """
        import os
        from pathlib import Path

        desc = getattr(self, "_pipeline_descriptor", None) or {}
        artifact = desc.get("loaded_artifact") or self._model_path
        rng = desc.get("assignment_layer_range")
        if not artifact or not rng or len(rng) != 2:
            return None
        cut_layer = int(rng[1]) - int(rng[0])
        key = (str(artifact), cut_layer)
        if (
            getattr(self, "_keep_head_key", None) == key
            and getattr(getattr(self, "_keep_head", None), "is_available", True)
        ):
            return getattr(self, "_keep_head", None)

        old = getattr(self, "_keep_head", None)
        if old is not None:
            old.close()
        self._keep_head = None
        self._keep_head_key = key

        # ★ 2026-10-08（#53）：定位收敛到 `keep_head_shim.resolve_keep_head_shim()` ——
        #   与本模块此前内联的推导等价（环境变量 > 仓库默认构建产物），
        #   但**两侧共用同一事实源**，避免 worker 侧只读环境变量而 master 侧能推导。
        shim = resolve_keep_head_shim()
        if not shim:
            # shim 本体是 `qlh_keep_head.dll`（`qlh_kh_*` 入口在它里面）；同目录的
            # `libllama.dll` 只是它依赖的**带补丁** llama.cpp，不是 shim。
            logger.error(
                "keep-head 上游不可用：%s；"
                "拒绝交付语义不兼容的 embeddings hidden（docs/已知问题记录.md #35）",
                missing_shim_hint())
            return None
        # ★ 2026-10-07：**删除了此前的 `qwen3*` 无条件拒绝判据**。
        #   旧判据依据「`nextn` 导出末层输出，各架构 `t_h_nextn` 挂点不同（qwen2 在
        #   `output_norm` 之前、qwen3* 在之后，多一次 RMSNorm）」—— 但
        #   `scripts/model_tools/keep_head_shim/qlh_keep_head.c:14-21` 记载 2026-09-23
        #   起已统一改用 `layer_inp` 的 `lid == n_layer` 槽位（不再走
        #   `llama_set_embeddings_nextn`），挂点即「模型自身最后一层的输出、
        #   `output_norm` 之前」，**各架构一致**。
        #   实测（2026-10-07，`scripts/model_tools/probes/probe_q35_hook.py`，qwen3_5）：
        #   `mode="nextn"`（head16 工件）与 `mode="layer_inp"`（整模 cut=16）输出**相同**，
        #   且都对应 `[0,16)` 层段的输出 —— `rel_err=7.169e-02`、**`cos=0.988463`**；
        #   该 rel_err 量级是 GGUF Q4_K_M 量化 vs HF fp32 的正常差异。
        #   ⇒ 判据已过时（会**误拒** qwen3.5 的 llama 上游），故移除。
        #   详见 `docs/已知问题记录.md` #54 与 `docs/archive/distributed/真三机产品路径攻坚-2026-10-07.md` §4.2。
        try:
            from llama_keep_head import KeepHeadCleanupError, KeepHeadUpstream

            # 不用 `layer_inp`：shim 把它限制为 `cut_layer < n_layer`
            # （scripts/model_tools/keep_head_shim/qlh_keep_head.c:111），取不到
            # `lid == n_layer`（= 末层输出）这个恰好需要的挂点。
            # ★ 2026-10-11（三机验收实测）：默认 `n_batch=512` 下，qwen3.5（M-RoPE）
            #   的 embd 批次按 `4 × n_tokens` 读 pos ⇒ `n_tokens_all > n_batch` ⇒
            #   llama.cpp 的 `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` 触发、worker
            #   在 init 后立刻退出（master 只见 `keep-head worker exited before response`）。
            #   这正是上游 #28963 家族（M-RoPE + embd）在本机的复现。这里把批次上限
            #   放大到能容纳 4× 放大，并保留 `QLH_KEEP_HEAD_N_BATCH` 旋钮。
            self._keep_head = KeepHeadUpstream(
                shim, str(artifact), mode="nextn",
                n_ctx=int(getattr(self, "_n_ctx", 4096) or 4096),
                n_batch=max(
                    512,
                    int(os.environ.get("QLH_KEEP_HEAD_N_BATCH", "2048") or 2048),
                ))
        except KeepHeadCleanupError as exc:
            self._keep_head = exc.cleanup_owner
            self._keep_head_key = None
            raise
        except Exception:  # noqa: BLE001 - 语义不兼容时由调用方 fail-closed
            logger.warning(
                "keep-head 上游构造失败；层接力将 fail-closed，不交付 embeddings hidden "
                "（docs/已知问题记录.md #35）", exc_info=True)
            self._keep_head = None
        return self._keep_head

    def forward_layers_to_hidden(self, input_ids, n_past: int = 0,
                                   all_positions: bool = False):
        """★ 层接力上游入口（2026-09-19）：只跑**本模型（裁层 GGUF = 前 k 层）**的层，
        返回**末位 hidden**（`np.float32`，长度 `n_embd`）。

        与 `forward_layers_from_hidden()`（吃 hidden）对称，两者合起来让 **llama 也能当上游**，
        从而支持「**交换上下游**」：llama 上游 → torch 下游
        （`model_module.forward_layers(hidden_states=..., apply_lm_head=...)` 原生支持从 hidden 起算）。

        实现走 llama.cpp 的 embeddings 通道：`llama_set_embeddings(ctx, True)` ⇒ decode ⇒
        `llama_get_embeddings_ith(ctx, -1)`。

        Args:
            input_ids: token id 序列（list / 1D array）。
            n_past: KV 里已有的位置数（接力首步传 0）。
            all_positions: True 时返回**每个位置**的 hidden（`[n_tokens, n_embd]`），
                上游 **prefill 必须**用它（下游要整段）；False（默认）只返回末位。

        Returns:
            末位 hidden 的 `np.ndarray`（长度 n_embd），或 `None`（未加载）。
        """
        if not self.is_loaded:
            return None

        # ★ docs/已知问题记录.md #35：层接力要求「末层输出、`output_norm` 之前」，
        #   而下面的 embeddings 通道返回的是 `output_norm(H)`。二者语义不同，直接
        #   交付会让下游段在已 norm 的残差流上继续算 ⇒ 接力第一步即分叉。优先走
        #   带补丁的 keep-head；它自己的 KV 由 shim 侧的 runner 维护，故这里不动
        #   `self._kv_used`（那是本进程 pip ctx 的账）。
        keep = self._keep_head_upstream()
        if keep is not None:
            import numpy as np

            toks = [int(t) for t in np.asarray(input_ids).reshape(-1).tolist()]
            if not toks:
                raise ValueError("input_ids 不能为空")
            if int(n_past) == 0:
                # 新任务从位置 0 重来。keep-head worker 在同一进程里跨任务复用
                # （`_keep_head_upstream` 按工件缓存），不清记忆就会撞上上一任务
                # 的残留位置 —— llama.cpp 报 "tokens have inconsistent sequence
                # positions"，表现为 `llama_decode rc=-1`。与 relay tail 段的
                # `_pos` 泄漏同族，只是这次在 master 自己的首段里。
                keep.reset()
            out = np.asarray(keep.forward_tokens_to_hidden(toks, n_past=int(n_past)),
                             dtype=np.float32)
            return out if all_positions else out[-1].copy()

        # ★ 2026-10-06 加固（docs/已知问题记录.md #35）：
        #   本函数末尾的 pip-embeddings 通道返回的是 `output_norm(H)`，而层接力要求
        #   「末层输出、`output_norm` **之前**」—— 二者语义不同。直接交付会让下游段在
        #   已归一化的残差流上继续算，**首步即分叉且不报错**（症状：token 与 prompt
        #   无关地交替，见 #35 原文）。
        #   此前只在 `partial_assignment` 时才 fail-closed ⇒ 「纯整模 GGUF +
        #   直接 forward_layers」这条形态（legacy 首段 `scheduler_pipeline.py:3596/6388`、
        #   peer 侧 `inference_service/peer.py:919`、relay 探针
        #   `scripts/relay_experiment.py:948`）会**静默**走进 embeddings 分支。
        #   现改为**默认一律 fail-closed**；确需旧行为者用 `QLH_ALLOW_NORMED_UPSTREAM=1`
        #   显式放行（与 handoff 的 `upstream_applies_output_norm` 口径一致）。
        _allow_normed = os.environ.get(
            "QLH_ALLOW_NORMED_UPSTREAM", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not _allow_normed:
            raise RuntimeError(
                "llama.cpp 层接力需要 keep-head 上游（output_norm 之前的 hidden）；"
                "当前 shim 不可用，拒绝使用语义不兼容的 embeddings 通道"
                "（确需旧行为请设 QLH_ALLOW_NORMED_UPSTREAM=1）"
            )

        import numpy as np
        import llama_cpp.llama_cpp as M

        llm = self._model
        native_model = getattr(getattr(llm, "_model", None), "model", None)
        native_ctx = getattr(getattr(llm, "_ctx", None), "ctx", None)
        if native_model is None or native_ctx is None:
            raise RuntimeError("llama_engine: 取不到原生 llama_model / llama_context")

        toks = [int(t) for t in np.asarray(input_ids).reshape(-1).tolist()]
        n_tokens = len(toks)
        if n_tokens == 0:
            raise ValueError("input_ids 不能为空")
        n_embd = int(M.llama_model_n_embd(native_model))

        # 开启 embeddings 通道（否则 llama_get_embeddings_ith 返回空指针）
        M.llama_set_embeddings(native_ctx, True)
        batch = M.llama_batch_init(n_tokens, 0, 1)
        try:
            for i, tid in enumerate(toks):
                batch.token[i] = tid
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 0          # 上游不需要 logits，只要 hidden
                batch.pos[i] = int(n_past) + i
            batch.n_tokens = n_tokens
            rc = M.llama_decode(native_ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode 失败 rc={rc}")
            if all_positions:
                rows = []
                for i in range(n_tokens):
                    p = M.llama_get_embeddings_ith(native_ctx, i)
                    if not p:
                        raise RuntimeError(
                            f"llama_get_embeddings_ith({i}) 返回空（embeddings 通道未生效？）")
                    rows.append(np.ctypeslib.as_array(p, shape=(n_embd,)).copy())
                self._kv_used = self._kv_used_cells() + n_tokens
                return np.stack(rows, axis=0)          # [n_tokens, n_embd]
            emb_ptr = M.llama_get_embeddings_ith(native_ctx, n_tokens - 1)
            if not emb_ptr:
                raise RuntimeError("llama_get_embeddings_ith 返回空（embeddings 通道未生效？）")
            self._kv_used = self._kv_used_cells() + n_tokens
            return np.ctypeslib.as_array(emb_ptr, shape=(n_embd,)).copy()
        finally:
            M.llama_batch_free(batch)

    def forward_layers_from_hidden(self, hidden, n_past: int = 0,
                                   seq_ids: Optional[list] = None,
                                   positions: Optional[list] = None,
                                   all_logits: bool = False):
        """★ 层接力下游入口（2026-09-19）：把**上游 hidden** 注入 llama.cpp 的 `embd` 通道，
        只跑**本模型（裁层 GGUF）的层**，返回末位 logits。

        为什么需要：QLH 的跨框架层接力（D→L / L→L）此前**只有实验脚本**用裸 `llama_cpp` API
        实现（`build/cross-framework-layer-poc/relay_mainrepo_upstream.py`），主仓
        `llama_engine` **没有入口** —— 与 D21 票面「生产接线后置」吻合。本方法把下游接回主仓，
        使「上下游都走主仓引擎」成立（上游＝`model_module.forward_layers`）。

        Args:
            hidden: 形状 `[n_tokens, n_embd]`（或 1D）的上游 `hidden_states`；内部转 f32。
            n_past: 本模型 KV 里已有的位置数（接力首步传 0）。**仅在 `positions` 为 None 时使用。**
            seq_ids: ★ **多序列（批量交叠）**：长度 `n_tokens` 的序列号列表；None ⇒ 全部为 0（单序列，
                与旧行为一致）。多条序列须使用**互不相同**的 id。
            positions: ★ 每个 token 的 KV 位置列表；None ⇒ `[n_past, n_past+1, ...]`（旧行为）。
                多序列交错推进时必须由调用方显式给出。
            all_logits: True ⇒ 返回**每个 token** 的 logits（形状 `[n_tokens, n_vocab]`）；
                False（默认）⇒ 只返回末位（形状 `[n_vocab]`，与旧行为一致）。
                多序列下需自行按 `seq_ids` 取末位。

        Returns:
            末位 logits（或全部，见 `all_logits`）的 `np.ndarray`，或 `None`（未加载）。

        ⚠️ **KV 位置由调用方管理**：本方法**直接在 KV 里占 `[n_past, n_past+n_tokens)`**；
        接力 decode 的常规用法是逐步推进 `n_past`。若要在**同一位置**重跑，先
        `self._model._ctx.kv_cache_clear()` —— 注意 `reset_kv_cache()` 是既有的 stateless
        no-op，**不能**用来清这里的 KV；同 `n_past` 重跑会得到 `llama_decode rc=-1`。
        """
        if not self.is_loaded:
            return None
        # ★ 2026-10-08：登记下游工件量化档 + 检查两侧精度对齐（默认只 WARN，不拦截）。
        _note_relay_downstream_precision(self)
        import ctypes

        import numpy as np
        import llama_cpp.llama_cpp as M

        llm = self._model                                    # llama_cpp.Llama
        # ⚠️ 实测（probe_llama_handles.py）：`llm._model` 是 `LlamaModel` 包装，
        #    **不能**直接喂给 `llama_*` C 函数（报 ctypes ArgumentError: wrong type）；
        #    原生 `llama_model_p` 是它的 `.model` 属性。`llm._ctx.ctx` 即原生 context。
        native_model = getattr(getattr(llm, "_model", None), "model", None)
        native_ctx = getattr(getattr(llm, "_ctx", None), "ctx", None)
        if native_model is None or native_ctx is None:
            raise RuntimeError("llama_engine: 取不到原生 llama_model / llama_context")

        h = np.ascontiguousarray(np.asarray(hidden, dtype=np.float32))
        if h.ndim == 1:
            h = h[None, :]
        n_tokens = int(h.shape[0])
        n_embd = int(M.llama_model_n_embd_inp(native_model))
        if h.shape[1] != n_embd:
            raise ValueError(f"hidden 宽度 {h.shape[1]} != 模型 n_embd {n_embd}")

        # ★ 多序列（批量交叠）：None 时退化为旧的单序列行为，保证向后兼容。
        if seq_ids is None:
            seq_list = [0] * n_tokens
        else:
            seq_list = [int(s) for s in seq_ids]
            if len(seq_list) != n_tokens:
                raise ValueError(f"seq_ids 长度 {len(seq_list)} != n_tokens {n_tokens}")
        if positions is None:
            pos_list = [int(n_past) + i for i in range(n_tokens)]
        else:
            pos_list = [int(p) for p in positions]
            if len(pos_list) != n_tokens:
                raise ValueError(f"positions 长度 {len(pos_list)} != n_tokens {n_tokens}")

        # ★ M-RoPE（Qwen3.5 等，n_pos_per_embd = 4）：**embd 注入通道**下 llama.cpp 按 planar 取
        #   位置 ⇒ 必须先用倍数分配、再把同一位置广播到各分量（与 token 通道语义一致）。
        #   只填 n_tokens 个会让 M-RoPE 模型读到越界位置 ⇒ 位置语义错、argmax 首步即分叉。
        n_pos_per_embd = _model_n_pos_per_embd(native_model, M)
        batch = M.llama_batch_init(n_tokens * n_pos_per_embd, n_embd, 1)
        try:
            for i in range(n_tokens):
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_list[i]
                # ★ 每个 token 都可能是「某条序列的末位」⇒ logits 全开（下游只跑少数 token，开销可接受）
                batch.logits[i] = 1
                for j in range(n_pos_per_embd):
                    batch.pos[j * n_tokens + i] = pos_list[i]
            batch.n_tokens = n_tokens
            ctypes.memmove(batch.embd, h.ctypes.data, h.nbytes)
            rc = M.llama_decode(native_ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode 失败 rc={rc}")
            n_vocab = int(M.llama_vocab_n_tokens(M.llama_model_get_vocab(native_model)))
            # KV 占用自行跟踪（见 `_kv_used_cells`）：本函数直接走原生 `llama_decode`，
            # `Llama.n_tokens` 不会更新。
            self._kv_used = self._kv_used_cells() + n_tokens
            if all_logits:
                rows = [np.ctypeslib.as_array(M.llama_get_logits_ith(native_ctx, i),
                                              shape=(n_vocab,)).copy() for i in range(n_tokens)]
                return np.stack(rows, 0)
            logits_ptr = M.llama_get_logits_ith(native_ctx, n_tokens - 1)
            return np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
        finally:
            M.llama_batch_free(batch)

    def reset_kv_cache(self) -> None:
        """
        清空 KV 缓存（用于多会话切换）。

        llama.cpp Python 绑定暂不直接提供 KV cache 重置 API。
        通过重新创建聊天上下文实现：下一次 create_chat_completion
        不传历史消息即为新会话。
        """
        # llama-cpp-python 在每次 create_chat_completion 时独立处理上下文，
        # 不保留跨调用的 KV cache。此方法为接口兼容预留。
        logger.debug("KV cache reset (llama.cpp — stateless, no-op)")

    def tokenize(self, text: str) -> List[int]:
        """将文本转换为 token ID 列表。"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        return self._model.tokenize(text.encode("utf-8"))

    @staticmethod
    def _merge_stop_sequences(stop: List[str] = None) -> List[str]:
        """Expose the same text stops to chat and the distributed decode loop."""
        return _merge_stop_sequences(stop)

    def _get_generation_eos_token_ids(self, stop_sequences: List[str]):
        """Use valid EOS ids and exact single-token stops; keep multi-token stops as text."""
        from numbers import Integral

        ids: List[int] = []
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, Integral) and not isinstance(eos_id, bool) and eos_id >= 0:
            ids.append(int(eos_id))
        for text in stop_sequences:
            try:
                encoded = self._model.tokenize(
                    text.encode("utf-8"), add_bos=False, special=True,
                )
            except TypeError:
                # Some keep-head tokenizer facades only expose tokenize(bytes).
                encoded = self._model.tokenize(text.encode("utf-8"))
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if len(encoded) != 1:
                continue
            token_id = encoded[0]
            if (
                isinstance(token_id, Integral)
                and not isinstance(token_id, bool)
                and token_id >= 0
                and int(token_id) not in ids
                and self.detokenize([int(token_id)], special=True) == text
            ):
                ids.append(int(token_id))
        return ids if len(ids) > 1 else ids[0] if ids else None

    def detokenize(self, tokens: List[int], *, special: bool = False) -> str:
        """将 token ID 列表转换为文本。"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        try:
            text = self._model.detokenize(tokens, special=special)
        except TypeError:
            # LL keep-head facades may return str and omit the special keyword.
            text = self._model.detokenize(tokens)
        return text if isinstance(text, str) else text.decode("utf-8", errors="replace")

    @property
    def tokenizer(self):
        """流水线执行循环要的 HF 风格 `tokenizer`（懒建）。

        Route A / relay 的执行循环按 HF 的用法取词：`tokenizer(text)` 拿 `input_ids`、
        `.eos_token_id`、`.decode(ids)`。llama.cpp 只有 `tokenize` / `detokenize`，这里
        做一层薄适配 —— 无 torch 的边缘主节点不必为了分词再装一份 transformers。
        """
        cached = getattr(self, "_tokenizer_adapter", None)
        if cached is None:
            cached = _LlamaCppTokenizerAdapter(self)
            self._tokenizer_adapter = cached
        return cached


def _kv_position(past_key_values) -> int:
    """把流水线传来的 KV 句柄折算成 `n_past`。

    llama.cpp 的 KV 活在 context 内、没有可跨调用传递的 cache 对象，所以本引擎用
    **整数位置数**当句柄（`forward_layers` 返回的 `cache` 就是它）。PyTorch 风格的层
    tuple / cache 对象在这里没有对应物 ⇒ 一律从 0 起算（接力首步的语义）。
    """
    if isinstance(past_key_values, bool) or past_key_values is None:
        return 0
    if isinstance(past_key_values, (int, float)):
        return max(0, int(past_key_values))
    return 0


class _LlamaCppTokenizerAdapter:
    """把 `LlamaCppEngine` 的分词接口适配成流水线要的 HF 用法。

    只实现执行循环真正用到的那几项（`__call__` / `eos_token_id` / `decode`）。刻意
    **不假装**是完整 tokenizer：多出来的接口一旦被调用会立刻 `AttributeError`，而不是
    给出一个语义不对的结果。
    """

    def __init__(self, engine: "LlamaCppEngine") -> None:
        self._engine = engine

    @property
    def eos_token_id(self):
        try:
            return int(self._engine._model.token_eos())
        except Exception:
            return None

    def __call__(self, text, return_tensors: str = None, **kwargs):
        """`tokenizer(text)["input_ids"]` —— 形状为 `[1, tokens]` 的 numpy int64。"""
        import numpy as _np

        ids = _np.asarray([self._engine.tokenize(str(text))], dtype=_np.int64)
        return {
            "input_ids": ids,
            "attention_mask": _np.ones_like(ids, dtype=_np.int64),
        }

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        """接受 numpy 数组 / 张量 / 嵌套列表，一律拍平后交给 `detokenize`。"""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        flat: List[int] = []
        for item in ids:
            if isinstance(item, (list, tuple)):
                flat.extend(int(value) for value in item)
            else:
                flat.append(int(item))
        return self._engine.detokenize(flat, special=not skip_special_tokens)


# ================================================================
# 便捷函数
# ================================================================

def check_llama_cpp_available() -> bool:
    """检测 llama-cpp-python 是否可用。"""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def get_gguf_model_path(models_dir: str = None) -> Optional[str]:
    """
    在模型目录中自动查找可用的 GGUF 文件。

    Args:
        models_dir: 模型目录路径，默认 "models/"

    Returns:
        找到的 .gguf 文件完整路径，或 None
    """
    if models_dir is None:
        # PyInstaller 打包后 models/ 与 exe 同级
        if getattr(sys, 'frozen', False):
            models_dir = os.path.join(
                os.path.dirname(os.path.abspath(sys.executable)),
                "models",
            )
        else:
            models_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "models",
            )

    models_dir = os.path.abspath(models_dir)

    if not os.path.isdir(models_dir):
        return None

    # 查找所有 .gguf 文件
    gguf_files = []
    for fname in os.listdir(models_dir):
        if fname.lower().endswith(".gguf"):
            gguf_files.append(os.path.join(models_dir, fname))

    if not gguf_files:
        return None

    # 优先选择 Q4_K_M
    for path in gguf_files:
        if "Q4_K_M" in os.path.basename(path):
            return path

    # 其次 Q5_K_M
    for path in gguf_files:
        if "Q5_K_M" in os.path.basename(path):
            return path

    # 否则返回第一个找到的
    return gguf_files[0]
