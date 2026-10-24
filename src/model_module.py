"""
模型模块 — 模型加载、量化、算子融合、层级拆分、前向推理
==========================================================
功能职责:
1. 双引擎推理 — 自动选择 PyTorch (CUDA) 或 llama.cpp (CPU/集显)
2. 原版 Qwen-1.8B 模型加载（PyTorch + HuggingFace Transformers）
3. INT4/INT8 量化（bitsandbytes 加载时量化，CUDA only）
4. 自动算子融合 torch.compile（可选）
5. 模型层级拆分（按配置分配给主/从节点）
6. 模型前向推理入口

引擎选择逻辑:
  - TP 孤岛启用 (QLH_ISLAND_ENABLED) → island（OpenAI 兼容端点整请求转发）
  - CUDA 可用 → PyTorch + bitsandbytes（INT4 量化，显存 ~1.75 GB）
  - CPU / 集显 → llama.cpp + GGUF（Q4_K_M 量化，内存 ~1.2 GB）
  - 手动覆盖: config.INFERENCE_ENGINE = "pytorch" | "llama_cpp" | "island"

依赖:
  PyTorch 栈: torch, transformers, bitsandbytes
  llama.cpp 栈: llama-cpp-python (pip install llama-cpp-python)
"""

import hashlib
import inspect
import contextlib
import json
import logging
import os
import re
import threading
import time
from functools import wraps
from dataclasses import replace
from product_runtime_receipt import LayerMaterializationProof, TensorMaterialization
from typing import Tuple, Optional, Dict, Any, List, Union


def _configure_huggingface_cache() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_home = os.path.join(project_root, ".hf-cache")
    hf_modules_cache = os.path.join(hf_home, "modules")
    os.makedirs(hf_modules_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_MODULES_CACHE", hf_modules_cache)


_configure_huggingface_cache()

import psutil
import torch
import torch.nn as nn

# ★ transformers 5.x 兼容 shim：必须在「导入 transformers 之后、使用它的远程代码加载之前」
#   执行。原因：transformers 5.x 的 dynamic_module_utils.check_imports 在加载任何 remote code
#   （如 Qwen-1.8B 的 modeling_qwen.py）时会逐个 import 它声明的依赖，而
#   transformers_stream_generator 顶层引用了 5.x 已移除的 5 个符号，其 ImportError 会被直接抛出
#   ⇒ 连模型都加载不了。详见 src/transformers5_compat.py。
import transformers as _transformers  # noqa: F401  (仅为确保 transformers 先于 shim 导入)
from transformers5_compat import install as _install_transformers5_compat

_install_transformers5_compat()

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import (
    MODEL_NAME, MODEL_PATH, GGUF_MODEL_PATH,
    COMPILE_RECOMPILE_LIMIT,
COMPILE_MIN_PARAMS,
    QUANT_TYPE, USE_COMPILE, USE_MONOLITHIC_FORWARD,
    USE_FUSED_RMSNORM,
    DEVICE, TRUST_REMOTE_CODE,
    INFERENCE_ENGINE,
    TOTAL_MODEL_LAYERS, DEFAULT_LAYER_CONFIG,
)
from koakuma_engine import backend_capabilities, select_backend
from model_load_resolver import (
    ModelLoadFacts,
    ModelLoadResolution,
    OP_FULL_MODEL,
    effective_pytorch_cuda_available,
    preferred_engine_for_artifacts,
    resolve_model_load,
)

import model_config as mc

try:  # ⚠️ **必须平铺导入**：`relay_precision` 带**进程级状态**（两侧档位登记 + 判据缓存）；
    #    若这里用 `src.relay_precision` 而别处用 `relay_precision`，Python 会当成两个模块 ⇒
    #    状态各存一份、登记互不可见、对齐检查永不触发（与 `llama_engine` 同规）。
    from relay_precision import note_upstream_precision as _note_upstream_precision
except ImportError:  # pragma: no cover - 兜底：极少数以 `src` 为根的导入场景
    from src.relay_precision import note_upstream_precision as _note_upstream_precision
logger = logging.getLogger(__name__)

#: 匹配 `<root>layers.<i>.` 形式的权重 key（Qwen 系各包装器共用该形态）。
_LAYER_KEY_RE = re.compile(r"^(?P<root>.*\.)layers\.\d+\.")


def _iter_safetensors_keys(model_path: str) -> List[str]:
    """只读枚举 safetensors 的权重 key（优先 index.json，避免逐个 shard 打开）。"""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            return list(json.load(handle).get("weight_map", {}).keys())
    from safetensors import safe_open

    keys: List[str] = []
    for filename in sorted(
        name for name in os.listdir(model_path) if name.endswith(".safetensors")
    ):
        with safe_open(os.path.join(model_path, filename),
                       framework="pt", device="cpu") as handle:
            keys.extend(handle.keys())
    return keys


def _is_tied_word_embeddings(config, model_path: Optional[str] = None) -> bool:
    """判断是否为 tied embeddings（`lm_head` 与 `embed_tokens` 共享权重）。

    ★ B16：tied 模型（Qwen3.5 等）的 safetensors 里**没有** `lm_head.weight`
    （权重与 `embed_tokens` 共用）⇒ 分层加载做「末节点」（`has_lm_head=True`）时
    **不能**要求该张量，否则必然报 `分层权重不完整: lm_head.weight`。

    判定顺序（先便宜的、再昂贵的）：
      1. `config.tie_word_embeddings` 显式为 True/False ⇒ 直接定论；
      2. 该字段缺失时扫 safetensors 的 key：**没有** `lm_head.weight` 且**有**
         `embed_tokens.weight` ⇒ 视为 tied（探测失败一律返回 False，保证对既有
         非 tied 模型**行为完全不变**）。
    """
    explicit = getattr(config, "tie_word_embeddings", None)
    if explicit is not None:
        return bool(explicit)
    if not model_path:
        return False
    try:
        keys = _iter_safetensors_keys(model_path)
    except Exception as exc:  # noqa: BLE001 - 探测失败必须无副作用
        logger.debug(f"tied embeddings 探测失败（按非 tied 处理）: {exc}")
        return False
    has_lm = any(k == "lm_head.weight" or k.endswith(".lm_head.weight") for k in keys)
    has_embed = any(k.endswith("embed_tokens.weight") for k in keys)
    return (not has_lm) and has_embed


def _detect_qwen_root_prefix(
    model_path: str,
    fallback: Optional[str] = "model.",
) -> Optional[str]:
    """探测 `<root>layers.<i>.` 的 root（出现次数最多者）。

    用于适配不同 Qwen 系包装器的 key 前缀（如 Qwen3.5 的
    ``model.language_model.layers.``）。探测失败（含异常）时返回 ``fallback``，
    因此对既有 Qwen2 系列**行为完全不变**。
    """
    try:
        keys = _iter_safetensors_keys(model_path)
    except Exception as exc:  # noqa: BLE001 - 探测失败必须无副作用
        logger.debug(f"层前缀探测失败（回退默认）: {exc}")
        return fallback
    counts: Dict[str, int] = {}
    for key in keys:
        match = _LAYER_KEY_RE.match(key)
        if match:
            root = match.group("root")
            counts[root] = counts.get(root, 0) + 1
    if not counts:
        return fallback
    return max(counts, key=counts.get)


#: Qwen 系中共享「按 key 过滤的层范围加载」的架构（key 形态 ``<root>layers.<i>.``，root 由探测得出）。
_QWEN_LAYER_RANGE_TYPES = frozenset({"qwen2", "qwen3", "qwen3_5", "qwen3_5_text"})


def _layer_idx_holders(layer) -> list:
    """返回该层里「持有 layer_idx 的子模块」列表（A3）。

    不同架构的 attention 子模块名不同：
      * Qwen2 / Qwen3 的 full attention —— ``self_attn``
      * Qwen3.5 的 linear_attention 层 —— ``linear_attn``（Qwen3_5GatedDeltaNet）
    这些子模块用 ``layer_idx`` 索引 KV / 递归状态 cache，因此分段加载时必须把它们
    改成**本地索引**（否则 DynamicCache 会出现稀疏空洞）。这里按属性探测，不写死名字。
    """
    return [sub for _name, sub in layer.named_children() if hasattr(sub, "layer_idx")]


def _is_hybrid_layer_types(layer_types) -> bool:
    """``layer_types`` 是否表示「混合层型」（A3 / v4-v5 共用的判定）。

    只有出现**既不是 full_attention、也不是 sliding_attention** 的层才算 hybrid
    —— 例如 Qwen3.5 的 `linear_attention`（需要 recurrent mask，而非 causal mask）。

    ⚠️ 必须排除 sliding_attention，且**不能只判断「有无 layer_types」**：
    transformers 5.x 给 `Qwen2Config` 也加上了 `layer_types`（值全为 `full_attention`，
    实测 24/24），若只判断「有无」会把纯 full-attention 模型误判为 hybrid ⇒
    走进 MRoPE/逐层 mask 分支 ⇒ RoPE 张量维度错乱。
    """
    if not layer_types:
        return False
    try:
        return any(
            str(item) not in ("full_attention", "sliding_attention")
            for item in layer_types
        )
    except TypeError:
        return False


def _hybrid_mask_index_per_layer(layer_types):
    """把 ``layer_types`` 映射成「去重层型顺序」+「每层的 mask 索引」（B14）。

    返回 ``(layer_type_order, index_per_layer)``：

    * ``layer_type_order``：去重后的层型名，其**下标顺序**就是 mask 元组的下标顺序；
    * ``index_per_layer``：每层对应的下标（用于从 mask 元组里取本层 mask）。

    ⚠️ ``_apply_compile()`` 与 ``forward_layers()`` **必须**用同一份顺序（前者编译进层循环、
    后者据此构造 mask 元组），否则 mask 会张冠李戴 ⇒ 所以两者都调用本函数，不各自实现。
    """
    order: list = []
    for item in layer_types:
        name = str(item)
        if name not in order:
            order.append(name)
    return tuple(order), tuple(order.index(str(item)) for item in layer_types)


def _new_dynamic_cache(config=None):
    """构造 ``DynamicCache``（A3：跨 transformers 版本兼容）。

    新版（实测 5.17）支持 ``DynamicCache(config=...)``，会按 ``config.layer_types`` 建出
    linear/full 混合层 —— hybrid 架构（Qwen3.5）**必须**这样建，否则 ``cache.layers[layer_idx]``
    越界。旧版（4.x）没有该关键字 ⇒ 回退到空构造（对纯 full-attention 的 Qwen2 足够）。

    ⚠️ 5.x 的 `DynamicCache.__init__` 会调用 `config.get_text_config()`，而测试里的假 config
    常是 `SimpleNamespace`（没有该方法）⇒ 除 `TypeError` 外还要兜住 `AttributeError`，
    否则会把「config 不够完整」误报成失败（实测 test_gemma4_pipeline_adapter 就是这样红的）。
    """
    from transformers.cache_utils import DynamicCache

    try:
        return DynamicCache(config=config)
    except (TypeError, AttributeError):
        return DynamicCache()


def _normalize_past_key_values(past_key_values):
    """★ B9：把「**空的** cache 对象」规范化成 ``None``（视同没有缓存）。

    为什么需要：调用方很容易传一个**尚未写入任何内容**的 cache 对象（例如
    ``DynamicCache()`` / ``DynamicCache(config=cfg)``）而不是 ``None``。各架构随后会：

      * ``len(past_key_values)`` ⇒ ``0`` ⇒ 报 ``Qwen2 KV cache 层数不匹配: cache=0, local=N``
        —— **报错信息不指向根因**（看起来像"层数算错了"，实际是"传了空 cache"）；
      * 或直接 ``cache.layers[layer_idx]`` ⇒ ``IndexError: list index out of range``
        （hybrid 架构上尤其容易踩）。

    两者是**同一个坑**。这里统一识别「没有任何已写入层的 cache」并视同 ``None``，
    由 ``forward_layers`` 自建正确形状的 cache。

    ⚠️ 只处理**完全空**的情况；**部分写入**的 cache（如已缓存若干层）原样返回，不做任何改动。
    """
    if past_key_values is None:
        return None

    # (1) cache 对象（DynamicCache-like）：**以「已缓存长度 == 0」为准**。
    #     ⚠️ 不能只看层容器是否为空：5.x 的 `DynamicCache(config=...)` 会**按 config 预先建出**
    #     每层的 `DynamicLayer` 占位对象（实测 `layers=[DynamicLayer×4]`、`len(cache)==4`），
    #     但 `get_seq_length()` 仍是 0 —— 那些占位对象不是 KV，直接 enumerate 会以
    #     `k, v = item ⇒ ValueError: too many values to unpack` 失败（本票实际踩到）。
    try:
        seq_len = None
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                seq_len = int(get_seq_length())
            except (TypeError, ValueError):
                # 部分实现要求传 layer_idx；无参调用失败时不算"空"
                seq_len = None
        if seq_len is not None:
            return None if seq_len == 0 else past_key_values
    except Exception as exc:  # noqa: BLE001 - 规范化失败不应影响主流程
        logger.debug(f"空 cache 规范化：读取 get_seq_length 失败（按原样使用）: {exc}")

    # (2) 无 `get_seq_length` 的层容器（空 list/tuple、或全 None 占位）⇒ 没有任何真实 KV。
    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, (list, tuple)) and (
        len(layers) == 0 or all(x is None for x in layers)
    ):
        return None

    # (3) 纯 tuple/list 形式：空、或全 None 占位（B14 在 hybrid 收集侧会留 None 占位）。
    if isinstance(past_key_values, (tuple, list)) and (
        len(past_key_values) == 0 or all(x is None for x in past_key_values)
    ):
        return None

    return past_key_values


def _locate_text_transformer(model) -> tuple:
    """定位「文本 Transformer 主体」，返回 ``(transformer, layers_attr, embedding_attr)``。

    兼容三类包装（A3 泛化；**只做属性探测，不改变既有 Qwen2 的行为**）：
      * ``model.model`` —— Qwen2 / Qwen3（``layers`` / ``embed_tokens``）
      * ``model.model.language_model`` —— Qwen3.5 多模态外壳下的文本塔
      * ``model.transformer`` —— 旧 Qwen / GPT-2（``h`` / ``wte``）

    找不到时抛 ``RuntimeError``（不静默返回错误对象）。
    """
    candidates = []
    for outer in ("model", "transformer"):
        holder = getattr(model, outer, None)
        if holder is None:
            continue
        candidates.append(holder)
        inner = getattr(holder, "language_model", None)  # Qwen3.5 的文本塔
        if inner is not None:
            candidates.append(inner)
    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return candidate, "layers", "embed_tokens"
        if hasattr(candidate, "h"):
            return candidate, "h", "wte"
    raise RuntimeError(
        "无法定位文本 Transformer 主体"
        "（已尝试 model / model.language_model / transformer）"
    )


class _LayerLoop(torch.nn.Module):
    """把「逐层循环」封装成可编译模块（A4：compile 层循环，而非整个 Qwen2Model）。

    为什么只编译层循环：`Qwen2Model.forward()` 的返回值经过 `self.norm`（完整模型语义），
    而分布式分段前向在 `has_lm_head=False` 时必须返回**未过 norm** 的 raw hidden。
    这里只包层循环 ⇒ 前置/后置仍由 `forward_layers()` 负责 ⇒ 语义与逐层版一致。

    注意：`torch.compile` 会把 `layer.self_attn.layer_idx` 等属性纳入 guard，
    因此调用方必须在**编译之前**把这些属性设成最终值（见 `_apply_compile`）。
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        cache_arg_name: Optional[str],
        mask_index_per_layer: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.cache_arg_name = cache_arg_name
        # ★ B14：hybrid（如 Qwen3.5）的**每层 mask 不同**（full_attention 用 causal、
        #   linear_attention 用 recurrent）⇒ 这里存「每层的 mask 索引」，forward 时从
        #   attention_mask **元组**里取本层所需的那份；非 hybrid 时为 None（沿用单一份 mask）。
        #   该索引序列来自 `_hybrid_mask_index_per_layer()`，与 forward_layers 构造元组的顺序同源。
        self.mask_index_per_layer = mask_index_per_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        cache: Optional[object] = None,
    ) -> torch.Tensor:
        # ★ B14：hybrid 时 attention_mask 是「按层型索引的元组」（由 forward_layers 构造，
        #   顺序与 `_hybrid_mask_index_per_layer()` 同源）；非 hybrid 时是单个张量。
        #   用元组 + 静态索引序列（而非 dict）是为了让 torch.compile 好 guard。
        for index, layer in enumerate(self.layers):
            if self.mask_index_per_layer is not None:
                layer_mask = attention_mask[self.mask_index_per_layer[index]]
            else:
                layer_mask = attention_mask
            layer_kwargs = {
                "attention_mask": layer_mask,
                "position_ids": position_ids,
                "position_embeddings": position_embeddings,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
            if self.cache_arg_name is not None and cache is not None:
                layer_kwargs[self.cache_arg_name] = cache
            layer_output = layer(hidden_states, **layer_kwargs)
            # transformers>=5.x: DecoderLayer 直接返回 tensor；4.x: 返回元组
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        return hidden_states
_IMPORTED_INFERENCE_ENGINE = INFERENCE_ENGINE


def _apply_fused_rmsnorm(model: torch.nn.Module) -> int:
    """★ R-R7 rank3：把模型里的 `RMSNorm` 模块换成 `F.rms_norm` 融合实现。返回替换个数。

    **只按实例替换**（把 `module.forward` 设成闭包），**不 patch transformers 的类** —— 后者会连
    同进程里其他模型与其他用途一起改，范围失控。用闭包而非 `types.MethodType`：`nn.Module.__call__`
    取的是 `self.forward` 属性，实例属性为普通函数时不会自动传 `self`，闭包直接捕获模块更稳。

    语义与 `transformers` 的 `Qwen2RMSNorm.forward` 一致：`weight * x * rsqrt(mean(x²) + eps)`；
    差别只在**算子数**（6 → 1）与不再走 `.to(float32)` 往返（实测那 2 次 `.to` 占全部 `.to` 的 90%）。
    torch < 2.4 没有 `F.rms_norm` ⇒ 返回 0，行为不变。
    """
    if not hasattr(torch.nn.functional, "rms_norm"):
        return 0

    replaced = 0
    for module in model.modules():
        if "RMSNorm" not in type(module).__name__:
            continue
        weight = getattr(module, "weight", None)
        eps = getattr(module, "variance_epsilon", None)
        if eps is None:
            eps = getattr(module, "eps", None)
        if weight is None or eps is None:
            continue

        def forward(hidden_states, _module=module, _eps=eps):  # noqa: ANN001
            return torch.nn.functional.rms_norm(
                hidden_states, (_module.weight.shape[0],), _module.weight, _eps)

        module.forward = forward
        replaced += 1
    return replaced


def _select_layer_runtime(quant_type: str | None = None) -> Tuple[str, torch.dtype]:
    """Choose a stable dtype for selectively loaded pipeline layers."""
    if torch.cuda.is_available():
        normalized = str(quant_type or "").strip().lower()
        if normalized in {"fp32", "f32", "float32"}:
            return "cuda:0", torch.float32
        return "cuda:0", torch.float16
    return "cpu", torch.float32


LAYER_OFFLOAD_MATERIALIZATION = "safetensors_layer_offload"


def _layer_offload_device_available(device):
    return (torch.cuda.is_available()
            and int(str(device).partition(":")[2]) < torch.cuda.device_count())


class _LayerOffloadRuntime:
    """Stage one execution unit, restoring the original CPU mmap owners.

    CPU source tensors keep their serialized dtype. Conversion happens on CPU
    one tensor at a time, so CUDA holds target tensors only. KV/activations are
    owned by the existing forward path on execution_device. No GPU copy is
    retained by this owner after a call, including a failed transfer/forward.
    """

    def __init__(self, execution_device, dtype):
        self.execution_device = torch.device(execution_device)
        self.dtype = dtype
        self.stage_count = 0
        self.target_weight_peak_bytes = 0
        self.host_conversion_peak_bytes = 0
        self.active = False

    def _promote(self, tensor, *, preserve_dtype=False):
        dtype = self.dtype if tensor.is_floating_point() and not preserve_dtype else tensor.dtype
        converted = None
        try:
            converted = tensor.to(dtype=dtype)
            if converted is not tensor:
                self.host_conversion_peak_bytes = max(
                    self.host_conversion_peak_bytes, converted.numel() * converted.element_size(),
                )
            # Separating the CPU cast from the transfer prevents a source-dtype
            # CUDA allocation in addition to the target-dtype allocation.
            return converted.to(device=self.execution_device)
        finally:
            # Transfer errors may be retained for diagnostics after unloading.
            converted = None

    def call(self, module, *args, **kwargs):
        if self.active:
            raise RuntimeError("layer offload execution units must not overlap")
        self.active = True
        saved = []
        promoted = {}
        replacement = None
        try:
            with torch.no_grad():
                for child in module.modules():
                    for is_parameter, entries in ((True, child._parameters), (False, child._buffers)):
                        for name, original in tuple(entries.items()):
                            if original is None:
                                continue
                            if original.device.type != "cpu":
                                raise RuntimeError("layer offload source tensor must reside on CPU")
                            token = id(original)
                            if token not in promoted:
                                if not is_parameter and name in child._non_persistent_buffers_set:
                                    # RoPE keeps its generated inverse frequencies
                                    # in FP32 even when weight execution uses FP16.
                                    replacement = self._promote(original, preserve_dtype=True)
                                else:
                                    replacement = self._promote(original)
                                if is_parameter:
                                    replacement = nn.Parameter(replacement, requires_grad=original.requires_grad)
                                promoted[token] = replacement
                            replacement = promoted[token]
                            saved.append((entries, name, original, replacement, is_parameter))
                            entries[name] = replacement
                self.target_weight_peak_bytes = max(
                    self.target_weight_peak_bytes,
                    sum(t.numel() * t.element_size() for t in promoted.values()),
                )
                self.stage_count += 1
                return module(*args, **kwargs)
        except Exception as error:
            # Forward/transfer frames can own views of staged weights even
            # after restoring the module. Preserve traceback locations while
            # clearing those locals before a caller retains the exception.
            import traceback
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            for entries, name, original, staged, is_parameter in reversed(saved):
                current = entries.get(name)
                # Dynamic rotary modules may replace their generated buffers.
                # Preserve that state on CPU along with their cached length.
                if not is_parameter and current is not None and current is not staged:
                    entries[name] = current.to(device="cpu")
                else:
                    entries[name] = original
            promoted.clear()
            saved.clear()
            # An exception traceback can keep this frame alive. Drop every
            # temporary weight reference even while the error is retained.
            replacement = current = staged = None
            self.active = False

    def observation(self):
        return {"materialization": LAYER_OFFLOAD_MATERIALIZATION,
                "execution_device": str(self.execution_device),
                "execution_dtype": str(self.dtype).removeprefix("torch."),
                "source_device": "cpu", "source_dtype": "artifact",
                "stage_count": self.stage_count,
                "target_weight_peak_bytes": self.target_weight_peak_bytes,
                "host_conversion_peak_bytes": self.host_conversion_peak_bytes,
                "active": self.active}


class _LayerRangeLoadTracker:
    """Enforce filtered safetensors materialization and capture load metrics."""

    MODE = "safetensors_key_filtered"

    def __init__(
        self,
        *,
        architecture: str,
        start_layer: int,
        end_layer: int,
        layer_prefix: str,
        selected_prefixes: List[str],
        target_dtype: torch.dtype,
        preserve_source_dtype: bool = False,
    ) -> None:
        self.preserve_source_dtype = preserve_source_dtype
        self.architecture = architecture
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.layer_prefix = layer_prefix
        self.selected_prefixes = tuple(selected_prefixes)
        self.target_element_size = torch.empty((), dtype=target_dtype).element_size()
        self.loaded_keys = set()
        self._tensor_metadata = {}
        self.loaded_layers = set()
        self.source_tensor_bytes = 0
        self.materialized_tensor_bytes = 0
        self.started_at = time.monotonic()
        self._process = psutil.Process(os.getpid())
        self._has_cuda = torch.cuda.is_available()
        self._baseline = self._memory_snapshot()
        self._peaks = dict(self._baseline)

    def _memory_snapshot(self) -> Dict[str, int]:
        snapshot = {
            "rss_bytes": int(self._process.memory_info().rss),
            "cuda_allocated_bytes": 0,
        }
        if self._has_cuda:
            snapshot["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
        return snapshot

    def observe(self) -> None:
        snapshot = self._memory_snapshot()
        for key, value in snapshot.items():
            self._peaks[key] = max(self._peaks.get(key, 0), value)

    def is_selected(self, key: str) -> bool:
        return any(key.startswith(prefix) for prefix in self.selected_prefixes)

    def materialize(self, handle: Any, key: str, source_file: str = "") -> torch.Tensor:
        # Check before get_tensor so post-load pruning cannot hide an
        # accidental full-model materialization by a future adapter.
        if not self.is_selected(key):
            raise RuntimeError(
                "层流水线拒绝物化未分配权重: "
                f"architecture={self.architecture}, key={key}"
            )
        if key in self.loaded_keys:
            raise RuntimeError(f"duplicate tensor materialization: {key}")
        tensor = handle.get_tensor(key)
        self._tensor_metadata[key] = TensorMaterialization(
            name=key, shape=tuple(int(n) for n in tensor.shape),
            element_size=int(tensor.element_size() if self.preserve_source_dtype else self.target_element_size),
            source_element_size=int(tensor.element_size()),
            source_file=os.path.abspath(source_file) if source_file else "",
        )
        self.loaded_keys.add(key)
        self.source_tensor_bytes += tensor.numel() * tensor.element_size()
        self.materialized_tensor_bytes += tensor.numel() * (
            tensor.element_size() if self.preserve_source_dtype else self.target_element_size
        )

        if key.startswith(self.layer_prefix):
            suffix = key[len(self.layer_prefix):]
            layer_text, separator, _ = suffix.partition(".")
            if not separator or not layer_text.isdigit():
                raise RuntimeError(f"无法解析层权重 key: {key}")
            layer_index = int(layer_text)
            if not self.start_layer <= layer_index < self.end_layer:
                raise RuntimeError(
                    "层流水线拒绝物化范围外权重: "
                    f"key={key}, range=[{self.start_layer}, {self.end_layer})"
                )
            self.loaded_layers.add(layer_index)

        self.observe()
        return tensor

    def finish(self) -> Dict[str, Any]:
        self.observe()
        expected_layers = set(range(self.start_layer, self.end_layer))
        if self.loaded_layers != expected_layers:
            missing = sorted(expected_layers - self.loaded_layers)
            unexpected = sorted(self.loaded_layers - expected_layers)
            raise RuntimeError(
                "层流水线物化层与分配不一致: "
                f"missing={missing}, unexpected={unexpected}"
            )
        rss_baseline = self._baseline["rss_bytes"]
        cuda_baseline = self._baseline["cuda_allocated_bytes"]
        return {
            "materialization_proof": LayerMaterializationProof(
                layer_range=(self.start_layer, self.end_layer),
                decoder_prefix=self.layer_prefix,
                tensors=tuple(self._tensor_metadata[key] for key in sorted(self._tensor_metadata)),
                materialized_bytes=self.materialized_tensor_bytes,
                allowed_boundary_prefixes=tuple(
                    prefix for prefix in self.selected_prefixes
                    if not prefix.startswith(self.layer_prefix)
                ),
            ),
            "mode": self.MODE,
            "architecture": self.architecture,
            "layer_range": (self.start_layer, self.end_layer),
            "selected_layer_indices": sorted(self.loaded_layers),
            "selected_tensor_count": len(self.loaded_keys),
            "source_tensor_bytes": self.source_tensor_bytes,
            "materialized_tensor_bytes": self.materialized_tensor_bytes,
            "sampling": "tensor_boundary",
            "duration_ms": round((time.monotonic() - self.started_at) * 1000, 3),
            "rss_baseline_bytes": rss_baseline,
            "rss_peak_bytes": self._peaks["rss_bytes"],
            "rss_peak_delta_bytes": max(0, self._peaks["rss_bytes"] - rss_baseline),
            "cuda_allocated_baseline_bytes": cuda_baseline,
            "cuda_allocated_peak_bytes": self._peaks["cuda_allocated_bytes"],
            "cuda_allocated_peak_delta_bytes": max(
                0,
                self._peaks["cuda_allocated_bytes"] - cuda_baseline,
            ),
        }


# ★ A6：`torch._dynamo.config` 是 **thread-local**（torch ≥2.12；本仓实测 2.13.0 亦然 ——
#   源码 `torch/utils/_config_module.py:53/:394-395/:812` 明确 "User overrides are thread-local"，
#   且实测「主线程设 64 ⇒ 新线程读回默认 8」）。
#   主仓 `_apply_compile()` 在**加载线程**设置 `recompile_limit`/`cache_size_limit`，而推理（尤其经
#   starlette `run_in_threadpool` 的 API 路径）发生在**worker 线程** ⇒ 设置**读不到**、退回默认 8
#   ⇒ 长序列下过早触发 `recompile_limit` 而放弃编译（与 B1/B2 观测的长序列劣化一致）。
#   对策：在**每个进入模型访问的线程**里幂等地补设一次（每线程一次，开销可忽略）。
_compile_limit_tls = threading.local()


@contextlib.contextmanager
def _premark_hf_initialized():
    """★ BUG 修复（2026-09-19）：`from_pretrained` 期间**禁止** `_init_weights` 覆盖已装载权重。

    ## 症状（用户报障）
    `qwen-1_8b` + int4 加载 ⇒ `NotImplementedError: "normal_kernel_cuda"
    not implemented for 'Byte'`（HTTP 500），**崩在 `Loading weights 1%` 处**。

    ## 根因（transformers 5.17 源码 + 实测调用栈）
    `modeling_utils.py:2379` `self._init_weights(module)` ⇒ `:2401-2402` `smart_apply` ⇒
    **remote code** `modeling_qwen.py:666 _init_weights` ⇒ `p.data.normal_(...)`，
    而量化后 `p.data` 是 **`uint8`** ⇒ CUDA 无该 dtype 的 normal 内核 ⇒ 抛异常。
    触发条件：`:2361` 的「是否已初始化」判定**看不到** remote code 用**递归**
    `named_parameters()` 写入的标记（`:521-524` 记录的同一 5.x 缺陷），
    于是把**已装载**的模块误判为未初始化。

    ## 为什么必须「预标记」而不是「事后修」
    仓库已有的 `_verify_and_repair_loaded_weights`（A7/B7）在**加载完成后**才跑，
    而本缺陷在**加载过程中**就崩溃 ⇒ 实测 `guard_calls: []`（守卫根本没被调到）。
    transformers 自己在「加载后修复 missing keys」时也用同一标记（`:4766-4769`）⇒
    本函数与官方做法**一致**，只是**提前**到加载窗口内。

    ## 安全性
    量化加载时权重**已从 ckpt 装载**，本就不需要 `_init_weights`；真缺失的键仍会由
    `missing_keys` 与现有守卫暴露（不会静默留下随机权重）。

    ⚠️ 调用方须**收窄作用域**（仅 transformers≥5 + remote code + 量化）；
    本函数用 `try/finally` 严格恢复 `PreTrainedModel.initialize_weights`。
    """
    try:
        from transformers import PreTrainedModel
    except Exception as exc:  # noqa: BLE001
        logger.debug("预标记：无法 import PreTrainedModel，跳过（%s）", exc)
        yield
        return

    original = PreTrainedModel.initialize_weights

    def _patched(self, *args, **kwargs):  # noqa: ANN001
        # 与 transformers :4766-4769 一致的标记方式
        for module in self.modules():
            try:
                module._is_hf_initialized = True
            except Exception:  # noqa: BLE001
                pass
            for param in module.parameters(recurse=False):
                try:
                    param._is_hf_initialized = True
                except Exception:  # noqa: BLE001
                    pass
        return original(self, *args, **kwargs)

    PreTrainedModel.initialize_weights = _patched
    try:
        yield
    finally:
        PreTrainedModel.initialize_weights = original


def _is_transformers_5_or_newer() -> bool:
    """当前 `transformers` 主版本是否 ≥5（用于决定是否需要「权重覆盖」守卫）。"""
    try:
        import transformers
        major = str(getattr(transformers, "__version__", "0")).split(".")[0]
        return int(major) >= 5
    except Exception:  # noqa: BLE001
        return False


def _model_declares_remote_code(model_dir) -> bool:
    """该模型目录的 ``config.json`` 是否声明 ``auto_map`` —— 即**真的**依赖 remote code。

    ★ BUG 修复（2026-09-19，实测单变量对照）：**不能**用全局 ``TRUST_REMOTE_CODE`` 当判据。

    ``TRUST_REMOTE_CODE`` 是**用户授权开关**（主仓默认 ``True``），只表示「允许加载
    remote code」，**不代表当前这个模型真的需要它**。用它做判据会把**原生模型**一并
    纳入 ``_premark_hf_initialized`` 的作用域 ⇒ 跳过整个 ``from_pretrained`` 窗口的
    权重初始化 ⇒ **原生模型的 buffer（如 RoPE ``inv_freq``）不被初始化**。

    实测（`qwen3-0.6b` + int4，主仓引擎，仅切换这一个变量）：

    ==================  ==================  ==========================================
    变量                权重审计            输出
    ==================  ==================  ==========================================
    允许 premark        全 0（健康）        ``<think>`` 重复 12 次 + ``رو``/``ismatic`` 乱码
    **禁用 premark**    全 0（健康）        ``<think>\\n好的，用户让我用一句话自我介绍…`` 通顺
    ==================  ==================  ==========================================

    ⚠️ 注意审计查 NaN/Inf/零值**全为 0** —— 这类「值不合理但非 NaN」的损坏**逃过审计**，
    所以必须靠**输出对照**才能发现（这正是用户报的「其他模型回复脏 token」）。

    实测各模型 ``auto_map``：**只有** ``qwen-1_8b-chat`` 与 ``minicpm4-0.5b`` 声明了
    ``auto_map``；``qwen3-0.6b`` / ``qwen3-4b`` / ``qwen3-5-2b`` / ``qwenseek-2b`` /
    ``qwen3-5-9b`` / ``qwen3-vl-4b`` / ``deepseek-r1-*`` / ``distilqwen*`` / ``gemma4-*`` /
    ``qwen2.5-*`` **全部是原生架构**。
    """
    try:
        import json
        import os as _os

        cfg_path = _os.path.join(str(model_dir), "config.json")
        if not _os.path.isfile(cfg_path):
            return False
        with open(cfg_path, encoding="utf-8") as fp:
            cfg = json.load(fp)
        return bool(cfg.get("auto_map"))
    except Exception:  # noqa: BLE001 —— 读不到就按「非 remote code」保守处理
        return False


def _verify_and_repair_loaded_weights(model, model_path: str) -> Optional[Dict[str, Any]]:
    """★ A7/B7：transformers 5.x 下 remote-code 模型的「**权重被 `_init_weights` 覆盖**」守卫。

    ## 为什么需要（已实测定位，见 `local_docs/CORE-RELAY-XFRAME-02-a6-and-leak-*.json`）
    transformers ≥5 把权重初始化从「装载**之前**」搬到了「装载**之后**」，且
    `_initialize_weights` 的「未初始化」检查只看 **`recurse=False`** 的 params/buffers，
    而 remote code 的 `_init_weights` 写入用的是**递归** `named_parameters()` ⇒ 两者范围不一致；
    当某个模块**只直接持有 non-persistent buffer**（`_move_missing_keys_from_meta_to_device`
    会把它们换成 `torch.empty_like` 的**新对象、丢掉标志**）时，该模块会被**误判为未初始化**。
    若 remote code 的 `_init_weights` 又按**相对名**匹配（如
    `modeling_qwen.py` 的 `if name == "c_proj.weight": p.data.normal_(...)`），
    就会**覆盖已经正确装载的权重** —— 且 **`missing_keys` 仍报 0**（**静默**，最危险的一类）。

    Qwen-1.8B 实测：**24 层 `transformer.h.*.attn.c_proj.weight` 全部中招**
    （`ok=171 / bad=24`），导致输出错误 + 跨进程非确定；本函数重载后输出与 4.47.1 **逐位一致**。

    ## 做法
    逐键把 `model.state_dict()` 与 safetensors 原值比对（`torch.equal`），**不符的键就地重载**。
    只在「5.x + `TRUST_REMOTE_CODE`」时被调用（4.x 无此缺陷，省掉一次全区读取）。

    ## 纪律
    ① **只修不抛**：任何异常都只记日志（fail-loud），绝不让加载失败；
    ② **绝不静默**：发现不符必然 `logger.warning`（这正是上游缺陷的可见性补救）；
    ③ 返回统计字典（供上层日志/报告），无问题或不可用时返回 None。
    """
    if not model_path:
        return None
    try:
        import torch
        from safetensors import safe_open
    except Exception as exc:  # noqa: BLE001
        logger.debug("权重守卫：缺少 torch/safetensors，跳过（%s）", exc)
        return None

    try:
        state = model.state_dict()
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        by_shard: Dict[str, List[str]] = {}
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
            for key, shard in weight_map.items():
                if key in state:
                    by_shard.setdefault(shard, []).append(key)
        else:
            for name in sorted(n for n in os.listdir(model_path) if n.endswith(".safetensors")):
                by_shard[name] = [k for k in _iter_safetensors_keys(model_path)]

        if not by_shard:
            logger.debug("权重守卫：%s 无可用 safetensors 索引，跳过", model_path)
            return None

        checked = repaired = 0
        mismatched: List[str] = []
        with torch.no_grad():
            for shard, keys in by_shard.items():
                shard_path = os.path.join(model_path, shard)
                if not os.path.isfile(shard_path):
                    continue
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    available = set(handle.keys())
                    for key in keys:
                        if key not in available:
                            continue
                        target = state.get(key)
                        if target is None or not hasattr(target, "copy_"):
                            continue
                        # ⚠️ **跨设备**：CUDA 上 `model.state_dict()` 的张量在 GPU，而 safetensors
                        # 读到的是 CPU。必须**显式搬到 target 的设备**再比较/写入 —— 否则
                        # `torch.equal` 跨设备恒为 False，会把**全部**键误判为「不一致」并做过量重载
                        # （2026-09-19 实测：CUDA 上曾误报 195/195）。
                        if getattr(target, "is_meta", False):
                            # `device_map` 分片可能把某些键留成 meta 占位，无法就地对写 ⇒ 跳过。
                            continue
                        raw = handle.get_tensor(key)
                        checked += 1
                        try:
                            src = raw.to(device=target.device, dtype=target.dtype)
                            same = raw.shape == target.shape and bool(torch.equal(src, target))
                        except Exception:  # noqa: BLE001
                            same = False
                            src = raw
                        if not same:
                            target.copy_(src)
                            repaired += 1
                            if len(mismatched) < 12:
                                mismatched.append(key)

        if repaired:
            logger.warning(
                "⚠️ 权重守卫：检测到 %d/%d 个张量的加载结果与 safetensors **不一致**（transformers 5.x 的 "
                "「初始化覆盖已装载权重」缺陷）⇒ **已从 safetensors 重载修复**：%s%s",
                repaired, checked, ", ".join(mismatched),
                " 等" if repaired > len(mismatched) else "",
            )
        else:
            logger.info("权重守卫：已逐键校验 %d 个张量，**全部与 safetensors 一致**", checked)
        return {"checked": checked, "repaired": repaired, "mismatched_head": mismatched}
    except Exception as exc:  # noqa: BLE001 - 守卫绝不使加载失败
        logger.warning("权重守卫执行失败（已忽略，继续使用原加载结果）: %s", exc)
        return None


def _ensure_compile_limits_in_current_thread() -> None:
    """★ A6：把 `recompile_limit`/`cache_size_limit` 在当前线程内**幂等**补设到配置值。

    `torch._dynamo.config` 是 thread-local ⇒ 仅在 `_apply_compile()`（加载线程）里设是不够的。
    本函数对每个线程记住「开关 + 目标值」状态；配置开关或目标值变化后会重新应用。
    只在「当前值小于配置值」时才写，避免覆盖用户显式调大的值。
    """
    compile_enabled = bool(globals().get("USE_COMPILE"))
    limit = globals().get("COMPILE_RECOMPILE_LIMIT")
    if not limit:
        return
    want = int(limit)
    state = (compile_enabled, want)
    if getattr(_compile_limit_tls, "state", None) == state:
        return
    if not compile_enabled:
        _compile_limit_tls.state = state
        return
    try:
        import torch._dynamo as _dynamo
        if int(_dynamo.config.recompile_limit or 0) < want:
            _dynamo.config.recompile_limit = want
        current_cache = getattr(_dynamo.config, "cache_size_limit", None)
        if current_cache is not None and int(current_cache or 0) < want:
            _dynamo.config.cache_size_limit = want
        _compile_limit_tls.state = state
        logger.debug("A6：已在本线程内补设 recompile_limit/cache_size_limit = %s", want)
    except Exception as exc:  # noqa: BLE001 - 补设失败不应影响前向
        logger.debug("A6：线程内补设 recompile_limit 失败（忽略）: %s", exc)


def _serialized_model_access(method):
    """Serialize model mutation and inference against the manager RLock."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        _ensure_compile_limits_in_current_thread()  # ★ A6
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


def _serialized_model_stream(method):
    """Keep the model lock for the complete lifetime of a stream."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        _ensure_compile_limits_in_current_thread()  # ★ A6
        with self._lock:
            yield from method(self, *args, **kwargs)
    return wrapper


def _resolve_pytorch_flash_attn() -> bool:
    """PyTorch 侧 flash-attn 开关：读同一张设备映射，**默认 False**（=接线前行为）。

    与 llama.cpp 的两条路径（绑定、`llama-server` 子进程）共用
    `src/inference_switches.py` 的显式映射；只有显式覆盖
    （`QLH_INFER_FLASH_ATTN=1`）才打开。映射或画像不可用时保持 False，
    即本函数引入前的硬编码值。
    """
    try:
        from inference_switches import resolve_from_device_profile

        # KV quantization is a llama.cpp option; its Flash dependency cannot
        # enable a PyTorch kernel. Only the PyTorch Flash preference applies.
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("QLH_INFER_KV_CACHE", "QLH_INFER_THREADS")}
        return bool(resolve_from_device_profile(env=environment,
                    execution_path="pytorch_remote_qwen").flash_attn)
    except Exception as exc:  # pragma: no cover - 依赖缺失时保持既有行为
        logger.warning("推理开关不可用，PyTorch flash-attn 保持关闭: %s", exc)
        return False


CHATML_STOP_SEQUENCES = [
    "<|im_end|>",
    "<|im_start|>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|endoftext|>",
    "</s>",
]


class ModelManager:
    """
    模型管理器：双引擎架构，自动选择最优推理后端。

    引擎:
      - PyTorch: CUDA + bitsandbytes INT4/INT8/FP16 量化（主节点 / 独显设备）
      - llama.cpp: GGUF Q4_K_M 等量化（边缘 / 集显 / CPU-only 设备）

    量化说明:
        bitsandbytes 采用"加载时量化"——权重在磁盘保持 FP16，
        加载到 GPU 时由 BitsAndBytesConfig 实时转换为 INT4/INT8。
        切换量化模式只需修改 config.QUANT_TYPE 并重新加载。
    """

    def __init__(self):
        # PyTorch 引擎
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.quant_type: Optional[str] = None
        self.layer_range: Optional[Tuple[int, int]] = None  # (start, end) 当前加载的层范围；None=完整模型
        self._layer_has_embedding: bool = True
        self._layer_has_lm_head: bool = True
        self._model_layers: int = 0          # 当前加载的层数（range 或 full）
        self._total_model_layers: int = 0    # 完整模型的总层数（加载时记录，load_layer_range 不覆盖）
        self._layer_architecture: str = ""   # "qwen" | "qwen2"，用于分层前向契约
        self._layer_load_metrics: Optional[Dict[str, Any]] = None
        self._layer_offload = None
        self._pipeline_descriptor: Optional[Dict[str, Any]] = None
        self._pipeline_distributed_only: bool = False
        #: torch.compile 后的内部 transformer（2026-09-18）。**不替换 self.model**，
        #: 否则 self.model.model / self.model.transformer 的结构访问会失效
        #: （forward_layers 与 _count_transformer_layers 都依赖它们）。
        self._compiled_transformer: Optional[nn.Module] = None
        # A4：编译版「层循环」（仅 forward_layers 使用；见 USE_MONOLITHIC_FORWARD）
        self._compiled_layer_loop: Optional[nn.Module] = None

        # llama.cpp 引擎（延迟导入 + 延迟加载）
        self._llama_engine = None   # LlamaCppEngine 实例
        # TP 孤岛引擎（延迟导入；OpenAI 兼容端点整请求转发）
        self._island_engine = None  # IslandEngine 实例
        self._engine_type: str = ""  # "pytorch" | "llama_cpp" | "island"

        # P3: 多模型支持 — 当前活跃的模型 ID
        # ★ 2026-09-19：默认模型**按设备画像**选择（边缘/轻薄本 <1B，PC ~2B）；
        #   分档与选型实测依据见 `model_config.DEFAULT_MODEL_BY_TIER`。
        self._active_model_id: str = mc.get_profile_default_model_id()
        self._previous_engine_type: str = ""      # 用于 rollback
        self._previous_quant_type: Optional[str] = None

        # 模型路径记录（供 SHA256 一致性校验等用途）
        self._model_path: Optional[str] = None
        # 主节点进入分层模式后，保留完整模型的加载参数供本地回退恢复。
        self._full_model_path: Optional[str] = None
        self._full_model_quant_type: Optional[str] = None
        self._load_fingerprint: Optional[Dict[str, Any]] = None
        self._load_fingerprint_sha256: str = ""

        # 并发保护锁 — 防止推理与模型切换之间的数据竞争
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载（兼容 PyTorch / llama.cpp / 孤岛三引擎）。"""
        if self._engine_type == "llama_cpp":
            return self._llama_engine is not None
        if self._engine_type == "island":
            return self._island_engine is not None
        return self.model is not None

    # ================================================================
    # 引擎选择
    # ================================================================

    @staticmethod
    def select_engine(profile: dict = None) -> str:
        """
        根据硬件环境和配置选择推理引擎。

        决策优先级:
          1. config.INFERENCE_ENGINE == "island" 显式指定孤岛引擎
          2. QLH_ISLAND_ENABLED + QLH_ISLAND_BASE_URL → "island"
             （孤岛网关节点整机作为孤岛前端，覆盖本地引擎默认值）
          3. config.INFERENCE_ENGINE 显式指定（"pytorch" / "llama_cpp"）
          4. "auto" → 检测 CUDA 可用性
             - CUDA 可用 → "pytorch"
             - CUDA 不可用 → "llama_cpp"

        Args:
            profile: 设备画像 dict（可选，用于更精确的判断）

        Returns:
            "pytorch" / "llama_cpp" / "island"
        """
        # 动态读取 config（api_server 会在运行时改写 config 模块属性）
        import config as _cfg

        runtime_requested = getattr(_cfg, "INFERENCE_ENGINE", INFERENCE_ENGINE)
        requested = (
            INFERENCE_ENGINE
            if INFERENCE_ENGINE != _IMPORTED_INFERENCE_ENGINE
            else runtime_requested
        )

        # ⚠️ 不要用 `torch.cuda.is_available()`：本仓 `.venv-test` 装的是 **CPU-only
        #   torch（2.13.0+cpu）**，于是明明有 RTX 4060 也返回 False ⇒ `auto` 会错选
        #   llama_cpp（本该按双档策略选 pytorch）。改用硬件级探测（nvidia-smi 优先），
        #   与 `inference_path_selector.detect_cuda_available()` 同源。
        from inference_path_selector import detect_cuda_available

        selected = select_backend(
            profile,
            requested=requested,
            cuda_available=detect_cuda_available(),
            island_enabled=bool(getattr(_cfg, "ISLAND_ENABLED", False)),
            island_base_url=str(getattr(_cfg, "ISLAND_BASE_URL", "") or ""),
        )
        logger.info("Koakuma backend: %s", selected)
        return selected

    # ================================================================
    # 量化配置工厂（PyTorch 专用）
    # ================================================================

    @staticmethod
    def _get_bnb_config(quant_type: str) -> Optional[BitsAndBytesConfig]:
        """
        获取 bitsandbytes 量化配置。

        Args:
            quant_type: "fp16" | "int8" | "int4"

        Returns:
            BitsAndBytesConfig 或 None（fp16 模式）
        """
        if quant_type == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant_type == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=10.0,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
            )
        else:
            return None  # fp16 — 不使用量化

    # ================================================================
    # 模型加载（双引擎入口）
    # ================================================================

    def configure_catalog(self, catalog_prepare, *, model_id: str = "") -> None:
        """Inject the catalog's prepare validator; no global registry lookup."""
        self._catalog_prepare = catalog_prepare

    def _verify_catalog_artifact(self, model_id, model_path, rows=None, *, expected=None):
        from qlh_core.adapters.artifacts import verify_registered_artifact

        prepare = getattr(self, "_catalog_prepare", None)
        resolved = prepare(model_id, model_path) if prepare is not None else None
        if resolved is None:
            resolved = verify_registered_artifact(model_id, model_path, rows=rows)
        if expected is not None and resolved != expected:
            from qlh_core.contracts.catalog import CatalogError
            raise CatalogError("artifact_revision_mismatch", "source binding changed since pipeline preparation")
        self._resolved_artifact = resolved
        return resolved

    @staticmethod
    def _normalize_load_path(path: str | None) -> str:
        if not path:
            return ""
        return os.path.normcase(
            os.path.realpath(os.path.abspath(os.path.expanduser(path)))
        )

    @staticmethod
    def _read_sha256_sidecar(path: str) -> str:
        candidates = []
        if os.path.isdir(path):
            candidates.append(os.path.join(path, "model.sha256"))
        elif path:
            candidates.extend((path + ".sha256", os.path.splitext(path)[0] + ".sha256"))
        for candidate in dict.fromkeys(candidates):
            try:
                if os.path.getsize(candidate) > 4096:
                    continue
                with open(candidate, "r", encoding="utf-8") as handle:
                    digest = handle.read(4097).strip().split()[0].lower()
            except (OSError, IndexError, UnicodeDecodeError):
                continue
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                return digest
        return ""

    @staticmethod
    def _registry_load_identity(
        model_id: str,
        db_experimental_models: list[dict] | None,
    ) -> Dict[str, str]:
        from qlh_core.contracts.catalog import registry_load_identity

        return registry_load_identity(model_id, db_experimental_models)

    @classmethod
    def _artifact_load_identity(
        cls,
        path: str | None,
        *,
        model_id: str,
        db_experimental_models: list[dict] | None,
    ) -> Dict[str, Any]:
        normalized = cls._normalize_load_path(path)
        identity: Dict[str, Any] = {
            "path": normalized,
            "registry": cls._registry_load_identity(model_id, db_experimental_models),
        }
        try:
            stat_result = os.stat(normalized)
        except OSError:
            identity["kind"] = "missing" if normalized else "none"
            return identity
        identity.update({
            "kind": "directory" if os.path.isdir(normalized) else "file",
            "size_bytes": int(stat_result.st_size),
            "mtime_ns": int(stat_result.st_mtime_ns),
        })
        from qlh_core.adapters.artifacts import artifact_load_identity

        identity.update(artifact_load_identity(
            normalized, model_id=model_id, rows=db_experimental_models,
        ))
        sidecar_sha256 = cls._read_sha256_sidecar(normalized)
        if sidecar_sha256:
            identity["sidecar_sha256"] = sidecar_sha256
        return identity

    @staticmethod
    def _profile_load_identity(profile: dict | None) -> Dict[str, Any]:
        value = profile if isinstance(profile, dict) else {}
        gpu = value.get("gpu") if isinstance(value.get("gpu"), dict) else {}
        cpu = value.get("cpu") if isinstance(value.get("cpu"), dict) else {}
        gpus = value.get("gpus") if isinstance(value.get("gpus"), list) else []

        def stable_int(raw: Any, default: int) -> int:
            try:
                return int(raw)
            except (TypeError, ValueError, OverflowError):
                return default

        def stable_float(raw: Any, default: float) -> float:
            try:
                result = float(raw)
            except (TypeError, ValueError, OverflowError):
                return default
            return result if result == result and abs(result) != float("inf") else default

        gpu_layers = value.get("gpu_layers")
        if gpu_layers is not None and not isinstance(gpu_layers, (bool, int, float, str)):
            gpu_layers = str(gpu_layers)
        return {
            "tier": str(value.get("tier", "laptop") or "laptop"),
            "selected_gpu_index": stable_int(value.get("selected_gpu_index", 0) or 0, 0),
            "gpu": {
                "index": stable_int(gpu.get("index", 0) or 0, 0),
                "cuda_available": bool(gpu.get("cuda_available", False)),
                "vram_total_gb": stable_float(gpu.get("vram_total_gb", 0) or 0, 0.0),
                "is_integrated": bool(gpu.get("is_integrated", False)),
                "gpu_type": str(gpu.get("gpu_type", "") or ""),
            },
            "any_cuda_gpu": any(
                isinstance(item, dict) and bool(item.get("cuda_available", False))
                for item in gpus
            ),
            "cpu_physical_cores": stable_int(cpu.get("physical_cores", 4) or 4, 4),
            "requested_device": str(value.get("device", "") or ""),
            "offload_profile": str(value.get("offload_profile", "") or ""),
            "gpu_layers": gpu_layers,
        }

    def _resolve_model_load_request(
        self,
        *,
        model_path: str | None,
        quant_type: str | None,
        profile: dict | None,
        model_id: str | None,
        engine: str | None,
        db_experimental_models: list[dict] | None,
        require_existing: bool,
        resolution: ModelLoadResolution | None = None,
    ) -> Dict[str, Any]:
        import config as runtime_config

        resolved_id = (
            resolution.model_id
            if resolution is not None
            else model_id or mc.get_profile_default_model_id()
        )
        cfg = mc.get_model_config(resolved_id, db_experimental_models)
        if resolution is not None:
            return {
                "model_id": resolved_id,
                "config": cfg,
                "engine": resolution.engine,
                "path": resolution.model_path or "",
                "requested_quantization": resolution.requested_quant.casefold(),
                "effective_quantization": resolution.quant_type,
                "runtime": resolution.runtime,
                "layer_range_mode": resolution.layer_range_mode,
                "reason_code": resolution.reason_code,
            }
        requested_quant = str(quant_type or QUANT_TYPE)
        cuda_ok = effective_pytorch_cuda_available(
            system_cuda_available=bool(torch.cuda.is_available()),
            profile=profile,
        )

        if model_path:
            explicit = os.path.abspath(model_path)
            is_gguf = explicit.lower().endswith(".gguf") and (
                os.path.isfile(explicit) or not require_existing
            )
            is_safetensors = not explicit.lower().endswith(".gguf") and (
                os.path.isdir(explicit) or not require_existing
            )
            has_gguf = is_gguf
            has_safetensors = is_safetensors
            gguf_path = explicit if is_gguf else None
            safetensors_path = explicit if is_safetensors else None
            registered = bool(cfg is not None or model_path)
        elif cfg is not None:
            status = mc.get_model_file_status(cfg)
            has_gguf = bool(status["has_gguf"] or (not require_existing and cfg.gguf_path))
            has_safetensors = bool(
                status["has_safetensors"] or (not require_existing and cfg.model_path)
            )
            gguf_path = mc.resolve_model_path(cfg.gguf_path) if has_gguf else None
            safetensors_path = (
                mc.resolve_model_path(cfg.model_path) if has_safetensors else None
            )
            registered = True
        else:
            has_gguf = False
            has_safetensors = False
            gguf_path = None
            safetensors_path = None
            registered = False

        selected_preference = (
            self.select_engine(profile)
            if not engine or str(engine).strip().lower() == "auto"
            else str(engine)
        )
        facts = ModelLoadFacts(
            model_id=resolved_id,
            model_name=getattr(cfg, "name", resolved_id),
            registered=registered,
            has_safetensors=has_safetensors,
            has_gguf=has_gguf,
            safetensors_path=safetensors_path,
            gguf_path=gguf_path,
            preferred_engine=selected_preference or preferred_engine_for_artifacts(
                has_safetensors=has_safetensors,
                has_gguf=has_gguf,
                cuda_available=cuda_ok,
            ),
            cuda_available=cuda_ok,
            island_enabled=bool(getattr(runtime_config, "ISLAND_ENABLED", False)),
            island_base_url=str(getattr(runtime_config, "ISLAND_BASE_URL", "") or ""),
        )
        resolution = resolve_model_load(
            facts,
            requested_engine=engine or "auto",
            requested_quant=requested_quant,
            operation=OP_FULL_MODEL,
        )
        return {
            "model_id": resolution.model_id,
            "config": cfg,
            "engine": resolution.engine,
            "path": resolution.model_path or "",
            "requested_quantization": resolution.requested_quant.casefold(),
            "effective_quantization": resolution.quant_type,
            "runtime": resolution.runtime,
            "layer_range_mode": resolution.layer_range_mode,
            "reason_code": resolution.reason_code,
        }

    def _build_load_fingerprint(
        self,
        request: Dict[str, Any],
        *,
        profile: dict | None,
        db_experimental_models: list[dict] | None,
    ) -> tuple[Dict[str, Any], str]:
        cfg = request.get("config")
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "model_id": request["model_id"],
            "engine": request["engine"],
            "requested_quantization": request["requested_quantization"],
            "effective_quantization": request.get("effective_quantization", ""),
            "artifact": self._artifact_load_identity(
                request.get("path"),
                model_id=request["model_id"],
                db_experimental_models=db_experimental_models,
            ),
            "registry_model_type": str(getattr(cfg, "model_type", "") or ""),
            "execution": {
                "profile": self._profile_load_identity(profile),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "configured_device": str(DEVICE),
                "use_compile": bool(USE_COMPILE),
                "trust_remote_code": bool(TRUST_REMOTE_CODE),
            },
        }
        canonical = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return payload, hashlib.sha256(canonical).hexdigest()

    def _record_load_fingerprint(
        self,
        request: Dict[str, Any],
        *,
        profile: dict | None,
        db_experimental_models: list[dict] | None,
    ) -> None:
        if not request.get("path") and self._model_path:
            request = dict(request)
            request["path"] = self._model_path
        payload, digest = self._build_load_fingerprint(
            request,
            profile=profile,
            db_experimental_models=db_experimental_models,
        )
        self._load_fingerprint = payload
        self._load_fingerprint_sha256 = digest

    @_serialized_model_access
    def load_model(
        self,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
        model_id: str = None,
        engine: str = None,
        db_experimental_models: list[dict] = None,
        resolution: ModelLoadResolution | None = None,
    ) -> None:
        """
        加载模型，自适应选择推理引擎。

        引擎选择:
          - PyTorch: 加载 Safetensors 格式 → 量化 → 融合
          - llama.cpp: 加载 GGUF 格式 → CPU 多线程推理

        Args:
            model_path: 本地模型路径，默认使用 config.MODEL_PATH
            quant_type: PyTorch 量化精度 "fp16" | "int8" | "int4"
            profile: 设备画像 dict
            model_id: 模型唯一标识（P3多模型支持）。
                      若提供且 model_path 未指定，从 model_config 查找路径。
            db_experimental_models: DB 注册的实验模型列表（P3修复：支持 DB 模型查找）。
        """
        request = self._resolve_model_load_request(
            model_path=model_path,
            quant_type=quant_type,
            profile=profile,
            model_id=model_id,
            engine=engine,
            db_experimental_models=db_experimental_models,
            require_existing=True,
            resolution=resolution,
        )
        resolved_path = request["path"] or None
        resolved_id = request["model_id"]
        resolved_engine = request["engine"]
        if resolved_engine != "island":
            self._verify_catalog_artifact(resolved_id, resolved_path or "", db_experimental_models)

        # ---- TP 孤岛引擎：无本地模型文件，"加载" = 健康检查 + 解析后端模型名 ----
        # 孤岛模型不进本地注册表（无落盘 artifact），跳过注册表/文件校验。
        if resolved_engine == "island":
            self._engine_type = "island"
            self._load_island(profile)
            self._active_model_id = resolved_id
            self._previous_engine_type = self._engine_type
            self._previous_quant_type = self.quant_type
            # 孤岛节点没有可回退的本地完整模型
            self._full_model_path = None
            self._full_model_quant_type = None
            self._record_load_fingerprint(
                request,
                profile=profile,
                db_experimental_models=db_experimental_models,
            )
            return

        self._engine_type = resolved_engine

        if self._engine_type == "llama_cpp":
            if resolved_id == "gemma4-native":
                self._load_gemma4_native(resolved_path, profile)
            else:
                self._load_llama_cpp(resolved_path, profile)
        else:
            self._load_pytorch(
                resolved_path,
                request["effective_quantization"],
                profile,
            )

        # 记录活跃模型 ID
        self._active_model_id = resolved_id
        self._previous_engine_type = self._engine_type
        self._previous_quant_type = self.quant_type
        self._full_model_path = self._model_path
        self._full_model_quant_type = self.quant_type
        self._pipeline_descriptor = None
        self._pipeline_distributed_only = False
        self._record_load_fingerprint(
            request,
            profile=profile,
            db_experimental_models=db_experimental_models,
        )

    @_serialized_model_access
    def unload_model(self) -> None:
        """
        卸载当前加载的模型，释放 GPU 显存和系统内存。

        同时清理 PyTorch 和 llama.cpp 引擎状态。
        调用后 is_loaded 返回 False，可安全加载新模型。
        """
        logger.info(f"卸载模型: {self._active_model_id} (引擎={self._engine_type})")
        # ★ 2026-10-24（#125）：卸载会清掉 `_pipeline_descriptor` /
        #   `_pipeline_distributed_only`，而它们正是"已 prepared"的唯一判据
        #   （`is_pipeline_prepared` = `_pipeline_distributed_only and _pipeline_descriptor`）。
        #   一旦被清，下一轮供给必然重新加载段并重跑 torch.compile（实测 138 s，
        #   内容与上一轮完全相同）。此前不知道供给链路里是谁触发的卸载，这里在
        #   **仅当确有 pipeline 状态被丢弃时**记录调用栈，便于定位并消除无谓卸载。
        if self._pipeline_descriptor is not None or self._pipeline_distributed_only:
            try:
                import traceback
                stack = " <- ".join(
                    line.strip() for line in traceback.format_stack()[-6:-1]
                ).replace("\n", " ")
                logger.info(
                    "event=pipeline_descriptor_cleared_by_unload active_model=%s stack=%s",
                    self._active_model_id, stack[:400],
                )
            except Exception:
                pass

        # --- llama.cpp 引擎清理 ---
        # Release native owners before clearing their state projection. A
        # failed close must retain the engine and remain retryable by callers.
        if self._llama_engine is not None:
            close = getattr(self._llama_engine, "close", None) or getattr(self._llama_engine, "unload", None)
            if callable(close):
                close()
            self._llama_engine = None

        # Compiled wrappers own the original parameters independently of self.model.
        # Drop every owner before GC/empty_cache, including an already-unloaded projection.
        self._compiled_transformer = None
        self._compiled_layer_loop = None
        # --- PyTorch 引擎清理 ---
        if self.model is not None:
            self.model = None

        if self.tokenizer is not None:
            self.tokenizer = None

        self.quant_type = None
        self.layer_range = None
        self._layer_has_embedding = True
        self._layer_has_lm_head = True
        self._model_layers = 0
        self._total_model_layers = 0
        self._layer_architecture = ""
        self._layer_load_metrics = None
        self._layer_offload = None
        self._pipeline_descriptor = None
        self._pipeline_distributed_only = False
        self._full_model_path = None
        self._full_model_quant_type = None
        self._load_fingerprint = None
        self._load_fingerprint_sha256 = ""

        # --- 孤岛引擎清理（断开 HTTP 客户端）---
        if self._island_engine is not None:
            try:
                self._island_engine.unload()
            except Exception:
                pass
            self._island_engine = None

        # --- GPU 显存回收 ---
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

        # 重置引擎类型
        self._engine_type = ""
        self._active_model_id = ""  # P3修复: 卸载后清空活跃模型ID
        logger.info("模型已卸载，显存已释放")

    @_serialized_model_access
    def prepare_pipeline_model(
        self,
        model_id: str,
        model_path: str,
        quant_type: str = None,
        layer_range: tuple[int, int] | None = None,
        model_sha256: str | None = None,
    ) -> dict:
        """Prepare a distributed model without materializing any weight tensor.

        This is an explicit distributed-only state. A later pipeline request
        may load the master's assigned range, but full-model fallback remains
        disabled until the caller performs a normal model load.
        """
        from model_sync import compute_model_sha256
        from pipeline_model_descriptor import inspect_pipeline_model

        resolved_path = os.path.abspath(model_path or "")
        self._verify_catalog_artifact(model_id, resolved_path)
        # Assignment manifests contain only the worker's layer range. Validate
        # that range instead of treating the filtered directory as complete.
        descriptor = inspect_pipeline_model(
            resolved_path,
            model_id=model_id,
            layer_range=layer_range,
        )
        if not descriptor.get("pipeline_runtime_supported", False):
            raise RuntimeError(
                descriptor.get("runtime_block_reason")
                or "该模型架构尚未实现流水线执行 adapter"
            )
        if not model_sha256:
            model_sha256 = compute_model_sha256(resolved_path)
        if not model_sha256:
            raise RuntimeError("无法计算流水线模型摘要")
        descriptor["model_sha256"] = model_sha256

        # Validate the complete artifact before replacing the current runtime.
        if self.is_loaded or self._pipeline_descriptor is not None:
            self.unload_model()
        self.model = None
        self.tokenizer = None
        self._engine_type = "pytorch"
        self._active_model_id = model_id
        self._model_path = resolved_path
        self._full_model_path = resolved_path
        self.quant_type = quant_type or QUANT_TYPE
        self._full_model_quant_type = self.quant_type
        self._total_model_layers = int(descriptor["total_layers"])
        self._model_layers = 0
        self._pipeline_descriptor = dict(descriptor)
        self._pipeline_distributed_only = True
        self._load_fingerprint = None
        self._load_fingerprint_sha256 = ""
        logger.info(
            "流水线模型元数据已准备: model=%s type=%s layers=%s "
            "runtime_supported=%s inspection=%s",
            model_id,
            descriptor["model_type"],
            descriptor["total_layers"],
            descriptor["pipeline_runtime_supported"],
            descriptor["inspection_mode"],
        )
        return dict(descriptor)

    @property
    def is_pipeline_prepared(self) -> bool:
        """Whether an explicit distributed-only artifact is active."""
        return bool(self._pipeline_distributed_only and self._pipeline_descriptor)

    def get_pipeline_descriptor(self) -> dict:
        """Return cached metadata, or inspect the active PyTorch artifact."""
        if self._pipeline_descriptor is not None:
            return dict(self._pipeline_descriptor)
        if self._engine_type != "pytorch":
            return {}
        model_path = self._full_model_path or self._model_path or ""
        if not model_path or not os.path.isdir(model_path):
            return {}
        from model_sync import compute_model_sha256
        from pipeline_model_descriptor import inspect_pipeline_model

        descriptor = inspect_pipeline_model(
            model_path,
            model_id=self._active_model_id,
        )
        descriptor["model_sha256"] = compute_model_sha256(model_path)
        self._pipeline_descriptor = dict(descriptor)
        return dict(descriptor)

    @_serialized_model_access
    def prepare_pipeline_tokenizer(self):
        """Load tokenizer metadata for a control-only pipeline coordinator."""
        if not self.is_pipeline_prepared:
            raise RuntimeError("当前没有已准备的 distributed-only 流水线模型")
        if self.tokenizer is None:
            model_path = self._full_model_path or self._model_path or ""
            if not model_path or not os.path.isdir(model_path):
                raise FileNotFoundError("流水线 tokenizer 的本地模型目录不存在")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=TRUST_REMOTE_CODE,
                local_files_only=True,
            )
        return self.tokenizer

    @_serialized_model_access
    def abort_pipeline_materialization(self) -> None:
        """Release a committed segment while preserving distributed metadata."""
        if not self._pipeline_descriptor or not self._pipeline_distributed_only:
            return
        self._compiled_transformer = None
        self._compiled_layer_loop = None
        self.model = None
        self.tokenizer = None
        self.layer_range = None
        self._layer_has_embedding = True
        self._layer_has_lm_head = True
        self._model_layers = 0
        self._layer_architecture = ""
        self._layer_load_metrics = None
        self._layer_offload = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def switch_model(
        self,
        model_id: str,
        quant_type: str = None,
        profile: dict = None,
        engine: str = None,
        model_path: str = None,
        db_experimental_models: list[dict] = None,
        resolution: ModelLoadResolution | None = None,
    ) -> dict:
        """
        切换到另一个模型（P3 多模型支持）。

        流程:
          1. 保存当前模型信息（用于失败时的 best-effort rollback）
          2. 调用 unload_model() 释放显存
          3. 查找新模型配置
          4. 调用 load_model() 加载新模型
          5. 失败时尝试回滚到上一个模型

        Args:
            model_id: 目标模型唯一标识
            quant_type: 量化精度（默认使用当前精度或 QUANT_TYPE）
            profile: 设备画像
            engine: 推理引擎 "pytorch" | "llama_cpp" | "island" | "auto" (None=auto)
            db_experimental_models: DB 注册的实验模型列表（P3修复：支持 DB 模型查找）。

        Returns:
            {"success": bool, "model_id": str, "model_name": str, "error": str | None}
        """
        with self._lock:
            # 保存回滚信息
            rollback_model_id = self._active_model_id
            rollback_model_name = self._active_model_id  # 将在下面尝试获取可读名称
            rollback_engine = self._previous_engine_type or self._engine_type
            rollback_quant = self._previous_quant_type or quant_type or QUANT_TYPE
            rollback_path = self._model_path
            had_model = self.is_loaded
            had_pipeline_preparation = self.is_pipeline_prepared

            if had_model:
                # 尝试获取回滚模型的可读名称
                try:
                    rollback_cfg = mc.get_model_config(rollback_model_id, db_experimental_models)
                    if rollback_cfg:
                        rollback_model_name = rollback_cfg.name
                except Exception:
                    pass

            logger.info(
                f"切换模型: {rollback_model_id} -> {model_id} "
                f"(quant={quant_type}, engine={engine or 'auto'}, "
                f"profile_tier={profile.get('tier', '?') if profile else '?'})"
            )

            # KIP-13: reuse requires an exact immutable load-fingerprint match.
            # The public result exposes only the digest; local paths stay private.
            target_request = None
            target_fingerprint = ""
            try:
                target_request = self._resolve_model_load_request(
                    model_path=model_path,
                    quant_type=quant_type,
                    profile=profile,
                    model_id=model_id,
                    engine=engine,
                    db_experimental_models=db_experimental_models,
                    require_existing=False,
                    resolution=resolution,
                )
                _, target_fingerprint = self._build_load_fingerprint(
                    target_request,
                    profile=profile,
                    db_experimental_models=db_experimental_models,
                )
            except (OSError, TypeError, ValueError):
                # The normal load path below returns the existing structured error.
                target_request = None
                target_fingerprint = ""

            # Reject an unknown registry target before releasing the active
            # runtime.  Previously this case returned from the load block only
            # after unload_model(), which silently discarded a healthy model
            # without attempting rollback.
            if (
                target_request is None
                and not model_path
                and mc.get_model_config(model_id, db_experimental_models) is None
            ):
                return {
                    "success": False,
                    "model_id": rollback_model_id if had_model else model_id,
                    "requested_model_id": model_id,
                    "model_name": rollback_model_name if had_model else model_id,
                    "error_code": "MODEL_NOT_REGISTERED",
                    "active_model_preserved": bool(had_model),
                    "error": f"模型 '{model_id}' 未在注册表中找到。请先注册或下载模型文件。",
                }

            if (
                had_model
                and not had_pipeline_preparation
                and self.layer_range is None
                and target_request is not None
                and target_request["engine"] != "island"
                and bool(self._load_fingerprint_sha256)
                and self._load_fingerprint_sha256 == target_fingerprint
            ):
                requested_quant = target_request["requested_quantization"]
                logger.info(
                    f"P6 同模型切换短路: {model_id} 已加载（quant={requested_quant}），直接复用"
                )
                cfg = target_request.get("config")
                return {
                    "success": True,
                    "model_id": self._active_model_id,
                    "model_name": cfg.name if cfg else model_id,
                    "error": None,
                    "reused": True,
                    "load_fingerprint": f"sha256:{target_fingerprint}",
                }

            # 步骤 1: 卸载当前模型
            if had_model or had_pipeline_preparation:
                try:
                    self.unload_model()
                except Exception as e:
                    logger.warning(f"卸载当前模型时出现异常（继续切换）: {e}")

            # 步骤 2: 加载新模型
            try:
                cfg = mc.get_model_config(model_id, db_experimental_models)
                self.load_model(model_id=model_id, model_path=model_path,
                                quant_type=quant_type, profile=profile,
                                engine=engine,
                                 db_experimental_models=db_experimental_models,
                                 resolution=resolution)
                return {
                    "success": True,
                    "model_id": self._active_model_id,
                    "model_name": cfg.name if cfg else model_id,
                    "error": None,
                    "load_fingerprint": self.load_fingerprint,
                }
            except Exception as e:
                logger.error(f"加载模型 '{model_id}' 失败: {e}")

                # 步骤 3: best-effort 回滚（含同模型重载失败的情况）
                if had_model and rollback_model_id:
                    logger.info(f"尝试回滚到上一个模型: {rollback_model_id}")
                    try:
                        self.unload_model()
                        self.load_model(
                            model_id=rollback_model_id,
                            model_path=rollback_path,
                            quant_type=rollback_quant,
                            profile=profile,
                            engine=rollback_engine if rollback_engine else None,
                            db_experimental_models=db_experimental_models,
                        )
                        return {
                            "success": False,
                            "model_id": rollback_model_id,
                            "model_name": rollback_model_name,
                            "error_code": "MODEL_LOAD_FAILED_ROLLED_BACK",
                            "error": f"模型 '{model_id}' 加载失败: {e}。已回滚到 '{rollback_model_name}'。",
                        }
                    except Exception as rollback_err:
                        logger.error(f"回滚也失败: {rollback_err}")
                        return {
                            "success": False,
                            "model_id": None,
                            "model_name": "",
                            "error_code": "MODEL_LOAD_AND_ROLLBACK_FAILED",
                            "error": f"模型 '{model_id}' 加载失败: {e}。回滚也失败: {rollback_err}。",
                        }

                return {
                    "success": False,
                    "model_id": None,
                    "model_name": "",
                    "error_code": "MODEL_LOAD_FAILED",
                    "error": f"模型 '{model_id}' 加载失败: {e}",
                }

    @property
    def active_model_id(self) -> str:
        """当前活跃的模型 ID。"""
        return self._active_model_id

    @property
    def load_fingerprint(self) -> str:
        """Return the public load identity without exposing local artifact paths."""
        if not self._load_fingerprint_sha256:
            return ""
        return f"sha256:{self._load_fingerprint_sha256}"

    @_serialized_model_access
    def load_layer_range(
        self,
        start_layer: int = 0,
        end_layer: int = 24,
        has_embedding: bool = True,
        has_lm_head: bool = True,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
        total_layers: int = None,
        model_id: str = None,
        load_spec=None,
    ) -> None:
        """
        加载模型的指定层范围（分布式流水线节点专用）。

        从 safetensors 按 key 仅物化 [start_layer, end_layer) 的
        Transformer 层，并按需物化 Embedding 和 LM Head。完整模型权重
        不得在此路径中物化；加载后的裁剪只作为防御性结构收缩。

        Args:
            start_layer: 起始层编号（0-based，含）
            end_layer: 结束层编号（0-based，不含）
            has_embedding: 是否保留 Token Embedding（首节点为 True）
            has_lm_head: 是否保留 LM Head 输出层（末节点为 True）
            model_path: 模型路径，默认使用 config.MODEL_PATH
            quant_type: 量化精度，默认使用 config.QUANT_TYPE
            profile: 设备画像 dict

        示例:
            # 主节点：Layer 0-7 + Embedding
            mgr.load_layer_range(0, 8, has_embedding=True, has_lm_head=False)

            # 中间节点：Layer 8-15
            mgr.load_layer_range(8, 16, has_embedding=False, has_lm_head=False)

            # 末节点：Layer 16-24 + LM Head
            mgr.load_layer_range(16, 24, has_embedding=False, has_lm_head=True)
        """
        import torch.nn as nn
        from qlh_core.contracts.load import RuntimeLoadSpec
        spec = RuntimeLoadSpec.from_dict(load_spec) if load_spec is not None else None
        if spec is not None:
            components = {"layers", "final_norm"}
            if has_embedding:
                components.add("embedding")
            if has_lm_head:
                components.add("lm_head")
            if (spec.materialization not in {"safetensors_key_filtered", LAYER_OFFLOAD_MATERIALIZATION}
                    or set(spec.components) != components
                    or spec.quant_type not in {"fp16", "fp32"} or spec.kv_policy != "dynamic"
                    or spec.kv_device != spec.device or spec.kv_dtype != spec.dtype
                    or {"fp16": "float16", "fp32": "float32"}.get(spec.quant_type) != spec.dtype
                    or (spec.materialization == LAYER_OFFLOAD_MATERIALIZATION
                        and (not spec.device.startswith("cuda:") or spec.sequence_count != 1))):
                raise ValueError("deployment_local_policy_unsupported")
            if (spec.materialization == LAYER_OFFLOAD_MATERIALIZATION
                    and not _layer_offload_device_available(spec.device)):
                raise ValueError("deployment_layer_offload_device_unavailable")
            quant_type = spec.quant_type
            self.quant_type = spec.quant_type

        # ★ 2026-10-08：登记**上游层段**的量化档，供两侧精度对齐诊断（见 `src/relay_precision.py`）。
        #    只记录，**不改变任何加载行为**；优先级：路径标记（`*-f16-dequant` 等）> `quant_type` 意图。
        _note_upstream_precision(
            model_path=model_path or self._full_model_path or self._model_path or MODEL_PATH,
            quant_type=quant_type or QUANT_TYPE,
        )

        # Resolve the active model before validating the assignment. DeepSeek
        # has 28 layers while the legacy Qwen project default has 24, and the
        # master normally reuses ``_full_model_path`` without passing a path.
        path = model_path or self._full_model_path or self._model_path or MODEL_PATH
        prepared = getattr(self, "_resolved_artifact", None)
        self._verify_catalog_artifact(model_id or self._active_model_id, path, expected=prepared)
        config_path = os.path.join(path, "config.json")
        if os.path.isfile(config_path):
            try:
                import json

                with open(config_path, "r", encoding="utf-8") as handle:
                    declared_model_type = str(
                        (json.load(handle) or {}).get("model_type", "") or ""
                    ).lower()
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                declared_model_type = ""
            if declared_model_type in {"gemma", "gemma4_unified"}:
                raise RuntimeError(
                    "Gemma 4 PyTorch 层流水线需要隔离 Transformers sidecar；"
                    "主运行时禁止把 gemma/gemma4_unified 复用为 Qwen2 执行器"
                )
        model_config = AutoConfig.from_pretrained(
            path,
            trust_remote_code=TRUST_REMOTE_CODE,
            local_files_only=True,
        )

        # ---- 参数校验 ----
        from config import TOTAL_MODEL_LAYERS
        # ★ A3：多模态外壳（如 Qwen3.5）把文本层数放在 text_config 下。
        _text_cfg = getattr(model_config, "text_config", None) or model_config
        config_total = int(getattr(_text_cfg, "num_hidden_layers", 0) or 0)
        declared_total = int(total_layers or config_total or TOTAL_MODEL_LAYERS)
        if total_layers and config_total and int(total_layers) != config_total:
            raise ValueError(
                f"层配置总数与模型不一致: "
                f"assignment={int(total_layers)}, model={config_total}"
            )
        if start_layer < 0 or end_layer > declared_total or start_layer >= end_layer:
            raise ValueError(
                f"无效的层范围: [{start_layer}, {end_layer})，"
                f"有效范围: [0, {declared_total})"
            )

        layers_count = end_layer - start_layer
        logger.info(
            f"🎯 层范围加载: Layer {start_layer}-{end_layer} ({layers_count}层), "
            f"embed={has_embedding}, lm_head={has_lm_head}"
        )
        self._layer_load_metrics = None
        self._layer_offload = None

        # Qwen-1.8B 与 Qwen2/DeepSeek 分别使用 transformer.h 和
        # model.layers，两种架构都从 safetensors 中只物化本节点需要的权重。
        model_type = str(getattr(model_config, "model_type", "") or "").lower()
        if (spec is not None and spec.materialization == LAYER_OFFLOAD_MATERIALIZATION
                and model_type not in _QWEN_LAYER_RANGE_TYPES):
            raise ValueError("deployment_layer_offload_architecture_unsupported")
        load_tracker = None
        # ★ A3：Qwen 系（qwen2 / qwen3 / qwen3_5 / qwen3_5_text）共享
        #   `<root>layers.<i>.` 的 key 形态（root 由 A2 探测得出），故复用同一按 key 过滤加载器。
        if model_type in _QWEN_LAYER_RANGE_TYPES:
            load_tracker = self._load_qwen2_layer_range(
                path,
                start_layer,
                end_layer,
                has_embedding=has_embedding,
                has_lm_head=has_lm_head,
                quant_type=quant_type,
                profile=profile,
                model_config=model_config,
                load_spec=spec,
            )
        elif model_type == "qwen":
            load_tracker = self._load_qwen_layer_range(
                path,
                start_layer,
                end_layer,
                has_embedding=has_embedding,
                has_lm_head=has_lm_head,
                quant_type=quant_type,
                profile=profile,
                model_config=model_config,
                load_spec=spec,
            )
        elif model_type in {"gemma", "gemma4_unified"}:
            raise RuntimeError(
                "Gemma 4 PyTorch 层流水线需要隔离 Transformers sidecar；"
                "主运行时禁止把 gemma/gemma4_unified 复用为 Qwen2 执行器"
            )
        else:
            raise RuntimeError(
                f"当前 PyTorch 层流水线不支持模型架构: {model_type or 'unknown'}；"
                "新架构必须按 docs/PyTorch层流水线加载峰值优化方案.md §4.2 "
                "实现 safetensors 按 key 过滤加载，禁止整模加载后裁剪"
            )
        if not isinstance(load_tracker, _LayerRangeLoadTracker):
            raise RuntimeError(
                f"{model_type} 层加载器未返回按 key 过滤守卫，拒绝继续；"
                "禁止绕过 _LayerRangeLoadTracker 后再裁剪完整模型"
            )
        self._engine_type = "pytorch"

        if self.model is None:
            raise RuntimeError("模型加载失败，无法进行层范围裁剪")

        # ---- 裁剪 Transformer 层 ----
        # ★ A3：改用属性探测（model / model.language_model / transformer），
        #   以支持 Qwen3.5 的 `model.model.language_model.layers` 包装；
        #   Qwen2 / 旧 qwen 的探测结果与原来的硬编码分支完全一致（行为不变）。
        transformer, layers_attr, embedding_attr = _locate_text_transformer(self.model)

        # 1. 保留指定范围的 Transformer 层
        all_layers = list(getattr(transformer, layers_attr))
        actual_total = len(all_layers)
        if actual_total != declared_total:
            raise RuntimeError(
                f"模型层数不一致: actual={actual_total}, assignment={declared_total}"
            )
        kept = all_layers[start_layer:end_layer]
        setattr(transformer, layers_attr, nn.ModuleList(kept))
        if len(getattr(transformer, layers_attr)) != layers_count:
            raise RuntimeError(
                "分层模型裁剪结果与分配不一致: "
                f"actual={len(getattr(transformer, layers_attr))}, expected={layers_count}"
            )

        # 释放被裁剪层的引用，帮助 GC 回收显存
        for layer in all_layers[:start_layer]:
            del layer
        for layer in all_layers[end_layer:]:
            del layer
        del all_layers

        # 2. 根据需要保留 Embedding
        if not has_embedding:
            if hasattr(transformer, embedding_attr):
                delattr(transformer, embedding_attr)
                setattr(transformer, embedding_attr, None)

        # 3. 根据需要保留 LM Head
        if not has_lm_head:
            if hasattr(self.model, 'lm_head'):
                del self.model.lm_head
                self.model.lm_head = None

        # 选择性构造时未选中的参数仍为 meta；裁剪完成后，剩余参数必须全部
        # 已物化，并将 rotary/logn 等非持久 buffer 移到同一设备。
        materialized_parameters = list(self.model.parameters())
        if not materialized_parameters:
            raise RuntimeError("分层模型没有可用参数")
        meta_parameters = [
            name for name, parameter in self.model.named_parameters()
            if parameter.device.type == "meta"
        ]
        if meta_parameters:
            raise RuntimeError(
                "分层模型仍有未物化参数: " + ", ".join(meta_parameters[:5])
            )
        target_device = materialized_parameters[0].device
        self.model.to(device=target_device)
        if spec is not None and spec.materialization == LAYER_OFFLOAD_MATERIALIZATION:
            self._layer_offload = _LayerOffloadRuntime(spec.device, getattr(torch, spec.dtype))
        self.model.eval()
        load_tracker.observe()

        # 4. 清理显存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self._layer_load_metrics = load_tracker.finish()
        descriptor = self._pipeline_descriptor or {}
        self._layer_load_metrics["materialization_proof"] = replace(
            self._layer_load_metrics["materialization_proof"],
            artifact_path=os.path.abspath(path),
            artifact_sha256=str(descriptor.get("model_sha256", "") or ""),
            total_layers=actual_total,
        )
        logger.info(
            "层流水线加载观测: mode=%s, tensors=%s, source_bytes=%s, "
            "materialized_bytes=%s, rss_peak_delta_bytes=%s, "
            "cuda_peak_delta_bytes=%s",
            self._layer_load_metrics["mode"],
            self._layer_load_metrics["selected_tensor_count"],
            self._layer_load_metrics["source_tensor_bytes"],
            self._layer_load_metrics["materialized_tensor_bytes"],
            self._layer_load_metrics["rss_peak_delta_bytes"],
            self._layer_load_metrics["cuda_allocated_peak_delta_bytes"],
        )

        # ---- 记录层范围 ----
        self.layer_range = (start_layer, end_layer)
        self._layer_has_embedding = bool(has_embedding)
        self._layer_has_lm_head = bool(has_lm_head)
        self._layer_architecture = model_type
        self._model_layers = layers_count
        self._total_model_layers = actual_total
        self._load_fingerprint = None
        self._load_fingerprint_sha256 = ""
        if model_id:
            self._active_model_id = model_id
        # _total_model_layers 在 _load_pytorch 中已设为完整模型总层数，此处不覆盖

        # ---- 显存统计 ----
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(
                f"✅ 层范围加载完成: 显存 {mem:.2f} GB, "
                f"层 {start_layer}-{end_layer}, "
                f"embed={has_embedding}, lm_head={has_lm_head}"
            )
        else:
            logger.info(
                f"✅ 层范围加载完成: Layer {start_layer}-{end_layer}, "
                f"embed={has_embedding}, lm_head={has_lm_head}"
            )

        # ---- 算子融合（2026-09-18 补齐）----
        # 此前本路径完全没有 compile 应用点；这里与 _load_pytorch 对齐。
        # 注意：forward_layers() 是**手动逐层**前向，吃不到整段 compile；
        # 此处注册的编译版本供 self.model(...) / 内部 transformer(...) 的整段调用使用。
        if self._layer_offload is not None:
            # Compiled layer loops retain GPU parameters across calls and break
            # the one-unit placement contract. The eager loop owns transfers.
            self._compiled_transformer = None
            self._compiled_layer_loop = None
            return
        self._maybe_apply_compile()
        # ★ R-R7 rank3：层分段路径（load_layer_range）同样应用融合 RMSNorm —— 与本方法并列，
        #   因为该路径的层循环同样吃不到整段 compile，但能吃到 RMSNorm 的算子合并。
        self._maybe_apply_fused_rmsnorm()

    def _load_qwen2_layer_range(
        self,
        model_path: str,
        start_layer: int,
        end_layer: int,
        *,
        has_embedding: bool,
        has_lm_head: bool,
        quant_type: str = None,
        profile: dict = None,
        model_config=None,
        load_spec=None,
        architecture: str = "qwen2",
    ) -> _LayerRangeLoadTracker:
        """Materialize only selected parameters from safetensors shards.

        ``architecture`` names the tracker identity for load metrics; the key
        layout is shared by Qwen2 (``model.layers.`` / ``model.embed_tokens.``
        / ``model.norm.`` / ``lm_head.``). Architectures with a different
        model wrapper or forward contract must use an isolated adapter.
        """
        import gc
        import json
        from collections import defaultdict

        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open

        config = model_config or AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=TRUST_REMOTE_CODE,
            local_files_only=True,
        )
        # ★ A3：多模态外壳（如 Qwen3.5）的文本层数在 text_config 下。
        _text_cfg = getattr(config, "text_config", None) or config
        total_layers = int(getattr(_text_cfg, "num_hidden_layers", 0) or 0)
        if total_layers <= 0:
            raise RuntimeError(f"{architecture} config 缺少 num_hidden_layers")

        old_path = os.path.abspath(self._model_path or "") if self._model_path else ""
        keep_tokenizer = self.tokenizer if old_path == os.path.abspath(model_path) else None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ★ A2：层前缀自动探测（探测失败或与默认一致时，行为与既往完全相同）。
        #   不同 Qwen 系包装器的 key 前缀不同（实测 Qwen3.5 为
        #   `model.language_model.layers.`），硬编码 `model.layers.` 会静默匹配不到张量。
        qwen_root = _detect_qwen_root_prefix(model_path)
        if qwen_root != "model.":
            logger.info(
                f"  🔎 探测到层前缀根: {qwen_root}（默认 model.；"
                f"仅影响本段的 key 过滤，语义不变）"
            )
        selected_prefixes = [
            f"{qwen_root}layers.{index}."
            for index in range(start_layer, end_layer)
        ]
        # model.norm is tiny and keeps parameter/device discovery valid on all segments.
        selected_prefixes.append(f"{qwen_root}norm.")
        # ★ B16：tied embeddings（Qwen3.5 等）的 safetensors 里**没有** `lm_head.weight`
        #   （与 embed_tokens 共用）⇒ 末节点不能要求它。此时改为加载 `embed_tokens`，
        #   并在加载后把 lm_head.weight 重绑成同一 Parameter（共享，无额外内存）。
        #   注意：**不**把 `lm_head.` 放进 selected_prefixes ⇒ 它也就不在完整性校验范围内
        #   （校验集 model_prefixes 由 selected_prefixes 派生），因此不会误报 missing。
        tied_lm_head = bool(has_lm_head) and _is_tied_word_embeddings(config, model_path)
        if tied_lm_head:
            logger.info(
                "  🔗 检测到 tied embeddings：末节点的 lm_head 将复用 embed_tokens 权重"
            )
        if has_embedding or tied_lm_head:
            selected_prefixes.append(f"{qwen_root}embed_tokens.")
        if has_lm_head and not tied_lm_head:
            selected_prefixes.append("lm_head.")

        # ★ A3：safetensors 的 key 与「模型属性路径」可能差一层 —— 实测 Qwen3.5 的 key 是
        #   `model.language_model.layers.*`，而 Qwen3_5ForCausalLM 的文本塔直接挂在 `model.`
        #   （即 `model.layers.*`）。因此需要把源前缀映射回模型属性前缀，否则
        #   `set_module_tensor_to_device` 找不到属性。非多模态（qwen_root == "model."）时
        #   该映射为恒等，行为与既往完全一致。
        def model_key_of(key: str) -> str:
            if qwen_root != "model." and key.startswith(qwen_root):
                return "model." + key[len(qwen_root):]
            return key

        model_prefixes = [model_key_of(prefix) for prefix in selected_prefixes]

        target_device, target_dtype = ((load_spec.device, getattr(torch, load_spec.dtype))
                                       if load_spec is not None else _select_layer_runtime(quant_type or QUANT_TYPE))
        offload = load_spec is not None and load_spec.materialization == LAYER_OFFLOAD_MATERIALIZATION
        load_tracker = _LayerRangeLoadTracker(
            architecture=architecture,
            start_layer=start_layer,
            end_layer=end_layer,
            layer_prefix=f"{qwen_root}layers.",
            selected_prefixes=selected_prefixes,
            target_dtype=target_dtype,
            preserve_source_dtype=offload,
        )

        with init_empty_weights():
            # ★ P6 实测：f16+sdpa 1.665 ms/层为最佳；此前未显式设置（依赖库默认）。
            # ⚠️ 但并非所有架构都支持 sdpa（如 remote-code 的 QWenLMHeadModel 会抛
            #    "does not support ... scaled_dot_product_attention"）⇒ **必须可降级**。
            try:
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=TRUST_REMOTE_CODE,
                    attn_implementation="sdpa",
                )
            except (ValueError, TypeError) as exc:
                logger.warning("⚠️ 该架构不支持 sdpa，回退 eager：%s", str(exc)[:160])
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=TRUST_REMOTE_CODE,
                    attn_implementation="eager",
                )
        load_tracker.observe()

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        files_to_keys = defaultdict(list)
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
            for key, filename in weight_map.items():
                if load_tracker.is_selected(key):
                    files_to_keys[filename].append(key)
        else:
            safetensor_files = sorted(
                name for name in os.listdir(model_path)
                if name.endswith(".safetensors")
            )
            for filename in safetensor_files:
                path = os.path.join(model_path, filename)
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if load_tracker.is_selected(key):
                            files_to_keys[filename].append(key)

        if not files_to_keys:
            raise FileNotFoundError(f"{architecture} 模型目录中未找到分层 safetensors 权重")

        runtime_quant = "fp16" if target_dtype == torch.float16 else "fp32"
        requested_quant = quant_type or QUANT_TYPE
        if requested_quant != runtime_quant:
            logger.info(
                "分层选择性加载以 %s 执行（请求量化=%s），仅加载 %s-%s 层",
                runtime_quant.upper(),
                requested_quant,
                start_layer,
                end_layer,
            )
        loaded_keys = set()
        for filename, keys in files_to_keys.items():
            shard_path = os.path.join(model_path, filename)
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in keys:
                    tensor = load_tracker.materialize(handle, key, source_file=shard_path)
                    # ★ A3：写入模型时用归一化后的「属性 key」（见 model_key_of）
                    set_module_tensor_to_device(
                        model,
                        model_key_of(key),
                        "cpu" if offload else target_device,
                        value=tensor,
                        dtype=tensor.dtype if offload else target_dtype,
                    )
                    loaded_keys.add(model_key_of(key))
                    load_tracker.observe()
                    del tensor

        required_parameter_names = {
            name for name, _ in model.named_parameters()
            # ★ A3：用「模型属性前缀」比对（多模态时与源前缀差一层，见 model_key_of）
            if any(name.startswith(prefix) for prefix in model_prefixes)
        }
        missing = sorted(required_parameter_names - loaded_keys)
        if missing:
            raise RuntimeError(
                f"{architecture} 分层权重不完整: " + ", ".join(missing[:5])
            )

        # ★ B16：tied 模型的末节点 —— `set_module_tensor_to_device` 会**替换** Parameter
        #   对象，从而**破坏** `lm_head` 与 `embed_tokens` 的共享（tied）关系
        #   （`lm_head.weight` 会留在 meta 上）⇒ 这里显式把 `lm_head.weight` 重绑为
        #   **同一个 Parameter 对象**（共享，不复制内存）。非 tied 时不进入该分支，
        #   行为与既往完全一致。
        if tied_lm_head:
            lm_head = getattr(model, "lm_head", None)
            embed_weight = None
            try:
                text_model, _la, embed_attr = _locate_text_transformer(model)
                embed_weight = getattr(
                    getattr(text_model, embed_attr, None), "weight", None
                )
            except RuntimeError as exc:
                logger.debug(f"tied 末节点定位 embed_tokens 失败: {exc}")
            if lm_head is None or embed_weight is None:
                raise RuntimeError(
                    "tied 模型做末节点需要在加载后把 lm_head 绑定到 embed_tokens.weight，"
                    f"但未能定位（lm_head={type(lm_head).__name__}, "
                    f"embed_weight={type(embed_weight).__name__}）"
                )
            lm_head.weight = embed_weight  # 同一 Parameter 对象 ⇒ tied 共享，零额外内存
            logger.info(
                f"  🔗 tied 末节点已就绪：lm_head 与 embed_tokens 共享权重 "
                f"({tuple(embed_weight.shape)}, {embed_weight.dtype})"
            )

        model.eval()
        self.model = model
        self.tokenizer = keep_tokenizer or AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=TRUST_REMOTE_CODE,
            local_files_only=True,
        )
        self._model_path = model_path
        self.quant_type = runtime_quant
        self._total_model_layers = total_layers
        self._model_layers = end_layer - start_layer
        logger.info(
            f"{architecture} 选择性权重加载完成: Layer %s-%s / %s",
            start_layer,
            end_layer,
            total_layers,
        )
        return load_tracker

    def _load_qwen_layer_range(
        self,
        model_path: str,
        start_layer: int,
        end_layer: int,
        *,
        has_embedding: bool,
        has_lm_head: bool,
        quant_type: str = None,
        profile: dict = None,
        model_config=None,
        load_spec=None,
    ) -> _LayerRangeLoadTracker:
        """Materialize one original Qwen-1.8B ``transformer.h`` segment."""
        import gc
        import json
        from collections import defaultdict

        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open

        config = model_config or AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=TRUST_REMOTE_CODE,
            local_files_only=True,
        )
        total_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        if total_layers <= 0:
            raise RuntimeError("Qwen config 缺少 num_hidden_layers")

        target_device, target_dtype = ((load_spec.device, getattr(torch, load_spec.dtype))
                                       if load_spec is not None else _select_layer_runtime(quant_type or QUANT_TYPE))
        use_cuda = target_device.startswith("cuda")
        runtime_quant = "fp16" if target_dtype == torch.float16 else "fp32"
        requested_quant = quant_type or QUANT_TYPE
        if requested_quant != runtime_quant:
            logger.info(
                "分层选择性加载以 %s 执行（请求量化=%s），仅加载 %s-%s 层",
                runtime_quant.upper(),
                requested_quant,
                start_layer,
                end_layer,
            )
        # Original Qwen reads these flags while constructing its remote-code model.
        # CPU workers use FP32; hidden states are cast at each pipeline boundary.
        config.bf16 = False
        config.fp16 = use_cuda
        config.fp32 = not use_cuda
        # flash-attn 由推理开关显式决定（票 20）：默认关闭等于接线前行为，
        # 只有显式覆盖才打开；与 llama.cpp 两条路径共用同一张映射。
        config.use_flash_attn = _resolve_pytorch_flash_attn()

        old_path = os.path.abspath(self._model_path or "") if self._model_path else ""
        keep_tokenizer = self.tokenizer if old_path == os.path.abspath(model_path) else None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        selected_prefixes = [
            f"transformer.h.{index}."
            for index in range(start_layer, end_layer)
        ]
        selected_prefixes.append("transformer.ln_f.")
        if has_embedding:
            selected_prefixes.append("transformer.wte.")
        if has_lm_head:
            selected_prefixes.append("lm_head.")

        load_tracker = _LayerRangeLoadTracker(
            architecture="qwen",
            start_layer=start_layer,
            end_layer=end_layer,
            layer_prefix="transformer.h.",
            selected_prefixes=selected_prefixes,
            target_dtype=target_dtype,
        )

        with init_empty_weights():
            # ★ P6 实测：f16+sdpa 1.665 ms/层为最佳；此前未显式设置（依赖库默认）。
            # ⚠️ 但并非所有架构都支持 sdpa（如 remote-code 的 QWenLMHeadModel 会抛
            #    "does not support ... scaled_dot_product_attention"）⇒ **必须可降级**。
            try:
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=TRUST_REMOTE_CODE,
                    attn_implementation="sdpa",
                )
            except (ValueError, TypeError) as exc:
                logger.warning("⚠️ 该架构不支持 sdpa，回退 eager：%s", str(exc)[:160])
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=TRUST_REMOTE_CODE,
                    attn_implementation="eager",
                )
        load_tracker.observe()

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        files_to_keys = defaultdict(list)
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
            for key, filename in weight_map.items():
                if load_tracker.is_selected(key):
                    files_to_keys[filename].append(key)
        else:
            safetensor_files = sorted(
                name for name in os.listdir(model_path)
                if name.endswith(".safetensors")
            )
            for filename in safetensor_files:
                shard_path = os.path.join(model_path, filename)
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if load_tracker.is_selected(key):
                            files_to_keys[filename].append(key)

        if not files_to_keys:
            raise FileNotFoundError("Qwen 模型目录中未找到分层 safetensors 权重")

        loaded_keys = set()
        for filename, keys in files_to_keys.items():
            shard_path = os.path.join(model_path, filename)
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in keys:
                    tensor = load_tracker.materialize(handle, key, source_file=shard_path)
                    set_module_tensor_to_device(
                        model,
                        key,
                        target_device,
                        value=tensor,
                        dtype=target_dtype,
                    )
                    loaded_keys.add(key)
                    load_tracker.observe()
                    del tensor

        required_parameter_names = {
            name for name, _ in model.named_parameters()
            if any(name.startswith(prefix) for prefix in selected_prefixes)
        }
        missing = sorted(required_parameter_names - loaded_keys)
        if missing:
            raise RuntimeError(
                "Qwen 分层权重不完整: " + ", ".join(missing[:5])
            )

        model.eval()
        self.model = model
        self.tokenizer = keep_tokenizer or AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=TRUST_REMOTE_CODE,
            local_files_only=True,
        )
        self._model_path = model_path
        self.quant_type = runtime_quant
        self._total_model_layers = total_layers
        self._model_layers = end_layer - start_layer
        logger.info(
            "Qwen 选择性权重加载完成: Layer %s-%s / %s",
            start_layer,
            end_layer,
            total_layers,
        )
        return load_tracker

    @_serialized_model_access
    def ensure_layer_range(
        self,
        start_layer: int,
        end_layer: int,
        has_embedding: bool,
        has_lm_head: bool,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
        total_layers: int = None,
        model_id: str = None,
    ) -> None:
        """确保当前 PyTorch 模型已裁剪为指定层范围，避免重复重载。"""
        desired_range = (start_layer, end_layer)
        if (
            self.is_loaded
            and self._engine_type == "pytorch"
            and self.layer_range == desired_range
            and self._layer_has_embedding == bool(has_embedding)
            and self._layer_has_lm_head == bool(has_lm_head)
        ):
            return
        self.load_layer_range(
            start_layer,
            end_layer,
            has_embedding=has_embedding,
            has_lm_head=has_lm_head,
            model_path=model_path or self._full_model_path or self._model_path,
            quant_type=quant_type,
            profile=profile,
            total_layers=total_layers,
            model_id=model_id,
        )

    def _llama_cpp_trimmed_range(self) -> str:
        """返回 llama.cpp 引擎的裁层描述；**空串表示未裁层**（可安全当整模用）。

        ★ 判据必须问 engine：裁层状态记在 `LlamaCppEngine._pipeline_descriptor` 上，
        而 ModelManager 自己的 `_pipeline_descriptor` / `_pipeline_distributed_only`
        只覆盖 PyTorch 的 `load_layer_range`（从 safetensors 物化层段）那条路径。
        这也是旧代码「llama.cpp 不存在层裁剪」假设失效的地方。
        """
        engine = getattr(self, "model", None)
        describe = getattr(engine, "get_pipeline_descriptor", None)
        if not callable(describe):
            return ""
        try:
            info = describe() or {}
        except Exception:  # noqa: BLE001 - 描述失败一律按「不可确认」处理
            return "descriptor_unavailable"
        rng = info.get("assignment_layer_range")
        if not info.get("partial_assignment") and not rng:
            return ""
        artifact = info.get("loaded_artifact") or info.get("model_path") or ""
        return f"range={rng} artifact={os.path.basename(str(artifact))}"

    @_serialized_model_access
    def ensure_full_model(self, quant_type: str = None,
                          profile: dict = None, engine: str = None) -> None:
        """确保当前模型为完整模型；流水线裁剪后回退本地推理前调用。"""
        if self._pipeline_distributed_only:
            raise RuntimeError(
                "当前模型以分布式专用模式准备，禁止自动整模回退；"
                "请等待流水线节点就绪或显式执行普通模型加载"
            )
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        # ★ 2026-10-04：`llama_cpp` **不再等于**「完整模型」语义 ——
        #   `LlamaCppEngine.load_layer_range()`（`src/llama_engine.py`）能加载裁层 GGUF。
        #   旧假设「llama.cpp 不存在层裁剪」让本方法对裁层状态**静默 return** ⇒ 回退路径
        #   把 head8（`[0,8)`、`lm_head=False`）当整模继续推理 ⇒ 产出垃圾却仍返回 HTTP 200。
        #   现在裁层一律 fail-closed（与 `_pipeline_distributed_only` 同口径，不自动重载
        #   整模）；`island` 仍按旧语义直接返回。
        if self._engine_type == "llama_cpp":
            trimmed = self._llama_cpp_trimmed_range()
            if trimmed:
                raise RuntimeError(
                    "当前 llama.cpp 引擎加载的是裁层工件（"
                    f"{trimmed}），禁止整模回退；"
                    "请等待流水线节点就绪或显式重新加载完整模型"
                )
            return
        if self._engine_type == "island":
            return
        if (
            self._engine_type == "pytorch"
            and self.layer_range is None
            and self._layer_has_embedding
            and self._layer_has_lm_head
        ):
            return

        model_id = self._active_model_id or mc.get_profile_default_model_id()
        q = quant_type or self._full_model_quant_type or self.quant_type or QUANT_TYPE
        model_path = self._full_model_path or self._model_path
        logger.info(
            f"🔄 当前为流水线裁剪模型 layer_range={self.layer_range}，"
            f"重新加载完整模型用于本地推理"
        )
        self.load_model(
            model_id=model_id,
            model_path=model_path,
            quant_type=q,
            profile=profile,
            engine=engine or "pytorch",
        )

    def _load_island(self, profile: dict = None) -> None:
        """
        连接 TP 孤岛引擎（OpenAI 兼容端点）。

        无本地模型文件："加载" = GET /v1/models 健康检查 + 解析后端模型名。
        端点/凭据/超时均取自 config.QLH_ISLAND_* 配置。
        """
        from island_engine import IslandEngine

        engine = IslandEngine()
        engine.load_model()
        self._island_engine = engine
        # model_path 记录脱敏后的端点，供状态上报/日志展示（不含凭据）
        self._model_path = engine.masked_base_url
        self.quant_type = None

        logger.info(
            f"✅ 孤岛引擎就绪 (整请求转发): endpoint={engine.masked_base_url}, "
            f"model={engine.model_name}"
        )

    def _load_llama_cpp(self, model_path: str = None, profile: dict = None) -> None:
        """
        加载 llama.cpp + GGUF 模型。

        GGUF 文件查找顺序:
          1. 显式指定的 model_path（如果是 .gguf 文件）
          2. config.GGUF_MODEL_PATH
          3. 自动搜索 models/ 目录下的 .gguf 文件
        """
        from llama_engine import LlamaCppEngine, get_gguf_model_path

        # 确定 GGUF 文件路径
        gguf_path = None
        if model_path and model_path.endswith(".gguf"):
            gguf_path = model_path
        else:
            # ★ 2026-09-19 BUG 修复：优先按**当前活跃模型**在注册表里的 `gguf_path` 解析。
            #   原逻辑直接落到模块级静态常量 `GGUF_MODEL_PATH`（兜底默认模型），
            #   于是**切到任意 GGUF 模型都会加载成兜底那个**（实测：切 `qwen3-5-2b`
            #   实际加载 `qwen3-0.6b-q8_0.gguf`）。safetensors 目录路径不会以 `.gguf`
            #   结尾，所以走 registered-model 分支正是 GGUF 模型的正常入口。
            _active_id = self._active_model_id or mc.get_profile_default_model_id()
            _registered = mc.get_builtin_model(_active_id)
            _reg_gguf = mc.resolve_model_path(_registered.gguf_path) if (
                _registered is not None and _registered.gguf_path
            ) else ""
            if _reg_gguf and os.path.isfile(_reg_gguf):
                gguf_path = _reg_gguf
            elif os.path.isfile(GGUF_MODEL_PATH):
                gguf_path = GGUF_MODEL_PATH
            else:
                gguf_path = get_gguf_model_path()

        if not gguf_path or not os.path.isfile(gguf_path):
            raise FileNotFoundError(
                f"GGUF 模型文件未找到。\n"
                f"  配置路径: {GGUF_MODEL_PATH}\n"
                f"  请下载 GGUF 格式的 Qwen-1.8B-Chat 模型:\n"
                f"  - 推荐: Q4_K_M (~1.16 GB) — 速度/质量最佳平衡\n"
                f"  - 下载: https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf\n"
                f"  - 或使用模型下载引导: python src/model_downloader.py"
            )

        # 自适应上下文窗口大小
        n_ctx = 4096
        if profile:
            tier = profile.get("tier", "laptop")
            if tier == "edge":
                n_ctx = 1024
            elif tier == "mobile":
                n_ctx = 512
            elif tier == "ultrabook":
                n_ctx = 2048

        # ★ 票 15a/15b 接线：整模 GGUF 走「绑定」还是「llama-server 子进程」，由
        #   `select_inference_path` 决定。默认 auto = llama.cpp 家族内挑传输：
        #   有 CUDA 走 server（绑定是 CPU-only，拿不到 GPU），无 CUDA 走绑定。
        #   两条硬规矩：层段接力强制绑定；**显式**请求 server 而无 CUDA / 无
        #   可执行文件时具名报错（绝不静默降级）。auto 的降级会在 reason 写明。
        import config as _cfg
        from inference_path_selector import (
            AUTO,
            BINDING,
            SERVER,
            select_inference_path,
        )
        from llama_server_engine import default_server_exe

        server_available = bool(default_server_exe())
        decision = select_inference_path(
            requested=str(getattr(_cfg, "INFER_PATH", AUTO) or AUTO),
            server_available=server_available,
        )
        if decision.path == SERVER:
            from llama_server_engine import LlamaServerEngine

            logger.info("推理路径裁决 -> llama-server 子进程（%s）", decision.reason)
            self._llama_engine = LlamaServerEngine(
                # 必须与决策同源：用硬件级探测，而不是 torch.cuda.is_available()
                # —— 本仓 .venv-test 是 CPU-only torch，后者会广告出"C 无 CUDA"
                # 却实际走了 server 路径，自相矛盾。
                cuda_available=decision.cuda_available,
            )
            self._llama_engine.load_model(model_path=gguf_path, context_limit=n_ctx)
        else:
            if decision.cuda_available and decision.requested == AUTO:
                # auto 降级必须可观测（有 GPU 却回落到 CPU 绑定，用户会奇怪为什么慢）
                logger.warning("推理路径 auto 降级到绑定：%s", decision.reason)
            self._llama_engine = LlamaCppEngine()
            self._llama_engine.load_model(
                model_path=gguf_path,
                context_limit=n_ctx,
            )
        self._model_path = gguf_path

        logger.info("✅ llama.cpp 引擎就绪 (CPU/集显 优化)")

    def _load_gemma4_native(self, gguf_path: str, profile: dict = None) -> None:
        """Load the frozen Gemma 4 GGUF/mmproj pair through native MTMD."""
        from llama_engine import LlamaCppEngine

        use_cuda = bool(torch.cuda.is_available())
        self._prepare_gemma4_native_binding(use_cuda=use_cuda)
        engine = LlamaCppEngine()
        try:
            engine.load_gemma4_native(
                gguf_path=gguf_path,
                n_ctx=768,
                gpu_layers=-1 if use_cuda else 0,
                require_gpu_layers=1 if use_cuda else 0,
                mtmd_use_gpu=use_cuda,
            )
        except Exception:
            try:
                engine.unload()
            except Exception:
                pass
            raise
        self._llama_engine = engine
        self._model_path = gguf_path
        self.quant_type = "Q4_K_M"
        logger.info("Gemma 4 native llama.cpp/MTMD 引擎就绪")

    @staticmethod
    def _prepare_gemma4_native_binding(*, use_cuda: bool) -> None:
        """Select and validate the isolated native binding before first import."""
        import importlib
        from pathlib import Path
        import sys

        managed_root = None
        if getattr(sys, "frozen", False):
            managed_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve(strict=False)
            site_packages = managed_root
        else:
            configured = os.environ.get("QLH_GEMMA4_SITE_PACKAGES", "").strip()
            if configured:
                site_packages = Path(configured).expanduser().resolve(strict=False)
            else:
                venv = Path(__file__).resolve().parents[1] / ".venv-gemma4-native"
                if os.name == "nt":
                    site_packages = venv / "Lib" / "site-packages"
                else:
                    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
                    site_packages = venv / "lib" / version / "site-packages"
            managed_root = site_packages

        if not site_packages.is_dir():
            raise RuntimeError(
                "gemma4-native requires an existing managed llama_cpp site-packages directory"
            )

        def _validate_module_root(module, label: str) -> None:
            module_file = str(getattr(module, "__file__", "") or "").strip()
            if not module_file:
                raise RuntimeError(f"gemma4-native {label} module has no __file__")
            resolved = Path(module_file).expanduser().resolve(strict=False)
            try:
                resolved.relative_to(managed_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"gemma4-native {label} module is outside the managed binding root"
                ) from exc

        loaded = sys.modules.get("llama_cpp")
        if loaded is not None:
            _validate_module_root(loaded, "llama_cpp")
        else:
            site_path = os.fspath(site_packages)
            sys.path[:] = [entry for entry in sys.path if entry != site_path]
            sys.path.insert(0, site_path)
            importlib.invalidate_caches()

        loaded = sys.modules.get("llama_cpp")

        try:
            import llama_cpp
            import llama_cpp.mtmd_cpp as mtmd
        except ImportError as exc:
            raise RuntimeError("gemma4-native requires the managed llama_cpp MTMD binding") from exc
        _validate_module_root(llama_cpp, "llama_cpp")
        _validate_module_root(mtmd, "mtmd_cpp")
        from scripts.model_tools.gemma4_native_binding import (
            expected_binding_marker,
            validate_binding_marker,
        )

        expected_marker = expected_binding_marker()
        expected_version = expected_marker["package"]["version"]
        if getattr(llama_cpp, "__version__", "") != expected_version:
            raise RuntimeError(
                f"gemma4-native requires llama-cpp-python {expected_version}"
            )
        validate_binding_marker(site_packages)
        for symbol in expected_marker["abi"]["mtmd_python_symbols"]:
            if not callable(getattr(mtmd, symbol, None)):
                raise RuntimeError(f"gemma4-native MTMD binding is missing {symbol}")
        if use_cuda:
            lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
            names = {path.name.lower() for path in lib_dir.iterdir()} if lib_dir.is_dir() else set()
            if os.name == "nt":
                required = {
                    "ggml-cuda.dll", "cudart64_13.dll", "cublas64_13.dll",
                    "cublaslt64_13.dll",
                }
                missing = sorted(required - names)
            else:
                missing = [] if any("ggml-cuda" in name for name in names) else ["ggml-cuda"]
            if missing:
                raise RuntimeError(
                    "gemma4-native CUDA binding is incomplete: " + ", ".join(missing)
                )

    def _load_pytorch(
        self,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
    ) -> None:
        """
        加载 PyTorch + Transformers 模型。

        核心逻辑:
        - CPU: 使用 torch.float32，规避半精度 CPU 算子不兼容
        - fp16: 直接以 torch.float16 加载，显存 ~3.5 GB
        - int8: bitsandbytes 8-bit 量化加载，显存 ~2.3 GB
        - int4: bitsandbytes 4-bit NF4 双重量化加载，显存 ~1.8 GB
        """
        path = model_path or MODEL_PATH
        if str(path).strip().lower().endswith(".gguf"):
            raise ValueError("PyTorch loader 拒绝 GGUF 路径；请先通过统一 resolver 选择引擎与工件")
        self._model_path = path
        self.quant_type = quant_type or QUANT_TYPE

        # ---- 自适应：根据设备画像调整加载策略 ----
        force_cpu = False
        if profile:
            tier = profile.get("tier", "laptop")
            gpu_info = profile.get("gpu", {})
            has_cuda = gpu_info.get("cuda_available", False)
            vram_gb = gpu_info.get("vram_total_gb", 0)

            if tier in ("ultrabook", "edge", "mobile") and not has_cuda:
                force_cpu = True
                logger.info(f"设备档位={tier} 无 CUDA，使用 CPU-only 模式")

            if has_cuda and vram_gb > 0:
                min_vram = {"fp16": 3.5, "int8": 2.3, "int4": 1.8}.get(self.quant_type, 3.5)
                if vram_gb < min_vram:
                    logger.warning(
                        f"⚠️ 显存 {vram_gb:.1f} GB 不足以加载 {self.quant_type} 模型"
                        f"（预计需要 {min_vram:.1f} GB），建议切换量化精度"
                    )

        use_cuda = torch.cuda.is_available() and not force_cpu

        logger.info(f"加载模型: {path}")
        logger.info(f"量化精度: {self.quant_type}  |  算子融合: {USE_COMPILE}")
        logger.info(f"推理设备: {'CUDA' if use_cuda else 'CPU'}")

        # ---- CPU-only 路径 ----
        if not use_cuda:
            if self.quant_type in ("int4", "int8"):
                logger.warning(
                    f"⚠️ bitsandbytes {self.quant_type} 量化不支持 CPU，回退到 FP32 CPU 推理"
                )
                logger.warning(
                    f"💡 建议切换引擎为 llama.cpp (设置 INFERENCE_ENGINE='llama_cpp') "
                    f"以获得更好的 CPU 推理性能（3-5x 加速）"
                )
            elif self.quant_type != "fp32":
                logger.info(
                    "CPU PyTorch 推理将请求精度 %s 调整为 FP32，以保证算子兼容性",
                    self.quant_type,
                )
            self.quant_type = "fp32"

            cpu_cores = profile.get("cpu", {}).get("physical_cores", 4) if profile else 4
            omp_threads = max(2, cpu_cores // 2)
            os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(omp_threads))
            logger.info(f"CPU 线程数: OMP={omp_threads}, MKL={omp_threads}")

            load_kwargs: Dict[str, Any] = dict(
                device_map={"": "cpu"},
                trust_remote_code=TRUST_REMOTE_CODE,
                torch_dtype=torch.float32,
            )
        else:
            # ---- CUDA 路径 ----
            bnb_config = self._get_bnb_config(self.quant_type)

            # ★ P6 实测 f16+sdpa 1.665 ms/层最佳，但**并非所有架构都支持**（remote-code 老架构会抛
            #   ValueError）。这里**直接试**而不是靠常量探测 —— 探测在 5.17 下不经实测不敢依赖。
            _attn_candidates = ("sdpa", "eager")
            load_kwargs: Dict[str, Any] = dict(
                device_map="auto",
                trust_remote_code=TRUST_REMOTE_CODE,
                # ★ P6 实测 f16+sdpa 1.665 ms/层最佳；`_attn_impl` 由调用方按架构降级决定
                #   （不支持的架构会用 "eager"，见下方 except 分支）。
                attn_implementation=_attn_candidates[0],
            )

            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = (
                    torch.float32 if self.quant_type == "fp32" else torch.float16
                )

        t0 = time.time()

        logger.info(f"加载 PyTorch 模型路径: {path}")

        # ★ BUG 修复（2026-09-19）：量化 + remote code + transformers≥5 时，
        #   加载窗口内**禁止** `_init_weights` 覆盖已装载权重（否则 Qwen-1.8B 会在
        #   `modeling_qwen.py:_init_weights` 里对 uint8 权重调 `normal_()` 而崩溃）。
        #   作用域收窄：仅三条件同时成立才打补丁。
        #
        #   ⚠️ 判据必须是 `_model_declares_remote_code(path)`（该模型自己声明了 `auto_map`），
        #   **不能**用全局 `TRUST_REMOTE_CODE` —— 那是**用户授权开关**且主仓默认 True，
        #   拿它当判据会把**原生模型**也纳入作用域 ⇒ 跳过其权重/buffer 初始化 ⇒ **脏 token**。
        #   实测（qwen3-0.6b + int4，单变量对照）：允许 premark ⇒ `<think>` 重复 12 次 + 乱码；
        #   禁用 ⇒ 输出通顺。见 `_model_declares_remote_code` 的 docstring。
        _needs_premark = (
            _model_declares_remote_code(path)
            and _is_transformers_5_or_newer()
            and self.quant_type in ("int4", "int8")
        )
        if _needs_premark:
            logger.info("  已启用「预标记已初始化」守卫（量化 + remote code + transformers≥5）")

        premark = _premark_hf_initialized() if _needs_premark else contextlib.nullcontext()
        with premark:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
            except (ValueError, TypeError) as exc:
                # ⚠️ 老架构（remote code）不支持 sdpa ⇒ 回退 eager 重试一次
                if load_kwargs.get("attn_implementation") != "eager":
                    logger.warning("⚠️ 该架构不支持 sdpa，回退 eager 重试：%s", str(exc)[:160])
                    load_kwargs = dict(load_kwargs, attn_implementation="eager")
                    self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
                else:
                    raise

        # ★ A7/B7：transformers 5.x 下 remote-code 模型的「权重被 `_init_weights` 覆盖」守卫。
        #   只在「transformers ≥5 且启用了 remote code」时执行 —— 4.x 无此缺陷（它有
        #   `set_initialized_submodules()`），因此可省掉一次全区读取（Qwen-1.8B 约 3.5 GB）。
        self._weight_guard_report = None
        #: 是否因 transformers 5.x 断裂而把 generate 改绑到 GenerationMixin
        self._generate_rebound = False
        if TRUST_REMOTE_CODE and _is_transformers_5_or_newer():
            self._weight_guard_report = _verify_and_repair_loaded_weights(self.model, path)

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=TRUST_REMOTE_CODE)

        # ★ BUG 修复（2026-09-19）：**旧版 remote code 模型缺 `generation_config`**。
        #   `_pytorch_chat()` 调 `self.model.generate(...)`，而 transformers>=5 的 `generate()`
        #   会读 `self.generation_config`；Qwen-1.8B 的 `QWenLMHeadModel`（继承
        #   `QWenPreTrainedModel` → `PreTrainedModel`）**从不设置该属性**（实测
        #   `hasattr(..., "generation_config") is False`）⇒ `AttributeError` ⇒ 聊天即时失败
        #   （上层表现为「收不到回复 / 超时」）。
        #   ⇒ **只补不覆盖**：已有该属性的模型完全不受影响。
        if getattr(self.model, "generation_config", None) is None:
            try:
                from transformers import GenerationConfig

                gen_cfg = GenerationConfig.from_model_config(self.model.config)
                # 尽量补齐特殊 token（`generate()` 的调用点虽显式传了 pad_token_id，
                # 但其他调用方可能依赖配置里的值）
                tok = self.tokenizer
                if getattr(gen_cfg, "pad_token_id", None) is None:
                    gen_cfg.pad_token_id = getattr(tok, "pad_token_id", None) \
                        or getattr(tok, "eos_token_id", None)
                if getattr(gen_cfg, "eos_token_id", None) is None:
                    gen_cfg.eos_token_id = getattr(tok, "eos_token_id", None)
                self.model.generation_config = gen_cfg
                logger.info("  已为 %s 补齐 generation_config（remote code 旧架构缺失）",
                            type(self.model).__name__)
            except Exception as exc:  # noqa: BLE001 —— 只补不抛
                logger.warning("  补齐 generation_config 失败（不影响加载）：%s", str(exc)[:160])

        # ★ BUG 修复（2026-09-19）：**旧 remote code 的 `generate` 与 transformers 5.x 断裂**。
        #   `QWenLMHeadModel.generate` 内部调 `super().generate(...)`，而 5.x 已把 `generate`
        #   从 `PreTrainedModel` 挪到 `GenerationMixin` ⇒ `AttributeError: 'super' object has
        #   no attribute 'generate'`（实测：补齐 generation_config 后紧接着出现）。
        #   ⇒ 这里**探测**该断裂，命中才把实例的 `generate` 绑到标准 `GenerationMixin.generate`。
        _gen = getattr(self.model, "generate", None)
        _broken = _gen is None
        if not _broken:
            try:
                import inspect as _inspect

                src_text = ""
                try:
                    src_text = _inspect.getsource(_gen) or ""
                except Exception:  # noqa: BLE001 —— 拿不到源码不算断裂
                    src_text = ""
                # 只有「源码里确实调 super().generate」才认为存在断裂风险
                _broken = "super().generate" in src_text
            except Exception:  # noqa: BLE001
                _broken = False
        if _broken:
            try:
                from transformers.generation import GenerationMixin

                self.model.generate = GenerationMixin.generate.__get__(self.model, type(self.model))
                self._generate_rebound = True
                logger.warning(
                    "  ⚠️ %s.generate 依赖已被 5.x 移除的 super().generate ⇒ "
                    "已改绑标准 GenerationMixin.generate",
                    type(self.model).__name__,
                )
            except Exception as exc:  # noqa: BLE001 —— 只补不抛
                logger.warning("  改绑 generate 失败（不影响加载）：%s", str(exc)[:160])
        else:
            self._generate_rebound = False

        self.layer_range = None
        self._layer_load_metrics = None
        self._layer_offload = None
        self._layer_has_embedding = True
        self._layer_has_lm_head = True

        load_time = time.time() - t0

        # 记录模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        param_dtype = next(self.model.parameters()).dtype
        layers = self._count_transformer_layers()
        self._model_layers = layers
        self._total_model_layers = layers

        logger.info(f"模型加载完成 ({load_time:.1f}s)")
        logger.info(f"  参数量: {total_params/1e9:.2f}B  |  类型: {param_dtype}")
        logger.info(f"  设备: {self.model.device}  |  Transformer层数: {self._model_layers}")

        # 显存统计
        if use_cuda and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"  GPU 显存占用: {mem:.2f} GB")
        elif not use_cuda:
            mem = psutil.virtual_memory().used / (1024 ** 3)
            logger.info(f"  CPU 内存占用 (进程): {mem:.1f} GB")

        # 算子融合 — 仅在 FP16 + CUDA 下生效
        if USE_COMPILE:
            if not use_cuda:
                logger.warning("⚠️ torch.compile 需要 CUDA，CPU 模式下已自动跳过")
            elif self.quant_type != "fp16":
                logger.warning(
                    f"⚠️ torch.compile 与 {self.quant_type} 量化不兼容（实测慢 13%），已自动跳过。"
                    f"如需融合，请设置 QUANT_TYPE='fp16'。"
                )
            else:
                self._apply_compile()

        # ★ R-R7 rank3：融合 RMSNorm —— 与 compile 无关的**独立开关**（见 config.USE_FUSED_RMSNORM）。
        #   不放在 `if USE_COMPILE` 内：它不需要 CUDA、也没有规模门，收益来自消掉每步的 `.to` 往返。
        self._maybe_apply_fused_rmsnorm()

    def _count_transformer_layers(self) -> int:
        """统计模型的 Transformer 层数（A3：改用统一的包装器探测）。"""
        if self.model is None:
            return 0
        try:
            transformer, layers_attr, _ = _locate_text_transformer(self.model)
        except RuntimeError:
            return 0
        return len(getattr(transformer, layers_attr, ()))

    def _apply_compile(self) -> None:
        """开启 torch.compile 自动算子融合（2026-09-18 修正）。

        两处实测修正：

        * **mode 用 ``default`` 而非 ``reduce-overhead``**：后者的 CUDA Graphs 与
          「KV cache 每步换 tensor」不兼容，解码路径会抛
          ``RuntimeError: accessing tensor output of CUDAGraphs ...``（实测）；
          而 ``default`` 在 12 层分段上实测 1.0443 → 0.3258 ms/层。
        * **不替换 ``self.model``**：原实现直接 ``self.model = torch.compile(self.model)``，
          会让 ``self.model.model`` / ``self.model.transformer`` 的结构访问失效，
          而 ``forward_layers()`` 与 ``_count_transformer_layers()`` 都依赖它们。
          改为编译内部 transformer 并另存 ``self._compiled_transformer``。

        ⚠️ 数值提示：``mode="default"`` **不是 bit-exact**（实测 hidden ``max|diff|≈7.8e-2``，
        fp16）。若调用方有「逐 token 一致」的验收判据（如跨框架接力），必须自行复验。
        """
        limit = globals().get("COMPILE_RECOMPILE_LIMIT")
        if limit:
            try:
                torch._dynamo.config.recompile_limit = int(limit)
                torch._dynamo.config.cache_size_limit = int(limit)
                logger.info(
                    f"  torch._dynamo recompile_limit → {limit}"
                    f"（2026-09-18 实测：非 hybrid 长序列下有 +44% 收益）")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  设置 recompile_limit 失败: {e}")
        logger.info("开启 torch.compile 算子融合 (mode='default')...")
        # ★ A3：改用统一的包装器探测（支持 Qwen3.5 的 `model.model.language_model`）——
        #   这同时让下面 A4 的「层循环」能在 Qwen3.5 上正确取到 `.layers`。
        try:
            inner, _inner_layers_attr, _ = _locate_text_transformer(self.model)
        except RuntimeError:
            logger.warning("  ❌ torch.compile 启用失败: 找不到内部 transformer，回退到普通模式")
            self._compiled_transformer = None
            return
        try:
            self._compiled_transformer = torch.compile(inner, mode="default")
            logger.info("  ✅ torch.compile 已启用 (mode='default'；解码路径不依赖 CUDA Graphs)")

            # ---- A4：额外编译「层循环」（供 forward_layers 使用）----
            # 与整模编译的区别：层循环**不含**末端 norm / lm_head ⇒ 与分段前向语义一致。
            self._compiled_layer_loop = None
            if USE_MONOLITHIC_FORWARD and hasattr(inner, "layers") and len(inner.layers):
                first_fwd = inner.layers[0].forward
                params = inspect.signature(first_fwd).parameters
                cache_arg_name = (
                    "past_key_values" if "past_key_values" in params
                    else "past_key_value" if "past_key_value" in params
                    else None
                )
                # ⚠️ 编译前必须定成最终值：compile 把属性当 guard，若每次调用临时改再恢复
                #    （原逐层版的补丁方式）会反复触发重编译。分段加载时本地索引才是正确的。
                # ★ A3：用通用探测（Qwen3.5 的 linear_attention 层持有者是 `linear_attn`）
                for local_idx, layer in enumerate(inner.layers):
                    for holder in _layer_idx_holders(layer):
                        holder.layer_idx = local_idx
                # ★ B14：hybrid（如 Qwen3.5）的每层 mask 不同 ⇒ 把「每层的 mask 索引」编译进
                #   层循环，这样 hybrid 也能吃到 A4 的收益（此前 hybrid 被整体跳过）。
                #   索引顺序由 `_hybrid_mask_index_per_layer()` 给出，forward_layers 用同一函数
                #   构造 mask 元组 ⇒ 两边顺序必然一致。
                mask_index_per_layer = None
                _inner_layer_types = getattr(inner.config, "layer_types", None)
                if _is_hybrid_layer_types(_inner_layer_types):
                    _order, mask_index_per_layer = _hybrid_mask_index_per_layer(
                        _inner_layer_types
                    )
                self._compiled_layer_loop = torch.compile(
                    _LayerLoop(inner.layers, cache_arg_name, mask_index_per_layer),
                    mode="default",
                )
                logger.info(
                    f"  ✅ A4 层循环已编译（{len(inner.layers)} 层；layer_idx 已永久本地化；"
                    f"cache 参数名={cache_arg_name}；"
                    f"per-layer mask={'是（hybrid）' if mask_index_per_layer else '否'}）"
                )
            # 退出路径配套（2026-09-18 用户裁定）：hybrid + compile 时解释器清理期可能崩溃
            #（`_PyModule_ClearDict` 调用栈）⇒ 登记 atexit，在清理**之前**有序释放编译对象
            # 与 CUDA 缓存，降低「退出码非零」的风险。
            import atexit

            def _release_compiled() -> None:  # pragma: no cover - 退出期路径
                try:
                    self._compiled_transformer = None
                    self._compiled_layer_loop = None
                    self.model = None
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 - 退出期不允许再抛
                    pass

            atexit.register(_release_compiled)
        except Exception as e:
            self._compiled_transformer = None
            self._compiled_layer_loop = None
            hint = ""
            if isinstance(e, UnicodeDecodeError) or "codec can't decode" in str(e):
                hint = (
                    "（Windows 下常见：torch/triton 内部按 GBK 解码源码失败。"
                    "请以 UTF-8 模式启动进程，例如设置环境变量 PYTHONUTF8=1 后重试）"
                )
            logger.warning(f"  ❌ torch.compile 启用失败: {e}，回退到普通模式{hint}")

    def _is_hybrid_architecture(self) -> bool:
        """模型是否为「hybrid」层型（部分层不是 full attention）。

        2026-09-18 B2 实测：Qwen3.5 这类 hybrid 架构 + torch.compile 会让
        ``torch._dynamo`` 的 ``recompile_limit`` 被 hybrid KV 的不稳定守卫
        （``transformers/cache_utils.py`` 的 ``lazy_initialization`` 里
        ``self.device is None``）打满 ⇒ 长序列退化；且 hybrid + compile 会在
        解释器退出期崩溃（不影响推理结果，但进程退出码非零）。
        """
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return False
        text_cfg = getattr(cfg, "text_config", None) or cfg
        # ★ A3：复用共享判定（排除 sliding_attention；不能只看「有无 layer_types」）
        return _is_hybrid_layer_types(getattr(text_cfg, "layer_types", None))

    def _estimate_total_params(self) -> float | None:
        """按 config 估算**整模**参数量（用于 compile 的规模门）。

        为什么不用 `sum(p.numel() for p in self.model.parameters())`：**分层加载只物化本节点
        的层段**（可能只有 12/24 层），直接求和会把大模型误判成小模型而错误跳过 compile。

        近似式：`layers × (4·h² + 3·h·i)`（attention 约 4 个 h×h 投影 + MLP 三个 h×i 矩阵），
        忽略 embedding/lm_head 与 hybrid 的 linear_attn 差异 —— 作为**数量级门槛**足够。
        """
        try:
            cfg = getattr(getattr(self.model, "config", None), "text_config", None) or \
                getattr(self.model, "config", None)
            if cfg is None:
                return None
            layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
            h = int(getattr(cfg, "hidden_size", 0) or 0)
            i = int(getattr(cfg, "intermediate_size", 0) or 0)
            if layers <= 0 or h <= 0 or i <= 0:
                return None
            return float(layers) * (4.0 * h * h + 3.0 * h * i)
        except Exception:  # noqa: BLE001
            return None

    def _maybe_apply_fused_rmsnorm(self) -> None:
        """★ R-R7 rank3：按 `USE_FUSED_RMSNORM` 决定是否把本模型的 RMSNorm 换成 `F.rms_norm`。

        与 `_maybe_apply_compile` 不同，这里**不设规模门**：实测收益来自「消掉每步 2×层数的 `.to`
        往返」，而该路径是 CPU/launch-bound，占比**在小模型上更高**，与参数量无关；也不需要 CUDA
        （`F.rms_norm` 在 CPU 上同样可用）。
        默认关闭 ⇒ 与旧版行为完全一致（见 `config.USE_FUSED_RMSNORM` 的实测依据与判据）。
        """
        if not USE_FUSED_RMSNORM:
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        replaced = _apply_fused_rmsnorm(model)
        if replaced:
            logger.info("✅ 已启用融合 RMSNorm（F.rms_norm）：替换 %d 个模块", replaced)
        else:
            logger.warning(
                "⚠️ USE_FUSED_RMSNORM=1 但未找到可替换的 RMSNorm 模块（保持原实现、行为不变）")

    def _maybe_apply_compile(self) -> None:
        """按与 ``_load_pytorch`` 相同的条件决定是否启用算子融合。

        抽出来的原因：``load_layer_range()`` 此前**没有 compile 应用点**，
        分层节点永远拿不到融合收益（2026-09-18 补齐）。

        两道保护（2026-09-18 用户裁定，依据 B1/B2 实测）：
          * ``COMPILE_MAX_SEQ_LEN`` —— 长序列下 compile 会负收益（141 步 0.672×）；
          * hybrid 架构 —— ``recompile_limit`` 触顶 + 退出期崩溃。
        """
        if not USE_COMPILE:
            return
        if not torch.cuda.is_available():
            logger.warning("⚠️ torch.compile 需要 CUDA，CPU 模式下已自动跳过")
            return
        if self.quant_type not in (None, "fp16"):
            logger.warning(
                f"⚠️ torch.compile 与 {self.quant_type} 量化不兼容（实测慢 13%），已自动跳过。"
                f"如需融合，请设置 QUANT_TYPE='fp16'。"
            )
            return
        # ★ 规模门（2026-09-19 用户裁定）：编译的收益**只在 ≥1.5B 上稳定为正**。
        #   ★ 2026-09-19 修正：旧依据（「0.5B/12 层反而慢约 6×」）是**误读** —— 那是把**约 6.45 s 的首次编译**
        #     摊进了短 gen 的每步。实测（`bench_compile_fresh.py`，主仓 `forward_layers`/12 层/f16）：编译层循环
        #     **稳态快 2.428×**（4.615 vs 11.204 ms/步），但**盈亏平衡 ≈979 步** ⇒ **门保留（结论对），理由改为
        #     「小模型每步省得少、摊不平编译开销」**；更贴切的判据是**预期生成长度**。
        #   （上游 84.2 vs 13.9 ms/step）—— 每层计算太轻时，编译的 guard/Python 开销压过
        #   kernel 收益；而 2B/24 层同一开关实测 **快 2.698×**（逐 token 一致）。
        #   1B 以内的小模型即便真能优化也会碰到边际效应 ⇒ 低于阈值直接跳过并告知。
        approx_params = self._estimate_total_params()
        if approx_params is not None and approx_params < COMPILE_MIN_PARAMS:
            logger.warning(
                "⚠️ 模型规模约 %.2fB < %.1fB ⇒ 自动跳过 torch.compile"
                "（实测该规模下编译反而慢；见 COMPILE_MIN_PARAMS 注释）",
                approx_params / 1e9, COMPILE_MIN_PARAMS / 1e9,
            )
            return
        if self._is_hybrid_architecture():
            # 2026-09-18 第五次更正后**放行**：官方 `is_compileable = False` 只影响
            # generate() 的**自动**编译决策，并不禁止手动 torch.compile；实测 hybrid
            # （Qwen3.5-2B）上 compile **快 1.57×** 且不随序列长度劣化。
            # 唯一风险是解释器退出期清理崩溃 —— 已在 _apply_compile 里登记 atexit 释放。
            logger.info("  ℹ️ 模型为 hybrid 层型：手动 compile 仍有效（实测快 1.57×），继续启用")
        self._apply_compile()

    # ================================================================
    # 对话补全（统一接口，内部委托给对应引擎）
    # ================================================================

    @_serialized_model_access
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
        对话补全 — 自动路由到当前活跃引擎。

        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 温度 (0-2)
            top_p: nucleus sampling
            stop: 停止词列表

        Returns:
            {"content": "模型回复文本", "usage": {...}, "tokens_per_second": float}
        """
        if self._engine_type == "llama_cpp":
            if self._llama_engine is None:
                raise RuntimeError("llama.cpp 引擎未加载，请先调用 load_model()")
            return self._llama_engine.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

        elif self._engine_type == "island":
            if self._island_engine is None:
                raise RuntimeError("孤岛引擎未连接，请先调用 load_model()")
            return self._island_engine.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

        elif self._engine_type == "pytorch":
            # PyTorch 路径: 使用 tokenizer + model.generate()
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("PyTorch 模型未加载，请先调用 load_model()")

            self.ensure_full_model()

            return self._pytorch_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

        else:
            raise RuntimeError(f"未知引擎类型: {self._engine_type}")

    @_serialized_model_access
    def chat_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delegate one local image request to the native MTMD engine."""
        if self._engine_type != "llama_cpp" or self._llama_engine is None:
            raise RuntimeError("native image chat requires a loaded llama.cpp engine")
        chat_image = getattr(self._llama_engine, "chat_image", None)
        if not callable(chat_image):
            raise RuntimeError("native MTMD image capability is unavailable")
        return chat_image(
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def native_vision_available(self) -> bool:
        """Return whether the active local engine has a registered MTMD vision path."""
        if self._engine_type != "llama_cpp" or self._llama_engine is None:
            return False
        capabilities = getattr(self._llama_engine, "get_capabilities", None)
        if not callable(capabilities):
            return False
        try:
            return bool(capabilities().get("vision"))
        except Exception:
            return False

    def _pytorch_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """PyTorch 路径的对话补全实现。"""
        t0 = time.time()

        # 使用 tokenizer 的 chat template 构建输入
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Qwen tokenizer 的 chat_template 可能不同，手动构建
            input_text = self._build_qwen_prompt(messages)

        stop_sequences = self._merge_stop_sequences(stop)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.get_device()) for k, v in inputs.items()}
        generation_kwargs = dict(kwargs)
        generation_kwargs.setdefault(
            "eos_token_id",
            self._get_generation_eos_token_ids(stop_sequences),
        )
        cancel_event = generation_kwargs.pop("_cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            return {
                "content": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "model": self.active_model_id,
                "finish_reason": "cancelled",
                "tokens_per_second": 0,
            }
        stop_criteria = self._build_stop_criteria(
            stop_sequences,
            inputs["input_ids"].shape[1],
            cancel_event=cancel_event,
        )
        if stop_criteria is not None:
            generation_kwargs.setdefault("stopping_criteria", stop_criteria)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
            )

        # 解码生成部分（去除输入部分）
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        content = self._decode_generated_ids(generated_ids, stop_sequences)

        elapsed = time.time() - t0
        completion_tokens = len(generated_ids)
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            f"推理完成 (PyTorch): {completion_tokens} tokens / {elapsed:.1f}s "
            f"= {tok_per_sec:.1f} tok/s"
        )
        cfg = mc.get_model_config(self._active_model_id) if self._active_model_id else None
        model_name = cfg.name if cfg else (self._active_model_id or MODEL_NAME)

        return {
            "content": content,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": completion_tokens,
                "total_tokens": input_len + completion_tokens,
            },
            "model": model_name,
            "finish_reason": "stop",
            "tokens_per_second": round(tok_per_sec, 1),
        }

    @_serialized_model_stream
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ):
        """
        流式对话补全。

        Yields:
            str: 增量文本 chunk
        """
        if self._engine_type == "llama_cpp":
            if self._llama_engine is None:
                raise RuntimeError("llama.cpp 引擎未加载")
            yield from self._llama_engine.chat_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )
        elif self._engine_type == "island":
            if self._island_engine is None:
                raise RuntimeError("孤岛引擎未连接")
            yield from self._island_engine.chat_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )
        elif self._engine_type == "pytorch":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("PyTorch 模型未加载，请先调用 load_model()")

            self.ensure_full_model()

            try:
                from transformers import TextIteratorStreamer
            except ImportError:
                # 降级：transformers 版本过旧，回退到非流式
                logger.warning("TextIteratorStreamer 不可用，回退到非流式")
                result = self._pytorch_chat(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, stop=stop, **kwargs,
                )
                yield result["content"]
                return

            # 构建输入
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                input_text = self._build_qwen_prompt(messages)

            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(self.get_device()) for k, v in inputs.items()}
            stop_sequences = self._merge_stop_sequences(stop)

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=False,
            )
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self._get_generation_eos_token_ids(stop_sequences),
                streamer=streamer,
            )
            cancel_event = kwargs.pop("_cancel_event", None)
            stop_criteria = self._build_stop_criteria(
                stop_sequences,
                inputs["input_ids"].shape[1],
                cancel_event=cancel_event,
            )
            if stop_criteria is not None:
                generation_kwargs["stopping_criteria"] = stop_criteria

            import threading
            t0 = time.time()
            thread = threading.Thread(
                target=self.model.generate, kwargs=generation_kwargs,
            )
            thread.start()

            chunk_count = 0
            try:
                for text in self._iter_stream_until_stop(streamer, stop_sequences):
                    if text:
                        chunk_count += 1
                        yield text
            finally:
                # A disconnected stream must not release the model lock while
                # generate() is still mutating/reading the current model.
                thread.join()
            elapsed = time.time() - t0
            logger.info(
                f"流式推理完成 (PyTorch): {chunk_count} chunks / {elapsed:.1f}s"
            )
        else:
            raise RuntimeError(f"未知引擎类型: {self._engine_type}")

    def _build_qwen_prompt(self, messages: List[Dict[str, str]]) -> str:
        """手动构建 Qwen ChatML 格式 prompt（fallback）。"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def _merge_stop_sequences(stop: List[str] = None) -> List[str]:
        """Return caller stop sequences plus ChatML sentinels, preserving order."""
        merged: List[str] = []
        for value in (stop or []) + CHATML_STOP_SEQUENCES:
            if value and value not in merged:
                merged.append(value)
        return merged

    def _get_generation_eos_token_ids(self, stop_sequences: List[str]) -> Optional[Union[List[int], int]]:
        """Map known stop strings to token ids when the tokenizer has them."""
        ids: List[int] = []
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if eos_id is not None:
            ids.append(int(eos_id))
        for text in stop_sequences:
            convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
            if convert is None:
                continue
            token_id = convert(text)
            if (
                isinstance(token_id, int)
                and token_id >= 0
                and token_id != unk_id
                and token_id not in ids
            ):
                ids.append(token_id)
        if not ids:
            return None
        return ids if len(ids) > 1 else ids[0]

    def _build_stop_criteria(self, stop_sequences: List[str], prompt_len: int,
                             cancel_event: threading.Event = None):
        """Stop generation when the new token tail matches any configured stop string."""
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except ImportError:
            return None

        stop_token_ids: List[List[int]] = []
        for text in stop_sequences:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            if encoded:
                stop_token_ids.append([int(token_id) for token_id in encoded])

        if not stop_token_ids and cancel_event is None:
            return None

        class StopOnTokenSequences(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                if cancel_event is not None and cancel_event.is_set():
                    return True
                generated = input_ids[0, prompt_len:].tolist()
                if not generated:
                    return False
                for seq in stop_token_ids:
                    if len(generated) >= len(seq) and generated[-len(seq):] == seq:
                        return True
                return False

        return StoppingCriteriaList([StopOnTokenSequences()])

    @staticmethod
    def _strip_known_special_tokens(text: str, trim: bool = True) -> str:
        """Remove ChatML/control tokens that may not be registered as special tokens."""
        if not text:
            return text
        result = text
        for marker in CHATML_STOP_SEQUENCES:
            result = result.replace(marker, "")
        result = re.sub(r"<\s*\|im_(?:start|end)\|\s*>", "", result)
        return result.strip() if trim else result

    def _decode_generated_ids(self, generated_ids, stop_sequences: List[str]) -> str:
        """Decode completion text, cut at stop text, then remove leaked control tokens."""
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        cut_idx = None
        for marker in stop_sequences:
            idx = content.find(marker)
            if idx != -1:
                cut_idx = idx if cut_idx is None else min(cut_idx, idx)
        if cut_idx is not None:
            content = content[:cut_idx]
        return self._strip_known_special_tokens(content)

    def _iter_stream_until_stop(self, streamer, stop_sequences: List[str]):
        """Yield streamed text while withholding enough tail to detect split stop strings."""
        max_stop_len = max((len(s) for s in stop_sequences), default=0)
        pending = ""
        for text in streamer:
            if not text:
                continue
            pending += text
            cut_idx = None
            for marker in stop_sequences:
                idx = pending.find(marker)
                if idx != -1:
                    cut_idx = idx if cut_idx is None else min(cut_idx, idx)
            if cut_idx is not None:
                chunk = self._strip_known_special_tokens(pending[:cut_idx], trim=False)
                if chunk:
                    yield chunk
                return
            if max_stop_len and len(pending) > max_stop_len:
                emit = pending[:-max_stop_len]
                pending = pending[-max_stop_len:]
                emit = self._strip_known_special_tokens(emit, trim=False)
                if emit:
                    yield emit
        pending = self._strip_known_special_tokens(pending, trim=False)
        if pending:
            yield pending

    # ================================================================
    # 模型层级拆分（PyTorch 专用）
    # ================================================================

    def split_model(self, layer_config: Tuple[int, int]) -> nn.Module:
        """
        [已废弃] 模型层级拆分 — 请使用 load_layer_range() 代替。

        此方法在加载完整模型后进行运行时拆分，已被 load_layer_range()
        的"加载即裁剪"方案取代（零额外显存峰值）。

        保留此方法仅为向后兼容，新代码请直接使用:
            mgr.load_layer_range(start, end, has_embedding=..., has_lm_head=...)
            result = mgr.forward_layers(input_ids=...)

        Args:
            layer_config: (start_layer, end_layer) 起止层编号，左闭右开

        Returns:
            None（始终抛 NotImplementedError）
        """
        raise NotImplementedError(
            "split_model() 已废弃，请使用 load_layer_range() 代替。\n"
            "示例: mgr.load_layer_range(start, end, has_embedding=True, has_lm_head=False)"
        )

    # ================================================================
    # 前向推理（PyTorch 分布式专用）
    # ================================================================

    def _forward_qwen_layers(
        self,
        *,
        input_ids: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: tuple = None,
        use_cache: bool = True,
        apply_lm_head: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run one original Qwen ``transformer.h`` segment."""
        transformer = self.model.transformer
        layers = transformer.h
        has_embed = getattr(transformer, "wte", None) is not None
        has_lm_head = getattr(self.model, "lm_head", None) is not None
        device = self.get_device()
        dtype = self._layer_compute_dtype()

        with torch.no_grad():
            if input_ids is not None:
                if not has_embed:
                    raise RuntimeError("当前 Qwen 分层不含 Embedding，请传入 hidden_states")
                input_ids = input_ids.to(device)
                batch_size, seq_len = input_ids.shape
                hidden_states = transformer.wte(input_ids).to(dtype=dtype)
                hidden_states = transformer.drop(hidden_states)
            else:
                batch_size, seq_len = hidden_states.shape[:2]
                hidden_states = hidden_states.to(device=device, dtype=dtype)

            local_layer_count = len(layers)
            if past_key_values is not None and len(past_key_values) != local_layer_count:
                raise RuntimeError(
                    f"Qwen KV cache 层数不匹配: cache={len(past_key_values)}, "
                    f"local={local_layer_count}"
                )
            local_past = (
                tuple(past_key_values)
                if past_key_values is not None
                else tuple([None] * local_layer_count)
            )

            past_length = 0
            first_past = next((item for item in local_past if item is not None), None)
            if first_past is not None:
                if getattr(transformer, "use_cache_quantization", False):
                    past_length = int(first_past[0][0].shape[2])
                else:
                    past_length = int(first_past[0].shape[1])
            kv_seq_len = past_length + seq_len

            prepared_mask = None
            if attention_mask is not None:
                prepared_mask = attention_mask.to(device).view(batch_size, -1)
                prepared_mask = prepared_mask[:, None, None, :].to(dtype=dtype)
                prepared_mask = (1.0 - prepared_mask) * torch.finfo(dtype).min

            if transformer.training or not transformer.use_dynamic_ntk:
                ntk_alpha_list = [1.0]
            elif past_length > 0:
                ntk_alpha_list = list(
                    getattr(transformer.rotary_emb, "_ntk_alpha_cached_list", None)
                    or [transformer.get_ntk_alpha(kv_seq_len)]
                )
            else:
                ntk_alpha_list = [transformer.get_ntk_alpha(kv_seq_len)]
            transformer.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
            rotary_pos_emb_list = [
                transformer.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
                for ntk_alpha in ntk_alpha_list
            ]

            presents = []
            for block, layer_past in zip(layers, local_past):
                outputs = block(
                    hidden_states,
                    rotary_pos_emb_list=rotary_pos_emb_list,
                    layer_past=layer_past,
                    attention_mask=prepared_mask,
                    head_mask=None,
                    use_cache=use_cache,
                    output_attentions=False,
                )
                hidden_states = outputs[0]
                if use_cache:
                    presents.append(outputs[1])

            result: Dict[str, torch.Tensor] = {}
            if has_lm_head and apply_lm_head:
                result["logits"] = self.model.lm_head(transformer.ln_f(hidden_states))
            else:
                result["hidden_states"] = hidden_states
            if use_cache:
                result["past_key_values"] = tuple(presents)
            return result

    @_serialized_model_access
    def layer_offload_observation(self):
        """Observe CPU source ownership and the largest target execution unit."""
        offload = getattr(self, "_layer_offload", None)
        if offload is None or self.model is None or offload.active:
            raise ValueError("layer_offload_runtime_unavailable")
        transformer, layers_attr, embedding_attr = _locate_text_transformer(self.model)
        units = list(getattr(transformer, layers_attr))
        units += [getattr(transformer, name, None)
                  for name in (embedding_attr, "norm", "rotary_emb")]
        units.append(getattr(self.model, "lm_head", None))
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        if any(t.device.type != "cpu" for t in tensors):
            raise ValueError("layer_offload_source_device_mismatch")
        size = torch.empty((), dtype=offload.dtype).element_size()
        derived = {id(tensor) for module in self.model.modules()
                   for name, tensor in module.named_buffers(recurse=False)
                   if name in module._non_persistent_buffers_set}
        def target_size(tensor):
            return size if tensor.is_floating_point() and id(tensor) not in derived else tensor.element_size()
        stage_weights = []
        for unit in units:
            if unit is not None:
                unique = {id(t): t for t in (*unit.parameters(), *unit.buffers())}
                stage_weights.append(sum(t.numel() * target_size(t) for t in unique.values()))
        unique = {id(t): t for t in tensors}
        return {**offload.observation(),
                "resident_source_bytes": sum(t.numel() * t.element_size() for t in unique.values()),
                "device_weight_bytes": max(stage_weights, default=0),
                "host_conversion_bytes": max((t.numel() * size for t in unique.values()
                                               if t.is_floating_point() and t.dtype != offload.dtype
                                               and id(t) not in derived), default=0),
                "target_weight_bytes": sum(t.numel() * target_size(t) for t in unique.values())}

    def _layer_compute_dtype(self):
        offload = getattr(self, "_layer_offload", None)
        return offload.dtype if offload is not None else next(self.model.parameters()).dtype

    def _call_layer_unit(self, module, *args, **kwargs):
        offload = getattr(self, "_layer_offload", None)
        if offload is None:
            return module(*args, **kwargs)
        return offload.call(module, *args, **kwargs)

    @_serialized_model_access
    def forward_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the active architecture's final norm and language-model head."""
        if self.model is None or self._engine_type != "pytorch":
            raise RuntimeError("PyTorch 模型未加载，无法执行 LM Head")
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("当前分层不含 LM Head")
        model_type = str(getattr(self.model.config, "model_type", "") or "").lower()
        if model_type == "qwen2":
            norm = getattr(self.model.model, "norm", None)
        elif model_type == "qwen":
            norm = getattr(self.model.transformer, "ln_f", None)
        else:
            norm = None
        # ★ #31 M5（2026-09-26）：再加一层**按架构属性**的兜底定位，而不是只认两个硬编码的
        #   `model_type`。hybrid（`qwen3_5_text`）此前在这里直接抛
        #   「模型架构 qwen3_5_text 缺少最终 Norm」⇒「master 保留 LM Head、层在 worker」
        #   那条路径对它完全不可用。
        #   与 `forward_layers` 的 Step 7（`transformer.norm(hidden_states)`，约 :4076）**同源**
        #   —— 用的是同一个 `_locate_text_transformer`，所以两条路径不会各自漂移。
        #   既有 `qwen2` / `qwen` 分支**保持不变**（先命中就先返回 ⇒ 行为零变化）。
        if norm is None:
            try:
                transformer, _layers_attr, _embed_attr = _locate_text_transformer(self.model)
            except RuntimeError:
                transformer = None
            if transformer is not None:
                norm = getattr(transformer, "norm", None)
        if norm is None:
            raise RuntimeError(f"模型架构 {model_type or 'unknown'} 缺少最终 Norm")
        device = self.get_device()
        dtype = self._layer_compute_dtype()
        with torch.no_grad():
            states = hidden_states.to(device=device, dtype=dtype)
            states = self._call_layer_unit(norm, states)
            return self._call_layer_unit(lm_head, states)

    @_serialized_model_access
    def forward_layers(
        self,
        input_ids: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: tuple = None,
        use_cache: bool = True,
        apply_lm_head: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        执行本节点层范围的单步前向传播（分布式流水线节点专用）。

        三种节点角色:
         - 首节点 (has_embedding): input_ids → embed_tokens → layers → hidden_states
         - 中间节点:              hidden_states → layers → hidden_states
         - 末节点 (has_lm_head):  hidden_states → layers → norm → lm_head → logits

        严格遵循 Qwen2Model.forward() 的调用链，复用其 RoPE、causal_mask、
        和 position_embeddings 计算逻辑，确保分布式与单机推理输出一致。

        **KV Cache 支持 (Phase 3 — 增量解码)**:
         - Prefill (past_key_values=None): 处理完整输入序列，构建 KV cache
         - Decode (past_key_values 存在): 仅处理新 token，基于缓存的 KV 增量计算
         - KV Cache 形状: 每层 (batch, num_heads, total_seq_len, head_dim)
         - past_key_values 索引 0..(N-1) 对应本节点的 N 个本地层

        Args:
            input_ids: 输入 token IDs (batch, seq_len)，仅首节点使用
            hidden_states: 中间隐藏状态 (batch, seq_len, hidden_dim)，中间/末节点使用
            attention_mask: 2D 注意力掩码 (batch, seq_len)，1=有效，0=填充
            position_ids: 位置 ID (batch, seq_len)，None 则自动生成
            past_key_values: 已缓存的 KV cache，tuple of (key, value) per local layer
            use_cache: 是否返回新的 KV cache（decode 阶段应为 True）
            apply_lm_head: 当前模型含 LM Head 时是否立即执行；master 首段保留
                LM Head 但需先把 hidden_states 发给 worker，因此传 False

        Returns:
            {"hidden_states": Tensor}       — 非末节点，形状 (batch, seq_len, hidden_dim)
            {"logits": Tensor}             — 末节点，形状 (batch, seq_len, vocab_size)
            当 use_cache=True 时附加:
            {"past_key_values": tuple}     — 更新后的 KV cache（每层一个 (k,v) 元组）
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model() 或 load_layer_range()")

        if self._engine_type != "pytorch":
            raise RuntimeError("forward_layers 仅支持 PyTorch 引擎")

        # ---- 输入校验 ----
        if input_ids is None and hidden_states is None:
            raise ValueError("必须提供 input_ids 或 hidden_states 之一")
        if input_ids is not None and hidden_states is not None:
            raise ValueError("input_ids 和 hidden_states 不能同时提供")

        # ★ B9：把「**空的** cache 对象」规范化成 None。调用方常传 `DynamicCache()`（而不是
        #   `None`），此时 `len(cache) == 0` ⇒ 会报「KV cache 层数不匹配: cache=0, local=N」，
        #   或（hybrid）直接 IndexError —— 两者报错都**不指向根因**。视同 None 由本方法自建即可。
        past_key_values = _normalize_past_key_values(past_key_values)

        model_type = str(getattr(self.model.config, "model_type", "") or "").lower()
        if model_type == "qwen":
            return self._forward_qwen_layers(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                apply_lm_head=apply_lm_head,
            )
        # ★ A3：Qwen 系（qwen2 / qwen3 / qwen3_5 / qwen3_5_text）共用「层序列 + 按层 mask」契约；
        #   hybrid（config 里有 layer_types）会在下面按层类型分别构造 mask。
        if model_type not in _QWEN_LAYER_RANGE_TYPES:
            raise RuntimeError(
                f"forward_layers 不支持模型架构: {model_type or 'unknown'}"
            )

        device = self.get_device()
        # ★ A3：用统一探测取文本 Transformer（支持 Qwen3.5 的 model.model.language_model）。
        #   到这里 model_type 已限定为本模块支持的 Qwen 系，其层属性均为 `layers`。
        transformer, _layers_attr, _embed_attr = _locate_text_transformer(self.model)
        dtype = self._layer_compute_dtype()

        # 检测节点角色
        has_embed = (
            hasattr(transformer, 'embed_tokens')
            and transformer.embed_tokens is not None
        )
        has_lm_head = (
            hasattr(self.model, 'lm_head')
            and self.model.lm_head is not None
        )

        batch_size: int
        seq_len: int

        # ============================================================
        # 整个前向传播在 torch.no_grad() 下执行，避免梯度追踪开销。
        # ============================================================
        with torch.no_grad():
            # ============================================================
            # Step 1: Embedding（首节点）
            # ============================================================
            if input_ids is not None:
                if not has_embed:
                    raise RuntimeError(
                        "当前节点不含 Embedding 层（加载时未设置 has_embedding=True），"
                        "请传入 hidden_states"
                    )
                input_ids = input_ids.to(device)
                batch_size, seq_len = input_ids.shape
                hidden_states = self._call_layer_unit(transformer.embed_tokens, input_ids).to(dtype=dtype)
            else:
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]
                if hidden_states.device != device:
                    hidden_states = hidden_states.to(device)
                if hidden_states.dtype != dtype:
                    hidden_states = hidden_states.to(dtype)

            # ============================================================
            # Step 2: KV Cache 初始化 — DynamicCache with local indices
            # ============================================================
            # Qwen2SdpaAttention / FlashAttention2 使用 DynamicCache，
            # 其内部以 self_attn.layer_idx 为索引存储 key/value。
            #
            # ★ 分布式节点仅加载部分层，若使用原始 global_idx（如 8-15），
            #    DynamicCache 会产生稀疏空洞（需 None 填充），导致
            #    get_seq_length() 对空洞条目抛 TypeError。
            #
            #    解决方案: 临时将各层的 self_attn.layer_idx 改为本地索引
            #    (0, 1, 2, ...)，使 DynamicCache 按连续本地索引存储。
            #    forward 后恢复原始 global_idx，确保后续调用不受影响。
            from transformers.cache_utils import DynamicCache

            # 保存并覆盖层索引为本地连续编号
            # ★ 先保存原始索引，再在 try 块内补丁，确保 finally 无论何路径都恢复
            # ★ A4：走编译版层循环时，layer_idx 已在 _apply_compile 里**永久**本地化
            #   （compile 把属性当 guard，每次临时改再恢复会反复重编译）⇒ 此时不打补丁。
            _use_compiled_loop = getattr(self, "_compiled_layer_loop", None) is not None
            # ★ A3：layer_idx 的持有者属性名随架构而变（Qwen2 是 `self_attn`；
            #   Qwen3.5 的 linear_attention 层是 `linear_attn`）⇒ 按「带 layer_idx 的子模块」
            #   通用处理，避免写死属性名。
            layer_index_holders = [
                (layer, _layer_idx_holders(layer)) for layer in transformer.layers
            ]
            saved_layer_indices: list = [
                (holder, holder.layer_idx)
                for _layer, holders in layer_index_holders
                for holder in holders
            ] if not _use_compiled_loop else []

            try:
                # ---- 补丁 layer_idx 为本地索引（必须在 try 内，确保异常时恢复） ----
                if not _use_compiled_loop:
                    for local_idx, (_layer, holders) in enumerate(layer_index_holders):
                        for holder in holders:
                            holder.layer_idx = local_idx

                import inspect
                first_layer_forward = transformer.layers[0].forward if len(transformer.layers) else None
                layer_forward_params = (
                    inspect.signature(first_layer_forward).parameters
                    if first_layer_forward is not None else {}
                )
                cache_arg_name = (
                    "past_key_values"
                    if "past_key_values" in layer_forward_params
                    else "past_key_value"
                    if "past_key_value" in layer_forward_params
                    else None
                )

                if use_cache:
                    if past_key_values is not None:
                        # ★ 2026-09-21（修 hybrid 增量解码）：若调用方回传的是**cache 对象**
                        #   （上一步 `result["cache"]`），**直接复用** —— 它带着 hybrid
                        #   （Qwen3.5）linear_attention 层的 recurrent state。转成 (k,v)
                        #   tuple 会把这类状态丢成 None ⇒ decode 第 2 步起退化为复读
                        #   （实测：`Paris of of of of …`，共享前缀 1/16）。
                        if not isinstance(past_key_values, (tuple, list)):
                            cache = past_key_values
                            n_local = len(transformer.layers)
                            if len(getattr(cache, "layers", [])) < n_local:
                                raise RuntimeError(
                                    f"cache 对象层槽不足: {len(getattr(cache, 'layers', []))} "
                                    f"< local={n_local}"
                                )
                        else:
                            # Decode: tuple of (k,v) → DynamicCache（本地索引 0..N-1）
                            # Phase 4.3: 验证缓存层数与本地层数一致
                            n_local = len(transformer.layers)
                            if len(past_key_values) != n_local:
                                raise RuntimeError(
                                    f"Qwen2 KV cache 层数不匹配: "
                                    f"cache={len(past_key_values)}, local={n_local}"
                                )
                            # ★ A3：传 config —— hybrid（如 Qwen3.5）需要按 layer_types 建出
                            #   linear/full 混合层；空 DynamicCache() 会让 `cache.layers[layer_idx]`
                            #   越界（IndexError）。旧版 transformers 无该关键字 ⇒ 辅助函数已兼容。
                            cache = _new_dynamic_cache(transformer.config)
                            for layer_idx, item in enumerate(past_key_values):
                                if item is None:
                                    # ★ B14（修 A3 遗留）：hybrid 的 linear_attention 层**没有 KV**
                                    #   （用 recurrent state）⇒ 收集侧留下了 None 占位以保持下标对齐，
                                    #   这里跳过它们即可（它们由模型内部的递归状态自行维护）。
                                    continue
                                k, v = item
                                cache.update(k, v, layer_idx)
                    else:
                        # Prefill: 创建空 DynamicCache（★ A3：传 config，理由同上）
                        cache = _new_dynamic_cache(transformer.config)
                else:
                    cache = None

                # ---- 从 cache 获取 past_seen_tokens ----
                if cache is not None:
                    try:
                        past_seen_tokens = cache.get_seq_length()
                    except (AttributeError, TypeError, IndexError) as e:
                        # Phase 4.4: DynamicCache 内部 API 变更 → 降级为 0
                        # (不捕获 MemoryError/SystemExit 等真实异常)
                        logger.debug(f"cache.get_seq_length() 失败: {e}")
                        past_seen_tokens = 0
                else:
                    past_seen_tokens = 0

                # ============================================================
                # Step 3: position_ids / cache_position
                # ============================================================
                if position_ids is None:
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + seq_len,
                        device=device
                    )
                    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
                else:
                    position_ids = position_ids.to(device)
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + seq_len,
                        device=device
                    )

                # ============================================================
                # Step 4: causal_mask — 复用 Qwen2 内置逻辑
                # ============================================================
                # transformers≥5.x: _update_causal_mask 已移除，改用 create_causal_mask
                # transformers 4.x: 无此函数，回退到手动构建因果掩码
                # 该函数根据 attention 实现自动选择:
                #   flash_attention_2 → None（flash 内核自行处理因果掩码）
                #   sdpa + 纯因果 → None（SDPA is_causal 路径）
                #   eager / 含填充 → 4D (batch,1,seq,seq) 因果掩码
                # ★ A3：hybrid 判定。**不能只看「有无 layer_types」** —— transformers 5.x 给
                #   Qwen2Config 也加上了 layer_types（实测 24/24 全是 full_attention），
                #   只判断有无会把纯 full-attention 模型误判为 hybrid，进而出错。
                #   非 hybrid 时显式置 None，使下游的 `if layer_types:` 分支自动走原路径。
                #   放在 try/else 之外 —— 任何分支（含 import 失败回退）下都必须有定义。
                _cfg_layer_types = getattr(transformer.config, "layer_types", None)
                layer_types = (
                    _cfg_layer_types if _is_hybrid_layer_types(_cfg_layer_types) else None
                )
                mask_fallback_error: Optional[Exception] = None
                mask_parameters = {}
                try:
                    from transformers.models.qwen2.modeling_qwen2 import create_causal_mask
                except (ImportError, AttributeError) as exc:
                    mask_fallback_error = exc
                else:
                    try:
                        mask_parameters = inspect.signature(
                            create_causal_mask
                        ).parameters
                    except (TypeError, ValueError) as exc:
                        mask_fallback_error = exc

                if mask_fallback_error is None:
                    mask_kwargs = {
                        "config": transformer.config,
                        "attention_mask": (
                            attention_mask.to(device)
                            if attention_mask is not None else None
                        ),
                        "past_key_values": cache,
                        "position_ids": position_ids,
                    }
                    if "input_embeds" in mask_parameters:
                        mask_kwargs["input_embeds"] = hidden_states
                    elif "inputs_embeds" in mask_parameters:
                        mask_kwargs["inputs_embeds"] = hidden_states
                    else:
                        mask_fallback_error = TypeError(
                            "create_causal_mask 缺少已知的输入张量参数"
                        )
                    if (
                        mask_fallback_error is None
                        and "cache_position" in mask_parameters
                    ):
                        mask_kwargs["cache_position"] = cache_position
                    # ★ A3：hybrid（如 Qwen3.5）按 layer_types 区分层型 —— full_attention 用
                    #   causal mask、linear_attention 用 recurrent mask ⇒ 建映射、层循环逐层取用。
                    if mask_fallback_error is None:
                        # Runtime input/cache failures are not version mismatches.
                        causal_mask = create_causal_mask(**mask_kwargs)
                        if layer_types:  # ★ A3/B14：hybrid ⇒ 按层型建 mask「元组」
                            try:
                                from transformers.masking_utils import (
                                    create_recurrent_attention_mask,
                                )
                            except ImportError as exc:  # 版本不符时明确报错，不静默出错
                                raise RuntimeError(
                                    "hybrid 架构需要 transformers.masking_utils."
                                    f"create_recurrent_attention_mask: {exc}"
                                ) from exc
                            # ★ B14：用「元组 + 每层索引」而非 dict —— 顺序由共享辅助
                            #   `_hybrid_mask_index_per_layer()` 给出，与 `_apply_compile` 编译进
                            #   层循环的索引序列**同源**（顺序错配会让 mask 张冠李戴）；
                            #   元组 + 静态索引也比 dict 查找更利于 torch.compile 的 guard。
                            _mask_order, mask_index_per_layer = _hybrid_mask_index_per_layer(
                                layer_types
                            )
                            _known_masks = {
                                "full_attention": causal_mask,
                                "linear_attention": create_recurrent_attention_mask(
                                    **mask_kwargs
                                ),
                            }
                            # 未预期的层型会直接 KeyError（fail-loud），避免静默用错 mask
                            causal_mask = tuple(
                                _known_masks[name] for name in _mask_order
                            )

                if mask_fallback_error is not None:
                    # transformers 4.x 回退：手动构建 4D 因果掩码
                    logger.debug(
                        "create_causal_mask unavailable or incompatible; "
                        "using manual mask: %s",
                        mask_fallback_error,
                    )
                    query_len = hidden_states.shape[1]
                    target_len = past_seen_tokens + query_len
                    causal_mask = torch.full(
                        (query_len, target_len),
                        float('-inf'),
                        device=device,
                        dtype=hidden_states.dtype,
                    )
                    causal_mask = torch.triu(
                        causal_mask, diagonal=past_seen_tokens + 1
                    )
                    causal_mask = causal_mask[None, None, :, :].expand(
                        hidden_states.shape[0], 1, query_len, target_len
                    )
                    if attention_mask is not None:
                        attn_mask = attention_mask.to(device)
                        # attn_mask shape: (batch, seq_len) → (batch, 1, 1, seq_len)
                        attn_mask = attn_mask[:, None, None, :]
                        causal_mask = causal_mask.masked_fill(
                            attn_mask == 0, float('-inf')
                        )

                # ============================================================
                # Step 5: position_embeddings — RoPE 旋转位置编码
                # ============================================================
                # Qwen2Model.rotary_emb 会将 position_ids 转换为 cos/sin 元组，
                # 各 attention 层内部通过 apply_rotary_pos_emb 应用到 Q/K 上。
                # 此步在层循环外仅计算一次，所有层共享同一份 position_embeddings。
                # ★ A3：hybrid（Qwen3.5）用 MRoPE —— rotary 需要 (3, bs, seq)，而层需要
                #   (bs, seq)。与 transformers 做法一致：把 position_ids 展开成 (4, bs, seq)，
                #   rotary 取后 3 维、层取第 0 维（text 位置）。
                layer_position_ids = position_ids
                if layer_types:
                    if position_ids.ndim == 2:
                        position_ids = position_ids[None, ...].expand(
                            4, position_ids.shape[0], -1
                        )
                    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                        layer_position_ids = position_ids[0]
                        position_ids = position_ids[1:]
                position_embeddings = self._call_layer_unit(transformer.rotary_emb, hidden_states, position_ids)

                # ============================================================
                # Step 6: Transformer 层前向传播（支持 KV cache）
                # ============================================================
                # DynamicCache 由 SDPA/FlashAttention 在 forward 时原地更新，
                # 每层的 key/value 按 layer_idx 写入 DynamicCache。
                # ★ A4：可用时走**编译版层循环**（前置/后置逻辑完全不变 ⇒ 语义一致）。
                # ★ B14：hybrid 也能走编译层循环了 —— `_LayerLoop` 现在按「每层 mask 索引」
                #   从 causal_mask **元组**里取本层 mask（索引与 `_hybrid_mask_index_per_layer()`
                #   同源）。此前 hybrid 被整体跳过，因此拿不到 A4 的 1.6–2.1× 收益。
                layer_loop = getattr(self, "_compiled_layer_loop", None)
                if layer_loop is not None:
                    hidden_states = layer_loop(
                        hidden_states,
                        attention_mask=causal_mask,
                        # ★ 层要的是 text 位置 (bs, seq)；MRoPE 的 (3, bs, seq) 只给 rotary 用
                        position_ids=layer_position_ids,
                        position_embeddings=position_embeddings,
                        cache_position=cache_position,
                        use_cache=use_cache,
                        cache=cache,
                    )
                else:
                    for i, layer in enumerate(transformer.layers):
                        layer_mask = (
                            causal_mask[mask_index_per_layer[i]]
                            if layer_types else causal_mask
                        )
                        layer_kwargs = {
                            "attention_mask": layer_mask,
                            # ★ A3：层用 text 位置 (bs, seq)，与 rotary 的 (3, bs, seq) 区分
                            "position_ids": layer_position_ids,
                            "position_embeddings": position_embeddings,
                            "use_cache": use_cache,
                            "cache_position": cache_position,
                        }
                        if cache_arg_name is not None:
                            layer_kwargs[cache_arg_name] = cache
                        layer_output = self._call_layer_unit(layer, hidden_states, **layer_kwargs)
                        # transformers≥5.x: DecoderLayer 直接返回 tensor
                        # transformers 4.x: 返回 (hidden_states, present_key_value) 元组
                        if isinstance(layer_output, tuple):
                            hidden_states = layer_output[0]
                        else:
                            hidden_states = layer_output
                        del layer_output

                # ============================================================
                # Step 7: 最终 Norm + LM Head（末节点）
                # ============================================================
                result: Dict[str, torch.Tensor] = {}

                if has_lm_head and apply_lm_head:
                    hidden_states = self._call_layer_unit(transformer.norm, hidden_states)
                    logits = self._call_layer_unit(self.model.lm_head, hidden_states)
                    result["logits"] = logits
                else:
                    result["hidden_states"] = hidden_states

                # ---- KV Cache: 转为 tuple 存储 ----
                # ★ B14（修 A3 遗留）：hybrid 的 linear_attention 层**不写 KV**（用 recurrent
                #   state）⇒ `cache.layers` 里那些位置是 None。**必须保留 None 占位**，否则
                #   tuple 的下标会与本地层号错位 ⇒ decode 时「层数不匹配」、mask/cache 张冠李戴。
                if use_cache and cache is not None:
                    cache_items = []
                    if hasattr(cache, "layers"):
                        # ★ B14：`DynamicCache(config=...)` 会按 **config** 的层数建槽，而分段加载时
                        #   config 仍是**完整模型**的层数（例如 4 层模型取 2 层）⇒ 多出来的槽是 None。
                        #   这里只取**本段实际拥有的层数**，与 `transformer.layers` 对齐
                        #   （layer_idx 已被补丁成本地索引，用到的正是前 N 个槽）。
                        for layer_cache in cache.layers[: len(transformer.layers)]:
                            if layer_cache is None:
                                cache_items.append(None)
                                continue
                            keys = getattr(layer_cache, "keys", None)
                            values = getattr(layer_cache, "values", None)
                            cache_items.append(
                                (keys, values) if keys is not None and values is not None else None
                            )
                    else:
                        key_cache = getattr(cache, "key_cache", [])
                        value_cache = getattr(cache, "value_cache", [])
                        for keys, values in zip(key_cache, value_cache):
                            cache_items.append(
                                (keys, values) if keys is not None and values is not None else None
                            )
                    if cache_items:
                        result["past_key_values"] = tuple(cache_items)
                    # ★ 2026-09-21（修 hybrid 增量解码）：**必须**同时给出 cache 对象本身。
                    #   上面的 (k,v) tuple **只承载 KV** —— hybrid（Qwen3.5）的
                    #   linear_attention 层用 **recurrent state**，其 `layer_cache` 没有
                    #   `.keys`/`.values` ⇒ 在 tuple 里只能是 `None` ⇒ **状态被丢弃**，
                    #   下一步 decode 等于从零开始（实测整模上游退化成 `Paris of of of of …`，
                    #   共享前缀 1/16）。调用方应优先用本键，并在下一步**原样回传**。
                    #   ⚠️ 对 hybrid 而言 `past_key_values`（tuple）是**有损**的兼容通道，
                    #   仅适用于纯 attention 模型。
                    result["cache"] = cache

                return result
            finally:
                # ★ 无论成功/异常，必须恢复原始 layer_idx
                #    （finally 覆盖了 layer_idx 补丁 + create_causal_mask +
                #      rotary_emb + 层循环 + LM Head 全路径）
                # ★ A3：saved_layer_indices 现在存的是 (holder, 原值) 对（见上方补丁处）
                for holder, orig_idx in saved_layer_indices:
                    holder.layer_idx = orig_idx

    # ================================================================
    # 工具方法
    # ================================================================

    @property
    def engine_type(self) -> str:
        """当前使用的推理引擎类型。"""
        return self._engine_type

    @property
    def is_llama_cpp(self) -> bool:
        """是否使用 llama.cpp 引擎。"""
        return self._engine_type == "llama_cpp"

    @property
    def is_pytorch(self) -> bool:
        """是否使用 PyTorch 引擎。"""
        return self._engine_type == "pytorch"

    @property
    def is_island(self) -> bool:
        """是否使用 TP 孤岛引擎。"""
        return self._engine_type == "island"

    @property
    def backend_id(self) -> str:
        return self.engine_type

    @property
    def capabilities(self):
        return backend_capabilities(self.engine_type)

    def supports(self, capability: str) -> bool:
        # 票 15a：能力必须反映**真实执行能力**。llama-server 子进程虽然同属
        # llama.cpp 家族（`engine_type == "llama_cpp"`，槽位也复用 `_llama_engine`），
        # 但它**不承担层段接力** —— 直接沿用 `self.capabilities`（llama_cpp 白名单里
        # `FORWARD_LAYERS` 为 True）会让调度层把层段派给一台执行不了的 worker。
        # 活跃引擎自带能力探测时以它为准；否则维持既有 backend 口径（行为不变）。
        if self._engine_type == "llama_cpp" and self._llama_engine is not None:
            probe = getattr(self._llama_engine, "supports", None)
            if callable(probe):
                return bool(probe(capability))
        return self.capabilities.supports(capability)

    def get_device(self) -> torch.device:
        """获取当前模型所在设备（PyTorch 引擎）"""
        offload = getattr(self, "_layer_offload", None)
        if offload is not None:
            return offload.execution_device
        if self.model is not None:
            model_device = getattr(self.model, "device", None)
            if model_device is not None:
                return torch.device(model_device)
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device(DEVICE)

    def get_model_info(self) -> dict:
        """获取模型基本信息，用于调试与日志（双引擎兼容）"""
        cfg = mc.get_model_config(self._active_model_id) if self._active_model_id else None
        model_name = cfg.name if cfg else (self._active_model_id or MODEL_NAME)
        if self._engine_type == "llama_cpp" and self._llama_engine:
            info = self._llama_engine.get_model_info()
            info["model_id"] = self._active_model_id
            info["model_name"] = model_name
            info["model_path"] = self._model_path
            if self.load_fingerprint:
                info["load_fingerprint"] = self.load_fingerprint
                info["load_fingerprint_schema"] = 1
            return info

        if self._engine_type == "island" and self._island_engine:
            info = self._island_engine.get_model_info()
            info["model_id"] = self._active_model_id
            # 孤岛节点对外展示后端模型名（凭据已在 base_url 中脱敏）
            info["model_name"] = info.get("model") or model_name
            info["model_path"] = info.get("base_url", "")
            if self.load_fingerprint:
                info["load_fingerprint"] = self.load_fingerprint
                info["load_fingerprint_schema"] = 1
            return info

        info = {
            "model_id": self._active_model_id,
            "engine": "pytorch",
            "model_name": model_name,
            "model_path": self._model_path or MODEL_PATH,
            "quant_type": self.quant_type,
            "compile": USE_COMPILE,
            "layer_range": self.layer_range,
            "total_layers": self._total_model_layers or self._model_layers,
            "loaded_layers": self._model_layers,
            "device": str(self.get_device()),
        }
        actual_config = getattr(self.model, "config", None)
        text_config = getattr(actual_config, "text_config", actual_config)
        implementation = getattr(text_config, "_attn_implementation", None)
        remote_flash = getattr(actual_config, "use_flash_attn", None)
        info["inference_switches"] = ({
            "schema_version": 2, "scope": "loaded_model", "execution_path": "pytorch",
            "requested": {"flash_attn": remote_flash if type(remote_flash) is bool else "engine_policy"},
            "effective": {"attention_implementation": implementation,
                          "remote_qwen_use_flash_attn": remote_flash,
                          "device": str(self.get_device())},
            "sources": {"attention": "loaded_model_config"},
            "selection": {"reason": "pytorch_independent_policy"},
        } if self.model is not None else None)
        if getattr(self, "_layer_offload", None) is not None:
            info["layer_offload"] = self.layer_offload_observation()
        if self._layer_load_metrics is not None:
            info["layer_load_metrics"] = dict(self._layer_load_metrics)
            proof = info["layer_load_metrics"].get("materialization_proof")
            if isinstance(proof, LayerMaterializationProof):
                info["layer_load_metrics"]["materialization_proof"] = proof.as_dict()
        if self.load_fingerprint:
            info["load_fingerprint"] = self.load_fingerprint
            info["load_fingerprint_schema"] = 1
        if torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
        return info

    def get_memory_usage(self) -> dict:
        """获取当前显存/内存占用，用于性能监控（多引擎兼容）"""
        if self._engine_type == "llama_cpp" and self._llama_engine:
            return self._llama_engine.get_memory_usage()
        if self._engine_type == "island" and self._island_engine:
            return self._island_engine.get_memory_usage()

        result = {}
        if torch.cuda.is_available():
            result["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            result["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
            result["gpu_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
        return result

    def reset_kv_cache(self) -> None:
        """清空 KV 缓存（多引擎兼容）"""
        if self._engine_type == "llama_cpp" and self._llama_engine:
            self._llama_engine.reset_kv_cache()
        if self._engine_type == "island" and self._island_engine:
            self._island_engine.reset_kv_cache()
        # PyTorch: KV cache 由 transformers generate() 内部管理，每次调用自动重置

    # ================================================================
    # KV Cache 工具方法（Phase 3 — 增量解码）
    # ================================================================

    @staticmethod
    def _tuple_to_dynamic_cache(past_key_values: tuple, start_layer: int = 0):
        """
        将旧格式 tuple of (k,v) 转换为 DynamicCache。

        NOTE: 此方法当前未被任何代码路径调用（死代码）。
        如果将来重新启用，请注意它使用全局层索引（start_layer + i），
        与 forward_layers 热路径中的本地索引补丁（0..N-1）不兼容。
        混用两者会导致 cache.layers 中出现 None 槽位，进而使
        get_seq_length() 崩溃。

        Qwen2 SDPA/Flash Attention 使用 DynamicCache（transformers >= 4.44），
        每层的 self_attn.layer_idx 为全局层编号，故 DynamicCache 内部
        以全局 layer_idx 为索引存储 key/value。

        Args:
            past_key_values: tuple of (key, value) per layer，索引 0..N-1
            start_layer: 第一个元素的全局层编号（分布式节点专用）

        Returns:
            DynamicCache 对象
        """
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i, kv in enumerate(past_key_values):
            if kv is None:
                continue
            k, v = kv
            global_idx = start_layer + i
            # transformers≥5.x: 使用 update() 按 global_idx 写入
            cache.update(k, v, global_idx)
        return cache
