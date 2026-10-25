"""Header-only dtype capacity projection for master and remote PyTorch supply.

Call ``project_runtime_descriptor(source_path, descriptor, dtype.itemsize)`` for
*each node's selected runtime dtype*. Keep the source descriptor for packaging
and disk accounting; pass the returned descriptor to capacity admission with
``runtime_multiplier=1``. Its usual weight-byte fields already include a
conservative load peak (source + materialized weights), even for equal dtypes.
No torch import or tensor materialization is needed here.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from pipeline_model_descriptor import (
        _ARCHITECTURE_LAYOUTS,
        _DTYPE_BYTES,
        _component_for_key,
        _read_json_object,
        _safe_shard_path,
        PipelineModelDescriptorError,
        canonical_pipeline_model_type,
    )
except ImportError:
    from .pipeline_model_descriptor import (
        _ARCHITECTURE_LAYOUTS,
        _DTYPE_BYTES,
        _component_for_key,
        _read_json_object,
        _safe_shard_path,
        PipelineModelDescriptorError,
        canonical_pipeline_model_type,
    )


# The production selective loader currently selects FP16 or FP32. Source FP8
# headers are supported without trying to create a CPU FP8 tensor.
_RUNTIME_ELEMENT_SIZES = frozenset({2, 4})
_FLOAT_DTYPES = frozenset({"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"})
_COMPONENTS = (
    "layers", "embedding", "final_norm", "lm_head", "visual", "mtp", "multimodal", "other",
)


class RuntimeCapacityProjectionError(PipelineModelDescriptorError):
    """A named supply failure rather than an unsafe byte estimate."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _shards(root: Path) -> tuple[dict[str, list[str] | None], int]:
    """Resolve the same index selection as the loader, without importing torch."""
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        shards = {path.name: None for path in sorted(root.glob("*.safetensors"))}
        if not shards:
            raise RuntimeCapacityProjectionError(
                "SUPPLY_RUNTIME_SOURCE_INVALID", "No Safetensors weights in source directory",
            )
        return shards, 0

    index = _read_json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_SOURCE_INVALID", "Safetensors index is missing weight_map",
        )
    shards: dict[str, list[str] | None] = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key or not isinstance(filename, str):
            raise RuntimeCapacityProjectionError(
                "SUPPLY_RUNTIME_SOURCE_INVALID", "Invalid Safetensors weight_map entry",
            )
        shards.setdefault(filename, []).append(key)
    metadata = index.get("metadata")
    declared_size = metadata.get("total_size", 0) if isinstance(metadata, dict) else 0
    if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size < 0:
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_SOURCE_INVALID", "Invalid Safetensors declared total_size",
        )
    return shards, declared_size


def _required_range(descriptor: dict[str, Any], total_layers: int) -> tuple[int, int]:
    if not descriptor.get("partial_assignment"):
        return 0, total_layers
    interval = descriptor.get("assignment_layer_range")
    if (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in interval)
        or not 0 <= interval[0] < interval[1] <= total_layers
    ):
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_DESCRIPTOR_INVALID", "Invalid partial assignment_layer_range",
        )
    return interval[0], interval[1]


def project_runtime_descriptor(
    source_path: str | Path,
    descriptor: dict[str, Any],
    target_element_size: int,
    *, target_dtype: str | None = None,
) -> dict[str, Any]:
    """Return an independent descriptor charged for a node's actual load dtype.

    ``layer_weight_bytes``, ``component_weight_bytes`` and ``weight_bytes`` are
    the conservative source-plus-target load peaks of participating tensors.
    ``runtime_*weight_bytes`` give the materialized target weights alone.
    ``source_*weight_bytes`` retain exact header-derived source sizes, including
    architecture-declared ignored components. All inherited identity and
    assignment metadata is deep-copied; the input descriptor remains unchanged.

    Only the architecture's existing ignore declaration may remove components.
    Unsupported source dtypes (including bool/integer weights) and target sizes
    fail with ``SUPPLY_RUNTIME_DTYPE_UNSUPPORTED``. An element size cannot
    identify a dtype by itself: callers must obtain it from the production
    runtime selector, which currently returns FP16 or FP32. Passing its exact
    ``target_dtype`` additionally quotes CPU aliasing and one-tensor conversion:
    matching source tensors reuse their mmap storage; BF16 to FP16 converts even
    though both elements occupy two bytes. Without a dtype name, conversion
    metadata conservatively treats every tensor as requiring a CPU cast.
    """
    if (
        isinstance(target_element_size, bool)
        or not isinstance(target_element_size, int)
        or target_element_size not in _RUNTIME_ELEMENT_SIZES
    ):
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_DTYPE_UNSUPPORTED",
            "Unsupported runtime target_element_size; FP16/FP32 require 2 or 4 bytes",
        )
    if target_dtype is not None and not isinstance(target_dtype, str):
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_DTYPE_UNSUPPORTED", "Runtime target dtype must be an explicit dtype name",
        )
    target_header_dtype = {"float16": "F16", "float32": "F32"}.get(target_dtype)
    if target_dtype is not None and (target_header_dtype is None
            or _DTYPE_BYTES[target_header_dtype] != target_element_size):
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_DTYPE_UNSUPPORTED", "Runtime target dtype and element size disagree",
        )
    if not isinstance(descriptor, dict):
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_DESCRIPTOR_INVALID", "Runtime source descriptor must be an object",
        )
    model_type = canonical_pipeline_model_type(descriptor.get("model_type"))
    layout = _ARCHITECTURE_LAYOUTS.get(model_type)
    if layout is None:
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_ARCHITECTURE_UNSUPPORTED",
            f"No pipeline tensor layout for {model_type or 'unknown'}",
        )
    total_layers = descriptor.get("total_layers")
    if isinstance(total_layers, bool) or not isinstance(total_layers, int) or total_layers <= 0:
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_DESCRIPTOR_INVALID", "Invalid descriptor total_layers",
        )
    required_start, required_end = _required_range(descriptor, total_layers)
    ignored = list(layout["runtime_ignored_components"])
    declared_ignored = descriptor.get("runtime_ignored_components", ignored)
    if (
        not isinstance(declared_ignored, (list, tuple))
        or any(not isinstance(name, str) for name in declared_ignored)
        or set(declared_ignored) != set(ignored)
    ):
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_DESCRIPTOR_INVALID",
            "Descriptor runtime_ignored_components differs from architecture layout",
        )
    root = Path(source_path).resolve()
    if not root.is_dir():
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_SOURCE_INVALID", "Safetensors source directory does not exist",
        )

    from safetensors import SafetensorError, safe_open

    source_layers = [0] * total_layers
    runtime_layers = [0] * total_layers
    source_components = dict.fromkeys(_COMPONENTS, 0)
    runtime_components = dict.fromkeys(_COMPONENTS, 0)
    conversion_layers = [0] * total_layers
    conversion_components = dict.fromkeys(_COMPONENTS, 0)
    conversion_layer_max = [0] * total_layers
    conversion_component_max = dict.fromkeys(_COMPONENTS, 0)
    dtype_statistics: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "element_count": 0, "weight_bytes": 0},
    )
    seen_keys: set[str] = set()
    try:
        shards, declared_size = _shards(root)
        for filename, indexed_keys in sorted(shards.items()):
            shard = _safe_shard_path(root, filename)
            # Unlike framework="pt", numpy header access does not import torch.
            # get_slice is used only for metadata, never sliced or materialized.
            with safe_open(str(shard), framework="numpy") as handle:
                available_keys = set(handle.keys())
                keys = sorted(available_keys) if indexed_keys is None else indexed_keys
                for key in keys:
                    if key not in available_keys or key in seen_keys:
                        raise RuntimeCapacityProjectionError(
                            "SUPPLY_RUNTIME_SOURCE_INVALID",
                            f"Missing or duplicate Safetensors tensor: {key}",
                        )
                    seen_keys.add(key)
                    tensor_slice = handle.get_slice(key)
                    dtype = str(tensor_slice.get_dtype()).upper()
                    shape = tensor_slice.get_shape()
                    del tensor_slice
                    if dtype not in _FLOAT_DTYPES:
                        raise RuntimeCapacityProjectionError(
                            "SUPPLY_RUNTIME_DTYPE_UNSUPPORTED",
                            f"Unsupported Safetensors runtime dtype {dtype} for tensor {key}",
                        )
                    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape):
                        raise RuntimeCapacityProjectionError(
                            "SUPPLY_RUNTIME_SOURCE_INVALID", f"Invalid tensor shape: {key}",
                        )
                    elements = math.prod(shape)
                    source_bytes = elements * _DTYPE_BYTES[dtype]
                    runtime_bytes = elements * target_element_size
                    component, layer_index = _component_for_key(key, layout)
                    # Accelerate and layer offload both cast on CPU before the
                    # device transfer. Equal element sizes alone cannot prove
                    # aliasing (BF16 -> FP16 still allocates a CPU tensor).
                    conversion_bytes = runtime_bytes if dtype != target_header_dtype else 0
                    if component not in ignored:
                        conversion_components[component] += conversion_bytes
                        conversion_component_max[component] = max(
                            conversion_component_max[component], conversion_bytes,
                        )
                    if layer_index is not None:
                        if layer_index >= total_layers:
                            raise RuntimeCapacityProjectionError(
                                "SUPPLY_RUNTIME_SOURCE_INVALID",
                                f"Tensor layer exceeds total_layers={total_layers}: {key}",
                            )
                        source_layers[layer_index] += source_bytes
                        runtime_layers[layer_index] += runtime_bytes
                        conversion_layers[layer_index] += conversion_bytes
                        conversion_layer_max[layer_index] = max(
                            conversion_layer_max[layer_index], conversion_bytes,
                        )
                    source_components[component] += source_bytes
                    if component not in ignored:
                        runtime_components[component] += runtime_bytes
                    statistics = dtype_statistics[dtype]
                    statistics["tensor_count"] += 1
                    statistics["element_count"] += elements
                    statistics["weight_bytes"] += source_bytes
    except RuntimeCapacityProjectionError:
        raise
    except (PipelineModelDescriptorError, SafetensorError, OSError) as exc:
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_SOURCE_INVALID", "Safetensors source headers or index are invalid",
        ) from exc

    missing_layers = [index for index in range(required_start, required_end) if source_layers[index] <= 0]
    if missing_layers:
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_SOURCE_INVALID",
            "Safetensors source is missing decoder layers: " + ", ".join(map(str, missing_layers[:8])),
        )
    source_weight_bytes = sum(source_components.values())
    if declared_size and not descriptor.get("partial_assignment") and declared_size != source_weight_bytes:
        raise RuntimeCapacityProjectionError(
            "SUPPLY_RUNTIME_SOURCE_INVALID",
            f"Safetensors index byte count differs from headers: declared={declared_size}, observed={source_weight_bytes}",
        )
    peak_components = {
        name: 0 if name in ignored else source_components[name] + runtime_components[name]
        for name in _COMPONENTS
    }
    projection = copy.deepcopy(descriptor)
    projection.update(
        runtime_projection_mode="safetensors_headers_source_plus_target",
        runtime_target_element_size=target_element_size,
        runtime_target_dtype=target_dtype,
        runtime_multiplier=1.0,
        runtime_conversion_layer_max_bytes=conversion_layer_max,
        runtime_conversion_component_max_bytes=conversion_component_max,
        runtime_cpu_layer_weight_bytes=[source + converted for source, converted
                                       in zip(source_layers, conversion_layers)],
        runtime_cpu_component_weight_bytes={
            name: 0 if name in ignored else source_components[name] + conversion_components[name]
            for name in _COMPONENTS
        },
        source_weight_bytes=source_weight_bytes,
        source_layer_weight_bytes=source_layers,
        source_component_weight_bytes=source_components,
        source_dtype_statistics=dict(sorted(dtype_statistics.items())),
        runtime_weight_bytes=sum(runtime_components.values()),
        runtime_layer_weight_bytes=runtime_layers,
        runtime_component_weight_bytes=runtime_components,
        weight_bytes=sum(peak_components.values()),
        layer_weight_bytes=[source + runtime for source, runtime in zip(source_layers, runtime_layers)],
        component_weight_bytes=peak_components,
        indexed_tensor_count=len(seen_keys),
        weight_file_count=len(shards),
        runtime_ignored_components=ignored,
        runtime_ignored_component_bytes={name: source_components[name] for name in ignored},
    )
    return projection


def runtime_range_capacity(
    projection: dict[str, Any], start_layer: int, end_layer: int, *,
    has_embedding: bool, has_lm_head: bool,
    materialization: str = "safetensors_layer_offload",
) -> dict[str, int]:
    """Quote separate weight budgets for a selected runtime range, without I/O.

    The caller adds its frozen KV/workspace reserve and safety rounding to
    each memory domain. Host capacity is never added to CUDA capacity. Offload
    charges the largest complete execution unit on CUDA, resident mode charges
    every selected target weight. Source mappings plus a single CPU conversion
    are the conservative host bound in both CUDA modes. Generated (non-checkpoint)
    buffers, activations and KV belong to the explicit workspace/KV reservation.
    """
    def invalid(text):
        raise RuntimeCapacityProjectionError("SUPPLY_RUNTIME_DESCRIPTOR_INVALID", text)

    if not isinstance(projection, dict):
        invalid("Runtime projection must be an object")
    total = projection.get("total_layers")
    if (type(total) is not int or total <= 0
            or type(start_layer) is not int or type(end_layer) is not int
            or not 0 <= start_layer < end_layer <= total
            or type(has_embedding) is not bool or type(has_lm_head) is not bool):
        invalid("Invalid selected runtime range or boundary ownership")
    if materialization not in {"safetensors_layer_offload", "safetensors_key_filtered"}:
        invalid("Unsupported runtime materialization")
    if (materialization == "safetensors_layer_offload" and projection.get("model_type")
            not in {"qwen2", "qwen3", "qwen3_5", "qwen3_5_text"}):
        invalid("Architecture has no layer offload execution adapter")
    arrays = {}
    for name in ("source_layer_weight_bytes", "runtime_layer_weight_bytes",
                 "runtime_conversion_layer_max_bytes"):
        values = projection.get(name)
        if (not isinstance(values, (list, tuple)) or len(values) != total
                or any(type(value) is not int or value < 0 for value in values)):
            invalid("Invalid " + name)
        arrays[name] = values[start_layer:end_layer]
    if any(value <= 0 for value in arrays["source_layer_weight_bytes"]):
        invalid("Selected runtime range has missing source layers")
    maps = {}
    for name in ("source_component_weight_bytes", "runtime_component_weight_bytes",
                 "runtime_conversion_component_max_bytes"):
        values = projection.get(name)
        if (not isinstance(values, dict) or any(key not in values for key in _COMPONENTS)
                or any(type(value) is not int or value < 0 for value in values.values())):
            invalid("Invalid " + name)
        maps[name] = values
    keys = ["final_norm", "other"]
    # The loader reuses the embedding tensor for tied lm_head on tail nodes.
    if has_embedding or (has_lm_head and projection.get("tie_word_embeddings") is True):
        keys.append("embedding")
    if has_lm_head and projection.get("tie_word_embeddings") is not True:
        keys.append("lm_head")
    host_source = sum(arrays["source_layer_weight_bytes"]) + sum(
        maps["source_component_weight_bytes"][key] for key in keys)
    host_conversion = max([*arrays["runtime_conversion_layer_max_bytes"], *(
        maps["runtime_conversion_component_max_bytes"][key] for key in keys)], default=0)
    device_units = [*arrays["runtime_layer_weight_bytes"], *(
        maps["runtime_component_weight_bytes"][key] for key in keys)]
    device_weights = (max(device_units, default=0) if materialization == "safetensors_layer_offload"
                      else sum(device_units))
    return {"host_source_bytes": host_source,
            "host_conversion_bytes": host_conversion,
            "host_weight_peak_bytes": host_source + host_conversion,
            "device_weight_bytes": device_weights,
            "target_weight_bytes": sum(device_units)}


def project_supply_capacity_nodes(
    descriptors: dict[str, dict], nodes: list[dict], *, master_engine: str,
    alignment: int = 1, safety_margin: float = 1.2,
    runtime_descriptors: dict[str, dict] | None = None,
    preferred_ranges: dict[str, list[list[int]]] | None = None,
    runtime_host_ranges: dict[str, list[list[int]]] | None = None,
) -> tuple[dict, list[dict], dict[str, dict]]:
    """Project a new supply operation, never an executable ready advertisement.

    The coordinator must first verify full source identity/provenance, available
    engines and fresh device budgets. ``runtime_descriptors`` must come from
    ``project_runtime_descriptor`` for the selected dtype (CUDA uses its target
    fields; host staging stays a separate range constraint). This function does
    not discover hardware. Explicit layer offload uses exact range quotes and
    an independent host RAM constraint, leaving every load_spec intact. Its
    three outputs replace the grouped/projected/by_id construction in ``model_supply.plan_segment_supply``.

    Ready ranges and old layer budgets describe previously supplied assets.
    Full source coverage allows generating new proper aligned segments, even
    when a PC lacks local_cut tools: the coordinator can cut and distribute.
    Explicit reuse preferences and host ranges remain independent hard bounds.
    Every node is charged its exact per-group bytes, without a worst-layer
    format ratio. Preserve the existing reserve total and separate its configured
    workspace and retained-boundary charges. Both keep their own upward rounding.
    """
    from qlh_core.contracts.load import RuntimeLoadSpec

    def fail(code, text):
        raise RuntimeCapacityProjectionError(code, text)

    reference = descriptors.get("llama_cpp") or descriptors.get(master_engine)
    if not reference or reference.get("pipeline_runtime_supported") is not True:
        fail("SUPPLY_DESCRIPTOR_UNAVAILABLE", "No supported source descriptor")
    total = reference.get("total_layers")
    if (type(total) is not int or total <= 0 or type(alignment) is not int
            or alignment < 1 or total % alignment):
        fail("SUPPLY_ALIGNMENT_INVALID", "Invalid decoder cut alignment")
    if (type(safety_margin) not in (int, float) or not math.isfinite(safety_margin)
            or safety_margin < 1):
        fail("SUPPLY_CAPACITY_INVALID", "Invalid supply safety margin")

    def source_sizes(source):
        if (source.get("pipeline_runtime_supported") is not True
                or source.get("total_layers") != total
                or source.get("hidden_size") != reference.get("hidden_size")):
            fail("SUPPLY_SOURCE_CONTRACT_MISMATCH", "Source dimensions or runtime support differ")
        if source.get("partial_assignment") or (
            "assignment_layer_range" in source and source["assignment_layer_range"] != [0, total]
        ):
            fail("SUPPLY_FULL_SOURCE_REQUIRED", "A ready segment cannot authorize a new full-source split")
        sizes = source.get("layer_weight_bytes")
        if (not isinstance(sizes, (list, tuple)) or len(sizes) != total
                or any(type(size) is not int or size <= 0 for size in sizes)):
            fail("SUPPLY_LAYER_BYTES_MISSING", "Source layer bytes are incomplete")
        return [sum(sizes[i:i + alignment]) for i in range(0, total, alignment)]

    group_sizes = source_sizes(reference)
    projected, by_id = [], {}
    for raw in nodes:
        node = copy.deepcopy(raw)
        node_id = node.get("node_id")
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：身份是 `(node_id, executor_id)`。
        #   同一台物理机可以承载多个执行器（master 的 GPU 段 + 本机 CPU 段），它们**共用
        #   `node_id`**、共享同一份主机内存，但各持有**不同的裁层工件** ⇒ 各自能承载的
        #   区间不同，必须分别建模。此前只认 `node_id`，本机 CPU 段一进来就撞
        #   "identities must be unique"（`SUPPLY_CAPACITY_INVALID`）。
        #   `executor_id` 缺省为空串 ⇒ 键退化为 `(node_id, "")` ⇒ 既有拓扑逐位等价。
        executor_id = str(node.get("executor_id") or "")
        if not isinstance(node_id, str) or not node_id:
            fail("SUPPLY_CAPACITY_INVALID", "Supply nodes must have unique nonempty identities")
        identity = (node_id, executor_id)
        if identity in by_id:
            fail("SUPPLY_CAPACITY_INVALID", "Supply nodes must have unique nonempty identities")
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**"用主引擎"是主执行器的性质，不是
        #   `node_id` 的性质**。
        #
        #   本机 CPU 段与主段共用 `node_id="master"`，但它的引擎是 `llama_cpp`
        #   （scheduler 的 `_local_cpu_executor_record` 明确声明 `engine="llama_cpp"`，
        #   与它持有的 keep-head GGUF 工件一致）。此前 `node_id == "master"` 就把
        #   `master_engine`（`pytorch`）强加给它 ⇒ 它被当成 PyTorch 段走
        #   `build_safetensors_segment` ⇒ 段虽然建出来了（磁盘上 master 下确实有两个
        #   segment 目录），但 manifest 形态与执行期期望的 GGUF 段不符 ⇒ 无法注册，
        #   也不会出现在部署视图里。
        #   实测（2026-10-26 06:19）：
        #     master\segment-7909e1b5…\supply.manifest.json   ← 主段
        #     master\segment-864ade28…\supply.manifest.json   ← CPU 段（形态错了）
        #   而 android / TABLET 都同时有 `segment.manifest.json`（llama_cpp 段形态）。
        is_primary_executor = node_id == "master" and not node.get("executor_id")
        engine = master_engine if is_primary_executor else node.get("engine")
        if engine not in {"llama_cpp", "pytorch"} or engine not in descriptors:
            fail("SUPPLY_ENGINE_SOURCE_MISSING", f"No source for {engine}")
        original = descriptors[engine]
        source_sizes(original)  # Full source must exist independently of its runtime quote.
        for key in ("source_model_sha256", "tokenizer_sha256"):
            expected = reference.get(key)
            if expected and original.get(key) != expected:
                fail("SUPPLY_SOURCE_CONTRACT_MISMATCH", f"Engine sources disagree on {key}")
        source = (runtime_descriptors or {}).get(node_id, original)
        sizes = source_sizes(source)
        for key in ("source_model_sha256", "tokenizer_sha256"):
            expected = original.get(key)
            if expected and source.get(key) != expected:
                fail("SUPPLY_SOURCE_CONTRACT_MISMATCH", f"Runtime projection changed {key}")
        multiplier = node.get("runtime_multiplier", 1.0)
        if (type(multiplier) not in (int, float) or not math.isfinite(multiplier)
                or multiplier <= 0):
            fail("SUPPLY_CAPACITY_INVALID", "Invalid runtime multiplier")
        spec = None
        if "load_spec" in node:
            try:
                policy = node["load_spec"]
                spec = policy if isinstance(policy, RuntimeLoadSpec) else RuntimeLoadSpec.from_dict(policy)
            except (TypeError, ValueError) as exc:
                raise RuntimeCapacityProjectionError("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", str(exc)) from exc
            materializations = ({"gguf_segment"} if engine == "llama_cpp"
                                else {"safetensors_key_filtered", "safetensors_layer_offload"})
            device = str(node.get("execution_device", ""))
            if (spec.materialization not in materializations
                    or device not in {spec.device, spec.device.partition(":")[0]}
                    or ("materialization" in node and node["materialization"] != spec.materialization)):
                fail("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Engine/device disagrees with selected load_spec")
            element_size = source.get("runtime_target_element_size")
            if element_size is not None and {"float16": 2, "float32": 4}.get(spec.dtype) != element_size:
                fail("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Runtime quote disagrees with selected dtype")
            node["load_spec"] = spec.to_dict()
            node["execution_device"] = spec.device
        offload = ((spec is not None and spec.materialization == "safetensors_layer_offload")
                   or node.get("materialization") == "safetensors_layer_offload")
        if offload:
            if (engine != "pytorch" or not str(node.get("execution_device", "")).startswith("cuda")
                    or multiplier != 1.0):
                fail("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Layer offload requires a CUDA target with exact byte accounting")
            if (spec is not None and (spec.sequence_count != 1 or spec.kv_device != spec.device
                    or spec.kv_dtype != spec.dtype
                    or {"fp16": "float16", "fp32": "float32"}.get(spec.quant_type) != spec.dtype)):
                fail("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Layer offload execution policy disagrees with its target")
            quoted_dtype = source.get("runtime_target_dtype")
            if spec is not None and quoted_dtype is not None and spec.dtype != quoted_dtype:
                fail("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Layer offload dtype disagrees with its header projection")
        components = source.get("component_weight_bytes") or {}
        keys = (("embedding", "final_norm", "lm_head", "other")
                if engine == "llama_cpp" or node_id != "master"
                else ("embedding", "final_norm", "other"))
        if any(type(components.get(key, 0)) is not int or components.get(key, 0) < 0 for key in keys):
            fail("SUPPLY_CAPACITY_INVALID", "Invalid boundary byte quote")
        overhead = sum(components.get(key, 0) for key in keys)
        for key in ("layer_ranges", "layer_artifacts", "layer_budget", "segment_mode", "models", "range_weight_bytes"):
            node.pop(key, None)
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**"必须以 0 开头"是主执行器的性质，不是
        #   `node_id` 的性质**。
        #
        #   本机 CPU 段与主段**共用 `node_id="master"`**，但它是 `middle`
        #   （`mid8-24.gguf` 里没有 embedding，也没有 `lm_head`/`final_norm`），必须能
        #   `start > 0`。此前判据写作 `node_id != "master"`，于是本机 CPU 段被限制成只能
        #   `start == 0` ⇒ 它的候选与主段完全一样（都是 `[0,k)`）⇒ **两段都只能当首段**
        #   ⇒ 32 层的中段与末段无人覆盖 ⇒ 整盘无解。
        #
        #   实测（2026-10-26 03:23，`event=supply_capacity_input`）：
        #     master       候选数=7   末项=[0,7]      ← 首段，正确
        #     TABLET       候选数=35  末项=[7,8]
        #     android      候选数=35  末项=[7,8]
        #     master#cpu   候选数=7   末项=[0,7]      ← 错：middle 段被卡死在首段
        #   而 `supply_capacity_exclusions` 显示只有 master 被排除
        #   （`node_candidate_layer_range_unavailable` 14 + `node_capacity_insufficient` 5）
        #   ⇒ 远端全可用、只有本机两段互相堵死，症状与证据完全吻合。
        is_primary_master = (node_id == "master" and not executor_id)
        candidates = [[start, end] for start in range(0, total, alignment)
                      for end in range(start + alignment, total + 1, alignment)
                      if end - start < total and (not is_primary_master or start == 0)]
        for mapping, code in ((preferred_ranges, "SUPPLY_PREFERRED_RANGE_INVALID"),
                              (runtime_host_ranges, "SUPPLY_RUNTIME_HOST_BUDGET_INVALID")):
            # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**host 暂存约束只适用于主执行器**。
            #
            #   它的 key 保持裸 `node_id`（既有契约 ——
            #   `test_host_ranges_remain_hard_after_pc_ready_hints_are_removed` 直接断言
            #   `runtime_host_ranges={"master": [[0,1]], "surface": [[2,4]]}` 必须让求解无解，
            #   我一度把 key 改成 `(node_id, executor_id)` 而打掉了这个测试）。
            #
            #   但**本机附加执行器要跳过**这条约束：它表达的是"把**源权重暂存到主机内存**
            #   需要多少 RAM"，只对需要跨 PCIe 暂存的物化（CUDA 上的 safetensors）成立。
            #   本机 CPU 段（llama.cpp）的权重**本来就常驻主机、就地计算**，不存在"暂存"。
            #   此前它对 CPU 段同样生效 ⇒ CPU 段继承了主段的 host 区间
            #   `[[0,4]..[0,28]]` ⇒ 候选被卡成与主段一样的 7 个、只能 `start == 0`
            #   ⇒ 32 层的中段无人承担 ⇒ 整盘无解。
            #   实测（2026-10-26 05:32，`event=supply_capacity_input`）：
            #     master      候选数=7      ← 首段，正确
            #     TABLET      候选数=35
            #     android     候选数=35
            #     master#cpu  候选数=7      ← 错！middle 段被 host 约束卡在首段
            if mapping is runtime_host_ranges and executor_id:
                continue
            if mapping is None or node_id not in mapping:
                continue
            spans = mapping[node_id]
            if (not isinstance(spans, list)
                    or any(not isinstance(span, list) or len(span) != 2
                           or any(type(value) is not int for value in span)
                           or not 0 <= span[0] < span[1] <= total
                           or span[1] - span[0] == total
                           or span[0] % alignment or span[1] % alignment for span in spans)
                    or (mapping is preferred_ranges and (node_id == "master" or not spans))):
                fail(code, "Invalid explicit candidate range constraint")
            # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**两个 mapping 都是层空间，直接比较**。
            #
            #   * `runtime_host_ranges` 来自 `allowed_host_layer_ranges`，值域是层。
            #   * `preferred_ranges` 由 `model_supply_plan_reuse._candidates` 构造，其
            #     `_span` 判据是 `0 <= start < end <= total`（`total` 是**层数**）⇒ 也是层。
            #   * `candidates` 由 `range(0, total, alignment)` 生成 ⇒ 仍是层。
            #   **只有输出** `candidate_layer_ranges` 时才 `// alignment` 转成组。
            #
            #   我在这上面错了两次，各自打掉一个既有测试，特此记下以免重蹈：
            #     第一次给 host 区间除了 `alignment` ⇒ 主段只剩 1 个候选
            #       （`test_alignment_and_explicit_host_ranges_are_preserved_in_exact_quotes`、
            #        `test_source_host_budget_constrains_gpu_plan_and_intersects_reuse_preferences`）；
            #     第二次给 preferred 乘了 `alignment` ⇒ `tail` 拿不到它的段
            #       （`test_existing_range_preference_is_grouped_without_imposing_old_inventory`）。
            #   教训：**先读那个字段的构造方判据（`_span` / `allowed_host_layer_ranges`），
            #   而不是从名字或调用位置猜它的空间。**
            candidates = [span for span in candidates if span in spans]
        configured_reserve = node.get("reserve_bytes", 0)
        if type(configured_reserve) is not int or configured_reserve < 0:
            fail("SUPPLY_CAPACITY_INVALID", "Invalid configured reserve")
        if offload:
            host_available = node.get("host_capacity_bytes")
            if host_available is not None and (type(host_available) is not int or host_available < 0):
                fail("SUPPLY_RUNTIME_HOST_BUDGET_INVALID", "Invalid independent host RAM budget")
            if host_available is None and (runtime_host_ranges is None or node_id not in runtime_host_ranges):
                fail("SUPPLY_RUNTIME_HOST_BUDGET_MISSING", "Layer offload needs an independent host RAM constraint")
            quotes = []
            admitted = []
            for start, end in candidates:
                has_embedding, has_lm_head = start == 0, end == total
                quote = runtime_range_capacity(source, start, end,
                    has_embedding=has_embedding, has_lm_head=has_lm_head)
                if (host_available is not None
                        and math.ceil(quote["host_weight_peak_bytes"] * safety_margin)
                            + configured_reserve > host_available):
                    continue
                admitted.append([start, end])
                quotes.append({"layer_range": [start // alignment, end // alignment],
                               "weight_bytes": quote["device_weight_bytes"],
                               "has_embedding": has_embedding, "has_lm_head": has_lm_head})
            candidates = admitted
            node["range_weight_bytes"] = quotes
            # The exact range quote includes every needed boundary execution
            # unit; summing boundary reservations would count it a second time.
            boundary_reserve = 0
            overhead = 0
        else:
            boundary_reserve = math.ceil(overhead * multiplier * safety_margin)
        node.update(
            engine=engine, role="master" if node_id == "master" else "client",
            layer_weight_bytes=sizes, runtime_multiplier=float(multiplier),
            reserve_bytes=configured_reserve + boundary_reserve,
            configured_reserve_bytes=configured_reserve,
            retained_component_reserve_bytes=boundary_reserve,
            retained_component_bytes=0,
            candidate_layer_ranges=[[start // alignment, end // alignment] for start, end in candidates],
        )
        # A preference is intentionally exact within this one candidate attempt;
        # the reuse owner may retry unrestricted if it makes admission impossible.
        if preferred_ranges and node_id in preferred_ranges:
            node["layer_ranges"] = [[start // alignment, end // alignment]
                                    for start, end in preferred_ranges[node_id]]
        by_id[identity] = {**copy.deepcopy(raw), "engine": engine, "overhead_bytes": overhead}
        projected.append(node)
    grouped = copy.deepcopy(reference)
    grouped.update(total_layers=len(group_sizes), layer_weight_bytes=group_sizes,
                   component_weight_bytes={"embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0},
                   runtime_ignored_components=[])
    return grouped, projected, by_id
