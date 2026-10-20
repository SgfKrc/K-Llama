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
    runtime selector, which currently returns FP16 or FP32.
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
                    if layer_index is not None:
                        if layer_index >= total_layers:
                            raise RuntimeCapacityProjectionError(
                                "SUPPLY_RUNTIME_SOURCE_INVALID",
                                f"Tensor layer exceeds total_layers={total_layers}: {key}",
                            )
                        source_layers[layer_index] += source_bytes
                        runtime_layers[layer_index] += runtime_bytes
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
        runtime_multiplier=1.0,
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
    not discover hardware or authorize RAM offload, and leaves every load_spec
    intact. Its three outputs replace the grouped/projected/by_id construction
    in ``model_supply.plan_segment_supply``.

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
        if not isinstance(node_id, str) or not node_id or node_id in by_id:
            fail("SUPPLY_CAPACITY_INVALID", "Supply nodes must have unique nonempty identities")
        engine = master_engine if node_id == "master" else node.get("engine")
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
        if "load_spec" in node:
            try:
                policy = node["load_spec"]
                spec = policy if isinstance(policy, RuntimeLoadSpec) else RuntimeLoadSpec.from_dict(policy)
            except (TypeError, ValueError) as exc:
                raise RuntimeCapacityProjectionError("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", str(exc)) from exc
            materialization = "gguf_segment" if engine == "llama_cpp" else "safetensors_key_filtered"
            device = str(node.get("execution_device", ""))
            if (spec.materialization != materialization
                    or device not in {spec.device, spec.device.partition(":")[0]}):
                fail("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Engine/device disagrees with selected load_spec")
            element_size = source.get("runtime_target_element_size")
            if element_size is not None and {"float16": 2, "float32": 4}.get(spec.dtype) != element_size:
                fail("SUPPLY_RUNTIME_LOAD_SPEC_INVALID", "Runtime quote disagrees with selected dtype")
            node["load_spec"] = spec.to_dict()
            node["execution_device"] = spec.device
        components = source.get("component_weight_bytes") or {}
        keys = (("embedding", "final_norm", "lm_head", "other")
                if engine == "llama_cpp" or node_id != "master"
                else ("embedding", "final_norm", "other"))
        if any(type(components.get(key, 0)) is not int or components.get(key, 0) < 0 for key in keys):
            fail("SUPPLY_CAPACITY_INVALID", "Invalid boundary byte quote")
        overhead = sum(components.get(key, 0) for key in keys)
        for key in ("layer_ranges", "layer_artifacts", "layer_budget", "segment_mode", "models"):
            node.pop(key, None)
        candidates = [[start, end] for start in range(0, total, alignment)
                      for end in range(start + alignment, total + 1, alignment)
                      if end - start < total and (node_id != "master" or start == 0)]
        for mapping, code in ((preferred_ranges, "SUPPLY_PREFERRED_RANGE_INVALID"),
                              (runtime_host_ranges, "SUPPLY_RUNTIME_HOST_BUDGET_INVALID")):
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
            candidates = [span for span in candidates if span in spans]
        configured_reserve = node.get("reserve_bytes", 0)
        if type(configured_reserve) is not int or configured_reserve < 0:
            fail("SUPPLY_CAPACITY_INVALID", "Invalid configured reserve")
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
        by_id[node_id] = {**copy.deepcopy(raw), "engine": engine, "overhead_bytes": overhead}
        projected.append(node)
    grouped = copy.deepcopy(reference)
    grouped.update(total_layers=len(group_sizes), layer_weight_bytes=group_sizes,
                   component_weight_bytes={"embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0},
                   runtime_ignored_components=[])
    return grouped, projected, by_id
