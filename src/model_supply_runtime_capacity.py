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
