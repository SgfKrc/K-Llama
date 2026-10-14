"""Metadata-only RAM constraints for staging CUDA source tensors.

GPU admission remains with the supply solver. This helper constrains that
solver to proper layer ranges whose exact source tensors also fit host RAM,
using the same boundary charges and rounding as the supply host post-check.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


_BOUNDARIES = ("embedding", "final_norm", "lm_head", "other")


class HostCapacityProjectionError(ValueError):
    """Invalid metadata must not silently remove the host capacity constraint."""

    code = "SUPPLY_RUNTIME_HOST_CAPACITY_INVALID"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def allowed_host_layer_ranges(
    projection: Mapping[str, Any],
    node_id: str,
    available_bytes: int,
    reserve_bytes: int,
    safety_margin: float,
    alignment: int,
) -> list[list[int]]:
    """Return all aligned proper ranges admitted by the real host RAM budget.

    Master ranges start at zero and charge embedding, final norm and other
    tensors. Every remote range charges all four boundary components, matching
    the conservative source staging post-check even for middle segments.
    Runtime/VRAM byte estimates and inventory ranges do not affect this bound.

    Malformed metadata raises HostCapacityProjectionError (a ValueError).
    Valid metadata with no feasible range returns an empty list. Inputs are
    never changed; each returned range is a fresh list in start/end order.
    """
    if not isinstance(projection, Mapping):
        raise HostCapacityProjectionError("projection must be a metadata mapping")
    total = projection.get("total_layers")
    if type(total) is not int or total <= 0:
        raise HostCapacityProjectionError("total_layers must be a positive integer")
    if type(alignment) is not int or alignment <= 0 or total % alignment:
        raise HostCapacityProjectionError("alignment must be positive and divide total_layers")
    if not isinstance(node_id, str) or not node_id or node_id.strip() != node_id:
        raise HostCapacityProjectionError("node_id must be a nonempty, unpadded string")
    for name, value in (("available_bytes", available_bytes), ("reserve_bytes", reserve_bytes)):
        if type(value) is not int or value < 0:
            raise HostCapacityProjectionError(f"{name} must be a nonnegative integer")
    if (type(safety_margin) not in (int, float)
            or not math.isfinite(safety_margin) or safety_margin < 1):
        raise HostCapacityProjectionError("safety_margin must be finite and at least 1")

    layers = projection.get("source_layer_weight_bytes")
    if (not isinstance(layers, (list, tuple)) or len(layers) != total
            or any(type(size) is not int or size <= 0 for size in layers)):
        raise HostCapacityProjectionError(
            "source_layer_weight_bytes must contain one positive integer per decoder layer",
        )
    components = projection.get("source_component_weight_bytes")
    if (not isinstance(components, Mapping)
            or any(key not in components for key in _BOUNDARIES)
            or any(not isinstance(key, str) or not key
                   or type(size) is not int or size < 0
                   for key, size in components.items())):
        raise HostCapacityProjectionError(
            "source_component_weight_bytes must declare all four boundaries with nonnegative integer sizes",
        )

    keys = ("embedding", "final_norm", "other") if node_id == "master" else _BOUNDARIES
    boundary_bytes = sum(components[key] for key in keys)
    prefix = [0]
    for size in layers:
        prefix.append(prefix[-1] + size)
    budget = available_bytes - reserve_bytes
    if budget <= 0:
        return []

    result = []
    starts = (0,) if node_id == "master" else range(0, total, alignment)
    for start in starts:
        for end in range(start + alignment, total + 1, alignment):
            if end - start == total:
                continue
            source_bytes = prefix[end] - prefix[start] + boundary_bytes
            # Costs increase with end. Never use a mean/max layer multiplier:
            # the source bytes of this particular interval govern host staging.
            if source_bytes > budget:
                break
            if math.ceil(source_bytes * safety_margin) > budget:
                break
            result.append([start, end])
    return result
