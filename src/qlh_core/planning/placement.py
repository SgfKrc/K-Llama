"""Execution placement heuristics, independent of HA/management scoring.

Scores distribute decoder work only after memory and artifact admission. They
are relative hardware estimates, not measured throughput or memory budgets.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def _number(value, default=0.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0 else default


def placement_score_breakdown(device_info, *, gpu, cuda_discrete,
                              execution_device=None):
    """Score absolute free host RAM and the selected execution backend.

    Free RAM uses GiB for both PC and Android profiles. Installed RAM is not a
    substitute for missing free-memory evidence. CPU execution gets no CUDA
    bonus even when its host also advertises a discrete GPU. CUDA selection
    still requires the profiler's discrete CUDA evidence. A caller selecting
    an indexed device must pass that device's GPU profile.
    """
    info = device_info if isinstance(device_info, Mapping) else {}
    ram = info.get("ram") or {}
    memory = info.get("memory") or {}
    cpu = info.get("cpu") or {}
    ram = ram if isinstance(ram, Mapping) else {}
    memory = memory if isinstance(memory, Mapping) else {}
    cpu = cpu if isinstance(cpu, Mapping) else {}
    gpu = gpu if isinstance(gpu, Mapping) else {}
    available_gb = (_number(ram["available_gb"]) if "available_gb" in ram
                    else _number(memory.get("available_bytes")) / 1024 ** 3)
    device = (execution_device.partition(":")[0] if execution_device
              else "cuda" if cuda_discrete else "cpu")
    cuda = device == "cuda" and cuda_discrete
    return {
        "ram_available_gb": available_gb,
        "ram": available_gb / 64.0 * 30.0,
        "vram": min(_number(gpu.get("vram_total_gb")) / 24.0, 1.0) * 50.0 if cuda else 0.0,
        "accelerator": 60.0 if cuda else 0.0,
        "cpu": (min(_number(cpu.get("physical_cores"), 2.0) / 16.0, 1.0) * 10.0
                + min(_number(cpu.get("freq_max_mhz"), 2000.0) / 4000.0, 1.0) * 10.0),
    }


def placement_weight(score):
    """A missing/nonpositive score receives unit weight, never extra capacity."""
    return max(float(score), 1.0)


def stage_placement_cost(work_fraction, score, required_bytes, capacity_bytes):
    """Additive costs keep the contiguous solver's suffix optimum valid.

    Squared shared decoder work divided by relative capability has a continuous
    minimum at work_i / work_j = weight_i / weight_j. Exact layers, component
    costs and capacity bounds determine the nearest feasible discrete split.
    Memory pressure is an additive tiebreaker, using each admitted byte quote.
    """
    return (work_fraction ** 2 / placement_weight(score),
            (required_bytes / capacity_bytes) ** 2)
