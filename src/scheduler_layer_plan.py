"""Stateless scoring and assignment helpers used by the scheduler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qlh_core.planning.placement import placement_score_breakdown


def gpu_is_integrated(gpu: dict) -> bool:
    """Classify a GPU conservatively when profiler metadata is incomplete."""
    if not isinstance(gpu, dict):
        return True
    if "is_integrated" in gpu:
        return bool(gpu.get("is_integrated"))
    gpu_type = str(gpu.get("gpu_type", "")).lower()
    if gpu_type == "discrete":
        return False
    if gpu_type == "integrated":
        return True
    name = str(gpu.get("name", "")).lower()
    if gpu.get("cuda_available") and any(
        marker in name
        for marker in ("nvidia", "geforce", "rtx", "gtx", "tesla", "quadro")
    ):
        return False
    # Unknown or incomplete GPU identities remain conservatively integrated.
    return True


def select_scoring_gpu(
    device_info: dict,
    *,
    gpu_is_integrated_fn: Callable[[dict], bool] = gpu_is_integrated,
) -> dict:
    """Choose the GPU relevant to CUDA execution and layer memory scoring."""
    if not device_info:
        return {}
    candidates = []
    selected = device_info.get("gpu", {})
    if isinstance(selected, dict) and selected:
        candidates.append(selected)
    for gpu in device_info.get("gpus", []) or []:
        if isinstance(gpu, dict) and gpu:
            key = (gpu.get("name"), gpu.get("vram_total_gb"), gpu.get("cuda_available"))
            if not any(
                (candidate.get("name"), candidate.get("vram_total_gb"),
                 candidate.get("cuda_available")) == key
                for candidate in candidates
            ):
                candidates.append(gpu)
    if not candidates:
        return {}

    def vram(gpu: dict) -> float:
        try:
            return float(gpu.get("vram_total_gb", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    cuda_discrete = [
        gpu for gpu in candidates
        if gpu.get("cuda_available") and not gpu_is_integrated_fn(gpu)
    ]
    if cuda_discrete:
        return max(cuda_discrete, key=vram)
    cuda_any = [gpu for gpu in candidates if gpu.get("cuda_available")]
    if cuda_any:
        return max(cuda_any, key=vram)
    discrete_any = [gpu for gpu in candidates if not gpu_is_integrated_fn(gpu)]
    if discrete_any:
        return max(discrete_any, key=vram)
    return max(candidates, key=vram)


def node_is_island_gateway(device_info: dict) -> bool:
    """Return whether a node profile advertises an island gateway."""
    if not isinstance(device_info, dict):
        return False
    island = device_info.get("island")
    return bool(isinstance(island, dict) and island.get("enabled"))


def compute_node_weight(
    device_info: dict,
    *,
    select_gpu: Callable[[dict], dict] = select_scoring_gpu,
    gpu_is_integrated_fn: Callable[[dict], bool] = gpu_is_integrated,
    execution_device: str | None = None,
    debug: Callable[..., Any] | None = None,
) -> float:
    """Estimate placement capability using free host RAM and the active backend.

    Omitting execution_device preserves inferred-backend scoring for legacy
    layer planning. Capacity callers pass their selected runtime device so a
    CPU GGUF executor cannot inherit its host's unused CUDA bonus.
    """
    gpu = select_gpu(device_info)
    cuda_discrete = bool(
        isinstance(gpu, dict)
        and gpu.get("cuda_available", False)
        and not gpu_is_integrated_fn(gpu)
    )
    parts = placement_score_breakdown(
        device_info, gpu=gpu, cuda_discrete=cuda_discrete,
        execution_device=execution_device,
    )
    vram_score, ram_score = parts["vram"], parts["ram"]
    cpu_score, accelerator_score = parts["cpu"], parts["accelerator"]

    island_score = 0.0
    island = device_info.get("island", {}) if device_info else {}
    if isinstance(island, dict) and island.get("enabled"):
        try:
            island_vram_gb = float(island.get("vram_gb", 0) or 0)
        except (TypeError, ValueError):
            island_vram_gb = 0.0
        try:
            island_gpu_count = int(island.get("gpu_count", 1) or 1)
        except (TypeError, ValueError):
            island_gpu_count = 1
        island_score = (
            min(island_vram_gb, 96.0) / 24.0 * 50.0
            + min(max(island_gpu_count, 1), 8) * 5.0
            + 60.0
        )

    weight = vram_score + ram_score + cpu_score + accelerator_score + island_score
    if debug is not None:
        gpu_name = gpu.get("name", "unknown") if isinstance(gpu, dict) else "unknown"
        debug(
            f"节点权重: GPU={(gpu.get('name', 'unknown') if isinstance(gpu, dict) else 'unknown')} VRAM={vram_score:.1f} RAM={ram_score:.1f} CPU={cpu_score:.1f} CUDA={accelerator_score:.1f} 孤岛={island_score:.1f} → {weight:.1f}"
        )
    return weight


def resequence_assignments(assignments: list) -> list:
    """Rebuild contiguous layer ranges and I/O ownership in-place copies."""
    cleaned = [item for item in assignments if item.get("layers_count", 0) > 0]
    master_index = next((
        index for index, item in enumerate(cleaned)
        if item.get("node_id") == "master" or item.get("role") == "master"
    ), None)
    lm_head_index = len(cleaned) - 1
    if master_index is not None and cleaned:
        try:
            master_score = float(cleaned[master_index].get("score", 0) or 0)
            tail_score = float(cleaned[-1].get("score", 0) or 0)
            if master_score >= tail_score:
                lm_head_index = master_index
        except (TypeError, ValueError):
            pass
    cursor = 0
    for index, item in enumerate(cleaned):
        count = int(item.get("layers_count", 0) or 0)
        item["layers_count"] = count
        item["start_layer"] = cursor
        item["end_layer"] = cursor + count
        item["has_embedding"] = index == 0
        item["has_lm_head"] = index == lm_head_index
        cursor += count
    return cleaned


def normalize_master_anchor(
    assignments: list,
    node_list: list,
    total_layers: int,
    *,
    compute_weight: Callable[[dict], float] = compute_node_weight,
    resequence: Callable[[list], list] = resequence_assignments,
    warning: Callable[..., Any] | None = None,
) -> list:
    """Keep master first with at least one layer while preserving coverage."""
    if not assignments:
        return []

    def is_master(item: dict) -> bool:
        return item.get("node_id") == "master" or item.get("role") == "master"

    def score(item: dict) -> float:
        try:
            return float(item.get("score", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    master = None
    rest = []
    for assignment in assignments:
        item = dict(assignment)
        if is_master(item) and master is None:
            master = item
        elif item.get("layers_count", 0) > 0:
            rest.append(item)

    if master is None:
        master_source = next((node for node in node_list if is_master(node)), None)
        if master_source is None:
            return resequence([
                dict(item) for item in assignments if item.get("layers_count", 0) > 0
            ])
        master = {
            "node_id": master_source.get("node_id", "master"),
            "role": master_source.get("role", "master"),
            "layers_count": 1,
            "score": round(
                master_source.get(
                    "score", compute_weight(master_source.get("device_info", {})),
                ),
                1,
            ),
        }
    else:
        master["layers_count"] = max(1, int(master.get("layers_count", 0) or 0))

    ordered = [master] + rest

    def total() -> int:
        return sum(int(item.get("layers_count", 0) or 0) for item in ordered)

    while total() > total_layers and len(ordered) > 1:
        reducible = [item for item in ordered[1:] if item.get("layers_count", 0) > 1]
        if reducible:
            victim = min(reducible, key=score)
            victim["layers_count"] -= 1
            continue
        victim = min(ordered[1:], key=score)
        if warning is not None:
            warning(
                f"节点 {victim.get('node_id')} 层数降至 0 将被移除，以保留 master 首段计算锚点"
            )
        ordered.remove(victim)

    while total() < total_layers and ordered:
        receivers = ordered[1:] or ordered
        target = max(receivers, key=score)
        target["layers_count"] = int(target.get("layers_count", 0) or 0) + 1

    while total() > total_layers and master.get("layers_count", 0) > 1:
        master["layers_count"] -= 1

    return resequence(ordered)


__all__ = [
    "compute_node_weight",
    "gpu_is_integrated",
    "node_is_island_gateway",
    "normalize_master_anchor",
    "resequence_assignments",
    "select_scoring_gpu",
]
