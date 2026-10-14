"""Final success gate for requests that require distributed execution."""

from __future__ import annotations

from typing import Any, Mapping

from release_contract import (
    DISTRIBUTED_REQUIRED_EXECUTION_MODES,
    release_profile_enforced,
)


class DistributedCompletionError(RuntimeError):
    pass


def validate_distributed_completion(
    routing_preference: str,
    metrics: Mapping[str, Any] | None,
    *,
    detail: str = "分布式执行未完成",
) -> None:
    """Reject successful local/external completion for ``distributed_required``.

    The source checkout accepts any real distributed mode for development and
    compatibility tests. A release artifact additionally constrains success to
    the execution modes named by its production profile.
    """
    if routing_preference != "distributed_required":
        return
    values = metrics if isinstance(metrics, Mapping) else {}
    if values.get("distributed_used") is not True:
        raise DistributedCompletionError(detail)
    if not release_profile_enforced():
        return
    execution_mode = str(values.get("execution_mode", "") or "")
    if execution_mode not in DISTRIBUTED_REQUIRED_EXECUTION_MODES:
        raise DistributedCompletionError(
            f"发行 profile 不允许 execution_mode={execution_mode or '<missing>'}"
        )
    if execution_mode == "route_a_stage_offer_v3" and not any(
        values.get(key)
        for key in ("workers_used", "layer_assignments", "layer_segments", "claimed_layers")
    ):
        raise DistributedCompletionError("Route-A 成功结果缺少 worker/层段执行证据")
