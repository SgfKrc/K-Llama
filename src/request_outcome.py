"""★ 2026-10-07（DIST-NEXT-8）：一次请求的**生命周期相位**与**互斥终态**的单一来源。

审计「两条产品路径的验收闸门 · 两者共同要求」要求：一次请求的日志与响应 metrics 能
重建「选了谁、哪个 assignment/generation、是否实际发送 offer、是否 ACK、是否执行、
失败属于可重试还是不可恢复、是否发生 fallback」。

现状缺口：metrics 里只有 `fallback` / `fallback_reason` / `distributed_used` 三个散字段，
而取消（`{'cancelled': True}` 事件）、链路错误（`{'error': ...}` 事件）与具名拒绝
（`_routing_gate_error`）**没有共同的终态字段** —— 聚合时「取消」「拒绝」「失败」会被
`fallback` 或 error 语义吞掉（真机排障时表现为「请求失败」看不出是用户取消还是链路错误）。

本模块提供两层，供 `api_server` / `routes_chat` / `engine_host` 共用：

* [request_phase_metrics] —— 相位（`admitted` / `started` / `completed` / `fallback` /
  `refused` / `cancelled` / `failed`）与由它们派生的**唯一终态** `outcome`；
* [derive_outcome] —— 互斥优先级：`cancelled` > `refused` > `failed` >
  `fallback_completed` > `completed` > `incomplete`。

**互斥的含义**：一个请求的 `outcome` 只有一个值；`fallback=True` 只描述「回退后仍完成」，
绝不用于表达取消或拒绝。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

OUTCOME_COMPLETED = "completed"
OUTCOME_FALLBACK_COMPLETED = "fallback_completed"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_REFUSED = "refused"
OUTCOME_FAILED = "failed"
OUTCOME_INCOMPLETE = "incomplete"

OUTCOMES = (
    OUTCOME_COMPLETED,
    OUTCOME_FALLBACK_COMPLETED,
    OUTCOME_CANCELLED,
    OUTCOME_REFUSED,
    OUTCOME_FAILED,
    OUTCOME_INCOMPLETE,
)

#: 相位字段名（稳定集合，供日志/测试按名索引）。
PHASE_FIELDS = (
    "admitted",
    "started",
    "completed",
    "fallback",
    "refused",
    "cancelled",
    "failed",
)

#: 取消的稳定 reason code（`_api_module.ChatGenerationCancelled` 路径）。
REASON_GENERATION_CANCELLED = "generation_cancelled"
#: 通用失败 reason code（链路错误等，具体原因仍见 `error` 文本）。
REASON_REQUEST_FAILED = "request_failed"
#: 具名拒绝的 reason code（路由门等：不可恢复、dispatch 前结束）。
REASON_REQUEST_REFUSED = "request_refused"
#: End-to-end request budget exhausted; distinct from worker lease expiry.
REASON_REQUEST_DEADLINE_EXCEEDED = "request_deadline_exceeded"

MAX_SECONDARY_OBSERVATIONS = 32


@dataclass
class TerminalOutcomeLatch:
    """Thread-safe first-writer-wins terminal reason.

    Later heartbeat, disconnect, lease, timeout or cancel observations are
    retained for diagnostics but cannot overwrite the request's terminal
    reason. This is intentionally a small value object rather than another
    scheduler state machine.
    """

    reason_code: str = ""
    source: str = ""
    observed_at: float = 0.0
    secondary_observations: list[dict[str, Any]] = field(default_factory=list)
    secondary_observations_dropped: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False,
    )

    def decide(self, reason_code: str, *, source: str = "") -> str:
        code = str(reason_code or "").strip()
        normalized_source = str(source or "")
        if not code:
            raise ValueError("terminal reason_code must not be empty")
        with self._lock:
            if not self.reason_code:
                self.reason_code = code
                self.source = normalized_source
                self.observed_at = time.time()
            elif code != self.reason_code or normalized_source != self.source:
                duplicate = any(
                    item.get("reason_code") == code
                    and item.get("source") == normalized_source
                    for item in self.secondary_observations
                )
                if duplicate:
                    return self.reason_code
                if len(self.secondary_observations) >= MAX_SECONDARY_OBSERVATIONS:
                    self.secondary_observations_dropped += 1
                    return self.reason_code
                self.secondary_observations.append({
                    "reason_code": code,
                    "source": normalized_source,
                    "observed_at": time.time(),
                })
            return self.reason_code

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "reason_code": self.reason_code,
                "source": self.source,
                "observed_at": self.observed_at,
                "secondary_observations": [
                    dict(item) for item in self.secondary_observations
                ],
                "secondary_observations_dropped": (
                    self.secondary_observations_dropped
                ),
            }


def derive_outcome(
    *,
    cancelled: bool = False,
    refused: bool = False,
    failed: bool = False,
    completed: bool = False,
    fallback: bool = False,
) -> str:
    """由相位派生**唯一**终态（互斥优先级见模块文档）。"""
    if cancelled:
        return OUTCOME_CANCELLED
    if refused:
        return OUTCOME_REFUSED
    if failed:
        return OUTCOME_FAILED
    if completed:
        return OUTCOME_FALLBACK_COMPLETED if fallback else OUTCOME_COMPLETED
    return OUTCOME_INCOMPLETE


def request_phase_metrics(
    *,
    admitted: bool = False,
    started: bool = False,
    completed: bool = False,
    fallback: bool = False,
    refused: bool = False,
    cancelled: bool = False,
    failed: bool = False,
    reason_code: str = "",
) -> dict[str, Any]:
    """相位 + 终态。

    `reason_code` 只在非 `completed` 时有意义（稳定、可 grep）；`completed_fallback`
    的细节仍由既有的 `fallback` / `fallback_reason` 承载。
    """
    return {
        "admitted": bool(admitted),
        "started": bool(started),
        "completed": bool(completed),
        "fallback": bool(fallback),
        "refused": bool(refused),
        "cancelled": bool(cancelled),
        "failed": bool(failed),
        "outcome": derive_outcome(
            cancelled=cancelled,
            refused=refused,
            failed=failed,
            completed=completed,
            fallback=fallback,
        ),
        "outcome_reason": str(reason_code or ""),
    }


def merge_phase_metrics(
    metrics: Mapping[str, Any] | None = None, **phases: Any,
) -> dict[str, Any]:
    """把相位写进既有 metrics（不改动其它键）。"""
    merged = dict(metrics or {})
    merged.update(request_phase_metrics(**phases))
    return merged
