"""Request-scoped deadline primitives shared by scheduler and task graph.

The deadline is created once from a duration and keeps both clocks:

* ``expires_at_monotonic`` is authoritative for local waiting and cannot move
  when the system clock is adjusted;
* ``expires_at_epoch`` is only the transport representation used by workers.

Heartbeat freshness and execution leases deliberately remain separate
concepts. A request deadline caps how long the caller is willing to wait;
leases still fence ownership of remote work, and heartbeat/TCP state still
describe peer liveness.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


REQUEST_DEADLINE_EXCEEDED = "request_deadline_exceeded"


class RequestDeadlineExceeded(TimeoutError):
    """Raised when an end-to-end request budget has been exhausted."""

    code = REQUEST_DEADLINE_EXCEEDED


@dataclass(frozen=True)
class RequestDeadline:
    """One immutable absolute deadline for a complete request."""

    expires_at_monotonic: float
    expires_at_epoch: float
    timeout_seconds: float

    @classmethod
    def start(cls, timeout_seconds: float) -> "RequestDeadline":
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request timeout must be finite and positive")
        return cls(
            expires_at_monotonic=time.monotonic() + timeout,
            expires_at_epoch=time.time() + timeout,
            timeout_seconds=timeout,
        )

    @classmethod
    def from_epoch_ms(cls, expires_at_epoch_ms: int) -> "RequestDeadline":
        """Rebuild a local monotonic deadline from its wire representation."""

        expires_at_epoch = float(expires_at_epoch_ms) / 1000.0
        remaining = expires_at_epoch - time.time()
        return cls(
            expires_at_monotonic=time.monotonic() + remaining,
            expires_at_epoch=expires_at_epoch,
            timeout_seconds=max(0.0, remaining),
        )

    def remaining(self) -> float:
        return max(0.0, self.expires_at_monotonic - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def clamp(self, timeout_seconds: float) -> float:
        """Return a stage timeout capped by the remaining request budget."""
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("stage timeout must be finite and non-negative")
        return min(timeout, self.remaining())

    def require_remaining(self) -> float:
        remaining = self.remaining()
        if remaining <= 0:
            raise RequestDeadlineExceeded("request deadline exceeded")
        return remaining

    @property
    def expires_at_epoch_ms(self) -> int:
        return int(self.expires_at_epoch * 1000)


def coerce_request_deadline(
    value: Optional[RequestDeadline],
    *,
    timeout_seconds: float,
) -> RequestDeadline:
    if value is None:
        return RequestDeadline.start(timeout_seconds)
    if not isinstance(value, RequestDeadline):
        raise TypeError("request deadline must be a RequestDeadline")
    return value


@contextmanager
def hold_lock_until_deadline(
    lock,
    deadline: Optional[RequestDeadline],
    *,
    operation: str = "request lock",
) -> Iterator[None]:
    """Acquire a standard threading lock without outliving the request.

    Lock ownership is never abandoned in the background: once acquired, the
    caller retains it until the guarded operation finishes. Only time spent
    waiting for ownership is bounded here.
    """

    if deadline is None:
        lock.acquire()
    else:
        remaining = deadline.require_remaining()
        if not lock.acquire(timeout=remaining):
            raise RequestDeadlineExceeded(
                f"request deadline exceeded while acquiring {operation}"
            )
        if deadline.expired():
            lock.release()
            raise RequestDeadlineExceeded(
                f"request deadline exceeded while acquiring {operation}"
            )
    try:
        yield
    finally:
        lock.release()


class DeadlineCancelEvent(threading.Event):
    """Event view that becomes set when either cancellation or deadline wins.

    ``set()`` remains available for queue/provider code that owns a local
    cancellation signal. The wrapped external event is read-only here so a
    scheduler timeout never masquerades as a user cancellation upstream.
    """

    def __init__(
        self,
        external_event: Optional[threading.Event],
        deadline: Optional[RequestDeadline],
    ) -> None:
        super().__init__()
        self._external_event = external_event
        self._deadline = deadline

    def is_set(self) -> bool:
        return bool(
            self.cancel_requested() or self.deadline_expired()
        )

    def cancel_requested(self) -> bool:
        return bool(
            super().is_set()
            or (
                self._external_event is not None
                and self._external_event.is_set()
            )
        )

    def deadline_expired(self) -> bool:
        return bool(
            self._deadline is not None and self._deadline.expired()
        )

    def wait(self, timeout: Optional[float] = None) -> bool:
        local_deadline = (
            None if timeout is None else time.monotonic() + max(0.0, timeout)
        )
        while not self.is_set():
            remaining = (
                None
                if local_deadline is None
                else local_deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                return False
            super().wait(
                0.05 if remaining is None else min(0.05, remaining)
            )
        return True


def request_stop_reason(
    cancel_event: Optional[threading.Event],
    deadline: Optional[RequestDeadline],
) -> str:
    """Return the first locally observable terminal stop class.

    Explicit cancellation is checked first. If both signals became visible
    between polls this deterministic order prevents two competing terminal
    reasons from being emitted for the same request.
    """
    if isinstance(cancel_event, DeadlineCancelEvent):
        if cancel_event.cancel_requested():
            return "generation_cancelled"
        if cancel_event.deadline_expired():
            return REQUEST_DEADLINE_EXCEEDED
    elif cancel_event is not None and cancel_event.is_set():
        return "generation_cancelled"
    if deadline is not None and deadline.expired():
        return REQUEST_DEADLINE_EXCEEDED
    return ""
