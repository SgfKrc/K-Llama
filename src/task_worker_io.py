"""Bounded FIFO sender for optional PC worker transport isolation.

Reservations cover queued work and the active owner check/action. Admission only
acquires a short mutex; it never waits for space or starts overflow workers.
``max_items`` and ``max_bytes`` are positive integers; ``charge_bytes`` is a
nonnegative integer (zero-byte control messages still consume one item).

Snapshot outcomes are disjoint: submitted == pending_items + completed + failed
+ retired + cancelled. ``completed`` counts successful actions, ``failed`` counts
owner-check/action exceptions, and ``retired`` counts stale owners. Queue wait
ends when the owner check starts; action wall/CPU totals include only actions.
Descriptions are accepted for caller compatibility and are never retained.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import threading
from time import perf_counter_ns, thread_time_ns
from typing import Any, Callable
import weakref


__all__ = ["WorkerIOError", "WorkerIOSender"]

_CLOSE_JOIN_SECONDS = 1.0


class WorkerIOError(RuntimeError):
    """Sender admission or ownership failure with a stable machine-readable code."""

    def __init__(self, message: str = "", *, code: str) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(eq=False)
class _Item:
    action: Callable[[], Any] | None
    is_current: Callable[[], bool] | None
    charge_bytes: int
    enqueued_ns: int
    future: Future
    state: str = "queued"
    # Exactly one participant may call set_running_or_notify_cancel, including
    # for removed cancelled work so concurrent.futures.wait/as_completed wake.
    future_claimed: bool = False

    def clear_callables(self) -> None:
        # Dropping caller objects can run destructors, so do this outside locks.
        self.action = None
        self.is_current = None


class WorkerIOSender:
    """One lazy daemon thread, bounded by item count and declared payload bytes.

    ``is_current`` runs immediately before committing an action to execution.
    Closing during an owner check cancels that Future and suppresses the action;
    the check stays charged until it returns. A committed action may finish even
    after close. Callers own transport/socket interruption and action timeouts.

    Future callbacks run synchronously on the thread completing/cancelling the
    Future, outside the queue lock. They should return promptly. Each close with
    ``wait=True`` joins for at most one second and is safe from the sender thread.
    """

    def __init__(
        self,
        *,
        max_items: int = 64,
        max_bytes: int = 64 * 1024 * 1024,
        name: str = "task-worker-io",
    ) -> None:
        for value, field in ((max_items, "max_items"), (max_bytes, "max_bytes")):
            if type(value) is not int or value < 1:
                raise WorkerIOError(
                    f"{field} must be a positive integer",
                    code="worker_io_budget_invalid",
                )
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._name = name
        self._condition = threading.Condition(threading.Lock())
        # dict order gives FIFO admission and O(1) removal on cancellation.
        self._queue: dict[_Item, None] = {}
        self._active: _Item | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._pending_items = 0
        self._pending_bytes = 0
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._retired = 0
        self._cancelled = 0
        self._peak_items = 0
        self._peak_bytes = 0
        self._queue_wait_ns = 0
        self._action_wall_ns = 0
        self._action_thread_cpu_ns = 0

    def submit(
        self,
        action: Callable[[], Any],
        *,
        charge_bytes: int,
        is_current: Callable[[], bool],
        description: str = "",
    ) -> Future:
        """Reserve the full declared charge or raise immediately without queuing."""
        if type(charge_bytes) is not int or charge_bytes < 0:
            raise WorkerIOError(
                "charge_bytes must be a nonnegative integer",
                code="worker_io_budget_invalid",
            )
        if not callable(action) or not callable(is_current):
            raise TypeError("action and is_current must be callable")
        if not isinstance(description, str):
            raise TypeError("description must be a string")
        future = Future()
        item = _Item(action, is_current, charge_bytes, perf_counter_ns(), future)
        sender_ref, item_ref = weakref.ref(self), weakref.ref(item)

        def on_done(done: Future) -> None:
            sender, pending = sender_ref(), item_ref()
            if sender is not None and pending is not None and done.cancelled():
                sender._on_cancel(pending)

        future.add_done_callback(on_done)
        try:
            with self._condition:
                if self._closed:
                    raise WorkerIOError(code="worker_io_closed")
                if (
                    self._pending_items >= self._max_items
                    or self._pending_bytes + charge_bytes > self._max_bytes
                ):
                    raise WorkerIOError(code="worker_io_full")
                self._queue[item] = None
                if self._thread is None:
                    try:
                        thread = threading.Thread(
                            target=self._run, name=self._name, daemon=True,
                        )
                        thread.start()
                    except BaseException:
                        del self._queue[item]
                        raise
                    self._thread = thread
                self._pending_items += 1
                self._pending_bytes += charge_bytes
                self._submitted += 1
                self._peak_items = max(self._peak_items, self._pending_items)
                self._peak_bytes = max(self._peak_bytes, self._pending_bytes)
                self._condition.notify()
        except BaseException:
            item.clear_callables()
            raise
        return future

    def close(self, wait: bool = False) -> None:
        """Close admission, cancel unstarted work and optionally join for <= 1s."""
        with self._condition:
            self._closed = True
            queued = list(self._queue)
            self._queue.clear()
            for item in queued:
                self._release_locked(item, "cancelled")
            checking = (
                self._active
                if self._active is not None and self._active.state == "checking"
                else None
            )
            thread = self._thread
            self._condition.notify_all()
        for item in queued:
            item.clear_callables()
            self._cancel_future(item.future)
        if checking is not None:
            self._cancel_future(checking.future)
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_CLOSE_JOIN_SECONDS)

    def snapshot(self) -> dict[str, int | bool]:
        """Return aggregate counts/timings; no descriptions, payloads or errors."""
        with self._condition:
            return {
                "closed": self._closed,
                "max_items": self._max_items,
                "max_bytes": self._max_bytes,
                "pending_items": self._pending_items,
                "pending_bytes": self._pending_bytes,
                "queued_items": len(self._queue),
                "thread_alive": self._thread is not None and self._thread.is_alive(),
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "retired": self._retired,
                "cancelled": self._cancelled,
                "peak_items": self._peak_items,
                "peak_bytes": self._peak_bytes,
                "queue_wait_ns": self._queue_wait_ns,
                "action_wall_ns": self._action_wall_ns,
                "action_thread_cpu_ns": self._action_thread_cpu_ns,
            }

    def _on_cancel(self, item: _Item) -> None:
        with self._condition:
            notify_cancel = not item.future_claimed
            item.future_claimed = True
            removed = item.state == "queued"
            if removed:
                del self._queue[item]
                self._release_locked(item, "cancelled")
        if removed:
            item.clear_callables()
        if notify_cancel:
            item.future.set_running_or_notify_cancel()

    @staticmethod
    def _cancel_future(future: Future) -> None:
        try:
            future.cancel()
        except BaseException:
            # Future has already transitioned before it invokes callbacks.
            # A caller's BaseException must not strand the remaining queue.
            pass

    def _release_locked(self, item: _Item, outcome: str) -> None:
        item.state = "finished"
        self._pending_items -= 1
        self._pending_bytes -= item.charge_bytes
        if outcome == "completed":
            self._completed += 1
        elif outcome == "failed":
            self._failed += 1
        elif outcome == "retired":
            self._retired += 1
        else:
            self._cancelled += 1

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue:
                    return
                item = next(iter(self._queue))
                del self._queue[item]
                item.state = "checking"
                self._active = item
                self._queue_wait_ns += max(0, perf_counter_ns() - item.enqueued_ns)
            self._execute(item)
            # An idle worker must not retain the previous Future/callables.
            del item

    def _execute(self, item: _Item) -> None:
        error: BaseException | None = None
        current = False
        try:
            if not item.future.cancelled():
                current = bool(item.is_current())
        except BaseException as exc:
            error = exc

        with self._condition:
            cancel = self._closed or item.future_claimed
            if not cancel:
                item.future_claimed = True
                item.state = "running"
        if cancel:
            self._finish(item, "cancelled")
            self._cancel_future(item.future)
            return
        if not item.future.set_running_or_notify_cancel():
            self._finish(item, "cancelled")
            return

        result: Any = None
        wall_ns = cpu_ns = 0
        if error is not None:
            outcome = "failed"
        elif not current:
            outcome = "retired"
            error = WorkerIOError(code="worker_io_owner_retired")
        else:
            wall_start, cpu_start = perf_counter_ns(), thread_time_ns()
            try:
                result = item.action()
                outcome = "completed"
            except BaseException as exc:
                error = exc
                outcome = "failed"
            finally:
                wall_ns = max(0, perf_counter_ns() - wall_start)
                cpu_ns = max(0, thread_time_ns() - cpu_start)
        self._finish(item, outcome, wall_ns, cpu_ns)
        try:
            if error is None:
                item.future.set_result(result)
            else:
                item.future.set_exception(error)
        except BaseException:
            # The Future is terminal even if a completion callback raises.
            pass

    def _finish(
        self, item: _Item, outcome: str, wall_ns: int = 0, cpu_ns: int = 0,
    ) -> None:
        with self._condition:
            self._release_locked(item, outcome)
            self._active = None
            self._action_wall_ns += wall_ns
            self._action_thread_cpu_ns += cpu_ns
        item.clear_callables()
