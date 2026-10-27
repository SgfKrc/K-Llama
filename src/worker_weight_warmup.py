"""Non-blocking page warming for layer-segment weight files.

The 27.9 s cold-start cost sits on the request critical path: the first forward
pass of a segment touches every weight page it needs, and until then each touch
is a synchronous major fault. The pages are needed anyway; the cost is *when*
they are faulted in, not *whether*.

`WeightWarmup` moves that work off the critical path by issuing page-in advice
from a background thread while the previous segment is still computing or while
the scheduler waits for the next hidden state. It is deliberately advisory:

* it never blocks the caller — `warm()` enqueues and returns immediately;
* it never raises into the request path — every failure is counted and logged;
* it never changes what is read, so results are unaffected;
* `disable()` is honoured before queueing, so an operator can turn it off
  without restarting the process.

The address-range approach follows the发射点 idea from prima.cpp's
`manage_graph_tensors` (advise after send, before the next compute), adapted to
a file-backed Python runtime: instead of per-tensor `data` pointers we advise
page-aligned file ranges, which is what this process actually faults on.

Note on scope: this only helps the *first* pass after a page cache eviction. Once
the segment is resident, repeated advice is a no-op. The layer-segment case is
exactly that — `[8,24)` is 3.47 GB in a 31.6 GB host, so it stays resident — and
therefore this module targets cold start only, not per-token decode.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass, field

__all__ = ["WarmupStats", "WeightWarmup"]

logger = logging.getLogger("worker_weight_warmup")

_PAGE = 4096


def _aligned(window: tuple[int, int]) -> tuple[int, int]:
    start, end = window
    return (start - (start % _PAGE), end + ((_PAGE - end % _PAGE) % _PAGE))


def merge_windows(windows, *, min_gap: int = 1 << 20):
    """Sort, merge and page-align byte windows; skip gaps below `min_gap`.

    Merging happens on the raw offsets first and alignment last, because aligning
    before merging collapses small windows into full pages and then makes
    genuinely adjacent ranges look separated by a page-sized gap.
    """
    cleaned = []
    for window in windows:
        if not isinstance(window, (tuple, list)) or len(window) != 2:
            continue
        try:
            start, end = int(window[0]), int(window[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        cleaned.append((start, end))
    cleaned.sort()
    merged = []
    for start, end in cleaned:
        if merged and start - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [_aligned(window) for window in merged]


@dataclass
class WarmupStats:
    requested: int = 0
    advised: int = 0
    bytes_advised: int = 0
    failures: int = 0
    skipped_disabled: int = 0
    last_error: str = ""
    windows: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "requested": self.requested,
                "advised": self.advised,
                "bytes_advised": self.bytes_advised,
                "failures": self.failures,
                "skipped_disabled": self.skipped_disabled,
                "last_error": self.last_error,
                "windows": self.windows,
            }


class WeightWarmup:
    """Background page-in advisor for one or more weight files."""

    def __init__(self, *, enabled: bool = True, min_gap: int = 1 << 20):
        self._enabled = bool(enabled)
        self._min_gap = min_gap
        self._queue: queue.Queue = queue.Queue()
        self._stats = WarmupStats()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- control ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def stats(self) -> dict:
        return self._stats.snapshot()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._thread is not None:
                return  # a stopped worker is not restarted implicitly
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="qlh-weight-warmup", daemon=True)
            self._thread.start()

    # -- public API -------------------------------------------------------

    def warm(self, path: str, windows=None) -> bool:
        """Queue a page-in for `path`.

        `windows` is an iterable of `(start, end)` byte offsets; `None` means the
        whole file. Returns True when the request was queued. Never blocks and
        never raises.
        """
        if not self._enabled:
            with self._stats.lock:
                self._stats.skipped_disabled += 1
            return False
        try:
            if not path or not os.path.isfile(path):
                return False
            size = os.path.getsize(path)
            if windows is None:
                merged = [(0, size)]
            else:
                merged = merge_windows(windows, min_gap=self._min_gap)
            if not merged:
                return False
            with self._stats.lock:
                self._stats.requested += 1
                self._stats.windows += len(merged)
            self._queue.put((path, merged, size))
            self._ensure_thread()
            return True
        except Exception:  # noqa: BLE001 — advisory only, never into the request path
            with self._stats.lock:
                self._stats.failures += 1
            return False

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until the queue drains. For tests and explicit sync points only."""
        deadline = threading.Event()
        try:
            self._queue.join()
            return True
        except Exception:  # pragma: no cover - join does not raise normally
            return False
        finally:
            deadline.set()

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._queue.put(None)
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # -- worker -----------------------------------------------------------

    def _advise(self, fd: int, start: int, end: int) -> None:
        """Advise the OS that this range is about to be read.

        POSIX exposes `posix_fadvise(POSIX_FADV_WILLNEED)`. Windows has no such
        call, and `os.pread` is also POSIX-only, so the fallback seeks and reads
        one byte per page. That both faults the page in and populates the
        standby list, which is what the OS would otherwise do lazily and
        synchronously inside the first forward pass.
        """
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_WILLNEED)
            return
        offset = start
        while offset < end:
            os.lseek(fd, offset, os.SEEK_SET)
            os.read(fd, 1)
            offset += _PAGE

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            try:
                if item is None:
                    break
                path, merged, size = item
                try:
                    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                except OSError as exc:
                    with self._stats.lock:
                        self._stats.failures += 1
                        self._stats.last_error = f"open:{exc.__class__.__name__}"
                    logger.debug("weight warmup open failed path=%s", path, exc_info=True)
                    continue
                try:
                    for start, end in merged:
                        start = max(0, min(start, size))
                        end = max(start, min(end, size))
                        if end <= start:
                            continue
                        try:
                            self._advise(fd, start, end)
                        except OSError as exc:
                            with self._stats.lock:
                                self._stats.failures += 1
                                self._stats.last_error = f"advise:{exc.__class__.__name__}"
                            continue
                        with self._stats.lock:
                            self._stats.advised += 1
                            self._stats.bytes_advised += end - start
                finally:
                    os.close(fd)
            except Exception:  # noqa: BLE001 — the worker must outlive any one item
                logger.debug("weight warmup item failed", exc_info=True)
                with self._stats.lock:
                    self._stats.failures += 1
            finally:
                self._queue.task_done()
