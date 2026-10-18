"""One advancing clock origin per externally timed operation."""
import time


def operation_clock(now_ms=None):
    if now_ms is None:
        return lambda: int(time.time() * 1000)
    if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
        raise ValueError("now_ms must be a nonnegative integer")
    wall_start = int(time.time() * 1000)
    monotonic_start = time.monotonic()
    return lambda: int(now_ms) + max(0, int(time.time() * 1000) - wall_start,
                                    int((time.monotonic() - monotonic_start) * 1000))
