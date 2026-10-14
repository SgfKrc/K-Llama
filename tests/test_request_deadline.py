import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import request_deadline as deadline_mod
from request_deadline import (
    REQUEST_DEADLINE_EXCEEDED,
    DeadlineCancelEvent,
    RequestDeadline,
    RequestDeadlineExceeded,
    request_stop_reason,
)
from request_outcome import (
    MAX_SECONDARY_OBSERVATIONS,
    REASON_GENERATION_CANCELLED,
    TerminalOutcomeLatch,
)


def test_request_deadline_uses_monotonic_clock_and_clamps_stage_budget(
    monkeypatch,
):
    clock = {"monotonic": 100.0, "epoch": 1000.0}
    monkeypatch.setattr(
        deadline_mod.time, "monotonic", lambda: clock["monotonic"],
    )
    monkeypatch.setattr(deadline_mod.time, "time", lambda: clock["epoch"])

    deadline = RequestDeadline.start(10.0)
    assert deadline.expires_at_monotonic == 110.0
    assert deadline.expires_at_epoch == 1010.0

    clock["monotonic"] = 104.0
    clock["epoch"] = -5000.0
    assert deadline.remaining() == 6.0
    assert deadline.clamp(30.0) == 6.0

    clock["epoch"] = 9000.0
    assert deadline.remaining() == 6.0
    assert deadline.expired() is False

    clock["monotonic"] = 110.0
    assert deadline.expired() is True
    with pytest.raises(RequestDeadlineExceeded):
        deadline.require_remaining()


def test_deadline_cancel_event_keeps_cancel_and_timeout_semantics_distinct():
    external_cancel = threading.Event()
    expired = RequestDeadline(
        expires_at_monotonic=0.0,
        expires_at_epoch=0.0,
        timeout_seconds=1.0,
    )
    combined = DeadlineCancelEvent(external_cancel, expired)

    assert combined.is_set() is True
    assert combined.cancel_requested() is False
    assert combined.deadline_expired() is True
    assert request_stop_reason(combined, expired) == REQUEST_DEADLINE_EXCEEDED

    external_cancel.set()
    assert request_stop_reason(combined, expired) == REASON_GENERATION_CANCELLED


def test_terminal_outcome_latch_keeps_one_primary_reason():
    latch = TerminalOutcomeLatch()

    assert latch.decide(
        REQUEST_DEADLINE_EXCEEDED, source="request_deadline",
    ) == REQUEST_DEADLINE_EXCEEDED
    assert latch.decide(
        "worker_tcp_disconnected", source="heartbeat",
    ) == REQUEST_DEADLINE_EXCEEDED
    assert latch.decide(
        REASON_GENERATION_CANCELLED, source="cancel_event",
    ) == REQUEST_DEADLINE_EXCEEDED

    snapshot = latch.snapshot()
    assert snapshot["reason_code"] == REQUEST_DEADLINE_EXCEEDED
    assert snapshot["source"] == "request_deadline"
    assert [
        item["reason_code"] for item in snapshot["secondary_observations"]
    ] == ["worker_tcp_disconnected", REASON_GENERATION_CANCELLED]


def test_terminal_outcome_latch_deduplicates_and_bounds_secondary_observations():
    latch = TerminalOutcomeLatch()
    latch.decide(REQUEST_DEADLINE_EXCEEDED, source="request_deadline")

    for _ in range(10):
        latch.decide("worker_tcp_disconnected", source="heartbeat")
    for index in range(MAX_SECONDARY_OBSERVATIONS + 5):
        latch.decide(f"late_signal_{index}", source=f"source_{index}")

    snapshot = latch.snapshot()
    assert len(snapshot["secondary_observations"]) == (
        MAX_SECONDARY_OBSERVATIONS
    )
    assert sum(
        item["reason_code"] == "worker_tcp_disconnected"
        for item in snapshot["secondary_observations"]
    ) == 1
    assert snapshot["secondary_observations_dropped"] == 6
