"""Contracts for task 22e's isolated, bounded worker sender.

Events/barriers hold queue boundaries; no sockets or native engines are needed.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future, wait
import gc
from pathlib import Path
import sys
import threading
import time
import weakref

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import task_worker_io as worker_io  # noqa: E402
from task_worker_io import WorkerIOError, WorkerIOSender  # noqa: E402

_TIMEOUT = 5.0
pytestmark = pytest.mark.refactor_contract


def _current() -> bool:
    return True


def _submit(sender, action=lambda: None, *, charge_bytes=1, is_current=_current, **kwargs):
    return sender.submit(
        action, charge_bytes=charge_bytes, is_current=is_current, **kwargs,
    )


def _eventually(predicate):
    deadline = time.monotonic() + _TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, "sender did not reach the expected state"
        threading.Event().wait(0.005)


def _snapshot(sender):
    state = sender.snapshot()
    assert type(state["closed"]) is bool
    assert type(state["thread_alive"]) is bool
    for key, value in state.items():
        if key not in {"closed", "thread_alive"}:
            assert type(value) is int, key
            assert value >= 0, key
    assert state["submitted"] == sum(state[key] for key in (
        "pending_items", "completed", "failed", "retired", "cancelled",
    ))
    assert 0 <= state["queued_items"] <= state["pending_items"] <= state["max_items"]
    assert state["pending_bytes"] <= state["max_bytes"]
    assert state["pending_items"] <= state["peak_items"] <= state["max_items"]
    assert state["pending_bytes"] <= state["peak_bytes"] <= state["max_bytes"]
    return state


class _Senders:
    def __init__(self):
        self.created = []
        self.gates = []

    def create(self, **kwargs):
        sender = WorkerIOSender(**kwargs)
        self.created.append(sender)
        return sender

    def hold(self, sender, *, charge_bytes=1):
        entered, release = threading.Event(), threading.Event()
        self.gates.append(release)

        def action():
            entered.set()
            assert release.wait(_TIMEOUT), "test did not release the running action"
            return "released"

        future = _submit(sender, action, charge_bytes=charge_bytes)
        assert entered.wait(_TIMEOUT)
        return future, release


@pytest.fixture
def senders():
    factory = _Senders()
    try:
        yield factory
    finally:
        for gate in factory.gates:
            gate.set()
        for sender in factory.created:
            sender.close(wait=True)


class _IntSubclass(int):
    pass


def test_defaults_are_lazy_and_close_before_first_submit_is_final(senders):
    sender = senders.create()
    state = _snapshot(sender)
    assert state == {
        "closed": False, "max_items": 64, "max_bytes": 64 * 1024 * 1024,
        "pending_items": 0, "pending_bytes": 0, "queued_items": 0,
        "thread_alive": False, "submitted": 0, "completed": 0, "failed": 0,
        "retired": 0, "cancelled": 0, "peak_items": 0, "peak_bytes": 0,
        "queue_wait_ns": 0, "action_wall_ns": 0, "action_thread_cpu_ns": 0,
    }
    sender.close()
    sender.close(wait=True)
    with pytest.raises(WorkerIOError) as error:
        _submit(sender)
    assert error.value.code == "worker_io_closed"
    assert _snapshot(sender) == {**state, "closed": True}


@pytest.mark.parametrize("field", ["max_items", "max_bytes"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None, _IntSubclass(1)])
def test_limits_require_positive_plain_integers(field, value):
    with pytest.raises(WorkerIOError) as error:
        WorkerIOSender(**{field: value})
    assert error.value.code == "worker_io_budget_invalid"


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None, _IntSubclass(1)])
def test_charge_requires_a_nonnegative_plain_integer(senders, value):
    sender = senders.create()
    with pytest.raises(WorkerIOError) as error:
        _submit(sender, charge_bytes=value)
    assert error.value.code == "worker_io_budget_invalid"
    state = _snapshot(sender)
    assert state["submitted"] == state["pending_items"] == state["pending_bytes"] == 0
    assert not state["thread_alive"]


@pytest.mark.parametrize("field", ["action", "is_current"])
@pytest.mark.parametrize("value", [None, True, 1, "callable", object()])
def test_action_and_owner_check_must_be_callable(senders, field, value):
    sender = senders.create()
    arguments = {"action": lambda: None, "is_current": _current, "charge_bytes": 1}
    arguments[field] = value
    with pytest.raises(TypeError):
        sender.submit(**arguments)
    assert _snapshot(sender)["submitted"] == 0
    assert not sender.snapshot()["thread_alive"]


def test_callable_instances_and_zero_charge_are_supported(senders):
    class CallableObject:
        def __init__(self, result):
            self.result = result

        def __call__(self):
            return self.result

    sender = senders.create(max_items=1, max_bytes=1)
    sentinel = object()
    future = _submit(
        sender, CallableObject(sentinel), charge_bytes=0,
        is_current=CallableObject(True),
    )
    assert isinstance(future, Future)
    assert future.result(timeout=_TIMEOUT) is sentinel
    state = _snapshot(sender)
    assert state["completed"] == state["peak_items"] == 1
    assert state["pending_bytes"] == state["peak_bytes"] == 0


@pytest.mark.parametrize("options,active_charge,next_charge", [
    ({"max_items": 1, "max_bytes": 16}, 1, 0),
    ({"max_items": 4, "max_bytes": 4}, 4, 1),
])
def test_inflight_work_consumes_both_budgets(senders, options, active_charge, next_charge):
    sender = senders.create(**options)
    running, release = senders.hold(sender, charge_bytes=active_charge)
    before = _snapshot(sender)
    assert before["pending_items"] == 1
    assert before["pending_bytes"] == active_charge
    assert before["queued_items"] == 0
    for _ in range(8):
        with pytest.raises(WorkerIOError) as error:
            _submit(sender, charge_bytes=next_charge)
        assert error.value.code == "worker_io_full"
    assert _snapshot(sender) == before
    release.set()
    assert running.result(timeout=_TIMEOUT) == "released"
    assert _submit(sender, charge_bytes=next_charge).result(timeout=_TIMEOUT) is None
    assert _snapshot(sender)["pending_items"] == 0


def test_full_reservation_is_atomic_and_oversize_never_starts_a_thread(senders):
    sender = senders.create(max_items=4, max_bytes=10)
    with pytest.raises(WorkerIOError) as error:
        _submit(sender, charge_bytes=11)
    assert error.value.code == "worker_io_full"
    assert not _snapshot(sender)["thread_alive"]
    running, release = senders.hold(sender, charge_bytes=4)
    with pytest.raises(WorkerIOError):
        _submit(sender, charge_bytes=7)
    exact = _submit(sender, lambda: "exact", charge_bytes=6)
    state = _snapshot(sender)
    assert state["submitted"] == state["pending_items"] == 2
    assert state["pending_bytes"] == state["peak_bytes"] == 10
    assert state["queued_items"] == 1
    release.set()
    running.result(timeout=_TIMEOUT)
    assert exact.result(timeout=_TIMEOUT) == "exact"


def test_one_lazy_named_daemon_executes_fifo_and_reuses_idle_thread(senders):
    sender = senders.create(name="worker-io-fifo-contract", max_items=12)
    running, release = senders.hold(sender)
    order, owners = [], []

    def owner(index):
        owners.append(index)
        return True

    def action(index):
        thread = threading.current_thread()
        order.append(index)
        return index, thread.ident, thread.name, thread.daemon

    futures = [
        _submit(sender, lambda i=i: action(i), is_current=lambda i=i: owner(i))
        for i in range(8)
    ]
    assert order == owners == []
    assert _snapshot(sender)["queued_items"] == 8
    release.set()
    running.result(timeout=_TIMEOUT)
    results = [future.result(timeout=_TIMEOUT) for future in futures]
    assert order == owners == list(range(8))
    assert [result[0] for result in results] == list(range(8))
    assert len({result[1] for result in results}) == 1
    assert all(result[2:] == ("worker-io-fifo-contract", True) for result in results)
    later = _submit(sender, lambda: threading.get_ident()).result(timeout=_TIMEOUT)
    assert later == results[0][1]
    assert later != threading.get_ident()
    assert _snapshot(sender)["completed"] == 10


def test_concurrent_submit_never_waits_for_space_or_exceeds_reservations(senders):
    name = "worker-io-concurrent-contract"
    sender = senders.create(name=name, max_items=8, max_bytes=8)
    running, release = senders.hold(sender)
    producers = 24
    barrier = threading.Barrier(producers + 1)
    result_lock = threading.Lock()
    accepted, rejected, unexpected = [], [], []

    def produce():
        try:
            barrier.wait(timeout=_TIMEOUT)
            try:
                future = _submit(sender, threading.get_ident)
            except WorkerIOError as exc:
                with result_lock:
                    rejected.append(exc.code)
            else:
                with result_lock:
                    accepted.append(future)
        except BaseException as exc:
            with result_lock:
                unexpected.append(exc)

    threads = [threading.Thread(target=produce, daemon=True) for _ in range(producers)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=_TIMEOUT)
    for thread in threads:
        thread.join(timeout=_TIMEOUT)
        assert not thread.is_alive(), "submit blocked behind the held action"
    assert unexpected == []
    assert len(accepted) == 7
    assert rejected == ["worker_io_full"] * 17
    state = _snapshot(sender)
    assert state["submitted"] == state["pending_items"] == state["peak_items"] == 8
    assert state["pending_bytes"] == state["peak_bytes"] == 8
    assert state["queued_items"] == 7
    assert len([thread for thread in threading.enumerate() if thread.name == name]) == 1
    assert not any(future.done() for future in accepted)
    release.set()
    running.result(timeout=_TIMEOUT)
    assert len({future.result(timeout=_TIMEOUT) for future in accepted}) == 1
    assert _snapshot(sender)["completed"] == 8


@pytest.mark.parametrize("error_type", [ValueError, SystemExit, KeyboardInterrupt])
def test_action_exception_reaches_future_and_does_not_kill_sender(senders, error_type):
    sender = senders.create()
    running, release = senders.hold(sender)
    error = error_type("private-action-payload-path")

    def fail():
        raise error

    failed = _submit(sender, fail)
    healthy = _submit(sender, lambda: "healthy")
    release.set()
    running.result(timeout=_TIMEOUT)
    with pytest.raises(error_type) as raised:
        failed.result(timeout=_TIMEOUT)
    assert raised.value is error
    assert healthy.result(timeout=_TIMEOUT) == "healthy"
    state = _snapshot(sender)
    assert state["failed"] == 1
    assert state["completed"] == 2
    assert state["thread_alive"]
    assert "private-action-payload-path" not in repr(state)


@pytest.mark.parametrize("error_type", [LookupError, SystemExit])
def test_owner_exception_is_failed_work_and_later_work_runs(senders, error_type):
    sender = senders.create()
    called = []
    error = error_type("private-owner-path")

    def owner():
        raise error

    failed = _submit(sender, lambda: called.append("forbidden"), is_current=owner)
    with pytest.raises(error_type) as raised:
        failed.result(timeout=_TIMEOUT)
    assert raised.value is error
    assert _submit(sender, lambda: "next").result(timeout=_TIMEOUT) == "next"
    assert called == []
    state = _snapshot(sender)
    assert state["failed"] == state["completed"] == 1
    assert state["retired"] == 0
    assert "private-owner-path" not in repr(state)


def test_owner_is_checked_at_execution_and_retired_action_never_runs(senders):
    sender = senders.create()
    running, release = senders.hold(sender)
    owner_live = threading.Event()
    owner_live.set()
    checked, called = [], []

    def owner():
        checked.append("checked")
        return owner_live.is_set()

    retired = _submit(sender, lambda: called.append("forbidden"), is_current=owner)
    assert checked == called == []
    owner_live.clear()
    healthy = _submit(sender, lambda: "next-owner")
    release.set()
    running.result(timeout=_TIMEOUT)
    with pytest.raises(WorkerIOError) as error:
        retired.result(timeout=_TIMEOUT)
    assert error.value.code == "worker_io_owner_retired"
    assert healthy.result(timeout=_TIMEOUT) == "next-owner"
    assert checked == ["checked"]
    assert called == []
    state = _snapshot(sender)
    assert state["retired"] == 1
    assert state["failed"] == state["cancelled"] == 0
    assert state["completed"] == 2


def test_cancelled_queued_future_releases_budget_immediately_and_notifies_waiters(senders):
    sender = senders.create(max_items=2, max_bytes=10)
    running, release = senders.hold(sender, charge_bytes=4)
    called, checked, callbacks = [], [], []
    cancelled = _submit(
        sender, lambda: called.append("forbidden"), charge_bytes=6,
        is_current=lambda: checked.append("forbidden") or True,
    )
    cancelled.add_done_callback(lambda future: callbacks.append(_snapshot(sender)))
    assert cancelled.cancel()
    assert cancelled.cancel()  # idempotent, including accounting
    assert cancelled.cancelled()
    with pytest.raises(CancelledError):
        cancelled.result(timeout=0)
    done, pending = wait([cancelled], timeout=0)
    assert done == {cancelled} and pending == set()
    state = _snapshot(sender)
    assert state["pending_items"] == 1 and state["pending_bytes"] == 4
    assert state["queued_items"] == 0 and state["cancelled"] == 1
    assert callbacks == [state]
    replacement = _submit(sender, lambda: "replacement", charge_bytes=6)
    release.set()
    running.result(timeout=_TIMEOUT)
    assert replacement.result(timeout=_TIMEOUT) == "replacement"
    assert called == checked == []
    assert _snapshot(sender)["cancelled"] == 1


def test_cancelling_a_running_action_fails_and_fifo_survives_a_cancelled_hole(senders):
    sender = senders.create(max_items=8)
    running, release = senders.hold(sender)
    assert running.running()
    assert not running.cancel()
    order = []
    futures = [_submit(sender, lambda i=i: order.append(i) or i) for i in range(5)]
    assert futures[2].cancel()
    release.set()
    running.result(timeout=_TIMEOUT)
    for index, future in enumerate(futures):
        if index == 2:
            assert future.cancelled()
        else:
            assert future.result(timeout=_TIMEOUT) == index
    assert order == [0, 1, 3, 4]
    state = _snapshot(sender)
    assert state["cancelled"] == 1 and state["completed"] == 5


def test_close_cancels_queue_but_allows_inflight_action_and_reentrant_callbacks(senders):
    sender = senders.create(max_items=4, max_bytes=16)
    running, release = senders.hold(sender, charge_bytes=4)
    called, callback_states = [], []
    queued = [
        _submit(sender, lambda: called.append("forbidden"), charge_bytes=charge)
        for charge in (3, 5)
    ]

    def callback(future):
        callback_states.append(_snapshot(sender))
        sender.close()

    for future in queued:
        future.add_done_callback(callback)
    sender.close(wait=False)
    sender.close(wait=False)
    assert all(future.cancelled() for future in queued)
    assert wait(queued, timeout=0).done == set(queued)
    assert running.running() and not running.cancel()
    state = _snapshot(sender)
    assert state["closed"] and state["thread_alive"]
    assert state["pending_items"] == 1 and state["pending_bytes"] == 4
    assert state["cancelled"] == 2 and state["queued_items"] == 0
    assert len(callback_states) == 2
    assert all(value == state for value in callback_states)
    with pytest.raises(WorkerIOError) as error:
        _submit(sender)
    assert error.value.code == "worker_io_closed"
    release.set()
    assert running.result(timeout=_TIMEOUT) == "released"
    sender.close(wait=True)
    assert called == []
    state = _snapshot(sender)
    assert state["pending_items"] == state["pending_bytes"] == 0
    assert not state["thread_alive"]
    assert state["completed"] == 1


def test_close_during_owner_check_cancels_future_and_suppresses_action(senders):
    sender = senders.create()
    entered, release = threading.Event(), threading.Event()
    senders.gates.append(release)
    called = []

    def owner():
        entered.set()
        assert release.wait(_TIMEOUT)
        return True

    future = _submit(sender, lambda: called.append("forbidden"), charge_bytes=12, is_current=owner)
    assert entered.wait(_TIMEOUT)
    sender.close()
    assert future.cancelled()
    assert wait([future], timeout=0).done == {future}
    state = _snapshot(sender)
    assert state["pending_items"] == 1 and state["pending_bytes"] == 12
    assert state["queued_items"] == 0
    release.set()
    sender.close(wait=True)
    state = _snapshot(sender)
    assert state["pending_items"] == state["pending_bytes"] == 0
    assert state["cancelled"] == 1 and state["completed"] == 0
    assert not state["thread_alive"]
    assert called == []


def test_cancel_during_owner_check_suppresses_action_and_keeps_active_charge(senders):
    sender = senders.create(max_items=1, max_bytes=12)
    entered, release = threading.Event(), threading.Event()
    senders.gates.append(release)
    called = []

    def owner():
        entered.set()
        assert release.wait(_TIMEOUT)
        return True

    future = _submit(sender, lambda: called.append("forbidden"), charge_bytes=12, is_current=owner)
    assert entered.wait(_TIMEOUT)
    assert future.cancel()
    assert wait([future], timeout=0).done == {future}
    assert _snapshot(sender)["pending_bytes"] == 12
    with pytest.raises(WorkerIOError) as error:
        _submit(sender, charge_bytes=0)
    assert error.value.code == "worker_io_full"
    release.set()
    _eventually(lambda: sender.snapshot()["pending_items"] == 0)
    assert _submit(sender, lambda: "next").result(timeout=_TIMEOUT) == "next"
    assert called == []
    assert _snapshot(sender)["cancelled"] == 1


def test_close_wait_is_bounded_and_repeated_close_can_join_later(senders):
    sender = senders.create()
    running, release = senders.hold(sender)
    start = time.monotonic()
    sender.close(wait=True)
    assert time.monotonic() - start < 2.5
    assert running.running()
    assert _snapshot(sender)["thread_alive"]
    release.set()
    assert running.result(timeout=_TIMEOUT) == "released"
    sender.close(wait=True)
    sender.close(wait=True)
    assert not _snapshot(sender)["thread_alive"]


def test_idle_worker_exits_after_nonwaiting_close(senders):
    sender = senders.create()
    assert _submit(sender).result(timeout=_TIMEOUT) is None
    assert _snapshot(sender)["thread_alive"]
    sender.close(wait=False)
    _eventually(lambda: not sender.snapshot()["thread_alive"])
    assert _snapshot(sender)["pending_items"] == 0


def _probe_from_another_thread(operation):
    """A separate observer also detects an incorrectly held reentrant mutex."""
    done = threading.Event()
    values, errors = [], []

    def observe():
        try:
            values.append(operation())
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    assert done.wait(_TIMEOUT), "caller code ran while the sender held its queue lock"
    observer.join(timeout=_TIMEOUT)
    assert not observer.is_alive()
    if errors:
        raise errors[0]
    return values[0]


def test_owner_action_and_completion_callback_run_outside_queue_lock(senders):
    sender = senders.create()
    running, release = senders.hold(sender)
    phases, nested, errors = [], [], []
    callback_done = threading.Event()

    def observe(phase):
        state = _probe_from_another_thread(lambda: _snapshot(sender))
        phases.append(phase)
        return state

    def owner():
        assert observe("owner")["pending_items"] == 1
        return True

    def action():
        assert observe("action")["pending_items"] == 1
        return 42

    def callback(future):
        try:
            assert observe("callback")["pending_items"] == 0
            nested.append(_probe_from_another_thread(lambda: _submit(sender, lambda: "nested")))
        except BaseException as exc:
            errors.append(exc)
        finally:
            callback_done.set()

    future = _submit(sender, action, is_current=owner)
    future.add_done_callback(callback)
    release.set()
    running.result(timeout=_TIMEOUT)
    assert future.result(timeout=_TIMEOUT) == 42
    assert callback_done.wait(_TIMEOUT)
    assert errors == []
    assert nested[0].result(timeout=_TIMEOUT) == "nested"
    assert phases == ["owner", "action", "callback"]


@pytest.mark.parametrize("phase", ["action", "owner", "callback"])
def test_close_wait_from_sender_thread_does_not_join_itself(senders, phase):
    sender = senders.create()
    running, release = senders.hold(sender)
    errors, callback_done = [], threading.Event()

    def owner():
        if phase == "owner":
            sender.close(wait=True)
        return True

    def action():
        if phase == "action":
            sender.close(wait=True)
        return "done"

    def callback(future):
        try:
            if phase == "callback":
                sender.close(wait=True)
        except BaseException as exc:
            errors.append(exc)
        finally:
            callback_done.set()

    future = _submit(sender, action, is_current=owner)
    future.add_done_callback(callback)
    release.set()
    running.result(timeout=_TIMEOUT)
    if phase == "owner":
        with pytest.raises(CancelledError):
            future.result(timeout=_TIMEOUT)
    else:
        assert future.result(timeout=_TIMEOUT) == "done"
    assert callback_done.wait(_TIMEOUT)
    assert errors == []
    sender.close(wait=True)
    assert not _snapshot(sender)["thread_alive"]


def test_baseexception_in_completion_callback_does_not_kill_sender(senders):
    sender = senders.create()
    running, release = senders.hold(sender)
    future = _submit(sender, lambda: "completed")

    def callback(done):
        raise SystemExit("private-callback-path")

    future.add_done_callback(callback)
    next_future = _submit(sender, lambda: "next")
    release.set()
    running.result(timeout=_TIMEOUT)
    assert future.result(timeout=_TIMEOUT) == "completed"
    assert next_future.result(timeout=_TIMEOUT) == "next"
    state = _snapshot(sender)
    assert state["completed"] == 3 and state["failed"] == 0
    assert state["thread_alive"]
    assert "private-callback-path" not in repr(state)


def test_baseexception_in_close_callback_does_not_strand_other_cancellations(senders):
    sender = senders.create()
    running, release = senders.hold(sender)
    futures = [_submit(sender) for _ in range(3)]

    def callback(done):
        _probe_from_another_thread(lambda: _snapshot(sender))
        raise KeyboardInterrupt("private-cancellation-path")

    futures[0].add_done_callback(callback)
    sender.close()
    assert all(future.cancelled() for future in futures)
    assert wait(futures, timeout=0).done == set(futures)
    assert _snapshot(sender)["cancelled"] == 3
    release.set()
    assert running.result(timeout=_TIMEOUT) == "released"


def test_concurrent_close_and_submit_has_one_terminal_outcome_per_reservation(senders):
    sender = senders.create(max_items=32, max_bytes=32)
    running, release = senders.hold(sender)
    producers = 16
    barrier = threading.Barrier(producers + 2)
    result_lock = threading.Lock()
    futures, rejected, errors = [], [], []

    def produce():
        try:
            barrier.wait(timeout=_TIMEOUT)
            try:
                future = _submit(sender)
            except WorkerIOError as exc:
                with result_lock:
                    rejected.append(exc.code)
            else:
                with result_lock:
                    futures.append(future)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    def close():
        try:
            barrier.wait(timeout=_TIMEOUT)
            sender.close()
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=produce, daemon=True) for _ in range(producers)]
    threads.append(threading.Thread(target=close, daemon=True))
    for thread in threads:
        thread.start()
    barrier.wait(timeout=_TIMEOUT)
    for thread in threads:
        thread.join(timeout=_TIMEOUT)
        assert not thread.is_alive()
    assert errors == []
    assert len(futures) + len(rejected) == producers
    assert all(code == "worker_io_closed" for code in rejected)
    assert all(future.cancelled() for future in futures)
    state = _snapshot(sender)
    assert state["closed"]
    assert state["submitted"] == 1 + len(futures)
    assert state["cancelled"] == len(futures)
    assert state["pending_items"] == state["pending_bytes"] == 1
    release.set()
    running.result(timeout=_TIMEOUT)


def test_concurrent_cancel_and_close_never_double_release(senders):
    sender = senders.create(max_items=32)
    running, release = senders.hold(sender)
    futures = [_submit(sender) for _ in range(16)]
    barrier = threading.Barrier(len(futures) + 2)
    errors = []
    error_lock = threading.Lock()

    def cancel(future):
        try:
            barrier.wait(timeout=_TIMEOUT)
            if future is None:
                sender.close()
            else:
                assert future.cancel()
        except BaseException as exc:
            with error_lock:
                errors.append(exc)

    threads = [threading.Thread(target=cancel, args=(future,), daemon=True) for future in futures]
    threads.append(threading.Thread(target=cancel, args=(None,), daemon=True))
    for thread in threads:
        thread.start()
    barrier.wait(timeout=_TIMEOUT)
    for thread in threads:
        thread.join(timeout=_TIMEOUT)
        assert not thread.is_alive()
    assert errors == []
    assert all(future.cancelled() for future in futures)
    assert wait(futures, timeout=0).done == set(futures)
    state = _snapshot(sender)
    assert state["cancelled"] == len(futures)
    assert state["pending_items"] == state["pending_bytes"] == 1
    release.set()
    running.result(timeout=_TIMEOUT)
    sender.close(wait=True)
    assert _snapshot(sender)["pending_bytes"] == 0


def test_snapshot_timings_are_aggregate_and_exclude_owner_check_from_action_time(senders, monkeypatch):
    clock = {"wall": 1_000, "cpu": 100}
    monkeypatch.setattr(worker_io, "perf_counter_ns", lambda: clock["wall"])
    monkeypatch.setattr(worker_io, "thread_time_ns", lambda: clock["cpu"])
    sender = senders.create(max_items=8)
    running, release = senders.hold(sender)

    def owner():
        clock["wall"] += 1_000
        return True

    def success():
        clock["wall"] += 700
        clock["cpu"] += 60
        return "ok"

    def retired_owner():
        clock["wall"] += 900
        return False

    def failure():
        clock["wall"] += 300
        clock["cpu"] += 30
        raise RuntimeError("private-exception-path")

    successful = _submit(sender, success, is_current=owner, description="private-payload-path")
    retired = _submit(sender, lambda: pytest.fail("retired action ran"), is_current=retired_owner)
    failed = _submit(sender, failure)
    cancelled = _submit(sender, lambda: pytest.fail("cancelled action ran"))
    assert cancelled.cancel()
    clock["wall"] = 1_100
    release.set()
    running.result(timeout=_TIMEOUT)
    assert successful.result(timeout=_TIMEOUT) == "ok"
    with pytest.raises(WorkerIOError):
        retired.result(timeout=_TIMEOUT)
    with pytest.raises(RuntimeError, match="private-exception-path"):
        failed.result(timeout=_TIMEOUT)
    state = _snapshot(sender)
    assert state["submitted"] == 5
    assert state["completed"] == 2
    assert state["retired"] == state["failed"] == state["cancelled"] == 1
    assert state["queue_wait_ns"] == 100 + 1_800 + 2_700
    assert state["action_wall_ns"] == 100 + 700 + 300
    assert state["action_thread_cpu_ns"] == 60 + 30
    assert state["peak_items"] == state["peak_bytes"] == 5
    assert state["pending_items"] == state["pending_bytes"] == 0
    assert "private-" not in repr(state)
    state["completed"] = -999
    assert _snapshot(sender)["completed"] == 2


class _Payload:
    def __init__(self):
        self.data = b"private-payload" * 16


def _submit_captured_payload(sender):
    payload = _Payload()

    def action(payload=payload):
        return len(payload.data)

    def owner(payload=payload):
        return bool(payload.data)

    references = [weakref.ref(value) for value in (payload, action, owner)]
    future = _submit(sender, action, is_current=owner, charge_bytes=len(payload.data))
    return future, references


def test_idle_worker_retains_no_completed_payload_callable_or_future(senders):
    sender = senders.create()
    futures, references = [], []
    for _ in range(16):
        future, refs = _submit_captured_payload(sender)
        assert future.result(timeout=_TIMEOUT) == len(b"private-payload" * 16)
        futures.append(future)
        references.extend(refs)
    _eventually(lambda: all(reference() is None for reference in references))
    future_refs = [weakref.ref(future) for future in futures]
    futures.clear()
    del future
    gc.collect()
    _eventually(lambda: all(reference() is None for reference in future_refs))
    state = _snapshot(sender)
    assert state["thread_alive"] and state["completed"] == 16
    assert state["pending_items"] == state["pending_bytes"] == 0


@pytest.mark.parametrize("close", [False, True])
def test_cancelled_queue_releases_payloads_even_while_action_is_blocked(senders, close):
    sender = senders.create()
    running, release = senders.hold(sender)
    future, references = _submit_captured_payload(sender)
    if close:
        sender.close()
    else:
        assert future.cancel()
    assert future.cancelled()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert _snapshot(sender)["pending_items"] == 1
    release.set()
    running.result(timeout=_TIMEOUT)


def test_sender_does_not_keep_failed_future_or_exception_payload(senders):
    sender = senders.create()

    def submit_failure():
        payload = _Payload()

        def fail(payload=payload):
            raise RuntimeError("private-error-path", payload)

        return _submit(sender, fail), weakref.ref(payload)

    failed, payload_ref = submit_failure()
    with pytest.raises(RuntimeError):
        failed.result(timeout=_TIMEOUT)
    failed_ref = weakref.ref(failed)
    del failed
    # Advancing the worker also ensures its previous stack frame has returned.
    assert _submit(sender, lambda: "next").result(timeout=_TIMEOUT) == "next"
    gc.collect()
    assert failed_ref() is None
    assert payload_ref() is None
    assert _snapshot(sender)["failed"] == 1


def test_completed_future_callback_does_not_retain_closed_sender():
    sender = WorkerIOSender()
    try:
        future = _submit(sender, lambda: "result")
        assert future.result(timeout=_TIMEOUT) == "result"
    finally:
        sender.close(wait=True)
    sender_ref = weakref.ref(sender)
    del sender
    gc.collect()
    assert sender_ref() is None
    assert future.result() == "result"


def test_thread_start_failure_rolls_back_reservation_and_allows_retry(senders, monkeypatch):
    name = "worker-io-start-failure-contract"
    original_start = threading.Thread.start
    attempts = []

    def start(thread):
        if thread.name == name and not attempts:
            attempts.append(thread.name)
            raise RuntimeError("thread unavailable")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    sender = senders.create(name=name, max_items=1, max_bytes=1)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        _submit(sender)
    state = _snapshot(sender)
    assert state["submitted"] == state["pending_items"] == state["pending_bytes"] == 0
    assert state["queued_items"] == state["peak_items"] == state["peak_bytes"] == 0
    assert not state["thread_alive"]
    assert _submit(sender, lambda: "retried").result(timeout=_TIMEOUT) == "retried"
    assert _snapshot(sender)["completed"] == 1
