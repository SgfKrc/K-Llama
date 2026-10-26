"""Cache maintenance follows finalization and yields to new execution work."""
import threading
from types import SimpleNamespace

import pytest

from tests.test_worker_prepare import command, drain, identity, run, setup as prepare_setup


def test_retention_runs_only_after_finalize_ack_and_command_queue_drain(prepare_setup):
    env = prepare_setup
    calls = []
    finished = threading.Event()

    def collect(*, cancelled):
        calls.append((env.scheduler._tcp_client.acks[-1]["phase"], cancelled()))
        assert env.controller._runner is None
        assert env.controller._current is None
        finished.set()
        return {"removed": 0}

    env.scheduler._model_supply_worker = SimpleNamespace(collect_garbage=collect)
    value = identity()
    run(env, "prepare", value)
    run(env, "commit", value)
    assert not calls
    run(env, "finalize", value)
    assert finished.wait(3)
    assert calls == [("finalize", False)]


def test_new_prepare_preempts_retention_holding_execution_admission(prepare_setup):
    env = prepare_setup
    started, interrupted = threading.Event(), threading.Event()

    def collect(*, cancelled):
        with env.scheduler._host.full_chat_execution_lock:
            started.set()
            for _ in range(300):
                if cancelled():
                    interrupted.set()
                    return {"removed": 0}
                threading.Event().wait(0.01)
            pytest.fail("maintenance did not yield to queued prepare")

    env.scheduler._model_supply_worker = SimpleNamespace(collect_garbage=collect)
    value = identity()
    run(env, "prepare", value)
    run(env, "commit", value)
    assert env.controller.handle(command("finalize", value))
    assert started.wait(3)
    assert env.controller.handle(command("prepare", identity(2)))
    assert interrupted.wait(3)
    drain(env.controller)
    assert env.scheduler._tcp_client.acks[-1]["status"] == "prepared"
    assert len(env.built) == 2


@pytest.mark.parametrize("change", ["attempt", "pipeline", "connection", "mode"])
def test_retention_callback_fences_new_work_and_scope_changes(prepare_setup, change):
    env = prepare_setup
    finished = threading.Event()

    def collect(*, cancelled):
        assert not cancelled()
        if change == "attempt":
            env.scheduler._task_worker_active_attempts = {"attempt": object()}
        elif change == "pipeline":
            env.scheduler._active_pipeline_task_ids = {"request"}
        elif change == "connection":
            env.scheduler._tcp_client._connection_generation += 1
        else:
            env.scheduler.enabled = False
        assert cancelled()
        finished.set()
        return {"removed": 0}

    env.scheduler._model_supply_worker = SimpleNamespace(collect_garbage=collect)
    value = identity()
    run(env, "prepare", value)
    run(env, "commit", value)
    run(env, "finalize", value)
    assert finished.wait(3)


def test_failed_finalize_never_starts_retention(prepare_setup, monkeypatch):
    env = prepare_setup
    calls = []
    env.scheduler._model_supply_worker = SimpleNamespace(
        collect_garbage=lambda **kwargs: calls.append(kwargs))
    value = identity()
    run(env, "prepare", value)
    run(env, "commit", value)

    def fail(identity):
        raise RuntimeError("native finalization failed")

    monkeypatch.setattr(env.host, "finalize_worker_runtime", fail)
    assert run(env, "finalize", value)["status"] == "error"
    assert calls == []


def test_retention_failure_cannot_change_finalize_ack(prepare_setup):
    env = prepare_setup
    finished = threading.Event()

    def collect(*, cancelled):
        finished.set()
        raise RuntimeError("maintenance failure")

    env.scheduler._model_supply_worker = SimpleNamespace(collect_garbage=collect)
    value = identity()
    run(env, "prepare", value)
    run(env, "commit", value)
    assert run(env, "finalize", value)["status"] == "finalized"
    assert finished.wait(3)
    assert [ack["phase"] for ack in env.scheduler._tcp_client.acks] == ["prepare", "commit", "finalize"]
