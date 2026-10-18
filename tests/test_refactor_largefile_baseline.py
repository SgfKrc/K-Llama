"""R5 ownership replaces the pinned large-file/facade characterization."""
import threading
import time
from types import SimpleNamespace
import pytest
import api_server
from scheduler import Scheduler, NodeInfo, NodeRole, NodeState
from scheduler_pipeline import SchedulerPipelineMixin
pytestmark = pytest.mark.refactor_contract

def _wait_for(event, timeout=3):
    assert event.wait(timeout)

def test_scheduler_uses_instance_settings_after_module_edits(monkeypatch):
    import scheduler as module
    owner = Scheduler()
    expected = owner._runtime_setting("RUN_MODE")
    monkeypatch.setattr(module, "RUN_MODE", "unrelated")
    assert owner._runtime_setting("RUN_MODE") == expected

def test_published_executor_has_no_legacy_queue_or_decode_entry():
    import scheduler as module
    assert not hasattr(module, "PipelineQueue")
    assert not hasattr(SchedulerPipelineMixin, "_run_route_a_stage_pipeline")
    assert not hasattr(SchedulerPipelineMixin, "_run_pipeline")

def test_handle_infer_forward_runs_pipeline_and_returns_result(monkeypatch) -> None:
    import local_store

    instance = Scheduler()
    sent: list[dict] = []
    completed: list[tuple[str, str, dict]] = []
    started = threading.Event()
    monkeypatch.setattr(local_store, "save_local_conversation_turn", lambda **_: None)
    monkeypatch.setattr(instance, "start_infer_task", lambda prompt, request_id=None: "task-forward")
    monkeypatch.setattr(
        instance,
        "complete_infer_task",
        lambda task_id, content, metrics: completed.append((task_id, content, metrics)),
    )
    monkeypatch.setattr(instance, "record_task_complete", lambda **_: None)
    monkeypatch.setattr(
        instance,
        "run_pipeline_safe",
        lambda **kwargs: started.set() or {
            "response": "forwarded",
            "thinking": "",
            "metrics": {"distributed_used": True, "engine": "test"},
        },
    )
    monkeypatch.setattr(
        instance,
        "_send_infer_result",
        lambda client_id, task_id, content, metrics=None, **kwargs: sent.append({
            "client_id": client_id,
            "task_id": task_id,
            "content": content,
            "metrics": metrics,
            **kwargs,
        }),
    )

    instance.handle_infer_forward("worker-1", {
        "data": {
            "prompt": "hello",
            "forward_request_id": "forward-1",
            "request_id": "request-1",
            "routing_preference": "distributed_required",
        },
    })

    _wait_for(started)
    deadline = time.monotonic() + 3.0
    while (not sent or instance._forward_cancel_events) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sent == [{
        "client_id": "worker-1",
        "task_id": "task-forward",
        "content": "forwarded",
        "metrics": {"distributed_used": True, "engine": "test"},
        "thinking_content": "",
        "forward_request_id": "forward-1",
    }]
    assert completed == [
        ("task-forward", "forwarded", {"distributed_used": True, "engine": "test"}),
    ]
    assert not instance._forward_cancel_events

def test_handle_infer_forward_rejects_full_concurrency_slot(monkeypatch) -> None:
    instance = Scheduler()
    sent: list[dict] = []

    class FullSlots:
        def acquire(self, blocking=False):
            return False

        def release(self):
            raise AssertionError("a rejected request must not release a slot")

    instance._forward_infer_slots = FullSlots()
    monkeypatch.setattr(
        instance,
        "_send_infer_result",
        lambda *args, **kwargs: sent.append({"args": args, "kwargs": kwargs}),
    )
    instance.handle_infer_forward("worker-1", {
        "data": {"forward_request_id": "full-1", "prompt": "hello"},
    })

    assert len(sent) == 1
    assert sent[0]["kwargs"]["status"] == "error"
    assert "并发上限" in sent[0]["kwargs"]["error"]

def test_handle_infer_forward_cancellation_reaches_pipeline(monkeypatch) -> None:
    instance = Scheduler()
    pipeline_started = threading.Event()
    sent: list[dict] = []
    observed_cancel: list[threading.Event] = []
    monkeypatch.setattr(instance, "start_infer_task", lambda prompt, request_id=None: "task-cancel")
    monkeypatch.setattr(instance, "fail_infer_task", lambda task_id, error: None)
    monkeypatch.setattr(
        instance,
        "run_pipeline_safe",
        lambda **kwargs: (
            observed_cancel.append(kwargs["_cancel_event"]),
            pipeline_started.set(),
            kwargs["_cancel_event"].wait(2.0),
            {"response": "", "metrics": {}, "error": "cancelled"},
        )[-1],
    )
    monkeypatch.setattr(
        instance,
        "_send_infer_result",
        lambda *args, **kwargs: sent.append({"args": args, "kwargs": kwargs}),
    )

    instance.handle_infer_forward("worker-1", {
        "data": {"forward_request_id": "cancel-1", "prompt": "hello"},
    })
    _wait_for(pipeline_started)
    instance._on_tcp_message("worker-1", {
        "type": "infer_cancel",
        "data": {"forward_request_id": "cancel-1"},
    })

    deadline = time.monotonic() + 3.0
    while (not sent or instance._forward_cancel_events) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert observed_cancel and observed_cancel[0].is_set()
    assert sent[0]["kwargs"]["status"] == "error"
    assert sent[0]["kwargs"]["error"] == "cancelled"
    assert not instance._forward_cancel_events
