import os
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference_service.engine_host import EngineHost
from worker_assignment_state import WorkerAssignmentRegistry


def test_engine_host_model_change_clears_assignment_control_plane():
    calls = []
    assignments = WorkerAssignmentRegistry()
    assignments.begin(
        "worker",
        expected_config={"config_id": "cfg-old", "generation": 7},
    )
    retry_state = {"worker": {"attempts": 2, "next_retry": 10.0}}
    scheduler = SimpleNamespace(
        _inference_lock=threading.RLock(),
        _layer_execution_lock=threading.RLock(),
        _layer_config_lock=threading.RLock(),
        _worker_assignments=assignments,
        _layer_config_retry_state=retry_state,
        _pipeline_load_transaction={
            "config_id": "cfg-old",
            "phase": "ready",
            "plan": {"admitted": True, "plan_id": "plan-old"},
        },
        _active_pipeline_capacity_plan={
            "admitted": True, "plan_id": "plan-old",
        },
        _active_layer_config={"config_id": "cfg-old"},
        _last_layer_config_ack_payload={"status": "ready"},
        _local_pipeline_steps={"task": {"step": 1}},
        _invalidate_pipeline_load_transaction=(
            lambda **kwargs: calls.append(("invalidated", kwargs))
        ),
    )
    host = object.__new__(EngineHost)
    host._host = SimpleNamespace(full_chat_execution_lock=threading.RLock())
    host._scheduler = scheduler
    host._refresh_pipeline_layer_config = (
        lambda current: calls.append(("refreshed", current))
    )

    result = host._run_exclusive_model_change(
        lambda: {"success": True},
    )

    assert result == {"success": True}
    assert assignments.snapshot() == {}
    assert retry_state == {}
    assert scheduler._pipeline_load_transaction is None
    assert scheduler._active_pipeline_capacity_plan is None
    assert scheduler._active_layer_config is None
    assert scheduler._last_layer_config_ack_payload is None
    assert scheduler._local_pipeline_steps == {}
    assert calls[0][0] == "invalidated"
    assert calls[-1] == ("refreshed", scheduler)
