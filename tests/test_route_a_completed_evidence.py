"""A dispatch attempt must never be counted as a successful physical handoff."""
import base64
import hashlib
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from scheduler_pipeline import SchedulerPipelineMixin


@pytest.mark.parametrize("result_mode", ["success", "invalid_result", "provider_error"])
def test_completed_receipt_is_emitted_only_after_result_validation(caplog, result_mode):
    released = []
    def execute(*args):
        if result_mode == "provider_error":
            raise RuntimeError("remote failed")
        return SimpleNamespace(output={"token_argmax": 2 if result_mode == "success" else -1})
    provider = SimpleNamespace(
        reserve=lambda request: SimpleNamespace(reservation_id="res_fixture"),
        execute=execute,
        release=released.append,
    )
    sched = SimpleNamespace(
        _worker_prepare_assignment_loaded=lambda *args: (True, ""),
        _ensure_remote_task_worker_provider=lambda node: provider,
    )
    identity = {"operation_id": "supply_fixture", "config_id": "cfg_fixture", "generation": 9,
                "assignment_id": "asg_fixture", "artifact_sha256": "a" * 64}
    hidden = np.zeros((1, 4), dtype=np.float32)
    with caplog.at_level(logging.INFO, logger="scheduler_pipeline"):
        kwargs = dict(node_id="android", assignment={"start_layer": 8, "end_layer": 32,
                      "worker_prepare_identity": identity}, hidden_states=hidden, model_identity=None,
                      workflow_id="wf_fixture", request_id="fixture:step:0", stage_id="android:step:0",
                      context_size=128, want_hidden=False)
        if result_mode == "success":
            result = SchedulerPipelineMixin._execute_layer_stage_offer(sched, **kwargs)
            assert result == {"kind": "token", "token_argmax": 2}
        else:
            with pytest.raises((ValueError, RuntimeError)):
                SchedulerPipelineMixin._execute_layer_stage_offer(sched, **kwargs)
    receipts = [json.loads(record.getMessage().split("Route-A stage completed: ", 1)[1])
                for record in caplog.records if "Route-A stage completed: " in record.getMessage()]
    assert len(receipts) == (1 if result_mode == "success" else 0)
    assert released == ["res_fixture"]
    if receipts:
        receipt = receipts[0]
        assert all(receipt[key] == value for key, value in identity.items())
        assert receipt["layer_range"] == [8, 32]
        assert receipt["result_kind"] == "token"
        assert receipt["hidden_sha256"] == hashlib.sha256(hidden.tobytes()).hexdigest()
        assert receipt["tokens"] == 1
        assert receipt["completed_at"] > 0
