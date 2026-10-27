"""Committed read reuse preserves live admission at stage-return checkpoints."""
from copy import deepcopy

import pytest

from qlh_core.control.deployment import DeploymentService
from tests.test_refactor_deployment_owner import control
from tests.test_refactor_execution import repository
from tests.test_refactor_execution_integration import application


pytestmark = pytest.mark.refactor_contract


@pytest.mark.parametrize("mode", ["legacy", "execution", "query", "combined"])
@pytest.mark.parametrize("fault", ["revoke", "reconnect"])
def test_stage_return_rejects_changed_deployment_or_connection(application, control, mode, fault):
    app, calls, state = application
    owner, model_repository, _ = control
    model_repository.execution_reads_enabled = mode in {"execution", "combined"}
    model_repository.query_reads_enabled = mode in {"query", "combined"}
    owner = DeploymentService(model_repository, boot_id="boot", clock=lambda: 1.0)
    owner.start()
    app.scheduler._model_control = owner
    live = {"connection": 1, "incarnation": "worker-boot"}
    app.scheduler._model_control_connections = lambda: {"worker": live["connection"]}
    app.scheduler._task_worker_control.worker_snapshot = lambda _node: {
        "capabilities": {"stage_kv_cleanup": True, "node_incarnation": live["incarnation"]}}
    model_repository.transact(lambda saved: saved.update(deepcopy(state)))
    original = app.scheduler._execute_layer_stage_offer
    def change_after_compute(**kwargs):
        result = original(**kwargs)
        if not kwargs.get("cleanup_only"):
            if fault == "revoke":
                model_repository.transact(lambda saved: saved.update(active_deployment=""))
            elif fault == "reconnect":
                live.update(connection=2, incarnation="worker-restarted")
        return result
    app.scheduler._execute_layer_stage_offer = change_after_compute
    result = app.run("hello", max_new_tokens=2)
    assert result["reason_code"] == "execution_deployment_fenced"
    assert result["response"] == ""
    assert [call[0] for call in calls].count("prefix") == 1
    saved = next(iter(app.requests.snapshot()["requests"].values()))
    assert saved["terminal"] == "execution_deployment_fenced"
    assert saved["execution_state"] == "stopped"

