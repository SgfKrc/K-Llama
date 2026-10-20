"""Tiny native ports wired through production publication and SQLite requests."""
from contextlib import contextmanager
from copy import deepcopy
import sqlite3
from types import SimpleNamespace
from model_host import ModelHost
from qlh_core.bootstrap.execution import ExecutionApplication
from qlh_core.adapters.request_repository import SqliteRequestRepository
from qlh_core.execution.requests import RequestService
from tests.test_refactor_execution_integration import spec

def make_application(path, manager, offer, callbacks, *, boundaries=(0, 4, 8, 12)):
    def connect():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection
    @contextmanager
    def transaction():
        connection = connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback(); raise
        finally: connection.close()
    selection = {"operation_id": "supply_" + "a" * 32, "config_id": "deployment", "generation": 1}
    stages = []
    for index in range(1, len(boundaries) - 1):
        name = "tail" if index == len(boundaries) - 2 else "middle"
        identity = {**selection, "assignment_id": name, "artifact_sha256": "a" * 64, "hidden_size": 4,
            "layer_range": list(boundaries[index:index+2]), "stage_model_identity": {"model_id": name,
            "engine": "llama_cpp", "format": "gguf", "revision": "1", "sha256": "b" * 64}}
        stages.append({"node_id": name, "start_layer": boundaries[index], "end_layer": boundaries[index+1],
            "load_spec": spec("lm_head", "final_norm") if name == "tail" else spec(), "worker_prepare_identity": identity})
    local = {"node_id": "master", "start_layer": 0, "end_layer": boundaries[1], "load_spec": spec("embedding"),
             "identity": {"assignment_id": "master", "layer_range": list(boundaries[:2])}}
    state = {"revision": 0, "boot_id": "boot", "commit_scope": "local_sqlite", "selected_operation": "",
        "operations": {}, "supplies": {}, "active_deployment": "deployment", "lifecycle": {"total_layers": boundaries[-1]},
        "deployments": {"deployment": {"published": True, "status": "active", "boot_id": "boot", "authority": {"term": 1},
            "connections": {s["node_id"]: 1 for s in stages}, "local_assignment": local,
            "assignments": {s["node_id"]: s for s in stages}}}}
    from llama_engine import LlamaCppEngine
    if isinstance(manager, LlamaCppEngine):
        manager._pipeline_descriptor = {
            **(manager._pipeline_descriptor or {}),
            "assignment_layer_range": list(boundaries[:2]),
        }
    else:
        manager.layer_range = tuple(boundaries[:2])
    host = ModelHost(manager=manager)
    scheduler = SimpleNamespace(_host=host, _model_control=SimpleNamespace(boot_id="boot", snapshot=lambda: deepcopy(state),
        runnable=lambda *a, **kw: state["deployments"]["deployment"]["status"] == "active"),
        _model_control_authority=lambda: {"term": 1}, _model_control_connections=lambda: {s["node_id"]: 1 for s in stages},
        _task_worker_control=SimpleNamespace(worker_snapshot=lambda n: {"capabilities": {"stage_kv_cleanup": True, "node_incarnation": n + "-boot"}}),
        get_effective_node_id=lambda: "master", get_distributed_inference_enabled=lambda: True,
        _execute_layer_stage_offer=offer, _require_callbacks=lambda: callbacks)
    requests = RequestService(SqliteRequestRepository(connect=connect, transaction=transaction), boot_id="boot")
    app = ExecutionApplication(scheduler, requests)
    scheduler._execution_app = app
    return app, stages, state
