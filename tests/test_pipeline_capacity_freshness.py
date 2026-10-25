"""Cached profiles cannot overrule fresh operation capacity or admit a real shortage."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from model_supply import ModelSupplyError
from scheduler import Scheduler
from tests.test_model_supply import coordinator, descriptor, node
from tests.test_pipeline_retirement import retirement_case  # noqa: F401


@pytest.fixture
def fresh_case(tmp_path, monkeypatch):
    c, op, _ = coordinator(tmp_path)
    c._operations[op].update(status="available", plan={"assignments": [
        {"node_id": "master", "engine": "llama_cpp", "start_layer": 0, "end_layer": 6},
        {"node_id": "worker", "engine": "llama_cpp", "start_layer": 6, "end_layer": 8},
    ]}, nodes={"master": {"artifact_sha256": "c" * 64}, "worker": {"artifact_sha256": "d" * 64}})
    c._operations[op].update(source_model_sha256="a" * 64, tokenizer_sha256="b" * 64)
    from task_worker_adapter import TaskWorkerControlPlane
    hello = TaskWorkerControlPlane().begin_worker_hello(node_id="worker", capabilities={
        "stage_types": ["layer_forward"], "engines": ["llama_cpp"], "max_concurrency": 1,
        "models": [{"model_id": "segment", "engine": "llama_cpp", "format": "gguf", "revision": "1", "sha256": "d" * 64}],
        "layer_ranges": [[6, 8]], "layer_artifacts": [{"model_id": "segment", "artifact_sha256": "d" * 64,
            "source_model_sha256": "a" * 64, "tokenizer_sha256": "b" * 64, "source_model_id": "fixture",
            "hidden_size": 16, "layer_range": [6, 8], "segment_mode": "tail", "layer_weight_bytes": [100, 100]}]})
    ack = c.scheduler._task_worker_control.receive_on_coordinator("worker", hello.snapshot(), coordinator_node_id="master")
    assert ack.payload["accepted"]
    monkeypatch.setattr(c, "prepared_source", lambda _op: {})
    ram = {"master": 10000, "worker": 1500}
    def inventory(node_id):
        return {"node_id": node_id, "distributed": True, "engines": ["llama_cpp"],
                "memory_available_bytes": ram[node_id], "sources": []}
    c.scheduler._model_supply_worker.inventory = lambda *, model_id: inventory("master")
    def send(node_id, payload, _message_type):
        assert payload["model_id"] == ""  # Resource-only; no multi-GB source rehash.
        assert payload["operation_id"] != op
        c.handle(node_id, {**inventory(node_id), "action": "inventory_result",
                           "operation_id": payload["operation_id"]})
        return True
    c.scheduler._tcp_server.send_to_client = send
    cached = [node("master", 10000), {**node("worker", 500), "layer_ranges": [[6, 8]]}]
    caps = c.scheduler._task_worker_control.worker_snapshot("worker")["capabilities"]
    cached[1].update(models=deepcopy(caps["models"]), layer_artifacts=deepcopy(caps["layer_artifacts"]))
    for record in cached:
        record.update(reserve_bytes=100, runtime_multiplier=2.0)
    model = descriptor(layer_bytes=100, components=0)
    model["component_weight_bytes"]["lm_head"] = 400
    master = Scheduler(host=SimpleNamespace(engine_type="llama_cpp"))
    master._role_override = "master"
    monkeypatch.setattr(master, "_get_pipeline_capacity_nodes", lambda _ids: deepcopy(cached))
    monkeypatch.setattr(master, "_pipeline_node_metadata", lambda: {
        "master": {"engine": "llama_cpp"}, "worker": {"engine": "llama_cpp"}})
    monkeypatch.setattr(master, "_relay_claimed_layer_prefix", lambda total: total)
    monkeypatch.setattr(master, "_relay_layer_claim_ranges", lambda _total: {})
    import config
    monkeypatch.setattr(config, "PIPELINE_CAPACITY_SAFETY_MARGIN", 1.0)
    return SimpleNamespace(c=c, op=op, ram=ram, cached=cached, model=model, master=master)


def plan(case, evidence=None):
    return case.master.get_pipeline_capacity_plan({"master", "worker"}, descriptor=case.model,
        require_distributed=True, capacity_evidence=evidence)


def test_real_scheduler_uses_fresh_operation_budget_without_mutating_profiles(fresh_case):
    case = fresh_case
    before = deepcopy(case.cached)
    stale = plan(case)
    assert not stale["admitted"]
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        result = plan(case, evidence)
    assert result["admitted"], result
    assert [(a["node_id"], a["start_layer"], a["end_layer"]) for a in result["assignments"]] == [
        ("master", 0, 6), ("worker", 6, 8)]
    worker = next(a for a in result["assignments"] if a["node_id"] == "worker")
    assert worker["required_bytes"] == 1300 and worker["capacity_bytes"] == 1500
    assert all(n["engine"] == "llama_cpp" and not n["cross_engine"]
               for n in result["pipeline_layout"]["nodes"])
    assert {e["node_id"]: e["capacity_bytes"] for e in result["capacity_evidence"]} == case.ram
    assert case.cached == before
    assert not case.c._capacity_queries


def test_fresh_shortage_remains_rejected_even_when_cached_profile_has_room(fresh_case):
    case = fresh_case
    case.cached[1]["capacity_bytes"] = 20000
    case.ram["worker"] = 1200
    assert plan(case)["admitted"], plan(case)
    with case.c.prepare_guard(case.op):
        result = plan(case, case.c.prepare_capacity_evidence(case.op))
    assert not result["admitted"]
    assert result["reason_code"] == "pipeline_capacity_single_node_insufficient"
    assert next(e for e in result["capacity_evidence"] if e["node_id"] == "worker")["capacity_bytes"] == 1200


def test_evidence_is_checked_at_solve_after_metadata_preparation(fresh_case, monkeypatch):
    import model_supply_capacity
    case = fresh_case
    clock = [100.0]
    monkeypatch.setattr(model_supply_capacity.time, "monotonic", lambda: clock[0])
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        clock[0] += model_supply_capacity.MAX_BUDGET_AGE_SECONDS + 1
        with pytest.raises(ModelSupplyError) as error:
            plan(case, evidence)
    assert error.value.code == "SUPPLY_RUNTIME_BUDGET_EXPIRED"


def test_prepare_budget_failure_aborts_candidate_and_preserves_error_code(retirement_case, monkeypatch):
    import api_server
    from api.routes_models import _prepare_pipeline_model_resolution
    from api_errors import CodedHTTPException
    case = retirement_case
    def reject(**_kwargs):
        raise ModelSupplyError("SUPPLY_RUNTIME_BUDGET_EXPIRED", "Runtime budget expired before model admission")
    monkeypatch.setattr(case.master, "start_pipeline_model_transaction", reject)
    resolution = SimpleNamespace(model_id="retirement-model", model_path=str(case.model_path),
                                 quant_type="fp16", engine="pytorch")
    with pytest.raises(CodedHTTPException) as error:
        asyncio.run(_prepare_pipeline_model_resolution(resolution,
                    operation_id="supply_probe", capacity_evidence=object()))
    assert error.value.error_code == "SUPPLY_RUNTIME_BUDGET_EXPIRED"
    assert case.host.peek_manager() is case.active
    assert case.host.pipeline_candidate_status()["phase"] == "aborted"


def test_direct_supplied_prepare_preserves_fresh_probe_error_before_candidate(retirement_case, monkeypatch):
    import model_supply_worker
    from api.routes_models import _prepare_pipeline_model_resolution
    from api_errors import CodedHTTPException
    from contextlib import contextmanager
    case = retirement_case
    op = "supply_direct_probe"
    entered = []
    @contextmanager
    def guard(operation_id):
        assert operation_id == op
        entered.append("enter")
        try:
            yield {}
        finally:
            entered.append("exit")
    def reject(operation_id):
        assert entered == ["enter"] and operation_id == op
        raise ModelSupplyError("SUPPLY_RUNTIME_CONNECTION_CHANGED", "Worker registration changed")
    monkeypatch.setattr(model_supply_worker, "get_selected_model", lambda _root: {"manifest": {"operation_id": op}})
    monkeypatch.setattr(case.master._model_supply_worker, "selected_model_path", lambda: case.model_path)
    monkeypatch.setattr(case.master._model_supply, "prepare_guard", guard)
    monkeypatch.setattr(case.master._model_supply, "prepare_capacity_evidence", reject)
    resolution = SimpleNamespace(model_id="retirement-model", model_path=str(case.model_path),
                                 quant_type="fp16", engine="pytorch")
    with pytest.raises(CodedHTTPException) as error:
        asyncio.run(_prepare_pipeline_model_resolution(resolution))
    assert error.value.status_code == 409
    assert error.value.error_code == "SUPPLY_RUNTIME_CONNECTION_CHANGED"
    assert entered == ["enter", "exit"]
    assert case.host.peek_manager() is case.active
    assert not case.managers  # No candidate was constructed before the failed probe.


def test_transfer_progress_renews_admission_window_but_stall_still_expires(fresh_case, monkeypatch):
    """#116：漫长的工件传输本身不是陈旧风险，停滞才是。

    工件传输（实测 5.4 GB、数十分钟）发生在证据准备之后、准入之前。窗口必须跟随
    「最后一次实际进展」，否则任何大工件的供给都必然以 SUPPLY_RUNTIME_BUDGET_EXPIRED
    失败；而一旦传输停滞超过窗口，原判据必须原样生效。
    """
    import model_supply_capacity
    case = fresh_case
    clock = [100.0]
    monkeypatch.setattr(model_supply_capacity.time, "monotonic", lambda: clock[0])
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        # 40 分钟传输，每 30 秒都有真实进展 ⇒ 准入窗口始终有效。
        for _ in range(80):
            clock[0] += 30.0
            case.c.touch_capacity_evidence(case.op)
        assert plan(case, evidence)["admitted"]
        # 传输停滞超过窗口 ⇒ 仍旧 fail-closed 过期。
        clock[0] += model_supply_capacity.MAX_BUDGET_AGE_SECONDS + 1
        with pytest.raises(ModelSupplyError) as error:
            plan(case, evidence)
    assert error.value.code == "SUPPLY_RUNTIME_BUDGET_EXPIRED"
