"""Selection lifecycle tests exercise its authoritative supply/native barriers."""
import asyncio
from copy import deepcopy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api_errors import CodedHTTPException, ERROR_CODE_HEADER
from model_intent import ModelIntentCoordinator
from model_supply import ModelSupplyCoordinator, ModelSupplyError
from worker_assignment_state import WorkerAssignmentRegistry, PHASE_READY
from worker_prepare_coordinator import WorkerPrepareCoordinatorMixin
from tests.test_pipeline_retirement import retirement_case  # noqa: F401
from tests.test_worker_prepare_coordinator import stage_case  # noqa: F401


class Supply:
    def __init__(self, root):
        self.root = root
        self.operations = {}
        self.cancellations = []
    def _check_master(self):
        if self.scheduler._effective_role() != "master":
            raise ModelSupplyError("SUPPLY_MASTER_REQUIRED", "master required")
        if not self.scheduler.get_distributed_inference_enabled():
            raise ModelSupplyError("SUPPLY_LOCAL_MODE", "local mode")
    def create(self, intent, *, wait_for_nodes=False):
        self._check_master()
        assert wait_for_nodes
        op_id = "supply_" + format(len(self.operations) + 1, "032x")
        value = {"operation_id": op_id, "intent": intent, "status": "waiting_nodes", "reason_code": "",
                 "nodes": {"master": {"status": "inventory"}, "tail": {"status": "inventory"}},
                 "master_model_path": "/private/master/segment.gguf", "plan": None}
        self.operations[op_id] = value
        return deepcopy(value)
    def get(self, op_id):
        if op_id not in self.operations:
            raise ModelSupplyError("SUPPLY_OPERATION_NOT_FOUND", "missing")
        return deepcopy(self.operations[op_id])
    def ensure_prepare_not_cancelled(self, op_id):
        if op_id in self.cancellations:
            raise ModelSupplyError("SUPPLY_PREPARE_CANCELLED", "cancelled")
    def cancel(self, op_id):
        self.cancellations.append(op_id)
        if self.operations[op_id]["status"] != "available":
            self.operations[op_id]["status"] = "cancelled"


class IntentOwner:
    """Selection repository test double; native barriers have separate real-owner tests."""
    boot_id = "intent-test"
    def __init__(self):
        self.state = {"boot_id":self.boot_id,"revision":0,"operations":{},"supplies":{},
            "selected_operation":"", "commit_scope":"test"}
    def runnable(self, *args, **kwargs): return True
    def start(self): return self.snapshot()
    def snapshot(self): return deepcopy(self.state)
    def save_operation(self, record, *, select=False, **options):
        self.state["revision"] += 1
        self.state["operations"][record["operation_id"]] = deepcopy(record)
        if select: self.state["selected_operation"] = record["operation_id"]
    def fence_operation_failure(self, record): self.save_operation(record)


def setup(tmp_path, prepare, *, retire=None, retire_timeout=300):
    supply = Supply(tmp_path)
    scheduler = SimpleNamespace(_model_supply=supply, _model_control=IntentOwner(), _model_control_authority=lambda: {}, _model_control_connections=lambda: {},
                                _model_control_input_revisions=lambda transaction: {}, _layer_config_lock=threading.RLock(),
                                _pipeline_load_transaction=None, _active_pipeline_capacity_plan=None,
                                _pipeline_model_cleanup_ledger={}, _worker_assignments=WorkerAssignmentRegistry(),
                                _effective_role=lambda: "master", get_distributed_inference_enabled=lambda: True,
                                _assignment_connection_generation=lambda node: 11,
                                _connected_client_ids=lambda: {"tail", "worker"})
    # Explicit owner evidence, using the production native readiness predicate.
    scheduler._worker_prepare_assignment_loaded = WorkerPrepareCoordinatorMixin._worker_prepare_assignment_loaded.__get__(scheduler)
    supply.scheduler = scheduler
    return ModelIntentCoordinator(scheduler, prepare, retire=retire, retire_timeout=retire_timeout,
                                  poll_interval=0.001), scheduler, supply


def transaction(op_id, phase="preparing"):
    from tests.test_worker_prepare_coordinator import identity
    value = identity("tail")
    value.update(operation_id=op_id, config_id="config", generation=42)
    return {"candidate_id": "candidate", "config_id": "config", "generation": 42, "phase": phase,
            "worker_ids": {"tail"}, "expected": {"tail": {"node_id": "tail", "config_id": "config",
                "generation": 42, "assignment_id": value["assignment_id"], "worker_prepare_identity": value}}, "acks": {}}


def publish_ready(scheduler):
    tx = scheduler._pipeline_load_transaction
    tx["phase"] = "ready"
    scheduler._active_pipeline_capacity_plan = {"config_id": tx["config_id"], "generation": tx["generation"],
        "transaction_phase": "ready", "assignments": [deepcopy(item) for item in tx["expected"].values()]}
    for node, assignment in tx["expected"].items():
        receipt = {**deepcopy(assignment), "phase": "commit", "loaded": True, "status": "ready"}
        scheduler._worker_assignments.begin(node, config_id=tx["config_id"],
            connection_generation=scheduler._assignment_connection_generation(node), expected_config=assignment)
        assert scheduler._worker_assignments.record_ack(node, receipt)
        assert scheduler._worker_assignments.transition(node, phase=PHASE_READY)
        tx["acks"][node] = receipt


async def finish_tasks(coordinator):
    tasks = list(coordinator._tasks.values())
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_one_selection_automatically_supplies_then_prepares_but_files_are_not_ready(tmp_path):
    async def scenario():
        prepared = []
        async def prepare(op_id):
            prepared.append(op_id)
            scheduler._pipeline_load_transaction = transaction(op_id)
            return {"transaction_id": "candidate"}
        c, scheduler, supply = setup(tmp_path, prepare)
        op = await c.create({"model_id": "fixture", "timeout_seconds": 30})
        op_id = op["operation_id"]
        await asyncio.sleep(0.005)
        assert prepared == [] and c.current()["status"] == "waiting_nodes"
        supply.operations[op_id]["status"] = "available"
        await asyncio.sleep(0.01)
        assert prepared == [op_id]
        assert c.get(op_id)["status"] == "preparing"
        assert c.get(op_id)["loaded_ready"] is False
        scheduler._pipeline_load_transaction["phase"] = "ready"
        # Native transaction alone cannot publish readiness without its active plan.
        assert not c.get(op_id)["loaded_ready"]
        publish_ready(scheduler)
        await asyncio.sleep(0.01)
        ready = c.get(op_id)
        assert ready["status"] == "ready" and ready["loaded_ready"]
        assert ready["config_id"] == "config" and ready["generation"] == 42
        assert "master_model_path" not in ready
        assert not c._tasks
        scheduler._pipeline_load_transaction = None
        assert c.get(op_id)["status"] == "ready" and not c.get(op_id)["loaded_ready"]
    asyncio.run(scenario())


def test_same_pending_selection_is_idempotent_and_other_selection_cannot_enter_prepare_gap(tmp_path):
    async def scenario():
        async def prepare(op_id):
            pytest.fail("files are not ready")
        c, _, supply = setup(tmp_path, prepare)
        intent = {"model_id": "fixture"}
        first = await c.create(intent)
        assert (await c.create(intent))["operation_id"] == first["operation_id"]
        supply.operations[first["operation_id"]]["status"] = "available"
        replacement = await c.create({"model_id": "other"})
        assert replacement["operation_id"] != first["operation_id"]
        assert c.get(first["operation_id"])["status"] == "superseded"
        assert first["operation_id"] in supply.cancellations
        c.cancel(replacement["operation_id"])
        await finish_tasks(c)
    asyncio.run(scenario())


def test_cancel_before_prepare_prevents_late_available_from_loading(tmp_path):
    async def scenario():
        calls = []
        async def prepare(op_id):
            calls.append(op_id)
        c, _, supply = setup(tmp_path, prepare)
        op_id = (await c.create({"model_id": "fixture"}))["operation_id"]
        c.cancel(op_id)
        supply.operations[op_id]["status"] = "available"
        await asyncio.sleep(0.01)
        assert not calls
        assert c.get(op_id)["status"] == "cancelled"
        assert supply.cancellations == [op_id]
    asyncio.run(scenario())


def test_cancel_during_prepare_keeps_terminal_intent_after_late_callback(tmp_path):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        async def prepare(op_id):
            entered.set()
            await release.wait()
            return {"transaction_id": "late"}
        c, _, supply = setup(tmp_path, prepare)
        op_id = (await c.create({"model_id": "fixture"}))["operation_id"]
        supply.operations[op_id]["status"] = "available"
        await entered.wait()
        task = c._tasks[op_id]
        assert c.cancel(op_id)["status"] == "cancelled"
        release.set()
        await asyncio.wait_for(task, timeout=1)
        assert c.get(op_id)["status"] == "cancelled"
        assert not c._tasks
    asyncio.run(scenario())


def test_native_failure_is_named_and_node_specific(tmp_path):
    async def scenario():
        async def prepare(op_id):
            scheduler._pipeline_load_transaction = transaction(op_id, "aborted")
            scheduler._pipeline_load_transaction.update(reason_code="pipeline_worker_prepare_failed", reason="tail native failed")
            scheduler._pipeline_load_transaction["acks"] = {"tail": {"status": "error", "error_code": "GGUF_DIGEST_MISMATCH", "error": "digest mismatch"}}
            return {"transaction_id": "candidate"}
        c, scheduler, supply = setup(tmp_path, prepare)
        op_id = (await c.create({"model_id": "fixture"}))["operation_id"]
        supply.operations[op_id]["status"] = "available"
        await asyncio.sleep(0.01)
        result = c.get(op_id)
        assert result["status"] == "failed" and not result["loaded_ready"]
        assert result["reason_code"] == "pipeline_worker_prepare_failed"
        # The first projection has per-node diagnostics, and final projection must retain them.
        assert result["nodes"]["tail"]["reason_code"] == "GGUF_DIGEST_MISMATCH"
    asyncio.run(scenario())


def test_master_restart_does_not_replay_prepare_or_claim_old_ready(tmp_path):
    async def scenario():
        async def prepare(op_id):
            pytest.fail("restart must not replay")
        c, scheduler, supply = setup(tmp_path, prepare)
        op_id = (await c.create({"model_id": "fixture"}))["operation_id"]
        await finish_tasks(c)
        # Recovery is an explicit owner command; GET only projects its terminal.
        scheduler._model_control.state["operations"][op_id].update(status="failed", reason_code="INTENT_MASTER_RESTARTED")
        recovered = ModelIntentCoordinator(scheduler, prepare)
        assert recovered.current()["reason_code"] == "INTENT_MASTER_RESTARTED"
        assert recovered._tasks == {}
    asyncio.run(scenario())


def test_progress_projects_verified_bytes_and_verification_stage(tmp_path):
    async def scenario():
        c, _, supply = setup(tmp_path, lambda op_id: None)
        op_id = (await c.create({"model_id": "fixture"}))["operation_id"]
        supply.operations[op_id]["nodes"]["tail"].update(status="supplying", phase="verify", downloaded_bytes=7, total_bytes=10)
        result = c.get(op_id)
        assert result["nodes"]["tail"]["status"] == "verifying"
        assert result["nodes"]["tail"]["bytes_received"] == 7
        c.cancel(op_id)
        await finish_tasks(c)
    asyncio.run(scenario())

@pytest.fixture
def native_intent(stage_case, monkeypatch):
    """Real supply/prepare/transaction owners; transport and model I/O stay local."""
    from api.routes_model_supply import prepare_supply
    case = stage_case
    supply = case.master._model_supply
    snapshot_worker = case.master._task_worker_control.worker_snapshot
    def worker_snapshot(node):
        value = snapshot_worker(node) or {}
        return {**value, "capabilities": {**value.get("capabilities", {}), "node_incarnation":node+"-boot", "stage_kv_cleanup":True, "load_spec_version":1}}
    monkeypatch.setattr(case.master._task_worker_control, "worker_snapshot", worker_snapshot)
    # stage_case substitutes receipt lookup for its native-only tests. Restore
    # the actual lookup, cancellation events, and selected-asset prepare guard.
    monkeypatch.setattr(supply, "get", lambda op: ModelSupplyCoordinator.get(supply, op))
    monkeypatch.setattr(supply, "_run", lambda op: None)  # No actual supply/network work.
    # Native lifecycle tests replace capacity planning below; its fresh network
    # inventory shares that boundary. Keep a snapshot marker through admission.
    capacity_evidence = SimpleNamespace()
    monkeypatch.setattr(supply, "prepare_capacity_evidence", lambda _op: capacity_evidence)
    monkeypatch.setattr(supply.scheduler._model_supply_worker, "selected_model_path", lambda: case.model_path)
    case.master._tcp_server.get_client_info = lambda node: {"connected_at": 11} if node in case.connected else None
    from tests.test_pipeline_retirement import _Manager
    original = _Manager.prepare_pipeline_model
    def descriptor(manager, **kwargs):
        kwargs["model_sha256"] = "a" * 64
        return original(manager, **kwargs)
    monkeypatch.setattr(_Manager, "prepare_pipeline_model", descriptor)
    c = ModelIntentCoordinator(case.master, lambda op: prepare_supply(op, None), poll_interval=0.001)
    case.intent = {"model_id": "retirement-model", "engine": "pytorch", "quant_type": "fp16", "timeout_seconds": 30}
    case.coordinator = c
    case.supply = supply
    def available(op):
        supply._update(op, status="available", master_model_path=str(case.model_path),
                       plan={"assignments": [{"engine": "pytorch"}]},
                       nodes={node: {"status": "available", "artifact_sha256": "b" * 64,
                                     "manifest_sha256": "c" * 64, "layer_range": span}
                              for node, span in (("master", [0, 4]), ("worker", [4, 6]), ("android", [6, 8]))})
    case.available = available
    return case


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), timeout=3)


@pytest.mark.parametrize("error,expected_code,expected_reason", [
    pytest.param(CodedHTTPException(409, "MODEL_TXN_CONFLICT", {
        "code": "LEGACY_IGNORED", "message": "Another model transaction is active",
        "model_path": "/private/master/segment.gguf",
    }), "MODEL_TXN_CONFLICT", "Another model transaction is active", id="coded-attribute-precedes-detail"),
    pytest.param(HTTPException(409, {"code": "LEGACY_IGNORED", "message": "Worker capacity is stale"},
        headers={ERROR_CODE_HEADER: "pipeline_remote_capacity_stale"}),
        "pipeline_remote_capacity_stale", "Worker capacity is stale", id="header-precedes-detail"),
    pytest.param(HTTPException(409, {"code": "SUPPLY_OPERATION_STALE", "message": "Selected asset changed"}),
        "SUPPLY_OPERATION_STALE", "Selected asset changed", id="legacy-detail-code"),
    pytest.param(HTTPException(409, {"reason_code": "SUPPLY_RECEIPT_MISMATCH", "message": "Worker returned another artifact"}),
        "SUPPLY_RECEIPT_MISMATCH", "Worker returned another artifact", id="legacy-detail-reason-code"),
    pytest.param(ModelSupplyError("SUPPLY_PREPARE_CANCELLED", "Model preparation cancelled by user"),
        "SUPPLY_PREPARE_CANCELLED", "Model preparation cancelled by user", id="supply-error"),
    pytest.param(CodedHTTPException(400, "MODEL_TXN_PREPARE_FAILED",
        "Native prepare failed at C:/private/master/segment.gguf; token=private-token"),
        "MODEL_TXN_PREPARE_FAILED", "Model preparation failed; inspect master diagnostics", id="coded-string-private"),
    pytest.param(HTTPException(409, "Worker capacity is stale",
        headers={ERROR_CODE_HEADER: "pipeline_remote_capacity_stale"}),
        "pipeline_remote_capacity_stale", "Model preparation failed; inspect master diagnostics", id="header-string"),
    pytest.param(HTTPException(500, "Cannot open /private/master/segment.gguf; token=private-token"),
        "INTENT_PREPARE_FAILED", "Model preparation failed; inspect master diagnostics", id="uncoded-string-private"),
    pytest.param(RuntimeError("Cannot open /private/master/segment.gguf; token=private-token"),
        "INTENT_PREPARE_FAILED", "Model preparation failed; inspect master diagnostics", id="unexpected-error-private"),
    pytest.param(HTTPException(409, {"code": "", "reason_code": "SUPPLY_OPERATION_STALE", "message": "Selected asset changed"}),
        "SUPPLY_OPERATION_STALE", "Selected asset changed", id="empty-code-uses-reason-code"),
    pytest.param(HTTPException(500, {"code": "", "model_path": "/private/master/segment.gguf"}),
        "INTENT_PREPARE_FAILED", "Model preparation failed", id="empty-code-fallback"),
])
def test_prepare_exception_preserves_code_and_safe_terminal_reason(native_intent, error, expected_code, expected_reason):
    async def scenario():
        case = native_intent
        c, supply = case.coordinator, case.supply
        calls = []
        async def prepare(operation_id):
            calls.append(operation_id)
            raise error
        c.prepare = prepare
        try:
            op = (await c.create(case.intent))["operation_id"]
            task = c._tasks[op]
            case.available(op)
            await asyncio.wait_for(task, timeout=3)
            assert calls == [op] and not c._tasks
            assert supply._events[op].is_set() and not supply._preparing
            with pytest.raises(ModelSupplyError) as cancelled:
                supply.ensure_prepare_not_cancelled(op)
            assert cancelled.value.code == "SUPPLY_PREPARE_CANCELLED"
            persisted = c.scheduler._model_control.snapshot()["operations"][op]
            recovered = ModelIntentCoordinator(case.master, prepare)
            for result in (c.get(op), c.current(), persisted, recovered.current()):
                assert result["status"] == "failed"
                assert result["reason_code"] == expected_code
                assert result["reason"] == expected_reason
                public = json.dumps(result)
                assert "private/master" not in public and "private-token" not in public
                assert "master_model_path" not in public
            assert c.get(op)["loaded_ready"] is False
            assert calls == [op] and not recovered._tasks
        finally:
            await c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("failure,expected_code", [
    ("capacity_rejected", "pipeline_remote_capacity_stale"),
    ("native_prepare", "MODEL_TXN_PREPARE_FAILED"),
])
def test_real_prepare_route_failure_keeps_code_without_private_details(native_intent, monkeypatch, failure, expected_code):
    async def scenario():
        case = native_intent
        c, supply, master = case.coordinator, case.supply, case.master
        if failure == "capacity_rejected":
            monkeypatch.setattr(master, "get_pipeline_capacity_plan", lambda *_args, **_kwargs: {
                "admitted": False, "status": "rejected", "assignments": [],
                "reason_code": expected_code, "reason": "Worker capacity is stale; retry the selection",
            })
        else:
            from tests.test_pipeline_retirement import _Manager
            def broken_prepare(_manager, **_kwargs):
                raise OSError("Cannot open C:/private/master/segment.gguf; token=private-token")
            monkeypatch.setattr(_Manager, "prepare_pipeline_model", broken_prepare)
        try:
            op = (await c.create(case.intent))["operation_id"]
            task = c._tasks[op]
            case.available(op)
            await asyncio.wait_for(task, timeout=3)
            assert case.managers  # The real route entered the candidate owner.
            if failure == "capacity_rejected":
                assert master._pipeline_load_transaction["phase"] == "rejected"
                assert case.host.pipeline_candidate_status()["phase"] == "aborted"
            assert master._active_pipeline_capacity_plan is None
            assert supply._events[op].is_set() and not supply._preparing and not c._tasks
            persisted = c.scheduler._model_control.snapshot()["operations"][op]
            for result in (c.get(op), c.current(), persisted):
                assert result["status"] == "failed"
                assert result["reason_code"] == expected_code
                assert result["reason"] == "Model preparation failed; inspect master diagnostics"
                public = json.dumps(result)
                assert "private/master" not in public and "private-token" not in public
            assert c.get(op)["loaded_ready"] is False
        finally:
            await c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("failed_file", ["intent.json", "current-intent.json"])
def test_create_disk_failure_revokes_supply_and_allows_explicit_retry(native_intent, monkeypatch, failed_file):
    async def scenario():
        case = native_intent
        c, supply = case.coordinator, case.supply
        owner = case.master._model_control
        original = owner.save_operation
        def broken_save(*args, **kwargs):
            raise OSError("injected atomic selection commit failure")
        monkeypatch.setattr(owner, "save_operation", broken_save)
        with pytest.raises(ModelSupplyError) as error:
            await c.create(case.intent)
        assert error.value.code == "INTENT_PERSIST_FAILED"
        old = supply._active
        assert supply._events[old].is_set() and not c._tasks
        assert c.current()["status"] == "failed"
        assert c.current()["reason_code"] == "INTENT_PERSIST_FAILED"
        monkeypatch.setattr(owner, "save_operation", original)
        fresh = await c.create(case.intent)
        assert fresh["operation_id"] != old and fresh["status"] == "queued"
        c.cancel(fresh["operation_id"])
        await c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["intent_disk", "supply_disk", "native_abort"])
def test_cancel_failure_still_revokes_real_owners_and_is_retryable(native_intent, monkeypatch, failure):
    async def scenario():
        case = native_intent
        c, supply, master = case.coordinator, case.supply, case.master
        op = (await c.create(case.intent))["operation_id"]
        case.available(op)
        await wait_until(lambda: master._pipeline_load_transaction is not None)
        task = c._tasks[op]
        transaction = master._pipeline_load_transaction
        persist = c._persist
        supply_persist = supply._persist
        abort = master._abort_pipeline_load_transaction
        def disk_error(*args, **kwargs):
            raise OSError("injected disk failure")
        if failure == "intent_disk":
            monkeypatch.setattr(c, "_persist", disk_error)
            assert c.cancel(op)["status"] == "cancelled"
        else:
            if failure == "supply_disk":
                # A live supply receipt save may fail before the native owner
                # entry. Intent must still call that existing native entry.
                supply._operations[op]["status"] = "supplying"
                monkeypatch.setattr(supply, "_persist", disk_error)
            else:
                monkeypatch.setattr(master, "_abort_pipeline_load_transaction", disk_error)
            with pytest.raises(ModelSupplyError):
                c.cancel(op)
        assert supply._events[op].is_set()
        monkeypatch.setattr(c, "_persist", persist)
        monkeypatch.setattr(supply, "_persist", supply_persist)
        monkeypatch.setattr(master, "_abort_pipeline_load_transaction", abort)
        assert c.cancel(op)["status"] == "cancelled"
        assert transaction["phase"] == "aborted"
        assert case.host.pipeline_candidate_status()["phase"] == "aborted"
        assert master._active_pipeline_capacity_plan is None
        await task
        assert not c._tasks
    asyncio.run(scenario())


@pytest.mark.parametrize("persistent", [False, True])
def test_candidate_disk_failure_aborts_installed_native_transaction(native_intent, monkeypatch, persistent):
    async def scenario():
        case = native_intent
        c, supply = case.coordinator, case.supply
        op = (await c.create(case.intent))["operation_id"]
        task = c._tasks[op]
        original = c._persist
        failed = False
        def broken(record):
            nonlocal failed
            if record.get("candidate_id") and (persistent or not failed):
                failed = True
                raise OSError("injected candidate save failure")
            original(record)
        monkeypatch.setattr(c, "_persist", broken)
        case.available(op)
        await task
        assert c.get(op)["status"] == "failed"
        assert c.get(op)["reason_code"] == "INTENT_PERSIST_FAILED"
        assert supply._events[op].is_set() and not c._tasks
        assert case.master._pipeline_load_transaction["phase"] == "aborted"
        assert case.host.pipeline_candidate_status()["phase"] == "aborted"
        assert any(record["action"] == "abort" for record in case.master._pipeline_model_cleanup_ledger.values())
    asyncio.run(scenario())


def test_first_async_lookup_failure_cleans_task_and_has_named_terminal(native_intent, monkeypatch):
    async def scenario():
        case = native_intent
        c, supply = case.coordinator, case.supply
        op = (await c.create(case.intent))["operation_id"]
        task = c._tasks[op]
        original = supply.get
        first = True
        def get(operation_id):
            nonlocal first
            if first:
                first = False
                raise ModelSupplyError("SUPPLY_WORKER_DISCONNECTED", "Worker connection changed")
            return original(operation_id)
        monkeypatch.setattr(supply, "get", get)
        await task
        assert not c._tasks and supply._events[op].is_set()
        assert c.get(op)["status"] == "failed"
        assert c.get(op)["reason_code"] == "SUPPLY_WORKER_DISCONNECTED"
    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["deadline", "cancel", "mode", "shutdown"])
@pytest.mark.parametrize("boundary", ["candidate", "transaction"])
def test_stop_fences_late_real_threadpool_prepare(native_intent, monkeypatch, stop, boundary):
    async def scenario():
        import model_intent
        case = native_intent
        c, supply, master = case.coordinator, case.supply, case.master
        entered, release = threading.Event(), threading.Event()
        finished = threading.Event()
        host_type = type(case.host)
        original = host_type.prepare_pipeline_candidate if boundary == "candidate" else master.start_pipeline_model_transaction
        def block(*args, **kwargs):
            entered.set()
            assert release.wait(5), "test did not release native thread"
            try:
                return original(*args, **kwargs)
            finally:
                finished.set()
        if boundary == "candidate":
            monkeypatch.setattr(host_type, "prepare_pipeline_candidate", block)
        else:
            monkeypatch.setattr(master, "start_pipeline_model_transaction", block)
        now = [0.0]
        monkeypatch.setattr(model_intent, "time", SimpleNamespace(monotonic=lambda: now[0], time=__import__("time").time))
        op = (await c.create(case.intent))["operation_id"]
        task = c._tasks[op]
        case.available(op)
        close_task = None
        try:
            await wait_until(entered.is_set)
            if stop == "deadline":
                now[0] = 31.0
                await wait_until(lambda: c.get(op)["status"] == "failed")
                assert c.get(op)["reason_code"] == "INTENT_DEADLINE_EXCEEDED"
            elif stop == "cancel":
                assert c.cancel(op)["status"] == "cancelled"
            elif stop == "mode":
                # Actual opt-out owner, including cancel_active while available.
                master.set_distributed_inference_enabled(False)
                assert supply._events[op].is_set()
                master._distributed_inference_enabled = True
                await wait_until(lambda: c.get(op)["status"] == "failed")
                assert c.get(op)["reason_code"] == "SUPPLY_PREPARE_CANCELLED"
            else:
                close_task = asyncio.create_task(c.close())
                await wait_until(lambda: supply._events[op].is_set())
            assert supply._events[op].is_set()
            assert supply._preparing == op and not task.done()
            assert not finished.is_set()
            # A new model intent may not overlap the still-live native guard.
            with pytest.raises(ModelSupplyError) as error:
                await c.create(dict(case.intent, model_id="new-selection"))
            assert error.value.code == ("INTENT_MASTER_STOPPED" if stop == "shutdown" else "SUPPLY_PREPARE_BUSY")
        finally:
            release.set()
            await asyncio.wait_for(task if stop != "shutdown" else close_task, timeout=3)
        assert finished.is_set() and not c._tasks and not supply._preparing
        assert master._pipeline_load_transaction is None
        assert master._active_pipeline_capacity_plan is None
        assert case.host.pipeline_candidate_status()["phase"] == "aborted"
        # The same cancelled/expired operation cannot acquire a new guard even
        # after rejoin; disk assets may be reused only by a fresh selection.
        with pytest.raises(ModelSupplyError):
            with supply.prepare_guard(op):
                pytest.fail("old prepare permission was replayed")
    asyncio.run(scenario())


def test_named_supply_disconnect_is_terminal_until_explicit_ui_retry(native_intent):
    async def scenario():
        case = native_intent
        c, supply = case.coordinator, case.supply
        op = (await c.create(case.intent))["operation_id"]
        task = c._tasks[op]
        supply._update(op, status="failed", reason_code="SUPPLY_WORKER_DISCONNECTED", reason="Worker Surface connection changed")
        await task
        for _ in range(3):
            current = c.current()
            assert current["operation_id"] == op and current["status"] == "failed"
            assert current["reason_code"] == "SUPPLY_WORKER_DISCONNECTED"
            assert not current["loaded_ready"] and not c._tasks
        fresh = await c.create(case.intent)
        assert fresh["operation_id"] != op
        with pytest.raises(ModelSupplyError):
            supply.ensure_prepare_not_cancelled(op)
        c.cancel(fresh["operation_id"])
        await c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("phase,status,expected", [
    ("receiving", "supplying", "downloading"), ("downloading", "supplying", "downloading"),
    ("download", "supplying", "downloading"), ("verify", "supplying", "verifying"),
    ("verifying", "supplying", "verifying"), ("registering", "supplying", "verifying"),
    ("downloading", "available", "available"), ("verifying", "available", "available"),
    ("downloading", "failed", "failed"),
])
def test_node_phase_projection_preserves_available_and_failure(tmp_path, phase, status, expected):
    async def scenario():
        c, _, supply = setup(tmp_path, lambda op: None)
        op = (await c.create({"model_id": "fixture"}))["operation_id"]
        supply.operations[op]["nodes"]["tail"].update(phase=phase, status=status, downloaded_bytes=7, total_bytes=10)
        projected = c.get(op)["nodes"]["tail"]
        assert projected["status"] == expected
        assert projected["bytes_received"] == (10 if status == "available" else 7)
        c.cancel(op)
        await c.close()
    asyncio.run(scenario())



def test_ready_receipt_remains_authoritative_when_final_projection_save_fails(native_intent, monkeypatch):
    async def scenario():
        from tests.test_worker_prepare_coordinator import ack, prepare_both
        case = native_intent
        c = case.coordinator
        op = (await c.create(case.intent))["operation_id"]
        task = c._tasks[op]
        case.available(op)
        await wait_until(lambda: case.master._pipeline_load_transaction is not None)
        prepare_both(case)
        original = c._persist
        def broken(record):
            if record["status"] == "ready":
                raise OSError("injected final projection save failure")
            original(record)
        monkeypatch.setattr(c, "_persist", broken)
        ack(case.master, "worker", "commit")
        ack(case.master, "android", "commit")
        await task
        projected = c.get(op)
        assert projected["status"] == "ready" and projected["loaded_ready"], json.dumps(projected)
        assert not c._tasks
        assert case.master._pipeline_load_transaction["phase"] == "ready"
        assert case.host.pipeline_candidate_status()["phase"] == "finalized"
        with pytest.raises(ModelSupplyError) as error:
            c.cancel(op)
        assert error.value.code == "INTENT_ALREADY_READY"
    asyncio.run(scenario())


def test_shutdown_before_advance_starts_revokes_and_removes_queued_task(native_intent):
    async def scenario():
        case = native_intent
        c = case.coordinator
        op = (await c.create(case.intent))["operation_id"]
        await c.close()
        assert not c._tasks and case.supply._events[op].is_set()
        assert c.current()["status"] == "failed"
        assert c.current()["reason_code"] == "INTENT_MASTER_STOPPED"
    asyncio.run(scenario())


@pytest.mark.parametrize("retired", ["restart", "generation", "mode"])
def test_persisted_ready_never_grants_readiness_after_owner_retirement(tmp_path, retired):
    async def scenario():
        c, scheduler, supply = setup(tmp_path, lambda op: None)
        op = (await c.create({"model_id": "fixture"}))["operation_id"]
        await c.close()
        supply.operations[op]["status"] = "available"
        c._records[op].update(status="ready", candidate_id="candidate", config_id="config", generation=42)
        c._persist(c._records[op])
        scheduler._pipeline_load_transaction = transaction(op, "ready")
        publish_ready(scheduler)
        if retired == "restart":
            scheduler._pipeline_load_transaction = None
        elif retired == "generation":
            scheduler._active_pipeline_capacity_plan["generation"] = 43
        else:
            scheduler.get_distributed_inference_enabled = lambda: False
        recovered = ModelIntentCoordinator(scheduler, lambda op: pytest.fail("old prepare was replayed"))
        projected = recovered.current()
        assert projected["status"] == "ready" and not projected["loaded_ready"]
        assert projected["deployment_reason_code"] == "INTENT_PLAN_RETIRED" and not recovered._tasks
    asyncio.run(scenario())


@pytest.fixture
def ready_intent(tmp_path):
    """Published receipts use the real registry/native owner; no API or model load."""
    c, scheduler, supply = setup(tmp_path, lambda op: None)
    op = supply.create({"model_id": "old"}, wait_for_nodes=True)["operation_id"]
    supply.operations[op]["status"] = "available"
    c._current = op
    c._records[op] = {"operation_id": op, "status": "ready", "candidate_id": "candidate",
                      "config_id": "config", "generation": 42}
    scheduler._pipeline_load_transaction = transaction(op, "ready")
    # Two worker receipts make a partial-ready regression observable.
    assignment = deepcopy(scheduler._pipeline_load_transaction["expected"]["tail"])
    assignment.update(node_id="worker", assignment_id="assignment-worker")
    assignment["worker_prepare_identity"].update(node_id="worker", assignment_id="assignment-worker")
    scheduler._pipeline_load_transaction["expected"]["worker"] = assignment
    scheduler._pipeline_load_transaction["worker_ids"].add("worker")
    publish_ready(scheduler)
    c._persist(c._records[op])
    return SimpleNamespace(c=c, scheduler=scheduler, supply=supply, op=op)


@pytest.mark.parametrize("node", ["tail", "worker"])
@pytest.mark.parametrize("stage", ["register_pending", "register_confirmed", "new_hello"])
def test_ready_projection_rejects_same_node_reconnect_without_fresh_native_commit(ready_intent, node, stage):
    case = ready_intent
    assert case.c.current()["loaded_ready"]
    # A replacement does not invoke disconnect or retire the old transaction.
    case.scheduler._assignment_connection_generation = lambda peer: (0 if stage == "register_pending" else 12) if peer == node else 11
    if stage == "new_hello":
        old = case.scheduler._pipeline_load_transaction["expected"][node]
        case.scheduler._worker_assignments.begin(node, config_id="config", connection_generation=12, expected_config=old)
    for read in (lambda: case.c.get(case.op), case.c.current):
        result = read()
        assert result["status"] == "ready" and not result["loaded_ready"]
        assert result["deployment_reason_code"] == "worker_native_ready_identity_mismatch"
    assert case.scheduler._pipeline_load_transaction["phase"] == "ready"


@pytest.mark.parametrize("evidence", ["owner_missing", "owner_error", "missing_worker", "empty_expected",
                                     "registry_missing", "ack_missing", "prepare_ack", "not_loaded", "identity", "plan_identity"])
def test_ready_projection_fails_closed_without_every_native_owner_proof(ready_intent, evidence):
    case = ready_intent
    scheduler = case.scheduler
    tx = scheduler._pipeline_load_transaction
    assignment = tx["expected"]["worker"]
    if evidence == "owner_missing":
        del scheduler._worker_prepare_assignment_loaded
    elif evidence == "owner_error":
        def unavailable(*_):
            raise RuntimeError("owner unavailable")
        scheduler._worker_prepare_assignment_loaded = unavailable
    elif evidence == "missing_worker":
        tx["expected"].pop("worker")
    elif evidence == "empty_expected":
        tx["expected"].clear(); tx["worker_ids"].clear()
    elif evidence == "registry_missing":
        scheduler._worker_assignments.clear()
    elif evidence == "plan_identity":
        scheduler._active_pipeline_capacity_plan["assignments"][1]["worker_prepare_identity"]["artifact_sha256"] = "f" * 64
    else:
        receipt = deepcopy(tx["acks"]["worker"])
        if evidence == "ack_missing":
            receipt = {}
        elif evidence == "prepare_ack":
            receipt["phase"] = "prepare"
        elif evidence == "not_loaded":
            receipt["loaded"] = False
        elif evidence == "identity":
            receipt["worker_prepare_identity"]["artifact_sha256"] = "f" * 64
        scheduler._worker_assignments.begin("worker", config_id="config", connection_generation=11,
            expected_config=assignment, acknowledgement=receipt)
        scheduler._worker_assignments.transition("worker", phase=PHASE_READY)
    result = case.c.current()
    assert result["status"] == "ready" and not result["loaded_ready"]
    assert result["deployment_reason_code"]


def test_normal_ready_and_reconnect_with_all_new_native_receipts_remain_ready(ready_intent):
    case = ready_intent
    assert case.c.current()["status"] == "ready"
    case.scheduler._assignment_connection_generation = lambda peer: 12
    assert not case.c.current()["loaded_ready"]
    publish_ready(case.scheduler)
    assert case.c.current()["status"] == "ready" and case.c.current()["loaded_ready"]


def test_terminal_ready_is_superseded_instead_of_waiting_on_new_hello_transaction(ready_intent):
    case = ready_intent
    case.scheduler._pipeline_load_transaction["phase"] = "preparing"
    for _ in range(3):
        result = case.c.current()
        assert result["status"] == "ready" and result["deployment_reason_code"] == "INTENT_PLAN_RETIRED"
        assert not result["loaded_ready"] and not case.c._tasks


def retire_local(case):
    case.scheduler._active_pipeline_capacity_plan = None
    case.scheduler._pipeline_load_transaction = None
    case.scheduler._worker_assignments.clear()


@pytest.mark.parametrize("retirement", ["recovered", "during_selection"])
def test_two_node_selection_preserves_offline_native_release_debt(native_intent, retirement):
    from tests.test_worker_prepare_coordinator import ack, prepare_both

    async def scenario():
        case = native_intent
        c, master, supply = case.coordinator, case.master, case.supply
        try:
            old = (await c.create(case.intent))["operation_id"]
            task = c._tasks[old]
            case.available(old)
            await wait_until(lambda: master._pipeline_load_transaction is not None)
            prepare_both(case)
            ack(master, "worker", "commit")
            ack(master, "android", "commit")
            await task
            for record in list(master._pipeline_model_cleanup_ledger.values()):
                master._handle_worker_prepare_ack(record["node_id"], {
                    "action": "ack", "phase": "finalize", "status": "finalized", "loaded": True,
                    "identity": record["payload"]["worker_prepare_identity"], "cleanup_id": record["cleanup_id"],
                })
            assert not master._pipeline_model_cleanup_ledger
            # Only the new healthy peer remains; both old native workers are offline.
            case.connected.clear()
            case.connected.add("rog")
            debts = {}
            retire_calls = []

            async def retire():
                retire_calls.append(old)
                if master._pipeline_load_transaction is None:
                    return
                assert case.unload()["unloaded"] is True
                debts.update(deepcopy(master._pipeline_model_cleanup_ledger))
                assert {record["node_id"] for record in debts.values()} == {"worker", "android"}
                assert all(record["action"] == "release" for record in debts.values())

            c.retire = retire
            c.retire_timeout = 0.03
            if retirement == "recovered":
                await retire()
            fresh = await c.create({**case.intent, "model_id": "two-node-replacement"})
            assert fresh["operation_id"] != old and len(supply._operations) == 2
            assert supply.get(fresh["operation_id"])["wait_for_nodes"] is True
            assert master._connected_client_ids() | {"master"} == {"master", "rog"}
            assert master._pipeline_model_cleanup_ledger == debts
            assert retire_calls == [old] and c._retire_task is None
            assert not c.get(old)["loaded_ready"]
            c.cancel(fresh["operation_id"])
        finally:
            await c.close()

    asyncio.run(scenario())


def test_reconnected_native_debt_blocks_both_gates_until_exact_release_ack(stage_case, monkeypatch):
    from tests.test_worker_prepare_coordinator import publish

    case = stage_case
    master = case.master
    publish(case)
    for record in list(master._pipeline_model_cleanup_ledger.values()):
        master._handle_worker_prepare_ack(record["node_id"], {
            "action": "ack", "phase": "finalize", "status": "finalized", "loaded": True,
            "identity": record["payload"]["worker_prepare_identity"], "cleanup_id": record["cleanup_id"],
        })
    master._invalidate_pipeline_load_transaction("test_two_node_retirement")
    debts = deepcopy(master._pipeline_model_cleanup_ledger)
    assert {record["action"] for record in debts.values()} == {"release"}
    case.connected.clear()
    case.connected.add("rog")
    c = ModelIntentCoordinator(master, lambda op: None)
    assert c._retirement_state() == (False, False)
    assert master._pipeline_model_cleanup_ledger == debts

    case.connected.add("worker")
    returning = next(record for record in debts.values() if record["node_id"] == "worker")
    assert master._replay_pipeline_model_cleanup_for_node("worker") is True
    assert case.sent[-1][1] == {
        "action": "release", "identity": returning["payload"]["worker_prepare_identity"],
        "cleanup_id": returning["cleanup_id"],
    }
    assert c._retirement_state() == (False, True)

    def assert_blocked():
        blocked = master.start_pipeline_model_transaction(
            candidate_id="after-reconnect", descriptor={}, model_path=str(case.model_path), quant_type="fp16")
        assert blocked["admitted"] is False and blocked["reason_code"] == "MODEL_TXN_CLEANUP_PENDING"
        assert c._retirement_state() == (False, True)
        assert master._pipeline_model_cleanup_ledger == debts

    assert_blocked()
    exact = {"action": "ack", "phase": "release", "status": "released", "loaded": False,
             "identity": returning["payload"]["worker_prepare_identity"], "cleanup_id": returning["cleanup_id"]}
    for field, value in (("artifact_sha256", "f" * 64), ("generation", exact["identity"]["generation"] - 1),
                         ("operation_id", "supply_stale"), ("assignment_id", "stale-assignment")):
        wrong = deepcopy(exact)
        wrong["identity"][field] = value
        master._handle_worker_prepare_ack("worker", wrong)
        assert_blocked()
    for change in ({"cleanup_id": "stale-cleanup"}, {"loaded": True}):
        master._handle_worker_prepare_ack("worker", {**exact, **change})
        assert_blocked()
    master._handle_layer_config_ack("worker", {"data": {**returning["payload"], "status": "released"}})
    assert_blocked()

    master._handle_worker_prepare_ack("worker", exact)
    remaining = {key: record for key, record in debts.items() if record["node_id"] == "android"}
    assert remaining and master._pipeline_model_cleanup_ledger == remaining
    assert c._retirement_state() == (False, False)
    # Exercise native admission after the fence without starting another model load.
    admitted = []
    def start_next(**kwargs):
        admitted.append(kwargs["candidate"]["candidate_id"])
        return {"admitted": True}
    monkeypatch.setattr(master, "_push_layer_config_to_clients_locked", start_next)
    result = master.start_pipeline_model_transaction(
        candidate_id="after-reconnect", descriptor={}, model_path=str(case.model_path), quant_type="fp16")
    assert result["admitted"] is True and admitted == ["after-reconnect"]
    assert master._pipeline_model_cleanup_ledger == remaining


def test_selection_waits_real_owner_cleanup_before_new_inventory_and_serializes_requests(ready_intent):
    async def scenario():
        case = ready_intent
        entered, local_free, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
        retire_calls = []
        async def retire():
            retire_calls.append(case.op)
            entered.set()
            await released.wait()
            retire_local(case)
            case.scheduler._pipeline_model_cleanup_ledger["remote-release"] = {"node_id": "tail", "state": "await_reconnect"}
            local_free.set()
        case.c.retire = retire
        intent = {"model_id": "new"}
        first = asyncio.create_task(case.c.create(intent))
        await entered.wait()
        same = asyncio.create_task(case.c.create(intent))
        other = asyncio.create_task(case.c.create({"model_id": "other"}))
        assert case.c.current()["status"] == "ready" and not case.c.current()["loaded_ready"]
        # Reads from another thread must not block behind an RLock held across await.
        assert (await asyncio.wait_for(asyncio.to_thread(case.c.current), timeout=1))["status"] == "ready"
        assert len(case.supply.operations) == 1
        released.set()
        await local_free.wait()
        await asyncio.sleep(0.01)
        assert not first.done() and len(case.supply.operations) == 1
        # Only the native lifecycle owner's exact release ACK clears this debt.
        case.scheduler._pipeline_model_cleanup_ledger.clear()
        fresh = await first
        assert (await same)["operation_id"] == fresh["operation_id"]
        replacement = await other
        assert replacement["operation_id"] != fresh["operation_id"]
        assert case.c.get(fresh["operation_id"])["status"] == "superseded"
        assert retire_calls == [case.op] and len(case.supply.operations) == 3
        case.c.cancel(replacement["operation_id"])
        await case.c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["no_callback", "callback_error", "no_retirement", "missing_ledger", "pending_ack"])
def test_retirement_failure_is_named_and_never_creates_supply(ready_intent, failure):
    async def scenario():
        case = ready_intent
        case.c.retire_timeout = 0.02
        async def retire():
            if failure == "callback_error":
                raise OSError("unload failed")
            if failure != "no_retirement":
                retire_local(case)
            if failure == "missing_ledger":
                del case.scheduler._pipeline_model_cleanup_ledger
            if failure == "pending_ack":
                case.scheduler._pipeline_model_cleanup_ledger["release"] = {"node_id": "master", "state": "sent"}
        if failure != "no_callback":
            case.c.retire = retire
        with pytest.raises(ModelSupplyError) as error:
            await case.c.create({"model_id": "new"})
        expected = "INTENT_RETIRE_UNAVAILABLE" if failure == "no_callback" else "INTENT_RETIRE_TIMEOUT" if failure == "pending_ack" else "INTENT_RETIRE_FAILED"
        assert error.value.code == expected
        assert len(case.supply.operations) == 1 and not case.c._tasks
        if failure != "no_callback":
            assert not case.c.current()["loaded_ready"]
    asyncio.run(scenario())


@pytest.mark.parametrize("exit", ["mode", "role", "shutdown"])
def test_mode_exit_or_demotion_during_retirement_cannot_start_late_inventory(ready_intent, exit):
    async def scenario():
        case = ready_intent
        entered, release = asyncio.Event(), asyncio.Event()
        async def retire():
            entered.set()
            await release.wait()
            retire_local(case)
        case.c.retire = retire
        selection = asyncio.create_task(case.c.create({"model_id": "new"}))
        await entered.wait()
        if exit == "mode":
            case.scheduler.get_distributed_inference_enabled = lambda: False
        elif exit == "role":
            case.scheduler._effective_role = lambda: "client"
        else:
            await case.c.close()
        with pytest.raises(ModelSupplyError) as error:
            await selection
        assert error.value.code == {"mode": "SUPPLY_LOCAL_MODE", "role": "SUPPLY_MASTER_REQUIRED", "shutdown": "INTENT_MASTER_STOPPED"}[exit]
        # Rejoin/demotion reversal does not replay the abandoned selection.
        if exit == "mode":
            case.scheduler.get_distributed_inference_enabled = lambda: True
        elif exit == "role":
            case.scheduler._effective_role = lambda: "master"
        task = case.c._retire_task
        release.set()
        await task
        await asyncio.sleep(0.005)
        assert len(case.supply.operations) == 1 and not case.c._tasks
    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
def test_late_unload_is_joined_before_explicit_retry_can_create_supply(ready_intent, stop):
    async def scenario():
        case = ready_intent
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def retire():
            calls.append(case.op)
            entered.set()
            await release.wait()
            retire_local(case)
        case.c.retire = retire
        case.c.retire_timeout = 0.02 if stop == "timeout" else 1
        selection = asyncio.create_task(case.c.create({"model_id": "new"}))
        await entered.wait()
        if stop == "cancel":
            selection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await selection
        else:
            with pytest.raises(ModelSupplyError) as error:
                await selection
            assert error.value.code == "INTENT_RETIRE_TIMEOUT"
        task = case.c._retire_task
        assert not task.done() and len(case.supply.operations) == 1
        case.c.retire_timeout = 1
        retry = asyncio.create_task(case.c.create({"model_id": "new"}))
        await asyncio.sleep(0.005)
        assert not retry.done() and calls == [case.op]
        release.set()
        fresh = await retry
        assert len(case.supply.operations) == 2 and case.c._retire_task is None
        case.c.cancel(fresh["operation_id"])
        await case.c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("state,expected", [
    (None, None), ({}, False), ({"cleanup": {"node_id": "master"}}, True),
    ({"cleanup": {"node_id": "tail", "state": "await_reconnect"}}, True),
    ({"cleanup": {"node_id": "offline", "state": "await_reconnect"}}, False),
    ({"cleanup": {"node_id": "offline", "state": "pending"}}, False),
])
def test_retirement_state_projects_cleanup_ledger_without_exporting_owner_dict(ready_intent, state, expected):
    case = ready_intent
    case.scheduler._pipeline_model_cleanup_ledger = state
    published, pending = case.c._retirement_state()
    assert published is True and pending is expected
    if state is not None:
        state.clear()
        assert pending is expected


@pytest.mark.parametrize("ready", [False, True])
def test_unassigned_inventory_candidate_never_projects_as_loaded_participant(tmp_path, ready):
    async def scenario():
        async def prepare(op_id):
            scheduler._pipeline_load_transaction = transaction(op_id)
            return {"transaction_id": "candidate"}
        c, scheduler, supply = setup(tmp_path, prepare)
        op = await c.create({"model_id": "fixture", "timeout_seconds": 30})
        op_id = op["operation_id"]
        supply.operations[op_id]["nodes"]["unassigned"] = {"status": "inventory"}
        assert "unassigned" in c.get(op_id)["nodes"]  # Inventory is still being collected.
        supply.operations[op_id].update(status="available", plan={"admitted": True, "assignments": [
            {"node_id": "master", "start_layer": 0, "end_layer": 4},
            {"node_id": "tail", "start_layer": 4, "end_layer": 8},
        ]})
        await asyncio.sleep(0.01)
        if ready:
            publish_ready(scheduler)
            await asyncio.sleep(0.01)
        result = c.get(op_id)
        assert set(result["nodes"]) == {"master", "tail"}
        assert result["loaded_ready"] is ready
        assert supply.get(op_id)["nodes"]["unassigned"] == {"status": "inventory"}
        if ready:
            assert all(node["status"] == "ready" for node in result["nodes"].values())
        await finish_tasks(c)
    asyncio.run(scenario())


def test_input_revision_change_replans_same_operation_after_exact_cleanup(tmp_path):
    async def scenario():
        calls = []
        async def prepare(op_id):
            calls.append(op_id)
            scheduler._pipeline_load_transaction = transaction(op_id)
            if len(calls) == 1:
                scheduler._pipeline_load_transaction.update(phase="aborted", reason_code="pipeline_global_commit_failed",
                    reason="deployment_input_revision_changed")
                scheduler._pipeline_model_cleanup_ledger["old-runtime"] = {"node_id": "tail"}
            return {"transaction_id": "candidate"}
        c, scheduler, supply = setup(tmp_path, prepare)
        op = (await c.create({"model_id": "fixture", "timeout_seconds": 30}))["operation_id"]
        supply.operations[op]["status"] = "available"
        await wait_until(lambda: c.get(op)["status"] == "waiting_capacity")
        assert calls == [op] and len(supply.operations) == 1
        scheduler._pipeline_model_cleanup_ledger.clear()  # Exact owner cleanup boundary fixture.
        await wait_until(lambda: len(calls) == 2)
        publish_ready(scheduler)
        await c._tasks[op]
        assert c.get(op)["status"] == "ready"
        assert calls == [op, op] and len(supply.operations) == 1
        await c.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["cancel", "deadline", "close"])
def test_retryable_disconnect_keeps_cause_and_original_deadline(native_intent, stop):
    async def scenario():
        case = native_intent
        c = case.coordinator
        async def prepare(operation_id):
            raise ModelSupplyError("SUPPLY_WORKER_DISCONNECTED", "Worker disconnected")
        c.prepare = prepare
        op = (await c.create({**case.intent, "timeout_seconds": 1 if stop == "deadline" else 30}))["operation_id"]
        task = c._tasks[op]
        case.master._pipeline_model_cleanup_ledger["waiting-runtime"] = {"node_id": "worker"}
        case.available(op)
        await wait_until(lambda: c.get(op)["status"] == "waiting_capacity")
        result = c.get(op)
        assert result["reason_code"] == "SUPPLY_WORKER_DISCONNECTED"
        assert result["reason"] == "Worker disconnected"
        assert result["commit_scope"] == "local_sqlite" and result["durable"] is True
        assert result["runnable"] is False and result["boot_id"]
        if stop == "cancel":
            c.cancel(op)
        elif stop == "close":
            await c.close()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)
        result = c.get(op)
        expected = {"cancel": "INTENT_CANCELLED", "deadline": "INTENT_DEADLINE_EXCEEDED", "close": "INTENT_MASTER_STOPPED"}[stop]
        assert result["reason_code"] == expected
        assert result["runnable"] is False
        assert len(case.master._model_control.snapshot()["operations"]) == 1
        await c.close()
    asyncio.run(scenario())


def test_second_selection_coordinator_reuses_durable_idempotency_receipt(native_intent):
    async def scenario():
        case = native_intent
        first = case.coordinator
        second = ModelIntentCoordinator(case.master, first.prepare)
        try:
            one = await first.create(case.intent, idempotency_key="same-user-key")
            two = await second.create(case.intent, idempotency_key="same-user-key")
            assert one["operation_id"] == two["operation_id"]
            assert not second._tasks
            state = case.master._model_control.snapshot()
            assert len(state["operations"]) == len(state["supplies"]) == 1
            case.supply._operations[one["operation_id"]]["reason_code"] = "UNCOMMITTED_CACHE"
            assert case.supply.get(one["operation_id"])["reason_code"] != "UNCOMMITTED_CACHE"
            with pytest.raises(ModelSupplyError) as conflict:
                await second.create({**case.intent, "model_id": "another-model"}, idempotency_key="same-user-key")
            assert conflict.value.code == "INTENT_IDEMPOTENCY_CONFLICT"
        finally:
            await first.close()
            await second.close()
    asyncio.run(scenario())
