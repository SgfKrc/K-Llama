"""User selection orchestration; supply and native transactions remain authoritative.

Only the chosen operation and the prepare invocation are durable here. Inventory,
file receipts, assignments and native readiness are projections of their owners.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import logging
from pathlib import Path
import threading
import time

from fastapi import HTTPException

from api_errors import error_code_from_exception
from model_supply import ModelSupplyError

TERMINAL = {"ready", "failed", "cancelled", "superseded"}
logger = logging.getLogger(__name__)


def _query_reads_enabled(owner):
    return (getattr(owner, "query_reads_enabled", False) is True
            and callable(getattr(owner, "query_snapshot", None))
            and callable(getattr(owner, "query_stamp", None)))


class ModelIntentCoordinator:
    def __init__(self, scheduler, prepare, *, retire=None, retire_timeout=300, root=None, poll_interval=0.25):
        self.scheduler = scheduler
        self.prepare = prepare
        self.retire = retire
        self.retire_timeout = retire_timeout
        self._selection_lane = asyncio.Lock()
        self._retire_task = None
        self._retire_requested = False
        self._closed = False
        self._root = root
        self.poll_interval = poll_interval
        self._lock = threading.RLock()
        self._records = {}
        self._tasks = {}
        self._current = ""

    @property
    def root(self):
        return Path(self._root) if self._root else self.scheduler._model_supply.root

    def _persist(self, record):
        with self._lock:
            owner = self.scheduler._model_control
            owner.start()
            snapshot = owner.snapshot()
            updating = record["operation_id"] in snapshot["operations"]
            owner.save_operation(record, select=self._current == record["operation_id"],
                                 expected_selected=record["operation_id"] if updating else snapshot["selected_operation"],
                                 idempotency_key=record.get("idempotency_key", ""))

    def _record(self, operation_id):
        self.scheduler._model_supply.get(operation_id)
        saved = self.scheduler._model_control.snapshot()["operations"].get(operation_id)
        if saved is None:
            raise ModelSupplyError("INTENT_OPERATION_NOT_FOUND", "Model selection operation not found")
        return deepcopy(saved)

    def current(self):
        owner = self.scheduler._model_control
        if _query_reads_enabled(owner):
            snapshot = owner.query_snapshot()
            selected = snapshot["selected_operation"]
            return self.get(selected, snapshot=snapshot) if selected else None
        selected = owner.snapshot()["selected_operation"]
        return self.get(selected) if selected else None

    def _transaction(self, operation_id, candidate_id):
        with self.scheduler._layer_config_lock:
            transaction = self.scheduler._pipeline_load_transaction or {}
            belongs = any(value.get("worker_prepare_identity", {}).get("operation_id") == operation_id
                          for value in transaction.get("expected", {}).values())
            if not belongs and (not candidate_id or transaction.get("candidate_id") != candidate_id):
                return None, None
            # Do not copy callbacks, runtime owners or previous control state.
            fields = ("phase", "config_id", "generation", "reason_code", "reason", "acks", "expected", "worker_ids")
            value = {key: deepcopy(transaction.get(key)) for key in fields}
            plan = self.scheduler._active_pipeline_capacity_plan or {}
            active = {key: deepcopy(plan.get(key)) for key in ("config_id", "generation", "transaction_phase")}
            if (transaction.get("phase") == "ready" and active == {
                    "config_id": transaction.get("config_id"), "generation": transaction.get("generation"),
                    "transaction_phase": "ready"} and self.scheduler.get_distributed_inference_enabled()):
                value["_native_ready"], value["_native_reason_code"], value["_native_reason"] = self._native_readiness(transaction)
            return value, active

    def get(self, operation_id, *, snapshot=None):
        from qlh_core.contracts.snapshots import envelope, reason
        owner = getattr(self.scheduler, "_model_control", None)
        query_reads = _query_reads_enabled(owner)
        for attempt in range(3):
            if query_reads:
                before = snapshot if attempt == 0 and snapshot is not None else owner.query_snapshot(operation_id=operation_id)
                result = self._project(operation_id, snapshot=before)
                after = owner.query_stamp()
            else:
                before = owner.snapshot() if owner is not None else None
                result = self._project(operation_id)
                after = owner.snapshot() if owner is not None else None
            if before is not None and (before["revision"] != after["revision"] or before["boot_id"] != after["boot_id"]):
                continue
            code = result.get("reason_code") or result.get("deployment_reason_code", "")
            return envelope(result, revision=after["revision"] if after is not None else 0,
                identity={"operation_id": operation_id, "boot_id": after["boot_id"] if after is not None else "legacy"},
                observed_at=time.time(), validity="valid" if after is None or after["boot_id"] == owner.boot_id else "invalid",
                reason=reason("model_operation", code, "model_control", details={"message": result.get("reason", "")}, cause=result.get("cause")))
        return envelope({"operation_id": operation_id, "status": "pending", "runnable": False, "loaded_ready": False},
                        revision=None, identity={"operation_id": operation_id}, observed_at=time.time(), validity="pending",
                        reason=reason("snapshot", "snapshot_revision_changed", "model_control", retryability="retry"))

    def _project(self, operation_id, *, snapshot=None):
        owner = getattr(self.scheduler, "_model_control", None)
        if snapshot is None:
            snapshot = owner.snapshot() if owner is not None else None
        if snapshot is not None and operation_id in snapshot["operations"] and operation_id in snapshot["supplies"]:
            record = snapshot["operations"][operation_id]
            supply = snapshot["supplies"][operation_id]
        else:
            record = self._record(operation_id)
            supply = self.scheduler._model_supply.get(operation_id)
        result = {key: deepcopy(supply[key]) for key in (
            "operation_id", "intent", "status", "reason_code", "reason", "created_at", "updated_at",
            "nodes", "plan", "source_progress", "capacity_diagnostic",
        ) if key in supply}
        result.update(config_id=record.get("config_id", ""), generation=record.get("generation", 0),
                      loaded_ready=False, reason=result.get("reason", ""), reason_code=result.get("reason_code", ""))
        result["runnable"] = False
        result["cause"] = deepcopy(record.get("cause"))
        if snapshot is not None:
            result.update(revision=snapshot["revision"], boot_id=snapshot["boot_id"],
                          commit_scope=record.get("commit_scope", snapshot["commit_scope"]),
                          durable=record.get("durable", True), deployment_id=result["config_id"])
        # Inventory candidates are not necessarily participants in the admitted
        # plan. Project only assigned nodes once the supply owner publishes it;
        # a global ready phase must never promote an unused candidate to ready.
        plan = supply.get("plan") or {}
        assigned = {item.get("node_id") for item in plan.get("assignments", [])
                    if isinstance(item, dict) and isinstance(item.get("node_id"), str)}
        if plan.get("admitted") is True and assigned:
            result["nodes"] = {node_id: node for node_id, node in result.get("nodes", {}).items()
                               if node_id in assigned}
        for node in result.get("nodes", {}).values():
            node["bytes_received"] = node.get("downloaded_bytes", 0)
            if node.get("status") == "available":
                node["bytes_received"] = node.get("total_bytes", node["bytes_received"])
            if node.get("phase") in {"receiving", "downloading", "download"} and node.get("status") in {"supplying", "receiving", "downloading"}:
                node["status"] = "downloading"
            if node.get("phase") in {"verify", "verifying", "registering"} and node.get("status") != "available":
                node["status"] = "verifying"
        if record["status"] == "waiting_capacity":
            result.update(status="waiting_capacity", phase="waiting_capacity", reason_code=record.get("reason_code", ""),
                          reason=record.get("reason", ""))
            return result
        if record["status"] in {"cancelled", "superseded"}:
            result.update(status=record["status"], reason_code=record.get("reason_code", ""), reason=record.get("reason", ""))
            return result
        if supply["status"] in {"failed", "cancelled"}:
            if record["status"] == "failed":
                result.update(status="failed", reason_code=record.get("reason_code", ""), reason=record.get("reason", ""))
            return result
        transaction, active = self._transaction(operation_id, record.get("candidate_id", ""))
        if transaction:
            result.update(config_id=transaction.get("config_id") or "", generation=transaction.get("generation") or 0)
            phase = transaction["phase"]
            if phase in {"aborted", "invalidated", "rejected"}:
                result.update(status="failed", reason_code=transaction.get("reason_code") or "INTENT_PREPARE_FAILED",
                              reason=transaction.get("reason") or "The selected model could not be prepared")
            elif self._is_ready(transaction, active):
                result.update(status="ready", loaded_ready=True, reason_code="", reason="")
            elif phase == "ready" or record["status"] == "ready":
                result.update(status="superseded", reason_code=transaction.get("_native_reason_code") or "INTENT_PLAN_RETIRED",
                              reason=transaction.get("_native_reason") or "The selected inference plan has been retired; retry the selection")
            else:
                result.update(status="preparing", reason_code="", reason="")
            for node_id, node in result.get("nodes", {}).items():
                ack = (transaction.get("acks") or {}).get(node_id, {})
                if result["status"] == "ready":
                    node["status"] = "ready"
                elif ack.get("status") == "error":
                    node.update(status="failed", reason_code=ack.get("error_code", "WORKER_PREPARE_FAILED"),
                                reason=ack.get("error", "Worker failed to load its segment"))
                elif node_id == "master" or node_id in (transaction.get("expected") or {}):
                    node["status"] = "prepared" if ack.get("phase") == "prepare" and ack.get("loaded") is True else "loading"
        elif record["status"] == "ready" or record.get("config_id"):
            result.update(status="superseded", reason_code="INTENT_PLAN_RETIRED", reason="The selected inference plan has been retired")
        elif supply["status"] == "available":
            result.update(status="preparing", reason_code="", reason="")
        if record["status"] == "failed":
            result.update(status="failed", loaded_ready=False, reason_code=record.get("reason_code", ""), reason=record.get("reason", ""))
        if record["status"] == "ready":
            result["status"] = "ready"
            if not result["loaded_ready"]:
                result["deployment_reason_code"] = result.get("reason_code", "INTENT_PLAN_RETIRED")
                result["reason_code"], result["reason"] = "", ""
        result["runnable"] = result["loaded_ready"]
        if snapshot is not None:
            result["deployment_id"] = result["config_id"]
        return result

    def _native_readiness(self, transaction):
        # Called under the native owner's lock; this is a projection, never a receipt.
        if self._retire_task is not None or self._retire_requested:
            return False, "INTENT_PLAN_RETIRING", "The previous inference plan is being retired; retry the selection"
        execution = getattr(self.scheduler, "_execution_app", None)
        if execution is not None:
            try:
                plan = execution.freeze()
                if plan.deployment_id != transaction.get("config_id"):
                    return False, "execution_deployment_mismatch", "The current deployment changed"
            except Exception as exc:
                return False, getattr(exc, "code", "execution_deployment_unavailable"), "Deployment admission is unavailable"
        loaded = getattr(self.scheduler, "_worker_prepare_assignment_loaded", None)
        expected = transaction.get("expected") or {}
        workers = transaction.get("worker_ids") or set()
        if not callable(loaded) or not expected or set(expected) != set(workers):
            return False, "INTENT_NATIVE_EVIDENCE_UNAVAILABLE", "Native worker readiness evidence is unavailable; retry the selection"
        try:
            for node_id, assignment in expected.items():
                ready, code = loaded(node_id, assignment)
                if ready is not True:
                    return False, code or "INTENT_NATIVE_READINESS_LOST", f"Worker {node_id} is no longer native-ready; retry the selection"
        except Exception:
            logger.exception("event=model_intent_native_readiness_unavailable")
            return False, "INTENT_NATIVE_EVIDENCE_UNAVAILABLE", "Native worker readiness evidence is unavailable; retry the selection"
        return True, "", ""

    def _is_ready(self, transaction, active):
        return bool(transaction and transaction.get("phase") == "ready" and active == {
            "config_id": transaction["config_id"], "generation": transaction["generation"],
            "transaction_phase": "ready",
        } and transaction.get("_native_ready") is True and self.scheduler.get_distributed_inference_enabled())

    def _persist_terminal(self, record):
        # Disk failure must not prevent the owner's cancellation or its retry.
        try:
            self._persist(record)
        except Exception:
            owner = getattr(self.scheduler, "_model_control", None)
            if owner is not None:
                owner.fence_operation_failure(record)
            logger.exception("event=model_intent_persist_failed operation_id=%s", record["operation_id"])

    def _revoke(self, operation_id):
        try:
            # Supply attempts native and receiver cleanup even when saving fails.
            self.scheduler._model_supply.cancel(operation_id)
        except Exception as exc:
            if isinstance(exc, ModelSupplyError) and exc.code == "SUPPLY_OPERATION_NOT_FOUND":
                # A recovered disk receipt has no live supply task. Still check
                # the existing native owner before allowing a fresh selection.
                cancel_prepare = getattr(self.scheduler, "cancel_worker_prepare_operation", None)
                if callable(cancel_prepare):
                    try:
                        cancel_prepare(operation_id)
                    except Exception as native_exc:
                        raise ModelSupplyError("INTENT_CANCEL_FAILED", "Native model cancellation failed; retry cancellation") from native_exc
                return
            raise ModelSupplyError("INTENT_CANCEL_FAILED", "Model selection cancellation failed; retry cancellation") from exc

    def _fail(self, operation_id, code, reason, *, cause=None):
        try:
            self._revoke(operation_id)
        except Exception:
            logger.exception("event=model_intent_cancel_failed operation_id=%s", operation_id)
        with self._lock:
            try:
                record = self._record(operation_id)
            except ModelSupplyError:
                record = self._records.get(operation_id)
            if record is None:
                return
            self._records[operation_id] = record
            if record["status"] not in {"cancelled", "failed", "superseded"}:
                transaction, active = self._transaction(operation_id, record.get("candidate_id", ""))
                if self._is_ready(transaction, active):
                    # A published native plan remains authoritative even when
                    # saving its final selection projection fails.
                    record.update(status="ready", config_id=transaction["config_id"],
                                  generation=transaction["generation"], reason_code="", reason="")
                else:
                    record.update(status="failed", reason_code=code, reason=reason, cause=deepcopy(cause))
            self._persist_terminal(record)

    def _task_done(self, operation_id, task):
        with self._lock:
            owned = self._tasks.get(operation_id) is task
            if owned:
                self._tasks.pop(operation_id, None)
        if task.cancelled():
            if owned:  # Cancellation before the coroutine's first instruction.
                self._fail(operation_id, "INTENT_MASTER_STOPPED", "The master stopped model selection")
        else:
            task.exception()  # Retrieve any failure, including during final cleanup.

    def _check_selection_current(self):
        if self._closed:
            raise ModelSupplyError("INTENT_MASTER_STOPPED", "The master stopped model selection")
        # Reuse the supply owner's live role/mode fence, including after every await.
        self.scheduler._model_supply._check_master()

    def _retirement_state(self):
        # Match native candidate admission; offline debts remain owned by the
        # cleanup ledger and become blockers again when their nodes reconnect.
        connected_for_cleanup = set(self.scheduler._connected_client_ids())
        with self.scheduler._layer_config_lock:
            transaction = self.scheduler._pipeline_load_transaction or {}
            plan = self.scheduler._active_pipeline_capacity_plan or {}
            # ★ 2026-10-23（#117）：**"待恢复"不等于"已发布"**。此前把
            #   `_pipeline_recovery_pending` 也算作 published，而重启后的恢复必然因
            #   供给身份不一致被拒（实测 `pipeline_supply_identity_mismatch:
            #   source_model_sha256`，`selected_operation` 指向一次失败的供给）⇒
            #   该标志永远是 True ⇒ `_retire_before_supply` 恒报"没有完成证据"
            #   （INTENT_RETIRE_FAILED）⇒ 每次重新选模都 409；又因为"没有已加载的
            #   模型"，正常 retire 也产生不了完成证据，形成死锁。
            #   恢复围栏由 `retire_pipeline_recovery_for_model_change()` 清理，
            #   不应在这里冒充发布状态。
            published = (transaction.get("phase") == "ready"
                         or plan.get("transaction_phase") == "ready")
            ledger = getattr(self.scheduler, "_pipeline_model_cleanup_ledger", None)
            pending = None if ledger is None else any(
                record.get("node_id") == "master" or record.get("node_id") in connected_for_cleanup
                for record in ledger.values()
            )
            return published, pending

    async def _retire_before_supply(self):
        self._retire_requested = True
        if self._retire_task is None:
            if not callable(self.retire):
                raise ModelSupplyError("INTENT_RETIRE_UNAVAILABLE", "Native plan retirement is unavailable; retry the selection")
            self._retire_task = asyncio.create_task(self.retire())
            # A timed-out/cancelled caller does not own cancellation of native unload.
            # Keep its task for the next explicit selection, and retrieve late errors.
            self._retire_task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        task = self._retire_task
        deadline = time.monotonic() + self.retire_timeout
        while True:
            self._check_selection_current()
            if task.done():
                try:
                    task.result()
                except ModelSupplyError:
                    self._retire_task = None
                    raise
                except (Exception, asyncio.CancelledError) as exc:
                    self._retire_task = None
                    raise ModelSupplyError("INTENT_RETIRE_FAILED", "Native model retirement failed; retry the selection") from exc
                published, ledger = self._retirement_state()
                if published or ledger is None:
                    self._retire_task = None
                    raise ModelSupplyError("INTENT_RETIRE_FAILED", "Native model retirement has no completed owner evidence; retry the selection")
                if not ledger:
                    self._retire_task = None
                    self._retire_requested = False
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelSupplyError("INTENT_RETIRE_TIMEOUT", "Waiting for native cleanup acknowledgements timed out; retry the selection")
            if task.done():
                await asyncio.sleep(min(self.poll_interval, remaining))
            else:
                await asyncio.wait({task}, timeout=min(self.poll_interval, remaining))

    async def create(self, intent, *, idempotency_key=""):
        # Serialize asynchronous retirement as well as the supply->prepare gap.
        async with self._selection_lane:
            self._check_selection_current()
            owner = getattr(self.scheduler, "_model_control", None)
            if idempotency_key and owner is not None:
                for prior in owner.snapshot()["operations"].values():
                    if prior.get("idempotency_key") == idempotency_key:
                        saved = self.scheduler._model_supply.get(prior["operation_id"])
                        if saved.get("intent") != intent:
                            raise ModelSupplyError("INTENT_IDEMPOTENCY_CONFLICT", "Idempotency key belongs to another selection")
                        return self.get(prior["operation_id"])
            with self._lock:
                current = self.current()
                if (current and not idempotency_key and current.get("intent") == intent
                        and current.get("status") == "ready" and current.get("runnable") is True
                        and current.get("validity") == "valid"):
                    saved = self.scheduler._model_supply.get(current["operation_id"])
                    connected = set(self.scheduler._connected_client_ids())
                    _, cleanup_pending = self._retirement_state()
                    with self.scheduler._layer_config_lock:
                        transaction = self.scheduler._pipeline_load_transaction or {}
                        candidates = saved.get("candidate_node_ids")
                        observed = (set(candidates) - {"master"} if candidates is not None
                                    else set(transaction.get("worker_ids") or ()))
                        if connected == observed and cleanup_pending is False:
                            reused = self.get(current["operation_id"])
                            if reused.get("runnable") is True and reused.get("validity") == "valid":
                                logger.info("event=model_intent_ready_reused operation=%s config=%s generation=%s",
                                            reused["operation_id"], reused["config_id"], reused["generation"])
                                return reused
                if current and current["status"] not in TERMINAL:
                    if current["intent"] == intent:
                        return current
                    record = self._record(current["operation_id"])
                    record.update(status="superseded", reason_code="INTENT_SUPERSEDED", reason="A newer model selection replaced this operation")
                    self._persist_terminal(record)
                    self._revoke(current["operation_id"])
            published, ledger = self._retirement_state()
            if published or ledger or self._retire_task is not None:
                try:
                    await self._retire_before_supply()
                except ModelSupplyError as exc:
                    if current:
                        with self._lock:
                            record = self._record(current["operation_id"])
                            if record["status"] not in TERMINAL:
                                record.update(status="superseded", reason_code=exc.code, reason=exc.message)
                                self._persist_terminal(record)
                    raise
            # No thread RLock is held over an await. A late unload cannot re-enable
            # participation or start inventory after mode exit/master demotion.
            self._check_selection_current()
            with self._lock:
                if current and current["status"] in {"failed", "cancelled", "superseded"}:
                    self._revoke(current["operation_id"])
                supply_owner = self.scheduler._model_supply
                accept = getattr(supply_owner, "accept", None)
                options = {"wait_for_nodes": True, **({"idempotency_key": idempotency_key} if idempotency_key else {})}
                if callable(accept):
                    supply, created = accept(intent, **options)
                    if not created:
                        return self.get(supply["operation_id"])
                else:
                    supply = supply_owner.create(intent, **options)
                operation_id = supply["operation_id"]
                record = {"operation_id": operation_id, "status": "queued", "reason_code": "", "reason": "",
                          "idempotency_key": idempotency_key}
                self._current = operation_id
                self._records[operation_id] = record
                try:
                    self._persist(record)
                except Exception as exc:
                    self._fail(operation_id, "INTENT_PERSIST_FAILED", "Model selection could not be saved; retry the selection")
                    raise ModelSupplyError("INTENT_PERSIST_FAILED", "Model selection could not be saved; retry the selection") from exc
                task = asyncio.create_task(self._advance(operation_id))
                self._tasks[operation_id] = task
                task.add_done_callback(lambda completed: self._task_done(operation_id, completed))
                return self.get(operation_id)

    def _update(self, operation_id, **values):
        with self._lock:
            record = self._record(operation_id)
            if record["status"] in TERMINAL:
                return False
            self._records[operation_id] = record
            if record["status"] in {"cancelled", "failed", "superseded"}:
                return False
            record.update(values)
            try:
                self._persist(record)
            except Exception as exc:
                raise ModelSupplyError("INTENT_PERSIST_FAILED", "Model selection could not be saved; retry the selection") from exc
            return True

    async def _drain_prepare(self, task):
        # Cancelling a threadpool await cannot stop its native thread. Keep the
        # callback and prepare_guard alive until transaction rejection/cleanup
        # completes; do not abandon its result or exception.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _retryable_prepare(code, reason=""):
        return code in {"SUPPLY_RUNTIME_CONNECTION_CHANGED", "SUPPLY_RUNTIME_CAPABILITY_CHANGED",
                        "SUPPLY_RUNTIME_NODE_UNAVAILABLE", "SUPPLY_CAPACITY_STALE", "SUPPLY_WORKER_DISCONNECTED",
                        "pipeline_worker_disconnected", "MODEL_TXN_CLEANUP_PENDING"} or reason in {
                            "deployment_input_revision_changed", "deployment_input_changed"}

    async def _wait_for_replan(self, operation_id, deadline, *, code="", reason=""):
        if not self._update(operation_id, status="waiting_capacity", candidate_id="",
                            reason_code=code or "INTENT_INPUTS_CHANGED",
                            reason=reason or "Deployment inputs changed; waiting to plan again"):
            return False
        while True:
            self._check_selection_current()
            if self._record(operation_id)["status"] in TERMINAL:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelSupplyError("INTENT_DEADLINE_EXCEEDED", "Model selection timed out while waiting to plan again")
            await asyncio.sleep(min(self.poll_interval, remaining))
            _, pending = self._retirement_state()
            if not pending:
                return True

    async def _advance(self, operation_id):
        prepare_task = None
        try:
            supply = self.scheduler._model_supply.get(operation_id)
            saved = self._record(operation_id)
            remaining = (saved["deadline"] - time.time() if saved.get("deadline") is not None
                         else supply["intent"].get("timeout_seconds", 3600))
            deadline = time.monotonic() + max(0, remaining)
            while True:
                record = self._record(operation_id)
                if record["status"] in {"cancelled", "failed", "superseded"}:
                    return
                supply = self.scheduler._model_supply.get(operation_id)
                if supply["status"] in {"failed", "cancelled"}:
                    self._update(operation_id, status=supply["status"], reason_code=supply.get("reason_code", ""), reason=supply.get("reason", ""))
                    return
                if time.monotonic() >= deadline:
                    diagnostic = supply.get("capacity_diagnostic")
                    if (isinstance(diagnostic, dict) and diagnostic.get("reason_code")
                            and diagnostic["reason_code"] == supply.get("reason_code")):
                        self._fail(operation_id, diagnostic["reason_code"],
                                   diagnostic["reason"] + " 本次选择已到截止时间，请释放资源或增加节点后重试。")
                    else:
                        self._fail(operation_id, "INTENT_DEADLINE_EXCEEDED", "Model selection timed out")
                    return
                if supply["status"] == "available":
                    break
                await asyncio.sleep(self.poll_interval)
            while True:
                self.scheduler._model_supply.ensure_prepare_not_cancelled(operation_id)
                if not self._update(operation_id, status="preparing"):
                    return
                prepare_task = asyncio.create_task(self.prepare(operation_id))
                while True:
                    if self._record(operation_id)["status"] in {"cancelled", "failed", "superseded"}:
                        return
                    self.scheduler._model_supply.ensure_prepare_not_cancelled(operation_id)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._fail(operation_id, "INTENT_DEADLINE_EXCEEDED", "Model preparation timed out")
                        return
                    if prepare_task.done():
                        break
                    await asyncio.wait({prepare_task}, timeout=min(self.poll_interval, remaining))
                try:
                    result = prepare_task.result()
                except Exception as exc:
                    code = error_code_from_exception(exc) if isinstance(exc, HTTPException) else getattr(exc, "code", "")
                    if not self._retryable_prepare(code):
                        raise
                    detail = getattr(exc, "detail", None)
                    reason = (detail.get("message", "Waiting for fresh deployment inputs") if isinstance(detail, dict)
                              else getattr(exc, "message", "Waiting for fresh deployment inputs"))
                    if not await self._wait_for_replan(operation_id, deadline, code=code, reason=reason):
                        return
                    continue
                if not self._update(operation_id, candidate_id=result.get("transaction_id", "")):
                    return
                while True:
                    projected = self.get(operation_id)
                    transaction, _ = self._transaction(operation_id, self._record(operation_id).get("candidate_id", ""))
                    if (projected["status"] == "failed" and self._retryable_prepare(
                            projected.get("reason_code", ""), (transaction or {}).get("reason", ""))):
                        if not await self._wait_for_replan(operation_id, deadline,
                                code=projected.get("reason_code", ""), reason=projected.get("reason", "")):
                            return
                        break
                    if projected["status"] in TERMINAL:
                        self._update(operation_id, status=projected["status"], config_id=projected["config_id"],
                                     generation=projected["generation"], reason_code=projected["reason_code"], reason=projected["reason"])
                        return
                    if time.monotonic() >= deadline:
                        self._fail(operation_id, "INTENT_DEADLINE_EXCEEDED", "Model preparation timed out")
                        return
                    await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            self._fail(operation_id, "INTENT_MASTER_STOPPED", "The master stopped model selection")
            raise
        except Exception as exc:
            detail = getattr(exc, "detail", None)
            code = (
                error_code_from_exception(exc) if isinstance(exc, HTTPException)
                else getattr(exc, "code", "")
            ) or "INTENT_PREPARE_FAILED"
            # Raw HTTP strings and unexpected exception text stay in diagnostics.
            reason = detail.get("message", "Model preparation failed") if isinstance(detail, dict) else getattr(exc, "message", "Model preparation failed; inspect master diagnostics")
            logger.exception("event=model_intent_prepare_failed operation_id=%s reason_code=%s", operation_id, code)
            self._fail(operation_id, code, reason, cause=(detail.get("reason_detail") if isinstance(detail, dict) else getattr(exc, "reason_detail", None)))
        finally:
            if prepare_task is not None:
                await self._drain_prepare(prepare_task)
                # Retry native cancellation after late callback completion, too.
                if self._records[operation_id]["status"] in {"failed", "cancelled", "superseded"}:
                    try:
                        self._revoke(operation_id)
                    except Exception:
                        logger.exception("event=model_intent_cancel_failed operation_id=%s", operation_id)
            self._tasks.pop(operation_id, None)

    def cancel(self, operation_id):
        with self._lock:
            current = self.get(operation_id)
            if current["status"] == "ready":
                raise ModelSupplyError("INTENT_ALREADY_READY", "Use unload to retire a ready inference plan")
            # Retry the owner even for a failed/cancelled selection. Neither a
            # previous disk failure nor a failed native abort is a success fence.
            try:
                self._revoke(operation_id)
            finally:
                record = self._record(operation_id)
                if current["status"] not in TERMINAL:
                    record.update(status="cancelled", reason_code="INTENT_CANCELLED", reason="Model selection cancelled")
                self._records[operation_id] = record
                self._persist_terminal(record)
            return self.get(operation_id)

    async def close(self):
        self._closed = True
        tasks = list(self._tasks.items())
        # Fence even tasks that have not yet run, before requesting cancellation.
        for operation_id, task in tasks:
            self._fail(operation_id, "INTENT_MASTER_STOPPED", "The master stopped model selection")
            task.cancel()
        await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
