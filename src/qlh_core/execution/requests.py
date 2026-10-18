"""Bounded FIFO admission, one terminal decision, and separately tracked draining."""
from __future__ import annotations
from collections import deque
from copy import deepcopy
import threading
import time
import uuid
from qlh_core.contracts.deadline import DeadlineCancelEvent, request_stop_reason
from qlh_core.contracts.execution import ExecutionError
from qlh_core.contracts.outcome import TerminalOutcomeLatch


class RequestService:
    def __init__(self, repository, *, boot_id, capacity=32):
        if capacity < 1:
            raise ValueError("request queue capacity must be positive")
        self.repository = repository
        self.boot_id = boot_id
        self.capacity = capacity
        self._condition = threading.Condition(threading.RLock())
        self._queue = deque()
        self._records = {}
        self._active = None
        self._paused = False
        self._closed = False
        self._started = False
        self._thread = None
        self._quarantine = set()
        self._persistence_failed = False
        self._revision = 0

    def start(self):
        with self._condition:
            if self._started:
                return
            recovery = self.repository.recovery_status()
            if recovery.get("corrupt"):
                raise ExecutionError("request_journal_corrupt")
            def recover(state):
                for record in state["requests"].values():
                    if not record.get("terminal"):
                        record.update(terminal="request_process_restarted", outcome="failed")
                    if record.get("execution_state") != "stopped":
                        record["execution_state"] = "recovery_required"
                state["boot_id"] = self.boot_id
            recovered = self.repository.transact(recover)
            self._revision = max(self._revision, recovered["revision"])
            self._started = True

    def _save(self, record):
        self._revision += 1
        record["revision"] = record.get("revision", 0) + 1
        record["observed_at"] = time.time()
        value = {key: deepcopy(item) for key, item in record.items()
                 if key not in {"execute", "cleanup", "cancel", "latch", "deadline", "result"}}
        def save(state):
            if state["boot_id"] != self.boot_id:
                raise ExecutionError("request_boot_superseded")
            state["requests"][value["request_id"]] = value
        try:
            persisted = self.repository.transact(save)
            self._revision = max(self._revision, persisted["revision"])
            return True
        except Exception:
            # Once durability fails, no new native work is admitted. The current
            # thread still owns physical cleanup and all waiting callers wake.
            self._persistence_failed = True
            self._quarantine.add(record["deployment_id"])
            record["latch"].decide("request_journal_write_failed", source="request_service")
            reason = record["latch"].reason_code
            record.update(terminal=reason, outcome=("completed" if reason == "completed" else
                          "cancelled" if reason == "generation_cancelled" else "failed"), durable=False)
            self._condition.notify_all()
            return False

    def _decide(self, record, reason, outcome):
        if record["terminal"]:
            return
        if record["phase"] == "queued":
            try:
                self._queue.remove(record["request_id"])
            except ValueError:
                pass
            record.update(phase="finished", execution_state="stopped")
            record.pop("execute", None)
            record.pop("cleanup", None)
        selected = record["latch"].reason_code or reason
        record.update(terminal=selected, outcome=("cancelled" if selected == "generation_cancelled" else outcome))
        if self._save(record):
            record["latch"].decide(selected, source="request_service")
        self._condition.notify_all()

    def run(self, *, plan, deadline, execute, cleanup, cancel_event=None, terminal=None,
            request_id=None, context_revision=None, sampling=None):
        self.start()
        with self._condition:
            reason = request_stop_reason(cancel_event, deadline)
            if reason:
                return self._refused(reason)
            if self._closed:
                return self._refused("request_service_closed")
            if self._persistence_failed:
                return self._refused("request_journal_write_failed")
            if self._paused:
                return self._refused("request_queue_paused")
            if plan.deployment_id in self._quarantine:
                return self._refused("request_runtime_cleanup_pending")
            if len(self._queue) >= self.capacity:
                return self._refused("request_queue_full")
            key = request_id or f"req_{uuid.uuid4().hex}"
            if key in self._records:
                raise ExecutionError("request_identity_conflict")
            record = {"request_id": key, "attempt_id": f"att_{uuid.uuid4().hex}",
                      "boot_id": self.boot_id, "deployment_id": plan.deployment_id,
                      "plan_digest": plan.digest, "context_revision": context_revision,
                      "authority": plan.authority,
                      "sampling": {"temperature": sampling.temperature, "top_p": sampling.top_p} if sampling else None,
                      "assignments": [{"node_id": s.node_id, "assignment_id": s.assignment_id,
                                       "incarnation": s.incarnation, "connection": s.connection} for s in plan.segments],
                      "deadline_epoch_ms": deadline.expires_at_epoch_ms,
                      "phase": "queued", "execution_state": "waiting", "terminal": "", "outcome": "",
                      "admitted_at": time.time(), "execute": execute, "cleanup": cleanup,
                      "deadline": deadline, "cancel": DeadlineCancelEvent(cancel_event, deadline),
                      "latch": terminal or TerminalOutcomeLatch(), "result": None}
            if not self._save(record):
                return self._refused("request_journal_write_failed")
            self._records[key] = record
            self._queue.append(key)
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="linear-request-executor", daemon=True)
                self._thread.start()
            self._condition.notify_all()
            while not record["terminal"]:
                reason = request_stop_reason(record["cancel"], deadline)
                if reason:
                    record["cancel"].set()
                    self._decide(record, reason, "cancelled" if reason == "generation_cancelled" else "failed")
                    break
                self._condition.wait(min(0.05, deadline.remaining()))
            if record["outcome"] == "completed":
                result = dict(record["result"] or {})
            else:
                result = {"response": "", "error": record["terminal"], "reason_code": record["terminal"],
                          "cancelled": record["outcome"] == "cancelled"}
            result.setdefault("metrics", {})["request"] = self._snapshot_record(record)
            return result

    def _refused(self, code):
        from qlh_core.contracts.snapshots import envelope, reason
        diagnostic = reason("request", code, "request_admission", retryability="retry")
        return {"response":"", "error":code, "reason_code":code, "reason_detail":diagnostic,
            "metrics":{"fallback":False, "request":envelope({"outcome":"refused", "terminal":code},
                revision=self._revision, identity={"boot_id":self.boot_id}, observed_at=time.time(), reason=diagnostic)}}

    def _loop(self):
        while True:
            with self._condition:
                while not self._queue:
                    if self._closed:
                        return
                    self._condition.wait()
                key = self._queue.popleft()
                record = self._records[key]
                reason = request_stop_reason(record["cancel"], record["deadline"])
                if not record["terminal"] and reason:
                    self._decide(record, reason, "cancelled" if reason == "generation_cancelled" else "failed")
                if not record["terminal"] and record["deployment_id"] in self._quarantine:
                    self._decide(record, "request_runtime_cleanup_pending", "refused")
                if record["terminal"]:
                    record.update(phase="finished", execution_state="stopped")
                    self._save(record)
                    continue
                self._active = key
                record.update(phase="running", execution_state="running")
                if not self._save(record):
                    record.update(phase="finished", execution_state="stopped")
                    self._active = None
                    continue
            result = None
            error = None
            try:
                result = record["execute"](record)
            except BaseException as exc:
                error = getattr(exc, "code", "request_execution_failed")
            finally:
                with self._condition:
                    record["execution_state"] = "draining"
                    self._save(record)
                try:
                    # Cleanup must positively confirm native stop and KV release.
                    clean = record["cleanup"](record) is True
                except BaseException:
                    clean = False
                with self._condition:
                    reason = request_stop_reason(record["cancel"], record["deadline"])
                    if not clean:
                        self._quarantine.add(record["deployment_id"])
                    record.update(phase="finished", execution_state="stopped" if clean else "cleanup_pending")
                    record["result"] = result
                    if reason:
                        self._decide(record, reason, "cancelled" if reason == "generation_cancelled" else "failed")
                    elif not clean:
                        self._decide(record, "request_runtime_cleanup_pending", "failed")
                    elif not isinstance(result, dict):
                        self._decide(record, error or "request_result_invalid", "failed")
                    elif error or (result or {}).get("error"):
                        self._decide(record, error or result.get("reason_code") or "request_execution_failed", "failed")
                    else:
                        self._decide(record, "completed", "completed")
                    self._save(record)
                    record.pop("execute", None)
                    record.pop("cleanup", None)
                    self._active = None
                    self._condition.notify_all()

    @staticmethod
    def _snapshot_record(record):
        from qlh_core.contracts.snapshots import envelope, reason
        payload = {key: deepcopy(record.get(key)) for key in ("request_id", "attempt_id", "deployment_id", "plan_digest",
                "context_revision", "phase", "execution_state", "terminal", "outcome", "deadline_epoch_ms",
                "revision", "observed_at", "boot_id", "authority", "sampling", "assignments")}
        code = record.get("terminal")
        return envelope(payload, revision=record.get("revision", 0),
            identity={"boot_id": record.get("boot_id", ""), "request_id": record["request_id"], "deployment_id": record.get("deployment_id", "")},
            observed_at=record.get("observed_at", time.time()),
            reason=reason("execution", code, "request_service") if code and code != "completed" else None)

    def snapshot(self):
        with self._condition:
            return {"schema_version": 2, "revision": self._revision, "boot_id": self.boot_id, "paused": self._paused, "capacity": self.capacity,
                    "active": self._active, "queue": list(self._queue),
                    "requests": {key: self._snapshot_record(r) for key, r in self._records.items()}}

    def pause(self):
        with self._condition:
            self._paused = True
            self._revision += 1

    def resume(self):
        with self._condition:
            self._paused = False
            self._revision += 1

    def cancel(self, request_id):
        with self._condition:
            record = self._records.get(request_id)
            if record is None or record["terminal"]:
                return False
            record["cancel"].set()
            self._decide(record, "generation_cancelled", "cancelled")
            return True

    def recent(self, limit=20):
        from qlh_core.contracts.snapshots import envelope
        limit = max(1, min(int(limit), 100))
        with self._condition:
            # Read durable history without starting recovery or native work.
            saved = self.repository.read()
            records = {**saved["requests"], **self._records}
            ordered = sorted(records.values(), key=lambda r: (r.get("observed_at", 0), r["request_id"]), reverse=True)[:limit]
            return envelope({"requests": [self._snapshot_record(r) for r in ordered]}, revision=max(self._revision, saved["revision"]),
                            identity={"boot_id": self.boot_id, "owner": "request_service"}, observed_at=time.time(),
                            validity="pending" if self._persistence_failed else "valid")

    def clear(self):
        with self._condition:
            keys = list(self._queue)
            for key in keys:
                self.cancel(key)
            return len(keys)

    def queue_snapshot(self):
        from qlh_core.contracts.snapshots import envelope, reason
        with self._condition:
            now = time.time()
            tasks = []
            for position, key in enumerate(self._queue, 1):
                record = self._records[key]
                tasks.append({**self._snapshot_record(record), "task_id": key, "position": position,
                              "wait_seconds": max(0, now - record["admitted_at"])})
            active = self._records.get(self._active) if self._active else None
            return envelope({"strategy": "fifo", "paused": self._paused, "capacity": self.capacity,
                             "queue_size": len(tasks), "is_busy": active is not None,
                             "current_task_id": self._active, "active": self._snapshot_record(active) if active else None,
                             "tasks": tasks}, revision=self._revision,
                            identity={"boot_id": self.boot_id, "owner": "request_service"}, observed_at=now,
                            validity="pending" if self._persistence_failed else "valid",
                            reason=reason("request", "request_journal_write_failed", "request_service") if self._persistence_failed else None)

    def close(self, timeout=30):
        with self._condition:
            self._closed = True
            for key, record in self._records.items():
                if not record["terminal"]:
                    record["cancel"].set()
                    self._decide(record, "generation_cancelled", "cancelled")
            self._condition.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("request execution is still draining")
