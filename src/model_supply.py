"""Master-owned, metadata-first segment supply. File availability is not loaded-ready.

Commands use the existing master/worker connection. Workers pull only immutable,
assignment-scoped files from that master; no worker-to-worker transfer is created.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import hmac
import json
import logging
import math
import re
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from atomic_json import atomic_write_json
from pipeline_capacity import solve_pipeline_capacity

logger = logging.getLogger(__name__)

# ★ 2026-10-24（#125）：`_worker_handshake_ready` 的诊断限流表（node_id -> 上次打印
#   的 monotonic 时刻）。握手判据不满足时每 5 秒最多打一条，避免 `waiting_nodes`
#   循环把日志刷爆，同时保留"卡在哪个条件上"的证据。
_HANDSHAKE_DIAG: dict[str, float] = {}


_STRATEGY_REASON_CODES = frozenset({
    "SUPPLY_CACHE_REUSED", "SUPPLY_LOCAL_CUT_PREFERRED", "SUPPLY_DISTRIBUTE_REQUESTED",
    "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "SUPPLY_LOCAL_CUT_TOOL_UNAVAILABLE",
    "SUPPLY_LOCAL_CUT_DISK_INSUFFICIENT", "SUPPLY_LOCAL_CUT_BUDGET_EXCEEDED",
    "SUPPLY_LOCAL_CUT_UNSUPPORTED", "SUPPLY_NODE_CUTTER_UNAVAILABLE",
})


class ModelSupplyError(ValueError):
    def __init__(self, code: str, message: str, *, diagnostic: dict | None = None):
        super().__init__(message)
        self.code, self.message = code, message
        self.diagnostic = copy.deepcopy(diagnostic)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":")).encode()).hexdigest()



def _segment_disk_weight_bytes(descriptor: dict, assignment: dict, source_path: Path | None = None) -> int:
    """Charge source bytes selected by the format's segment builder exactly once.

    ``components.layers`` is the complete decoder aggregate, already represented
    by the per-layer slice. Safetensors excludes visual/MTP/multimodal weights
    and selects boundaries with the same head/tail rules as _selected_keys.
    GGUF uses the builder's header-only tensor selection, retaining non-block
    tensors while excluding decoder-external MTP blocks. Builder checks still
    account for support files and headers before writing.
    """
    start, end = assignment["start_layer"], assignment["end_layer"]
    components = descriptor.get("component_weight_bytes") or {}
    if assignment["engine"] == "llama_cpp":
        from model_supply_assets import gguf_segment_disk_weight_bytes
        if source_path is None:
            raise ModelSupplyError("SUPPLY_SOURCE_MISSING", "GGUF disk budget requires its source header")
        return gguf_segment_disk_weight_bytes(source_path, start, end)
    else:
        tail = end == descriptor["total_layers"]
        keys = ["final_norm"]
        if start == 0 or (tail and descriptor.get("tie_word_embeddings") is True):
            keys.append("embedding")
        if tail:
            keys.extend(("lm_head", "other"))
    return sum(descriptor["layer_weight_bytes"][start:end]) + sum(
        int(components.get(key, 0)) for key in keys
    )


def plan_segment_supply(descriptors: dict[str, dict], nodes: list[dict], *,
                        master_engine: str, alignment: int = 1,
                        safety_margin: float = 1.2,
                        runtime_descriptors: dict[str, dict] | None = None,
                        preferred_ranges: dict[str, list[list[int]]] | None = None,
                        runtime_host_ranges: dict[str, list[list[int]]] | None = None,
                        prefer_all_workers: bool | None = None) -> dict:
    """Reuse capacity admission while planning missing, not just existing, assets.

    GGUF segments retain non-block tensors. Charge those tensors on *each* GGUF
    executor, including the middle, and use aligned groups for hybrid models.
    Each executor uses its exact projected per-layer bytes for the selected
    engine and load specification; packaged size is rechecked before dispatch.
    """
    from model_supply_runtime_capacity import (
        RuntimeCapacityProjectionError, project_supply_capacity_nodes,
    )
    try:
        grouped, projected, by_id = project_supply_capacity_nodes(
            descriptors, nodes, master_engine=master_engine, alignment=alignment,
            safety_margin=safety_margin, runtime_descriptors=runtime_descriptors,
            preferred_ranges=preferred_ranges, runtime_host_ranges=runtime_host_ranges,
        )
    except RuntimeCapacityProjectionError as exc:
        raise ModelSupplyError(exc.code, exc.message) from exc
    if prefer_all_workers is None:
        from config import prefer_all_workers_enabled
        prefer_all_workers = prefer_all_workers_enabled()
    if type(prefer_all_workers) is not bool:
        raise ModelSupplyError("SUPPLY_PLACEMENT_POLICY_INVALID", "prefer_all_workers must be a boolean")
    reference = descriptors.get("llama_cpp") or descriptors.get(master_engine)
    total = int(reference["total_layers"])
    result = solve_pipeline_capacity(grouped, projected, safety_margin=safety_margin,
                                    require_distributed=True, prefer_all_workers=prefer_all_workers,
                                    required_node_id="master")
    if result.get("reason_code") == "pipeline_required_node_unavailable":
        result = {**result, "reason_code": "SUPPLY_MASTER_CAPACITY_INSUFFICIENT",
                  "reason": "主节点当前可用容量不足以执行接力的首段。"}
    if not result.get("admitted"):
        from model_supply_diagnostics import explain_supply_capacity
        diagnostic = explain_supply_capacity(
            result, grouped_descriptor=grouped, projected_nodes=projected,
            alignment=alignment, total_layers=total, safety_margin=safety_margin,
            prefer_all_workers=prefer_all_workers,
        )
        return {**result, "total_layers": total, "loaded_ready": False, "supply_only": True,
                "capacity_diagnostic": diagnostic}
    assignments = []
    for raw in result["assignments"]:
        item = dict(raw)
        item["start_layer"] *= alignment
        item["end_layer"] *= alignment
        item["layers_count"] = item["end_layer"] - item["start_layer"]
        if "range_weight_quote" in item:
            item["range_weight_quote"] = {**item["range_weight_quote"],
                "layer_range": [item["start_layer"], item["end_layer"]]}
        if "layer_ranges" in item:
            item["layer_ranges"] = [[start * alignment, end * alignment] for start, end in item["layer_ranges"]]
        selected_node = by_id[item["node_id"]]
        item["engine"] = selected_node["engine"]
        if selected_node.get("materialization") == "safetensors_layer_offload":
            from model_supply_runtime_capacity import runtime_range_capacity
            from qlh_core.planning.load_spec import assignment_load_spec
            quote = runtime_range_capacity(runtime_descriptors[item["node_id"]],
                item["start_layer"], item["end_layer"],
                has_embedding=item["has_embedding"], has_lm_head=item["has_lm_head"])
            host_required = math.ceil(quote["host_weight_peak_bytes"] * safety_margin) + selected_node["reserve_bytes"]
            host_capacity = selected_node.get("host_capacity_bytes")
            if type(host_capacity) is not int or not 0 < host_required <= host_capacity:
                raise ModelSupplyError("SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT", "Layer offload host budget changed")
            item.update(quote, runtime_materialization="safetensors_layer_offload",
                        host_required_bytes=host_required, host_capacity_bytes=host_capacity,
                        runtime_target_dtype=runtime_descriptors[item["node_id"]]["runtime_target_dtype"],
                        runtime_target_element_size=runtime_descriptors[item["node_id"]]["runtime_target_element_size"])
            spec = assignment_load_spec(item, artifact_format="safetensors")
            item.update(load_spec=spec.to_dict(), load_spec_digest=spec.digest)
        if item["engine"] == "llama_cpp":
            # Native cut artifacts retain all non-block tensors on every node.
            # Keep this header-derived quote for the later prepare admission.
            item["retained_component_bytes"] = by_id[item["node_id"]]["overhead_bytes"]
        item["segment_mode"] = ("head" if item["start_layer"] == 0 else
                                "tail" if item["end_layer"] == total else "middle")
        assignments.append(item)
    if not assignments or assignments[0]["node_id"] != "master":
        raise ModelSupplyError("SUPPLY_MASTER_CAPACITY_INSUFFICIENT", "Master must execute the first segment")
    if any(item["layers_count"] == total for item in assignments):
        raise ModelSupplyError("SUPPLY_FULL_MODEL_FORBIDDEN", "Distributed supply requires proper segments")
    return {**result, "assignments": assignments, "total_layers": total,
            "plan_id": _digest({"source": reference.get("source_model_sha256", reference.get("model_sha256")),
                                "assignments": assignments}),
            "cut_alignment": alignment, "loaded_ready": False, "supply_only": True}


class ModelSupplyCoordinator:
    """One active master operation; immutable per-node file leases are private."""
    TERMINAL = {"available", "failed", "cancelled"}

    def __init__(self, scheduler, root: Path | None = None):
        self.scheduler = scheduler
        self._root = root
        self._condition = threading.Condition(threading.RLock())
        self._operations: dict[str, dict] = {}
        self._events: dict[str, threading.Event] = {}
        self._responses: dict[tuple[str, str, str], dict] = {}
        self._leases: dict[tuple[str, str], dict] = {}
        self._connections: dict[tuple[str, str], object] = {}
        self._active = ""
        self._preparing = ""
        self._prepare_cancelled = False
        self._capacity_queries: dict[str, dict] = {}
        # ★ 2026-10-23（#116）：最近一次准入证据，供工件传输进度续期使用。
        #   传输（可能数十分钟）是流程的正常组成，不应被读成"预算过期"。
        self._capacity_evidence: dict[str, object] = {}

    @property
    def root(self) -> Path:
        if self._root is None:
            from config import STATE_DIR
            self._root = Path(STATE_DIR) / "model-supply" / "operations"
        return self._root

    def cancel_active(self):
        with self._condition:
            op_id = self._active
            # Available is only a file receipt. Opt-out must also revoke the
            # supply->prepare gap and an already-entered prepare guard.
            # Repeated opt-out also retries cleanup after a failed save/abort.
            active = op_id and op_id in self._operations
        if active:
            self.cancel(op_id)

    def _check_master(self):
        if self.scheduler._effective_role() != "master":
            raise ModelSupplyError("SUPPLY_MASTER_REQUIRED", "Submit supply intent to the master")
        if not self.scheduler.get_distributed_inference_enabled():
            raise ModelSupplyError("SUPPLY_LOCAL_MODE", "Node has opted out of distributed inference")

    def create(self, intent: dict, *, wait_for_nodes: bool = False, idempotency_key: str = "") -> dict:
        return self.accept(intent, wait_for_nodes=wait_for_nodes, idempotency_key=idempotency_key)[0]

    def accept(self, intent: dict, *, wait_for_nodes: bool = False, idempotency_key: str = "") -> tuple[dict, bool]:
        """Return the durable receipt and whether this call owns its background work."""
        self._check_master()
        owner = getattr(self.scheduler, "_model_control", None)
        if owner is not None and idempotency_key:
            for prior in owner.snapshot()["supplies"].values():
                if prior.get("idempotency_key") == idempotency_key:
                    if prior["intent"] != intent:
                        raise ModelSupplyError("INTENT_IDEMPOTENCY_CONFLICT", "Idempotency key belongs to another selection")
                    return copy.deepcopy(prior), False
        if str(intent.get("quant_type", "")).lower() in {"fp32", "f32", "float32"}:
            raise ModelSupplyError("SUPPLY_RUNTIME_PRECISION_UNSUPPORTED", "FP32 supply requires per-tensor materialized byte admission")
        if any(re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value or "") is None for value in intent.get("node_ids", [])):
            raise ModelSupplyError("SUPPLY_NODE_ID_INVALID", "Invalid target node identity")
        host = getattr(self.scheduler, "_host", None)
        host_lock = getattr(host, "full_chat_execution_lock", None)
        if host_lock is not None and not host_lock.acquire(blocking=False):
            raise ModelSupplyError("SUPPLY_NODE_BUSY", "Wait for active inference before selecting a new model")
        try:
            with self.scheduler._task_worker_stage_lock:
                if self.scheduler._task_worker_active_attempts or self.scheduler._active_pipeline_task_ids:
                    raise ModelSupplyError("SUPPLY_NODE_BUSY", "Wait for active pipeline work before selecting a new model")
        finally:
            if host_lock is not None:
                host_lock.release()
        with self._condition:
            transaction = getattr(self.scheduler, "_pipeline_load_transaction", None)
            if self._preparing or (isinstance(transaction, dict) and transaction.get("phase") in {
                    "preparing", "committing_local", "committing", "committing_global"}):
                raise ModelSupplyError("SUPPLY_PREPARE_BUSY", "Wait for the selected operation's model transaction")
            if self._active and self._operations[self._active]["status"] not in self.TERMINAL:
                existing = self._operations[self._active]
                if existing["intent"] == intent:
                    return copy.deepcopy(existing), False
                raise ModelSupplyError("SUPPLY_OPERATION_BUSY", "Another model supply operation is active")
            op_id = "supply_" + uuid.uuid4().hex
            operation = {"operation_id": op_id, "intent": copy.deepcopy(intent), "idempotency_key": idempotency_key,
                         "status": "queued", "reason_code": "", "created_at": time.time(),
                         "updated_at": time.time(), "loaded_ready": False, "nodes": {},
                         "plan": None, "capacity_diagnostic": None,
                         "requires_prepare": True, "wait_for_nodes": wait_for_nodes}
            try:
                self._persist(operation)
            except Exception as exc:
                if owner is None or getattr(exc, "code", "") not in {
                        "operation_idempotency_reused", "operation_idempotency_conflict"}:
                    raise
                for prior in owner.snapshot()["supplies"].values():
                    if prior.get("idempotency_key") == idempotency_key:
                        if prior["intent"] != intent:
                            raise ModelSupplyError("INTENT_IDEMPOTENCY_CONFLICT", "Idempotency key belongs to another selection") from exc
                        return copy.deepcopy(prior), False
                raise
            self._operations[op_id] = operation
            self._events[op_id] = threading.Event()
            self._active = op_id
            # Bound in-memory history without deleting user artifacts.
            for old_id in list(self._operations)[:-64]:
                if self._operations[old_id]["status"] in self.TERMINAL:
                    self._operations.pop(old_id, None)
                    self._events.pop(old_id, None)
            threading.Thread(target=self._run, args=(op_id,), name="model-supply", daemon=True).start()
            return copy.deepcopy(operation), True

    def _persist(self, operation):
        # Keep the owner lock through backoff so a delayed write cannot land
        # after a newer update or cancellation.
        with self._condition:
            owner = getattr(self.scheduler, "_model_control", None)
            if owner is not None:
                owner.start()
                owner.save_supply(operation)
                return
            directory = self.root / operation["operation_id"]
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(directory / "operation.json", operation, ensure_ascii=False, indent=2)

    def get(self, op_id: str) -> dict:
        with self._condition:
            if not op_id.startswith("supply_") or len(op_id) != 39 or not all(c in "0123456789abcdef" for c in op_id[7:]):
                raise ModelSupplyError("SUPPLY_OPERATION_NOT_FOUND", "Supply operation not found")
            owner = getattr(self.scheduler, "_model_control", None)
            if owner is not None:
                saved = owner.snapshot()["supplies"].get(op_id)
                if saved is None:
                    raise ModelSupplyError("SUPPLY_OPERATION_NOT_FOUND", "Supply operation not found")
                return copy.deepcopy(saved)
            op = self._operations.get(op_id)
            if op is None:
                path = self.root / op_id / "operation.json"
                if not path.is_file():
                    raise ModelSupplyError("SUPPLY_OPERATION_NOT_FOUND", "Supply operation not found")
                op = json.loads(path.read_text(encoding="utf-8"))
                if op["status"] not in self.TERMINAL:
                    op.update(status="failed", reason_code="SUPPLY_MASTER_RESTARTED")
                return op
            return copy.deepcopy(op)

    def _update(self, op_id, **values):
        with self._condition:
            operation = self._operations[op_id]
            if operation["status"] in {"failed", "cancelled", "available"}:
                return
            if self._events[op_id].is_set() and values.get("status") not in {"cancelled", "failed"}:
                return
            operation.update(values, updated_at=time.time())
            self._persist(self._operations[op_id])
            self._condition.notify_all()

    def _check(self, op_id):
        self._check_master()
        owner = getattr(self.scheduler, "_model_control", None)
        if owner is not None:
            snapshot = owner.snapshot()
            operation = snapshot["operations"].get(op_id)
            if (snapshot["boot_id"] != owner.boot_id
                    or operation and (snapshot["selected_operation"] != op_id
                                      or operation["status"] in {"failed", "cancelled", "superseded"})):
                raise ModelSupplyError("SUPPLY_OPERATION_STALE", "Supply operation was superseded")
        if self._events[op_id].is_set():
            raise ModelSupplyError("SUPPLY_CANCELLED", "Supply operation cancelled")
        if self._active != op_id:
            raise ModelSupplyError("SUPPLY_OPERATION_STALE", "Supply operation was superseded")

    def cancel(self, op_id):
        # Verified assets stay available on disk. Cancellation still fences the
        # pending prepare, including the interval before its transaction exists.
        errors = []
        with self._condition:
            op = self._operations.get(op_id)
            if not op:
                raise ModelSupplyError("SUPPLY_OPERATION_NOT_FOUND", "Supply operation not found")
            if self._preparing == op_id:
                self._prepare_cancelled = True
            terminal = op["status"] in self.TERMINAL
            self._events.setdefault(op_id, threading.Event()).set()
            if not terminal:
                op.update(status="cancelled", reason_code="SUPPLY_CANCELLED",
                          reason="Supply operation cancelled", updated_at=time.time())
                # Preserve a marked historical snapshot for intent deadlines.
                # Current cause/status still decide what consumers may display.
                if isinstance(op.get("capacity_diagnostic"), dict):
                    op["capacity_diagnostic"] = {**op["capacity_diagnostic"], "historical": True}
            # Revocation cannot depend on a successful disk write. Repeated
            # cancellation retries a previously failed save and receiver cleanup.
            for key in list(self._leases):
                if key[0] == op_id:
                    self._leases.pop(key, None)
            self._condition.notify_all()
            if not terminal or op["status"] == "cancelled":
                try:
                    self._persist(op)
                except Exception as exc:
                    errors.append(("persist", exc))
            result = copy.deepcopy(op)
        # Never take the native transaction lock while holding the condition.
        # Attempt both owners even if persistence or the other cleanup fails.
        cancel_prepare = getattr(self.scheduler, "cancel_worker_prepare_operation", None)
        if callable(cancel_prepare):
            try:
                cancel_prepare(op_id)
            except Exception as exc:
                errors.append(("native", exc))
        if result["status"] != "available":
            try:
                self._cancel_receivers(op_id, list(result["nodes"]))
            except Exception as exc:
                errors.append(("receivers", exc))
        for stage, exc in errors:
            logger.error("event=model_supply_cancel_failed operation_id=%s stage=%s reason_code=%s",
                         op_id, stage, getattr(exc, "code", "SUPPLY_CANCEL_FAILED"),
                         exc_info=(type(exc), exc, exc.__traceback__))
        if errors:
            raise errors[0][1]
        return result

    def _cancel_receivers(self, op_id, node_ids):
        self.scheduler._model_supply_worker.cancel(op_id)
        from transport_port import MessageType
        server = self.scheduler._tcp_server
        if server:
            for node in node_ids:
                if node != "master":
                    try:
                        server.send_to_client(node, {"action": "cancel", "operation_id": op_id}, MessageType.MODEL_SUPPLY)
                    except (OSError, ConnectionError):
                        pass

    def handle(self, node_id: str, payload: dict):
        op_id = str(payload.get("operation_id", ""))
        action = payload.get("action")
        if action not in {"inventory_result", "result", "progress"}:
            return
        with self._condition:
            from model_supply_capacity import handle_capacity_inventory
            if handle_capacity_inventory(self, node_id, payload):
                return
            op = self._operations.get(op_id)
            if not op or op["status"] in self.TERMINAL or node_id not in op["nodes"]:
                return
            server = self.scheduler._tcp_server
            info = server.get_client_info(node_id) if server else None
            if not info or payload.get("node_id") != node_id:
                return
            # Connection identity was captured when the request was dispatched.
            expected = self._connections.get((op_id, node_id))
            if expected is not None and info.get("connected_at") != expected:
                return
            if action == "progress":
                allowed = {key: payload[key] for key in (
                    "downloaded_bytes", "actual_downloaded_bytes", "reused_bytes",
                    "resumed_bytes", "total_bytes", "files_completed", "files_total",
                ) if type(payload.get(key)) is int and payload[key] >= 0}
                if payload.get("phase") in {"receiving", "downloading", "building", "verifying", "registering",
                                            "download", "verify", "build"}:
                    allowed["phase"] = payload["phase"]
                if payload.get("strategy") in {"distribute", "local_cut", "reuse"}:
                    allowed["strategy"] = payload["strategy"]
                if payload.get("strategy_reason_code") in _STRATEGY_REASON_CODES:
                    allowed["strategy_reason_code"] = payload["strategy_reason_code"]
                op["nodes"][node_id].update(allowed)
            else:
                self._responses[(op_id, node_id, action)] = copy.deepcopy(payload)
            self._condition.notify_all()

    def _worker_connections(self, selected):
        from scheduler_types import WORKER_HEARTBEAT_MAX_AGE
        with self.scheduler._nodes_lock:
            all_nodes = list(self.scheduler.nodes.items())
        now = time.time()
        server = self.scheduler._tcp_server
        return {node_id: server.get_client_info(node_id) for node_id, node in all_nodes
                if node_id != "master" and node.is_available()
                and node.is_heartbeat_fresh(now, WORKER_HEARTBEAT_MAX_AGE)
                and (not selected or node_id in selected) and server}

    def _worker_handshake_ready(self, node_id, connection):
        # Project the existing control/REGISTER owners, without requiring a model
        # or layer range that this supply operation is about to install.
        server = self.scheduler._tcp_server
        control = getattr(self.scheduler, "_task_worker_control", None)
        if not server or not control or not connection:
            return False
        try:
            epoch = server.get_authenticated_peer_epoch(node_id)
            worker = control.worker_snapshot(node_id)
            ready = bool(
                type(epoch) is int and epoch > 0
                and epoch == connection.get("registration_epoch")
                and worker.get("manual_stage_dispatch_enabled") is True
                # A same-id reconnect may retain the previous accepted hello
                # until the REGISTER callback marks it pending. It is stale.
                and worker.get("hello_received_at", 0) >= connection["connected_at"]
                # The original hello handler resolves pending after sending ACK.
                and node_id not in control.pending_worker_ids()
                and server.get_authenticated_peer_epoch(node_id) == epoch
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return False
        if not ready:
            # ★ 2026-10-24（#125）：供给曾在 `waiting_nodes` 与 `planning` 之间反复
            #   循环（实测 6→58→63→172→177 s），却**没有任何日志**说明卡在六个判据的
            #   哪一个上。这里把各自的实际值打出来（限流 5 秒一次，避免刷屏）。
            now = time.monotonic()
            if now - _HANDSHAKE_DIAG.get(node_id, 0.0) >= 5.0:
                _HANDSHAKE_DIAG[node_id] = now
                try:
                    pending = node_id in control.pending_worker_ids()
                except (AttributeError, TypeError):
                    pending = None
                logger.info(
                    "event=supply_handshake_not_ready node=%s epoch=%r conn_epoch=%r "
                    "manual_stage=%r layer_stage=%r healthy=%r connected=%r accepted=%r "
                    "selected_version=%r pending=%r hello_received_at=%r connected_at=%r",
                    node_id, epoch, connection.get("registration_epoch"),
                    worker.get("manual_stage_dispatch_enabled"),
                    worker.get("layer_stage_dispatch_enabled"),
                    worker.get("healthy"), worker.get("connected"), worker.get("accepted"),
                    worker.get("selected_version"), pending,
                    worker.get("hello_received_at"), connection.get("connected_at"),
                )
        return ready

    def _send(self, op_id, node_id, payload):
        from transport_port import MessageType
        # Cancellation and dispatch share one ordering boundary. A returned
        # cancel cannot be followed by a newly dispatched ensure.
        with self._condition:
            self._check(op_id)
            server = self.scheduler._tcp_server
            info = server.get_client_info(node_id) if server else None
            if not info:
                raise ModelSupplyError("SUPPLY_WORKER_DISCONNECTED", f"Worker {node_id} disconnected")
            key = (op_id, node_id)
            if key in self._connections and self._connections[key] != info.get("connected_at"):
                raise ModelSupplyError("SUPPLY_WORKER_DISCONNECTED", f"Worker {node_id} connection changed")
            if payload.get("action") == "inventory" and not self._worker_handshake_ready(node_id, info):
                raise ModelSupplyError("SUPPLY_WORKERS_UNAVAILABLE", f"Worker {node_id} has not finished joining")
            self._connections.setdefault(key, info.get("connected_at"))
            sent = server.send_to_client(node_id, {**payload, "operation_id": op_id}, MessageType.MODEL_SUPPLY)
            if sent is False:
                raise ModelSupplyError("SUPPLY_WORKER_DISCONNECTED", f"Worker {node_id} send failed")

    def _check_receivers(self, op_id):
        operation = self._operations[op_id]
        for (response_op, response_node, response_action), result in self._responses.items():
            if response_op != op_id or response_action != "result":
                continue
            if result.get("status") != "available":
                raise ModelSupplyError(str(result.get("reason_code") or "SUPPLY_WORKER_FAILED"),
                                       f"Worker {response_node} could not supply its segment")
            target = operation["nodes"].get(response_node, {})
            for field in ("manifest_sha256", "artifact_sha256", "layer_range"):
                if field in target and result.get(field) != target[field]:
                    raise ModelSupplyError("SUPPLY_RECEIPT_MISMATCH", f"Worker {response_node} returned a different segment identity")
            source = operation.get("source_model_sha256")
            if source and result.get("source_model_sha256") != source:
                raise ModelSupplyError("SUPPLY_RECEIPT_MISMATCH", f"Worker {response_node} returned a different source identity")
        server = self.scheduler._tcp_server
        for (connection_op, node_id), connected_at in self._connections.items():
            if connection_op != op_id:
                continue
            info = server.get_client_info(node_id) if server else None
            if not info or info.get("connected_at") != connected_at:
                raise ModelSupplyError("SUPPLY_WORKER_DISCONNECTED", f"Worker {node_id} connection changed")

    def _wait(self, op_id, node_id, action, timeout):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._check(op_id)
                if action in {"inventory_result", "result"}:
                    self._check_receivers(op_id)
                value = self._responses.pop((op_id, node_id, action), None)
                if value is not None:
                    return value
                server = self.scheduler._tcp_server
                if not server or not server.get_client_info(node_id):
                    raise ModelSupplyError("SUPPLY_WORKER_DISCONNECTED", f"Worker {node_id} disconnected")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModelSupplyError("SUPPLY_WORKER_TIMEOUT", f"Worker {node_id} did not return {action}")
                self._condition.wait(min(remaining, 0.5))

    def _extend_auto_inventory(self, op_id, intent, worker_ids, nodes, inventories, deadline, *,
                               wait_for_new=False, capacity_diagnostic=None):
        """Extend only the unpublished automatic pool, using the live join owners."""
        # ★ 2026-10-24（#117）：本函数内的清点等待也用同一上限（见 `_execute` 顶部注释）。
        from config import MODEL_SUPPLY_INVENTORY_TIMEOUT_S
        expanded = False
        next_refresh = time.monotonic() + 1.0
        while True:
            with self._condition:
                self._check(op_id)
                self._check_receivers(op_id)
            connections = self._worker_connections(set())
            ready = [node_id for node_id, info in connections.items()
                     if self._worker_handshake_ready(node_id, info)]
            pending = any(info and node_id not in ready for node_id, info in connections.items())
            added = [node_id for node_id in ready if node_id not in worker_ids]
            if not pending and not added and not wait_for_new:
                return expanded
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if wait_for_new and (capacity_diagnostic is not None or not pending):
                    return expanded
                raise ModelSupplyError("SUPPLY_WORKERS_TIMEOUT", "No joined supply workers before the operation deadline")
            if added and not pending:
                if any(re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", node_id) is None for node_id in added):
                    raise ModelSupplyError("SUPPLY_NODE_ID_INVALID", "Registered worker identity is unsafe for supply")
                # Publish recipients before dispatch so immediate inventory replies
                # pass handle(), and cancellation cleans up every added receiver.
                with self._condition:
                    self._check(op_id)
                    self._check_receivers(op_id)
                    nodes.update({node_id: {"status": "inventory"} for node_id in added})
                    self._update(op_id, status="inventory", nodes=nodes, reason_code="", reason="",
                                 capacity_diagnostic=None)
                for node_id in added:
                    self._send(op_id, node_id, {"action": "inventory", "model_id": intent["model_id"],
                                               "source_model_sha256": ""})
                for node_id in added:
                    # ★ 2026-10-24（#117）：清点等待做成设置（默认 300 s）。此前硬编码
                    #   90 s，而"未登记候选"的本地源核对要对整模做一次全量 sha256
                    #   （5.78 GB，实测 90~120 s）⇒ 首次清点必然 SUPPLY_WORKER_TIMEOUT。
                    # ★ 2026-10-24（#125）：清点是"并行发送、逐个 `_wait`"，故**总耗时
                    #   由最慢节点决定**；此前这一段完全没有日志，导致"清点为什么慢"
                    #   只能靠猜（实测 master 侧自身清点仅约 0.4 s、android 读 2 GiB 不到
                    #   1 s，而 inventory 阶段仍要 108~142 s）。记录每个节点的真实往返。
                    _inventory_started = time.monotonic()
                    inventory = self._wait(op_id, node_id, "inventory_result",
                                           min(MODEL_SUPPLY_INVENTORY_TIMEOUT_S,
                                               max(0, deadline - time.monotonic())))
                    logger.info(
                        "event=supply_inventory_done node=%s elapsed_ms=%d sources=%d",
                        node_id, int((time.monotonic() - _inventory_started) * 1000),
                        len(inventory.get("sources") or []),
                    )
                    if not inventory.get("distributed"):
                        raise ModelSupplyError("SUPPLY_WORKER_LOCAL_MODE", f"Worker {node_id} opted out")
                    with self._condition:
                        self._check(op_id)
                        self._check_receivers(op_id)
                        inventories[node_id] = inventory
                        worker_ids.append(node_id)
                expanded, wait_for_new = True, False
                # Inventory/source inspection can take long enough for another
                # peer to register. Drain new joined peers before planning once.
                continue
            if wait_for_new and not pending and time.monotonic() >= next_refresh:
                # Re-observe existing nodes as well as joins. Capacity may become
                # available without a membership change; inventory owns live RAM,
                # VRAM and local artifact evidence. No build has started here.
                refreshed = {"master": self.scheduler._model_supply_worker.inventory(model_id=intent["model_id"])}
                for node_id in worker_ids:
                    self._send(op_id, node_id, {"action": "inventory", "model_id": intent["model_id"],
                                               "source_model_sha256": ""})
                for node_id in worker_ids:
                    refreshed[node_id] = self._wait(op_id, node_id, "inventory_result",
                                                   min(MODEL_SUPPLY_INVENTORY_TIMEOUT_S,
                                                       max(0, deadline - time.monotonic())))
                    if not refreshed[node_id].get("distributed"):
                        raise ModelSupplyError("SUPPLY_WORKER_LOCAL_MODE", f"Worker {node_id} opted out")
                with self._condition:
                    self._check(op_id)
                    self._check_receivers(op_id)
                fields = ("engines", "pytorch_runtime", "memory_available_bytes", "disk_free_bytes",
                          "sources", "artifacts", "local_cut", "distributed")
                changed = any(any(inventories[node_id].get(key) != value.get(key) for key in fields)
                              for node_id, value in refreshed.items())
                inventories.update(refreshed)
                if changed:
                    return True
                next_refresh = time.monotonic() + 1.0
            if wait_for_new and capacity_diagnostic is not None:
                diagnostic = copy.deepcopy(capacity_diagnostic)
                diagnostic["pending_node_ids"] = sorted(
                    node_id for node_id, info in connections.items() if info and node_id not in ready
                )
                self._update(op_id, status="waiting_nodes",
                             reason_code=diagnostic["reason_code"], reason=diagnostic["reason"],
                             capacity_diagnostic=diagnostic)
            else:
                self._update(op_id, status="waiting_nodes", reason_code="SUPPLY_WAITING_NODES",
                             reason="Waiting for additional distributed workers to finish joining",
                             capacity_diagnostic=None)
            self._events[op_id].wait(min(remaining, 0.25))

    def authorize_file(self, op_id, node_id, filename, token):
        with self._condition:
            lease = self._leases.get((op_id, node_id))
            if not lease or not token or not hmac.compare_digest(lease["token"], token):
                raise ModelSupplyError("SUPPLY_FILE_FORBIDDEN", "Invalid or expired supply file lease")
            self._check(op_id)
            info = self.scheduler._tcp_server.get_client_info(node_id)
            if not info or info.get("connected_at") != self._connections.get((op_id, node_id)):
                raise ModelSupplyError("SUPPLY_FILE_FORBIDDEN", "Worker connection changed")
            path = lease["files"].get(filename)
            if not path or not Path(path).is_file():
                raise ModelSupplyError("SUPPLY_FILE_NOT_FOUND", "File is outside the assigned package")
            return Path(path)

    def _sources(self, op_id, intent, *, want_gguf=True):
        import model_config as mc
        import local_store
        from model_sync import compute_model_sha256, compute_file_sha256
        from pipeline_model_descriptor import inspect_pipeline_model
        from llama_engine import LlamaCppEngine
        from relay_segment_info import read_gguf_layer_info
        model_id = intent["model_id"]
        from model_supply_sources import ensure_registered_sources
        source_result = ensure_registered_sources(
            model_id, operation_dir=self.root / op_id / "source",
            want_gguf=want_gguf, require_cross_format=intent.get("engine") != "llama_cpp",
            cancelled=lambda: self._events[op_id].is_set(),
            progress=lambda value: self._update(op_id, source_progress=value),
        )
        cfg = source_result["model_config"]
        hf, gguf = source_result.get("hf"), source_result.get("gguf")
        proof = source_result.get("conversion_proof")
        descriptors = {}
        if hf and (hf / "config.json").is_file():
            descriptor = inspect_pipeline_model(hf, model_id=model_id)
            descriptor.setdefault("model_sha256", compute_model_sha256(str(hf)))
            descriptors["pytorch"] = descriptor
        if gguf and gguf.is_file():
            info = read_gguf_layer_info(gguf) or {}
            engine = LlamaCppEngine()
            engine._model_path = str(gguf)
            engine.active_model_id = model_id
            descriptor = engine.get_pipeline_descriptor()
            if not descriptor or any(value <= 0 for value in descriptor.get("layer_weight_bytes", [])):
                raise ModelSupplyError("SUPPLY_FULL_SOURCE_REQUIRED", "Supply needs a complete on-disk source, not a pre-cut segment")
            descriptor["source_model_id"] = model_id
            descriptors["llama_cpp"] = descriptor
            from model_supply_assets import _gguf_table
            gguf_metadata, _ = _gguf_table(gguf)
            alignment = int(gguf_metadata.get(str(info.get("architecture")) + ".full_attention_interval") or 1)
        else:
            alignment = 1
        if not descriptors:
            raise ModelSupplyError("SUPPLY_SOURCE_MISSING", "No verified on-disk model source")
        selected = intent.get("engine", "auto")
        master_engine = selected if selected != "auto" else ("pytorch" if "pytorch" in descriptors else "llama_cpp")
        if master_engine not in descriptors:
            raise ModelSupplyError("SUPPLY_ENGINE_SOURCE_MISSING", "Selected source format is absent")
        if "llama_cpp" in descriptors:
            gguf_descriptor = descriptors["llama_cpp"]
            identity = gguf_descriptor["source_model_sha256"]
            tokenizer = gguf_descriptor["tokenizer_sha256"]
            if "pytorch" in descriptors and master_engine == "pytorch":
                py_descriptor = descriptors["pytorch"]
                if py_descriptor.get("source_model_sha256") != identity or py_descriptor.get("tokenizer_sha256") != tokenizer:
                    raise ModelSupplyError("SUPPLY_CONVERSION_PROOF_REQUIRED", "Mixed-engine supply requires a verified cross-format provenance")
        else:
            descriptor = descriptors[master_engine]
            identity = descriptor.get("source_model_sha256", descriptor["model_sha256"])
            tokenizer = descriptor["tokenizer_sha256"]
        return {"hf": hf, "gguf": gguf, "descriptors": descriptors,
                "master_engine": master_engine, "alignment": alignment,
                "source_model_id": model_id, "source_model_sha256": identity,
                "tokenizer_sha256": tokenizer, "conversion_proof": proof}

    def collect_operation_directories(self, *, keep_operation_ids=(), keep_recent=None) -> dict:
        """★ 2026-10-24（#125）：回收历史供给操作目录。

        `self._root` 下每个 `<op_id>/` 都是一次供给为各节点构建的段工件（单次
        3~15 GB），而**此前从不回收**：实测本机积压 **146 个目录、453.78 GB**，
        C 盘一度只剩 108 GB 可用（内存里的 `_operations` 只淘汰记录，不碰磁盘）。

        保留策略（全部保留，其余整目录删除）：
          - 显式给出的 `keep_operation_ids`（正在服务的那次）；
          - 仍登记在内存 `_operations` 中的（尚未终结）；
          - 按 mtime 最新的 `keep_recent` 个，默认取设置
            `MODEL_SUPPLY_KEEP_OPERATIONS`（0 = 清空全部历史）。

        只处理本根目录的**直接子目录**，跳过符号链接；删除失败只记数不抛错，
        因为回收是维护动作，绝不能让一次已成功的供给因此失败。
        """
        import shutil
        from config import MODEL_SUPPLY_KEEP_OPERATIONS
        recent = MODEL_SUPPLY_KEEP_OPERATIONS if keep_recent is None else int(keep_recent)
        keep = {str(value) for value in keep_operation_ids if value}
        with self._condition:
            keep.update(str(key) for key in self._operations)
        try:
            candidates = [path for path in self._root.iterdir() if path.is_dir()]
        except OSError:
            return {"removed": 0, "reclaimed_bytes": 0, "kept": sorted(keep),
                    "examined": 0, "failed": 0}
        try:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            pass
        for path in candidates[:max(0, recent)]:
            keep.add(path.name)
        removed, reclaimed, failed = 0, 0, 0
        for path in candidates:
            if path.name in keep or path.is_symlink() or path.parent != self._root:
                continue
            size = 0
            try:
                for child in path.rglob("*"):
                    if child.is_file() and not child.is_symlink():
                        size += child.stat().st_size
                shutil.rmtree(path)
            except OSError:
                failed += 1
                continue
            removed += 1
            reclaimed += size
        if removed or failed:
            logger.info(
                "event=supply_operations_reclaimed removed=%d reclaimed_bytes=%d "
                "failed=%d kept=%d examined=%d",
                removed, reclaimed, failed, len(keep), len(candidates),
            )
        return {"removed": removed, "reclaimed_bytes": reclaimed, "kept": sorted(keep),
                "examined": len(candidates), "failed": failed}

    def _run(self, op_id):
        try:
            self._execute(op_id)
        except Exception as exc:
            code = str(getattr(exc, "code", "SUPPLY_FAILED"))
            # File paths and one-operation capabilities are not public diagnostics.
            message = str(getattr(exc, "message", "Segment supply failed; inspect local diagnostics"))
            import logging
            logging.getLogger(__name__).exception("event=model_supply_failed operation_id=%s code=%s", op_id, code)
            with self._condition:
                for key in list(self._leases):
                    if key[0] == op_id:
                        self._leases.pop(key, None)
                cancelled = self._events[op_id].is_set()
                self._events[op_id].set()
            owner = getattr(self.scheduler, "_model_control", None)
            if owner is not None and owner.snapshot()["boot_id"] != owner.boot_id:
                return  # A recovered owner has already durably fenced this boot.
            self._update(op_id, status="cancelled" if cancelled else "failed",
                         reason_code="SUPPLY_CANCELLED" if cancelled else code, reason=message,
                         capacity_diagnostic=None if cancelled else getattr(exc, "diagnostic", None))
            self._cancel_receivers(op_id, list(self.get(op_id)["nodes"]))
        finally:
            # ★ 2026-10-24（#125）：**无论成败都回收历史操作目录**。此前每次供给都会在
            #   `operations/<op_id>/` 留下 3~15 GB 的段工件且从不回收（实测积压
            #   146 个 / 453.78 GB）。保留当前这次与设置指定的最近若干次。
            try:
                self.collect_operation_directories(keep_operation_ids=(op_id,))
            except Exception:
                logger.exception("event=supply_operations_reclaim_failed operation_id=%s", op_id)


    def _execute(self, op_id):
        # ★ 2026-10-24（#117）：清点等待上限在此统一取一次，供本函数内所有
        #   `inventory_result` 等待点使用（此前是散落的硬编码 90 秒，且改了一处漏了
        #   另一处 ⇒ 主路径仍超时）。
        from config import MODEL_SUPPLY_INVENTORY_TIMEOUT_S
        self._check(op_id)
        intent = self.get(op_id)["intent"]
        self._update(op_id, status="inspecting")
        selected = set(intent.get("node_ids") or [])
        auto_pool = not selected and self.get(op_id).get("wait_for_nodes") is True
        created_at = self.get(op_id).get("created_at", time.time())
        deadline = time.monotonic() + max(0, created_at + int(intent.get("timeout_seconds", 3600)) - time.time())
        while True:
            self._check(op_id)
            connections = self._worker_connections(selected)
            worker_ids = [node_id for node_id, info in connections.items()
                          if self._worker_handshake_ready(node_id, info)]
            missing = selected - set(worker_ids) - {"master"}
            # Auto selection must wait for connected peers still negotiating,
            # instead of silently planning only the faster PC before Android hello.
            pending = any(info and node_id not in worker_ids for node_id, info in connections.items())
            if worker_ids and not missing and not pending:
                break
            if not self.get(op_id).get("wait_for_nodes"):
                if missing:
                    raise ModelSupplyError("SUPPLY_REQUESTED_NODE_UNAVAILABLE", "A requested node has not finished joining")
                raise ModelSupplyError("SUPPLY_WORKERS_UNAVAILABLE", "No joined supply workers")
            if time.monotonic() >= deadline:
                raise ModelSupplyError("SUPPLY_WORKERS_TIMEOUT", "No joined supply workers before the operation deadline")
            self._update(op_id, status="waiting_nodes", reason_code="SUPPLY_WAITING_NODES",
                         reason="Waiting for distributed workers to finish joining")
            self._events[op_id].wait(0.25)
        self._update(op_id, reason_code="", reason="")
        if any(re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is None for value in worker_ids):
            raise ModelSupplyError("SUPPLY_NODE_ID_INVALID", "Registered worker identity is unsafe for supply")
        if not worker_ids:
            raise ModelSupplyError("SUPPLY_WORKERS_UNAVAILABLE", "No joined supply workers")
        nodes = {node_id: {"status": "inventory"} for node_id in ["master", *worker_ids]}
        self._update(op_id, status="inventory", nodes=nodes)
        # ★ 2026-10-24（#125）：记录**整个清点阶段**的耗时。实测存在一段 45 s 的
        #   静默空档（最后一个 worker 的 inventory_done 之后、planning 之前），
        #   期间 supply 内部没有任何日志，无法判断是二次观察、wait_for_new 轮询
        #   还是别的等待。这里给出阶段总耗时，配合每节点的
        #   `supply_inventory_done` 即可算出空档归属。
        _inventory_phase_started = time.monotonic()
        for node_id in worker_ids:
            self._send(op_id, node_id, {"action": "inventory", "model_id": intent["model_id"],
                                     "source_model_sha256": ""})
        inventories = {"master": self.scheduler._model_supply_worker.inventory(model_id=intent["model_id"])}
        # ★ 2026-10-24（#117）：这条**主路径**此前也硬编码 90 秒，而"未登记候选"的
        #   本地源核对要对整模做一次全量 sha256（5.78 GB，实测 90~120 s）⇒ 首次清点
        #   必然 SUPPLY_WORKER_TIMEOUT。与上面那处一起用同一设置（函数顶部已取）。
        for node_id in worker_ids:
            # ★ 2026-10-24（#125）：**这才是主路径**（上面 L582 那处是另一条分支）。
            #   清点是「并行发送、逐个 `_wait`」，故总耗时 = 最慢节点，此前完全没有
            #   日志可查（实测 master 自身约 0.4 s、android 读 2 GiB 不到 1 s，而整个
            #   inventory 阶段仍要 108~142 s）。逐节点记录往返耗时以定位瓶颈。
            _inventory_started = time.monotonic()
            inventories[node_id] = self._wait(op_id, node_id, "inventory_result",
                                              MODEL_SUPPLY_INVENTORY_TIMEOUT_S)
            logger.info(
                "event=supply_inventory_done node=%s elapsed_ms=%d sources=%d",
                node_id, int((time.monotonic() - _inventory_started) * 1000),
                len(inventories[node_id].get("sources") or []),
            )
            if not inventories[node_id].get("distributed"):
                raise ModelSupplyError("SUPPLY_WORKER_LOCAL_MODE", f"Worker {node_id} opted out")
        logger.info(
            "event=supply_inventory_phase_done nodes=%d elapsed_ms=%d",
            len(inventories), int((time.monotonic() - _inventory_phase_started) * 1000),
        )
        want_gguf = intent.get("engine") != "pytorch" or any(
            "pytorch" not in inventories[node_id].get("engines", []) for node_id in worker_ids)
        sources = self._sources(op_id, intent, want_gguf=want_gguf)
        while True:
            if auto_pool:
                self._extend_auto_inventory(op_id, intent, worker_ids, nodes, inventories, deadline)
                needs_gguf = intent.get("engine") != "pytorch" or any(
                    "pytorch" not in inventories[node_id].get("engines", []) for node_id in worker_ids)
                if needs_gguf and not want_gguf:
                    # The original source owner supplies conversion/provenance;
                    # a late Android never changes the operation's source identity.
                    extended = self._sources(op_id, intent, want_gguf=True)
                    with self._condition:
                        self._check(op_id)
                        self._check_receivers(op_id)
                    if any(extended[key] != sources[key] for key in (
                            "source_model_id", "source_model_sha256", "tokenizer_sha256")):
                        raise ModelSupplyError("SUPPLY_SOURCE_CONTRACT_MISMATCH", "Source identity changed while workers joined")
                    sources, want_gguf = extended, True
                    continue
            with self._condition:
                self._check(op_id)
                self._check_receivers(op_id)
            self._update(op_id, status="planning", source_model_sha256=sources["source_model_sha256"],
                         tokenizer_sha256=sources["tokenizer_sha256"], reason_code="", reason="",
                         capacity_diagnostic=None)
            capacity = self.scheduler._get_pipeline_capacity_nodes(set(worker_ids) | {"master"})
            capacity_by_id = {record["node_id"]: record for record in capacity}
            records = []
            runtime_descriptors = {}
            runtime_host_descriptors = {}
            for node_id, inventory in inventories.items():
                base = dict(capacity_by_id.get(node_id) or {})
                if not base:
                    from config import PIPELINE_CAPACITY_RESERVE_MB
                    base = {"node_id": node_id, "role": "client", "score": 0.0,
                            "reserve_bytes": int(PIPELINE_CAPACITY_RESERVE_MB * 1024**2)}
                engines = inventory.get("engines") or []
                engine = sources["master_engine"] if node_id == "master" else (
                    "llama_cpp" if "llama_cpp" in engines and "llama_cpp" in sources["descriptors"] else
                    "pytorch" if "pytorch" in engines and "pytorch" in sources["descriptors"] else "")
                if not engine:
                    raise ModelSupplyError("SUPPLY_ENGINE_SOURCE_MISSING", f"No compatible source for {node_id}")
                if engine == "llama_cpp":
                    # ★ 2026-10-23（#117）：**自裁节点按原生量化宽度物化**。GGUF 段是
                    #   从整模按原生 Q4_K_M 直接切出来的，worker 侧没有 fp16→fp32 的
                    #   投影，所以不该乘 2。此前对 llama_cpp 一律硬编码
                    #   `runtime_multiplier=2.0`，把自裁 worker 的可用内存凭空减半
                    #   ⇒ 三机只凑得出 18 层 < 32 ⇒ 求解器恒无解、`waiting_nodes`
                    #   反复空转。非自裁路径保留既有倍率，避免改动无关行为。
                    multiplier = 1.0 if inventory.get("local_cut") else 2.0
                    base.update(capacity_bytes=int(inventory.get("memory_available_bytes", 0)),
                                capacity_source="supply.ram.available_bytes_live", execution_device="cpu", runtime_multiplier=multiplier)
                if engine == "pytorch":
                    from model_supply_runtime_capacity import project_runtime_descriptor
                    runtime = inventory.get("pytorch_runtime")
                    if not isinstance(runtime, dict):
                        raise ModelSupplyError("SUPPLY_RUNTIME_BUDGET_MISSING", f"Worker {node_id} has no selected PyTorch runtime budget")
                    device = str(runtime.get("execution_device", ""))
                    element_size = runtime.get("target_element_size")
                    available = runtime.get("available_bytes")
                    if (device not in {"cpu", "cuda"} or type(available) is not int or available <= 0):
                        raise ModelSupplyError("SUPPLY_RUNTIME_BUDGET_MISSING", f"Worker {node_id} has an invalid runtime memory budget")
                    projection = project_runtime_descriptor(sources["hf"], sources["descriptors"]["pytorch"], element_size,
                        target_dtype={2: "float16", 4: "float32"}.get(element_size))
                    if device == "cuda":
                        # Selective loaders read Safetensors on CPU and transfer one
                        # tensor at a time. Charge target weights to VRAM, source
                        # staging separately to RAM, never RAM bytes to VRAM.
                        runtime_host_descriptors[node_id] = projection
                        projection = {**projection,
                            "weight_bytes": projection["runtime_weight_bytes"],
                            "layer_weight_bytes": projection["runtime_layer_weight_bytes"],
                            "component_weight_bytes": projection["runtime_component_weight_bytes"]}
                    runtime_descriptors[node_id] = projection
                    from model_supply_capacity import runtime_capacity_source
                    base.update(capacity_bytes=available, execution_device=device, runtime_multiplier=1.0,
                                capacity_source=runtime_capacity_source(runtime, device, prefix="supply"))
                    if device == "cuda" and isinstance(runtime.get("budget_observations"), dict):
                        base["capacity_observations"] = copy.deepcopy(runtime["budget_observations"])
                base["engine"] = engine
                # Source resolution may select a different engine from the
                # previous execution. Score its actual device and fresh RAM.
                node_lock = getattr(self.scheduler, "_nodes_lock", None)
                if node_lock is not None:
                    with node_lock:
                        observed_node = self.scheduler.nodes.get(node_id)
                        profile = getattr(observed_node, "device_info", None)
                    if isinstance(profile, dict):
                        from scheduler_layer_plan import compute_node_weight
                        profile = copy.deepcopy(profile)
                        profile["ram"] = {**(profile.get("ram") or {}),
                            "available_gb": max(0, int(inventory.get("memory_available_bytes", 0))) / 1024**3}
                        base["score"] = compute_node_weight(profile, execution_device=base["execution_device"])
                logger.info("event=supply_runtime_budget operation=%s node=%s engine=%s device=%s capacity_bytes=%s reserve_bytes=%s multiplier=%s score=%s",
                            op_id, node_id, engine, base.get("execution_device"), base.get("capacity_bytes"),
                            base.get("reserve_bytes"), base.get("runtime_multiplier"), base.get("score"))
                if engine == "pytorch" and device == "cuda":
                    # ★ 2026-10-23（#117）：**主机内存按物理总量封顶**（默认 60%）。
                    from config import HOST_RAM_MAX_FRACTION
                    #   主节点还要跑别的程序，不能把"当前可用内存"整块当成可分配容量
                    #   —— 此前实测一次供给吃掉 ~14 GB / 31.6 GB（45%）。上限取
                    #   `min(当前可用, 物理总量 × HOST_RAM_MAX_FRACTION)`，其余留给
                    #   系统与其他程序；总量未知时退回可用内存（保持旧行为不更差）。
                    available_ram = int(inventory.get("memory_available_bytes", 0))
                    total_ram = int(inventory.get("memory_total_bytes", 0) or 0)
                    if total_ram > 0:
                        allowed = int(total_ram * HOST_RAM_MAX_FRACTION)
                        base["host_capacity_bytes"] = max(0, min(available_ram, allowed))
                        base["host_capacity_source"] = "host_ram_total_x_fraction"
                    else:
                        base["host_capacity_bytes"] = max(0, available_ram)
                    logger.info(
                        "event=host_ram_budget operation=%s node=%s available_gb=%.1f "
                        "total_gb=%.1f fraction=%.2f host_capacity_gb=%.1f",
                        op_id, node_id, available_ram / 1024**3, total_ram / 1024**3,
                        HOST_RAM_MAX_FRACTION, base["host_capacity_bytes"] / 1024**3,
                    )
                records.append(base)
            from config import PIPELINE_CAPACITY_SAFETY_MARGIN, API_PORT
            from model_supply_host_capacity import allowed_host_layer_ranges
            runtime_host_ranges = {}
            for node_id, projection in runtime_host_descriptors.items():
                runtime_host_ranges[node_id] = allowed_host_layer_ranges(
                    projection, node_id=node_id,
                    available_bytes=int(inventories[node_id].get("memory_available_bytes", 0)),
                    reserve_bytes=int(capacity_by_id.get(node_id, {}).get("reserve_bytes", 0)),
                    safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN, alignment=sources["alignment"],
                )
                if not runtime_host_ranges[node_id]:
                    from model_supply_diagnostics import explain_host_capacity
                    diagnostic = explain_host_capacity(
                        projection, node_id=node_id,
                        available_bytes=int(inventories[node_id].get("memory_available_bytes", 0)),
                        reserve_bytes=int(capacity_by_id.get(node_id, {}).get("reserve_bytes", 0)),
                        safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN, alignment=sources["alignment"],
                    )
                    raise ModelSupplyError(diagnostic["reason_code"], diagnostic["reason"],
                                           diagnostic=diagnostic)
            from model_supply_plan_reuse import plan_segment_supply_reuse
            def capacity_plan(*, preferred_ranges, constrain_host=True, plan_records=None):
                return plan_segment_supply(sources["descriptors"], plan_records or records,
                    master_engine=sources["master_engine"], alignment=sources["alignment"],
                    safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN,
                    runtime_descriptors=runtime_descriptors, preferred_ranges=preferred_ranges,
                    runtime_host_ranges=runtime_host_ranges if constrain_host else None)
            plan = plan_segment_supply_reuse(sources["descriptors"], capacity, records,
                planner=capacity_plan, source_model_id=sources["source_model_id"],
                source_model_sha256=sources["source_model_sha256"], tokenizer_sha256=sources["tokenizer_sha256"],
                engine_by_node={record["node_id"]:record["engine"] for record in records},
                alignment=sources["alignment"])
            # ★ 2026-10-23（#117）：**本机 RAM 是一等承载资源，不是失败兜底**。
            #   此前只在 resident 规划**失败之后**才尝试 `safetensors_layer_offload`
            #   （注释原文 "Prefer resident execution when feasible"），而 resident
            #   几乎总能"成功"，于是本机那 ~32 GB 主机内存全程闲置、层段被整体压给
            #   远端 —— 实测 master 只拿 `[0,8)`，android 被迫扛 `[8,32)` 共 24 层。
            #   这里把两种物化**并列求解**，再按"本机承担更多层"选优。
            # ★ 2026-10-24（#117）：**但 offload 会显著拖慢推理** —— 权重放在主机内存
            #   时每步前向都要跨 PCIe 搬运，实测 master 以 offload 扛 28 层时每 token
            #   约 5 秒（master 成了串行瓶颈），而 17-18 点 resident（master 只占 8 层）
            #   的三机明显更快。所以是否启用由 `CAPACITY_PLANNING_MODE` 决定：
            #   `balanced` 允许（资源优先），`fastest` 关闭（速度优先）。
            local_runtime = inventories["master"].get("pytorch_runtime") or {}
            local_projection = runtime_descriptors.get("master") or {}
            # 运行时读（节点级 `capacity_planning_mode` 在 config 之后才写入 env）。
            from config import host_ram_offload_enabled
            if (host_ram_offload_enabled()
                    and sources["master_engine"] == "pytorch"
                    and local_runtime.get("execution_device") == "cuda"
                    and "safetensors_layer_offload" in local_runtime.get("materializations", [])
                    and local_projection.get("model_type") in {"qwen2", "qwen3", "qwen3_5", "qwen3_5_text"}):
                resident_plan = plan
                resident_diagnostic = copy.deepcopy(plan.get("capacity_diagnostic"))
                offload_records = [
                    {**record, "materialization": "safetensors_layer_offload"}
                    if record["node_id"] == "master" else record
                    for record in records
                ]
                offload_plan = plan_segment_supply_reuse(
                    sources["descriptors"], capacity, offload_records,
                    planner=lambda *, preferred_ranges, constrain_host=True: capacity_plan(
                        preferred_ranges=preferred_ranges, constrain_host=constrain_host,
                        plan_records=offload_records),
                    source_model_id=sources["source_model_id"],
                    source_model_sha256=sources["source_model_sha256"],
                    tokenizer_sha256=sources["tokenizer_sha256"],
                    engine_by_node={record["node_id"]: record["engine"] for record in offload_records},
                    alignment=sources["alignment"])

                def _local_layers(candidate):
                    """本机承担的层数；越大越优（本机算力最强）。"""
                    total = 0
                    for item in candidate.get("assignments", []) or []:
                        if item.get("node_id") == "master":
                            total += max(0, int(item.get("end_layer", 0)) - int(item.get("start_layer", 0)))
                    return total

                if offload_plan.get("admitted") and (
                        not plan.get("admitted")
                        or _local_layers(offload_plan) > _local_layers(resident_plan)):
                    plan = offload_plan
                    plan["resident_capacity_diagnostic"] = resident_diagnostic
                    logger.info(
                        "本机 RAM 参与承载: materialization=safetensors_layer_offload "
                        "resident_local_layers=%s offload_local_layers=%s",
                        _local_layers(resident_plan), _local_layers(offload_plan),
                    )
                elif resident_plan.get("admitted"):
                    plan = resident_plan
                    if offload_plan is not None:
                        plan["offload_capacity_diagnostic"] = copy.deepcopy(
                            offload_plan.get("capacity_diagnostic"))
                else:
                    # 两种物化都不可行：保留 offload 的诊断（它把主机内存计入，
                    # 更接近"真正差多少"的真相）。
                    plan = offload_plan
                    plan["resident_capacity_diagnostic"] = resident_diagnostic
            if not plan.get("admitted"):
                diagnostic = plan["capacity_diagnostic"]
                # A host range is a staging constraint, not an absent artifact.
                # Prove its effect with the same runtime solver before naming it.
                if runtime_host_ranges:
                    unconstrained = capacity_plan(preferred_ranges={}, constrain_host=False)
                    if unconstrained.get("admitted"):
                        from model_supply_diagnostics import explain_host_capacity
                        shortages = []
                        for assignment in unconstrained["assignments"]:
                            node_id = assignment["node_id"]
                            if node_id not in runtime_host_descriptors:
                                continue
                            explanation = explain_host_capacity(
                                runtime_host_descriptors[node_id], node_id=node_id,
                                available_bytes=int(inventories[node_id].get("memory_available_bytes", 0)),
                                reserve_bytes=int(capacity_by_id.get(node_id, {}).get("reserve_bytes", 0)),
                                safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN, alignment=sources["alignment"],
                                layer_range=[assignment["start_layer"], assignment["end_layer"]],
                            )
                            shortages.extend(item for item in explanation["nodes"] if item.get("deficit_bytes", 0) > 0)
                        if shortages:
                            diagnostic = {**explanation, "nodes": shortages,
                                          "origin_reason_code": plan.get("reason_code", ""),
                                          "origin_reason": plan.get("reason", ""),
                                          "reason": "当前节点的显存可形成接力计划，但主机 RAM 不足以暂存该候选计划的源权重。"}
                code = str(diagnostic["reason_code"])
                if (auto_pool and self._extend_auto_inventory(
                        op_id, intent, worker_ids, nodes, inventories, deadline,
                        wait_for_new=True, capacity_diagnostic=diagnostic)):
                    continue
                raise ModelSupplyError(code, diagnostic["reason"], diagnostic=diagnostic)
            if auto_pool and self._extend_auto_inventory(op_id, intent, worker_ids, nodes, inventories, deadline):
                continue
            # CUDA still needs host source mappings. Gate the selected segment's
            # complete source bytes against real RAM plus the configured reserve.
            for assignment in plan["assignments"]:
                node_id = assignment["node_id"]
                projection = runtime_host_descriptors.get(node_id)
                if projection is None:
                    continue
                from model_supply_runtime_capacity import runtime_range_capacity
                from qlh_core.planning.load_spec import assignment_load_spec
                materialization = assignment.get("runtime_materialization", "safetensors_key_filtered")
                quote = runtime_range_capacity(
                    projection, assignment["start_layer"], assignment["end_layer"],
                    has_embedding=assignment["has_embedding"], has_lm_head=assignment["has_lm_head"],
                    materialization=materialization,
                )
                source_bytes = quote["host_source_bytes"]
                conversion_bytes = quote["host_conversion_bytes"]
                reserve = int(capacity_by_id.get(node_id, {}).get("reserve_bytes", 0))
                required = math.ceil((source_bytes + conversion_bytes) * PIPELINE_CAPACITY_SAFETY_MARGIN) + reserve
                if required > int(inventories[node_id].get("memory_available_bytes", 0)):
                    from model_supply_diagnostics import explain_host_capacity
                    diagnostic = explain_host_capacity(
                        projection, node_id=node_id,
                        available_bytes=int(inventories[node_id].get("memory_available_bytes", 0)),
                        reserve_bytes=reserve, safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN,
                        alignment=sources["alignment"],
                        layer_range=[assignment["start_layer"], assignment["end_layer"]],
                    )
                    raise ModelSupplyError(diagnostic["reason_code"], diagnostic["reason"],
                                           diagnostic=diagnostic)
                # Freeze the exact runtime quote as well as the source identity.
                # Fresh admission must retain target-dtype costs for resident
                # CUDA too, rather than reverting to source artifact precision.
                assignment.update(quote, host_source_bytes=source_bytes,
                                  host_conversion_bytes=conversion_bytes,
                                  host_required_bytes=required,
                                  host_capacity_bytes=int(inventories[node_id].get("memory_available_bytes", 0)),
                                  runtime_materialization=materialization,
                                  runtime_target_element_size=projection["runtime_target_element_size"],
                                  runtime_target_dtype=projection["runtime_target_dtype"])
                spec = assignment_load_spec(assignment, artifact_format="safetensors")
                assignment.update(load_spec=spec.to_dict(), load_spec_digest=spec.digest)
            reference = sources["descriptors"].get("llama_cpp") or sources["descriptors"][sources["master_engine"]]
            plan["plan_id"] = _digest({
                "source": reference.get("source_model_sha256", reference.get("model_sha256")),
                "assignments": plan["assignments"],
            })
            break
        with self._condition:
            self._check(op_id)
            self._check_receivers(op_id)
            assigned = {item["node_id"] for item in plan["assignments"]}
            candidate_node_ids = sorted(nodes)
            nodes = {node_id: value for node_id, value in nodes.items() if node_id in assigned}
            # Inventory candidates are observations. Once placement is published,
            # only its executors own connection/receipt obligations.
            for key in list(self._connections):
                if key[0] == op_id and key[1] not in assigned:
                    self._connections.pop(key)
            self._update(op_id, status="building", plan=plan, nodes=nodes,
                         candidate_node_ids=candidate_node_ids)
        from model_supply_assets import build_gguf_segment, build_safetensors_segment, segment_cache_hint
        pending_receipts = []
        for assignment in plan["assignments"]:
            self._check(op_id)
            node_id = assignment["node_id"]
            directory = self.root / op_id / node_id
            directory.mkdir(parents=True, exist_ok=True)
            builder = build_gguf_segment if assignment["engine"] == "llama_cpp" else build_safetensors_segment
            src = sources["gguf"] if assignment["engine"] == "llama_cpp" else sources["hf"]
            # Disk budget covers the temporary segment + final publish. Source
            # is already on disk; a missing source is never assumed free.
            disk_descriptor = sources["descriptors"][assignment["engine"]]
            estimated = _segment_disk_weight_bytes(disk_descriptor, assignment, src)
            free = shutil.disk_usage(directory).free
            identity = {"node_id": node_id, "source_model_id": sources["source_model_id"],
                "source_model_sha256": sources["source_model_sha256"],
                "tokenizer_sha256": sources["tokenizer_sha256"], "total_layers": plan["total_layers"],
                "layer_range": [assignment["start_layer"], assignment["end_layer"]],
                "segment_mode": "head" if assignment["start_layer"] == 0 else
                                "tail" if assignment["end_layer"] == plan["total_layers"] else "middle",
                "format": "gguf" if assignment["engine"] == "llama_cpp" else "safetensors",
                "conversion_proof": sources["conversion_proof"]}
            if free < max(estimated * 2, 64 * 1024**2) and not segment_cache_hint(self.root, identity):
                raise ModelSupplyError("SUPPLY_MASTER_DISK_INSUFFICIENT", "Master staging disk is insufficient")
            build_options = ({"cancelled": lambda: self._events[op_id].is_set(), "deadline": deadline}
                             if assignment["engine"] == "llama_cpp" else {})
            manifest = builder(src, directory, source_model_id=sources["source_model_id"],
                               source_model_sha256=sources["source_model_sha256"],
                               tokenizer_sha256=sources["tokenizer_sha256"],
                               start_layer=assignment["start_layer"], end_layer=assignment["end_layer"],
                               total_layers=plan["total_layers"], operation_id=op_id, node_id=node_id,
                               conversion_proof=sources["conversion_proof"], reuse_root=self.root, **build_options)
            self._check(op_id)
            reused_bytes = manifest.pop("_reused_bytes", 0)
            files = manifest.pop("local_files")
            total_bytes = sum(int(item["size_bytes"]) for item in manifest["files"])
            with self._condition:
                nodes[node_id] = {"status": "supplying", "layer_range": manifest["layer_range"],
                                  "segment_mode": manifest["segment_mode"], "total_bytes": total_bytes,
                                  "manifest_sha256": manifest["manifest_sha256"], "artifact_sha256": manifest["artifact_sha256"],
                                  "transfer_granularity": "segment_tensors", "engine": assignment["engine"]}
                self._update(op_id, nodes=nodes, status="supplying")
            if node_id == "master":
                self._check(op_id)
                result = self.scheduler._model_supply_worker.activate_local(
                    manifest, directory, cancelled=lambda: self._events[op_id].is_set(),
                )
                if result.get("status") != "available":
                    raise ModelSupplyError(str(result.get("reason_code") or "SUPPLY_LOCAL_REGISTRATION_FAILED"), "Master could not register its segment")
                with self._condition:
                    nodes[node_id].update(status="available", strategy="reuse" if reused_bytes else "local_build",
                                          actual_downloaded_bytes=0, reused_bytes=reused_bytes)
                    self._update(op_id, nodes=nodes, master_model_path=str(result["model_path"]))
                continue
            available_sources = inventories[node_id].get("sources") or []
            # ★ 2026-10-23（#117）：**"未声明 tokenizer" ≠ "tokenizer 不匹配"**。
            #   源身份的核心是权重摘要 `source_model_sha256`，tokenizer 摘要是附加
            #   校验。此前用 `entry.get("tokenizer_sha256", default)` 判等，当 worker
            #   上报的键**存在但值为 None/""**（实测安卓那份整模 GGUF：
            #   `read_gguf_tokenizer_sha256` 读不到元数据 ⇒ tokenizer_sha256=None）
            #   时不会回退到默认值 ⇒ 被判"无匹配本地源" ⇒ 明明握着整模却只能从网络
            #   以 ~0.67 MB/s 拉 4 层（实测卡 `bytes=0` 超过 10 分钟）。
            #   现在：源没给出 tokenizer 就不据此否决它。
            matching_source = any(
                entry.get("source_model_sha256") == sources["source_model_sha256"]
                and (not entry.get("tokenizer_sha256")
                     or entry.get("tokenizer_sha256") == sources["tokenizer_sha256"])
                for entry in available_sources)
            can_cut = assignment["engine"] == "llama_cpp" and inventories[node_id].get("local_cut") and matching_source
            requested = intent.get("strategy", "auto")
            if requested == "local_cut" and not can_cut:
                raise ModelSupplyError("SUPPLY_LOCAL_CUT_UNAVAILABLE", f"Worker {node_id} lacks the verified source or cutter")
            strategy = "local_cut" if can_cut and requested != "distribute" else "distribute"
            reason = ("SUPPLY_LOCAL_CUT_PREFERRED" if strategy == "local_cut" else
                      "SUPPLY_DISTRIBUTE_REQUESTED" if requested == "distribute" else
                      "SUPPLY_LOCAL_SOURCE_UNAVAILABLE" if not matching_source else
                      "SUPPLY_NODE_CUTTER_UNAVAILABLE")
            # ★ 2026-10-24（#124）：**复用显式参与决策**。此前 master 只在
            #   local_cut / distribute 间二选一，从不告知节点"复用你已有的段"：
            #   Python worker 好在收到 local_cut 时会先试 `reuse_segment`，而 Kotlin
            #   worker（android）的 local_cut 路径**没有任何复用检查** ⇒ 每轮白裁约
            #   3 GB（实测 ~105 s），即使它磁盘上已有身份完全一致的同段缓存
            #   （已用两轮 marker 逐字段 diff 证明仅 manifest_sha256/operation_id 两个
            #   收据字段不同，而这两者不在接收端 `sameContentIdentity` 判据内）。
            #
            #   接收端的复用能力挂在 distribute 路径上（`ModelSupplyReceiver.receive()`
            #   在下载前会先按 `sameContentIdentity` 扫已发布段并 hardlink），所以这里
            #   把命中的节点改判为 distribute ⇒ 零网络传输地复用已有段。
            #
            #   只在此处**确认**层段与三项摘要全等时才切换：若复用意外未命中，
            #   distribute 会退化为真下载 3 GB（比裁切更慢），故判据从严。
            cached_hit = False
            if strategy == "local_cut" and requested == "auto":
                try:
                    control = getattr(self.scheduler, "_task_worker_control", None)
                    snapshot = (control.worker_snapshot(node_id) or {}) if control else {}
                    # 快照可能把 capabilities 嵌在 `capabilities` 下，也可能就是它本身。
                    caps = snapshot.get("capabilities") or snapshot
                except (AttributeError, KeyError, TypeError, ValueError):
                    caps = {}
                want_range = list(manifest.get("layer_range") or [])
                for artifact in caps.get("layer_artifacts") or []:
                    if (isinstance(artifact, dict)
                            and list(artifact.get("layer_range") or []) == want_range
                            and artifact.get("artifact_sha256") == manifest.get("artifact_sha256")
                            and artifact.get("source_model_id") == manifest.get("source_model_id")
                            and artifact.get("source_model_sha256") == manifest.get("source_model_sha256")
                            and artifact.get("tokenizer_sha256") == manifest.get("tokenizer_sha256")):
                        cached_hit = True
                        break
            if cached_hit:
                strategy, reason = "distribute", "SUPPLY_CACHE_REUSED"
                logger.info(
                    "event=supply_cache_reuse node=%s layer_range=%s artifact=%s "
                    "reason=%s（改走接收端复用路径，避免 local_cut 重裁）",
                    node_id, want_range, str(manifest.get("artifact_sha256"))[:12], reason,
                )
            elif strategy == "local_cut" and requested == "auto":
                # 未命中时把两侧事实打出来，避免"看起来该复用却没复用"无处定位。
                logger.info(
                    "event=supply_cache_miss node=%s layer_range=%s artifact=%s "
                    "advertised=%d artifact_shas=%s ranges=%s",
                    node_id, want_range, str(manifest.get("artifact_sha256"))[:12],
                    len(caps.get("layer_artifacts") or []),
                    [str(item.get("artifact_sha256"))[:12]
                     for item in (caps.get("layer_artifacts") or []) if isinstance(item, dict)],
                    [item.get("layer_range")
                     for item in (caps.get("layer_artifacts") or []) if isinstance(item, dict)],
                )
            # Do not attempt a new local cut when its staging reserve is already
            # known infeasible; the receiver can still reuse its cached files.
            if (strategy == "local_cut" and requested == "auto" and
                    int(inventories[node_id].get("disk_free_bytes", 0)) < total_bytes + 64 * 1024**2):
                strategy, reason = "distribute", "SUPPLY_LOCAL_CUT_DISK_INSUFFICIENT"
            token = secrets.token_urlsafe(32)
            with self._condition:
                self._leases[(op_id, node_id)] = {"token": token, "files": files}
            with self._condition:
                nodes[node_id].update(strategy=strategy, strategy_reason_code=reason)
            self._send(op_id, node_id, {"action": "ensure", "node_id": node_id, "strategy": strategy,
                                     "manifest": manifest, "api_port": int(API_PORT), "token": token,
                                     "allow_fallback": requested == "auto", "strategy_reason_code": reason,
                                     "local_cut_timeout_seconds": max(1, min(300, int((deadline - time.monotonic()) / 3)))})
            pending_receipts.append((node_id, manifest, total_bytes, strategy, time.monotonic()))
        # All receivers pull from the master concurrently. File availability
        # is published only after every assignment returns its exact identity.
        for node_id, manifest, total_bytes, strategy, dispatched_at in pending_receipts:
            remaining = max(0, int(intent.get("timeout_seconds", 3600)) - (time.monotonic() - dispatched_at))
            result = self._wait(op_id, node_id, "result", remaining)
            if result.get("status") != "available":
                raise ModelSupplyError(str(result.get("reason_code", "SUPPLY_WORKER_FAILED")), f"Worker {node_id} could not supply its segment")
            for field in ("manifest_sha256", "artifact_sha256", "layer_range", "source_model_sha256"):
                if result.get(field) != manifest.get(field):
                    raise ModelSupplyError("SUPPLY_RECEIPT_MISMATCH", f"Worker {node_id} returned a different segment identity")
            with self._condition:
                nodes[node_id].update(status="available", downloaded_bytes=total_bytes if strategy == "distribute" else 0)
                if result.get("strategy") in {"distribute", "local_cut", "reuse"}:
                    nodes[node_id]["strategy"] = result["strategy"]
                if result.get("strategy_reason_code") in _STRATEGY_REASON_CODES:
                    nodes[node_id]["strategy_reason_code"] = result["strategy_reason_code"]
                for counter in ("actual_downloaded_bytes", "reused_bytes", "resumed_bytes"):
                    if type(result.get(counter)) is int and result[counter] >= 0:
                        nodes[node_id][counter] = result[counter]
                self._leases.pop((op_id, node_id), None)
                self._update(op_id, nodes=nodes)
        self._check(op_id)
        self._collect_garbage(op_id)
        self._check(op_id)
        self._update(op_id, status="available", nodes=nodes, reason_code="SUPPLY_ASSETS_AVAILABLE",
                     loaded_ready=False, requires_prepare=True)

    def _collect_garbage(self, op_id):
        collector = getattr(self.scheduler._model_supply_worker, "collect_garbage", None)
        if not callable(collector):
            return
        # Snapshot under the condition; live SHA can take minutes on slow
        # storage and must leave status, progress and cancellation responsive.
        # The receiver holds native owner admission throughout collection.
        with self._condition:
            protected = [self.root / op_id]
            preparing = self._preparing
            event = self._events[op_id]
            if preparing:
                protected.append(self.root / preparing)
            leased = frozenset(path for lease in self._leases.values()
                               for path in lease["files"].values())
            protected.extend(Path(path) for path in leased)

        def invalidated():
            # New selection/prepare/lease ownership invalidates this snapshot.
            # The collector checks this during hashing and before deletion.
            with self._condition:
                return (event.is_set() or self._active != op_id
                        or self._preparing != preparing
                        or any(path not in leased for lease in self._leases.values()
                               for path in lease["files"].values()))

        report = collector(root=self.root, protected_paths=protected, cancelled=invalidated)
        self._update(op_id, retention=report)

    @contextmanager
    def prepare_guard(self, op_id):
        # Bind the available selection until the existing candidate/transaction
        # entry has accepted it. Loaded-ready ACK ownership remains with 12c.
        with self._condition:
            if self._preparing:
                raise ModelSupplyError("SUPPLY_PREPARE_BUSY", "Another supplied model preparation is entering its transaction")
            self._preparing = op_id
            self._prepare_cancelled = False
        try:
            source = self.prepared_source(op_id)
            yield source
        finally:
            with self._condition:
                if self._preparing == op_id:
                    self._preparing = ""
                    self._prepare_cancelled = False

    def ensure_prepare_not_cancelled(self, op_id):
        with self._condition:
            event = self._events.get(op_id)
            # A live guard owns its attempt. Keep the operation event as a fence
            # after that guard exits, when an abandoned native thread may return.
            cancelled = self._prepare_cancelled if self._preparing == op_id else bool(event and event.is_set())
            if cancelled:
                raise ModelSupplyError("SUPPLY_PREPARE_CANCELLED", "Model preparation cancelled by user")
            if self._active and self._active != op_id:
                raise ModelSupplyError("SUPPLY_OPERATION_STALE", "Model preparation was superseded")
        self._check_master()

    def prepare_capacity_evidence(self, op_id):
        from model_supply_capacity import prepare_capacity_evidence
        return prepare_capacity_evidence(self, op_id)

    def register_capacity_evidence(self, operation_id: str, evidence) -> None:
        """★ 2026-10-23（#116）：保留活动证据，使传输进展可以给它续期。"""
        self._capacity_evidence[operation_id] = evidence
        if len(self._capacity_evidence) > 32:
            for stale in list(self._capacity_evidence)[:-32]:
                self._capacity_evidence.pop(stale, None)

    def touch_capacity_evidence(self, operation_id: str) -> None:
        """传输有实际进展就续期；停滞超过窗口仍会被准入判为过期。"""
        evidence = self._capacity_evidence.get(operation_id)
        if evidence is not None:
            evidence.touch()

    def discard_capacity_evidence(self, operation_id: str) -> None:
        self._capacity_evidence.pop(operation_id, None)

    def prepared_source(self, op_id) -> dict:
        self._check_master()
        with self._condition:
            op = self.get(op_id)
            if (op["status"] != "available" or self._active != op_id
                    or self._events[op_id].is_set()):
                raise ModelSupplyError("SUPPLY_ASSETS_NOT_AVAILABLE", "Current supply operation is not available")
        path = self.scheduler._model_supply_worker.selected_model_path()
        if str(path) != op.get("master_model_path"):
            raise ModelSupplyError("SUPPLY_OPERATION_STALE", "Master selected asset has changed")
        return {"model_id": op["intent"]["model_id"], "model_path": str(path),
                "engine": op["plan"]["assignments"][0]["engine"],
                "quant_type": op["intent"].get("quant_type") or ("gguf" if op["plan"]["assignments"][0]["engine"] == "llama_cpp" else "fp16")}
