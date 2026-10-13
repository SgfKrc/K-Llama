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

from pipeline_capacity import solve_pipeline_capacity

logger = logging.getLogger(__name__)


class ModelSupplyError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


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
                        runtime_host_ranges: dict[str, list[list[int]]] | None = None) -> dict:
    """Reuse capacity admission while planning missing, not just existing, assets.

    GGUF segments retain non-block tensors. Charge those tensors on *each* GGUF
    executor, including the middle, and use aligned groups for hybrid models.
    Mixed precision nodes are charged a conservative per-group ratio; the actual
    packaged size is rechecked before dispatch.
    """
    reference = descriptors.get("llama_cpp") or descriptors.get(master_engine)
    if not reference or not reference.get("pipeline_runtime_supported"):
        raise ModelSupplyError("SUPPLY_DESCRIPTOR_UNAVAILABLE", "No supported source descriptor")
    total = int(reference["total_layers"])
    if isinstance(alignment, bool) or alignment < 1 or total % alignment:
        raise ModelSupplyError("SUPPLY_ALIGNMENT_INVALID", "Invalid decoder cut alignment")
    group_sizes = [sum(reference["layer_weight_bytes"][i:i + alignment])
                   for i in range(0, total, alignment)]
    projected = []
    by_id = {}
    for raw in nodes:
        node = dict(raw)
        node_id = str(node["node_id"])
        engine = master_engine if node_id == "master" else str(node.get("engine", "llama_cpp"))
        source = (runtime_descriptors or {}).get(node_id) or descriptors.get(engine)
        if not source:
            raise ModelSupplyError("SUPPLY_ENGINE_SOURCE_MISSING", f"No source for {engine}")
        if int(source["total_layers"]) != total or int(source["hidden_size"]) != int(reference["hidden_size"]):
            raise ModelSupplyError("SUPPLY_SOURCE_CONTRACT_MISMATCH", "Source dimensions differ")
        sizes = [sum(source["layer_weight_bytes"][i:i + alignment])
                 for i in range(0, total, alignment)]
        if any(value <= 0 for value in sizes + group_sizes):
            raise ModelSupplyError("SUPPLY_LAYER_BYTES_MISSING", "Source layer bytes are incomplete")
        multiplier = float(node.get("runtime_multiplier", 1.0))
        components = source.get("component_weight_bytes") or {}
        # GGUF cut files retain every non-block tensor. Selective PyTorch
        # loading charges the head's input components and each node's norm;
        # remote PyTorch nodes conservatively reserve both possible boundaries.
        boundary_keys = (("embedding", "final_norm", "lm_head", "other")
                         if engine == "llama_cpp" or node_id != "master" else
                         ("embedding", "final_norm", "other"))
        overhead = sum(int(components.get(key, 0)) for key in boundary_keys)
        # A target plan may create a new segment. Inventory ranges constrain
        # executable plans, never this supply plan. No execution advert is changed.
        for key in ("layer_ranges", "layer_artifacts", "layer_budget", "segment_mode", "models"):
            node.pop(key, None)
        wanted_ranges = (preferred_ranges or {}).get(node_id)
        if wanted_ranges is not None:
            if (node_id == "master" or not isinstance(wanted_ranges, list) or not wanted_ranges
                    or any(not isinstance(span, list) or len(span) != 2
                           or any(type(value) is not int for value in span)
                           or not 0 <= span[0] < span[1] <= total
                           or span[1] - span[0] == total
                           or span[0] % alignment or span[1] % alignment
                           for span in wanted_ranges)):
                raise ModelSupplyError("SUPPLY_PREFERRED_RANGE_INVALID", "Invalid reusable segment preference")
            node["layer_ranges"] = [[span[0] // alignment, span[1] // alignment] for span in wanted_ranges]
        host_ranges = (runtime_host_ranges or {}).get(node_id)
        if host_ranges is not None:
            if (not isinstance(host_ranges, list)
                    or any(not isinstance(span, list) or len(span) != 2
                           or any(type(value) is not int for value in span)
                           or not 0 <= span[0] < span[1] <= total
                           or span[1] - span[0] == total
                           or span[0] % alignment or span[1] % alignment for span in host_ranges)):
                raise ModelSupplyError("SUPPLY_RUNTIME_HOST_BUDGET_INVALID", "Invalid source staging ranges")
            admitted_ranges = [[span[0] // alignment, span[1] // alignment] for span in host_ranges]
            if "layer_ranges" in node:
                admitted_ranges = [span for span in admitted_ranges if span in node["layer_ranges"]]
            node["layer_ranges"] = admitted_ranges
        node["role"] = "master" if node_id == "master" else "client"
        node["reserve_bytes"] = int(node.get("reserve_bytes", 0)) + math.ceil(overhead * multiplier * safety_margin)
        node["runtime_multiplier"] = multiplier * max(a / b for a, b in zip(sizes, group_sizes))
        node["engine"] = engine
        by_id[node_id] = {**raw, "engine": engine, "overhead_bytes": overhead}
        projected.append(node)
    grouped = {**reference, "total_layers": len(group_sizes),
               "layer_weight_bytes": group_sizes,
               "component_weight_bytes": {"embedding": 0, "final_norm": 0, "lm_head": 0, "other": 0},
               "runtime_ignored_components": []}
    result = solve_pipeline_capacity(grouped, projected, safety_margin=safety_margin,
                                    require_distributed=True, prefer_all_workers=True)
    if not result.get("admitted"):
        return {**result, "total_layers": total, "loaded_ready": False, "supply_only": True}
    assignments = []
    for raw in result["assignments"]:
        item = dict(raw)
        item["start_layer"] *= alignment
        item["end_layer"] *= alignment
        item["layers_count"] = item["end_layer"] - item["start_layer"]
        if "layer_ranges" in item:
            item["layer_ranges"] = [[start * alignment, end * alignment] for start, end in item["layer_ranges"]]
        item["engine"] = by_id[item["node_id"]]["engine"]
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

    @property
    def root(self) -> Path:
        if self._root is None:
            from config import STATE_DIR
            self._root = Path(STATE_DIR) / "model-supply" / "operations"
        return self._root

    def cancel_active(self):
        with self._condition:
            op_id = self._active
            active = op_id and self._operations[op_id]["status"] not in self.TERMINAL
        if active:
            self.cancel(op_id)

    def _check_master(self):
        if self.scheduler._effective_role() != "master":
            raise ModelSupplyError("SUPPLY_MASTER_REQUIRED", "Submit supply intent to the master")
        if not self.scheduler.get_distributed_inference_enabled():
            raise ModelSupplyError("SUPPLY_LOCAL_MODE", "Node has opted out of distributed inference")

    def create(self, intent: dict) -> dict:
        self._check_master()
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
                    return copy.deepcopy(existing)
                raise ModelSupplyError("SUPPLY_OPERATION_BUSY", "Another model supply operation is active")
            op_id = "supply_" + uuid.uuid4().hex
            operation = {"operation_id": op_id, "intent": copy.deepcopy(intent),
                         "status": "queued", "reason_code": "", "created_at": time.time(),
                         "updated_at": time.time(), "loaded_ready": False, "nodes": {},
                         "plan": None, "requires_prepare": True}
            self._operations[op_id] = operation
            self._events[op_id] = threading.Event()
            self._active = op_id
            self._persist(operation)
            # Bound in-memory history without deleting user artifacts.
            for old_id in list(self._operations)[:-64]:
                if self._operations[old_id]["status"] in self.TERMINAL:
                    self._operations.pop(old_id, None)
                    self._events.pop(old_id, None)
            threading.Thread(target=self._run, args=(op_id,), name="model-supply", daemon=True).start()
            return copy.deepcopy(operation)

    def _persist(self, operation):
        directory = self.root / operation["operation_id"]
        directory.mkdir(parents=True, exist_ok=True)
        temp = directory / "operation.json.tmp"
        temp.write_text(json.dumps(operation, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(directory / "operation.json")

    def get(self, op_id: str) -> dict:
        with self._condition:
            op = self._operations.get(op_id)
            if op is None:
                if not op_id.startswith("supply_") or len(op_id) != 39 or not all(c in "0123456789abcdef" for c in op_id[7:]):
                    raise ModelSupplyError("SUPPLY_OPERATION_NOT_FOUND", "Supply operation not found")
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
        if self._events[op_id].is_set():
            raise ModelSupplyError("SUPPLY_CANCELLED", "Supply operation cancelled")
        if self._active != op_id:
            raise ModelSupplyError("SUPPLY_OPERATION_STALE", "Supply operation was superseded")

    def cancel(self, op_id):
        # Verified assets stay available on disk. Cancellation still fences the
        # pending prepare, including the interval before its transaction exists.
        with self._condition:
            op = self._operations.get(op_id)
            if not op:
                raise ModelSupplyError("SUPPLY_OPERATION_NOT_FOUND", "Supply operation not found")
            if self._preparing == op_id:
                self._prepare_cancelled = True
            terminal = op["status"] in self.TERMINAL
            if not terminal:
                self._events[op_id].set()
                op.update(status="cancelled", reason_code="SUPPLY_CANCELLED", updated_at=time.time())
                self._persist(op)
                for key in list(self._leases):
                    if key[0] == op_id:
                        self._leases.pop(key, None)
                self._condition.notify_all()
            result = copy.deepcopy(op)
        # Do not hold the supply condition while acquiring the transaction lock.
        cancel_prepare = getattr(self.scheduler, "cancel_worker_prepare_operation", None)
        if callable(cancel_prepare):
            cancel_prepare(op_id)
        if not terminal:
            self._cancel_receivers(op_id, list(result["nodes"]))
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
                if payload.get("phase") in {"receiving", "building", "verifying", "registering",
                                            "download", "verify", "build"}:
                    allowed["phase"] = payload["phase"]
                op["nodes"][node_id].update(allowed)
            else:
                self._responses[(op_id, node_id, action)] = copy.deepcopy(payload)
            self._condition.notify_all()

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
            self._connections[(op_id, node_id)] = info.get("connected_at")
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
                if action == "result":
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
            want_gguf=want_gguf, cancelled=lambda: self._events[op_id].is_set(),
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
            self._update(op_id, status="cancelled" if cancelled else "failed",
                         reason_code="SUPPLY_CANCELLED" if cancelled else code, reason=message)
            self._cancel_receivers(op_id, list(self.get(op_id)["nodes"]))

    def _execute(self, op_id):
        self._check(op_id)
        intent = self.get(op_id)["intent"]
        self._update(op_id, status="inspecting")
        with self.scheduler._nodes_lock:
            all_nodes = list(self.scheduler.nodes.items())
        selected = set(intent.get("node_ids") or [])
        from scheduler_types import WORKER_HEARTBEAT_MAX_AGE
        now = time.time()
        worker_ids = [node_id for node_id, node in all_nodes
                      if node_id != "master" and node.is_available()
                      and node.is_heartbeat_fresh(now, WORKER_HEARTBEAT_MAX_AGE)
                      and (not selected or node_id in selected)
                      and self.scheduler._tcp_server and self.scheduler._tcp_server.get_client_info(node_id)]
        if selected and selected - set(worker_ids) - {"master"}:
            raise ModelSupplyError("SUPPLY_REQUESTED_NODE_UNAVAILABLE", "A requested node is disconnected")
        if any(re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is None for value in worker_ids):
            raise ModelSupplyError("SUPPLY_NODE_ID_INVALID", "Registered worker identity is unsafe for supply")
        if not worker_ids:
            raise ModelSupplyError("SUPPLY_WORKERS_UNAVAILABLE", "No connected supply workers")
        nodes = {node_id: {"status": "inventory"} for node_id in ["master", *worker_ids]}
        self._update(op_id, status="inventory", nodes=nodes)
        for node_id in worker_ids:
            self._send(op_id, node_id, {"action": "inventory", "model_id": intent["model_id"],
                                     "source_model_sha256": ""})
        inventories = {"master": self.scheduler._model_supply_worker.inventory(model_id=intent["model_id"])}
        for node_id in worker_ids:
            inventories[node_id] = self._wait(op_id, node_id, "inventory_result", 90)
            if not inventories[node_id].get("distributed"):
                raise ModelSupplyError("SUPPLY_WORKER_LOCAL_MODE", f"Worker {node_id} opted out")
        want_gguf = intent.get("engine") != "pytorch" or any(
            "pytorch" not in inventories[node_id].get("engines", []) for node_id in worker_ids)
        sources = self._sources(op_id, intent, want_gguf=want_gguf)
        self._check(op_id)
        self._update(op_id, status="planning", source_model_sha256=sources["source_model_sha256"],
                     tokenizer_sha256=sources["tokenizer_sha256"])
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
                base.update(capacity_bytes=int(inventory.get("memory_available_bytes", 0)),
                            capacity_source="supply.ram.available_bytes_live", execution_device="cpu", runtime_multiplier=2.0)
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
                projection = project_runtime_descriptor(sources["hf"], sources["descriptors"]["pytorch"], element_size)
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
                base.update(capacity_bytes=available, execution_device=device, runtime_multiplier=1.0,
                            capacity_source="supply.cuda.mem_get_info" if device == "cuda" else "supply.ram.available_bytes_live")
            base["engine"] = engine
            logger.info("event=supply_runtime_budget operation=%s node=%s engine=%s device=%s capacity_bytes=%s reserve_bytes=%s multiplier=%s score=%s",
                        op_id, node_id, engine, base.get("execution_device"), base.get("capacity_bytes"),
                        base.get("reserve_bytes"), base.get("runtime_multiplier"), base.get("score"))
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
                raise ModelSupplyError("SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT", f"Worker {node_id} cannot stage a proper source segment in RAM")
        from model_supply_plan_reuse import plan_segment_supply_reuse
        def capacity_plan(*, preferred_ranges):
            return plan_segment_supply(sources["descriptors"], records,
                master_engine=sources["master_engine"], alignment=sources["alignment"],
                safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN,
                runtime_descriptors=runtime_descriptors, preferred_ranges=preferred_ranges,
                runtime_host_ranges=runtime_host_ranges)
        plan = plan_segment_supply_reuse(sources["descriptors"], capacity, records,
            planner=capacity_plan, source_model_id=sources["source_model_id"],
            source_model_sha256=sources["source_model_sha256"], tokenizer_sha256=sources["tokenizer_sha256"],
            engine_by_node={record["node_id"]:record["engine"] for record in records},
            alignment=sources["alignment"])
        if not plan.get("admitted"):
            raise ModelSupplyError(str(plan.get("reason_code", "SUPPLY_CAPACITY_INSUFFICIENT")), "No feasible segment supply plan")
        # CUDA still needs host source mappings. Gate the selected segment's
        # complete source bytes against real RAM plus the configured reserve.
        for assignment in plan["assignments"]:
            node_id = assignment["node_id"]
            projection = runtime_host_descriptors.get(node_id)
            if projection is None:
                continue
            components = projection["source_component_weight_bytes"]
            keys = ("embedding", "final_norm", "other") if node_id == "master" else ("embedding", "final_norm", "lm_head", "other")
            source_bytes = sum(projection["source_layer_weight_bytes"][assignment["start_layer"]:assignment["end_layer"]])
            source_bytes += sum(int(components.get(key, 0)) for key in keys)
            reserve = int(capacity_by_id.get(node_id, {}).get("reserve_bytes", 0))
            required = math.ceil(source_bytes * PIPELINE_CAPACITY_SAFETY_MARGIN) + reserve
            if required > int(inventories[node_id].get("memory_available_bytes", 0)):
                raise ModelSupplyError("SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT", f"Worker {node_id} cannot stage its selected source tensors in RAM")
            assignment.update(host_source_bytes=source_bytes, host_required_bytes=required,
                              runtime_target_element_size=projection["runtime_target_element_size"])
        self._update(op_id, status="building", plan=plan)
        from model_supply_assets import build_gguf_segment, build_safetensors_segment
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
            if free < max(estimated * 2, 64 * 1024**2):
                raise ModelSupplyError("SUPPLY_MASTER_DISK_INSUFFICIENT", "Master staging disk is insufficient")
            manifest = builder(src, directory, source_model_id=sources["source_model_id"],
                               source_model_sha256=sources["source_model_sha256"],
                               tokenizer_sha256=sources["tokenizer_sha256"],
                               start_layer=assignment["start_layer"], end_layer=assignment["end_layer"],
                               total_layers=plan["total_layers"], operation_id=op_id, node_id=node_id,
                               conversion_proof=sources["conversion_proof"])
            self._check(op_id)
            files = manifest.pop("local_files")
            total_bytes = sum(int(item["size_bytes"]) for item in manifest["files"])
            if int(inventories[node_id].get("disk_free_bytes", 0)) < total_bytes * 2 + 64 * 1024**2:
                raise ModelSupplyError("SUPPLY_WORKER_DISK_INSUFFICIENT", f"Worker {node_id} staging disk is insufficient")
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
                    nodes[node_id].update(status="available", strategy="local_build")
                    self._update(op_id, nodes=nodes, master_model_path=str(result["model_path"]))
                continue
            available_sources = inventories[node_id].get("sources") or []
            can_cut = assignment["engine"] == "llama_cpp" and inventories[node_id].get("local_cut") and any(
                entry.get("source_model_sha256") == sources["source_model_sha256"] for entry in available_sources)
            requested = intent.get("strategy", "auto")
            if requested == "local_cut" and not can_cut:
                raise ModelSupplyError("SUPPLY_LOCAL_CUT_UNAVAILABLE", f"Worker {node_id} lacks the verified source or cutter")
            strategy = "local_cut" if can_cut and requested != "distribute" else "distribute"
            token = secrets.token_urlsafe(32)
            with self._condition:
                self._leases[(op_id, node_id)] = {"token": token, "files": files}
            with self._condition:
                nodes[node_id]["strategy"] = strategy
            self._send(op_id, node_id, {"action": "ensure", "node_id": node_id, "strategy": strategy,
                                     "manifest": manifest, "api_port": int(API_PORT), "token": token})
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
                for counter in ("actual_downloaded_bytes", "reused_bytes", "resumed_bytes"):
                    if type(result.get(counter)) is int and result[counter] >= 0:
                        nodes[node_id][counter] = result[counter]
                self._leases.pop((op_id, node_id), None)
                self._update(op_id, nodes=nodes)
        self._check(op_id)
        self._update(op_id, status="available", nodes=nodes, reason_code="SUPPLY_ASSETS_AVAILABLE",
                     loaded_ready=False, requires_prepare=True)

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
            if self._preparing == op_id and self._prepare_cancelled:
                raise ModelSupplyError("SUPPLY_PREPARE_CANCELLED", "Model preparation cancelled by user")

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
