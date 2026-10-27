"""PC segment preparation: asynchronous, connection-fenced runtime ownership."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from worker_prepare_contract import (
    WorkerPrepareError, stage_identity, validate_ack, validate_command,
    validate_identity, validate_stage_identity,
)

logger = logging.getLogger(__name__)


@dataclass
class LayerRuntime:
    identity: dict | None
    selected: dict | None = None
    upstream: object | None = None
    manager: object | None = None
    workflow: object | None = None
    caches: dict = field(default_factory=dict)

    @property
    def owned(self):
        # Even a dead child or a partially closed manager still needs retirement.
        return self.upstream is not None or self.manager is not None or bool(self.caches)

    @property
    def loaded(self):
        if self.upstream is not None:
            return bool(self.upstream.is_available)
        return self.manager is not None and bool(getattr(self.manager, "is_loaded", getattr(self.manager, "model", None) is not None))


def close_layer_runtime(runtime: LayerRuntime | None) -> None:
    """Do not wait on the shim's unbounded close-response readline."""
    if runtime is None:
        return
    upstream = runtime.upstream
    if upstream is not None:
        process = getattr(upstream, "_worker", None)
        if process is not None:
            if process.poll() is None:
                try:
                    if process.stdin is not None and not process.stdin.closed:
                        process.stdin.write('{"op":"close"}\n')
                        process.stdin.flush()
                except (OSError, BrokenPipeError):
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
            if process.poll() is None:
                raise WorkerPrepareError("worker_prepare_close_failed")
        upstream.close()
        if process is not None:
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    pipe.close()
        runtime.upstream = None
    # Drop KV before unload_model's GC/device reclamation.
    runtime.caches.clear()
    runtime.workflow = None
    if runtime.manager is not None:
        runtime.manager.unload_model()
        if getattr(runtime.manager, "is_loaded", getattr(runtime.manager, "model", None) is not None):
            raise WorkerPrepareError("worker_prepare_close_failed")
        runtime.manager = None


def resolve_prepare_selection(scheduler, identity: dict) -> dict:
    """Bind the verified supply registration, or an explicitly verified env crop."""
    from model_supply_worker import get_selected_model
    worker = getattr(scheduler, "_model_supply_worker", None)
    selected = get_selected_model(getattr(worker, "root", None))
    if identity["manifest_sha256"]:
        if selected is None:
            raise WorkerPrepareError("worker_prepare_registration_unavailable")
        manifest = selected["manifest"]
        for key in (
            "operation_id", "node_id", "artifact_sha256", "manifest_sha256",
            "source_model_id", "source_model_sha256", "tokenizer_sha256",
            "layer_range", "total_layers", "hidden_size", "segment_mode",
        ):
            if manifest.get(key) != identity[key]:
                raise WorkerPrepareError("worker_prepare_manifest_mismatch")
        if manifest["format"] != identity["stage_model_identity"]["format"]:
            raise WorkerPrepareError("worker_prepare_model_mismatch")
    else:
        if identity["stage_model_identity"]["format"] != "gguf":
            raise WorkerPrepareError("worker_prepare_manifest_required")
        # A nonempty supply identity must never fall back to an env asset.
        path = Path(os.environ.get("QLH_LAYER_GGUF", "").strip())
        if not path.is_file():
            raise WorkerPrepareError("worker_prepare_artifact_unavailable")
        marker = path.with_suffix(".manifest.json")
        if not marker.is_file():
            marker = Path(str(path) + ".manifest.json")
        try:
            native = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise WorkerPrepareError("worker_prepare_manifest_required") from None
        for key in ("source_model_id", "source_model_sha256", "tokenizer_sha256", "hidden_size", "artifact_sha256"):
            if native.get(key) != identity[key]:
                raise WorkerPrepareError("worker_prepare_manifest_mismatch")
        if (native.get("source_layer_range") != identity["layer_range"]
                or native.get("n_layer") != identity["total_layers"]
                or native.get("mode") != identity["segment_mode"]):
            raise WorkerPrepareError("worker_prepare_manifest_mismatch")
        from model_supply_assets import _gguf_layers, _gguf_table, _hash_file
        from model_supply_digest_cache import cached_sha256
        from relay_segment_info import read_gguf_tokenizer_sha256
        if (cached_sha256(path, hash_file=_hash_file) != identity["artifact_sha256"]
                or read_gguf_tokenizer_sha256(path) != identity["tokenizer_sha256"]):
            raise WorkerPrepareError("worker_prepare_artifact_mismatch")
        metadata, _ = _gguf_table(path)
        _, layers, hidden = _gguf_layers(metadata)
        if layers != identity["layer_range"][1] - identity["layer_range"][0] or hidden != identity["hidden_size"]:
            raise WorkerPrepareError("worker_prepare_layer_range_mismatch")
        selected = {"model_path": str(path.resolve()), "directory": str(path.parent.resolve()),
                    "manifest": {**deepcopy(identity), "format": "gguf"},
                    "manifest_sha256": "", "artifact_sha256": identity["artifact_sha256"]}
    artifact = scheduler._configured_layer_artifact()
    if scheduler._task_worker_layer_artifact_identity(artifact) != identity["stage_model_identity"]:
        raise WorkerPrepareError("worker_prepare_model_mismatch")
    return deepcopy(selected)


@dataclass(eq=False)
class _Retirement:
    epoch: int
    done: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    unloaded: bool = False
    resolve_runtime: bool = False


@dataclass(eq=False)
class _Command:
    payload: dict
    client: object
    registration_epoch: object
    connection_generation: object
    epoch: int
    cancelled: threading.Event = field(default_factory=threading.Event)
    retirement: _Retirement | None = None


class WorkerPrepareController:
    """One serialized command worker; admission and invalidation never load weights."""

    def __init__(self, scheduler, runtime_factory=None):
        self.scheduler = scheduler
        self._runtime_factory = runtime_factory
        self._runtime = None
        self._lock = threading.RLock()
        self._queue = deque()
        self._runner = None
        self._current = None
        self._epoch = 0
        self._closed = False
        self._managed = False
        self._candidate = None
        self._candidate_phase = "idle"
        self._prepare_command = None
        self._active_identity = None
        self._previous_identity = None
        self._active_scope = None
        self._latest_prepare = None
        self._retired = {}
        self._pending_retirements = {}
        self._stage_fenced = True
        self._cleanup_retirement = None
        self._retire_waiters = 0

    def configure_runtime(self, factory):
        if not callable(factory):
            raise TypeError("worker prepare runtime factory must be callable")
        with self._lock:
            if self._runtime is not None:
                raise RuntimeError("worker prepare runtime is already bound")
            self._runtime_factory = factory

    @property
    def managed(self):
        with self._lock:
            return self._managed

    def _scope(self, payload, client):
        return _Command(payload, client, getattr(client, "registration_epoch", None),
                        getattr(client, "_connection_generation", None), self._epoch)

    def _valid(self, command, *, check_mode=True):
        sched, client = self.scheduler, command.client
        if (command.epoch != self._epoch or self._closed
                or getattr(sched, "_tcp_client", None) is not client or client is None
                or not getattr(client, "is_registered", False)
                or getattr(client, "registration_epoch", None) != command.registration_epoch
                or getattr(client, "_connection_generation", None) != command.connection_generation
                or sched._effective_role() != "client"):
            return False
        if check_mode and not sched.get_distributed_inference_enabled():
            return False
        return True

    def handle(self, payload, client=None):
        """Validate and enqueue; return without waiting for native work."""
        if not isinstance(payload, dict):
            return False
        data = deepcopy(payload)
        try:
            data = validate_command(data)
        except WorkerPrepareError:
            return False
        with self._lock:
            retirement = self._cleanup_retirement
            if data["action"] == "prepare" and (self._retire_waiters or (retirement is not None and not retirement.done.is_set())):
                return False
            captured = client if client is not None else getattr(self.scheduler, "_tcp_client", None)
            command = self._scope(data, captured)
            if not self._valid(command) or data["identity"]["node_id"] != self.scheduler.get_effective_node_id():
                return False
            if len(self._queue) >= 64:
                return False
            self._managed = True
            if data["action"] == "prepare":
                latest = self._latest_prepare
                generation = data["identity"]["generation"]
                if latest is None or generation > latest["generation"]:
                    self._latest_prepare = deepcopy(data["identity"])
                    self._stage_fenced = True
                    if self._runtime is not None:
                        self._runtime.fence_worker_runtime()
                    if self._current is not None and self._current.payload is not None and self._current.payload["action"] == "prepare":
                        self._current.cancelled.set()
                # Lower generations are rejected in the worker and receive an ACK.
            elif data["action"] in {"abort", "release"}:
                if self._current is not None and self._current.payload is not None and self._current.payload["identity"] == data["identity"] and self._current.payload["action"] == "prepare":
                    self._current.cancelled.set()
            self._queue.append(command)
            self._start_locked()
        return True

    def _start_locked(self):
        if self._runner is None or not self._runner.is_alive():
            self._runner = threading.Thread(target=self._run, name="worker-model-prepare", daemon=True)
            self._runner.start()

    def _get_runtime(self):
        if self._runtime is None:
            if self._runtime_factory is None:
                raise WorkerPrepareError("worker_prepare_executor_unavailable")
            runtime = self._runtime_factory()
            if runtime is None:
                raise WorkerPrepareError("worker_prepare_executor_unavailable")
            with self._lock:
                self._runtime = runtime
                if self._stage_fenced:
                    runtime.fence_worker_runtime()
        return self._runtime

    def _ack(self, command, *, error=""):
        phase = command.payload["action"]
        identity = command.payload["identity"]
        loaded = bool(self._runtime and self._runtime.worker_runtime_loaded(identity))
        if phase in {"release", "abort"} and not error:
            loaded = False
        statuses = {"prepare": "prepared", "commit": "ready", "abort": "released", "finalize": "finalized", "release": "released"}
        ack = {"action": "ack", "phase": phase, "identity": deepcopy(identity),
               "status": "error" if error else statuses[phase], "loaded": loaded}
        if not error and phase in {"prepare", "commit"} and "load_spec_digest" in identity:
            observer = getattr(self._runtime, "worker_runtime_receipt", None)
            receipt = observer(identity) if callable(observer) else None
            if receipt is None:
                raise WorkerPrepareError("prepare_runtime_receipt_missing")
            ack["runtime_receipt"] = receipt
        if "cleanup_id" in command.payload:
            ack["cleanup_id"] = command.payload["cleanup_id"]
        if error:
            ack["error"] = error
        return validate_ack(ack)

    def _send(self, command, ack):
        # Never redirect a receipt to a replacement connection or authority.
        if not self._valid(command, check_mode=False):
            return
        try:
            from transport_port import MessageType
            command.client.send_data(ack, MessageType.MODEL_PREPARE)
        except Exception:
            self.invalidate("worker_prepare_ack_failed", client=command.client)

    @staticmethod
    def _identity_key(identity):
        return json.dumps(identity, sort_keys=True)

    def _publish(self, command, update):
        with self._lock:
            if not self._valid(command):
                raise WorkerPrepareError("worker_prepare_fence_changed")
            update()

    def _record_retired(self, identities):
        with self._lock:
            for identity in identities:
                if identity is not None:
                    self._retired[self._identity_key(identity)] = True

    def _retire_all(self, *, resolve_runtime=False):
        # The serialized command worker is the only native lifecycle writer.
        # Keep proofs across repeated invalidation and collect native ownership,
        # since invalidation may have already cleared controller identities.
        with self._lock:
            identities = list(self._pending_retirements.values())
        runtime = self._runtime
        # A synchronous caller also retires an API singleton used before this
        # controller first prepared it. Binding the factory loads no weights.
        if runtime is None and resolve_runtime and self._runtime_factory is not None:
            runtime = self._get_runtime()
        if runtime is not None:
            with self.scheduler._host.full_chat_execution_lock:
                identities.extend(runtime.worker_runtime_owned_identities())
                runtime.release_worker_runtime()
                if runtime.worker_runtime_present():
                    raise WorkerPrepareError("worker_prepare_close_failed")
        self._record_retired(identities)
        with self._lock:
            self._pending_retirements.clear()

    def _apply(self, command):
        payload, identity = command.payload, command.payload["identity"]
        action = payload["action"]
        with self._lock:
            if not self._valid(command):
                raise WorkerPrepareError("worker_prepare_fence_changed")
        runtime = self._get_runtime()
        identity_key = self._identity_key(identity)
        if action == "prepare":
            with self._lock:
                if identity_key in self._retired:
                    raise WorkerPrepareError("worker_prepare_retired_identity")
                if identity != self._latest_prepare:
                    raise WorkerPrepareError("worker_prepare_stale_identity")
                phase, candidate = self._candidate_phase, self._candidate
                if self._pending_retirements:
                    raise WorkerPrepareError("worker_prepare_cleanup_pending")
                if candidate == identity and phase in {"prepared", "committed", "finalized"}:
                    if self._prepare_command != payload:
                        raise WorkerPrepareError("worker_prepare_contract_conflict")
                    duplicate = True
                else:
                    duplicate = False
                    if phase == "committed":
                        raise WorkerPrepareError("worker_prepare_cleanup_pending")
            if duplicate:
                if not runtime.worker_runtime_loaded(identity):
                    raise WorkerPrepareError("worker_prepare_runtime_unavailable")
                return
            if candidate is not None and phase in {"prepared", "failed"}:
                runtime.abort_worker_runtime(candidate, restore_previous=False)
                self._record_retired([candidate])
            def preparing():
                self._candidate = deepcopy(identity)
                self._candidate_phase = "preparing"
                self._prepare_command = deepcopy(payload)
                self._stage_fenced = True
            self._publish(command, preparing)
            runtime.fence_worker_runtime()
            started_at = time.monotonic()
            selection_started_at = started_at
            selected = resolve_prepare_selection(self.scheduler, identity)
            logger.info(
                "event=worker_prepare_selection node=%s operation=%s config=%s "
                "range=%s elapsed_ms=%.1f",
                identity["node_id"], identity["operation_id"], identity["config_id"],
                identity["layer_range"], (time.monotonic() - selection_started_at) * 1000,
            )
            runtime_started_at = time.monotonic()
            runtime.prepare_worker_runtime(payload, selected, command.cancelled)
            logger.info(
                "event=worker_prepare_runtime_loaded node=%s operation=%s config=%s "
                "range=%s elapsed_ms=%.1f",
                identity["node_id"], identity["operation_id"], identity["config_id"],
                identity["layer_range"], (time.monotonic() - runtime_started_at) * 1000,
            )
            if command.cancelled.is_set() or not self._valid(command) or identity != self._latest_prepare:
                runtime.abort_worker_runtime(identity, restore_previous=False)
                raise WorkerPrepareError("worker_prepare_fence_changed")
            selection_recheck_started_at = time.monotonic()
            if resolve_prepare_selection(self.scheduler, identity) != selected:
                runtime.abort_worker_runtime(identity, restore_previous=False)
                raise WorkerPrepareError("worker_prepare_registration_changed")
            logger.info(
                "event=worker_prepare_selection_recheck node=%s operation=%s config=%s "
                "range=%s elapsed_ms=%.1f total_elapsed_ms=%.1f",
                identity["node_id"], identity["operation_id"], identity["config_id"],
                identity["layer_range"], (time.monotonic() - selection_recheck_started_at) * 1000,
                (time.monotonic() - started_at) * 1000,
            )
            self._publish(command, lambda: setattr(self, "_candidate_phase", "prepared"))
        elif action == "commit":
            with self._lock:
                if identity != self._latest_prepare:
                    raise WorkerPrepareError("worker_prepare_stale_identity")
                if self._candidate != identity or self._candidate_phase not in {"prepared", "committed", "finalized"}:
                    raise WorkerPrepareError("worker_prepare_not_prepared")
                promote = self._candidate_phase == "prepared"
            if promote:
                runtime.commit_worker_runtime(identity)
                def committed():
                    if identity != self._latest_prepare:
                        runtime.fence_worker_runtime()
                        raise WorkerPrepareError("worker_prepare_stale_identity")
                    self._previous_identity = self._active_identity
                    self._active_identity = deepcopy(identity)
                    self._active_scope = command
                    self._candidate_phase = "committed"
                    self._stage_fenced = False
                try:
                    self._publish(command, committed)
                except Exception:
                    with self._lock:
                        restore = self._active_scope is not None and self._valid(self._active_scope)
                    runtime.abort_worker_runtime(identity, restore_previous=restore)
                    runtime.fence_worker_runtime()
                    raise
            if not runtime.worker_runtime_loaded(identity):
                raise WorkerPrepareError("worker_prepare_runtime_unavailable")
        else:
            with self._lock:
                retired = identity_key in self._retired
                phase, candidate = self._candidate_phase, self._candidate
                active, previous = self._active_identity, self._previous_identity
                restore = self._active_scope is not None and self._valid(self._active_scope)
                pending = identity_key in self._pending_retirements
            owned = runtime.worker_runtime_owned_identities()
            if retired and identity not in owned:
                return
            if pending:
                if identity in owned:
                    runtime.release_worker_runtime(identity)
                elif runtime.worker_runtime_present():
                    raise WorkerPrepareError("worker_prepare_identity_mismatch")
                self._record_retired([identity])
                with self._lock:
                    self._pending_retirements.pop(identity_key, None)
                return
            # Absence is an ownership proof, not a loaded/child-liveness check.
            if not runtime.worker_runtime_present() and phase != "preparing":
                self._record_retired([identity])
                def empty():
                    if self._candidate == identity:
                        self._candidate = None
                        self._candidate_phase = "released"
                self._publish(command, empty)
                return
            if identity == previous and identity != candidate:
                runtime.retire_previous_worker_runtime(identity)
                self._record_retired([identity])
                self._publish(command, lambda: setattr(self, "_previous_identity", None))
                return
            if action == "release":
                if identity not in (active, candidate):
                    raise WorkerPrepareError("worker_prepare_identity_mismatch")
                runtime.release_worker_runtime(identity)
                self._record_retired([identity, previous if identity == active else None])
                def released():
                    if self._active_identity == identity:
                        self._previous_identity = None
                        self._active_identity = None
                        self._active_scope = None
                    if self._candidate == identity:
                        self._candidate = None
                        self._candidate_phase = "released"
                    self._stage_fenced = True
                self._publish(command, released)
            elif candidate != identity:
                raise WorkerPrepareError("worker_prepare_identity_mismatch")
            elif action == "abort":
                runtime.abort_worker_runtime(identity, restore_previous=restore)
                self._record_retired([identity])
                def aborted():
                    if phase == "committed":
                        self._active_identity = previous if restore else None
                    if not restore:
                        self._active_identity = None
                    self._active_scope = command if self._active_identity is not None else None
                    self._previous_identity = None
                    self._candidate = None
                    self._candidate_phase = "aborted"
                    self._stage_fenced = self._active_identity is None
                self._publish(command, aborted)
            else:
                if phase not in {"committed", "finalized"}:
                    raise WorkerPrepareError("worker_prepare_not_committed")
                runtime.finalize_worker_runtime(identity)
                self._record_retired([previous])
                def finalized():
                    self._previous_identity = None
                    self._candidate_phase = "finalized"
                self._publish(command, finalized)

    def _collect_finalized_cache(self, command):
        worker = getattr(self.scheduler, "_model_supply_worker", None)
        collect = getattr(worker, "collect_garbage", None)
        if not callable(collect):
            return

        def interrupted():
            with self._lock:
                return (not self._valid(command) or self._runner is not None
                        or self._current is not None or bool(self._queue)
                        or self._candidate_phase != "finalized"
                        or self._active_identity != command.payload["identity"]
                        or bool(getattr(self.scheduler, "_task_worker_active_attempts", {}))
                        or bool(getattr(self.scheduler, "_active_pipeline_task_ids", set())))

        if interrupted():
            return
        started = time.monotonic()
        try:
            report = collect(cancelled=interrupted)
            logger.info("event=worker_post_finalize_gc elapsed_ms=%d report=%s",
                        int((time.monotonic() - started) * 1000), report)
        except Exception:
            logger.warning("event=worker_post_finalize_gc_failed", exc_info=True)

    def _run(self):
        finalized = None
        while True:
            with self._lock:
                if not self._queue:
                    self._current = None
                    self._runner = None
                    break
                command = self._queue.popleft()
                self._current = command
            if command.payload is None:
                retirement = command.retirement
                error = None
                try:
                    self._retire_all(resolve_runtime=bool(retirement and retirement.resolve_runtime))
                except Exception as exc:
                    error = exc
                    logger.warning("worker prepare cleanup failed", exc_info=True)
                finally:
                    with self._lock:
                        if retirement is not None and not retirement.done.is_set():
                            if retirement.epoch != self._epoch:
                                retirement.error = WorkerPrepareError("worker_prepare_retire_epoch_changed")
                            else:
                                retirement.error = error
                                retirement.unloaded = error is None
                            retirement.done.set()
                continue
            error = ""
            try:
                with self.scheduler._host.full_chat_execution_lock:
                    self._apply(command)
                    if not self._valid(command):
                        self._retire_all()
                        raise WorkerPrepareError("worker_prepare_fence_changed")
                    if command.payload["action"] == "prepare" and (command.cancelled.is_set() or command.payload["identity"] != self._latest_prepare):
                        self._runtime.abort_worker_runtime(command.payload["identity"], restore_previous=False)
                        raise WorkerPrepareError("worker_prepare_stale_identity")
            except Exception as exc:
                if not self._valid(command) and self._runtime is not None:
                    try:
                        self._retire_all()
                    except Exception:
                        logger.warning("worker prepare fence cleanup failed", exc_info=True)
                with self._lock:
                    if command.epoch == self._epoch and command.payload["action"] == "prepare" and self._candidate == command.payload["identity"] and self._candidate_phase not in {"committed", "finalized"}:
                        self._candidate_phase = "failed"
                code = getattr(exc, "reason_code", "worker_prepare_runtime_failed")
                error = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,256}", code) else "worker_prepare_runtime_failed"
                # ★ 2026-10-23：原始异常必须留痕。此前只记错误码，于是
                #   "reason_code 非法 ⇒ 降级成 worker_prepare_runtime_failed" 的真因
                #   被彻底吞掉（实测 ROG 在 prepare [8,16) 时 297 ms 失败，日志里
                #   没有任何堆栈，只能反推）。降级本身保持原样，只是不再丢现场。
                logger.warning(
                    "worker prepare action failed: action=%s code=%s raw_code=%s",
                    command.payload["action"], error, getattr(exc, "reason_code", "<none>"),
                    exc_info=True,
                )
            self._send(command, self._ack(command, error=error))
            finalized = command if not error and command.payload["action"] == "finalize" else None
        if finalized is not None:
            self._collect_finalized_cache(finalized)

    def check_stage(self, offer):
        """All production layer stages require a committed identity and scope."""
        with self._lock:
            identity = validate_stage_identity(offer.get("prepare_identity"))
            active, runtime, epoch = self._active_identity, self._runtime, self._epoch
            if self._stage_fenced:
                raise WorkerPrepareError("worker_prepare_stage_fenced")
            if active is None or identity != stage_identity(active):
                raise WorkerPrepareError("worker_prepare_stage_identity_mismatch")
            if self._active_scope is None or not self._valid(self._active_scope):
                raise WorkerPrepareError("worker_prepare_fence_changed")
            if offer.get("model_identity") != active["stage_model_identity"] or offer.get("layer_range") != active["layer_range"]:
                raise WorkerPrepareError("worker_prepare_stage_contract_mismatch")
        # Do not hold the controller lock while acquiring the native owner lock.
        if runtime is None or not runtime.worker_runtime_ready(active):
            raise WorkerPrepareError("worker_prepare_runtime_unavailable")
        with self._lock:
            if epoch != self._epoch or self._stage_fenced or active != self._active_identity:
                raise WorkerPrepareError("worker_prepare_stage_fenced")
            if self._active_scope is None or not self._valid(self._active_scope):
                raise WorkerPrepareError("worker_prepare_fence_changed")

    def invalidate(self, reason_code, *, client=None, close=False):
        with self._lock:
            if client is not None and getattr(self.scheduler, "_tcp_client", None) is not client:
                return
            for identity in (self._candidate, self._active_identity, self._previous_identity):
                if identity is not None:
                    self._pending_retirements[self._identity_key(identity)] = deepcopy(identity)
            previous_retirement = self._cleanup_retirement
            if previous_retirement is not None and not previous_retirement.done.is_set():
                previous_retirement.error = WorkerPrepareError("worker_prepare_retire_epoch_changed")
                previous_retirement.done.set()
            self._epoch += 1
            self._closed = self._closed or close
            self._stage_fenced = True
            if self._runtime is not None:
                self._runtime.fence_worker_runtime()
            if self._current is not None:
                self._current.cancelled.set()
            self._queue.clear()
            self._active_identity = self._previous_identity = self._candidate = None
            self._active_scope = None
            self._latest_prepare = None
            self._candidate_phase = "released"
            cleanup = self._scope(None, getattr(self.scheduler, "_tcp_client", None))
            cleanup.retirement = self._cleanup_retirement = _Retirement(self._epoch)
            self._queue.append(cleanup)
            self._start_locked()
            return cleanup.retirement

    def retire_and_join(self, reason_code: str, timeout: float = 120) -> bool:
        """Fence and physically retire the factory's runtime within a deadline.

        Call outside full_chat_execution_lock. True proves that this epoch's
        cleanup verified worker_runtime_present() == False. Native calls may
        finish later after a timeout; timeout never grants unload readiness.
        """
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or not 0 < timeout <= threading.TIMEOUT_MAX):
            raise WorkerPrepareError("invalid_worker_prepare_retire_timeout")
        execution_lock = self.scheduler._host.full_chat_execution_lock
        is_owned = getattr(execution_lock, "_is_owned", None)
        if callable(is_owned) and is_owned():
            raise WorkerPrepareError("worker_prepare_retire_lock_held")
        if threading.current_thread() is self._runner:
            raise WorkerPrepareError("worker_prepare_retire_from_worker")
        deadline = time.monotonic() + timeout
        with self._lock:
            self._retire_waiters += 1
        try:
            with self._lock:
                retirement = self.invalidate(reason_code)
                retirement.resolve_runtime = True
            if not retirement.done.wait(max(0.0, deadline - time.monotonic())):
                raise WorkerPrepareError("worker_prepare_retire_timeout")
            with self._lock:
                if retirement is not self._cleanup_retirement or retirement.epoch != self._epoch:
                    raise WorkerPrepareError("worker_prepare_retire_epoch_changed")
                if retirement.error is not None:
                    raise WorkerPrepareError("worker_prepare_retire_failed") from retirement.error
                if not retirement.unloaded:
                    raise WorkerPrepareError("worker_prepare_retire_failed")
                # New prepare remains barred through the epoch/proof check.
                # The worker checked native ownership, so the caller never
                # blocks on a native lock after the bounded event wait.
                return True
        except WorkerPrepareError:
            raise
        except Exception as exc:
            raise WorkerPrepareError("worker_prepare_retire_failed") from exc
        finally:
            with self._lock:
                self._retire_waiters -= 1

    def reopen(self):
        # Scheduler can stop/start during a listener or role transition. Preserve
        # retirement proofs and let queued native cleanup finish before prepare.
        with self._lock:
            if self._closed:
                self._closed = False
                self._epoch += 1

    def close(self, *, wait=True):
        self.invalidate("worker_prepare_stopped", close=True)
        with self._lock:
            runner = self._runner
        if wait and runner is not None and runner is not threading.current_thread():
            runner.join(timeout=15)
