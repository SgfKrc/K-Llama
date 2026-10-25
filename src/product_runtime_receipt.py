"""Read-only runtime receipts; never load models, hash artifacts or infer from plans."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
import os
import re
import sys
import time


class RuntimeObservationError(ValueError):
    """A required observation is missing, inconsistent or changing."""


@dataclass(frozen=True)
class TensorMaterialization:
    name: str
    shape: tuple[int, ...]
    element_size: int
    source_element_size: int
    source_file: str


@dataclass(frozen=True)
class LayerMaterializationProof:
    layer_range: tuple[int, int]
    decoder_prefix: str
    tensors: tuple[TensorMaterialization, ...]
    materialized_bytes: int
    allowed_boundary_prefixes: tuple[str, ...] = ()
    artifact_path: str = ""
    artifact_sha256: str = ""
    total_layers: int = 0

    def as_dict(self):
        # Every returned container is detached from the immutable load receipt.
        return {
            "source": "safetensors_key_filtered",
            "layer_range": list(self.layer_range),
            "decoder_prefix": self.decoder_prefix,
            "loaded_tensor_names": [tensor.name for tensor in self.tensors],
            "tensor_metadata": {
                tensor.name: {"shape": list(tensor.shape), "element_size": tensor.element_size,
                              "source_element_size": tensor.source_element_size,
                              "source_file": tensor.source_file}
                for tensor in self.tensors
            },
            "materialized_bytes": self.materialized_bytes,
            "allowed_boundary_prefixes": list(self.allowed_boundary_prefixes),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "total_layers": self.total_layers,
        }



@dataclass(frozen=True)
class NativeLayerLoadProof:
    artifact_path: str
    artifact_sha256: str
    layer_range: tuple[int, int]
    total_layers: int


def _require(condition, reason):
    if not condition:
        raise RuntimeObservationError(reason)


def _state(owner, name, default=None):
    # Bypass lazy proxies and __getattr__; observation must never create owners.
    try:
        return vars(owner).get(name, default)
    except TypeError:
        return default


@contextmanager
def _hold(owner, name):
    lock = _state(owner, name)
    _require(lock is not None, f"missing existing runtime lock: {name}")
    acquired = lock.acquire(timeout=2.0)
    _require(acquired, f"runtime busy: {name}")
    try:
        yield
    finally:
        lock.release()


def _selection(identity):
    _require(isinstance(identity, dict), "missing active operation identity")
    result = {key: identity.get(key) for key in ("operation_id", "config_id", "generation")}
    _require(all(isinstance(result[key], str) and result[key].strip()
                 for key in ("operation_id", "config_id")), "missing active operation/config identity")
    _require(type(result["generation"]) is int and result["generation"] > 0,
             "missing active generation")
    return result


def _digest(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "missing loaded artifact SHA256 identity (probe must independently rehash)")
    return value


def _path(value):
    _require(isinstance(value, (str, os.PathLike)) and bool(os.fspath(value)),
             "missing actual loaded artifact path")
    return os.path.normcase(os.path.abspath(value))


def _span(value, total):
    _require(type(total) is int and total > 0, "missing actual source total_layers")
    _require(isinstance(value, (tuple, list)) and len(value) == 2
             and all(type(n) is int for n in value)
             and 0 <= value[0] < value[1] <= total, "missing/invalid actual loaded layer range")
    _require(value[1] - value[0] < total, "actual runtime contains the full decoder")
    return list(value)


def _pytorch(manager, selection, process):
    model = _state(manager, "model")
    _require(model is not None, "PyTorch manager has no loaded model")
    metrics = _state(manager, "_layer_load_metrics") or {}
    proof = metrics.get("materialization_proof")
    _require(isinstance(proof, LayerMaterializationProof),
             "missing immutable selective tensor materialization proof; empty mmap is insufficient")
    total = _state(manager, "_total_model_layers")
    span = _span(_state(manager, "layer_range"), total)
    _require(proof.layer_range == tuple(span) and proof.total_layers == total,
             "materialization proof differs from loaded manager range/total")
    _require(_path(proof.artifact_path) == _path(_state(manager, "_model_path")),
             "materialization artifact path differs from loaded manager")
    digest = _digest(proof.artifact_sha256)
    # Count actual live modules, not the scheduler plan or a request descriptor.
    transformer = getattr(model, "model", None)
    if transformer is not None and getattr(transformer, "language_model", None) is not None:
        transformer = transformer.language_model
    if transformer is not None and hasattr(transformer, "layers"):
        count = len(transformer.layers)
    else:
        transformer = getattr(model, "transformer", None)
        _require(transformer is not None and hasattr(transformer, "h"),
                 "cannot observe actual PyTorch decoder modules")
        count = len(transformer.h)
    _require(count == span[1] - span[0] == _state(manager, "_model_layers"),
             "actual PyTorch decoder count differs from load proof")
    _require(proof.tensors and proof.decoder_prefix.endswith("."), "empty selective tensor proof")
    names, layers, byte_count = set(), set(), 0
    # Both existing selective adapters materialize the final norm on every
    # segment (it remains in the constructed text tower even without lm_head).
    # This fixed list reflects actual loader behavior, not caller-selected keys.
    boundaries = ["model.norm.", "model.language_model.norm.", "transformer.ln_f."]
    if span[0] == 0:
        boundaries += ["model.embed_tokens.", "model.language_model.embed_tokens.", "transformer.wte."]
    if span[1] == total:
        boundaries += ["lm_head."]
    for tensor in proof.tensors:
        _require(tensor.name not in names, "duplicate tensor in materialization proof")
        names.add(tensor.name)
        if tensor.name.startswith(proof.decoder_prefix):
            suffix = tensor.name[len(proof.decoder_prefix):]
            layer, separator, _ = suffix.partition(".")
            _require(separator and layer.isdigit() and span[0] <= int(layer) < span[1],
                     "materialized tensor outside assigned decoder range")
            layers.add(int(layer))
        else:
            _require(any(tensor.name.startswith(prefix) for prefix in boundaries),
                     "materialized tensor outside owned boundaries")
        _require(all(type(n) is int and n > 0 for n in tensor.shape)
                 and type(tensor.element_size) is int and tensor.element_size > 0,
                 "invalid materialized tensor shape/element_size")
        _require(bool(tensor.source_file) and os.path.isabs(tensor.source_file),
                 "missing actual safetensors source shard path")
        shard = _path(tensor.source_file)
        artifact_root = _path(proof.artifact_path)
        _require(os.path.commonpath((shard, artifact_root)) == artifact_root,
                 "materialized shard is outside the loaded artifact directory")
        byte_count += math.prod(tensor.shape) * tensor.element_size
    _require(layers == set(range(*span)), "selective tensor proof omits loaded decoder layers")
    _require(byte_count == proof.materialized_bytes == metrics.get("materialized_tensor_bytes"),
             "selective tensor materialization byte accounting differs")
    materialization = {**proof.as_dict(), **process, **selection}
    if _state(manager, "_layer_offload") is not None:
        # Observe the live CPU source owners and CUDA unit sizes under the
        # existing manager lock. A requested policy is never mmap evidence.
        try:
            observation = deepcopy(manager.layer_offload_observation())
        except (ValueError, RuntimeError) as exc:
            raise RuntimeObservationError("actual layer offload observation unavailable") from exc
        _require(isinstance(observation, dict)
                 and observation.get("materialization") == "safetensors_layer_offload",
                 "actual layer offload materialization missing")
        materialization["layer_offload"] = observation
    return {"engine": "pytorch", "loaded": True, "full_model_loaded": False,
            "layer_range": span, "total_layers": total, "loader_pid": process["pid"],
            "artifact": {"path": proof.artifact_path, "sha256": digest,
                         "layer_range": span, "decoder_layers": count},
            "materialization": materialization}


def _native_count(binding):
    # Read the already-loaded native model; no new llama import or model open.
    internal = _state(binding, "_model")
    pointer = _state(internal, "model")
    native = sys.modules.get("llama_cpp.llama_cpp")
    counter = getattr(native, "llama_model_n_layer", None)
    _require(pointer is not None and callable(counter), "missing actual native decoder count API/handle")
    count = counter(pointer)
    _require(type(count) is int and count > 0, "invalid actual native decoder count")
    return count


def _upstream(upstream, process):
    _require(upstream is not None and upstream.is_available, "native upstream is not loaded/alive")
    path, count = _state(upstream, "model_path"), _state(upstream, "n_layer")
    _require(type(count) is int and count > 0, "missing native upstream decoder count")
    child = _state(upstream, "_worker")
    if child is not None:
        _require(child.poll() is None and type(child.pid) is int and child.pid > 0,
                 "native loader child is not alive")
        loader = child.pid
    else:
        _require(_state(upstream, "_lib") is not None and _state(upstream, "_handle") is not None,
                 "native in-process loader handle unavailable")
        loader = process["pid"]
    return path, count, loader


def _llama(manager, process):
    engine = _state(manager, "_llama_engine") or manager
    _require(_state(engine, "_loaded") is True and _state(engine, "_model") is not None,
             "llama.cpp binding has no actual loaded model; server-only runtime lacks segment receipt")
    _require(_state(engine, "_rpc_session") is None, "RPC loader ownership is not locally observable")
    path = _state(engine, "_model_path")
    descriptor = _state(engine, "_pipeline_descriptor") or {}
    proof = _state(engine, "_layer_load_proof")
    _require(isinstance(proof, NativeLayerLoadProof),
             "missing load-bound GGUF artifact path/SHA/range/total receipt")
    span = _span(proof.layer_range, proof.total_layers)
    _require(_path(proof.artifact_path) == _path(path)
             and descriptor.get("assignment_layer_range") == span
             and descriptor.get("partial_assignment") is True
             and _path(descriptor.get("loaded_artifact")) == _path(path),
             "loaded GGUF artifact differs from immutable load receipt")
    _require(descriptor.get("total_layers") == proof.total_layers,
             "source descriptor total differs from loaded artifact manifest total")
    digest = _digest(proof.artifact_sha256)
    from qlh_core.adapters.llama_vocabulary import LlamaVocabulary
    binding = _state(engine, "_model")
    keep_head = _state(engine, "_keep_head")
    loader = process["pid"]
    if isinstance(binding, LlamaVocabulary):
        # A vocab-only handle has no decoder weights. Count the existing native
        # weight owner; a missing owner remains an unavailable observation.
        upstream_path, count, loader = _upstream(keep_head, process)
        _require(_path(upstream_path) == _path(path), "keep-head loaded a different artifact")
    else:
        count = _native_count(binding)
        if keep_head is not None:
            upstream_path, upstream_count, loader = _upstream(keep_head, process)
            _require(_path(upstream_path) == _path(path) and upstream_count == count,
                     "keep-head child and binding loaded different artifacts/counts")
    _require(count == span[1] - span[0], "actual llama.cpp native count differs from loaded segment")
    return {"engine": "llama_cpp", "loaded": True, "full_model_loaded": False,
            "layer_range": span, "total_layers": proof.total_layers,
            "loader_pid": loader, "native_layer_count": count,
            "artifact": {"path": os.fspath(path), "sha256": digest,
                         "layer_range": span, "decoder_layers": count}}


def _manager_runtime(manager, selection, process):
    _require(manager is not None, "no existing model manager (diagnostics never initializes one)")
    engine = _state(manager, "_engine_type")
    _require(engine != "pytorch" or _state(manager, "_lock") is not None,
             "missing existing PyTorch manager lock")
    with _hold(manager, "_lock") if _state(manager, "_lock") is not None else _no_lock():
        engine = _state(manager, "_engine_type")
        if engine == "pytorch":
            return _pytorch(manager, selection, process)
        _require(engine == "llama_cpp" or type(manager).__name__ == "LlamaCppEngine",
                 "actual loaded engine is absent or unsupported")
        return _llama(manager, process)


@contextmanager
def _no_lock():
    yield


def _master_transaction(scheduler):
    with _hold(scheduler, "_layer_config_lock"):
        transaction = _state(scheduler, "_pipeline_load_transaction")
        _require(isinstance(transaction, dict) and transaction.get("phase") == "ready"
                 and transaction.get("published") is True and not transaction.get("cancel_requested"),
                 "missing ready/published master pipeline transaction")
        config, generation = transaction.get("config_id"), transaction.get("generation")
        identities = [item.get("worker_prepare_identity")
                      for item in transaction.get("expected", {}).values()
                      if isinstance(item, dict) and isinstance(item.get("worker_prepare_identity"), dict)]
        _require(identities, "master transaction has no committed operation identity")
        selection = _selection(identities[0])
        _require(selection["config_id"] == config and selection["generation"] == generation
                 and all(_selection(identity) == selection for identity in identities),
                 "master transaction operation/config/generation disagree")
        candidate = transaction.get("candidate_id")
        _require(isinstance(candidate, str) and bool(candidate), "master lacks installed candidate binding")
        return {"selection": selection, "candidate_id": candidate,
                "expected": deepcopy(identities)}


def _master(host, scheduler, peek_manager, process):
    before = _master_transaction(scheduler)
    with _hold(host, "_pipeline_candidate_lock"):
        candidate = _state(host, "_pipeline_candidate_transaction")
        manager = peek_manager()
        _require(candidate is not None and candidate.transaction_id == before["candidate_id"]
                 and candidate.phase in {"committed", "finalized"} and candidate.manager is manager,
                 "master published transaction differs from actual installed manager")
        _require(candidate.previous_manager is None, "previous master runtime is pending retirement")
        runtime = _manager_runtime(manager, before["selection"], process)
    _require(before == _master_transaction(scheduler), "master identity changed during runtime capture")
    return {**before["selection"], **runtime}


def _worker_scope(controller):
    with _hold(controller, "_lock"):
        identity = deepcopy(_state(controller, "_active_identity"))
        scope = _state(controller, "_active_scope")
        _require(not _state(controller, "_stage_fenced", True) and identity is not None
                 and scope is not None and controller._valid(scope),
                 "worker active identity is absent, cancelled, disconnected or fenced")
        return identity, _state(controller, "_runtime"), _state(controller, "_epoch"), scope


def _worker(scheduler, process, node_id):
    controller = _state(scheduler, "_worker_prepare")
    _require(controller is not None, "worker prepare controller is absent")
    identity, owner, epoch, scope = _worker_scope(controller)
    selection = _selection(identity)
    _require(identity.get("node_id") == node_id, "active worker identity belongs to another node")
    _require(owner is not None, "no existing worker runtime owner")
    # Never hold controller._lock while entering the native runtime lock.
    with _hold(owner, "_worker_runtime_lock"):
        active = _state(owner, "_worker_runtime_active")
        fence = _state(owner, "_worker_runtime_fenced")
        _require(active is not None and active.identity == identity
                 and fence is not None and not fence.is_set()
                 and _state(owner, "_worker_runtime_phase") in {"committed", "finalized", "aborted"},
                 "worker actual runtime is absent, changed, or fenced")
        _require(_state(owner, "_worker_runtime_candidate") is None
                 and _state(owner, "_worker_runtime_previous") is None
                 and not _state(owner, "_worker_runtime_orphans"),
                 "worker has additional candidate/previous/orphan loader owners")
        selected = active.selected or {}
        manifest = selected.get("manifest") or {}
        for key in ("operation_id", "node_id", "artifact_sha256", "layer_range", "total_layers"):
            _require(manifest.get(key) == identity.get(key), f"loaded supply manifest differs: {key}")
        stage = identity.get("stage_model_identity") or {}
        span = _span(manifest.get("layer_range"), manifest.get("total_layers"))
        digest = _digest(manifest.get("artifact_sha256"))
        if active.manager is not None:
            _require(active.upstream is None and stage.get("engine") == "pytorch",
                     "ambiguous worker engine ownership")
            runtime = _manager_runtime(active.manager, selection, process)
            _require(_path(selected.get("directory")) == _path(runtime["artifact"]["path"]),
                     "worker supplied directory differs from actual materialized directory")
        else:
            _require(stage.get("engine") == "llama_cpp", "worker engine identity differs from actual native owner")
            path, count, loader = _upstream(active.upstream, process)
            _require(_path(path) == _path(selected.get("model_path")) and count == span[1] - span[0],
                     "actual worker native artifact/count differs from loaded selection")
            runtime = {"engine": "llama_cpp", "loaded": True, "full_model_loaded": False,
                       "layer_range": span, "total_layers": manifest["total_layers"],
                       "loader_pid": loader, "native_layer_count": count,
                       "artifact": {"path": os.fspath(path), "sha256": digest,
                                    "layer_range": span, "decoder_layers": count}}
        _require(runtime["artifact"]["sha256"] == digest and runtime["layer_range"] == span
                 and runtime["total_layers"] == manifest["total_layers"],
                 "actual worker load differs from committed identity")
    after_identity, after_owner, after_epoch, after_scope = _worker_scope(controller)
    _require(identity == after_identity and owner is after_owner and epoch == after_epoch
             and scope is after_scope, "worker identity/epoch changed during runtime capture")
    return {**selection, **runtime}


def capture_runtime(api_module):
    """Return v1 runtime from existing SSOT, or raise RuntimeObservationError.

    HTTP/source authorization and independent artifact rehash are caller-owned.
    Lock order follows model replacement: full-chat -> runtime execution -> layer execution.
    Config/controller locks are released before acquiring model/native-owner locks,
    then rechecked. No factories, lazy managers, plans, hashes or model API calls.
    """
    import psutil
    process = psutil.Process(os.getpid())
    identity = {"pid": process.pid, "created_at": process.create_time()}
    host, scheduler = api_module.model_host, api_module.scheduler
    from qlh_core.bootstrap.runtime import LazyResource
    if isinstance(scheduler, LazyResource):
        services = _state(_state(api_module, "_app_runtime"), "_services")
        scheduler = _state(services, "scheduler")
        _require(scheduler is not None, "no existing runtime scheduler")
    with _hold(host, "full_chat_execution_lock"):
        with _hold(_state(host, "_runtime_controller"), "execution_lock"):
            with _hold(scheduler, "_layer_execution_lock"):
                role = scheduler._effective_role()
                node_id = scheduler.get_effective_node_id()
                _require(isinstance(node_id, str) and bool(node_id), "missing actual runtime node_id")
                if role == "master":
                    runtime = _master(host, scheduler, api_module._peek_model_manager, identity)
                else:
                    _require(role == "client", "runtime node role is unsupported/unresolved")
                    runtime = _worker(scheduler, identity, node_id)
                _require(role == scheduler._effective_role() and node_id == scheduler.get_effective_node_id(),
                         "node role/identity changed during runtime capture")
                return {**identity, "captured_at": time.time(), "node_id": node_id, **runtime}
