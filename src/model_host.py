"""推理宿主（ModelHost）—— 阶段 0 边界治理（0.2/0.4）

统一持有：
  - ModelManager 实例（经 _LazyModelManager 延迟导入 model_module，
    避免冷启动 import 全量加载——原 api_server.py:113-155 迁入）
  - 模型运行时可变状态：model_loaded / generation_config
  - 推理执行锁：full_chat_execution_lock（原 api_server._full_chat_execution_lock）

api_server 与 scheduler 共享同一 model_host 单例，消除 scheduler 对
api_server 的运行时反向 import（scheduler.py 中 12 处 `import api_server`）。

原生访问使用显式端口；运行态由 RuntimeController 发布。
"""
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from koakuma_engine import (
    BackendCapabilities,
    backend_capabilities,
    backend_id_for,
    normalize_backend_request,
    select_backend,
)
from model_load_resolver import ModelLoadResolution


class InferenceHost(Protocol):
    """推理宿主协议：行为由 model_module.ModelManager 提供（阶段 1 由
    inference-svc 以 HTTP 契约实现同构接口）。"""

    def select_engine(self, profile): ...

    @property
    def engine_type(self) -> str: ...

    @property
    def backend_id(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def supports(self, capability: str) -> bool: ...

    def load_model(self, engine, quant_type, use_compile, model_id): ...

    def unload_model(self): ...

    def prepare_pipeline_candidate(
        self, transaction_id, model_id, model_path, quant_type=None,
        layer_range=None, model_sha256=None,
    ): ...

    def materialize_pipeline_candidate(
        self, transaction_id, start_layer, end_layer, has_embedding,
        has_lm_head, total_layers=None, model_id=None,
    ): ...

    def commit_pipeline_candidate(self, transaction_id): ...

    def abort_pipeline_candidate(self, transaction_id): ...

    def finalize_pipeline_candidate(self, transaction_id): ...

    def load_layer_range(self, layer_range, embed, lm_head): ...

    def forward_layers(self, layer_range, hidden, past_key_values, **kw): ...

    def chat(self, messages, **kw): ...

    def chat_image(self, image_path, prompt, **kw): ...

    def chat_stream(self, messages, **kw): ...

    def native_vision_available(self) -> bool: ...

    def ensure_full_model(self): ...

    @property
    def scheduler_callbacks(self) -> Optional["SchedulerCallbacks"]: ...


class SchedulerCallbacks(Protocol):
    """Explicit callbacks required by scheduler control-plane features."""

    active_execution_model_identity: Callable[[], Any]
    execute_task_worker_stage: Callable[[Any, Any], dict]
    build_model_chat_prompt: Callable[..., str]
    thinking_system_prompt: str
    snapshot_recent_logs: Callable[[], tuple[list[dict], int]]
    filter_recent_logs: Callable[..., list[dict]]
    format_model_response: Callable[..., tuple[str, Optional[str]]]
    supports_layer_multi_position_argmax: Callable[[], bool]


@dataclass(frozen=True)
class SchedulerCallbackSet:
    """Immutable scheduler dependency bundle assembled by the API composition root."""

    active_execution_model_identity: Callable[[], Any]
    execute_task_worker_stage: Callable[[Any, Any], dict]
    build_model_chat_prompt: Callable[..., str]
    thinking_system_prompt: str
    snapshot_recent_logs: Callable[[], tuple[list[dict], int]]
    filter_recent_logs: Callable[..., list[dict]]
    format_model_response: Callable[..., tuple[str, Optional[str]]]
    supports_layer_multi_position_argmax: Callable[[], bool] = lambda: False


class _LazyModelManager:
    """Delay importing model_module until the model manager is first used."""

    __slots__ = ("_instance", "_lock")
    _instance: Any
    _lock: Any

    def __init__(self):
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_lock", threading.RLock())

    def _get_instance(self):
        instance = self._instance
        if instance is not None:
            return instance
        with self._lock:
            instance = self._instance
            if instance is None:
                from model_module import ModelManager

                instance = ModelManager()
                object.__setattr__(self, "_instance", instance)
        return instance

    def __getattr__(self, name):
        try:
            instance = self._get_instance()
        except ImportError as exc:
            # 无 torch 的边缘构建（免安装版）里 `model_module` 根本 import 不了
            # （它顶部就 `import torch`）。调用方大多是**探测语义**的
            # `getattr(host, "layer_range", None)` —— 它们要的是「没有这个值」，
            # 不是异常。转成 `AttributeError` 让 `getattr` 的默认值生效；其他
            # 异常照旧抛出，不掩盖真错误。
            raise AttributeError(
                f"{name!r} 需要 PyTorch 引擎，但当前节点没有 torch: {exc}"
            ) from None
        return getattr(instance, name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        setattr(self._get_instance(), name, value)

    def __delattr__(self, name):
        if name in self.__slots__:
            raise AttributeError(name)
        delattr(self._get_instance(), name)

    def __repr__(self):
        instance = self._instance
        if instance is None:
            return "<_LazyModelManager unloaded>"
        return repr(instance)


@dataclass
class _PipelineCandidateTransaction:
    transaction_id: str
    model_id: str
    model_path: str
    quant_type: Optional[str]
    layer_range: Optional[tuple[int, int]]
    model_sha256: Optional[str]
    manager: Any
    descriptor: dict
    phase: str = "preparing"
    materialization_key: Optional[tuple[Any, ...]] = None
    materialization_result: Any = None
    previous_manager: Any = None
    previous_host_state: Optional[dict] = None
    error: Optional[str] = None


_PIPELINE_HOST_MIRROR_ATTRS = (
    "_engine_type",
    "quant_type",
    "model_path",
    "active_model_id",
    "_active_model_id",
)
_PIPELINE_CANDIDATE_TERMINAL_PHASES = frozenset({"aborted", "finalized"})


def _runtime_mutation(method):
    from functools import wraps
    @wraps(method)
    def run(host, *args, **kwargs):
        with host.full_chat_execution_lock, host.runtime_controller.execution_lock:
            return method(host, *args, **kwargs)
    return run


def _runtime_publication(method):
    from functools import wraps
    @wraps(method)
    def publish(host, *args, **kwargs):
        with host.runtime_controller.publication():
            return method(host, *args, **kwargs)
    return publish


def _runtime_metadata_property(name):
    def read(host):
        return host._runtime_owner().read_metadata(name)
    def write(host, value):
        host._runtime_owner().publish_metadata(name, value)
    return property(read, write)


def _manager_port(name):
    def read(host):
        return getattr(host._materialize_manager(), name)
    def write(host, value):
        setattr(host._materialize_manager(), name, value)
    return property(read, write)


def _manager_state(name, empty=None):
    """Observe an existing owner; explicit writes may construct the manager."""
    def read(host):
        manager = host.peek_manager()
        return empty if manager is None else getattr(manager, name)

    def write(host, value):
        setattr(host._materialize_manager(), name, value)

    return property(read, write)

class ModelHost:
    """推理宿主：统一持有 ModelManager + 模型运行时状态 + 执行锁。

    原生访问使用下面列明的执行端口与只读描述符。
    """

    abort_pipeline_materialization = _manager_port("abort_pipeline_materialization")
    clear_pipeline_kv = _manager_port("clear_pipeline_kv")
    tokenizer = _manager_state("tokenizer")
    model = _manager_state("model")
    role = _manager_state("role")
    _total_model_layers = _manager_state("_total_model_layers", 0)
    _model_path = _manager_state("_model_path")
    _full_model_path = _manager_state("_full_model_path")
    _island_engine = _manager_state("_island_engine")
    _chat_template = _manager_state("_chat_template")
    _thinking_controlled = _manager_state("_thinking_controlled", False)
    chat = _manager_port("chat")
    chat_image = _manager_port("chat_image")
    chat_stream = _manager_port("chat_stream")
    native_vision_available = _manager_port("native_vision_available")
    load_layer_range = _manager_port("load_layer_range")
    ensure_layer_range = _manager_port("ensure_layer_range")
    forward_layers = _manager_port("forward_layers")
    forward_lm_head = _manager_port("forward_lm_head")
    get_device = _manager_port("get_device")
    get_memory_usage = _manager_port("get_memory_usage")
    get_model_info = _manager_port("get_model_info")
    _get_generation_eos_token_ids = _manager_port("_get_generation_eos_token_ids")
    _merge_stop_sequences = _manager_port("_merge_stop_sequences")
    _build_stop_criteria = _manager_port("_build_stop_criteria")
    _decode_generated_ids = _manager_port("_decode_generated_ids")
    _ensure_device_profile = _manager_port("_ensure_device_profile")

    _engine_type = _runtime_metadata_property("_engine_type")
    quant_type = _runtime_metadata_property("quant_type")
    model_path = _runtime_metadata_property("model_path")
    active_model_id = _runtime_metadata_property("active_model_id")
    _active_model_id = _runtime_metadata_property("_active_model_id")

    def _runtime_owner(self):
        state = object.__getattribute__(self, "__dict__")
        if "_runtime_controller" not in state:
            from qlh_core.execution.runtime import RuntimeController
            state["_runtime_controller"] = RuntimeController(write_metadata=lambda manager, name, value: setattr(manager, name, value))
        return state["_runtime_controller"]

    @property
    def runtime_controller(self):
        return self._runtime_owner()

    @property
    def _manager(self):
        return self._runtime_owner().manager()

    @_manager.setter
    def _manager(self, value):
        self._runtime_owner().install(value)

    @property
    def model_loaded(self):
        return self._runtime_owner().snapshot().model_loaded

    @model_loaded.setter
    def model_loaded(self, value):
        self._runtime_owner().update(model_loaded=value)

    @property
    def current_quant(self):
        return self._runtime_owner().snapshot().current_quant

    @current_quant.setter
    def current_quant(self, value):
        self._runtime_owner().set_quant(value)

    def __init__(self, manager: Any = None, *, initial_quant: str | None = None, environment=None):
        object.__setattr__(self, "_runtime_environment", None if environment is None else dict(environment))
        object.__setattr__(self, "_manager", manager if manager is not None else _LazyModelManager())
        object.__setattr__(self, "model_loaded", False)
        try:
            import config as _cfg
            _initial_quant = str(getattr(_cfg, "QUANT_TYPE", "int4"))
        except Exception:
            _initial_quant = "int4"
        object.__setattr__(self, "current_quant", initial_quant if initial_quant is not None else _initial_quant)
        object.__setattr__(self, "generation_config", {
            "max_new_tokens": 1024,          # laptop 档默认值
            "tier_max_new_tokens": 1024,     # 设备档位上限（auto_configure 后更新）
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        })
        object.__setattr__(self, "full_chat_execution_lock", threading.RLock())
        object.__setattr__(self, "scheduler_callbacks", None)
        object.__setattr__(self, "_pipeline_candidate_lock", threading.RLock())
        object.__setattr__(self, "_pipeline_candidate_transaction", None)
        object.__setattr__(self, "_catalog_prepare", None)

    @property
    def engine_type(self) -> str:
        """Public backend identity; implementation details stay behind the host."""

        own_id = self._runtime_owner().read_metadata("_engine_type") if "_engine_type" in dict(self._runtime_owner().snapshot().metadata) else None
        if own_id:
            return str(own_id)
        manager = object.__getattribute__(self, "_manager")
        if isinstance(manager, _LazyModelManager):
            manager = object.__getattribute__(manager, "_instance")
            if manager is None:
                return ""
        return backend_id_for(manager, default="")

    @property
    def backend_id(self) -> str:
        return self.engine_type

    @property
    def capabilities(self) -> BackendCapabilities:
        return backend_capabilities(self.engine_type)

    def supports(self, capability: str) -> bool:
        return self.capabilities.supports(capability)

    def select_engine(self, profile: dict | None = None) -> str:
        """Select the node backend through the Koakuma boundary."""

        try:
            import config as _cfg

            requested = getattr(_cfg, "INFERENCE_ENGINE", "auto")
            island_enabled = bool(getattr(_cfg, "ISLAND_ENABLED", False))
            island_base_url = str(getattr(_cfg, "ISLAND_BASE_URL", "") or "")
        except Exception:
            requested = "auto"
            island_enabled = False
            island_base_url = ""
        return select_backend(
            profile,
            requested=requested,
            island_enabled=island_enabled,
            island_base_url=island_base_url,
        )

    def configure_catalog(self, catalog_prepare) -> None:
        """Bind the owning runtime without materializing the lazy manager."""
        object.__setattr__(self, "_catalog_prepare", catalog_prepare)
        manager = self.peek_manager()
        if manager is not None:
            self._configure_manager_catalog(manager)

    def _configure_manager_catalog(self, manager, *, model_id="") -> None:
        prepare = object.__getattribute__(self, "_catalog_prepare")
        if prepare is not None:
            configure = getattr(manager, "configure_catalog", None)
            if callable(configure):
                configure(prepare, model_id=model_id)

    def configure_runtime_environment(self, environment):
        object.__setattr__(self, "_runtime_environment", dict(environment))

    def configure_scheduler_callbacks(self, callbacks: SchedulerCallbacks) -> None:
        """Install the scheduler's complete, typed callback contract atomically."""

        if callbacks is None:
            raise TypeError("scheduler callbacks are required")
        object.__setattr__(self, "scheduler_callbacks", callbacks)

    def peek_manager(self) -> Any:
        """Return an already-created manager without triggering lazy import."""

        manager = object.__getattribute__(self, "_manager")
        if isinstance(manager, _LazyModelManager):
            return object.__getattribute__(manager, "_instance")
        return manager

    def _materialize_manager(self) -> Any:
        """Materialize the torch-backed manager for an explicit torch operation."""

        manager = object.__getattribute__(self, "_manager")
        # Tests and embedders may replace the lazy-factory symbol; use the
        # stable proxy type name here instead of requiring that symbol remain a
        # class object.
        if type(manager).__name__ != "_LazyModelManager":
            self._configure_manager_catalog(manager)
            return manager
        try:
            instance = manager._get_instance()
            self._configure_manager_catalog(instance)
            return instance
        except ImportError as exc:
            raise RuntimeError(
                "当前发行版没有 PyTorch，无法使用 torch ModelManager；"
                "请选择 llama_cpp/GGUF 引擎"
            ) from exc

    @property
    def layer_range(self):
        """Read the existing native owner's range without creating a manager."""
        manager = self.peek_manager()
        return None if manager is None else getattr(manager, "layer_range", None)

    def get_pipeline_descriptor(self) -> dict:
        """Describe the existing owner without constructing a torch runtime."""
        manager = self.peek_manager()
        describe = getattr(manager, "get_pipeline_descriptor", None)
        return describe() if callable(describe) else {}

    def has_loaded_model(self) -> bool:
        """Check LLM ownership without materializing the lazy manager."""

        if bool(object.__getattribute__(self, "model_loaded")):
            return True
        manager = self.peek_manager()
        return bool(manager is not None and getattr(manager, "is_loaded", False))

    @property
    def is_loaded(self) -> bool:
        """Expose a lazy-safe loaded flag for scheduler and API callers.

        Reading this state must never instantiate ``model_module``.  In a torch-free
        distribution the lazy manager is intentionally unavailable; that is an
        unloaded host, not an AttributeError that callers might accidentally treat
        as a successful fallback.
        """

        return self.has_loaded_model()

    @property
    def is_pipeline_prepared(self) -> bool:
        """Expose distributed-only preparation without warming the lazy manager."""

        manager = self.peek_manager()
        return bool(manager is not None and getattr(manager, "is_pipeline_prepared", False))

    def ensure_full_model(self, *args, **kwargs):
        """Ensure a local full-model runtime, failing closed when it is unavailable.

        ``ModelHost`` may be backed directly by ``LlamaCppEngine`` on the torch-free
        path.  That engine does not need a separate ``ensure_full_model`` method when
        it owns a whole GGUF, but a layer artifact (for example ``head8``) must never
        be sent through the full-model chat fallback.
        """

        manager = self.peek_manager()
        if manager is None:
            raise RuntimeError(
                "完整模型回退不可用：当前节点没有可用的 ModelManager（可能为无 torch 发行版）"
            )

        ensure = getattr(manager, "ensure_full_model", None)
        if callable(ensure):
            return ensure(*args, **kwargs)

        # The torch-free GGUF engine is installed directly as the manager.  Inspect
        # its pipeline descriptor before treating the loaded artifact as full-model.
        describe = getattr(manager, "get_pipeline_descriptor", None)
        if callable(describe):
            descriptor = describe() or {}
            if descriptor.get("partial_assignment") or descriptor.get("assignment_layer_range"):
                layer_range = descriptor.get("assignment_layer_range")
                raise RuntimeError(
                    "当前 llama.cpp 引擎加载的是裁层工件"
                    f"（layer_range={layer_range!r}），禁止整模回退；"
                    "请等待流水线节点就绪或显式重新加载完整模型"
                )
        if backend_id_for(manager, default="") == "llama_cpp":
            return None
        raise RuntimeError(
            "完整模型回退不可用：当前推理引擎未提供 ensure_full_model()"
        )

    def runtime_status(self) -> dict:
        """Return the public, lazy-safe runtime state for diagnostics and tests."""

        manager = self.peek_manager()
        return {
            "manager_loaded": manager is not None,
            # Keep whole-request readiness separate from the broader question
            # of whether this process owns any materialized runtime.  A
            # distributed-only layer segment is materialized and prepared, but
            # must never be advertised as a complete local chat model.
            "model_loaded": bool(
                object.__getattribute__(self, "model_loaded")
            ),
            "runtime_materialized": bool(
                manager is not None and getattr(manager, "is_loaded", False)
            ),
            "pipeline_prepared": bool(
                manager is not None
                and getattr(manager, "is_pipeline_prepared", False)
            ),
            "current_quant": object.__getattribute__(self, "current_quant"),
        }

    @staticmethod
    def _close_pipeline_manager(manager: Any) -> None:
        if type(manager).__name__ == "_LazyModelManager":
            manager = object.__getattribute__(manager, "_instance")
        if manager is None:
            return
        close = (
            getattr(manager, "unload_model", None)
            or getattr(manager, "unload", None)
            or getattr(manager, "close", None)
        )
        if callable(close):
            close()

    def _abort_failed_pipeline_candidate(
        self,
        transaction: _PipelineCandidateTransaction,
        manager: Any,
        error: Exception,
    ) -> Optional[Exception]:
        transaction.error = str(error)
        # Unwound observer/loader frames can retain the model and tensor lists.
        # Release those locals before unload runs GC/empty_cache, retaining the
        # original traceback locations and error for diagnostics.
        import traceback
        traceback.clear_frames(error.__traceback__)
        try:
            self._close_pipeline_manager(manager)
        except Exception as cleanup_error:
            transaction.manager = manager
            transaction.phase = "cleanup_pending"
            transaction.error = f"{error}; cleanup failed: {cleanup_error}"
            return cleanup_error
        transaction.manager = None
        transaction.phase = "aborted"
        return None

    @staticmethod
    def _pipeline_candidate_conflict(
        transaction_id: str,
        active_transaction_id: str,
    ) -> RuntimeError:
        return RuntimeError(
            "MODEL_TXN_CONFLICT: "
            f"transaction {transaction_id!r} conflicts with active transaction "
            f"{active_transaction_id!r}"
        )

    def _require_pipeline_candidate(
        self,
        transaction_id: str,
    ) -> _PipelineCandidateTransaction:
        transaction = object.__getattribute__(
            self, "_pipeline_candidate_transaction",
        )
        if transaction is None:
            raise RuntimeError(
                f"MODEL_TXN_NOT_FOUND: transaction {transaction_id!r} has no candidate"
            )
        if transaction.transaction_id != transaction_id:
            raise self._pipeline_candidate_conflict(
                transaction_id, transaction.transaction_id,
            )
        return transaction

    def _snapshot_pipeline_host_state(self) -> dict:
        instance_state = object.__getattribute__(self, "__dict__")
        return {
            "model_loaded": object.__getattribute__(self, "model_loaded"),
            "current_quant": object.__getattribute__(self, "current_quant"),
            "mirrors": {
                name: (name in dict(self._runtime_owner().snapshot().metadata), dict(self._runtime_owner().snapshot().metadata).get(name))
                for name in _PIPELINE_HOST_MIRROR_ATTRS
            },
        }

    @_runtime_publication
    def _restore_pipeline_host_state(self, snapshot: dict) -> None:
        object.__setattr__(self, "model_loaded", snapshot["model_loaded"])
        object.__setattr__(self, "current_quant", snapshot["current_quant"])
        instance_state = object.__getattribute__(self, "__dict__")
        for name, (was_present, value) in snapshot["mirrors"].items():
            if was_present:
                object.__setattr__(self, name, value)
            else:
                instance_state.pop(name, None)

    def _pipeline_candidate_host_state(
        self,
        transaction: _PipelineCandidateTransaction,
    ) -> dict:
        manager = transaction.manager
        backend_id = backend_id_for(manager, default="")
        quant_type = (
            getattr(manager, "quant_type", None)
            or getattr(manager, "_quant_type", None)
            or transaction.quant_type
        )
        if not quant_type and backend_id == "llama_cpp":
            quant_type = "gguf"
        loaded = getattr(manager, "is_loaded", None)
        pipeline_prepared = bool(
            getattr(manager, "is_pipeline_prepared", False)
        )
        return {
            # `model_loaded` means a whole-request local runtime.  A materialized
            # pipeline segment is loaded for layer forwarding but must never be
            # advertised as a complete model.
            "model_loaded": (
                False
                if pipeline_prepared
                else (True if loaded is None else bool(loaded))
            ),
            "current_quant": str(quant_type) if quant_type else None,
            "mirrors": {
                "_engine_type": backend_id or None,
                "quant_type": quant_type,
                "model_path": transaction.model_path,
                "active_model_id": transaction.model_id,
                "_active_model_id": transaction.model_id,
            },
        }

    @_runtime_publication
    def _apply_pipeline_candidate_host_state(self, state: dict) -> None:
        object.__setattr__(self, "model_loaded", state["model_loaded"])
        if state["current_quant"]:
            object.__setattr__(self, "current_quant", state["current_quant"])

        instance_state = object.__getattribute__(self, "__dict__")
        for name in _PIPELINE_HOST_MIRROR_ATTRS:
            instance_state.pop(name, None)
        for name, value in state["mirrors"].items():
            if value is not None:
                object.__setattr__(self, name, value)

    def pipeline_candidate_status(self, transaction_id: str | None = None) -> dict:
        """Return candidate transaction state without exposing runtime objects."""

        with object.__getattribute__(self, "_pipeline_candidate_lock"):
            transaction = object.__getattribute__(
                self, "_pipeline_candidate_transaction",
            )
            if transaction is None:
                return {"active": False, "phase": "idle"}
            return {
                "active": transaction.phase not in _PIPELINE_CANDIDATE_TERMINAL_PHASES,
                "matches": (
                    transaction_id is None
                    or transaction.transaction_id == transaction_id
                ),
                "transaction_id": transaction.transaction_id,
                "phase": transaction.phase,
                "model_id": transaction.model_id,
                "model_path": transaction.model_path,
                "quant_type": transaction.quant_type,
                "descriptor": dict(transaction.descriptor),
                "error": transaction.error,
            }

    def prepare_pipeline_candidate(
        self,
        transaction_id: str,
        model_id: str,
        model_path: str,
        quant_type: str = None,
        layer_range: tuple[int, int] | None = None,
        model_sha256: str | None = None,
    ) -> dict:
        """Prepare one isolated pipeline runtime without touching the active one."""

        import os

        transaction_id = str(transaction_id or "").strip()
        if not transaction_id:
            raise ValueError("transaction_id is required")
        resolved_path = os.path.abspath(model_path or "")
        normalized_range = tuple(layer_range) if layer_range is not None else None
        prepare_key = (
            str(model_id or ""),
            resolved_path,
            quant_type,
            normalized_range,
            model_sha256,
        )
        lock = object.__getattribute__(self, "_pipeline_candidate_lock")
        with lock:
            current = object.__getattribute__(
                self, "_pipeline_candidate_transaction",
            )
            if current is not None and current.transaction_id == transaction_id:
                current_key = (
                    current.model_id,
                    current.model_path,
                    current.quant_type,
                    current.layer_range,
                    current.model_sha256,
                )
                if current_key != prepare_key:
                    raise self._pipeline_candidate_conflict(
                        transaction_id, current.transaction_id,
                    )
                if current.phase in {"aborted", "cleanup_pending"}:
                    raise RuntimeError(
                        f"MODEL_TXN_STATE: transaction {transaction_id!r} "
                        f"is {current.phase!r}"
                    )
                return dict(current.descriptor)
            if (
                current is not None
                and current.phase not in _PIPELINE_CANDIDATE_TERMINAL_PHASES
            ):
                raise self._pipeline_candidate_conflict(
                    transaction_id, current.transaction_id,
                )

            if os.path.isfile(resolved_path):
                from llama_engine import LlamaCppEngine

                manager = LlamaCppEngine()
                manager._environment = self._runtime_environment
            else:
                lazy_manager = _LazyModelManager()
                if type(lazy_manager).__name__ == "_LazyModelManager":
                    try:
                        manager = lazy_manager._get_instance()
                    except ImportError as exc:
                        raise RuntimeError(
                            "当前发行版没有 PyTorch，无法准备目录模型候选运行时"
                        ) from exc
                else:
                    manager = lazy_manager

            transaction = _PipelineCandidateTransaction(
                transaction_id=transaction_id,
                model_id=str(model_id or ""),
                model_path=resolved_path,
                quant_type=quant_type,
                layer_range=normalized_range,
                model_sha256=model_sha256,
                manager=manager,
                descriptor={},
            )
            object.__setattr__(self, "_pipeline_candidate_transaction", transaction)
            try:
                self._configure_manager_catalog(manager, model_id=model_id)
                descriptor = manager.prepare_pipeline_model(
                    model_id=model_id,
                    model_path=resolved_path,
                    quant_type=quant_type,
                    layer_range=normalized_range,
                    model_sha256=model_sha256,
                )
            except Exception as exc:
                cleanup_error = self._abort_failed_pipeline_candidate(
                    transaction, manager, exc,
                )
                if cleanup_error is not None:
                    raise RuntimeError(
                        "MODEL_TXN_CLEANUP_FAILED: candidate cleanup failed"
                    ) from cleanup_error
                raise
            transaction.descriptor = dict(descriptor or {})
            transaction.phase = "prepared"
            return dict(transaction.descriptor)

    def freeze_pipeline_candidate_assignment(self, transaction_id, assignment, *, config_id, generation):
        from copy import deepcopy
        from qlh_core.planning.load_spec import assignment_load_spec
        transaction = self._require_pipeline_candidate(transaction_id)
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**引擎按该 assignment 判，不是事务的 manager**。
        #
        #   同机 CPU 段（llama_cpp）与主段（pytorch）共用**同一个** candidate 事务，
        #   而 `transaction.manager` 是主段的引擎。此前用 manager 判 `native`
        #   ⇒ CPU 段被当成 safetensors 段 ⇒ `artifact_format` 错、也不去取 GGUF 摘要。
        #   实测（2026-10-26 11:52）：plan 里四段齐全、供给 `ready`，但 `/api/chat`
        #   立刻 503 `execution_segment_coverage_invalid`，缺的正是 CPU 段的 `[8,24)`。
        #
        #   缺省（assignment 未声明 engine）时退回原判据，保持既有行为不变。
        assignment_engine = str(assignment.get("engine") or "").strip()
        native = (assignment_engine == "llama_cpp") if assignment_engine else (
            getattr(transaction.manager, "engine_type", "pytorch") == "llama_cpp"
        )
        artifact_sha = transaction.model_sha256 or transaction.descriptor.get("model_sha256")
        if native:
            # CPU 段的 GGUF 由它自己加载（本机文件 `QLH_LAYER_GGUF`），**不在**
            # `transaction.manager` 里 —— 那里只有主段（pytorch）的层工件，而且
            # `ModelManager` 根本没有 `_find_layer_artifact` 方法（原 `native` 判据
            # 从未为真，这段代码是第一次被执行 ⇒ 一跑就暴露）。
            #   优先级：求解结果带的 `layer_artifact.artifact_sha256` > 本机 GGUF 文件摘要。
            selected = assignment.get("layer_artifact")
            selected_sha = (
                str(selected.get("artifact_sha256") or "").strip().lower()
                if isinstance(selected, Mapping) else ""
            )
            if selected_sha:
                artifact_sha = selected_sha
            else:
                raw_layer = (os.environ.get("QLH_LAYER_GGUF") or "").strip()
                if not raw_layer:
                    raise ValueError("deployment_local_artifact_missing")
                layer_path = Path(raw_layer)
                if not layer_path.is_absolute():
                    # 相对路径按仓库根解析（`.env` 里写的是 `models/...`）。
                    layer_path = Path(__file__).resolve().parents[1] / layer_path
                if not layer_path.is_file():
                    raise ValueError("deployment_local_artifact_missing")
                digest = hashlib.sha256()
                with open(layer_path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                artifact_sha = digest.hexdigest()
        spec = assignment_load_spec(assignment, artifact_format="gguf" if native else "safetensors")
        result = deepcopy(assignment)
        result["load_spec"] = spec.to_dict()
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**`stage_model_identity` 必须自建**。
        #
        #   `bind_model_preflight` 只给**有 binding 的 worker** 注入这个键（按
        #   `(node_id, start_layer, end_layer)` 查表，`model_preflight.py:571`），
        #   而本机执行器根本不在 worker 列表里 ⇒ 永远取不到。
        #   执行期要拿它构造 `ModelIdentity(**...)`，缺了就直接
        #   `execution_stage_model_identity_missing`（实测 14:14）。
        #   字段集与 `model_preflight._MODEL_IDENTITY_FIELDS` 完全一致
        #   （model_id/engine/format/revision/sha256），取值与远端同源：描述器 + 本段工件摘要。
        stage_model = dict(assignment.get("stage_model_identity") or {})
        if not stage_model:
            _descriptor = transaction.descriptor or {}
            stage_model = {
                "model_id": str(
                    _descriptor.get("model_id")
                    or assignment.get("model_id")
                    or "local"
                ),
                "engine": "llama_cpp" if native else "pytorch",
                "format": "gguf" if native else "safetensors",
                "revision": str(
                    assignment.get("quant_type")
                    or _descriptor.get("quant_type")
                    or "local"
                ),
                "sha256": str(artifact_sha or ""),
            }
        result["identity"] = {
            # ★ `stage_identity()` 要求 `STAGE_IDENTITY_FIELDS` 全集
            #   （operation_id/config_id/generation/assignment_id/artifact_sha256），
            #   本机执行器没有供给侧的 `operation_id`，用本机前缀派生一个合法值
            #   （`_ID` 允许 `[A-Za-z0-9_.:/-]`）。缺了它，执行期会在
            #   `validate_stage_identity` 抛 `invalid_prepare_identity`。
            "operation_id": "local-" + config_id,
            "config_id": config_id, "generation": generation,
            "assignment_id": "local-" + config_id, "node_id": "master",
            "layer_range": [int(assignment["start_layer"]), int(assignment["end_layer"])],
            "artifact_sha256": artifact_sha, "load_spec_digest": spec.digest,
            # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**与远端 `worker_prepare_identity` 同构**。
            #
            #   执行期（`qlh_core/bootstrap/execution.py` 的 `stage()` 与
            #   `_reset_pipeline_kv()`）直接消费这两个键：
            #     * `stage_model_identity` → `ModelIdentity(**...)`，由
            #       `bind_model_preflight` 注入到 assignment **顶层**，这里搬进来；
            #     * `hidden_size` → `np.zeros((1, hidden_size))` 的清理占位张量。
            #       远端是 worker 自报的，本机则取自描述器。
            #   缺了它们，本机 CPU 段一被 stage 就
            #   `KeyError: 'worker_prepare_identity'` / `KeyError: 'hidden_size'`
            #   （实测 11:31:27：四段已就位、段列表正确，却死在第一个 CPU 段）。
            "stage_model_identity": stage_model,
            "hidden_size": int(transaction.descriptor.get("hidden_size", 0) or 0),
        }
        return result

    def materialize_pipeline_candidate(
        self,
        transaction_id: str,
        start_layer: int | None,
        end_layer: int | None,
        has_embedding: bool,
        has_lm_head: bool,
        total_layers: int = None,
        model_id: str = None,
        load_assignment: dict | None = None,
    ) -> Any:
        """Materialize the candidate's assigned range while active stays live."""

        lock = object.__getattribute__(self, "_pipeline_candidate_lock")
        with lock:
            transaction = self._require_pipeline_candidate(transaction_id)
            if transaction.phase == "aborted":
                raise RuntimeError(
                    f"MODEL_TXN_ABORTED: transaction {transaction_id!r} was aborted"
                )
            if (start_layer is None) != (end_layer is None):
                raise ValueError(
                    "start_layer and end_layer must both be set or both be None"
                )
            normalized_start = (
                None if start_layer is None else int(start_layer)
            )
            normalized_end = None if end_layer is None else int(end_layer)
            materialization_key = (
                normalized_start,
                normalized_end,
                bool(has_embedding),
                bool(has_lm_head),
                total_layers,
                model_id or transaction.model_id,
                load_assignment,
            )
            if transaction.materialization_key is not None:
                if transaction.materialization_key != materialization_key:
                    raise self._pipeline_candidate_conflict(
                        transaction_id, transaction.transaction_id,
                    )
                return transaction.materialization_result
            if transaction.phase != "prepared" or transaction.manager is None:
                raise RuntimeError(
                    f"MODEL_TXN_STATE: transaction {transaction_id!r} "
                    f"cannot materialize from {transaction.phase!r}"
                )
            try:
                if normalized_start is None:
                    prepare_tokenizer = getattr(
                        transaction.manager,
                        "prepare_pipeline_tokenizer",
                        None,
                    )
                    result = (
                        prepare_tokenizer()
                        if callable(prepare_tokenizer) else None
                    )
                else:
                    result = transaction.manager.load_layer_range(
                        start_layer=normalized_start,
                        end_layer=normalized_end,
                        has_embedding=bool(has_embedding),
                        has_lm_head=bool(has_lm_head),
                        model_path=transaction.model_path,
                        quant_type=(load_assignment["load_spec"]["quant_type"] if load_assignment else transaction.quant_type),
                        total_layers=total_layers,
                        model_id=model_id or transaction.model_id,
                        **({"load_spec": load_assignment["load_spec"]} if load_assignment else {}),
                    )
                    if load_assignment:
                        from qlh_core.adapters.runtime_observation import observe_runtime
                        result = dict(result or {"success": True})
                        result["runtime_receipt"] = observe_runtime(transaction.manager, load_assignment)
            except Exception as exc:
                manager = transaction.manager
                cleanup_error = self._abort_failed_pipeline_candidate(
                    transaction, manager, exc,
                )
                if cleanup_error is not None:
                    raise RuntimeError(
                        "MODEL_TXN_CLEANUP_FAILED: candidate cleanup failed"
                    ) from cleanup_error
                raise
            if result is None:
                result = {
                    "success": True,
                    "layer_start": normalized_start,
                    "layer_end": normalized_end,
                    "tokenizer_only": normalized_start is None,
                }
            transaction.materialization_key = materialization_key
            transaction.materialization_result = result
            transaction.phase = "materialized"
            return result

    @_runtime_mutation
    def commit_pipeline_candidate(self, transaction_id: str) -> dict:
        """Atomically publish the materialized manager, retaining rollback state."""

        lock = object.__getattribute__(self, "_pipeline_candidate_lock")
        # API unload/invalidation already owns full-chat before aborting a
        # candidate. Direct commit/abort must follow the same lifecycle order.
        with object.__getattribute__(self, "full_chat_execution_lock"), lock, self.runtime_controller.publication():
            transaction = self._require_pipeline_candidate(transaction_id)
            if transaction.phase in {"committed", "finalized"}:
                return self.pipeline_candidate_status(transaction_id)
            if transaction.phase != "materialized" or transaction.manager is None:
                raise RuntimeError(
                    f"MODEL_TXN_STATE: transaction {transaction_id!r} "
                    f"cannot commit from {transaction.phase!r}"
                )
            candidate_host_state = self._pipeline_candidate_host_state(transaction)
            transaction.previous_manager = object.__getattribute__(self, "_manager")
            transaction.previous_host_state = self._snapshot_pipeline_host_state()
            object.__setattr__(self, "_manager", transaction.manager)
            self._apply_pipeline_candidate_host_state(candidate_host_state)
            transaction.phase = "committed"
            return self.pipeline_candidate_status(transaction_id)

    @_runtime_mutation
    def abort_pipeline_candidate(self, transaction_id: str) -> dict:
        """Discard an uncommitted candidate or restore the exact prior runtime."""

        lock = object.__getattribute__(self, "_pipeline_candidate_lock")
        with object.__getattribute__(self, "full_chat_execution_lock"), lock, self.runtime_controller.publication():
            transaction = self._require_pipeline_candidate(transaction_id)
            if transaction.phase == "aborted":
                return self.pipeline_candidate_status(transaction_id)
            if transaction.phase == "finalized":
                raise RuntimeError(
                    f"MODEL_TXN_STATE: transaction {transaction_id!r} is finalized"
                )
            manager = transaction.manager
            if transaction.phase == "committed":
                object.__setattr__(self, "_manager", transaction.previous_manager)
                self._restore_pipeline_host_state(
                    transaction.previous_host_state or {},
                )
            self._close_pipeline_manager(manager)
            transaction.manager = None
            transaction.previous_manager = None
            transaction.previous_host_state = None
            transaction.phase = "aborted"
            return self.pipeline_candidate_status(transaction_id)

    @_runtime_mutation
    def finalize_pipeline_candidate(self, transaction_id: str) -> dict:
        """Make a committed candidate permanent and release the prior runtime."""

        lock = object.__getattribute__(self, "_pipeline_candidate_lock")
        with lock:
            transaction = self._require_pipeline_candidate(transaction_id)
            if transaction.phase == "finalized":
                return self.pipeline_candidate_status(transaction_id)
            if transaction.phase != "committed":
                raise RuntimeError(
                    f"MODEL_TXN_STATE: transaction {transaction_id!r} "
                    f"cannot finalize from {transaction.phase!r}"
                )
            previous_manager = transaction.previous_manager
            self._close_pipeline_manager(previous_manager)
            transaction.previous_manager = None
            transaction.previous_host_state = None
            transaction.phase = "finalized"
            return self.pipeline_candidate_status(transaction_id)

    # ------------------------------------------------------------ engine dispatch
    @_runtime_mutation
    def load_model(
        self,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
        model_id: str = None,
        engine: str = None,
        db_experimental_models: list = None,
        resolution: ModelLoadResolution | None = None,
    ) -> None:
        """Load a model, routing the GGUF engine *around* the torch-backed manager.

        The GGUF/llama.cpp engine has its own implementation (``llama_engine``) and needs
        no PyTorch, so it is loaded directly here and installed as this host's manager.
        Every other engine keeps using ModelManager unchanged.

        Routing GGUF through ModelManager would import ``model_module`` -- which imports
        torch at module scope -- leaving the L-tier (no-torch) environment unable to load
        a model at all.  Argument order matches ModelManager.load_model, so existing
        positional and keyword callers keep working.
        """

        requested_engine = normalize_backend_request(engine or "auto")
        if requested_engine == "auto":
            requested_engine = normalize_backend_request(self.select_engine(profile))
        if requested_engine == "llama_cpp":
            self._load_gguf_model(model_path=model_path, model_id=model_id, profile=profile)
            return
        if requested_engine == "pytorch" and str(model_path or "").lower().endswith(".gguf"):
            raise ValueError("PyTorch loader 拒绝 GGUF 路径")
        manager = self._materialize_manager()
        manager.load_model(
            model_path=model_path,
            quant_type=quant_type,
            profile=profile,
            model_id=model_id,
            engine=engine,
            db_experimental_models=db_experimental_models,
            resolution=resolution,
        )

    @_runtime_mutation
    def prepare_pipeline_model(
        self,
        model_id: str,
        model_path: str,
        quant_type: str = None,
        layer_range: tuple[int, int] | None = None,
        model_sha256: str | None = None,
    ) -> dict:
        """Prepare directory or GGUF pipeline metadata without a full load.

        GGUF is routed directly to the import-safe llama.cpp engine, so the
        edge distribution does not import the torch-backed ``ModelManager``.
        """
        import os

        resolved_path = os.path.abspath(model_path or "")
        if os.path.isfile(resolved_path):
            from llama_engine import LlamaCppEngine

            current = self.peek_manager()
            if current is not None:
                unload = (
                    getattr(current, "unload_model", None)
                    or getattr(current, "unload", None)
                )
                if callable(unload):
                    unload()
            engine_obj = LlamaCppEngine()
            engine_obj._environment = self._runtime_environment
            self._configure_manager_catalog(engine_obj, model_id=model_id)
            descriptor = engine_obj.prepare_pipeline_model(
                model_id=model_id,
                model_path=resolved_path,
                quant_type=quant_type,
                layer_range=layer_range,
                model_sha256=model_sha256,
            )
            object.__setattr__(self, "_manager", engine_obj)
            object.__setattr__(self, "model_loaded", False)
            object.__setattr__(self, "active_model_id", model_id)
            object.__setattr__(self, "_active_model_id", model_id)
            object.__setattr__(self, "_engine_type", "llama_cpp")
            object.__setattr__(self, "quant_type", quant_type or "GGUF")
            object.__setattr__(self, "model_path", resolved_path)
            object.__setattr__(self, "current_quant", quant_type or "gguf")
            return descriptor

        manager = self.peek_manager()
        prepare = (
            getattr(manager, "prepare_pipeline_model", None)
            if manager is not None else None
        )
        if manager is not None and (
            not callable(prepare)
            or backend_id_for(manager, default="") == "llama_cpp"
        ):
            unload = (
                getattr(manager, "unload_model", None)
                or getattr(manager, "unload", None)
            )
            if callable(unload):
                unload()
            object.__setattr__(self, "_manager", _LazyModelManager())
            for name in (
                "_engine_type", "quant_type", "model_path",
                "active_model_id", "_active_model_id",
            ):
                object.__getattribute__(self, "__dict__").pop(name, None)
        manager = self._materialize_manager()
        descriptor = manager.prepare_pipeline_model(
            model_id=model_id,
            model_path=resolved_path,
            quant_type=quant_type,
            layer_range=layer_range,
            model_sha256=model_sha256,
        )
        object.__setattr__(self, "model_loaded", False)
        if quant_type:
            object.__setattr__(self, "current_quant", quant_type)
        return descriptor

    def _load_gguf_model(self, *, model_path: str = None, model_id: str = None, profile: dict = None) -> None:
        """Load a GGUF file via llama_engine and make it this host's manager."""

        import os

        from llama_engine import LlamaCppEngine, get_gguf_model_path

        try:
            from config import GGUF_MODEL_PATH
        except Exception:  # pragma: no cover - config defines it in practice
            GGUF_MODEL_PATH = ""

        if model_path:
            if not str(model_path).lower().endswith(".gguf"):
                raise ValueError("llama.cpp loader 只接受显式 GGUF 文件路径")
            gguf_path = model_path
        elif GGUF_MODEL_PATH and os.path.isfile(GGUF_MODEL_PATH):
            gguf_path = GGUF_MODEL_PATH
        else:
            gguf_path = get_gguf_model_path()
        if not gguf_path or not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF 模型文件未找到: {gguf_path or '(未解析到路径)'}")

        resolved_id = model_id
        if not resolved_id:
            try:
                from model_config import get_builtin_models, resolve_model_path

                absolute_path = os.path.abspath(gguf_path)
                for candidate in get_builtin_models():
                    if candidate.gguf_path and os.path.abspath(
                        resolve_model_path(candidate.gguf_path),
                    ) == absolute_path:
                        resolved_id = candidate.model_id
                        break
            except Exception:
                resolved_id = None

        model_metadata = {}
        if resolved_id:
            try:
                from model_config import get_model_profile_metadata

                model_metadata = get_model_profile_metadata(resolved_id)
            except Exception:
                model_metadata = {}

        tier = (profile or {}).get("tier", "laptop")
        n_ctx = {"edge": 1024, "mobile": 512, "ultrabook": 2048}.get(tier, 4096)

        engine_obj = LlamaCppEngine()
        engine_obj._environment = self._runtime_environment
        self._configure_manager_catalog(engine_obj, model_id=resolved_id or "")
        engine_obj.load_model(
            model_path=gguf_path,
            context_limit=n_ctx,
            model_profile=model_metadata,
        )
        # Install the engine as the manager: chat/chat_stream/is_loaded are then proxied
        # to it unchanged (their signatures already match ModelManager's).
        object.__setattr__(self, "_manager", engine_obj)
        object.__setattr__(self, "model_loaded", True)
        # Callers (api_server.get_status, scheduler) read these manager-side fields, so
        # mirror them here.  __getattr__ only fires when the instance dict misses, so
        # setting them on the instance keeps the proxy working normally otherwise.
        resolved_id = resolved_id or gguf_path
        object.__setattr__(self, "active_model_id", resolved_id)
        object.__setattr__(self, "_active_model_id", resolved_id)
        object.__setattr__(self, "_engine_type", "llama_cpp")
        object.__setattr__(self, "quant_type", "gguf")
        object.__setattr__(self, "model_path", gguf_path)
        object.__setattr__(self, "current_quant", "gguf")

    def _gguf_available(self) -> bool:
        """Whether a GGUF file can be resolved *without* importing torch."""

        import os

        try:
            from config import GGUF_MODEL_PATH
            from llama_engine import get_gguf_model_path
        except Exception:  # pragma: no cover - llama_engine is import-safe
            return False
        if GGUF_MODEL_PATH and os.path.isfile(GGUF_MODEL_PATH):
            return True
        return bool(get_gguf_model_path())

    @_runtime_mutation
    def switch_model(
        self,
        model_id: str = None,
        quant_type: str = None,
        profile: dict = None,
        engine: str = None,
        model_path: str = None,
        db_experimental_models: list = None,
        resolution: ModelLoadResolution | None = None,
    ) -> dict:
        """Switch models, keeping the GGUF engine on the torch-free path.

        ``api_server`` calls ``switch_model`` (for its rollback protection), not
        ``load_model``, so the GGUF routing has to live here too.  ``engine=None``/"auto"
        asks the backend to decide: when a GGUF file resolves *and* torch is absent, GGUF
        is the only engine that can actually run, so prefer it.  With torch present the
        D-tier default is kept exactly as before.
        """

        import importlib.util

        requested = normalize_backend_request(engine or "auto")
        wants_gguf = requested == "llama_cpp"
        if not wants_gguf and requested == "auto":
            torch_absent = importlib.util.find_spec("torch") is None
            wants_gguf = torch_absent and self._gguf_available()
        if wants_gguf:
            self._load_gguf_model(model_path=model_path, model_id=model_id, profile=profile)
            # Mirror ModelManager.switch_model's contract -- api_server reads "success".
            return {
                "success": True,
                "model_id": model_id,
                "model_name": model_id,
                "error": None,
            }
        if requested == "pytorch" and str(model_path or "").lower().endswith(".gguf"):
            raise ValueError("PyTorch loader 拒绝 GGUF 路径")
        # ★ 切回非 GGUF 引擎前，先确认 `_manager` 仍是模型管理器：`_load_gguf_model`
        #   会把 `LlamaCppEngine` 装成 `_manager`（它没有 `switch_model`），此时直接
        #   转发会抛 AttributeError。
        manager = self._materialize_manager()
        if not hasattr(manager, "switch_model"):
            if hasattr(manager, "unload"):
                manager.unload()
            object.__setattr__(self, "_manager", _LazyModelManager())
            object.__setattr__(self, "model_loaded", False)
            # `_load_gguf_model` 用 object.__setattr__ 把这些镜像属性写进了**实例字典**
            # （绕过 __getattr__ 代理），读取时命中实例字典。留着会把上一档的值带进
            # 新引擎 —— 尤其 `_engine_type` 停在 "llama_cpp" 时，api_server 会按 GGUF
            # 分支处理请求，把请求 kwargs 原样喂给 torch。移除后恢复代理语义。
            for _name in ("_engine_type", "quant_type", "model_path",
                          "active_model_id", "_active_model_id"):
                object.__getattribute__(self, "__dict__").pop(_name, None)
            manager = self._materialize_manager()
        return manager.switch_model(
            model_id=model_id,
            quant_type=quant_type,
            profile=profile,
            engine=engine,
            model_path=model_path,
            db_experimental_models=db_experimental_models,
            resolution=resolution,
        )

    @_runtime_mutation
    def unload_model(self) -> None:
        """Unload whichever engine is active -- GGUF (direct) or ModelManager."""

        manager = object.__getattribute__(self, "_manager")
        if isinstance(manager, _LazyModelManager):
            manager = object.__getattribute__(manager, "_instance")
        if manager is None:
            pass
        elif hasattr(manager, "unload_model"):  # ModelManager
            manager.unload_model()
        elif hasattr(manager, "unload"):  # LlamaCppEngine
            manager.unload()
        object.__setattr__(self, "model_loaded", False)
        self.runtime_controller.clear_metadata()
        for name in (
            "_engine_type", "quant_type", "model_path",
            "active_model_id", "_active_model_id",
        ):
            object.__getattribute__(self, "__dict__").pop(name, None)




    def __repr__(self):
        return "<ModelHost manager=%r model_loaded=%s>" % (
            self._manager, self.model_loaded)


# 全局单例：api_server 与 scheduler 共享（0.4）
model_host = ModelHost()


def get_model_host() -> ModelHost:
    """返回全局单例（显式获取入口，避免 from-import 时绑定旧实例）。"""
    return model_host
