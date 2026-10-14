"""Execution-provider contracts and the local full-model provider."""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from koakuma_engine import normalize_backend_request
from request_deadline import RequestDeadline


PROVIDER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
RESERVATION_ID_PATTERN = re.compile(r"^res_[A-Za-z0-9_-]{8,96}$")
MODEL_IDENTITY_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
MODEL_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCY_FAILURES_KEY = "__qlh_failed_dependencies__"

_NON_LOCAL_MODEL_ENGINES = frozenset({
    "island", "external_api", "speculative_assisted",
})


def canonical_model_engine(value: object) -> str:
    """Return the canonical task/pipeline engine identity or fail closed.

    Only local runtime aliases are folded.  Non-local execution identities are
    already protocol-level identifiers and therefore pass through unchanged.
    """

    raw = str(value or "").strip()
    folded = raw.casefold()
    if folded in _NON_LOCAL_MODEL_ENGINES:
        return folded
    try:
        return normalize_backend_request(raw, allow_auto=False)
    except ValueError as exc:
        raise ValueError("model identity engine is unsupported") from exc


class _CombinedCancelEvent(threading.Event):
    def __init__(
        self,
        workflow_event: threading.Event,
        request_deadline: Optional[RequestDeadline] = None,
    ):
        super().__init__()
        self._workflow_event = workflow_event
        self._request_deadline = request_deadline

    def set(self) -> None:
        """Preserve the callback contract that setting this cancels workflow."""
        self._workflow_event.set()
        super().set()

    def set_local(self) -> None:
        """Stop only this Provider attempt without inventing a user cancel."""
        super().set()

    def is_set(self) -> bool:
        return bool(
            super().is_set()
            or self._workflow_event.is_set()
            or (
                self._request_deadline is not None
                and self._request_deadline.expired()
            )
        )

    def wait(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                return False
            super().wait(
                0.05 if remaining is None else min(0.05, remaining)
            )
        return True


class ProviderError(RuntimeError):
    """Base error with stable fields suitable for workflow error handling."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        provider_id: str = "",
        retryable: bool = False,
        same_provider_retryable: bool = False,
    ):
        self.code = code
        self.provider_id = provider_id
        self.retryable = retryable
        self.same_provider_retryable = same_provider_retryable
        super().__init__(message)


class ProviderRegistrationError(ProviderError):
    pass


class ProviderNotFound(ProviderError):
    pass


class ProviderUnavailable(ProviderError):
    pass


class ProviderBusy(ProviderError):
    pass


class ProviderExecutionError(ProviderError):
    pass


class ProviderReservationError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderCapabilities:
    provider_id: str
    provider_kind: str
    supported_stage_types: tuple[str, ...]
    max_concurrency: int
    active_reservations: int
    healthy: bool
    available: bool
    node_id: str = ""
    updated_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "supported_stage_types": list(self.supported_stage_types),
            "max_concurrency": self.max_concurrency,
            "active_reservations": self.active_reservations,
            "healthy": self.healthy,
            "available": self.available,
            "node_id": self.node_id,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    engine: str
    format: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        # "island": TP 孤岛引擎（OpenAI 兼容端点整请求转发），
        # 其 sha256 为端点指纹摘要而非本地文件摘要（见调研方案 §2.2）。
        # "external_api": 外部推理服务（路线 B，集群信任域之外的 OpenAI 兼容
        # 端点），同样以"端点指纹 + 服务端模型名"的摘要替代本地文件摘要。
        # "speculative_assisted": 投机解码（路线 C-1，本地 draft + 外部 verify，
        # 见调研方案 §2.3），无单一本地 artifact，指纹 = draft 模型 + verify
        # 端点 + verify 模型名；输出分布等于 verify 模型。
        object.__setattr__(self, "engine", canonical_model_engine(self.engine))
        if any(
            MODEL_IDENTITY_VALUE_PATTERN.fullmatch(value) is None
            for value in (self.model_id, self.format, self.revision)
        ):
            raise ValueError("model identity contains an invalid value")
        if MODEL_SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("model identity sha256 is invalid")

    def snapshot(self) -> dict:
        return {
            "model_id": self.model_id,
            "engine": self.engine,
            "format": self.format,
            "revision": self.revision,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class StageRequest:
    workflow_id: str
    request_id: str
    stage_id: str
    stage_type: str
    provider_id: str
    dependencies: dict[str, dict]
    root_input: dict
    model_identity: Optional[ModelIdentity] = None
    runtime_context: dict = field(default_factory=dict, compare=False, repr=False)
    # Protocol fields that belong to the selected stage, rather than its root
    # input.  Layer-forward requests use this to carry the hidden contract.
    stage_fields: dict = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    provider_id: str
    workflow_id: str
    stage_id: str
    created_at: float
    selection_reason: str
    provider_kind: str = ""
    provider_node_id: str = ""


@dataclass(frozen=True)
class StageAttempt:
    attempt_id: str
    request: StageRequest
    provider_id: str
    lease_id: str = ""
    lease_epoch: int = 0
    lease_expires_at: float = 0.0
    accept_timeout_seconds: float = 10.0
    request_deadline: Optional[RequestDeadline] = field(
        default=None, compare=False, repr=False,
    )


@dataclass(frozen=True)
class StageResult:
    output: dict
    provider_id: str
    metadata: dict = field(default_factory=dict)
    attempt_id: str = ""
    lease_epoch: int = 0


class ExecutionProvider(Protocol):
    provider_id: str

    def inspect(self) -> ProviderCapabilities: ...
    def reserve(self, request: StageRequest) -> Reservation: ...
    def execute(
        self,
        attempt: StageAttempt,
        reservation: Reservation,
        cancel_event: threading.Event,
    ) -> StageResult: ...
    def renew_lease(
        self,
        attempt_id: str,
        lease_id: str,
        lease_epoch: int,
        lease_expires_at: float,
    ) -> bool: ...
    def cancel(self, attempt_id: str) -> None: ...
    def release(self, reservation_id: str) -> None: ...
    def close(self) -> None: ...


ProviderExecutor = Callable[[StageRequest, threading.Event], dict]


def _safe_number(value) -> Optional[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sanitize_result_metadata(metadata: dict) -> dict:
    raw_metadata = metadata
    usage = {}
    raw_usage = raw_metadata.get("usage", {})
    if isinstance(raw_usage, dict):
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
        ):
            value = _safe_number(raw_usage.get(key))
            if value is not None:
                usage[key] = value
    sanitized = {
        "usage": usage,
        "usage_estimated": bool(raw_metadata.get("usage_estimated", False)),
    }
    tokens_per_second = _safe_number(raw_metadata.get("tokens_per_second"))
    if tokens_per_second is not None:
        sanitized["tokens_per_second"] = tokens_per_second
    model_id = str(raw_metadata.get("model", "") or "")[:256]
    if model_id:
        sanitized["model"] = model_id
    return sanitized


class LocalFullModelProvider:
    """Synchronous local model adapter with an atomic concurrency slot."""

    provider_id = "local_full_model"

    def __init__(
        self,
        executor: ProviderExecutor,
        *,
        provider_id: str = "local_full_model",
        node_id: str = "",
        supported_stage_types: tuple[str, ...] = (
            "full_inference",
            "aggregate",
        ),
        max_concurrency: int = 1,
        provider_kind: str = "local_full_model",
    ):
        if not PROVIDER_ID_PATTERN.fullmatch(provider_id):
            raise ProviderRegistrationError(
                "provider_id contains unsupported characters",
                code="invalid_provider_id",
                provider_id=provider_id,
            )
        self.provider_id = provider_id
        self._executor = executor
        self._node_id = str(node_id or "")
        self._supported_stage_types = tuple(dict.fromkeys(
            str(value) for value in supported_stage_types if str(value)
        ))
        self._max_concurrency = max(1, int(max_concurrency))
        self._provider_kind = str(provider_kind or "local_full_model")
        self._reservations: dict[str, Reservation] = {}
        self._executed_reservations: set[str] = set()
        self._active_attempts: dict[str, threading.Event] = {}
        self._healthy = True
        self._lock = threading.RLock()

    def inspect(self) -> ProviderCapabilities:
        with self._lock:
            active = len(self._reservations)
            return ProviderCapabilities(
                provider_id=self.provider_id,
                provider_kind=self._provider_kind,
                supported_stage_types=self._supported_stage_types,
                max_concurrency=self._max_concurrency,
                active_reservations=active,
                healthy=self._healthy,
                available=self._healthy and active < self._max_concurrency,
                node_id=self._node_id,
            )

    def reserve(self, request: StageRequest) -> Reservation:
        with self._lock:
            if not self._healthy:
                raise ProviderUnavailable(
                    f"provider {self.provider_id} is unavailable",
                    code="provider_unavailable",
                    provider_id=self.provider_id,
                    retryable=True,
                )
            if request.provider_id and request.provider_id != self.provider_id:
                raise ProviderReservationError(
                    "stage request targets a different provider",
                    code="provider_request_mismatch",
                    provider_id=self.provider_id,
                )
            if request.stage_type not in self._supported_stage_types:
                raise ProviderUnavailable(
                    f"provider {self.provider_id} does not support stage type "
                    f"{request.stage_type}",
                    code="unsupported_stage_type",
                    provider_id=self.provider_id,
                )
            if len(self._reservations) >= self._max_concurrency:
                raise ProviderBusy(
                    f"provider {self.provider_id} has no free reservation slot",
                    code="provider_busy",
                    provider_id=self.provider_id,
                    retryable=True,
                )
            reservation = Reservation(
                reservation_id=f"res_{uuid.uuid4().hex}",
                provider_id=self.provider_id,
                workflow_id=request.workflow_id,
                stage_id=request.stage_id,
                created_at=time.time(),
                selection_reason=(
                    "requested_provider" if request.provider_id
                    else "first_compatible_provider"
                ),
                provider_kind=self._provider_kind,
                provider_node_id=self._node_id,
            )
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    def execute(
        self,
        attempt: StageAttempt,
        reservation: Reservation,
        cancel_event: threading.Event,
    ) -> StageResult:
        attempt_cancel_event = _CombinedCancelEvent(
            cancel_event, attempt.request_deadline,
        )
        with self._lock:
            owned = self._reservations.get(reservation.reservation_id)
            if owned != reservation:
                raise ProviderReservationError(
                    "reservation is unknown or no longer active",
                    code="invalid_reservation",
                    provider_id=self.provider_id,
                )
            if (
                reservation.provider_id != self.provider_id
                or attempt.provider_id != self.provider_id
                or reservation.workflow_id != attempt.request.workflow_id
                or reservation.stage_id != attempt.request.stage_id
            ):
                raise ProviderReservationError(
                    "reservation does not match the stage attempt",
                    code="reservation_mismatch",
                    provider_id=self.provider_id,
                )
            if reservation.reservation_id in self._executed_reservations:
                raise ProviderReservationError(
                    "reservation has already been executed",
                    code="reservation_already_executed",
                    provider_id=self.provider_id,
                )
            self._executed_reservations.add(reservation.reservation_id)
            self._active_attempts[attempt.attempt_id] = attempt_cancel_event
        try:
            output = self._executor(attempt.request, attempt_cancel_event)
            if not isinstance(output, dict):
                raise ProviderExecutionError(
                    "provider output must be a dict",
                    code="invalid_provider_output",
                    provider_id=self.provider_id,
                )
            return StageResult(
                output=output,
                provider_id=self.provider_id,
                metadata=sanitize_result_metadata(output),
                attempt_id=attempt.attempt_id,
                lease_epoch=attempt.lease_epoch,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderExecutionError(
                f"provider {self.provider_id} execution failed: {exc}",
                code="provider_execution_failed",
                provider_id=self.provider_id,
            ) from exc
        finally:
            with self._lock:
                self._active_attempts.pop(attempt.attempt_id, None)

    def cancel(self, attempt_id: str) -> None:
        with self._lock:
            event = self._active_attempts.get(attempt_id)
            if event is not None:
                event.set_local()

    def renew_lease(
        self,
        attempt_id: str,
        lease_id: str,
        lease_epoch: int,
        lease_expires_at: float,
    ) -> bool:
        del attempt_id, lease_id, lease_epoch, lease_expires_at
        return False

    def release(self, reservation_id: str) -> None:
        with self._lock:
            self._reservations.pop(reservation_id, None)
            self._executed_reservations.discard(reservation_id)

    def close(self) -> None:
        with self._lock:
            self._healthy = False
            for event in self._active_attempts.values():
                event.set()
            self._active_attempts.clear()
            self._reservations.clear()
            self._executed_reservations.clear()


class CallbackExecutionProvider(LocalFullModelProvider):
    """Compatibility adapter for the pre-N1.3 StageExecutor interface."""

    def __init__(
        self,
        provider_id: str,
        executor: ProviderExecutor,
        supported_stage_types: tuple[str, ...],
    ):
        super().__init__(
            executor,
            provider_id=provider_id,
            supported_stage_types=supported_stage_types,
            provider_kind="callback_compatibility",
        )


class InProcessWorkerProvider(LocalFullModelProvider):
    """Worker-shaped provider that keeps transport inside the current process."""

    def __init__(
        self,
        provider_id: str,
        executor: ProviderExecutor,
        *,
        node_id: str = "",
        supported_stage_types: tuple[str, ...] = (
            "full_inference",
            "aggregate",
        ),
        max_concurrency: int = 1,
    ):
        super().__init__(
            executor,
            provider_id=provider_id,
            node_id=node_id,
            supported_stage_types=supported_stage_types,
            max_concurrency=max_concurrency,
            provider_kind="in_process_worker",
        )


class DeterministicFakeProvider(LocalFullModelProvider):
    """Controllable provider for deterministic concurrency and failure tests."""

    def __init__(
        self,
        provider_id: str,
        *,
        output_factory: Optional[ProviderExecutor] = None,
        node_id: str = "",
        supported_stage_types: tuple[str, ...] = (
            "full_inference",
            "aggregate",
        ),
        max_concurrency: int = 1,
        delay_seconds: float = 0.0,
        start_barrier: Optional[threading.Barrier] = None,
        block_event: Optional[threading.Event] = None,
        fail_stage_ids: tuple[str, ...] = (),
        accept_failures: int = 0,
        accept_error_code: str = "fake_accept_timeout",
        execution_failures: int = 0,
        execution_error_code: str = "fake_worker_disconnected",
    ):
        self._output_factory = output_factory
        self._delay_seconds = max(0.0, float(delay_seconds))
        self._start_barrier = start_barrier
        self._block_event = block_event
        self._fail_stage_ids = frozenset(fail_stage_ids)
        self._accept_failures = max(0, int(accept_failures))
        self._accept_error_code = str(
            accept_error_code or "fake_accept_timeout"
        )
        self._execution_failures = max(0, int(execution_failures))
        self._execution_error_code = str(
            execution_error_code or "fake_worker_disconnected"
        )
        self._call_records: list[dict] = []
        self._fake_lock = threading.RLock()
        super().__init__(
            self._execute_fake,
            provider_id=provider_id,
            node_id=node_id,
            supported_stage_types=supported_stage_types,
            max_concurrency=max_concurrency,
            provider_kind="deterministic_fake",
        )

    def reserve(self, request: StageRequest) -> Reservation:
        with self._fake_lock:
            if self._accept_failures > 0:
                self._accept_failures -= 1
                raise ProviderReservationError(
                    f"deterministic accept failure for stage {request.stage_id}",
                    code=self._accept_error_code,
                    provider_id=self.provider_id,
                    retryable=True,
                )
        return super().reserve(request)

    def _execute_fake(
        self,
        request: StageRequest,
        cancel_event: threading.Event,
    ) -> dict:
        record = {
            "workflow_id": request.workflow_id,
            "stage_id": request.stage_id,
            "started_at": time.time(),
            "finished_at": None,
            "state": "running",
        }
        with self._fake_lock:
            self._call_records.append(record)
        try:
            if self._start_barrier is not None:
                try:
                    self._start_barrier.wait(timeout=5.0)
                except threading.BrokenBarrierError as exc:
                    raise ProviderExecutionError(
                        "deterministic provider start barrier failed",
                        code="fake_barrier_failed",
                        provider_id=self.provider_id,
                    ) from exc
            if self._delay_seconds > 0 and cancel_event.wait(
                self._delay_seconds,
            ):
                raise ProviderExecutionError(
                    "deterministic provider execution cancelled",
                    code="provider_cancelled",
                    provider_id=self.provider_id,
                )
            while (
                self._block_event is not None
                and not self._block_event.wait(0.01)
            ):
                if cancel_event.is_set():
                    raise ProviderExecutionError(
                        "deterministic provider execution cancelled",
                        code="provider_cancelled",
                        provider_id=self.provider_id,
                    )
            if cancel_event.is_set():
                raise ProviderExecutionError(
                    "deterministic provider execution cancelled",
                    code="provider_cancelled",
                    provider_id=self.provider_id,
                )
            with self._fake_lock:
                fail_execution = self._execution_failures > 0
                if fail_execution:
                    self._execution_failures -= 1
            if fail_execution:
                raise ProviderExecutionError(
                    f"deterministic execution failure for stage "
                    f"{request.stage_id}",
                    code=self._execution_error_code,
                    provider_id=self.provider_id,
                    retryable=True,
                )
            if request.stage_id in self._fail_stage_ids:
                raise ProviderExecutionError(
                    f"deterministic failure for stage {request.stage_id}",
                    code="deterministic_provider_failure",
                    provider_id=self.provider_id,
                )
            if self._output_factory is not None:
                output = self._output_factory(request, cancel_event)
            else:
                output = {"content": request.stage_id}
            record["state"] = "completed"
            return output
        except Exception:
            record["state"] = (
                "cancelled" if cancel_event.is_set() else "failed"
            )
            raise
        finally:
            record["finished_at"] = time.time()

    def call_records(self) -> list[dict]:
        with self._fake_lock:
            return [dict(record) for record in self._call_records]


class ProviderRegistry:
    """Thread-safe provider lifecycle and reservation owner."""

    def __init__(self):
        self._providers: dict[str, ExecutionProvider] = {}
        self._reservations: dict[str, tuple[ExecutionProvider, Reservation]] = {}
        self._executed_reservations: set[str] = set()
        self._closed = False
        self._lock = threading.RLock()

    def register(self, provider: ExecutionProvider) -> None:
        provider_id = str(getattr(provider, "provider_id", "") or "")
        if not PROVIDER_ID_PATTERN.fullmatch(provider_id):
            raise ProviderRegistrationError(
                "provider_id contains unsupported characters",
                code="invalid_provider_id",
                provider_id=provider_id,
            )
        try:
            capabilities = provider.inspect()
        except Exception as exc:
            raise ProviderRegistrationError(
                f"provider {provider_id} capability inspection failed",
                code="provider_inspection_failed",
                provider_id=provider_id,
            ) from exc
        if capabilities.provider_id != provider_id:
            raise ProviderRegistrationError(
                "provider capability identity does not match provider_id",
                code="provider_identity_mismatch",
                provider_id=provider_id,
            )
        with self._lock:
            if self._closed:
                raise ProviderRegistrationError(
                    "provider registry is closed",
                    code="provider_registry_closed",
                    provider_id=provider_id,
                )
            if provider_id in self._providers:
                raise ProviderRegistrationError(
                    f"provider_id already registered: {provider_id}",
                    code="duplicate_provider_id",
                    provider_id=provider_id,
                )
            self._providers[provider_id] = provider

    def replace(self, provider: ExecutionProvider) -> None:
        """Replace an idle Provider while preserving its stable identity."""
        provider_id = str(getattr(provider, "provider_id", "") or "")
        if not PROVIDER_ID_PATTERN.fullmatch(provider_id):
            raise ProviderRegistrationError(
                "provider_id contains unsupported characters",
                code="invalid_provider_id",
                provider_id=provider_id,
            )
        try:
            capabilities = provider.inspect()
        except Exception as exc:
            raise ProviderRegistrationError(
                f"provider {provider_id} capability inspection failed",
                code="provider_inspection_failed",
                provider_id=provider_id,
            ) from exc
        if capabilities.provider_id != provider_id:
            raise ProviderRegistrationError(
                "provider capability identity does not match provider_id",
                code="provider_identity_mismatch",
                provider_id=provider_id,
            )
        previous = None
        with self._lock:
            if self._closed:
                raise ProviderRegistrationError(
                    "provider registry is closed",
                    code="provider_registry_closed",
                    provider_id=provider_id,
                )
            previous = self._providers.get(provider_id)
            if previous is provider:
                return
            if any(
                reservation.provider_id == provider_id
                for _owner, reservation in self._reservations.values()
            ):
                raise ProviderBusy(
                    f"provider {provider_id} still owns active reservations",
                    code="provider_has_active_reservations",
                    provider_id=provider_id,
                    retryable=True,
                )
            self._providers[provider_id] = provider
        if previous is not None:
            previous.close()

    def has_provider(self, provider_id: str) -> bool:
        with self._lock:
            return provider_id in self._providers

    def provider_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._providers)

    def unregister(self, provider_id: str) -> bool:
        with self._lock:
            provider = self._providers.get(provider_id)
            if provider is None:
                return False
            if any(
                reservation.provider_id == provider_id
                for _provider, reservation in self._reservations.values()
            ):
                raise ProviderBusy(
                    f"provider {provider_id} still owns active reservations",
                    code="provider_has_active_reservations",
                    provider_id=provider_id,
                    retryable=True,
                )
            self._providers.pop(provider_id)
        provider.close()
        return True

    def inspect(self) -> list[dict]:
        with self._lock:
            providers = [
                (provider_id, self._providers[provider_id])
                for provider_id in sorted(self._providers)
            ]
        statuses = []
        for provider_id, provider in providers:
            try:
                capabilities = provider.inspect()
                if capabilities.provider_id != provider_id:
                    raise ValueError("provider identity mismatch")
                statuses.append(capabilities.snapshot())
            except Exception:
                statuses.append({
                    "provider_id": provider_id,
                    "provider_kind": "unknown",
                    "supported_stage_types": [],
                    "max_concurrency": 0,
                    "active_reservations": 0,
                    "healthy": False,
                    "available": False,
                    "node_id": "",
                    "updated_at": time.time(),
                    "error_code": "provider_inspection_failed",
                })
        return statuses

    @staticmethod
    def _supports(capabilities: ProviderCapabilities, stage_type: str) -> bool:
        return stage_type in capabilities.supported_stage_types

    @staticmethod
    def _inspect_for_selection(
        provider: ExecutionProvider,
    ) -> ProviderCapabilities:
        try:
            capabilities = provider.inspect()
        except Exception as exc:
            raise ProviderUnavailable(
                f"provider {provider.provider_id} capability inspection failed",
                code="provider_inspection_failed",
                provider_id=provider.provider_id,
                retryable=True,
            ) from exc
        if (
            capabilities.provider_id != provider.provider_id
            or capabilities.max_concurrency < 1
            or capabilities.active_reservations < 0
        ):
            raise ProviderUnavailable(
                f"provider {provider.provider_id} returned invalid capabilities",
                code="invalid_provider_capabilities",
                provider_id=provider.provider_id,
            )
        return capabilities

    def _reserve_once(
        self,
        request: StageRequest,
        abandoned: threading.Event,
        operation: dict,
    ) -> Reservation:
        with self._lock:
            if self._closed:
                raise ProviderUnavailable(
                    "provider registry is closed",
                    code="provider_registry_closed",
                )
            if request.provider_id:
                provider = self._providers.get(request.provider_id)
                if provider is None:
                    raise ProviderNotFound(
                        f"provider is not registered: {request.provider_id}",
                        code="provider_not_found",
                        provider_id=request.provider_id,
                    )
                candidates = [provider]
            else:
                candidates = [
                    self._providers[key] for key in sorted(self._providers)
                ]

        compatible = []
        inspection_errors = []
        for provider in candidates:
            if abandoned.is_set():
                raise ProviderReservationError(
                    "provider reservation acceptance timed out",
                    code="provider_accept_timeout",
                    provider_id=request.provider_id,
                    retryable=True,
                )
            try:
                capabilities = self._inspect_for_selection(provider)
            except ProviderUnavailable as exc:
                inspection_errors.append(exc)
                continue
            if self._supports(capabilities, request.stage_type):
                compatible.append((provider, capabilities))
        if not compatible:
            if request.provider_id and inspection_errors:
                raise inspection_errors[0]
            raise ProviderUnavailable(
                f"no provider supports stage type {request.stage_type}",
                code="no_compatible_provider",
                provider_id=request.provider_id,
            )
        healthy = [item for item in compatible if item[1].healthy]
        if not healthy:
            raise ProviderUnavailable(
                "all compatible providers are unhealthy",
                code="all_providers_unhealthy",
                provider_id=request.provider_id,
                retryable=True,
            )
        available = [item for item in healthy if item[1].available]
        if not available:
            raise ProviderBusy(
                "all compatible providers are busy or unhealthy",
                code="all_providers_busy",
                provider_id=request.provider_id,
                retryable=True,
            )
        provider = available[0][0]
        reservation = provider.reserve(request)
        rejection: Optional[ProviderReservationError] = None
        if (
            not RESERVATION_ID_PATTERN.fullmatch(reservation.reservation_id)
            or reservation.provider_id != provider.provider_id
            or reservation.workflow_id != request.workflow_id
            or reservation.stage_id != request.stage_id
        ):
            rejection = ProviderReservationError(
                "provider returned a reservation for a different request",
                code="reservation_identity_mismatch",
                provider_id=provider.provider_id,
            )
        with self._lock:
            if rejection is None and abandoned.is_set():
                rejection = ProviderReservationError(
                    "provider reservation acceptance timed out",
                    code="provider_accept_timeout",
                    provider_id=provider.provider_id,
                    retryable=True,
                )
            if rejection is None and (
                self._closed
                or self._providers.get(provider.provider_id) is not provider
            ):
                rejection = ProviderReservationError(
                    "provider registry changed while accepting reservation",
                    code="provider_registry_changed",
                    provider_id=provider.provider_id,
                    retryable=True,
                )
            if (
                rejection is None
                and reservation.reservation_id in self._reservations
            ):
                rejection = ProviderReservationError(
                    "provider returned a duplicate reservation_id",
                    code="duplicate_reservation_id",
                    provider_id=provider.provider_id,
                )
            if rejection is None:
                self._reservations[reservation.reservation_id] = (
                    provider,
                    reservation,
                )
                operation["reservation"] = reservation
                return reservation
        if rejection is None:
            raise ProviderReservationError(
                "provider reservation could not be committed",
                code="reservation_commit_failed",
                provider_id=provider.provider_id,
            )
        try:
            provider.release(reservation.reservation_id)
        except Exception as cleanup_exc:
            rejection.add_note(
                f"late reservation cleanup also failed: {cleanup_exc}"
            )
        raise rejection

    def reserve(
        self,
        request: StageRequest,
        timeout_seconds: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Reservation:
        abandoned = threading.Event()
        operation: dict = {}
        if timeout_seconds is None:
            return self._reserve_once(request, abandoned, operation)
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        done = threading.Event()
        result: dict = {}

        def accept_reservation() -> None:
            try:
                result["reservation"] = self._reserve_once(
                    request, abandoned, operation,
                )
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(
            target=accept_reservation,
            name=f"provider-accept-{request.stage_id}",
            daemon=True,
        ).start()
        deadline = time.monotonic() + timeout
        timeout_code = "provider_accept_timeout"
        while True:
            remaining = deadline - time.monotonic()
            if done.wait(max(0.0, min(0.05, remaining))):
                error = result.get("error")
                if isinstance(error, BaseException):
                    raise error
                reservation = result.get("reservation")
                if isinstance(reservation, Reservation):
                    return reservation
                raise ProviderReservationError(
                    "provider reservation returned no result",
                    code="invalid_reservation_result",
                    provider_id=request.provider_id,
                )
            if cancel_event is not None and cancel_event.is_set():
                timeout_code = "provider_accept_cancelled"
                break
            if remaining <= 0:
                timeout_code = "provider_accept_timeout"
                break
        with self._lock:
            committed = operation.get("reservation")
            if committed is None:
                abandoned.set()
        if isinstance(committed, Reservation):
            return committed
        raise ProviderReservationError(
            "provider reservation acceptance did not complete",
            code=timeout_code,
            provider_id=request.provider_id,
            retryable=timeout_code == "provider_accept_timeout",
        )

    def execute(
        self,
        attempt: StageAttempt,
        reservation: Reservation,
        cancel_event: threading.Event,
    ) -> StageResult:
        with self._lock:
            entry = self._reservations.get(reservation.reservation_id)
            if entry is None or entry[1] != reservation:
                raise ProviderReservationError(
                    "reservation is not owned by this registry",
                    code="invalid_reservation",
                    provider_id=reservation.provider_id,
                )
            provider = entry[0]
            if (
                attempt.provider_id != reservation.provider_id
                or attempt.request.workflow_id != reservation.workflow_id
                or attempt.request.stage_id != reservation.stage_id
            ):
                raise ProviderReservationError(
                    "stage attempt does not match its reservation",
                    code="attempt_reservation_mismatch",
                    provider_id=reservation.provider_id,
                )
            if reservation.reservation_id in self._executed_reservations:
                raise ProviderReservationError(
                    "reservation has already been executed",
                    code="reservation_already_executed",
                    provider_id=reservation.provider_id,
                )
            self._executed_reservations.add(reservation.reservation_id)
        result = provider.execute(attempt, reservation, cancel_event)
        if result.provider_id != reservation.provider_id:
            raise ProviderExecutionError(
                "provider result identity does not match reservation",
                code="provider_result_identity_mismatch",
                provider_id=reservation.provider_id,
            )
        if not isinstance(result.output, dict) or not isinstance(
            result.metadata, dict,
        ):
            raise ProviderExecutionError(
                "provider result has an invalid schema",
                code="invalid_provider_result",
                provider_id=reservation.provider_id,
            )
        legacy_identity_missing = (
            not result.attempt_id and result.lease_epoch == 0
        )
        return StageResult(
            output=result.output,
            provider_id=result.provider_id,
            metadata=sanitize_result_metadata(result.metadata),
            attempt_id=(
                attempt.attempt_id
                if legacy_identity_missing else result.attempt_id
            ),
            lease_epoch=(
                attempt.lease_epoch
                if legacy_identity_missing else result.lease_epoch
            ),
        )

    def cancel(self, provider_id: str, attempt_id: str) -> None:
        with self._lock:
            provider = self._providers.get(provider_id)
        if provider is not None:
            provider.cancel(attempt_id)

    def discard_result(
        self,
        provider_id: str,
        attempt_id: str,
        output: dict,
    ) -> bool:
        """Ask an owning Provider to release an uncommitted external result."""
        with self._lock:
            provider = self._providers.get(provider_id)
        if provider is None:
            raise ProviderNotFound(
                f"provider not found: {provider_id}",
                code="provider_not_found",
                provider_id=provider_id,
            )
        discard = getattr(provider, "discard_result", None)
        if not callable(discard):
            return False
        try:
            discard(attempt_id, output)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderExecutionError(
                f"provider {provider_id} failed to discard an uncommitted result",
                code="provider_result_discard_failed",
                provider_id=provider_id,
            ) from exc
        return True

    def renew_lease(
        self,
        provider_id: str,
        attempt_id: str,
        lease_id: str,
        lease_epoch: int,
        lease_expires_at: float,
    ) -> bool:
        """Ask a Provider to extend a lease on its transport, if supported."""
        with self._lock:
            provider = self._providers.get(provider_id)
        if provider is None:
            raise ProviderNotFound(
                f"provider not found: {provider_id}",
                code="provider_not_found",
                provider_id=provider_id,
            )
        renew = getattr(provider, "renew_lease", None)
        if not callable(renew):
            return False
        return bool(renew(
            attempt_id,
            lease_id,
            int(lease_epoch),
            float(lease_expires_at),
        ))

    def release(self, reservation_id: str) -> None:
        with self._lock:
            entry = self._reservations.pop(reservation_id, None)
            if entry is None:
                return
            provider, _reservation = entry
            self._executed_reservations.discard(reservation_id)
        try:
            provider.release(reservation_id)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderReservationError(
                f"provider {provider.provider_id} failed to release reservation",
                code="provider_release_failed",
                provider_id=provider.provider_id,
            ) from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            reservations = list(self._reservations.items())
            providers = list(self._providers.values())
            self._reservations.clear()
            self._providers.clear()
            self._executed_reservations.clear()
        first_error: Optional[Exception] = None
        for reservation_id, (provider, _reservation) in reservations:
            try:
                provider.release(reservation_id)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        for provider in providers:
            try:
                provider.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise ProviderReservationError(
                "provider registry close encountered a cleanup error",
                code="provider_close_failed",
            ) from first_error
