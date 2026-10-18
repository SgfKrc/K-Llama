"""Koakuma engine boundary and backend selection.

This module is deliberately dependency-light.  It describes the common engine
contract and chooses a backend from a node profile without importing either
runtime implementation.  The actual llama.cpp and PyTorch implementations
remain behind the host/manager adapters for now.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from types import MappingProxyType

from qlh_core.contracts.engines import (
    COMMON_CAPABILITIES, ENGINE_REGISTRY, LEGACY_ENGINE_SPECS,
    LLAMA_CPP_BINDING, LLAMA_CPP_KEEP_HEAD, PYTORCH_CPU, PYTORCH_CUDA, ISLAND_REMOTE,
)
from typing import Any, Mapping, Protocol, FrozenSet


class BackendId:
    """Canonical backend identifiers used in API and model contracts."""

    LLAMA_CPP = "llama_cpp"
    PYTORCH = "pytorch"
    ISLAND = "island"


class Capability:
    CHAT = "chat"
    CHAT_STREAM = "chat_stream"
    CHAT_IMAGE = "chat_image"
    LOAD_MODEL = "load_model"
    UNLOAD_MODEL = "unload_model"
    LOAD_LAYER_RANGE = "load_layer_range"
    FORWARD_LAYERS = "forward_layers"
    ENSURE_FULL_MODEL = "ensure_full_model"


@dataclass(frozen=True)
class BackendCapabilities:
    """Capabilities exposed by an engine, independent of its implementation."""

    backend_id: str
    capabilities: FrozenSet[str]

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


_BACKEND_CAPABILITIES = MappingProxyType({
    family: BackendCapabilities(family, spec.capabilities)
    for family, spec in LEGACY_ENGINE_SPECS.items()
})
_EMPTY_CAPABILITIES = BackendCapabilities(
    backend_id="",
    capabilities=frozenset(),
)


def registered_backends() -> tuple[str, ...]:
    """Return backend IDs in stable API display order."""

    return tuple(_BACKEND_CAPABILITIES)


def accepted_backend_requests(*, include_auto: bool = True) -> tuple[str, ...]:
    backends = registered_backends()
    return ("auto", *backends) if include_auto else backends


_BACKEND_REQUEST_ALIASES = {
    "gguf": BackendId.LLAMA_CPP,
    "llama": BackendId.LLAMA_CPP,
    "llama_cpp": BackendId.LLAMA_CPP,
    "pytorch": BackendId.PYTORCH,
    "torch": BackendId.PYTORCH,
    "island": BackendId.ISLAND,
    "tp_island": BackendId.ISLAND,
}


def normalize_backend_request(value: Any, *, allow_auto: bool = True) -> str:
    """Strictly normalize a public backend request.

    Request parsing must fail closed: an unknown spelling must never silently
    become llama.cpp.
    """

    normalized = str(value or "auto").strip().lower().replace("-", "_").replace(".", "_")
    if allow_auto and normalized == "auto":
        return "auto"
    backend = _BACKEND_REQUEST_ALIASES.get(normalized)
    if backend is None:
        raise ValueError(f"unsupported backend request: {value!r}")
    return backend


class Koakuma(Protocol):
    """Common host-facing engine contract.

    Backend-specific features must be checked with ``supports`` instead of
    reaching into a private engine-type field from control-plane code.
    """

    @property
    def backend_id(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def supports(self, capability: str) -> bool: ...

    def load_model(self, *args: Any, **kwargs: Any) -> Any: ...

    def unload_model(self) -> None: ...

    def chat(self, messages: Any, **kwargs: Any) -> Any: ...

    def chat_stream(self, messages: Any, **kwargs: Any) -> Any: ...


def canonical_backend(value: Any, default: str = "") -> str:
    """Normalize public and historical backend aliases without guessing."""

    normalized = str(value or "").strip().lower().replace("-", "_").replace(".", "_")
    return _BACKEND_REQUEST_ALIASES.get(normalized, default)


def backend_capabilities(backend: Any, runtime_kind: str | None = None) -> BackendCapabilities:
    """Return the public capability set for a backend identifier."""

    if not str(backend or "").strip():
        return _EMPTY_CAPABILITIES
    backend_id = canonical_backend(backend, default="")
    if runtime_kind is not None:
        spec = ENGINE_REGISTRY.get(backend_id, runtime_kind)
        return (BackendCapabilities(backend_id, spec.capabilities)
                if spec is not None else _EMPTY_CAPABILITIES)
    return _BACKEND_CAPABILITIES.get(backend_id, _EMPTY_CAPABILITIES)


def _profile_has_cuda(node_profile: Mapping[str, Any] | None) -> bool:
    if not node_profile:
        return False

    gpus = node_profile.get("gpus") or []
    if any(bool(gpu.get("cuda_available")) for gpu in gpus if isinstance(gpu, Mapping)):
        return True

    gpu = node_profile.get("gpu") or {}
    return bool(isinstance(gpu, Mapping) and gpu.get("cuda_available"))


def select_backend(
    node_profile: Mapping[str, Any] | None = None,
    *,
    requested: Any = "auto",
    cuda_available: bool | None = None,
    island_enabled: bool = False,
    island_base_url: str = "",
) -> str:
    """Select one backend for a node.

    ``requested`` is an explicit user/configuration choice.  ``auto`` uses
    node capabilities: CUDA nodes default to PyTorch, while CPU/edge/mobile
    nodes use llama.cpp.  ``llama_cpp`` is therefore also the explicit opt-out
    from the CUDA default.  The island route remains a separately enabled
    compatibility route and is selected before local backends.
    """

    requested_backend = normalize_backend_request(requested, allow_auto=True)

    if requested_backend == BackendId.ISLAND or (
        island_enabled and island_base_url
    ):
        return BackendId.ISLAND

    if requested_backend == BackendId.LLAMA_CPP:
        return BackendId.LLAMA_CPP
    if requested_backend == BackendId.PYTORCH:
        return BackendId.PYTORCH

    has_cuda = _profile_has_cuda(node_profile)
    if cuda_available is not None:
        has_cuda = bool(cuda_available) or has_cuda
    return BackendId.PYTORCH if has_cuda else BackendId.LLAMA_CPP


def backend_id_for(runtime: Any, default: str = "") -> str:
    """Read a runtime backend through its public property with legacy fallback.

    The fallback is intentionally kept here, at the boundary, so callers do
    not need to know about the historical ``_engine_type`` field.  It can be
    removed after all external test doubles and sidecars expose ``engine_type``.
    """

    if runtime is None:
        return default
    missing = object()
    public_id = getattr(runtime, "engine_type", missing)
    if public_id is not missing:
        if public_id:
            return canonical_backend(public_id, default=default)
        return default
    legacy_id = getattr(runtime, "_engine_type", None)
    if legacy_id:
        return canonical_backend(legacy_id, default=default)
    public_id = getattr(runtime, "backend_id", None)
    if public_id:
        return canonical_backend(public_id, default=default)
    return default


def _runtime_attribute(runtime: Any, name: str) -> Any:
    """Read a declared property without triggering a lazy host's __getattr__."""
    if inspect.getattr_static(runtime, name, None) is None:
        return None
    return getattr(runtime, name, None)


def runtime_identity_for(runtime: Any) -> tuple[str, str]:
    """Identify the concrete execution path without importing/materializing it."""
    if runtime is None:
        return "", ""
    peek = _runtime_attribute(runtime, "peek_manager")
    if callable(peek):
        manager = peek()
        if manager is not None and manager is not runtime:
            return runtime_identity_for(manager)
    family = backend_id_for(runtime)
    declared_kind = _runtime_attribute(runtime, "runtime_kind")
    inference_path = _runtime_attribute(runtime, "inference_path")
    if declared_kind and inference_path and declared_kind != inference_path:
        return family, ""
    kind = declared_kind or inference_path
    if kind:
        return family, str(kind)
    # Existing implementations predate runtime_kind. Keep this adaptation at
    # the legacy boundary, based on the actual implementation, not its family.
    cls = type(runtime)
    if cls.__name__ == "LlamaCppEngine" and cls.__module__ == "llama_engine":
        kind = LLAMA_CPP_KEEP_HEAD if vars(runtime).get("_keep_head") is not None else LLAMA_CPP_BINDING
    elif cls.__name__ == "ModelManager" and cls.__module__ == "model_module":
        native = vars(runtime).get("_llama_engine") or vars(runtime).get("_island_engine")
        if native is not None:
            return runtime_identity_for(native)
        if family == BackendId.PYTORCH:
            model = vars(runtime).get("model")
            device = str(getattr(model, "device", None) or vars(runtime).get("device", "cpu"))
            kind = PYTORCH_CUDA if device.startswith("cuda") else PYTORCH_CPU
    elif cls.__name__ == "IslandEngine" and cls.__module__ == "island_engine":
        kind = ISLAND_REMOTE
    return family, str(kind or "")


def runtime_supports(runtime: Any, capability: str) -> bool:
    """Intersect a runtime probe with its concrete static contract."""
    if runtime is None:
        return False
    try:
        family, kind = runtime_identity_for(runtime)
        spec = ENGINE_REGISTRY.get(family, kind)
        abi = _runtime_attribute(runtime, "native_abi")
        if spec is None or (abi is not None and abi != spec.native_abi):
            return False
        if not backend_capabilities(family, kind).supports(capability):
            return False
        supports = _runtime_attribute(runtime, "supports")
        if callable(supports) and not bool(supports(capability)):
            return False
        peek = _runtime_attribute(runtime, "peek_manager")
        if callable(peek):
            manager = peek()
            if manager is not None and manager is not runtime:
                return runtime_supports(manager, capability)
        return True
    except Exception:
        return False


__all__ = [
    "BackendCapabilities",
    "BackendId",
    "Capability",
    "COMMON_CAPABILITIES",
    "Koakuma",
    "backend_capabilities",
    "backend_id_for",
    "canonical_backend",
    "normalize_backend_request",
    "accepted_backend_requests",
    "registered_backends",
    "runtime_supports",
    "runtime_identity_for",
    "select_backend",
]
