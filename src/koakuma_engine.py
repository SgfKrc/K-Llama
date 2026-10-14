"""Koakuma engine boundary and backend selection.

This module is deliberately dependency-light.  It describes the common engine
contract and chooses a backend from a node profile without importing either
runtime implementation.  The actual llama.cpp and PyTorch implementations
remain behind the host/manager adapters for now.
"""

from __future__ import annotations

from dataclasses import dataclass
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


COMMON_CAPABILITIES: FrozenSet[str] = frozenset({
    Capability.CHAT,
    Capability.CHAT_STREAM,
    Capability.LOAD_MODEL,
    Capability.UNLOAD_MODEL,
})


@dataclass(frozen=True)
class BackendCapabilities:
    """Capabilities exposed by an engine, independent of its implementation."""

    backend_id: str
    capabilities: FrozenSet[str]

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


_BACKEND_CAPABILITIES = {
    BackendId.LLAMA_CPP: BackendCapabilities(
        backend_id=BackendId.LLAMA_CPP,
        capabilities=frozenset({
            *COMMON_CAPABILITIES,
            Capability.CHAT_IMAGE,
            # ★ 2026-09-19：llama.cpp 现在**也能做层前向**，故补齐层接力能力位。
            #   理由（本会话实测）：
            #     * 当**下游** —— `LlamaCppEngine.forward_layers_from_hidden()` 经 `llama_batch.embd`
            #       注入上游 hidden，跑本 GGUF（裁层）的层并出 logits；
            #     * 当**上游** —— `LlamaCppEngine.forward_layers_to_hidden()` 出 hidden
            #       （取「本文件最后一层之后」，故上游需用**切点处的裁层 GGUF**）。
            #   ⚠️ 这里只是**声明能力**：llama.cpp 不需要 torch ⇒
            #   **Android/边缘设备（无 torch、有 GGUF 引擎）因此可以参与层流水线**。
            #   ⚠️ 不含 `LOAD_LAYER_RANGE` / `ENSURE_FULL_MODEL` —— 那两个是 PyTorch 侧语义
            #   （按 key 物化层段 / 需要整模），llama.cpp 用「裁层 GGUF」表达同一意图。
            Capability.FORWARD_LAYERS,
        }),
    ),
    BackendId.PYTORCH: BackendCapabilities(
        backend_id=BackendId.PYTORCH,
        capabilities=frozenset({
            *COMMON_CAPABILITIES,
            Capability.CHAT_IMAGE,
            Capability.LOAD_LAYER_RANGE,
            Capability.FORWARD_LAYERS,
            Capability.ENSURE_FULL_MODEL,
        }),
    ),
    BackendId.ISLAND: BackendCapabilities(
        backend_id=BackendId.ISLAND,
        capabilities=COMMON_CAPABILITIES,
    ),
}
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


def backend_capabilities(backend: Any) -> BackendCapabilities:
    """Return the public capability set for a backend identifier."""

    if not str(backend or "").strip():
        return _EMPTY_CAPABILITIES
    backend_id = canonical_backend(backend, default="")
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


def runtime_supports(runtime: Any, capability: str) -> bool:
    """Probe a runtime capability while keeping legacy compatibility local."""

    if runtime is None:
        return False
    supports = getattr(runtime, "supports", None)
    if callable(supports):
        return bool(supports(capability))
    return backend_capabilities(backend_id_for(runtime)).supports(capability)


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
    "select_backend",
]
