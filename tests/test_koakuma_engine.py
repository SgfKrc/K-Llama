from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from koakuma_engine import (
    BackendId,
    Capability,
    accepted_backend_requests,
    backend_capabilities,
    backend_id_for,
    canonical_backend,
    registered_backends,
    runtime_supports,
    select_backend,
)
from model_host import ModelHost


def test_select_backend_uses_cuda_profile_as_single_decision() -> None:
    assert select_backend({"gpus": [{"cuda_available": True}]}) == BackendId.PYTORCH
    assert select_backend({"tier": "edge", "gpus": []}) == BackendId.LLAMA_CPP
    assert select_backend({"tier": "mobile", "gpus": []}) == BackendId.LLAMA_CPP


def test_select_backend_allows_explicit_llama_fallback_on_cuda() -> None:
    assert select_backend(
        {"gpus": [{"cuda_available": True}]},
        requested=BackendId.LLAMA_CPP,
    ) == BackendId.LLAMA_CPP


def test_select_backend_keeps_island_as_explicit_compatibility_route() -> None:
    assert select_backend(requested=BackendId.ISLAND) == BackendId.ISLAND
    assert select_backend(
        requested="auto",
        island_enabled=True,
        island_base_url="http://127.0.0.1:8000",
    ) == BackendId.ISLAND


def test_backend_capabilities_are_public_and_backend_specific() -> None:
    llama = backend_capabilities(BackendId.LLAMA_CPP)
    torch = backend_capabilities(BackendId.PYTORCH)

    assert llama.supports(Capability.CHAT)
    # ★ 2026-09-19：llama.cpp 现在**也能做层前向**（`forward_layers_from_hidden` 当下游、
    #   `forward_layers_to_hidden` 当上游）⇒ 能力位由「不支持」改为「支持」。
    #   意义：llama.cpp **不需要 torch** ⇒ Android/边缘设备（无 torch、有 GGUF 引擎）
    #   可以参与层流水线。
    assert llama.supports(Capability.FORWARD_LAYERS)
    # 但 llama.cpp **不**提供 PyTorch 侧的层段物化语义（它用「裁层 GGUF」表达同一意图）。
    assert not llama.supports(Capability.LOAD_LAYER_RANGE)
    assert not llama.supports(Capability.ENSURE_FULL_MODEL)
    assert torch.supports(Capability.FORWARD_LAYERS)


def test_unknown_and_nonlocal_backends_have_no_local_capabilities() -> None:
    for backend in ("external_api", "speculative_assisted", "relay_middle", "typo"):
        assert canonical_backend(backend) == ""
        assert not backend_capabilities(backend).supports(Capability.FORWARD_LAYERS)


def test_select_backend_rejects_unknown_request() -> None:
    with pytest.raises(ValueError, match="unsupported backend request"):
        select_backend(requested="llama_magic", cuda_available=False)


def test_backend_registry_is_the_api_allowlist_source() -> None:
    assert registered_backends() == (
        BackendId.LLAMA_CPP,
        BackendId.PYTORCH,
        BackendId.ISLAND,
    )
    assert accepted_backend_requests() == ("auto", *registered_backends())


def test_backend_id_for_legacy_doubles_stays_at_boundary() -> None:
    class LegacyRuntime:
        _engine_type = "gguf"

    assert backend_id_for(LegacyRuntime()) == BackendId.LLAMA_CPP
    assert canonical_backend("llama.cpp") == BackendId.LLAMA_CPP


def test_runtime_supports_uses_capabilities_with_legacy_fallback() -> None:
    class LegacyLayerRuntime:
        _engine_type = "pytorch"

    class ExplicitRuntime:
        def supports(self, capability: str) -> bool:
            return capability == "custom"

    assert runtime_supports(LegacyLayerRuntime(), Capability.FORWARD_LAYERS)
    assert runtime_supports(ExplicitRuntime(), "custom")
    assert not runtime_supports(ExplicitRuntime(), Capability.FORWARD_LAYERS)

    class UnknownRuntime:
        _engine_type = "external_api"

    assert backend_id_for(UnknownRuntime()) == ""
    assert not runtime_supports(UnknownRuntime(), Capability.FORWARD_LAYERS)


def test_model_host_engine_status_does_not_materialize_lazy_manager() -> None:
    host = ModelHost()

    assert host.engine_type == ""
    assert not host.supports(Capability.CHAT)
    assert host.runtime_status()["manager_loaded"] is False


def test_upper_layers_do_not_read_private_backend_field() -> None:
    root = Path(__file__).resolve().parents[1]
    scheduler_sources = [root / "src" / "scheduler.py"]
    scheduler_sources.extend(sorted((root / "src").glob("scheduler_*.py")))
    assert scheduler_sources
    paths = [
        *scheduler_sources,
        root / "src" / "api_server.py",
        root / "src" / "inference_service" / "engine_host.py",
    ]
    for path in paths:
        assert "_engine_type" not in path.read_text(encoding="utf-8")
