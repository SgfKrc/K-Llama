import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from execution_provider import (
    LocalFullModelProvider,
    ModelIdentity,
    ProviderBusy,
    ProviderNotFound,
    ProviderRegistrationError,
    ProviderReservationError,
    ProviderRegistry,
    ProviderUnavailable,
    StageAttempt,
    StageRequest,
)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("llama_cpp", "llama_cpp"),
        ("llama.cpp", "llama_cpp"),
        ("llama-cpp", "llama_cpp"),
        ("llama", "llama_cpp"),
        ("gguf", "llama_cpp"),
        ("pytorch", "pytorch"),
        ("torch", "pytorch"),
    ],
)
def test_model_identity_normalizes_local_engine_aliases(alias, canonical):
    identity = ModelIdentity(
        model_id="model-a",
        engine=alias,
        format="gguf" if canonical == "llama_cpp" else "safetensors",
        revision="rev-a",
        sha256="a" * 64,
    )

    assert identity.engine == canonical
    assert identity.snapshot()["engine"] == canonical


@pytest.mark.parametrize("engine", ["island", "external_api"])
def test_model_identity_preserves_non_local_engine_identifiers(engine):
    identity = ModelIdentity(
        model_id="model-a", engine=engine, format="remote",
        revision="rev-a", sha256="a" * 64,
    )

    assert identity.engine == engine


@pytest.mark.parametrize("engine", ["llama_magic", "speculative_assisted"])
def test_model_identity_rejects_unknown_engine_alias(engine):
    with pytest.raises(ValueError, match="engine is unsupported"):
        ModelIdentity(
            model_id="model-a", engine=engine, format="gguf",
            revision="rev-a", sha256="a" * 64,
        )


def _request(
    *,
    workflow_id="wf_provider01",
    stage_id="candidate_a",
    stage_type="full_inference",
    provider_id="local_full_model",
):
    return StageRequest(
        workflow_id=workflow_id,
        request_id="req-provider",
        stage_id=stage_id,
        stage_type=stage_type,
        provider_id=provider_id,
        dependencies={},
        root_input={"message": "question"},
    )


def test_registry_enforces_unique_identity_and_stage_capabilities():
    registry = ProviderRegistry()
    provider = LocalFullModelProvider(
        lambda request, cancel_event: {"content": request.stage_id},
        node_id="node-local",
    )
    registry.register(provider)

    status = registry.inspect()[0]
    assert status["provider_id"] == "local_full_model"
    assert status["provider_kind"] == "local_full_model"
    assert status["node_id"] == "node-local"
    assert status["max_concurrency"] == 1
    assert status["available"] is True
    with pytest.raises(ProviderRegistrationError) as duplicate:
        registry.register(provider)
    assert duplicate.value.code == "duplicate_provider_id"
    with pytest.raises(ProviderNotFound) as missing:
        registry.reserve(_request(provider_id="missing_provider"))
    assert missing.value.code == "provider_not_found"
    with pytest.raises(ProviderUnavailable) as unsupported:
        registry.reserve(_request(stage_type="unsupported"))
    assert unsupported.value.code == "no_compatible_provider"
    reservation = registry.reserve(_request())
    with pytest.raises(ProviderBusy) as active:
        registry.unregister("local_full_model")
    assert active.value.code == "provider_has_active_reservations"
    registry.release(reservation.reservation_id)
    assert registry.unregister("local_full_model") is True
    assert registry.unregister("local_full_model") is False
    registry.close()


def test_reservation_accept_timeout_does_not_block_registry_or_leak_slot():
    release_accept = threading.Event()
    late_release = threading.Event()

    class SlowAcceptProvider(LocalFullModelProvider):
        def reserve(self, request):
            release_accept.wait()
            return super().reserve(request)

        def release(self, reservation_id):
            super().release(reservation_id)
            late_release.set()

    registry = ProviderRegistry()
    provider = SlowAcceptProvider(
        lambda request, cancel_event: {"content": "unused"},
    )
    registry.register(provider)

    started = time.perf_counter()
    with pytest.raises(ProviderReservationError) as captured:
        registry.reserve(_request(), timeout_seconds=0.03)
    assert captured.value.code == "provider_accept_timeout"
    assert time.perf_counter() - started < 0.5

    status_started = time.perf_counter()
    assert registry.inspect()[0]["healthy"] is True
    assert time.perf_counter() - status_started < 0.5

    release_accept.set()
    assert late_release.wait(1.0)
    assert provider.inspect().active_reservations == 0
    registry.close()


def test_local_provider_reservation_is_atomic_across_threads():
    registry = ProviderRegistry()
    registry.register(LocalFullModelProvider(
        lambda request, cancel_event: {"content": "unused"},
    ))
    barrier = threading.Barrier(3)
    winner_reserved = threading.Event()
    loser_finished = threading.Event()
    release_winner = threading.Event()
    reservations = []
    errors = []

    def reserve_slot(index):
        barrier.wait()
        try:
            reservation = registry.reserve(_request(
                workflow_id=f"wf_atomic0{index}",
            ))
            reservations.append(reservation)
            winner_reserved.set()
            assert release_winner.wait(5)
            registry.release(reservation.reservation_id)
        except ProviderBusy as exc:
            errors.append(exc)
            loser_finished.set()

    threads = [
        threading.Thread(target=reserve_slot, args=(index,))
        for index in (1, 2)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    assert winner_reserved.wait(5)
    assert loser_finished.wait(5)
    release_winner.set()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(reservations) == 1
    assert len(errors) == 1
    assert errors[0].retryable is True
    assert registry.inspect()[0]["active_reservations"] == 0
    registry.close()


def test_provider_inspection_failure_is_isolated_and_selection_continues():
    class ToggleInspectProvider(LocalFullModelProvider):
        fail_inspection = False

        def inspect(self):
            if self.fail_inspection:
                raise RuntimeError("inspection failed")
            return super().inspect()

    registry = ProviderRegistry()
    broken = ToggleInspectProvider(
        lambda request, cancel_event: {"content": "broken"},
        provider_id="a_broken",
    )
    healthy = LocalFullModelProvider(
        lambda request, cancel_event: {"content": "healthy"},
        provider_id="b_healthy",
    )
    registry.register(broken)
    registry.register(healthy)
    broken.fail_inspection = True

    statuses = {item["provider_id"]: item for item in registry.inspect()}
    assert statuses["a_broken"]["healthy"] is False
    assert statuses["a_broken"]["error_code"] == "provider_inspection_failed"
    assert statuses["b_healthy"]["available"] is True
    automatic = registry.reserve(_request(provider_id=""))
    assert automatic.provider_id == "b_healthy"
    registry.release(automatic.reservation_id)
    with pytest.raises(ProviderUnavailable) as explicit:
        registry.reserve(_request(provider_id="a_broken"))
    assert explicit.value.code == "provider_inspection_failed"
    registry.close()


def test_local_provider_result_metadata_excludes_output_content():
    registry = ProviderRegistry()
    provider = LocalFullModelProvider(
        lambda request, cancel_event: {
            "content": "secret candidate text",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "untrusted": "secret",
            },
            "tokens_per_second": 2.5,
            "model": "test-model",
        },
    )
    registry.register(provider)
    request = _request()
    reservation = registry.reserve(request)
    result = registry.execute(
        StageAttempt("att_provider01", request, reservation.provider_id),
        reservation,
        threading.Event(),
    )

    assert result.output["content"] == "secret candidate text"
    assert result.metadata == {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "usage_estimated": False,
        "tokens_per_second": 2.5,
        "model": "test-model",
    }
    assert "secret candidate text" not in str(result.metadata)
    with pytest.raises(ProviderReservationError) as duplicate_execute:
        registry.execute(
            StageAttempt("att_provider02", request, reservation.provider_id),
            reservation,
            threading.Event(),
        )
    assert duplicate_execute.value.code == "reservation_already_executed"
    registry.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0
    registry.close()


def test_provider_cancel_reaches_active_attempt():
    started = threading.Event()

    def execute(request, cancel_event):
        started.set()
        assert cancel_event.wait(5)
        return {"content": "cancel observed"}

    registry = ProviderRegistry()
    registry.register(LocalFullModelProvider(execute))
    request = _request()
    reservation = registry.reserve(request)
    result = []

    def run_provider():
        result.append(registry.execute(
            StageAttempt("att_cancel01", request, reservation.provider_id),
            reservation,
            threading.Event(),
        ))

    thread = threading.Thread(target=run_provider)
    thread.start()
    assert started.wait(5)
    registry.cancel("local_full_model", "att_cancel01")
    thread.join(5)

    assert not thread.is_alive()
    assert result[0].output["content"] == "cancel observed"
    registry.release(reservation.reservation_id)
    registry.close()
