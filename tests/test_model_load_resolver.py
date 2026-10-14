import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from koakuma_engine import normalize_backend_request
from model_load_resolver import (
    LAYER_RANGE_DYNAMIC,
    LAYER_RANGE_NONE,
    LAYER_RANGE_PRECUT_ARTIFACT,
    ModelLoadFacts,
    ModelLoadResolutionError,
    OP_DISTRIBUTED_LOAD,
    OP_DYNAMIC_LAYER_RANGE,
    OP_LAYER_RANGE,
    effective_pytorch_cuda_available,
    resolve_model_load,
)


def _facts(*, cuda=True, safetensors=True, gguf=True):
    return ModelLoadFacts(
        model_id="fixture",
        model_name="Fixture",
        has_safetensors=safetensors,
        has_gguf=gguf,
        safetensors_path="C:/models/fixture" if safetensors else None,
        gguf_path="C:/models/fixture.Q4_K_M.gguf" if gguf else None,
        preferred_engine="pytorch" if cuda else "llama_cpp",
        cuda_available=cuda,
    )


@pytest.mark.parametrize("alias", ["llama_cpp", "llama.cpp", "llama-cpp", "gguf", "llama"])
def test_llama_cpp_aliases_share_one_canonical_answer(alias):
    assert normalize_backend_request(alias) == "llama_cpp"
    resolution = resolve_model_load(
        _facts(),
        requested_engine=alias,
        requested_quant="Q4_K_M",
    )
    assert resolution.engine == "llama_cpp"
    assert resolution.model_path.endswith(".gguf")
    assert resolution.requested_quant == "Q4_K_M"
    assert resolution.quant_type == "gguf"
    assert resolution.layer_range_mode == LAYER_RANGE_PRECUT_ARTIFACT


@pytest.mark.parametrize("alias", ["pytorch", "torch"])
def test_pytorch_aliases_share_one_canonical_answer(alias):
    resolution = resolve_model_load(
        _facts(),
        requested_engine=alias,
        requested_quant="int4",
    )
    assert resolution.engine == "pytorch"
    assert resolution.model_path == "C:/models/fixture"
    assert resolution.quant_type == "int4"
    assert resolution.layer_range_mode == LAYER_RANGE_DYNAMIC


def test_unknown_engine_fails_closed():
    with pytest.raises(ModelLoadResolutionError) as exc:
        resolve_model_load(
            _facts(),
            requested_engine="tensorrt",
            requested_quant="int4",
        )
    assert exc.value.code == "MODEL_ENGINE_UNSUPPORTED"


def test_pytorch_never_accepts_gguf_as_its_artifact():
    facts = ModelLoadFacts(
        model_id="bad",
        has_safetensors=True,
        safetensors_path="C:/models/wrong.gguf",
        cuda_available=True,
    )
    with pytest.raises(ModelLoadResolutionError) as exc:
        resolve_model_load(
            facts,
            requested_engine="pytorch",
            requested_quant="fp16",
        )
    assert exc.value.code == "MODEL_PATH_ENGINE_MISMATCH"


@pytest.mark.parametrize("requested", ["auto", "pytorch", "torch"])
def test_dynamic_layer_range_resolves_supported_request_to_pytorch(requested):
    resolution = resolve_model_load(
        _facts(),
        requested_engine=requested,
        requested_quant="int4",
        operation=OP_DYNAMIC_LAYER_RANGE,
    )
    assert resolution.engine == "pytorch"
    assert resolution.model_path == "C:/models/fixture"
    assert resolution.quant_type == "fp16"
    assert resolution.runtime_quant == "fp16"
    assert resolution.layer_range_mode == LAYER_RANGE_DYNAMIC


@pytest.mark.parametrize("requested", ["llama_cpp", "gguf", "llama.cpp"])
def test_dynamic_layer_range_rejects_explicit_precut_engine(requested):
    with pytest.raises(ModelLoadResolutionError) as exc:
        resolve_model_load(
            _facts(),
            requested_engine=requested,
            requested_quant="int4",
            operation=OP_DYNAMIC_LAYER_RANGE,
        )
    assert exc.value.code == "LAYER_RANGE_MODE_MISMATCH"


def test_distributed_auto_prefers_pytorch_for_dual_format_model():
    resolution = resolve_model_load(
        _facts(),
        requested_engine="auto",
        requested_quant="int4",
        operation=OP_DISTRIBUTED_LOAD,
    )
    assert resolution.engine == "pytorch"
    assert resolution.model_path == "C:/models/fixture"
    assert resolution.quant_type == "int4"
    assert resolution.layer_range_mode == LAYER_RANGE_DYNAMIC


@pytest.mark.parametrize("requested", ["llama_cpp", "gguf", "llama.cpp"])
def test_distributed_explicit_llama_cpp_keeps_precut_mode(requested):
    resolution = resolve_model_load(
        _facts(),
        requested_engine=requested,
        requested_quant="fp32",
        operation=OP_DISTRIBUTED_LOAD,
    )
    assert resolution.engine == "llama_cpp"
    assert resolution.quant_type == "gguf"
    assert resolution.layer_range_mode == LAYER_RANGE_PRECUT_ARTIFACT


@pytest.mark.parametrize(
    ("requested", "expected_engine", "expected_mode"),
    [
        ("pytorch", "pytorch", LAYER_RANGE_DYNAMIC),
        ("llama_cpp", "llama_cpp", LAYER_RANGE_PRECUT_ARTIFACT),
    ],
)
def test_general_layer_range_preserves_selected_implementation(
    requested, expected_engine, expected_mode,
):
    resolution = resolve_model_load(
        _facts(),
        requested_engine=requested,
        requested_quant="int4",
        operation=OP_LAYER_RANGE,
    )
    assert resolution.engine == expected_engine
    assert resolution.layer_range_mode == expected_mode
    if expected_engine == "pytorch":
        assert resolution.quant_type == "fp16"


@pytest.mark.parametrize("requested", ["fp32", "fp16", "int8", "int4"])
def test_cpu_pytorch_quant_is_explicitly_resolved_to_fp32(requested):
    resolution = resolve_model_load(
        _facts(cuda=False, gguf=False),
        requested_engine="pytorch",
        requested_quant=requested,
    )
    assert resolution.quant_type == "fp32"
    assert resolution.runtime_quant == "fp32"
    expected_reason = "RUNTIME_QUANT_CONVERTED" if requested != "fp32" else "ENGINE_EXPLICIT"
    assert resolution.reason_code == expected_reason


@pytest.mark.parametrize("requested", ["fp32", "fp16", "int8", "int4"])
def test_cuda_pytorch_preserves_supported_full_model_quant(requested):
    resolution = resolve_model_load(
        _facts(cuda=True, gguf=False),
        requested_engine="pytorch",
        requested_quant=requested,
    )
    assert resolution.quant_type == requested
    assert resolution.runtime_quant == requested


def test_pytorch_rejects_gguf_quant_name():
    with pytest.raises(ModelLoadResolutionError) as exc:
        resolve_model_load(
            _facts(),
            requested_engine="pytorch",
            requested_quant="Q4_K_M",
        )
    assert exc.value.code == "MODEL_QUANT_UNSUPPORTED"


def test_island_has_no_local_artifact_or_layer_range_mode():
    facts = ModelLoadFacts(
        model_id="remote",
        island_enabled=True,
        island_base_url="http://island.invalid",
    )
    resolution = resolve_model_load(
        facts,
        requested_engine="island",
        requested_quant="int4",
    )
    assert resolution.engine == "island"
    assert resolution.model_path is None
    assert resolution.quant_type == "island"
    assert resolution.layer_range_mode == LAYER_RANGE_NONE


def test_island_rejects_dynamic_layer_materialization():
    facts = ModelLoadFacts(
        model_id="remote",
        island_enabled=True,
        island_base_url="http://island.invalid",
    )
    with pytest.raises(ModelLoadResolutionError) as exc:
        resolve_model_load(
            facts,
            requested_engine="island",
            requested_quant="island",
            operation=OP_DYNAMIC_LAYER_RANGE,
        )
    assert exc.value.code == "LAYER_RANGE_MODE_UNSUPPORTED"


def test_llama_cpp_layer_range_capability_is_precut_not_none():
    resolution = resolve_model_load(
        _facts(),
        requested_engine="llama_cpp",
        requested_quant="Q4_K_M",
    )
    assert resolution.layer_range_mode == LAYER_RANGE_PRECUT_ARTIFACT


def test_gguf_effective_quant_never_claims_requested_pytorch_dtype():
    resolution = resolve_model_load(
        _facts(),
        requested_engine="llama_cpp",
        requested_quant="fp32",
    )
    assert resolution.requested_quant == "fp32"
    assert resolution.quant_type == "gguf"
    assert resolution.runtime_quant == "gguf"


def test_edge_profile_without_assigned_cuda_forces_cpu_resolution():
    assert effective_pytorch_cuda_available(
        system_cuda_available=True,
        profile={"tier": "edge", "gpu": {"cuda_available": False}},
    ) is False
    assert effective_pytorch_cuda_available(
        system_cuda_available=True,
        profile={"tier": "laptop", "gpu": {"cuda_available": False}},
    ) is True

