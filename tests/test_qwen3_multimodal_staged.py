"""MM1.20 staged visual-to-text contract and lifecycle regressions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

# CI 不拉模型工件（本机 models 是 Junction）。缺工件按项目约定跳过。
pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "qwen3-vl-4b-instruct" / "config.json").exists(),
    reason="缺本地工件 models/qwen3-vl-4b-instruct（CI 不拉模型）",
)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen3_multimodal_contract import (  # noqa: E402
    build_mm1_model_manifest,
    build_mm1_model_profile,
)
from qwen3_multimodal_runtime import (  # noqa: E402
    Qwen3MultimodalRuntimeError,
    Qwen3MultimodalStagedTextFixture,
    build_mm1_staged_text_contract,
    execute_mm1_staged_text_contract,
    validate_mm1_staged_text_contract,
)


def _component(component_id: str, kind: str, digest: str, size: int) -> dict:
    return {
        "component_id": component_id,
        "artifact_id": f"{component_id}-artifact",
        "component_kind": kind,
        "format": "tokenizer" if kind == "processor" else "safetensors",
        "revision": "fixture-revision",
        "size_bytes": size,
        "sha256": digest * 64,
    }


def _manifest() -> dict:
    config = json.loads(
        (ROOT / "models" / "qwen3-vl-4b-instruct" / "config.json").read_text(encoding="utf-8"),
    )
    profile = build_mm1_model_profile(config)
    return build_mm1_model_manifest(
        model_id="fixture-qwen3-vl-mm120",
        model_family=profile["model_family"],
        runtime="transformers_sidecar",
        revision="fixture-revision",
        components=[
            _component("processor", "processor", "a", 128),
            _component("text", "text_weights", "b", 2_000_000),
            _component("vision", "vision_weights", "c", 1_000_000),
        ],
        text=profile["text"],
        vision=profile["vision"],
        processor=profile["processor"],
    )


def _feature(manifest: dict) -> dict:
    return {
        "feature_kind": "qwen3_visual_feature_placeholder",
        "model_id": manifest["model_id"],
        "media_reference_sha256": "d" * 64,
        "tensor": {
            "shape": [1, 64, manifest["text"]["hidden_size"]],
            "dtype": "float32",
            "device": "cpu",
        },
        "synthetic": False,
        "weight_materialized": True,
        "full_model_materialized": False,
    }


def _segments(*, first_capacity: int = 2_000_000) -> list[dict]:
    return [
        {
            "node_id": "text-node-a",
            "layer_range": [0, 18],
            "has_embedding": True,
            "has_lm_head": False,
            "device": "cpu",
            "dtype": "float32",
            "required_bytes": 1_000_000,
            "activation_bytes": 700_000,
            "node_capacity_bytes": first_capacity,
            "assignment_manifest_sha256": "1" * 64,
        },
        {
            "node_id": "text-node-b",
            "layer_range": [18, 36],
            "has_embedding": False,
            "has_lm_head": True,
            "device": "cpu",
            "dtype": "float32",
            "required_bytes": 1_000_000,
            "activation_bytes": 700_000,
            "node_capacity_bytes": 2_000_000,
            "assignment_manifest_sha256": "2" * 64,
        },
    ]


def _contract() -> tuple[dict, dict]:
    manifest = _manifest()
    contract = build_mm1_staged_text_contract(
        vision_feature=_feature(manifest),
        manifest=manifest,
        segments=_segments(),
        text_chain_id="e" * 64,
        generation=20,
        source_node_id="vision-node",
    )
    return manifest, contract


def test_mm120_contract_binds_visual_feature_to_first_segment_without_paths():
    manifest, contract = _contract()

    assert contract["entry_segment_index"] == 0
    assert contract["segment_plan"][0]["layer_range"] == [0, 18]
    assert contract["segment_plan"][0]["peak_bytes"] == 1_700_000
    assert contract["input_layout"]["visual_span"] == [0, 64]
    assert contract["input_layout"]["text_span"] == [64, 68]
    assert contract["input_layout"]["total_sequence"] == 68
    assert contract["visual_handoff"]["boundary"] == "visual_to_text"
    assert contract["visual_handoff"]["target_node_id"] == "text-node-a"
    assert contract["execution"]["segment_materialized"] is False
    assert contract["execution"]["full_model_materialized"] is False
    assert validate_mm1_staged_text_contract(contract, manifest=manifest) == contract
    encoded = json.dumps(contract, ensure_ascii=True).lower()
    assert "path" not in encoded
    assert "prompt_text" not in encoded
    assert "prompt_content" not in encoded


def test_mm120_contract_fails_closed_on_capacity_and_full_model_state():
    manifest = _manifest()
    with pytest.raises(Qwen3MultimodalRuntimeError) as capacity_error:
        build_mm1_staged_text_contract(
            vision_feature=_feature(manifest),
            manifest=manifest,
            segments=_segments(first_capacity=1_000_000),
            text_chain_id="e" * 64,
            generation=20,
        )
    assert capacity_error.value.reason_code == "qwen3_mm1_staged_capacity_rejected"

    feature = _feature(manifest)
    feature["full_model_materialized"] = True
    with pytest.raises(Qwen3MultimodalRuntimeError) as materialization_error:
        build_mm1_staged_text_contract(
            vision_feature=feature,
            manifest=manifest,
            segments=_segments(),
            text_chain_id="e" * 64,
            generation=20,
        )
    assert materialization_error.value.reason_code == "qwen3_mm1_staged_full_model_forbidden"

    with pytest.raises(Qwen3MultimodalRuntimeError) as sequence_error:
        build_mm1_staged_text_contract(
            vision_feature=_feature(manifest),
            manifest=manifest,
            segments=_segments(),
            text_chain_id="e" * 64,
            generation=20,
            prompt_tokens=4,
            sequence_length=64,
        )
    assert sequence_error.value.reason_code == "qwen3_mm1_staged_sequence_overflow"


def test_mm120_contract_accepts_three_contiguous_text_segments():
    manifest = _manifest()
    segments = _segments()
    segments[0]["layer_range"] = [0, 12]
    segments[1]["layer_range"] = [12, 24]
    segments[1]["has_lm_head"] = False
    segments.append({
        **segments[1],
        "node_id": "text-node-c",
        "layer_range": [24, 36],
        "has_lm_head": True,
        "assignment_manifest_sha256": "3" * 64,
    })

    contract = build_mm1_staged_text_contract(
        vision_feature=_feature(manifest),
        manifest=manifest,
        segments=segments,
        text_chain_id="e" * 64,
        generation=20,
    )

    assert [item["layer_range"] for item in contract["segment_plan"]] == [
        [0, 12], [12, 24], [24, 36],
    ]


def test_mm120_fixture_executes_one_segment_builds_next_request_and_releases():
    manifest, contract = _contract()
    executor = Qwen3MultimodalStagedTextFixture()

    result = execute_mm1_staged_text_contract(
        contract, manifest=manifest, executor=executor,
    )

    assert result["status"] == "staged_text_segment_fixture_executed"
    assert result["execution"]["evidence_kind"] == "cpu_fixture"
    assert result["execution"]["text_weights_loaded"] is False
    assert result["execution"]["segment_materialized"] is False
    assert result["execution"]["fixture_segment_materialized"] is True
    assert result["execution"]["full_model_materialized"] is False
    assert result["next_segment_request"]["segment_index"] == 1
    assert result["next_segment_request"]["node_id"] == "text-node-b"
    assert result["cleanup"]["completed"] is True
    assert result["cleanup"]["segment_materialized"] is False
    assert executor.cleanup_reasons == ["completed"]
    assert "path" not in json.dumps(result, ensure_ascii=True).lower()


def test_mm120_execution_failure_still_releases_segment():
    manifest, contract = _contract()
    executor = Qwen3MultimodalStagedTextFixture(fail_execution=True)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        execute_mm1_staged_text_contract(contract, manifest=manifest, executor=executor)

    assert caught.value.reason_code == "qwen3_mm1_staged_execution_failed"
    assert executor.cleanup_reasons == ["execution_failed"]
    assert executor.fixture_segment_materialized is False


def test_mm120_incomplete_cleanup_fails_closed():
    manifest, contract = _contract()
    executor = Qwen3MultimodalStagedTextFixture(fail_cleanup=True)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        execute_mm1_staged_text_contract(contract, manifest=manifest, executor=executor)

    assert caught.value.reason_code == "qwen3_mm1_staged_cleanup_failed"
    assert executor.fixture_segment_materialized is True


def test_mm120_tampered_contract_is_rejected_before_execution():
    manifest, contract = _contract()
    contract["segment_plan"][0]["required_bytes"] += 1
    executor = Qwen3MultimodalStagedTextFixture()

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        execute_mm1_staged_text_contract(contract, manifest=manifest, executor=executor)

    assert caught.value.reason_code == "qwen3_mm1_staged_contract_invalid"
    assert executor.cleanup_reasons == []
