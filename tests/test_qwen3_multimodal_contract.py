from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

# CI 不拉模型工件（本机 models 是 Junction）；本文件按工作目录相对路径读模型配置。
pytestmark = pytest.mark.skipif(
    not Path("models/qwen3-vl-4b-instruct/config.json").exists()
    or not Path("models/qwen3-5-2b/config.json").exists(),
    reason="缺本地工件 models/{qwen3-vl-4b-instruct,qwen3-5-2b}（CI 不拉模型）",
)

sys.path.insert(0, "src")

from qwen3_multimodal_contract import (  # noqa: E402
    Qwen3MultimodalContractError,
    build_mm1_handoff_contract,
    build_mm1_model_manifest,
    build_mm1_model_profile,
    build_mm1_transfer_binding,
    estimate_mm1_capacity,
    validate_mm1_handoff_contract,
    validate_mm1_model_manifest,
    validate_mm1_transfer_binding,
)


def _profile(model_path: str) -> dict:
    return build_mm1_model_profile(
        json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8")),
    )


def _component(component_id: str, artifact_id: str, kind: str, size: int, *, fmt: str = "safetensors"):
    digest_prefix = {
        "processor": "a",
        "text_weights": "b",
        "vision_weights": "c",
        "mmproj": "d",
        "mtp": "e",
    }[kind]
    if artifact_id == "shared-main":
        digest_prefix = "b"
    return {
        "component_id": component_id,
        "artifact_id": artifact_id,
        "component_kind": kind,
        "format": fmt,
        "revision": "fixture-revision",
        "size_bytes": size,
        "sha256": digest_prefix * 64,
    }


def _manifest(*, family="qwen3_vl", runtime="transformers_sidecar"):
    profile = _profile(
        "models/qwen3-vl-4b-instruct" if family == "qwen3_vl" else "models/qwen3-5-2b",
    )
    return build_mm1_model_manifest(
        model_id=f"fixture-{family}",
        model_family=family,
        runtime=runtime,
        revision="fixture-revision",
        components=[
            _component("processor", "processor-artifact", "processor", 128, fmt="tokenizer"),
            _component("text", "shared-main", "text_weights", 1_000),
            _component("vision", "shared-main", "vision_weights", 1_000),
        ],
        text=profile["text"],
        vision=profile["vision"],
        processor=profile["processor"],
    )


def test_real_qwen3_vl_and_qwen35_profiles_are_path_free():
    vl = _profile("models/qwen3-vl-4b-instruct")
    qwen35 = _profile("models/qwen3-5-2b")
    assert vl["model_family"] == "qwen3_vl"
    assert vl["text"]["num_hidden_layers"] == 36
    assert vl["vision"]["output_hidden_size"] == 2560
    assert qwen35["model_family"] == "qwen3_5"
    assert qwen35["text"]["num_hidden_layers"] == 24
    assert qwen35["vision"]["output_hidden_size"] == 2048
    assert "path" not in json.dumps(vl)
    assert "path" not in json.dumps(qwen35)


def test_manifest_and_capacity_deduplicate_shared_native_weights():
    manifest = _manifest(family="qwen3_5")
    report = estimate_mm1_capacity(
        manifest,
        batch_size=1,
        visual_tokens=64,
        sequence_length=32,
        dtype="bfloat16",
        safety_margin=1.2,
    )
    assert report["unique_artifact_bytes"] == 1_000 + 128
    assert report["visual_weight_bytes"] == 1_000
    assert report["text_weight_bytes"] == 1_000
    assert report["handoff_bytes"] == 64 * 2048 * 2
    assert report["full_model_materialized"] is False


def test_visual_to_text_handoff_is_strict_and_path_free():
    manifest = _manifest()
    contract = build_mm1_handoff_contract(
        manifest=manifest,
        text_chain_id="a" * 64,
        generation=3,
        phase="prefill",
        source_node_id="vision-node",
        target_node_id="text-node",
        artifact={
            "artifact_id": "visual-hidden-1",
            "mode": "network",
            "size_bytes": 64 * 2560 * 2,
            "sha256": "b" * 64,
            "status": "committed",
        },
        shape=[1, 64, 2560],
        dtype="bfloat16",
        device="cpu",
        modality="image",
    )
    assert validate_mm1_handoff_contract(contract, manifest) == contract
    assert contract["full_model_materialized"] is False
    encoded = json.dumps(contract, ensure_ascii=True)
    assert "path" not in encoded
    assert "ticket" not in encoded
    changed = copy.deepcopy(contract)
    changed["tensor"]["shape"][2] = 2048
    with pytest.raises(Qwen3MultimodalContractError, match="visual hidden size"):
        validate_mm1_handoff_contract(changed, manifest)


def test_mm1_rejects_unsupported_runtime_and_media_shape():
    with pytest.raises(Qwen3MultimodalContractError, match="transformers_sidecar"):
        _manifest(family="qwen3_5", runtime="llama_cpp_mtmd")
    manifest = _manifest()
    with pytest.raises(Qwen3MultimodalContractError, match="video frames"):
        build_mm1_handoff_contract(
            manifest=manifest,
            text_chain_id="a" * 64,
            generation=3,
            phase="prefill",
            source_node_id="vision-node",
            target_node_id="text-node",
            artifact={
                "artifact_id": "visual-hidden-1", "mode": "local",
                "size_bytes": 128, "sha256": "c" * 64, "status": "committed",
            },
            shape=[1, 4, 2560], dtype="bfloat16", device="cpu",
            modality="video", frame_count=0,
        )


def test_mm1_manifest_rejects_component_identity_drift():
    manifest = _manifest()
    tampered = copy.deepcopy(manifest)
    tampered["components"][1]["size_bytes"] += 1
    with pytest.raises(Qwen3MultimodalContractError, match="shared MM1 artifact identity"):
        validate_mm1_model_manifest(tampered)


def test_mm1_transfer_binding_is_committed_and_path_free():
    manifest = _manifest()
    handoff = build_mm1_handoff_contract(
        manifest=manifest,
        text_chain_id="a" * 64,
        generation=2,
        phase="prefill",
        source_node_id="vision-node",
        target_node_id="text-node",
        artifact={
            "artifact_id": "qtx_visual_1",
            "mode": "network",
            "size_bytes": 256,
            "sha256": "d" * 64,
            "status": "committed",
        },
        shape=[1, 4, 2560], dtype="bfloat16", device="cpu", modality="image",
    )
    reference = {
        "artifact_id": "qtx_visual_1",
        "mode": "network",
        "source_node_id": "vision-node",
        "target_node_id": "text-node",
        "chain_id": "e" * 64,
        "generation": 2,
        "phase": "prefill",
        "from_segment": 0,
        "to_segment": 1,
        "size_bytes": 256,
        "sha256": "d" * 64,
    }
    binding = build_mm1_transfer_binding(
        handoff=handoff, manifest=manifest, transfer_reference=reference,
    )
    assert validate_mm1_transfer_binding(binding, manifest=manifest, handoff=handoff) == binding
    encoded = json.dumps(binding, ensure_ascii=True)
    assert "path" not in encoded and "ticket" not in encoded
    changed = copy.deepcopy(reference)
    changed["size_bytes"] = 128
    with pytest.raises(Qwen3MultimodalContractError, match="does not match"):
        build_mm1_transfer_binding(
            handoff=handoff, manifest=manifest, transfer_reference=changed,
        )
