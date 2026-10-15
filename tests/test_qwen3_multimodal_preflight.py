from __future__ import annotations

import copy
import hashlib
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
sys.path.insert(0, str(ROOT / "src"))

from qwen3_multimodal_contract import (  # noqa: E402
    build_mm1_model_manifest,
    build_mm1_model_profile,
)
from qwen3_multimodal_preflight import (  # noqa: E402
    Qwen3MultimodalPreflightError,
    build_mm1_visual_worker_request,
    build_mm1_visual_worker_response,
    inspect_mm1_processor_assets,
    validate_mm1_processor_inspection,
    validate_mm1_visual_worker_request,
    validate_mm1_visual_worker_response,
)


def _component(component_id: str, kind: str, digest: str) -> dict:
    return {
        "component_id": component_id,
        "artifact_id": f"{component_id}-artifact",
        "component_kind": kind,
        "format": "tokenizer" if kind == "processor" else "safetensors",
        "revision": "fixture-revision",
        "size_bytes": 128 if kind == "processor" else 1024,
        "sha256": digest * 64,
    }


def _manifest(model_name: str) -> dict:
    config = json.loads(
        (ROOT / "models" / model_name / "config.json").read_text(encoding="utf-8"),
    )
    profile = build_mm1_model_profile(config)
    return build_mm1_model_manifest(
        model_id=f"fixture-{profile['model_family']}-mm1",
        model_family=profile["model_family"],
        runtime="transformers_sidecar",
        revision="fixture-revision",
        components=[
            _component("processor", "processor", "a"),
            _component("text", "text_weights", "b"),
            _component("vision", "vision_weights", "c"),
        ],
        text=profile["text"],
        vision=profile["vision"],
        processor=profile["processor"],
    )


def _resign(value: dict, digest_field: str) -> None:
    unsigned = copy.deepcopy(value)
    unsigned.pop(digest_field, None)
    encoded = json.dumps(
        unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    value[digest_field] = hashlib.sha256(encoded).hexdigest()


def _prepared(model_name: str = "qwen3-vl-4b-instruct") -> tuple[dict, dict]:
    manifest = _manifest(model_name)
    inspection = inspect_mm1_processor_assets(ROOT / "models" / model_name, manifest)
    return manifest, inspection


@pytest.mark.parametrize(
    ("model_name", "model_family", "hidden_size"),
    [
        ("qwen3-vl-4b-instruct", "qwen3_vl", 2560),
        ("qwen3-5-2b", "qwen3_5", 2048),
    ],
)
def test_processor_asset_inspection_uses_real_local_metadata_and_stays_path_free(
    model_name: str,
    model_family: str,
    hidden_size: int,
):
    manifest, inspection = _prepared(model_name)
    assert validate_mm1_processor_inspection(inspection, manifest=manifest) == inspection
    assert inspection["model_family"] == model_family
    assert manifest["text"]["hidden_size"] == hidden_size
    assert inspection["processor"]["processor_class"] == "Qwen3VLProcessor"
    assert inspection["processor"]["tokenizer_class"] == "Qwen2Tokenizer"
    assert inspection["processor"]["patch_size"] == 16
    assert inspection["processor"]["temporal_patch_size"] == 2
    assert inspection["processor"]["spatial_merge_size"] == 2
    assert inspection["processor"]["image_min_pixels"] == 65_536
    assert inspection["processor"]["image_max_pixels"] == 4_194_304
    assert inspection["processor"]["video_min_pixels"] == 4_096
    assert inspection["processor"]["video_max_pixels"] == 4_194_304
    assert inspection["full_model_materialized"] is False
    assert inspection["weight_materialized"] is False
    encoded = json.dumps(inspection, ensure_ascii=True).lower()
    assert "path" not in encoded
    assert model_name.lower() not in encoded


def test_visual_worker_image_and_video_preflight_are_offline_and_path_free():
    manifest, inspection = _prepared()
    image = build_mm1_visual_worker_request(
        request_id="mm1-image-1",
        node_id="vision-node",
        manifest=manifest,
        inspection=inspection,
        component_ids=["processor", "vision"],
        modality="image",
        item_count=1,
        frame_count=0,
        width=256,
        height=256,
    )
    assert validate_mm1_visual_worker_request(
        image, manifest=manifest, inspection=inspection,
    ) == image
    image_response = build_mm1_visual_worker_response(
        image, manifest=manifest, inspection=inspection,
    )
    assert validate_mm1_visual_worker_response(image_response, request=image) == image_response
    assert image_response["status"] == "ready_for_offline_start"
    assert image_response["security"] == {
        "offline": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "network_disabled": True,
    }
    assert image_response["full_model_materialized"] is False
    assert image_response["weight_materialized"] is False

    video = build_mm1_visual_worker_request(
        request_id="mm1-video-1",
        node_id="vision-node",
        manifest=manifest,
        inspection=inspection,
        component_ids=["processor", "vision"],
        modality="video",
        item_count=1,
        frame_count=8,
        width=128,
        height=128,
    )
    assert video["media"]["frame_count"] == 8
    assert video["media"]["pixel_count"] == 16_384
    encoded = json.dumps({"request": image, "response": image_response, "video": video}).lower()
    for forbidden in ("model_path", "image_bytes", "video_bytes", "pixel_values", "prompt"):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    ("modality", "frame_count", "width", "height", "message"),
    [
        ("image", 0, 64, 64, "outside processor limits"),
        ("image", 1, 256, 256, "cannot declare video frames"),
        ("video", 0, 128, 128, "outside limits"),
        ("video", 129, 128, 128, "outside limits"),
        ("video", 1, 4096, 4096, "outside processor limits"),
    ],
)
def test_visual_worker_rejects_media_outside_processor_budget(
    modality: str,
    frame_count: int,
    width: int,
    height: int,
    message: str,
):
    manifest, inspection = _prepared()
    with pytest.raises(Qwen3MultimodalPreflightError, match=message):
        build_mm1_visual_worker_request(
            request_id="mm1-media-reject",
            node_id="vision-node",
            manifest=manifest,
            inspection=inspection,
            component_ids=["processor", "vision"],
            modality=modality,
            item_count=1,
            frame_count=frame_count,
            width=width,
            height=height,
        )


@pytest.mark.parametrize(
    "component_ids",
    [
        ["processor"],
        ["processor", "text", "vision"],
        ["text", "vision"],
    ],
)
def test_visual_worker_rejects_non_minimal_component_assignment(component_ids: list[str]):
    manifest, inspection = _prepared()
    with pytest.raises(Qwen3MultimodalPreflightError, match="least-privilege"):
        build_mm1_visual_worker_request(
            request_id="mm1-components-reject",
            node_id="vision-node",
            manifest=manifest,
            inspection=inspection,
            component_ids=component_ids,
            modality="image",
            item_count=1,
            frame_count=0,
            width=256,
            height=256,
        )


def test_mtmd_visual_worker_selects_processor_and_mmproj_only():
    config = json.loads(
        (ROOT / "models" / "qwen3-vl-4b-instruct" / "config.json").read_text(
            encoding="utf-8",
        ),
    )
    profile = build_mm1_model_profile(config)
    components = [
        _component("mmproj", "mmproj", "d"),
        _component("processor", "processor", "a"),
        _component("text", "text_weights", "b"),
    ]
    components[0]["format"] = "gguf"
    components[2]["format"] = "gguf"
    manifest = build_mm1_model_manifest(
        model_id="fixture-qwen3-vl-mtmd",
        model_family="qwen3_vl",
        runtime="llama_cpp_mtmd",
        revision="fixture-revision",
        components=components,
        text=profile["text"],
        vision=profile["vision"],
        processor=profile["processor"],
    )
    inspection = inspect_mm1_processor_assets(
        ROOT / "models" / "qwen3-vl-4b-instruct", manifest,
    )
    request = build_mm1_visual_worker_request(
        request_id="mm1-mtmd",
        node_id="vision-node",
        manifest=manifest,
        inspection=inspection,
        component_ids=["mmproj", "processor"],
        modality="image",
        item_count=1,
        frame_count=0,
        width=256,
        height=256,
    )
    assert request["runtime"] == "llama_cpp_mtmd"
    assert request["component_ids"] == ["mmproj", "processor"]


def test_visual_worker_rejects_network_or_remote_code_policy():
    manifest, inspection = _prepared()
    request = build_mm1_visual_worker_request(
        request_id="mm1-security-reject",
        node_id="vision-node",
        manifest=manifest,
        inspection=inspection,
        component_ids=["processor", "vision"],
        modality="image",
        item_count=1,
        frame_count=0,
        width=256,
        height=256,
    )
    for field, value in (("network_disabled", False), ("trust_remote_code", True)):
        tampered = copy.deepcopy(request)
        tampered["security"][field] = value
        _resign(tampered, "request_sha256")
        with pytest.raises(Qwen3MultimodalPreflightError, match="offline-safe"):
            validate_mm1_visual_worker_request(
                tampered, manifest=manifest, inspection=inspection,
            )


def test_visual_worker_detects_inspection_request_and_response_tampering():
    manifest, inspection = _prepared("qwen3-5-2b")
    damaged_inspection = copy.deepcopy(inspection)
    damaged_inspection["assets"][0]["size_bytes"] += 1
    with pytest.raises(Qwen3MultimodalPreflightError, match="digest mismatch"):
        validate_mm1_processor_inspection(damaged_inspection, manifest=manifest)

    request = build_mm1_visual_worker_request(
        request_id="mm1-tamper",
        node_id="vision-node",
        manifest=manifest,
        inspection=inspection,
        component_ids=["processor", "vision"],
        modality="image",
        item_count=1,
        frame_count=0,
        width=256,
        height=256,
    )
    damaged_request = copy.deepcopy(request)
    damaged_request["media"]["width"] = 512
    with pytest.raises(Qwen3MultimodalPreflightError, match="pixel_count does not match"):
        validate_mm1_visual_worker_request(
            damaged_request, manifest=manifest, inspection=inspection,
        )

    response = build_mm1_visual_worker_response(
        request, manifest=manifest, inspection=inspection,
    )
    damaged_response = copy.deepcopy(response)
    damaged_response["component_count"] = 3
    _resign(damaged_response, "response_sha256")
    with pytest.raises(Qwen3MultimodalPreflightError, match="component count differs"):
        validate_mm1_visual_worker_response(damaged_response, request=request)


def test_visual_worker_request_schema_rejects_path_or_payload_fields():
    manifest, inspection = _prepared()
    request = build_mm1_visual_worker_request(
        request_id="mm1-sensitive",
        node_id="vision-node",
        manifest=manifest,
        inspection=inspection,
        component_ids=["processor", "vision"],
        modality="image",
        item_count=1,
        frame_count=0,
        width=256,
        height=256,
    )
    for field in ("model_path", "image_bytes"):
        tampered = copy.deepcopy(request)
        tampered[field] = "forbidden"
        _resign(tampered, "request_sha256")
        with pytest.raises(Qwen3MultimodalPreflightError, match="fields are invalid"):
            validate_mm1_visual_worker_request(
                tampered, manifest=manifest, inspection=inspection,
            )
