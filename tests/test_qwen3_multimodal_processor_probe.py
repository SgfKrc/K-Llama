from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]

# CI 不拉模型工件（本机 models 是 Junction）。缺工件按项目约定跳过。
pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "qwen3-vl-4b-instruct" / "config.json").exists(),
    reason="缺本地工件 models/qwen3-vl-4b-instruct（CI 不拉模型）",
)
sys.path.insert(0, str(ROOT / "src"))

from qwen3_multimodal_contract import build_mm1_model_manifest, build_mm1_model_profile  # noqa: E402
from qwen3_multimodal_preflight import (  # noqa: E402
    Qwen3MultimodalPreflightError,
    build_mm1_visual_worker_request,
    inspect_mm1_processor_assets,
    validate_mm1_processor_smoke_response,
)
from scripts.model_tools.qwen3_multimodal_processor_probe import (  # noqa: E402
    run_qwen3_multimodal_processor_probe,
)
from scripts.model_tools.qwen3_multimodal_processor_probe_worker import (  # noqa: E402
    execute_request,
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


def _manifest(model_name: str = "qwen3-vl-4b-instruct") -> dict:
    config = json.loads(
        (ROOT / "models" / model_name / "config.json").read_text(encoding="utf-8"),
    )
    profile = build_mm1_model_profile(config)
    return build_mm1_model_manifest(
        model_id=f"fixture-{profile['model_family']}-processor-smoke",
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


def _prepared(model_name: str = "qwen3-vl-4b-instruct") -> tuple[dict, dict, dict]:
    manifest = _manifest(model_name)
    inspection = inspect_mm1_processor_assets(ROOT / "models" / model_name, manifest)
    request = build_mm1_visual_worker_request(
        request_id=f"processor-smoke-{manifest['model_family']}",
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
    return manifest, inspection, request


def _fake_transformers(*, version: str = "4.57.6", kwargs_sink: list[dict] | None = None):
    class Qwen2VLImageProcessorFast:
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2

        def __call__(self, images, **kwargs):
            # MM1.7：合成媒体预处理——返回模拟 pixel_values（不加载权重）
            import numpy as np
            arr = np.asarray(images)
            h, w = arr.shape[:2]
            return {"pixel_values": np.zeros((1, 3, h, w), dtype=np.float16)}

    class Qwen3VLVideoProcessor:
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2

        def __call__(self, frames, **kwargs):
            import numpy as np
            arr = np.asarray(frames)
            n, h, w = arr.shape[:3]
            return {"pixel_values": np.zeros((1, n, 3, h, w), dtype=np.float16)}

    class Qwen2TokenizerFast:
        image_token_id = 151655
        video_token_id = 151656

    class Qwen3VLProcessor:
        image_token_id = 151655
        video_token_id = 151656

        def __init__(self):
            self.image_processor = Qwen2VLImageProcessorFast()
            self.video_processor = Qwen3VLVideoProcessor()
            self.tokenizer = Qwen2TokenizerFast()

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            if kwargs_sink is not None:
                kwargs_sink.append(kwargs)
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return Qwen3VLProcessor()

    return SimpleNamespace(__version__=version, AutoProcessor=AutoProcessor)


def _worker_request(model_name: str = "qwen3-vl-4b-instruct") -> tuple[dict, dict, dict]:
    manifest, inspection, visual_request = _prepared(model_name)
    return ({
        "schema_version": 1,
        "tool": "qwen3_multimodal_processor_probe",
        "operation": "qwen3_visual_worker_processor_smoke",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(ROOT / "models" / model_name),
        "manifest": manifest,
        "visual_request": visual_request,
        "controller_python": str(Path(sys.executable).with_name("controller.exe")),
    }, manifest, visual_request)


def test_processor_worker_constructs_real_contract_with_fake_isolated_transformers():
    request, _manifest_value, visual_request = _worker_request()
    kwargs_seen: list[dict] = []
    result = execute_request(
        request,
        module_loader=lambda name: _fake_transformers(kwargs_sink=kwargs_seen),
    )
    assert result["status"] == "ready_for_offline_start"
    assert result["gate_passed"] is True
    assert result["response"]["runtime"]["processor_class"] == "Qwen3VLProcessor"
    assert result["response"]["runtime"]["tokenizer_class"] == "Qwen2TokenizerFast"
    assert result["response"]["runtime"]["image_token_id"] == visual_request["processor"]["image_token_id"]
    assert result["response"]["cleanup"]["completed"] is True
    assert result["response"]["cleanup"]["weight_materialized"] is False
    assert kwargs_seen == [{"local_files_only": True, "trust_remote_code": False}]
    assert str(ROOT) not in json.dumps(result)
    assert "model_path" not in json.dumps(result).lower()


def test_processor_worker_rejects_old_runtime_and_contract_drift():
    request, manifest, visual_request = _worker_request("qwen3-5-2b")
    old = execute_request(
        request,
        module_loader=lambda name: _fake_transformers(version="4.47.1"),
    )
    assert old["status"] == "runtime_rejected"
    assert old["errors"][0]["code"] == "transformers_too_old"

    drifted = dict(visual_request)
    drifted["processor"] = dict(visual_request["processor"])
    drifted["processor"]["patch_size"] = 32
    request["visual_request"] = drifted
    rejected = execute_request(
        request,
        module_loader=lambda name: _fake_transformers(),
    )
    assert rejected["status"] == "artifact_rejected"
    assert rejected["errors"][0]["code"] == "mm1_preflight_rejected"


def test_processor_worker_rejects_missing_processor_metadata(tmp_path: Path):
    source = ROOT / "models" / "qwen3-vl-4b-instruct"
    target = tmp_path / "qwen3-vl-metadata"
    target.mkdir()
    for name in (
        "config.json", "preprocessor_config.json", "video_preprocessor_config.json",
        "tokenizer_config.json",
    ):
        shutil.copy2(source / name, target / name)
    (target / "video_preprocessor_config.json").unlink()
    request, _manifest_value, _visual_request = _worker_request()
    request["model_path"] = str(target)
    result = execute_request(
        request,
        module_loader=lambda name: _fake_transformers(),
    )
    assert result["status"] == "artifact_rejected"
    assert result["errors"][0]["code"] == "mm1_preflight_rejected"
    assert str(target) not in json.dumps(result)


def test_processor_worker_rejects_class_drift_and_network_protocol():
    request, _manifest_value, _visual_request = _worker_request()

    class WrongImageProcessor:
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2

    transformers = _fake_transformers()
    original = transformers.AutoProcessor.from_pretrained

    class DriftedAutoProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            processor = original(path, **kwargs)
            processor.image_processor = WrongImageProcessor()
            return processor

    transformers.AutoProcessor = DriftedAutoProcessor
    drifted = execute_request(request, module_loader=lambda name: transformers)
    assert drifted["status"] == "processor_contract_rejected"
    assert drifted["errors"][0]["code"] == "processor_contract_rejected"

    request["network_access"] = "enabled"
    unsafe = execute_request(request, module_loader=lambda name: _fake_transformers())
    assert unsafe["valid"] is False
    assert unsafe["status"] == "invalid_request"


def test_processor_smoke_response_contract_rejects_incomplete_cleanup():
    manifest, inspection, request = _prepared("qwen3-5-2b")
    runtime = {
        "transformers_version": "4.57.6",
        "isolated": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "processor_class": "Qwen3VLProcessor",
        "image_processor_class": "Qwen2VLImageProcessorFast",
        "video_processor_class": "Qwen3VLVideoProcessor",
        "tokenizer_class": "Qwen2TokenizerFast",
        "declared_tokenizer_class": "Qwen2Tokenizer",
        "image_token_id": manifest["processor"]["image_token_id"],
        "video_token_id": manifest["processor"]["video_token_id"],
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
    }
    from qwen3_multimodal_preflight import build_mm1_processor_smoke_response

    response = build_mm1_processor_smoke_response(
        request, manifest=manifest, inspection=inspection, runtime=runtime,
    )
    response["cleanup"]["completed"] = False
    with pytest.raises(Qwen3MultimodalPreflightError, match="cleanup is incomplete"):
        validate_mm1_processor_smoke_response(response, request=request)


def test_processor_controller_maps_worker_failure_and_repeat_cleanup():
    manifest, _inspection, request = _prepared()
    failed = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        worker_runner=lambda payload, timeout: (_ for _ in ()).throw(RuntimeError("worker exited")),
    )
    assert failed["status"] == "worker_failed"
    assert failed["errors"][0]["code"] == "worker_runner_failed"

    def runner(payload, timeout):
        isolated_payload = dict(payload)
        isolated_payload["controller_python"] = str(Path(sys.executable).with_name("controller.exe"))
        return execute_request(isolated_payload, module_loader=lambda name: _fake_transformers())

    first = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        worker_runner=runner,
    )
    second = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        worker_runner=runner,
    )
    assert first["response"] == second["response"]
    assert first["response"]["cleanup"]["completed"] is True


@pytest.mark.real_model
@pytest.mark.parametrize("model_name", ["qwen3-vl-4b-instruct", "qwen3-5-2b"])
def test_real_isolated_autoprocessor_smoke(model_name: str):
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.6 requires the isolated Qwen3 pipeline sidecar")
    manifest, _inspection, request = _prepared(model_name)
    report = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / model_name,
        manifest=manifest,
        visual_request=request,
        timeout_seconds=180,
        media_smoke={"image_size": [32, 32], "video_size": [32, 32], "video_frames": 2},
    )
    assert report["status"] == "ready_for_offline_start"
    assert report["gate_passed"] is True
    response = report["response"]
    assert response["processor_constructed"] is True
    # MM1.7：真实 processor 媒体预处理摘要（shape/dtype/token 数，无权重）
    summary = response["media_summary"]
    assert summary is not None
    assert summary["weight_materialized"] is False
    # 真实 processor 会 resize/网格化（shape 维度数因实现而异）——
    # 契约只投影存在性、dtype 与正数 token 数
    assert len(summary["image"]["pixel_values_shape"]) >= 2
    assert "float" in summary["image"]["dtype"]
    assert summary["image"]["token_count_estimate"] and summary["image"]["token_count_estimate"] > 0
    if summary["video"]["pixel_values_shape"]:
        # 真实 video_processor 对合成帧可能不产张量——非空时校验
        assert len(summary["video"]["pixel_values_shape"]) >= 2
        assert "float" in summary["video"]["dtype"]
        assert summary["video"]["token_count_estimate"] and summary["video"]["token_count_estimate"] > 0
    # 版本：sidecar 现随主运行时统一到 5.17（此前 4.57.6）⇒ 断言「不低于契约下限」比钉死小版本更稳
    # （契约本身的检查在 src/qwen3_multimodal_preflight.py，下限 4.51.0）。
    _ver = tuple(int(p) for p in response["runtime"]["transformers_version"].split(".")[:2])
    assert _ver >= (4, 51), response["runtime"]["transformers_version"]
    assert response["runtime"]["processor_class"] == "Qwen3VLProcessor"
    # 5.x 去掉了 `Fast` 后缀（`Qwen2VLImageProcessorFast` → `Qwen2VLImageProcessor`；
    # `Qwen2TokenizerFast` → `Qwen2Tokenizer`）⇒ 两者都接受，与 preflight 的允许集合一致。
    assert response["runtime"]["image_processor_class"] in {
        "Qwen2VLImageProcessorFast", "Qwen2VLImageProcessor"}
    assert response["runtime"]["video_processor_class"] == "Qwen3VLVideoProcessor"
    assert response["runtime"]["tokenizer_class"] in {"Qwen2TokenizerFast", "Qwen2Tokenizer"}
    assert response["cleanup"]["completed"] is True
    assert response["cleanup"]["weight_materialized"] is False
    encoded = json.dumps(report, ensure_ascii=True).lower()
    assert str(ROOT).lower() not in encoded
    assert "model_path" not in encoded


# ================================================================
# MM1.7：CPU 合成媒体预处理与张量摘要合同
# ================================================================

def test_media_preprocess_projects_shape_dtype_and_tokens():
    """合成图像/帧经 processor 预处理后投影摘要（shape/dtype/token 数）。"""
    request, _manifest, _visual = _worker_request()
    request["media_smoke"] = {"image_size": [64, 64], "video_size": [32, 32], "video_frames": 3}
    result = execute_request(
        request, module_loader=lambda name: _fake_transformers(),
    )
    assert result["gate_passed"] is True
    summary = result["response"]["media_summary"]
    assert summary is not None
    assert summary["weight_materialized"] is False
    assert summary["full_model_materialized"] is False
    # 图像：shape [1,3,H,W]、fp16、token 网格 = H*W
    assert summary["image"]["pixel_values_shape"][-2:] == [64, 64]
    assert summary["image"]["dtype"] == "float16"
    assert summary["image"]["token_count_estimate"] == (64 // 16) * (64 // 16)  # patch=16
    # 视频：shape [1,F,3,H,W]
    assert summary["video"]["pixel_values_shape"][1] == 3
    assert summary["video"]["pixel_values_shape"][-2:] == [32, 32]
    assert summary["video"]["token_count_estimate"] == (32 // 16) * (32 // 16)
    assert summary["output_bytes_estimate"] > 0


def test_media_preprocess_rejects_out_of_bounds_media():
    """超限尺寸/帧数 fail-closed（MM1.7 契约限制）。"""
    for bad_media in (
        {"image_size": [2048, 64]},            # 图像边长超 1024
        {"video_size": [64, 4096]},            # 视频边长超 1024
        {"video_frames": 64},                  # 帧数超 32
        {"video_frames": 0},                   # 空媒体
        {"image_size": [2, 2]},                # 过小
    ):
        request, _manifest, _visual = _worker_request()
        request["media_smoke"] = bad_media
        result = execute_request(
            request, module_loader=lambda name: _fake_transformers(),
        )
        assert result["gate_passed"] is False, bad_media
        assert result["errors"][0]["code"] in (
            "processor_contract_rejected", "processor_smoke_failed",
        )


def test_media_summary_contains_no_sensitive_data():
    """摘要不含像素值/原始媒体/路径（path-free 合同）。"""
    request, _manifest, _visual = _worker_request()
    request["media_smoke"] = {"image_size": [32, 32], "video_frames": 2}
    result = execute_request(
        request, module_loader=lambda name: _fake_transformers(),
    )
    payload = json.dumps(result, ensure_ascii=True)
    # 字段名 pixel_values_shape 允许；实际像素数组/原始媒体/路径禁止
    assert '"pixel_values":' not in payload
    assert "models/" not in payload
    assert "rng" not in payload


# ================================================================
# MM1.8：媒体张量跨边界投影与视觉组件占位
# ================================================================

def test_media_tensor_reference_projects_capacity_and_stays_path_free():
    """媒体摘要投影为 path-free 张量参考（shape/dtype/token/容量预算）。"""
    from qwen3_multimodal_preflight import (
        build_mm1_media_tensor_reference,
        validate_mm1_media_tensor_reference,
    )

    summary = {
        "image": {"pixel_values_shape": [1, 3, 64, 64], "dtype": "float16",
                  "token_count_estimate": 16},
        "video": {"pixel_values_shape": [1, 2, 3, 32, 32], "dtype": "float16",
                  "token_count_estimate": 8},
        "output_bytes_estimate": 4096,
        "weight_materialized": False,
        "full_model_materialized": False,
    }
    reference = build_mm1_media_tensor_reference(
        summary, model_id="qwen3-vl-4b-instruct",
        component_ids=["vision_tower", "text_segment_0"],
    )
    validated = validate_mm1_media_tensor_reference(
        reference, model_id="qwen3-vl-4b-instruct",
        component_ids=["vision_tower", "text_segment_0"],
    )
    assert validated["reference_kind"] == "qwen3_visual_media_tensor_placeholder"
    assert validated["media"]["image"]["token_count_estimate"] == 16
    assert validated["media"]["video"]["token_count_estimate"] == 8
    assert validated["capacity"]["total_media_tokens"] == 24
    assert validated["capacity"]["output_bytes_estimate"] == 4096
    assert validated["weight_materialized"] is False
    payload = json.dumps(validated, ensure_ascii=True)
    assert "pixel_values\"" not in payload
    assert "models/" not in payload


def test_media_tensor_reference_rejects_bad_input(monkeypatch):
    """空/畸形媒体摘要或身份不符 fail-closed。"""
    from qwen3_multimodal_preflight import (
        Qwen3MultimodalPreflightError,
        build_mm1_media_tensor_reference,
        validate_mm1_media_tensor_reference,
    )

    for bad in (None, {}, {"image": {}, "video": {}}):
        try:
            build_mm1_media_tensor_reference(
                bad, model_id="qwen3-vl-4b-instruct", component_ids=["vision_tower"],
            )
        except Qwen3MultimodalPreflightError:
            pass
        else:
            raise AssertionError(f"应拒绝 {bad!r}")
    summary = {
        "image": {"pixel_values_shape": [1, 3, 64, 64], "token_count_estimate": 16},
        "video": {"pixel_values_shape": [], "token_count_estimate": 0},
    }
    reference = build_mm1_media_tensor_reference(
        summary, model_id="qwen3-vl-4b-instruct", component_ids=["vision_tower"],
    )
    # 身份不符（model_id 不同）拒绝
    try:
        validate_mm1_media_tensor_reference(
            reference, model_id="other-model", component_ids=["vision_tower"],
        )
    except Qwen3MultimodalPreflightError:
        pass
    else:
        raise AssertionError("身份不符应拒绝")
    # 篡改 digest 拒绝
    tampered = dict(reference)
    tampered["capacity"] = dict(reference["capacity"])
    tampered["capacity"]["total_media_tokens"] = 999
    try:
        validate_mm1_media_tensor_reference(
            tampered, model_id="qwen3-vl-4b-instruct", component_ids=["vision_tower"],
        )
    except Qwen3MultimodalPreflightError:
        pass
    else:
        raise AssertionError("篡改应拒绝")


def test_real_media_tensor_reference_in_response():
    """真实双模型：响应含 media_tensor_reference（与摘要解耦、无权重）。"""
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.8 requires the isolated Qwen3 pipeline sidecar")
    manifest, _inspection, request = _prepared("qwen3-vl-4b-instruct")
    report = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        timeout_seconds=180,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
    )
    assert report["gate_passed"] is True
    reference = report["response"]["media_tensor_reference"]
    assert reference is not None
    assert reference["reference_kind"] == "qwen3_visual_media_tensor_placeholder"
    assert reference["weight_materialized"] is False
    assert reference["capacity"]["total_media_tokens"] > 0
    encoded = json.dumps(report, ensure_ascii=True).lower()
    assert "pixel_values\"" not in encoded
    assert str(ROOT).lower() not in encoded


# ================================================================
# MM1.9：视觉塔执行器输入占位与容量预算接线
# ================================================================

def _sample_reference():
    from qwen3_multimodal_preflight import build_mm1_media_tensor_reference
    return build_mm1_media_tensor_reference(
        {
            "image": {"pixel_values_shape": [1, 3, 64, 64], "dtype": "float16",
                      "token_count_estimate": 16},
            "video": {"pixel_values_shape": [1, 2, 3, 32, 32], "dtype": "float16",
                      "token_count_estimate": 8},
            "output_bytes_estimate": 4096,
            "weight_materialized": False,
            "full_model_materialized": False,
        },
        model_id="qwen3-vl-4b-instruct",
        component_ids=["vision_tower", "text_segment_0"],
    )


def test_visual_input_contract_admitted_within_budget():
    """容量充足 → admitted=True，契约含 grid/token/字节预算且 path-free。"""
    from qwen3_multimodal_preflight import (
        build_mm1_visual_input_contract,
        validate_mm1_visual_input_contract,
    )
    reference = _sample_reference()
    contract = build_mm1_visual_input_contract(
        reference, node_capacity_bytes=8192, safety_margin=1.2,
    )
    validated = validate_mm1_visual_input_contract(contract, media_tensor_reference=reference)
    assert validated["contract_kind"] == "qwen3_visual_input_placeholder"
    assert validated["media_reference_sha256"] == reference["reference_sha256"]
    assert validated["input"]["total_media_tokens"] == 24
    assert validated["capacity"]["admitted"] is True
    assert validated["capacity"]["required_bytes"] == int(4096 * 1.2)
    payload = json.dumps(validated, ensure_ascii=True)
    assert "pixel_values\"" not in payload
    assert "models/" not in payload


def test_visual_input_contract_fails_closed_on_budget_shortage():
    """预算不足 → admitted=False（fail-closed，视觉塔不执行）。"""
    from qwen3_multimodal_preflight import (
        Qwen3MultimodalPreflightError,
        build_mm1_visual_input_contract,
        validate_mm1_visual_input_contract,
    )
    reference = _sample_reference()
    contract = build_mm1_visual_input_contract(
        reference, node_capacity_bytes=1024, safety_margin=1.2,
    )
    assert contract["capacity"]["admitted"] is False
    validated = validate_mm1_visual_input_contract(contract, media_tensor_reference=reference)
    assert validated["capacity"]["admitted"] is False
    # 篡改 admitted（不足却标 True）→ 拒绝
    tampered = dict(contract)
    tampered["capacity"] = dict(contract["capacity"])
    tampered["capacity"]["admitted"] = True
    try:
        validate_mm1_visual_input_contract(tampered, media_tensor_reference=reference)
    except Qwen3MultimodalPreflightError:
        pass
    else:
        raise AssertionError("admitted 不一致应拒绝")


def test_real_visual_input_contract_in_response():
    """真实模型：响应含 visual_input_contract（容量比对，无权重）。"""
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.9 requires the isolated Qwen3 pipeline sidecar")
    manifest, _inspection, request = _prepared("qwen3-vl-4b-instruct")
    report = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        timeout_seconds=180,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
    )
    assert report["gate_passed"] is True
    # 未给 node_capacity → 契约为 None（接线点可选）
    assert report["response"]["visual_input_contract"] is None
    # 带容量再跑：契约非 None（接线点生效）
    report2 = run_qwen3_multimodal_processor_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        manifest=manifest,
        visual_request=request,
        timeout_seconds=180,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
        node_capacity_bytes=1024 * 1024,
    )
    contract = report2["response"]["visual_input_contract"]
    assert contract is not None
    assert contract["contract_kind"] == "qwen3_visual_input_placeholder"
    assert contract["capacity"]["node_capacity_bytes"] == 1024 * 1024
    encoded = json.dumps(report2, ensure_ascii=True).lower()
    assert "pixel_values\"" not in encoded
    assert str(ROOT).lower() not in encoded


# ================================================================
# MM1.10：视觉塔组件容量规划与多模态资源账本
# ================================================================

def test_resource_ledger_commits_and_is_idempotent():
    """统一入账：视觉塔权重+媒体输入+文本段；重复入账不重复计费。"""
    from qwen3_multimodal_preflight import (
        mm1_ledger_commit,
        validate_mm1_resource_ledger,
    )
    entries = [
        {"entry_id": "vision_tower", "kind": "vision_tower_weights", "bytes": 800_000_000},
        {"entry_id": "media_1", "kind": "media_input", "bytes": 4_000_000},
        {"entry_id": "text_0", "kind": "text_segment", "bytes": 200_000_000},
    ]
    ledger = mm1_ledger_commit(entries, ledger_id="node-b", node_capacity_bytes=1_200_000_000)
    validated = validate_mm1_resource_ledger(ledger)
    assert validated["capacity"]["total_bytes"] == 1_004_000_000
    assert validated["capacity"]["admitted"] is True
    assert validated["capacity"]["remaining_bytes"] == 1_200_000_000 - 1_004_000_000
    # 幂等：同 ledger_id 重复入账（相同 entry_id）不翻倍
    again = mm1_ledger_commit(
        entries + entries, ledger_id="node-b", node_capacity_bytes=1_200_000_000,
    )
    assert validate_mm1_resource_ledger(again)["capacity"]["total_bytes"] == 1_004_000_000


def test_resource_ledger_fails_closed_on_combined_overrun():
    """组合超限 → admitted=false（fail-closed）；admitted 篡改拒绝。"""
    from qwen3_multimodal_preflight import (
        Qwen3MultimodalPreflightError,
        mm1_ledger_commit,
        validate_mm1_resource_ledger,
    )
    entries = [
        {"entry_id": "vision_tower", "kind": "vision_tower_weights", "bytes": 900_000_000},
        {"entry_id": "media_1", "kind": "media_input", "bytes": 400_000_000},
    ]
    ledger = mm1_ledger_commit(entries, ledger_id="node-c", node_capacity_bytes=1_000_000_000)
    assert ledger["capacity"]["admitted"] is False
    validate_mm1_resource_ledger(ledger)  # admitted=False 是合法终态
    tampered = dict(ledger)
    tampered["capacity"] = dict(ledger["capacity"])
    tampered["capacity"]["admitted"] = True
    try:
        validate_mm1_resource_ledger(tampered)
    except Qwen3MultimodalPreflightError:
        pass
    else:
        raise AssertionError("超限却标 admitted=True 应拒绝")


def test_resource_ledger_release_is_idempotent():
    """释放条目：total 减少；重复释放为 no-op（幂等）。"""
    from qwen3_multimodal_preflight import (
        mm1_ledger_commit,
        mm1_ledger_release,
        validate_mm1_resource_ledger,
    )
    entries = [
        {"entry_id": "vision_tower", "kind": "vision_tower_weights", "bytes": 800_000_000},
        {"entry_id": "media_1", "kind": "media_input", "bytes": 4_000_000},
    ]
    ledger = mm1_ledger_commit(entries, ledger_id="node-d", node_capacity_bytes=1_000_000_000)
    released = mm1_ledger_release(ledger, entry_id="media_1")
    validated = validate_mm1_resource_ledger(released)
    assert validated["capacity"]["total_bytes"] == 800_000_000
    # 重复释放（media_1 已不在）→ no-op
    again = mm1_ledger_release(released, entry_id="media_1")
    assert validate_mm1_resource_ledger(again)["capacity"]["total_bytes"] == 800_000_000


# ================================================================
# MM1.11：视觉塔组件放置与纯文本请求守卫
# ================================================================

def _ledger(*, total_entries, capacity_bytes):
    from qwen3_multimodal_preflight import mm1_ledger_commit
    return mm1_ledger_commit(
        total_entries, ledger_id="node-e", node_capacity_bytes=capacity_bytes,
    )


def test_pure_text_request_never_activates_vision_tower():
    """纯文本请求守卫：无媒体 → 视觉塔不激活（文本段独立可执行）。"""
    from qwen3_multimodal_preflight import (
        mm1_vision_tower_placement,
        validate_mm1_vision_tower_placement,
    )
    ledger = _ledger(
        total_entries=[
            {"entry_id": "text_0", "kind": "text_segment", "bytes": 200_000_000},
        ],
        capacity_bytes=1_000_000_000,
    )
    decision = mm1_vision_tower_placement(
        ledger, request_has_media=False, vision_tower_bytes=800_000_000,
    )
    validated = validate_mm1_vision_tower_placement(decision, ledger=ledger)
    assert validated["vision_tower_active"] is False
    assert validated["reason"] == "text_only_request_guard"
    assert validated["capacity"]["admitted"] is True  # 文本段独立可执行
    payload = json.dumps(validated, ensure_ascii=True)
    assert "models/" not in payload


def test_vision_tower_placement_admitted_with_capacity():
    """有媒体 + 容量足够 → 视觉塔可放置（active=true）。"""
    from qwen3_multimodal_preflight import (
        mm1_vision_tower_placement,
        validate_mm1_vision_tower_placement,
    )
    ledger = _ledger(
        total_entries=[
            {"entry_id": "media_1", "kind": "media_input", "bytes": 4_000_000},
            {"entry_id": "text_0", "kind": "text_segment", "bytes": 200_000_000},
        ],
        capacity_bytes=1_500_000_000,
    )
    decision = mm1_vision_tower_placement(
        ledger, request_has_media=True, vision_tower_bytes=800_000_000,
    )
    validated = validate_mm1_vision_tower_placement(decision, ledger=ledger)
    assert validated["vision_tower_active"] is True
    assert validated["reason"] == "capacity_admitted"


def test_vision_tower_placement_fails_closed_without_capacity():
    """有媒体 + 容量不足 → active=false（fail-closed，视觉塔不执行）。"""
    from qwen3_multimodal_preflight import (
        Qwen3MultimodalPreflightError,
        mm1_vision_tower_placement,
        validate_mm1_vision_tower_placement,
    )
    ledger = _ledger(
        total_entries=[
            {"entry_id": "media_1", "kind": "media_input", "bytes": 900_000_000},
            {"entry_id": "text_0", "kind": "text_segment", "bytes": 200_000_000},
        ],
        capacity_bytes=1_000_000_000,
    )
    decision = mm1_vision_tower_placement(
        ledger, request_has_media=True, vision_tower_bytes=800_000_000,
    )
    assert decision["vision_tower_active"] is False
    assert decision["reason"] == "vision_tower_capacity_insufficient"
    validate_mm1_vision_tower_placement(decision, ledger=ledger)
    # 篡改：无媒体却标 active=True → 拒绝（纯文本守卫）
    tampered = dict(decision)
    tampered["request_has_media"] = False
    try:
        validate_mm1_vision_tower_placement(tampered, ledger=ledger)
    except Qwen3MultimodalPreflightError:
        pass
    else:
        raise AssertionError("纯文本守卫被绕过应拒绝")


# ================================================================
# MM1.12：视觉塔执行器骨架与 text-only 会话回归
# ================================================================

def _placement(*, has_media: bool, active: bool):
    from qwen3_multimodal_preflight import (
        mm1_ledger_commit,
        mm1_vision_tower_placement,
    )
    ledger = mm1_ledger_commit(
        [{"entry_id": "text_0", "kind": "text_segment", "bytes": 100_000_000}],
        ledger_id="node-f", node_capacity_bytes=1_000_000_000,
    )
    return mm1_vision_tower_placement(
        ledger, request_has_media=has_media,
        # inactive = 视觉塔字节超剩余容量（ledger 1GB、text 100MB → 剩 900MB）
        vision_tower_bytes=(
            400_000_000 if active else 1_200_000_000
        ) if has_media else 0,
    )


def _media_reference():
    from qwen3_multimodal_preflight import build_mm1_media_tensor_reference
    return build_mm1_media_tensor_reference(
        {
            "image": {"pixel_values_shape": [1, 3, 64, 64], "dtype": "float16",
                      "token_count_estimate": 16},
            "video": {"pixel_values_shape": [], "token_count_estimate": 0},
            "output_bytes_estimate": 4096,
            "weight_materialized": False,
            "full_model_materialized": False,
        },
        model_id="qwen3-vl-4b-instruct",
        component_ids=["vision_tower", "text_segment_0"],
    )


def test_visual_skeleton_skips_vision_tower_for_text_only():
    """text-only 会话：visual_path=skipped，全程不触碰视觉塔。"""
    from qwen3_multimodal_runtime import (
        Qwen3MultimodalRuntimeError,
        run_mm1_visual_tower_skeleton,
    )
    placement = _placement(has_media=False, active=False)
    result = run_mm1_visual_tower_skeleton(placement, None, text_only=True)
    assert result["visual_path"] == "skipped"
    assert result["vision_tower_active"] is False
    assert result["weight_materialized"] is False
    # text-only 带媒体参考 → 拒绝（守卫）
    try:
        run_mm1_visual_tower_skeleton(placement, _media_reference(), text_only=True)
    except Qwen3MultimodalRuntimeError:
        pass
    else:
        raise AssertionError("text-only 携带媒体参考应拒绝")


def test_visual_skeleton_placeholder_ready_for_media():
    """media + active：visual_path=placeholder_ready（占位执行路径）。"""
    from qwen3_multimodal_runtime import run_mm1_visual_tower_skeleton
    placement = _placement(has_media=True, active=True)
    result = run_mm1_visual_tower_skeleton(placement, _media_reference(), text_only=False)
    assert result["visual_path"] == "placeholder_ready"
    assert result["vision_tower_active"] is True
    assert result["total_media_tokens"] == 16
    assert result["weight_materialized"] is False


def test_visual_skeleton_fails_closed_when_tower_inactive():
    """media + inactive：fail-closed（视觉塔不执行）。"""
    from qwen3_multimodal_runtime import (
        Qwen3MultimodalRuntimeError,
        run_mm1_visual_tower_skeleton,
    )
    placement = _placement(has_media=True, active=False)
    try:
        run_mm1_visual_tower_skeleton(placement, _media_reference(), text_only=False)
    except Qwen3MultimodalRuntimeError as exc:
        assert exc.reason_code == "qwen3_mm1_vision_tower_inactive"
    else:
        raise AssertionError("视觉塔未激活的 media 请求应 fail-closed")


def test_visual_skeleton_rejects_placement_contradiction():
    """一致性：放置决策与 text-only 标志矛盾 → 拒绝。"""
    from qwen3_multimodal_runtime import (
        Qwen3MultimodalRuntimeError,
        run_mm1_visual_tower_skeleton,
    )
    media_placement = _placement(has_media=True, active=True)
    try:
        run_mm1_visual_tower_skeleton(media_placement, None, text_only=True)
    except Qwen3MultimodalRuntimeError:
        pass
    else:
        raise AssertionError("放置决策与 text-only 矛盾应拒绝")


# ================================================================
# MM1.13：视觉塔占位执行路径与媒体参考消费契约
# ================================================================

def _ready_skeleton():
    from qwen3_multimodal_preflight import (
        mm1_ledger_commit,
        mm1_vision_tower_placement,
    )
    ledger = mm1_ledger_commit(
        [{"entry_id": "text_0", "kind": "text_segment", "bytes": 100_000_000}],
        ledger_id="node-g", node_capacity_bytes=1_000_000_000,
    )
    placement = mm1_vision_tower_placement(
        ledger, request_has_media=True, vision_tower_bytes=400_000_000,
    )
    from qwen3_multimodal_runtime import run_mm1_visual_tower_skeleton
    return run_mm1_visual_tower_skeleton(placement, _media_reference(), text_only=False)


def test_placeholder_execution_projects_feature_summary():
    """占位执行产出视觉特征摘要（shape 绑定媒体 token 与视觉 hidden）。"""
    from qwen3_multimodal_runtime import run_mm1_visual_placeholder_execution
    skeleton = _ready_skeleton()
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    feature = run_mm1_visual_placeholder_execution(
        skeleton, _media_reference(), manifest=manifest,
    )
    assert feature["feature_kind"] == "qwen3_visual_feature_placeholder"
    assert feature["synthetic"] is True
    assert feature["weight_materialized"] is False
    # 特征 shape = [1, media_tokens, vision.output_hidden_size]
    assert feature["tensor"]["shape"][1] == 16  # 媒体参考 token 数
    assert feature["tensor"]["shape"][2] == manifest["vision"]["output_hidden_size"]
    payload = json.dumps(feature, ensure_ascii=True)
    assert "models/" not in payload


def test_placeholder_execution_requires_ready_skeleton():
    """骨架非 placeholder_ready（skipped）→ fail-closed。"""
    from qwen3_multimodal_runtime import (
        Qwen3MultimodalRuntimeError,
        run_mm1_visual_placeholder_execution,
    )
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    try:
        run_mm1_visual_placeholder_execution(
            {"visual_path": "skipped"}, _media_reference(), manifest=manifest,
        )
    except Qwen3MultimodalRuntimeError as exc:
        assert exc.reason_code == "qwen3_mm1_skeleton_not_ready"
    else:
        raise AssertionError("skipped 骨架不应执行占位")


def test_visual_feature_binds_to_text_handoff_consistently():
    """特征绑定回文本段 hidden handoff：visual_to_text + token 数一致。"""
    from qwen3_multimodal_runtime import (
        bind_mm1_visual_feature_handoff,
        run_mm1_visual_placeholder_execution,
    )
    skeleton = _ready_skeleton()
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    feature = run_mm1_visual_placeholder_execution(
        skeleton, _media_reference(), manifest=manifest,
    )
    contract = bind_mm1_visual_feature_handoff(
        feature,
        manifest=manifest,
        text_chain_id="a" * 64,
        generation=1,
        phase="prefill",
        source_node_id="node-a",
        target_node_id="node-b",
        modality="image",
    )
    assert contract["boundary"] == "visual_to_text"
    # 消费一致性：handoff token 数 == 特征 token 数
    assert contract["tensor"]["shape"][1] == feature["tensor"]["shape"][1]
    # 绑定证据：artifact.sha256 == 媒体参考（path-free 消费契约）
    assert contract["artifact"]["sha256"] == feature["media_reference_sha256"]
    assert contract["artifact"]["mode"] == "local"
    assert contract["full_model_materialized"] is False


# ================================================================
# MM1.14：视觉占位链端到端合成回归（CPU）
# ================================================================

def test_synthetic_visual_chain_end_to_end():
    """MM1.7→MM1.13 全链串联：合成媒体 → 摘要 → 参考 → 账本 → 放置 → 骨架
    → 占位执行 → visual_to_text handoff（零权重加载）。"""
    from qwen3_multimodal_runtime import run_mm1_synthetic_visual_chain
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    result = run_mm1_synthetic_visual_chain(
        manifest=manifest,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
        node_capacity_bytes=2_000_000_000,
    )
    assert result["chain_kind"] == "qwen3_synthetic_visual_chain"
    assert result["text_only"] is False
    assert result["media_tokens"] > 0
    assert result["ledger_admitted"] is True
    assert result["vision_tower_active"] is True
    assert result["visual_path"] == "placeholder_ready"
    # 消费一致性：handoff token == 特征 token == 媒体 token
    assert result["feature_shape"][1] == result["media_tokens"]
    assert result["handoff_tokens"] == result["media_tokens"]
    assert result["handoff_boundary"] == "visual_to_text"
    assert result["weight_materialized"] is False


def test_synthetic_visual_chain_fails_closed_on_capacity():
    """容量不足 → 账本 admitted=false → 视觉塔不激活 → fail-closed。"""
    from qwen3_multimodal_runtime import (
        Qwen3MultimodalRuntimeError,
        run_mm1_synthetic_visual_chain,
    )
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    try:
        run_mm1_synthetic_visual_chain(
            manifest=manifest,
            media_smoke={"image_size": [32, 32], "video_frames": 2},
            node_capacity_bytes=100_000,  # 远小于视觉塔需求
        )
    except Qwen3MultimodalRuntimeError:
        pass  # fail-closed（视觉塔不执行）
    else:
        # 或 admitted=false 路径（骨架 fail-closed）——二选一都算关闭
        raise AssertionError("容量不足应 fail-closed")


def test_synthetic_visual_chain_text_only_skips_vision():
    """text-only 路径：全链跳过视觉塔（骨架 skipped，零媒体参考）。"""
    from qwen3_multimodal_runtime import run_mm1_synthetic_visual_chain
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    result = run_mm1_synthetic_visual_chain(
        manifest=manifest,
        media_smoke=None,
        node_capacity_bytes=2_000_000_000,
        text_only=True,
    )
    assert result["text_only"] is True
    assert result["visual_path"] == "skipped"
    assert result["vision_tower_active"] is False
    assert result["weight_materialized"] is False


# ================================================================
# MM1.15：视觉塔真实权重接入与合成/真实对照
# ================================================================

@pytest.mark.real_model
def test_real_vision_tower_weights_load_and_forward():
    """真实视觉塔权重加载 + 合成图前向 + 与合成占位对照。"""
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.15 requires the isolated Qwen3 pipeline sidecar")
    import psutil
    if psutil.virtual_memory().available < 4 * 2**30:
        pytest.skip("MM1.15 资源门：可用 RAM < 4GB")
    from scripts.model_tools.qwen3_multimodal_vision_tower_probe import (
        run_qwen3_multimodal_vision_tower_probe,
    )
    report = run_qwen3_multimodal_vision_tower_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        timeout_seconds=300,
    )
    assert report["status"] == "vision_tower_weights_loaded"
    assert report["gate_passed"] is True
    response = report["response"]
    # 视觉塔真实信息
    assert response["vision_tower"]["class_name"] == "Qwen3VLVisionModel"
    assert response["vision_tower"]["depth"] == 24
    assert response["vision_tower"]["hidden_size"] == 1024
    assert response["vision_tower"]["loaded_weights"] > 0
    # 真实特征与合成占位对照：merger 投影到文本 hidden（2560）
    assert response["real_feature"]["shape"][-1] == 2560
    assert response["consistency"]["hidden_matches_text_config"] is True
    assert response["consistency"]["tokens_match"] is True
    # 边界：视觉塔权重已加载（如实登记），文本零加载
    assert response["weight_materialized"] is True
    assert response["text_weights_loaded"] is False
    assert response["full_model_materialized"] is False
    encoded = json.dumps(report, ensure_ascii=True).lower()
    assert "pixel_values\"" not in encoded
    assert str(ROOT).lower() not in encoded


# ================================================================
# MM1.16：视觉特征接入文本段合成解码（CPU 混合链）
# ================================================================

def _real_vision_feature():
    """构造 MM1.15 口径的真实视觉特征摘要（[1, tokens, text_hidden]）。"""
    return {
        "feature_kind": "qwen3_visual_feature_placeholder",
        "model_id": "qwen3-vl-4b-instruct",
        "media_reference_sha256": "b" * 64,
        "tensor": {"shape": [1, 64, 2560], "dtype": "float32", "device": "cpu"},
        "synthetic": False,
        "weight_materialized": True,
        "full_model_materialized": False,
    }


def test_synthetic_text_decode_aligns_visual_tokens():
    """视觉特征接入文本段合成解码：visual token 位置对齐 + 文本零权重。"""
    from qwen3_multimodal_runtime import run_mm1_synthetic_text_decode
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    result = run_mm1_synthetic_text_decode(
        vision_feature=_real_vision_feature(),
        manifest=manifest,
        prompt_tokens=4,
        sequence_length=128,
    )
    assert result["decode_kind"] == "qwen3_synthetic_text_decode"
    layout = result["input"]["layout"]
    # 视觉区 [0:64] + 文本区 [64:68]
    assert layout["visual_span"] == [0, 64]
    assert layout["text_span"] == [64, 68]
    assert layout["total_sequence"] == 68
    assert result["input"]["visual_tokens"] == 64
    assert result["synthetic"] is True
    assert result["text_weights_loaded"] is False
    assert result["weight_materialized"] is False


def test_synthetic_text_decode_fails_closed_on_mismatch():
    """hidden 与文本段不符 / 序列超预算 → fail-closed。"""
    from qwen3_multimodal_runtime import (
        Qwen3MultimodalRuntimeError,
        run_mm1_synthetic_text_decode,
    )
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    # hidden 不匹配（特征 128 vs 文本 2560）
    bad_feature = dict(_real_vision_feature())
    bad_feature["tensor"] = {"shape": [1, 64, 128], "dtype": "float32", "device": "cpu"}
    try:
        run_mm1_synthetic_text_decode(
            vision_feature=bad_feature, manifest=manifest, sequence_length=128,
        )
    except Qwen3MultimodalRuntimeError as exc:
        assert exc.reason_code == "qwen3_mm1_hidden_mismatch"
    else:
        raise AssertionError("hidden 不匹配应拒绝")
    # 序列超预算（visual 64 + prompt 4096 > 128）
    try:
        run_mm1_synthetic_text_decode(
            vision_feature=_real_vision_feature(),
            manifest=manifest, prompt_tokens=4096, sequence_length=128,
        )
    except Qwen3MultimodalRuntimeError as exc:
        assert exc.reason_code == "qwen3_mm1_sequence_overflow"
    else:
        raise AssertionError("序列超预算应拒绝")


# ================================================================
# MM1.17：CPU 混合链端到端合成回归（视觉塔真实 + 文本合成）
# ================================================================

def test_synthetic_hybrid_chain_end_to_end():
    """混合链：合成媒体链 + 真实视觉特征 + 文本段合成解码（token 对齐）。"""
    from qwen3_multimodal_runtime import run_mm1_synthetic_hybrid_chain
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    result = run_mm1_synthetic_hybrid_chain(
        vision_feature=_real_vision_feature(),
        manifest=manifest,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
        node_capacity_bytes=2_000_000_000,
        prompt_tokens=4,
    )
    assert result["chain_kind"] == "qwen3_synthetic_hybrid_chain"
    assert result["ledger_admitted"] is True
    assert result["vision_tower_active"] is True
    assert result["visual_path"] == "placeholder_ready"
    # token 一致性：特征 token == 文本段视觉 token
    assert result["consistency"]["tokens_match"] is True
    decode = result["decode"]
    assert decode["input"]["layout"]["visual_span"] == [0, 64]
    assert decode["input"]["layout"]["text_span"] == [64, 68]
    # 权重登记：视觉塔 true（真实特征携带）、文本 false
    assert result["vision_tower_weight_materialized"] is True
    assert result["text_weight_materialized"] is False
    assert result["full_model_materialized"] is False


def test_synthetic_hybrid_chain_fails_closed_without_capacity():
    """容量不足 → 骨架 fail-closed（视觉塔不执行）。"""
    from qwen3_multimodal_runtime import (
        Qwen3MultimodalRuntimeError,
        run_mm1_synthetic_hybrid_chain,
    )
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    try:
        run_mm1_synthetic_hybrid_chain(
            vision_feature=_real_vision_feature(),
            manifest=manifest,
            media_smoke={"image_size": [32, 32], "video_frames": 2},
            node_capacity_bytes=100_000,
        )
    except Qwen3MultimodalRuntimeError:
        pass  # fail-closed（视觉塔不执行）
    else:
        raise AssertionError("容量不足应 fail-closed")


@pytest.mark.real_model
def test_real_vision_feature_feeds_hybrid_chain():
    """真实串联：MM1.15 真实视觉特征 → MM1.17 混合链（端到端回归）。"""
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.17 requires the isolated Qwen3 pipeline sidecar")
    import psutil
    if psutil.virtual_memory().available < 4 * 2**30:
        pytest.skip("MM1.17 资源门：可用 RAM < 4GB")
    from scripts.model_tools.qwen3_multimodal_vision_tower_probe import (
        run_qwen3_multimodal_vision_tower_probe,
    )
    from qwen3_multimodal_runtime import run_mm1_synthetic_hybrid_chain

    probe = run_qwen3_multimodal_vision_tower_probe(
        model=ROOT / "models" / "qwen3-vl-4b-instruct", timeout_seconds=300,
    )
    assert probe["status"] == "vision_tower_weights_loaded"
    feature = {
        "feature_kind": "qwen3_visual_feature_placeholder",
        "model_id": "qwen3-vl-4b-instruct",
        "media_reference_sha256": probe["response"]["synthetic_reference_sha256"],
        "tensor": {
            "shape": list(probe["response"]["real_feature"]["shape"]),
            "dtype": "float32",
            "device": "cpu",
        },
        "synthetic": False,
        "weight_materialized": True,
        "full_model_materialized": False,
    }
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    result = run_mm1_synthetic_hybrid_chain(
        vision_feature=feature,
        manifest=manifest,
        media_smoke={"image_size": [32, 32], "video_frames": 2},
        node_capacity_bytes=2_000_000_000,
        prompt_tokens=4,
    )
    assert result["consistency"]["tokens_match"] is True
    assert result["vision_tower_weight_materialized"] is True
    assert result["text_weight_materialized"] is False
    assert result["full_model_materialized"] is False


# ================================================================
# MM1.18：真实视觉语义 smoke 与跨节点 hidden handoff 前置
# ================================================================

@pytest.mark.skip(reason="MM1.18 full-model semantic smoke is explicit opt-in only")
@pytest.mark.real_model
def test_real_vision_semantics_and_handoff():
    """首次文本权重加载：固定测试图真实语义 + 真实特征 handoff 绑定。"""
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        pytest.skip("MM1.18 requires the isolated Qwen3 pipeline sidecar")
    import psutil
    if psutil.virtual_memory().available < 6 * 2**30:
        pytest.skip("MM1.18 资源门：可用 RAM < 6GB")
    from scripts.model_tools.qwen3_multimodal_vision_text_smoke import (
        run_qwen3_multimodal_vision_text_smoke,
    )
    from qwen3_multimodal_runtime import bind_mm1_visual_feature_handoff

    image = ROOT / "build" / "exp-calibration" / "round-1" / "images" / "sd-001_seed19950101.png"
    if not image.is_file():
        pytest.skip("MM1.18 requires the fixed test image (sd-001)")
    report = run_qwen3_multimodal_vision_text_smoke(
        model=ROOT / "models" / "qwen3-vl-4b-instruct",
        image=image,
        timeout_seconds=900,
    )
    assert report["status"] == "vision_semantics_loaded"
    response = report["response"]
    assert response["text_weights_loaded"] is True
    assert response["weight_materialized"] is True
    assert response["full_model_materialized"] is False
    # 固定测试图语义基线：red apple / wooden surface
    assert response["keyword_hits"]["apple"] is True
    assert response["keyword_hits"]["red"] is True
    assert response["keyword_hits"]["wood"] is True
    assert len(response["description"]) >= 20

    # 真实特征 → visual_to_text hidden handoff（MM1.13 契约）
    manifest, _inspection, _request = _prepared("qwen3-vl-4b-instruct")
    feature = {
        "feature_kind": "qwen3_visual_feature_placeholder",
        "model_id": "qwen3-vl-4b-instruct",
        "media_reference_sha256": "c" * 64,
        "tensor": {
            "shape": [1, 64, manifest["text"]["hidden_size"]],
            "dtype": "float32",
            "device": "cpu",
        },
        "synthetic": False,
        "weight_materialized": True,
        "full_model_materialized": False,
    }
    handoff = bind_mm1_visual_feature_handoff(
        feature,
        manifest=manifest,
        text_chain_id="a" * 64,
        generation=4,
        phase="prefill",
        source_node_id="node-a",
        target_node_id="node-b",
        modality="image",
    )
    assert handoff["boundary"] == "visual_to_text"
    assert handoff["tensor"]["shape"][-1] == manifest["text"]["hidden_size"]
    assert handoff["contract_sha256"]
    assert handoff["artifact"]["status"] == "committed"
    assert handoff["artifact"]["sha256"] == feature["media_reference_sha256"]
    assert handoff["artifact"]["size_bytes"] > 0


def test_real_vision_semantics_requires_explicit_full_model_opt_in():
    """MM1.18 must not silently materialize a full model on an 8GB node."""
    from scripts.model_tools.qwen3_multimodal_vision_text_smoke_worker import execute_request

    report = execute_request({
        "schema_version": 1,
        "tool": "qwen3_multimodal_vision_text_smoke",
        "operation": "qwen3_vision_text_real_semantics",
        "read_only": True,
        "network_access": "disabled",
        "model_path": "not-used",
        "image_path": "not-used",
        "text_chain_id": "a" * 64,
        "generation": 4,
        "allow_full_model_materialization": False,
    })
    assert report["status"] == "resource_rejected"
    assert report["errors"][0]["code"] == "full_model_materialization_disabled"
    assert "response" not in report
