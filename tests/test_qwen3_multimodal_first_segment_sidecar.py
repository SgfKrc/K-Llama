"""MM1.21 first text-segment local artifact sidecar regressions."""

from __future__ import annotations

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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen3_multimodal_contract import (  # noqa: E402
    build_mm1_model_manifest,
    build_mm1_model_profile,
)
from qwen3_multimodal_runtime import (  # noqa: E402
    Qwen3MultimodalFirstSegmentSidecarAdapter,
    Qwen3MultimodalRuntimeError,
    build_mm1_first_segment_artifact_binding,
    build_mm1_staged_text_contract,
    validate_mm1_first_segment_artifact_binding,
)
from qwen3_pipeline_sidecar import Qwen3PipelineSidecarSession  # noqa: E402


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
        (ROOT / "models" / "qwen3-vl-4b-instruct" / "config.json").read_text(
            encoding="utf-8",
        ),
    )
    profile = build_mm1_model_profile(config)
    return build_mm1_model_manifest(
        model_id="fixture-qwen3-vl-mm121",
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


def _staged() -> tuple[dict, dict]:
    manifest = _manifest()
    hidden = manifest["text"]["hidden_size"]
    feature = {
        "feature_kind": "qwen3_visual_feature_placeholder",
        "model_id": manifest["model_id"],
        "media_reference_sha256": "d" * 64,
        "tensor": {"shape": [1, 64, hidden], "dtype": "float32", "device": "cpu"},
        "synthetic": False,
        "weight_materialized": True,
        "full_model_materialized": False,
    }
    segments = [
        {
            "node_id": "text-node-a",
            "layer_range": [0, 18],
            "has_embedding": True,
            "has_lm_head": False,
            "device": "cpu",
            "dtype": "float32",
            "required_bytes": 1_000_000,
            "activation_bytes": 700_000,
            "node_capacity_bytes": 2_000_000,
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
    return manifest, build_mm1_staged_text_contract(
        vision_feature=feature,
        manifest=manifest,
        segments=segments,
        text_chain_id="e" * 64,
        generation=21,
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _binding(manifest: dict, staged: dict, data: bytes) -> dict:
    return build_mm1_first_segment_artifact_binding(
        staged,
        manifest=manifest,
        artifact_id="mm121-combined-hidden",
        size_bytes=len(data),
        sha256=_digest(data),
    )


def _session(tmp_path: Path, staged: dict, runner) -> Qwen3PipelineSidecarSession:
    model_root = tmp_path / "model"
    model_root.mkdir(exist_ok=True)
    first = staged["segment_plan"][0]
    return Qwen3PipelineSidecarSession(
        model_path=model_root,
        model_id=staged["model_id"],
        model_sha256=staged["manifest_sha256"],
        config_id="mm121-config",
        plan_id="mm121-plan",
        node_id=first["node_id"],
        layer_range=first["layer_range"],
        total_layers=staged["total_layers"],
        has_embedding=first["has_embedding"],
        has_lm_head=first["has_lm_head"],
        execution_device=first["device"],
        dtype=first["dtype"],
        generation=staged["generation"],
        assignment_manifest_sha256=first["assignment_manifest_sha256"],
        worker_runner=runner,
    )


def _runner(
    phases: list[str],
    *,
    fail_prefill: bool = False,
    bad_output_sha: bool = False,
    incomplete_cleanup_phase: str | None = None,
):
    def run(request: dict, _timeout: float) -> dict:
        phase = request["phase"]
        phases.append(phase)
        if phase in {"prepare", "commit", "release", "abort"}:
            return {
                "schema_version": 1,
                "operation": "qwen3_pipeline_sidecar",
                "phase": phase,
                "status": {
                    "prepare": "prepared",
                    "commit": "committed",
                    "release": "released",
                    "abort": "aborted",
                }[phase],
                "gate_passed": True,
                "cleanup_complete": (
                    phase in {"release", "abort"} and phase != incomplete_cleanup_phase
                ),
                "execution": {
                    "segment_materialized": phase == "commit",
                    "full_model_materialized": False,
                },
            }
        output = Path(request["output_ref"])
        output.write_bytes(b"mm121-first-segment-output")
        if fail_prefill:
            return {
                "schema_version": 1,
                "operation": "qwen3_pipeline_sidecar",
                "phase": phase,
                "status": "execution_failed",
                "gate_passed": False,
                "errors": [{"code": "fixture_prefill_failed", "message": "prefill failed"}],
            }
        output_data = output.read_bytes()
        output_sha256 = "f" * 64 if bad_output_sha else _digest(output_data)
        return {
            "schema_version": 1,
            "operation": "qwen3_pipeline_sidecar",
            "phase": phase,
            "status": "executed",
            "gate_passed": True,
            "execution": {
                "data_plane": "local_artifact",
                "segment_materialized": True,
                "full_model_materialized": False,
                "artifact_bytes": len(output_data),
                "artifact_sha256": output_sha256,
            },
            "hidden_handoff": {
                "schema_version": 1,
                "chain_id": request["chain_id"],
                "from_segment": 0,
                "to_segment": 1,
                "shape": [request["batch_size"], request["sequence_length"], 2560],
                "batch_size": request["batch_size"],
                "sequence_length": request["sequence_length"],
                "hidden_size": 2560,
                "dtype": "torch.float32",
                "device": request["device"],
            },
        }

    return run


def test_mm121_binding_and_sidecar_lifecycle_are_path_free(tmp_path):
    manifest, staged = _staged()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    data = b"mm121-combined-hidden-input"
    input_ref = artifact_root / "input.pt"
    input_ref.write_bytes(data)
    binding = _binding(manifest, staged, data)
    phases: list[str] = []
    adapter = Qwen3MultimodalFirstSegmentSidecarAdapter(
        _session(tmp_path, staged, _runner(phases)), artifact_root=artifact_root,
    )

    result = adapter.execute(
        staged, manifest=manifest, artifact_binding=binding, input_ref=input_ref,
    )

    assert phases == ["prepare", "commit", "prefill", "release"]
    assert result["lifecycle"] == phases
    assert result["input_artifact"]["sha256"] == _digest(data)
    assert result["hidden_handoff"]["shape"] == [1, 68, 2560]
    output_path = adapter.output_path(result["output_artifact"]["artifact_id"])
    assert output_path.is_file()
    assert "path" not in json.dumps(binding, ensure_ascii=True).lower()
    assert "path" not in json.dumps(result, ensure_ascii=True).lower()
    cleanup = adapter.cleanup("downstream_committed")
    assert cleanup["removed_artifacts"] == 1
    assert not output_path.exists()


def test_mm121_rejects_tampered_input_before_prepare(tmp_path):
    manifest, staged = _staged()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    input_ref = artifact_root / "input.pt"
    original = b"committed-input"
    input_ref.write_bytes(original)
    binding = _binding(manifest, staged, original)
    input_ref.write_bytes(b"tampered-input")
    phases: list[str] = []
    adapter = Qwen3MultimodalFirstSegmentSidecarAdapter(
        _session(tmp_path, staged, _runner(phases)), artifact_root=artifact_root,
    )

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute(
            staged, manifest=manifest, artifact_binding=binding, input_ref=input_ref,
        )

    assert caught.value.reason_code == "qwen3_mm1_sidecar_artifact_mismatch"
    assert phases == []


def test_mm121_scope_fence_rejects_external_input_before_prepare(tmp_path):
    manifest, staged = _staged()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    external = tmp_path / "external.pt"
    data = b"external-input"
    external.write_bytes(data)
    phases: list[str] = []
    adapter = Qwen3MultimodalFirstSegmentSidecarAdapter(
        _session(tmp_path, staged, _runner(phases)), artifact_root=artifact_root,
    )

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute(
            staged,
            manifest=manifest,
            artifact_binding=_binding(manifest, staged, data),
            input_ref=external,
        )

    assert caught.value.reason_code == "qwen3_mm1_sidecar_artifact_scope"
    assert phases == []


def test_mm121_cancel_after_commit_aborts_without_output(tmp_path):
    manifest, staged = _staged()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    data = b"cancel-input"
    input_ref = artifact_root / "input.pt"
    input_ref.write_bytes(data)
    phases: list[str] = []
    adapter = Qwen3MultimodalFirstSegmentSidecarAdapter(
        _session(tmp_path, staged, _runner(phases)), artifact_root=artifact_root,
    )

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute(
            staged,
            manifest=manifest,
            artifact_binding=_binding(manifest, staged, data),
            input_ref=input_ref,
            cancel_after_commit=True,
        )

    assert caught.value.reason_code == "qwen3_mm1_sidecar_cancelled"
    assert phases == ["prepare", "commit", "abort"]
    assert list(artifact_root.glob("mm1-first-*.pt")) == []


@pytest.mark.parametrize("bad_output_sha", [False, True])
def test_mm121_prefill_failure_or_output_mismatch_aborts_and_cleans(
    tmp_path, bad_output_sha,
):
    manifest, staged = _staged()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    data = b"failure-input"
    input_ref = artifact_root / "input.pt"
    input_ref.write_bytes(data)
    phases: list[str] = []
    adapter = Qwen3MultimodalFirstSegmentSidecarAdapter(
        _session(
            tmp_path,
            staged,
            _runner(phases, fail_prefill=not bad_output_sha, bad_output_sha=bad_output_sha),
        ),
        artifact_root=artifact_root,
    )

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute(
            staged,
            manifest=manifest,
            artifact_binding=_binding(manifest, staged, data),
            input_ref=input_ref,
        )

    assert caught.value.reason_code == "qwen3_mm1_sidecar_execution_failed"
    assert phases == ["prepare", "commit", "prefill", "abort"]
    assert list(artifact_root.glob("mm1-first-*.pt")) == []


def test_mm121_binding_tamper_is_rejected_before_sidecar_use():
    manifest, staged = _staged()
    binding = _binding(manifest, staged, b"binding-input")
    binding["tensor"]["shape"][1] += 1

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        validate_mm1_first_segment_artifact_binding(
            binding, staged_contract=staged, manifest=manifest,
        )

    assert caught.value.reason_code == "qwen3_mm1_sidecar_binding_invalid"


@pytest.mark.parametrize(
    ("cleanup_phase", "cancel_after_commit", "expected_phases"),
    [
        ("release", False, ["prepare", "commit", "prefill", "release"]),
        ("abort", True, ["prepare", "commit", "abort"]),
    ],
)
def test_mm121_incomplete_sidecar_cleanup_fails_closed(
    tmp_path, cleanup_phase, cancel_after_commit, expected_phases,
):
    manifest, staged = _staged()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    data = b"incomplete-cleanup-input"
    input_ref = artifact_root / "input.pt"
    input_ref.write_bytes(data)
    phases: list[str] = []
    adapter = Qwen3MultimodalFirstSegmentSidecarAdapter(
        _session(
            tmp_path,
            staged,
            _runner(phases, incomplete_cleanup_phase=cleanup_phase),
        ),
        artifact_root=artifact_root,
    )

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute(
            staged,
            manifest=manifest,
            artifact_binding=_binding(manifest, staged, data),
            input_ref=input_ref,
            cancel_after_commit=cancel_after_commit,
        )

    assert caught.value.reason_code == "qwen3_mm1_sidecar_cleanup_failed"
    assert phases == expected_phases
    assert list(artifact_root.glob("mm1-first-*.pt")) == []
