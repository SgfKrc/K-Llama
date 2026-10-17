"""MM1.22 local two/three-segment sidecar-chain regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

# CI 不拉模型工件（本机 models 是 Junction）。缺工件按项目约定跳过，
# 不用失败冒充"测过"；有工件的开发机照常执行。
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
from qwen3_multimodal_multisidecar import (  # noqa: E402
    Qwen3MultimodalDecodeArtifactConsumer,
    Qwen3MultimodalMultiSidecarAdapter,
    build_mm1_sampling_policy_snapshot,
    build_mm1_decode_artifact_binding,
)
from qwen3_multimodal_runtime import (  # noqa: E402
    Qwen3MultimodalRuntimeError,
    build_mm1_first_segment_artifact_binding,
    build_mm1_staged_text_contract,
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
        model_id="fixture-qwen3-vl-mm122",
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


def _staged(segment_count: int) -> tuple[dict, dict]:
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
    boundaries = {2: [0, 18, 36], 3: [0, 12, 24, 36]}[segment_count]
    segments = []
    for index in range(segment_count):
        segments.append({
            "node_id": f"text-node-{chr(97 + index)}",
            "layer_range": [boundaries[index], boundaries[index + 1]],
            "has_embedding": index == 0,
            "has_lm_head": index == segment_count - 1,
            "device": "cpu",
            "dtype": "float32",
            "required_bytes": 1_000_000,
            "activation_bytes": 700_000,
            "node_capacity_bytes": 2_000_000,
            "assignment_manifest_sha256": f"{index + 1:x}" * 64,
        })
    staged = build_mm1_staged_text_contract(
        vision_feature=feature,
        manifest=manifest,
        segments=segments,
        text_chain_id="e" * 64,
        generation=22,
    )
    return manifest, staged


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runner(
    index: int,
    calls: dict[int, list[str]],
    *,
    hidden_size: int,
    fail_prefill: bool = False,
    corrupt_handoff: bool = False,
    corrupt_decode_kv: bool = False,
    bounded_tokens: tuple[int, ...] | None = None,
    fail_decode_step: int | None = None,
):
    def run(request: dict, _timeout: float) -> dict:
        phase = request["phase"]
        calls[index].append(phase)
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
                "cleanup_complete": phase in {"release", "abort"},
                "execution": {
                    "segment_materialized": phase == "commit",
                    "full_model_materialized": False,
                },
            }
        output = Path(request["output_ref"])
        decode_step = calls[index].count("decode")
        if fail_prefill or (
            phase == "decode" and fail_decode_step == decode_step
        ):
            return {
                "schema_version": 1,
                "operation": "qwen3_pipeline_sidecar",
                "phase": phase,
                "status": "execution_failed",
                "gate_passed": False,
                "errors": [{"code": "fixture_segment_failed", "message": "segment failed"}],
            }
        handoff_tokens = 1 if phase == "decode" else request["sequence_length"]
        if bounded_tokens is None:
            output.write_bytes(f"mm122-segment-{index}".encode("ascii"))
        else:
            cache_tokens = request["sequence_length"]
            local_layers = request["layer_range"][1] - request["layer_range"][0]
            cache = tuple((
                torch.zeros((request["batch_size"], 2, cache_tokens, 4)),
                torch.zeros((request["batch_size"], 2, cache_tokens, 4)),
            ) for _layer in range(local_layers))
            if request["has_next_segment"]:
                payload = {
                    "hidden_states": torch.zeros((
                        request["batch_size"], handoff_tokens, hidden_size,
                    )),
                    "logits": None,
                    "past_key_values": cache,
                }
            else:
                vocab_size = max(8, max(bounded_tokens, default=0) + 2)
                logits = torch.zeros((request["batch_size"], handoff_tokens, vocab_size))
                if phase == "decode":
                    selected = bounded_tokens[min(decode_step - 1, len(bounded_tokens) - 1)]
                    logits[:, -1, selected] = 10.0
                payload = {
                    "hidden_states": None,
                    "logits": logits,
                    "past_key_values": cache,
                }
            torch.save(payload, output)
        data = output.read_bytes()
        hidden = None
        if request["has_next_segment"]:
            shape_hidden = hidden_size + 1 if corrupt_handoff else hidden_size
            hidden = {
                "schema_version": 1,
                "chain_id": request["chain_id"],
                "from_segment": index,
                "to_segment": index + 1,
                "shape": [request["batch_size"], handoff_tokens, shape_hidden],
                "batch_size": request["batch_size"],
                "sequence_length": handoff_tokens,
                "hidden_size": shape_hidden,
                "dtype": "torch.float32",
                "device": "cpu",
            }
        kv_segment_index = 99 if phase == "decode" and corrupt_decode_kv else index
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
                "artifact_bytes": len(data),
                "artifact_sha256": _sha(data),
            },
            "hidden_handoff": hidden,
            "kv_contract": {
                "schema_version": 1,
                "chain_id": request["chain_id"],
                "segment_index": kv_segment_index,
                "layer_range": list(request["layer_range"]),
                "sequence_length": request["sequence_length"],
                "batch_size": request["batch_size"],
                "dtype": "torch.float32",
                "device": "cpu",
                "phase": phase,
                "generation": request["generation"],
            },
        }

    return run


def _fixture(tmp_path: Path, segment_count: int = 2, *, fail_index: int | None = None,
             corrupt_index: int | None = None, corrupt_decode_kv_index: int | None = None,
             bounded_tokens: tuple[int, ...] | None = None,
             fail_decode_index: int | None = None,
             fail_decode_step: int | None = None):
    manifest, staged = _staged(segment_count)
    root = tmp_path / "artifacts"
    root.mkdir()
    input_ref = root / "input.pt"
    input_data = b"mm122-combined-hidden-input"
    input_ref.write_bytes(input_data)
    binding = build_mm1_first_segment_artifact_binding(
        staged,
        manifest=manifest,
        artifact_id="mm122-input",
        size_bytes=len(input_data),
        sha256=_sha(input_data),
    )
    calls = {index: [] for index in range(segment_count)}
    model_root = tmp_path / "model"
    model_root.mkdir()
    sessions = []
    for index, segment in enumerate(staged["segment_plan"]):
        sessions.append(Qwen3PipelineSidecarSession(
            model_path=model_root,
            model_id=staged["model_id"],
            model_sha256=staged["manifest_sha256"],
            config_id="mm122-config",
            plan_id="mm122-plan",
            node_id=segment["node_id"],
            layer_range=segment["layer_range"],
            total_layers=staged["total_layers"],
            has_embedding=segment["has_embedding"],
            has_lm_head=segment["has_lm_head"],
            execution_device=segment["device"],
            dtype=segment["dtype"],
            generation=staged["generation"],
            assignment_manifest_sha256=segment["assignment_manifest_sha256"],
            worker_runner=_runner(
                index,
                calls,
                hidden_size=staged["input_layout"]["hidden_size"],
                fail_prefill=index == fail_index,
                corrupt_handoff=index == corrupt_index,
                corrupt_decode_kv=index == corrupt_decode_kv_index,
                bounded_tokens=bounded_tokens,
                fail_decode_step=(
                    fail_decode_step if index == fail_decode_index else None
                ),
            ),
        ))
    adapter = Qwen3MultimodalMultiSidecarAdapter(
        staged_contract=staged,
        manifest=manifest,
        artifact_binding=binding,
        sessions=sessions,
        artifact_root=root,
    )
    return adapter, manifest, staged, binding, input_ref, calls, root


def _decode_fixture(adapter, root: Path, *, torch_input: bool = False):
    path = root / "decode-input.pt"
    if torch_input:
        torch.save({"input_ids": torch.tensor([[1]], dtype=torch.long)}, path)
        data = path.read_bytes()
    else:
        data = b"mm122-decode-input-ids"
        path.write_bytes(data)
    binding = build_mm1_decode_artifact_binding(
        adapter.staged,
        manifest=adapter.manifest,
        artifact_id="mm122-decode-input",
        size_bytes=len(data),
        sha256=_sha(data),
        batch_size=1,
        token_count=1,
    )
    return path, binding


@pytest.mark.parametrize("segment_count", [2, 3])
def test_mm122_runs_two_or_three_segments_and_retains_only_final_output(
    tmp_path, segment_count,
):
    adapter, manifest, staged, binding, input_ref, calls, root = _fixture(
        tmp_path, segment_count,
    )

    result = adapter.execute_prefill(input_ref=input_ref)

    assert result["status"] == "multimodal_text_chain_prefilled"
    assert result["segment_count"] == segment_count
    assert result["lifecycle"] == ["prepare", "commit", "prefill", "release"]
    assert len(result["segment_reports"]) == segment_count
    assert len([
        report["artifact_reference"]
        for report in result["segment_reports"][:-1]
        if report["artifact_reference"] is not None
    ]) == segment_count - 1
    assert result["final_kv_contract"]["segment_index"] == segment_count - 1
    assert result["full_model_materialized"] is False
    assert all(value == ["prepare", "commit", "prefill", "release"] for value in calls.values())
    final_path = adapter.output_path(result["final_artifact"]["artifact_id"])
    assert final_path.is_file()
    assert list(root.glob("qwen3-*.pt")) == []
    encoded = json.dumps(result, ensure_ascii=True).lower()
    assert "path" not in encoded
    cleanup = adapter.cleanup("downstream_committed")
    assert cleanup["removed_artifacts"] == 1
    assert not final_path.exists()


def test_mm122_input_scope_and_sha_are_checked_before_chain_prepare(tmp_path):
    adapter, manifest, staged, binding, input_ref, calls, _root = _fixture(tmp_path)
    input_ref.write_bytes(b"changed-input")

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_prefill(input_ref=input_ref)

    assert caught.value.reason_code == "qwen3_mm1_multisidecar_artifact_mismatch"
    assert all(value == [] for value in calls.values())


def test_mm122_middle_segment_failure_aborts_every_sidecar_and_cleans(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, 3, fail_index=1,
    )

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_prefill(input_ref=input_ref)

    assert caught.value.reason_code == "qwen3_mm1_multisidecar_execution_failed"
    assert calls[0] == ["prepare", "commit", "prefill", "abort"]
    assert calls[1] == ["prepare", "commit", "prefill", "abort"]
    assert calls[2] == ["prepare", "commit", "abort"]
    assert adapter.chain.phase == "aborted"
    assert list(root.glob("qwen3-*.pt")) == []
    assert list(root.glob("mm1-*.pt")) == []


def test_mm122_handoff_shape_mismatch_aborts_before_next_segment(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, 2, corrupt_index=0,
    )

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_prefill(input_ref=input_ref)

    assert caught.value.reason_code == "qwen3_mm1_multisidecar_handoff_mismatch"
    assert calls[0] == ["prepare", "commit", "prefill", "abort"]
    assert calls[1] == ["prepare", "commit", "abort"]
    assert list(root.glob("qwen3-*.pt")) == []


def test_mm122_cancel_after_commit_aborts_all_sessions(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, 3,
    )

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_prefill(input_ref=input_ref, cancel_after_commit=True)

    assert caught.value.reason_code == "qwen3_mm1_multisidecar_cancelled"
    assert all(value == ["prepare", "commit", "abort"] for value in calls.values())
    assert list(root.glob("qwen3-*.pt")) == []


def test_mm122_restart_recovery_removes_generic_and_mm1_retained_artifacts(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, _input_ref, calls, root = _fixture(
        tmp_path, 2,
    )
    stale_chain = root / f"qwen3-{adapter.chain._chain_token}-0-prefill-stale.pt"
    stale_mm1 = root / f"{adapter._retained_prefix}stale.pt"
    stale_chain.write_bytes(b"stale-chain")
    stale_mm1.write_bytes(b"stale-mm1")

    result = adapter.recover_after_restart()

    assert result["completed"] is True
    assert result["removed_retained_artifacts"] == 1
    assert all(value == [] for value in calls.values())
    assert list(root.glob("qwen3-*.pt")) == []
    assert list(root.glob("mm1-*.pt")) == []


@pytest.mark.parametrize("segment_count", [2, 3])
def test_mm123_runs_prefill_decode_and_retains_final_decode_artifact(tmp_path, segment_count):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, segment_count,
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root)

    result = adapter.execute_prefill_decode(
        input_ref=input_ref,
        decode_input_ref=decode_ref,
        decode_binding=decode_binding,
    )

    assert result["status"] == "multimodal_text_chain_decoded"
    assert result["segment_count"] == segment_count
    assert result["lifecycle"] == ["prepare", "commit", "prefill", "decode", "release"]
    assert len(result["prefill_segment_reports"]) == segment_count
    assert len(result["decode_segment_reports"]) == segment_count
    assert result["final_kv_contract"]["phase"] == "decode"
    assert result["final_kv_contract"]["generation"] == 23
    assert result["decode_sequence"]["decode_length"] == 69
    assert all(
        value == ["prepare", "commit", "prefill", "decode", "release"]
        for value in calls.values()
    )
    final_path = adapter.output_path(result["final_artifact"]["artifact_id"])
    assert final_path.is_file()
    assert list(root.glob("qwen3-*.pt")) == []
    assert "path" not in json.dumps(result, ensure_ascii=True).lower()
    assert adapter.cleanup("decode_committed")["removed_artifacts"] == 1
    assert not final_path.exists()


def test_mm123_decode_input_evidence_is_checked_before_prepare(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(tmp_path)
    decode_ref, decode_binding = _decode_fixture(adapter, root)
    decode_ref.write_bytes(b"tampered-decode-input")

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_prefill_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
        )

    assert caught.value.reason_code == "qwen3_mm1_multisidecar_artifact_mismatch"
    assert all(value == [] for value in calls.values())


def test_mm123_cancel_after_prefill_aborts_without_decode(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, 3,
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_prefill_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
            cancel_after_prefill=True,
        )

    assert caught.value.reason_code == "qwen3_mm1_multisidecar_cancelled"
    assert all(value == ["prepare", "commit", "prefill", "abort"] for value in calls.values())
    assert list(root.glob("qwen3-*.pt")) == []


def test_mm123_decode_binding_generation_tamper_is_rejected_before_prepare(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(tmp_path)
    decode_ref, decode_binding = _decode_fixture(adapter, root)
    decode_binding["decode_generation"] = 99

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_prefill_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
        )

    assert caught.value.reason_code == "qwen3_mm1_decode_binding_invalid"
    assert all(value == [] for value in calls.values())


def test_mm123_decode_kv_tamper_aborts_the_full_chain(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, 3, corrupt_decode_kv_index=0,
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_prefill_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
        )

    assert caught.value.reason_code == "qwen3_mm1_multisidecar_execution_failed"
    assert calls[0] == ["prepare", "commit", "prefill", "decode", "abort"]
    assert calls[1] == ["prepare", "commit", "prefill", "abort"]
    assert calls[2] == ["prepare", "commit", "prefill", "abort"]
    assert list(root.glob("qwen3-*.pt")) == []


def _consumer_fixture(
    tmp_path: Path,
    *,
    output_tokens: int = 1,
    cache_tokens: int = 69,
):
    root = tmp_path / "consumer-artifacts"
    root.mkdir()
    artifact = root / "final-decode.pt"
    payload = {
        "hidden_states": None,
        "logits": torch.zeros((1, output_tokens, 17), dtype=torch.float32),
        "past_key_values": ((
            torch.zeros((1, 2, cache_tokens, 4), dtype=torch.float32),
            torch.zeros((1, 2, cache_tokens, 4), dtype=torch.float32),
        ),),
    }
    torch.save(payload, artifact)
    data = artifact.read_bytes()
    metadata = {
        "artifact_id": "mm124-final-decode",
        "size_bytes": len(data),
        "sha256": _sha(data),
        "status": "committed",
        "content_kind": "final_decode_output",
    }
    kv_contract = {
        "phase": "decode",
        "generation": 23,
        "sequence_length": 69,
        "batch_size": 1,
        "layer_range": [0, 1],
        "dtype": "float32",
        "device": "cpu",
    }
    return root, artifact, metadata, kv_contract


def test_mm124_consumes_decode_artifact_into_path_free_quality_summary(tmp_path):
    root, artifact, metadata, kv_contract = _consumer_fixture(tmp_path)
    consumer = Qwen3MultimodalDecodeArtifactConsumer(artifact_root=root)

    result = consumer.consume(
        artifact_ref=artifact,
        artifact_metadata=metadata,
        kv_contract=kv_contract,
        expected_generation=23,
        expected_decode_tokens=1,
    )

    assert result["status"] == "decode_artifact_consumed"
    assert result["output_kind"] == "logits"
    assert result["logits"]["shape"] == [1, 1, 17]
    assert result["logits"]["dtype"] == "float32"
    assert result["logits"]["device"] == "cpu"
    assert result["logits"]["finite"] is True
    assert result["kv"]["layer_count"] == 1
    assert result["kv"]["sequence_length"] == 69
    assert result["full_model_materialized"] is False
    assert "path" not in json.dumps(result, ensure_ascii=True).lower()


def test_mm124_rejects_repeat_consumption_until_local_replay_state_is_reset(tmp_path):
    root, artifact, metadata, kv_contract = _consumer_fixture(tmp_path)
    consumer = Qwen3MultimodalDecodeArtifactConsumer(artifact_root=root)
    request = {
        "artifact_ref": artifact,
        "artifact_metadata": metadata,
        "kv_contract": kv_contract,
        "expected_generation": 23,
        "expected_decode_tokens": 1,
    }
    consumer.consume(**request)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        consumer.consume(**request)

    assert caught.value.reason_code == "qwen3_mm1_decode_consume_repeated"
    consumer.reset()
    assert consumer.consume(**request)["status"] == "decode_artifact_consumed"


def test_mm124_checks_decode_artifact_evidence_before_loading(tmp_path):
    root, artifact, metadata, kv_contract = _consumer_fixture(tmp_path)
    artifact.write_bytes(b"tampered")
    consumer = Qwen3MultimodalDecodeArtifactConsumer(artifact_root=root)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        consumer.consume(
            artifact_ref=artifact,
            artifact_metadata=metadata,
            kv_contract=kv_contract,
            expected_generation=23,
            expected_decode_tokens=1,
        )

    assert caught.value.reason_code == "qwen3_mm1_decode_consume_artifact_mismatch"


def test_mm124_rejects_missing_decode_artifact(tmp_path):
    root, artifact, metadata, kv_contract = _consumer_fixture(tmp_path)
    artifact.unlink()
    consumer = Qwen3MultimodalDecodeArtifactConsumer(artifact_root=root)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        consumer.consume(
            artifact_ref=artifact,
            artifact_metadata=metadata,
            kv_contract=kv_contract,
            expected_generation=23,
            expected_decode_tokens=1,
        )

    assert caught.value.reason_code == "qwen3_mm1_decode_consume_artifact_missing"


def test_mm124_rejects_artifact_outside_the_local_root(tmp_path):
    root, _artifact, metadata, kv_contract = _consumer_fixture(tmp_path)
    external = tmp_path / "external.pt"
    external.write_bytes(b"external")
    consumer = Qwen3MultimodalDecodeArtifactConsumer(artifact_root=root)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        consumer.consume(
            artifact_ref=external,
            artifact_metadata=metadata,
            kv_contract=kv_contract,
            expected_generation=23,
            expected_decode_tokens=1,
        )

    assert caught.value.reason_code == "qwen3_mm1_decode_consume_scope_invalid"


def test_mm124_rejects_output_shape_that_exceeds_decode_token_contract(tmp_path):
    root, artifact, metadata, kv_contract = _consumer_fixture(
        tmp_path, output_tokens=2,
    )
    consumer = Qwen3MultimodalDecodeArtifactConsumer(artifact_root=root)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        consumer.consume(
            artifact_ref=artifact,
            artifact_metadata=metadata,
            kv_contract=kv_contract,
            expected_generation=23,
            expected_decode_tokens=1,
        )

    assert caught.value.reason_code == "qwen3_mm1_decode_consume_tensor_invalid"


def test_mm124_rejects_output_dtype_that_differs_from_decode_contract(tmp_path):
    root, artifact, metadata, kv_contract = _consumer_fixture(tmp_path)
    kv_contract["dtype"] = "float16"
    consumer = Qwen3MultimodalDecodeArtifactConsumer(artifact_root=root)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        consumer.consume(
            artifact_ref=artifact,
            artifact_metadata=metadata,
            kv_contract=kv_contract,
            expected_generation=23,
            expected_decode_tokens=1,
        )

    assert caught.value.reason_code == "qwen3_mm1_decode_consume_tensor_invalid"


def test_mm124_rejects_kv_length_that_differs_from_decode_contract(tmp_path):
    root, artifact, metadata, kv_contract = _consumer_fixture(
        tmp_path, cache_tokens=68,
    )
    consumer = Qwen3MultimodalDecodeArtifactConsumer(artifact_root=root)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        consumer.consume(
            artifact_ref=artifact,
            artifact_metadata=metadata,
            kv_contract=kv_contract,
            expected_generation=23,
            expected_decode_tokens=1,
        )

    assert caught.value.reason_code == "qwen3_mm1_decode_consume_kv_invalid"


def test_mm124_consumer_does_not_take_artifact_cleanup_ownership(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, _calls, root = _fixture(
        tmp_path,
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root)
    execution = adapter.execute_prefill_decode(
        input_ref=input_ref,
        decode_input_ref=decode_ref,
        decode_binding=decode_binding,
    )
    retained = adapter.output_path(execution["final_artifact"]["artifact_id"])

    assert retained.is_file()
    assert adapter.cleanup("consumer_finished")["removed_artifacts"] == 1
    assert not retained.exists()


@pytest.mark.parametrize("segment_count", [2, 3])
def test_mm125_runs_bounded_multistep_decode_with_atomic_kv_rotation(
    tmp_path, segment_count,
):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, segment_count, bounded_tokens=(5, 6, 7),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)

    result = adapter.execute_bounded_decode(
        input_ref=input_ref,
        decode_input_ref=decode_ref,
        decode_binding=decode_binding,
        max_new_tokens=3,
    )

    assert result["status"] == "multimodal_text_chain_generated"
    assert result["generated_token_count"] == 3
    assert result["stop_reason"] == "max_new_tokens"
    assert result["final_generation"] == 25
    assert result["final_kv_contract"]["sequence_length"] == 71
    assert result["token_ledger"]["token_count"] == 3
    assert result["token_ledger"]["content_kind"] == "generated_token_ledger"
    assert result["lifecycle"] == [
        "prepare", "commit", "prefill",
        "decode_step_1", "decode_step_2", "decode_step_3", "release",
    ]
    assert result["decode_trace"] == {
        "step_count": 3,
        "first_generation": 23,
        "final_generation": 25,
        "first_sequence_length": 69,
        "final_sequence_length": 71,
        "sha256": result["decode_trace"]["sha256"],
    }
    assert len(result["decode_trace"]["sha256"]) == 64
    assert result["final_decode_quality"]["generation"] == 25
    assert result["final_decode_quality"]["sequence_length"] == 71
    assert all(
        value == ["prepare", "commit", "prefill", "decode", "decode", "decode", "release"]
        for value in calls.values()
    )
    assert "path" not in json.dumps(result, ensure_ascii=True).lower()
    assert "input_ids" not in json.dumps(result, ensure_ascii=True).lower()
    assert list(root.glob(f"{adapter._retained_prefix}next-*.pt")) == []
    final_path = adapter.output_path(result["final_artifact"]["artifact_id"])
    assert final_path.is_file()
    assert adapter.cleanup("generation_committed")["removed_artifacts"] == 2
    assert not final_path.exists()


def test_mm125_stops_on_eos_without_exposing_selected_token(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, 3, bounded_tokens=(5, 2, 7),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)

    result = adapter.execute_bounded_decode(
        input_ref=input_ref,
        decode_input_ref=decode_ref,
        decode_binding=decode_binding,
        max_new_tokens=4,
        eos_token_ids=(2,),
    )

    assert result["generated_token_count"] == 2
    assert result["stop_reason"] == "eos"
    assert result["final_generation"] == 24
    assert result["final_kv_contract"]["sequence_length"] == 70
    assert result["decode_trace"]["step_count"] == 2
    assert result["decode_trace"]["final_sequence_length"] == 70
    assert all(value.count("decode") == 2 for value in calls.values())
    encoded = json.dumps(result, ensure_ascii=True).lower()
    assert "selected_token" not in encoded
    assert "token_id" not in encoded
    adapter.cleanup("eos_committed")


def test_mm125_cancel_after_committed_step_aborts_and_removes_temporary_inputs(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, 2, bounded_tokens=(5, 6, 7),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_bounded_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
            max_new_tokens=3,
            cancel_after_step=2,
        )

    assert caught.value.reason_code == "qwen3_mm1_multisidecar_cancelled"
    assert adapter.chain.phase == "aborted"
    assert all(value[-1] == "abort" and value.count("decode") == 2 for value in calls.values())
    assert list(root.glob("qwen3-*.pt")) == []
    assert list(root.glob(f"{adapter._retained_prefix}next-*.pt")) == []


def test_mm125_second_step_segment_failure_aborts_and_removes_all_generations(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path,
        3,
        bounded_tokens=(5, 6, 7),
        fail_decode_index=1,
        fail_decode_step=2,
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_bounded_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
            max_new_tokens=3,
        )

    assert caught.value.reason_code == "qwen3_mm1_bounded_decode_execution_failed"
    assert calls[0][-1] == "abort"
    assert calls[1][-1] == "abort"
    assert calls[2][-1] == "abort"
    assert calls[0].count("decode") == 2
    assert calls[1].count("decode") == 2
    assert calls[2].count("decode") == 1
    assert list(root.glob("qwen3-*.pt")) == []
    assert list(root.glob(f"{adapter._retained_prefix}next-*.pt")) == []


def test_mm125_rejects_unbounded_or_duplicate_eos_contract_before_prepare(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, bounded_tokens=(5,),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_bounded_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
            max_new_tokens=3,
            eos_token_ids=(2, 2),
        )

    assert caught.value.reason_code == "qwen3_mm1_bounded_decode_contract_invalid"
    assert all(value == [] for value in calls.values())


def test_mm126_adapter_decodes_retained_ledger_through_isolated_worker_boundary(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, _calls, root = _fixture(
        tmp_path, 2, bounded_tokens=(5, 2),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)
    generated = adapter.execute_bounded_decode(
        input_ref=input_ref,
        decode_input_ref=decode_ref,
        decode_binding=decode_binding,
        max_new_tokens=2,
        eos_token_ids=(2,),
    )
    model_root = root / "tokenizer-model"
    model_root.mkdir()
    seen: dict[str, object] = {}

    class FakeTokenizer:
        def decode(self, ids, *, skip_special_tokens):
            seen["ids"] = list(ids)
            seen["skip_special_tokens"] = skip_special_tokens
            return "generated text"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_path, **kwargs):
            seen["kwargs"] = kwargs
            return FakeTokenizer()

    from scripts.model_tools.qwen3_token_ledger_worker import execute_request

    def isolated_runner(request, _timeout):
        isolated_request = dict(request)
        isolated_request["controller_python"] = str(root / "controller-python")
        return execute_request(
            isolated_request,
            module_loader=lambda _name: SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

    result = adapter.decode_token_ledger(
        ledger_id=generated["token_ledger"]["ledger_id"],
        model=model_root,
        worker_runner=isolated_runner,
    )

    assert result["status"] == "decoded"
    assert result["text"] == "generated text"
    assert seen["ids"] == [5, 2]
    assert seen["skip_special_tokens"] is True
    assert seen["kwargs"] == {"local_files_only": True, "trust_remote_code": False}
    assert "token_id" not in json.dumps(result, ensure_ascii=True).lower()
    assert adapter.cleanup("tokenizer_decoded")["removed_artifacts"] == 2


def test_mm127_sampling_binds_sampler_and_draw_evidence_without_control_token_ids(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, _calls, root = _fixture(
        tmp_path, 2, bounded_tokens=(5, 6, 7),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)
    policy = build_mm1_sampling_policy_snapshot(
        temperature=1.0, top_k=1, top_p=1.0, seed=123,
        issued_at=int(time.time()) - 1, expires_at=int(time.time()) + 3600,
        policy_id="mm127-policy-fixture",
    )

    result = adapter.execute_bounded_decode(
        input_ref=input_ref,
        decode_input_ref=decode_ref,
        decode_binding=decode_binding,
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        seed=123,
        policy_snapshot=policy,
    )

    assert result["sampling"]["mode"] == "multinomial"
    assert result["sampling"]["top_k"] == 1
    assert result["sampling"]["seed"] == 123
    assert result["sampling"]["draw_count"] == 3
    assert len(result["sampling"]["sha256"]) == 64
    encoded = json.dumps(result, ensure_ascii=True).lower()
    assert "token_id" not in encoded
    ledger_path = adapter.ledger_path(result["token_ledger"]["ledger_id"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["sampling"]["mode"] == "multinomial"
    assert ledger["sampling_sha256"] == result["sampling"]["sha256"]
    assert len(ledger["draw_evidence"]) == 3
    assert all(len(item["sha256"]) == 64 for item in ledger["draw_evidence"])
    assert result["sampling_quality"]["step_count"] == 3
    assert result["sampling_quality"]["sampling_sha256"] == result["sampling"]["sha256"]
    assert result["sampling_quality"]["candidate_count_min"] == 1
    assert ledger["quality_summary"]["sha256"] == result["sampling_quality"]["sha256"]
    assert result["sampling_policy"]["policy_id"] == "mm127-policy-fixture"
    assert ledger["policy_snapshot_sha256"] == result["sampling_policy"]["snapshot_sha256"]
    assert adapter.cleanup("sampling_committed")["removed_artifacts"] == 2


def test_mm128_replay_boundary_validates_ledger_without_model_or_tokenizer(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, _calls, root = _fixture(
        tmp_path, 2, bounded_tokens=(5, 6),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)
    policy = build_mm1_sampling_policy_snapshot(
        temperature=1.0, top_k=1, top_p=1.0, seed=17,
        issued_at=int(time.time()) - 1, expires_at=int(time.time()) + 3600,
        policy_id="mm128-policy-fixture",
    )
    generated = adapter.execute_bounded_decode(
        input_ref=input_ref,
        decode_input_ref=decode_ref,
        decode_binding=decode_binding,
        max_new_tokens=2,
        temperature=1.0,
        top_k=1,
        seed=17,
        policy_snapshot=policy,
    )
    seen: dict[str, object] = {}
    from scripts.model_tools.qwen3_token_ledger_worker import execute_request

    def replay_runner(request, _timeout):
        seen.update(request)
        return execute_request(request)

    replay = adapter.replay_token_ledger(
        generated["token_ledger"]["ledger_id"],
        expected_sampling_sha256=generated["sampling"]["sha256"],
        expected_quality_sha256=generated["sampling_quality"]["sha256"],
        worker_runner=replay_runner,
    )

    assert replay["status"] == "replay_validated"
    assert replay["operation"] == "qwen3_token_ledger_replay"
    assert replay["replay"] == {
        "sampler_validated": True,
        "quality_validated": True,
        "policy_validated": True,
        "full_model_materialized": False,
        "weights_loaded": False,
    }
    assert "model_path" not in seen
    encoded = json.dumps(replay, ensure_ascii=True).lower()
    assert "token_id" not in encoded
    assert "text" not in replay
    rejected = adapter.replay_token_ledger(
        generated["token_ledger"]["ledger_id"],
        expected_sampling_sha256="f" * 64,
        expected_quality_sha256=generated["sampling_quality"]["sha256"],
        worker_runner=replay_runner,
    )
    assert rejected["status"] == "replay_failed"
    adapter.cleanup("replay_validated")


def test_mm127_same_seed_repeats_local_sampling_and_top_p_keeps_boundary_token():
    from qwen3_multimodal_multisidecar import Qwen3MultimodalMultiSidecarAdapter

    logits = torch.tensor([[[2.0, 1.0, 0.5, 0.0]]])
    sampling = {
        "mode": "multinomial",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.8,
        "seed": 17,
    }
    first_generator = torch.Generator(device="cpu").manual_seed(17)
    second_generator = torch.Generator(device="cpu").manual_seed(17)
    first = Qwen3MultimodalMultiSidecarAdapter._select_next_token(
        logits, sampling=sampling, generator=first_generator,
    )
    second = Qwen3MultimodalMultiSidecarAdapter._select_next_token(
        logits, sampling=sampling, generator=second_generator,
    )
    assert first == second
    assert first in {0, 1}


def test_mm129_rejects_implicit_non_greedy_sampling_before_prepare(tmp_path):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, bounded_tokens=(5,),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_bounded_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
            max_new_tokens=1,
            temperature=1.0,
        )

    assert caught.value.reason_code == "qwen3_mm1_sampling_policy_required"
    assert all(value == [] for value in calls.values())


def test_mm129_rejects_expired_sampling_policy():
    snapshot = build_mm1_sampling_policy_snapshot(
        temperature=1.0,
        issued_at=int(time.time()) - 7200,
        expires_at=int(time.time()) - 1,
        policy_id="mm129-expired-policy",
    )
    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        from qwen3_multimodal_multisidecar import validate_mm1_sampling_policy_snapshot

        validate_mm1_sampling_policy_snapshot(snapshot)
    assert caught.value.reason_code == "qwen3_mm1_sampling_policy_expired"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"top_k": -1},
        {"top_k": 4097},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_mm127_sampling_contract_rejects_invalid_bounds_before_prepare(tmp_path, kwargs):
    adapter, _manifest_value, _staged_value, _binding, input_ref, calls, root = _fixture(
        tmp_path, bounded_tokens=(5,),
    )
    decode_ref, decode_binding = _decode_fixture(adapter, root, torch_input=True)

    with pytest.raises(Qwen3MultimodalRuntimeError) as caught:
        adapter.execute_bounded_decode(
            input_ref=input_ref,
            decode_input_ref=decode_ref,
            decode_binding=decode_binding,
            max_new_tokens=1,
            **kwargs,
        )

    assert caught.value.reason_code == "qwen3_mm1_bounded_decode_contract_invalid"
    assert all(value == [] for value in calls.values())
    assert list(root.glob("qwen3-*.pt")) == []
