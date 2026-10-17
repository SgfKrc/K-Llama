from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
from qwen3_multimodal_runtime import (  # noqa: E402
    Qwen3MultimodalRuntimeError,
    Qwen3MultimodalSidecarAdapter,
    Qwen3MultimodalSyntheticExecutor,
)
from qwen3_pipeline_control import router as network_control_router  # noqa: E402
from qwen3_pipeline_data_plane import (  # noqa: E402
    Qwen3ArtifactTransferRuntime,
    router as transfer_router,
)
from qwen3_pipeline_network import (  # noqa: E402
    Qwen3NetworkError,
    Qwen3NetworkTransferCoordinator,
)
from qwen3_pipeline_peer_auth import (  # noqa: E402
    Qwen3PeerAuthMiddleware,
    Qwen3PeerRequestVerifier,
)
from qwen3_pipeline_transaction import build_qwen3_dry_run_contract  # noqa: E402


SECRET = "qwen3-network-contract-secret-value!!"


def _manifest() -> dict:
    config = json.loads(
        (ROOT / "models" / "qwen3-vl-4b-instruct" / "config.json").read_text(
            encoding="utf-8",
        ),
    )
    profile = build_mm1_model_profile(config)
    return build_mm1_model_manifest(
        model_id="fixture-qwen3-vl-mm1",
        model_family="qwen3_vl",
        runtime="transformers_sidecar",
        revision="fixture-revision",
        components=[
            {
                "component_id": "processor",
                "artifact_id": "processor-artifact",
                "component_kind": "processor",
                "format": "tokenizer",
                "revision": "fixture-revision",
                "size_bytes": 128,
                "sha256": "a" * 64,
            },
            {
                "component_id": "text",
                "artifact_id": "text-artifact",
                "component_kind": "text_weights",
                "format": "safetensors",
                "revision": "fixture-revision",
                "size_bytes": 1024,
                "sha256": "b" * 64,
            },
            {
                "component_id": "vision",
                "artifact_id": "vision-artifact",
                "component_kind": "vision_weights",
                "format": "safetensors",
                "revision": "fixture-revision",
                "size_bytes": 2048,
                "sha256": "c" * 64,
            },
        ],
        text=profile["text"],
        vision=profile["vision"],
        processor=profile["processor"],
    )


def _contract(generation: int = 70) -> dict:
    return build_qwen3_dry_run_contract(
        config_id=f"cfg-mm1-{generation}",
        plan_id=f"plan-mm1-{generation}",
        generation=generation,
        model_id="qwen3-vl-mm1",
        model_sha256="d" * 64,
        total_layers=2,
        hidden_size=4,
        execution_mode="node_local_sidecar",
        segments=[
            {
                "node_id": "node-a", "layer_range": [0, 1],
                "has_embedding": True, "has_lm_head": False,
                "required_bytes": 100, "assignment_manifest_sha256": "1" * 64,
                "execution_device": "cpu", "dtype": "float32",
            },
            {
                "node_id": "node-b", "layer_range": [1, 2],
                "has_embedding": False, "has_lm_head": True,
                "required_bytes": 100, "assignment_manifest_sha256": "2" * 64,
                "execution_device": "cpu", "dtype": "float32",
            },
        ],
    )


def _target(tmp_path: Path, *, now: list[float] | None = None):
    clock = (lambda: now[0]) if now is not None else None
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path / "node-b",
        cluster_secret=SECRET,
        clock=clock,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id="node-b", runtime=runtime,
    )
    state = {
        "value": {
            "schema_version": 1, "local_node_id": "node-b", "last_generation": -1,
            "active_contract": {}, "transfers": {}, "outputs": {}, "updated_at": "",
        },
    }

    def load():
        return json.loads(json.dumps(state["value"]))

    def save(value):
        state["value"] = json.loads(json.dumps(value))
        return load()

    coordinator.configure_persistent_ledger(load=load, save=save)
    app = FastAPI()
    app.state.qwen3_artifact_transfer = runtime
    app.add_middleware(
        Qwen3PeerAuthMiddleware,
        verifier=Qwen3PeerRequestVerifier(
            SECRET,
            is_authenticated_peer=lambda peer: peer == "node-a",
            **({"clock": clock} if clock is not None else {}),
        ),
    )
    app.include_router(transfer_router)
    app.include_router(network_control_router)
    return runtime, coordinator, state, TestClient(app)


def _committed_transfer(runtime, coordinator, contract, data=b"visual-input"):
    coordinator.activate(contract)
    coordinator.begin_phase("prefill", contract["generation"])
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill", from_segment=0, to_segment=1,
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )
    runtime.receiver.write(
        plan["transfer_id"], ticket=plan["ticket"],
        authenticated_peer_id="node-a", offset=0, data=data,
    )
    runtime.receiver.commit(
        plan["transfer_id"], ticket=plan["ticket"],
        authenticated_peer_id="node-a",
    )
    return plan, coordinator.commit_reference(plan["transfer_id"])


def test_mm1_synthetic_consume_binds_reference_and_cleans_output(tmp_path):
    runtime, coordinator, _state, _client = _target(tmp_path)
    contract = _contract()
    plan, _reference = _committed_transfer(runtime, coordinator, contract)
    executor = Qwen3MultimodalSyntheticExecutor(
        manifest=_manifest(), artifact_root=runtime.receiver.root, visual_tokens=8,
    )
    result = coordinator.consume_transfer(
        plan["transfer_id"], phase="prefill", generation=contract["generation"],
        batch_size=1, sequence_length=4, dtype="float32", device="cpu",
        has_next_segment=False, executor=executor,
    )
    assert len(result["mm1_binding_sha256"]) == 64
    assert result["full_model_materialized"] is False
    assert "path" not in json.dumps(result).lower()
    assert list(runtime.receiver.root.glob("qwen3-consume-mm1-*.pt"))
    cleanup = coordinator.release()
    assert cleanup["cleanup_complete"] is True
    assert not list(runtime.receiver.root.glob("qwen3-consume-mm1-*.pt"))


def test_mm1_synthetic_failure_is_terminal_and_cleans_output(tmp_path):
    runtime, coordinator, state, _client = _target(tmp_path)
    contract = _contract(71)
    plan, _reference = _committed_transfer(runtime, coordinator, contract)
    executor = Qwen3MultimodalSyntheticExecutor(
        manifest=_manifest(), artifact_root=runtime.receiver.root, fail_phase="prefill",
    )
    with pytest.raises(Qwen3NetworkError, match="synthetic MM1 execution failed"):
        coordinator.consume_transfer(
            plan["transfer_id"], phase="prefill", generation=contract["generation"],
            batch_size=1, sequence_length=4, dtype="float32", device="cpu",
            has_next_segment=False, executor=executor,
        )
    assert state["value"]["transfers"][plan["transfer_id"]]["status"] == "failed"
    assert not list(runtime.receiver.root.glob("qwen3-consume-mm1-*.pt"))


def test_mm1_epoch_and_ttl_fence_terminal_state(tmp_path):
    runtime, coordinator, state, _client = _target(tmp_path)
    contract = _contract(72)
    coordinator.activate(contract)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"epoch-input"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876", source_peer_id="node-a",
        chain_id=contract["contract_sha256"], generation=contract["generation"],
        phase="prefill", from_segment=0, to_segment=1,
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )
    fenced = coordinator.authorize_control_peer(
        "node-a", contract=contract, peer_epoch=1,
    )
    assert fenced["generation"] == contract["generation"]
    assert state["value"]["transfers"][plan["transfer_id"]]["status"] == "invalidated"
    assert not list(runtime.receiver.root.glob("*.part"))

    now = [1000.0]
    runtime2, coordinator2, state2, _client2 = _target(tmp_path / "ttl", now=now)
    contract2 = _contract(73)
    coordinator2.activate(contract2)
    coordinator2.begin_phase("prefill", contract2["generation"])
    plan2 = coordinator2.begin_receive(
        base_url="http://127.0.0.1:9876", source_peer_id="node-a",
        chain_id=contract2["contract_sha256"], generation=contract2["generation"],
        phase="prefill", from_segment=0, to_segment=1,
        size_bytes=4, sha256=hashlib.sha256(b"ttl!").hexdigest(), ttl_seconds=1,
    )
    now[0] += 2
    expired = coordinator2.cleanup_expired()
    assert expired["reconciled_transfers"] == 1
    assert state2["value"]["transfers"][plan2["transfer_id"]]["status"] == "expired"
    assert not list(runtime2.receiver.root.glob("*.part"))


def test_mm1_sidecar_adapter_projects_binding_and_forwards_cleanup(tmp_path):
    runtime, coordinator, _state, _client = _target(tmp_path)
    contract = _contract(74)
    plan, reference = _committed_transfer(runtime, coordinator, contract)
    source = tmp_path / "adapter-input.pt"
    source.write_bytes(b"sidecar-input")
    calls = []

    class _Delegate:
        def __call__(self, _input_path, _request):
            return {
                "status": "isolated-sidecar-metadata",
                "hidden_handoff": {
                    "shape": [1, 4, 2560], "dtype": "float32", "device": "cpu",
                },
            }

        def cleanup(self, _request, reason_code):
            calls.append(reason_code)

    adapter = Qwen3MultimodalSidecarAdapter(
        _Delegate(), manifest=_manifest(), visual_tokens=4,
    )
    report = adapter(
        source,
        {
            "reference": reference,
            "chain_id": contract["contract_sha256"],
            "transfer_id": plan["transfer_id"],
            "generation": contract["generation"], "phase": "prefill",
            "batch_size": 1, "sequence_length": 4,
            "dtype": "float32", "device": "cpu",
        },
    )
    assert report["status"] == "isolated-sidecar-metadata"
    assert len(report["mm1_binding_sha256"]) == 64
    assert report["mm1_metadata"]["visual_shape"] == [1, 4, 2560]
    assert "path" not in json.dumps(report).lower()
    adapter.cleanup({"transfer_id": plan["transfer_id"]}, "test-cleanup")
    assert calls == ["test-cleanup"]
