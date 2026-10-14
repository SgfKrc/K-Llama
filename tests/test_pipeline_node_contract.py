"""Unified layer-node contract tests for CORE-LAYER-CONTRACT-01."""

import json

import pytest

from src.llama_rpc_contract import RpcShardLeaseBook
from src.pipeline_capacity import solve_pipeline_capacity
from src.pipeline_node_contract import (
    PipelineArtifactRef,
    PipelineNode,
    PipelineNodeCapacity,
    PipelineNodeContractError,
    build_aggregate_resource_view,
    pipeline_layout_from_capacity_plan,
    pipeline_layout_from_relay_handoff,
    pipeline_node_from_assignment_manifest,
    pipeline_node_from_rpc_lease,
    validate_pipeline_nodes,
)
from src.relay_contract import RelayModelIdentity, build_relay_handoff


MODEL_SHA = "a" * 64


def _capacity(capacity=1000, required=200):
    return PipelineNodeCapacity(
        capacity_bytes=capacity,
        required_bytes=required,
        reserve_bytes=50,
        execution_device="cpu",
    )


def _artifact(start, end, *, model_sha=MODEL_SHA, architecture="qwen3"):
    return PipelineArtifactRef(
        model_sha256=model_sha,
        artifact_kind="safetensors_assignment",
        architecture=architecture,
        source_layer_range=(start, end),
    )


def _node(node_id, start, end, *, embedding=False, head=False, **overrides):
    fields = {
        "node_id": node_id,
        "kind": "local" if node_id == "local" else "remote_pipeline",
        "layer_range": (start, end),
        "engine": "pytorch",
        "location": "local" if node_id == "local" else f"node:{node_id}",
        "capacity": _capacity(),
        "artifact": _artifact(start, end),
        "has_embedding": embedding,
        "has_lm_head": head,
        "federated": node_id != "local",
    }
    fields.update(overrides)
    return PipelineNode(**fields)


def test_layout_requires_exact_coverage_and_stable_public_contract():
    layout = validate_pipeline_nodes([
        _node("remote", 2, 4, head=True),
        _node("local", 0, 2, embedding=True),
    ], total_layers=4)

    public = layout.to_dict()
    assert [node["layer_range"] for node in public["nodes"]] == [[0, 2], [2, 4]]
    assert public["is_distributed"] is True
    assert public["engines"] == ["pytorch"]
    assert public["aggregate_capacity"]["required_bytes"] == 400
    assert len(public["contract_sha256"]) == 64
    assert public["contract_sha256"] == layout.to_dict()["contract_sha256"]


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("llama.cpp", "llama_cpp"),
        ("llama-cpp", "llama_cpp"),
        ("gguf", "llama_cpp"),
        ("llama", "llama_cpp"),
        ("torch", "pytorch"),
    ],
)
def test_pipeline_node_normalizes_local_engine_aliases(alias, canonical):
    node = _node("local", 0, 1, embedding=True, head=True, engine=alias)

    assert node.engine == canonical
    assert node.to_dict()["engine"] == canonical


def test_pipeline_node_preserves_non_local_engine_and_rejects_unknown():
    assert _node(
        "local", 0, 1, embedding=True, head=True, engine="external_api",
    ).engine == "external_api"

    with pytest.raises(PipelineNodeContractError, match="engine is unsupported"):
        _node("local", 0, 1, embedding=True, head=True, engine="llama_magic")


@pytest.mark.parametrize("second_range", [(3, 4), (1, 4)])
def test_layout_rejects_gap_or_overlap(second_range):
    with pytest.raises(PipelineNodeContractError, match="cover every layer"):
        validate_pipeline_nodes([
            _node("local", 0, 2, embedding=True),
            _node("remote", *second_range, head=True),
        ], total_layers=4)


def test_layout_rejects_identity_architecture_and_boundary_owner_mismatch():
    with pytest.raises(PipelineNodeContractError, match="model identity"):
        validate_pipeline_nodes([
            _node("local", 0, 2, embedding=True),
            _node(
                "remote", 2, 4, head=True,
                artifact=_artifact(2, 4, model_sha="b" * 64),
            ),
        ], total_layers=4)
    with pytest.raises(PipelineNodeContractError, match="architecture"):
        validate_pipeline_nodes([
            _node("local", 0, 2, embedding=True),
            _node(
                "remote", 2, 4, head=True,
                artifact=_artifact(2, 4, architecture="llama"),
            ),
        ], total_layers=4)
    with pytest.raises(PipelineNodeContractError, match="embedding"):
        validate_pipeline_nodes([
            _node("local", 0, 2),
            _node("remote", 2, 4, head=True),
        ], total_layers=4)


def test_artifact_geometry_rejects_stale_mtp_metadata():
    valid = validate_pipeline_nodes([
        _node(
            "local", 0, 2, embedding=True,
            artifact=PipelineArtifactRef(
                model_sha256=MODEL_SHA,
                artifact_kind="gguf",
                architecture="qwen35",
                source_layer_range=(0, 2),
                block_count=2,
            ),
        ),
        _node(
            "remote", 2, 4, head=True,
            artifact=PipelineArtifactRef(
                model_sha256=MODEL_SHA,
                artifact_kind="gguf",
                architecture="qwen35",
                source_layer_range=(2, 4),
                block_count=3,
                nextn_predict_layers=1,
            ),
        ),
    ], total_layers=4)
    assert valid.total_layers == 4

    with pytest.raises(PipelineNodeContractError, match="must set nextn"):
        validate_pipeline_nodes([
            _node(
                "local", 0, 2, embedding=True,
                artifact=PipelineArtifactRef(
                    model_sha256=MODEL_SHA,
                    artifact_kind="gguf",
                    architecture="qwen35",
                    source_layer_range=(0, 2),
                    block_count=3,
                    nextn_predict_layers=1,
                ),
            ),
            _node(
                "remote", 2, 4, head=True,
                artifact=_artifact(2, 4, architecture="qwen35"),
            ),
        ], total_layers=4)


def test_capacity_plan_adapter_preserves_solver_assignment():
    descriptor = {
        "model_id": "fixture",
        "model_type": "qwen3",
        "model_sha256": MODEL_SHA,
        "total_layers": 2,
        "layer_weight_bytes": [100, 100],
        "component_weight_bytes": {
            "embedding": 10, "final_norm": 5, "lm_head": 10,
            "visual": 0, "mtp": 0, "multimodal": 0, "other": 0,
        },
    }
    capacity_nodes = [
        {
            "node_id": "local", "role": "master", "capacity_bytes": 150,
            "reserve_bytes": 10, "runtime_multiplier": 1.0,
            "execution_device": "cpu", "capacity_source": "test", "score": 10,
        },
        {
            "node_id": "remote", "role": "client", "capacity_bytes": 150,
            "reserve_bytes": 10, "runtime_multiplier": 1.0,
            "execution_device": "cpu", "capacity_source": "test", "score": 9,
        },
    ]
    plan = solve_pipeline_capacity(
        descriptor, capacity_nodes, safety_margin=1.0, require_distributed=True,
    )
    layout = pipeline_layout_from_capacity_plan(
        plan,
        node_metadata={
            "local": {"is_local": True, "location": "local"},
            "remote": {"is_local": False, "location": "node:remote"},
        },
    )

    assert layout.model_sha256 == MODEL_SHA
    assert [node.layer_range for node in layout.nodes] == [(0, 1), (1, 2)]
    assert layout.nodes[-1].federated is True


def test_capacity_plan_adapter_preserves_selected_segment_artifact():
    plan = {
        "admitted": True,
        "total_layers": 2,
        "model_sha256": MODEL_SHA,
        "model_type": "qwen2",
        "assignments": [
            {
                "node_id": "local", "role": "master",
                "start_layer": 0, "end_layer": 1,
                "has_embedding": True, "has_lm_head": False,
                "capacity_bytes": 100, "required_bytes": 10,
            },
            {
                "node_id": "edge", "role": "client",
                "start_layer": 1, "end_layer": 2,
                "has_embedding": False, "has_lm_head": True,
                "capacity_bytes": 100, "required_bytes": 10,
                "layer_artifact": {
                    "layer_range": [1, 2],
                    "segment_mode": "tail",
                    "model_id": "tail1-2.gguf",
                    "artifact_sha256": "b" * 64,
                    "source_model_sha256": "c" * 64,
                },
            },
        ],
    }

    layout = pipeline_layout_from_capacity_plan(plan)

    assert layout.nodes[-1].artifact.model_sha256 == MODEL_SHA
    assert layout.nodes[-1].artifact.artifact_kind == "gguf_segment"
    assert layout.nodes[-1].artifact.artifact_sha256 == "b" * 64
    assert layout.nodes[-1].artifact.source_layer_range == (1, 2)
    assert layout.nodes[-1].engine == "llama_cpp"
    assert layout.nodes[-1].cross_engine is True
    assert layout.nodes[-1].handoff_at == 1


def test_relay_and_rpc_contracts_map_without_control_plane_imports():
    upstream = RelayModelIdentity(
        model_sha256=MODEL_SHA, architecture="qwen35", block_count=5,
        n_embd=128, nextn_predict_layers=1, artifact_sha256="b" * 64,
    )
    downstream = RelayModelIdentity(
        model_sha256=MODEL_SHA, architecture="qwen35", block_count=3,
        n_embd=128, nextn_predict_layers=1, artifact_sha256="c" * 64,
    )
    decision = build_relay_handoff(upstream, downstream, 2)
    relay_layout = pipeline_layout_from_relay_handoff(decision.handoff)
    assert [node.layer_range for node in relay_layout.nodes] == [(0, 2), (2, 4)]
    assert relay_layout.nodes[-1].artifact.source_layer_range == (2, 4)
    assert relay_layout.is_distributed is False

    lease = RpcShardLeaseBook().assign(
        "shard-2", "rpc-edge", MODEL_SHA,
        {
            "layer_range": [2, 4], "engine": "llama.cpp",
            "artifact_kind": "gguf", "has_lm_head": True,
        },
    )
    rpc_node = pipeline_node_from_rpc_lease(lease, location="node:rpc-edge")
    assert rpc_node.kind == "remote_rpc"
    assert rpc_node.layer_range == (2, 4)
    assert rpc_node.federated is True
    assert rpc_node.engine == "llama_cpp"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("llama_cpp", "llama_cpp"),
        ("llama.cpp", "llama_cpp"),
        ("llama-cpp", "llama_cpp"),
        ("gguf", "llama_cpp"),
        ("llama", "llama_cpp"),
        ("torch", "pytorch"),
        ("pytorch", "pytorch"),
    ],
)
def test_rpc_lease_normalizes_local_engine_aliases(alias, canonical):
    lease = RpcShardLeaseBook().assign(
        "shard-2", "rpc-edge", MODEL_SHA,
        {"layer_range": [2, 4], "engine": alias, "artifact_kind": "gguf"},
    )

    assert pipeline_node_from_rpc_lease(lease).engine == canonical


def test_rpc_lease_preserves_non_local_engine_identifier():
    lease = RpcShardLeaseBook().assign(
        "shard-2", "rpc-edge", MODEL_SHA,
        {
            "layer_range": [2, 4], "engine": "external_api",
            "artifact_kind": "remote",
        },
    )

    assert pipeline_node_from_rpc_lease(lease).engine == "external_api"


def test_rpc_lease_rejects_unknown_engine_alias():
    lease = RpcShardLeaseBook().assign(
        "shard-2", "rpc-edge", MODEL_SHA,
        {"layer_range": [2, 4], "engine": "llama_magic", "artifact_kind": "gguf"},
    )

    with pytest.raises(PipelineNodeContractError, match="allocation.engine is unsupported"):
        pipeline_node_from_rpc_lease(lease)


def test_assignment_manifest_maps_to_the_same_node_contract():
    node = pipeline_node_from_assignment_manifest(
        {
            "manifest_kind": "pytorch_pipeline_assignment",
            "manifest_sha256": "d" * 64,
            "model_sha256": MODEL_SHA,
            "model_type": "qwen3",
            "node_id": "worker-a",
            "layer_range": [0, 2],
            "has_embedding": True,
            "has_lm_head": True,
        },
        kind="local",
        location="local",
        capacity=_capacity(),
    )

    assert node.layer_range == (0, 2)
    assert node.artifact.manifest_sha256 == "d" * 64
    assert node.artifact.artifact_kind == "safetensors_assignment"


def test_aggregate_resource_view_counts_only_available_nodes_and_hides_addresses():
    view = build_aggregate_resource_view([
        {
            "node_id": "master", "role": "master", "node_type": "pc",
            "state": "online", "address": "127.0.0.1:8888",
            "device_info": {
                "cpu": {"physical_cores": 4, "logical_cores": 8},
                "ram": {"total_gb": 16, "available_gb": 10},
                "gpu": {
                    "cuda_available": True, "vram_total_gb": 8,
                    "vram_free_gb": 6,
                },
            },
        },
        {
            "node_id": "edge", "role": "client", "node_type": "android",
            "state": "online", "address": "100.64.0.2:8888",
            "device_info": {
                "cpu": {"physical_cores": 4, "logical_cores": 4},
                "ram": {"total_gb": 8, "available_gb": 2},
            },
        },
        {
            "node_id": "offline", "state": "offline",
            "device_info": {"ram": {"total_gb": 128, "available_gb": 128}},
        },
    ], local_node_id="master")

    assert view["is_distributed"] is True
    assert view["remote_available_count"] == 1
    assert view["unavailable_node_count"] == 1
    assert [node["node_id"] for node in view["available"]["remote"]] == ["edge"]
    assert view["totals"]["logical_cores"] == 12
    assert view["totals"]["ram_available_gb"] == 12
    serialized = json.dumps(view)
    assert "127.0.0.1" not in serialized
    assert "100.64.0.2" not in serialized
