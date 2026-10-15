"""Layout engine identity follows the selected execution, without loading models."""
from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from model_host import ModelHost
from pipeline_node_contract import (
    PipelineNodeContractError,
    pipeline_layout_from_capacity_plan,
)
from scheduler import NodeInfo, NodeState, Scheduler


def _plan():
    return {
        "admitted": True, "total_layers": 4,
        "model_sha256": "a" * 64, "model_type": "qwen35",
        "assignments": [
            {"node_id": "master", "role": "master", "start_layer": 0,
             "end_layer": 2, "layers_count": 2, "has_embedding": True,
             "has_lm_head": False, "capacity_bytes": 100, "required_bytes": 10},
            {"node_id": "edge", "role": "client", "start_layer": 2,
             "end_layer": 4, "layers_count": 2, "has_embedding": False,
             "has_lm_head": True, "capacity_bytes": 100, "required_bytes": 10,
             "layer_artifact": {"layer_range": [2, 4], "segment_mode": "tail",
                                "artifact_sha256": "b" * 64}},
        ],
    }


def _scheduler(host):
    scheduler = Scheduler(host=host)
    scheduler._role_override = "master"
    scheduler.nodes = {
        "master": NodeInfo(node_id="master", role="master", state=NodeState.ONLINE,
                           device_info={"backend_id": "pytorch"}),
        "edge": NodeInfo(node_id="edge", role="client", state=NodeState.ONLINE),
    }
    return scheduler


@pytest.mark.parametrize("engine", ["llama_cpp", "pytorch"])
def test_cached_layout_uses_runtime_engine_and_actual_cross_engine_boundary(engine):
    scheduler = _scheduler(SimpleNamespace(engine_type=engine))
    plan = _plan()
    scheduler._active_pipeline_capacity_plan = deepcopy(plan)

    public = scheduler.get_pipeline_capacity_plan()

    assert public["admitted"] is True
    master, edge = public["pipeline_layout"]["nodes"]
    assert master["engine"] == engine
    assert master["artifact"]["artifact_kind"] == (
        "gguf_segment" if engine == "llama_cpp" else "safetensors_assignment"
    )
    assert edge["engine"] == "llama_cpp"
    assert edge["cross_engine"] is (engine == "pytorch")
    assert edge["handoff_at"] == (2 if engine == "pytorch" else None)
    assert scheduler._active_pipeline_capacity_plan == plan


@pytest.mark.parametrize("phase", ["prepared", "materialized", "committed"])
def test_prepared_candidate_does_not_relabel_cached_active_plan(phase):
    host = SimpleNamespace(
        engine_type="pytorch", _pipeline_candidate_lock=threading.RLock(),
        _pipeline_candidate_transaction=SimpleNamespace(
            phase=phase, manager=SimpleNamespace(engine_type="llama_cpp"),
        ),
    )
    scheduler = _scheduler(host)
    scheduler._active_pipeline_capacity_plan = deepcopy(_plan())
    public = scheduler.get_pipeline_capacity_plan()
    assert public["pipeline_layout"]["nodes"][0]["engine"] == "pytorch"
    assert public["pipeline_layout"]["nodes"][1]["cross_engine"] is True
    assert public["pipeline_layout"]["nodes"][1]["handoff_at"] == 2


def test_fresh_candidate_plan_binds_engine_from_admission_records():
    scheduler = _scheduler(SimpleNamespace(engine_type="pytorch"))
    plan = _plan()
    public = scheduler._attach_pipeline_node_contract(plan, capacity_nodes=[
        {"node_id": "master", "engine": "llama_cpp"},
        {"node_id": "edge", "engine": "llama_cpp"},
    ])
    assert public["assignments"][0]["engine"] == "llama_cpp"
    assert [n["engine"] for n in public["pipeline_layout"]["nodes"]] == ["llama_cpp"] * 2
    assert public["pipeline_layout"]["nodes"][1]["cross_engine"] is False
    assert "engine" not in plan["assignments"][0]
    scheduler._active_pipeline_capacity_plan = public
    assert scheduler.get_pipeline_capacity_plan()["pipeline_layout"] == public["pipeline_layout"]


def test_layout_metadata_does_not_materialize_lazy_manager(monkeypatch):
    host = ModelHost()
    lazy = object.__getattribute__(host, "_manager")
    def unexpected_load(*args, **kwargs):
        raise AssertionError("layout read must not instantiate the lazy manager")
    monkeypatch.setattr(type(lazy), "_get_instance", unexpected_load)
    scheduler = _scheduler(host)

    assert scheduler._pipeline_node_metadata()["master"]["engine"] == "pytorch"
    assert host.peek_manager() is None


@pytest.mark.parametrize("engine, stale_engine", [
    ("llama.cpp", "pytorch"), ("pytorch", "llama_cpp"),
])
def test_assignment_engine_overrides_stale_node_engine_and_artifact_kind(engine, stale_engine):
    plan = _plan()
    plan["assignments"][0]["engine"] = engine
    layout = pipeline_layout_from_capacity_plan(plan, node_metadata={"master": {
        "engine": stale_engine,
        "artifact_kind": "gguf_segment" if stale_engine == "llama_cpp" else "safetensors_assignment",
    }})
    master, edge = layout.nodes
    assert master.engine == ("llama_cpp" if engine == "llama.cpp" else engine)
    assert master.artifact.artifact_kind == (
        "gguf_segment" if master.engine == "llama_cpp" else "safetensors_assignment"
    )
    assert edge.cross_engine is (master.engine == "pytorch")


def test_selected_gguf_segment_overrides_installation_engine_metadata():
    plan = _plan()
    plan["assignments"][1]["engine"] = "pytorch"
    layout = pipeline_layout_from_capacity_plan(plan, node_metadata={
        "edge": {"engine": "pytorch", "artifact_kind": "safetensors_assignment"},
    })
    assert layout.nodes[-1].engine == "llama_cpp"
    assert layout.nodes[-1].artifact.artifact_kind == "gguf_segment"


def test_invalid_assignment_engine_does_not_fall_back_to_node_default():
    plan = _plan()
    plan["assignments"][0]["engine"] = "unknown_backend"
    with pytest.raises(PipelineNodeContractError, match="engine is unsupported"):
        pipeline_layout_from_capacity_plan(plan)


@pytest.mark.parametrize("selected", [False, True])
def test_assignment_identity_cannot_reuse_stale_complete_metadata_artifact(selected):
    plan = _plan()
    index = 1 if selected else 0
    if not selected:
        plan["assignments"][0]["engine"] = "pytorch"
    layout = pipeline_layout_from_capacity_plan(plan, node_metadata={
        plan["assignments"][index]["node_id"]: {
            "artifact": {"model_sha256": "c" * 64,
                         "artifact_kind": "safetensors_assignment" if selected else "gguf_segment",
                         "artifact_sha256": "d" * 64},
            "artifact_sha256": "d" * 64,
        },
    })
    node = layout.nodes[index]
    assert node.artifact.model_sha256 == plan["model_sha256"]
    assert node.artifact.artifact_kind == ("gguf_segment" if selected else "safetensors_assignment")
    assert node.artifact.artifact_sha256 == ("b" * 64 if selected else "")
