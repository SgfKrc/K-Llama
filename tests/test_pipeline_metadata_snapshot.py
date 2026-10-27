from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from pipeline_node_contract import PipelineNodeContractError, pipeline_layout_from_capacity_plan
from scheduler import NodeInfo, NodeState, Scheduler


MODEL_SHA = "a" * 64


def _assignment(node_id, start, end, *, executor_id="", engine=""):
    return {
        "node_id": node_id, "executor_id": executor_id,
        "role": "master" if node_id == "master" and not executor_id else "client",
        "start_layer": start, "end_layer": end, "layers_count": end - start,
        "has_embedding": start == 0, "has_lm_head": end == 6,
        "capacity_bytes": 100000, "required_bytes": 1000,
        **({"engine": engine} if engine else {}),
    }


def _plan():
    return {
        "admitted": True, "plan_id": "metadata-fixture", "total_layers": 6,
        "model_sha256": MODEL_SHA, "model_type": "qwen3",
        "assignments": [
            _assignment("master", 0, 2),
            _assignment("master", 2, 4, executor_id="cpu"),
            _assignment("edge", 4, 5),
            _assignment("lost", 5, 6),
        ],
    }


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setattr("config.LOCAL_CPU_EXECUTOR_ID", "cpu")
    scheduler = Scheduler(host=SimpleNamespace(engine_type="pytorch"))
    scheduler._role_override = "master"
    scheduler.nodes = {
        node_id: NodeInfo(
            node_id=node_id, role="master" if node_id == "master" else "client",
            state=NodeState.ONLINE, device_info={"backend_id": "pytorch"},
        )
        for node_id in ("master", "edge", "lost")
    }
    return scheduler


def test_metadata_distinguishes_local_executors_and_keeps_primary_legacy_alias(scheduler):
    metadata = scheduler._pipeline_node_metadata()

    assert metadata[("master", "")]["engine"] == "pytorch"
    assert metadata[("master", "cpu")]["engine"] == "llama_cpp"
    assert metadata["master"] == metadata[("master", "")]
    assert metadata["edge"] == metadata[("edge", "")]
    for identity in (("master", ""), ("master", "cpu")):
        assert metadata[identity]["is_local"] is True
        assert metadata[identity]["location"] == "local"
        assert metadata[identity]["federated"] is False


@pytest.mark.parametrize("source", ["active", "preparing", "committing", "ready"])
def test_cached_and_transaction_layout_freezes_both_local_executors(scheduler, source):
    plan = _plan()
    if source == "active":
        scheduler._active_pipeline_capacity_plan = deepcopy(plan)
    else:
        scheduler._pipeline_load_transaction = {"phase": source, "plan": deepcopy(plan)}

    public = scheduler.get_pipeline_capacity_plan()
    assert public["admitted"] is True
    primary, cpu, edge, lost = public["pipeline_layout"]["nodes"]
    assert [node["engine"] for node in (primary, cpu, edge, lost)] == [
        "pytorch", "llama_cpp", "pytorch", "pytorch",
    ]
    assert cpu["location"] == "local"
    assert cpu["federated"] is False
    assert cpu["artifact"]["artifact_kind"] == "gguf_segment"
    assert cpu["cross_engine"] is True
    assert cpu["handoff_at"] == 2
    assert edge["handoff_at"] == 4
    if source == "active":
        scheduler._active_pipeline_capacity_plan = deepcopy(public)
    else:
        scheduler._pipeline_load_transaction["plan"] = deepcopy(public)
    scheduler._host.engine_type = "llama_cpp"
    scheduler.nodes.clear()

    rebound = scheduler.get_pipeline_capacity_plan()
    assert rebound["pipeline_layout"] == public["pipeline_layout"]
    scheduler._activate_pipeline_reshard_coordinator(rebound)
    assert scheduler._pipeline_reshard_coordinator.layout.to_dict() == public["pipeline_layout"]


def _stage(scheduler, monkeypatch):
    descriptor = {
        "model_id": "metadata-fixture", "model_type": "qwen3",
        "model_sha256": MODEL_SHA, "total_layers": 6,
        "layer_weight_bytes": [100] * 6,
        "component_weight_bytes": {"embedding": 10, "final_norm": 10, "lm_head": 10},
    }
    records = []
    for node_id, executor_id, start, end, engine in (
        ("master", "", 0, 2, "pytorch"),
        ("master", "cpu", 2, 4, "llama_cpp"),
        ("edge", "", 4, 6, "pytorch"),
    ):
        record = {
            "node_id": node_id, "executor_id": executor_id, "engine": engine,
            "role": "master" if node_id == "master" else "client",
            "capacity_bytes": 100000, "reserve_bytes": 0, "runtime_multiplier": 1.0,
            "score": 10, "execution_device": "cpu" if executor_id else "cuda",
            "capacity_source": "metadata-test", "layer_ranges": [[start, end]],
        }
        if executor_id:
            record["segment_mode"] = "middle"
            record["layer_artifacts"] = [{
                "layer_range": [start, end], "segment_mode": "middle",
                "model_id": "cpu-middle.gguf", "artifact_sha256": "b" * 64,
                "source_model_sha256": MODEL_SHA, "layer_weight_bytes": [100] * 2,
            }]
        records.append(record)
    monkeypatch.setattr("config.PIPELINE_CAPACITY_SAFETY_MARGIN", 1.0)
    monkeypatch.setattr("config.prefer_all_workers_enabled", lambda: True)
    monkeypatch.setattr(scheduler, "_task_worker_full_model_ids", lambda: set())
    monkeypatch.setattr(scheduler, "_get_pipeline_capacity_nodes", lambda ids: deepcopy(records))
    scheduler._host.get_pipeline_descriptor = lambda: descriptor
    scheduler._tcp_server = SimpleNamespace(get_client_ids=lambda: ["edge"])
    initial = scheduler._attach_pipeline_node_contract(_plan())
    assert initial["admitted"] is True
    scheduler._activate_pipeline_reshard_coordinator(initial)
    report = scheduler._stage_pipeline_reshard_after_disconnect("lost")
    assert report["accepted"] is True, report
    coordinator = scheduler._pipeline_reshard_coordinator
    staged = coordinator.staged_plan(report["plan"]["plan_id"])
    assert staged is not None
    return coordinator, staged


@pytest.mark.parametrize("source", ["active", "transaction", "reattach", "bare_ready"])
def test_reshard_stage_ready_commit_uses_one_mixed_engine_contract(scheduler, monkeypatch, source):
    coordinator, staged = _stage(scheduler, monkeypatch)
    before_epoch = coordinator.epoch
    frozen = staged.candidate_layout.to_dict()
    assert staged.capacity_plan["pipeline_layout"] == frozen
    assert [(node.node_id, node.executor_id, node.engine) for node in staged.candidate_layout.nodes] == [
        ("master", "", "pytorch"), ("master", "cpu", "llama_cpp"), ("edge", "", "pytorch"),
    ]
    assert staged.candidate_layout.nodes[1].artifact.artifact_sha256 == "b" * 64
    if source == "active":
        scheduler._active_pipeline_capacity_plan = deepcopy(staged.capacity_plan)
    elif source == "transaction":
        scheduler._pipeline_load_transaction = {
            "phase": "committing", "plan": deepcopy(staged.capacity_plan),
        }
    scheduler._host.engine_type = "llama_cpp"
    scheduler.nodes.clear()
    if source in {"reattach", "bare_ready"}:
        ready = deepcopy(staged.capacity_plan)
        ready.pop("pipeline_layout")
        if source == "reattach":
            ready = scheduler._attach_pipeline_node_contract(ready, capacity_nodes=[
                {"node_id": "master", "engine": "llama_cpp"},
            ])
    else:
        ready = scheduler.get_pipeline_capacity_plan()
    ready["transaction_phase"] = "ready"
    if source != "bare_ready":
        assert ready["pipeline_layout"] == frozen
    commit = coordinator.commit
    committed_snapshots = []

    def commit_after_ready(plan_id, *, expected_epoch):
        snapshot = coordinator.snapshot()
        assert snapshot["status"] == "ready"
        assert snapshot["staged"][0]["missing_requirements"] == []
        committed_snapshots.append(snapshot)
        return commit(plan_id, expected_epoch=expected_epoch)

    monkeypatch.setattr(coordinator, "commit", commit_after_ready)
    assert scheduler._commit_ready_pipeline_reshard(ready) is True
    assert len(committed_snapshots) == 1
    assert coordinator.layout.to_dict() == frozen
    assert coordinator.epoch == before_epoch + 1
    assert coordinator.fence(before_epoch).accepted is False
    assert coordinator.fence(coordinator.epoch).accepted is True
    assert coordinator.snapshot()["staged"] == []


@pytest.mark.parametrize("change", ["engine", "executor", "capacity", "digest", "model", "boundary"])
def test_frozen_layout_does_not_hide_assignment_changes(scheduler, change):
    plan = _plan()
    plan["assignments"][1]["layer_artifact"] = {"artifact_sha256": "b" * 64}
    bound = scheduler._attach_pipeline_node_contract(plan)
    assert bound["admitted"] is True
    changed = deepcopy(bound)
    if change == "engine":
        changed["assignments"][0]["engine"] = "llama_cpp"
    elif change == "executor":
        changed["assignments"][1]["executor_id"] = "other"
    elif change == "capacity":
        changed["assignments"][0]["capacity_bytes"] += 1
    elif change == "digest":
        changed["assignments"][1]["layer_artifact"]["artifact_sha256"] = "c" * 64
    elif change == "model":
        changed["model_sha256"] = "d" * 64
    else:
        changed["assignments"][0]["end_layer"] = 1
        changed["assignments"][1]["start_layer"] = 1

    with pytest.raises(PipelineNodeContractError):
        pipeline_layout_from_capacity_plan(changed)


def test_reshard_commit_still_rejects_a_different_valid_layout(scheduler, monkeypatch):
    coordinator, staged = _stage(scheduler, monkeypatch)
    changed = deepcopy(staged.capacity_plan)
    changed.pop("pipeline_layout", None)
    changed["assignments"][0]["engine"] = "llama_cpp"
    changed["pipeline_layout"] = pipeline_layout_from_capacity_plan(
        changed, node_metadata=scheduler._pipeline_node_metadata(),
    ).to_dict()
    ready = scheduler._attach_pipeline_node_contract(changed)
    assert ready["admitted"] is True
    assert ready["pipeline_layout"]["contract_sha256"] != staged.candidate_layout.contract_sha256

    assert scheduler._commit_ready_pipeline_reshard(ready) is False
    assert coordinator.epoch == staged.base_epoch
    assert coordinator.layout.contract_sha256 == staged.base_contract_sha256


def test_worker_ready_ack_publishes_the_staged_layout_after_local_backend_swap(scheduler, monkeypatch):
    coordinator, staged = _stage(scheduler, monkeypatch)
    scheduler._host.full_chat_execution_lock = threading.RLock()
    calls = []

    def commit_candidate(candidate_id):
        calls.append(candidate_id)
        scheduler._host.engine_type = "llama_cpp"

    scheduler._host.commit_pipeline_candidate = commit_candidate
    monkeypatch.setattr(scheduler, "get_distributed_inference_enabled", lambda: True)
    monkeypatch.setattr(scheduler, "_dispatch_pipeline_model_cleanup", lambda record: None)
    expected = {
        "node_id": "edge", "config_id": "metadata-ready", "generation": 7,
        "assignment_id": "asg_metadata_ready", "phase": "commit",
        "plan_id": staged.capacity_plan["plan_id"], "start_layer": 4, "end_layer": 6,
        "model_sha256": MODEL_SHA, "model_type": "qwen3", "engine": "pytorch",
        "has_embedding": False, "has_lm_head": True,
    }
    scheduler._pipeline_load_transaction = {
        "config_id": "metadata-ready", "generation": 7, "phase": "committing",
        "candidate_id": "metadata-candidate", "plan": deepcopy(staged.capacity_plan),
        "worker_ids": {"edge"}, "ready_nodes": set(), "expected": {"edge": expected},
        "acks": {}, "connection_generations": {"edge": 0},
    }
    assert coordinator.snapshot()["status"] == "staged"
    scheduler._handle_layer_config_ack("edge", {"data": {
        **expected, "status": "ready", "layer_range": [4, 6],
    }})

    transaction = scheduler._pipeline_load_transaction
    assert transaction["phase"] == "ready", transaction
    assert transaction["ready_nodes"] == {"edge"}
    assert calls == ["metadata-candidate"]
    assert scheduler._active_pipeline_capacity_plan["pipeline_layout"] == staged.candidate_layout.to_dict()
    assert coordinator.layout.to_dict() == staged.candidate_layout.to_dict()
    assert coordinator.epoch == staged.base_epoch + 1
    assert coordinator.fence(staged.base_epoch).accepted is False


@pytest.mark.parametrize("change", ["range", "engine", "executor", "artifact", "location", "hash"])
def test_modified_snapshot_payload_cannot_keep_its_old_hash(scheduler, change):
    bound = scheduler._attach_pipeline_node_contract(_plan())
    changed = deepcopy(bound)
    snapshot = changed["pipeline_layout"]
    if change == "range":
        snapshot["nodes"][0]["layer_range"][1] = 1
    elif change == "engine":
        snapshot["nodes"][0]["engine"] = "llama_cpp"
    elif change == "executor":
        snapshot["nodes"][1]["executor_id"] = "other"
    elif change == "artifact":
        snapshot["nodes"][1]["artifact"]["artifact_sha256"] = "c" * 64
    elif change == "location":
        snapshot["nodes"][1]["location"] = "node:master"
    else:
        snapshot["contract_sha256"] = "0" * 64

    with pytest.raises(PipelineNodeContractError):
        pipeline_layout_from_capacity_plan(changed)
    rejected = scheduler._attach_pipeline_node_contract(changed)
    assert rejected["admitted"] is False
    assert rejected["reason_code"] == "pipeline_node_contract_invalid"


@pytest.mark.parametrize("with_snapshot", [False, True])
@pytest.mark.parametrize("change", ["range", "engine", "executor", "artifact"])
def test_staged_snapshot_cannot_authorize_changed_ready_assignments(
    scheduler, monkeypatch, with_snapshot, change,
):
    coordinator, staged = _stage(scheduler, monkeypatch)
    changed = deepcopy(staged.capacity_plan)
    if not with_snapshot:
        changed.pop("pipeline_layout")
    if change == "range":
        changed["assignments"][0]["end_layer"] = 1
        changed["assignments"][1]["start_layer"] = 1
    elif change == "engine":
        changed["assignments"][0]["engine"] = "llama_cpp"
    elif change == "executor":
        changed["assignments"][1]["executor_id"] = "other"
    else:
        changed["assignments"][1]["layer_artifact"]["artifact_sha256"] = "c" * 64
    before = coordinator.snapshot()

    assert scheduler._attach_pipeline_node_contract(changed)["admitted"] is False
    assert scheduler._commit_ready_pipeline_reshard(changed) is False
    assert coordinator.snapshot() == before
