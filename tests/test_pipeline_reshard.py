"""Automatic layer recovery tests for CORE-RESHARD-01."""

from src.pipeline_node_contract import (
    PipelineArtifactRef,
    PipelineNode,
    PipelineNodeCapacity,
    validate_pipeline_nodes,
)
from src.pipeline_reshard import (
    PipelineArtifactAvailability,
    PipelineArtifactRequirement,
    PipelineReshardCoordinator,
    plan_pipeline_reshard,
)


MODEL_SHA = "a" * 64


def _descriptor(total_layers=6):
    return {
        "model_id": "reshard-fixture",
        "model_type": "qwen3",
        "model_sha256": MODEL_SHA,
        "total_layers": total_layers,
        "layer_weight_bytes": [100] * total_layers,
        "component_weight_bytes": {
            "embedding": 10,
            "final_norm": 10,
            "lm_head": 10,
            "visual": 0,
            "mtp": 0,
            "multimodal": 0,
            "other": 0,
        },
    }


def _capacity_record(node_id, capacity=360, *, role="client", score=10, executor_id=""):
    item = {
        "node_id": node_id,
        "role": role,
        "capacity_bytes": capacity,
        "reserve_bytes": 0,
        "runtime_multiplier": 1.0,
        "execution_device": "cpu",
        "capacity_source": "recovery-test",
        "score": score,
    }
    # 只有**附加**执行器带 `executor_id`（主执行器记录没有该键），与
    # `scheduler._get_pipeline_capacity_nodes` 的产出形态一致。
    if executor_id:
        item["executor_id"] = executor_id
    return item


def _node(
    node_id, start, end, *, embedding=False, head=False, executor_id="",
    kind=None, engine="pytorch", artifact_kind="safetensors_assignment",
    handoff_at=None,
):
    is_local = node_id == "master"
    return PipelineNode(
        node_id=node_id,
        kind=kind or ("local" if is_local else "remote_pipeline"),
        layer_range=(start, end),
        engine=engine,
        location="local" if is_local else f"node:{node_id}",
        capacity=PipelineNodeCapacity(
            capacity_bytes=260,
            required_bytes=220,
            execution_device="cpu",
            capacity_source="initial-test",
        ),
        artifact=PipelineArtifactRef(
            model_sha256=MODEL_SHA,
            artifact_kind=artifact_kind,
            architecture="qwen3",
            source_layer_range=(start, end),
        ),
        has_embedding=embedding,
        has_lm_head=head,
        federated=not is_local,
        executor_id=executor_id,
        handoff_at=handoff_at,
    )


def _layout():
    return validate_pipeline_nodes([
        _node("master", 0, 2, embedding=True),
        _node("worker-a", 2, 4),
        _node("worker-b", 4, 6, head=True),
    ], total_layers=6)


def _survivors(master_capacity=360, worker_capacity=360):
    return [
        _capacity_record(
            "master", master_capacity, role="master", score=20,
        ),
        _capacity_record("worker-b", worker_capacity, score=10),
    ]


def _availability_for(plan):
    return [
        PipelineArtifactAvailability(
            node_id=node.node_id,
            model_sha256=MODEL_SHA,
            artifact_kind="safetensors_assignment",
            layer_range=node.layer_range,
            has_embedding=node.has_embedding,
            has_lm_head=node.has_lm_head,
            # 身份是 `(node_id, executor_id)`；漏掉它会让同机附加执行器的
            # requirement 永远等不到证明（A3）。
            executor_id=node.executor_id,
        )
        for node in plan.candidate_layout.nodes
    ]


def test_failure_reassigns_exact_ranges_only_after_capacity_preflight():
    decision = plan_pipeline_reshard(
        _layout(),
        base_epoch=1,
        failed_node_ids={"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )

    assert decision.accepted is True
    assert decision.ready_to_commit is False
    assert decision.reason_code == "pipeline_reshard_assets_required"
    assert [node.node_id for node in decision.plan.candidate_layout.nodes] == [
        "master", "worker-b",
    ]
    assert [node.layer_range for node in decision.plan.candidate_layout.nodes] == [
        (0, 3), (3, 6),
    ]
    assert any(
        move.source_node_id == "worker-a" for move in decision.plan.moves
    )
    assert decision.plan.candidate_layout.total_layers == 6


def test_insufficient_survivor_capacity_rejects_without_partial_layout():
    decision = plan_pipeline_reshard(
        _layout(),
        base_epoch=1,
        failed_node_ids={"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(250, 250),
        safety_margin=1.0,
    )

    assert decision.accepted is False
    assert decision.reason_code == "pipeline_distributed_capacity_insufficient"
    assert decision.plan is None


def test_spare_node_can_join_recovery_but_must_prove_its_artifact():
    decision = plan_pipeline_reshard(
        validate_pipeline_nodes([
            _node("master", 0, 3, embedding=True),
            _node("worker-a", 3, 6, head=True),
        ], total_layers=6),
        base_epoch=1,
        failed_node_ids={"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=[
            _capacity_record("master", role="master", score=20),
            _capacity_record("spare", score=5),
        ],
        node_metadata={
            "spare": {
                "location": "node:spare",
                "kind": "remote_pipeline",
                "engine": "pytorch",
                "federated": True,
                "artifact_kind": "safetensors_assignment",
                "architecture": "qwen3",
            },
        },
        safety_margin=1.0,
    )

    assert decision.accepted is True
    assert "spare" in {
        node.node_id for node in decision.plan.candidate_layout.nodes
    }
    assert "spare" in {
        requirement.node_id for requirement in decision.plan.missing_requirements
    }


def test_commit_waits_for_assets_then_atomically_fences_old_epoch():
    coordinator = PipelineReshardCoordinator(_layout())
    old_layout = coordinator.layout
    old_lease = coordinator.topology_lease
    staged = coordinator.stage_failure(
        {"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )

    blocked = coordinator.commit(staged.plan.plan_id, expected_epoch=1)
    assert blocked.accepted is False
    assert blocked.reason_code == "pipeline_reshard_assets_not_ready"
    assert coordinator.layout.contract_sha256 == old_layout.contract_sha256

    for availability in _availability_for(staged.plan):
        coordinator.record_artifact(availability)
    assert coordinator.snapshot()["staged"][0]["assets_ready"] is True

    stale = coordinator.commit(staged.plan.plan_id, expected_epoch=0)
    assert stale.reason_code == "pipeline_reshard_stale_epoch"
    assert coordinator.layout.contract_sha256 == old_layout.contract_sha256

    committed = coordinator.commit(staged.plan.plan_id, expected_epoch=1)
    assert committed.accepted is True
    assert committed.epoch == 2
    assert coordinator.layout.contract_sha256 == staged.plan.candidate_layout.contract_sha256
    assert coordinator.fence(2).accepted is True
    assert coordinator._lease_book.check(old_lease.lease_id, 1).reason == "stale_lease"


def test_wrong_model_or_unverified_artifact_never_opens_commit_gate():
    coordinator = PipelineReshardCoordinator(_layout())
    staged = coordinator.stage_failure(
        {"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )
    for requirement in staged.plan.missing_requirements:
        coordinator.record_artifact(PipelineArtifactAvailability(
            node_id=requirement.node_id,
            model_sha256="b" * 64,
            artifact_kind=requirement.artifact_kind,
            layer_range=requirement.layer_range,
            verified=True,
            has_embedding=requirement.has_embedding,
            has_lm_head=requirement.has_lm_head,
        ))
        coordinator.record_artifact(PipelineArtifactAvailability(
            node_id=requirement.node_id,
            model_sha256=requirement.model_sha256,
            artifact_kind=requirement.artifact_kind,
            layer_range=requirement.layer_range,
            verified=False,
            has_embedding=requirement.has_embedding,
            has_lm_head=requirement.has_lm_head,
        ))

    result = coordinator.commit(staged.plan.plan_id, expected_epoch=1)
    assert result.accepted is False
    assert result.reason_code == "pipeline_reshard_assets_not_ready"


def test_layer_coverage_without_required_boundary_weights_is_not_ready():
    coordinator = PipelineReshardCoordinator(_layout())
    staged = coordinator.stage_failure(
        {"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )
    for requirement in staged.plan.missing_requirements:
        coordinator.record_artifact(PipelineArtifactAvailability(
            node_id=requirement.node_id,
            model_sha256=requirement.model_sha256,
            artifact_kind=requirement.artifact_kind,
            layer_range=requirement.layer_range,
            verified=True,
            has_embedding=False,
            has_lm_head=False,
        ))

    missing = coordinator.snapshot()["staged"][0]["missing_requirements"]
    assert any(item["has_embedding"] or item["has_lm_head"] for item in missing)
    assert coordinator.commit(
        staged.plan.plan_id, expected_epoch=1,
    ).reason_code == "pipeline_reshard_assets_not_ready"


def test_later_failure_supersedes_earlier_plan_in_the_same_epoch():
    coordinator = PipelineReshardCoordinator(_layout())
    first = coordinator.stage_failure(
        {"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )

    second = coordinator.stage_failure(
        {"worker-a", "worker-b"},
        descriptor=_descriptor(),
        capacity_nodes=[
            _capacity_record("master", role="master", score=20),
            _capacity_record("spare", score=5),
        ],
        node_metadata={
            "spare": {
                "location": "node:spare",
                "kind": "remote_pipeline",
                "engine": "pytorch",
                "federated": True,
                "artifact_kind": "safetensors_assignment",
                "architecture": "qwen3",
            },
        },
        safety_margin=1.0,
    )

    assert second.accepted is True
    assert second.plan.plan_id != first.plan.plan_id
    assert coordinator.staged_plan(first.plan.plan_id) is None
    assert coordinator.commit(
        first.plan.plan_id, expected_epoch=1,
    ).reason_code == "pipeline_reshard_unknown_plan"
    assert [item["plan_id"] for item in coordinator.snapshot()["staged"]] == [
        second.plan.plan_id,
    ]


# --------------------------------------------------------------------------
# 2026-10-27：`node_id` 身份残留（审计 A3 / A4 / :446）
# --------------------------------------------------------------------------


def _local_executor_layout(total_layers=32):
    """同一台机器上的两个执行器（主执行器 + CPU 执行器）+ 两个远端 worker。"""
    return validate_pipeline_nodes([
        _node("master", 0, 8, embedding=True),
        _node(
            "master", 8, 12, executor_id="cpu", kind="cross_framework",
            engine="llama_cpp", artifact_kind="gguf_segment", handoff_at=8,
        ),
        _node("worker-a", 12, 20),
        _node("worker-b", 20, total_layers, head=True),
    ], total_layers=total_layers)


def _local_executor_records():
    # 容量必须覆盖 32 层 × 100 字节 + 组件，否则求解器先报容量不足，
    # 测不到"哪个执行器退场"这一层。
    return [
        _capacity_record("master", capacity=100000, role="master", score=20),
        _capacity_record("master", capacity=100000, executor_id="cpu", score=15),
        _capacity_record("worker-a", capacity=100000, score=10),
        _capacity_record("worker-b", capacity=100000, score=10),
    ]


def test_survivor_count_counts_capacity_records_not_distinct_node_ids():
    """:446 —— reason 说的是 "capacity records"，判据就必须是记录数。

    旧写法数的是不同 `node_id` 的个数：只剩 master 的两个执行器时记录数=2、
    node 数=1 ⇒ 误判 `pipeline_reshard_survivors_unavailable`，而这正是
    "同机两段"的目标形态。
    """
    layout = validate_pipeline_nodes([
        _node("master", 0, 8, embedding=True),
        _node(
            "master", 8, 24, executor_id="cpu", kind="cross_framework",
            engine="llama_cpp", artifact_kind="gguf_segment", handoff_at=8,
        ),
        _node("worker", 24, 32, head=True),
    ], total_layers=32)

    decision = plan_pipeline_reshard(
        layout,
        base_epoch=1,
        failed_node_ids={"worker"},
        descriptor=_descriptor(32),
        capacity_nodes=[
            _capacity_record("master", capacity=100000, role="master", score=20),
            _capacity_record(
                "master", capacity=100000, executor_id="cpu", score=15,
            ),
        ],
        safety_margin=1.0,
    )

    assert decision.accepted is True, decision.reason_code
    assert decision.reason_code == "pipeline_reshard_assets_required"
    assert [
        (node.node_id, node.executor_id)
        for node in decision.plan.candidate_layout.nodes
    ] == [("master", ""), ("master", "cpu")]


def test_executor_level_failure_retires_only_that_executor():
    """A4 —— `(node_id, executor_id)` 只退场一个执行器；裸 `node_id` 仍是整机。"""
    layout = _local_executor_layout()

    single = plan_pipeline_reshard(
        layout,
        base_epoch=1,
        failed_node_ids={("master", "cpu")},
        descriptor=_descriptor(32),
        capacity_nodes=_local_executor_records(),
        safety_margin=1.0,
    )
    assert single.accepted is True, single.reason_code
    assert single.plan.failed_node_ids == ()
    assert single.plan.failed_executor_ids == (("master", "cpu"),)
    candidate = {
        (node.node_id, node.executor_id) for node in single.plan.candidate_layout.nodes
    }
    # 健康的主执行器段必须保留 —— 旧契约表达不了"只坏 CPU 执行器"。
    assert ("master", "") in candidate
    assert ("master", "cpu") not in candidate

    whole = plan_pipeline_reshard(
        layout,
        base_epoch=1,
        failed_node_ids={"master"},
        descriptor=_descriptor(32),
        capacity_nodes=_local_executor_records(),
        safety_margin=1.0,
    )
    assert whole.accepted is True, whole.reason_code
    assert whole.plan.failed_node_ids == ("master",)
    assert whole.plan.failed_executor_ids == ()
    # 整机失联时两个执行器都要退场（"剔除执行器"不能被缩成"剔除半台机器"）。
    assert all(
        node.node_id != "master" for node in whole.plan.candidate_layout.nodes
    )

    # `(node_id, "")` 是"**只**主执行器坏"，与裸 `node_id`（整机）刻意不等价：
    # 它必须只退场主执行器，留下 CPU 执行器继续参与候选。
    primary_only = plan_pipeline_reshard(
        layout,
        base_epoch=1,
        failed_node_ids={("master", "")},
        descriptor=_descriptor(32),
        capacity_nodes=_local_executor_records(),
        safety_margin=1.0,
    )
    assert primary_only.accepted is True, primary_only.reason_code
    assert primary_only.plan.failed_node_ids == ()
    assert primary_only.plan.failed_executor_ids == (("master", ""),)
    assert all(
        (node.node_id, node.executor_id) != ("master", "")
        for node in primary_only.plan.candidate_layout.nodes
    )


def test_unknown_executor_failure_is_rejected_before_planning():
    decision = plan_pipeline_reshard(
        _local_executor_layout(),
        base_epoch=1,
        failed_node_ids={("master", "npu")},
        descriptor=_descriptor(32),
        capacity_nodes=_local_executor_records(),
        safety_margin=1.0,
    )

    assert decision.accepted is False
    assert decision.reason_code == "pipeline_reshard_input_invalid"
    assert "master#npu" in decision.reason


def test_whole_model_primary_proof_does_not_cover_the_cpu_executor():
    """A3 —— 主段的整模工件（`source_layer_range=[0,total)`）不得当作 CPU 段的证明。"""
    availability = PipelineArtifactAvailability(
        node_id="master",
        model_sha256=MODEL_SHA,
        artifact_kind="safetensors_assignment",
        layer_range=(0, 32),
        has_embedding=True,
        has_lm_head=True,
        executor_id="",
    )
    cpu_requirement = PipelineArtifactRequirement(
        node_id="master",
        model_sha256=MODEL_SHA,
        artifact_kind="safetensors_assignment",
        layer_range=(8, 24),
        executor_id="cpu",
    )
    primary_requirement = PipelineArtifactRequirement(
        node_id="master",
        model_sha256=MODEL_SHA,
        artifact_kind="safetensors_assignment",
        layer_range=(8, 24),
        executor_id="",
    )

    # 同 node_id、区间包含、kind 相同 —— 旧判据在这里返回 True（伪就绪）。
    assert availability.covers(cpu_requirement) is False
    assert availability.covers(primary_requirement) is True


def test_commit_requires_a_proof_for_each_executor_identity():
    """A3 端到端：主执行器的证明不能替 CPU 段开 commit 门。"""
    layout = validate_pipeline_nodes([
        _node("master", 0, 16, embedding=True),
        _node(
            "master", 16, 24, executor_id="cpu", kind="cross_framework",
            engine="llama_cpp", artifact_kind="gguf_segment", handoff_at=16,
        ),
        _node("worker", 24, 32, head=True),
    ], total_layers=32)
    coordinator = PipelineReshardCoordinator(layout)
    staged = coordinator.stage_failure(
        {"worker"},
        descriptor=_descriptor(32),
        capacity_nodes=[
            _capacity_record("master", capacity=100000, role="master", score=20),
            _capacity_record(
                "master", capacity=100000, executor_id="cpu", score=15,
            ),
        ],
        safety_margin=1.0,
    )
    assert staged.accepted is True, staged.reason_code
    requirements = staged.plan.requirements

    def prove(requirement, *, executor_id):
        coordinator.record_artifact(PipelineArtifactAvailability(
            node_id=requirement.node_id,
            model_sha256=requirement.model_sha256,
            artifact_kind=requirement.artifact_kind,
            layer_range=requirement.layer_range,
            artifact_sha256=requirement.artifact_sha256,
            verified=True,
            has_embedding=requirement.has_embedding,
            has_lm_head=requirement.has_lm_head,
            executor_id=executor_id,
        ))

    # 1) 所有 requirement 都用主执行器身份"证明"一遍：CPU 段的证明必须不被认。
    for requirement in requirements:
        prove(requirement, executor_id="")
    assert coordinator.snapshot()["staged"][0]["assets_ready"] is False
    assert coordinator.commit(
        staged.plan.plan_id, expected_epoch=1,
    ).reason_code == "pipeline_reshard_assets_not_ready"

    # 2) 按**各自的执行器身份**补齐证明后才允许发布。
    for requirement in requirements:
        prove(requirement, executor_id=requirement.executor_id)
    assert coordinator.snapshot()["staged"][0]["assets_ready"] is True
    assert coordinator.commit(
        staged.plan.plan_id, expected_epoch=1,
    ).accepted is True
