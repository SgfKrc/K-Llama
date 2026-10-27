import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity


MIB = 1024 * 1024


def descriptor(layer_sizes=(100, 100, 100, 100)):
    return {
        "model_id": "synthetic-qwen2",
        "model_type": "qwen2",
        "model_sha256": "a" * 64,
        "total_layers": len(layer_sizes),
        "layer_weight_bytes": [value * MIB for value in layer_sizes],
        "component_weight_bytes": {
            "embedding": 40 * MIB,
            "final_norm": 5 * MIB,
            "lm_head": 40 * MIB,
            "visual": 0,
            "mtp": 0,
            "other": 0,
        },
    }


def node(node_id, capacity_mb, *, role="client", score=10):
    return {
        "node_id": node_id,
        "role": role,
        "capacity_bytes": capacity_mb * MIB,
        "reserve_bytes": 10 * MIB,
        "runtime_multiplier": 1.0,
        "score": score,
        "execution_device": "cuda",
        "capacity_source": "test",
    }


@pytest.mark.parametrize("executor_score,executor_capacity", [(20, 500), (10, 800)])
@pytest.mark.parametrize("executor_mode", [None, "middle"])
def test_primary_precedes_additional_master_executor(executor_score, executor_capacity, executor_mode):
    primary = {**node("master", 500, role="master"), "candidate_layer_ranges": [[0, 1]]}
    additional = {
        **node("master", executor_capacity, role="master", score=executor_score),
        "executor_id": "cpu", "layer_bounds": [1, 3], "candidate_layer_ranges": [[1, 3]],
    }
    if executor_mode is not None:
        additional["segment_mode"] = executor_mode
    tail = {**node("tail", 500), "candidate_layer_ranges": [[3, 4]]}
    plan = solve_pipeline_capacity(
        descriptor(), [additional, primary, tail], safety_margin=1.0,
        require_distributed=True, prefer_all_workers=True, required_node_id="master",
    )
    assert plan["admitted"] is True
    assert [(item["node_id"], item.get("executor_id", ""), item["start_layer"], item["end_layer"])
            for item in plan["assignments"]] == [
                ("master", "", 0, 1), ("master", "cpu", 1, 3), ("tail", "", 3, 4),
            ]


def test_partial_master_merges_local_worker_bytes_without_full_model():
    partial = descriptor(layer_sizes=(100, 0, 0, 0))
    partial["source_model_sha256"] = "a" * 64
    workers = [
        {**node("master", 500, role="master"), "layer_ranges": [[0, 1]]},
        {
            **node("middle", 500),
            "layer_ranges": [[1, 3]],
            "layer_artifacts": [{
                "layer_range": [1, 3], "segment_mode": "middle",
                "model_id": "middle.gguf", "artifact_sha256": "b" * 64,
                "source_model_sha256": "a" * 64,
                "layer_weight_bytes": [100 * MIB, 100 * MIB],
            }],
        },
        {
            **node("tail", 500),
            "layer_ranges": [[3, 4]],
            "layer_artifacts": [{
                "layer_range": [3, 4], "segment_mode": "tail",
                "model_id": "tail.gguf", "artifact_sha256": "c" * 64,
                "source_model_sha256": "a" * 64,
                "layer_weight_bytes": [100 * MIB],
            }],
        },
    ]
    plan = solve_pipeline_capacity(
        partial, workers, safety_margin=1.0,
        require_distributed=True, prefer_all_workers=True,
    )
    assert plan["admitted"] is True
    assert [(item["node_id"], item["start_layer"], item["end_layer"])
            for item in plan["assignments"]] == [
                ("master", 0, 1), ("middle", 1, 3), ("tail", 3, 4),
            ]
    assert partial["layer_weight_bytes"] == [100 * MIB, 0, 0, 0]

    workers[2]["layer_artifacts"][0].pop("layer_weight_bytes")
    with pytest.raises(PipelineCapacityError, match="every decoder layer"):
        solve_pipeline_capacity(partial, workers, safety_margin=1.0)


def test_partial_bytes_require_a_matching_source():
    partial = descriptor(layer_sizes=(100, 0))
    master = {**node("master", 200, role="master"), "layer_ranges": [[0, 1]]}
    tail = {**node("tail", 500), "layer_ranges": [[1, 2]], "layer_artifacts": [{
        "layer_range": [1, 2], "segment_mode": "tail", "model_id": "tail.gguf",
        "artifact_sha256": "b" * 64, "source_model_sha256": "c" * 64,
        "layer_weight_bytes": [1 * MIB],
    }]}
    with pytest.raises(PipelineCapacityError, match="source_model_sha256"):
        solve_pipeline_capacity(partial, [master, tail], safety_margin=1.0)
    partial["source_model_sha256"] = "a" * 64
    with pytest.raises(PipelineCapacityError, match="every decoder layer"):
        solve_pipeline_capacity(partial, [master, tail], safety_margin=1.0)

    correct = {**node("correct", 400, score=50), "layer_ranges": [[1, 2]],
               "layer_artifacts": [{**tail["layer_artifacts"][0],
                                    "model_id": "correct.gguf",
                                    "artifact_sha256": "d" * 64,
                                    "source_model_sha256": "a" * 64,
                                    "layer_weight_bytes": [200 * MIB]}]}
    plan = solve_pipeline_capacity(partial, [master, tail, correct],
                                   safety_margin=1.0, require_distributed=True)
    assert plan["admitted"] is True
    assert plan["assignments"][-1]["node_id"] == "correct"
    assert plan["assignments"][-1]["raw_weight_bytes"] == 245 * MIB


def test_overlap_charges_selected_artifact_instead_of_first_quote():
    partial = descriptor(layer_sizes=(100, 0, 0, 0))
    partial["source_model_sha256"] = "a" * 64
    middle = {
        "layer_range": [1, 3], "segment_mode": "middle",
        "model_id": "middle.gguf", "artifact_sha256": "b" * 64,
        "source_model_sha256": "a" * 64,
    }
    cheap = {
        **node("cheap", 120, score=11), "layer_ranges": [[1, 3]],
        "layer_artifacts": [{**middle, "layer_weight_bytes": [40 * MIB] * 2}],
    }
    expensive = {
        **node("expensive", 400, score=50), "layer_ranges": [[1, 3]],
        "layer_artifacts": [{
            **middle, "model_id": "expensive.gguf", "artifact_sha256": "c" * 64,
            "layer_weight_bytes": [150 * MIB] * 2,
        }],
    }
    tail = {
        **node("tail", 200), "layer_ranges": [[3, 4]],
        "layer_artifacts": [{
            "layer_range": [3, 4], "segment_mode": "tail",
            "model_id": "tail.gguf", "artifact_sha256": "d" * 64,
            "source_model_sha256": "a" * 64,
            "layer_weight_bytes": [100 * MIB],
        }],
    }
    master = {**node("master", 200, role="master"), "layer_ranges": [[0, 1]]}
    for workers in ([cheap, expensive], [expensive, cheap]):
        plan = solve_pipeline_capacity(
            partial, [master, *workers, tail], safety_margin=1.0,
            require_distributed=True,
        )
        assert plan["admitted"] is True
        chosen = next(item for item in plan["assignments"] if item["node_id"] == "expensive")
        assert chosen["raw_weight_bytes"] == 305 * MIB
        assert chosen["required_bytes"] == 315 * MIB
        assert plan["raw_model_bytes"] == 585 * MIB

    expensive["capacity_bytes"] = 250 * MIB
    cheaper_plan = solve_pipeline_capacity(
        partial, [master, cheap, expensive, tail], safety_margin=1.0,
        require_distributed=True,
    )
    assert cheaper_plan["admitted"] is True
    cheap_assignment = next(item for item in cheaper_plan["assignments"]
                            if item["node_id"] == "cheap")
    assert cheap_assignment["raw_weight_bytes"] == 85 * MIB
    assert cheaper_plan["raw_model_bytes"] == 365 * MIB

    cheap["capacity_bytes"] = 50 * MIB
    rejected = solve_pipeline_capacity(
        partial, [master, cheap, expensive, tail], safety_margin=1.0,
        require_distributed=True,
    )
    assert rejected["admitted"] is False


def test_same_artifact_digest_with_conflicting_quotes_is_rejected():
    partial = descriptor(layer_sizes=(100, 0))
    partial["source_model_sha256"] = "a" * 64
    artifact = {
        "layer_range": [1, 2], "segment_mode": "tail",
        "model_id": "tail.gguf", "artifact_sha256": "b" * 64,
        "source_model_sha256": "a" * 64,
    }
    workers = [
        {**node("first", 500), "layer_ranges": [[1, 2]],
         "layer_artifacts": [{**artifact, "layer_weight_bytes": [100 * MIB]}]},
        {**node("second", 500), "layer_ranges": [[1, 2]],
         "layer_artifacts": [{**artifact, "layer_weight_bytes": [200 * MIB]}]},
    ]
    with pytest.raises(PipelineCapacityError, match="conflicting layer_weight_bytes"):
        solve_pipeline_capacity(partial, workers, safety_margin=1.0)


def test_partial_bytes_cannot_be_borrowed_by_an_unpriced_artifact():
    partial = descriptor(layer_sizes=(100, 0))
    partial["source_model_sha256"] = "a" * 64
    artifact = {
        "layer_range": [1, 2], "segment_mode": "tail",
        "model_id": "tail.gguf", "artifact_sha256": "b" * 64,
        "source_model_sha256": "a" * 64,
        "layer_weight_bytes": [100 * MIB],
    }
    lender = {**node("lender", 50), "layer_ranges": [[1, 2]],
              "layer_artifacts": [artifact]}
    selected = {**node("selected", 300), "layer_ranges": [[1, 2]],
                "layer_artifacts": [{key: value for key, value in artifact.items()
                                     if key != "layer_weight_bytes"}]}
    plan = solve_pipeline_capacity(
        partial, [node("master", 200, role="master"), lender, selected],
        safety_margin=1.0, require_distributed=True,
    )
    assert plan["admitted"] is False
    lender["capacity_bytes"] = 5 * MIB
    with pytest.raises(PipelineCapacityError, match="every decoder layer"):
        solve_pipeline_capacity(
            partial, [node("master", 200, role="master"), lender, selected],
            safety_margin=1.0, require_distributed=True,
        )


def test_complete_descriptor_charges_selected_cross_format_artifact():
    complete = descriptor(layer_sizes=(100, 100))
    complete["source_model_sha256"] = "a" * 64
    worker = {
        **node("tail", 300), "layer_ranges": [[1, 2]],
        "layer_artifacts": [{
            "layer_range": [1, 2], "segment_mode": "tail",
            "model_id": "tail.gguf", "artifact_sha256": "b" * 64,
            "source_model_sha256": "a" * 64,
            "layer_weight_bytes": [200 * MIB],
        }],
    }
    master = {**node("master", 200, role="master"), "layer_ranges": [[0, 1]]}
    plan = solve_pipeline_capacity(
        complete, [master, worker], safety_margin=1.0,
        require_distributed=True,
    )
    assert plan["admitted"] is True
    assert plan["assignments"][-1]["required_bytes"] == 255 * MIB
    assert plan["raw_model_bytes"] == 385 * MIB
    worker["capacity_bytes"] = 200 * MIB
    rejected = solve_pipeline_capacity(
        complete, [master, worker], safety_margin=1.0,
        require_distributed=True,
    )
    assert rejected["admitted"] is False

    worker["capacity_bytes"] = 300 * MIB
    worker["layer_artifacts"][0].pop("layer_weight_bytes")
    unpriced = solve_pipeline_capacity(
        complete, [master, worker], safety_margin=1.0,
        require_distributed=True,
    )
    assert unpriced["admitted"] is False
    worker["layer_artifacts"][0]["layer_weight_bytes"] = [100 * MIB]
    worker["layer_artifacts"][0]["source_model_sha256"] = "c" * 64
    foreign = solve_pipeline_capacity(
        complete, [master, worker], safety_margin=1.0,
        require_distributed=True,
    )
    assert foreign["admitted"] is False


def test_declared_layer_budget_caps_that_node_layers():
    """设备自荐的层数上限必须被分配器尊重 —— 它是"本地裁层后能承载的上限"，
    不是"当前工件已就绪的区间"，所以分配器可以给它任意连续区间，但不能超过层数。"""
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_budget": {
                    "available_bytes": 500 * MIB,
                    "per_layer_bytes": 100 * MIB,
                    "max_layers": 2,
                    "local_cut": True,
                },
            },
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    by_node = {item["node_id"]: item for item in plan["assignments"]}
    assert by_node["worker-a"]["layers_count"] == 2
    assert sum(item["layers_count"] for item in plan["assignments"]) == 4


def test_layer_budget_rejects_malformed_payload():
    bad_budgets = (
        {"available_bytes": 1 * MIB, "per_layer_bytes": 0, "max_layers": 2},
        {"available_bytes": 1 * MIB, "per_layer_bytes": 1 * MIB, "max_layers": 0},
        {"available_bytes": 1 * MIB, "per_layer_bytes": 1 * MIB},
        {
            "available_bytes": 1 * MIB,
            "per_layer_bytes": 1 * MIB,
            "max_layers": 2,
            "local_cut": "yes",
        },
    )
    for bad in bad_budgets:
        with pytest.raises(PipelineCapacityError):
            solve_pipeline_capacity(
                descriptor(),
                [
                    {**node("worker-a", 500), "layer_budget": bad},
                    node("worker-b", 400),
                ],
                safety_margin=1.0,
            )


def test_layer_budget_supersedes_fixed_layer_ranges():
    """申报了 `layer_budget` 的节点：`layer_ranges` 只代表"当前已就绪"，
    不再作为分配硬约束 —— 否则设备永远被它预置的那一段钉死。"""
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_ranges": [[0, 1]],
                "layer_artifacts": [{
                    "layer_range": [0, 1], "segment_mode": "head",
                    "model_id": "stale-cache.gguf", "artifact_sha256": "b" * 64,
                }],
                "layer_budget": {
                    "available_bytes": 500 * MIB,
                    "per_layer_bytes": 100 * MIB,
                    "max_layers": 3,
                    "local_cut": True,
                },
            },
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    by_node = {item["node_id"]: item for item in plan["assignments"]}
    # Equal capability balances decoder work; local_cut still unlocks the old [0,1).
    assert by_node["worker-a"]["layers_count"] == 2
    assert by_node["worker-a"]["start_layer"] == 0
    assert by_node["worker-a"]["layers_count"] <= 3


def test_layer_budget_without_local_cut_keeps_fixed_ranges():
    """设备没声明"能本地裁层"时，`layer_ranges` 仍是硬约束 —— 否则会派给它一个
    手上没有工件的区间，请求必然失败。"""
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_ranges": [[0, 2]],
                "layer_budget": {
                    "available_bytes": 500 * MIB,
                    "per_layer_bytes": 100 * MIB,
                    "max_layers": 3,
                    "local_cut": False,
                },
            },
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )

    by_node = {item["node_id"]: item for item in plan["assignments"]}
    assert by_node["worker-a"]["start_layer"] == 0
    assert by_node["worker-a"]["end_layer"] == 2


def test_fixed_artifact_range_cannot_be_sliced_by_prefer_all_workers():
    """A ready GGUF artifact is executable only at its exact manifest range."""
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {**node("master", 180, role="master"), "layer_ranges": [[0, 1]]},
            {
                **node("middle", 500),
                "layer_ranges": [[1, 4]],
            },
            {
                **node("tail", 500),
                "layer_ranges": [[3, 4]],
            },
        ],
        safety_margin=1.0,
        require_distributed=True,
        prefer_all_workers=True,
    )

    assert plan["admitted"] is True
    by_node = {item["node_id"]: item for item in plan["assignments"]}
    assert (by_node["middle"]["start_layer"], by_node["middle"]["end_layer"]) == (1, 4)
    assert "tail" in plan["control_only_nodes"]


def test_middle_segment_artifact_cannot_take_the_tail_slot():
    """★ DIST-3（2026-10-05 三机实测）：中间段工件不能接末段。

    `layer_ranges` 只声明"本节点覆盖哪些层"，**不区分工件含不含
    `embedding` / `lm_head`**。Y700 广告 `[0, 3]`（实为 `mid8-24`，
    `mode=middle`）时，旧判据认为末段落在区间内就照分 —— 而中间段工件没有
    `lm_head` / `final_norm`，请求必然失败（实测报
    `remote worker reported a Stage error`）。声明的 `segment_mode` 必须让求解器
    拒绝这种分配。
    """
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_ranges": [[0, 4]],
                "segment_mode": "middle",
            },
        ],
        safety_margin=1.0,
    )

    assert plan["admitted"] is False
    assert plan["reason_code"] == "pipeline_segment_contract_unsatisfied"


def test_per_artifact_modes_select_the_matching_range_and_identity():
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {**node("master", 180, role="master"), "layer_ranges": [[0, 1]]},
            {
                **node("edge", 500),
                "layer_ranges": [[1, 3], [3, 4]],
                "layer_artifacts": [
                    {
                        "layer_range": [1, 3],
                        "segment_mode": "middle",
                        "model_id": "middle.gguf",
                        "artifact_sha256": "a" * 64,
                        "source_model_sha256": "c" * 64,
                    },
                    {
                        "layer_range": [3, 4],
                        "segment_mode": "tail",
                        "model_id": "tail.gguf",
                        "artifact_sha256": "b" * 64,
                        "source_model_sha256": "c" * 64,
                    },
                ],
            },
            {
                **node("tail", 500),
                "layer_ranges": [[3, 4]],
                "segment_mode": "tail",
            },
        ],
        safety_margin=1.0,
        require_distributed=True,
        prefer_all_workers=True,
    )

    assert plan["admitted"] is True
    edge = next(item for item in plan["assignments"] if item["node_id"] == "edge")
    assert edge["layer_artifact"]["model_id"] == "middle.gguf"
    assert edge["layer_artifact"]["segment_mode"] == "middle"


def test_per_artifact_source_digest_mismatch_is_rejected():
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("middle", 500),
                "layer_ranges": [[0, 2]],
                "layer_artifacts": [{
                    "layer_range": [0, 2],
                    "segment_mode": "head",
                    "model_id": "head.gguf",
                    "artifact_sha256": "a" * 64,
                    "source_model_sha256": "c" * 64,
                }],
            },
            {
                **node("tail", 500),
                "layer_ranges": [[2, 4]],
                "layer_artifacts": [{
                    "layer_range": [2, 4],
                    "segment_mode": "tail",
                    "model_id": "tail.gguf",
                    "artifact_sha256": "b" * 64,
                    "source_model_sha256": "d" * 64,
                }],
            },
        ],
        safety_margin=1.0,
        require_distributed=True,
    )

    assert plan["admitted"] is False
    assert plan["reason_code"] == "pipeline_segment_contract_unsatisfied"


def test_invalid_segment_mode_fails_closed():
    with pytest.raises(PipelineCapacityError):
        solve_pipeline_capacity(
            descriptor(),
            [{**node("worker", 500), "segment_mode": "middle-ish"}],
            safety_margin=1.0,
        )


def test_absent_segment_declaration_keeps_legacy_behaviour():
    """未声明 `segment_mode` 时**不得新增任何约束**（旧设备 / 旧 manifest 不变）。

    取形沿用 `test_layer_budget_without_local_cut_keeps_fixed_ranges`：`worker-a`
    只声明 `[0, 2]`，`worker-b` 不声明区间。这条同时是段类型约束的**反证** ——
    若约束无条件生效（把缺失当 `middle` 处理），`worker-a` 拿首段就会被拒。
    """
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {**node("worker-a", 500), "layer_ranges": [[0, 2]]},
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    by_node = {item["node_id"]: item for item in plan["assignments"]}
    assert by_node["worker-a"]["start_layer"] == 0

    # 同一拓扑、但显式声明 `middle` ⇒ 首段必须被拒：段类型约束确实生效。
    rejected = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_ranges": [[0, 2]],
                "segment_mode": "middle",
            },
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )
    assert rejected["admitted"] is False


def test_aggregate_capacity_admits_when_no_single_node_fits():
    plan = solve_pipeline_capacity(
        descriptor(),
        [node("master", 80, role="master", score=100), node("worker-a", 300), node("worker-b", 300)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    assert plan["aggregate_only"] is True
    assert plan["single_node_full_model_candidates"] == []
    assert sum(item["layers_count"] for item in plan["assignments"]) == 4
    assert plan["assignments"][0]["has_embedding"] is True
    assert plan["assignments"][-1]["has_lm_head"] is True
    assert "master" in plan["control_only_nodes"]


# ── ★ #31 M3：被运行时忽略的分量可以不计入容量账（只有**显式声明**的架构才行）──────


def test_hybrid_visual_and_mtp_no_longer_block_capacity() -> None:
    """★ M3 的核心：`qwen3_5` 声明了 visual / mtp 不参与层执行 ⇒ 容量账**放行**。

    修复前这里必然 `PipelineCapacityError`（"separately placeable components without a
    runtime plan"）⇒ hybrid 在层流水线上完全不可用。
    """
    model = descriptor()
    model["model_type"] = "qwen3_5"
    model["component_weight_bytes"]["visual"] = 600 * MIB
    model["component_weight_bytes"]["mtp"] = 100 * MIB
    model["runtime_ignored_components"] = ["visual", "mtp"]
    model["runtime_ignored_component_bytes"] = {"visual": 600 * MIB, "mtp": 100 * MIB}

    plan = solve_pipeline_capacity(
        model,
        [node("master", 300, role="master", score=100), node("worker", 300)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True


def test_unignored_component_still_blocks_capacity() -> None:
    """★ 闸门**没被削弱**：只声明了 visual / mtp，但 `multimodal` 也非零 ⇒ 仍必须拒绝。"""
    model = descriptor()
    model["component_weight_bytes"]["visual"] = 600 * MIB
    model["component_weight_bytes"]["multimodal"] = 1 * MIB
    model["runtime_ignored_components"] = ["visual", "mtp"]

    with pytest.raises(PipelineCapacityError, match="without a runtime plan"):
        solve_pipeline_capacity(
            model, [node("master", 300, role="master", score=100)], safety_margin=1.0
        )


def test_unknown_ignored_component_is_rejected() -> None:
    """★ 标记里写错名字必须报错 —— 否则一个笔误就等于**悄悄放行**了一个分量。"""
    model = descriptor()
    model["runtime_ignored_components"] = ["visualx"]

    with pytest.raises(PipelineCapacityError, match="unknown entries"):
        solve_pipeline_capacity(
            model, [node("master", 300, role="master", score=100)], safety_margin=1.0
        )


def test_descriptor_without_the_marker_behaves_as_before() -> None:
    """★ 兼容性：**没有**这个字段的 descriptor（老工件 / 其它架构）行为与修复前一致。"""
    model = descriptor()          # 不带 runtime_ignored_components
    assert "runtime_ignored_components" not in model

    plan = solve_pipeline_capacity(
        model,
        [node("master", 300, role="master", score=100), node("worker", 300)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True


def test_required_distributed_rejects_single_node_shortcut():
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            node("master", 600, role="master", score=100),
            node("worker-a", 300, score=10),
        ],
        safety_margin=1.0,
        require_distributed=True,
    )

    assert plan["admitted"] is True
    assert plan["reason_code"] == "distributed_forced"
    assert plan["require_distributed"] is True
    assert plan["participating_node_count"] == 2
    assert {item["node_id"] for item in plan["assignments"]} == {
        "master", "worker-a",
    }


def test_relay_exempt_assignment_counts_as_participating_node():
    relay = node("relay", 1)
    relay["capacity_source"] = "relay_exempt"
    plan = solve_pipeline_capacity(
        descriptor(),
        [node("master", 600, role="master", score=100), relay],
        safety_margin=1.0,
        require_distributed=True,
        local_layer_budget=4,
        relay_claims={"relay": (4, 4)},
    )

    assert plan["admitted"] is True
    assert plan["participating_node_count"] == 2
    assert plan["aggregate_only"] is True


def test_capacity_failure_returns_no_partial_assignment():
    plan = solve_pipeline_capacity(
        descriptor(),
        [node("master", 180, role="master"), node("worker", 180)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is False
    assert plan["reason_code"] == "pipeline_cluster_capacity_insufficient"
    assert plan["assignments"] == []


def test_cpu_runtime_multiplier_is_charged_to_required_bytes():
    cuda = node("cuda", 600)
    cpu = node("cpu", 600)
    cpu["runtime_multiplier"] = 2.0
    plan = solve_pipeline_capacity(descriptor((100,)), [cuda, cpu], safety_margin=1.0)

    assert plan["admitted"] is True
    assert plan["assignments"][0]["node_id"] == "cuda"
    assert plan["assignments"][0]["required_bytes"] == 195 * MIB


@pytest.mark.parametrize("component", ["visual", "mtp", "multimodal"])
def test_component_that_requires_an_unimplemented_runtime_is_rejected(component):
    item = descriptor((100,))
    item["component_weight_bytes"][component] = 20 * MIB

    with pytest.raises(PipelineCapacityError, match=component):
        solve_pipeline_capacity(item, [node("worker", 500)])


def test_capacity_plan_id_is_stable_for_same_inputs():
    nodes = [node("worker-a", 300), node("worker-b", 300)]
    first = solve_pipeline_capacity(descriptor(), nodes, safety_margin=1.0)
    second = solve_pipeline_capacity(descriptor(), list(reversed(nodes)), safety_margin=1.0)

    assert first["plan_id"] == second["plan_id"]


def test_advertised_layer_ranges_constrain_worker_assignment():
    master = node("master", 160, role="master", score=100)
    android = node("android", 400, score=1)
    android["layer_ranges"] = [[1, 4]]

    plan = solve_pipeline_capacity(
        descriptor(), [master, android], safety_margin=1.0,
        require_distributed=True,
    )

    assert plan["admitted"] is True
    assert [
        (item["node_id"], item["start_layer"], item["end_layer"])
        for item in plan["assignments"]
    ] == [("master", 0, 1), ("android", 1, 4)]
    assert plan["assignments"][1]["layer_ranges"] == [[1, 4]]


def test_advertised_layer_ranges_reject_uncovered_cursor():
    master = node("master", 160, role="master", score=100)
    android = node("android", 400, score=1)
    android["layer_ranges"] = [[2, 4]]

    plan = solve_pipeline_capacity(
        descriptor(), [master, android], safety_margin=1.0,
        require_distributed=True,
    )

    assert plan["admitted"] is False
    assert plan["reason_code"] == "pipeline_layer_range_coverage_insufficient"
    assert plan["assignments"] == []
    # ★ 2026-10-08（真机复测）：拒绝时要把**未覆盖区间**一并给出 —— 现场多次把本码误读成
    #   "工件区间配置错"，真因往往是 worker 掉线/未准入（Y700 每 4–16 分钟重建一次）。
    #   这里 `android` 只声明 `[2,4)`，`[0,2)` 无人覆盖（master 是整模、无区间声明）。
    assert plan["uncovered_layer_ranges"] == [[0, 2]]
    assert "未覆盖区间: [0,2)" in plan["reason"]
    assert "检查该节点是否掉线/未准入" in plan["reason"]


def test_uncovered_layer_ranges_helper_reports_head_gap_and_tail():
    from pipeline_capacity import _uncovered_layer_ranges

    # 全覆盖 / 空声明 ⇒ 无缺口（空声明表示"未约束"，不是"不允许任何区间"）。
    assert _uncovered_layer_ranges([(0, 24)], 24) == []
    assert _uncovered_layer_ranges([], 24) == [(0, 24)]
    # 缺中间段 + 缺尾段 ⇒ 两段缺口（按起点升序）。
    assert _uncovered_layer_ranges([(0, 16), (20, 24)], 24) == [(16, 20)]
    assert _uncovered_layer_ranges([(0, 16)], 24) == [(16, 24)]
    # 区间重叠/乱序也按并集处理。
    assert _uncovered_layer_ranges([(16, 20), (0, 16)], 24) == [(20, 24)]
    # 超出 total_layers 的声明不制造负缺口。
    assert _uncovered_layer_ranges([(0, 32)], 24) == []


def test_tied_embedding_is_charged_to_output_capacity():
    item = descriptor((100,))
    item["tie_word_embeddings"] = True
    item["component_weight_bytes"]["lm_head"] = 0
    plan = solve_pipeline_capacity(
        item,
        [node("worker", 250)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    assert plan["raw_model_bytes"] == 185 * MIB
    assert plan["assignments"][0]["raw_weight_bytes"] == 185 * MIB
