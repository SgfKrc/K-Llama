"""B1：附加执行器被陈旧切点挤出时，放开切点锁并按"共存"优先级取优。

★ 2026-10-27（LOCAL-EXECUTORS-01）。这里覆盖两件事：
  1. `executor_coexistence` 的取优优先级链（admitted → 期望附加执行器参与数 →
     参与执行器总数 → 目标函数；打平保留 baseline）；
  2. `scheduler.get_pipeline_capacity_plan` 真的会在"可行但缺少 CPU 段"时重解一次，
     并选中包含 CPU 段的那份方案。
"""
import copy
from types import SimpleNamespace

from executor_coexistence import (
    additional_executor_identities,
    assigned_executor_identities,
    choose_plan,
    missing_additional_executors,
    plan_rank,
)
from scheduler import Scheduler


def _assignment(node_id, executor_id, start, end):
    item = {"node_id": node_id, "start_layer": start, "end_layer": end}
    if executor_id:
        item["executor_id"] = executor_id
    return item


def _plan(assignments, *, admitted=True, objective=None):
    return {
        "admitted": admitted,
        "assignments": assignments,
        "placement_objective": objective or {
            "node_count": len(assignments),
            "weighted_work_cost": 1.0,
            "memory_pressure_cost": 1.0,
            "score_total": 1.0,
        },
    }


def test_missing_additional_executor_ignores_primary_and_zero_layer_records():
    nodes = [
        {"node_id": "master", "executor_id": ""},
        {"node_id": "master", "executor_id": "cpu"},
        {"node_id": "worker"},
    ]
    assert additional_executor_identities(nodes) == {("master", "cpu")}
    plan = _plan([
        _assignment("master", "", 0, 8),
        # 零层的占位/relay 条目不算"参与"：CPU 段并没有真的拿到层。
        _assignment("master", "cpu", 8, 8),
        _assignment("worker", "cpu", 8, 16),
    ])
    assert assigned_executor_identities(plan) == {
        ("master", ""), ("worker", "cpu")}
    # 零层的 CPU 段条目不算参与 ⇒ 仍然被判为"被挤出"。
    assert missing_additional_executors(plan, nodes) == {("master", "cpu")}


def test_missing_additional_executor_detects_the_squeezed_local_executor():
    nodes = [{"node_id": "master", "executor_id": ""},
             {"node_id": "master", "executor_id": "cpu"}]
    plan = _plan([_assignment("master", "", 0, 8),
                  _assignment("worker", "", 8, 16)])
    assert missing_additional_executors(plan, nodes) == {("master", "cpu")}


def test_choose_plan_priority_chain_prefers_more_additional_executors():
    expected = {("master", "cpu")}
    base = _plan([_assignment("master", "", 0, 8),
                  _assignment("worker", "", 8, 16)],
                 objective={"node_count": 2, "weighted_work_cost": 0.1,
                            "memory_pressure_cost": 0.1, "score_total": 99.0})
    released = _plan([_assignment("master", "", 0, 4),
                      _assignment("master", "cpu", 4, 12),
                      _assignment("worker", "", 12, 16)],
                     objective={"node_count": 3, "weighted_work_cost": 9.0,
                                "memory_pressure_cost": 9.0, "score_total": 0.0})
    # 目标函数更差，但"共存"优先：期望的附加执行器参与了，就该选它。
    assert plan_rank(released, expected) < plan_rank(base, expected)
    assert choose_plan(base, released, expected) == (released, "released_better")


def test_choose_plan_prefers_more_participating_executors_before_objective():
    base = _plan([_assignment("master", "", 0, 8),
                  _assignment("worker", "", 8, 16)],
                 objective={"node_count": 2, "weighted_work_cost": 0.1,
                            "memory_pressure_cost": 0.1, "score_total": 99.0})
    released = _plan([_assignment("master", "", 0, 4),
                      _assignment("master", "cpu", 4, 12),
                      _assignment("worker", "", 12, 16)],
                     objective={"node_count": 3, "weighted_work_cost": 9.0,
                                "memory_pressure_cost": 9.0, "score_total": 0.0})
    assert choose_plan(base, released, set()) == (released, "released_better")


def test_choose_plan_keeps_baseline_when_released_is_unavailable_or_tied():
    base = _plan([_assignment("master", "", 0, 8)])
    assert choose_plan(base, None) == (base, "baseline_released_unavailable")
    rejected = _plan([], admitted=False)
    assert choose_plan(base, rejected)[0] is base
    # 完全相同（含目标函数）⇒ 不为了重切而换布局。
    assert choose_plan(base, copy.deepcopy(base))[0] is base
    # 目标函数更好才换。
    better = copy.deepcopy(base)
    better["placement_objective"]["weighted_work_cost"] = 0.0
    assert choose_plan(base, better)[0] is better


def _node(node_id, *, role="client", executor="", **changes):
    item = {"node_id": node_id, "role": role, "capacity_bytes": 10 ** 9,
            "reserve_bytes": 0, "runtime_multiplier": 1.0, "score": 10.0,
            "execution_device": "cuda", "capacity_source": "test"}
    if executor:
        item["executor_id"] = executor
    item.update(changes)
    return item


def _descriptor():
    return {
        "model_id": "fixture", "model_type": "qwen2", "model_sha256": "a" * 64,
        "source_model_sha256": "a" * 64, "total_layers": 8,
        "pipeline_runtime_supported": True,
        "layer_weight_bytes": [100] * 8,
        "component_weight_bytes": {"embedding": 10, "final_norm": 10,
                                   "lm_head": 10, "other": 0},
    }


def test_scheduler_releases_stale_locks_and_plans_the_squeezed_local_executor(monkeypatch):
    """陈旧切点让"不含 CPU 段"的方案仍可行时，也必须重切一次并选中 CPU 段。"""
    import config

    monkeypatch.setattr(config, "PIPELINE_CAPACITY_SAFETY_MARGIN", 1.0)
    monkeypatch.setattr(config, "prefer_all_workers_enabled", lambda: True)

    capacity_nodes = [
        # 上一轮供给写下的切点锁：主段独占 [0,4)，worker 独占 [4,8)。
        _node("master", role="master", candidate_layer_ranges=[[0, 4]]),
        # 本机 CPU 段（middle）：它的候选被锁在 [2,4)，在 [0,8) 被占满时挤不进去。
        _node("master", role="master", executor="cpu", segment_mode="middle",
              candidate_layer_ranges=[[2, 4]]),
        _node("worker", candidate_layer_ranges=[[4, 8]]),
    ]
    master = SimpleNamespace(
        _get_pipeline_capacity_nodes=lambda ids: copy.deepcopy(capacity_nodes),
        _relay_claimed_layer_prefix=lambda total: total,
        _relay_layer_claim_ranges=lambda total: {},
        _pipeline_node_metadata=lambda: {
            "master": {"engine": "pytorch"}, "worker": {"engine": "pytorch"},
        },
        _attach_pipeline_node_contract=lambda plan, **kwargs: plan,
    )
    plan = Scheduler.get_pipeline_capacity_plan(
        master, {"master", "worker"}, descriptor=_descriptor(),
        require_distributed=True,
    )
    assert plan["admitted"], plan
    identities = [(item["node_id"], item.get("executor_id", ""))
                  for item in plan["assignments"]]
    assert ("master", "cpu") in identities
    assert plan["placement_objective"]["node_count"] == 3


def test_scheduler_keeps_baseline_when_released_solve_raises(monkeypatch):
    """放开切点锁后的求解异常不得把本来可行的 baseline 打掉。

    为了修"僵化布局"反而让能跑的布局失败，是最坏的反向回归（外层 except 会把整个
    描述器判成非法）。这里让第二次求解抛 `PipelineCapacityError`，断言仍返回 baseline。
    """
    import config
    import pipeline_capacity
    import scheduler as scheduler_module

    monkeypatch.setattr(config, "PIPELINE_CAPACITY_SAFETY_MARGIN", 1.0)
    monkeypatch.setattr(config, "prefer_all_workers_enabled", lambda: True)
    real_solver = scheduler_module.solve_pipeline_capacity
    calls = []

    def flaky_solver(descriptor, nodes, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise pipeline_capacity.PipelineCapacityError("released solve exploded")
        return real_solver(descriptor, nodes, **kwargs)

    monkeypatch.setattr(scheduler_module, "solve_pipeline_capacity", flaky_solver)
    capacity_nodes = [
        _node("master", role="master", candidate_layer_ranges=[[0, 4]]),
        _node("master", role="master", executor="cpu", segment_mode="middle",
              candidate_layer_ranges=[[2, 4]]),
        _node("worker", candidate_layer_ranges=[[4, 8]]),
    ]
    master = SimpleNamespace(
        _get_pipeline_capacity_nodes=lambda ids: copy.deepcopy(capacity_nodes),
        _relay_claimed_layer_prefix=lambda total: total,
        _relay_layer_claim_ranges=lambda total: {},
        _pipeline_node_metadata=lambda: {
            "master": {"engine": "pytorch"}, "worker": {"engine": "pytorch"},
        },
        _attach_pipeline_node_contract=lambda plan, **kwargs: plan,
    )
    plan = Scheduler.get_pipeline_capacity_plan(
        master, {"master", "worker"}, descriptor=_descriptor(),
        require_distributed=True,
    )
    assert len(calls) == 2
    assert plan["admitted"], plan
    assert [(item["node_id"], item["start_layer"], item["end_layer"])
            for item in plan["assignments"]] == [("master", 0, 4), ("worker", 4, 8)]


def test_scheduler_keeps_baseline_when_no_additional_executor_is_configured(monkeypatch):
    """没有附加执行器时，行为与旧逻辑逐位一致（只用一次求解）。"""
    import config

    monkeypatch.setattr(config, "PIPELINE_CAPACITY_SAFETY_MARGIN", 1.0)
    monkeypatch.setattr(config, "prefer_all_workers_enabled", lambda: False)
    capacity_nodes = [
        _node("master", role="master", candidate_layer_ranges=[[0, 4]]),
        _node("worker", candidate_layer_ranges=[[4, 8]]),
    ]
    master = SimpleNamespace(
        _get_pipeline_capacity_nodes=lambda ids: copy.deepcopy(capacity_nodes),
        _relay_claimed_layer_prefix=lambda total: total,
        _relay_layer_claim_ranges=lambda total: {},
        _pipeline_node_metadata=lambda: {
            "master": {"engine": "pytorch"}, "worker": {"engine": "pytorch"},
        },
        _attach_pipeline_node_contract=lambda plan, **kwargs: plan,
    )
    plan = Scheduler.get_pipeline_capacity_plan(
        master, {"master", "worker"}, descriptor=_descriptor(),
        require_distributed=True,
    )
    assert plan["admitted"], plan
    assert [(item["node_id"], item["start_layer"], item["end_layer"])
            for item in plan["assignments"]] == [("master", 0, 4), ("worker", 4, 8)]
