"""附加执行器（同机第二执行器）的**共存判定**与**两案取优**。

★ 2026-10-27（LOCAL-EXECUTORS-01 / 审计 B1）：本模块是 scheduler 与供给侧
（`model_supply_plan_reuse`）**唯一**的取优口径，避免两处各写一份比较逻辑后又跑偏。

只放纯函数：不读环境、不读时钟、不做 IO。两条上层调用点负责：
  * `scheduler.get_pipeline_capacity_plan`  —— prepare 阶段的容量求解；
  * `model_supply_plan_reuse.plan_segment_supply_reuse` —— 供给阶段的重用偏好求解。

## 为什么需要"两案取优"

`candidate_layer_ranges` 是**上一轮供给写下的切点**（`SupplyCapacityEvidence.project`），
目的是锁定布局、避免 prepare 重新求解时切点漂移。它表达的是"**我手上确实有这个段**"，
在求解器里是**硬约束**（`capacity.py` 有完整论证：放宽它等于把算力规划到并不存在的
工件上）。

问题不在硬约束，而在**释放时机**：此前只在"解不出来"时才放开它。可是"不含本机 CPU
段的方案仍然可行"时不会触发释放 ⇒ 陈旧切点作为硬约束永久保留 ⇒ 附加执行器在后续
轮次被**结构性挤出**拓扑。典型现场（`model_supply.py:1236-1245` 记的同一形态）：
`android [[8,20]] + TABLET [[20,32]]` 占满 `[8,32)`，CPU 段在候选里却从未被选。
**"可行但已不该沿用"的布局永远不会重切。**

所以触发条件从"无解"扩展为"**本次解里缺少配置了的附加执行器**"，释放后再**认真比较**
两案，而不是盲目采用 released。

## 取优优先级链（从高到低，`plan_rank` 的元组顺序就是这个顺序）

1. **`admitted` 优先** —— 解不出来的一方永远输（`admitted=False` 的 rank 第一项为 1）。
2. **期望的附加执行器实际参与得多者优先** —— 这是本票"共存"目标的直接体现：
   配置了 CPU 段却被挤出，方案再"可行"也不该沿用。
3. **参与的执行器数多者优先** —— 同样表达了"共存"：能多用一段就多用一段，
   而不是把全部层压回少数节点。
4. **求解器目标函数** —— `weighted_work_cost` ↑ 更差、`memory_pressure_cost` ↑ 更差、
   `score_total` ↑ 更好（与 `capacity.placement_objective` 同一套字段）。
5. **完全相同 ⇒ 保留 baseline**（`choose_plan` 用严格 `<` 比较）。

第 5 条是**机制稳定性**的一部分：两案在目标上打平时不换布局，于是"上一轮锁定的切点"
下一轮仍被接受；一旦换了布局，下一轮的候选锁就来自新布局，release 求解会得到同一个
解 ⇒ 自然收敛、不来回震荡。

## 为什么没有新增 `QLH_*` 阈值

这条链里**没有任何可调数值**：`admitted` 是布尔、参与数是整数、目标函数是求解器给出的
确定性数值（同一输入逐位相同，不需要容差）。刻意不引入"改进多少才换"的门槛 ——
那会让"配了 CPU 段却永远差一点点"重新变成不可解释的现场问题。若将来确实需要，
应加在 `config.py` 并用 `_env_float`/`_env_int` 读取，不要在业务层直读 `os.environ`。
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

#: 缺失/非法目标函数值的"最差"占位。用 `math.inf` 而不是 0 —— 0 会让缺失的目标
#: 看起来"完美"，从而在比较中胜出；这里让它在比较中必输（fail-safe 方向）。
_WORST_COST = math.inf
_BEST_GAIN = -math.inf


def executor_identity(record: Mapping[str, Any] | None) -> tuple[str, str]:
    """`(node_id, executor_id)` —— 完整的执行器身份。

    `executor_id` 缺省 `""`（= 该节点的主/默认执行器）。**不要**退回裸 `node_id`：
    同机 CPU 段与主段共用 `node_id="master"`，裸 `node_id` 会把两段塌成一段
    （这正是 A8/A9 的根因形态）。
    """
    if not isinstance(record, Mapping):
        return "", ""
    return (
        str(record.get("node_id") or ""),
        str(record.get("executor_id") or ""),
    )


def executor_label(identity: tuple[str, str]) -> str:
    """对外的**可读标签**：主执行器裸 `node_id`，附加执行器 `node_id#executor_id`。

    仅用于日志与诊断字段（`reuse_preferred_node_ids` 等既有输出按裸 `node_id` 读）。
    身份判定一律用 `executor_identity()` 的元组，不得用标签反推。
    """
    node_id, executor_id = identity
    return f"{node_id}#{executor_id}" if executor_id else node_id


def additional_executor_identities(nodes: Iterable[Mapping[str, Any]] | None) -> set:
    """配置了的**附加执行器**身份（`executor_id` 非空）。

    主执行器（`executor_id=""`）不在此列 —— 它天然在场，不需要"共存"判定。
    """
    expected: set[tuple[str, str]] = set()
    for node in nodes or ():
        identity = executor_identity(node)
        if identity[0] and identity[1]:
            expected.add(identity)
    return expected


def _has_positive_layers(assignment: Mapping[str, Any]) -> bool:
    """该 assignment 是否**真的承载层**（零层 relay/占位条目不算参与）。

    字段缺失/非法时保守地视为"有层"（宁可多算参与，不要把真实参与漏掉）。
    """
    try:
        return int(assignment.get("end_layer", 0)) > int(assignment.get("start_layer", 0))
    except (TypeError, ValueError):
        return True


def assigned_executor_identities(
    plan: Mapping[str, Any] | None, *, with_layers_only: bool = True,
) -> set:
    """计划里**实际参与**的执行器身份集合。"""
    if not isinstance(plan, Mapping):
        return set()
    assigned: set[tuple[str, str]] = set()
    for assignment in plan.get("assignments") or ():
        if not isinstance(assignment, Mapping):
            continue
        if with_layers_only and not _has_positive_layers(assignment):
            continue
        identity = executor_identity(assignment)
        if identity[0]:
            assigned.add(identity)
    return assigned


def missing_additional_executors(
    plan: Mapping[str, Any] | None, nodes: Iterable[Mapping[str, Any]] | None,
) -> set:
    """**配置了、但本次解里没有拿到层**的附加执行器身份。

    这就是 B1 的触发判据："配了却被挤出"，而不是"容量不够"——后者会在 released
    求解里同样出不来，下面的取优会自然保留 baseline。
    """
    return additional_executor_identities(nodes) - assigned_executor_identities(plan)


def has_candidate_locks(nodes: Iterable[Mapping[str, Any]] | None) -> bool:
    """是否存在上一轮写下的切点锁（`candidate_layer_ranges`）。

    没有锁时"释放后求解"与 baseline 是**同一个问题**，重解只会浪费一次 DFS。
    """
    return any(
        isinstance(node, Mapping) and "candidate_layer_ranges" in node
        for node in nodes or ()
    )


def _cost(value: Any) -> float:
    """代价类目标值：越小越好；缺失/非法 ⇒ 最差。"""
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    return _WORST_COST


def _gain(value: Any) -> float:
    """收益类目标值（如 `score_total`）：越大越好；缺失/非法 ⇒ 最差。"""
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    return _BEST_GAIN


def plan_rank(
    plan: Mapping[str, Any] | None, expected_additional: Iterable[tuple[str, str]] = (),
) -> tuple:
    """比较用排序键：**越小越好**。元组顺序即取优优先级链（见模块 docstring）。

    `expected_additional` 是"配置了的附加执行器身份"；其中实际参与的个数排在第 2 位。
    第 3 位是参与执行器总数，第 4~6 位是求解器目标函数，最后没有"字典序"之类的
    任意 tie-break —— 打平由 `choose_plan` 保留 baseline。
    """
    expected = set(expected_additional or ())
    assigned = assigned_executor_identities(plan)
    objective = plan.get("placement_objective") if isinstance(plan, Mapping) else None
    objective = objective if isinstance(objective, Mapping) else {}
    return (
        0 if isinstance(plan, Mapping) and plan.get("admitted") is True else 1,
        -len(expected & assigned),
        -len(assigned),
        _cost(objective.get("weighted_work_cost")),
        _cost(objective.get("memory_pressure_cost")),
        -_gain(objective.get("score_total")),
    )


def choose_plan(
    baseline: Mapping[str, Any], released: Mapping[str, Any] | None,
    expected_additional: Iterable[tuple[str, str]] = (),
) -> tuple[Mapping[str, Any], str]:
    """在 baseline 与"放开切点锁后"的 released 之间取优。

    返回 `(选中方案, 决策标签)`。**打平保留 baseline**：不为了没有收益的重切而换布局。
    """
    if not isinstance(released, Mapping) or released.get("admitted") is not True:
        return baseline, "baseline_released_unavailable"
    if baseline.get("admitted") is not True:
        return released, "released_admitted"
    if plan_rank(released, expected_additional) < plan_rank(baseline, expected_additional):
        return released, "released_better"
    return baseline, "baseline_kept"
