"""Bounded metadata-only preferences for existing, verified segment assets.

Admission remains owned by the injected supply planner, including fresh runtime
budgets, alignment, proper segments and distributed/all-worker policy. Advertised
ranges describe registered assets; they never establish a loaded-ready ACK.
"""
from __future__ import annotations

import copy
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from executor_coexistence import (
    additional_executor_identities,
    choose_plan,
    executor_label,
    has_candidate_locks,
    missing_additional_executors,
)

logger = logging.getLogger(__name__)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_FORMATS = {"llama_cpp": "gguf", "pytorch": "safetensors"}


@dataclass(frozen=True)
class _Candidate:
    """一个已验证的缓存段。**身份是 `(node_id, executor_id)`**。

    ★ 2026-10-27（LOCAL-EXECUTORS-01 / A8）：此前只有 `node_id`。同一台物理机现在可以
    同时承载主执行器（`executor_id=""`，GPU 段、PyTorch）与附加执行器（`executor_id="cpu"`，
    CPU 段、llama.cpp keep-head），两者共用 `node_id`。复用偏好若只按 `node_id` 记，
    两段的缓存段会互相覆盖/互相排斥——本机 CPU 段那份已验证的 GGUF 段永远进不了候选
    （见 `_candidates` 的说明）。
    """
    node_id: str
    executor_id: str
    start: int
    end: int
    weight_bytes: int
    model_id: str
    artifact_sha256: str


def _sequence(value: Any) -> list | tuple:
    return value if isinstance(value, (list, tuple)) else ()


def _identity_key(key: Any) -> tuple[str, str]:
    """把偏好键归一成 `(node_id, executor_id)`。

    内部一律用元组键；字符串键是**旧调用方兼容**（`node_id` ⇒ 主执行器 `""`）。
    绝不把字符串键套到附加执行器上：`{"master": ...}` 指主段，不能变成 CPU 段的偏好。
    """
    if isinstance(key, tuple) and len(key) == 2:
        return (str(key[0]), str(key[1]))
    if isinstance(key, str):
        return (key, "")
    return ("", "")


def _engine_for(engine_by_node: Mapping[Any, str] | None, node_id: str,
                executor_id: str) -> str | None:
    """按执行器身份取引擎；元组键优先，旧字符串键只在主执行器上兼容。

    ★ A8 问题 3：调用方此前用 `{record["node_id"]: record["engine"]}` 建映射，本机 CPU 段
    （`llama_cpp`）会**覆盖**主段（`pytorch`）的同名键。修好 ① ② 之后，`_preserves`
    仍会因 `engine_by_node["master"] == "llama_cpp"` 把主段的 `pytorch` 判成不等而返回
    False ⇒ 复用照旧失效。所以键必须同步改成 `(node_id, executor_id)`。
    """
    if not isinstance(engine_by_node, Mapping):
        return None
    key = (node_id, executor_id)
    if key in engine_by_node:
        return engine_by_node[key]
    if not executor_id:
        return engine_by_node.get(node_id)
    return None


def _span(value: Any, total: int, alignment: int) -> tuple[int, int] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(bound) is not int for bound in value)):
        return None
    start, end = value
    if not (0 <= start < end <= total and end - start < total
            and start % alignment == end % alignment == 0):
        return None
    return start, end


def _model_matches(models: Any, artifact: Mapping, engine: str) -> bool:
    matches = [model for model in _sequence(models)
               if isinstance(model, Mapping)
               and model.get("model_id") == artifact.get("model_id")]
    return (len(matches) == 1 and matches[0].get("sha256") == artifact.get("artifact_sha256")
            and matches[0].get("engine") == engine
            and matches[0].get("format") == _FORMATS.get(engine))


def _candidates(descriptors, capacity_records, nodes, *, source_model_id,
                source_model_sha256, tokenizer_sha256, engine_by_node,
                capabilities_by_node, alignment):
    reference = descriptors.get("llama_cpp") or descriptors.get(
        _engine_for(engine_by_node, "master", "")) or {}
    total = reference.get("total_layers")
    if (type(total) is not int or total < 2 or type(alignment) is not int
            or alignment < 1 or total % alignment
            or not isinstance(source_model_id, str) or not _ID.fullmatch(source_model_id)
            or any(not isinstance(value, str) or not _SHA.fullmatch(value)
                   for value in (source_model_sha256, tokenizer_sha256))):
        return [], total
    active = {
        (node.get("node_id"), str(node.get("executor_id") or "")): node
        for node in nodes if isinstance(node, Mapping)
    }
    candidates = []
    seen = set()
    for record in capacity_records:
        if not isinstance(record, Mapping):
            continue
        node_id = record.get("node_id")
        executor_id = str(record.get("executor_id") or "")
        identity = (node_id, executor_id)
        # ★ A8 问题 2：**只跳过主执行器，不跳过 `node_id == "master"`**。
        #
        #   主段（`executor_id=""`）的切点是每轮新裁的，没有"已验证的缓存段"可复用；
        #   但本机 CPU 段与它共用 `node_id="master"`，此前被这条判据一起跳过 ⇒
        #   "本机 CPU 段那份已验证的 GGUF 段永远不进复用候选"。
        #   注意也不能只改成"master 非空才允许"：其它节点的附加执行器同样要按
        #   `(node_id, executor_id)` 判在不在线，所以统一用身份查 `active`。
        if (not isinstance(node_id, str) or not node_id
                or (node_id == "master" and not executor_id)
                or identity not in active):
            continue
        engine = _engine_for(engine_by_node, node_id, executor_id)
        source = descriptors.get(engine) or {}
        if (engine not in _FORMATS or active[identity].get("engine") != engine
                or source.get("total_layers") != total
                or source.get("hidden_size") != reference.get("hidden_size")):
            continue
        if capabilities_by_node is None:
            caps = record
        else:
            # `capabilities_by_node` 的既有契约是 `worker_snapshot(node_id)`（键为裸
            # `node_id`）。**附加执行器不能退回裸键**：那会拿到同机主段的快照，
            # 用主段的 engines/artifacts/models 去"证明" CPU 段的工件（串味）。
            # 主执行器（`executor_id=""`）仍兼容裸键 ⇒ 既有调用逐位不变；
            # 若调用方未来提供身份键快照，优先用它。
            caps = capabilities_by_node.get(identity)
            if caps is None and not executor_id:
                caps = capabilities_by_node.get(node_id, {})
        if not isinstance(caps, Mapping):
            continue
        artifacts = _sequence(record.get("layer_artifacts"))
        if len(artifacts) > 64:
            continue
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            if any(artifact.get(key) != expected for key, expected in (
                    ("source_model_id", source_model_id),
                    ("source_model_sha256", source_model_sha256),
                    ("tokenizer_sha256", tokenizer_sha256))):
                continue
            interval = _span(artifact.get("layer_range"), total, alignment)
            if interval is None:
                continue
            start, end = interval
            mode = "head" if start == 0 else "tail" if end == total else "middle"
            if (artifact.get("segment_mode") != mode
                    or type(artifact.get("hidden_size")) is not int
                    or artifact["hidden_size"] <= 0
                    or artifact["hidden_size"] != source.get("hidden_size")):
                continue
            if not any(_span(value, total, alignment) == interval
                       for value in _sequence(record.get("layer_ranges"))):
                continue
            artifact_id, digest = artifact.get("model_id"), artifact.get("artifact_sha256")
            if (not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id)
                    or not isinstance(digest, str) or not _SHA.fullmatch(digest)
                    or not _model_matches(caps.get("models"), artifact, engine)):
                continue
            if capabilities_by_node is not None:
                if (engine not in _sequence(caps.get("engines"))
                        or not any(item == artifact for item in _sequence(caps.get("layer_artifacts")))
                        or not any(_span(value, total, alignment) == interval
                                   for value in _sequence(caps.get("layer_ranges")))
                        or ("models" in record and not _model_matches(record["models"], artifact, engine))):
                    continue
            sizes = artifact.get("layer_weight_bytes")
            if (not isinstance(sizes, (list, tuple)) or len(sizes) != end - start
                    or any(type(size) is not int or size <= 0 for size in sizes)):
                continue
            identity_key = (node_id, executor_id, start, end, artifact_id, digest)
            if identity_key not in seen:
                seen.add(identity_key)
                candidates.append(_Candidate(
                    node_id, executor_id, start, end, sum(sizes), artifact_id, digest,
                ))
    candidates.sort(key=lambda item: (-item.weight_bytes, item.node_id, item.executor_id,
                                      item.start, item.end, item.model_id,
                                      item.artifact_sha256))
    return candidates, total


def _preserves(plan, preferred, total, alignment, engine_by_node):
    """方案是否**在保留全部已裁段的前提下**仍成立。

    ★ 2026-10-27（LOCAL-EXECUTORS-01 / A8 问题 1）：唯一性判据从 `node_id` 改为
    **执行器身份** `(node_id, executor_id)`。

    目标形态就是 `master` 出现两次（`master#""` GPU 段 + `master#cpu` CPU 段）。
    此前写 `node_id in assigned` ⇒ 第二次出现的 `master` 被当成重复 ⇒ 恒 False
    ⇒ 复用偏好**一律**退回 `unrestricted`，已裁好的段每轮重新规划。
    "一段一个执行器"是真正的契约（同一个执行器不能占两段），同机的**另一个执行器**
    当然可以占一段。
    """
    if not plan.get("admitted"):
        return False
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or len(assignments) < 2:
        return False
    cursor, assigned = 0, {}
    for item in assignments:
        if not isinstance(item, Mapping):
            return False
        node_id = item.get("node_id")
        executor_id = str(item.get("executor_id") or "")
        identity = (node_id, executor_id)
        interval = _span([item.get("start_layer"), item.get("end_layer")], total, alignment)
        if (not isinstance(node_id, str) or not node_id or identity in assigned
                or interval is None or interval[0] != cursor
                or item.get("engine") != _engine_for(engine_by_node, node_id, executor_id)):
            return False
        assigned[identity] = list(interval)
        cursor = interval[1]
    first = assignments[0]
    return (first.get("node_id") == "master"
            and not str(first.get("executor_id") or "")
            and cursor == total
            and all(assigned.get(_identity_key(key)) in ranges
                    for key, ranges in preferred.items()))


def _annotate(plan, preferred, strategy):
    result = copy.deepcopy(plan)
    # 元组键是内部身份，对外字段仍按既有读法输出裸 `node_id`（附加执行器才带
    # `node#executor`）—— 既有单执行器拓扑的输出逐字节不变。
    result.update(reuse_preferred_node_ids=sorted(
        executor_label(_identity_key(key)) for key in preferred),
        target_preference=strategy, loaded_ready=False)
    return result


def _same_placement_quality(candidate, baseline):
    """Cached assets break placement ties; they cannot change node policy/work."""
    policy = baseline.get("placement_policy")
    if not isinstance(policy, Mapping) or candidate.get("placement_policy") != policy:
        return False
    left, right = candidate.get("placement_objective"), baseline.get("placement_objective")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if left.get("node_count") != right.get("node_count"):
        return False
    for field in ("weighted_work_cost", "memory_pressure_cost", "score_total"):
        lhs, rhs = left.get(field), right.get(field)
        if (type(lhs) not in (int, float) or type(rhs) not in (int, float)
                or not math.isfinite(lhs) or not math.isfinite(rhs)
                or not math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-15)):
            return False
    return True


def plan_segment_supply_reuse(
    descriptors: dict[str, dict], capacity_records: list[dict], nodes: list[dict], *,
    planner: Callable[..., dict], source_model_id: str, source_model_sha256: str,
    tokenizer_sha256: str, engine_by_node: Mapping[Any, str],
    capabilities_by_node: Mapping[str, dict] | None = None, alignment: int = 1,
) -> dict:
    """Select preferences without solving capacity or modifying advertisements.

    ``nodes`` are the new target engines/runtime budgets. Bind them, descriptors,
    runtime projections and safety margin into ``planner``; it must call existing
    ``plan_segment_supply`` with distributed/all-worker admission unchanged.
    The only callback argument is ``preferred_ranges`` in absolute source layer
    coordinates; ``{}`` requests unrestricted planning. Exceptions propagate.

    ★ 2026-10-27（LOCAL-EXECUTORS-01 / A8）：`preferred_ranges` 的键是**执行器身份**
    `(node_id, executor_id)`，与 `plan_segment_supply` 的入口契约一致（那里兼容旧的
    裸 `node_id` 键）。`engine_by_node` 同样按 `(node_id, executor_id)` 传入；旧的
    裸 `node_id` 映射仍被兼容，但只作用于主执行器 —— 同机 CPU 段的引擎不得覆盖主段。

    ``capabilities_by_node`` contains worker_snapshot(node_id)['capabilities'].
    Pass an empty body after a failed snapshot: no engine proof means no reuse.
    If the map is omitted, models inherited into capacity records provide proof.
    A model's physical id/digest and supported engine/format must match its
    artifact. Installed engines, node.engine and guessed formats are insufficient.

    Solve unrestricted placement first. Verified cached ranges may break ties
    only when the node-count policy and capability/memory objective are equal
    to that baseline. Then try ranges jointly and individually by descending
    advertised decoder bytes. One worker executes one segment. Calls remain
    bounded by candidate_count + 3（baseline、B1 的 release 尝试、joint、各 greedy 尝试）;
    cache metadata cannot override capability.
    Metadata distinguishes joint_fixed, greedy_fixed and unrestricted selection;
    it describes target preference, never execution or loaded readiness.

    ★ B1：若 baseline 可行但**缺少配置了的附加执行器**（同机 CPU 段被陈旧切点挤出），
    会再放开 `candidate_layer_ranges` 解一次并按 `executor_coexistence.plan_rank`
    取优；因此最坏多一次求解，且打平时保留 baseline。
    """
    candidates, total = _candidates(
        descriptors, capacity_records, nodes, source_model_id=source_model_id,
        source_model_sha256=source_model_sha256, tokenizer_sha256=tokenizer_sha256,
        engine_by_node=engine_by_node, capabilities_by_node=capabilities_by_node,
        alignment=alignment,
    )
    baseline = planner(preferred_ranges={})
    if not baseline.get("admitted"):
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**连无偏好的 baseline 都无解 ⇒ 放开切点锁**。
        #
        #   `candidate_layer_ranges` 由上一轮供给写入（锁定布局、避免 prepare 重切，见
        #   #117）。但拓扑变化时它是死锁：旧切点把区间占满，新执行器（本机 CPU 段）或
        #   换过段的设备（surface 取代 ROG）挤不进自己工件覆盖的区间 ⇒ baseline 无解
        #   ⇒ 供给直接失败，而设备手上还拿着**过期工件**。
        #   scheduler 侧的同名机制早已放开重排（`get_pipeline_capacity_plan` 里
        #   "锁定的切点不再满足容量，放开候选约束重排"），供给侧此前缺这一步。
        #   planner 未声明该关键字时退化为原行为，保持对旧调用方的兼容。
        try:
            released = planner(preferred_ranges={}, release_candidates=True)
        except TypeError:
            released = baseline
        logger.info(
            "event=supply_plan_release_candidates baseline_admitted=%s "
            "released_admitted=%s baseline_reason=%s released_reason=%s",
            baseline.get("admitted"), released.get("admitted"),
            baseline.get("reason_code"), released.get("reason_code"),
        )
        if released.get("admitted"):
            return _annotate(released, {}, "released_candidates")
        return _annotate(baseline, {}, "unrestricted")
    # ★ 2026-10-27（LOCAL-EXECUTORS-01 / B1）：**"可行但已不该沿用"的布局也要重切**。
    #
    #   上面那条只在**解不出来**时释放切点锁。问题：只要"不含本机 CPU 段的方案仍然
    #   可行"，就永远不释放 ⇒ 陈旧切点作为硬约束永久保留 ⇒ 附加执行器在后续轮次被
    #   **结构性挤出**拓扑。`model_supply.py:1236-1245` 记的
    #   "`android[[8,20]]+TABLET[[20,32]]` 占满 `[8,32)`，CPU 段在候选里却从未被选"
    #   在另一层的表现正是这条。
    #
    #   与 `scheduler.get_pipeline_capacity_plan` 里的同名机制**同口径**（两边都用
    #   `executor_coexistence` 的判定与取优）：baseline 里缺少配置了的附加执行器时，
    #   放开切点锁再解一次，然后**认真比较**两案，不盲目取 released。优先级链见
    #   `executor_coexistence.plan_rank`：admitted → 期望附加执行器参与数 → 参与执行器
    #   总数 → 目标函数；打平保留 baseline（机制稳定性）。
    #
    #   没有候选锁时"释放后求解"就是同一个问题 ⇒ 不重解（避免每次规划多一次 DFS）。
    expected_additional = additional_executor_identities(nodes)
    missing_additional = missing_additional_executors(baseline, nodes)
    if expected_additional and missing_additional and has_candidate_locks(capacity_records):
        try:
            released = planner(preferred_ranges={}, release_candidates=True)
        except TypeError:
            # planner 未声明该关键字 ⇒ 旧调用方，维持原行为。
            released = None
        if released is not None:
            chosen, decision = choose_plan(baseline, released, expected_additional)
            logger.info(
                "event=supply_release_candidates_coexistence decision=%s "
                "missing_additional=%s expected_additional=%s "
                "baseline_admitted=%s released_admitted=%s baseline_locks=%s",
                decision,
                sorted(executor_label(item) for item in missing_additional),
                sorted(executor_label(item) for item in expected_additional),
                baseline.get("admitted"), released.get("admitted"),
                [node.get("candidate_layer_ranges") for node in capacity_records
                 if isinstance(node, Mapping) and "candidate_layer_ranges" in node],
            )
            if chosen is released:
                return _annotate(released, {}, "released_candidates")
    if candidates:
        joint = {}
        for item in candidates:
            key = (item.node_id, item.executor_id)
            joint.setdefault(key, []).append([item.start, item.end])
        joint = {key: sorted({tuple(span) for span in spans})
                 for key, spans in sorted(joint.items())}
        joint = {key: [list(span) for span in spans] for key, spans in joint.items()}
        plan = planner(preferred_ranges=copy.deepcopy(joint))
        if (_preserves(plan, joint, total, alignment, engine_by_node)
                and _same_placement_quality(plan, baseline)):
            return _annotate(plan, joint, "joint_fixed")
        fixed, selected = {}, None
        for item in candidates:
            key = (item.node_id, item.executor_id)
            if key in fixed:
                continue
            trial = {**fixed, key: [[item.start, item.end]]}
            plan = planner(preferred_ranges=copy.deepcopy(trial))
            if (_preserves(plan, trial, total, alignment, engine_by_node)
                    and _same_placement_quality(plan, baseline)):
                fixed, selected = trial, plan
        if selected is not None:
            return _annotate(selected, fixed, "greedy_fixed")
    return _annotate(baseline, {}, "unrestricted")
