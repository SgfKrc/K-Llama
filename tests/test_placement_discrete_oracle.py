"""Small exhaustive oracle for the capacity solver's discrete placement policy.

Enumerate node subsets and positive contiguous partitions independently of the
solver's recursive search. Inputs use complete additive byte quotes, optional
candidate/ready ranges, layer budgets and segment modes. No production helper,
profile evaluator, model, or runtime is used to construct the expected plans.
"""
from __future__ import annotations

import itertools
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction

import pytest

from qlh_core.planning.capacity import solve_pipeline_capacity


_COST_REL = 1e-12
_COST_ABS = 1e-14


@dataclass(frozen=True)
class _Stage:
    node_id: str
    start: int
    end: int
    raw: int
    required: int
    work: Fraction


@dataclass(frozen=True)
class _Candidate:
    stages: tuple[_Stage, ...]
    objective: tuple


def _descriptor(sizes, *, embedding=0, norm=0, head=0, other=0, tied=False):
    return {
        "model_id": "discrete-oracle",
        "model_type": "qwen2",
        "model_sha256": "a" * 64,
        "total_layers": len(sizes),
        "layer_weight_bytes": list(sizes),
        "component_weight_bytes": {
            "embedding": embedding, "final_norm": norm,
            "lm_head": head, "other": other,
        },
        "tie_word_embeddings": tied,
    }


def _node(node_id, score=1, *, capacity=128, reserve=0, multiplier=1.0,
          role=None, **contracts):
    return {
        "node_id": node_id,
        "role": role or ("master" if node_id == "master" else "client"),
        "score": score,
        "capacity_bytes": capacity,
        "reserve_bytes": reserve,
        "runtime_multiplier": multiplier,
        "execution_device": "cpu",
        **contracts,
    }


def _layer_budget(max_layers, *, local_cut):
    return {"available_bytes": 128, "per_layer_bytes": 1,
            "max_layers": max_layers, "local_cut": local_cut}


def _ordered_nodes(nodes):
    """Reconstruct the public input semantics, including free-byte ordering."""
    normalized = []
    for raw in nodes:
        capacity = int(raw.get("capacity_bytes", 0))
        reserve = int(raw.get("reserve_bytes", 0))
        if capacity <= reserve:
            continue
        normalized.append({
            **raw,
            "node_id": str(raw.get("node_id", "") or "").strip(),
            "role": str(raw.get("role", "client") or "client"),
            "score": float(raw.get("score", 0) or 0),
            "capacity_bytes": capacity,
            "reserve_bytes": reserve,
            "runtime_multiplier": float(raw.get("runtime_multiplier", 1.0)),
        })
    # ★ 2026-10-25（LOCAL-EXECUTORS-01）：与 `_normalize_nodes` 的排序保持**逐字一致**。
    #   段类型必须参与：head → 未声明 → middle → tail。
    #   - 排序决定 DFS 遍历次序，而 DFS 是**单向**的（`cursor` 只前进、不回头）⇒
    #     `middle` 段若排在 `head` 段之前，在 `cursor=0` 会被 `cursor != 0` 拒掉，
    #     之后 `cursor` 前进而 DFS 不回头 ⇒ 它永远拿不到本可承载的中间区间。
    #   - **未声明 `segment_mode` 的节点排在 head 之后、middle 之前**：它最灵活，
    #     选择多，越早安排越不会被别人挤掉。放到最后反而会无解（有约束的 tail 先占
    #     末段 ⇒ 灵活节点无处安放）。
    #   本函数复刻求解器的排序，漏掉这一项就会让 oracle 与求解器对
    #   `control_only_nodes` 的期望分歧（实测 seed-11 报
    #   `['worker-a','worker-z'] == ['worker-z','worker-a']`）。
    mode_order = {"head": 0, "": 0.5, "middle": 1, "tail": 2}
    return sorted(normalized, key=lambda node: (
        node["role"] != "master",
        mode_order.get(str(node.get("segment_mode") or "").lower(), 0.5),
        -node["score"],
        -(node["capacity_bytes"] - node["reserve_bytes"]),
        node["node_id"],
    ))


def _enumerate_candidates(descriptor, nodes, *, safety_margin=1.0,
                          require_distributed=False, prefer_all_workers=False,
                          required_node_id=None):
    """Enumerate complete plans; never reuse a suffix or a solver helper.

    Byte scaling and objective arithmetic use rational numbers. All multipliers
    and margins in this suite are dyadic, so a rational ceiling also agrees
    with the solver's single floating-point multiplication/ceiling at this size.

    ★ 2026-10-26（LOCAL-EXECUTORS-01）：**`candidate_layer_ranges` 是硬约束**，本枚举
    按此建模（受 `test_candidate_ranges_and_layer_limits_remain_hard_with_local_cut`
    约束）。一度试过在这里也枚举"去掉候选"的解空间以迁就真机上的自锁，但那方向是错的：
    候选表达的是"**我手上确实有这个段**"，放宽它等于把设备规划到并不存在的工件上。
    真机自锁要修的是**写入侧**（谁把上一轮切点写成了这一轮的候选）。
    """
    sizes = descriptor["layer_weight_bytes"]
    total_layers, total_work = len(sizes), sum(sizes)
    components = descriptor["component_weight_bytes"]
    embedding = components.get("embedding", 0)
    norm = components.get("final_norm", 0)
    output = components.get("lm_head", 0) + components.get("other", 0)
    if descriptor.get("tie_word_embeddings") and output == 0:
        output = embedding
    ordered = _ordered_nodes(nodes)
    candidates = []
    for count in range(1, min(len(ordered), total_layers) + 1):
        if require_distributed and count < 2:
            continue
        for selected in itertools.combinations(ordered, count):
            if required_node_id is not None and required_node_id not in {
                node["node_id"] for node in selected
            }:
                continue
            for cuts in itertools.combinations(range(1, total_layers), count - 1):
                bounds = (0, *cuts, total_layers)
                stages = []
                for node, start, end in zip(selected, bounds, bounds[1:]):
                    span = (start, end)
                    if "candidate_layer_ranges" in node and span not in {
                        tuple(item) for item in node["candidate_layer_ranges"]
                    }:
                        break
                    budget = node.get("layer_budget")
                    if budget is not None and end - start > budget["max_layers"]:
                        break
                    local_cut = budget is not None and budget.get("local_cut", False)
                    if not local_cut:
                        if "layer_ranges" in node and span not in {
                            tuple(item) for item in node["layer_ranges"]
                        }:
                            break
                        mode = node.get("segment_mode")
                        allowed_position = {
                            None: True,
                            "head": start == 0,
                            "middle": start > 0 and end < total_layers,
                            "tail": end == total_layers,
                        }
                        if not allowed_position[mode.lower() if mode else None]:
                            break
                    quoted_sizes = node.get("layer_weight_bytes", sizes)
                    raw = sum(quoted_sizes[start:end])
                    if "retained_component_bytes" in node:
                        raw += node["retained_component_bytes"]
                    else:
                        raw += norm
                        raw += embedding if start == 0 else 0
                        raw += output if end == total_layers else 0
                    scaled = (raw * Fraction(str(node["runtime_multiplier"]))
                              * Fraction(str(safety_margin)))
                    required = node["reserve_bytes"] + math.ceil(scaled)
                    if required > node["capacity_bytes"]:
                        break
                    stages.append(_Stage(
                        node["node_id"], start, end, raw, required,
                        Fraction(sum(sizes[start:end]), total_work),
                    ))
                else:
                    objective = (
                        -count if prefer_all_workers else count,
                        sum((stage.work ** 2 / max(Fraction(str(node["score"])), 1)
                             for node, stage in zip(selected, stages)), Fraction()),
                        sum((Fraction(stage.required, node["capacity_bytes"]) ** 2
                             for node, stage in zip(selected, stages)), Fraction()),
                        -sum((Fraction(str(node["score"])) for node in selected),
                             Fraction()),
                        tuple((stage.node_id, stage.start, stage.end) for stage in stages),
                    )
                    candidates.append(_Candidate(tuple(stages), objective))
    return tuple(candidates)


def _clearly_unique(best, candidates):
    """Require a separated numeric optimum before general identity assertions."""
    for other in candidates:
        if other is best:
            continue
        for index in range(4):
            left, right = best.objective[index], other.objective[index]
            if left == right:
                continue
            if index and math.isclose(float(left), float(right),
                                      rel_tol=_COST_REL, abs_tol=_COST_ABS):
                return False
            break
        else:
            # Equal numeric costs leave identity as the only discriminator.
            return False
    return True


def _identity(assignments):
    return tuple((item["node_id"], item["start_layer"], item["end_layer"])
                 for item in assignments)


def _assert_matches_oracle(descriptor, nodes, **options):
    options = {"safety_margin": 1.0, **options}
    before = deepcopy((descriptor, nodes))
    candidates = _enumerate_candidates(descriptor, nodes, **options)
    plan = solve_pipeline_capacity(descriptor, nodes, **options)
    assert (descriptor, nodes) == before
    assert plan["admitted"] is bool(candidates), plan
    if not candidates:
        assert plan["status"] == "rejected"
        assert plan["assignments"] == []
        return plan

    assert plan["status"] == "admitted"
    identity = _identity(plan["assignments"])
    by_identity = {candidate.objective[4]: candidate for candidate in candidates}
    assert identity in by_identity, f"solver returned an infeasible plan: {identity}"
    actual = by_identity[identity]
    best = min(candidates, key=lambda candidate: candidate.objective)
    assert actual.objective[0] == best.objective[0]
    assert plan["participating_node_count"] == len(actual.stages)
    assert plan["placement_policy"]["prefer_all_workers"] is bool(
        options.get("prefer_all_workers", False)
    )
    ordered = _ordered_nodes(nodes)
    by_id = {node["node_id"]: node for node in ordered}
    assert plan["control_only_nodes"] == [
        node["node_id"] for node in ordered if node["node_id"] not in {
            stage.node_id for stage in actual.stages
        }
    ]
    for index, (item, stage) in enumerate(zip(plan["assignments"], actual.stages)):
        node = by_id[stage.node_id]
        assert item["layers_count"] == stage.end - stage.start > 0
        assert item["has_embedding"] is (index == 0)
        assert item["has_lm_head"] is (index == len(actual.stages) - 1)
        assert item["raw_weight_bytes"] == stage.raw
        assert item["required_bytes"] == stage.required
        assert item["capacity_bytes"] == node["capacity_bytes"]
        assert item["reserve_bytes"] == item["reserved_bytes"] == node["reserve_bytes"]
        assert item["runtime_multiplier"] == node["runtime_multiplier"]
        assert item["available_bytes"] == node["capacity_bytes"]
        assert item["allocatable_bytes"] == node["capacity_bytes"] - node["reserve_bytes"]
        assert item["headroom_bytes"] == node["capacity_bytes"] - stage.required >= 0
        assert item["shortfall_bytes"] == 0
        assert item["placement_weight"] == max(node["score"], 1)
        assert item["placement_work_fraction"] == pytest.approx(float(stage.work))

    objective = plan["placement_objective"]
    assert objective["node_count"] == len(actual.stages)
    for field, index, sign in (
        ("weighted_work_cost", 1, 1),
        ("memory_pressure_cost", 2, 1),
        ("score_total", 3, -1),
    ):
        assert objective[field] == pytest.approx(
            sign * float(actual.objective[index]), rel=_COST_REL, abs=_COST_ABS,
        )
        assert objective[field] == pytest.approx(
            sign * float(best.objective[index]), rel=_COST_REL, abs=_COST_ABS,
        )
    if _clearly_unique(best, candidates):
        assert identity == best.objective[4]
    return plan


def _assert_order_invariance(descriptor, nodes, *, all_orders=False, **options):
    orders = (itertools.permutations(nodes) if all_orders
              else (nodes, list(reversed(nodes))))
    first = None
    for order in orders:
        plan = _assert_matches_oracle(descriptor, list(order), **options)
        if first is None:
            first = plan
        else:
            for field in ("admitted", "assignments", "control_only_nodes"):
                assert plan[field] == first[field]
            if plan["admitted"]:
                assert plan["plan_id"] == first["plan_id"]
                assert plan["placement_objective"] == first["placement_objective"]
    return first


@pytest.mark.parametrize("nodes, expected", [
    ([_node("a-fast", 16), _node("z-slow", 4),
      _node("input", -4, role="master")], ["input", "a-fast", "z-slow"]),
    ([_node("a-low", 0), _node("m-mid", 2), _node("z-high", 4)],
     ["z-high", "m-mid", "a-low"]),
    ([_node("a", 2, capacity=128, reserve=112),
      _node("m", 2, capacity=64, reserve=16), _node("z", 2, capacity=64)],
     ["z", "m", "a"]),
    ([_node("z", 2, reserve=1), _node("m", 2, reserve=1),
      _node("a", 2, reserve=1)], ["a", "m", "z"]),
], ids=["master-role", "descending-score", "free-not-total-bytes", "node-id"])
def test_execution_order_matches_every_normalization_tiebreak(nodes, expected):
    plan = _assert_order_invariance(
        _descriptor((1, 2, 5)), nodes, all_orders=True,
        require_distributed=True, prefer_all_workers=True,
    )
    assert [item["node_id"] for item in plan["assignments"]] == expected


def test_unequal_reference_bytes_define_work_independently_of_engine_byte_quotes():
    plan = _assert_order_invariance(
        _descriptor((5, 1, 1, 1)),
        [_node("master", layer_weight_bytes=[1, 1, 1, 1]),
         _node("worker", layer_weight_bytes=[20, 20, 20, 20])],
        require_distributed=True, required_node_id="master",
    )
    assert [item["layers_count"] for item in plan["assignments"]] == [1, 3]
    assert [item["raw_weight_bytes"] for item in plan["assignments"]] == [1, 60]
    assert [item["placement_work_fraction"] for item in plan["assignments"]] == (
        pytest.approx([5 / 8, 3 / 8])
    )


@pytest.mark.parametrize("capacity, admitted", [(4, False), (5, True), (6, True)])
def test_reserve_is_outside_scaling_and_rounding_happens_once(capacity, admitted):
    # 3 + ceil(1 * 1.25 * 1.25) == 5. Rounding after each factor gives 6.
    plan = _assert_matches_oracle(
        _descriptor((1,)),
        [_node("master", capacity=capacity, reserve=3, multiplier=1.25)],
        safety_margin=1.25, required_node_id="master",
    )
    assert plan["admitted"] is admitted
    if admitted:
        assert plan["assignments"][0]["required_bytes"] == 5


@pytest.mark.parametrize("changed_node", ["master", "tail"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_component_and_reserve_budgets_obey_each_nodes_exact_boundary(changed_node, delta):
    nodes = [
        _node("master", 9, capacity=15, reserve=3, multiplier=1.5,
              candidate_layer_ranges=[[0, 1]]),
        _node("tail", 2, capacity=15, reserve=2, multiplier=0.75,
              candidate_layer_ranges=[[1, 3]]),
    ]
    next(node for node in nodes if node["node_id"] == changed_node)["capacity_bytes"] += delta
    plan = _assert_order_invariance(
        _descriptor((1, 2, 3), embedding=3, norm=2, head=5, other=1), nodes,
        safety_margin=1.25, require_distributed=True, required_node_id="master",
    )
    assert plan["admitted"] is (delta >= 0)
    if plan["admitted"]:
        # Norm is charged per stage; embedding/output belong to the boundaries.
        assert [item["raw_weight_bytes"] for item in plan["assignments"]] == [6, 13]
        assert [item["required_bytes"] for item in plan["assignments"]] == [15, 15]


@pytest.mark.parametrize("ranges, local_cut, max_layers, admitted", [
    ([[0, 1]], True, 4, True),
    ([[0, 1]], False, 4, False),
    ([[0, 4]], True, 4, False),
    ([], True, 4, False),
    ([[0, 1], [0, 1]], True, 4, True),
    ([[0, 2]], True, 1, False),
    ([[0, 2]], True, 2, True),
], ids=["exact-candidate", "ready-range-still-hard", "containment-is-insufficient",
        "empty-candidates", "duplicate-candidates", "layer-limit", "local-cut"])
def test_candidate_ranges_and_layer_limits_remain_hard_with_local_cut(
    ranges, local_cut, max_layers, admitted,
):
    nodes = [
        _node("master", 64, candidate_layer_ranges=ranges, layer_ranges=[[0, 4]],
              layer_budget=_layer_budget(max_layers, local_cut=local_cut)),
        _node("tail", layer_ranges=[[2, 4], [1, 4]]),
    ]
    plan = _assert_order_invariance(
        _descriptor((5, 1, 1, 1)), nodes,
        require_distributed=True, required_node_id="master", prefer_all_workers=True,
    )
    assert plan["admitted"] is admitted


@pytest.mark.parametrize("required", ["master", "tail"])
@pytest.mark.parametrize("distributed", [False, True])
def test_required_executor_cannot_be_skipped_or_receive_an_empty_range(required, distributed):
    nodes = [_node("master", 0.5 if required == "master" else 8),
             _node("tail", 0.5 if required == "tail" else 8)]
    descriptor = _descriptor((1, 1))
    ordinary = _assert_matches_oracle(descriptor, nodes)
    assert ordinary["assignments"][0]["node_id"] != required
    plan = _assert_order_invariance(
        descriptor, nodes, required_node_id=required, require_distributed=distributed,
    )
    assert required in {item["node_id"] for item in plan["assignments"]}
    assert len(plan["assignments"]) == (2 if distributed else 1)


@pytest.mark.parametrize("failure", ["missing", "reserved", "too-small", "no-range"])
def test_required_master_rejects_even_when_an_optional_worker_can_host_everything(failure):
    master = _node("master", 0)
    if failure == "reserved":
        master.update(capacity_bytes=4, reserve_bytes=4)
    elif failure == "too-small":
        master.update(capacity_bytes=5, reserve_bytes=4)
    elif failure == "no-range":
        master["candidate_layer_ranges"] = []
    nodes = ([] if failure == "missing" else [master]) + [_node("worker", 32)]
    descriptor = _descriptor((2, 6))
    assert _assert_matches_oracle(descriptor, nodes)["admitted"]
    plan = _assert_order_invariance(descriptor, nodes, required_node_id="master")
    assert plan["admitted"] is False


@pytest.mark.parametrize("scores", [(1, 1, 1), (64, 1, 0.5)], ids=["equal", "skewed"])
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("prefer", [False, True])
def test_node_count_is_lexicographically_prior_to_work_cost(scores, distributed, prefer):
    descriptor = _descriptor((6, 1, 1))
    nodes = [_node(name, score) for name, score in zip(("master", "middle", "edge"), scores)]
    options = {"require_distributed": distributed, "prefer_all_workers": prefer,
               "required_node_id": "master"}
    plan = _assert_order_invariance(descriptor, nodes, **options)
    expected_count = 3 if prefer else (2 if distributed else 1)
    assert len(plan["assignments"]) == expected_count
    if (prefer and scores[0] == 64) or (not prefer and scores[0] == 1):
        alternatives = _enumerate_candidates(descriptor, nodes, **options)
        chosen = next(candidate for candidate in alternatives
                      if candidate.objective[4] == _identity(plan["assignments"]))
        assert any(len(candidate.stages) != expected_count
                   and candidate.objective[1] < chosen.objective[1]
                   for candidate in alternatives)


@pytest.mark.parametrize("tiebreak", ["memory", "score", "identity"])
def test_all_remaining_objective_terms_break_exact_work_ties(tiebreak):
    nodes = [
        _node("master", 2, candidate_layer_ranges=[[0, 1]]),
        _node("a", 0.25 if tiebreak == "score" else 1,
              capacity=4 if tiebreak == "memory" else 16,
              candidate_layer_ranges=[[1, 2]]),
        _node("z", 0.5 if tiebreak == "score" else 1,
              capacity=16, candidate_layer_ranges=[[1, 2]]),
    ]
    descriptor = _descriptor((1, 1))
    options = {"require_distributed": True, "required_node_id": "master"}
    candidates = _enumerate_candidates(descriptor, nodes, **options)
    assert len(candidates) == 2
    assert candidates[0].objective[1] == candidates[1].objective[1]
    if tiebreak in {"score", "identity"}:
        assert candidates[0].objective[2] == candidates[1].objective[2]
    if tiebreak == "identity":
        assert candidates[0].objective[:4] == candidates[1].objective[:4]
    plan = _assert_order_invariance(descriptor, nodes, all_orders=True, **options)
    # Identity is asserted here only after proving earlier costs tie exactly;
    # these halves and power-of-two capacities are exactly representable.
    assert plan["assignments"][1]["node_id"] == ("a" if tiebreak == "identity" else "z")


def test_identity_tiebreak_includes_layer_boundaries_after_node_id():
    descriptor = _descriptor((1, 2, 1))
    nodes = [_node("master", capacity=16), _node("worker", capacity=16)]
    options = {"require_distributed": True, "required_node_id": "master"}
    candidates = _enumerate_candidates(descriptor, nodes, **options)
    assert len(candidates) == 2
    assert candidates[0].objective[:4] == candidates[1].objective[:4]
    plan = _assert_order_invariance(descriptor, nodes, **options)
    assert _identity(plan["assignments"]) == (("master", 0, 1), ("worker", 1, 3))


@pytest.mark.parametrize("score", [-7, 0, 0.25, 0.5, 1])
def test_nonpositive_and_subunit_scores_have_unit_work_weight(score):
    plan = _assert_matches_oracle(
        _descriptor((1, 1, 2)), [_node("master", score), _node("worker", score)],
        require_distributed=True, required_node_id="master",
    )
    assert [item["layers_count"] for item in plan["assignments"]] == [2, 1]
    assert all(item["placement_weight"] == 1 for item in plan["assignments"])


@pytest.mark.parametrize("retained", [False, True])
def test_tied_output_and_retained_component_quotes_use_the_same_capacity_constraints(retained):
    nodes = [_node("master", candidate_layer_ranges=[[0, 1]]),
             _node("tail", candidate_layer_ranges=[[1, 2]])]
    if retained:
        nodes[0]["retained_component_bytes"] = 5
        nodes[1]["retained_component_bytes"] = 0
    plan = _assert_matches_oracle(
        _descriptor((1, 3), embedding=2, norm=1, tied=True), nodes,
        require_distributed=True, required_node_id="master",
    )
    assert [item["raw_weight_bytes"] for item in plan["assignments"]] == (
        [6, 3] if retained else [4, 6]
    )


@pytest.mark.parametrize("mode", ["head", "middle", "tail"])
@pytest.mark.parametrize("local_cut", [False, True])
def test_segment_position_contract_is_relaxed_only_by_local_cut(mode, local_cut):
    nodes = [
        _node("master", 4, segment_mode="head", candidate_layer_ranges=[[0, 1]]),
        _node("middle", 2, segment_mode=mode, candidate_layer_ranges=[[1, 3]],
              layer_budget=_layer_budget(2, local_cut=local_cut)),
        _node("tail", 1, segment_mode="tail", candidate_layer_ranges=[[3, 4]]),
    ]
    plan = _assert_matches_oracle(
        _descriptor((1, 2, 4, 1)), nodes,
        require_distributed=True, required_node_id="master", prefer_all_workers=True,
    )
    assert plan["admitted"] is (local_cut or mode == "middle")


def _seeded_case(seed):
    rng = random.Random(0x22C000 + seed)
    layer_count = 1 + seed % 5
    total_bytes = rng.choice((16, 32))
    cuts = (0, *sorted(rng.sample(range(1, total_bytes), layer_count - 1)), total_bytes)
    sizes = [end - start for start, end in zip(cuts, cuts[1:])]
    if layer_count > 1 and len(set(sizes)) == 1:
        sizes[0] -= 1
        sizes[-1] += 1
    descriptor = _descriptor(sizes, embedding=rng.randrange(5), norm=rng.randrange(3),
                             head=rng.randrange(5), other=rng.randrange(3))
    if seed % 9 == 0:
        descriptor["tie_word_embeddings"] = True
        descriptor["component_weight_bytes"].update(lm_head=0, other=0)
    names = ("master", "worker-z", "worker-a")[:1 + (seed // 5) % 3]
    spans = [[start, end] for start in range(layer_count)
             for end in range(start + 1, layer_count + 1)]
    nodes = []
    for name in names:
        node = _node(
            name, rng.choice((-2, 0, 0.25, 0.5, 1, 2, 4, 8)),
            capacity=rng.choice((8, 16, 32, 64, 128)), reserve=rng.randrange(8),
            multiplier=rng.choice((0.5, 0.75, 1.0, 1.25, 1.5, 2.0)),
        )
        if rng.random() < 0.4:
            ranges = [list(span) for span in spans if rng.random() < 0.5]
            rng.shuffle(ranges)
            if ranges and rng.random() < 0.5:
                ranges.append(list(ranges[0]))
            node["candidate_layer_ranges"] = ranges
        if rng.random() < 0.25:
            node["layer_ranges"] = [list(span) for span in spans if rng.random() < 0.5]
        if rng.random() < 0.45:
            node["layer_budget"] = _layer_budget(
                rng.randint(1, layer_count), local_cut=rng.choice((False, True)),
            )
        if rng.random() < 0.25:
            node["segment_mode"] = rng.choice(("head", "middle", "tail"))
        if rng.random() < 0.35:
            node["layer_weight_bytes"] = [rng.randint(1, 24) for _ in sizes]
        if rng.random() < 0.25:
            node["retained_component_bytes"] = rng.randrange(12)
        nodes.append(node)
    if seed % 13 == 0:
        nodes[0]["capacity_bytes"] = nodes[0]["reserve_bytes"]
    rng.shuffle(nodes)
    options = {
        "safety_margin": rng.choice((1.0, 1.25, 1.5)),
        "required_node_id": rng.choice((None, None, "master", names[-1], "missing")),
    }
    return descriptor, nodes, options


@pytest.mark.parametrize("seed", range(32), ids=lambda seed: f"seed-{seed:02d}")
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("prefer", [False, True])
def test_seeded_small_cases_match_complete_discrete_oracle(seed, distributed, prefer):
    # Power-of-two reference totals/capacities and dyadic scores make both cost
    # sums exact in binary. Nonuniform layer bytes, rounded admission and hard
    # range restrictions still vary independently across the deterministic cases.
    descriptor, nodes, options = _seeded_case(seed)
    assert 1 <= descriptor["total_layers"] <= 5
    assert 1 <= len(nodes) <= 3
    _assert_order_invariance(
        descriptor, nodes, require_distributed=distributed,
        prefer_all_workers=prefer, **options,
    )
