"""Only existing supply admission decides whether a reuse preference fits."""
import copy

import pytest

from model_supply import ModelSupplyError, plan_segment_supply
from model_supply_plan_reuse import plan_segment_supply_reuse


SOURCE_ID = "q35-9b-local"
SOURCE_SHA = "a" * 64
TOKENIZER_SHA = "b" * 64


def _case(master_capacity=165):
    sizes = [10] * 16 + [20] * 8 + [10] * 8
    common = {"model_id": SOURCE_ID, "source_model_id": SOURCE_ID,
              "source_model_sha256": SOURCE_SHA, "model_sha256": SOURCE_SHA,
              "tokenizer_sha256": TOKENIZER_SHA, "total_layers": 32,
              "hidden_size": 16, "pipeline_runtime_supported": True,
              "component_weight_bytes": {"embedding": 2, "final_norm": 2,
                                         "lm_head": 2, "other": 0}}
    descriptors = {"llama_cpp": {**common, "layer_weight_bytes": sizes},
                   "pytorch": {**common, "layer_weight_bytes": [40] * 32}}
    nodes = [{"node_id": name, "role": "master" if name == "master" else "client",
              "engine": "pytorch" if name == "master" else "llama_cpp",
              "capacity_bytes": capacity, "reserve_bytes": 1,
              "runtime_multiplier": 1.0, "score": score}
             for name, capacity, score in (("master", master_capacity, 100),
                                           ("surface", 300, 80), ("android", 180, 60))]
    records = copy.deepcopy(nodes)
    for record, start, end, digest, filename in (
        (records[1], 16, 24, "c" * 64, "segment.gguf"),
        (records[2], 24, 32, "d" * 64, "q35-9b-local-tail24-32.gguf"),
    ):
        artifact = {"layer_range": [start, end], "segment_mode": "tail" if end == 32 else "middle",
                    "model_id": filename, "artifact_sha256": digest,
                    "source_model_id": SOURCE_ID, "source_model_sha256": SOURCE_SHA,
                    "tokenizer_sha256": TOKENIZER_SHA, "hidden_size": 16,
                    "layer_weight_bytes": sizes[start:end]}
        record.update(layer_ranges=[[start, end]], layer_artifacts=[artifact],
                      models=[{"model_id": filename, "sha256": digest, "engine": "llama_cpp",
                               "format": "gguf", "revision": "local"}])
    return descriptors, records, nodes


def _select(case, *, planner=None, capabilities_by_node=None, **overrides):
    descriptors, records, nodes = case
    calls = []
    def callback(*, preferred_ranges):
        calls.append(copy.deepcopy(preferred_ranges))
        if planner is not None:
            return planner(preferred_ranges=preferred_ranges)
        return plan_segment_supply(descriptors, nodes, master_engine="pytorch",
                                   alignment=4, safety_margin=1.0, prefer_all_workers=True,
                                   preferred_ranges=preferred_ranges)
    arguments = {"planner": callback, "source_model_id": SOURCE_ID,
                 "source_model_sha256": SOURCE_SHA, "tokenizer_sha256": TOKENIZER_SHA,
                 "engine_by_node": {node["node_id"]: node["engine"] for node in nodes},
                 "capabilities_by_node": capabilities_by_node, "alignment": 4}
    arguments.update(overrides)
    return plan_segment_supply_reuse(descriptors, records, nodes, **arguments), calls


def _ranges(plan):
    return {item["node_id"]: [item["start_layer"], item["end_layer"]]
            for item in plan["assignments"]}


def _caps(case):
    return {record["node_id"]: {"engines": ["llama_cpp"],
            **copy.deepcopy({key: record[key] for key in ("models", "layer_ranges", "layer_artifacts")})}
            for record in case[1] if "layer_artifacts" in record}


def test_small_pt_head_rebalances_instead_of_locking_a_suboptimal_cached_tail():
    case = _case()
    before = copy.deepcopy(case)
    plan, calls = _select(case)
    assert plan["admitted"]
    assert _ranges(plan) == {"master": [0, 4], "surface": [4, 20], "android": [20, 32]}
    assert calls == [{}, {("android", ""): [[24, 32]], ("surface", ""): [[16, 24]]},
                     {("surface", ""): [[16, 24]]}, {("android", ""): [[24, 32]]}]
    assert plan["reuse_preferred_node_ids"] == []
    assert plan["target_preference"] == "unrestricted"
    fixed = plan_segment_supply(case[0], case[2], master_engine="pytorch", alignment=4,
                                safety_margin=1.0, prefer_all_workers=True,
                                preferred_ranges={"android": [[24, 32]]})
    assert fixed["admitted"]
    assert plan["placement_objective"]["weighted_work_cost"] < fixed["placement_objective"]["weighted_work_cost"]
    assert plan["loaded_ready"] is False and plan["supply_only"] is True
    assert plan["require_distributed"] is True
    assert all(item["required_bytes"] <= item["capacity_bytes"] for item in plan["assignments"])
    assert case == before


def test_joint_fixed_retains_three_segments_when_pt_head_fits():
    plan, calls = _select(_case(master_capacity=645))
    assert plan["admitted"]
    assert _ranges(plan) == {"master": [0, 16], "surface": [16, 24], "android": [24, 32]}
    assert len(calls) == 2 and calls[0] == {}
    assert plan["target_preference"] == "joint_fixed"
    assert plan["reuse_preferred_node_ids"] == ["android", "surface"]
    assert plan["loaded_ready"] is False


@pytest.mark.parametrize("field,value", [
    ("source_model_id", "another-model"), ("source_model_id", SOURCE_ID + " "),
    ("source_model_sha256", "e" * 64), ("source_model_sha256", SOURCE_SHA.upper()),
    ("tokenizer_sha256", "f" * 64), ("tokenizer_sha256", TOKENIZER_SHA.upper()),
])
def test_identity_is_exact_and_never_normalized(field, value):
    case = _case()
    for record in case[1][1:]:
        record["layer_artifacts"][0][field] = value
    plan, calls = _select(case)
    assert plan["admitted"] and calls == [{}]
    assert plan["target_preference"] == "unrestricted"
    assert plan["reuse_preferred_node_ids"] == []


@pytest.mark.parametrize("change", [
    {"layer_range": [17, 24]}, {"layer_range": [16, 23]},
    {"layer_range": [0, 32]}, {"layer_range": [-4, 24]},
    {"layer_range": [16, 36]}, {"layer_range": [24, 24]},
    {"layer_range": [True, 24]}, {"layer_range": [16.0, 24]},
    {"segment_mode": "tail"}, {"hidden_size": 32}, {"hidden_size": True},
    {"layer_weight_bytes": [0] * 8}, {"layer_weight_bytes": [True] * 8},
    {"layer_weight_bytes": [20] * 7}, {"layer_weight_bytes": [20.0] * 8},
    {"layer_weight_bytes": None}, {"artifact_sha256": "bad"}, {"model_id": "../segment.gguf"},
])
def test_unsafe_or_unaligned_asset_falls_back_to_original_planner(change):
    case = _case()
    case[1][2].pop("layer_artifacts")
    case[1][1]["layer_artifacts"][0].update(change)
    plan, calls = _select(case)
    assert plan["admitted"] and calls == [{}]
    assert plan["target_preference"] == "unrestricted"
    assert plan["loaded_ready"] is False


@pytest.mark.parametrize("field", ["source_model_id", "source_model_sha256", "tokenizer_sha256"])
def test_legacy_asset_without_full_identity_is_not_reused(field):
    case = _case()
    for record in case[1][1:]:
        record["layer_artifacts"][0].pop(field)
    assert _select(case)[1] == [{}]


@pytest.mark.parametrize("change", [
    {"engine": "pytorch", "format": "safetensors"},
    {"engine": "llama_cpp", "format": "safetensors"},
    {"engine": "pytorch", "format": "gguf"},
    {"engine": "llama_cpp", "format": "unknown"},
    {"sha256": "e" * 64}, {"model_id": SOURCE_ID},
])
def test_physical_model_identity_proves_engine_and_format(change):
    case = _case()
    for record in case[1][1:]:
        record["models"][0].update(change)
    assert _select(case)[1] == [{}]


def test_engine_installation_or_node_format_is_not_physical_identity_proof():
    case = _case()
    for record in case[1][1:]:
        record.pop("models")
        record.update(engines=["llama_cpp"], runtime_model_format="gguf")
    assert _select(case)[1] == [{}]


def test_duplicate_physical_identity_is_ambiguous():
    case = _case()
    for record in case[1][1:]:
        record["models"].append(copy.deepcopy(record["models"][0]))
    assert _select(case)[1] == [{}]


def test_range_must_be_advertised_with_its_artifact():
    case = _case()
    for record in case[1][1:]:
        record["layer_ranges"] = []
    assert _select(case)[1] == [{}]


def test_master_asset_is_never_fixed_and_fresh_nodes_own_capacity():
    case = _case()
    master = case[1][0]
    master.update(layer_ranges=[[0, 16]],
                  layer_artifacts=copy.deepcopy(case[1][1]["layer_artifacts"]),
                  models=copy.deepcopy(case[1][1]["models"]))
    master["layer_artifacts"][0].update(layer_range=[0, 16], segment_mode="head",
                                       layer_weight_bytes=[40] * 16)
    for record in case[1]:
        record["capacity_bytes"] = 1000000
    plan, calls = _select(case)
    assert _ranges(plan)["master"] == [0, 4]
    assert all("master" not in call for call in calls)
    assert _ranges(plan)["android"] == [20, 32]


def test_disconnected_inventory_and_target_engine_mismatch_are_not_reused():
    case = _case()
    del case[2][2:]
    case[2][1]["engine"] = "pytorch"
    assert _select(case)[1] == [{}]


def test_optional_capabilities_proof_is_checked_without_mutation():
    case = _case(master_capacity=645)
    caps = _caps(case)
    before = copy.deepcopy(caps)
    plan, _ = _select(case, capabilities_by_node=caps)
    assert plan["reuse_preferred_node_ids"] == ["android", "surface"]
    assert caps == before


@pytest.mark.parametrize("field", ["models", "layer_ranges", "layer_artifacts", "engines"])
def test_inconsistent_second_snapshot_disables_reuse(field):
    case = _case()
    caps = _caps(case)
    for body in caps.values():
        body[field] = []
    assert _select(case, capabilities_by_node=caps)[1] == [{}]


def test_failed_snapshot_map_does_not_fall_back_to_record_models():
    assert _select(_case(), capabilities_by_node={})[1] == [{}]


def test_pytorch_asset_requires_pytorch_physical_identity():
    case = _case(master_capacity=645)
    case[2][2].update(engine="pytorch", capacity_bytes=500)
    case[1][2]["models"][0].update(engine="pytorch", format="safetensors")
    case[1][2]["layer_artifacts"][0]["layer_weight_bytes"] = [40] * 8
    plan, _ = _select(case)
    assert plan["target_preference"] == "joint_fixed"
    assert plan["reuse_preferred_node_ids"] == ["android", "surface"]
    assert plan["assignments"][-1]["engine"] == "pytorch"


def test_greedy_order_uses_actual_asset_bytes_and_is_deterministic():
    case = _case()
    case[1][2]["layer_artifacts"][0]["layer_weight_bytes"] = [30] * 8
    plan, calls = _select(case)
    assert calls[0] == {}
    assert calls[2] == {("android", ""): [[24, 32]]}
    assert calls[3] == {("surface", ""): [[16, 24]]}
    assert plan["reuse_preferred_node_ids"] == []
    reversed_case = (case[0], list(reversed(case[1])), case[2])
    reversed_plan, reversed_calls = _select(reversed_case)
    assert reversed_calls == calls and reversed_plan == plan


def test_rejected_preferences_are_bounded_and_preserve_original_rejection():
    rejected = {"admitted": False, "reason_code": "real-capacity-failure", "assignments": []}
    before = copy.deepcopy(rejected)
    plan, calls = _select(_case(), planner=lambda **kwargs: rejected)
    assert calls == [{}]
    assert plan["reason_code"] == "real-capacity-failure" and not plan["admitted"]
    assert plan["target_preference"] == "unrestricted" and plan["loaded_ready"] is False
    assert rejected == before


def test_admitted_plan_that_skips_a_fixed_worker_is_not_accepted():
    case = _case()
    reduced_nodes = copy.deepcopy(case[2])
    reduced_nodes[1]['capacity_bytes'] = 0
    reduced_nodes[2]['capacity_bytes'] = 400
    baseline = plan_segment_supply(case[0], reduced_nodes, master_engine="pytorch",
                                   alignment=4, safety_margin=1.0, prefer_all_workers=True)
    def ignores_preferences(**kwargs):
        return baseline
    plan, calls = _select(case, planner=ignores_preferences)
    assert plan["target_preference"] == "unrestricted"
    assert calls[0] == {}
    assert baseline.get("target_preference") is None


def test_full_model_or_unaligned_candidate_result_falls_back():
    case = _case()
    safe = plan_segment_supply(case[0], case[2], master_engine="pytorch",
                               alignment=4, safety_margin=1.0, prefer_all_workers=True)
    unsafe = {**safe, "assignments": [{"node_id": "master", "engine": "pytorch",
                                       "start_layer": 0, "end_layer": 32}]}
    def callback(*, preferred_ranges):
        return unsafe if preferred_ranges else safe
    plan, calls = _select(case, planner=callback)
    assert plan["target_preference"] == "unrestricted" and calls[0] == {}
    assert len(plan["assignments"]) == 3


def test_planner_errors_propagate_instead_of_weakening_admission():
    def broken(**kwargs):
        raise ModelSupplyError("SUPPLY_TEST_GATE", "capacity gate failed")
    with pytest.raises(ModelSupplyError) as error:
        _select(_case(), planner=broken)
    assert error.value.code == "SUPPLY_TEST_GATE"


def test_invalid_alignment_stays_owned_by_existing_planner():
    case = _case()
    def callback(*, preferred_ranges):
        return plan_segment_supply(case[0], case[2], master_engine="pytorch",
                                   alignment=3, preferred_ranges=preferred_ranges)
    with pytest.raises(ModelSupplyError) as error:
        _select(case, planner=callback, alignment=3)
    assert error.value.code == "SUPPLY_ALIGNMENT_INVALID"


def test_cached_three_stage_chain_cannot_override_fewest_nodes_configuration():
    case = _case(master_capacity=645)
    def planner(*, preferred_ranges):
        return plan_segment_supply(case[0], case[2], master_engine="pytorch", alignment=4,
                                   safety_margin=1.0, prefer_all_workers=False,
                                   preferred_ranges=preferred_ranges)
    plan, calls = _select(case, planner=planner)
    assert plan["admitted"] and plan["participating_node_count"] == 2
    assert plan["target_preference"] == "unrestricted"
    assert plan["reuse_preferred_node_ids"] == []
    assert calls[0] == {}


def test_equal_quality_single_cached_range_still_gets_reused():
    case = _case()
    android = case[1][2]
    android["layer_ranges"] = [[20, 32]]
    artifact = android["layer_artifacts"][0]
    artifact.update(layer_range=[20, 32], layer_weight_bytes=case[0]["llama_cpp"]["layer_weight_bytes"][20:32])
    plan, calls = _select(case)
    assert _ranges(plan) == {"master": [0, 4], "surface": [4, 20], "android": [20, 32]}
    assert plan["reuse_preferred_node_ids"] == ["android"]
    assert plan["target_preference"] == "greedy_fixed"
    assert len(calls) <= 4


# ---------------------------------------------------------------------------
# 同机附加执行器（LOCAL-EXECUTORS-01 / 审计 A8 + B1）
# ---------------------------------------------------------------------------
def _cpu_reuse_case(master_capacity=645):
    """master 主段 + 本机 CPU 段（`executor_id="cpu"`）+ android 尾段。

    CPU 段与主段**共用 `node_id="master"`**，这是 A8 的目标形态：`_Candidate`、
    `engine_by_node`、`_preserves` 三处都必须按 `(node_id, executor_id)` 区分，
    否则本机那份已验证的 GGUF 段永远进不了复用候选 / 方案恒被判不保留。
    """
    descriptors, records, nodes = _case(master_capacity=master_capacity)
    del nodes[1]
    del records[1]  # 去掉 surface，只留 master 主段 + android 尾段
    sizes = descriptors["llama_cpp"]["layer_weight_bytes"]
    cpu_node = {"node_id": "master", "executor_id": "cpu", "role": "master",
                "engine": "llama_cpp", "capacity_bytes": 1000, "reserve_bytes": 1,
                "runtime_multiplier": 1.0, "score": 100}
    nodes.append(cpu_node)
    cpu_record = copy.deepcopy(cpu_node)
    cpu_record.update(
        layer_ranges=[[8, 24]],
        layer_artifacts=[{
            "layer_range": [8, 24], "segment_mode": "middle", "model_id": "mid8-24.gguf",
            "artifact_sha256": "e" * 64, "source_model_id": SOURCE_ID,
            "source_model_sha256": SOURCE_SHA, "tokenizer_sha256": TOKENIZER_SHA,
            "hidden_size": 16, "layer_weight_bytes": sizes[8:24],
        }],
        models=[{"model_id": "mid8-24.gguf", "sha256": "e" * 64,
                 "engine": "llama_cpp", "format": "gguf", "revision": "local"}],
    )
    records.append(cpu_record)
    return descriptors, records, nodes


def _cpu_engines(nodes):
    return {(node["node_id"], str(node.get("executor_id") or "")): node["engine"]
            for node in nodes}


def _segment(node_id, executor_id, engine, start, end):
    item = {"node_id": node_id, "engine": engine, "start_layer": start, "end_layer": end}
    if executor_id:
        item["executor_id"] = executor_id
    return item


def test_candidates_include_the_local_cpu_executor_artifact():
    """A8 问题 2：`_candidates` 不能再按 `node_id == "master"` 跳过本机 CPU 段。"""
    from model_supply_plan_reuse import _candidates
    descriptors, records, nodes = _cpu_reuse_case()
    candidates, total = _candidates(
        descriptors, records, nodes, source_model_id=SOURCE_ID,
        source_model_sha256=SOURCE_SHA, tokenizer_sha256=TOKENIZER_SHA,
        engine_by_node=_cpu_engines(nodes), capabilities_by_node=None, alignment=4,
    )
    assert total == 32
    assert any((item.node_id, item.executor_id, item.start, item.end)
               == ("master", "cpu", 8, 24) for item in candidates)


def test_local_cpu_segment_is_reused_by_executor_identity():
    """A8 问题 1 + 3：`master` 出现两次合法、引擎按执行器取、偏好键是身份元组。"""
    descriptors, records, nodes = _cpu_reuse_case()
    objective = {"node_count": 3, "weighted_work_cost": 1.0,
                 "memory_pressure_cost": 2.0, "score_total": 3.0}
    policy = {"prefer_all_workers": True, "work_basis": "x", "objective": "y"}
    common = {"admitted": True, "placement_policy": policy,
              "placement_objective": objective}
    baseline = {**copy.deepcopy(common), "assignments": [
        _segment("master", "", "pytorch", 0, 8),
        _segment("master", "cpu", "llama_cpp", 8, 16),
        _segment("android", "", "llama_cpp", 16, 32),
    ]}
    pinned = {**copy.deepcopy(common), "assignments": [
        _segment("master", "", "pytorch", 0, 8),
        _segment("master", "cpu", "llama_cpp", 8, 24),
        _segment("android", "", "llama_cpp", 24, 32),
    ]}
    calls = []

    def planner(*, preferred_ranges, release_candidates=False):
        calls.append(copy.deepcopy(preferred_ranges))
        return copy.deepcopy(pinned if preferred_ranges else baseline)

    plan = plan_segment_supply_reuse(
        descriptors, records, nodes, planner=planner,
        source_model_id=SOURCE_ID, source_model_sha256=SOURCE_SHA,
        tokenizer_sha256=TOKENIZER_SHA, engine_by_node=_cpu_engines(nodes),
        alignment=4,
    )
    assert plan["admitted"] and plan["target_preference"] == "joint_fixed"
    assert plan["reuse_preferred_node_ids"] == ["android", "master#cpu"]
    assert calls[0] == {}
    assert calls[1] == {("android", ""): [[24, 32]],
                        ("master", "cpu"): [[8, 24]]}


def test_missing_local_executor_releases_stale_locks_and_is_chosen():
    """B1：baseline 可行但缺少配置了的 CPU 段 ⇒ 放开陈旧切点锁并取优。"""
    descriptors, records, nodes = _cpu_reuse_case()
    # 模拟上一轮供给写下的切点锁（两段恰好占满 [0,32)）。
    for record in records:
        if record.get("executor_id"):
            continue
        record["candidate_layer_ranges"] = (
            [[0, 4]] if record["node_id"] == "master" else [[4, 8]])
    objective = {"node_count": 3, "weighted_work_cost": 9.0,
                 "memory_pressure_cost": 9.0, "score_total": 1.0}
    baseline = {"admitted": True, "placement_objective": objective, "assignments": [
        _segment("master", "", "pytorch", 0, 16),
        _segment("android", "", "llama_cpp", 16, 32),
    ]}
    released = {"admitted": True, "placement_objective": objective, "assignments": [
        _segment("master", "", "pytorch", 0, 8),
        _segment("master", "cpu", "llama_cpp", 8, 24),
        _segment("android", "", "llama_cpp", 24, 32),
    ]}
    calls = []

    def planner(*, preferred_ranges, release_candidates=False):
        calls.append((copy.deepcopy(preferred_ranges), release_candidates))
        return copy.deepcopy(released if release_candidates else baseline)

    plan = plan_segment_supply_reuse(
        descriptors, records, nodes, planner=planner,
        source_model_id=SOURCE_ID, source_model_sha256=SOURCE_SHA,
        tokenizer_sha256=TOKENIZER_SHA, engine_by_node=_cpu_engines(nodes),
        alignment=4,
    )
    assert calls[0] == ({}, False)
    assert calls[1] == ({}, True)
    assert plan["target_preference"] == "released_candidates"
    assert [(item["node_id"], item.get("executor_id", ""))
            for item in plan["assignments"]] == [
        ("master", ""), ("master", "cpu"), ("android", "")]
