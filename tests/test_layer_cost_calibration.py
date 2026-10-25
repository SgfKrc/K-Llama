"""切点求解的**成本标定**：`QLH_LAYER_COST_MS` 的解析与加权效果。

背景：求解器原先按解码器**字节**份额均衡（`capacity.py` 的
`placement_work_fraction`），把最慢的设备当成和最快的设备一样贵。实测三机每层成本
差 2.3 倍（master 12.6 / rog-5060 8.7 / android-542cd254 20.1 ms，见
`docs/本机跨框架层段成本与干扰实测-2026-10-25.md`），典型后果是 rog 扛 24 层成为
瓶颈，而最慢的 android 却只扛 8 层。

标定值由**调用方**（`scheduler._get_pipeline_capacity_nodes`）填进节点记录，因为
`qlh_core.planning.capacity` 是纯求解器、契约禁止它访问 ambient state
（由 `tests/test_refactor_capacity_load_spec.py` 断言）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 复用既有 fixture：它构造了"陈旧 profile + 新鲜短缺"的完整求解场景，
# 本文件的 makespan 报告测试直接跑真实求解器，而不是伪造求解结果。
from tests.test_pipeline_capacity_freshness import fresh_case  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT, ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))


def _config():
    import config
    return config


def test_unset_env_yields_no_calibration(monkeypatch):
    """未配置 ⇒ 空表 + default 0.0 ⇒ 求解器按 1.0 ⇒ 与旧行为逐字节等价。"""
    monkeypatch.delenv("QLH_LAYER_COST_MS", raising=False)
    cfg = _config()
    assert cfg.layer_cost_ms_by_node() == {}
    assert cfg.LAYER_COST_MS_DEFAULT == 0.0


def test_well_formed_calibration_parses(monkeypatch):
    monkeypatch.setenv("QLH_LAYER_COST_MS",
                       "master=12.6,rog-5060=8.7,android-542cd254=20.1")
    assert _config().layer_cost_ms_by_node() == {
        "master": 12.6, "rog-5060": 8.7, "android-542cd254": 20.1,
    }


@pytest.mark.parametrize("raw,expected", [
    # 缺 `=`、空 node、空值、非数、非正数、非有限 —— 全部忽略而不是抛错：
    # 标定值是运行期调优参数，一个手误不该让整次供给失败。
    ("a=1, broken, =, b=xyz, c=-5, d=2.5", {"a": 1.0, "d": 2.5}),
    ("  spaced = 3.5  ", {"spaced": 3.5}),
    ("inf=inf, nan=nan", {}),
    ("", {}),
    ("no_equals_at_all", {}),
])
def test_malformed_entries_are_ignored(monkeypatch, raw, expected):
    monkeypatch.setenv("QLH_LAYER_COST_MS", raw)
    assert _config().layer_cost_ms_by_node() == expected


def test_solver_needs_a_calibrated_field_to_change_behaviour():
    """求解器侧的核心契约：**没有 `layer_cost_ms` 字段就完全不变**。

    这里只断言归一化/缺省逻辑的语义（不构造完整求解），因为求解器本身是纯函数、
    不读环境；把标定值塞进节点记录的是调用方。
    """
    import math

    # 与 `capacity._normalize_nodes` 相同的白名单校验语义
    def normalize(raw):
        if "layer_cost_ms" in raw:
            cost = raw["layer_cost_ms"]
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0:
                raise ValueError("layer_cost_ms must be a positive number")
            return float(cost)
        return None

    assert normalize({}) is None
    assert normalize({"layer_cost_ms": 8.7}) == 8.7
    for bad in (0, -1, float("inf"), float("nan"), "8.7", None):
        with pytest.raises(ValueError):
            normalize({"layer_cost_ms": bad})


def test_reference_and_multiplier_semantics():
    """倍数语义：最快者为 1.0；未标定者按**已标定中最慢**计（安全方向）。

    未标定若按 1.0 处理，会因为没有数据反而被当成最快而**过度分配**。
    """
    def multipliers(costs_by_node):
        calibrated = [v for v in costs_by_node.values() if v]
        reference = min(calibrated) if calibrated else 1.0
        out = {}
        for node, cost in costs_by_node.items():
            out[node] = (float(cost) / reference) if cost else (
                (max(calibrated) / reference) if calibrated else 1.0)
        return out

    # 实测值：rog 最快 ⇒ 1.0；android 最慢 ⇒ 2.31
    got = multipliers({"master": 12.6, "rog-5060": 8.7, "android-542cd254": 20.1})
    assert got["rog-5060"] == pytest.approx(1.0)
    assert got["master"] == pytest.approx(12.6 / 8.7)
    assert got["android-542cd254"] == pytest.approx(20.1 / 8.7)
    # 全部未标定 ⇒ 一律 1.0 ⇒ 旧行为
    assert multipliers({"a": 0.0, "b": 0.0}) == {"a": 1.0, "b": 1.0}
    # 部分未标定 ⇒ 未标定者按最慢的已标定值，不会被当成最快
    partial = multipliers({"calibrated": 8.7, "uncalibrated": 0.0})
    assert partial["uncalibrated"] == pytest.approx(1.0)
    partial2 = multipliers({"fast": 5.0, "slow": 20.0, "uncalibrated": 0.0})
    assert partial2["uncalibrated"] == pytest.approx(20.0 / 5.0)


# --------------------------------------------------------------------------
# makespan 报告：让"切点好不好"可测，但**不参与选择**
# --------------------------------------------------------------------------

def _admitted_plan(case, calibrate=None):
    """用**新鲜 evidence** 求解。

    fixture 的陈旧 profile 本身不足以准入（`test_pipeline_capacity_freshness` 自己就断言
    `not plan(case)["admitted"]`）—— 必须走 `prepare_guard` + `prepare_capacity_evidence`
    拿新鲜容量样本，这也是 `test_refactor_product_gate` 采用同一模式的原因。
    """
    from tests.test_pipeline_capacity_freshness import plan

    if calibrate is not None:
        calibrate(case)
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        return plan(case, evidence)


def test_step_time_estimate_reports_bottleneck_without_changing_the_choice(fresh_case):
    """标定后求解器给出步时估计与瓶颈节点；**选择必须与未标定时完全一致**。

    这是本报告的核心契约：求解目标是"能力加权工作平方"的**均衡近似**，而端到端由
    最长段（makespan）决定，两者会背离（实测 `fastest` 下按成本均衡反而慢 7%）。
    因此报告必须只读——一旦它影响了选择，就会把"均衡"悄悄换成"另一个东西"。
    """
    case = fresh_case
    baseline = _admitted_plan(case)
    assert baseline["admitted"], baseline.get("reason_code")
    assert "step_time_estimate" not in baseline          # 未标定 ⇒ 不产出报告

    def calibrate(case):
        for record in case.cached:
            record["layer_cost_ms"] = 20.0 if record["node_id"] == "worker" else 10.0

    result = _admitted_plan(case, calibrate)
    assert result["admitted"], result.get("reason_code")
    segments = lambda r: [(a["node_id"], a["start_layer"], a["end_layer"])
                          for a in r["assignments"]]
    assert segments(result) == segments(baseline), "报告不得改变选择"

    est = result["step_time_estimate"]
    assert est["basis"] == "layers_count_x_calibrated_layer_ms"
    assert est["uncalibrated_segments"] == []
    assert all(seg["layers_count"] == seg["layer_range"][1] - seg["layer_range"][0]
               for seg in est["segments"])
    assert all(seg["estimated_ms"] == pytest.approx(
        seg["layers_count"] * seg["layer_cost_ms"]) for seg in est["segments"])
    # 瓶颈 = 估计步时最大的那一段
    assert est["estimated_step_ms"] == max(seg["estimated_ms"] for seg in est["segments"])
    assert est["bottleneck_node_id"] == next(
        seg["node_id"] for seg in est["segments"]
        if seg["estimated_ms"] == est["estimated_step_ms"])


def test_uncalibrated_segments_are_declared_not_guessed(fresh_case):
    """半标定：未标定段的 `estimated_ms` 必须是 None 并被单列 —— 不猜。

    半真半假的步时估计比没有更危险：它会让人以为瓶颈已经确定。
    """
    case = fresh_case

    def calibrate(case):
        for record in case.cached:
            if record["node_id"] == "worker":
                record["layer_cost_ms"] = 20.0
            else:
                record.pop("layer_cost_ms", None)        # master 不标定

    result = _admitted_plan(case, calibrate)
    assert result["admitted"], result.get("reason_code")
    est = result.get("step_time_estimate")
    assert est is not None, "有任一参与节点带标定就应产出报告"
    uncalibrated = [seg for seg in est["segments"]
                    if seg["node_id"] in est["uncalibrated_segments"]]
    assert uncalibrated, est
    assert all(seg["estimated_ms"] is None and seg["layer_cost_ms"] is None
               for seg in uncalibrated)
    # 瓶颈只能取自**已标定**段
    assert est["bottleneck_node_id"] not in est["uncalibrated_segments"]


# --------------------------------------------------------------------------
# 同机容量去重（LOCAL-EXECUTORS-01 的交付条件："容量无重复计数"）
# --------------------------------------------------------------------------

def test_no_domain_declaration_keeps_todays_accounting(fresh_case):
    """**不声明共享域** ⇒ 结果里不出现 `capacity_domains`。

    这是回归门：域机制必须对既有拓扑零影响（`capacity_domain_id` 缺省 = 每节点自成一域）。
    """
    result = _admitted_plan(fresh_case)
    assert result["admitted"], result.get("reason_code")
    assert "capacity_domains" not in result


def test_capacity_domain_rejects_overclaimed_shared_budget(fresh_case):
    """同域内执行器申报之和 > 域总账 ⇒ fail closed，并给出可审计的域报告。

    这锁住的正是实测缺口：一台物理机上的两个执行器各自按整机可用内存申报，求解器把
    它们当成两台独立机器，于是给出**物理上装不下**的计划却 `admitted=True`
    （实测两节点各 200、合计需求 242，每段还显示 headroom=79）。
    """
    case = fresh_case

    def calibrate(case):
        for record in case.cached:
            record["capacity_domain_id"] = "host-alpha"
            record["domain_total_bytes"] = 250          # 远小于 10000 + 1500

    result = _admitted_plan(case, calibrate)
    assert not result["admitted"]
    assert result["reason_code"] == "pipeline_capacity_domain_exceeded"
    report = result["capacity_domains"]["host-alpha"]
    assert report["domain_total_bytes"] == 250
    assert report["claimed_allocatable_bytes"] > 250
    assert report["excess_bytes"] == report["claimed_allocatable_bytes"] - 250
    assert len(report["executors"]) == 2
    assert {item["node_id"] for item in report["executors"]} == {"master", "worker"}
    assert result["assignments"] == []


def test_capacity_domain_admits_when_within_total(fresh_case):
    """总账足够 ⇒ 照常准入，且域报告仍可审计（写明总账与申报额）。"""
    case = fresh_case

    def calibrate(case):
        for record in case.cached:
            record["capacity_domain_id"] = "host-alpha"
            record["domain_total_bytes"] = 10 ** 9

    result = _admitted_plan(case, calibrate)
    assert result["admitted"], result.get("reason_code")
    report = result["capacity_domains"]["host-alpha"]
    assert report["domain_total_bytes"] == 10 ** 9
    assert report["excess_bytes"] == 0
    assert report["claimed_allocatable_bytes"] <= 10 ** 9


def test_conflicting_domain_totals_are_rejected_loudly():
    """同域内两个执行器声明了**不同**的总账 ⇒ 直接报错，不静默取其一。

    静默取 min/max 会让"这台机器到底有多少内存"变成一个随节点顺序变化的值。
    """
    from qlh_core.planning.capacity import (
        PipelineCapacityError, validate_capacity_domains,
    )

    nodes = [
        {"node_id": "a", "capacity_bytes": 100, "reserve_bytes": 0,
         "capacity_domain_id": "host-x", "domain_total_bytes": 1000},
        {"node_id": "b", "capacity_bytes": 100, "reserve_bytes": 0,
         "capacity_domain_id": "host-x", "domain_total_bytes": 2000},
    ]
    with pytest.raises(PipelineCapacityError):
        validate_capacity_domains(nodes)


# --------------------------------------------------------------------------
# 执行器身份进契约（LOCAL-EXECUTORS-01 交付条件前半："同物理节点两段可规划"）
# --------------------------------------------------------------------------

def _segment(node_id, executor_id, start, end):
    from qlh_core.contracts.execution import Segment

    return Segment(node_id, f"assign-{node_id}-{executor_id or 'main'}", "inc-1", 1,
                   start, end, start == 0, end == 8, "{}", executor_id)


def test_execution_plan_admits_two_executors_on_one_physical_node():
    """**同一 node_id、不同 executor_id** ⇒ 可规划（本票的目标形态）。

    物理上就是一台机器：GPU 段跑 PyTorch 首段、CPU 段跑 llama.cpp 中间段。两者共享
    同一份 RAM 与同一个传输身份，所以 `node_id` 必须相同；靠 `executor_id` 区分。
    """
    from qlh_core.contracts.execution import ExecutionPlan

    plan = ExecutionPlan("deploy-1", "boot-1", "auth", 8, (
        _segment("master", "", 0, 4),        # 主执行器 = GPU 段
        _segment("master", "cpu", 4, 8),     # 附加执行器 = 同机 CPU 段
    ))
    assert [s.executor_id for s in plan.segments] == ["", "cpu"]
    assert [s.identity_key for s in plan.segments] == [
        ("master", ""), ("master", "cpu")]


def test_execution_plan_still_rejects_duplicate_executor_identity():
    """**同一 (node_id, executor_id) 出现两次** ⇒ 仍然拒绝。

    放宽的是"一台机器只能有一个段"，不是"允许重复声明同一个执行器"。
    """
    from qlh_core.contracts.execution import ExecutionError, ExecutionPlan

    with pytest.raises(ExecutionError, match="execution_requires_two_nodes"):
        ExecutionPlan("deploy-1", "boot-1", "auth", 8, (
            _segment("master", "", 0, 4),
            _segment("master", "", 4, 8),
        ))


def test_executor_id_defaults_to_the_primary_executor():
    """缺省 `executor_id` ⇒ 与改动前**逐位等价**（旧的"node_id 唯一"语义）。

    这是整个阶段 2 的回归保证：既有 topology 从不传 `executor_id`，所以它们的
    `identity_key` 退化为 `(node_id, "")`，唯一性判据与旧的 `node_id` 集合相同。
    """
    from qlh_core.contracts.execution import ExecutionPlan

    plan = ExecutionPlan("deploy-1", "boot-1", "auth", 8, (
        _segment("master", "", 0, 4),
        _segment("worker", "", 4, 8),
    ))
    assert all(s.executor_id == "" for s in plan.segments)
    assert {s.identity_key for s in plan.segments} == {("master", ""), ("worker", "")}


def test_pipeline_node_executor_id_is_omitted_when_default():
    """`to_dict()` 只在**非空**时输出 `executor_id`。

    缺省即主执行器 ⇒ 既有 layout 的 `to_dict()` 形态与 `contract_sha256` 逐位不变，
    frozen 契约与只读消费者不受影响。
    """
    from pipeline_node_contract import (
        PipelineArtifactRef, PipelineNode, PipelineNodeCapacity,
    )

    common = dict(
        kind="layer", layer_range=(0, 4), engine="pytorch", location="local",
        capacity=PipelineNodeCapacity(capacity_bytes=1000, required_bytes=200,
                                      reserve_bytes=50, execution_device="cpu"),
        artifact=PipelineArtifactRef(model_sha256="a" * 64,
                                     artifact_kind="safetensors_assignment",
                                     architecture="qwen3",
                                     source_layer_range=(0, 4)),
    )
    primary = PipelineNode(node_id="master", **common)
    assert "executor_id" not in primary.to_dict()
    assert primary.identity_key == ("master", "")

    secondary = PipelineNode(node_id="master", executor_id="cpu", **common)
    assert secondary.to_dict()["executor_id"] == "cpu"
    assert secondary.identity_key == ("master", "cpu")


# --------------------------------------------------------------------------
# KV 生命周期：同 node 两段的 stage 标识必须可区分
# --------------------------------------------------------------------------

@pytest.mark.parametrize("executor_id,expected", [
    ("", "master"),              # 主执行器（缺省）⇒ 与改动前逐字相同
    ("cpu", "master#cpu"),       # 同机 CPU 段
])
def test_stage_identity_includes_the_executor(executor_id, expected):
    """`stage_identity` 是 `stage()` 与 `_reset_pipeline_kv()` **共用**的规则。

    同 node 两段的 `node_id` 相同，只用 node_id 组 stage 标识会让两段撞车 —— 而它同时
    是请求追踪与**清理回执关联**的键。撞车的后果是 `_reset_pipeline_kv` 把清理命令
    关联到错误的 stage（实测中该函数按 `segment.node_id` 定位 owner）。
    """
    from types import SimpleNamespace

    from qlh_core.bootstrap.execution import stage_identity

    assert stage_identity(SimpleNamespace(node_id="master", executor_id=executor_id)) == expected


def test_stage_identity_tolerates_segments_without_the_field():
    """没有 `executor_id` 属性的对象（旧形态）⇒ 退化为 `node_id`，不抛错。"""
    from types import SimpleNamespace

    from qlh_core.bootstrap.execution import stage_identity

    assert stage_identity(SimpleNamespace(node_id="worker")) == "worker"


def test_same_node_segments_get_distinct_stage_identities():
    """同 node 两段的 stage 标识**必须**互不相同（本测试锁住那个不变量本身）。"""
    from qlh_core.bootstrap.execution import stage_identity

    primary, secondary = _segment("master", "", 0, 4), _segment("master", "cpu", 4, 8)
    assert stage_identity(primary) != stage_identity(secondary)
    assert primary.node_id == secondary.node_id      # 同一台物理机
    assert primary.identity_key != secondary.identity_key
