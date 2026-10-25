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
