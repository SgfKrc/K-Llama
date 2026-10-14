"""22c 缺口①：把在线心跳 RTT 接进切点求解的数据入口（2026-10-14）。

只测纯计算部分（`rtt_ms_from_heartbeat` / `SegmentProfile.with_rtt` /
`from_device_profile` 的 RTT 字段识别），不依赖任何网络模块。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relay_cut_objective import (  # noqa: E402
    SegmentEconomics,
    SegmentProfile,
    rtt_ms_from_heartbeat,
)


# ---------- rtt_ms_from_heartbeat ----------

def test_empty_samples_returns_zero():
    assert rtt_ms_from_heartbeat([]) == 0.0


def test_single_sample_returns_itself():
    assert rtt_ms_from_heartbeat([19.0]) == 19.0


def test_median_odd_count():
    # 5 个样本的中位就是第 3 个
    assert rtt_ms_from_heartbeat([3.0, 19.0, 43.0, 19.0, 3.0]) == 19.0


def test_median_even_count_interpolates():
    # 4 个样本 [3,3,19,43] ⇒ 中位在 pos=1.5 ⇒ 3 + (19-3)*0.5 = 11
    assert rtt_ms_from_heartbeat([3.0, 3.0, 19.0, 43.0]) == 11.0


def test_p95_percentile():
    values = [float(i) for i in range(1, 21)]  # 1..20
    # pos = 0.95*19 = 18.05 ⇒ 19 + (20-19)*0.05 = 19.05
    assert rtt_ms_from_heartbeat(values, percentile=0.95) == 19.05


def test_negative_samples_are_filtered():
    assert rtt_ms_from_heartbeat([-5.0, 19.0]) == 19.0
    assert rtt_ms_from_heartbeat([-5.0, -1.0]) == 0.0


def test_percentile_clamped():
    assert rtt_ms_from_heartbeat([1.0, 3.0], percentile=2.0) == 3.0
    assert rtt_ms_from_heartbeat([1.0, 3.0], percentile=-1.0) == 1.0


# ---------- SegmentProfile.with_rtt ----------

def test_with_rtt_replaces_only_rtt_and_keeps_frozen_original():
    base = SegmentProfile(
        node_id="surface",
        capacity_bytes=8 * 1024 ** 3,
        ms_per_layer_decode=11.75,
        rtt_ms=3.0,          # 人工默认（MEASURED_TABLET_RTT_MS）
        bandwidth_mbps=150.0,
    )
    updated = base.with_rtt(19.0, source="heartbeat")

    # 只改了 rtt 与 source
    assert updated.rtt_ms == 19.0
    assert updated.source == "heartbeat"
    assert updated.node_id == base.node_id
    assert updated.ms_per_layer_decode == base.ms_per_layer_decode
    assert updated.capacity_bytes == base.capacity_bytes
    # frozen ⇒ 原对象不变
    assert base.rtt_ms == 3.0
    assert base.source == "declared"


def test_with_rtt_clamps_negative_and_keeps_source_when_omitted():
    base = SegmentProfile(node_id="y700", rtt_ms=19.0, source="measured-constant")
    assert base.with_rtt(-1.0).rtt_ms == 0.0
    assert base.with_rtt(43.0).source == "measured-constant"


def test_with_rtt_changes_round_trip_tax():
    """接线的意义：分出去要付的 `cost_ms` 里含固定往返税 = STAGE_ROUND_TRIPS(2) × rtt_ms。"""
    base = SegmentProfile(node_id="x", rtt_ms=3.0)
    econ_before = SegmentEconomics(n_layers=4, c_master_ms=3.06, c_device_ms=11.75,
                                   rtt_ms=base.rtt_ms)
    econ_after = SegmentEconomics(n_layers=4, c_master_ms=3.06, c_device_ms=11.75,
                                  rtt_ms=base.with_rtt(19.0).rtt_ms)
    assert econ_after.round_trips == econ_before.round_trips == 2
    # 段本身的计算没变，变的只是往返税 ⇒ cost_ms 必须变大（等价于净亏判据更严）
    assert econ_after.cost_ms > econ_before.cost_ms
    # 且 19ms 那一档仍"不值得"（c_device 11.75 > c_master 3.06 ⇒ 任何 n 都不值得）
    assert econ_after.min_layers is None


# ---------- from_device_profile 认分位字段 ----------

def test_from_device_profile_reads_rtt_ms():
    profile = SegmentProfile.from_device_profile(
        {"ram": {"available_gb": 8.0}, "network": {"rtt_ms": 19.0}}
    )
    assert profile.rtt_ms == 19.0


def test_from_device_profile_reads_rtt_ms_p50_from_heartbeat_window():
    """`network_path.HeartbeatQualityWindow` 风格的分位字段。"""
    profile = SegmentProfile.from_device_profile(
        {"ram": {"available_gb": 8.0}, "network": {"rtt_ms_p50": 21.0, "rtt_ms_p95": 43.0}}
    )
    assert profile.rtt_ms == 21.0


def test_from_device_profile_falls_back_to_p95_then_zero():
    p95_only = SegmentProfile.from_device_profile(
        {"network": {"rtt_ms_p95": 43.0}}
    )
    assert p95_only.rtt_ms == 43.0
    # 全缺 ⇒ 如实置 0（不猜）
    assert SegmentProfile.from_device_profile({"network": {}}).rtt_ms == 0.0


def test_explicit_rtt_ms_wins_over_percentile_fields():
    profile = SegmentProfile.from_device_profile(
        {"network": {"rtt_ms": 5.0, "rtt_ms_p50": 21.0}}
    )
    assert profile.rtt_ms == 5.0
