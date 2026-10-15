"""DIST-NEXT-7 回归：A1 relay 必须被隔离在生产状态机之外。

判据来自 `docs/archive/misc/DIST后续核心链路缺陷审计-2026-10-07.md` 的「设计债务与遗留隔离」：

1. **默认不可被 scheduler assignment 选中**（`PIPELINE_RELAY_PROBE_ONLY` 默认 1）；
2. 门只有一个（`relay_a1_legacy.a1_production_enabled`），生产供给只经它；
3. **独立诊断 namespace**（`relay_a1.*`），与 Route A 的原因码分开；
4. 探针入口仍可用（显式 `QLH_RELAY_PROBE_ONLY=0` + `QLH_RELAY_ENABLED=1`）。

被隔离的是「生产状态判定」而不是探针本身：A1 的段映射解析迁入隔离模块，判据与
既有入口行为必须逐条不变。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server  # noqa: F401,E402  （与 tests/test_scheduler.py 相同的导入副作用）
import relay_a1_legacy as a1  # noqa: E402
import scheduler_pipeline  # noqa: E402
from scheduler import Scheduler  # noqa: E402

RELAY_SEGMENTS = "worker-2=middle@127.0.0.1:50183#896#8-16"


def test_production_gate_truth_table():
    """门的真值表：只有「总开关开 + 探针闸门关」才允许进生产。"""
    assert a1.a1_production_enabled(probe_only=True, relay_enabled=True) is False
    assert a1.a1_production_enabled(probe_only=True, relay_enabled=False) is False
    assert a1.a1_production_enabled(probe_only=False, relay_enabled=False) is False
    assert a1.a1_production_enabled(probe_only=False, relay_enabled=True) is True


def test_default_configuration_keeps_a1_out_of_assignments(monkeypatch):
    """默认：即使配置齐全，A1 段也拿不到、节点也不会被当 relay 宿主。"""
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_PROBE_ONLY", True)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(
        scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS", RELAY_SEGMENTS,
    )
    sched = Scheduler()

    assert sched._relay_segment_for_worker("worker-2") is None
    assert sched._is_relay_host("worker-2") is False

    status = sched.get_relay_a1_status()
    assert status["namespace"] == "relay_a1"
    assert status["production_enabled"] is False
    assert status["assignment_selectable"] is False
    # 配置本身是齐的：拒绝来自隔离门，而不是「没配」
    assert status["segments_configured"] is True
    assert status["configured_node_count"] == 1


def test_probe_entry_restores_a1(monkeypatch):
    """探针入口：显式关掉闸门后 A1 行为恢复（含角色判定与段供给）。"""
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_PROBE_ONLY", False)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(
        scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS", RELAY_SEGMENTS,
    )
    sched = Scheduler()

    spec = sched._relay_segment_for_worker("worker-2")
    assert spec is not None and spec["port"] == 50183
    assert sched._is_relay_host("worker-2") is True

    status = sched.get_relay_a1_status()
    assert status["production_enabled"] is True
    assert status["assignment_selectable"] is True
    assert status["relay_host_count"] == 0  # 该节点未注册进 self.nodes


def test_diagnostic_namespace_is_separate_from_route_a_reasons():
    """诊断字段集合固定、namespace 独立 —— 不与 Route A 的原因码混用。"""
    status = a1.a1_isolation_status(probe_only=True, relay_enabled=True)

    assert status["namespace"] == "relay_a1"
    assert set(status) == {
        "namespace", "production_enabled", "assignment_selectable", "probe_only",
        "relay_enabled", "segments_configured", "configured_node_count",
        "relay_host_count", "probe_entry",
    }
    assert status["probe_entry"] == a1.PROBE_ENTRY_ENV


def test_segment_map_parser_moved_without_behaviour_change():
    """解析实现迁入隔离模块；合法/非法条目判定与既有入口逐条一致。"""
    raw = (
        f"{RELAY_SEGMENTS};broken;"
        "worker-3=head@127.0.0.1:50184#896#8-16"   # head 尚不被接受 ⇒ 整条丢弃
    )
    parsed = a1.parse_relay_segment_map(raw, Scheduler._normalize_relay_segment)

    assert set(parsed) == {"worker-2"}
    assert Scheduler._parse_relay_segment_map(raw) == parsed
    assert a1.parse_relay_segment_map("", Scheduler._normalize_relay_segment) == {}


def test_relay_host_status_counts_registered_nodes(monkeypatch):
    """已注册的 relay 宿主会被计入独立诊断（供 health 侧判断占用）。"""
    from scheduler import NodeInfo, NodeState

    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_PROBE_ONLY", False)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(
        scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS", RELAY_SEGMENTS,
    )
    sched = Scheduler()
    sched.nodes["worker-2"] = NodeInfo(
        node_id="worker-2", role="client", state=NodeState.ONLINE,
    )

    status = sched.get_relay_a1_status()
    assert status["relay_host_count"] == 1
