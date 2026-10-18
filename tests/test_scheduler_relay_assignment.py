"""★ A1 / X 档（Y 档第二条缺口 1）：relay 段节点必须**不占层**、且被标 `engine="relay_middle"`。

背景（实测，2026-09-28）：产品路径跨机 relay 第一次跑时，master 把 relay worker 当**普通层节点**
分层（日志 `client_TABLET-2TLUCNU8: Layer 18-24 (6层)`），而它没有本地模型 ⇒ 无法确认
「分层释放 ACK」⇒ `pipeline_distributed_workers_unavailable` ⇒ `/api/chat` 503。
根因：「动态分层」这条路（`compute_layer_assignment` / `_simple_weight_assignment`）**不认 relay 段** ——
第 2/3 批的 `engine="relay_middle"` 只接到了 `scheduler_pipeline` 里的 assignment 构造路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import api_server  # noqa: F401,E402 - 与 tests/test_scheduler.py 保持同样的导入副作用
from tests.test_scheduler import _engine_contract
from scheduler import NodeInfo, NodeState, Scheduler  # noqa: E402

RELAY_SPEC = {
    "role": "middle", "host": "127.0.0.1", "port": 50283,
    "n_embd": 896, "timeout": 60.0, "layer_start": 8, "layer_end": 16,
}


def _node(node_id: str, role: str) -> NodeInfo:
    return NodeInfo(
        node_id=node_id, role=role, state=NodeState.ONLINE,
        node_type="pc", device_info=_engine_contract("pytorch", "pytorch_cpu"),
    )


def _scheduler(monkeypatch: pytest.MonkeyPatch, relay_ids: set[str]) -> Scheduler:
    """造一个三节点调度器；`relay_ids` 里的节点被 `_relay_segment_for_worker` 认成 relay 段。"""
    sched = Scheduler()
    sched.nodes = {
        "master": _node("master", "master"),
        "client1": _node("client1", "client"),
        "client2": _node("client2", "client"),
    }

    def _relay_for(node_id, routing_preference="auto"):
        return RELAY_SPEC if node_id in relay_ids else None

    monkeypatch.setattr(sched, "_relay_segment_for_worker", _relay_for, raising=True)
    return sched


def test_relay_worker_gets_no_layers_and_is_marked_relay_middle(monkeypatch):
    """★ 缺口 1 的判据：relay 段节点 `layers_count == 0` 且 `engine == "relay_middle"`。"""
    sched = _scheduler(monkeypatch, {"client2"})

    result = {a["node_id"]: a for a in sched.compute_layer_assignment()}

    relay = result["client2"]
    assert relay["layers_count"] == 0, relay
    assert relay["engine"] == "relay_middle"
    assert relay["start_layer"] == relay["end_layer"]
    assert relay["has_embedding"] is False
    assert relay["has_lm_head"] is False

    # 真正跑层的两个节点仍覆盖全部 24 层（relay 节点**不**吃层）。
    learning = [a for a in result.values() if a["node_id"] != "client2"]
    assert sum(a["layers_count"] for a in learning) == 24
    assert max(a["end_layer"] for a in learning) == 24


def test_opted_out_relay_worker_is_restored_before_assignment(monkeypatch):
    """An endpoint-backed relay must survive a stale Full Worker opt-out."""
    sched = _scheduler(monkeypatch, {"client2"})
    sched._pipeline_worker_opt_out.add("client2")

    result = {a["node_id"]: a for a in sched.compute_layer_assignment()}

    assert result["client2"]["engine"] == "relay_middle"
    assert result["client2"]["layers_count"] == 0
    assert sum(
        item["layers_count"] for node_id, item in result.items()
        if node_id != "client2"
    ) == 24


def test_opted_out_relay_worker_is_a_capacity_candidate(monkeypatch):
    """Relay capacity is an endpoint placeholder, not local model capacity."""
    sched = _scheduler(monkeypatch, {"client2"})
    sched._pipeline_worker_opt_out.add("client2")

    candidates = sched._get_pipeline_capacity_nodes()

    relay = next(item for item in candidates if item["node_id"] == "client2")
    assert relay["capacity_source"] == "relay_exempt"
    assert relay["reserve_bytes"] == 0


def test_opted_out_relay_worker_is_not_filtered_from_manual_assignments(monkeypatch):
    sched = _scheduler(monkeypatch, {"client2"})
    sched._pipeline_worker_opt_out.add("client2")
    sched._runtime_layer_override = [{
        "node_id": "client2",
        "start_layer": 0,
        "end_layer": 0,
        "layers_count": 0,
    }]

    result = sched.get_layer_assignments()

    assert any(item["node_id"] == "client2" for item in result["assignments"])


def test_relay_worker_alone_still_appears_in_the_plan(monkeypatch):
    """只有 relay 节点参与时也必须出条目 —— 否则流水线直接空转（`return []`）。"""
    sched = _scheduler(monkeypatch, {"master", "client1", "client2"})

    result = {a["node_id"]: a for a in sched.compute_layer_assignment()}

    assert set(result) == {"master", "client1", "client2"}
    assert all(a["layers_count"] == 0 for a in result.values())
    assert all(a["engine"] == "relay_middle" for a in result.values())


def test_switch_off_keeps_layer_plan_unchanged(monkeypatch):
    """开关默认关（`_relay_segment_for_worker` 恒 None）⇒ 分层结果里**没有** relay 条目。"""
    sched = _scheduler(monkeypatch, set())

    result = sched.compute_layer_assignment()

    assert all(a.get("engine") != "relay_middle" for a in result)
    assert all(a["layers_count"] > 0 for a in result)
    assert sum(a["layers_count"] for a in result) == 24


def test_guard_split_relay_nodes_uses_the_real_predicate(monkeypatch):
    """★「该红必须红」守卫：`_split_relay_nodes` 必须用**真判据**摘节点。

    若把它改成恒返回 `(node_list, [])`，上面两条会**静默失效**（relay 节点又会被分层），
    而本用例立刻红 —— 它直接断言"谁被摘出来"。
    """
    sched = _scheduler(monkeypatch, {"client2"})

    remaining, relay = sched._split_relay_nodes([
        {"node_id": "master", "role": "master"},
        {"node_id": "client2", "role": "client"},
    ])

    assert [n["node_id"] for n in remaining] == ["master"]
    assert [n["node_id"] for n in relay] == ["client2"]


def _claim(role: str, start: int, end: int, port: int = 50283) -> dict:
    """一条 relay 段规格（与 `_normalize_relay_segment` 的返回形状一致）。"""
    return {"role": role, "host": "127.0.0.1", "port": port, "n_embd": 896,
            "timeout": 60.0, "layer_start": start, "layer_end": end}


def _claims(sched: Scheduler, claims: dict) -> None:
    """注入 relay 段映射（`_relay_claimed_layer_prefix` 的输入真源）。"""
    sched._relay_segment_map_cache = claims


# ---- ★ Y 档第二条：层区间认领的校验（fail-closed）-------------------------


def test_claimed_prefix_rejects_middle_plus_tail_until_chain_is_wired(monkeypatch):
    """合法形态：本机跑 head `[0,8)`，middle 段 `[8,16)`，tail 段 `[16,24)` ⇒ k = 8。

    这正是产品形态（master 做 head，Surface 上的 middle + tail 段覆盖其余）。
    """
    sched = _scheduler(monkeypatch, set())
    _claims(sched, {"mid": _claim("middle", 8, 16), "tail": _claim("tail", 16, 24)})

    with pytest.raises(ValueError, match="单段 relay"):
        sched._relay_claimed_layer_prefix(24)


def test_claimed_prefix_accepts_middle_covering_to_top(monkeypatch):
    """★★ **当前产品形态**：本机跑 `[0,8)`，**单段 `middle[8,24)`** 覆盖到顶 ⇒ k = 8。

    `middle` 贴顶是合法的 —— 它只输出 hidden，`lm_head` 由主节点在收到末节点回的 hidden
    后自己跑（`scheduler_pipeline.py:4594-4595` 的推荐拓扑）。此前把它误判为"应 tail"
    导致真实配置被拒（实测 `reason_code=relay_layer_claim_invalid`），本用例锁住该回归。
    """
    sched = _scheduler(monkeypatch, set())
    _claims(sched, {"mid": _claim("middle", 8, 24)})

    assert sched._relay_claimed_layer_prefix(24) == 8


def test_claimed_prefix_rejects_relay_only_multi_segment_topology(monkeypatch):
    """极端但合法：`head` 段从 0 起 + `tail` 段到顶 ⇒ 本机无层（k = 0）。"""
    sched = _scheduler(monkeypatch, set())
    _claims(sched, {"head": _claim("head", 0, 8), "tail": _claim("tail", 8, 24)})

    with pytest.raises(ValueError, match="relay head segment"):
        sched._relay_claimed_layer_prefix(24)


def test_claimed_prefix_without_claims_returns_total(monkeypatch):
    """没有认领 ⇒ k = total（与接线前一致，对既有路径零影响）。"""
    sched = _scheduler(monkeypatch, set())
    _claims(sched, {})

    assert sched._relay_claimed_layer_prefix(24) == 24


@pytest.mark.parametrize(
    ("claims", "why"),
    [
        ({"a": _claim("middle", 8, 16), "b": _claim("middle", 12, 20)}, "两段重叠"),
        ({"a": _claim("middle", 0, 8), "b": _claim("middle", 8, 24)}, "middle 从 0 起（吃不到 hidden）"),
        ({"a": _claim("head", 4, 8), "b": _claim("tail", 8, 24)}, "head 未从 0 起"),
        ({"a": _claim("middle", 8, 16), "b": _claim("tail", 16, 20)}, "tail 未覆盖到顶"),
        ({"a": _claim("middle", 8, 16)}, "中间空洞（8-16 之外无人认领）"),
        ({"a": _claim("middle", 8, 30)}, "区间越界"),
        ({"a": _claim("middle", 8, 8)}, "空区间"),
    ],
)
def test_claimed_prefix_rejects_invalid(monkeypatch, claims, why):
    """★ fail-closed：任一不成立的认领都必须**抛错**，绝不静默放过（`why` 即判据）。"""
    sched = _scheduler(monkeypatch, set())
    _claims(sched, claims)

    with pytest.raises(ValueError):
        sched._relay_claimed_layer_prefix(24)


# ---- ★ Y 档第二条：认领层必须从本机节点的分配里**扣除** -------------------


def test_assignment_deducts_claimed_layers_from_local_nodes(monkeypatch):
    """★★ 真根因的回归判据：认领的层**被扣除** —— 本机节点只分到 `[0, k)`。

    改造前：master 拿到 0-24（随后远端段在「已过全部层」的 hidden 上重算 8-16，
    输出语义崩坏）。改造后：本机节点合计只分到 8 层，且上界 `<= 8`。
    """
    sched = _scheduler(monkeypatch, set())
    _claims(sched, {"mid": _claim("middle", 8, 24)})

    result = sched.compute_layer_assignment()

    learning = [a for a in result if a["layers_count"] > 0]
    assert learning, "本机层节点应当仍有层可跑（0-8）"
    assert sum(a["layers_count"] for a in learning) == 8
    assert max(a["end_layer"] for a in learning) <= 8


def test_assignment_with_all_layers_claimed_yields_no_local_layers(monkeypatch):
    """全部层都被认领 ⇒ 本机**无层条目**产出（`layer_budget == 0` 的早返回）。"""
    sched = _scheduler(monkeypatch, set())
    _claims(sched, {"tail": _claim("tail", 0, 24)})

    assert sched.compute_layer_assignment() == []
