"""#51 回归：本地（master）节点也要声明自己的层段工件区间。

背景
----
`src/pipeline_capacity.py` 的判据是
`range_constrained = any(node.get("layer_ranges") is not None for node in usable)`：
**只要任一**节点声明了区间，全链就必须由区间拼满 `[0, total)`（契约见
`src/pipeline_node_contract.py` 的「cover every layer exactly once」）。

而 `src/scheduler.py:_get_pipeline_capacity_nodes` 的 `layer_ranges` **只**从远端
task worker 的 v3 hello 取，**本地 master 节点此前从不带** ⇒ 一旦远端声明区间，
本地这段就成了覆盖缺口，求解器返回
`status=rejected reason=pipeline_layer_range_coverage_insufficient`（真机实测）。

修复：本地节点若配了裁层工件（`QLH_LAYER_GGUF`），用与 PC worker 同一份推导
（`SchedulerTaskWorkerMixin._configured_layer_artifact`）声明自己的区间与 `segment_mode`。
⚠️ 判据必须用字面量 `"master"`（与 `_pipeline_node_metadata` 一致）——
用 `get_effective_node_id()` 会因它返回 `_configured_node_id()`（主机名）而永不匹配。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

import api_server  # noqa: F401,E402  （加载 API composition root，配置 SchedulerCallbackSet）
from scheduler import Scheduler, NodeInfo, NodeState, NodeRole  # noqa: E402
from koakuma_engine import Capability  # noqa: E402
from qlh_core.contracts.engines import ENGINE_REGISTRY, LLAMA_CPP_BINDING


def _master_sched():
    sched = Scheduler()
    sched._role_override = "master"
    sched.nodes = {
        "master": NodeInfo(
            node_id="master",
            role=NodeRole.MASTER,
            state=NodeState.ONLINE,
            node_type="pc",
            device_info={
                "engine_capabilities": ENGINE_REGISTRY.advertisement([
                    ENGINE_REGISTRY.require("llama_cpp", LLAMA_CPP_BINDING),
                ]),
                "ram": {"total_gb": 64.0, "available_gb": 32.0},
                "capabilities": [Capability.FORWARD_LAYERS],
            },
        ),
    }
    return sched


def test_local_node_declares_layer_ranges_from_artifact(monkeypatch):
    """配了裁层工件 ⇒ 本地节点把自己的区间交给求解器。"""
    monkeypatch.setattr(
        Scheduler,
        "_configured_layer_artifact",
        lambda self: {
            "start": 0,
            "end": 16,
            "model_id": "q35-2b-head16.gguf",
            "mode": "head",
        },
        raising=True,
    )

    records = _master_sched()._get_pipeline_capacity_nodes()
    master = next(r for r in records if r["node_id"] == "master")

    assert master.get("layer_ranges") == [[0, 16]]
    assert master.get("segment_mode") == "head"


def test_local_node_without_artifact_declares_nothing(monkeypatch):
    """没配工件 ⇒ 不声明区间（保持既有行为：求解器走无约束分支）。"""
    monkeypatch.setattr(
        Scheduler, "_configured_layer_artifact", lambda self: None, raising=True,
    )

    records = _master_sched()._get_pipeline_capacity_nodes()
    master = next(r for r in records if r["node_id"] == "master")

    assert "layer_ranges" not in master
    assert "segment_mode" not in master


def test_local_node_keeps_unrecognized_mode_out_of_segment_mode(monkeypatch):
    """工件 `mode` 不在 {head,middle,tail} 时，只声明区间、不写 segment_mode。"""
    monkeypatch.setattr(
        Scheduler,
        "_configured_layer_artifact",
        lambda self: {"start": 20, "end": 24, "model_id": "x.gguf", "mode": "weird"},
        raising=True,
    )

    records = _master_sched()._get_pipeline_capacity_nodes()
    master = next(r for r in records if r["node_id"] == "master")

    assert master.get("layer_ranges") == [[20, 24]]
    assert "segment_mode" not in master
