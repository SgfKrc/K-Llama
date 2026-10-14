"""DIST-NEXT-3 回归：`WorkerAssignmentState` 的换代际、相位推进与单一 reason code。

审计 P1-1 / P0-3：同一节点重连或重配时只有「config/generation 的组合」可以间接隔离，
且撤销原因散落多处。本文件的判据：

1. 每次下发新配置都换 `assignment_id`（旧 lease 不被复用）；
2. 相位按序前进，越级/回退/终止态变更一律拒绝（这类事件就是「不属于当前 assignment」）；
3. 进入终止态只记**一个** reason code，且重复撤销幂等；
4. 现有写路径（publish / ACK / clear）已同步进权威视图（读路径尚未切换 ⇒ 零行为变化）。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server  # noqa: F401,E402
from scheduler import Scheduler  # noqa: E402
from worker_assignment_state import (  # noqa: E402
    PHASE_ABORTED,
    PHASE_ACKED,
    PHASE_PUSHING,
    PHASE_READY,
    PHASE_RELEASED,
    REASON_CONFIG_ACKED,
    REASON_CONFIG_CLEARED,
    REASON_CONFIG_PUBLISHED,
    REASON_RECONFIGURED,
    REASON_WORKER_RELEASED,
    WorkerAssignmentRegistry,
    WorkerAssignmentState,
    assignment_state_summary,
)


def _registry():
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"asg_test{counter['n']:04d}"

    return WorkerAssignmentRegistry(id_factory=next_id), counter


def test_begin_assigns_a_fresh_identity_every_time():
    """同一节点重配 ⇒ 新 assignment_id、旧 lease 不复用（换代际）。"""
    registry, counter = _registry()

    first = registry.begin(
        "worker_01", config_id="cfg-1", connection_generation=7,
    )
    assert first.assignment_id == "asg_test0001"
    assert first.phase == PHASE_PUSHING
    assert first.config_id == "cfg-1"
    assert first.connection_generation == 7
    assert first.reason_code == REASON_CONFIG_PUBLISHED

    registry.attach_lease(
        "worker_01", lease_id="lease_aaa", lease_epoch=3,
    )
    assert registry.state("worker_01").lease_id == "lease_aaa"

    second = registry.begin(
        "worker_01", config_id="cfg-2", connection_generation=8,
    )
    assert second.assignment_id == "asg_test0002"
    assert second.assignment_id != first.assignment_id
    # 旧 lease 不得跟着新 assignment 走
    assert second.lease_id == ""
    assert second.lease_epoch == 0
    assert second.reason_code == REASON_RECONFIGURED
    assert counter["n"] == 2


def test_phase_progress_only_forward_and_only_on_the_current_assignment():
    registry, _ = _registry()
    registry.begin("worker_01", config_id="cfg-1")

    # 允许**跨级前进**（真实路径里 ACK 可能直接给出就绪，没有单独的 acked 阶段）
    assert registry.transition("worker_01", phase=PHASE_READY) is not None
    assert registry.is_ready("worker_01")

    # 回退被拒
    assert registry.transition("worker_01", phase=PHASE_ACKED) is None
    assert registry.phase("worker_01") == PHASE_READY
    # 未知相位被拒
    assert registry.transition("worker_01", phase="nonsense") is None
    # 未登记的节点不凭空产生状态
    assert registry.transition("worker_99", phase=PHASE_ACKED) is None

    # 正常顺序前进也成立
    registry.begin("worker_02", config_id="cfg-1")
    assert registry.transition("worker_02", phase=PHASE_ACKED) is not None
    assert registry.transition("worker_02", phase=PHASE_READY) is not None
    assert registry.phase("worker_02") == PHASE_READY


def test_release_records_exactly_one_reason_and_is_idempotent():
    registry, _ = _registry()
    registry.begin("worker_01", config_id="cfg-1")
    registry.transition("worker_01", phase=PHASE_ACKED)

    released = registry.release("worker_01", reason_code=REASON_CONFIG_CLEARED)

    assert released.phase == PHASE_RELEASED
    assert released.reason_code == REASON_CONFIG_CLEARED
    assert released.terminal
    assert released.lease_id == ""
    # 重复撤销：状态不变、原因不被覆盖
    again = registry.release("worker_01", reason_code="something_else")
    assert again.reason_code == REASON_CONFIG_CLEARED
    # 终止态不接受相位推进，也不接受新 lease
    assert registry.transition("worker_01", phase=PHASE_READY) is None
    assert registry.attach_lease(
        "worker_01", lease_id="lease_late", lease_epoch=9,
    ) is None


def test_aborted_release_uses_the_aborted_phase():
    registry, _ = _registry()
    registry.begin("worker_01", config_id="cfg-1")

    state = registry.release(
        "worker_01", reason_code="pipeline_local_commit_failed", aborted=True,
    )

    assert state.phase == PHASE_ABORTED
    assert state.reason_code == "pipeline_local_commit_failed"


def test_snapshot_and_summary_are_serializable():
    registry, _ = _registry()
    registry.begin("worker_01", config_id="cfg-1")
    registry.transition("worker_01", phase=PHASE_READY)
    registry.begin("worker_02", config_id="cfg-1")

    snapshot = registry.snapshot()

    assert set(snapshot) == {"worker_01", "worker_02"}
    assert snapshot["worker_01"]["phase"] == PHASE_READY
    assert snapshot["worker_01"]["assignment_id"] == "asg_test0001"
    assert assignment_state_summary(snapshot) == {PHASE_READY: 1, PHASE_PUSHING: 1}
    # 传 dataclass 也支持
    assert assignment_state_summary({
        "worker_01": WorkerAssignmentState(
            node_id="worker_01", assignment_id="asg_x", phase=PHASE_RELEASED,
        ),
    }) == {PHASE_RELEASED: 1}


def test_registry_owns_expected_contract_and_receipt_by_assignment_identity():
    registry, _ = _registry()
    source = {
        "config_id": "cfg-1",
        "generation": 7,
        "layer_range": [8, 24],
    }

    state = registry.begin(
        "worker_01",
        config_id="cfg-1",
        connection_generation=3,
        expected_config=source,
    )
    source["layer_range"][0] = 99

    expected = registry.expected("worker_01")
    assert expected["assignment_id"] == state.assignment_id
    assert expected["layer_range"] == [8, 24]
    assert registry.identity_matches(
        "worker_01",
        assignment_id=state.assignment_id,
        config_id="cfg-1",
        generation=7,
    )
    assert not registry.identity_matches(
        "worker_01",
        assignment_id="asg_stale",
        config_id="cfg-1",
        generation=7,
    )

    assert registry.record_ack("worker_01", {
        "assignment_id": "asg_stale",
        "config_id": "cfg-1",
        "generation": 7,
        "status": "ready",
    }) is None
    assert registry.acknowledgement("worker_01") == {}

    receipt = {
        "assignment_id": state.assignment_id,
        "config_id": "cfg-1",
        "generation": 7,
        "status": "ready",
    }
    assert registry.record_ack("worker_01", receipt) is not None
    receipt["status"] = "error"
    assert registry.acknowledgement("worker_01")["status"] == "ready"


def test_prepare_to_commit_updates_contract_without_changing_assignment_id():
    registry, _ = _registry()
    prepared = registry.begin(
        "worker_01",
        expected_config={
            "config_id": "cfg-1",
            "generation": 7,
            "phase": "prepare",
        },
    )

    committed = registry.update_expected("worker_01", {
        "config_id": "cfg-1",
        "generation": 7,
        "phase": "commit",
    })

    assert committed is not None
    assert committed.assignment_id == prepared.assignment_id
    assert registry.expected("worker_01")["assignment_id"] == prepared.assignment_id
    assert registry.expected("worker_01")["phase"] == "commit"
    assert registry.acknowledgement("worker_01") == {}


def test_snapshot_restore_preserves_contract_receipt_and_assignment_identity():
    registry, _ = _registry()
    state = registry.begin(
        "worker_01",
        connection_generation=4,
        expected_config={"config_id": "cfg-1", "generation": 9},
    )
    registry.record_ack("worker_01", {
        "assignment_id": state.assignment_id,
        "config_id": "cfg-1",
        "generation": 9,
        "status": "ready",
    })
    registry.transition("worker_01", phase=PHASE_READY)
    snapshot = registry.snapshot()

    registry.begin(
        "worker_01",
        expected_config={"config_id": "cfg-2", "generation": 10},
    )
    registry.restore(snapshot)

    restored = registry.state("worker_01")
    assert restored is not None
    assert restored.assignment_id == state.assignment_id
    assert restored.connection_generation == 4
    assert restored.phase == PHASE_READY
    assert registry.expected("worker_01")["config_id"] == "cfg-1"
    assert registry.acknowledgement("worker_01")["status"] == "ready"


def test_versioned_ready_ack_advances_the_authority_view():
    sched = Scheduler()
    config = {
        "config_id": "cfg-1", "generation": 11, "nodes": [],
        "phase": "commit", "plan_id": "plan-1",
        "start_layer": 8, "end_layer": 24,
        "model_sha256": "a" * 64, "model_type": "qwen2", "engine": "pytorch",
    }
    sched._publish_layer_configs({"worker_01": config})
    expected = sched._worker_assignments.expected("worker_01")

    sched._handle_layer_config_ack("worker_01", {"data": {
        "node_id": "worker_01",
        "assignment_id": expected["assignment_id"],
        "config_id": "cfg-1", "generation": 11,
        "status": "ready", "phase": "commit", "plan_id": "plan-1",
        "layer_range": [8, 24], "model_sha256": "a" * 64, "model_type": "qwen2",
        "engine": "pytorch",
    }})

    assert "worker_01" in sched._layer_config_pushed
    derived = {
        node for node, state in sched._worker_assignments.snapshot().items()
        if state["phase"] in (PHASE_ACKED, PHASE_READY)
    }
    # 不变量：旧集合与权威视图派生的集合一致（达成后即可把旧集合降级为派生视图）
    assert derived == set(sched._layer_config_pushed)


def test_retry_scan_ignores_terminal_assignments():
    sched = Scheduler()
    registry = sched._worker_assignments
    sent = []

    class _Server:
        _running = True

        def get_client_ids(self):
            return ["stale"]

        def send_layer_config(self, node_id, payload):
            sent.append((node_id, payload))

    sched._tcp_server = _Server()
    with sched._layer_config_lock:
        sched._layer_config_retry_state["stale"] = {"attempts": 0, "next_retry": 0.0}
        registry.begin(
            "stale",
            expected_config={"config_id": "c", "generation": 1},
        )
        registry.release("stale", reason_code=REASON_CONFIG_CLEARED)

    assert sched._retry_pending_layer_configs(now=1_000_000.0) == 0
    assert sent == []


def test_retry_scan_still_skips_genuinely_pushed_nodes():
    """对照：旧集合与权威视图都说 pushed ⇒ 重发照旧跳过（零行为漂移）。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    sent = []

    class _Server:
        _running = True

        def get_client_ids(self):
            return ["fresh"]

        def send_layer_config(self, node_id, payload):
            sent.append((node_id, payload))

    sched._tcp_server = _Server()
    with sched._layer_config_lock:
        sched._layer_config_retry_state["fresh"] = {"attempts": 0, "next_retry": 0.0}
        registry.begin(
            "fresh",
            expected_config={"config_id": "c", "generation": 1},
        )
        registry.transition("fresh", phase=PHASE_ACKED)

    assert sched._retry_pending_layer_configs(now=1_000_000.0) == 0
    assert sent == []


def test_pushed_view_and_ready_nodes_are_derived_only_from_registry():
    sched = Scheduler()
    assert sched._effective_layer_config_pushed("unknown_node", True) is False
    sched._worker_assignments.begin("w_acked", config_id="c")
    sched._worker_assignments.transition("w_acked", phase=PHASE_ACKED)
    assert sched._effective_layer_config_pushed("w_acked", True) is True
    sched._worker_assignments.begin("w_released", config_id="c")
    sched._worker_assignments.release("w_released", reason_code=REASON_CONFIG_CLEARED)
    assert sched._effective_layer_config_pushed("w_released", True) is False
    assert sched._effective_layer_config_pushed_nodes(
        {"legacy_only"}, {"legacy_only": {"config_id": "legacy"}},
    ) == {"w_acked"}


def test_scheduler_write_paths_feed_the_authority_view():
    """现有写路径（publish / ACK / clear）已同步进权威视图。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    assert isinstance(registry, WorkerAssignmentRegistry)

    sched._publish_layer_configs({
        "worker_01": {
            "config_id": "cfg-1",
            "generation": 11,
            "nodes": [],
        },
    })
    state = registry.state("worker_01")
    assert state is not None
    assert state.phase == PHASE_PUSHING
    assert state.config_id == "cfg-1"
    assert state.assignment_id.startswith("asg_")
    assert registry.expected("worker_01")["assignment_id"] == state.assignment_id

    # 清配置 ⇒ 终止态 + 单一 reason
    sched._clear_layer_config_state("worker_01")
    released = registry.state("worker_01")
    assert released.phase == PHASE_RELEASED
    assert released.reason_code == REASON_CONFIG_CLEARED


def test_reconfiguration_replaces_the_previous_assignment():
    """同节点二次下发 ⇒ 权威视图换成新代际（旧 ACK 无法冒充）。"""
    sched = Scheduler()
    registry = sched._worker_assignments

    sched._publish_layer_configs({
        "worker_01": {"config_id": "cfg-1", "generation": 11, "nodes": []},
    })
    first = registry.assignment_id("worker_01")
    sched._publish_layer_configs({
        "worker_01": {"config_id": "cfg-2", "generation": 12, "nodes": []},
    })
    second = registry.assignment_id("worker_01")

    assert first and second and first != second
    assert registry.state("worker_01").config_id == "cfg-2"
    assert registry.state("worker_01").reason_code == REASON_RECONFIGURED


def test_stale_same_config_generation_ack_cannot_advance_new_assignment():
    sched = Scheduler()
    config = {
        "config_id": "cfg-1", "generation": 11, "phase": "commit",
        "plan_id": "plan-1", "start_layer": 8, "end_layer": 24,
        "model_sha256": "a" * 64, "model_type": "qwen2", "engine": "pytorch",
    }
    sched._publish_layer_configs({"worker_01": dict(config)})
    old_id = sched._worker_assignments.assignment_id("worker_01")
    sched._publish_layer_configs({"worker_01": dict(config)})
    current = sched._worker_assignments.state("worker_01")
    assert current.assignment_id != old_id

    sched._handle_layer_config_ack("worker_01", {"data": {
        "node_id": "worker_01", "assignment_id": old_id,
        "config_id": "cfg-1", "generation": 11,
        "status": "ready", "phase": "commit", "plan_id": "plan-1",
        "layer_range": [8, 24], "model_sha256": "a" * 64,
        "model_type": "qwen2", "engine": "pytorch",
    }})

    assert sched._worker_assignments.phase("worker_01") == PHASE_PUSHING
    assert sched._worker_assignments.acknowledgement("worker_01") == {}
    assert "worker_01" in sched._layer_config_retry_state


def test_versioned_ack_without_assignment_id_fails_closed():
    sched = Scheduler()
    sched._publish_layer_configs({"worker_01": {
        "config_id": "cfg-1", "generation": 11, "phase": "commit",
        "start_layer": 0, "end_layer": 1,
        "model_sha256": "a" * 64, "model_type": "qwen2", "engine": "pytorch",
    }})

    sched._handle_layer_config_ack("worker_01", {"data": {
        "node_id": "worker_01", "config_id": "cfg-1", "generation": 11,
        "status": "ready", "phase": "commit", "layer_range": [0, 1],
        "model_sha256": "a" * 64, "model_type": "qwen2", "engine": "pytorch",
    }})

    assert sched._worker_assignments.phase("worker_01") == PHASE_PUSHING
    assert sched._worker_assignments.acknowledgement("worker_01") == {}


def test_retry_reuses_the_same_assignment_identity():
    sched = Scheduler()
    sent = []

    class _Server:
        _running = True

        def get_client_ids(self):
            return ["worker_01"]

        def send_layer_config(self, node_id, payload):
            sent.append((node_id, dict(payload)))

    sched._tcp_server = _Server()
    sched._publish_layer_configs({"worker_01": {
        "config_id": "cfg-1", "generation": 11, "release": True,
    }})
    assignment_id = sched._worker_assignments.assignment_id("worker_01")
    sched._layer_config_retry_state["worker_01"]["next_retry"] = 0.0

    assert sched._retry_pending_layer_configs(now=1_000_000.0) == 1
    assert len(sent) == 2
    assert {payload["assignment_id"] for _, payload in sent} == {assignment_id}


def test_late_release_ack_cannot_clear_replacement_assignment():
    sched = Scheduler()
    sched._publish_layer_configs({"worker_01": {
        "config_id": "cfg-release", "generation": 11, "release": True,
    }})
    old = sched._worker_assignments.expected("worker_01")
    sched._publish_layer_configs({"worker_01": {
        "config_id": "cfg-new", "generation": 12,
        "phase": "commit", "start_layer": 0, "end_layer": 1,
        "model_sha256": "a" * 64, "model_type": "qwen2", "engine": "pytorch",
    }})

    sched._handle_layer_config_ack("worker_01", {"data": {
        "node_id": "worker_01", "assignment_id": old["assignment_id"],
        "config_id": "cfg-release", "generation": 11,
        "status": "released", "release": True,
    }})

    current = sched._worker_assignments.state("worker_01")
    assert current.config_id == "cfg-new"
    assert current.phase == PHASE_PUSHING
    assert "worker_01" in sched._layer_config_retry_state


def test_ack_path_reaches_acked_and_released_states():
    """ACK 路径：非 released ⇒ acked；released ⇒ 终止态（reason=worker_released）。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    # `release: True` 的配置才会走 released ACK 分支（否则要求层区间等 legacy 字段）
    sched._publish_layer_configs({
        "worker_01": {
            "config_id": "cfg-1", "generation": 11, "release": True, "nodes": [],
        },
    })

    def ack(**extra):
        assignment_id = registry.assignment_id("worker_01")
        return {"data": {
            "node_id": "worker_01",
            "assignment_id": assignment_id,
            "config_id": "cfg-1",
            "generation": 11,
            **extra,
        }}

    # released ACK（worker 确认释放该层配置）⇒ 权威视图进终止态 + 单一 reason。
    # 非 released 的 ready ACK 还要求层区间等 legacy 字段，其相位推进由 registry
    # 单测直接覆盖（见 test_phase_progress_only_forward...）。
    sched._handle_layer_config_ack("worker_01", ack(status="released", release=True))
    state = registry.state("worker_01")
    assert state.phase == PHASE_RELEASED
    assert state.reason_code == REASON_WORKER_RELEASED
