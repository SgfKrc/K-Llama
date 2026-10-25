"""DIST-NEXT-4b 回归：reservation 的**原子撤销**与**单一 reason code**。

审计 P0-3：「断线、心跳过期、服务重建都必须原子撤销 assignment 和 reservation，
并记录单一 reason code。」

* 断线：`notify_disconnect()` 全量回收（2026-10-05 已修，这里锁定 reason code）；
* 心跳过期但 TCP 仍在线：`_reap_stale_worker_reservations()` 只回收**已终结**的条目，
  不打断 in-flight 执行 —— TCP 巡检窗口（约 131s）里这类节点的槽位否则会被永久占住。
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server  # noqa: F401,E402  （与 tests/test_scheduler.py 相同的导入副作用）
from scheduler import NodeInfo, NodeState, Scheduler  # noqa: E402
from task_worker_adapter import (  # noqa: E402
    RELEASE_REASON_DISCONNECTED,
    RELEASE_REASON_HEARTBEAT_STALE,
    RELEASE_REASON_SERVICE_RESTART,
    RELEASE_REASONS,
    RemoteFullWorkerProvider,
    TaskWorkerControlPlane,
)
from execution_provider import ModelIdentity, StageRequest  # noqa: E402


def _capabilities():
    return {
        "stage_types": ["full_inference"],
        "engines": ["pytorch"],
        "models": [{
            "model_id": "qwen-1_8b",
            "engine": "pytorch",
            "format": "safetensors",
            "revision": "r1",
            "sha256": "a" * 64,
        }],
        "max_concurrency": 1,
    }


def _admitted_control_plane():
    worker = TaskWorkerControlPlane()
    coordinator = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(node_id="worker_01", capabilities=_capabilities())
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    worker.receive_on_worker(ack.snapshot())
    return coordinator


def _provider(coordinator, sent=None):
    return RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator.worker_snapshot("worker_01"),
        send_message=(sent.append if sent is not None else (lambda _m: None)),
    )


def _request(provider_id):
    return StageRequest(
        workflow_id="wf_release001",
        request_id="request-release-01",
        stage_id="candidate_a",
        stage_type="full_inference",
        provider_id=provider_id,
        dependencies={},
        root_input={"message": "hello"},
        model_identity=ModelIdentity(
            model_id="qwen-1_8b", engine="pytorch", format="safetensors",
            revision="r1", sha256="a" * 64,
        ),
    )


def _stale_node(age_seconds=10_000.0):
    return NodeInfo(
        node_id="worker_01",
        role="client",
        state=NodeState.ONLINE,
        connected_at=time.time() - age_seconds,
        last_heartbeat=time.time() - age_seconds,
    )


def _fresh_node():
    return NodeInfo(
        node_id="worker_01",
        role="client",
        state=NodeState.ONLINE,
        connected_at=time.time(),
        last_heartbeat=time.time(),
    )


def test_release_reasons_are_a_closed_set():
    assert RELEASE_REASONS == (
        "worker_tcp_disconnected",
        "worker_heartbeat_stale",
        "worker_service_restarted",
    )
    assert RELEASE_REASON_DISCONNECTED in RELEASE_REASONS
    assert RELEASE_REASON_HEARTBEAT_STALE in RELEASE_REASONS
    assert RELEASE_REASON_SERVICE_RESTART in RELEASE_REASONS


def test_idle_reservation_is_released_with_the_given_reason():
    coordinator = _admitted_control_plane()
    provider = _provider(coordinator)
    reservation = provider.reserve(_request(provider.provider_id))
    assert provider.inspect().active_reservations == 1

    freed = provider.release_stale_reservations(RELEASE_REASON_HEARTBEAT_STALE)

    assert freed == [reservation.reservation_id]
    assert provider.inspect().active_reservations == 0


def test_in_flight_reservation_is_not_reaped():
    """有在跑 attempt（offer 已发、未接受）的条目不能被打断。"""
    coordinator = _admitted_control_plane()
    sent = []
    provider = _provider(coordinator, sent=sent)
    reservation = provider.reserve(_request(provider.provider_id))

    from execution_provider import StageAttempt

    stage_attempt = StageAttempt(
        attempt_id="att_release001",
        request=_request(provider.provider_id),
        provider_id=provider.provider_id,
        lease_id="lease_release001",
        lease_epoch=1,
        lease_expires_at=time.time() + 5.0,
    )
    finished = threading.Event()

    def _execute():
        try:
            provider.execute(stage_attempt, reservation, threading.Event())
        except BaseException:
            pass
        finally:
            finished.set()

    threading.Thread(target=_execute, daemon=True).start()
    deadline = time.monotonic() + 2.0
    while not sent and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sent, "offer 未发出，测试前置条件不成立"

    # in-flight ⇒ 不回收
    assert provider.release_stale_reservations(RELEASE_REASON_HEARTBEAT_STALE) == []
    assert provider.inspect().active_reservations == 1

    provider.notify_disconnect(reason_code=RELEASE_REASON_DISCONNECTED)
    finished.wait(2)
    assert provider.inspect().active_reservations == 0


def test_stale_heartbeat_reaps_reservations_with_a_single_reason(caplog):
    sched = Scheduler()
    coordinator = _admitted_control_plane()
    provider = _provider(coordinator)
    reservation = provider.reserve(_request(provider.provider_id))
    sched._remote_task_worker_providers["worker_01"] = provider
    sched.nodes["worker_01"] = _stale_node()

    with caplog.at_level("WARNING", logger="scheduler"):
        released = sched._reap_stale_worker_reservations()

    assert released == {"worker_01": [reservation.reservation_id]}
    assert provider.inspect().active_reservations == 0
    # 单一 reason code 出现在事件里（可 grep、可对账）
    assert "event=task_worker_reservations_released" in caplog.text
    assert f"reason={RELEASE_REASON_HEARTBEAT_STALE}" in caplog.text


def test_fresh_heartbeat_keeps_reservations(caplog):
    sched = Scheduler()
    coordinator = _admitted_control_plane()
    provider = _provider(coordinator)
    provider.reserve(_request(provider.provider_id))
    sched._remote_task_worker_providers["worker_01"] = provider
    sched.nodes["worker_01"] = _fresh_node()

    with caplog.at_level("WARNING", logger="scheduler"):
        released = sched._reap_stale_worker_reservations()

    assert released == {}
    assert provider.inspect().active_reservations == 1
    assert "task_worker_reservations_released" not in caplog.text


def test_reap_is_idempotent():
    sched = Scheduler()
    coordinator = _admitted_control_plane()
    provider = _provider(coordinator)
    provider.reserve(_request(provider.provider_id))
    sched._remote_task_worker_providers["worker_01"] = provider
    sched.nodes["worker_01"] = _stale_node()

    assert sched._reap_stale_worker_reservations() != {}
    # 第二次没有可回收的条目 ⇒ 空结果，不报错
    assert sched._reap_stale_worker_reservations() == {}
