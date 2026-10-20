"""Automatic supervision must converge through the existing quorum authority."""
from __future__ import annotations

import threading
import time

from cluster_auto_role import AutoRoleController
from cluster_auto_role_runtime import AutoRoleRuntime
from cluster_control_contract import ControlPlaneAuthority
from cluster_quorum import QuorumPolicy
from tests.test_ha_role_auto import _fixture


def _controllers(tmp_path):
    voter_set, collector, authority, fence = _fixture(tmp_path)
    collector.policy = QuorumPolicy(lease_duration_ms=400)
    old = AutoRoleController("voter-0", authority=authority, collector=collector, fence=fence)
    next_authority = ControlPlaneAuthority(cluster_id=voter_set.cluster_id, voter_set=voter_set)
    candidate = AutoRoleController("voter-1", authority=next_authority, collector=collector)
    runtime = AutoRoleRuntime(voter=collector.voters["voter-1"], collector=collector, secret="test")
    return collector, old, candidate, runtime


def test_supervisor_takes_over_after_live_leader_lease_expires(tmp_path):
    collector, old, candidate, runtime = _controllers(tmp_path)
    voters = tuple(collector.voters)
    assert old.start(available_voter_ids=voters).accepted
    original = old.certificate
    assert not candidate.start(available_voter_ids=voters).accepted
    candidate.on_disconnect()
    elected = threading.Event()
    decisions = []

    def observed(decision):
        decisions.append(decision)
        if decision.state == "leader":
            elected.set()

    runtime.start_lease_renewal(candidate, on_decision=observed)
    try:
        assert elected.wait(4), candidate.snapshot()
        assert candidate.can_write()
        assert candidate.certificate.term > original.term
        assert not old.can_write()
        assert decisions[-1].leader_id == candidate.node_id
    finally:
        runtime.close()


def test_supervisor_without_majority_stays_read_only_and_keeps_voter_set(tmp_path):
    collector, old, candidate, runtime = _controllers(tmp_path)
    configured = collector.voter_set
    collector.voters = {"voter-1": collector.voters["voter-1"]}
    assert not candidate.start(available_voter_ids=tuple(collector.voters)).accepted
    observed = threading.Event()
    runtime.start_lease_renewal(candidate, on_decision=lambda decision: observed.set())
    try:
        assert observed.wait(2)
        assert not candidate.can_write()
        assert collector.voter_set == configured
        assert candidate.state == "read_only"
    finally:
        runtime.close()


def test_stopped_controller_never_reenters_election(tmp_path):
    collector, old, candidate, runtime = _controllers(tmp_path)
    candidate.stop()
    runtime.start_lease_renewal(candidate)
    try:
        time.sleep(.2)
        assert candidate.state == "stopped"
        assert all(v.snapshot().promised_term == 0 for v in collector.voters.values())
    finally:
        runtime.close()


def test_failed_runtime_binding_revokes_new_authority(tmp_path):
    collector, old, candidate, runtime = _controllers(tmp_path)
    attempted = threading.Event()

    def fail_binding(decision):
        if decision.state == "leader":
            attempted.set()
            raise RuntimeError("runtime role could not be bound")

    runtime.start_lease_renewal(candidate, on_decision=fail_binding)
    try:
        assert attempted.wait(2)
        deadline = time.monotonic() + 2
        while candidate.state == "leader" and time.monotonic() < deadline:
            time.sleep(.01)
        assert not candidate.can_write()
        assert candidate.state == "read_only"
    finally:
        runtime.close()


def test_follower_transport_consumes_proof_but_cannot_write_http(tmp_path):
    import pytest
    from cluster_fence import ControlFence, ControlFenceError
    from tcp_comm import TCPClient, MessageType

    collector, old, candidate, runtime = _controllers(tmp_path)
    collector.policy = QuorumPolicy(lease_duration_ms=10_000)
    assert old.start(available_voter_ids=tuple(collector.voters)).accepted
    fence = ControlFence(candidate.authority, required=True)
    fence.set_role_guard(candidate.can_write)
    fence.set_peer_guard(candidate.can_follow, consume_certificate=candidate.observe_peer_certificate)
    client = TCPClient(server_host="127.0.0.1", server_port=8888, client_id=candidate.node_id)
    client.set_control_fence(fence)
    frame = {"type": MessageType.NODE_LIST_SYNC.value,
             "data": {"control_certificate": old.certificate.to_dict(), "nodes": []}}
    client._admit_inbound_control(frame)
    assert candidate.state == "follower"
    assert candidate.can_follow() and not candidate.can_write()
    with pytest.raises(ControlFenceError) as denied:
        fence.admit(old.certificate, action="models.intent", source="http")
    assert denied.value.code == "control_certificate_not_current"
    candidate.on_disconnect()
    with pytest.raises(ControlFenceError):
        client._admit_inbound_control(frame)
    assert candidate.state == "read_only"


def test_scheduler_binding_fences_old_connections_and_rebuilds_master_resources(tmp_path, monkeypatch):
    from scheduler import Scheduler
    from types import SimpleNamespace

    collector, old, candidate, runtime = _controllers(tmp_path)
    collector.policy = QuorumPolicy(lease_duration_ms=10_000)
    assert candidate.start(available_voter_ids=tuple(collector.voters)).accepted
    scheduler = Scheduler()
    scheduler._auto_role_controller = candidate
    scheduler._auto_role_bound_role = "client"
    scheduler._running = True
    actions = []
    previous = SimpleNamespace(on_disconnect=lambda: None, disconnect=lambda: actions.append("old_socket_closed"))
    scheduler._tcp_client = previous
    scheduler._tcp_server = SimpleNamespace(disconnect_all_clients=lambda reason: actions.append(reason))
    scheduler._execution_app = SimpleNamespace(requests=SimpleNamespace(
        queue_snapshot=lambda: {"tasks": [{"request_id": "queued"}], "current_task_id": "active"},
        cancel=lambda request_id: actions.append(("cancel", request_id))))
    scheduler._model_supply = SimpleNamespace(cancel_active=lambda: actions.append("supply_cancelled"))
    scheduler._model_supply_worker = SimpleNamespace(cancel=lambda: actions.append("worker_supply_cancelled"))
    scheduler._worker_prepare = SimpleNamespace(retire_and_join=lambda *_a, **_kw: actions.append("native_retired"))
    monkeypatch.setattr(scheduler, "get_distributed_inference_enabled", lambda: True)
    monkeypatch.setattr(scheduler, "_on_master_connection_lost", lambda client, **kw: actions.append(("worker_cleanup", client is previous, kw)))
    monkeypatch.setattr(scheduler, "init_nodes", lambda: actions.append("init_master"))
    monkeypatch.setattr(scheduler, "_load_pipeline_recovery_state", lambda: actions.append("recovery"))
    monkeypatch.setattr(scheduler, "_restore_pipeline_model_for_recovery", lambda: actions.append("restore_metadata"))
    monkeypatch.setattr(scheduler, "_start_deferred_network_identity", lambda: actions.append("advertise"))
    scheduler._reconcile_auto_role_resources()
    assert scheduler._auto_role_bound_role == "master" and scheduler._auto_role_binding_ready
    assert scheduler._tcp_client is None and previous.on_disconnect is None
    assert actions.index("old_socket_closed") < actions.index("init_master") < actions.index("recovery")
    assert ("cancel", "queued") in actions and ("cancel", "active") in actions
    assert "authority_changed" in actions
    assert ("worker_cleanup", True, {"notify_auto_role": False}) in actions
    candidate.stop()


def test_scheduler_role_binding_does_not_block_lease_renewal(tmp_path, monkeypatch):
    from scheduler import Scheduler
    collector, old, candidate, runtime = _controllers(tmp_path)
    scheduler = Scheduler()
    scheduler._auto_role_controller = candidate
    scheduler._auto_role_bound_role = "client"
    scheduler._running = True
    entered, release = threading.Event(), threading.Event()
    def bind():
        entered.set()
        assert release.wait(3)
    monkeypatch.setattr(scheduler, "_reconcile_auto_role_resources", bind)
    runtime.start_lease_renewal(candidate, on_decision=scheduler._on_auto_role_decision)
    try:
        assert entered.wait(2)
        first = candidate.certificate
        deadline = time.monotonic() + 2
        while candidate.certificate.expires_at_ms <= first.expires_at_ms and time.monotonic() < deadline:
            time.sleep(.01)
        assert candidate.can_write()
        assert candidate.certificate.expires_at_ms > first.expires_at_ms
        assert not scheduler._auto_role_binding_ready
    finally:
        scheduler._running = False
        release.set()
        runtime.close()
        scheduler._stop_auto_role_binding()
        candidate.stop()


def test_auto_bootstrap_preserves_voter_identity_and_role_mode(tmp_path):
    import pytest
    from qlh_core.bootstrap.runtime import AppRuntime
    from tests.test_refactor_bootstrap import settings
    runtime = AppRuntime(settings(tmp_path, QLH_NODE_ROLE="auto", QLH_CLUSTER_SECRET="auto-secret"))
    runtime.open_identity()
    try:
        runtime.accept_bootstrap({"node": {"node_id": runtime.config.node_id, "role": "client"},
            "cluster": {"cluster_id": "auto-cluster", "cluster_secret": "auto-secret", "cluster_secret_epoch": 1}})
        assert runtime.config.node_role == "auto"
        assert runtime.config.environment_snapshot()["QLH_NODE_ROLE"] == "auto"
        with pytest.raises(ValueError, match="voter.*identity"):
            runtime.accept_bootstrap({"node": {"node_id": "another-voter", "role": "client"},
                "cluster": {"cluster_id": "auto-cluster", "cluster_secret": "auto-secret", "cluster_secret_epoch": 1}})
    finally:
        runtime.close()


def test_role_readout_does_not_emit_control_write_audit(tmp_path):
    """The API log handler reads runtime_role while holding its logging lock."""
    collector, old, candidate, runtime = _controllers(tmp_path)
    collector.policy = QuorumPolicy(lease_duration_ms=10_000)
    assert candidate.start(available_voter_ids=tuple(collector.voters)).accepted
    from cluster_fence import ControlFence
    audits = []
    candidate.fence = ControlFence(candidate.authority, required=True, audit_sink=audits.append)
    try:
        assert candidate.can_write()
        assert candidate.runtime_role == "master"
        assert candidate.snapshot()["writable"]
        assert audits == [], "role observation re-enters the logging handler and can deadlock"
    finally:
        candidate.stop()
        runtime.close()
