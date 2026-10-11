from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from cluster_auto_role import AutoRoleController  # noqa: E402
from cluster_control_contract import (  # noqa: E402
    ControlPlaneAuthority,
    VoterIdentity,
    VoterSet,
)
from cluster_fence import ControlFence  # noqa: E402
from cluster_quorum import (  # noqa: E402
    QuorumCollector,
    QuorumPolicy,
    QuorumVoter,
    SQLiteVoterLedger,
)


def _fixture(root: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    identities = []
    for index, key in enumerate(keys):
        public = base64.urlsafe_b64encode(
            key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("ascii").rstrip("=")
        identities.append(VoterIdentity(voter_id=f"voter-{index}", public_key=public))
    voter_set = VoterSet(cluster_id="auto-role-test", voter_set_epoch=1, voters=identities)
    voters = {}
    for index, key in enumerate(keys):
        voter_id = f"voter-{index}"
        voters[voter_id] = QuorumVoter(
            voter_id=voter_id,
            private_key=key,
            voter_set=voter_set,
            ledger=SQLiteVoterLedger(
                root / f"{voter_id}.sqlite3",
                voter_id=voter_id,
                cluster_id=voter_set.cluster_id,
                voter_set_epoch=voter_set.voter_set_epoch,
            ),
        )
    collector = QuorumCollector(
        voter_set=voter_set,
        voters=voters,
        policy=QuorumPolicy(lease_duration_ms=100),
    )
    authority = ControlPlaneAuthority(
        cluster_id=voter_set.cluster_id,
        voter_set=voter_set,
    )
    fence = ControlFence(authority, required=True)
    return voter_set, collector, authority, fence


def test_auto_role_without_majority_is_read_only(tmp_path: Path):
    _, collector, authority, fence = _fixture(tmp_path)
    controller = AutoRoleController(
        "voter-0", authority=authority, collector=collector, fence=fence
    )

    missing = controller.start()
    assert missing.state == "read_only"
    assert missing.reason == "quorum_unavailable"
    assert controller.can_write(now_ms=1_000) is False

    minority = controller.start(available_voter_ids=("voter-0",), now_ms=1_000)
    assert minority.state == "read_only"
    assert minority.reason == "quorum_unavailable"
    assert controller.snapshot(now_ms=1_000)["writable"] is False


def test_auto_role_installs_quorum_certificate_and_can_write(tmp_path: Path):
    voter_set, collector, authority, fence = _fixture(tmp_path)
    controller = AutoRoleController(
        "voter-0", authority=authority, collector=collector, fence=fence
    )

    decision = controller.start(
        available_voter_ids=tuple(voter_set.voter_map),
        now_ms=1_000,
    )
    assert decision.accepted is True
    assert decision.state == "leader"
    assert decision.term == 1
    assert controller.can_write(now_ms=1_001) is True
    assert controller.snapshot(now_ms=1_001)["writable"] is True


def test_auto_role_foreign_leader_is_follower_and_not_writable(tmp_path: Path):
    voter_set, collector, authority, fence = _fixture(tmp_path)
    issued = collector.acquire(
        "voter-0", available_voter_ids=tuple(voter_set.voter_map), now_ms=1_000
    )
    assert issued.accepted and issued.certificate is not None
    controller = AutoRoleController(
        "voter-2", authority=authority, collector=collector, fence=fence
    )

    decision = controller.consume_certificate(issued.certificate, now_ms=1_001)
    assert decision.state == "follower"
    assert decision.runtime_role == "client"
    assert controller.can_write(now_ms=1_001) is False


def test_disconnect_requires_new_quorum_before_reclaiming_write(tmp_path: Path):
    voter_set, collector, authority, fence = _fixture(tmp_path)
    controller = AutoRoleController(
        "voter-0", authority=authority, collector=collector, fence=fence
    )
    initial = controller.start(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=1_000
    )
    assert initial.accepted and controller.certificate is not None
    old_certificate = controller.certificate

    assert controller.on_disconnect().state == "read_only"
    assert controller.can_write(now_ms=1_001) is False
    replay = controller.on_reconnect(certificate=old_certificate, now_ms=1_001)
    assert replay.state == "read_only"
    assert replay.reason == "quorum_unavailable"

    recovered = controller.on_reconnect(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=1_101
    )
    assert recovered.state == "leader"
    assert recovered.term == 2
    assert controller.can_write(now_ms=1_102) is True


def test_expired_old_certificate_cannot_restore_write_after_new_term(tmp_path: Path):
    voter_set, collector, authority, fence = _fixture(tmp_path)
    controller = AutoRoleController(
        "voter-0", authority=authority, collector=collector, fence=fence
    )
    first = controller.start(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=1_000
    )
    assert first.accepted and controller.certificate is not None
    old_certificate = controller.certificate
    newer = collector.acquire(
        "voter-2", available_voter_ids=("voter-1", "voter-2"), now_ms=1_101
    )
    assert newer.accepted and newer.certificate is not None
    controller.consume_certificate(newer.certificate, now_ms=1_101)

    rejected = controller.consume_certificate(old_certificate, now_ms=1_101)
    assert rejected.accepted is False
    assert rejected.state == "read_only"
    assert rejected.reason == "control_certificate_expired"
    assert controller.can_write(now_ms=1_101) is False


def test_static_master_and_client_remain_compatible(tmp_path: Path):
    master = AutoRoleController("master", mode="master")
    client = AutoRoleController("client", mode="client")
    assert master.start().state == "leader"
    assert master.can_write() is True
    assert client.start().state == "follower"
    assert client.can_write() is False


def test_scheduler_can_attach_controller_without_enabling_global_auto_mode(tmp_path: Path):
    from scheduler import Scheduler

    voter_set, collector, authority, fence = _fixture(tmp_path)
    collector.policy = QuorumPolicy(lease_duration_ms=10_000)
    controller = AutoRoleController(
        "voter-0", authority=authority, collector=collector, fence=fence
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._role_override = None
    scheduler._auto_role_controller = None
    scheduler.set_auto_role_controller(controller)

    assert scheduler._effective_role() == "client"
    now_ms = int(time.time() * 1000)
    started = scheduler.start_auto_role(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=now_ms
    )
    assert started["accepted"] is True
    assert scheduler._effective_role() == "master"
    assert scheduler.get_auto_role_snapshot(now_ms=now_ms + 1)["enabled"] is True
    assert scheduler.auto_role_on_disconnect()["state"] == "read_only"
    assert scheduler.auto_role_on_reconnect(
        certificate=controller.certificate, now_ms=now_ms + 1
    )["reason"] == "quorum_unavailable"
    scheduler.set_auto_role_controller(None)
    assert scheduler.get_auto_role_snapshot()["enabled"] is False


def test_renewal_updates_authority_and_losing_quorum_revokes_old_permits(tmp_path: Path):
    voter_set, collector, authority, fence = _fixture(tmp_path)
    controller = AutoRoleController("voter-0", authority=authority, collector=collector, fence=fence)
    assert controller.start(available_voter_ids=tuple(voter_set.voter_map), now_ms=1_000).accepted
    first = controller.certificate

    renewed = controller.renew_lease(available_voter_ids=("voter-0", "voter-1"), now_ms=1_050)
    assert renewed.accepted and renewed.term == first.term
    assert controller.certificate.expires_at_ms == 1_150
    assert authority.current_certificate() == controller.certificate
    assert controller.can_write(now_ms=1_101)
    with pytest.raises(Exception) as stale:
        fence.admit(first, action="test.old", now_ms=1_051)
    assert stale.value.code == "control_certificate_not_current"

    lost = controller.renew_lease(available_voter_ids=("voter-0",), now_ms=1_110)
    assert lost.state == "read_only"
    assert controller.can_write(now_ms=1_111) is False
    assert authority.current_certificate() is None
    with pytest.raises(Exception) as revoked:
        fence.admit(controller.certificate, action="test.revoked", now_ms=1_111)
    assert revoked.value.code == "control_certificate_not_current"


def test_stopping_auto_role_fences_unexpired_certificate(tmp_path: Path):
    voter_set, collector, authority, fence = _fixture(tmp_path)
    controller = AutoRoleController("voter-0", authority=authority, collector=collector, fence=fence)
    assert controller.start(available_voter_ids=tuple(voter_set.voter_map), now_ms=1_000).accepted
    certificate = controller.certificate
    assert controller.stop().state == "stopped"
    assert controller.on_quorum_loss().state == "stopped"
    assert controller.renew_lease(available_voter_ids=tuple(voter_set.voter_map)).state == "stopped"
    assert authority.current_certificate() is None
    with pytest.raises(Exception) as stopped:
        fence.admit(certificate, action="test.stopped", now_ms=1_001)
    assert stopped.value.code == "control_certificate_not_current"


def test_in_flight_renewal_cannot_restore_authority_after_stop(tmp_path: Path, monkeypatch):
    import threading

    voter_set, collector, authority, fence = _fixture(tmp_path)
    controller = AutoRoleController("voter-0", authority=authority, collector=collector, fence=fence)
    assert controller.start(available_voter_ids=tuple(voter_set.voter_map), now_ms=1_000).accepted
    entered = threading.Event()
    release = threading.Event()
    real_renew = collector.renew

    def slow_renew(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return real_renew(*args, **kwargs)

    monkeypatch.setattr(collector, "renew", slow_renew)
    results = []
    thread = threading.Thread(target=lambda: results.append(controller.renew_lease(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=1_050,
    )))
    thread.start()
    try:
        assert entered.wait(timeout=5)
        controller.stop()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert results[0].accepted is False
    assert controller.state == "stopped"
    assert authority.current_certificate() is None


def test_auto_role_override_cannot_claim_master_without_a_certificate(monkeypatch):
    import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "NODE_ROLE", "auto")
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler._auto_role_controller = None
    scheduler._role_override = "master"
    assert scheduler._effective_role() == "client"
