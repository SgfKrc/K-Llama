from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

from cluster_auto_role import AutoRoleController, AutoRoleDecision  # noqa: E402
from cluster_control_contract import (  # noqa: E402
    ControlPlaneAuthority,
    VoterIdentity,
    VoterSet,
)
from cluster_fence import ControlFence  # noqa: E402
from cluster_handoff import HandoffCoordinator, HandoffError  # noqa: E402
from cluster_quorum import (  # noqa: E402
    QuorumCollector,
    QuorumPolicy,
    QuorumVoter,
    SQLiteVoterLedger,
)


@pytest.fixture(autouse=True)
def _fixed_handoff_operation_time(monkeypatch):
    """Explicit times in these state transitions must not depend on SQLite IO.

    Freeze only the operation clock's sources, leaving process timers intact.
    Lease boundaries remain the explicit 1000/1101 ms used by each test.
    """
    from qlh_core.control import clock

    monkeypatch.setattr(clock, "time", SimpleNamespace(time=lambda: 1.0, monotonic=lambda: 1.0))


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
    voter_set = VoterSet(cluster_id="handoff-test", voter_set_epoch=1, voters=identities)
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
    old = AutoRoleController(
        "voter-0", authority=authority, collector=collector, fence=fence
    )
    new = AutoRoleController(
        "voter-1", authority=authority, collector=collector, fence=fence
    )
    started = old.start(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=1_000
    )
    assert started.accepted
    return voter_set, collector, authority, fence, old, new


def _coordinator(tmp_path: Path, events: list[dict]):
    voter_set, collector, authority, fence, old, new = _fixture(tmp_path)
    coordinator = HandoffCoordinator(
        "voter-0",
        authority=authority,
        collector=collector,
        fence=fence,
        old_role_controller=old,
        new_role_controller=new,
        event_sink=events.append,
    )
    return voter_set, coordinator, old, new


def test_prepare_is_idempotent_and_records_only_manifest_digest(tmp_path: Path):
    events: list[dict] = []
    _, coordinator, _, _ = _coordinator(tmp_path, events)
    manifest = {"layout_sha256": "abc", "journal_sequence": 4}
    first = coordinator.prepare(
        "voter-1", manifest, reason="planned maintenance", operator="admin", now_ms=1_001,
        handoff_id="handoff-1",
    )
    replay = coordinator.prepare(
        "voter-1", manifest, reason="planned maintenance", operator="admin", now_ms=1_002,
        handoff_id="handoff-1-replay",
    )
    assert replay == first
    assert first.old_term == 1
    assert first.new_term is None
    assert first.manifest_digest != "abc"
    assert "layout_sha256" not in coordinator.snapshot()["record"]
    assert [event["event_type"] for event in events] == ["handoff_prepared"]


def test_commit_fences_old_role_before_new_term_is_installed(tmp_path: Path):
    events: list[dict] = []
    voter_set, coordinator, old, new = _coordinator(tmp_path, events)
    coordinator.prepare(
        "voter-1", {"layout_sha256": "abc"}, reason="maintenance", operator="admin", now_ms=1_001,
    )
    result = coordinator.commit(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=1_101, lease_id="handoff-lease",
    )
    assert result.state == "committed"
    assert result.old_term == 1
    assert result.new_term == 2
    assert result.old_leader_id == "voter-0"
    assert result.new_leader_id == "voter-1"
    assert old.state == "read_only"
    assert old.can_write(now_ms=1_101) is False
    assert new.state == "leader"
    assert new.can_write(now_ms=1_101) is True
    assert [event["event_type"] for event in events] == [
        "handoff_prepared", "handoff_old_leader_fenced", "handoff_committed"
    ]


def test_no_majority_keeps_handoff_awaiting_and_old_role_fenced(tmp_path: Path):
    events: list[dict] = []
    _, coordinator, old, new = _coordinator(tmp_path, events)
    coordinator.prepare(
        "voter-1", {"layout_sha256": "abc"}, reason="maintenance", operator="admin", now_ms=1_001,
    )
    result = coordinator.commit(
        available_voter_ids=("voter-0",), now_ms=1_101,
    )
    assert result.state == "awaiting_quorum"
    assert result.result == "quorum_unavailable"
    assert old.state == "read_only"
    assert old.can_write(now_ms=1_101) is False
    assert new.state == "starting"
    with pytest.raises(HandoffError) as error:
        coordinator.abort(now_ms=1_102)
    assert error.value.code == "handoff_old_leader_fenced"


def test_recovery_timeout_keeps_old_role_fenced(tmp_path: Path):
    events: list[dict] = []
    voter_set, coordinator, old, new = _coordinator(tmp_path, events)
    coordinator.prepare(
        "voter-1", {"layout_sha256": "abc"}, reason="maintenance", operator="admin", now_ms=1_001,
    )
    waiting = coordinator.commit(
        available_voter_ids=("voter-0",), now_ms=1_101,
    )
    assert waiting.state == "awaiting_quorum"

    still_waiting = coordinator.recover_pending(now_ms=1_101 + 29_999, timeout_ms=30_000)
    assert still_waiting is not None
    assert still_waiting.state == "awaiting_quorum"
    timed_out = coordinator.recover_pending(now_ms=1_101 + 30_000, timeout_ms=30_000)
    assert timed_out is not None
    assert timed_out.state == "failed"
    assert timed_out.result == "handoff_timeout"
    assert old.can_write(now_ms=1_101 + 30_000) is False
    assert new.state == "starting"
    assert events[-1]["event_type"] == "handoff_timeout"


def test_old_certificate_change_aborts_commit_without_releasing_old_state(tmp_path: Path):
    events: list[dict] = []
    voter_set, coordinator, old, _ = _coordinator(tmp_path, events)
    coordinator.prepare(
        "voter-1", {"layout_sha256": "abc"}, reason="maintenance", operator="admin", now_ms=1_001,
    )
    replacement = old.collector.acquire(
        "voter-2", available_voter_ids=("voter-1", "voter-2"), now_ms=1_101,
    )
    assert replacement.accepted and replacement.certificate is not None
    old.authority.install_certificate(replacement.certificate, now_ms=1_101)
    with pytest.raises(HandoffError) as error:
        coordinator.commit(available_voter_ids=tuple(voter_set.voter_map), now_ms=1_102)
    assert error.value.code == "handoff_certificate_changed"
    assert old.state == "leader"


def test_target_rejection_is_terminal_and_audited(tmp_path: Path, monkeypatch):
    events: list[dict] = []
    voter_set, coordinator, old, new = _coordinator(tmp_path, events)
    coordinator.prepare(
        "voter-1", {"layout_sha256": "abc"}, reason="maintenance", operator="admin", now_ms=1_001,
    )
    monkeypatch.setattr(
        new,
        "consume_certificate",
        lambda *args, **kwargs: AutoRoleDecision(
            accepted=False,
            state="read_only",
            runtime_role="client",
            reason="target_not_ready",
        ),
    )
    result = coordinator.commit(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=1_101,
    )
    assert result.state == "failed"
    assert result.result == "handoff_target_rejected"
    assert old.state == "read_only"
    assert events[-1]["event_type"] == "handoff_failed"


def test_invalid_target_and_manifest_are_rejected(tmp_path: Path):
    _, coordinator, _, _ = _coordinator(tmp_path, [])
    with pytest.raises(HandoffError) as target_error:
        coordinator.prepare(
            "unknown", {"layout_sha256": "abc"}, reason="maintenance", operator="admin", now_ms=1_001,
        )
    assert target_error.value.code == "handoff_target_invalid"
    with pytest.raises(HandoffError) as manifest_error:
        coordinator.prepare(
            "voter-1", {}, reason="maintenance", operator="admin", now_ms=1_001,
        )
    assert manifest_error.value.code == "handoff_manifest_invalid"


def test_scheduler_exposes_handoff_only_when_explicitly_attached(tmp_path: Path):
    from scheduler import Scheduler

    events: list[dict] = []
    voter_set, coordinator, _, _ = _coordinator(tmp_path, events)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._handoff_coordinator = None
    assert scheduler.get_handoff_snapshot()["enabled"] is False
    assert scheduler.prepare_leader_handoff(
        "voter-1", {"layout_sha256": "abc"}, reason="maintenance", operator="admin",
    )["status"] == "disabled"

    persisted: list[tuple[str, str, dict]] = []
    coordinator.event_sink = None
    scheduler._append_ha_log = lambda category, direction, details: persisted.append(
        (category, direction, details)
    )
    scheduler.set_handoff_coordinator(coordinator)
    assert coordinator.event_sink == scheduler._persist_handoff_event
    prepared = scheduler.prepare_leader_handoff(
        "voter-1", {"layout_sha256": "abc"}, reason="maintenance", operator="admin", now_ms=1_001,
    )
    assert prepared["state"] == "prepared"
    committed = scheduler.commit_leader_handoff(
        available_voter_ids=tuple(voter_set.voter_map), now_ms=1_101,
    )
    assert committed["state"] == "committed"
    assert scheduler.get_handoff_snapshot()["enabled"] is True
    assert [item[1] for item in persisted] == [
        "handoff_prepared", "handoff_old_leader_fenced", "handoff_committed"
    ]
