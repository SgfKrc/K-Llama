"""HA-CROSSHOST-01 local process-boundary development gate."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cluster_crosshost import (  # noqa: E402
    CrossHostProcessHarness,
    write_crosshost_evidence,
)
from cluster_auto_role import AutoRoleController  # noqa: E402
from cluster_control_contract import ControlPlaneAuthority, VoterIdentity, VoterSet  # noqa: E402
from cluster_fence import ControlFence, ControlFenceError  # noqa: E402
from cluster_handoff import HandoffCoordinator  # noqa: E402
from cluster_quorum import QuorumCollector, QuorumPolicy, QuorumVoter, SQLiteVoterLedger  # noqa: E402
from cluster_recovery import decide_recovery  # noqa: E402


def _ha_fixture(root: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    identities = []
    for index, key in enumerate(keys):
        public_key = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        identities.append(VoterIdentity(
            voter_id=f"voter-{index}",
            public_key=base64.urlsafe_b64encode(public_key).decode("ascii").rstrip("="),
        ))
    voter_set = VoterSet(cluster_id="crosshost-test", voter_set_epoch=1, voters=identities)
    voters = {
        f"voter-{index}": QuorumVoter(
            voter_id=f"voter-{index}",
            private_key=key,
            voter_set=voter_set,
            ledger=SQLiteVoterLedger(
                root / f"voter-{index}.sqlite3",
                voter_id=f"voter-{index}",
                cluster_id=voter_set.cluster_id,
                voter_set_epoch=voter_set.voter_set_epoch,
            ),
        )
        for index, key in enumerate(keys)
    }
    collector = QuorumCollector(
        voter_set=voter_set,
        voters=voters,
        policy=QuorumPolicy(lease_duration_ms=100),
    )
    authority = ControlPlaneAuthority(cluster_id=voter_set.cluster_id, voter_set=voter_set)
    fence = ControlFence(authority, required=True)
    old = AutoRoleController("voter-0", authority=authority, collector=collector, fence=fence)
    new = AutoRoleController("voter-1", authority=authority, collector=collector, fence=fence)
    started = old.start(available_voter_ids=tuple(voter_set.voter_map), now_ms=1_000)
    assert started.accepted and old.certificate is not None
    coordinator = HandoffCoordinator(
        "voter-0",
        authority=authority,
        collector=collector,
        fence=fence,
        old_role_controller=old,
        new_role_controller=new,
    )
    return voter_set, authority, fence, old, new, coordinator


def test_dual_process_restart_fences_old_generation_and_reconnects():
    evidence = CrossHostProcessHarness(timeout_seconds=5).run(
        term_history=(1, 2),
        write_rejections=({"source": "control_fence", "code": "control_certificate_stale"},),
        task_outcomes=({"workflow_id": "wf-crosshost-1", "action": "retry", "result": "retry_pending"},),
        rpo_last_durable_sequence=7,
        rpo_lost_events=0,
    )

    assert evidence.safe
    assert evidence.physical_nodes is False
    assert evidence.process_count == 2
    assert evidence.term_history == (1, 2)
    assert evidence.rto_ms >= 0
    assert evidence.rpo_lost_events == 0
    assert {event["event"] for event in evidence.events} >= {
        "old_generation_rejected",
        "control_frame_accepted_after_reconnect",
    }
    assert {item["code"] for item in evidence.write_rejections} >= {
        "generation_stale",
        "control_certificate_stale",
    }
    assert evidence.task_outcomes[0]["result"] == "retry_pending"


def test_crosshost_metadata_is_attached_from_real_ha_components(tmp_path: Path, monkeypatch):
    # Metadata assertions use a deterministic clock; lease expiry has separate coverage.
    from qlh_core.control import clock
    monkeypatch.setattr(clock, "operation_clock", lambda now_ms=None: lambda: now_ms)
    voter_set, authority, fence, old, new, coordinator = _ha_fixture(tmp_path / "quorum")
    old_certificate = old.certificate
    assert old_certificate is not None
    coordinator.prepare(
        "voter-1",
        {"journal_sequence": 7},
        reason="crosshost-reconnect",
        operator="test",
        now_ms=1_001,
        handoff_id="crosshost-handoff",
    )
    waiting = coordinator.commit(available_voter_ids=("voter-0",), now_ms=1_101)
    assert waiting.state == "awaiting_quorum"
    assert old.can_write(now_ms=1_101) is False

    committed = coordinator.commit(
        available_voter_ids=tuple(voter_set.voter_map),
        now_ms=1_202,
        lease_id="crosshost-lease",
    )
    assert committed.state == "committed"
    assert committed.new_term is not None
    with pytest.raises(ControlFenceError) as rejected:
        fence.admit(old_certificate, action="crosshost.old_write", now_ms=1_202)
    assert rejected.value.code == "control_certificate_expired"

    recovery = decide_recovery({
        "workflow_id": "wf-crosshost-recovery",
        "last_sequence": 7,
        "state": "running",
        "stages": [{
            "stage_id": "pure-stage",
            "state": "running",
            "pure": True,
            "retry_safe": True,
            "attempts": [{"attempt_id": "attempt-1", "state": "running"}],
        }],
    })
    evidence = CrossHostProcessHarness(timeout_seconds=5).run()
    evidence = evidence.with_control_metadata(
        term_history=(old_certificate.term, committed.new_term),
        write_rejections=({"source": "control_fence", "code": rejected.value.code},),
        task_outcomes=(recovery.to_dict(),),
        rpo_last_durable_sequence=7,
    )

    assert evidence.term_history == (1, 2)
    assert {item["code"] for item in evidence.write_rejections} == {
        "generation_stale",
        "control_certificate_expired",
    }
    assert evidence.task_outcomes[0]["action"] == "retry"
    assert evidence.rpo_last_durable_sequence == 7


def test_crosshost_evidence_is_atomic_and_does_not_include_payload(tmp_path: Path):
    evidence = CrossHostProcessHarness(timeout_seconds=5).run()
    path = write_crosshost_evidence(tmp_path / "ha" / "crosshost.json", evidence)
    assert path.is_file()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["schema_version"] == "qlh.cluster.crosshost.v1"
    assert body["physical_nodes"] is False
    assert "crosshost-control-metadata" not in path.read_text(encoding="utf-8")
    assert not path.with_name(".crosshost.json.part").exists()
