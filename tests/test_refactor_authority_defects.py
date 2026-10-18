"""Authority target contracts using real prepare, quorum and replica transitions."""
import base64
from dataclasses import replace
from types import SimpleNamespace
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from cluster_auto_role import AutoRoleController
from cluster_auto_role_runtime import JournalPrefixAuthority
from cluster_control_contract import (
    CommitWatermark, ControlContractError, ControlPlaneAuthority, QuorumCertificate,
    VoterIdentity, VoterSet, sign_certificate, validate_certificate,
)
from cluster_fence import ControlFence
from cluster_quorum import QuorumPolicy
from journal_replication import build_checkpoint
from tests.test_cluster_auto_role_watermark import _voters
from tests.test_journal_commit_truncate import SOURCE, STREAM, _replica_with, _snapshot
from tests.test_worker_prepare import setup, identity, run, offer  # noqa: F401
from worker_prepare_contract import WorkerPrepareError

pytestmark = pytest.mark.refactor_contract


class LeaseRenewalInvalidated(AssertionError):
    pass


class CommittedFloorDropped(AssertionError):
    pass


class UnauthorizedRecoveryMutation(AssertionError):
    pass


@pytest.fixture
def fixed_clock(monkeypatch):
    now = [1000]
    monkeypatch.setattr(time, "time", lambda: now[0] / 1000)
    return now


@pytest.fixture
def certs():
    keys = [Ed25519PrivateKey.from_private_bytes(bytes([i + 1]) * 32) for i in range(3)]
    voter_set = VoterSet(
        cluster_id="contract-r0", voter_set_epoch=1,
        voters=tuple(VoterIdentity(voter_id=f"voter-{i}", public_key=base64.urlsafe_b64encode(
            key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        ).decode("ascii").rstrip("=")) for i, key in enumerate(keys)),
    )

    def issue(*, expires_at_ms, term=7, lease_id="lease-7", watermarks=()):
        unsigned = QuorumCertificate(
            cluster_id=voter_set.cluster_id, voter_set_epoch=1, term=term, leader_id="voter-0",
            lease_id=lease_id, expires_at_ms=expires_at_ms, signatures=(), commit_watermark=tuple(watermarks),
        )
        return unsigned.with_signatures(tuple(
            sign_certificate(unsigned, voter_id=f"voter-{i}", private_key=keys[i]) for i in (0, 1)
        ))

    return SimpleNamespace(voter_set=voter_set, issue=issue,
                           authority=ControlPlaneAuthority(cluster_id=voter_set.cluster_id, voter_set=voter_set))


def test_same_lease_renewal_preserves_stage_admission(setup, fixed_clock, certs):
    first = certs.issue(expires_at_ms=2000)
    renewed = certs.issue(expires_at_ms=3000)
    authority = certs.authority
    authority.install_certificate(first, now_ms=1000)
    setup.scheduler._control_fence = ControlFence(authority, required=True)
    value = identity()
    assert run(setup, "prepare", value, control_certificate=first.to_dict())["status"] == "prepared"
    assert run(setup, "commit", value, control_certificate=first.to_dict())["status"] == "ready"
    setup.controller.check_stage(offer(value))
    fixed_clock[0] = 1100
    authority.install_certificate(renewed, now_ms=1100)
    authority.admit_control_write(renewed, now_ms=1100)
    assert (first.term, first.leader_id, first.lease_id) == (renewed.term, renewed.leader_id, renewed.lease_id)
    assert first.digest() != renewed.digest()
    assert setup.host.worker_runtime_ready(value) and len(setup.built) == 1
    try:
        setup.controller.check_stage(offer(value))
    except WorkerPrepareError as exc:
        if type(exc) is WorkerPrepareError and exc.reason_code == "worker_prepare_fence_changed":
            raise LeaseRenewalInvalidated(exc.reason_code) from exc
        raise


def test_partial_renewal_proposal_cannot_drop_committed_floor(tmp_path, fixed_clock):
    voter_set, collector, authority = _voters(tmp_path)
    collector.policy = QuorumPolicy(lease_duration_ms=1000)
    a = CommitWatermark(workflow_id="wf-a", last_sequence=10, checkpoint_digest="a" * 64)
    b = CommitWatermark(workflow_id="wf-b", last_sequence=20, checkpoint_digest="b" * 64)
    held = {"wf-a": a, "wf-b": b}
    for voter in collector.voters.values():
        voter.watermark_source = held.get
    unreadable = set()

    def recovery_status(workflow_id):
        if workflow_id in unreadable:
            raise OSError("injected journal read failure")
        watermark = held[workflow_id]
        return {"verified": True, "durable_sequence": watermark.last_sequence,
                "checkpoint_digest": watermark.checkpoint_digest}

    prefix = JournalPrefixAuthority(SimpleNamespace(recovery_status=recovery_status), workflows=("wf-a", "wf-b"))
    controller = AutoRoleController("voter-0", authority=authority, collector=collector,
                                    commit_watermark_provider=prefix.propose)
    voters = tuple(voter_set.voter_map)
    assert controller.attempt_takeover(available_voter_ids=voters, now_ms=1000, lease_id="lease-wm").accepted
    old = controller.certificate
    assert old is not None and set(old.commit_watermark) == {a, b}
    unreadable.add("wf-b")
    assert prefix.propose() == (a,)
    fixed_clock[0] = 1100
    decision = controller.renew_lease(available_voter_ids=voters, now_ms=1100)
    if not decision.accepted:
        assert not controller.can_write(now_ms=1100)
        return
    current = controller.certificate
    assert current is not None
    actual = {w.workflow_id: w for w in current.commit_watermark}
    if actual == {"wf-a": a} and controller.can_write(now_ms=1100):
        raise CommittedFloorDropped("renewal accepted and writable after wf-b disappeared")
    for floor in old.commit_watermark:
        assert floor.workflow_id in actual
        value = actual[floor.workflow_id]
        assert value.last_sequence >= floor.last_sequence
        if value.last_sequence == floor.last_sequence:
            assert value.checkpoint_digest == floor.checkpoint_digest


@pytest.mark.parametrize("kind,expected_code", [
    ("expired", "control_certificate_expired"),
    ("bad_signature", "control_certificate_signature_invalid"),
    ("stale_term", "control_certificate_stale"),
])
def test_invalid_certificate_cannot_mutate_replica(tmp_path, fixed_clock, certs, kind, expected_code):
    wf = "wf-tail"
    source, replica, _ = _replica_with(tmp_path, wf, 5)
    try:
        prefix3 = SimpleNamespace(get_snapshot=lambda workflow_id: _snapshot(workflow_id, 3),
                                  list_events=lambda workflow_id: source.list_events(workflow_id)[:3])
        checkpoint3 = build_checkpoint(prefix3, source_node_id=SOURCE, stream_id=STREAM,
                                       workflow_id=wf, created_at=1.0)
        floor = CommitWatermark(workflow_id=wf, last_sequence=3, checkpoint_digest=checkpoint3.checkpoint_digest)
        candidate = certs.issue(expires_at_ms=1000 if kind == "expired" else 2000, watermarks=(floor,))
        if kind == "bad_signature":
            candidate = replace(candidate, expires_at_ms=2001)
        authority = certs.authority
        if kind == "stale_term":
            authority.install_certificate(certs.issue(term=8, lease_id="lease-8", expires_at_ms=3000), now_ms=1000)
        with pytest.raises(ControlContractError) as error:
            if kind == "stale_term":
                authority.install_certificate(candidate, now_ms=1000)
            else:
                validate_certificate(candidate, certs.voter_set, now_ms=1000)
        assert type(error.value) is ControlContractError and error.value.code == expected_code
        prefix = JournalPrefixAuthority(replica, workflows=(wf,))
        controller = AutoRoleController("voter-0", authority=authority, committed_prefix_guard=prefix.enforce)

        def journal_state():
            return (replica.durable_prefix(wf), replica.truncated_watermark(wf), replica.recovery_status(wf))

        before = journal_state()
        assert before[0]["last_sequence"] == 5 and before[1] == "" and before[2]["verified"] is True
        decision = controller.consume_certificate(candidate, now_ms=1000)
        assert not decision.accepted and not controller.can_write(now_ms=1000)
        after = journal_state()
        if (after[0]["last_sequence"] == 3 and after[0]["event_count"] == 3 and after[1] == "3@1"
                and after[2]["verified"] is False and prefix.truncated == {wf: 2}
                and decision.reason == "committed_prefix_unverified"):
            raise UnauthorizedRecoveryMutation(f"{kind}: replica 5 -> 3 before authority validation")
        assert after == before
        assert prefix.truncated == {} and decision.reason == expected_code
    finally:
        source.close()
        replica.close()
