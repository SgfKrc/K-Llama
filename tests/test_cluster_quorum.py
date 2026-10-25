from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, "src")

from cluster_control_contract import VoterIdentity, VoterSet  # noqa: E402
from cluster_quorum import (  # noqa: E402
    QuorumCollector,
    QuorumError,
    QuorumPolicy,
    QuorumVoter,
    SQLiteVoterLedger,
)


@pytest.fixture(autouse=True)
def _fixed_quorum_transition_clock(monkeypatch):
    """Keep explicit state-machine times independent of durable IO latency.

    Cross-IO clock advance and expiry are injected in the R1 authority tests.
    """
    from qlh_core.control import clock
    monkeypatch.setattr(clock, "time", SimpleNamespace(time=lambda: 1.0, monotonic=lambda: 1.0))


def _fixture(tmp_path: Path, *, count: int = 3):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pairs = []
    identities = []
    for index in range(count):
        private = Ed25519PrivateKey.generate()
        public = base64.urlsafe_b64encode(
            private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("ascii").rstrip("=")
        voter_id = f"voter-{index}"
        pairs.append((voter_id, private))
        identities.append(VoterIdentity(voter_id=voter_id, public_key=public))
    voter_set = VoterSet(cluster_id="cluster-quorum", voter_set_epoch=1, voters=identities)
    voters = {
        voter_id: QuorumVoter(
            voter_id=voter_id,
            private_key=private,
            voter_set=voter_set,
            ledger=SQLiteVoterLedger(
                tmp_path / f"{voter_id}.sqlite3",
                voter_id=voter_id,
                cluster_id=voter_set.cluster_id,
                voter_set_epoch=voter_set.voter_set_epoch,
            ),
        )
        for voter_id, private in pairs
    }
    return voter_set, voters


def test_three_voters_issue_durable_quorum_certificate(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", now_ms=1_000, lease_id="lease-1"
    )
    assert result.accepted is True
    assert result.reason == "quorum_certificate_issued"
    assert result.term == 1
    assert result.certificate is not None
    assert len(result.signed_voters) >= voter_set.quorum_size
    assert set(result.signed_voters) <= set(voters)
    assert {voter.snapshot().active_term for voter in voters.values()} == {1}
    assert voters["voter-0"].snapshot().promised_term == 1


@pytest.mark.parametrize("count", [1, 2])
def test_single_or_two_voters_without_witness_are_read_only(tmp_path, count):
    voter_set, voters = _fixture(tmp_path, count=count)
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", now_ms=1_000
    )
    assert result.accepted is False
    assert result.reason == "quorum_requires_witness"


def test_two_reachable_voters_can_issue_when_third_configured_voter_is_witness(tmp_path):
    voter_set, voters = _fixture(tmp_path, count=3)
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", available_voter_ids=("voter-0", "voter-1"),
        now_ms=1_000, lease_id="lease-witness",
    )
    assert result.accepted is True
    assert result.term == 1


def test_partition_has_at_most_one_certificate_and_minority_is_read_only(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    left = QuorumCollector(voter_set=voter_set, voters=voters)
    right = QuorumCollector(voter_set=voter_set, voters=voters)
    barrier = threading.Barrier(3)
    results = []

    def acquire(collector, leader, available):
        barrier.wait(timeout=5)
        results.append(collector.acquire(leader, available_voter_ids=available, now_ms=1_000))

    left_thread = threading.Thread(target=acquire, args=(left, "voter-0", ("voter-0", "voter-1")))
    right_thread = threading.Thread(target=acquire, args=(right, "voter-2", ("voter-1", "voter-2")))
    left_thread.start()
    right_thread.start()
    barrier.wait(timeout=5)
    left_thread.join(timeout=5)
    right_thread.join(timeout=5)
    assert len(results) == 2
    assert sum(result.accepted for result in results) <= 1
    assert any(result.reason == "quorum_certificate_issued" for result in results)


def test_term_is_durable_and_old_term_cannot_reuse_a_voter_for_another_leader(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    collector = QuorumCollector(voter_set=voter_set, voters=voters)
    first = collector.acquire("voter-0", now_ms=1_000, lease_id="lease-1")
    assert first.accepted
    second = collector.acquire(
        "voter-2", available_voter_ids=("voter-1", "voter-2"),
        now_ms=1_001, lease_id="lease-2",
    )
    assert second.accepted is False
    assert second.reason == "quorum_unavailable"
    assert voters["voter-1"].snapshot().promised_term == 1


def test_same_term_is_persistently_bound_to_one_leader(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    voters["voter-0"].prepare("voter-0", 1, now_ms=1_000)
    with pytest.raises(QuorumError) as error:
        voters["voter-0"].prepare("voter-2", 1, now_ms=1_001)
    assert getattr(error.value, "code", None) == "quorum_term_conflict"
    reopened = SQLiteVoterLedger(
        tmp_path / "voter-0.sqlite3", voter_id="voter-0",
        cluster_id="cluster-quorum", voter_set_epoch=1,
    )
    assert reopened.snapshot().promised_leader == "voter-0"


def test_voter_identity_and_leader_must_match_the_configured_voter_set(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    with pytest.raises(QuorumError) as identity_error:
        QuorumVoter(
            voter_id="voter-0",
            private_key=voters["voter-0"].private_key,
            voter_set=voter_set,
            ledger=SQLiteVoterLedger(
                tmp_path / "wrong-cluster.sqlite3", voter_id="voter-0",
                cluster_id="other-cluster", voter_set_epoch=1,
            ),
        )
    assert identity_error.value.code == "quorum_invalid_identity"
    with pytest.raises(QuorumError) as leader_error:
        voters["voter-0"].reserve_term("not-a-voter", now_ms=1_000)
    assert leader_error.value.code == "quorum_invalid_leader"


def test_expired_certificate_allows_higher_term_and_old_certificate_expires(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    collector = QuorumCollector(
        voter_set=voter_set, voters=voters, policy=QuorumPolicy(lease_duration_ms=100)
    )
    first = collector.acquire("voter-0", now_ms=1_000, lease_id="lease-1")
    assert first.accepted and first.certificate is not None
    second = collector.acquire(
        "voter-2", available_voter_ids=("voter-1", "voter-2"),
        now_ms=1_101, lease_id="lease-2",
    )
    assert second.accepted is True
    assert second.term == 2
    from cluster_control_contract import ControlContractError, validate_certificate
    with pytest.raises(ControlContractError) as expired:
        validate_certificate(first.certificate, voter_set, now_ms=1_101)
    assert expired.value.code == "control_certificate_expired"


def test_renew_keeps_term_and_lease_id_but_extends_expiry(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    collector = QuorumCollector(
        voter_set=voter_set, voters=voters, policy=QuorumPolicy(lease_duration_ms=100)
    )
    first = collector.acquire("voter-0", now_ms=1_000, lease_id="lease-1")
    renewed = collector.renew(first.certificate, now_ms=1_050)
    assert renewed.accepted is True
    assert renewed.certificate is not None
    assert renewed.term == first.term
    assert renewed.certificate.lease_id == "lease-1"
    assert renewed.certificate.expires_at_ms == 1_150


def test_renew_can_use_recovered_voter_outside_original_majority(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    collector = QuorumCollector(
        voter_set=voter_set, voters=voters, policy=QuorumPolicy(lease_duration_ms=100)
    )
    first = collector.acquire(
        "voter-0", available_voter_ids=("voter-0", "voter-1"), now_ms=1_000
    )
    assert first.accepted
    renewed = collector.renew(
        first.certificate, available_voter_ids=("voter-0", "voter-2"), now_ms=1_050
    )
    assert renewed.accepted
    assert renewed.signed_voters == ("voter-0", "voter-2")
    assert voters["voter-2"].snapshot().active_certificate_digest == renewed.certificate.digest()


def test_ledger_reopen_preserves_term_and_active_certificate(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", now_ms=1_000, lease_id="lease-1"
    )
    assert result.certificate is not None
    reopened = SQLiteVoterLedger(
        tmp_path / "voter-0.sqlite3", voter_id="voter-0",
        cluster_id="cluster-quorum", voter_set_epoch=1,
    )
    snapshot = reopened.snapshot()
    assert snapshot.promised_term == 1
    assert snapshot.active_term == 1
    assert snapshot.active_lease_id == "lease-1"


def test_certificate_attempt_with_one_signature_is_not_writable(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    result = QuorumCollector(
        voter_set=voter_set, voters=voters
    ).acquire("voter-0", available_voter_ids=("voter-0",), now_ms=1_000)
    assert result.accepted is False
    assert result.reason == "quorum_unavailable"


def test_commit_requires_the_local_durable_vote(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    collector = QuorumCollector(voter_set=voter_set, voters=voters)
    result = collector.acquire("voter-0", now_ms=1_000, lease_id="lease-1")
    assert result.accepted and result.certificate is not None
    voter = voters["voter-0"]
    connection = voter.ledger._connect()
    try:
        connection.execute("DELETE FROM quorum_votes WHERE voter_id = ?", ("voter-0",))
    finally:
        connection.close()
    with pytest.raises(QuorumError) as error:
        voter.commit_certificate(result.certificate, now_ms=1_000)
    assert error.value.code == "quorum_vote_rejected"
