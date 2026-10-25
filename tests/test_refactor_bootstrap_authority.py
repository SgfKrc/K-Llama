"""R1 recovery fault injection and persistent voting safety contracts."""
from __future__ import annotations
from dataclasses import replace
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from cluster_auto_role import AutoRoleController
from cluster_control_contract import CommitWatermark, ControlContractError, QuorumCertificate
from cluster_quorum import QuorumCollector, QuorumError, SQLiteVoterLedger
from tests.test_cluster_quorum import _fixture
from tests.test_refactor_authority_defects import certs, fixed_clock  # noqa: F401

pytestmark = pytest.mark.refactor_contract


@pytest.mark.parametrize("action", ["stop", "disconnect"])
@pytest.mark.parametrize("concurrent", [False, True])
def test_first_takeover_cannot_publish_after_cancel(certs, fixed_clock, action, concurrent):
    entered, resume = threading.Event(), threading.Event()
    decisions = []
    def guard(certificate, *, permit):
        assert certificate is permit.certificate
        if concurrent:
            entered.set()
            assert resume.wait(5)
        else:
            cancel()
    controller = AutoRoleController("voter-0", authority=certs.authority, committed_prefix_guard=guard)
    cancel = controller.stop if action == "stop" else controller.on_disconnect
    candidate = certs.issue(expires_at_ms=3000)
    def consume():
        decisions.append(controller.consume_certificate(candidate, now_ms=1000))
    if concurrent:
        worker = threading.Thread(target=consume)
        worker.start()
        try:
            assert entered.wait(5)
            cancel()
        finally:
            resume.set()
            worker.join(5)
        assert not worker.is_alive()
    else:
        consume()
    assert len(decisions) == 1 and not decisions[0].accepted
    assert controller.state == ("stopped" if action == "stop" else "read_only")
    assert certs.authority.current_certificate() is None
    assert not controller.can_write(now_ms=1100)


def test_failed_takeover_never_restores_old_write_proof(certs, fixed_clock):
    old = certs.issue(expires_at_ms=2000)
    certs.authority.install_certificate(old, now_ms=1000)
    new = certs.issue(term=8, lease_id="lease-8", expires_at_ms=3000)
    def fail(permit):
        with pytest.raises(ControlContractError):
            certs.authority.admit_control_write(old, now_ms=1000)
        raise OSError("recovery disk error")
    with pytest.raises(OSError):
        certs.authority.install_certificate(new, now_ms=1000, recovery=fail)
    assert certs.authority.current_certificate() is None
    for candidate in (old, new):
        with pytest.raises(ControlContractError):
            certs.authority.admit_control_write(candidate, now_ms=1100)



@pytest.mark.parametrize("interruption", ["failure", "cancel", "revoke", "expiry"])
def test_interrupted_recovery_cannot_be_bypassed_by_recovery_free_install(certs, fixed_clock, interruption):
    from cluster_fence import ControlFence
    from qlh_core.contracts.authority import AuthorityError

    authority = certs.authority
    authority._service._clock = lambda: fixed_clock[0]
    fence = ControlFence(authority, required=True)
    candidate = certs.issue(expires_at_ms=2000)
    permits, writes = [], []

    def interrupted(permit):
        permits.append(permit)
        permit.mutate(lambda: writes.append("partial recovery"))
        if interruption == "failure":
            raise OSError("recovery failed after a partial write")
        if interruption == "cancel":
            authority.cancel_recovery()
        elif interruption == "revoke":
            assert authority.revoke_certificate(candidate)
        else:
            fixed_clock[0] = 2000

    with pytest.raises(OSError if interruption == "failure" else ControlContractError):
        fence.install_certificate(candidate, now_ms=1000, recovery=interrupted)
    assert authority.current_certificate() is None
    newer = certs.issue(term=8, lease_id="next", expires_at_ms=4000)
    for retry in (candidate, certs.issue(expires_at_ms=4000), newer):
        with pytest.raises(ControlContractError):
            fence.install_certificate(retry, now_ms=fixed_clock[0])
        assert authority.current_certificate() is None
        assert authority.snapshot(now_ms=fixed_clock[0]).mode == "read_only"
        with pytest.raises(ControlContractError):
            authority.admit_control_write(retry, now_ms=fixed_clock[0])
    assert authority.snapshot(now_ms=fixed_clock[0]).committed_term == candidate.term

    def recovered(permit):
        permit.check()
        permit.mutate(lambda: writes.append("protected recovery completed"))

    fence.install_certificate(newer, now_ms=fixed_clock[0], recovery=recovered)
    assert authority.current_certificate() == newer
    authority.admit_control_write(newer, now_ms=fixed_clock[0])
    with pytest.raises(AuthorityError):
        permits[0].mutate(lambda: writes.append("late recovery"))
    renewed = certs.issue(term=8, lease_id="next", expires_at_ms=5000)
    fence.install_certificate(renewed, now_ms=fixed_clock[0])
    assert authority.current_certificate() == renewed
    assert writes == ["partial recovery", "protected recovery completed"]


def test_failed_recovery_can_retry_same_proof_with_protected_recovery(certs, fixed_clock):
    candidate = certs.issue(expires_at_ms=3000)

    def fail(permit):
        raise OSError("retryable disk error")

    with pytest.raises(OSError):
        certs.authority.install_certificate(candidate, now_ms=1000, recovery=fail)
    recovered = []
    certs.authority.install_certificate(
        candidate, now_ms=1000, recovery=lambda permit: permit.mutate(lambda: recovered.append(candidate.digest())))
    assert recovered == [candidate.digest()]
    certs.authority.admit_control_write(candidate, now_ms=1000)


def test_initial_recovery_free_install_and_renewal_remain_supported(certs, fixed_clock):
    initial = certs.issue(expires_at_ms=2000)
    certs.authority.install_certificate(initial, now_ms=1000)
    renewed = certs.issue(expires_at_ms=3000)
    certs.authority.install_certificate(renewed, now_ms=1100)
    certs.authority.admit_control_write(renewed, now_ms=1100)


def test_renewal_updates_proof_without_replaying_takeover(certs, fixed_clock):
    calls = []
    first = certs.issue(expires_at_ms=2000)
    certs.authority.install_certificate(first, now_ms=1000, recovery=lambda permit: calls.append(permit))
    identity = certs.authority.authority_identity()
    renewed = certs.issue(expires_at_ms=3000)
    certs.authority.install_certificate(renewed, now_ms=1100, recovery=lambda permit: pytest.fail("renewal reran recovery"))
    assert len(calls) == 1 and certs.authority.authority_identity() == identity
    assert certs.authority.admit_bound_identity(identity, now_ms=1100).certificate_digest == renewed.digest()
    with pytest.raises(ControlContractError):
        certs.authority.admit_control_write(first, now_ms=1100)


def test_recovery_uses_immutable_validated_input(certs, fixed_clock):
    original = certs.issue(expires_at_ms=3000)
    mutable = original.to_dict()
    def recover(permit):
        mutable["leader_id"] = "voter-2"
        mutable["expires_at_ms"] = 999999
        assert permit.certificate == original
        permit.check()
    certs.authority.install_certificate(mutable, now_ms=1000, recovery=recover)
    assert certs.authority.current_certificate() == original


def test_certificate_expiring_during_recovery_stays_fenced(certs, fixed_clock):
    clock = [1000]
    certs.authority._service._clock = lambda: clock[0]
    def expire(permit):
        clock[0] = 2000
    with pytest.raises(ControlContractError) as error:
        certs.authority.install_certificate(certs.issue(expires_at_ms=2000), recovery=expire)
    assert error.value.code == "control_certificate_expired"
    assert certs.authority.current_certificate() is None
    assert certs.authority.snapshot(now_ms=2000).mode == "read_only"


@pytest.mark.parametrize("changed", ["omitted", "regressed", "forked"])
def test_signed_floor_survives_crash_before_certificate_commit(tmp_path, changed):
    voter_set, voters = _fixture(tmp_path)
    voter = voters["voter-0"]
    floor = CommitWatermark(workflow_id="workflow", last_sequence=8, checkpoint_digest="a" * 64)
    voter.watermark_source = lambda workflow: floor
    certificate = QuorumCertificate(
        cluster_id=voter_set.cluster_id, voter_set_epoch=1, term=1, leader_id="voter-0",
        lease_id="lease", expires_at_ms=2000, signatures=(), commit_watermark=(floor,))
    voter.prepare("voter-0", 1, now_ms=1000)
    voter.sign_vote(certificate, now_ms=1000)
    # Reopen the actual SQLite file without delivering commit_certificate.
    reopened = SQLiteVoterLedger(voter.ledger.path, voter_id=voter.voter_id,
                                 cluster_id=voter_set.cluster_id, voter_set_epoch=1)
    assert reopened.recovery_floors() == (floor,)
    assert reopened.committed_floors() == ()
    proposed = () if changed == "omitted" else (replace(floor, last_sequence=7) if changed == "regressed"
                                               else replace(floor, checkpoint_digest="b" * 64),)
    next_certificate = replace(certificate, term=2, lease_id="next", expires_at_ms=4000, commit_watermark=proposed)
    reopened.prepare("voter-1", 2, now_ms=2100)
    next_certificate = replace(next_certificate, leader_id="voter-1")
    with pytest.raises(QuorumError) as rejected:
        reopened.sign_vote(next_certificate, private_key=voter.private_key, now_ms=2100)
    assert rejected.value.code == "quorum_watermark_unheld"
    assert reopened.recovery_floors() == (floor,)


def test_legacy_ledger_requires_verified_matching_manifest(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    floor = CommitWatermark(workflow_id="workflow", last_sequence=8, checkpoint_digest="a" * 64)
    for voter in voters.values():
        voter.watermark_source = lambda workflow: floor
    issued = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", now_ms=1000, lease_id="old", commit_watermark=(floor,))
    assert issued.accepted
    certificate = issued.certificate
    voter = voters["voter-0"]
    with sqlite3.connect(voter.ledger.path) as connection:
        connection.execute("DELETE FROM quorum_metadata WHERE key='recovery_floor_schema'")
        connection.execute("DELETE FROM quorum_recovery_floors")
    reopened = SQLiteVoterLedger(voter.ledger.path, voter_id=voter.voter_id,
                                 cluster_id=voter_set.cluster_id, voter_set_epoch=1)
    renewed = replace(certificate, expires_at_ms=certificate.expires_at_ms + 1000, signatures=())
    with pytest.raises(QuorumError) as error:
        reopened.sign_vote(renewed, private_key=voter.private_key, now_ms=1100)
    assert error.value.code == "quorum_watermark_unavailable"
    with pytest.raises(ControlContractError):
        reopened.import_recovery_manifest(replace(certificate, commit_watermark=()), voter_set)
    assert reopened.recovery_floors() == ()
    reopened.import_recovery_manifest(certificate, voter_set)
    assert reopened.recovery_floors() == reopened.committed_floors() == (floor,)
    reopened.sign_vote(renewed, private_key=voter.private_key, now_ms=1100)


def test_renewal_close_timeout_preserves_live_owner():
    from cluster_auto_role_runtime import AutoRoleRuntime
    entered, finish = threading.Event(), threading.Event()
    thread = threading.Thread(target=lambda: (entered.set(), finish.wait(5)))
    runtime = AutoRoleRuntime(voter=None, collector=None, secret="secret")
    runtime._renewal_thread = thread
    runtime._renewal_stop = threading.Event()
    thread.start()
    assert entered.wait(5)
    try:
        with pytest.raises(TimeoutError):
            runtime.stop_lease_renewal(timeout=0)
        assert runtime._renewal_thread is thread
    finally:
        finish.set()
        runtime.stop_lease_renewal(timeout=5)
    assert runtime._renewal_thread is None


def test_invalid_auto_role_settings_open_no_durable_resources(tmp_path):
    import json
    from cluster_auto_role_runtime import AutoRoleRuntimeError, build_auto_role_runtime
    from tests.test_cluster_auto_role_runtime import _key_material, _private_key_text
    keys, voter_set = _key_material()
    directory = tmp_path / "never-opened"
    environment = {"QLH_CONTROL_VOTER_SET": json.dumps(voter_set.to_dict()),
                   "QLH_CONTROL_VOTER_ID": "voter-0", "QLH_CONTROL_VOTER_PRIVATE_KEY": _private_key_text(keys[0]),
                   "QLH_CONTROL_VOTER_LEDGER": str(directory / "ledger.sqlite3"),
                   "QLH_JOURNAL_REPLICA": str(directory / "replica.sqlite3")}
    with pytest.raises(AutoRoleRuntimeError) as error:
        build_auto_role_runtime(environment=environment)
    assert error.value.code == "auto_role_rpc_secret_missing"
    assert not directory.exists()


@pytest.mark.parametrize("cancel_kind", ["stop", "expire"])
def test_reentrant_cancel_or_expiry_between_prefix_read_and_truncate(certs, fixed_clock, cancel_kind):
    from cluster_auto_role_runtime import JournalPrefixAuthority
    sequence = [5]
    writes = []
    def durable_prefix(workflow):
        if cancel_kind == "stop":
            controller.stop()
        else:
            fixed_clock[0] = 3000
        return {"last_sequence": sequence[0]}
    def truncate(workflow, *, last_sequence):
        writes.append(last_sequence)
        sequence[0] = last_sequence
        return {"removed_events": 2}
    replica = SimpleNamespace(durable_prefix=durable_prefix, truncate_to=truncate,
                              recovery_status=lambda workflow: {"verified": True, "durable_sequence": sequence[0],
                                                                 "checkpoint_digest": "a" * 64})
    prefix = JournalPrefixAuthority(replica, workflows=("wf",))
    controller = AutoRoleController("voter-0", authority=certs.authority, committed_prefix_guard=prefix.enforce)
    certs.authority._service._clock = lambda: fixed_clock[0]
    floor = CommitWatermark(workflow_id="wf", last_sequence=3, checkpoint_digest="a" * 64)
    decision = controller.consume_certificate(certs.issue(expires_at_ms=2000, watermarks=(floor,)), now_ms=1000)
    assert not decision.accepted and writes == [] and sequence == [5]


def test_cancelled_recovery_drains_before_new_authority_can_activate(certs, fixed_clock):
    entered, resume = threading.Event(), threading.Event()
    mutations, errors = [], []
    def recover(permit):
        entered.set()
        assert resume.wait(5)
        permit.mutate(lambda: mutations.append("old"))
    def install_old():
        try:
            certs.authority.install_certificate(certs.issue(expires_at_ms=2000), now_ms=1000, recovery=recover)
        except ControlContractError as exc:
            errors.append(exc.code)
    worker = threading.Thread(target=install_old)
    worker.start()
    try:
        assert entered.wait(5)
        certs.authority.cancel_recovery()
        newer = certs.issue(term=8, lease_id="new", expires_at_ms=3000)
        with pytest.raises(ControlContractError) as error:
            certs.authority.install_certificate(newer, now_ms=1000)
        assert error.value.code == "control_certificate_not_current"
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive() and mutations == [] and errors == ["control_certificate_not_current"]
    with pytest.raises(ControlContractError) as error:
        certs.authority.install_certificate(newer, now_ms=1000)
    assert error.value.code == "control_certificate_not_current"
    certs.authority.install_certificate(
        newer, now_ms=1000, recovery=lambda permit: permit.mutate(lambda: mutations.append("new")))
    assert mutations == ["new"] and certs.authority.current_certificate() == newer


def test_checkpoint_fetched_after_cancellation_is_not_applied(certs, fixed_clock, monkeypatch):
    import journal_replication_service as replication
    from tests.test_journal_replication_recovery import _settings
    writes = []
    def request(*args, **kwargs):
        certs.authority.cancel_recovery()
        return {"fetched": "late"}
    monkeypatch.setattr(replication, "request_checkpoint", request)
    replica = SimpleNamespace(apply_checkpoint=lambda checkpoint: writes.append(checkpoint))
    def recover(permit):
        replication.recover_journal_prefix(replica, workflow_id="wf", settings=_settings((("127.0.0.1", 1),)),
                                            mutation_guard=permit.mutate)
    with pytest.raises(ControlContractError):
        certs.authority.install_certificate(certs.issue(expires_at_ms=2000), now_ms=1000, recovery=recover)
    assert writes == [] and certs.authority.current_certificate() is None


@pytest.mark.parametrize("renewal", [False, True])
def test_majority_signed_uncommitted_lease_excludes_next_leader_after_reopen(tmp_path, renewal):
    from cluster_control_contract import validate_certificate
    voter_set, voters = _fixture(tmp_path)
    collector = QuorumCollector(voter_set=voter_set, voters=voters)
    if renewal:
        initial = collector.acquire("voter-0", now_ms=1000)
        assert initial.accepted
        unsigned = replace(initial.certificate, expires_at_ms=14000, signatures=())
        signed_at, contender_at = 1100, 12000
    else:
        unsigned = QuorumCertificate(cluster_id=voter_set.cluster_id, voter_set_epoch=1, term=1,
                                     leader_id="voter-0", lease_id="signed-only", expires_at_ms=5000,
                                     signatures=(), commit_watermark=())
        signed_at, contender_at = 1000, 1001
        for voter in voters.values():
            voter.prepare("voter-0", 1, now_ms=signed_at)
    signed = unsigned.with_signatures(tuple(voter.sign_vote(unsigned, now_ms=signed_at) for voter in voters.values()))
    validate_certificate(signed, voter_set, now_ms=contender_at)
    for voter in voters.values():
        voter.ledger = SQLiteVoterLedger(voter.ledger.path, voter_id=voter.voter_id,
                                         cluster_id=voter_set.cluster_id, voter_set_epoch=1)
    contender = collector.acquire("voter-1", now_ms=contender_at)
    assert not contender.accepted
    assert collector.acquire("voter-1", now_ms=signed.expires_at_ms + 1).accepted


def test_quorum_io_expiry_rejected_before_any_recovery(tmp_path, fixed_clock, monkeypatch):
    from cluster_control_contract import ControlPlaneAuthority
    voter_set, voters = _fixture(tmp_path)
    collector = QuorumCollector(voter_set=voter_set, voters=voters)
    original = collector.acquire
    def delayed(*args, **kwargs):
        outcome = original(*args, **kwargs)
        assert outcome.accepted
        fixed_clock[0] = outcome.certificate.expires_at_ms + 1
        return outcome
    monkeypatch.setattr(collector, "acquire", delayed)
    recovered = []
    authority = ControlPlaneAuthority(cluster_id=voter_set.cluster_id, voter_set=voter_set)
    controller = AutoRoleController("voter-0", authority=authority, collector=collector,
                                    committed_prefix_guard=lambda certificate: recovered.append(certificate))
    decision = controller.attempt_takeover(available_voter_ids=tuple(voters), now_ms=1000)
    assert not decision.accepted and recovered == [] and authority.current_certificate() is None


def floor_pair(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    floor5 = CommitWatermark(workflow_id="wf", last_sequence=5, checkpoint_digest="a" * 64)
    floor10 = replace(floor5, last_sequence=10, checkpoint_digest="b" * 64)
    for voter in voters.values():
        voter.watermark_source = lambda workflow: floor10
    collector = QuorumCollector(voter_set=voter_set, voters=voters)
    initial = collector.acquire("voter-0", now_ms=1000, commit_watermark=(floor5,))
    assert initial.accepted
    pending = replace(initial.certificate, expires_at_ms=12000, signatures=(), commit_watermark=(floor10,))
    pending = pending.with_signatures(tuple(voter.sign_vote(pending, now_ms=1100) for voter in voters.values()))
    return voter_set, voters["voter-0"], initial.certificate, pending, floor10


def test_import_cannot_overwrite_new_schema_accepted_floor(tmp_path):
    voter_set, voter, active, pending, floor = floor_pair(tmp_path)
    with pytest.raises(QuorumError) as rejected:
        voter.ledger.import_recovery_manifest(active, voter_set)
    assert rejected.value.code == "quorum_migration_not_required"
    assert voter.ledger.recovery_floors() == (floor,)


@pytest.mark.parametrize("retain_floor_table", [False, True])
def test_legacy_import_cannot_unlock_without_pending_signed_manifest(tmp_path, retain_floor_table):
    voter_set, voter, active, pending, floor = floor_pair(tmp_path)
    with sqlite3.connect(voter.ledger.path) as connection:
        connection.execute("DELETE FROM quorum_metadata WHERE key='recovery_floor_schema'")
        if not retain_floor_table:
            connection.execute("DELETE FROM quorum_recovery_floors")
    reopened = SQLiteVoterLedger(voter.ledger.path, voter_id=voter.voter_id,
                                 cluster_id=voter_set.cluster_id, voter_set_epoch=1)
    with pytest.raises(QuorumError) as rejected:
        reopened.import_recovery_manifest(active, voter_set)
    assert rejected.value.code == "quorum_watermark_unavailable"
    reopened.import_recovery_manifest(active, voter_set, accepted_certificates=(pending,))
    assert reopened.recovery_floors() == (floor,)
    assert reopened.committed_floors() == ()
    with pytest.raises(QuorumError):
        reopened.import_recovery_manifest(active, voter_set, accepted_certificates=(pending,))


@pytest.mark.parametrize("operation", ["truncate", "checkpoint"])
def test_recovery_transaction_rolls_back_if_permission_expires_before_commit(tmp_path, operation):
    from qlh_core.contracts.authority import AuthorityError
    from journal_replication import DurableJournalReplica, build_checkpoint
    from tests.test_journal_commit_truncate import _replica_with, SOURCE, STREAM
    source, existing, _ = _replica_with(tmp_path, "wf", 5)
    replica = existing if operation == "truncate" else DurableJournalReplica(
        tmp_path / "empty.sqlite3", source_node_id=SOURCE, stream_id=STREAM)
    calls = []
    def check():
        calls.append(1)
        if len(calls) == 2:
            raise AuthorityError("control_certificate_expired", "expired before durable commit")
    before = replica.durable_prefix("wf")
    try:
        with pytest.raises(AuthorityError):
            if operation == "truncate":
                replica.truncate_to("wf", last_sequence=3, permission_check=check)
            else:
                checkpoint = build_checkpoint(source, source_node_id=SOURCE, stream_id=STREAM,
                                               workflow_id="wf", created_at=1.0)
                replica.apply_checkpoint(checkpoint, permission_check=check)
        assert len(calls) == 2 and replica.durable_prefix("wf") == before
        assert replica.truncated_watermark("wf") == ""
    finally:
        source.close()
        existing.close()
        if replica is not existing:
            replica.close()
