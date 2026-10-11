from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
from jsonschema import validate

sys.path.insert(0, "src")

from cluster_control_contract import (
    CONTROL_SCHEMA_VERSION,
    ControlContractError,
    ControlPlaneAuthority,
    QuorumCertificate,
    VoterIdentity,
    VoterSet,
    sign_certificate,
    validate_certificate,
)


def _keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public = base64.urlsafe_b64encode(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii").rstrip("=")
    return private, public


def _fixture():
    keys = [_keypair() for _ in range(3)]
    voter_set = VoterSet(
        cluster_id="cluster-test",
        voter_set_epoch=4,
        voters=tuple(
            VoterIdentity(voter_id=f"node-{index}", public_key=public)
            for index, (_, public) in enumerate(keys)
        ),
    )
    unsigned = QuorumCertificate(
        cluster_id="cluster-test",
        voter_set_epoch=4,
        term=7,
        leader_id="node-0",
        lease_id="lease-7",
        expires_at_ms=2_000,
        signatures=tuple(),
    )
    signed = unsigned.with_signatures(
        tuple(
            sign_certificate(unsigned, voter_id=f"node-{index}", private_key=keys[index][0])
            for index in (0, 1)
        )
    )
    return keys, voter_set, unsigned, signed


def test_contract_roundtrip_and_json_schema():
    _, voter_set, _, certificate = _fixture()
    schema = json.loads(
        Path("schemas/cluster-control-v1.schema.json").read_text(encoding="utf-8")
    )
    validate(voter_set.to_dict(), schema)
    validate(certificate.to_dict(), schema)
    assert QuorumCertificate.from_dict(certificate.to_dict()) == certificate
    assert VoterSet.from_dict(voter_set.to_dict()) == voter_set
    assert certificate.signing_bytes() == QuorumCertificate.from_dict(
        certificate.to_dict()
    ).signing_bytes()
    reversed_certificate = certificate.with_signatures(tuple(reversed(certificate.signatures)))
    assert reversed_certificate.digest() == certificate.digest()
    assert certificate.to_dict()["schema_version"] == CONTROL_SCHEMA_VERSION


def test_strict_majority_validation_and_snapshot_gate():
    _, voter_set, unsigned, certificate = _fixture()
    validation = validate_certificate(certificate, voter_set, now_ms=1_000)
    assert validation.valid_voter_ids == ("node-0", "node-1")
    assert validation.quorum_required == 2

    authority = ControlPlaneAuthority(
        cluster_id="cluster-test",
        voter_set=voter_set,
        static_role="master",
    )
    initial = authority.snapshot(now_ms=1_000)
    assert initial.mode == "read_only"
    assert initial.static_role == "master"
    assert initial.read_only_reason == "control_certificate_missing"
    with pytest.raises(ControlContractError) as missing:
        authority.admit_control_write(None, now_ms=1_000)
    assert missing.value.code == "control_certificate_missing"

    authority.install_certificate(certificate, now_ms=1_000)
    current = authority.snapshot(now_ms=1_000)
    assert current.mode == "writable"
    assert current.committed_term == 7
    permit = authority.admit_control_write(certificate.to_dict(), now_ms=1_000)
    assert permit.term == 7
    assert permit.lease_id == "lease-7"
    assert permit.certificate_digest == certificate.digest()

    expired = authority.snapshot(now_ms=2_000)
    assert expired.mode == "read_only"
    assert expired.read_only_reason == "control_certificate_expired"
    with pytest.raises(ControlContractError) as expired_error:
        authority.admit_control_write(certificate, now_ms=2_000)
    assert expired_error.value.code == "control_certificate_expired"
    assert unsigned.signatures == tuple()


def test_one_signature_is_not_a_quorum_and_duplicate_votes_fail():
    keys, voter_set, unsigned, _ = _fixture()
    one_signature = unsigned.with_signatures(
        (sign_certificate(unsigned, voter_id="node-0", private_key=keys[0][0]),)
    )
    with pytest.raises(ControlContractError) as no_quorum:
        validate_certificate(one_signature, voter_set, now_ms=1_000)
    assert no_quorum.value.code == "control_certificate_not_quorum"

    duplicate = unsigned.with_signatures(
        (
            sign_certificate(unsigned, voter_id="node-0", private_key=keys[0][0]),
            sign_certificate(unsigned, voter_id="node-0", private_key=keys[0][0]),
        )
    )
    with pytest.raises(ControlContractError) as duplicate_error:
        validate_certificate(duplicate, voter_set, now_ms=1_000)
    assert duplicate_error.value.code == "control_certificate_duplicate_vote"


def test_invalid_identity_epoch_cluster_and_signature_fail_closed():
    keys, voter_set, unsigned, certificate = _fixture()
    bad_signature = unsigned.with_signatures(
        (sign_certificate(unsigned, voter_id="node-0", private_key=keys[0][0]),)
    )
    bad_signature = bad_signature.with_signatures(
        bad_signature.signatures + (
            sign_certificate(
                QuorumCertificate(
                    cluster_id="cluster-test",
                    voter_set_epoch=4,
                    term=8,
                    leader_id="node-0",
                    lease_id="lease-8",
                    expires_at_ms=2_000,
                    signatures=tuple(),
                ),
                voter_id="node-1",
                private_key=keys[1][0],
            ),
        )
    )
    with pytest.raises(ControlContractError) as signature_error:
        validate_certificate(bad_signature, voter_set, now_ms=1_000)
    assert signature_error.value.code == "control_certificate_signature_invalid"

    with pytest.raises(ControlContractError) as cluster_error:
        validate_certificate(
            QuorumCertificate(
                cluster_id="other-cluster",
                voter_set_epoch=4,
                term=7,
                leader_id="node-0",
                lease_id="lease-7",
                expires_at_ms=2_000,
                signatures=certificate.signatures,
            ),
            voter_set,
            now_ms=1_000,
        )
    assert cluster_error.value.code == "control_certificate_cluster_mismatch"

    other_epoch = VoterSet(
        cluster_id="cluster-test",
        voter_set_epoch=5,
        voters=voter_set.voters,
    )
    with pytest.raises(ControlContractError) as epoch_error:
        validate_certificate(certificate, other_epoch, now_ms=1_000)
    assert epoch_error.value.code == "control_certificate_epoch_mismatch"


def test_authority_does_not_auto_activate_or_accept_same_term_conflict():
    keys, voter_set, unsigned, certificate = _fixture()
    authority = ControlPlaneAuthority(cluster_id="cluster-test", voter_set=voter_set)
    with pytest.raises(ControlContractError) as not_current:
        authority.admit_control_write(certificate, now_ms=1_000)
    assert not_current.value.code == "control_certificate_not_current"

    authority.install_certificate(certificate, now_ms=1_000)
    conflicting_unsigned = QuorumCertificate(
        cluster_id="cluster-test",
        voter_set_epoch=4,
        term=7,
        leader_id="node-1",
        lease_id="other-lease-7",
        expires_at_ms=2_000,
        signatures=tuple(),
    )
    conflicting = conflicting_unsigned.with_signatures(
        tuple(
            sign_certificate(conflicting_unsigned, voter_id=f"node-{index}", private_key=keys[index][0])
            for index in (0, 1)
        )
    )
    with pytest.raises(ControlContractError) as conflict:
        authority.install_certificate(conflicting, now_ms=1_000)
    assert conflict.value.code == "control_certificate_conflict"


def test_same_term_renewal_only_extends_identical_lease_and_revocation_is_final():
    keys, voter_set, _, original = _fixture()
    authority = ControlPlaneAuthority(cluster_id="cluster-test", voter_set=voter_set)
    authority.install_certificate(original, now_ms=1_000)
    extension = QuorumCertificate(
        cluster_id="cluster-test", voter_set_epoch=4, term=7,
        leader_id="node-0", lease_id="lease-7", expires_at_ms=3_000,
        signatures=tuple(),
    )
    renewed = extension.with_signatures(tuple(
        sign_certificate(extension, voter_id=f"node-{index}", private_key=keys[index][0])
        for index in (0, 2)
    ))
    authority.install_certificate(renewed, now_ms=1_001)
    assert authority.current_certificate() == renewed
    with pytest.raises(ControlContractError) as old:
        authority.admit_control_write(original, now_ms=1_002)
    assert old.value.code == "control_certificate_not_current"
    with pytest.raises(ControlContractError) as regression:
        authority.install_certificate(original, now_ms=1_002)
    assert regression.value.code == "control_certificate_conflict"
    assert authority.revoke_certificate(original) is False
    assert authority.revoke_certificate(renewed) is True
    with pytest.raises(ControlContractError) as replay:
        authority.install_certificate(renewed, now_ms=1_003)
    assert replay.value.code == "control_certificate_stale"


def test_certificate_mapping_rejects_noncanonical_fields():
    _, _, _, certificate = _fixture()
    payload = certificate.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ControlContractError) as error:
        QuorumCertificate.from_dict(payload)
    assert error.value.code == "control_certificate_invalid"
