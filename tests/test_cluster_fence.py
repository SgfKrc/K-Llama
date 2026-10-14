from __future__ import annotations

import base64
import sys
import time

import pytest

sys.path.insert(0, "src")

from cluster_control_contract import (  # noqa: E402
    ControlPlaneAuthority,
    VoterIdentity,
    VoterSet,
    QuorumCertificate,
    sign_certificate,
)
from cluster_fence import (  # noqa: E402
    ControlFence,
    ControlFenceError,
    certificate_from_headers,
    encode_certificate_header,
)
from tcp_comm import MessageType, _decorate_control_payload  # noqa: E402


def _fixture():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    voters = []
    for index, key in enumerate(keys):
        public = base64.urlsafe_b64encode(
            key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("ascii").rstrip("=")
        voters.append(VoterIdentity(voter_id=f"voter-{index}", public_key=public))
    voter_set = VoterSet(cluster_id="fence-test", voter_set_epoch=1, voters=voters)
    unsigned = QuorumCertificate(
        cluster_id="fence-test", voter_set_epoch=1, term=3,
        leader_id="voter-0", lease_id="lease-3", expires_at_ms=int(time.time() * 1000) + 60_000,
        signatures=tuple(),
    )
    certificate = unsigned.with_signatures(tuple(
        sign_certificate(unsigned, voter_id=f"voter-{index}", private_key=keys[index])
        for index in (0, 1)
    ))
    return voter_set, certificate


def _fence():
    voter_set, certificate = _fixture()
    authority = ControlPlaneAuthority(cluster_id=voter_set.cluster_id, voter_set=voter_set)
    fence = ControlFence(authority, required=True)
    fence.install_certificate(certificate)
    return fence, certificate


def test_fence_rejects_missing_and_admits_exact_current_certificate():
    fence, certificate = _fence()
    with pytest.raises(ControlFenceError) as missing:
        fence.admit(None, action="cluster.layers.override")
    assert missing.value.code == "control_certificate_missing"
    permit = fence.admit(certificate, action="cluster.layers.override")
    assert permit is not None
    assert permit.term == 3


def test_fence_rejects_expired_certificate_and_header_roundtrip():
    fence, certificate = _fence()
    with pytest.raises(ControlFenceError) as expired:
        fence.admit(certificate, action="cluster.role.transfer", now_ms=certificate.expires_at_ms)
    assert expired.value.code == "control_certificate_expired"
    encoded = encode_certificate_header(certificate)
    assert certificate_from_headers({"X-QLH-Control-Certificate": encoded}) == certificate.to_dict()


def test_request_permit_is_invalidated_by_local_lease_revocation():
    fence, certificate = _fence()
    try:
        fence.admit_runtime(certificate, action="test.begin")
        assert fence.require_current_permit(action="test.continue") is not None
        assert fence.authority.revoke_certificate(certificate)
        with pytest.raises(ControlFenceError) as rejected:
            fence.require_current_permit(action="test.continue")
        assert rejected.value.code == "control_certificate_not_current"
    finally:
        fence.clear_context()


def test_auto_role_guard_rejects_a_valid_certificate_on_a_follower():
    fence, certificate = _fence()
    leader = {"active": True}
    fence.set_role_guard(lambda: leader["active"])
    try:
        fence.admit_runtime(certificate, action="test.start")
        leader["active"] = False
        with pytest.raises(ControlFenceError) as blocked:
            fence.admit(certificate, action="test.followup", source="http")
        assert blocked.value.code == "control_certificate_not_current"
        with pytest.raises(ControlFenceError) as in_flight:
            fence.require_current_permit(action="test.followup")
        assert in_flight.value.code == "control_certificate_not_current"
    finally:
        fence.clear_context()


def test_install_invalid_certificate_uses_stable_fence_error():
    fence = ControlFence(required=True)
    with pytest.raises(ControlFenceError) as rejected:
        fence.install_certificate({"invalid": True})
    assert rejected.value.code == "control_fence_unavailable"

    voter_set, _ = _fixture()
    authority = ControlPlaneAuthority(cluster_id=voter_set.cluster_id, voter_set=voter_set)
    fence = ControlFence(authority, required=True)
    with pytest.raises(ControlFenceError) as malformed:
        fence.install_certificate({"invalid": True})
    assert malformed.value.code == "control_certificate_invalid"


def test_tcp_control_payload_is_decorated_and_requires_current_certificate():
    fence, certificate = _fence()
    payload = _decorate_control_payload({"node_id": "worker"}, MessageType.LAYER_CONFIG, fence)
    assert payload["control_certificate"] == certificate.to_dict()
    with pytest.raises(ControlFenceError) as rejected:
        _decorate_control_payload({"node_id": "worker", "control_certificate": {"bad": True}}, MessageType.LAYER_CONFIG, fence)
    assert rejected.value.code == "control_certificate_invalid"


def test_disabled_fence_preserves_legacy_payloads():
    fence = ControlFence(required=False)
    assert _decorate_control_payload({"node_id": "worker"}, MessageType.LAYER_CONFIG, fence) == {"node_id": "worker"}


def test_auto_role_without_voter_set_requires_an_unavailable_fence(monkeypatch):
    monkeypatch.setenv("QLH_NODE_ROLE", "auto")
    monkeypatch.delenv("QLH_CONTROL_VOTER_SET", raising=False)
    fence = ControlFence.from_environment()
    assert fence.required is True
    with pytest.raises(ControlFenceError) as rejected:
        fence.admit(None, action="test.auto")
    assert rejected.value.code == "control_fence_unavailable"
