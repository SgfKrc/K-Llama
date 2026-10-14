from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "src")

from bootstrap_credentials import (
    BootstrapCredentialError,
    create_bootstrap_credential_request,
    open_bootstrap_credential,
    seal_bootstrap_credential,
)


def _envelope(*, now: int = 1_000):
    request = create_bootstrap_credential_request(now=now)
    envelope = seal_bootstrap_credential(
        public_key=request.public_key,
        request_nonce=request.request_nonce,
        requested_at=request.requested_at,
        cluster_id="cluster-a",
        node_id="client-a",
        cluster_secret="s" * 32,
        secret_epoch=4,
        now=now + 1,
    )
    return request, envelope


def test_bootstrap_credential_roundtrip_never_serializes_plaintext_secret():
    request, envelope = _envelope()

    encoded = json.dumps(envelope, sort_keys=True)
    assert "s" * 32 not in encoded
    assert "cluster_secret" not in encoded

    opened = open_bootstrap_credential(
        envelope,
        private_key=request.private_key,
        expected_request_nonce=request.request_nonce,
        expected_cluster_id="cluster-a",
        expected_node_id="client-a",
        now=1_002,
    )
    assert opened["cluster_secret"] == "s" * 32
    assert opened["cluster_secret_epoch"] == 4


def test_bootstrap_credential_rejects_expiry_binding_and_ciphertext_tamper():
    request, envelope = _envelope()

    with pytest.raises(BootstrapCredentialError) as expired:
        open_bootstrap_credential(
            envelope,
            private_key=request.private_key,
            expected_request_nonce=request.request_nonce,
            now=1_121,
        )
    assert expired.value.code == "credential_expired"

    with pytest.raises(BootstrapCredentialError) as wrong_node:
        open_bootstrap_credential(
            envelope,
            private_key=request.private_key,
            expected_request_nonce=request.request_nonce,
            expected_node_id="client-b",
            now=1_002,
        )
    assert wrong_node.value.code == "binding_mismatch"

    tampered = json.loads(json.dumps(envelope))
    tampered["ciphertext"] = tampered["ciphertext"][:-1] + (
        "A" if tampered["ciphertext"][-1] != "A" else "B"
    )
    with pytest.raises(BootstrapCredentialError) as bad_ciphertext:
        open_bootstrap_credential(
            tampered,
            private_key=request.private_key,
            expected_request_nonce=request.request_nonce,
            now=1_002,
        )
    assert bad_ciphertext.value.code == "credential_auth_failed"


def test_bootstrap_request_rejects_stale_requested_at():
    request = create_bootstrap_credential_request(now=1_000)
    with pytest.raises(BootstrapCredentialError) as stale:
        seal_bootstrap_credential(
            public_key=request.public_key,
            request_nonce=request.request_nonce,
            requested_at=request.requested_at,
            cluster_id="cluster-a",
            node_id="client-a",
            cluster_secret="s" * 32,
            secret_epoch=1,
            now=1_301,
        )
    assert stale.value.code == "request_expired"
