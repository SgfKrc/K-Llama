"""#93：把「已提交前缀」（commit watermark）接入 quorum 证书。

覆盖三件事：
1. voter 端在签名**前**必须确认自己**耐久持有**被提议的前缀（否则拒签，fail-closed）；
2. collector 把「多数不持有」与「多数不可用」区分开（前者不能被误当作容量/网络问题）；
3. 空水位的证书与引入本字段之前逐字节一致（旧证书/旧签名继续有效）。
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, "src")

from cluster_control_contract import (  # noqa: E402
    CommitWatermark,
    QuorumCertificate,
    VoterIdentity,
    VoterSet,
)
from cluster_quorum import (  # noqa: E402
    QuorumCollector,
    QuorumVoter,
    SQLiteVoterLedger,
)

WF = "wf-watermark"
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _watermark(sequence: int = 10, digest: str = DIGEST) -> CommitWatermark:
    return CommitWatermark(workflow_id=WF, last_sequence=sequence, checkpoint_digest=digest)


def _fixture(tmp_path: Path, *, count: int = 3, watermarks: dict | None = None):
    """同 `test_cluster_quorum._fixture`，但可选地给每个 voter 接上耐久水位来源。

    ``watermarks=None`` ⇒ 所有 voter **没有**耐久日志来源（模拟未接线的场景）；
    传入 ``{"voter-0": (sequence, digest), ...}`` ⇒ 该 voter 耐久持有到该点。
    """
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
    voter_set = VoterSet(cluster_id="cluster-watermark", voter_set_epoch=1, voters=identities)

    def _held_for(voter_id: str):
        if watermarks is None:
            return None
        entry = watermarks.get(voter_id)
        if entry is None:
            return None
        return _watermark(*entry)

    voters = {}
    for voter_id, private in pairs:
        voters[voter_id] = QuorumVoter(
            voter_id=voter_id,
            private_key=private,
            voter_set=voter_set,
            ledger=SQLiteVoterLedger(
                tmp_path / f"{voter_id}.sqlite3",
                voter_id=voter_id,
                cluster_id=voter_set.cluster_id,
                voter_set_epoch=voter_set.voter_set_epoch,
            ),
            watermark_source=(
                None if watermarks is None
                else (lambda workflow_id, vid=voter_id: _held_for(vid))
            ),
        )
    return voter_set, voters


def test_empty_watermark_needs_no_durable_source(tmp_path):
    """不提议任何前缀时，没有耐久日志的 voter 也必须照常成证（向后兼容）。"""
    voter_set, voters = _fixture(tmp_path)
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire("voter-0", now_ms=1_000)
    assert result.accepted is True
    assert result.certificate is not None
    assert result.certificate.commit_watermark == ()


def test_proposed_watermark_without_durable_source_is_rejected(tmp_path):
    voter_set, voters = _fixture(tmp_path)
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", now_ms=1_000, commit_watermark=(_watermark(),),
    )
    assert result.accepted is False
    assert result.reason == "quorum_watermark_unavailable"
    assert result.signed_voters == ()


def test_majority_durable_holders_issue_watermarked_certificate(tmp_path):
    """持有者过半即可成证；落后的少数拒签但不阻塞（= 多数一致前缀）。"""
    voter_set, voters = _fixture(tmp_path, watermarks={
        "voter-0": (10, DIGEST), "voter-1": (10, DIGEST), "voter-2": (4, DIGEST),
    })
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", now_ms=1_000, lease_id="lease-wm", commit_watermark=(_watermark(),),
    )
    assert result.accepted is True
    assert result.certificate is not None
    assert result.certificate.commit_watermark == (_watermark(),)
    assert "voter-2" not in result.signed_voters
    assert {"voter-0", "voter-1"} <= set(result.signed_voters)


def test_behind_majority_cannot_attest(tmp_path):
    """多数还没耐久追上前缀 ⇒ 不得成证，且 reason 必须点明是水位问题。"""
    voter_set, voters = _fixture(tmp_path, watermarks={
        "voter-0": (3, DIGEST), "voter-1": (4, DIGEST), "voter-2": (10, DIGEST),
    })
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", now_ms=1_000, commit_watermark=(_watermark(),),
    )
    assert result.accepted is False
    assert result.reason == "quorum_watermark_unheld"


def test_forked_digest_cannot_attest(tmp_path):
    """同一 sequence 但 digest 不同 = 分叉 ⇒ 只有真正持有该点的一方算数。"""
    voter_set, voters = _fixture(tmp_path, watermarks={
        "voter-0": (10, OTHER_DIGEST),
        "voter-1": (10, OTHER_DIGEST),
        "voter-2": (10, DIGEST),
    })
    result = QuorumCollector(voter_set=voter_set, voters=voters).acquire(
        "voter-0", now_ms=1_000, commit_watermark=(_watermark(),),
    )
    assert result.accepted is False
    assert result.reason == "quorum_watermark_unheld"


def test_renew_preserves_commit_watermark(tmp_path):
    """续期不得丢掉提交证明。"""
    voter_set, voters = _fixture(tmp_path, watermarks={
        "voter-0": (10, DIGEST), "voter-1": (10, DIGEST), "voter-2": (10, DIGEST),
    })
    collector = QuorumCollector(voter_set=voter_set, voters=voters)
    first = collector.acquire(
        "voter-0", now_ms=1_000, lease_id="lease-wm", commit_watermark=(_watermark(),),
    )
    assert first.accepted is True and first.certificate is not None
    renewed = collector.renew(first.certificate, now_ms=first.certificate.expires_at_ms - 1)
    assert renewed.accepted is True
    assert renewed.certificate is not None
    assert renewed.certificate.commit_watermark == (_watermark(),)


def test_empty_watermark_keeps_legacy_signing_payload():
    """回归：空水位不得改变被签名的 payload（旧证书/旧签名必须继续有效）。"""
    base = dict(
        cluster_id="c1", voter_set_epoch=1, term=1, leader_id="l1",
        lease_id="le1", expires_at_ms=10**12, signatures=(),
    )
    assert QuorumCertificate(**base).signing_payload() == {
        "schema_version": "qlh.cluster.control.v1",
        "document_type": "quorum_certificate",
        "cluster_id": "c1",
        "voter_set_epoch": 1,
        "term": 1,
        "leader_id": "l1",
        "lease_id": "le1",
        "expires_at_ms": 10**12,
    }
    watermarked = QuorumCertificate(**base, commit_watermark=(_watermark(),))
    assert watermarked.signing_payload()["commit_watermark"] == [_watermark().to_dict()]
    assert QuorumCertificate.from_dict(watermarked.to_dict()).commit_watermark == (_watermark(),)
    legacy = watermarked.to_dict()
    legacy.pop("commit_watermark")
    assert QuorumCertificate.from_dict(legacy).commit_watermark == ()


def test_duplicate_workflow_watermark_is_rejected():
    """同一 workflow 出现两条水位 ⇒ 契约拒绝（歧义即分叉）。"""
    import pytest

    from cluster_control_contract import ControlContractError

    with pytest.raises(ControlContractError) as excinfo:
        QuorumCertificate(
            cluster_id="c1", voter_set_epoch=1, term=1, leader_id="l1",
            lease_id="le1", expires_at_ms=10**12, signatures=(),
            commit_watermark=(_watermark(1, DIGEST), _watermark(2, OTHER_DIGEST)),
        )
    assert excinfo.value.code == "control_certificate_watermark_invalid"
