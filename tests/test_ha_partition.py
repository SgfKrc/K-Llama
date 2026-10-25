from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from cluster_control_contract import VoterIdentity, VoterSet  # noqa: E402
from cluster_partition import (  # noqa: E402
    PartitionHarness,
    PartitionScenarioResult,
    write_partition_evidence,
)
from cluster_quorum import (  # noqa: E402
    QuorumCollector,
    QuorumPolicy,
    QuorumVoter,
    SQLiteVoterLedger,
)


def _harness(root: Path, *, seed: int) -> PartitionHarness:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root.mkdir(parents=True, exist_ok=True)
    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    identities = []
    for index, key in enumerate(keys):
        public_key = base64.urlsafe_b64encode(
            key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("ascii").rstrip("=")
        identities.append(VoterIdentity(voter_id=f"voter-{index}", public_key=public_key))
    voter_set = VoterSet(cluster_id="partition-test", voter_set_epoch=1, voters=tuple(identities))
    voters = {}
    for index, key in enumerate(keys):
        voter_id = f"voter-{index}"
        ledger = SQLiteVoterLedger(
            root / f"{voter_id}.sqlite3",
            voter_id=voter_id,
            cluster_id=voter_set.cluster_id,
            voter_set_epoch=voter_set.voter_set_epoch,
        )
        voters[voter_id] = QuorumVoter(
            voter_id=voter_id,
            private_key=key,
            voter_set=voter_set,
            ledger=ledger,
        )
    collector = QuorumCollector(
        voter_set=voter_set,
        voters=voters,
        # 租约必须明显长于一次 acquire 的实际耗时：operation_clock(now_ms) 返回的是
        # 「假起点 + 真实流逝」的时钟（见 qlh_core/control/clock.py），而 acquire 内部
        # 要走 3 个 voter 的 reserve/prepare/sign/commit 多次 SQLite 往返。原先的 100 ms
        # 会被自身耗时挤爆，证书签完即判过期（本机与 CI 都稳定失败）。
        policy=QuorumPolicy(lease_duration_ms=60_000),
    )
    return PartitionHarness(collector, voter_set=voter_set, seed=seed)


def _assert_safe(result: PartitionScenarioResult, tmp_path: Path) -> None:
    if not result.safe:
        evidence = write_partition_evidence(
            tmp_path / f"{result.scenario}-{result.seed}.json",
            result,
        )
        pytest.fail(f"partition scenario failed; evidence={evidence}; violations={result.violations}")


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_double_primary_race_never_has_two_writable_certificates(tmp_path: Path, parallel: bool):
    for seed in range(3):
        harness = _harness(tmp_path / f"double-{parallel}-{seed}", seed=seed)
        _assert_safe(harness.double_primary_race(parallel=parallel), tmp_path)


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_renewal_race_preserves_one_lease_identity(tmp_path: Path, parallel: bool):
    for seed in range(3):
        harness = _harness(tmp_path / f"renew-{parallel}-{seed}", seed=seed)
        initial = harness.collector.acquire(
            "voter-0",
            available_voter_ids=harness.voter_ids,
            now_ms=harness.clock.now_ms,
            lease_id=f"renew-{seed}",
        )
        assert initial.accepted and initial.certificate is not None
        _assert_safe(harness.renewal_race(initial.certificate, parallel=parallel), tmp_path)


def test_duplicate_term_recovery_merge_and_delayed_old_packet_are_fenced(tmp_path: Path):
    duplicate = _harness(tmp_path / "duplicate", seed=10).duplicate_term()
    _assert_safe(duplicate, tmp_path)

    recovery = _harness(tmp_path / "recovery", seed=11).recovery_merge()
    _assert_safe(recovery, tmp_path)
    assert {event.get("code") for event in recovery.events} & {
        "control_certificate_stale",
        "control_certificate_not_current",
    }

    delayed = _harness(tmp_path / "delayed", seed=12).delayed_old_packet()
    _assert_safe(delayed, tmp_path)
    assert "generation_stale" in {event.get("code") for event in delayed.events}


def test_failure_evidence_is_atomic_and_replayable(tmp_path: Path):
    result = PartitionScenarioResult(
        scenario="synthetic-failure",
        seed=99,
        events=[{"event": "barrier_released"}],
        violations=["synthetic_violation"],
    )
    path = write_partition_evidence(
        tmp_path / "evidence" / "partition.json",
        result,
        error=RuntimeError("replay me"),
    )
    assert path.is_file()
    assert path.read_text(encoding="utf-8").count("synthetic_violation") == 1
    assert not path.with_name(f".{path.name}.part").exists()
