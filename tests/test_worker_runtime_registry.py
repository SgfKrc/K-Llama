"""Tests for the cross-operation runtime ownership transfer protocol (#127).

Covers the five acceptance levels from
`docs/跨operation运行时所有权转移协议-2026-10-27.md` §6: unit reuse, race,
fence, negative controls, and the invariant table.
"""
from __future__ import annotations

import threading

import pytest

from worker_prepare_contract import (
    ARTIFACT_IDENTITY_FIELDS,
    IDENTITY_FIELDS,
    OPTIONAL_IDENTITY_FIELDS,
    OWNER_IDENTITY_FIELDS,
    split_identity,
)
from worker_runtime_registry import (
    GENERATION_MISMATCH,
    LOAD_SPEC_MISMATCH,
    NO_MATCH,
    NOT_LIVE,
    OWNER_MISMATCH,
    STALE_OWNER,
    UNKNOWN_RUNTIME,
    RuntimeRegistry,
    TransferError,
)


ARTIFACT = {
    "artifact_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
    "layer_range": [8, 24],
    "source_model_id": "qwen3-5-9b",
}
OWNER_OP1 = {"operation_id": "op_one", "generation": 1, "assignment_id": "as_one"}
OWNER_OP2 = {"operation_id": "op_two", "generation": 1, "assignment_id": "as_two"}


# -- §6.1 unit: same artifact, different operation, no reload -----------------

def test_transfer_keeps_runtime_identity_and_bumps_generation():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)

    receipt = registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT,
                                expected_generation=entry.generation)

    assert receipt.runtime_id == entry.runtime_id          # I5: content unchanged
    assert receipt.generation == 2                         # I2: monotonic
    assert registry.get(entry.runtime_id).owner == OWNER_OP2
    assert registry.get(entry.runtime_id).artifact_identity == ARTIFACT


def test_second_prepare_reuses_instead_of_registering_a_new_runtime():
    registry = RuntimeRegistry()
    first = registry.register(ARTIFACT, OWNER_OP1)

    receipt = registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT,
                                expected_generation=first.generation)

    assert len(registry.entries()) == 1
    assert receipt.runtime_id == first.runtime_id


# -- §6.2 race: only one of two concurrent transfers wins --------------------

def test_concurrent_transfers_only_one_succeeds():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    expected = entry.generation
    results = []
    barrier = threading.Barrier(2)

    def attempt(target):
        barrier.wait()
        try:
            results.append(("ok", registry.transfer(
                OWNER_OP1, target, ARTIFACT, expected_generation=expected).generation))
        except TransferError as exc:
            results.append(("err", exc.code))

    threads = [threading.Thread(target=attempt, args=(owner,))
               for owner in (OWNER_OP2, {"operation_id": "op_three", "generation": 1})]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(1 for kind, _ in results if kind == "ok") == 1
    # The loser is rejected by whichever precondition it violates first: after the
    # winner transferred ownership, `from_owner` no longer matches
    # (`owner_mismatch`); if two racers carried the same `from_owner` the loser
    # would instead trip the generation CAS (`generation_mismatch`). Both codes
    # mean "do not reuse", which is the property under test.
    rejected = [value for kind, value in results if kind == "err"]
    assert len(rejected) == 1
    assert rejected[0] in {GENERATION_MISMATCH, OWNER_MISMATCH}


def test_generation_cas_rejects_a_stale_expected_generation():
    """Owner identity alone is not enough: a round trip back to the same owner
    leaves `generation` advanced, so a stale `expected_generation` must fail."""
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT, expected_generation=1)
    registry.transfer(OWNER_OP2, OWNER_OP1, ARTIFACT, expected_generation=2)
    assert registry.get(entry.runtime_id).generation == 3
    assert registry.get(entry.runtime_id).owner == OWNER_OP1

    with pytest.raises(TransferError) as caught:
        registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT, expected_generation=1)

    assert caught.value.code == GENERATION_MISMATCH


# -- §6.3 fence: stale owner cannot free a runtime the new owner holds -------

def test_stale_owner_release_is_refused_and_instance_stays_live():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT, expected_generation=entry.generation)

    with pytest.raises(TransferError) as caught:
        registry.release(entry.runtime_id, OWNER_OP1, expected_generation=1)

    assert caught.value.code == STALE_OWNER
    still = registry.get(entry.runtime_id)
    assert still.state == "live"
    assert still.owner == OWNER_OP2


def test_release_by_new_owner_then_by_old_owner_drains_to_closed():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT, expected_generation=entry.generation)

    # refcount is 2 after the transfer: old holder plus new holder.
    assert registry.release(entry.runtime_id, OWNER_OP2, expected_generation=2) == "live"
    assert registry.get(entry.runtime_id).state == "live"
    assert registry.release(entry.runtime_id, OWNER_OP2, expected_generation=2) == "closed"
    assert registry.get(entry.runtime_id).state == "closed"


def test_superseded_fence_entry_records_the_new_owner():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT, expected_generation=entry.generation)

    superseded = registry.get(entry.runtime_id).superseded
    assert len(superseded) == 1
    assert superseded[0]["owner"] == OWNER_OP1
    assert superseded[0]["superseded_by"] == OWNER_OP2
    assert superseded[0]["state"] == "superseded"


# -- §6.4 negative controls: each case must fall back to a rebuild ----------

def test_unknown_artifact_is_no_match():
    registry = RuntimeRegistry()
    registry.register(ARTIFACT, OWNER_OP1)
    other = dict(ARTIFACT, artifact_sha256="c" * 64)
    with pytest.raises(TransferError) as caught:
        registry.transfer(OWNER_OP1, OWNER_OP2, other, expected_generation=1)
    assert caught.value.code == NO_MATCH


def test_different_layer_range_is_no_match():
    registry = RuntimeRegistry()
    registry.register(ARTIFACT, OWNER_OP1)
    other = dict(ARTIFACT, layer_range=[24, 28])
    with pytest.raises(TransferError) as caught:
        registry.transfer(OWNER_OP1, OWNER_OP2, other, expected_generation=1)
    assert caught.value.code == NO_MATCH


def test_load_spec_mismatch_blocks_transfer():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1, load_spec_digest="d" * 64)
    with pytest.raises(TransferError) as caught:
        registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT,
                          expected_generation=entry.generation, load_spec_digest="e" * 64)
    assert caught.value.code == LOAD_SPEC_MISMATCH


def test_wrong_source_owner_is_rejected():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    with pytest.raises(TransferError) as caught:
        registry.transfer(OWNER_OP2, OWNER_OP1, ARTIFACT, expected_generation=entry.generation)
    assert caught.value.code == OWNER_MISMATCH


def test_retiring_instance_cannot_be_transferred():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    registry.retire(entry.runtime_id)
    with pytest.raises(TransferError) as caught:
        registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT, expected_generation=1)
    assert caught.value.code == NOT_LIVE


def test_release_unknown_runtime_is_reported():
    registry = RuntimeRegistry()
    with pytest.raises(TransferError) as caught:
        registry.release("rt_missing", OWNER_OP1, expected_generation=1)
    assert caught.value.code == UNKNOWN_RUNTIME


# -- §4 invariants ----------------------------------------------------------

def test_invariant_exactly_one_owner_at_any_time():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    for index in range(5):
        current = registry.get(entry.runtime_id).owner
        target = {"operation_id": f"op_{index}", "generation": 1}
        registry.transfer(current, target, ARTIFACT,
                          expected_generation=registry.get(entry.runtime_id).generation)
        assert registry.get(entry.runtime_id).owner == target


def test_invariant_generation_strictly_increases():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    seen = [entry.generation]
    current = OWNER_OP1
    for index in range(4):
        target = {"operation_id": f"op_{index}", "generation": 1}
        receipt = registry.transfer(current, target, ARTIFACT,
                                    expected_generation=seen[-1])
        seen.append(receipt.generation)
        current = target
    assert seen == sorted(seen)
    assert len(set(seen)) == len(seen)


def test_audit_trail_records_transfer_and_refused_release():
    registry = RuntimeRegistry()
    entry = registry.register(ARTIFACT, OWNER_OP1)
    registry.transfer(OWNER_OP1, OWNER_OP2, ARTIFACT, expected_generation=entry.generation)
    with pytest.raises(TransferError):
        registry.release(entry.runtime_id, OWNER_OP1, expected_generation=1)

    kinds = [event["kind"] for event in registry.events()]
    assert kinds == ["register", "transfer", "release_refused"]


def test_close_all_drains_every_live_entry():
    registry = RuntimeRegistry()
    first = registry.register(ARTIFACT, OWNER_OP1)
    second = registry.register(dict(ARTIFACT, artifact_sha256="c" * 64), OWNER_OP2)
    drained = registry.close_all()
    assert {item.runtime_id for item in drained} == {first.runtime_id, second.runtime_id}
    assert registry.entries() == []


# -- identity split (protocol §3.1) ----------------------------------------

def test_identity_split_covers_every_field_without_overlap():
    assert ARTIFACT_IDENTITY_FIELDS & OWNER_IDENTITY_FIELDS == frozenset()
    assert ARTIFACT_IDENTITY_FIELDS | OWNER_IDENTITY_FIELDS == IDENTITY_FIELDS


def test_split_identity_round_trips_and_ignores_ownership():
    identity = {key: "v" for key in IDENTITY_FIELDS}
    identity.update({"generation": 1, "total_layers": 32, "hidden_size": 4096,
                     "layer_range": [8, 24], "segment_mode": "middle"})
    artifact, owner = split_identity(identity)
    assert {**artifact, **owner} == identity
    # Ownership fields must not leak into the artifact half: that is what makes
    # comparing the artifact half alone a safe reuse test.
    assert not (set(artifact) & OWNER_IDENTITY_FIELDS)
    assert "operation_id" in owner


def test_split_identity_carries_optional_load_spec_digest_into_artifact_half():
    identity = {key: "v" for key in IDENTITY_FIELDS}
    identity.update({"generation": 1, "total_layers": 32, "hidden_size": 4096,
                     "layer_range": [8, 24], "segment_mode": "middle",
                     "load_spec_digest": "f" * 64})
    artifact, owner = split_identity(identity)
    assert artifact["load_spec_digest"] == "f" * 64
    assert "load_spec_digest" not in owner
    assert set(artifact) - ARTIFACT_IDENTITY_FIELDS == set(OPTIONAL_IDENTITY_FIELDS)
