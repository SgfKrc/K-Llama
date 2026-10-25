from contextlib import contextmanager
import json
import sqlite3

import pytest

from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
from qlh_core.adapters.request_repository import SqliteRequestRepository
from qlh_core.contracts.deployment import ModelControlError, canonical
from qlh_core.contracts.execution import ExecutionError


pytestmark = pytest.mark.refactor_contract


@pytest.fixture(params=[("model_control", SqliteModelControlRepository, ModelControlError),
                        ("request_journal", SqliteRequestRepository, ExecutionError)])
def journal_case(tmp_path, request):
    prefix, adapter, error = request.param
    path = tmp_path / "control.sqlite3"
    def connect():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection
    @contextmanager
    def transaction():
        connection = connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
    return prefix, adapter, error, path, connect, transaction


def append(repository, count, *, payload=""):
    for index in range(count):
        repository.transact(lambda state: state.update(boot_id=str(index), probe=payload))


def test_legacy_compaction_preserves_exact_head_and_state(journal_case):
    prefix, adapter, _, _, connect, transaction = journal_case
    legacy = adapter(connect=connect, transaction=transaction, max_events=1000)
    append(legacy, 25)
    before = legacy.recovery_status()
    state = legacy.read()
    with connect() as connection:
        connection.execute(f"DROP TABLE {prefix}_events_checkpoint")
    bounded = adapter(connect=connect, transaction=transaction, max_events=4)
    assert bounded.prepare_recovery() == before
    assert bounded.recovery_status() == before
    assert bounded.read() == state
    with connect() as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {prefix}_events").fetchone()[0] == 2
        anchor = connection.execute(f"SELECT * FROM {prefix}_events_checkpoint").fetchone()
        assert anchor["sequence"] == 23
        assert json.loads(anchor["payload"])["sequence"] == 23
    bounded.transact(lambda state: state.update(boot_id="continued"))
    assert bounded.recovery_status()["durable_sequence"] == before["durable_sequence"] + 1


def test_checkpoint_alone_proves_state_and_continues_sequence(journal_case):
    prefix, adapter, _, _, connect, transaction = journal_case
    repository = adapter(connect=connect, transaction=transaction, max_events=2)
    append(repository, 5)
    status = repository.recovery_status()
    with connect() as connection:
        tail = connection.execute(f"SELECT * FROM {prefix}_events ORDER BY sequence DESC LIMIT 1").fetchone()
        connection.execute(f"UPDATE {prefix}_events_checkpoint SET sequence=?,domain_id=?,digest=?,payload=? WHERE id=1",
                           (tail["sequence"], tail["domain_id"], tail["digest"], tail["payload"]))
        connection.execute(f"DELETE FROM {prefix}_events")
    assert repository.recovery_status() == status
    repository.transact(lambda state: state.update(boot_id="next"))
    with connect() as connection:
        row = connection.execute(f"SELECT * FROM {prefix}_events").fetchone()
        assert row["sequence"] == status["durable_sequence"] + 1
        assert row["previous_digest"] == status["checkpoint_digest"]
    assert repository.recovery_status()["verified"]


@pytest.mark.parametrize("damage", ["anchor_payload", "anchor_digest", "tail_payload", "tail_gap",
                                    "snapshot", "missing_anchor", "missing_events", "missing_state", "missing_state_row"])
def test_corruption_after_checkpoint_still_fails_closed(journal_case, damage):
    prefix, adapter, error, _, connect, transaction = journal_case
    repository = adapter(connect=connect, transaction=transaction, max_events=4)
    append(repository, 6)
    assert repository.recovery_status()["verified"]
    with connect() as connection:
        if damage == "anchor_payload":
            connection.execute(f"UPDATE {prefix}_events_checkpoint SET payload='{{}}'")
        elif damage == "anchor_digest":
            connection.execute(f"UPDATE {prefix}_events_checkpoint SET digest=?", ("a" * 64,))
        elif damage == "tail_payload":
            connection.execute(f"UPDATE {prefix}_events SET payload='{{}}' WHERE sequence=(SELECT MAX(sequence) FROM {prefix}_events)")
        elif damage == "tail_gap":
            connection.execute(f"DELETE FROM {prefix}_events WHERE sequence=(SELECT MIN(sequence) FROM {prefix}_events)")
        elif damage == "snapshot":
            state = repository.read()
            state["boot_id"] = "tampered"
            connection.execute(f"UPDATE {prefix}_state SET payload=?", (canonical(state),))
        elif damage == "missing_anchor":
            connection.execute(f"DROP TABLE {prefix}_events_checkpoint")
        elif damage == "missing_events":
            connection.execute(f"DROP TABLE {prefix}_events")
        elif damage == "missing_state":
            connection.execute(f"DROP TABLE {prefix}_state")
        else:
            connection.execute(f"DELETE FROM {prefix}_state")
    with pytest.raises(error, match="journal_corrupt"):
        repository.recovery_status()
    with pytest.raises(error, match="journal_corrupt"):
        repository.prepare_recovery()


def test_compaction_and_append_rollback_together(journal_case):
    prefix, adapter, _, _, connect, transaction = journal_case
    repository = adapter(connect=connect, transaction=transaction, max_events=4)
    append(repository, 4)
    before = repository.recovery_status()
    @contextmanager
    def failing_transaction():
        with transaction() as connection:
            yield connection
            raise OSError("commit failed")
    repository._transaction = failing_transaction
    with pytest.raises(OSError, match="commit failed"):
        repository.transact(lambda state: state.update(boot_id="rollback"))
    assert repository.recovery_status() == before
    with connect() as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {prefix}_events_checkpoint").fetchone()[0] == 0


def test_byte_budget_and_vacuum_reclaim_without_changing_head(journal_case):
    prefix, adapter, _, path, connect, transaction = journal_case
    legacy = adapter(connect=connect, transaction=transaction, max_events=1000, max_event_bytes=100_000_000)
    append(legacy, 40, payload="x" * 100_000)
    before_size = path.stat().st_size
    before = legacy.recovery_status()
    bounded = adapter(connect=connect, transaction=transaction, max_events=64, max_event_bytes=350_000)
    bounded.prepare_recovery()
    with connect() as connection:
        rows, payload_bytes = connection.execute(f"SELECT COUNT(*),SUM(length(payload)) FROM {prefix}_events").fetchone()
        assert rows >= 1 and payload_bytes <= 350_000
        connection.execute("VACUUM")
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert path.stat().st_size < before_size // 3
    assert bounded.recovery_status() == before


def test_model_checkpoint_keeps_quorum_watermark_identity(journal_case):
    prefix, adapter, _, _, connect, transaction = journal_case
    if prefix != "model_control":
        pytest.skip("model owner watermark")
    from types import SimpleNamespace
    from cluster_auto_role_runtime import JournalPrefixAuthority
    from qlh_core.contracts.deployment import MODEL_CONTROL_DOMAIN
    repository = adapter(connect=connect, transaction=transaction, max_events=1000)
    append(repository, 12)
    authority = JournalPrefixAuthority(SimpleNamespace())
    authority.model_control = SimpleNamespace(repository=repository)
    held = authority.held(MODEL_CONTROL_DOMAIN)
    repository._journal.max_events = 4
    repository.prepare_recovery()
    assert authority.held(MODEL_CONTROL_DOMAIN) == held
    authority.enforce(SimpleNamespace(commit_watermark=(held,)))


@pytest.mark.parametrize("max_events", [4, 1000])
def test_legacy_snapshot_damage_is_rejected_before_mutation(journal_case, max_events):
    prefix, adapter, error, _, connect, transaction = journal_case
    repository = adapter(connect=connect, transaction=transaction, max_events=1000)
    append(repository, 6)
    with connect() as connection:
        state = repository.read()
        state["boot_id"] = "damaged"
        connection.execute(f"UPDATE {prefix}_state SET payload=?", (canonical(state),))
    repository._journal.max_events = max_events
    with pytest.raises(error, match="journal_corrupt"):
        repository.transact(lambda state: state.update(boot_id="would-mask-damage"))
    assert repository.read()["boot_id"] == "damaged"
    with connect() as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {prefix}_events").fetchone()[0] == 6
        assert connection.execute(f"SELECT COUNT(*) FROM {prefix}_events_checkpoint").fetchone()[0] == 0


def test_repeated_appends_remain_bounded_and_oversized_latest_is_preserved(journal_case):
    prefix, adapter, _, _, connect, transaction = journal_case
    repository = adapter(connect=connect, transaction=transaction, max_events=4, max_event_bytes=1000)
    append(repository, 40)
    with connect() as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {prefix}_events").fetchone()[0] <= 4
    repository.transact(lambda state: state.update(probe="x" * 2000))
    repository.transact(lambda state: state.update(boot_id="large-latest"))
    assert repository.read()["probe"] == "x" * 2000
    assert repository.recovery_status()["verified"]
    with connect() as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {prefix}_events").fetchone()[0] == 1
