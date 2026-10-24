"""Hot history bounds preserve admission fences, debt and transactional archives."""
from contextlib import contextmanager
from copy import deepcopy
import sqlite3

import pytest

from qlh_core.adapters.model_control_repository import SqliteModelControlRepository, empty_state
from qlh_core.adapters.model_control_retention import (
    HISTORY_KEEP, guard_retired_records, initialize_retention, retain_model_control,
)
from qlh_core.contracts.deployment import ModelControlError, canonical, command_key, digest
from qlh_core.control.deployment import DeploymentService


pytestmark = pytest.mark.refactor_contract
ZERO_KEEP = {kind: 0 for kind in HISTORY_KEEP}


@pytest.fixture
def connection():
    database = sqlite3.connect(":memory:")
    database.execute("BEGIN IMMEDIATE")
    initialize_retention(database)
    try:
        yield database
    finally:
        database.rollback()
        database.close()


@pytest.fixture
def control(tmp_path):
    path = tmp_path / "retention.sqlite3"

    def connect():
        database = sqlite3.connect(path)
        database.row_factory = sqlite3.Row
        return database

    @contextmanager
    def transaction():
        database = connect()
        try:
            database.execute("BEGIN IMMEDIATE")
            yield database
            database.commit()
        except BaseException:
            database.rollback()
            raise
        finally:
            database.close()

    repository = SqliteModelControlRepository(connect=connect, transaction=transaction)
    owner = DeploymentService(repository, boot_id="boot", clock=lambda: 1000)
    owner.start()
    return owner, repository, connect


def record(kind, record_id, *, status=None, created_at=0, **fields):
    identity_field = {"supplies": "operation_id", "operations": "operation_id",
                      "deployments": "deployment_id", "outbox": "command_id"}[kind]
    value = {identity_field: record_id, "created_at": created_at}
    if kind == "outbox":
        value.update(state=status or "acknowledged", node_id="worker", action="release", payload={})
    else:
        value["status"] = status or {"supplies": "available", "operations": "ready", "deployments": "retired"}[kind]
    if kind in {"supplies", "operations"}:
        value.update(intent={"model_id": "tiny"}, idempotency_key="")
    return {**value, **fields}


def history(count=100):
    state = empty_state()
    for kind in HISTORY_KEEP:
        for index in range(count):
            record_id = f"{kind}-{index:04}"
            state[kind][record_id] = record(kind, record_id, created_at=index)
    return state


def test_default_history_limits_choose_latest_and_archive_only_minimal_identity(connection):
    state = history()
    original = deepcopy(state)
    removed = retain_model_control(connection, state)
    for kind, limit in HISTORY_KEEP.items():
        assert len(state[kind]) == limit
        assert removed[kind] == 100 - limit
        assert set(state[kind]) == {f"{kind}-{index:04}" for index in range(100 - limit, 100)}
    rows = connection.execute("SELECT * FROM model_control_tombstones").fetchall()
    assert len(rows) == sum(removed.values())
    for kind, record_id, status, key, intent_digest in rows:
        old = original[kind][record_id]
        assert status == old.get("status", old.get("state"))
        assert key == old.get("idempotency_key", "")
        assert intent_digest == digest(old.get("intent"))
    assert state.keys() == original.keys()
    before_changes = connection.total_changes
    assert retain_model_control(connection, state) == ZERO_KEEP
    assert connection.total_changes == before_changes


@pytest.mark.parametrize("kind,statuses", [
    ("supplies", ["queued", "inspecting", "inventory", "waiting_nodes", "waiting_capacity", "planning",
                  "building", "supplying", "superseded", "unknown"]),
    ("operations", ["accepted", "queued", "preparing", "waiting_capacity", "unknown"]),
    ("deployments", ["candidate", "preparing", "prepared", "committing", "active", "invalidated",
                     "retiring", "aborting", "unknown"]),
    ("outbox", ["pending", "delivered", "failed", "unknown"]),
])
def test_nonterminal_and_unrecognized_statuses_never_expire(connection, kind, statuses):
    state = empty_state()
    for index, status in enumerate(statuses):
        record_id = f"old-{index}"
        state[kind][record_id] = record(kind, record_id, status=status, created_at=-1_000_000)
    state[kind]["terminal"] = record(kind, "terminal")
    expected = deepcopy(state[kind])
    del expected["terminal"]
    retain_model_control(connection, state, keep=ZERO_KEEP)
    assert state[kind] == expected


@pytest.mark.parametrize("root", ["selected", "active", "lifecycle", "cleanup", "artifact_cleanup"])
def test_roots_pin_supply_operation_deployment_and_both_command_encodings(connection, root):
    state = empty_state()
    state["supplies"]["operation"] = record("supplies", "operation")
    state["operations"]["operation"] = record("operations", "operation", config_id="deployment")
    state["deployments"]["deployment"] = record("deployments", "deployment", assignments={
        "worker": {"worker_prepare_identity": {"operation_id": "operation", "config_id": "deployment"}}})
    state["outbox"]["wire"] = record("outbox", "wire", payload={
        "cleanup_id": "cleanup", "identity": {"config_id": "deployment", "operation_id": "operation"}})
    state["outbox"]["master"] = record("outbox", "master", node_id="master", payload={
        "config_id": "cleanup", "candidate_config_id": "deployment", "cleanup_action": "release"})
    if root == "selected":
        state["selected_operation"] = "operation"
    elif root == "active":
        state["active_deployment"] = "deployment"
    elif root == "lifecycle":
        state["lifecycle"] = {"opaque_future_field": {"target": ["deployment"]}}
    else:
        state[root]["cleanup"] = {"opaque_future_field": {"target": ["deployment"]}}
    for kind in HISTORY_KEEP:
        state[kind]["unreferenced-" + kind] = record(kind, "unreferenced-" + kind)
    retain_model_control(connection, state, keep=ZERO_KEEP)
    assert set(state["supplies"]) == set(state["operations"]) == {"operation"}
    assert set(state["deployments"]) == {"deployment"}
    assert set(state["outbox"]) == {"wire", "master"}


def test_unfinished_related_record_and_pending_cleanup_pin_terminal_history(connection):
    state = empty_state()
    state["supplies"]["operation"] = record("supplies", "operation", status="building")
    state["operations"]["operation"] = record("operations", "operation", config_id="deployment")
    state["deployments"]["deployment"] = record("deployments", "deployment")
    state["outbox"]["pending"] = record("outbox", "pending", status="pending", payload={
        "cleanup_id": "missing-debt", "identity": {"config_id": "deployment"}})
    before = deepcopy(state)
    assert retain_model_control(connection, state, keep=ZERO_KEEP) == ZERO_KEEP
    assert state == before
    assert state["outbox"]["pending"]["state"] == "pending"


def test_newest_retained_record_keeps_older_references_hot(connection):
    state = empty_state()
    state["operations"]["new"] = record("operations", "new", created_at=100, config_id="old")
    state["deployments"]["old"] = record("deployments", "old", created_at=1)
    state["deployments"]["recent"] = record("deployments", "recent", created_at=100)
    retain_model_control(connection, state, keep={**ZERO_KEEP, "operations": 1})
    assert set(state["operations"]) == {"new"}
    assert set(state["deployments"]) == {"old"}


@pytest.mark.parametrize("kind", HISTORY_KEEP)
def test_unknown_fields_and_malformed_records_stay_pinned(connection, kind):
    state = empty_state()
    state[kind]["unknown"] = record(kind, "unknown", future_obligations={"opaque": True})
    state[kind]["malformed"] = record(kind, "malformed", created_at=None)
    state[kind]["missing-status"] = record(kind, "missing-status")
    del state[kind]["missing-status"]["state" if kind == "outbox" else "status"]
    state[kind]["wrong-identity"] = record(kind, "other")
    state[kind]["missing-record"] = None
    before = deepcopy(state)
    assert retain_model_control(connection, state, keep=ZERO_KEEP) == ZERO_KEEP
    assert state == before


def test_unknown_aggregate_fields_prevent_guessing_about_hidden_debts(connection):
    state = history(2)
    state["future_obligations"] = {"opaque": True}
    before = deepcopy(state)
    assert retain_model_control(connection, state, keep=ZERO_KEEP) == ZERO_KEEP
    assert state == before


@pytest.mark.parametrize("location,path", [
    ("/cache/publication", "/cache/publication/model.gguf"),
    ("C:\\Cache\\Publication", "c:/cache/publication/model.gguf"),
])
def test_artifact_location_references_pin_history_without_touching_files(connection, location, path):
    state = empty_state()
    state["supplies"]["referenced"] = record("supplies", "referenced", master_model_path=path)
    state["supplies"]["sibling"] = record("supplies", "sibling", master_model_path=location + "-other/model.gguf")
    state["artifact_cleanup"]["debt"] = {"location": location, "status": "pending", "evidence": {}}
    retain_model_control(connection, state, keep=ZERO_KEEP)
    assert set(state["supplies"]) == {"referenced"}
    assert state["artifact_cleanup"]["debt"]["status"] == "pending"


@pytest.mark.parametrize("kind,code", [
    ("supplies", "supply_terminal_conflict"), ("operations", "operation_terminal_conflict"),
    ("deployments", "deployment_retired"), ("outbox", "deployment_retired"),
])
@pytest.mark.parametrize("terminal_replay", [False, True])
def test_retired_ids_cannot_reappear_even_as_the_same_terminal_record(connection, kind, code, terminal_replay):
    state = empty_state()
    old = record(kind, "retired")
    state[kind]["retired"] = deepcopy(old)
    retain_model_control(connection, state, keep=ZERO_KEEP)
    before = deepcopy(state)
    if not terminal_replay:
        old["state" if kind == "outbox" else "status"] = {
            "supplies": "queued", "operations": "preparing", "deployments": "preparing", "outbox": "pending",
        }[kind]
    state[kind]["retired"] = old
    with pytest.raises(ModelControlError) as error:
        guard_retired_records(connection, before, state)
    assert error.value.code == code


@pytest.mark.parametrize("kind", ["supplies", "operations"])
@pytest.mark.parametrize("same_intent", [True, False])
def test_archived_idempotency_keys_preserve_reused_and_conflict_codes(connection, kind, same_intent):
    state = empty_state()
    state[kind]["retired"] = record(kind, "retired", idempotency_key="one-key", intent={"model_id": "tiny", "node_ids": ["worker"]})
    retain_model_control(connection, state, keep=ZERO_KEEP)
    before = deepcopy(state)
    intent = {"node_ids": ["worker"], "model_id": "tiny" if same_intent else "different"}
    state[kind]["fresh"] = record(kind, "fresh", status="queued", idempotency_key="one-key", intent=intent)
    with pytest.raises(ModelControlError) as error:
        guard_retired_records(connection, before, state)
    assert error.value.code == ("operation_idempotency_reused" if same_intent else "operation_idempotency_conflict")


def test_guard_checks_new_identity_and_key_on_existing_record(connection):
    state = empty_state()
    state["supplies"]["retired"] = record("supplies", "retired", idempotency_key="old-key")
    retain_model_control(connection, state, keep=ZERO_KEEP)
    state["supplies"]["live"] = record("supplies", "live", status="queued")
    before = deepcopy(state)
    state["supplies"]["live"]["operation_id"] = "retired"
    with pytest.raises(ModelControlError, match="^supply_terminal_conflict$"):
        guard_retired_records(connection, before, state)
    state = deepcopy(before)
    state["supplies"]["live"]["idempotency_key"] = "old-key"
    with pytest.raises(ModelControlError, match="^operation_idempotency_reused$"):
        guard_retired_records(connection, before, state)
    guard_retired_records(connection, before, before)


def test_retention_and_schema_changes_require_an_existing_transaction():
    database = sqlite3.connect(":memory:")
    try:
        for invoke in (lambda: initialize_retention(database),
                       lambda: guard_retired_records(database, empty_state(), empty_state()),
                       lambda: retain_model_control(database, empty_state())):
            with pytest.raises(ModelControlError, match="^model_control_retention_transaction_required$"):
                invoke()
        assert database.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        database.execute("BEGIN IMMEDIATE")
        initialize_retention(database)
        database.rollback()
        assert database.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
    finally:
        database.close()


def test_hot_payload_and_archive_are_smaller_than_repeated_full_history(connection):
    state = history(400)
    for index, supply in enumerate(state["supplies"].values()):
        supply["nodes"] = {"worker": {"receipt": "x" * 8192, "index": index}}
    before_size = len(canonical(state))
    retain_model_control(connection, state)
    after_size = len(canonical(state))
    archive_size = connection.execute("SELECT SUM(length(record_kind)+length(record_id)+length(status)"
                                      "+length(idempotency_key)+length(intent_digest)) FROM model_control_tombstones").fetchone()[0]
    assert after_size < before_size / 10
    assert after_size + archive_size < before_size / 5
    assert "tombstones" not in canonical(state)


def install_history(state, count=100):
    for kind, values in history(count).items():
        if kind in HISTORY_KEEP:
            state[kind].update(values)


def test_repository_archives_guard_late_owner_updates_and_survive_reopen(control):
    owner, repository, connect = control
    state = repository.transact(install_history)
    for kind, limit in HISTORY_KEEP.items():
        assert len(state[kind]) == limit
    assert repository.recovery_status()["verified"]
    reopened = SqliteModelControlRepository(connect=connect, transaction=repository._transaction)
    stale_updates = (
        (lambda: owner.save_supply(record("supplies", "supplies-0000", status="queued")), "supply_terminal_conflict"),
        (lambda: owner.save_operation(record("operations", "operations-0000", status="preparing")), "operation_terminal_conflict"),
        (lambda: owner.save_execution({"config_id": "deployments-0000", "phase": "preparing"},
                                     expected={}, acknowledgements={}, authority={}, connections={}), "deployment_retired"),
    )
    for update, code in stale_updates:
        with pytest.raises(ModelControlError) as error:
            update()
        assert error.value.code == code
        assert reopened.read() == state


@pytest.mark.parametrize("same_intent", [True, False])
def test_owner_supply_idempotency_remains_atomic_after_archival(control, same_intent):
    owner, repository, _ = control

    def seed(state):
        install_history(state)
        state["supplies"]["supplies-0000"]["idempotency_key"] = "archived-key"

    before = repository.transact(seed)
    value = record("supplies", "fresh", status="queued", idempotency_key="archived-key",
                   intent={"model_id": "tiny" if same_intent else "different"})
    with pytest.raises(ModelControlError) as error:
        owner.save_supply(value)
    assert error.value.code == ("operation_idempotency_reused" if same_intent else "operation_idempotency_conflict")
    assert repository.read() == before


def test_failed_commit_rolls_back_hot_state_and_tombstones_together(control, monkeypatch):
    _, repository, connect = control
    before = repository.read()
    recovery_before = repository.recovery_status()
    original = repository._transaction

    @contextmanager
    def failed_commit():
        with original() as database:
            yield database
            assert database.execute("SELECT COUNT(*) FROM model_control_tombstones").fetchone()[0] > 0
            raise OSError("injected commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(repository, "_transaction", failed_commit)
        with pytest.raises(OSError, match="injected commit failure"):
            repository.transact(install_history)
    assert repository.read() == before
    assert repository.recovery_status() == recovery_before
    with connect() as database:
        assert database.execute("SELECT COUNT(*) FROM model_control_tombstones").fetchone()[0] == 0
    after = repository.transact(install_history)
    assert all(len(after[kind]) == limit for kind, limit in HISTORY_KEEP.items())


@pytest.mark.parametrize("encoding", ["master", "wire"])
@pytest.mark.parametrize("evidence_kind", ["receipt", "process_ended"])
def test_settle_cleanup_matches_exact_master_or_wire_command(control, encoding, evidence_kind):
    owner, repository, _ = control
    identity = {"config_id": "deployment", "assignment_id": "assignment"}
    payload = {"config_id": "cleanup", "cleanup_action": "release", "candidate_config_id": "deployment"}
    if encoding == "wire":
        payload["worker_prepare_identity"] = identity
    debt = {"cleanup_id": "cleanup", "node_id": "master" if encoding == "master" else "worker",
            "candidate_config_id": "deployment", "action": "release", "payload": payload,
            "process_boot_id": "old-boot", "process_id": 123, "process_start_time": 456.0}
    command_payload = owner._wire(debt["node_id"], payload)
    command_id = command_key(debt["node_id"], command_payload)

    def seed(state):
        state["cleanup"]["cleanup"] = deepcopy(debt)
        state["lifecycle"] = {"config_id": "deployment", "cleanup_ledger": [deepcopy(debt)]}
        state["deployments"]["deployment"] = record("deployments", "deployment", status="retiring")
        state["outbox"][command_id] = record("outbox", command_id, status="pending", node_id=debt["node_id"],
                                             payload=deepcopy(command_payload))
        for record_id, change in (("other-node", {"node_id": "other"}),
                                  ("other-action", {"payload": {**command_payload, "action": "abort", "cleanup_action": "abort"}}),
                                  ("other-identity", {"payload": {**command_payload, "identity": {"config_id": "other"}}})):
            state["outbox"][record_id] = record("outbox", record_id, status="pending", node_id=debt["node_id"],
                                                payload=deepcopy(command_payload))
            state["outbox"][record_id].update(change)

    repository.transact(seed)
    if evidence_kind == "receipt":
        evidence = {"kind": "receipt", "identity": identity if encoding == "wire" else None, "action": "release"}
    else:
        evidence = {"kind": "process_ended", "boot_id": "old-boot", "process_id": 123, "process_start_time": 456.0}
    state = owner.settle_cleanup("cleanup", evidence=evidence)
    assert state["cleanup"] == {}
    assert state["lifecycle"]["cleanup_ledger"] == []
    assert state["deployments"]["deployment"]["status"] == "retired"
    assert state["outbox"][command_id]["state"] == "acknowledged"
    assert all(value["state"] == "pending" for key, value in state["outbox"].items() if key != command_id)
    assert owner.settle_cleanup("cleanup", evidence=evidence) == state


def test_missing_cleanup_debt_never_authorizes_guessing_old_pending_command_completed(control):
    owner, repository, _ = control

    def seed(state):
        state["outbox"]["orphan"] = record("outbox", "orphan", status="pending", payload={
            "cleanup_id": "missing", "action": "release", "identity": {"config_id": "old"}})

    before = repository.transact(seed)
    assert owner.settle_cleanup("missing", evidence={"kind": "receipt", "action": "release", "identity": {"config_id": "old"}}) == before
    assert repository.transact(lambda state: None)["outbox"]["orphan"]["state"] == "pending"
