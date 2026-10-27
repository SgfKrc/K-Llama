"""PERF-CONTROL-READ-01: committed, bounded model-control read projections."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from copy import deepcopy
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from tests import test_refactor_deployment_owner as owner_cases
from qlh_core.adapters import model_control_repository as repository_module
from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
from qlh_core.contracts.deployment import ModelControlError, canonical
from qlh_core.control.deployment import DeploymentService


pytestmark = pytest.mark.refactor_contract
READ_MODES = ("legacy", "execution", "query", "combined")


@pytest.fixture(params=READ_MODES)
def read_mode(request):
    return request.param


@pytest.fixture
def control_factory(tmp_path):
    def create(mode, *, start=True):
        path = tmp_path / "read-views.sqlite3"

        def connect():
            connection = sqlite3.connect(path, timeout=10)
            connection.row_factory = sqlite3.Row
            return connection

        @contextmanager
        def transaction():
            with closing(connect()) as connection:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    yield connection
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

        with closing(connect()) as connection:
            assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        repository = SqliteModelControlRepository(
            connect=connect, transaction=transaction, read_mode=mode)
        owner = DeploymentService(repository, boot_id="boot-1", clock=lambda: 1.0)
        if start:
            owner.start()
        return SimpleNamespace(path=path, connect=connect, transaction=transaction,
                               repository=repository, owner=owner, mode=mode)

    return create


@pytest.fixture
def read_control(control_factory, read_mode):
    return control_factory(read_mode)


def _second_repository(control):
    return SqliteModelControlRepository(
        connect=control.connect, transaction=control.transaction, read_mode=control.mode)


def _publish(control, *, input_revisions=None):
    for phase in ("preparing", "committing", "ready"):
        lifecycle, expected = owner_cases.candidate(phase)
        lifecycle.update(total_layers=8, input_revisions=deepcopy(input_revisions or {}))
        receipts = {"worker": {"phase": "prepare" if phase == "preparing" else "commit",
                               "loaded": True, "status": "ready",
                               "identity": expected["worker"]["worker_prepare_identity"]}}
        control.owner.save_execution(lifecycle, expected=expected, acknowledgements=receipts,
                                     authority={"term": 1}, connections={"worker": 1})


def _raw_update(control, mutation):
    with closing(control.connect()) as connection, connection:
        state = json.loads(connection.execute(
            "SELECT payload FROM model_control_state WHERE id=1").fetchone()[0])
        mutation(state)
        connection.execute("UPDATE model_control_state SET payload=? WHERE id=1", (canonical(state),))
    return state


def _invalidate(state):
    state["active_deployment"] = ""
    state["deployments"]["deployment-1"].update(status="invalidated", validity="unknown")


def _views(control):
    return (control.owner.execution_snapshot(), control.owner.query_snapshot(), control.owner.query_stamp())


def _runnable(owner, **changes):
    arguments = {"authority": {"term": 1}, "connections": {"worker": 1}}
    arguments.update(changes)
    return owner.runnable("deployment-1", **arguments)


def _count_reads(monkeypatch, repository):
    original = repository._read
    revisions = []

    def counted(connection):
        state = original(connection)
        revisions.append(state["revision"])
        return state

    monkeypatch.setattr(repository, "_read", counted)
    return revisions


def _seed_query(control, *, history=False):
    marker = "historic-payload:" + "x" * (160 * 1024)

    def seed(state):
        state["selected_operation"] = "live"
        state["operations"]["live"] = {
            "operation_id": "live", "status": "preparing", "intent": {"model_id": "tiny"}}
        state["supplies"]["live"] = {
            "operation_id": "live", "status": "available", "intent": {"model_id": "tiny"},
            "nodes": {"worker": {"layer_range": [4, 8]}}, "plan": {"total_layers": 8},
            "master_model_path": marker if history else "private/model", "created_at": 1.0}
        state["lifecycle"]["local_assignment"] = {"start_layer": 0, "end_layer": 4}
        if history:
            state["operations"]["old"] = {
                "operation_id": "old", "status": "failed", "reason": marker}
            state["supplies"]["old"] = {
                "operation_id": "old", "status": "failed", "reason": marker}
            state["deployments"]["old"] = {
                "deployment_id": "old", "status": "retired", "reason_code": marker}
            state["outbox"]["old"] = {
                "command_id": "old", "state": "superseded", "payload": {"data": marker}}
            state["cleanup"]["pending"] = {"payload": marker}
            state["artifact_cleanup"]["pending"] = {"evidence": marker}
            state["lifecycle"]["cleanup_ledger"] = [{"payload": marker}]

    control.repository.transact(seed)
    return marker


def test_modes_select_the_requested_read_paths(read_control, read_mode):
    repository = read_control.repository
    assert repository.execution_reads_enabled is (read_mode in {"execution", "combined"})
    assert repository.query_reads_enabled is (read_mode in {"query", "combined"})


@pytest.mark.parametrize("mode", ["", "both", "COMBINED", False])
def test_invalid_read_mode_is_rejected(control_factory, mode):
    with pytest.raises(ValueError, match="model_control_read_mode_invalid"):
        control_factory(mode)


def test_projections_bound_historic_payload_and_allow_explicit_operation_lookup(read_control):
    _publish(read_control)
    marker = _seed_query(read_control, history=True)
    full = read_control.repository.read()
    assert len(canonical(full)) > 1024 * 1024
    execution, query, stamp = _views(read_control)
    for view, enabled in ((execution, read_control.repository.execution_reads_enabled),
                          (query, read_control.repository.query_reads_enabled)):
        if enabled:
            assert len(canonical(view)) < 8192
            assert marker not in canonical(view)
            assert set(view["deployments"]) == {"deployment-1"}
            assert view["deployments"]["deployment-1"] == full["deployments"]["deployment-1"]
            assert view["lifecycle"] == {
                "total_layers": 8, "local_assignment": {"start_layer": 0, "end_layer": 4}}
            assert not {"outbox", "cleanup", "artifact_cleanup"}.intersection(view)
        else:
            assert view == full
    if read_control.repository.execution_reads_enabled:
        assert not {"operations", "supplies"}.intersection(execution)
    if read_control.repository.query_reads_enabled:
        assert set(query["operations"]) == set(query["supplies"]) == {"live"}
        assert query["operations"]["live"] == full["operations"]["live"]
        assert "master_model_path" not in query["supplies"]["live"]
        assert query["supplies"]["live"]["nodes"] == {"worker": {"layer_range": [4, 8]}}
        assert stamp == {"revision": full["revision"], "boot_id": "boot-1"}
        old = read_control.owner.query_snapshot("old")
        assert set(old["operations"]) == set(old["supplies"]) == {"old"}
        assert old["operations"]["old"]["reason"] == marker
        assert old["active_deployment"] == "deployment-1"
        missing = read_control.owner.query_snapshot("missing")
        assert missing["operations"] == missing["supplies"] == {}
    else:
        assert stamp == full


def test_hot_reads_decode_once_per_committed_generation(read_control, monkeypatch):
    _publish(read_control)
    _seed_query(read_control)
    repository = read_control.repository
    reads = _count_reads(monkeypatch, repository)
    optimized = (repository.execution_reads_enabled, repository.query_reads_enabled,
                 repository.query_reads_enabled)
    expected = int(any(optimized)) + 5 * optimized.count(False)
    revision = None
    for _ in range(5):
        execution, query, stamp = _views(read_control)
        revision = execution["revision"]
        assert query["revision"] == stamp["revision"] == revision
    assert reads == [revision] * expected
    committed = _second_repository(read_control).transact(
        lambda state: state["operations"]["live"].update(reason="next generation"))
    reads.clear()
    for _ in range(5):
        assert all(view["revision"] == committed["revision"] for view in _views(read_control))
    assert reads == [revision + 1] * expected


def test_returned_nested_mutations_cannot_change_cached_views(read_control):
    _publish(read_control)
    _seed_query(read_control)
    expected = _views(read_control)
    for view in _views(read_control):
        view["boot_id"] = "mutated"
        if "deployments" in view:
            view["deployments"]["deployment-1"]["authority"]["term"] = 999
            view["deployments"]["deployment-1"]["assignments"]["worker"][
                "worker_prepare_identity"]["assignment_id"] = "mutated"
            view["lifecycle"]["local_assignment"]["start_layer"] = 99
        if "operations" in view:
            view["operations"]["live"]["intent"]["model_id"] = "mutated"
            view["supplies"]["live"]["nodes"]["worker"]["layer_range"].append(99)
    assert _views(read_control) == expected
    assert _runnable(read_control.owner)


@pytest.mark.parametrize("writer", ["connection", "repository", "process"])
def test_external_writes_invalidate_warm_views(read_control, writer):
    _publish(read_control)
    before = _views(read_control)[0]
    assert _runnable(read_control.owner)
    with closing(read_control.connect()) as connection:
        generation = connection.execute(
            "SELECT generation FROM model_control_read_revision WHERE id=1").fetchone()[0]
    if writer == "repository":
        changed = _second_repository(read_control).transact(_invalidate)
        assert changed["revision"] == before["revision"] + 1
    elif writer == "connection":
        assert _raw_update(read_control, _invalidate)["revision"] == before["revision"]
    else:
        program = (
            "import json, sqlite3, sys\n"
            "with sqlite3.connect(sys.argv[1]) as connection:\n"
            "    state = json.loads(connection.execute('SELECT payload FROM model_control_state WHERE id=1').fetchone()[0])\n"
            "    state['active_deployment'] = ''\n"
            "    state['deployments']['deployment-1']['status'] = 'invalidated'\n"
            "    connection.execute('UPDATE model_control_state SET payload=? WHERE id=1', (json.dumps(state),))\n"
        )
        subprocess.run([sys.executable, "-c", program, str(read_control.path)],
                       check=True, capture_output=True, text=True, timeout=15)
    with closing(read_control.connect()) as connection:
        assert connection.execute(
            "SELECT generation FROM model_control_read_revision WHERE id=1").fetchone()[0] > generation
    for view in _views(read_control)[:2]:
        assert view["active_deployment"] == ""
    assert not _runnable(read_control.owner)
    if writer != "repository":
        assert read_control.owner.query_stamp()["revision"] == before["revision"]


@pytest.mark.parametrize("damage", ["delete_state", "drop_state", "drop_trigger", "replace_trigger"])
def test_schema_damage_cannot_reuse_cached_readiness(read_control, damage, monkeypatch):
    _publish(read_control)
    _views(read_control)
    assert _runnable(read_control.owner)
    reads = _count_reads(monkeypatch, read_control.repository)
    with closing(read_control.connect()) as connection, connection:
        if damage == "delete_state":
            connection.execute("DELETE FROM model_control_state WHERE id=1")
        elif damage == "drop_state":
            connection.execute("DROP TABLE model_control_state")
        else:
            connection.execute("DROP TRIGGER model_control_read_update")
            if damage == "replace_trigger":
                connection.execute(
                    "CREATE TRIGGER model_control_read_update AFTER UPDATE ON model_control_state "
                    "BEGIN UPDATE model_control_read_revision SET generation=generation WHERE id=1; END")
    if damage in {"drop_trigger", "replace_trigger"}:
        _raw_update(read_control, _invalidate)
    for _ in range(3):
        assert all(view["active_deployment"] == "" for view in _views(read_control)[:2])
        assert not _runnable(read_control.owner)
    if damage in {"drop_trigger", "replace_trigger"}:
        assert len(reads) >= 3


@pytest.mark.parametrize("damage", ["delete_version", "drop_version"])
def test_missing_version_falls_back_until_transaction_repairs_it(read_control, damage, monkeypatch):
    _publish(read_control)
    before = _views(read_control)
    with closing(read_control.connect()) as connection, connection:
        if damage == "delete_version":
            connection.execute("DELETE FROM model_control_read_revision WHERE id=1")
        else:
            connection.execute("DROP TABLE model_control_read_revision")
    reads = _count_reads(monkeypatch, read_control.repository)
    for _ in range(3):
        assert _views(read_control) == before
    assert len(reads) == 9
    repaired = _second_repository(read_control).transact(_invalidate)
    reads.clear()
    assert all(view["revision"] == repaired["revision"] for view in _views(read_control))
    assert not _runnable(read_control.owner)
    with closing(read_control.connect()) as connection:
        assert connection.execute(
            "SELECT incarnation,generation FROM model_control_read_revision WHERE id=1").fetchone()
        assert len(connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='model_control_state'").fetchall()) == 3


@pytest.mark.parametrize("payload_schema", [None, 2])
def test_invalid_state_schema_cannot_return_cached_ready(read_control, payload_schema):
    _publish(read_control)
    _views(read_control)

    def corrupt(state):
        if payload_schema is None:
            state.pop("schema_version")
        else:
            state["schema_version"] = payload_schema

    _raw_update(read_control, corrupt)
    for read in (read_control.owner.execution_snapshot, read_control.owner.query_snapshot,
                 read_control.owner.query_stamp):
        with pytest.raises(ModelControlError, match="model_control_schema_invalid"):
            read()
    with pytest.raises(ModelControlError, match="model_control_schema_invalid"):
        _runnable(read_control.owner)


@pytest.mark.parametrize("outcome", ["rollback", "commit_failure", "commit"])
def test_views_publish_only_after_a_successful_commit(read_control, monkeypatch, outcome):
    _publish(read_control)
    before = _views(read_control)
    journal = read_control.repository.recovery_status()
    attempted_commits = []

    def deny_commit(action, argument, _other, _database, _source):
        if action == sqlite3.SQLITE_TRANSACTION and argument == "COMMIT":
            attempted_commits.append(argument)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @contextmanager
    def transaction():
        with read_control.transaction() as connection:
            yield connection
            pending = json.loads(connection.execute(
                "SELECT payload FROM model_control_state WHERE id=1").fetchone()[0])
            assert pending["revision"] == before[0]["revision"] + 1
            assert pending["active_deployment"] == ""
            assert _views(read_control) == before
            assert _runnable(read_control.owner)
            if outcome == "rollback":
                raise OSError("injected rollback before commit")
            if outcome == "commit_failure":
                connection.set_authorizer(deny_commit)

    monkeypatch.setattr(read_control.repository, "_transaction", transaction)
    if outcome == "commit":
        read_control.repository.transact(_invalidate)
        assert all(view["revision"] == before[0]["revision"] + 1 for view in _views(read_control))
        assert not _runnable(read_control.owner)
    else:
        error = OSError if outcome == "rollback" else sqlite3.DatabaseError
        with pytest.raises(error, match="injected rollback|not authorized"):
            read_control.repository.transact(_invalidate)
        assert _views(read_control) == before
        assert _runnable(read_control.owner)
        assert read_control.repository.recovery_status() == journal
    assert attempted_commits == (["COMMIT"] if outcome == "commit_failure" else [])


def _schema(control):
    with closing(control.connect()) as connection:
        schema = connection.execute("PRAGMA schema_version").fetchone()[0]
        rows = [tuple(row) for row in connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
        return schema, rows


@pytest.mark.parametrize("state", ["empty", "ready", "missing_version", "missing_trigger"])
def test_reads_never_write_or_repair_schema(control_factory, read_mode, monkeypatch, state):
    control = control_factory(read_mode, start=state != "empty")
    if state != "empty":
        _publish(control)
        _views(control)
        with closing(control.connect()) as connection, connection:
            if state == "missing_version":
                connection.execute("DROP TABLE model_control_read_revision")
            elif state == "missing_trigger":
                connection.execute("DROP TRIGGER model_control_read_update")
    before = _schema(control)
    forbidden = []
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
               sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(action, argument, other, database, source):
        if action in allowed or (action == sqlite3.SQLITE_PRAGMA and other is None):
            return sqlite3.SQLITE_OK
        forbidden.append((action, argument, other, database, source))
        return sqlite3.SQLITE_DENY

    def read_only_connect():
        connection = control.connect()
        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(control.repository, "_connect", read_only_connect)
    for _ in range(3):
        full = control.owner.snapshot()
        execution, query, stamp = _views(control)
        assert all(view["revision"] == full["revision"] for view in (execution, query, stamp))
        assert _runnable(control.owner) is (state != "empty")
    assert forbidden == []
    assert _schema(control) == before


def test_restart_invalidates_warm_views_and_old_owner_writes(read_control):
    _publish(read_control)
    _seed_query(read_control)
    before = _views(read_control)
    assert _runnable(read_control.owner)
    restarted = DeploymentService(_second_repository(read_control), boot_id="boot-2", clock=lambda: 2.0)
    recovered = restarted.start()
    assert recovered["revision"] > before[0]["revision"]
    for view in _views(read_control):
        assert view["boot_id"] == "boot-2"
    assert read_control.owner.execution_snapshot()["active_deployment"] == ""
    assert read_control.owner.query_snapshot()["operations"]["live"]["status"] == "failed"
    assert not _runnable(read_control.owner)
    assert not _runnable(restarted)
    with pytest.raises(ModelControlError, match="deployment_boot_superseded"):
        owner_cases.save(read_control.owner, "preparing")


def test_volatile_operation_fence_does_not_mutate_repository_cache(read_control, monkeypatch):
    _publish(read_control)
    _seed_query(read_control)
    before = read_control.repository.read_query()
    original = read_control.repository._transaction

    @contextmanager
    def fail_commit():
        with original() as connection:
            yield connection
            raise OSError("injected operation commit failure")

    with monkeypatch.context() as failure:
        failure.setattr(read_control.repository, "_transaction", fail_commit)
        with pytest.raises(OSError, match="injected operation commit failure"):
            read_control.owner.save_operation({
                "operation_id": "live", "status": "cancelled", "reason_code": "disk_full"})
    read_control.owner.fence_operation_failure({
        "operation_id": "live", "status": "cancelled", "reason_code": "disk_full"})
    fenced = read_control.owner.query_snapshot()
    assert fenced["revision"] == before["revision"]
    assert fenced["operations"]["live"]["durable"] is False
    assert fenced["operations"]["live"]["commit_scope"] == "volatile_failure"
    assert read_control.repository.read_query() == before
    second_owner = DeploymentService(read_control.repository, boot_id="boot-1", clock=lambda: 1.0)
    assert second_owner.query_snapshot()["operations"]["live"]["status"] == "preparing"
    read_control.owner.save_operation({"operation_id": "live", "status": "preparing",
                                       "reason_code": "retry_committed"})
    retried = read_control.owner.query_snapshot()["operations"]["live"]
    assert retried["status"] == "preparing"
    assert retried["reason_code"] == "retry_committed"
    assert retried.get("commit_scope") != "volatile_failure"
    assert read_control.owner.query_snapshot() == read_control.repository.read_query()
    restarted = DeploymentService(_second_repository(read_control), boot_id="boot-2", clock=lambda: 2.0)
    restarted.start()
    assert read_control.owner.query_snapshot()["operations"]["live"]["status"] == "failed"


@pytest.mark.parametrize("changed", ["authority", "connections", "input_revisions"])
def test_runnable_checks_live_arguments_on_every_cached_read(read_control, changed, monkeypatch):
    revisions = {"catalog": 3, "workers": 7}
    _publish(read_control, input_revisions=revisions)
    reads = _count_reads(monkeypatch, read_control.repository)
    arguments = {"input_revisions": revisions}
    assert _runnable(read_control.owner, **arguments)
    replacement = {"authority": {"term": 2}, "connections": {"worker": 2},
                   "input_revisions": {"catalog": 4, "workers": 7}}[changed]
    for _ in range(4):
        assert not _runnable(read_control.owner, **{**arguments, changed: replacement})
        assert _runnable(read_control.owner, **arguments)
    assert len(reads) == (1 if read_control.repository.execution_reads_enabled else 9)
    _raw_update(read_control, lambda state: state["deployments"]["deployment-1"].update(
        {changed: replacement}))
    assert not _runnable(read_control.owner, **arguments)
    assert _runnable(read_control.owner, **{**arguments, changed: replacement})


def _bind_revision(state):
    revision = state["revision"] + 1
    state["deployments"]["deployment-1"]["authority"] = {"term": revision}
    state["lifecycle"]["total_layers"] = revision + 8
    state["operations"]["live"]["intent"] = {"model_id": f"model-{revision}"}


def _assert_bound(view):
    assert view["boot_id"] == "boot-1"
    if "deployments" in view:
        assert view["deployments"]["deployment-1"]["authority"]["term"] == view["revision"]
        assert view["lifecycle"]["total_layers"] == view["revision"] + 8
    if "operations" in view:
        assert view["operations"]["live"]["intent"]["model_id"] == f"model-{view['revision']}"


def test_concurrent_readers_keep_payload_bound_to_its_revision(read_control):
    _publish(read_control)
    _seed_query(read_control)
    initial = read_control.repository.transact(_bind_revision)["revision"]
    writer = _second_repository(read_control)
    barrier = threading.Barrier(5, timeout=10)
    rounds = 12

    def read():
        observed = []
        try:
            for _ in range(rounds):
                barrier.wait()
                for view in _views(read_control):
                    _assert_bound(view)
                    observed.append(view["revision"])
                barrier.wait()
            return observed
        except BaseException:
            barrier.abort()
            raise

    def write():
        try:
            for _ in range(rounds):
                barrier.wait()
                writer.transact(_bind_revision)
                barrier.wait()
        except BaseException:
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=5) as pool:
        readers = [pool.submit(read) for _ in range(4)]
        writing = pool.submit(write)
        for future in readers:
            observed = future.result(timeout=30)
            assert observed == sorted(observed)
            assert all(initial <= revision <= initial + rounds for revision in observed)
        writing.result(timeout=30)
    for view in _views(read_control):
        _assert_bound(view)
        assert view["revision"] == initial + rounds


@pytest.mark.parametrize("read_mode", ["execution", "query", "combined"], indirect=True)
@pytest.mark.parametrize("warm", [False, True])
def test_generation_and_payload_share_one_sqlite_snapshot(read_control, monkeypatch, warm):
    _publish(read_control)
    _seed_query(read_control)
    initial = read_control.repository.transact(_bind_revision)["revision"]
    repository = read_control.repository
    read = repository.read_execution if repository.execution_reads_enabled else repository.read_query
    if warm:
        _assert_bound(read())
    token_seen = threading.Event()
    committed = threading.Event()
    original = repository._committed_read_token

    def pause(connection):
        token = original(connection)
        if not token_seen.is_set():
            token_seen.set()
            assert committed.wait(10), "writer did not commit while the reader held its snapshot"
        return token

    monkeypatch.setattr(repository, "_committed_read_token", pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reading = pool.submit(read)
        try:
            assert token_seen.wait(10), "reader did not reach generation lookup"
            updated = _second_repository(read_control).transact(_bind_revision)
        finally:
            committed.set()
        view = reading.result(timeout=15)
    assert view["revision"] == initial
    _assert_bound(view)
    current = read()
    assert current["revision"] == updated["revision"] == initial + 1
    _assert_bound(current)


def _connection_control(connect, mode):
    @contextmanager
    def transaction():
        with closing(connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    repository = SqliteModelControlRepository(
        connect=connect, transaction=transaction, read_mode=mode)
    return SimpleNamespace(connect=connect, transaction=transaction, repository=repository,
                           owner=DeploymentService(repository, boot_id="boot-1", clock=lambda: 1.0),
                           mode=mode)


def _legacy_generation_token(control):
    with closing(control.connect()) as connection:
        schema = connection.execute("PRAGMA schema_version").fetchone()[0]
        row = connection.execute(
            "SELECT incarnation,generation FROM model_control_read_revision WHERE id=1").fetchone()
        return schema, row["incarnation"], row["generation"]


@pytest.fixture
def forked_file_controls(read_control, tmp_path):
    _publish(read_control)
    _seed_query(read_control)
    with closing(read_control.connect()) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    branch_path = tmp_path / "forked-read-views.sqlite3"
    shutil.copyfile(read_control.path, branch_path)

    def connect():
        connection = sqlite3.connect(branch_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    branch = _connection_control(connect, read_control.mode)
    branch.path = branch_path
    active = read_control.repository.transact(
        lambda state: state["operations"]["live"].update(reason="active branch"))
    invalid = branch.repository.transact(_invalidate)
    assert active["revision"] == invalid["revision"]
    assert _legacy_generation_token(read_control) == _legacy_generation_token(branch)
    assert read_control.repository.recovery_status()["verified"]
    assert branch.repository.recovery_status()["verified"]
    return read_control, branch


def test_switching_cloned_database_files_cannot_reuse_active_view(forked_file_controls, monkeypatch):
    control, branch = forked_file_controls
    before = _views(control)
    assert before[0]["active_deployment"] == "deployment-1"
    assert _runnable(control.owner)
    monkeypatch.setattr(control.repository, "_connect", branch.connect)
    assert control.repository.read()["active_deployment"] == ""
    for _ in range(3):
        execution, query, stamp = _views(control)
        assert execution["revision"] == query["revision"] == stamp["revision"] == before[0]["revision"]
        assert execution["active_deployment"] == query["active_deployment"] == ""
        assert not _runnable(control.owner)
    assert control.repository.recovery_status()["verified"]
    monkeypatch.setattr(control.repository, "_connect", control.connect)
    assert _runnable(control.owner)
    assert control.owner.execution_snapshot()["active_deployment"] == "deployment-1"


def test_replacing_database_at_same_path_cannot_reuse_active_view(forked_file_controls):
    control, branch = forked_file_controls
    before = _views(control)
    assert _runnable(control.owner)
    original_path = control.path.resolve()
    replacement_path = branch.path.resolve()
    assert original_path.parent == replacement_path.parent
    assert original_path != replacement_path
    os.replace(replacement_path, original_path)
    assert control.repository.read()["active_deployment"] == ""
    assert control.repository.recovery_status()["verified"]
    for _ in range(3):
        execution, query, stamp = _views(control)
        assert execution["revision"] == query["revision"] == stamp["revision"] == before[0]["revision"]
        assert execution["active_deployment"] == query["active_deployment"] == ""
        assert not _runnable(control.owner)


def test_cloned_memory_databases_use_uncached_reads(read_control, tmp_path, monkeypatch):
    _publish(read_control)
    _seed_query(read_control)
    anchors = []
    controls = []
    try:
        for index in range(2):
            uri = f"file:control-read-{tmp_path.name}-{index}?mode=memory&cache=shared"
            anchor = sqlite3.connect(uri, uri=True, timeout=10)
            anchors.append(anchor)
            with closing(read_control.connect()) as source:
                source.backup(anchor)

            def connect(database=uri):
                connection = sqlite3.connect(database, uri=True, timeout=10)
                connection.row_factory = sqlite3.Row
                return connection

            controls.append(_connection_control(connect, read_control.mode))
        control, branch = controls
        active = control.repository.transact(
            lambda state: state["operations"]["live"].update(reason="active memory branch"))
        invalid = branch.repository.transact(_invalidate)
        assert active["revision"] == invalid["revision"]
        assert _legacy_generation_token(control) == _legacy_generation_token(branch)
        assert control.repository.recovery_status()["verified"]
        assert branch.repository.recovery_status()["verified"]
        with closing(control.connect()) as connection:
            assert connection.execute("PRAGMA database_list").fetchone()["file"] == ""
            assert control.repository._committed_read_token(connection) is None
        reads = _count_reads(monkeypatch, control.repository)
        for _ in range(3):
            assert all(view["revision"] == active["revision"] for view in _views(control))
        assert len(reads) == 9
        assert _runnable(control.owner)
        monkeypatch.setattr(control.repository, "_connect", branch.connect)
        reads.clear()
        for _ in range(3):
            execution, query, stamp = _views(control)
            assert execution["active_deployment"] == query["active_deployment"] == ""
            assert stamp["revision"] == invalid["revision"]
            assert not _runnable(control.owner)
        assert len(reads) == 12
    finally:
        for anchor in anchors:
            anchor.close()


def test_permission_ctime_changes_do_not_invalidate_same_generation(read_control, monkeypatch):
    _publish(read_control)
    _seed_query(read_control)
    repository = read_control.repository
    generation = _legacy_generation_token(read_control)
    filename = os.path.normcase(os.path.realpath(read_control.path))
    baseline = os.stat(filename)
    observed_ctimes = []

    def permission_only_stat(path):
        assert path == filename
        ctime = baseline.st_ctime_ns + len(observed_ctimes) + 1
        observed_ctimes.append(ctime)
        return SimpleNamespace(st_dev=baseline.st_dev, st_ino=baseline.st_ino,
                               st_mtime_ns=baseline.st_mtime_ns, st_size=baseline.st_size,
                               st_ctime_ns=ctime)

    monkeypatch.setattr(repository_module, "os", SimpleNamespace(
        path=os.path, environ=os.environ, stat=permission_only_stat))
    reads = _count_reads(monkeypatch, repository)
    optimized = (repository.execution_reads_enabled, repository.query_reads_enabled,
                 repository.query_reads_enabled)
    expected_reads = int(any(optimized)) + 5 * optimized.count(False)
    revision = repository.read()["revision"]
    reads.clear()
    for _ in range(5):
        assert all(view["revision"] == revision for view in _views(read_control))
    assert _legacy_generation_token(read_control) == generation
    assert reads == [revision] * expected_reads
    committed = _second_repository(read_control).transact(
        lambda state: state["operations"]["live"].update(reason="after permission changes"))
    assert committed["revision"] == revision + 1
    reads.clear()
    for _ in range(5):
        assert all(view["revision"] == committed["revision"] for view in _views(read_control))
    assert reads == [committed["revision"]] * expected_reads
    assert len(observed_ctimes) == 10 * optimized.count(True)
    assert len(set(observed_ctimes)) == len(observed_ctimes)
