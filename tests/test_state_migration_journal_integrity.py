"""Offline migration rejects incomplete journal domains before creating outputs."""
from contextlib import contextmanager
import hashlib
import sqlite3

import pytest

from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
from qlh_core.adapters.request_repository import SqliteRequestRepository
from qlh_core.adapters.state_migration import MigrationError, migrate


pytestmark = pytest.mark.refactor_contract


def tree_digest(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def repository_at(path, adapter):
    def connect():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def transaction():
        connection = connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    return adapter(connect=connect, transaction=transaction, max_events=2)


@pytest.fixture
def legacy_source(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'qlh-control.sqlite3'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE sessions(session_id TEXT PRIMARY KEY, title TEXT)')
        connection.execute("INSERT INTO sessions VALUES ('saved', 'retained history')")
    return source, path


@pytest.fixture(params=[('model_control', SqliteModelControlRepository),
                        ('request_journal', SqliteRequestRepository)])
def journal_source(legacy_source, request):
    source, path = legacy_source
    prefix, adapter = request.param
    repository = repository_at(path, adapter)
    for index in range(4):
        repository.transact(lambda state: state.update(boot_id=str(index)))
    assert repository.recovery_status()['verified']
    with sqlite3.connect(path) as connection:
        assert connection.execute(f'SELECT COUNT(*) FROM {prefix}_events_checkpoint').fetchone()[0] == 1
    return source, path, prefix, adapter


@pytest.mark.parametrize('dry_run', [True, False])
@pytest.mark.parametrize('remaining', [
    ('state',), ('events',), ('events_checkpoint',),
    ('state', 'events_checkpoint'), ('events', 'events_checkpoint'),
])
def test_partial_journal_domain_fails_before_creating_outputs(journal_source, tmp_path, dry_run, remaining):
    source, path, prefix, _ = journal_source
    with sqlite3.connect(path) as connection:
        for suffix in ('state', 'events', 'events_checkpoint'):
            if suffix not in remaining:
                connection.execute(f'DROP TABLE {prefix}_{suffix}')
    before = tree_digest(source)
    target, backup = tmp_path / 'target', tmp_path / 'backup'

    with pytest.raises(MigrationError, match='migration_database_(corrupt|journal_missing)'):
        migrate(source, target, backup, dry_run=dry_run)

    assert tree_digest(source) == before
    assert not target.exists() and not backup.exists()
    assert not list(tmp_path.glob('target.pending-*'))


@pytest.mark.parametrize('dry_run', [True, False])
def test_missing_state_row_fails_before_creating_outputs(journal_source, tmp_path, dry_run):
    source, path, prefix, _ = journal_source
    with sqlite3.connect(path) as connection:
        connection.execute(f'DELETE FROM {prefix}_state WHERE id=1')
    before = tree_digest(source)
    target, backup = tmp_path / 'target', tmp_path / 'backup'

    with pytest.raises(MigrationError, match='migration_database_(corrupt|journal_missing)'):
        migrate(source, target, backup, dry_run=dry_run)

    assert tree_digest(source) == before
    assert not target.exists() and not backup.exists()
    assert not list(tmp_path.glob('target.pending-*'))


@pytest.mark.parametrize('dry_run', [True, False])
@pytest.mark.parametrize('remaining', ['events', 'events_checkpoint'])
def test_empty_evidence_table_still_requires_verification(journal_source, tmp_path, dry_run, remaining):
    source, path, prefix, _ = journal_source
    with sqlite3.connect(path) as connection:
        for suffix in ('state', 'events', 'events_checkpoint'):
            if suffix != remaining:
                connection.execute(f'DROP TABLE {prefix}_{suffix}')
        connection.execute(f'DELETE FROM {prefix}_{remaining}')
    before = tree_digest(source)
    target, backup = tmp_path / 'target', tmp_path / 'backup'

    with pytest.raises(MigrationError, match='migration_database_(corrupt|journal_missing)'):
        migrate(source, target, backup, dry_run=dry_run)

    assert tree_digest(source) == before
    assert not target.exists() and not backup.exists()
    assert not list(tmp_path.glob('target.pending-*'))


@pytest.mark.parametrize('dry_run', [True, False])
def test_valid_checkpoint_domain_migrates_and_remains_verified(journal_source, tmp_path, dry_run):
    source, path, _, adapter = journal_source
    before = tree_digest(source)
    target, backup = tmp_path / 'target', tmp_path / 'backup'

    plan = migrate(source, target, backup, dry_run=dry_run)

    assert plan['dry_run'] is dry_run
    assert tree_digest(source) == before
    if dry_run:
        assert not target.exists() and not backup.exists()
    else:
        backup_digest = tree_digest(backup)
        assert all(backup_digest[name] == value for name, value in before.items())
        repository = repository_at(target / path.name, adapter)
        assert repository.recovery_status()['verified']
        assert repository.read()['boot_id'] == 'migration'


@pytest.mark.parametrize('dry_run', [True, False])
def test_legacy_database_without_journal_evidence_still_migrates(legacy_source, tmp_path, dry_run):
    source, path = legacy_source
    before = tree_digest(source)
    target, backup = tmp_path / 'target', tmp_path / 'backup'

    plan = migrate(source, target, backup, dry_run=dry_run)

    assert plan['dry_run'] is dry_run
    assert tree_digest(source) == before
    if dry_run:
        assert not target.exists() and not backup.exists()
    else:
        backup_digest = tree_digest(backup)
        assert all(backup_digest[name] == value for name, value in before.items())
        with sqlite3.connect(target / path.name) as connection:
            assert connection.execute('SELECT title FROM sessions').fetchone()[0] == 'retained history'
            assert {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {'sessions'}
