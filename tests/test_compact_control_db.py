"""Offline control maintenance preserves user data, live state and recovery evidence."""
from contextlib import closing
from copy import deepcopy
import json
import sqlite3

import pytest

from scripts import compact_control_db
from qlh_core.adapters.model_control_repository import SqliteModelControlRepository, empty_state as model_state
from qlh_core.adapters.request_repository import SqliteRequestRepository, empty_state as request_state
from qlh_core.contracts.deployment import ModelControlError, canonical, digest
from qlh_core.contracts.execution import ExecutionError


pytestmark = pytest.mark.refactor_contract
ADAPTERS = {'model_control': SqliteModelControlRepository, 'request_journal': SqliteRequestRepository}
USER_TABLES = ('sessions', 'session_messages', 'settings', 'sqlite_sequence')


def repository_at(path, prefix):
    def connect():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection

    return ADAPTERS[prefix](connect=connect, transaction=None)


def database_snapshot(path):
    with closing(sqlite3.connect(path)) as connection:
        tables = {}
        for name, sql in connection.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            quoted = '"' + name.replace('"', '""') + '"'
            rows = connection.execute(f'SELECT * FROM {quoted}').fetchall()
            tables[name] = {'sql': sql, 'rows': sorted(rows, key=repr)}
        return {'tables': tables, 'user_version': connection.execute('PRAGMA user_version').fetchone()[0]}


def live_model_state(*, history):
    state = model_state()
    state.update(boot_id='live-boot', selected_operation='live-operation', active_deployment='live-deployment',
                 lifecycle={'config_id': 'live-deployment', 'phase': 'ready', 'published': True})
    state['operations']['live-operation'] = {
        'operation_id': 'live-operation', 'status': 'preparing', 'config_id': 'live-deployment'}
    state['operations']['cleanup-operation'] = {
        'operation_id': 'cleanup-operation', 'status': 'failed', 'config_id': 'retiring-deployment'}
    state['supplies']['live-operation'] = {'operation_id': 'live-operation', 'status': 'building'}
    state['deployments']['live-deployment'] = {
        'deployment_id': 'live-deployment', 'status': 'active', 'published': True, 'validity': 'valid'}
    state['deployments']['retiring-deployment'] = {
        'deployment_id': 'retiring-deployment', 'status': 'retiring', 'published': False}
    state['cleanup']['cleanup-debt'] = {
        'cleanup_id': 'cleanup-debt', 'node_id': 'worker', 'candidate_config_id': 'retiring-deployment',
        'action': 'release', 'state': 'pending'}
    state['artifact_cleanup']['artifact-debt'] = {'location': '/assets/held', 'status': 'pending'}
    state['outbox']['live-command'] = {
        'command_id': 'live-command', 'node_id': 'worker', 'action': 'commit', 'state': 'pending',
        'payload': {'config_id': 'live-deployment'}}
    state['outbox']['cleanup-command'] = {
        'command_id': 'cleanup-command', 'node_id': 'worker', 'action': 'release', 'state': 'pending',
        'payload': {'cleanup_id': 'cleanup-debt', 'config_id': 'retiring-deployment'}}
    if history:
        for index in range(40):
            identity = f'old-operation-{index:03}'
            state['operations'][identity] = {
                'operation_id': identity, 'status': 'failed', 'created_at': index,
                'intent': {'model_id': f'old-model-{index}'}, 'idempotency_key': f'old-key-{index}',
                'reason': 'x' * 512}
    return state


@pytest.fixture
def database_factory(tmp_path):
    def create(*, domains=tuple(ADAPTERS), event_count=4, checkpoint=True, history=False):
        directory = tmp_path / 'database'
        directory.mkdir()
        path = directory / 'control.sqlite3'
        with closing(sqlite3.connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA user_version=7')
            connection.execute('CREATE TABLE sessions(session_id TEXT PRIMARY KEY, title TEXT)')
            connection.execute("INSERT INTO sessions VALUES ('saved', 'retained history')")
            connection.execute('CREATE TABLE session_messages(message_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                               'session_id TEXT, content TEXT)')
            connection.execute("INSERT INTO session_messages(session_id,content) VALUES ('saved', 'preserve me')")
            connection.execute('CREATE TABLE settings(setting_key TEXT PRIMARY KEY, value BLOB) WITHOUT ROWID')
            connection.executemany('INSERT INTO settings VALUES (?,?)', [('theme', b'\x00\xff'), ('optional', None)])
            for prefix in domains:
                repository = repository_at(path, prefix)
                repository._initialize(connection)
                if checkpoint:
                    repository._journal.initialize(connection)
                state = live_model_state(history=history) if prefix == 'model_control' else request_state()
                if prefix == 'request_journal':
                    state.update(boot_id='live-boot', requests={
                        'running-request': {'request_id': 'running-request', 'attempt_id': 'attempt',
                                            'execution_state': 'running', 'terminal': None,
                                            'assignments': [{'node_id': 'worker'}]},
                        'cleanup-request': {'request_id': 'cleanup-request', 'execution_state': 'cleanup_pending',
                                            'terminal': 'completed', 'assignments': [{'node_id': 'worker'}]}})
                previous = '0' * 64
                for sequence in range(1, event_count + 1):
                    state['revision'] = sequence
                    event = {'sequence': sequence, 'domain_id': repository._journal.domain,
                             'previous_digest': previous, 'state': state}
                    event_digest = digest(event)
                    connection.execute(f'INSERT INTO {prefix}_events VALUES (?,?,?,?,?)',
                                       (sequence, repository._journal.domain, previous, event_digest, canonical(event)))
                    previous = event_digest
                connection.execute(f'INSERT INTO {prefix}_state VALUES (1,?)', (canonical(state),))
                if checkpoint:
                    anchor_sequence = max(1, event_count // 2)
                    anchor = connection.execute(f'SELECT * FROM {prefix}_events WHERE sequence=?',
                                                (anchor_sequence,)).fetchone()
                    connection.execute(f'INSERT INTO {prefix}_events_checkpoint VALUES (1,?,?,?,?)',
                                       (anchor['sequence'], anchor['domain_id'], anchor['digest'], anchor['payload']))
                    connection.execute(f'DELETE FROM {prefix}_events WHERE sequence<=?', (anchor_sequence,))
            connection.commit()
        for prefix in domains:
            assert repository_at(path, prefix).recovery_status()['verified']
        return path

    return create


@pytest.mark.parametrize('supply_backup', [False, True])
def test_dry_run_never_changes_database_or_creates_backup(database_factory, tmp_path, supply_backup):
    path = database_factory()
    backup, output = tmp_path / 'backup.sqlite3', tmp_path / 'report.json'
    before_bytes, before = path.read_bytes(), database_snapshot(path)
    before_files = set(path.parent.iterdir())
    args = ['--database', str(path), '--output', str(output)]
    if supply_backup:
        args.extend(['--offline', '--backup', str(backup)])

    assert compact_control_db.main(args) == 0

    assert path.read_bytes() == before_bytes
    assert database_snapshot(path) == before
    assert set(path.parent.iterdir()) == before_files
    assert not backup.exists()
    assert json.loads(output.read_text(encoding='utf-8'))['dry_run'] is True


@pytest.mark.parametrize('offline,backup_argument', [(False, False), (True, False), (False, True)])
def test_apply_requires_offline_and_backup(database_factory, tmp_path, capsys, offline, backup_argument):
    path = database_factory()
    backup = tmp_path / 'backup.sqlite3'
    before = path.read_bytes()
    args = ['--database', str(path), '--apply']
    if offline:
        args.append('--offline')
    if backup_argument:
        args.extend(['--backup', str(backup)])

    with pytest.raises(SystemExit) as error:
        compact_control_db.main(args)

    assert error.value.code == 2
    assert '--apply requires --offline and --backup' in capsys.readouterr().err
    assert path.read_bytes() == before
    assert not backup.exists()


@pytest.mark.parametrize('destination', ['existing', 'database', 'missing_parent'])
def test_apply_requires_new_separate_backup_file(database_factory, tmp_path, destination):
    path = database_factory()
    backup = tmp_path / 'backup.sqlite3'
    if destination == 'existing':
        backup.write_bytes(b'existing backup must survive')
    elif destination == 'database':
        backup = path
    else:
        backup = tmp_path / 'absent' / 'backup.sqlite3'
    before = path.read_bytes()
    backup_before = backup.read_bytes() if backup.exists() else None

    with pytest.raises(ValueError, match='backup'):
        compact_control_db.main(['--database', str(path), '--apply', '--offline', '--backup', str(backup)])

    assert path.read_bytes() == before
    if backup_before is None:
        assert not backup.exists() and not backup.parent.exists()
    else:
        assert backup.read_bytes() == backup_before


@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize('collision', ['database', '-wal', '-shm', '-journal', 'backup'])
def test_cli_rejects_output_collisions_before_touching_files(database_factory, tmp_path, capsys, apply, collision):
    path = database_factory()
    backup = tmp_path / 'backup.sqlite3'
    output = path if collision == 'database' else backup if collision == 'backup' else path.with_name(path.name + collision)
    before = {entry: entry.read_bytes() for entry in path.parent.iterdir() if entry.is_file()}
    args = ['--database', str(path), '--backup', str(backup), '--output', str(output)]
    if apply:
        args.extend(['--apply', '--offline'])

    with pytest.raises(SystemExit) as error:
        compact_control_db.main(args)

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert '--output must be separate from the database and backup' in captured.err
    assert not captured.out
    assert {entry: entry.read_bytes() for entry in path.parent.iterdir() if entry.is_file()} == before
    assert not backup.exists()


def test_apply_compacts_and_vacuums_with_original_backup_and_live_state_preserved(database_factory, tmp_path):
    path = database_factory(event_count=160, checkpoint=False, history=True)
    backup, output = tmp_path / 'backup.sqlite3', tmp_path / 'report.json'
    before, before_bytes = database_snapshot(path), path.stat().st_size
    repositories = {prefix: repository_at(path, prefix) for prefix in ADAPTERS}
    states = {prefix: repository.read() for prefix, repository in repositories.items()}
    statuses = {prefix: repository.recovery_status() for prefix, repository in repositories.items()}

    assert compact_control_db.main(['--database', str(path), '--apply', '--offline', '--backup', str(backup),
                                    '--output', str(output)]) == 0

    report = json.loads(output.read_text(encoding='utf-8'))
    after = database_snapshot(path)
    assert report['dry_run'] is False
    assert database_snapshot(backup) == before
    assert after['user_version'] == before['user_version']
    assert {name: after['tables'][name] for name in USER_TABLES} == {
        name: before['tables'][name] for name in USER_TABLES}
    assert path.stat().st_size < before_bytes
    assert report['after']['file_bytes'] == path.stat().st_size
    assert report['after']['free_pages'] == 0
    model_after = repositories['model_control'].read()
    for key in ('boot_id', 'selected_operation', 'active_deployment', 'cleanup', 'artifact_cleanup', 'lifecycle',
                'supplies', 'deployments', 'outbox'):
        assert model_after[key] == states['model_control'][key]
    for identity in ('live-operation', 'cleanup-operation'):
        assert model_after['operations'][identity] == states['model_control']['operations'][identity]
    assert len(model_after['operations']) < len(states['model_control']['operations'])
    assert repositories['request_journal'].read() == states['request_journal']
    for prefix, repository in repositories.items():
        status = repository.recovery_status()
        assert status['verified'] and status['commit_scope'] == 'local_sqlite'
        assert repository_at(backup, prefix).recovery_status() == statuses[prefix]
        rows = after['tables'][prefix + '_events']['rows']
        assert 1 <= len(rows) < len(before['tables'][prefix + '_events']['rows'])
        assert after['tables'][prefix + '_events_checkpoint']['rows']
        assert report['journals'][prefix + '_state']['after'] == status
    assert repositories['request_journal'].recovery_status() == statuses['request_journal']


def test_fingerprints_support_without_rowid_and_detect_changed_binary_data():
    with closing(sqlite3.connect(':memory:')) as connection:
        connection.execute('CREATE TABLE settings(setting_key TEXT PRIMARY KEY, value BLOB) WITHOUT ROWID')
        connection.executemany('INSERT INTO settings VALUES (?,?)', [('theme', b'\x00\xff'), ('optional', None)])

        before = compact_control_db.fingerprints(connection)

        assert before['settings']['rows'] == 2
        assert compact_control_db.fingerprints(connection) == before
        connection.execute("UPDATE settings SET value=? WHERE setting_key='theme'", (b'\x00\xfe',))
        assert compact_control_db.fingerprints(connection) != before


@pytest.mark.parametrize('prefix,error', [('model_control', ModelControlError), ('request_journal', ExecutionError)])
@pytest.mark.parametrize('damage', ['event_payload', 'event_domain', 'checkpoint_payload', 'snapshot',
                                    'missing_state_row', 'missing_events', 'missing_state',
                                    'orphan_events', 'orphan_checkpoint'])
def test_apply_rejects_corrupt_or_orphaned_domains(database_factory, tmp_path, prefix, error, damage):
    path = database_factory(domains=(prefix,))
    with closing(sqlite3.connect(path)) as connection:
        if damage == 'event_payload':
            connection.execute(f"UPDATE {prefix}_events SET payload='{{}}' WHERE sequence=4")
        elif damage == 'event_domain':
            connection.execute(f"UPDATE {prefix}_events SET domain_id='foreign:v1' WHERE sequence=4")
        elif damage == 'checkpoint_payload':
            connection.execute(f"UPDATE {prefix}_events_checkpoint SET payload='{{}}'")
        elif damage == 'snapshot':
            state = deepcopy(repository_at(path, prefix).read())
            state['boot_id'] = 'tampered'
            connection.execute(f'UPDATE {prefix}_state SET payload=? WHERE id=1', (canonical(state),))
        elif damage == 'missing_state_row':
            connection.execute(f'DELETE FROM {prefix}_state WHERE id=1')
        elif damage == 'missing_events':
            connection.execute(f'DROP TABLE {prefix}_events')
        else:
            connection.execute(f'DROP TABLE {prefix}_state')
            if damage == 'orphan_events':
                connection.execute(f'DROP TABLE {prefix}_events_checkpoint')
            elif damage == 'orphan_checkpoint':
                connection.execute(f'DROP TABLE {prefix}_events')
        connection.commit()
    before, before_bytes = database_snapshot(path), path.read_bytes()
    backup, output = tmp_path / 'backup.sqlite3', tmp_path / 'report.json'

    with pytest.raises(error, match='journal_corrupt'):
        compact_control_db.main(['--database', str(path), '--apply', '--offline', '--backup', str(backup),
                                '--output', str(output)])

    assert path.read_bytes() == before_bytes
    assert database_snapshot(path) == before
    assert not output.exists()
    if backup.exists():
        assert database_snapshot(backup) == before
