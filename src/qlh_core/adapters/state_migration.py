"""Explicit offline state migration; dry-run reads only and original files stay intact."""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sqlite3
import time
import uuid

class MigrationError(ValueError):
    pass

def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()

def _read_json(path):
    try: return json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, OSError) as exc: raise MigrationError('migration_document_unreadable') from exc

def _credentials(path):
    value = _read_json(path)
    if not isinstance(value, dict) or value.get('schema') != 'qlh.local-secret-store.v1':
        raise MigrationError('migration_credential_schema_unknown')
    key_path = path.with_suffix(path.suffix + '.key')
    if not key_path.is_file(): raise MigrationError('migration_credential_key_missing')
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    def decode(value): return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
    try:
        key = decode(key_path.read_text(encoding='ascii').strip())
        for name, record in value['records'].items():
            text = AESGCM(key).decrypt(decode(record['nonce']), decode(record['ciphertext']), name.encode()).decode()
            if name == 'cluster_secret':
                credential = json.loads(text) if text.startswith('{') else None
                if credential and (credential.get('schema') != 'qlh.cluster-credential.v1' or not credential.get('secret') or int(credential.get('epoch', 0)) < 1):
                    raise ValueError('credential invalid')
    except Exception as exc: raise MigrationError('migration_credential_corrupt') from exc

def _connect(path):
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    return connection

def _database(path):
    connection = _connect(path)
    try:
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok': raise MigrationError('migration_database_corrupt')
        if connection.execute('PRAGMA user_version').fetchone()[0] != 0: raise MigrationError('migration_database_schema_unknown')
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ('model_control_state', 'request_journal_state'):
            if table in tables:
                row = connection.execute('SELECT payload FROM ' + table + ' WHERE id=1').fetchone()
                state = json.loads(row[0]) if row else {}
                if state.get('schema_version') != 1: raise MigrationError('migration_control_schema_unknown')
                for record in state.get('requests', {}).values():
                    if record.get('execution_state') not in {'stopped', 'waiting'}:
                        raise MigrationError('migration_runtime_reconciliation_required')
        # Verify the complete chain before creating either backup or staging.
        from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
        from qlh_core.adapters.request_repository import SqliteRequestRepository
        for table, adapter in (('model_control_state', SqliteModelControlRepository), ('request_journal_state', SqliteRequestRepository)):
            if table in tables:
                repository = adapter(connect=lambda: _connect(path), transaction=None)
                if not repository.recovery_status()['verified']:
                    raise MigrationError('migration_database_journal_missing')
        return sorted(tables)
    except MigrationError:
        raise
    except Exception as exc:
        raise MigrationError('migration_database_corrupt') from exc
    finally: connection.close()

def inspect(source, target, backup):
    source, target, backup = [Path(p).expanduser().resolve() for p in (source, target, backup)]
    if not source.is_dir(): raise MigrationError('migration_source_missing')
    for a, b in ((source,target),(source,backup),(target,backup)):
        if a == b or a.is_relative_to(b) or b.is_relative_to(a): raise MigrationError('migration_paths_overlap')
    files = []
    for path in sorted(source.rglob('*')):
        if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024):
            raise MigrationError('migration_symlink_unsupported')
        if not path.is_file(): continue
        if path.name.endswith(('-wal', '-journal')) and path.stat().st_size:
            raise MigrationError('migration_source_busy')
        if path.name == 'node_config.json':
            value = _read_json(path)
            if not isinstance(value,dict) or value.get('schema_version',1) not in {1,2}:
                raise MigrationError('migration_config_schema_unknown')
        if path.name == 'node_secrets.json': _credentials(path)
        tables = _database(path) if path.suffix in {'.db','.sqlite','.sqlite3'} else None
        files.append({'path':path.relative_to(source).as_posix(),'bytes':path.stat().st_size,'sha256':_hash(path), **({'tables':tables} if tables is not None else {})})
    rewrites = []
    for entry in files:
        if Path(entry['path']).name == 'node_config.json':
            _remap_config(_read_json(source / entry['path']), source, target, rewrites, entry['path'])
    fingerprint = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if target.exists():
        report = target/'migration-report.json'
        if report.is_file() and _read_json(report).get('source_fingerprint') == fingerprint:
            return {**_read_json(report), 'replayed':True}
        raise MigrationError('migration_target_exists')
    if backup.exists(): raise MigrationError('migration_backup_exists')
    size = sum(f['bytes'] for f in files)
    for destination in (target,backup):
        ancestor=destination.parent
        while not ancestor.exists(): ancestor=ancestor.parent
        if shutil.disk_usage(ancestor).free < 2 * size + 1024 * 1024:
            raise MigrationError('migration_space_insufficient')
    return {'schema_version':2,'source':str(source),'target':str(target),'backup':str(backup),
        'source_fingerprint':fingerprint,'files':files,'config_path_rewrites':rewrites,'required_bytes':2*size+1024*1024,
        'preserved':['node_identity','credential_epoch','sessions','preferences','artifact_references','workbench_files'],
        'runtime_policy':'invalidate_readiness_and_kv; terminate_unfinished_requests; reconcile_cleanup_before_reuse',
        'android_policy':'Room and system SAF grants remain with Android; URI references alone confer no grant', 'replayed':False}

def _sanitize_database(path, tables):
    def connect():
        c=sqlite3.connect(path);c.row_factory=sqlite3.Row;return c
    @contextmanager
    def transaction():
        c=connect()
        try:
            c.execute('BEGIN IMMEDIATE');yield c;c.commit()
        except BaseException: c.rollback();raise
        finally:c.close()
    from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
    from qlh_core.adapters.request_repository import SqliteRequestRepository
    debts=[]
    def invalidate(state):
        state.update(boot_id='migration', active_deployment='', selected_operation='', lifecycle=None)
        for operation in state['operations'].values():
            if operation['status'] not in {'ready','failed','cancelled','superseded'}:
                operation.update(status='failed',phase='failed',reason_code='operation_state_migrated')
        for deployment in state['deployments'].values():
            deployment.update(status='invalidated',validity='unknown',published=False,reason_code='deployment_state_migrated')
        for command in state['outbox'].values():
            if command['action'] in {'prepare','commit'}:command['state']='superseded'
        debts.extend({'domain':'model_control','cleanup_id':key,'state':'reconciliation_required'} for key in state['cleanup'])
    if 'model_control_state' in tables:
        model=SqliteModelControlRepository(connect=connect,transaction=transaction)
        model.transact(invalidate)
    def terminate(state):
        state['boot_id']='migration'
        for value in state['requests'].values():
            if not value.get('terminal'):value.update(terminal='request_state_migrated',outcome='failed')
            if value.get('execution_state') == 'waiting': value['execution_state'] = 'stopped'
            if value.get('execution_state')!='stopped':
                value['execution_state']='recovery_required'
                debts.append({'domain':'request','request_id':value['request_id'],'assignments':deepcopy(value['assignments']),'state':'reconciliation_required'})
    if 'request_journal_state' in tables:
        request=SqliteRequestRepository(connect=connect,transaction=transaction)
        request.transact(terminate)
    return debts

# Only owned runtime roots are invalidated. User workbench/operations remains data.
def _runtime_receipt(relative):
    parts = Path(relative).parts
    return parts[0] in {'kv_cache', 'readiness', 'model-intents'} or (
        len(parts) > 1 and parts[0] == 'runtime' and parts[1] in {'kv_cache', 'readiness', 'operations'})

_OWNED_PATH_KEYS = frozenset({'state_dir', 'sqlite_path', 'db_path', 'local_store_path',
    'node_config_path', 'secret_store_path', 'node_secret_store_path', 'log_dir',
    'QLH_STATE_DIR', 'QLH_SQLITE_PATH', 'QLH_NODE_CONFIG_PATH', 'QLH_NODE_SECRET_STORE_PATH', 'QLH_LOG_DIR'})

def _remap_config(document, source, target, changes, file, trail=''):
    result = deepcopy(document)
    for key, value in result.items():
        name = trail + '.' + key if trail else key
        if isinstance(value, dict):
            result[key] = _remap_config(value, source, target, changes, file, name)
        elif key in _OWNED_PATH_KEYS and isinstance(value, str) and Path(value).is_absolute():
            resolved = Path(value).resolve()
            if resolved.is_relative_to(source):
                destination = target / resolved.relative_to(source)
                result[key] = str(destination)
                changes.append({'file':file, 'key':name, 'source':str(resolved), 'target':str(destination)})
    return result

def migrate(source, target, backup, *, dry_run=True):
    plan=inspect(source,target,backup)
    if dry_run or plan['replayed']:return {**plan,'dry_run':dry_run}
    source,target,backup=[Path(plan[k]) for k in ('source','target','backup')]
    staging=target.with_name(target.name+'.pending-'+uuid.uuid4().hex)
    backup.mkdir(parents=True,exist_ok=False);staging.mkdir(parents=True,exist_ok=False)
    for entry in plan['files']:
        path=source/entry['path'];saved=backup/entry['path'];saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,saved)
        if _hash(saved)!=entry['sha256'] or _hash(path)!=entry['sha256']:raise MigrationError('migration_source_changed')
        # Old runtime receipts and KV stay in backup, outside the new live state tree.
        if _runtime_receipt(entry['path']):continue
        copied=staging/entry['path'];copied.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(saved,copied)
    final=inspect(source,target,target.with_name(target.name+'.verify-backup'))
    if final['source_fingerprint']!=plan['source_fingerprint']:raise MigrationError('migration_source_changed')
    debts=[]
    for entry in plan['files']:
        copied=staging/entry['path']
        if 'tables' in entry and copied.is_file():debts.extend(_sanitize_database(copied, entry['tables']))
        if copied.name == 'node_config.json' and copied.is_file():
            remapped = _remap_config(_read_json(copied), source, target, [], entry['path'])
            copied.write_text(json.dumps(remapped, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report={**plan,'dry_run':False,'completed_at':time.time(),'cleanup_reconciliation':debts}
    for folder in (backup,staging):
        (folder/'migration-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.rename(staging,target)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--target',required=True,type=Path)
    parser.add_argument('--backup',required=True,type=Path);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    try: print(json.dumps(migrate(args.source,args.target,args.backup,dry_run=args.dry_run),ensure_ascii=False,indent=2));return 0
    except MigrationError as exc:print(json.dumps({'error_code':str(exc)}));return 2
