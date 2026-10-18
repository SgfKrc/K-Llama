"""R5 client snapshots, session isolation, migration and retired compatibility surfaces."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
import api_server
from chat_context import ConversationContextService, current_client_id
from qlh_core.adapters.state_migration import MigrationError, migrate
from qlh_core.bootstrap.client_snapshots import ClientSnapshots
from qlh_core.control.participation import ParticipationService
from tests.test_refactor_execution_integration import application
from tests.test_refactor_execution import repository, launch, service, wait_for
pytestmark = pytest.mark.refactor_contract

def digest_tree(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}

def test_client_context_selection_is_independent_and_linear_prepare_does_not_touch_kv():
    owner=ConversationContextService(load_history=lambda _: [],persist_turn=lambda *a: True)
    first=owner.prepare('one',client_id='alpha',runtime_kv=False)
    second=owner.prepare('two',client_id='beta',runtime_kv=False)
    assert owner.resolve_session(client_id='alpha')=='one'
    assert owner.resolve_session(client_id='beta')=='two'
    assert owner._kv_session_id is None
    owner.commit_turn(first.session_id,'question','answer',expected_revision=first.revision)
    assert owner.history_for(second.session_id)==[]
    owner.drop_session('one')
    assert owner.resolve_session(client_id='alpha')=='client-' + hashlib.sha256(b'alpha').hexdigest()[:40]
    assert owner.resolve_session(client_id='beta')=='two'

@pytest.mark.parametrize('path', ['/api/workflows','/api/workflows/example','/api/experimental/speculative','/api/cluster/queue/strategy'])
def test_retired_client_paths_return_named_version_error_without_loading(path,monkeypatch):
    monkeypatch.setattr(api_server.model_host,'load_model',lambda *a,**kw:pytest.fail('retired command loaded model'))
    response=TestClient(api_server.app).get(path)
    assert response.status_code==410
    assert response.json()['error_code']=='CLIENT_FEATURE_REMOVED'
    assert response.json()['reason_detail']['code']=='CLIENT_FEATURE_REMOVED'

def test_bad_client_header_does_not_leak_request_context():
    token=api_server._request_id_ctx.set('outer')
    try:
        response=TestClient(api_server.app).get('/api/client/state',headers={'X-QLH-Client-ID':'bad client'})
        assert response.status_code==400
        assert api_server._request_id_ctx.get()=='outer'
    finally:api_server._request_id_ctx.reset(token)

def test_request_history_distinguishes_terminal_from_native_drain_and_has_no_content(service):
    entered,release=threading.Event(),threading.Event()
    def execute(_):entered.set();assert release.wait(3);return {'response':'private'}
    result,thread=launch(service,'request',execute)
    assert entered.wait(3)
    service.cancel('request');thread.join(2)
    value=service.recent()
    assert value['schema_version']==2 and value['validity']=='valid'
    assert value['requests'][0]['terminal']=='generation_cancelled'
    assert value['requests'][0]['execution_state']=='running'
    assert 'private' not in json.dumps(value)
    release.set();wait_for(lambda:service.queue_snapshot()['active'] is None)
    assert service.recent()['requests'][0]['execution_state']=='stopped'

def test_client_projection_uses_execution_admission_and_never_repairs(application):
    app,_,state=application
    state.update(revision=1,boot_id='boot',selected_operation='',operations={},supplies={},commit_scope='local_sqlite')
    scheduler=app.scheduler;scheduler._execution_app=app
    scheduler.get_distributed_inference_enabled=lambda:True
    scheduler._participation=ParticipationService(enabled=True)
    before=deepcopy(state)
    value=ClientSnapshots(scheduler).current()
    assert value['runnable'] is True and value['deployment']['validity']=='valid'
    assert state==before and app.requests._started is False
    state['deployments']['deployment']['status']='invalidated'
    value=ClientSnapshots(scheduler).current()
    assert value['runnable'] is False
    assert value['reason_detail']['code']=='execution_deployment_unavailable'

def test_snapshot_revision_race_returns_pending_instead_of_mixed_read(application):
    app,_,state=application
    state.update(revision=1,boot_id='boot',selected_operation='',operations={},supplies={},commit_scope='local_sqlite')
    scheduler=app.scheduler;scheduler._execution_app=app;scheduler.get_distributed_inference_enabled=lambda:True
    original=scheduler._model_control.snapshot
    def racing():state['revision']+=1;return original()
    scheduler._model_control.snapshot=racing
    result=ClientSnapshots(scheduler).current()
    assert result['validity']=='pending' and not result['runnable']
    assert result['reason_detail']['code']=='snapshot_revision_changed'

def test_participation_persists_intent_and_exposes_transition_before_runtime_returns():
    calls=[];owner=ParticipationService(enabled=True,persist=calls.append)
    def retire(mode):
        snapshot=owner.snapshot()
        assert snapshot['desired']=='local' and snapshot['phase']=='leaving' and not snapshot['eligible']
        assert calls==['local']
        return {'mode':mode,'success':True}
    result=owner.change('local',retire)
    assert result['participation']['phase']=='stable' and result['participation']['revision']==2

def test_participation_reboot_restores_explicit_local_mode(tmp_path):
    from qlh_core.adapters.settings import IdentityStore, SettingsStore
    config=tmp_path/'node_config.json';secrets=tmp_path/'node_secrets.json'
    IdentityStore(str(config),str(secrets)).persist_participation('local')
    resolved=SettingsStore({'QLH_NODE_CONFIG_PATH':str(config),'QLH_NODE_SECRET_STORE_PATH':str(secrets),'QLH_STATE_DIR':str(tmp_path/'state')},dotenv_path=tmp_path/'absent').load()
    assert resolved.legacy_settings()['DISTRIBUTED_INFERENCE_ENABLED'] is False

def test_tui_rejects_stale_and_old_boot_snapshots():
    from tui_api import ApiClient,ApiError
    client=ApiClient();value={'schema_version':2,'revision':8,'identity':{'boot_id':'boot'},'observed_at':10,'validity':'valid'}
    assert client.accept_snapshot(value)==value
    for bad in [dict(value,revision=7),dict(value,observed_at=9,identity={'boot_id':'previous'}),dict(value,schema_version=1)]:
        with pytest.raises(ApiError):client.accept_snapshot(bad)
    assert client.accept_snapshot(dict(value,revision=1,observed_at=11,identity={'boot_id':'new'}))['revision']==1

def migration_source(tmp_path):
    root=tmp_path/'original';root.mkdir()
    (root/'node_config.json').write_text(json.dumps({'schema_version':1,'node':{'node_id':'preserved','desired_inference_mode':'local'},'preferences':{'theme':'saved'}}),encoding='utf-8')
    from local_secret_store import LocalSecretStore
    LocalSecretStore(root/'node_secrets.json').set('cluster_secret',json.dumps({'schema':'qlh.cluster-credential.v1','secret':'not-for-report','epoch':7}))
    (root/'workbench.json').write_text('{"draft":"untouched"}',encoding='utf-8')
    connection=sqlite3.connect(root/'qlh-control.sqlite3')
    connection.execute('CREATE TABLE sessions(session_id TEXT PRIMARY KEY,title TEXT)')
    connection.execute("INSERT INTO sessions VALUES ('session','history')")
    connection.commit();connection.close()
    return root

def test_migration_dry_run_is_read_only_and_apply_is_backed_up_and_idempotent(tmp_path):
    source=migration_source(tmp_path);target=tmp_path/'new';backup=tmp_path/'backup';before=digest_tree(source)
    plan=migrate(source,target,backup,dry_run=True)
    assert not target.exists() and not backup.exists() and digest_tree(source)==before
    assert 'not-for-report' not in json.dumps(plan)
    report=migrate(source,target,backup,dry_run=False)
    assert digest_tree(source)==before
    assert all(digest_tree(backup)[name]==value for name,value in before.items())
    assert (target/'workbench.json').read_bytes()==(source/'workbench.json').read_bytes()
    assert json.loads((target/'node_config.json').read_text())['node']['node_id']=='preserved'
    c=sqlite3.connect(target/'qlh-control.sqlite3')
    assert c.execute('SELECT title FROM sessions').fetchone()[0]=='history'
    assert c.execute("SELECT name FROM sqlite_master WHERE name='model_control_state'").fetchone() is None
    c.close()
    assert migrate(source,target,backup,dry_run=False)['replayed'] is True
    assert report['schema_version']==2

@pytest.mark.parametrize('failure',['unknown_schema','missing_key','corrupt_credential','overlap','insufficient_space'])
def test_migration_stops_before_writing_for_invalid_inputs(tmp_path,monkeypatch,failure):
    source=migration_source(tmp_path);target=tmp_path/'new';backup=tmp_path/'backup'
    if failure=='unknown_schema':(source/'node_config.json').write_text('{"schema_version":999}',encoding='utf-8')
    elif failure=='missing_key':(source/'node_secrets.json.key').unlink()
    elif failure=='corrupt_credential':(source/'node_secrets.json.key').write_text('corrupt',encoding='ascii')
    elif failure=='overlap':target=source/'nested'
    else:
        import qlh_core.adapters.state_migration as module
        monkeypatch.setattr(module.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    before=digest_tree(source)
    with pytest.raises(MigrationError):migrate(source,target,backup,dry_run=False)
    assert not target.exists() and not backup.exists() and digest_tree(source)==before


def migration_repositories(source):
    from contextlib import contextmanager
    from qlh_core.adapters.model_control_repository import SqliteModelControlRepository
    from qlh_core.adapters.request_repository import SqliteRequestRepository
    path = source / 'qlh-control.sqlite3'
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
    return (SqliteModelControlRepository(connect=connect, transaction=transaction),
            SqliteRequestRepository(connect=connect, transaction=transaction))


def test_migration_invalidates_runtime_ends_requests_and_preserves_debts_and_user_files(tmp_path):
    source = migration_source(tmp_path)
    model, requests = migration_repositories(source)
    def prepare(state):
        state.update(boot_id='old', active_deployment='deployment', selected_operation='operation', lifecycle={'phase':'ready'})
        state['operations']['operation'] = {'status':'preparing'}
        state['deployments']['deployment'] = {'status':'active','validity':'valid','published':True}
        state['outbox']['command'] = {'action':'commit','state':'pending'}
        state['cleanup'].update(local={'node_id':'local','state':'pending'}, remote={'node_id':'remote','state':'pending'})
    model.transact(prepare)
    requests.transact(lambda state: state['requests'].update(queued={
        'request_id':'queued', 'execution_state':'waiting', 'terminal':None, 'assignments':[]}))
    for relative in ('kv_cache/stale', 'runtime/readiness/ready', 'workbench/operations/draft', 'operations/user-draft'):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('retained-in-backup')
    config_path = source/'node_config.json'
    config = json.loads(config_path.read_text())
    config['paths'] = {'state_dir':str(source/'state'), 'sqlite_path':str(source/'qlh-control.sqlite3'),
        'model_path':str(source/'assets/model.gguf')}
    config_path.write_text(json.dumps(config), encoding='utf-8')
    before = digest_tree(source)
    target, backup = tmp_path/'new', tmp_path/'backup'
    report = migrate(source, target, backup, dry_run=False)
    assert digest_tree(source) == before
    assert all(digest_tree(backup)[name] == value for name, value in before.items())
    assert not (target/'kv_cache').exists() and not (target/'runtime/readiness').exists()
    assert (target/'workbench/operations/draft').read_text() == 'retained-in-backup'
    assert (target/'operations/user-draft').is_file()
    connection = sqlite3.connect(target/'qlh-control.sqlite3')
    state = json.loads(connection.execute('SELECT payload FROM model_control_state').fetchone()[0])
    saved_requests = json.loads(connection.execute('SELECT payload FROM request_journal_state').fetchone()[0])
    connection.close()
    assert state['active_deployment'] == state['selected_operation'] == ''
    assert state['lifecycle'] is None and state['boot_id'] == 'migration'
    assert state['deployments']['deployment']['published'] is False
    assert state['operations']['operation']['reason_code'] == 'operation_state_migrated'
    assert state['outbox']['command']['state'] == 'superseded'
    assert set(state['cleanup']) == {'local','remote'}
    assert {debt['cleanup_id'] for debt in report['cleanup_reconciliation']} == {'local','remote'}
    assert saved_requests['requests']['queued']['terminal'] == 'request_state_migrated'
    assert saved_requests['requests']['queued']['execution_state'] == 'stopped'
    mapped = json.loads((target/'node_config.json').read_text())['paths']
    assert mapped['state_dir'] == str(target/'state')
    assert mapped['sqlite_path'] == str(target/'qlh-control.sqlite3')
    assert mapped['model_path'] == str(source/'assets/model.gguf')
    assert len(report['config_path_rewrites']) == 2


@pytest.mark.parametrize('failure', ['corrupt_chain','missing_chain','live_runtime','unknown_database_schema'])
def test_migration_journal_and_runtime_fail_before_backup(tmp_path, failure):
    source = migration_source(tmp_path)
    model, requests = migration_repositories(source)
    model.transact(lambda state: state.update(boot_id='old'))
    if failure == 'live_runtime':
        requests.transact(lambda state: state['requests'].update(live={
            'request_id':'live', 'execution_state':'running', 'terminal':None, 'assignments':[]}))
    else:
        connection = sqlite3.connect(source/'qlh-control.sqlite3')
        if failure == 'corrupt_chain': connection.execute("UPDATE model_control_events SET digest='corrupt'")
        elif failure == 'missing_chain': connection.execute('DROP TABLE model_control_events')
        else: connection.execute('PRAGMA user_version=999')
        connection.commit(); connection.close()
    target, backup = tmp_path/'new', tmp_path/'backup'
    before = digest_tree(source)
    with pytest.raises(MigrationError): migrate(source, target, backup, dry_run=False)
    assert not target.exists() and not backup.exists() and digest_tree(source) == before


def test_current_model_query_is_versioned_without_identity_repair(monkeypatch):
    monkeypatch.setattr(api_server.model_host, 'model_loaded', False)
    monkeypatch.setattr(api_server, '_pipeline_model_is_prepared', lambda: False)
    monkeypatch.setattr(api_server, '_active_execution_model_identity', lambda: pytest.fail('query recomputed identity'))
    result = TestClient(api_server.app).get('/api/models/current').json()
    assert result['schema_version'] == 2 and result['revision'] >= 0
    assert result['identity']['boot_id'] and result['loaded'] is False and not result['runnable']


def test_static_frontend_serves_assets_and_keeps_api_unknown_paths_closed(tmp_path, monkeypatch):
    (tmp_path/'index.html').write_text('<html>application</html>', encoding='utf-8')
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/app.js').write_text('window.app=true', encoding='utf-8')
    monkeypatch.setattr(api_server, '_resolve_frontend_dist', lambda: tmp_path)
    client = TestClient(api_server.app)
    assert client.get('/').text == '<html>application</html>'
    assert client.get('/assets/app.js').text == 'window.app=true'
    assert client.get('/api/no-such-endpoint').status_code == 404
    assert client.get('/v1/no-such-endpoint').status_code == 404
    assert client.get('/assets/%2e%2e/%2e%2e/secret').status_code == 404


def test_implicit_client_sessions_persist_and_remain_isolated():
    import local_store
    owner=ConversationContextService(load_history=local_store.load_local_conversation,
        persist_turn=local_store.save_local_conversation_turn)
    session=owner.resolve_session(client_id="tui:" + "a"*64)
    assert local_store._valid_session_id(session)==session
    assert owner.commit_turn(session,"question","answer")
    assert [m["content"] for m in local_store.load_local_conversation(session)]==["question","answer"]
    assert owner.resolve_session(client_id="web:" + "a"*64)!=session


def test_application_shutdown_requests_owned_listener_and_waits_for_cleanup(monkeypatch):
    from types import SimpleNamespace
    import app_bootstrap
    server=SimpleNamespace(should_exit=False)
    monkeypatch.setattr(api_server,"_uvicorn_servers",[server])
    stopped=threading.Event()
    def serve():
        while not server.should_exit:
            stopped.wait(0.001)
        stopped.set()
    thread=threading.Thread(target=serve)
    thread.start()
    try:
        app_bootstrap._shutdown_owned_backend(thread,timeout=2)
        assert stopped.is_set() and not thread.is_alive()
    finally:
        server.should_exit=True
        thread.join(2)


def test_application_shutdown_preserves_pending_native_cleanup(monkeypatch):
    from types import SimpleNamespace
    import app_bootstrap
    monkeypatch.setattr(api_server,"_uvicorn_servers",[])
    thread=SimpleNamespace(join=lambda **kw:None,is_alive=lambda:True)
    with pytest.raises(RuntimeError,match="application_shutdown_cleanup_pending"):
        app_bootstrap._shutdown_owned_backend(thread,timeout=0)


@pytest.mark.parametrize("fails",[False,True])
def test_tui_exit_stops_its_supervisor_after_normal_or_failed_ui(monkeypatch,fails):
    from types import SimpleNamespace
    import tui_textual
    stopped=[]
    def run():
        if fails: raise RuntimeError("ui-failure")
    monkeypatch.setattr(tui_textual,"KoakumaApp",lambda *a,**kw:SimpleNamespace(run=run))
    supervisor=SimpleNamespace(stop=lambda:stopped.append(True))
    if fails:
        with pytest.raises(RuntimeError,match="ui-failure"):
            tui_textual.run(supervisor=supervisor)
    else:
        assert tui_textual.run(supervisor=supervisor)==0
    assert stopped==[True]


@pytest.mark.parametrize("status,payload,expected",[
    (200,b'{"ready":true}',True),(200,b'{"status":"ok"}',False),
    (200,b'{"ready":false}',False),(200,b'{"ready":"true"}',False),
    (503,b'{"ready":true}',False),(200,b'invalid',False),
])
def test_application_opens_ui_only_after_readiness(status,payload,expected):
    from types import SimpleNamespace
    import app_bootstrap
    assert app_bootstrap._application_ready(SimpleNamespace(status=status,read=lambda:payload)) is expected
