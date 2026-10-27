"""R6 regressions: real shared control schemas and rejection of misleading evidence."""
from copy import deepcopy
import sqlite3
import time
import pytest
from qlh_core.adapters.state_migration import migrate, MigrationError
from qlh_core.contracts.snapshots import envelope
from scripts.package_contract.refactor_product_gate import completed, ProductGateError
from tests.test_refactor_client_cutover import migration_source, digest_tree


@pytest.mark.parametrize('version', [5,7])
def test_migrate_existing_shared_control_version_preserves_user_data(tmp_path, version):
    source=migration_source(tmp_path)
    # Initialize the actual Python shared schema, then retain control-svc's version.
    from local_store import initialize_local_store, bind_local_store
    with bind_local_store(source/'qlh-control.sqlite3'):
        connection=sqlite3.connect(source/'qlh-control.sqlite3')
        connection.execute('DROP TABLE sessions'); connection.commit(); connection.close()
        initialize_local_store()
    connection=sqlite3.connect(source/'qlh-control.sqlite3')
    connection.execute("INSERT INTO sessions VALUES ('saved','Real history','now','now',1)")
    connection.execute("INSERT INTO session_messages(session_id,role,content) VALUES ('saved','user','retained')")
    connection.execute(f'PRAGMA user_version={version}'); connection.commit();connection.close()
    before=digest_tree(source); target=tmp_path/'target'; backup=tmp_path/'backup'
    migrate(source,target,backup,dry_run=False)
    assert digest_tree(source)==before
    with sqlite3.connect(target/'qlh-control.sqlite3') as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0]==version
        assert connection.execute('SELECT content FROM session_messages').fetchone()[0]=='retained'
    assert all(digest_tree(backup)[name]==value for name,value in before.items())


@pytest.mark.parametrize('damage', ['missing_tables','wrong_column','future_version'])
def test_shared_schema_rejects_wrong_structure_before_any_copy(tmp_path, damage):
    source=migration_source(tmp_path)
    with sqlite3.connect(source/'qlh-control.sqlite3') as connection:
        connection.execute('PRAGMA user_version=7' if damage!='future_version' else 'PRAGMA user_version=999')
        if damage=='wrong_column': connection.execute('ALTER TABLE sessions RENAME COLUMN title TO obsolete')
    before=digest_tree(source); target=tmp_path/'target'; backup=tmp_path/'backup'
    with pytest.raises(MigrationError,match='migration_database_schema_unknown'):
        migrate(source,target,backup,dry_run=False)
    assert digest_tree(source)==before and not target.exists() and not backup.exists()


def env(payload, **identity):
    return envelope(payload,revision=4,identity=identity or {'boot_id':'boot'},observed_at=10)


@pytest.fixture
def evidence():
    assignments=[{'node_id':'master','start_layer':0,'end_layer':4,'engine':'pytorch'},
                 {'node_id':'surface','start_layer':4,'end_layer':32,'engine':'llama_cpp'}]
    operation=env({'operation_id':'op','config_id':'config','generation':3,'status':'ready','loaded_ready':True,
                   'runnable':True,'plan':{'admitted':True,'total_layers':32,'assignments':assignments}})
    state=env({'operation':operation,'runnable':True,'participation':{'desired':'distributed','phase':'stable'},
               'deployment':env({'runnable':True},deployment_id='config')})
    request=env({'request_id':'req','attempt_id':'attempt','plan_digest':'digest','deployment_id':'config',
                 'outcome':'completed','terminal':'completed','execution_state':'stopped',
                 'assignments':[{'node_id':a['node_id']} for a in assignments]})
    response={'content':'2','metrics':{'execution_mode':'distributed_pipeline','distributed_used':True,'fallback':False,
        'stop_reason':'eos_token','model_operation_id':'op','model_config_id':'config','model_generation':3,
        'workers_used':['surface'],'layer_segments':[[0,4],[4,32]],'request':request}}
    return response,state,deepcopy(state),env({'requests':[deepcopy(request)]})


def test_linear_product_evidence_binds_exact_request_and_native_cleanup(evidence):
    assert completed(*evidence)['request_id']=='req'


@pytest.mark.parametrize('damage', ['fallback','missing_fallback','length_stop','old_selection','different_worker',
    'wrong_range','different_request','draining','legacy_snapshot','not_runnable','bool_generation'])
def test_linear_product_evidence_refuses_missing_or_stale_facts(evidence, damage):
    reply,before,after,history=evidence
    if damage=='fallback':reply['metrics']['fallback']=True
    elif damage=='missing_fallback':reply['metrics'].pop('fallback')
    elif damage=='length_stop':reply['metrics']['stop_reason']='max_new_tokens'
    elif damage=='old_selection':after['operation']['config_id']='old'
    elif damage=='different_worker':reply['metrics']['workers_used']=['other']
    elif damage=='wrong_range':reply['metrics']['layer_segments']=[[0,32]]
    elif damage=='different_request':history['requests'][0]['request_id']='previous'
    elif damage=='draining':history['requests'][0]['execution_state']='draining'
    elif damage=='legacy_snapshot':before['schema_version']=1
    elif damage=='not_runnable':before['runnable']=False
    else:before['operation']['generation']=True
    with pytest.raises(ProductGateError):completed(*evidence)


def test_linear_product_evidence_does_not_accept_old_success(evidence):
    with pytest.raises(ProductGateError,match='earlier request'):
        completed(*evidence,previous_request_ids={'req'})


def test_native_llama_assignment_is_observable_without_pytorch_layer_field(monkeypatch):
    from types import SimpleNamespace
    from llama_engine import LlamaCppEngine
    from qlh_core.planning.load_spec import assignment_load_spec
    from qlh_core.adapters.runtime_observation import observe_runtime
    engine=LlamaCppEngine()
    assert engine.layer_range is None
    engine._loaded=True;engine._model=SimpleNamespace(close=lambda:None)
    engine._pipeline_descriptor={'assignment_layer_range':[0,4]}
    policy=assignment_load_spec({'has_embedding':True},artifact_format='gguf')
    assignment={'load_spec':policy.to_dict(),'identity':{'layer_range':[0,4],'artifact_sha256':'a'*64,'load_spec_digest':policy.digest}}
    observation={**policy.to_dict(),'artifact_sha256':'a'*64,'materialized_bytes':1024}
    monkeypatch.setattr(engine,'deployment_materialization_observation',lambda:observation)
    assert observe_runtime(engine,assignment)['layer_range']==[0,4]
    engine._loaded=False
    assert engine.layer_range is None


@pytest.mark.parametrize('scope', ['state', 'request', 'history'])
@pytest.mark.parametrize('timestamp', [float('inf'), float('nan')])
def test_product_evidence_rejects_nonfinite_observation_time(evidence, scope, timestamp):
    reply, before, after, history = evidence
    target = before if scope == 'state' else reply['metrics']['request'] if scope == 'request' else history
    target['observed_at'] = timestamp
    with pytest.raises(ProductGateError):
        completed(*evidence)


@pytest.mark.parametrize('scope', ['response', 'current'])
def test_product_evidence_requires_integer_generation_on_every_surface(evidence, scope):
    reply, before, after, history = evidence
    if scope == 'response':
        reply['metrics']['model_generation'] = 3.0
    else:
        after['operation']['generation'] = 3.0
    with pytest.raises(ProductGateError):
        completed(*evidence)


def test_llama_vocab_only_cleanup_resets_the_existing_native_owner():
    from types import SimpleNamespace
    from llama_engine import LlamaCppEngine
    calls = []
    engine = LlamaCppEngine()
    engine._model = SimpleNamespace()  # vocab-only binding owns no context
    engine._keep_head = SimpleNamespace(reset=lambda: calls.append('reset'))
    engine._kv_used = 7
    engine.clear_pipeline_kv()
    assert calls == ['reset'] and engine._kv_used == 0
    engine._keep_head.reset = lambda: (_ for _ in ()).throw(RuntimeError('native reset failed'))
    with pytest.raises(RuntimeError, match='native reset failed'):
        engine.clear_pipeline_kv()
    engine._keep_head = None
    with pytest.raises(RuntimeError, match='pipeline_kv_owner_unavailable'):
        engine.clear_pipeline_kv()
    engine._model = None


@pytest.mark.parametrize("damage", [None, "parameter_dtype", "buffer_device", "persistent_buffer_dtype"])
def test_real_fp16_layer_owner_preserves_native_float32_rope(tiny_supply, damage):
    import torch
    from model_module import ModelManager
    from qlh_core.adapters.runtime_observation import observe_runtime
    from qlh_core.contracts.load import RuntimeLoadSpec
    manifest, directory = tiny_supply.segments[0]
    policy = RuntimeLoadSpec(device="cpu", dtype="float16", quant_type="fp16",
        context_size=64, batch_size=16, kv_device="cpu", kv_dtype="float16",
        kv_policy="dynamic", components=("layers", "final_norm", "embedding"),
        materialization="safetensors_key_filtered")
    manager = ModelManager()
    try:
        manager.prepare_pipeline_model(model_id="tiny-qwen2", model_path=str(directory),
            quant_type="fp16", layer_range=(0, 2), model_sha256=manifest["artifact_sha256"])
        manager.load_layer_range(0, 2, has_embedding=True, has_lm_head=False,
            model_path=str(directory), quant_type="fp16", total_layers=4,
            model_id="tiny-qwen2", load_spec=policy.to_dict())
        assert all(p.dtype == torch.float16 for p in manager.model.parameters())
        assert any(b.dtype == torch.float32 for b in manager.model.buffers())
        assignment = {"load_spec": policy.to_dict(), "identity": {
            "layer_range": [0, 2], "artifact_sha256": manifest["artifact_sha256"],
            "load_spec_digest": policy.digest}}
        if damage == "parameter_dtype":
            parameter = next(manager.model.parameters())
            parameter.data = parameter.data.float()
        elif damage == "persistent_buffer_dtype":
            manager.model.register_buffer("persistent_weight", torch.ones(1, dtype=torch.float32))
        elif damage == "buffer_device":
            module = next(m for m in manager.model.modules() if m._buffers)
            name = next(iter(module._buffers))
            module._buffers[name] = module._buffers[name].to("meta")
        if damage:
            with pytest.raises(ValueError, match="deployment_local_policy_mismatch"):
                observe_runtime(manager, assignment)
        else:
            receipt = observe_runtime(manager, assignment)
            tensors = list(manager.model.parameters()) + list(manager.model.buffers())
            assert receipt["materialized_bytes"] == sum(t.numel() * t.element_size() for t in tensors)
            assert receipt["load_spec"] == policy.to_dict()
    finally:
        manager.unload_model()


from tests.test_safetensors_supply_worker import tiny_supply


def test_native_retained_components_are_charged_on_middle_and_each_boundary():
    from qlh_core.planning.capacity import solve_pipeline_capacity
    source = {"model_id": "native", "model_sha256": "a" * 64, "model_type": "qwen2",
        "total_layers": 6, "layer_weight_bytes": [10] * 6,
        "component_weight_bytes": {"embedding": 20, "final_norm": 3, "lm_head": 30, "other": 7}}
    nodes = [{"node_id": name, "role": "master" if i == 0 else "client", "score": 10-i,
        "capacity_bytes": 1000, "reserve_bytes": 5, "retained_component_bytes": 60,
        "layer_ranges": [[2*i, 2*i+2]]} for i, name in enumerate(("master", "pc", "android"))]
    plan = solve_pipeline_capacity(source, nodes, safety_margin=1, require_distributed=True,
        prefer_all_workers=True)
    assert plan["admitted"]
    assert [a["raw_weight_bytes"] for a in plan["assignments"]] == [80, 80, 80]
    assert [a["required_bytes"] for a in plan["assignments"]] == [85, 85, 85]
    assert all(a["retained_component_bytes"] == 60 for a in plan["assignments"])
    nodes[1]["capacity_bytes"] = 84
    assert not solve_pipeline_capacity(source, nodes, safety_margin=1,
        require_distributed=True, prefer_all_workers=True)["admitted"]


@pytest.mark.parametrize("quote", [-1, True, "60"])
def test_native_component_quote_requires_nonnegative_integer(quote):
    from qlh_core.planning.capacity import solve_pipeline_capacity, PipelineCapacityError
    from tests.test_refactor_capacity_load_spec import descriptor, node
    with pytest.raises(PipelineCapacityError, match="retained_component_bytes"):
        solve_pipeline_capacity(descriptor(), [node(retained_component_bytes=quote)])


def test_supply_native_component_quote_reaches_fresh_final_admission(fresh_case):
    from tests.test_pipeline_capacity_freshness import plan
    case = fresh_case
    for assignment in case.c._operations[case.op]["plan"]["assignments"]:
        assignment["retained_component_bytes"] = 500
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        result = plan(case, evidence)
    assert result["admitted"], result
    worker = next(a for a in result["assignments"] if a["node_id"] == "worker")
    # GGUF 段（`llama_cpp`）按**原生量化宽度**计费，`runtime_multiplier` 只适用于
    # **未投影**描述符（见 `docs/已知问题记录.md` 的「×2 倍率误用」）⇒ 倍率 1.0：
    # 账 = (2 层 × 100 + retained 500) × 1.0 + reserve 100 = 800。
    # 该断言同时锁住两件事：native 组件的 500 B 报价确实抵达最终准入，且没有被
    # 退化成 ×2（那会是 1500，见 test_pipeline_capacity_freshness 的同源说明）。
    assert worker["retained_component_bytes"] == 500 and worker["required_bytes"] == 800
    case.ram["worker"] = 799
    with case.c.prepare_guard(case.op):
        result = plan(case, case.c.prepare_capacity_evidence(case.op))
    assert not result["admitted"]


def test_supply_plan_carries_exact_native_non_block_quote():
    from model_supply import plan_segment_supply
    from tests.test_model_supply import descriptor, node
    plan = plan_segment_supply({"llama_cpp": descriptor()},
        [node("master"), node("middle"), node("tail")], master_engine="llama_cpp", alignment=2)
    assert plan["admitted"]
    assert all(a["retained_component_bytes"] == 12 for a in plan["assignments"])


def test_local_assignment_uses_inspected_artifact_sha_when_caller_omits_it(monkeypatch):
    from types import SimpleNamespace
    from model_host import ModelHost
    host = ModelHost.__new__(ModelHost)
    transaction = SimpleNamespace(model_sha256=None, descriptor={"model_sha256": "a" * 64},
        manager=SimpleNamespace(engine_type="pytorch"))
    monkeypatch.setattr(host, "_require_pipeline_candidate", lambda key: transaction)
    frozen = host.freeze_pipeline_candidate_assignment("candidate",
        {"start_layer": 0, "end_layer": 2, "has_embedding": True}, config_id="config", generation=1)
    assert frozen["identity"]["artifact_sha256"] == "a" * 64


@pytest.mark.parametrize("explicit_engine", [False, True])
@pytest.mark.parametrize("legacy_path", [None, "stale-legacy.gguf"])
def test_native_primary_freeze_uses_candidate_artifact(monkeypatch, explicit_engine, legacy_path):
    from types import SimpleNamespace
    from model_host import ModelHost
    calls = []
    manager = SimpleNamespace(engine_type="llama_cpp",
        _find_layer_artifact=lambda start, end: calls.append((start, end)) or "supplied-head.gguf",
        _pipeline_file_sha256=lambda path: calls.append(path) or "b" * 64)
    host = SimpleNamespace(_require_pipeline_candidate=lambda key: SimpleNamespace(
        manager=manager, model_sha256="a" * 64, descriptor={"hidden_size": 4096}))
    if legacy_path is None:
        monkeypatch.delenv("QLH_LAYER_GGUF", raising=False)
    else:
        monkeypatch.setenv("QLH_LAYER_GGUF", legacy_path)
    assignment = {"start_layer": 0, "end_layer": 24, "has_embedding": True}
    if explicit_engine:
        assignment["engine"] = "llama_cpp"
    frozen = ModelHost.freeze_pipeline_candidate_assignment(
        host, "candidate", assignment, config_id="config", generation=1)
    assert calls == [(0, 24), "supplied-head.gguf"]
    assert frozen["identity"]["artifact_sha256"] == "b" * 64
    assert frozen["identity"]["stage_model_identity"]["sha256"] == "b" * 64
    assert frozen["load_spec"]["materialization"] == "gguf_segment"


def test_native_primary_freeze_rejects_missing_candidate_despite_legacy_env(monkeypatch):
    from types import SimpleNamespace
    from model_host import ModelHost
    monkeypatch.setenv("QLH_LAYER_GGUF", "legacy.gguf")
    host = SimpleNamespace(_require_pipeline_candidate=lambda key: SimpleNamespace(
        manager=SimpleNamespace(engine_type="llama_cpp", _find_layer_artifact=lambda start, end: ""),
        model_sha256="a" * 64, descriptor={}))
    with pytest.raises(ValueError, match="deployment_local_artifact_missing"):
        ModelHost.freeze_pipeline_candidate_assignment(host, "candidate",
            {"start_layer": 0, "end_layer": 24}, config_id="config", generation=1)


@pytest.mark.parametrize("changed", [False, True])
def test_additional_native_freeze_fallback_shares_scheduler_digest(tmp_path, monkeypatch, changed):
    import hashlib
    import os
    from types import SimpleNamespace
    import config
    import model_supply_digest_cache as digests
    import scheduler
    from model_host import ModelHost

    artifact = tmp_path / "middle.gguf"
    original = b"cpu middle 8-24"
    replacement = b"new middle 8-24"
    artifact.write_bytes(original)
    monkeypatch.setenv("QLH_LAYER_GGUF", str(artifact))
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(digests, "_STORES", {})
    reads = []
    hash_file = digests._hash_file

    def observed_hash(path, cancelled=None):
        reads.append(path)
        return hash_file(path, cancelled)

    monkeypatch.setattr(digests, "_hash_file", observed_hash)
    assert scheduler._local_cpu_artifact_sha256() == hashlib.sha256(original).hexdigest()
    if changed:
        # ★ 2026-10-27：`original` 与 `replacement` **都是 15 字节**，且下面把
        #   mtime 拨回原值 ⇒ size 与 mtime 都不体现差异，只有 ChangeTime 能。
        #   而紧随其后的 write 很容易落在**同一个时间戳 tick** 里（Windows
        #   ChangeTime 是 FILETIME，实际分辨率取决于卷），于是
        #   `file_stamp` 完全相等 ⇒ 缓存层认为"文件没变" ⇒ 冻结出来的
        #   `artifact_sha256` 还是旧摘要 ⇒ 与 `expected` 不等。
        #
        #   症状是随机失败（Heisenbug：加一行 print 就过）。这里先跨 tick 再写，
        #   并校验 stamp 真的变了；不变就重试，2 秒仍不变则显式失败说明前置条件
        #   不成立，而不是让 `artifact_sha256` 的断言去背这个锅。
        before = artifact.stat()
        deadline = time.monotonic() + 2.0
        while True:
            original_stamp = digests.file_stamp(artifact)
            time.sleep(0.005)
            artifact.write_bytes(replacement)
            os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
            if digests.file_stamp(artifact) != original_stamp:
                break
            if time.monotonic() > deadline:
                pytest.fail(
                    "ChangeTime 在改写后没有推进（重试 2 秒）：这次改写对 "
                    "file_stamp 不可见，测试的前置条件不成立"
                )
    host = SimpleNamespace(_require_pipeline_candidate=lambda key: SimpleNamespace(
        manager=SimpleNamespace(engine_type="pytorch"), model_sha256="a" * 64, descriptor={}))
    assignment = {"engine": "llama_cpp", "executor_id": "cpu", "start_layer": 8, "end_layer": 24}
    expected = hashlib.sha256(replacement if changed else original).hexdigest()
    for _ in range(2):
        frozen = ModelHost.freeze_pipeline_candidate_assignment(
            host, "candidate", assignment, config_id="config", generation=1)
        assert frozen["identity"]["artifact_sha256"] == expected
        assert frozen["identity"]["stage_model_identity"]["sha256"] == expected
    assert len(reads) == (2 if changed else 1)


def test_additional_native_executor_keeps_selected_identity_with_pytorch_primary(monkeypatch):
    from types import SimpleNamespace
    from model_host import ModelHost
    monkeypatch.delenv("QLH_LAYER_GGUF", raising=False)
    host = SimpleNamespace(_require_pipeline_candidate=lambda key: SimpleNamespace(
        manager=SimpleNamespace(engine_type="pytorch"), model_sha256="a" * 64, descriptor={}))
    assignment = {"engine": "llama_cpp", "executor_id": "cpu", "start_layer": 8, "end_layer": 24,
                  "layer_artifact": {"artifact_sha256": "b" * 64}}
    frozen = ModelHost.freeze_pipeline_candidate_assignment(
        host, "candidate", assignment, config_id="config", generation=1)
    assert frozen["identity"]["artifact_sha256"] == "b" * 64
    assert frozen["load_spec"]["materialization"] == "gguf_segment"


from tests.test_pipeline_capacity_freshness import fresh_case


def test_failed_materialization_releases_tensor_owners_before_cleanup_returns(monkeypatch, tmp_path):
    import gc
    import traceback
    import weakref
    from types import SimpleNamespace
    import torch
    import model_host
    from qlh_core.contracts.load import RuntimeLoadSpec
    directory = tmp_path / "model"
    directory.mkdir()
    refs, released = [], []
    manager = SimpleNamespace(engine_type="pytorch", is_loaded=False, layer_range=(0, 2),
                              quant_type="fp16", model=None)
    manager.prepare_pipeline_model = lambda **_kw: {"model_sha256": "a" * 64}
    def load(**_kw):
        manager.model = torch.nn.Linear(2, 2)  # Real wrong-dtype tensor owner.
        refs.append(weakref.ref(next(manager.model.parameters())))
        manager.is_loaded = True
        return {"success": True}
    def unload():
        manager.model = None
        manager.is_loaded = False
        gc.collect()
        released.append(refs[0]() is None)
    manager.load_layer_range, manager.unload_model = load, unload
    monkeypatch.setattr(model_host._LazyModelManager, "_get_instance", lambda _self: manager)
    host = model_host.ModelHost()
    host.prepare_pipeline_candidate("candidate", "tiny", str(directory), quant_type="fp16")
    policy = RuntimeLoadSpec(device="cpu", dtype="float16", quant_type="fp16",
        context_size=64, kv_device="cpu", kv_dtype="float16", kv_policy="dynamic",
        components=("layers", "embedding"), materialization="safetensors_key_filtered")
    assignment = {"load_spec": policy.to_dict(), "identity": {"layer_range": [0, 2],
        "artifact_sha256": "a" * 64, "load_spec_digest": policy.digest}}
    with pytest.raises(ValueError, match="deployment_local_policy_mismatch") as caught:
        host.materialize_pipeline_candidate("candidate", 0, 2, True, False,
                                            load_assignment=assignment)
    assert released == [True], "failed observer traceback retained materialized weight tensors during unload"
    assert any(frame.name == "observe_runtime" for frame in traceback.extract_tb(caught.value.__traceback__))
