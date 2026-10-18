"""Capacity evidence stays with the supply owner across waits and intent projection."""
import asyncio
from copy import deepcopy
import json
import time

import pytest

from tests.test_model_supply import startup_pool_case  # noqa: F401
from tests.test_model_intent import setup, finish_tasks


def test_online_capacity_wait_keeps_cause_and_is_cleared_by_new_plan(startup_pool_case):
    case = startup_pool_case
    case.budgets = {node_id: 45 for node_id in ('master', 'worker', 'android')}
    case.release_source.set()
    runner = case.start()
    assert case.waiting.wait(2)
    waiting = case.c.get(case.op)
    assert waiting['status'] == 'waiting_nodes'
    assert waiting['reason_code'] == 'pipeline_distributed_capacity_insufficient'
    diagnostic = waiting['capacity_diagnostic']
    assert diagnostic['reason_code'] == waiting['reason_code']
    assert diagnostic['scope'] == 'candidate_plan'
    assert diagnostic['pending_node_ids'] == []
    assert any(node['deficit_bytes'] > 0 for node in diagnostic['nodes'])
    assert all(node['available_bytes'] == 45 for node in diagnostic['nodes'])
    assert waiting['plan'] is None and not waiting.get('loaded_ready')
    persisted = json.loads((case.c.root / case.op / 'operation.json').read_text(encoding='utf-8'))
    assert persisted['capacity_diagnostic'] == diagnostic
    case.join()
    runner.join(3)
    assert not runner.is_alive()
    result = case.c.get(case.op)
    assert result['status'] == 'available'
    assert result['capacity_diagnostic'] is None and result['reason_code'] == 'SUPPLY_ASSETS_AVAILABLE'
    assert len(result['plan']['assignments']) == 3


def test_actual_join_wait_never_claims_an_unmeasured_memory_deficit(startup_pool_case):
    case = startup_pool_case
    runner = case.start()
    case.join(acknowledge=False)
    case.release_source.set()
    assert case.waiting.wait(2)
    value = case.c.get(case.op)
    assert value['reason_code'] == 'SUPPLY_WAITING_NODES'
    assert value.get('capacity_diagnostic') is None
    assert not case.capacity_pools and not case.build_calls
    case.c.cancel(case.op)
    runner.join(3)
    assert not runner.is_alive()
    assert case.c.get(case.op)['status'] == 'cancelled'


def test_capacity_deadline_keeps_exact_evidence_instead_of_join_message(startup_pool_case):
    case = startup_pool_case
    case.c._operations[case.op]['intent']['timeout_seconds'] = 1
    case.budgets = {'master': 45, 'worker': 45}
    case.release_source.set()
    runner = case.start()
    assert case.waiting.wait(2)
    waiting = case.c.get(case.op)
    runner.join(3)
    assert not runner.is_alive()
    result = case.c.get(case.op)
    assert result['status'] == 'failed'
    assert result['reason_code'] == waiting['reason_code'] == 'pipeline_distributed_capacity_insufficient'
    assert result['capacity_diagnostic']['nodes'] == waiting['capacity_diagnostic']['nodes']
    assert not case.build_calls


def test_capacity_shortage_and_late_pending_hello_keep_both_facts(startup_pool_case):
    case = startup_pool_case
    case.budgets = {'master': 45, 'worker': 45}
    case.release_source.set()
    runner = case.start()
    assert case.waiting.wait(2)
    case.join(acknowledge=False)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        value = case.c.get(case.op)
        if value.get('capacity_diagnostic', {}).get('pending_node_ids') == ['android']:
            break
        case.c._events[case.op].wait(0.02)
    else:
        pytest.fail('pending hello was not projected alongside the capacity cause')
    assert value['reason_code'] == 'pipeline_distributed_capacity_insufficient'
    assert not case.build_calls
    case.c.cancel(case.op)
    runner.join(3)
    assert not runner.is_alive()


def _diagnostic():
    return {'reason_code': 'pipeline_distributed_capacity_insufficient', 'reason': '当前节点容量不足',
            'scope': 'candidate_plan', 'nodes': [{'node_id': 'tail', 'resource': 'ram',
            'available_bytes': 45, 'required_bytes': 60, 'deficit_bytes': 15, 'layer_range': [3, 6]}]}


def test_intent_projects_supply_diagnostic_without_duplicating_authority(tmp_path):
    async def scenario():
        async def prepare(_):
            pytest.fail('rejected supply must not prepare')
        c, _, supply = setup(tmp_path, prepare)
        try:
            value = await c.create({'model_id': 'fixture', 'timeout_seconds': 30})
            op = value['operation_id']
            diagnostic = _diagnostic()
            supply.operations[op].update(reason_code=diagnostic['reason_code'], reason=diagnostic['reason'],
                                         capacity_diagnostic=deepcopy(diagnostic))
            result = c.get(op)
            assert result['capacity_diagnostic'] == diagnostic
            result['capacity_diagnostic']['nodes'][0]['available_bytes'] = 9999
            assert c.get(op)['capacity_diagnostic'] == diagnostic
            assert 'capacity_diagnostic' not in c._records[op]
            assert not result['loaded_ready'] and result['config_id'] == ''
        finally:
            await finish_tasks(c)
    asyncio.run(scenario())


def test_intent_deadline_keeps_capacity_cause_when_it_preempts_supply(tmp_path):
    async def scenario():
        async def prepare(_):
            pytest.fail('rejected supply must not prepare')
        c, _, supply = setup(tmp_path, prepare)
        value = await c.create({'model_id': 'fixture', 'timeout_seconds': 0})
        op = value['operation_id']
        diagnostic = _diagnostic()
        supply.operations[op].update(reason_code=diagnostic['reason_code'], reason=diagnostic['reason'],
                                     capacity_diagnostic=deepcopy(diagnostic))
        await asyncio.wait_for(asyncio.gather(*list(c._tasks.values())), 2)
        result = c.get(op)
        assert result['status'] == 'failed'
        assert result['reason_code'] == diagnostic['reason_code']
        assert result['capacity_diagnostic'] == diagnostic
        assert '截止' in result['reason'] and not result['loaded_ready']
    asyncio.run(scenario())


def _cuda_inventory(case, *, host_available, runtime_available=10000):
    case.c._operations[case.op]['intent']['engine'] = 'pytorch'
    case.engines = {'master': ['pytorch'], 'worker': ['llama_cpp']}
    case.budgets = {'master': host_available, 'worker': 10000}
    original = case.scheduler._model_supply_worker.inventory
    def inventory(**kwargs):
        result = original(**kwargs)
        result['pytorch_runtime'].update(execution_device='cuda', available_bytes=runtime_available)
        return result
    case.scheduler._model_supply_worker.inventory = inventory


def test_host_staging_failure_reports_source_bytes_separately_from_vram(startup_pool_case, tmp_path):
    from tests.test_model_supply_runtime_capacity import _write_shard
    case = startup_pool_case
    case.runtime_source = tmp_path / 'hf'
    case.runtime_source.mkdir()
    _write_shard(case.runtime_source, {f'model.layers.{i}.weight': ('F16', [5]) for i in range(6)})
    _cuda_inventory(case, host_available=9)
    case.release_source.set()
    runner = case.start()
    runner.join(3)
    assert not runner.is_alive()
    result = case.c.get(case.op)
    assert result['status'] == 'failed'
    assert result['reason_code'] == 'SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT'
    row = result['capacity_diagnostic']['nodes'][0]
    assert row['resource'] == 'host_ram'
    assert row['node_id'] == 'master' and row['layer_range'] == [0, 1]
    assert row['available_bytes'] == 9 and row['required_bytes'] == 10 and row['deficit_bytes'] == 1
    assert not case.build_calls


def test_partial_host_ranges_are_reported_as_staging_not_artifact_coverage(startup_pool_case, tmp_path):
    from tests.test_model_supply_runtime_capacity import _write_shard
    case = startup_pool_case
    case.c._operations[case.op]['intent']['timeout_seconds'] = 1
    case.runtime_source = tmp_path / 'hf'
    case.runtime_source.mkdir()
    _write_shard(case.runtime_source, {f'model.layers.{i}.weight': ('F16', [5]) for i in range(6)})
    _cuda_inventory(case, host_available=10)
    case.budgets['worker'] = 60
    case.release_source.set()
    runner = case.start()
    assert case.waiting.wait(2)
    result = case.c.get(case.op)
    assert result['reason_code'] == 'SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT'
    diagnostic = result['capacity_diagnostic']
    assert diagnostic['scope'] == 'candidate_plan'
    assert diagnostic['origin_reason_code'] == 'pipeline_layer_range_coverage_insufficient'
    assert isinstance(diagnostic['origin_reason'], str) and diagnostic['origin_reason']
    assert result['plan'] is None and not result.get('loaded_ready')
    assert all(row['resource'] == 'host_ram' for row in diagnostic['nodes'])
    row = diagnostic['nodes'][0]
    assert row['node_id'] == 'master' and row['available_bytes'] == 10
    assert row['required_bytes'] == 10 * row['layer_range'][1]
    assert row['deficit_bytes'] == row['required_bytes'] - 10 > 0
    runner.join(3)
    assert not runner.is_alive()
    assert case.c.get(case.op)['status'] == 'failed'
    assert not case.build_calls


def test_model_intent_and_supply_api_include_same_owner_capacity_diagnostic(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from api import routes_model_intent, routes_model_supply
    async def prepare(_):
        pytest.fail('rejected supply must not prepare')
    c, scheduler, supply = setup(tmp_path, prepare)
    value = supply.create({'model_id': 'fixture'}, wait_for_nodes=True)
    op = value['operation_id']
    c._current = op
    c._records[op] = {'operation_id': op, 'status': 'queued'}
    c._persist(c._records[op])
    diagnostic = _diagnostic()
    supply.operations[op].update(reason_code=diagnostic['reason_code'], reason=diagnostic['reason'],
                                 capacity_diagnostic=deepcopy(diagnostic))
    api = SimpleNamespace(scheduler=scheduler, require_model_api_source=lambda _: None)
    monkeypatch.setattr(routes_model_intent, '_api_module', api)
    monkeypatch.setattr(routes_model_intent, '_coordinator', c)
    monkeypatch.setattr(routes_model_intent, 'require_model_intent_source', lambda *_: None)
    monkeypatch.setattr(routes_model_supply, '_api_module', api)
    app = FastAPI()
    app.add_api_route('/api/models/intent/current', routes_model_intent.current, methods=['GET'])
    app.add_api_route('/api/models/intent/{operation_id}', routes_model_intent.get, methods=['GET'])
    app.add_api_route('/api/models/supply/{operation_id}', routes_model_supply.get_supply, methods=['GET'])
    client = TestClient(app)
    for endpoint in ('/api/models/intent/current', f'/api/models/intent/{op}', f'/api/models/supply/{op}'):
        response = client.get(endpoint)
        assert response.status_code == 200
        body = response.json()
        assert body['capacity_diagnostic'] == diagnostic
        assert body['reason_code'] == diagnostic['reason_code']
        assert not body.get('loaded_ready')
        assert 'master_model_path' not in body


def test_depleted_master_cannot_be_hidden_by_two_well_provisioned_workers():
    from model_supply import plan_segment_supply
    from tests.test_model_supply import descriptor, node
    plan = plan_segment_supply({'llama_cpp': descriptor()},
                               [node('master', 1), node('worker-a', 10000), node('worker-b', 10000)],
                               master_engine='llama_cpp')
    assert not plan['admitted'] and plan['assignments'] == []
    assert plan['reason_code'] == 'SUPPLY_MASTER_CAPACITY_INSUFFICIENT'
    assert not plan['loaded_ready']
    diagnostic = plan['capacity_diagnostic']
    master = next(row for row in diagnostic['nodes'] if row['node_id'] == 'master')
    assert master['available_bytes'] == 1
    assert master['deficit_bytes'] == master['required_bytes'] - 1 > 0
    assert master['layer_range'][0] == 0 and master['layer_range'][1] < 8


def test_cancelled_capacity_wait_does_not_display_old_numeric_cause(tmp_path):
    from tests.test_model_supply import coordinator
    from tui_shared import format_model_intent
    c, op, _ = coordinator(tmp_path)
    diagnostic = _diagnostic()
    reason = "容量不足，缺口 123456789 字节"
    c._operations[op].update(status="waiting_nodes", reason_code=diagnostic["reason_code"],
                             reason=reason, capacity_diagnostic=deepcopy(diagnostic))
    value = c.cancel(op)
    assert value["reason_code"] == "SUPPLY_CANCELLED"
    assert value["reason"] == "Supply operation cancelled"
    historical = {**diagnostic, "historical": True}
    assert value["capacity_diagnostic"] == historical
    assert "historical" not in diagnostic
    c._run(op)
    assert c.get(op)["capacity_diagnostic"] == historical
    persisted = json.loads((c.root / op / "operation.json").read_text(encoding="utf-8"))
    assert persisted["reason"] == "Supply operation cancelled"
    assert persisted["capacity_diagnostic"] == historical
    assert c.cancel(op)["capacity_diagnostic"] == historical
    rendered = format_model_intent(value)
    assert "123456789" not in rendered and "容量不足" not in rendered
