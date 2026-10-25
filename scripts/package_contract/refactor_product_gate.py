"""R6 development product checks for the v2 linear execution contract.

These pure checks bind a real reply to the selected deployment and durable request.
They supplement OS/native observations; release cleanliness/signing is checked by
product_relay_gate and is never inferred here.
"""
from __future__ import annotations

import math


class ProductGateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProductGateError(message)


def snapshot(value, name):
    require(isinstance(value, dict), f'{name}: expected object')
    require(type(value.get('schema_version')) is int and value['schema_version'] == 2,
            f'{name}: requires client contract v2')
    require(type(value.get('revision')) is int and value['revision'] >= 0,
            f'{name}: missing revision')
    require(value.get('validity') == 'valid', f'{name}: observation is not valid')
    require(isinstance(value.get('identity'), dict) and bool(value['identity']),
            f'{name}: missing identity')
    require(type(value.get('observed_at')) in (int, float)
            and math.isfinite(value['observed_at']) and value['observed_at'] > 0,
            f'{name}: missing observation time')
    return value


def prepared(state, *, minimum_nodes=2):
    snapshot(state, 'client state')
    require(state.get('runnable') is True, 'deployment is not runnable')
    participation = state.get('participation') or {}
    require(participation.get('desired') == 'distributed' and participation.get('phase') == 'stable',
            'participation is not stable distributed')
    operation = snapshot(state.get('operation'), 'operation')
    require(operation.get('status') == 'ready' and operation.get('loaded_ready') is True,
            'operation is not natively ready')
    require(operation.get('runnable') is True, 'operation is not runnable')
    deployment = snapshot(state.get('deployment'), 'deployment')
    require(deployment.get('runnable') is True, 'published deployment is not runnable')
    require(deployment['identity'].get('deployment_id') == operation.get('config_id'),
            'operation and deployment differ')
    plan = operation.get('plan') or {}
    require(plan.get('admitted') is True, 'plan was not admitted')
    total = plan.get('total_layers')
    require(type(total) is int and total > 1, 'missing decoder layer count')
    assignments = plan.get('assignments') or []
    require(len(assignments) >= minimum_nodes, 'too few assigned physical nodes')
    ids, cursor = set(), 0
    for item in assignments:
        node, start, end = item.get('node_id'), item.get('start_layer'), item.get('end_layer')
        require(isinstance(node, str) and node and node not in ids, 'duplicate/missing node')
        require(type(start) is int and type(end) is int and start == cursor and start < end <= total,
                'segments must be continuous ordered ranges')
        require(end-start < total, 'a segment contains the full model')
        require(item.get('engine') in {'pytorch', 'llama_cpp'}, 'unknown execution engine')
        ids.add(node); cursor = end
    require(cursor == total and assignments[0]['node_id'] == 'master', 'missing master prefix or tail')
    return operation


def completed(response, before, after, history, *, previous_request_ids=()):
    operation = prepared(before)
    current = prepared(after)
    identity = lambda op: tuple(op.get(k) for k in ('operation_id', 'config_id', 'generation'))
    require(identity(operation) == identity(current), 'selection changed during request')
    require(bool(operation.get('operation_id')) and bool(operation.get('config_id'))
            and type(operation.get('generation')) is int and operation['generation'] > 0,
            'selection lacks operation/config/integer generation')
    require(isinstance(response, dict) and not response.get('error') and not response.get('cancelled'),
            'request failed or was cancelled')
    text = response.get('content', response.get('response', ''))
    require(isinstance(text, str) and bool(text.strip()), 'empty response')
    metrics = response.get('metrics') or {}
    require(metrics.get('distributed_used') is True and metrics.get('fallback') is False,
            'distributed execution / fallback evidence missing')
    require(metrics.get('execution_mode') == 'distributed_pipeline', 'response did not use linear relay')
    require(metrics.get('stop_reason') in {'eos_token', 'text_stop'}, 'request did not stop normally')
    require(type(metrics.get('model_generation')) is int, 'response lacks integer generation')
    require(type(current.get('generation')) is int, 'current selection lacks integer generation')
    require(tuple(metrics.get(k) for k in ('model_operation_id', 'model_config_id', 'model_generation'))
            == identity(operation), 'response belongs to another model selection')
    assignments = operation['plan']['assignments']
    require(metrics.get('workers_used') == [a['node_id'] for a in assignments[1:]],
            'actual workers differ from selection')
    require(metrics.get('layer_segments') == [[a['start_layer'], a['end_layer']] for a in assignments],
            'actual ranges differ from selection')
    request = snapshot(metrics.get('request'), 'response request')
    key = request.get('request_id')
    require(isinstance(key, str) and bool(key) and key not in previous_request_ids,
            'reply reused an earlier request')
    snapshot(history, 'request history')
    matches = [r for r in history.get('requests', []) if r.get('request_id') == key]
    require(len(matches) == 1, 'durable history lacks the current request')
    durable = snapshot(matches[0], 'durable request')
    for record in (request, durable):
        require(record.get('deployment_id') == operation['config_id'], 'request deployment mismatch')
        require(record.get('terminal') == 'completed' and record.get('outcome') == 'completed'
                and record.get('execution_state') == 'stopped', 'request/native cleanup did not complete')
        require([a['node_id'] for a in record.get('assignments', [])] == [a['node_id'] for a in assignments],
                'request assignments differ from selected nodes')
    require(request.get('attempt_id') == durable.get('attempt_id')
            and request.get('plan_digest') == durable.get('plan_digest')
            and bool(request.get('attempt_id')) and bool(request.get('plan_digest')), 'request identity mismatch')
    return {'request_id':key, 'operation_id':operation['operation_id'], 'config_id':operation['config_id'],
            'generation':operation['generation'], 'workers':metrics['workers_used'],
            'ranges':metrics['layer_segments'], 'stop_reason':metrics['stop_reason'], 'fallback':False}
