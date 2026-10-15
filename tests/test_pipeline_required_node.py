"""Mandatory executors are constraints of the same pipeline solve, including replay."""
from copy import deepcopy

import pytest

from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity
from model_supply import plan_segment_supply
from tests.test_model_supply import descriptor, node


def test_extra_worker_cannot_optimize_away_a_feasible_master_head():
    desc = descriptor(layers=8, layer_bytes=10, components=0)
    records = [{**node('master', 100), 'runtime_multiplier': 2.0},
               {**node('worker-a', 10000), 'runtime_multiplier': 2.0}]
    before = deepcopy((desc, records))
    two = plan_segment_supply({'llama_cpp': desc}, records, master_engine='llama_cpp', alignment=4)
    three = plan_segment_supply({'llama_cpp': desc}, records + [{**node('worker-b', 10000), 'runtime_multiplier': 2.0}],
                                master_engine='llama_cpp', alignment=4)
    assert two['admitted'] and three['admitted']
    for plan in (two, three):
        assert len(plan['assignments']) == 2
        head = plan['assignments'][0]
        assert head['node_id'] == 'master'
        assert [head['start_layer'], head['end_layer']] == [0, 4]
        assert head['required_bytes'] == 97 <= head['capacity_bytes'] == 100
        assert plan['assignments'][1]['end_layer'] == 8
        assert not plan['loaded_ready']
    assert (desc, records) == before


def test_mandatory_later_executor_cannot_be_avoided_by_early_completion():
    desc = descriptor(layers=4, layer_bytes=10, components=0)
    records = [node('master', 10000), node('required-tail', 25)]
    plain = solve_pipeline_capacity(desc, records, safety_margin=1)
    assert len(plain['assignments']) == 1 and plain['assignments'][0]['node_id'] == 'master'
    result = solve_pipeline_capacity(desc, records, safety_margin=1, required_node_id='required-tail')
    assert result['admitted']
    head, tail = result['assignments']
    assert head['node_id'] == 'master' and head['start_layer'] == 0
    assert tail['node_id'] == 'required-tail' and tail['end_layer'] == 4
    assert head['end_layer'] == tail['start_layer'] and 2 <= tail['start_layer'] < 4


@pytest.mark.parametrize('records', [[node('master', 0), node('worker', 10000)], [node('worker', 10000)]])
def test_missing_or_depleted_required_executor_is_explicit(records):
    value = solve_pipeline_capacity(descriptor(), records, required_node_id='master', require_distributed=True)
    assert not value['admitted'] and value['assignments'] == []
    assert value['reason_code'] == 'pipeline_required_node_unavailable'
    assert value['required_node_id'] == 'master'


@pytest.mark.parametrize('contract', ['range', 'segment'])
def test_diagnostic_replays_preserve_mandatory_executor(contract):
    # Without a mandatory head, two larger workers can cover the decoder. With
    # it, even the range/segment diagnostic replay must preserve its real budget.
    desc = descriptor(layers=4, layer_bytes=10, components=0)
    records = [node('master', 5), node('worker-a', 1000), node('worker-b', 1000)]
    if contract == 'range':
        records[1]['layer_ranges'] = [[1, 3]]
        records[2]['layer_ranges'] = [[3, 4]]
    else:
        records[1]['segment_mode'] = 'middle'
        records[2]['segment_mode'] = 'tail'
    value = solve_pipeline_capacity(desc, records, safety_margin=1,
                                    require_distributed=True, required_node_id='master')
    assert not value['admitted'] and value['assignments'] == []
    assert value['reason_code'] == 'pipeline_distributed_capacity_insufficient'


@pytest.mark.parametrize('bad', ['', ' master ', 1, True, [], {}])
def test_invalid_mandatory_identity_is_rejected(bad):
    with pytest.raises(PipelineCapacityError, match='required_node_id'):
        solve_pipeline_capacity(descriptor(), [node('master'), node('worker')], required_node_id=bad)
