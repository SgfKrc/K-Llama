"""Nonce/PID-bound Android observations; synthetic fixtures are not device acceptance."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('android_probe_test', ROOT / 'scripts/package_contract/android_runtime_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
NONCE = 'a' * 32
PACKAGE = 'com.qlh.inference'


def observation():
    return {'schema_version': 1, 'nonce': NONCE, 'pid': 123, 'captured_at': 100,
            'package_name': PACKAGE, 'version_name': '0.1.8.3', 'version_code': 6,
            'service_running': True, 'inference_mode': 'distributed', 'role': 'client', 'error_code': None,
            'worker': {'process_running': False, 'connection_alive': False, 'connection': 'STOPPED', 'active_attempt_state': 'IDLE'},
            'native': {'available': True, 'system_info': 'NEON = 1', 'backend_device_count': 1,
                       'backend_devices': 'CPU(CPU)', 'error_code': None},
            'model': {'loaded': False, 'layer_range': [], 'full_model_loaded': False, 'native_layer_count': None}}


def validate(value, **kwargs):
    options = {'nonce': NONCE, 'pid': 123, 'package': PACKAGE, 'started_at': 99, 'finished_at': 101,
               'version_name': '0.1.8.3', 'version_code': 6}
    options.update(kwargs)
    return probe.validate_observation('ACTIVITY MANAGER SERVICES\n  ' + probe.PREFIX + json.dumps(value), **options)


def test_real_idle_backend_can_be_observed_without_loading_any_model():
    result = validate(observation(), require_idle=True)
    assert result['role'] == 'client' and result['model']['loaded'] is False


@pytest.mark.parametrize('change', [
    lambda v: v.update(schema_version=True), lambda v: v.update(nonce='b'*32),
    lambda v: v.update(pid=True), lambda v: v.update(pid=124),
    lambda v: v.update(captured_at=50), lambda v: v.update(captured_at=float('inf')),
    lambda v: v.update(package_name=PACKAGE+'.debug'), lambda v: v.update(version_name='old'),
    lambda v: v.update(version_code=True), lambda v: v.update(version_code=5),
    lambda v: v.update(service_running=False), lambda v: v.update(error_code='timeout'),
    lambda v: v.update(inference_mode='auto'), lambda v: v.update(role='worker'),
    lambda v: v['worker'].update(connection_alive=True),
    lambda v: v['worker'].update(connection='READY'),
    lambda v: v['worker'].update(process_running=True),
    lambda v: v['worker'].update(active_attempt_state='RUNNING'),
    lambda v: v['worker'].update(connection='UNKNOWN'),
    lambda v: v['native'].update(available=False), lambda v: v['native'].update(error_code='stub'),
    lambda v: v['native'].update(system_info=''), lambda v: v['native'].update(backend_devices=''),
    lambda v: v['native'].update(backend_device_count=0), lambda v: v['native'].update(backend_device_count=True),
    lambda v: v['model'].update(loaded='false'), lambda v: v['model'].update(full_model_loaded=True),
    lambda v: v['model'].update(layer_range=[0, 2]), lambda v: v['model'].update(native_layer_count=2),
])
def test_incomplete_stale_or_contradictory_runtime_facts_are_rejected(change):
    value = observation()
    change(value)
    with pytest.raises(ValueError):
        validate(value)


def test_runtime_refuses_duplicate_records_or_duplicate_json_keys():
    line = probe.PREFIX + json.dumps(observation())
    args = dict(nonce=NONCE, pid=123, package=PACKAGE, started_at=99, finished_at=101)
    with pytest.raises(ValueError):
        probe.validate_observation(line+'\n'+line, **args)
    duplicate = line.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    with pytest.raises(ValueError):
        probe.validate_observation(duplicate, **args)


def test_cluster_segment_receipt_requires_actual_native_count_and_worker_connection():
    value = observation()
    value.update(role='worker')
    value['worker'].update(process_running=True, connection_alive=True, connection='READY')
    value['model'].update(loaded=True, layer_range=[24,32], native_layer_count=8)
    assert validate(value)['model']['full_model_loaded'] is False
    with pytest.raises(ValueError):
        validate(value, require_idle=True)
    value['model']['native_layer_count'] = 32
    with pytest.raises(ValueError):
        validate(value)


def test_full_model_requires_explicit_local_mode_and_no_cluster_worker():
    value = observation()
    value['model'].update(loaded=True, full_model_loaded=True, native_layer_count=32)
    with pytest.raises(ValueError):
        validate(value)
    value.update(inference_mode='local', role='local')
    assert validate(value)['model']['full_model_loaded'] is True
    value['worker'].update(process_running=True, connection='CONNECTING')
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize('bad', [True, 0, -1, 1.0])
def test_pid_expectation_is_a_positive_integer(bad):
    with pytest.raises(ValueError):
        validate(observation(), pid=bad)


def execution(output='', rc=0):
    return {'executed': True, 'ok': rc == 0, 'returncode': rc, 'output': output}


def test_read_only_collection_binds_both_pid_observations_and_new_nonce(tmp_path, monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if 'pidof' in args:
            return execution('123\n')
        value = observation()
        value.update(nonce=args[-1], captured_at=probe.time.time())
        return execution(probe.PREFIX + json.dumps(value))
    monkeypatch.setattr(probe, 'run_command', run)
    result = probe.observe_runtime(adb='adb', serial='Y700', package=PACKAGE, work=tmp_path, env={'fixture':'1'}, require_idle=True)
    assert result['passed'] and len(calls) == 3
    assert calls[0][-2:] == ['pidof', PACKAGE] and calls[2][-2:] == ['pidof', PACKAGE]
    assert calls[1][3:8] == ['shell', 'dumpsys', 'activity', 'service', PACKAGE+'/com.qlh.inference.service.InferenceService']
    assert result['observation']['nonce'] != NONCE
    assert not any(token in {'install','uninstall','start','force-stop','clear'} for call in calls for token in call)


@pytest.mark.parametrize('failure', ['restarted','dump-failed','no-service','stub'])
def test_process_restart_missing_service_and_stub_never_pass(tmp_path, monkeypatch, failure):
    ticks = iter([0, 0, 0, 0, 0, 2, 2, 2, 2])
    monkeypatch.setattr(probe.time, 'monotonic', lambda: next(ticks, 2))
    seen_pid = 0
    def run(args, **kwargs):
        nonlocal seen_pid
        if 'pidof' in args:
            seen_pid += 1
            return execution('124' if failure == 'restarted' and seen_pid > 1 else '123')
        if failure == 'dump-failed':
            return execution('ignored', 1)
        if failure == 'no-service':
            return execution('No services match')
        value = observation()
        value.update(nonce=args[-1], captured_at=probe.time.time())
        value['native']['backend_device_count'] = 0
        return execution(probe.PREFIX+json.dumps(value))
    monkeypatch.setattr(probe, 'run_command', run)
    result = probe.observe_runtime(adb='adb', serial='Y700', package=PACKAGE, work=tmp_path, env={'fixture':'1'}, timeout=1)
    assert result['passed'] is False and result['observation'] is None


def test_collector_refuses_arbitrary_shell_package_or_serial(tmp_path):
    with pytest.raises(ValueError):
        probe.observe_runtime(adb='adb', serial='Y700', package='com.qlh.inference; id', work=tmp_path)
    with pytest.raises(ValueError):
        probe.observe_runtime(adb='adb', serial='Y700; id', package=PACKAGE, work=tmp_path)


def test_terminal_worker_failure_can_be_observed_without_claiming_worker_readiness():
    value = observation()
    value['worker'].update(connection='RETRY_EXHAUSTED', active_attempt_state='FAILED')
    assert validate(value)['role'] == 'client'
    with pytest.raises(ValueError):
        validate(value, require_idle=True)


@pytest.mark.parametrize('field', ['system_info','backend_devices'])
def test_stub_backend_label_cannot_claim_native_readiness(field):
    value = observation()
    value['native'][field] = 'test STUB backend'
    with pytest.raises(ValueError):
        validate(value)
