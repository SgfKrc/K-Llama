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



def product_observation():
    value = observation()
    value.update(probe_kind="product", role="worker")
    value["worker"].update(process_running=True, connection_alive=True, connection="READY")
    value["model"].update(loaded=True, layer_range=[24, 32], native_layer_count=8)
    identity = {"pid": 123, "created_at": 1234500}
    proc = {**identity, "parent_pid": 1, "identity_after": dict(identity), "command": ["fixture"], "cwd": "/data/fixture",
            "task_ids": [123, 124], "child_pids": [], "children_complete": True,
            "maps_complete": True, "model_mappings": ["/data/fixture/segment.gguf"], "native_mappings": [],
            "connections_complete": True, "connections": [{"status": "ESTABLISHED", "local": ["127.0.0.1", 40000],
                "remote": ["127.0.0.2", 8888]}], "errors": [], "connection_errors": []}
    value["process"] = {"pid": 123, "created_at": identity["created_at"], "source": "android_proc_task_children",
                        "identity_before": dict(identity), "identity_after": dict(identity), "complete": True,
                        "processes": [proc], "errors": []}
    value["runtime_receipt"] = {**identity, "schema_version": 1, "source": "committed_worker_prepare", "engine": "llama_cpp",
        "operation_id": "supply_real", "config_id": "config_real", "generation": 1791800000000000013,
        "node_id": "android_real", "assignment_id": "assignment_real", "package_name": PACKAGE,
        "loader_pid": 123, "loaded": True, "full_model_loaded": False, "layer_range": [24, 32],
        "source_layer_range": [24, 32], "range_source": "verified_loaded_native_manifest", "total_layers": 32,
        "native_layer_count": 8, "captured_at": 100, "artifact": {
            "path": "/data/fixture/segment.gguf", "load_path": "/proc/self/fd/42", "size_bytes": 4500000000,
            "sha256": "b"*64, "rehashed_sha256": "b"*64, "native_manifest_sha256": "c"*64,
            "layer_range": [24, 32], "decoder_layers": 8, "device": 259, "inode": 456}}
    return value


def test_product_receipt_retains_exact_generation_rehash_and_complete_os_tree():
    value = product_observation()
    result = validate(value, product=True)
    assert result["runtime_receipt"]["generation"] == 1791800000000000013
    assert result["runtime_receipt"]["artifact"]["size_bytes"] == 4500000000
    assert result["process"]["processes"][0]["task_ids"] == [123, 124]


@pytest.mark.parametrize("change", [
    lambda v: v.pop("runtime_receipt"), lambda v: v.update(probe_kind="release"),
    lambda v: v["runtime_receipt"].update(generation=True),
    lambda v: v["runtime_receipt"].update(generation=0),
    lambda v: v["runtime_receipt"].update(loader_pid=124),
    lambda v: v["runtime_receipt"].update(created_at=1234501),
    lambda v: v["runtime_receipt"].update(total_layers=8),
    lambda v: v["runtime_receipt"].update(source_layer_range=[0, 32]),
    lambda v: v["runtime_receipt"].update(native_layer_count=32),
    lambda v: v["runtime_receipt"]["artifact"].update(rehashed_sha256="d"*64),
    lambda v: v["runtime_receipt"]["artifact"].update(decoder_layers=32),
    lambda v: v["runtime_receipt"]["artifact"].update(path="/data/fixture/segment.gguf (deleted)"),
    lambda v: v["runtime_receipt"]["artifact"].update(inode=0),
    lambda v: v["process"].update(complete=False),
    lambda v: v["process"]["processes"][0].update(children_complete=False),
    lambda v: v["process"]["processes"][0].update(maps_complete=False),
    lambda v: v["process"]["processes"][0].update(connections_complete=False),
    lambda v: v["process"]["processes"][0].update(child_pids=[999]),
    lambda v: v["process"]["processes"][0]["identity_after"].update(created_at=1234501),
    lambda v: v["process"]["processes"][0]["connections"][0].update(status="UNKNOWN"),
])
def test_product_rejects_missing_plan_only_stale_or_incomplete_receipts(change):
    value = product_observation()
    change(value)
    with pytest.raises(ValueError):
        validate(value, product=True)


def test_idle_release_diagnostics_do_not_require_product_hash_or_complete_process_claim():
    value = observation()
    value["process"] = {"pid": 123, "maps_complete": True, "children_complete": False}
    assert validate(value, require_idle=True)["model"]["loaded"] is False
    with pytest.raises(ValueError):
        validate(value, product=True)


def test_product_poll_uses_same_nonce_and_does_not_reschedule_hash(tmp_path, monkeypatch):
    calls, dumps = [], 0
    def run(args, **kwargs):
        nonlocal dumps
        calls.append(args)
        if "pidof" in args:
            return execution("123")
        dumps += 1
        value = product_observation()
        value.update(nonce=args[-1], captured_at=probe.time.time())
        value["runtime_receipt"]["captured_at"] = probe.time.time()
        if dumps <= 2:
            value.update(error_code="PROBE_PENDING", runtime_receipt=None)
        return execution(probe.PREFIX + json.dumps(value))
    monkeypatch.setattr(probe, "run_command", run)
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    result = probe.observe_runtime(adb="adb", serial="Y700", package=PACKAGE, work=tmp_path,
                                   env={"fixture":"1"}, product=True)
    assert result["passed"] and dumps == 3 and len(result["attempts"]) == 3
    assert len({attempt["nonce"] for attempt in result["attempts"]}) == 1
    assert [a.get("pending", False) for a in result["attempts"]] == [True, True, False]
    assert all("--qlh-product-probe" in args for args in calls if "dumpsys" in args)
    assert not any(token in {"install", "start", "uninstall", "force-stop", "clear"} for call in calls for token in call)


@pytest.mark.parametrize("failure", ["restart", "pending-nonce", "incomplete-tree"])
def test_product_poll_never_accepts_restart_wrong_pending_or_permission_denial(tmp_path, monkeypatch, failure):
    dumps = 0
    def run(args, **kwargs):
        nonlocal dumps
        if "pidof" in args:
            return execution("124" if failure == "restart" and dumps else "123")
        dumps += 1
        value = product_observation()
        value.update(nonce=args[-1], captured_at=probe.time.time())
        value["runtime_receipt"]["captured_at"] = probe.time.time()
        if failure == "pending-nonce":
            value.update(error_code="PROBE_PENDING", nonce="wrong")
        if failure == "incomplete-tree":
            value["process"].update(complete=False)
        return execution(probe.PREFIX + json.dumps(value))
    monkeypatch.setattr(probe, "run_command", run)
    result = probe.observe_runtime(adb="adb", serial="Y700", package=PACKAGE, work=tmp_path,
                                   env={"fixture":"1"}, product=True)
    assert result["passed"] is False and result["observation"] is None and len(result["attempts"]) == 1
    assert probe.PREFIX in result["attempts"][0]["executions"]["dump"]["output"]


def uid_fallback_observation():
    import copy
    value = product_observation()
    proc = value["process"]["processes"][0]
    scan = {"visible_pids": [123, 222, 999], "owners": [
        {"pid": 123, "uid": 10000}, {"pid": 222, "uid": 10000}, {"pid": 999, "uid": 20000}],
        "roster": [{"pid": 123, "uid": 10000, "created_at": proc["created_at"], "parent_pid": 1},
                   {"pid": 222, "uid": 10000, "created_at": 222000, "parent_pid": 1}],
        "outside_uid": [{"pid": 999, "uid": 20000, "created_at": 999000, "parent_pid": 1}], "errors": []}
    value["process"].update(source="android_proc_uid_ppid", enumeration={"method": "proc_uid_ppid",
        "scope": "app_uid_descendants", "reason": "task_children_enoent", "uid": 10000,
        "complete": True, "before": copy.deepcopy(scan), "after": copy.deepcopy(scan)})
    return value


def test_product_accepts_double_scanned_uid_ppid_fallback_with_unrelated_same_uid_process():
    value = uid_fallback_observation()
    assert validate(value, product=True)["process"]["source"] == "android_proc_uid_ppid"


@pytest.mark.parametrize("change", [
    lambda t: t.pop("enumeration"),
    lambda t: t["enumeration"].update(complete=False),
    lambda t: t["enumeration"].update(reason="permission_denied"),
    lambda t: t["enumeration"].update(uid=True),
    lambda t: t["enumeration"]["before"].update(errors=["123:stat_denied"]),
    lambda t: t["enumeration"]["before"]["owners"].pop(),
    lambda t: t["enumeration"]["before"]["roster"].pop(),
    lambda t: t["enumeration"]["after"]["roster"][0].update(created_at=1234501),
    lambda t: t["enumeration"]["after"]["roster"][1].update(parent_pid=123),
    lambda t: t["enumeration"]["before"]["owners"][0].update(uid=20000),
])
def test_uid_fallback_rejects_missing_denied_omitted_or_racing_scan_proof(change):
    value = uid_fallback_observation()
    change(value["process"])
    with pytest.raises(ValueError):
        validate(value, product=True)


def test_uid_fallback_rejects_stably_omitted_descendant_even_when_tree_says_complete():
    value = uid_fallback_observation()
    for scan in (value["process"]["enumeration"]["before"], value["process"]["enumeration"]["after"]):
        scan["roster"][1]["parent_pid"] = 123
    with pytest.raises(ValueError, match="omits a UID descendant"):
        validate(value, product=True)


@pytest.mark.parametrize("change", [
    lambda s: s.pop("outside_uid"),
    lambda s: s["outside_uid"].clear(),
    lambda s: s["outside_uid"][0].update(parent_pid=123),
])
def test_uid_fallback_never_silently_omits_different_uid_child_or_unreadable_process(change):
    value = uid_fallback_observation()
    change(value["process"]["enumeration"]["before"])
    with pytest.raises(ValueError):
        validate(value, product=True)
