"""Real artifact gate regressions. Synthetic fixtures never count as release acceptance."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/package_contract/release_artifacts_gate.py"
spec = importlib.util.spec_from_file_location("release_gate_regression", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
runtime = sys.modules["release_gate_runtime"]
REVISION = "a" * 40
CONTRACT = json.loads((ROOT / "release-contract.json").read_text(encoding="utf-8"))


def options(tmp_path):
    return {"work_dir": tmp_path / "probe", "require_revision": REVISION}


def command(output="", rc=0):
    return {"executed": True, "returncode": rc, "ok": rc == 0, "output": output}


def make_apk(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = (183).to_bytes(2, "little")
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in {"AndroidManifest.xml": b"fixture manifest", "classes.dex": b"fixture dex",
            "lib/arm64-v8a/libqlh_llama_jni.so": bytes(header),
            "lib/arm64-v8a/libc++_shared.so": bytes(header),
            "assets/build-info.json": json.dumps({"schema_version": 1, "source_revision": REVISION, "source_dirty": False}).encode()}.items():
            archive.writestr(name, value)
    return path


@pytest.mark.parametrize("platform", ["windows", "linux", "android"])
def test_missing_final_artifacts_never_pass(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(gate.shutil, "which", lambda *args: None)
    monkeypatch.setattr(gate, "android_tool", lambda *args: None)
    report = gate.GATES[platform](tmp_path, CONTRACT, options(tmp_path)).as_dict()
    assert report["passed"] is False and report["status"] in ("failed", "unverified")


def test_scripts_and_source_versions_are_not_android_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "android_tool", lambda *args: "fixture-tool")
    report = gate.gate_android(tmp_path, CONTRACT, options(tmp_path)).as_dict()
    assert not report["passed"]
    assert any(i["check"] == "android.artifact" and not i["passed"] for i in report["items"])


def test_linux_docker_existence_does_not_equal_installed_artifact(tmp_path, monkeypatch):
    (tmp_path / "qlh-edge-inference-cpu_0.1.8.3_amd64.deb").write_bytes(b"fixture")
    monkeypatch.setattr(gate.shutil, "which", lambda *args: "fixture-docker")
    monkeypatch.setattr(gate, "run_command", lambda *args, **kwargs: command("daemon unavailable", 1))
    report = gate.gate_linux(tmp_path, CONTRACT, options(tmp_path)).as_dict()
    assert not report["passed"]
    assert any(i["check"] == "linux.execution" and i["status"] == "unverified" for i in report["items"])


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(returncode=2, ok=True),
    lambda r: r.update(returncode=False, ok=True),
    lambda r: r.update(returncode=None, ok=True),
    lambda r: r.update(executed=False),
    lambda r: r.update(ok=False),
])
def test_success_json_cannot_override_verifier_process_failure(tmp_path, monkeypatch, mutation):
    result = command(json.dumps({"ok": True, "level": "deep"}))
    mutation(result)
    monkeypatch.setattr(gate, "run_command", lambda *args, **kwargs: result)
    with pytest.raises(ValueError):
        gate.verify_tree(tmp_path, tmp_path / "work")


@pytest.mark.parametrize("authenticated,revision,required,expected", [
    (False, REVISION, REVISION, "unverified"),
    (True, None, REVISION, "failed"),
    (True, "b"*40, REVISION, "failed"),
    (True, REVISION, REVISION, "passed"),
])
def test_build_identity_requires_authenticated_matching_revision(authenticated, revision, required, expected):
    report = gate.Report("windows")
    gate.check_revision(report, "windows.cpu.revision", {"schema_version": 1, "source_revision": revision, "source_dirty": False},
                        authenticated=authenticated, required=required)
    assert report.items[0]["status"] == expected


def test_conflicting_artifact_revisions_fail_report():
    report = gate.Report("windows")
    for name, revision in (("cpu", REVISION), ("cuda", "b"*40)):
        report.ok(f"windows.{name}.revision", source_revision=revision, revision_verified=True)
    assert not report.as_dict()["passed"]


def linux_report():
    names = ["package.version", "package.install", "package.audit", "bundle.integrity",
             "bundle.contract", "bundle.revision", "runtime.dependencies", "runtime.clean", "runtime.worker"]
    return {"schema_version": 1, "gate": "linux-installed-artifact", "nonce": "current-nonce",
            "artifact_sha256": "c"*64, "variant": "cpu", "items": [
                {"check": name, "status": "passed", "evidence": {"runtime": {"executed": True, "ready": True}}}
                for name in names]}


@pytest.mark.parametrize("change", [
    lambda o: o.update(nonce="old-run"),
    lambda o: o.update(artifact_sha256="d"*64),
    lambda o: o.update(variant="cuda"),
    lambda o: o["items"].pop(),
    lambda o: o["items"].append(o["items"][0]),
    lambda o: o["items"][-1]["evidence"]["runtime"].update(executed=False),
    lambda o: o["items"][-1]["evidence"]["runtime"].update(ready=False),
    lambda o: o["items"][0].update(status="skipped"),
])
def test_container_result_must_be_complete_executed_and_bound(change):
    value = linux_report()
    change(value)
    with pytest.raises(ValueError):
        gate._linux_result(gate.Report("linux"), value, nonce="current-nonce", digest="c"*64, variant="cpu")


def test_complete_container_result_is_projected():
    report = gate.Report("linux")
    gate._linux_result(report, linux_report(), nonce="current-nonce", digest="c"*64, variant="cpu")
    assert len(report.items) == 9 and all(i["passed"] for i in report.items)


def test_apk_inspection_requires_actual_arm64_native_elf(tmp_path):
    apk = make_apk(tmp_path / "release.apk")
    assert gate.inspect_apk(apk)["abi"] == "arm64-v8a"
    with zipfile.ZipFile(tmp_path / "empty.apk", "w") as archive:
        archive.writestr("AndroidManifest.xml", "fixture")
    with pytest.raises(ValueError):
        gate.inspect_apk(tmp_path / "empty.apk")


@pytest.mark.parametrize("header", [b"not ELF".ljust(64,b"\0"), b"\x7fELF\x01\x01".ljust(64,b"\0")])
def test_apk_wrong_architecture_is_rejected(tmp_path, header):
    apk = tmp_path / "bad.apk"
    with zipfile.ZipFile(apk,"w") as archive:
        for name, value in {"AndroidManifest.xml":b"fixture","classes.dex":b"fixture",
                           "lib/arm64-v8a/libqlh_llama_jni.so":header,
                           "lib/arm64-v8a/libc++_shared.so":header}.items():
            archive.writestr(name,value)
    with pytest.raises(ValueError): gate.inspect_apk(apk)


def test_embedded_debug_apk_is_rejected():
    identity = "package: name='com.qlh.inference' versionCode='6' versionName='0.1.8.3'\n"
    assert gate.apk_metadata(identity)["version_code"] == 6
    with pytest.raises(ValueError): gate.apk_metadata(identity + "application-debuggable\n")


def test_apk_signer_must_be_single_known_certificate():
    signer = f"Signer #1 certificate SHA-256 digest: {'A'*64}\n"
    assert gate.signer_digest(signer) == "a"*64
    with pytest.raises(ValueError): gate.signer_digest(signer + signer)


def test_android_static_checks_do_not_invent_native_role_or_startup(tmp_path, monkeypatch):
    make_apk(tmp_path / "app-release.apk")
    monkeypatch.setattr(gate, "android_tool", lambda name: name)
    def run(args, **kwargs):
        if args[0] == "aapt":
            return command("package: name='com.qlh.inference' versionCode='6' versionName='0.1.8.3'\n")
        if args[0] == "apksigner":
            return command(f"Signer #1 certificate SHA-256 digest: {'a'*64}\n")
        return command("List of devices attached\n")
    monkeypatch.setattr(gate, "run_command", run)
    opts = {**options(tmp_path), "android_cert_sha256": "a"*64}
    report = gate.gate_android(tmp_path, CONTRACT, opts)
    checks = {i["check"]:i for i in report.items}
    assert checks["android.version"]["passed"] and checks["android.signature"]["passed"]
    assert checks["android.revision"]["passed"]
    for name in ("startup", "roles", "native-runtime"):
        assert checks["android."+name]["status"] == "unverified"
    assert report.as_dict()["passed"] is False


def test_android_probe_never_replaces_or_stops_existing_app(tmp_path, monkeypatch):
    calls=[]
    def run(args, **kwargs):
        calls.append(args)
        return command("package:/data/app/existing/base.apk\n")
    monkeypatch.setattr(gate,"run_command",run)
    report=gate.Report("android")
    gate.android_device_smoke(report,adb="adb",serial="fixture",apk=tmp_path/"a.apk",work=tmp_path,env={})
    assert len(calls)==1 and calls[0][-3:]==["pm","path","com.qlh.inference"]
    assert not report.as_dict()["passed"]


def test_readiness_requires_every_component_and_actual_role():
    healthy={"status":"ok"}
    ready={"process_ready":True,"ready":True,"status":"ready","error":None,
           "components":{"local_store":True,"scheduler":True,"device_profile":True}}
    assert runtime.readiness_matches(healthy,ready,{"node_role":"master"},"master")
    assert not runtime.readiness_matches(healthy,ready,{"node_role":"client"},"master")
    ready["components"]["scheduler"]=False
    assert not runtime.readiness_matches(healthy,ready,{"node_role":"master"},"master")


@pytest.mark.external
def test_owned_command_runs_hidden_and_times_out_without_success(tmp_path):
    env=runtime.isolated_environment(tmp_path/"state")
    result=runtime.run_command([sys.executable,"-B","-c","import json; print(json.dumps({'ok':True}))"],
                              cwd=tmp_path,env=env,log=tmp_path/"success.log",timeout=10)
    assert result["ok"] and runtime.json_output(result)=={"ok":True}
    timeout=runtime.run_command([sys.executable,"-B","-c","import time; time.sleep(30)"],
                               cwd=tmp_path,env=env,log=tmp_path/"timeout.log",timeout=0.3)
    assert timeout["executed"] and not timeout["ok"]
    assert timeout["returncode"] is None


def test_runtime_state_is_isolated_from_production(tmp_path,monkeypatch):
    monkeypatch.setenv("QLH_CLUSTER_SECRET","sensitive")
    monkeypatch.setenv("PYTHONPATH","production")
    env=runtime.isolated_environment(tmp_path/"state")
    assert "QLH_CLUSTER_SECRET" not in env and "PYTHONPATH" not in env
    assert Path(env["QLH_SQLITE_PATH"]).is_relative_to(tmp_path)
    assert Path(env["QLH_NODE_CONFIG_PATH"]).is_relative_to(tmp_path)


@pytest.mark.parametrize("platform",["windows","linux","android"])
def test_cli_missing_dist_writes_failure_evidence(tmp_path,platform):
    output=tmp_path/"result.json"
    assert gate.main(["--platforms",platform,"--dist",str(tmp_path/"absent"),"--json",str(output),
                      "--work-dir",str(tmp_path/"work"),"--require-revision",REVISION])==1
    result=json.loads(output.read_text())
    assert result["passed"] is False and result["requested_platforms"]==[platform]


def test_cli_requires_explicit_platform_and_valid_options():
    assert gate.main([])==1
    assert gate.main(["--platforms","solaris"])==2
    with pytest.raises(SystemExit): gate.main(["--platforms","linux","--linux-variants","cpu,cpu"])
    with pytest.raises(SystemExit): gate.main(["--platforms","android","--android-device-probe"])


def test_cuda_is_explicitly_deferred_by_default(tmp_path, monkeypatch):
    seen = {}
    def inspect(dist, contract, settings):
        seen.update(settings)
        report = gate.Report("windows")
        report.unverified("runtime", "synthetic parser-only test")
        return report
    monkeypatch.setitem(gate.GATES, "windows", inspect)
    assert gate.main(["--platforms", "windows", "--dist", str(tmp_path)]) == 1
    assert seen["windows_variants"] == ["cpu"]
    assert seen["linux_variants"] == ["cpu"]


def _device_smoke_commands(tmp_path, apk, monkeypatch):
    """A bounded fake device; no actual package install is performed by this test."""
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if args[-3:] == ['pm', 'path', 'com.qlh.inference']:
            seen = sum(call[-3:] == ['pm', 'path', 'com.qlh.inference'] for call in calls)
            return command('' if seen == 1 else 'package:/data/app/accepted/base.apk\n')
        if 'pull' in args:
            shutil.copyfile(apk, Path(args[-1]))
            return command('1 file pulled')
        if 'install' in args or 'uninstall' in args:
            return command('Success\n')
        if 'am' in args:
            return command('Status: ok\n')
        if 'pidof' in args:
            return command('123\n')
        if args[-3:] == ['dumpsys', 'activity', 'activities']:
            return command('mResumedActivity: com.qlh.inference/.MainActivity\n')
        raise AssertionError(args)
    monkeypatch.setattr(gate, 'run_command', run)
    monkeypatch.setattr(gate.time, 'sleep', lambda _: None)
    return calls


def test_fresh_apk_role_and_native_acceptance_requires_real_runtime_observation(tmp_path, monkeypatch):
    apk = make_apk(tmp_path/'accepted.apk')
    calls = _device_smoke_commands(tmp_path, apk, monkeypatch)
    observed = {}
    def observe(**kwargs):
        observed.update(kwargs)
        return {'passed': True, 'observation': {'role': 'client'}, 'attempts': [{'nonce': 'n'}]}
    monkeypatch.setattr(gate, 'observe_runtime', observe)
    report = gate.Report('android')
    gate.android_device_smoke(report, adb='adb', serial='Y700', apk=apk, work=tmp_path, env={},
                              version_name='0.1.8.3', version_code=6)
    checks = {item['check']: item for item in report.items}
    assert all(checks['android.'+name]['passed'] for name in ('startup','roles','native-runtime','cleanup'))
    assert observed['require_idle'] is True and observed['version_name'] == '0.1.8.3' and observed['version_code'] == 6
    assert checks['android.native-runtime']['evidence']['artifact_sha256'] == gate.sha256(apk)
    assert calls[-1][-2:] == ['uninstall', 'com.qlh.inference']


def test_resumed_activity_does_not_override_native_probe_failure(tmp_path, monkeypatch):
    apk = make_apk(tmp_path/'accepted.apk')
    calls = _device_smoke_commands(tmp_path, apk, monkeypatch)
    monkeypatch.setattr(gate, 'observe_runtime', lambda **_: {'passed': False, 'attempts': [{'error':'JNI missing'}]})
    report = gate.Report('android')
    gate.android_device_smoke(report, adb='adb', serial='Y700', apk=apk, work=tmp_path, env={})
    checks = {item['check']: item for item in report.items}
    assert checks['android.startup']['passed'] is True
    assert checks['android.roles']['passed'] is False and checks['android.native-runtime']['passed'] is False
    assert checks['android.cleanup']['passed'] is True and report.as_dict()['passed'] is False
    assert calls[-1][-2:] == ['uninstall', 'com.qlh.inference']


def test_installed_apk_mismatch_never_observes_or_launches_runtime(tmp_path, monkeypatch):
    apk = make_apk(tmp_path/'accepted.apk')
    calls = _device_smoke_commands(tmp_path, apk, monkeypatch)
    fake_run = gate.run_command
    def tampered(args, **kwargs):
        result = fake_run(args, **kwargs)
        if 'pull' in args:
            Path(args[-1]).write_bytes(b'another apk')
        return result
    monkeypatch.setattr(gate, 'run_command', tampered)
    monkeypatch.setattr(gate, 'observe_runtime', lambda **_: pytest.fail('APK mismatch must stop before runtime observation'))
    report = gate.Report('android')
    gate.android_device_smoke(report, adb='adb', serial='Y700', apk=apk, work=tmp_path, env={})
    assert not report.as_dict()['passed']
    assert not any('am' in call for call in calls)
    assert calls[-1][-2:] == ['uninstall', 'com.qlh.inference']


def test_dirty_authenticated_build_cannot_claim_release_revision():
    report = gate.Report('android')
    assert gate.check_revision(report, 'android.revision',
        {'schema_version':1,'source_revision':REVISION,'source_dirty':True}, authenticated=True, required=REVISION) is None
    assert report.as_dict()['passed'] is False


@pytest.mark.parametrize("dirty", [None, True, 0, 1, "false", [], {}])
def test_authenticated_artifact_requires_explicit_boolean_clean_source(dirty):
    report = gate.Report("android")
    info = {"schema_version": 1, "source_revision": REVISION, "source_dirty": dirty}
    assert gate.check_revision(report, "android.revision", info,
                               authenticated=True, required=REVISION) is None
    assert report.items[0]["status"] == "failed"


def test_signed_artifact_missing_clean_source_fact_is_rejected():
    report = gate.Report("windows")
    info = {"schema_version": 1, "source_revision": REVISION}
    assert gate.check_revision(report, "windows.cpu.revision", info,
                               authenticated=True, required=REVISION) is None
    assert report.items[0]["status"] == "failed"
