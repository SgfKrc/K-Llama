"""Offline build-chain regressions; synthetic files never count as release evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

provenance_fixtures = fixture_module("test_build_provenance")
checkout = provenance_fixtures.checkout
seed_checkout = provenance_fixtures.seed_checkout
git, put = provenance_fixtures.git, provenance_fixtures.put
product_fixture = fixture_module("test_product_relay_gate").bundle

SPEC = importlib.util.spec_from_file_location("release_build_test", ROOT / "scripts/package_contract/release_build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)
REV = "a" * 40
INFO = {"schema_version": 1, "source_revision": REV, "source_dirty": False, "source_tree_sha256": "b" * 64,
        "components": {".": {"source_dirty": False, "revision_matches_pin": True},
                       "android": {"source_dirty": False, "revision_matches_pin": True}}}
CONTRACT = json.loads((ROOT / "release-contract.json").read_text(encoding="utf-8"))


@pytest.fixture
def run_env(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("EXPECTED_REVISION", REV)
    monkeypatch.setenv("GITHUB_SHA", REV)


def load_yaml(path):
    # BaseLoader preserves the Actions `on` key (YAML 1.1 bool coercion is wrong).
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_manual_only_workflow_produces_exact_acceptance_inputs():
    workflow = load_yaml(ROOT / ".github/workflows/release-build.yml")
    acceptance = load_yaml(ROOT / ".github/workflows/release-acceptance.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert jobs["artifacts"]["needs"] == ["revision", "product-evidence"]
    expected_matrix = acceptance["jobs"]["final-artifacts"]["strategy"]["matrix"]["include"]
    assert jobs["artifacts"]["strategy"]["matrix"]["include"] == expected_matrix
    produced = []
    for name, job in jobs.items():
        for step in job.get("steps", []):
            if step.get("uses") == "actions/checkout@v4":
                assert step["with"]["ref"] == "${{ github.sha }}"
                assert step["with"]["clean"] == "true"
                assert step["with"]["submodules"] == "recursive"
                assert step["with"]["persist-credentials"] == "false"
            if step.get("uses") == "actions/upload-artifact@v4":
                assert "if" not in step  # upload only after successful verification
                assert step["with"]["if-no-files-found"] == "error"
                assert step["with"]["include-hidden-files"] == "true"
                assert step["with"]["path"].endswith("/stage/")
                produced.append(step["with"]["name"])
    consumed = [step["with"]["name"] for job in acceptance["jobs"].values() for step in job.get("steps", [])
                if step.get("uses") == "actions/download-artifact@v4" and step.get("with", {}).get("run-id")]
    assert set(produced) == set(consumed) == {"qlh-product-relay-evidence", "qlh-final-${{ matrix.platform }}"}


@pytest.mark.parametrize("revision,event,success", [(REV, REV, True), ("a" * 8, REV, False),
    ("A" * 40, REV, False), ("a" * 39, REV, False), (REV, "c" * 40, False), ("$(id)", REV, False)])
def test_actual_dispatch_revision_step_cannot_relabel_run(monkeypatch, revision, event, success):
    workflow = load_yaml(ROOT / ".github/workflows/release-build.yml")
    script = workflow["jobs"]["revision"]["steps"][0]["run"]
    monkeypatch.setenv("EXPECTED_REVISION", revision)
    monkeypatch.setenv("GITHUB_SHA", event)
    compile(script, "workflow-revision", "exec")
    if success:
        exec(script, {})
        build.revision_pair(revision, event)
    else:
        with pytest.raises(SystemExit):
            exec(script, {})
        with pytest.raises(build.BuildError):
            build.revision_pair(revision, event)


def test_build_commands_use_real_entrypoints_with_noninteractive_windows():
    workflow = load_yaml(ROOT / ".github/workflows/release-build.yml")
    job = workflow["jobs"]["artifacts"]
    assert job["env"]["QLH_BUILD_PROFILE"] == "llama_cpp_only"
    assert job["env"]["QLH_NONINTERACTIVE"] == "1"
    assert job["env"]["QLH_CORE_ROOT"] == "${{ github.workspace }}"
    assert job["env"]["QLH_SHELL_ROOT"] == "${{ github.workspace }}/frontend_cybergothic"
    assert job["env"]["QLH_SIGNING_KEY"] == "${{ vars.QLH_RELEASE_SIGNING_KEY_PATH }}"
    runs = [(s.get("shell"), s.get("run", "")) for s in job["steps"]]
    cmd = next(script for shell, script in runs if shell == "cmd")
    for name in ("build-cpu.bat", "build-launcher.bat"):
        assert f'call "%GITHUB_WORKSPACE%\\packaging\\packaging\\{name}" <NUL\n' in cmd
        assert (ROOT / "packaging/packaging" / name).is_file()
    assert cmd.count("if errorlevel 1 exit /b 1") == 2
    combined = "\n".join(script for _, script in runs)
    assert "bash packaging/packaging/linux/build-deb.sh cpu" in combined
    assert "bash android/gradlew -p android :app:assembleRelease --console=plain" in combined
    assert "connected" not in combined and "adb " not in combined
    assert "release_artifacts_gate.py --platforms" not in combined
    assert "keygen" not in combined and "signing.py sign" not in combined
    # The actual CPU entrypoint switches to the packaging repository root.
    cpu = (ROOT / "packaging/packaging/build-cpu.bat").read_text(encoding="utf-8")
    assert 'cd /d "%~dp0\\.."' in cpu and 'dist\\QLH-Edge-Inference' in cpu
    assert 'if not defined QLH_NONINTERACTIVE pause' in cpu


@pytest.mark.skipif(os.name != "nt", reason="Execute the actual cmd workflow on Windows")
@pytest.mark.parametrize("cpu_code,launcher_code", [(0, 0), (7, 0), (0, 9)])
def test_windows_workflow_survives_cpu_cwd_change_and_propagates_failures(tmp_path, cpu_code, launcher_code):
    workflow = load_yaml(ROOT / ".github/workflows/release-build.yml")
    script = next(s["run"] for s in workflow["jobs"]["artifacts"]["steps"] if s.get("shell") == "cmd")
    # Controlled offline entrypoints only; never invoke the real builders.
    workspace = tmp_path / "checkout with spaces"
    builders = workspace / "packaging/packaging"
    builders.mkdir(parents=True)
    (builders / "build-cpu.bat").write_text(
        '@echo off\ncd /d "%~dp0.."\nexit /b ' + str(cpu_code) + '\n', encoding="utf-8")
    (builders / "build-launcher.bat").write_text(
        '@echo off\n>"%GITHUB_WORKSPACE%\\launcher-ran.txt" echo reached\nexit /b ' + str(launcher_code) + '\n',
        encoding="utf-8")
    command = workspace / "workflow.cmd"
    command.write_text(script, encoding="utf-8")
    result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(command)],
        cwd=workspace, env={**os.environ, "GITHUB_WORKSPACE": str(workspace)}, capture_output=True, timeout=15)
    assert (result.returncode == 0) == (cpu_code == launcher_code == 0), result
    assert (workspace / "launcher-ran.txt").exists() == (cpu_code == 0)


@pytest.mark.parametrize("change", [lambda i: i.pop("source_dirty"), lambda i: i.update(source_dirty=0),
    lambda i: i.update(source_dirty=True), lambda i: i.update(source_revision="c" * 40),
    lambda i: i["components"]["android"].update(revision_matches_pin=False),
    lambda i: i["components"]["android"].update(source_dirty=True), lambda i: i.update(schema_version=True)])
def test_clean_identity_never_defaults_missing_facts_to_true(change):
    info = copy.deepcopy(INFO)
    change(info)
    with pytest.raises(build.BuildError):
        build.clean_info(info, REV)


def test_real_fixed_git_graph_and_worktree_drift_are_checked(checkout, monkeypatch):
    provider = build.tool(ROOT, "scripts/package_contract/build_provenance.py")
    monkeypatch.setattr(build, "tool", lambda *_: provider)
    revision = git(checkout, "rev-parse", "HEAD")
    observed = build.observe(checkout, revision)
    assert observed["source_dirty"] is False and len(observed["components"]) == 6
    put(checkout / "android", "app/src/main/Production.kt", "modified Kotlin\n")
    with pytest.raises(build.BuildError, match="source_dirty"):
        build.observe(checkout, revision)


def test_postbuild_reuses_strict_original_snapshot_and_rejects_source_drift(checkout, monkeypatch, run_env):
    provider = build.tool(ROOT, "scripts/package_contract/build_provenance.py")
    monkeypatch.setattr(build, "tool", lambda *_: provider)
    revision = git(checkout, "rev-parse", "HEAD")
    snapshot = build.work_path(checkout, "linux") / "source/build-info.json"
    provider.write_build_info(checkout, snapshot)
    original = snapshot.read_bytes()
    assert build.expected_info(checkout, "linux", revision)["source_dirty"] is False
    put(checkout, "src/core.py", "VALUE = 2\n")
    with pytest.raises(provider.ProvenanceError):
        build.expected_info(checkout, "linux", revision)
    assert snapshot.read_bytes() == original


@pytest.mark.parametrize("platform,stale", [("windows", "packaging/dist/QLH-Edge-Inference/old.exe"),
    ("windows", "packaging/dist/QLH-Launcher/old.exe"), ("linux", "packaging/packaging/linux/qlh-edge-inference-cpu_old.deb"),
    ("android", "android/app/build/outputs/apk/release/old.apk")])
def test_preflight_refuses_stale_outputs_without_deleting_them(tmp_path, monkeypatch, run_env, platform, stale):
    key = tmp_path / "existing.key"
    key.write_bytes(b"never read this fixture")
    monkeypatch.setenv("QLH_SIGNING_KEY", str(key))
    monkeypatch.setenv("QLH_BUILD_PROFILE", "llama_cpp_only")
    put(tmp_path, "packaging/.venv-packaging/Scripts/python.exe", "fixture")
    put(tmp_path, stale, "old output")
    monkeypatch.setattr(build, "observe", lambda *_: pytest.fail("must reject stale output before source snapshot"))
    with pytest.raises(build.BuildError, match="[Oo]ld"):
        build.preflight(tmp_path, platform, REV, REV)
    assert (tmp_path / stale).read_text() == "old output"


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_missing_signing_configuration_fails_before_build(tmp_path, monkeypatch, run_env, platform):
    monkeypatch.delenv("QLH_SIGNING_KEY", raising=False)
    with pytest.raises(build.BuildError, match="QLH_SIGNING_KEY"):
        build.preflight(tmp_path, platform, REV, REV)


def test_android_missing_properties_is_failure_not_unsigned_build(tmp_path, monkeypatch, run_env):
    monkeypatch.setenv("ANDROID_RELEASE_CERT_SHA256", "d" * 64)
    monkeypatch.delenv("QLH_RELEASE_ANDROID_KEYSTORE_PROPERTIES", raising=False)
    with pytest.raises(build.BuildError, match="KEYSTORE_PROPERTIES"):
        build.preflight(tmp_path, "android", REV, REV)
    assert not (tmp_path / "android/keystore.properties").exists()


def test_android_uses_existing_external_configuration_bytes_and_no_key_reads(tmp_path, monkeypatch, run_env):
    root = tmp_path / "checkout"
    (root / "android").mkdir(parents=True)
    keystore = tmp_path / "lab.jks"
    keystore.write_bytes(b"fixture key must never be read")
    config = tmp_path / "lab.properties"
    # Java Properties encoding for Windows absolute paths.
    path_value = str(keystore).replace("\\", "\\\\").replace(":", "\\:")
    raw = f"storeFile={path_value}\nstorePassword=private-fixture\nkeyAlias=release\nkeyPassword=private-fixture\n".encode()
    config.write_bytes(raw)
    monkeypatch.setenv("ANDROID_RELEASE_CERT_SHA256", "d" * 64)
    monkeypatch.setenv("QLH_RELEASE_ANDROID_KEYSTORE_PROPERTIES", str(config))
    monkeypatch.setattr(build, "observe", lambda *_: copy.deepcopy(INFO))
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path != keystore, "preflight must not open private keystore"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    work = build.preflight(root, "android", REV, REV)
    assert (root / "android/keystore.properties").read_bytes() == raw
    assert json.loads((work / "source/build-info.json").read_text()) == INFO
    assert config.read_bytes() == raw


@pytest.fixture
def signed_bundle(tmp_path, monkeypatch):
    # Synthetic unit-test trust, isolated in pytest tmp_path. Production code
    # only consumes the existing pinned trust directory; it never creates keys.
    monkeypatch.syspath_prepend(str(ROOT / "packaging/packaging"))
    import install_manifest
    import signing
    keys = tmp_path / "synthetic-keys"
    public = tmp_path / "synthetic-trust"
    public.mkdir()
    signing.generate_keypair(keys, key_id="root", role="root")
    (public / "root.pub.json").write_bytes((keys / "root.pub.json").read_bytes())
    gate = build.tool(ROOT, "scripts/package_contract/release_artifacts_gate.py")
    monkeypatch.setattr(gate, "TRUSTED_KEYS", public)
    app = tmp_path / "app"
    app.mkdir()
    (app / "version.txt").write_text(CONTRACT["version"]["product"], encoding="utf-8")
    (app / "release-contract.json").write_text(json.dumps(CONTRACT), encoding="utf-8")
    (app / "build-info.json").write_text(json.dumps(INFO), encoding="utf-8")
    (app / "QLH-Edge-Inference.exe").write_bytes(b"synthetic application, never executed")
    def sign():
        return install_manifest.write_signed_install_manifest(app, app_id="qlh-edge-inference",
            version=CONTRACT["version"]["product"], platform="windows", variant="cpu", package_kind="application",
            private_key_path=keys / "root.key", trusted_keys_dir=public, runtime_profile="llama_cpp_only")
    sign()
    return app, gate, sign


def test_static_build_checks_execute_trusted_deep_verifier_without_app_launch(signed_bundle, tmp_path, monkeypatch):
    app, gate, _ = signed_bundle
    monkeypatch.setattr(gate, "probe_service", lambda *a, **k: pytest.fail("no application execution in build gate"))
    manifest = build.verified_bundle(gate, app, "windows", "cpu", CONTRACT, INFO, tmp_path / "verify")
    assert "build-info.json" in {entry["path"] for entry in manifest["files"]}
    assert (tmp_path / "verify/verify.log").is_file()


@pytest.mark.parametrize("mutation", ["payload", "unsigned-info", "wrong-tree", "untrusted"])
def test_actual_signed_bundle_corruption_and_binding_fail(signed_bundle, tmp_path, mutation):
    app, gate, sign = signed_bundle
    if mutation == "payload":
        (app / "QLH-Edge-Inference.exe").write_bytes(b"tampered")
    elif mutation in ("unsigned-info", "wrong-tree"):
        info = copy.deepcopy(INFO)
        info["source_tree_sha256"] = "c" * 64
        (app / "build-info.json").write_text(json.dumps(info))
        if mutation == "wrong-tree":
            (app / "manifest/install-manifest.json").unlink()
            sign()  # same revision but different authenticated source bytes
    else:
        gate.TRUSTED_KEYS = tmp_path / "empty-trust"
        gate.TRUSTED_KEYS.mkdir()
    with pytest.raises(build.BuildError):
        build.verified_bundle(gate, app, "windows", "cpu", CONTRACT, INFO, tmp_path / "verify")


def test_windows_stages_only_signed_file_set_from_packaging_dist(tmp_path, monkeypatch):
    calls = []
    def verified(gate, app, platform, variant, contract, expected, work):
        calls.append((app, variant))
        return {"fixture": True}
    def clone(app, manifest, target):
        target.mkdir(parents=True)
        (target / "authenticated.exe").write_bytes(b"signed fixture")
    monkeypatch.setattr(build, "verified_bundle", verified)
    output = tmp_path / "out"
    build.stage_windows(tmp_path, output, INFO, types.SimpleNamespace(_clone_bundle=clone), CONTRACT, tmp_path / "work")
    assert calls == [(tmp_path / "packaging/dist/QLH-Edge-Inference", "cpu"),
                     (output / "QLH-Edge-Inference", "cpu"),
                     (tmp_path / "packaging/dist/QLH-Launcher", "launcher"),
                     (output / "QLH-Launcher", "launcher")]


def test_linux_extracts_and_authenticates_deb_without_installing(tmp_path, monkeypatch):
    put(tmp_path, "packaging/packaging/linux/qlh-edge-inference-cpu_fixture.deb", "synthetic deb")
    commands = []
    def command(args):
        commands.append([str(arg) for arg in args])
        if "--field" in args:
            return f"Package: qlh-edge-inference-cpu\nVersion: {CONTRACT['version']['product']}\nArchitecture: amd64\n"
        assert "--extract" in args
        return ""
    monkeypatch.setattr(build, "capture", command)
    checked = []
    monkeypatch.setattr(build, "verified_bundle", lambda *a: checked.append(a[1:4]))
    build.stage_linux(tmp_path, tmp_path / "out", INFO, object(), CONTRACT, tmp_path / "work")
    assert [c[1] for c in commands] == ["--field", "--extract"]
    assert checked[0][0] == tmp_path / "work/extracted/opt/qlh-edge-inference"
    assert checked[0][1:] == ("linux", "cpu")
    assert (tmp_path / "out/qlh-edge-inference-cpu_fixture.deb").read_text() == "synthetic deb"


def apk(tmp_path, name="app-release.apk", info=None):
    path = tmp_path / "android/app/build/outputs/apk/release" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = (183).to_bytes(2, "little")
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("lib/arm64-v8a/libqlh_llama_jni.so", "lib/arm64-v8a/libc++_shared.so"):
            archive.writestr(name, header)
        archive.writestr("AndroidManifest.xml", b"fixture")
        archive.writestr("classes.dex", b"fixture")
        archive.writestr("assets/build-info.json", json.dumps(info or INFO))
    return path


@pytest.mark.parametrize("change", ["valid", "debug", "unsigned", "wrong-pin", "wrong-source", "signature-fails"])
def test_android_actual_inspection_and_static_signature_metadata_failures(tmp_path, monkeypatch, change):
    info = copy.deepcopy(INFO)
    if change == "wrong-source":
        info["source_revision"] = "c" * 40
    apk(tmp_path, "app-release-unsigned.apk" if change == "unsigned" else "app-release.apk", info)
    gate = build.tool(ROOT, "scripts/package_contract/release_artifacts_gate.py")
    monkeypatch.setattr(gate, "android_tool", lambda name: name)
    monkeypatch.setattr(gate, "android_device_smoke", lambda *a, **k: pytest.fail("no device execution"))
    pin = "d" * 64
    monkeypatch.setenv("ANDROID_RELEASE_CERT_SHA256", pin)
    commands = []
    def capture(args):
        commands.append(args)
        if args[0] == "aapt":
            return f"package: name='com.qlh.inference' versionCode='{CONTRACT['version']['android_code']}' versionName='{CONTRACT['version']['product']}'\n" + ("application-debuggable\n" if change == "debug" else "")
        assert args[0] == "apksigner"
        if change == "signature-fails":
            raise build.BuildError("signature rejected")
        return "Signer #1 certificate SHA-256 digest: " + ("e" * 64 if change == "wrong-pin" else pin) + "\n"
    monkeypatch.setattr(build, "capture", capture)
    if change == "valid":
        build.stage_android(tmp_path, tmp_path / "out", INFO, gate, CONTRACT, tmp_path / "work")
        assert (tmp_path / "out/app-release.apk").is_file()
        assert [c[0] for c in commands] == ["aapt", "apksigner"]
    else:
        with pytest.raises((ValueError, build.BuildError)):
            build.stage_android(tmp_path, tmp_path / "out", INFO, gate, CONTRACT, tmp_path / "work")
        assert not (tmp_path / "out/app-release.apk").exists()


def add_ui_transcripts(bundle, data):
    refs = []
    for run in data["runs"]:
        ui_path = bundle.parent / run["ui"]["path"]
        ui = json.loads(ui_path.read_text(encoding="utf-8"))
        for moment in ("before", "after"):
            name = run["mode"] + "/transcript-" + moment + ".txt"
            path = bundle.parent / name
            path.write_bytes(("synthetic transcript " + moment).encode())
            ref = {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            ui["transcript_" + moment] = ref
            refs.append(ref)
        ui_path.write_text(json.dumps(ui), encoding="utf-8")
        run["ui"]["sha256"] = hashlib.sha256(ui_path.read_bytes()).hexdigest()
    bundle.write_text(json.dumps(data), encoding="utf-8")
    return refs


def test_product_complete_attachment_closure_roundtrips_without_fact_mutation(product_fixture, tmp_path, monkeypatch, run_env):
    bundle, original = product_fixture
    transcript_refs = add_ui_transcripts(bundle, original)
    product_gate = build.tool(ROOT, "scripts/product_relay_gate.py")
    raw = bundle.read_bytes()
    monkeypatch.setattr(build, "tool", lambda *_: product_gate)
    monkeypatch.setattr(build, "observe", lambda *_: copy.deepcopy(INFO))
    monkeypatch.setenv("QLH_RELEASE_PRODUCT_BUNDLE", str(bundle))
    # An unrelated private fixture in the lab root must never be uploaded.
    (bundle.parent / "private.key").write_bytes(b"unreferenced synthetic private data")
    output = build.product(tmp_path, REV, REV)
    assert (output / "bundle.json").read_bytes() == raw == bundle.read_bytes()
    assert not (output / "private.key").exists()
    assert product_gate.validate_bundle(output / "bundle.json", REV)["passed"]
    for run in original["runs"]:
        topology = json.loads((bundle.parent / run["topology"]["path"]).read_text(encoding="utf-8"))
        for node in topology["nodes"]:
            name = node["raw_evidence"]["path"]
            assert (output / name).read_bytes() == (bundle.parent / name).read_bytes()
    assert (output / "LL/screen.svg").is_file()
    for ref in transcript_refs:
        assert (output / ref["path"]).read_bytes() == (bundle.parent / ref["path"]).read_bytes()
    assert json.loads((output / "bundle.json").read_text()) == original


@pytest.mark.parametrize("mutation", ["missing-variable", "missing-attachment", "changed-attachment", "nested-attachment", "dirty", "missing-clean", "wrong-revision"])
def test_product_missing_or_unverifiable_input_is_fatal_before_upload(product_fixture, tmp_path, monkeypatch, run_env, mutation):
    bundle, data = product_fixture
    gate = build.tool(ROOT, "scripts/product_relay_gate.py")
    monkeypatch.setattr(build, "tool", lambda *_: gate)
    monkeypatch.setattr(build, "observe", lambda *_: copy.deepcopy(INFO))
    monkeypatch.setenv("QLH_RELEASE_PRODUCT_BUNDLE", str(bundle))
    if mutation == "missing-variable":
        monkeypatch.delenv("QLH_RELEASE_PRODUCT_BUNDLE")
    elif mutation == "missing-attachment":
        (bundle.parent / "LL/screen.svg").unlink()
    elif mutation == "changed-attachment":
        topology = json.loads((bundle.parent / data["runs"][0]["topology"]["path"]).read_text(encoding="utf-8"))
        (bundle.parent / topology["nodes"][0]["raw_evidence"]["path"]).write_text("modified")
    elif mutation == "nested-attachment":
        refs = add_ui_transcripts(bundle, data)
        (bundle.parent / refs[0]["path"]).write_bytes(b"changed transcript")
    elif mutation == "dirty":
        data["source_dirty"] = True
    elif mutation == "missing-clean":
        data.pop("source_dirty")
    else:
        data["source_revision"] = "c" * 40
    bundle.write_text(json.dumps(data))
    before = bundle.read_bytes()
    with pytest.raises(build.BuildError):
        build.product(tmp_path, REV, REV)
    assert bundle.read_bytes() == before
    assert not (build.work_path(tmp_path, "product") / "stage").exists()


def test_copy_rejects_mutation_and_never_overwrites_staged_artifact(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"new")
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build.copy_file(source, target)
    assert target.read_bytes() == b"existing"


def test_verifier_nonzero_cannot_be_overridden_by_success_json(monkeypatch):
    monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=7, stdout='{"ok":true}'))
    with pytest.raises(build.BuildError):
        build.capture(["synthetic-verifier"])

@pytest.mark.parametrize("value", [None, "relative-secret-directory/private-release.key"])
def test_cli_reports_missing_signing_interface_without_values_or_paths(tmp_path, monkeypatch, capsys, run_env, value):
    if value is None:
        monkeypatch.delenv("QLH_SIGNING_KEY", raising=False)
    else:
        monkeypatch.setenv("QLH_SIGNING_KEY", value)
    assert build.main(["preflight", "--platform", "windows", "--root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    report = json.loads(output.err)
    assert report["command"] == "preflight" and report["platform"] == "windows"
    assert report["passed"] is False and "QLH_SIGNING_KEY" in report["reason"]
    assert not output.out and str(tmp_path) not in output.err
    assert value is None or value not in output.err


@pytest.mark.parametrize("exception", [ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError])
def test_cli_suppresses_external_diagnostic_paths_and_secret_values(tmp_path, monkeypatch, capsys, exception):
    sensitive = "PASSWORD=DO_NOT_ECHO C:/private-runner/config/keystore.properties"
    def fail(*args):
        raise exception(sensitive)
    monkeypatch.setattr(build, "stage", fail)
    assert build.main(["stage", "--platform", "android", "--root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    report = json.loads(output.err)
    assert report["command"] == "stage" and report["platform"] == "android" and report["passed"] is False
    assert report["reason"] == "Source inventory or static artifact verifier failed; external error details suppressed"
    assert not output.out and sensitive not in output.err and str(tmp_path) not in output.err
