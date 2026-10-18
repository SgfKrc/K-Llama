"""Execute the CPU export wiring with a synthetic APP; never build or sign."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "packaging/packaging/build-cpu.bat"
HELPER = ROOT / "src/runtime_frozen_source.py"
EXPORT = '"%PYTHON%" -B "%QLH_CORE_ROOT%\\src\\runtime_frozen_source.py"'
PROFILE_GUARD = 'if /i "!QLH_BUILD_PROFILE!"=="llama_cpp_only" ('


def _script():
    return BATCH.read_text(encoding="utf-8")


def _integration_block():
    text = _script()
    start = text.index('    "%PYTHON%" -B packaging\\release_build_provenance.py stage')
    signing = text.index('    "%PYTHON%" packaging\\install_manifest.py build', start)
    end = text.index("    if errorlevel 1 exit /b 1", signing)
    return text[start:end + len("    if errorlevel 1 exit /b 1")]


def test_cpu_exports_complete_staged_edge_tree_before_signing():
    text = _script()
    export = text.index(EXPORT)
    stage = text.index("release_build_provenance.py stage")
    signing = text.index("packaging\\install_manifest.py build")
    assert stage < export < signing
    for previous in (
        "qlh-tui-chat.spec", "qlh-model-tools.spec", "qlh-data-retention.spec",
        "qlh-install-manifest.spec", 'xcopy /e /i /y "packaging\\pubkeys"',
    ):
        assert text.index(previous) < stage
    assert PROFILE_GUARD in text[stage:export]
    assert '--root "dist\\QLH-Edge-Inference"' in text[export:signing]
    assert "if errorlevel 1 exit /b !errorlevel!" in text[export:signing]
    assert text.count("runtime_frozen_source.py") == 1
    assert text.index("if not defined QLH_SIGNING_KEY") < stage


def _put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def pipeline(tmp_path):
    if os.name != "nt":
        pytest.skip("executes the real cmd.exe integration block")
    # Paths contain spaces to exercise the exact quoted production command.
    core = tmp_path / "source with spaces"
    packaging = core / "packaging"
    app = packaging / "dist/QLH-Edge-Inference"
    for relative in (
        "QLH-Edge-Inference.exe", "QLH-TUI-Chat/QLH-TUI-Chat.exe",
        "QLH-Model-Tools/QLH-Model-Tools.exe", "tools/QLH-Data-Retention.exe",
        "tools/QLH-Install-Manifest.exe", "pubkeys/test-public.json",
    ):
        _put(app, relative, "synthetic bytes, never executed\n")
    _put(packaging, "packaging/release_build_provenance.py", '''
import json, os, pathlib, sys
with open(os.environ["QLH_TEST_EVENTS"], "a") as stream: stream.write("stage\\n")
if os.environ.get("QLH_TEST_STAGE_FAIL"): raise SystemExit(31)
root = pathlib.Path(sys.argv[sys.argv.index("--app-root") + 1])
if os.environ.get("QLH_TEST_NO_BUILD_INFO"): raise SystemExit(0)
(root / "build-info.json").write_text(json.dumps({"schema_version": 1,
    "source_revision": "5d157b05eab6911328b3edcb87d9188eb7d09e85", "source_dirty": False}))
''')
    _put(core, "src/runtime_frozen_source.py", '''
import os, runpy
with open(os.environ["QLH_TEST_EVENTS"], "a") as stream: stream.write("export\\n")
if os.environ.get("QLH_TEST_EXPORT_FAIL"): raise SystemExit(47)
runpy.run_path(os.environ["QLH_TEST_REAL_EXPORTER"], run_name="__main__")
''')
    # Use the real unsigned inventory API as a signing-step substitute. No key
    # is read; this proves the source observation will belong to the signed set.
    _put(packaging, "packaging/install_manifest.py", '''
import json, os, pathlib, sys
with open(os.environ["QLH_TEST_EVENTS"], "a") as stream: stream.write("sign\\n")
sys.path.insert(0, os.environ["QLH_TEST_MANIFEST_MODULES"])
from install_manifest import build_install_manifest
root = pathlib.Path(sys.argv[sys.argv.index("--root") + 1])
manifest = build_install_manifest(root, app_id="qlh-edge-inference", version="0.1.8.3",
    platform="windows", variant="cpu", package_kind="application", runtime_profile=os.environ["QLH_BUILD_PROFILE"].lower())
path = root / "manifest/install-manifest.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest), encoding="utf-8")
''')
    batch = _put(packaging, "wiring.bat", "@echo off\nsetlocal enabledelayedexpansion\n"
                 + _integration_block() + "\nexit /b 0\n")
    # cmd.exe expects CRLF. Preserve the actual nested parentheses from the
    # enclosing production block so delayed errorlevel expansion is exercised.
    raw = batch.read_text(encoding="utf-8")
    batch.write_bytes(raw.replace("\n", "\r\n").encode("utf-8"))
    events = tmp_path / "events.log"
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("QLH_")}
    env.update(PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               QLH_CORE_ROOT=str(core), QLH_BUILD_INFO=str(core / "snapshot.json"),
               QLH_SIGNING_KEY="unused-test-placeholder", APP_VERSION="0.1.8.3",
               QLH_TEST_EVENTS=str(events), QLH_TEST_REAL_EXPORTER=str(HELPER),
               QLH_TEST_MANIFEST_MODULES=str(ROOT / "packaging/packaging"))

    def run(profile="llama_cpp_only", **overrides):
        result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(batch)],
                                cwd=packaging, env={**env, "QLH_BUILD_PROFILE": profile, **overrides},
                                capture_output=True, text=True, errors="replace", timeout=20)
        observed = events.read_text().splitlines() if events.exists() else []
        return result, observed, app

    return run


@pytest.mark.parametrize("profile", ["llama_cpp_only", "LLAMA_CPP_ONLY", "torch_cpu"])
def test_real_cmd_orders_stage_export_and_sign_only_for_edge(pipeline, profile):
    result, events, app = pipeline(profile)
    assert result.returncode == 0, result.stdout + result.stderr
    edge = profile.lower() == "llama_cpp_only"
    assert events == (["stage", "export", "sign"] if edge else ["stage", "sign"])
    observation = app / "product-package-observation.json"
    install = json.loads((app / "manifest/install-manifest.json").read_text())
    files = {entry["path"]: entry for entry in install["files"]}
    assert (observation.name in files) is edge
    if edge:
        assert files[observation.name]["sha256"] == hashlib.sha256(observation.read_bytes()).hexdigest()
        exported = json.loads(observation.read_text())
        assert exported["executable_relative"] == "QLH-Edge-Inference.exe"
        assert exported["source_dirty"] is False
        assert "manifest/install-manifest.json" not in {entry["path"] for entry in exported["files"]}
        assert {"build-info.json", "pubkeys/test-public.json", "tools/QLH-Install-Manifest.exe"} <= {
            entry["path"] for entry in exported["files"]}


@pytest.mark.parametrize(("override", "expected_code", "events"), [
    ({"QLH_TEST_STAGE_FAIL": "1"}, 1, ["stage"]),
    ({"QLH_TEST_EXPORT_FAIL": "1"}, 47, ["stage", "export"]),
    ({"QLH_TEST_NO_BUILD_INFO": "1"}, 1, ["stage", "export"]),
])
def test_real_cmd_failure_stops_before_signing(pipeline, override, expected_code, events):
    result, observed, app = pipeline(**override)
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert observed == events
    assert not (app / "manifest/install-manifest.json").exists()
    assert not (app / "product-package-observation.json").exists()
