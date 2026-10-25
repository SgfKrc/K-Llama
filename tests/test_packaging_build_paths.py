"""Execute packaging entrypoints against synthetic, isolated toolchains only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINUX_BUILD = ROOT / "packaging/packaging/linux/build-deb.sh"


def put(root: Path, relative: str, contents: str = "fixture\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8", newline="\n")
    return path


@pytest.fixture(scope="module")
def docker():
    executable = shutil.which("docker")
    if not executable:
        pytest.skip("Docker is required to execute Linux packaging paths")
    image = "qlh-release-gate:bookworm"
    result = subprocess.run([executable, "image", "inspect", image],
                            capture_output=True, timeout=20)
    if result.returncode:
        pytest.skip("Pre-provisioned Linux gate image/daemon unavailable; never pull images")
    return executable, image


# This driver runs in the existing Linux image. The mount contains only the
# fixture checkout and mocks, never repository source, host configuration or keys.
LINUX_DRIVER = r'''
import json, os, pathlib, shutil, subprocess, sys
case = pathlib.Path('/case')
core = case / 'checkout with spaces'
events = case / 'events.jsonl'
bin_dir = case / 'bin'
bin_dir.mkdir()
shim = "#!" + sys.executable + "\n" + (case / 'linux_tools.py').read_text()
for name in ['python3', 'node', 'npm', 'npx', 'git', 'cmake', 'c++', 'make', 'dpkg-deb']:
    path = bin_dir / name
    path.write_text(shim)
    path.chmod(0o755)
script = core / 'packaging/packaging/linux/build-deb.sh'
syntax = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
if syntax.returncode:
    raise SystemExit(syntax.stderr)
env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
       'CASE_ROOT': str(case), 'CASE_CORE': str(core), 'CASE_EVENTS': str(events),
       'QLH_CORE_ROOT': str(core), 'QLH_SIGNING_KEY': str(case / 'unopened-placeholder')}
args = ['bash', str(script), 'cpu']
if os.environ['CASE_MODE'] == 'preflight':
    args.append('--preflight-only')
result = subprocess.run(args, cwd=case, env=env, capture_output=True, text=True, timeout=30)
print(json.dumps({'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
                  'events': [json.loads(s) for s in events.read_text().splitlines()] if events.exists() else [],
                  'deb_outputs': [p.name for p in script.parent.glob('*.deb')],
                  'build_dir_exists': (core / 'build').exists()}))
'''

LINUX_TOOLS = r'''
import json, os, pathlib, sys
name, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
core = pathlib.Path(os.environ['CASE_CORE'])
mode = os.environ['CASE_MODE']
event = {'tool': name, 'args': args, 'cwd': str(pathlib.Path.cwd())}
with open(os.environ['CASE_EVENTS'], 'a') as f:
    f.write(json.dumps(event) + '\n')
if name == 'node':
    print('20')
elif name == 'npm':
    assert args == ['ci', '--silent']
    assert pathlib.Path('package.json').is_file() and pathlib.Path('package-lock.json').is_file()
    if mode == 'npm_failure':
        raise SystemExit(23)
elif name == 'npx':
    assert args == ['vite', 'build']
    if mode != 'missing_dist':
        out = pathlib.Path('dist')
        out.mkdir(exist_ok=True)
        marker = 'nested-frontend' if pathlib.Path.cwd() == core / 'frontend_cybergothic/frontend_cybergothic' else 'wrong-root'
        (out / 'index.html').write_text(marker)
elif name in ('python3', 'python'):
    if '-m' in args and args[args.index('-m') + 1] == 'venv':
        target = pathlib.Path(args[-1]) / 'bin'
        target.mkdir(parents=True)
        for tool in ('python', 'pip'):
            path = target / tool
            path.write_bytes(pathlib.Path(sys.argv[0]).read_bytes())
            path.chmod(0o755)
    elif any(a.endswith('build_llama_quantize.py') for a in args):
        out = pathlib.Path(args[args.index('--output-root') + 1]) / 'packages/linux-x86_64'
        out.mkdir(parents=True)
        (out / 'manifest.json').write_text('{}')
        (out / 'llama-quantize').write_text('offline placeholder')
        (out / 'llama-quantize').chmod(0o755)
elif name == 'dpkg-deb':
    assert args[0] == '--build'
    stage = pathlib.Path(args[1])
    staged = stage / 'opt/qlh-edge-inference/frontend_cybergothic/dist/index.html'
    with open(os.environ['CASE_EVENTS'], 'a') as f:
        f.write(json.dumps({'tool': 'staged-frontend', 'contents': staged.read_text()}) + '\n')
    pathlib.Path(args[2]).write_text('OFFLINE MOCK ONLY, NOT A DEBIAN PACKAGE\n')
'''


def linux_fixture(tmp_path: Path, mode: str) -> Path:
    core = tmp_path / "checkout with spaces"
    script = put(core, "packaging/packaging/linux/build-deb.sh", LINUX_BUILD.read_text(encoding="utf-8"))
    put(core, ".git", "fixture marker; not a provenance claim\n")
    put(core, "src/fixture.py")
    put(core, "release-contract.json", "{}")
    put(core, "scripts/version_contract.py")
    put(core, "packaging/packaging/version.txt", "0.1.8.3\n")
    frontend = "frontend_cybergothic/frontend_cybergothic/"
    for relative in ("package.json", "package-lock.json"):
        if mode != "missing_" + relative:
            put(core, frontend + relative, "{}")
        # A plausible obsolete root must never substitute for the nested root.
        put(core, "frontend_cybergothic/" + relative, "{}")
    put(core, "frontend_cybergothic/dist/index.html", "stale-root-decoy")
    for name in ("app_maintenance.py", "packaging_release_contract.py", "runtime_guard.py", "diagnose.py", "repair.py", "data_retention.py",
                 "update_core.py", "updater.py", "version_store.py",
                 "install_manifest.py", "signing.py", "launcher.py", "requirements-cpu.txt",
                 "release_build_provenance.py", "model_supply_resources.py", "pubkeys/fixture.pub.json"):
        put(core, "packaging/packaging/" + name)
    for name in ("launcher.py", "bjtu", "qlh-env-register", "qlh-edge-inference.desktop",
                 "qlh-edge-inference.service", "postinst", "prerm", "postrm"):
        put(script.parent, name)
    put(script.parent, "control-cpu", "Package: offline-fixture\nVersion: 0\n")
    put(tmp_path, "unopened-placeholder", "not a key; tools never open this file\n")
    put(tmp_path, "linux_driver.py", LINUX_DRIVER)
    put(tmp_path, "linux_tools.py", LINUX_TOOLS)
    return core


def run_linux(tmp_path: Path, docker, mode: str) -> dict:
    linux_fixture(tmp_path, mode)
    executable, image = docker
    result = subprocess.run([
        executable, "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,exec,nosuid,size=64m",
        "--mount", f"type=bind,source={tmp_path},target=/case",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "CASE_MODE=" + mode,
        image, "python3", "-B", "/case/linux_driver.py",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_linux_executes_nested_build_and_copies_its_real_output(tmp_path, docker):
    result = run_linux(tmp_path, docker, "build")
    assert result["returncode"] == 0, result
    frontend = "/case/checkout with spaces/frontend_cybergothic/frontend_cybergothic"
    calls = [e for e in result["events"] if e["tool"] in ("npm", "npx")]
    assert [e["tool"] for e in calls] == ["npm", "npx"]
    assert all(e["cwd"] == frontend for e in calls)
    assert {"tool": "staged-frontend", "contents": "nested-frontend"} in result["events"]
    assert len(result["deb_outputs"]) == 1


@pytest.mark.parametrize("mode", ["missing_package.json", "missing_package-lock.json"])
def test_linux_rejects_missing_nested_inputs_before_any_build(tmp_path, docker, mode):
    result = run_linux(tmp_path, docker, mode)
    assert result["returncode"] != 0, result
    assert "FRONTEND_SOURCE_MISSING" in result["stderr"]
    assert not result["events"] and not result["deb_outputs"]
    assert result["build_dir_exists"] is False


def test_linux_preflight_checks_inputs_without_building_or_signing(tmp_path, docker):
    result = run_linux(tmp_path, docker, "preflight")
    assert result["returncode"] == 0, result
    assert not result["deb_outputs"] and not result["build_dir_exists"]
    assert all(e["tool"] not in ("npm", "npx", "dpkg-deb") for e in result["events"])
    assert not any("release_build_provenance.py" in a for e in result["events"] for a in e["args"])


def test_linux_successful_tool_without_dist_cannot_reuse_root_dist(tmp_path, docker):
    result = run_linux(tmp_path, docker, "missing_dist")
    assert result["returncode"] != 0, result
    assert "FRONTEND_BUILD_MISSING" in result["stderr"]
    assert not result["deb_outputs"]
    assert not any(e["tool"] == "staged-frontend" for e in result["events"])


def test_linux_propagates_frontend_failure_without_packaging(tmp_path, docker):
    result = run_linux(tmp_path, docker, "npm_failure")
    assert result["returncode"] == 23, result
    assert not any(e["tool"] in ("npx", "dpkg-deb", "staged-frontend") for e in result["events"])
    assert not result["deb_outputs"]
