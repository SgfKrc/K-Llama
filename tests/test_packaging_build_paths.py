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
LAUNCHER_BUILD = ROOT / "packaging/packaging/build-launcher.bat"


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
    for name in ("qlh_launcher.py", "diagnose.py", "repair.py", "data_retention.py",
                 "update_core.py", "updater.py", "version_store.py", "launcher_slots.py",
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


WINDOWS_TOOL = r'''
import json, os, pathlib, sys
root = pathlib.Path(os.environ['CASE_CORE'])
event = {'tool': pathlib.Path(sys.argv[0]).name, 'args': sys.argv[1:], 'cwd': str(pathlib.Path.cwd())}
with open(os.environ['CASE_EVENTS'], 'a') as f:
    f.write(json.dumps(event) + '\n')
args = sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == 'PyInstaller.py':
    if args[0] == 'packaging\\qlh-launcher.spec':
        out = root / 'packaging/dist/QLH-Launcher/QLH-Launcher.exe'
    else:
        assert args[0] == 'packaging\\qlh-install-manifest.spec'
        out = root / 'packaging/dist/QLH-Launcher/tools/QLH-Install-Manifest.exe'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('OFFLINE MOCK ONLY, NOT AN EXECUTABLE')
elif pathlib.Path(sys.argv[0]).name == 'install_manifest.py':
    out = root / 'packaging/dist/QLH-Launcher/manifest/install-manifest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{}')
'''


@pytest.fixture
def windows_checkout(tmp_path):
    if os.name != "nt":
        pytest.skip("Actual Windows batch execution requires cmd.exe")
    core = tmp_path / "checkout with spaces"
    packaging = core / "packaging"
    builder = core / "packaging/packaging/build-launcher.bat"
    builder.parent.mkdir(parents=True)
    shutil.copyfile(LAUNCHER_BUILD, builder)
    # The actual test interpreter executes only isolated stub modules/scripts.
    # No packaging venv, pip, compiler or signer from the workspace is invoked.
    interpreter = packaging / ".venv-packaging/Scripts/python.exe"
    interpreter.parent.mkdir(parents=True)
    shutil.copyfile(sys.executable, interpreter)
    config = Path(sys.prefix) / "pyvenv.cfg"
    if config.is_file():
        shutil.copyfile(config, interpreter.parent.parent / "pyvenv.cfg")
    else:
        put(interpreter.parent.parent, "pyvenv.cfg", "home = " + str(Path(sys.executable).parent) + "\n")
    for name in ("scripts/version_contract.py", "packaging/packaging/release_build_provenance.py",
                 "packaging/packaging/install_manifest.py"):
        put(core, name, WINDOWS_TOOL)
    put(core, "packaging/packaging/launcher-version.txt", "0.1.8.2\n")
    # Copy the real Inno input: the mock resolves its actual source/output paths.
    for name in ("setup-launcher.iss", "release-versions.issinc"):
        shutil.copyfile(ROOT / "packaging/packaging" / name, builder.parent / name)
    put(core, "packaging/packaging/pubkeys/fixture.pub.json", "{}")
    modules = tmp_path / "mock modules"
    for name in ("pip.py", "PyInstaller.py"):
        put(modules, name, WINDOWS_TOOL)
    # Read the real .iss declarations to resolve its source tree and outputs;
    # the mock never invokes Inno or packages an executable.
    compiler = put(tmp_path, "mock tools/iscc.py", textwrap.dedent(r'''
        import os, pathlib, re, sys
        assert len(sys.argv) == 3, sys.argv
        version = sys.argv[1].removeprefix('/DMyAppVersion=')
        assert version == '0.1.8.2'
        source = pathlib.Path(sys.argv[2]).resolve()
        assert source.is_file() and pathlib.Path.cwd() == source.parent
        contents = source.read_text(encoding='utf-8-sig')
        include = re.search(r'^#include "([^"]+)"', contents, re.M).group(1)
        assert (source.parent / include).is_file()
        app = re.search(r'^#define MySourceDir\s+"([^"]+)"', contents, re.M).group(1)
        assert (source.parent / app / 'QLH-Launcher.exe').is_file()
        output_dir = re.search(r'^OutputDir=(.+)$', contents, re.M).group(1).strip()
        output_name = re.search(r'^OutputBaseFilename=(.+)$', contents, re.M).group(1).strip()
        output_name = output_name.replace('{#MyAppVersion}', version) + '.exe'
        core = pathlib.Path(os.environ['CASE_CORE'])
        (core / 'iscc-arguments.txt').write_text('\n'.join(sys.argv[1:]))
        mode = os.environ['CASE_MODE']
        if mode == 'compiler_failure':
            raise SystemExit(37)
        if mode == 'no_setup_output':
            raise SystemExit(0)
        target = source.parent / output_dir / output_name
        if mode == 'wrong_setup_output':
            target = core / 'wrong-output' / output_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('OFFLINE MOCK ONLY, NOT AN EXECUTABLE')
    ''').lstrip())
    iscc = put(tmp_path, "mock tools/iscc.cmd",
               '@echo off\n"%CASE_PYTHON%" -B "%CASE_COMPILER%" %*\nexit /b %errorlevel%\n')
    iscc.write_bytes(iscc.read_bytes().replace(b"\n", b"\r\n"))
    events = tmp_path / "events.jsonl"
    env = {**os.environ, "CASE_CORE": str(core), "CASE_EVENTS": str(events),
           "QLH_CORE_ROOT": str(core), "QLH_SIGNING_KEY": "unused-offline-placeholder",
           "ISCC": str(iscc), "CASE_PYTHON": sys.executable, "CASE_COMPILER": str(compiler),
           "PYTHONPATH": str(modules), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUTF8": "1"}
    return core, builder, env, events


def run_windows(fixture, mode="build"):
    core, builder, env, events = fixture
    result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(builder)],
                            cwd=core.parent, env={**env, "CASE_MODE": mode},
                            capture_output=True, timeout=35)
    calls = [json.loads(s) for s in events.read_text().splitlines()] if events.exists() else []
    return result, calls


def test_launcher_executes_real_batch_with_consistent_inno_input_output(windows_checkout):
    core, _, _, _ = windows_checkout
    result, calls = run_windows(windows_checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(e["tool"] == "PyInstaller.py" for e in calls)
    assert (core / "packaging/packaging/dist/QLH-Launcher-Setup-v0.1.8.2.exe").is_file()
    assert (core / "packaging/packaging/dist/QLH-Launcher-v0.1.8.2.zip").is_file()
    assert not (core / "packaging/packaging/packaging").exists()
    args = (core / "iscc-arguments.txt").read_text()
    assert '/DMyAppVersion=0.1.8.2' in args
    assert str(core / "packaging/packaging/setup-launcher.iss") in args


@pytest.mark.parametrize("name", ["setup-launcher.iss", "release-versions.issinc"])
def test_launcher_missing_inno_input_fails_before_python_or_build(windows_checkout, name):
    core, builder, _, _ = windows_checkout
    (builder.parent / name).unlink()
    result, calls = run_windows(windows_checkout)
    assert result.returncode != 0
    assert not calls
    assert not (core / "iscc-arguments.txt").exists()
    assert not (core / "packaging/dist").exists()


@pytest.mark.parametrize("mode", ["compiler_failure", "no_setup_output", "wrong_setup_output"])
def test_launcher_rejects_inno_failure_or_missing_output(windows_checkout, mode):
    core, _, _, _ = windows_checkout
    result, _ = run_windows(windows_checkout, mode)
    assert result.returncode != 0, result.stdout + result.stderr
    assert (core / "iscc-arguments.txt").is_file()
    assert not (core / "packaging/packaging/dist/QLH-Launcher-Setup-v0.1.8.2.exe").exists()


def test_launcher_rejects_stale_setup_before_building(windows_checkout):
    core, _, _, _ = windows_checkout
    setup = put(core, "packaging/packaging/dist/QLH-Launcher-Setup-v0.1.8.2.exe", "stale fixture")
    result, calls = run_windows(windows_checkout)
    assert result.returncode != 0
    assert [e["tool"] for e in calls] == ["version_contract.py"]
    assert setup.read_text() == "stale fixture"
    assert not (core / "iscc-arguments.txt").exists()


def test_launcher_without_inno_keeps_existing_optional_setup_behavior(windows_checkout):
    core, builder, env, events = windows_checkout
    env["ISCC"] = str(core / "missing compiler/ISCC.exe")
    result, calls = run_windows((core, builder, env, events))
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(e["tool"] == "install_manifest.py" for e in calls)
    assert (core / "packaging/packaging/dist/QLH-Launcher-v0.1.8.2.zip").is_file()
    assert not (core / "iscc-arguments.txt").exists()
