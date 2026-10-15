#!/usr/bin/env python3
"""CI-RELEASE-01: fail-closed checks of built Windows/Linux/Android artifacts.

Explicit platforms are required. Missing artifacts, tools, execution or readiness
produce failed/unverified evidence and a nonzero exit. Source-only checks,
--help and process survival are never release acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
from release_gate_runtime import Report, isolated_environment, json_output, probe_service, run_command
from android_runtime_probe import observe_runtime

ROOT = MODULE_DIR.parents[1]
CONTRACT_PATH = ROOT / "release-contract.json"
LAUNCH_PROBE_ENV = "QLH_RELEASE_GATE_LAUNCH_PROBE"
SCHEMA_VERSION = 2
TRUSTED_KEYS = ROOT / "packaging" / "packaging" / "pubkeys"
DEFAULT_IMAGE = "qlh-release-gate:bookworm"
WINDOWS_BUNDLES = (
    ("cpu", "QLH-Edge-Inference", "QLH-Edge-Inference.exe", "product", "application"),
    ("cuda", "QLH-Edge-Inference-CUDA", "QLH-Edge-Inference.exe", "product", "application"),
    ("launcher", "QLH-Launcher", "QLH-Launcher.exe", "launcher", "launcher"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def validate_contract(contract: dict[str, Any]) -> None:
    version, profile = contract.get("version"), contract.get("profile")
    if contract.get("schema_version") != 1 or not isinstance(version, dict) or not isinstance(profile, dict):
        raise ValueError("release contract requires schema_version=1, version and profile objects")
    for key in ("product", "launcher"):
        if not isinstance(version.get(key), str) or not re.fullmatch(r"\d+(?:\.\d+){1,3}", version[key]):
            raise ValueError(f"invalid version.{key}")
    code = version.get("android_code")
    if isinstance(code, bool) or not isinstance(code, int) or code < 1:
        raise ValueError("invalid version.android_code")
    if profile.get("default_node_role") not in ("master", "worker"):
        raise ValueError("invalid profile.default_node_role")
    if not isinstance(profile.get("fixed_environment"), dict):
        raise ValueError("profile.fixed_environment must be an object")


def _work(options: dict[str, Any] | None, platform: str) -> Path:
    base = (options or {}).get("work_dir", ROOT / "build" / "release-artifacts-gate")
    work = Path(base).resolve() / f"{platform}-{uuid.uuid4().hex}"
    work.mkdir(parents=True)
    return work


def _record_command(report: Report, check: str, result: dict[str, Any], **expected: Any) -> dict:
    try:
        value = json_output(result)
        if any(value.get(key) != val for key, val in expected.items()):
            raise ValueError(f"unexpected command result; required {expected}")
        report.ok(check, "real command returned validated JSON", execution=result, result=value)
        return value
    except ValueError as exc:
        report.fail(check, str(exc), execution=result)
        return {}


def verify_tree(app: Path, work: Path) -> tuple[dict, dict]:
    """Reuse canonical verifier and canonical trust, never the artifact's own trust root."""
    result = run_command([
        sys.executable, "-B", str(ROOT / "packaging" / "packaging" / "install_manifest.py"),
        "verify", "--root", str(app), "--level", "deep", "--trusted-keys-dir",
        str(TRUSTED_KEYS), "--json",
    ], cwd=work, env=isolated_environment(work / "verify-state"), log=work / "verify.log", timeout=600)
    if (result.get("executed") is not True or type(result.get("returncode")) is not int
            or result["returncode"] != 0 or result.get("ok") is not True):
        raise ValueError("canonical signature verifier did not execute successfully")
    return result, json_output(result)


def check_revision(report: Report, check: str, info: dict | None, *, authenticated: bool,
                   required: str | None = None, **evidence: Any) -> str | None:
    revision = info.get("source_revision") if isinstance(info, dict) else None
    if not authenticated or not isinstance(info, dict) or info.get("schema_version") != 1:
        report.unverified(check, "authenticated build-info.json unavailable; CLI revision is not build identity",
                          required_revision=required, **evidence)
        return None
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", revision):
        report.fail(check, "build-info.json source_revision must be a full lowercase Git object id", **evidence)
        return None
    if required and revision != required:
        report.fail(check, f"artifact source_revision {revision} != required {required}", **evidence)
        return None
    if "source_dirty" in info and info["source_dirty"] is not False:
        report.fail(check, "artifact was built from modified or unverifiable source", **evidence)
        return None
    report.ok(check, "source revision authenticated with artifact", source_revision=revision,
              revision_verified=True, **evidence)
    return revision


def check_bundle(report: Report, *, app: Path, contract: dict, platform: str,
                 variant: str, version_key: str, kind: str, work: Path,
                 required_revision: str | None = None) -> dict | None:
    prefix = f"{platform}.{variant}"
    version_file = app / "version.txt"
    want = str(contract["version"][version_key])
    try:
        actual = version_file.read_text(encoding="utf-8-sig").strip()
        if actual != want:
            raise ValueError(f"declared {actual!r} != contract {want!r}")
        report.ok(prefix + ".version", want, path=str(version_file))
    except (OSError, ValueError) as exc:
        report.fail(prefix + ".version", str(exc), path=str(version_file))
    manifest_path = app / "manifest" / "install-manifest.json"
    result = None
    try:
        manifest = read_object(manifest_path)
        expected = dict(version=want, platform=platform, package_kind=kind,
                        app_id="qlh-launcher" if kind == "launcher" else "qlh-edge-inference")
        if kind == "application":
            expected["variant"] = variant
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(f"manifest identity mismatch; expected {expected}")
        result, verified = verify_tree(app, work)
        if (result.get("executed") is not True or type(result.get("returncode")) is not int
                or result["returncode"] != 0 or result.get("ok") is not True
                or verified.get("ok") is not True or verified.get("level") != "deep"):
            raise ValueError("deep install verification failed")
        # Bind metadata to the object that the existing verifier actually authenticated.
        if any(verified.get("manifest", {}).get(k) != v for k, v in expected.items()):
            raise ValueError("verified manifest identity mismatch")
        report.ok(prefix + ".integrity", "trusted Ed25519 signature + every file SHA-256",
                  manifest_sha256=sha256(manifest_path), verification=verified, execution=result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.fail(prefix + ".integrity", str(exc), path=str(manifest_path), execution=result)
        report.unverified(prefix + ".revision", "artifact integrity not established", required_revision=required_revision)
        return None
    contract_candidates = (app / "release-contract.json", app / "_internal" / "release-contract.json")
    packaged = next((p for p in contract_candidates if p.is_file()), contract_candidates[0])
    try:
        relative = packaged.relative_to(app).as_posix()
        if relative not in {entry["path"] for entry in manifest["files"]}:
            raise ValueError("bundled release-contract.json is not in the signed file set")
        if read_object(packaged) != contract:
            raise ValueError("bundled release contract differs from canonical contract")
        report.ok(prefix + ".contract", "signed bundled contract matches canonical contract",
                  path=str(packaged), sha256=sha256(packaged))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.fail(prefix + ".contract", str(exc), path=str(packaged))
        report.unverified(prefix + ".revision", "artifact contract not established", required_revision=required_revision)
        return None
    if any(i["check"] == prefix + ".version" for i in report.failures):
        report.unverified(prefix + ".revision", "artifact version not established", required_revision=required_revision)
        return None
    info_path = app / "build-info.json"
    try:
        relative = info_path.relative_to(app).as_posix()
        entries = {entry["path"]: entry for entry in manifest["files"]}
        authenticated = relative in entries and sha256(info_path) == entries[relative]["sha256"]
        info = read_object(info_path) if authenticated else None
    except (OSError, ValueError, KeyError):
        info, authenticated = None, False
    check_revision(report, prefix + ".revision", info, authenticated=authenticated,
                   required=required_revision, path=str(info_path))
    return manifest


def _clone_bundle(app: Path, manifest: dict, target: Path) -> Path:
    """Copy only authenticated application files; user models/state never enter a probe."""
    target.mkdir(parents=True)
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        source, destination = app / relative, target / relative
        if relative.is_absolute() or not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError(f"unsafe manifest path: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != entry["sha256"]:
            raise ValueError(f"artifact changed after verification: {relative}")
    destination = target / "manifest" / "install-manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(app / "manifest" / "install-manifest.json", destination)
    return target


def gate_windows(dist: Path, contract: dict[str, Any], options: dict | None = None) -> Report:
    report, work = Report("windows"), _work(options, "windows")
    enabled = os.environ.get(LAUNCH_PROBE_ENV, "1") != "0"
    selected = (options or {}).get("windows_variants", ("cpu", "launcher"))
    prepared_cpu = None
    for variant, directory, executable, version_key, kind in WINDOWS_BUNDLES:
        if variant not in selected:
            continue
        app = dist.resolve() / directory
        prefix = f"windows.{variant}"
        bundle_work = work / variant
        bundle_work.mkdir()
        manifest = check_bundle(report, app=app, contract=contract, platform="windows",
                                variant=variant, version_key=version_key, kind=kind, work=bundle_work,
                                required_revision=(options or {}).get("require_revision"))
        if not enabled or os.name != "nt" or manifest is None:
            reason = (f"{LAUNCH_PROBE_ENV}=0" if not enabled else
                      "Windows host required" if os.name != "nt" else "artifact prerequisites failed")
            for check in ("dependencies", "roles", "startup"):
                report.unverified(prefix + "." + check, reason + "; not executed")
            continue
        try:
            clone = _clone_bundle(app, manifest, bundle_work / "app")
            if kind == "launcher":
                # Exercise the actual standalone Launcher and its dependency imports, no GUI.
                result = run_command([
                    str(clone / executable), "verify", "--root", str(clone), "--level", "deep",
                    "--trusted-keys-dir", str(TRUSTED_KEYS), "--json",
                ], cwd=clone, env=isolated_environment(bundle_work / "launcher-state"),
                    log=bundle_work / "launcher-verify.log", timeout=600)
                verified = _record_command(report, prefix + ".startup", result, ok=True, level="deep")
                if verified:
                    report.ok(prefix + ".dependencies", "Launcher executed signed-tree verifier", execution=result)
                    if prepared_cpu is None:
                        report.unverified(prefix + ".roles", "no ready CPU application for real Launcher role forwarding")
                    else:
                        forwarding = []
                        default = "client" if contract["profile"]["default_node_role"] == "worker" else "master"
                        for label, role in (("clean", default), ("confirmed-worker", "client")):
                            runtime = probe_service([str(clone / executable), "--headless"], app=clone,
                                target_app=prepared_cpu, work=bundle_work / label, role=role,
                                confirmed_worker=label == "confirmed-worker",
                                timeout=(options or {}).get("startup_timeout", 90))
                            forwarding.append(runtime)
                        (report.ok if all(r["ready"] for r in forwarding) else report.fail)(
                            prefix + ".roles", "real Launcher child readiness and observed application roles",
                            executions=forwarding)
                else:
                    report.unverified(prefix + ".dependencies", "Launcher command did not complete")
                    report.unverified(prefix + ".roles", "Launcher command did not complete")
                continue
            # Complete bundles can run offline; slim bootstrappers would pip-install an external
            # venv and may hardcode port 8000. Refuse that path until a separately staged runtime exists.
            if (clone / "_internal" / "src" / "api_server.py").is_file() and not (
                    clone / "_internal" / "fastapi" / "__init__.py").is_file():
                for check in ("dependencies", "roles", "startup"):
                    report.unverified(prefix + "." + check, "slim external runtime has not been staged/verified")
                continue
            results = []
            default_role = "client" if contract["profile"]["default_node_role"] == "worker" else "master"
            for label, role in (("clean", default_role), ("confirmed-worker", "client")):
                result = probe_service([str(clone / executable), "--headless"], app=clone,
                                       work=bundle_work / label, role=role, confirmed_worker=label == "confirmed-worker",
                                       timeout=(options or {}).get("startup_timeout", 90))
                results.append(result)
                (report.ok if result["ready"] else report.fail)(
                    prefix + ".role." + label, "owned API ready and observed role=" + role
                    if result["ready"] else result.get("error", "not ready"), runtime=result)
            if variant == "cpu" and all(r["ready"] for r in results):
                prepared_cpu = clone
            for check in ("dependencies", "roles", "startup"):
                (report.ok if all(r["ready"] for r in results) else report.fail)(
                    prefix + "." + check, "signed files + actual startup components and role observations",
                    executions=results)
        except (OSError, ValueError) as exc:
            report.fail(prefix + ".runner", str(exc))
    return report


def _linux_result(report: Report, value: dict, *, nonce: str, digest: str, variant: str) -> None:
    """Only a result from this invocation, artifact and mandatory runtime faces is accepted."""
    expected = dict(schema_version=1, gate="linux-installed-artifact", nonce=nonce,
                    artifact_sha256=digest, variant=variant)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("Linux runner schema/challenge/artifact binding mismatch")
    items = value.get("items")
    if not isinstance(items, list):
        raise ValueError("Linux runner items missing")
    required = {"package.version", "package.install", "package.audit", "bundle.integrity",
                "bundle.contract", "bundle.revision", "runtime.dependencies", "runtime.clean", "runtime.worker"}
    names = [i.get("check") for i in items if isinstance(i, dict)]
    if len(names) != len(items) or len(set(names)) != len(names) or not required.issubset(names):
        raise ValueError("Linux runner omitted mandatory checks or returned duplicate checks")
    for item in items:
        if item.get("status") not in {"passed", "failed", "unverified"}:
            raise ValueError("invalid Linux runner check status")
        if item["check"] in {"runtime.clean", "runtime.worker"} and item["status"] == "passed":
            runtime = item.get("evidence", {}).get("runtime", {})
            if runtime.get("executed") is not True or runtime.get("ready") is not True:
                raise ValueError("Linux runner claimed runtime success without execution/readiness")
    for item in items:
        report.add(f"linux.{variant}.{item['check']}", item["status"], item.get("detail", ""),
                   **item.get("evidence", {}))


def gate_linux(dist: Path, contract: dict[str, Any], options: dict | None = None) -> Report:
    report, work = Report("linux"), _work(options, "linux")
    env = isolated_environment(work / "host-state")
    env["DOCKER_CONFIG"] = os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker"))
    docker = shutil.which("docker")
    variants = (options or {}).get("linux_variants", ("cpu",))
    artifacts: dict[str, Path] = {}
    for variant in variants:
        # Select by variant only; dpkg-deb inspects embedded metadata, never trust filename version.
        matches = sorted(p for p in dist.rglob(f"qlh-edge-inference-{variant}_*.deb") if p.is_file())
        if len(matches) != 1:
            report.fail(f"linux.{variant}.artifact", f"expected exactly one .deb, found {len(matches)}")
        else:
            artifacts[variant] = matches[0].resolve()
            report.ok(f"linux.{variant}.artifact", str(matches[0]), sha256=sha256(matches[0]))
    if not docker:
        report.unverified("linux.docker", "docker executable unavailable")
        report.unverified("linux.execution", "no final .deb installation or runtime executed")
        return report
    server = run_command([docker, "info", "--format", "{{json .OSType}}"], cwd=work, env=env,
                         log=work / "docker-info.log", timeout=25)
    if not server.get("ok") or server.get("output", "").strip() != '"linux"':
        report.unverified("linux.docker", "Linux Docker daemon unavailable", execution=server)
        report.unverified("linux.execution", "no final .deb installation or runtime executed")
        return report
    report.ok("linux.docker", "Linux daemon answered", execution=server)
    if not artifacts:
        report.unverified("linux.execution", "no final .deb installation or runtime executed")
        return report
    if os.environ.get(LAUNCH_PROBE_ENV, "1") == "0":
        report.unverified("linux.execution", f"{LAUNCH_PROBE_ENV}=0; installation/startup not executed")
        return report
    image = (options or {}).get("linux_image", DEFAULT_IMAGE)
    inspection = run_command([docker, "image", "inspect", image], cwd=work, env=env,
                             log=work / "docker-image.log", timeout=25)
    if not inspection["ok"]:
        report.unverified("linux.image", "preprovisioned image unavailable; see Dockerfile.release-gate",
                          execution=inspection)
        return report
    report.ok("linux.image", image, execution=inspection)
    profile_path = ROOT / "packaging" / "packaging" / "runtime_guard.py"
    spec = importlib.util.spec_from_file_location("release_gate_canonical_runtime_guard", profile_path)
    if spec is None or spec.loader is None:
        report.fail("linux.profiles", "canonical runtime_guard unavailable")
        return report
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    profiles_file = work / "runtime-profiles.json"
    profiles_file.write_text(json.dumps({
        profile: list(guard.required_modules(guard.RuntimeContext(
            root=work, engine="cuda" if profile == "torch_cuda" else "cpu", profile=profile)))
        for profile in guard.RUNTIME_PROFILES
    }), encoding="utf-8")
    contract_file = work / "release-contract.json"
    contract_file.write_text(json.dumps(contract), encoding="utf-8")
    for variant, artifact in artifacts.items():
        nonce, container = uuid.uuid4().hex, "qlh-release-gate-" + uuid.uuid4().hex
        output = work / variant
        output.mkdir()
        digest = sha256(artifact)
        command = [docker, "create", "--name", container, "--network", "none",
                   "--security-opt", "no-new-privileges", "--pids-limit", "256",
                   "--memory", "4g", "--tmpfs", "/tmp:rw,nosuid,size=512m"]
        for source, target, readonly in (
            (artifact, "/artifact.deb", True), (MODULE_DIR, "/gate", True),
            (TRUSTED_KEYS, "/trusted-keys", True), (contract_file, "/release-contract.json", True),
            (profiles_file, "/runtime-profiles.json", True),
            (output, "/evidence", False),
        ):
            command.extend(("--mount", f"type=bind,source={source},target={target}" +
                            (",readonly" if readonly else "")))
        command.extend((image, "python3", "-B", "/gate/release_linux_runner.py", "--variant", variant,
                        "--nonce", nonce, "--sha256", digest, "--startup-timeout",
                        str((options or {}).get("startup_timeout", 90))))
        if (options or {}).get("require_revision"):
            command.extend(("--require-revision", options["require_revision"]))
        created = run_command(command, cwd=work, env=env, log=output / "docker-create.log", timeout=30)
        if not created["ok"]:
            report.fail(f"linux.{variant}.container", "isolated container creation failed", execution=created)
            # A timed-out create can still have created our uniquely named container.
            cleanup = run_command([docker, "rm", "-f", container], cwd=work, env=env,
                                  log=output / "docker-cleanup.log", timeout=30)
            report.unverified(f"linux.{variant}.execution", "container did not run", cleanup=cleanup)
            continue
        try:
            execution = run_command([docker, "start", "--attach", container], cwd=work, env=env,
                                    log=output / "docker-start.log", timeout=900)
            report_file = output / "result.json"
            if not report_file.is_file():
                raise ValueError("container produced no current installed-artifact report")
            value = read_object(report_file)
            _linux_result(report, value, nonce=nonce, digest=digest, variant=variant)
            if not execution["ok"]:
                report.fail(f"linux.{variant}.execution", "container command failed", execution=execution)
            else:
                report.ok(f"linux.{variant}.execution", "installed .deb and executed packaged runtime",
                          execution=execution, container=container, artifact_sha256=digest)
        except (OSError, ValueError, TypeError) as exc:
            report.fail(f"linux.{variant}.runner", str(exc))
        finally:
            cleanup = run_command([docker, "rm", "-f", container], cwd=work, env=env,
                                  log=output / "docker-cleanup.log", timeout=30)
            (report.ok if cleanup["ok"] else report.fail)(
                f"linux.{variant}.cleanup", "remove only owned container " + container, execution=cleanup)
    return report


def android_tool(name: str) -> str | None:
    direct = shutil.which(name)
    if direct:
        return direct
    sdk_roots = [os.environ.get("ANDROID_SDK_ROOT", ""), os.environ.get("ANDROID_HOME", ""),
                 str(Path.home() / "Android" / "Sdk"),
                 str(Path.home() / "AppData" / "Local" / "Android" / "Sdk")]
    extensions = (".exe", ".bat", "") if os.name == "nt" else ("",)
    for sdk in sdk_roots:
        if not sdk:
            continue
        base = Path(sdk)
        folders = ([base / "platform-tools"] if name == "adb" else
                   sorted((base / "build-tools").glob("*"), reverse=True))
        for folder in folders:
            for extension in extensions:
                tool = folder / (name + extension)
                if tool.is_file():
                    return str(tool)
    return None


def inspect_apk(apk: Path) -> dict:
    with zipfile.ZipFile(apk) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("APK has duplicate ZIP entries")
        if "AndroidManifest.xml" not in names or "classes.dex" not in names:
            raise ValueError("APK lacks binary manifest or classes.dex")
        native = [n for n in names if n.startswith("lib/") and n.endswith(".so")]
        required = "lib/arm64-v8a/libqlh_llama_jni.so"
        if required not in native or "lib/arm64-v8a/libc++_shared.so" not in native:
            raise ValueError("APK lacks arm64 JNI runtime or libc++_shared.so")
        for name in native:
            with archive.open(name) as stream:
                header = stream.read(64)
            if (not name.startswith("lib/arm64-v8a/") or len(header) < 64
                    or header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183):
                raise ValueError(f"not an arm64 ELF64 library: {name}")
        return dict(abi="arm64-v8a", native_libraries=native, sha256=sha256(apk))


def apk_metadata(output: str) -> dict:
    package = re.search(r"^package: name='([^']+)' versionCode='(\d+)' versionName='([^']+)'", output, re.M)
    if not package:
        raise ValueError("aapt returned no embedded APK identity")
    if re.search(r"^application-debuggable\s*$", output, re.M):
        raise ValueError("debuggable APK is not a release artifact")
    return dict(package=package[1], version_code=int(package[2]), version=package[3])


def android_command(tool: str, arguments: list[str]) -> list[str]:
    # Call the SDK JAR directly on Windows: no cmd.exe quoting or visible window.
    if Path(tool).suffix.lower() == ".bat":
        jar = Path(tool).parent / "lib" / "apksigner.jar"
        java = shutil.which("java")
        if not java and os.environ.get("JAVA_HOME"):
            candidate = Path(os.environ["JAVA_HOME"]) / "bin" / "java.exe"
            if candidate.is_file():
                java = str(candidate)
        if not java or not jar.is_file():
            raise ValueError("apksigner requires Java and SDK lib/apksigner.jar")
        return [java, "-jar", str(jar), *arguments]
    return [tool, *arguments]


def signer_digest(output: str) -> str:
    digests = re.findall(r"^Signer #\d+ certificate SHA-256 digest:\s*([0-9a-fA-F]{64})\s*$", output, re.M)
    if len(digests) != 1:
        raise ValueError("expected exactly one verified APK signer")
    return digests[0].lower()


def android_device_smoke(report: Report, *, adb: str, serial: str, apk: Path,
                         work: Path, env: dict[str, str], version_name: str | None = None,
                         version_code: int | None = None) -> None:
    """Explicit test device only; never replace or stop a pre-existing installation."""
    package = "com.qlh.inference"
    def run(name: str, arguments: list[str], timeout: float = 30) -> dict:
        return run_command([adb, "-s", serial, *arguments], cwd=work, env=env,
                           log=work / (name + ".log"), timeout=timeout)
    existing = run("device-existing", ["shell", "pm", "path", package])
    if not existing.get("executed") or existing.get("returncode") not in (0, 1):
        report.unverified("android.startup", "cannot establish clean device package state", execution=existing)
        return
    if existing.get("output", "").strip():
        report.unverified("android.startup", "device already has release package; refuse replacement/launch/stop",
                          execution=existing)
        return
    installed = run("device-install", ["install", "--no-streaming", str(apk)], timeout=180)
    if not installed["ok"] or not re.search(r"^Success\s*$", installed.get("output", ""), re.M):
        report.fail("android.startup", "adb install did not succeed; no launch performed", execution=installed)
        return
    executions = {"install": installed}
    try:
        # Bind runtime to exactly the accepted APK, rather than an already installed package.
        installed_path = run("device-apk-path", ["shell", "pm", "path", package])
        executions["package_path"] = installed_path
        paths = [line.removeprefix("package:").strip() for line in installed_path.get("output", "").splitlines()
                 if line.startswith("package:")]
        if not installed_path["ok"] or len(paths) != 1:
            raise ValueError("installed APK path unavailable or split package unexpected")
        pulled = work / "installed-base.apk"
        transfer = run("device-pull", ["pull", paths[0], str(pulled)], timeout=120)
        executions["installed_artifact"] = transfer
        if not transfer["ok"] or not pulled.is_file() or sha256(pulled) != sha256(apk):
            raise ValueError("installed APK differs from accepted artifact")
        started = run("device-start", ["shell", "am", "start", "-W", "-n", package + "/.MainActivity"])
        executions["launch"] = started
        if not started["ok"] or "Status: ok" not in started.get("output", ""):
            raise ValueError("Activity manager did not confirm successful start")
        for index in range(2):
            pid = run(f"device-pid-{index}", ["shell", "pidof", package])
            focus = run(f"device-focus-{index}", ["shell", "dumpsys", "activity", "activities"])
            executions[f"pid_{index}"], executions[f"focus_{index}"] = pid, focus
            if not pid["ok"] or not re.fullmatch(r"\d+(?:\s+\d+)*", pid.get("output", "").strip()):
                raise ValueError("release process is not running after Activity start")
            if not focus["ok"] or not any(package in line and "ResumedActivity" in line
                                          for line in focus.get("output", "").splitlines()):
                raise ValueError("release Activity not observed resumed")
            if index == 0:
                time.sleep(2)
        report.ok("android.startup", "real accepted APK installed, launched and Activity observed resumed twice",
                  device_serial=serial, artifact_sha256=sha256(apk), executions=executions)
        runtime = observe_runtime(adb=adb, serial=serial, package=package, work=work / "runtime",
                                  env=env, version_name=version_name, version_code=version_code,
                                  require_idle=True, timeout=20)
        if runtime["passed"] is not True:
            for check in ("roles", "native-runtime"):
                report.fail("android." + check, "fresh process runtime observation did not pass", runtime=runtime)
        else:
            for check, detail in (
                ("roles", "actual idle default distributed client; no worker or full model was started"),
                ("native-runtime", "real JNI backend devices and system information from the accepted APK process"),
            ):
                report.ok("android." + check, detail, artifact_sha256=sha256(apk), runtime=runtime)
    except (OSError, ValueError) as exc:
        report.fail("android.startup", str(exc), executions=executions)
    finally:
        cleanup = run("device-cleanup", ["uninstall", package], timeout=60)
        (report.ok if cleanup["ok"] and "Success" in cleanup.get("output", "") else report.fail)(
            "android.cleanup", "remove only package freshly installed by this probe", execution=cleanup)


def gate_android(dist: Path, contract: dict[str, Any], options: dict | None = None) -> Report:
    report, work = Report("android"), _work(options, "android")
    env = isolated_environment(work / "host-state")
    tools = {name: android_tool(name) for name in ("aapt", "apksigner", "adb")}
    for name, tool in tools.items():
        (report.ok if tool else report.unverified)("android.tool." + name, tool or "platform tool unavailable")
    selected = (options or {}).get("android_apk")
    matches = [Path(selected).resolve()] if selected else sorted(dist.rglob("*.apk"))
    if len(matches) != 1 or not matches[0].is_file():
        report.fail("android.artifact", f"expected exactly one release APK, found {len(matches)}")
        for check in ("version", "dependencies", "signature", "revision", "roles", "startup", "native-runtime"):
            report.unverified("android." + check, "no release APK inspected/installed/executed")
        return report
    apk = matches[0].resolve()
    authenticated = False
    if not apk.is_relative_to(dist.resolve()):
        report.fail("android.artifact", "APK must be inside --dist")
        return report
    try:
        native = inspect_apk(apk)
        report.ok("android.dependencies", "APK contains arm64 ELF JNI + C++ runtime; loading checked separately",
                  artifact=str(apk), **native)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        report.fail("android.dependencies", str(exc))
    if tools["aapt"]:
        result = run_command([tools["aapt"], "dump", "badging", str(apk)], cwd=work, env=env,
                             log=work / "aapt.log", timeout=45)
        try:
            if not result["ok"]:
                raise ValueError("aapt failed to inspect release APK")
            identity = apk_metadata(result["output"])
            expected = dict(package="com.qlh.inference", version=contract["version"]["product"],
                            version_code=contract["version"]["android_code"])
            if identity != expected:
                raise ValueError(f"embedded APK identity {identity} != {expected}")
            report.ok("android.version", "embedded release APK metadata matches contract",
                      identity=identity, execution=result)
        except ValueError as exc:
            report.fail("android.version", str(exc), execution=result)
    else:
        report.unverified("android.version", "aapt unavailable; source version.properties is not evidence")
    expected_signer = (options or {}).get("android_cert_sha256", "")
    if tools["apksigner"]:
        result = run_command(android_command(tools["apksigner"], ["verify", "--verbose", "--print-certs", str(apk)]),
                             cwd=work, env=env, log=work / "apksigner.log", timeout=60)
        try:
            if not result["ok"]:
                raise ValueError("apksigner rejected release APK")
            actual = signer_digest(result["output"])
            if not expected_signer:
                report.unverified("android.signature", "signature cryptographically valid; release certificate pin missing",
                                  certificate_sha256=actual, execution=result)
            elif actual != expected_signer:
                report.fail("android.signature", "APK signer differs from trusted release certificate",
                            certificate_sha256=actual, execution=result)
            else:
                authenticated = True
                report.ok("android.signature", "apksigner verified pinned release certificate",
                          certificate_sha256=actual, execution=result)
        except ValueError as exc:
            report.fail("android.signature", str(exc), execution=result)
    else:
        report.unverified("android.signature", "apksigner unavailable")
    info = None
    try:
        with zipfile.ZipFile(apk) as archive:
            metadata = archive.getinfo("assets/build-info.json")
            if metadata.file_size > 65536:
                raise ValueError("APK build-info.json too large")
            info = json.loads(archive.read(metadata))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        pass
    check_revision(report, "android.revision", info, authenticated=authenticated,
                   required=(options or {}).get("require_revision"), artifact=str(apk))
    # Role and native readiness require the existing service's nonce/PID-bound
    # dumpsys observation; Activity lifetime alone never proves those checks.
    serial = (options or {}).get("android_serial")
    online = []
    if tools["adb"]:
        result = run_command([tools["adb"], "devices", "-l"], cwd=work, env=env,
                             log=work / "adb-devices.log", timeout=25)
        online = [line.split()[0] for line in result.get("output", "").splitlines()
                  if len(line.split()) >= 2 and line.split()[1] == "device"]
        report.add("android.device", "passed" if result["ok"] and serial in online else "unverified",
                   "explicit serial online" if serial in online else "no explicit online acceptance device",
                   execution=result, requested_serial=serial, online_devices=online)
    enabled = (options or {}).get("android_device_probe", False)
    prerequisites = all(any(i["check"] == "android." + name and i["passed"] for i in report.items)
                        for name in ("version", "dependencies", "signature", "revision"))
    if (enabled and serial in online and prerequisites and tools["adb"]
            and os.environ.get(LAUNCH_PROBE_ENV, "1") != "0"):
        android_device_smoke(report, adb=tools["adb"], serial=serial, apk=apk, work=work, env=env,
                             version_name=contract["version"]["product"],
                             version_code=contract["version"]["android_code"])
    else:
        report.unverified("android.startup", "device smoke requires --android-device-probe, explicit online serial, "
                          "valid APK/pinned signature and enabled launch probe; not executed")
    for check in ("roles", "native-runtime"):
        if not any(item["check"] == "android." + check for item in report.items):
            report.unverified("android." + check,
                              "accepted APK process runtime observation not executed; Activity smoke is insufficient")
    return report


GATES = {"windows": gate_windows, "linux": gate_linux, "android": gate_android}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platforms", default="", help="explicit comma-separated windows,linux,android")
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build" / "release-artifacts-gate")
    parser.add_argument("--linux-image", default=DEFAULT_IMAGE)
    parser.add_argument("--windows-variants", default="cpu,launcher", help="cpu,launcher; cuda explicitly opt-in")
    parser.add_argument("--linux-variants", default="cpu", help="cpu; cuda explicitly opt-in")
    parser.add_argument("--require-revision", help="full Git object id; must match authenticated artifact build-info.json")
    parser.add_argument("--startup-timeout", type=float, default=90)
    parser.add_argument("--android-apk", type=Path, help="one release APK inside --dist")
    parser.add_argument("--android-cert-sha256", default="", help="trusted release signer SHA-256 pin")
    parser.add_argument("--android-serial", help="explicit acceptance device serial")
    parser.add_argument("--android-device-probe", action="store_true", help="opt in to install/launch on explicit CLEAN test device")
    args = parser.parse_args(argv)
    requested = list(dict.fromkeys(s.strip().lower() for s in args.platforms.split(",") if s.strip()))
    if not requested:
        print("::error::必须用 --platforms 显式声明平台", file=sys.stderr)
        return 1
    if any(name not in GATES for name in requested):
        print("::error::未知平台", file=sys.stderr)
        return 2
    variants = [v.strip() for v in args.linux_variants.split(",") if v.strip()]
    if not variants or len(variants) != len(set(variants)) or any(v not in {"cpu", "cuda"} for v in variants):
        parser.error("--linux-variants must contain unique cpu/cuda values")
    if not 1 <= args.startup_timeout <= 600:
        parser.error("--startup-timeout must be between 1 and 600 seconds")
    windows_variants = [v.strip() for v in args.windows_variants.split(",") if v.strip()]
    if (not windows_variants or len(windows_variants) != len(set(windows_variants))
            or any(v not in {"cpu", "cuda", "launcher"} for v in windows_variants)):
        parser.error("--windows-variants must contain unique cpu/cuda/launcher values")
    if args.require_revision and not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", args.require_revision):
        parser.error("--require-revision must be a full lowercase Git object id")
    pin = args.android_cert_sha256.replace(":", "").lower()
    if pin and not re.fullmatch(r"[a-f0-9]{64}", pin):
        parser.error("--android-cert-sha256 must be a SHA-256 hex digest")
    if args.android_device_probe and not args.android_serial:
        parser.error("--android-device-probe requires --android-serial")
    started = time.time()
    try:
        contract = read_object(CONTRACT_PATH)
        validate_contract(contract)
    except (OSError, ValueError, TypeError) as exc:
        print(f"::error::发行合同错误: {exc}", file=sys.stderr)
        return 2
    options = {**vars(args), "linux_variants": variants, "windows_variants": windows_variants,
               "android_cert_sha256": pin}
    reports = []
    for name in requested:
        try:
            if not args.dist.is_dir():
                report = Report(name)
                report.fail(name + ".dist", "dist directory missing", path=str(args.dist))
            else:
                report = GATES[name](args.dist.resolve(), contract, options)
        except Exception as exc:
            report = Report(name)
            report.fail(name + ".runner", f"{type(exc).__name__}: {exc}")
        if args.require_revision and (report.as_dict()["source_revision"] != args.require_revision
                                      or not report.as_dict()["source_revision_verified"]):
            report.unverified(name + ".revision-required", "not every selected artifact has authenticated required revision",
                              required_revision=args.require_revision)
        reports.append(report)
        print(f"[{name}] {report.as_dict()['status'].upper()}")
        for item in report.items:
            print(f"  [{item['status']}] {item['check']}: {item['detail']}")
    passed = all(r.as_dict()["passed"] for r in reports)
    revisions = {r.as_dict()["source_revision"] for r in reports}
    revision = next(iter(revisions)) if len(revisions) == 1 and None not in revisions else None
    evidence = dict(schema_version=SCHEMA_VERSION, gate="release-artifacts", dist=str(args.dist.resolve()),
                    source_revision=revision, required_revision=args.require_revision,
                    revision_verified=revision is not None, source_revision_verified=revision is not None,
                    contract_version=contract["version"], contract_sha256=sha256(CONTRACT_PATH),
                    requested_platforms=requested, windows_variants=windows_variants, linux_variants=variants,
                    started_at_unix=started, finished_at_unix=time.time(),
                    reports=[r.as_dict() for r in reports], passed=passed,
                    status="passed" if passed else "failed")
    if args.json:
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"::error::cannot write evidence: {exc}", file=sys.stderr)
            return 1
        print(f"[evidence] {args.json}")
    return 0 if passed else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raise SystemExit(main())
