#!/usr/bin/env python3
"""Install one final Debian artifact inside the gate's disposable Docker container.

Only installed files provide the application/runtime. The host source tree is
never mounted. Results are bound to the controller nonce and artifact SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from release_gate_runtime import Report, isolated_environment, json_output, probe_service, run_command

APP = Path("/opt/qlh-edge-inference")
WORK = Path("/evidence")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--startup-timeout", type=float, default=90)
    parser.add_argument("--require-revision")
    args = parser.parse_args(argv)
    report = Report("linux")
    artifact = Path("/artifact.deb")
    started = time.time()
    env = isolated_environment(WORK / "install-state")

    def run(name: str, command: list[str], timeout: float = 120) -> dict:
        return run_command(command, cwd=WORK, env=env, log=WORK / (name + ".log"), timeout=timeout)

    try:
        if digest(artifact) != args.sha256:
            raise ValueError("mounted artifact SHA-256 differs from controller")
        contract = json.loads(Path("/release-contract.json").read_text(encoding="utf-8"))
        result = run("metadata", ["dpkg-deb", "--field", str(artifact), "Package", "Version", "Architecture"])
        fields = dict(line.split(": ", 1) for line in result.get("output", "").splitlines() if ": " in line)
        expected = {"Package": "qlh-edge-inference-" + args.variant,
                    "Version": contract["version"]["product"], "Architecture": "amd64"}
        if not result["ok"] or fields != expected:
            raise ValueError(f"embedded Debian metadata {fields} != {expected}")
        report.ok("package.version", "embedded dpkg metadata matches contract", metadata=fields, execution=result)
        result = run("install", ["dpkg", "--install", str(artifact)], timeout=600)
        if not result["ok"]:
            report.fail("package.install", "dpkg installation/postinst failed", execution=result)
        else:
            status = run("installed-status", ["dpkg-query", "--show", "--showformat=${Status}", expected["Package"]])
            if not status["ok"] or status.get("output", "").strip() != "install ok installed":
                report.fail("package.install", "dpkg did not configure installed package", execution=status)
            else:
                report.ok("package.install", "real dpkg --install, including maintainer scripts", execution=result, status=status)
        audit = run("audit", ["dpkg", "--audit"])
        (report.ok if audit["ok"] and not audit.get("output", "").strip() else report.fail)(
            "package.audit", "dpkg --audit must be empty", execution=audit)
        python = APP / "venv" / "bin" / "python3"
        manifest_path = APP / "manifest" / "install-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = dict(version=contract["version"]["product"], platform="linux", variant=args.variant,
                        app_id="qlh-edge-inference", package_kind="application")
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("installed signed manifest identity mismatch")
        verification = run("verify-deep", [str(python), "-B", str(APP / "bin" / "install_manifest.py"),
                           "verify", "--root", str(APP), "--level", "deep", "--trusted-keys-dir",
                           "/trusted-keys", "--json"], timeout=600)
        verified = json_output(verification)
        if verified.get("ok") is not True or verified.get("level") != "deep" or any(
                verified.get("manifest", {}).get(k) != v for k, v in identity.items()):
            raise ValueError("installed tree did not pass trusted deep verification")
        report.ok("bundle.integrity", "installed verifier checked trusted signature and every file SHA-256",
                  execution=verification, verification=verified)
        contract_path = APP / "release-contract.json"
        entries = {e["path"]: e for e in manifest["files"]}
        if ("release-contract.json" not in entries
                or digest(contract_path) != entries["release-contract.json"]["sha256"]
                or json.loads(contract_path.read_text(encoding="utf-8")) != contract):
            raise ValueError("installed release contract is missing, unsigned or mismatched")
        report.ok("bundle.contract", "installed signed release contract matches canonical contract")
        build_path = APP / "build-info.json"
        try:
            info = json.loads(build_path.read_text(encoding="utf-8"))
            revision = info.get("source_revision")
            import re
            if ("build-info.json" not in entries or digest(build_path) != entries["build-info.json"]["sha256"]
                    or info.get("schema_version") != 1 or not isinstance(revision, str)
                    or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", revision)):
                raise ValueError("invalid or unsigned installed build-info.json")
            if args.require_revision and revision != args.require_revision:
                raise ValueError("installed source_revision differs from --require-revision")
            if info.get("source_dirty") is not False:
                raise ValueError("installed artifact was built from modified or unverifiable source")
            report.ok("bundle.revision", "source revision authenticated by installed signed file set",
                      source_revision=revision, revision_verified=True)
        except (OSError, ValueError) as exc:
            report.unverified("bundle.revision", str(exc))
        profiles = json.loads(Path("/runtime-profiles.json").read_text(encoding="utf-8"))
        profile = manifest.get("runtime_profile")
        if profile not in profiles:
            report.unverified("runtime.dependencies", "signed runtime_profile missing/unsupported")
        else:
            script = (
                "import importlib,json,sys; "
                f"modules={profiles[profile]!r}; "
                "loaded={n:getattr(importlib.import_module(n),'__file__',None) for n in modules}; "
                "print(json.dumps({'prefix':sys.prefix,'modules':loaded,'imports_succeeded':True}))"
            )
            dependencies = run("runtime-imports", [str(python), "-I", "-B", "-c", script], timeout=180)
            imported = json_output(dependencies)
            if imported.get("imports_succeeded") is not True or imported.get("prefix") != str(APP / "venv"):
                raise ValueError("dependency imports did not execute in installed venv")
            report.ok("runtime.dependencies", "actual imports in installed venv, using runtime_guard profile list",
                      profile=profile, imported=imported, execution=dependencies)
        if any(i["check"] != "bundle.revision" for i in report.failures):
            raise ValueError("installation/dependency prerequisites failed; runtime not launched")
        default_role = "client" if contract["profile"]["default_node_role"] == "worker" else "master"
        command = [str(python), str(APP / "bin" / "qlh-launcher"), "--headless"]
        for name, role in (("runtime.clean", default_role), ("runtime.worker", "client")):
            runtime = probe_service(command, app=APP, work=WORK / name, role=role, timeout=args.startup_timeout,
                                    confirmed_worker=name == "runtime.worker")
            (report.ok if runtime["ready"] else report.fail)(
                name, "owned API /health + /ready + observed role" if runtime["ready"] else runtime.get("error", "not ready"),
                runtime=runtime)
    except Exception as exc:
        report.fail("runner", f"{type(exc).__name__}: {exc}")
    required = {"package.version", "package.install", "package.audit", "bundle.integrity",
                "bundle.contract", "bundle.revision", "runtime.dependencies", "runtime.clean", "runtime.worker"}
    for check in sorted(required - {i["check"] for i in report.items}):
        report.unverified(check, "not executed because an earlier artifact prerequisite failed")
    result = dict(schema_version=1, gate="linux-installed-artifact", nonce=args.nonce,
                  artifact_sha256=args.sha256, variant=args.variant, started_at_unix=started,
                  finished_at_unix=time.time(), **report.as_dict())
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
