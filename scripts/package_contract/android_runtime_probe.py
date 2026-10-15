"""Read-only Android runtime observation through the existing service's dumpsys.

Never installs, launches, stops, reconfigures, or loads a model. A fresh nonce,
PID observations on both sides, and real JNI backend facts bind each response.
Debug observations are diagnostics; release acceptance additionally authenticates
and matches the installed APK in release_artifacts_gate.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from release_gate_runtime import isolated_environment, run_command

PREFIX = "QLH_RELEASE_PROBE: "
PACKAGE = re.compile(r"com\.qlh\.inference(?:\.debug)?")
CONNECTIONS = {"STOPPED", "CONNECTING", "HELLO_SENT", "READY", "BACKING_OFF", "RETRY_EXHAUSTED"}
ATTEMPTS = {"IDLE", "OFFERED", "RUNNING", "CANCELLING", "SUCCEEDED", "FAILED", "CANCELLED", "LOST"}


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key in Android runtime observation")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("non-finite JSON constant in Android runtime observation")


def _pid(result: dict) -> int:
    require(result.get("executed") is True and type(result.get("returncode")) is int
            and result["returncode"] == 0 and result.get("ok") is True,
            "Android PID command did not execute successfully")
    raw = result.get("output", "").strip()
    require(bool(re.fullmatch(r"[1-9][0-9]*", raw)), "expected one running Android application PID")
    return int(raw)


def validate_observation(output: str, *, nonce: str, pid: int, package: str,
                         started_at: float, finished_at: float,
                         version_name: str | None = None, version_code: int | None = None,
                         require_idle: bool = False) -> dict:
    require(bool(re.fullmatch(r"[a-f0-9]{32}", nonce)), "invalid probe nonce")
    require(type(pid) is int and pid > 0 and bool(PACKAGE.fullmatch(package)), "invalid expected process identity")
    lines = [line.strip()[len(PREFIX):] for line in output.splitlines() if line.strip().startswith(PREFIX)]
    require(len(lines) == 1, "missing or ambiguous Android runtime observation")
    value = json.loads(lines[0], object_pairs_hook=_object, parse_constant=_constant)
    require(isinstance(value, dict), "Android runtime observation must be an object")
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
            "unsupported Android runtime observation schema")
    require(value.get("nonce") == nonce and type(value.get("pid")) is int and value["pid"] == pid,
            "Android observation nonce or PID mismatch")
    require(value.get("package_name") == package and value.get("service_running") is True,
            "Android service or package identity mismatch")
    captured = value.get("captured_at")
    require(type(captured) in (int, float) and math.isfinite(captured)
            and 0 < started_at <= finished_at and started_at - 30 <= captured <= finished_at + 30,
            "Android runtime observation timestamp is outside the request window")
    require(value.get("error_code") is None, "Android runtime observation reports an error")
    require(isinstance(value.get("version_name"), str) and bool(value["version_name"])
            and type(value.get("version_code")) is int and value["version_code"] > 0,
            "missing Android process version")
    if version_name is not None:
        require(value["version_name"] == version_name, "Android process version differs from accepted APK")
    if version_code is not None:
        require(value["version_code"] == version_code, "Android process version code differs from accepted APK")
    mode, role = value.get("inference_mode"), value.get("role")
    require(mode in ("distributed", "local"), "unknown Android inference mode")
    worker = value.get("worker")
    require(isinstance(worker, dict), "missing Android worker facts")
    running, alive = worker.get("process_running"), worker.get("connection_alive")
    require(type(running) is bool and type(alive) is bool
            and worker.get("connection") in CONNECTIONS and worker.get("active_attempt_state") in ATTEMPTS,
            "invalid Android worker facts")
    require(not alive or (running and worker["connection"] == "READY"),
            "worker connection is alive without a ready running worker")
    require(worker["connection"] != "READY" or (alive and running), "READY worker lacks a real connection")
    require(worker["connection"] != "STOPPED" or (not alive and not running), "STOPPED worker still running")
    require(running or worker["connection"] in ("STOPPED", "RETRY_EXHAUSTED"),
            "inactive worker contains an active connection state")
    require(worker["active_attempt_state"] not in ("OFFERED", "RUNNING", "CANCELLING") or
            (running and alive and worker["connection"] == "READY"),
            "active attempt lacks a ready live worker connection")
    expected_role = ("worker" if running else "client") if mode == "distributed" else "local"
    require(role == expected_role, "Android role contradicts current mode and worker lifecycle")
    require(mode != "local" or not running, "local Android process still participates as a cluster worker")
    native = value.get("native")
    require(isinstance(native, dict) and native.get("available") is True and native.get("error_code") is None,
            "Android JNI backend is unavailable")
    require(isinstance(native.get("system_info"), str) and bool(native["system_info"].strip())
            and isinstance(native.get("backend_devices"), str) and bool(native["backend_devices"].strip())
            and type(native.get("backend_device_count")) is int and native["backend_device_count"] > 0,
            "Android JNI did not expose actual backend devices and system information")
    require("stub" not in native["system_info"].lower() and "stub" not in native["backend_devices"].lower(),
            "Android JNI backend is a stub")
    model = value.get("model")
    require(isinstance(model, dict) and type(model.get("loaded")) is bool
            and type(model.get("full_model_loaded")) is bool and isinstance(model.get("layer_range"), list),
            "missing native model load facts")
    span = model["layer_range"]
    require(not span or (len(span) == 2 and all(type(n) is int for n in span) and 0 <= span[0] < span[1]),
            "invalid native model layer range")
    require(model["full_model_loaded"] == (model["loaded"] and not span), "contradictory full-model load fact")
    if model["loaded"]:
        require(type(model.get("native_layer_count")) is int and model["native_layer_count"] > 0,
                "loaded Android context lacks a native decoder count")
        require(not span or model["native_layer_count"] == span[1] - span[0],
                "native decoder count differs from loaded segment")
    else:
        require(not span and model.get("native_layer_count") is None, "unloaded context retains model facts")
    require(mode != "distributed" or model["full_model_loaded"] is False,
            "default distributed Android process loaded a full model")
    if require_idle:
        require(mode == "distributed" and role == "client" and not model["loaded"]
                and worker["connection"] == "STOPPED", "fresh release must expose the idle default distributed client")
    return value


def observe_runtime(*, adb: str, serial: str, package: str, work: Path,
                    env: dict | None = None, version_name: str | None = None,
                    version_code: int | None = None, require_idle: bool = False,
                    timeout: float = 20) -> dict:
    require(bool(PACKAGE.fullmatch(package)), "unsupported Android package")
    require(isinstance(serial, str) and bool(re.fullmatch(r"[A-Za-z0-9_.:\-]+", serial)), "invalid Android serial")
    require(math.isfinite(timeout) and 0 < timeout <= 60, "runtime observation timeout must be within 60 seconds")
    work.mkdir(parents=True, exist_ok=True)
    env = env or isolated_environment(work / "host-state")
    attempts, observation = [], None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        index, nonce = len(attempts), uuid.uuid4().hex
        attempt = {"nonce": nonce, "started_at": time.time(), "executions": {}}
        attempts.append(attempt)
        def run(label, arguments):
            execution = run_command([adb, "-s", serial, *arguments], cwd=work, env=env,
                                    log=work / f"runtime-{index}-{label}.log",
                                    timeout=max(0.1, min(8, deadline - time.monotonic())))
            attempt["executions"][label] = execution
            return execution
        try:
            before = _pid(run("pid-before", ["shell", "pidof", package]))
            result = run("dump", ["shell", "dumpsys", "activity", "service", package + "/com.qlh.inference.service.InferenceService",
                                  "--qlh-release-probe", nonce])
            require(result.get("executed") is True and type(result.get("returncode")) is int
                    and result["returncode"] == 0 and result.get("ok") is True, "runtime dumpsys did not complete")
            after = _pid(run("pid-after", ["shell", "pidof", package]))
            require(before == after, "Android application restarted during runtime observation")
            attempt["finished_at"] = time.time()
            observation = validate_observation(result.get("output", ""), nonce=nonce, pid=before,
                package=package, started_at=attempt["started_at"], finished_at=attempt["finished_at"],
                version_name=version_name, version_code=version_code, require_idle=require_idle)
            attempt["passed"] = True
            break
        except (OSError, ValueError) as exc:
            attempt.update(passed=False, error=str(exc), finished_at=time.time())
            if time.monotonic() < deadline:
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    return {"passed": observation is not None, "package": package, "serial": serial,
            "observation": observation, "attempts": attempts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--package", default="com.qlh.inference")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing to replace an existing observation report")
    result = observe_runtime(adb=args.adb, serial=args.serial, package=args.package,
                             work=args.output.parent / (args.output.stem + "-raw"), timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "attempts": len(result["attempts"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
