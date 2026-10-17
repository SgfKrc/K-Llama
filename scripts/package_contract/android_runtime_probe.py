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


def _positive_int(value, name, minimum=1):
    require(type(value) is int and value >= minimum, "invalid " + name)
    return value


def _identity(value, pid=None):
    require(isinstance(value, dict), "missing process identity")
    result = (_positive_int(value.get("pid"), "process PID"),
              _positive_int(value.get("created_at"), "process start ticks"))
    require(pid is None or result[0] == pid, "process identity PID differs")
    return result


def validate_process_tree(tree, *, pid, required=False):
    if tree is None:
        require(not required, "complete Android process tree required")
        return
    require(isinstance(tree, dict) and type(tree.get("complete")) is bool,
            "invalid Android process tree")
    if not tree["complete"]:
        require(not required, "Android process tree incomplete/permission denied")
        return
    require(tree.get("source") in {"android_proc_task_children", "android_proc_uid_ppid"} and not tree.get("errors"),
            "Android process tree has errors or unsupported enumeration")
    root = _identity(tree.get("identity_before"), pid)
    require(root == _identity(tree.get("identity_after"), pid), "Android process root identity changed")
    entries = tree.get("processes")
    require(isinstance(entries, list) and 0 < len(entries) <= 128, "missing/oversized process tree")
    indexed = {}
    for proc in entries:
        ident = _identity(proc)
        require(ident == _identity(proc.get("identity_after")), "child process identity changed")
        require(ident[0] not in indexed, "duplicate process PID")
        indexed[ident[0]] = proc
        require(all(proc.get(key) is True for key in ("children_complete", "maps_complete", "connections_complete"))
                and not proc.get("errors") and not proc.get("connection_errors"), "complete process tree contains incomplete facts")
        require(isinstance(proc.get("command"), list) and bool(proc["command"])
                and all(isinstance(v, str) and bool(v) for v in proc["command"]), "missing process command")
        require(isinstance(proc.get("cwd"), str) and proc["cwd"].startswith("/"), "missing process cwd")
        for key in ("model_mappings", "native_mappings"):
            require(isinstance(proc.get(key), list) and all(isinstance(v, str) for v in proc[key]), "missing process mappings")
        children = proc.get("child_pids")
        require(isinstance(children, list) and len(set(children)) == len(children)
                and all(type(v) is int and v > 0 for v in children), "invalid child PID closure")
        connections = proc.get("connections")
        require(isinstance(connections, list), "missing observed connections")
        for connection in connections:
            require(isinstance(connection, dict) and connection.get("status") in
                    {"ESTABLISHED", "SYN_SENT", "SYN_RECV", "FIN_WAIT1", "FIN_WAIT2", "TIME_WAIT",
                     "CLOSE", "CLOSE_WAIT", "LAST_ACK", "LISTEN", "CLOSING"}, "unknown connection state")
            for key in ("local", "remote"):
                endpoint = connection.get(key)
                if key == "remote" and endpoint is None:
                    require(connection["status"] != "ESTABLISHED", "established socket missing peer")
                    continue
                require(isinstance(endpoint, list) and len(endpoint) == 2 and isinstance(endpoint[0], str)
                        and type(endpoint[1]) is int and 0 <= endpoint[1] <= 65535, "invalid observed endpoint")
                import ipaddress
                ipaddress.ip_address(endpoint[0].split("%", 1)[0])
    require(pid in indexed and _identity(indexed[pid]) == root, "root process not in closure")
    visited, pending = set(), [pid]
    while pending:
        parent = pending.pop()
        require(parent not in visited, "cyclic/multiply owned process tree")
        visited.add(parent)
        for child in indexed[parent]["child_pids"]:
            require(child in indexed and indexed[child].get("parent_pid") == parent, "missing/misowned descendant")
            pending.append(child)
    require(visited == set(indexed), "disconnected process tree")

    if tree["source"] == "android_proc_uid_ppid":
        validate_uid_enumeration(tree.get("enumeration"), root_pid=pid, indexed=indexed)


def validate_uid_enumeration(proof, *, root_pid, indexed):
    require(isinstance(proof, dict) and proof.get("method") == "proc_uid_ppid"
            and proof.get("scope") == "app_uid_descendants" and proof.get("reason") == "task_children_enoent"
            and proof.get("complete") is True, "missing complete Android UID enumeration proof")
    uid = _positive_int(proof.get("uid"), "enumerated UID", 0)
    scans = []
    for key in ("before", "after"):
        scan = proof.get(key)
        require(isinstance(scan, dict) and scan.get("errors") == [], "UID scan contains errors")
        visible = scan.get("visible_pids")
        require(isinstance(visible, list) and 0 < len(visible) <= 32768
                and all(type(p) is int and p > 0 for p in visible)
                and len(set(visible)) == len(visible), "invalid visible /proc PID set")
        owners, roster = {}, {}
        require(isinstance(scan.get("owners"), list) and isinstance(scan.get("roster"), list), "missing UID roster")
        for owner in scan["owners"]:
            require(isinstance(owner, dict), "invalid PID owner")
            p = _positive_int(owner.get("pid"), "owner PID")
            require(p not in owners, "duplicate PID owner")
            owners[p] = _positive_int(owner.get("uid"), "process UID", 0)
        require(set(owners) == set(visible), "listed /proc process has no readable UID")
        for proc in scan["roster"]:
            p, start = _identity(proc)
            require(proc.get("uid") == uid and type(proc.get("uid")) is int and p not in roster,
                    "duplicate or differently owned UID roster entry")
            roster[p] = (start, _positive_int(proc.get("parent_pid"), "roster parent PID", 0))
        require(set(roster) == {p for p, owner in owners.items() if owner == uid}, "UID roster omits a matching process")
        require(root_pid in roster, "root absent from UID roster")
        require(isinstance(scan.get("outside_uid"), list), "missing facts for other visible UIDs")
        external = set()
        for proc in scan["outside_uid"]:
            p, _ = _identity(proc)
            parent = _positive_int(proc.get("parent_pid"), "external parent PID", 0)
            require(p not in external and p in owners and proc.get("uid") == owners[p]
                    and type(proc.get("uid")) is int and owners[p] != uid, "invalid external UID fact")
            require(parent not in roster, "cross-UID descendant omitted from closure")
            external.add(p)
        require(external == {p for p, owner in owners.items() if owner != uid}, "unreadable external process cannot be excluded")
        scans.append(roster)
    require(scans[0] == scans[1], "UID roster changed during observation")
    roster = scans[0]
    reachable, pending = set(), [root_pid]
    while pending:
        p = pending.pop()
        require(p not in reachable, "cyclic UID descendant closure")
        reachable.add(p)
        pending.extend(child for child, (_, parent) in roster.items() if parent == p)
    require(reachable == set(indexed), "process tree omits a UID descendant")
    for p, proc in indexed.items():
        require(roster[p] == (proc["created_at"], proc.get("parent_pid")), "observed process differs from UID roster")


def validate_receipt(receipt, *, value, pid, started_at, finished_at):
    require(isinstance(receipt, dict) and type(receipt.get("schema_version")) is int
            and receipt["schema_version"] == 1 and receipt.get("source") == "committed_worker_prepare",
            "missing committed Android runtime receipt")
    for key in ("operation_id", "config_id", "node_id", "assignment_id"):
        require(isinstance(receipt.get(key), str) and bool(receipt[key].strip()), "missing committed " + key)
    _positive_int(receipt.get("generation"), "committed generation")
    _identity(receipt, pid)
    require(receipt.get("loader_pid") == pid and type(receipt.get("loader_pid")) is int
            and receipt.get("package_name") == value["package_name"], "receipt native loader identity differs")
    require(receipt.get("loaded") is True and receipt.get("full_model_loaded") is False
            and receipt.get("engine") == "llama_cpp", "receipt is not an actual loaded GGUF segment")
    total = _positive_int(receipt.get("total_layers"), "source total_layers", 2)
    span = receipt.get("source_layer_range")
    require(isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span)
            and 0 <= span[0] < span[1] <= total and span[1] - span[0] < total,
            "receipt source range is not a proper segment")
    count = _positive_int(receipt.get("native_layer_count"), "receipt native count")
    require(receipt.get("range_source") == "verified_loaded_native_manifest" and receipt.get("layer_range") == span
            and span == value["model"]["layer_range"] and count == span[1] - span[0]
            and count == value["model"]["native_layer_count"], "receipt/native model source range differs")
    captured = receipt.get("captured_at")
    require(type(captured) in (int, float) and math.isfinite(captured)
            and started_at - 30 <= captured <= finished_at + 30, "receipt outside observation window")
    artifact = receipt.get("artifact")
    require(isinstance(artifact, dict), "missing actual loaded artifact receipt")
    for key in ("sha256", "rehashed_sha256", "native_manifest_sha256"):
        require(isinstance(artifact.get(key), str) and bool(re.fullmatch(r"[0-9a-f]{64}", artifact[key])), "missing artifact " + key)
    require(artifact["sha256"] == artifact["rehashed_sha256"], "actual artifact rehash differs from prepared bytes")
    require(artifact.get("layer_range") == span and type(artifact.get("decoder_layers")) is int
            and artifact["decoder_layers"] == count, "artifact decoder geometry differs")
    for key in ("path", "load_path"):
        require(isinstance(artifact.get(key), str) and artifact[key].startswith("/")
                and not artifact[key].endswith(" (deleted)"), "missing/deleted loaded artifact path")
    for key in ("size_bytes", "inode"):
        _positive_int(artifact.get(key), "artifact " + key)
    _positive_int(artifact.get("device"), "artifact device", 0)
    tree = value.get("process")
    require(isinstance(tree, dict) and _identity(tree.get("identity_before"), pid) == _identity(receipt)
            and _identity(tree.get("identity_after"), pid) == _identity(receipt), "receipt not bound to observed root process")


def _decode(output):
    lines = [line.strip()[len(PREFIX):] for line in output.splitlines() if line.strip().startswith(PREFIX)]
    require(len(lines) == 1, "missing or ambiguous Android runtime observation")
    value = json.loads(lines[0], object_pairs_hook=_object, parse_constant=_constant)
    require(isinstance(value, dict), "Android runtime observation must be an object")
    return value


def validate_observation(output: str, *, nonce: str, pid: int, package: str,
                         started_at: float, finished_at: float,
                         version_name: str | None = None, version_code: int | None = None,
                         require_idle: bool = False, require_process_tree: bool = False, product: bool = False) -> dict:
    require(bool(re.fullmatch(r"[a-f0-9]{32}", nonce)), "invalid probe nonce")
    require(type(pid) is int and pid > 0 and bool(PACKAGE.fullmatch(package)), "invalid expected process identity")
    value = _decode(output)
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
    if product:
        require(value.get("probe_kind") == "product", "explicit product probe response required")
        require(model["loaded"] and bool(span), "product probe requires an actually loaded proper segment")
    receipt = value.get("runtime_receipt")
    if model["loaded"] and span and (product or receipt is not None):
        validate_receipt(receipt, value=value, pid=pid, started_at=started_at, finished_at=finished_at)
    else:
        require(receipt is None, "idle/full local model must not claim committed worker receipt")
    tree = value.get("process")
    # Old release probes carry self-only diagnostics without a tree-complete claim.
    if product or require_process_tree or isinstance(tree, dict) and "complete" in tree:
        validate_process_tree(tree, pid=pid, required=require_process_tree or product)
    if require_idle:
        require(mode == "distributed" and role == "client" and not model["loaded"]
                and worker["connection"] == "STOPPED", "fresh release must expose the idle default distributed client")
    return value


def observe_runtime(*, adb: str, serial: str, package: str, work: Path,
                    env: dict | None = None, version_name: str | None = None,
                    version_code: int | None = None, require_idle: bool = False,
                    require_process_tree: bool = False, product: bool = False, timeout: float | None = None) -> dict:
    require(bool(PACKAGE.fullmatch(package)), "unsupported Android package")
    require(isinstance(serial, str) and bool(re.fullmatch(r"[A-Za-z0-9_.:\-]+", serial)), "invalid Android serial")
    timeout = timeout if timeout is not None else (120 if product else 20)
    require(math.isfinite(timeout) and 0 < timeout <= 150, "runtime observation timeout must be within 150 seconds")
    work.mkdir(parents=True, exist_ok=True)
    env = env or isolated_environment(work / "host-state")
    attempts, observation = [], None
    started_at, product_nonce, product_pid = time.time(), uuid.uuid4().hex if product else None, None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        index, nonce = len(attempts), product_nonce or uuid.uuid4().hex
        attempt = {"nonce": nonce, "started_at": time.time(), "executions": {}}
        attempts.append(attempt)
        def run(label, arguments):
            execution = run_command([adb, "-s", serial, *arguments], cwd=work, env=env,
                                    log=work / f"runtime-{index}-{label}.log",
                                    timeout=max(0.1, min(95 if label == "dump" else 8, deadline - time.monotonic())))
            attempt["executions"][label] = execution
            return execution
        try:
            before = _pid(run("pid-before", ["shell", "pidof", package]))
            if product:
                if product_pid is None:
                    product_pid = before
                require(before == product_pid, "Android product process restarted while hashing")
            result = run("dump", ["shell", "dumpsys", "activity", "service", package + "/com.qlh.inference.service.InferenceService",
                                  "--qlh-product-probe" if product else "--qlh-release-probe", nonce])
            require(result.get("executed") is True and type(result.get("returncode")) is int
                    and result["returncode"] == 0 and result.get("ok") is True, "runtime dumpsys did not complete")
            after = _pid(run("pid-after", ["shell", "pidof", package]))
            require(before == after, "Android application restarted during runtime observation")
            attempt["finished_at"] = time.time()
            if product:
                pending = _decode(result.get("output", ""))
                if pending.get("error_code") == "PROBE_PENDING":
                    require(pending.get("schema_version") == 1 and pending.get("nonce") == nonce
                            and type(pending.get("pid")) is int and pending["pid"] == before
                            and pending.get("probe_kind") == "product" and pending.get("package_name") == package,
                            "pending product observation identity differs")
                    attempt.update(passed=False, pending=True)
                    if time.monotonic() < deadline:
                        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
                    continue
            observation = validate_observation(result.get("output", ""), nonce=nonce, pid=before,
                package=package, started_at=started_at if product else attempt["started_at"], finished_at=attempt["finished_at"],
                version_name=version_name, version_code=version_code, require_idle=require_idle,
                require_process_tree=require_process_tree, product=product)
            attempt["passed"] = True
            break
        except (OSError, ValueError) as exc:
            attempt.update(passed=False, error=str(exc), finished_at=time.time())
            if product:
                break
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
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--product", action="store_true", help="Poll one explicit 120-second product receipt task; idle release probes keep their short budget")
    parser.add_argument("--require-process-tree", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing to replace an existing observation report")
    result = observe_runtime(adb=args.adb, serial=args.serial, package=args.package,
                             work=args.output.parent / (args.output.stem + "-raw"), timeout=args.timeout,
                             require_process_tree=args.require_process_tree, product=args.product)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(args.output), "attempts": len(result["attempts"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
