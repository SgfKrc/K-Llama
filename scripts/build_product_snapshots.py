"""Collect product observations and derive snapshots, refusing incomplete evidence.

--revision and the model intent are expectations, never running-process facts.
--node-addresses is an independent JSON mapping of node IDs to observed IP lists.
Legacy probes are retained in full as diagnostics and cannot produce passing
snapshots. A current deployment without PID-bound source/materialization evidence
will fail closed; this collector does not add state to the production runtime.
Existing output files are never overwritten, including historical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from package_contract.product_observations import (  # noqa: E402
    ObservationError, derive_node_observation, parse_json_output,
)

BASE = "http://127.0.0.1:8000"
ANDROID_PROBE = ROOT / "scripts/package_contract/android_runtime_probe.py"


def api(path):
    request = urllib.request.Request(f"{BASE}/api/{path}")
    token = os.environ.get("QLH_PRODUCT_AUTH_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def run(command, timeout=60):
    """Preserve the complete output and actual command window, including failures."""
    record = {"command": list(command), "started_at": time.time()}
    try:
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
        record.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else value or ""
        record.update(returncode=None, stdout=text(exc.stdout), stderr=text(exc.stderr), error="command timed out")
    except OSError as exc:
        record.update(returncode=None, stdout="", stderr=str(exc), error=type(exc).__name__)
    record["finished_at"] = time.time()
    return record


def write_evidence(out_dir, subdir, name, payload):
    path = pathlib.Path(out_dir) / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # An existing success or failure record is never repurposed for a new run.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"path": path.relative_to(out_dir).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _identity(proc):
    return {"pid": proc.pid, "created_at": proc.create_time()}


def collect_master_processes():
    """Observe the listener owner and every descendant, including native children."""
    import psutil
    start = time.time()
    result = {"started_at": start, "listeners": [], "process_tree": {"complete": False, "processes": []}}
    try:
        listeners = [c for c in psutil.net_connections("tcp") if c.status == "LISTEN"
                     and c.laddr and c.laddr.port in (8000, 8888)]
        result["listeners"] = [{"pid": c.pid, "address": c.laddr.ip, "port": c.laddr.port} for c in listeners]
        owners = {c.pid for c in listeners if c.pid is not None}
        if len(owners) != 1 or {c.laddr.port for c in listeners} != {8000, 8888}:
            raise ObservationError("master API/TCP listeners do not have one observed process owner")
        root = psutil.Process(owners.pop())
        result["identity_before"] = _identity(root)
        processes = [root, *root.children(recursive=True)]
        complete = True
        for proc in processes:
            entry = {"pid": proc.pid}
            try:
                entry.update(_identity(proc), parent_pid=proc.ppid(), command=proc.cmdline(), cwd=proc.cwd())
                children = sorted(c.pid for c in proc.children())
                entry.update(child_pids=children, children_complete=True)
                entry["model_mappings"] = sorted({m.path for m in proc.memory_maps(grouped=False)
                                                   if m.path.lower().endswith((".gguf", ".safetensors", ".bin", ".pt", ".pth"))})
                entry["maps_complete"] = True
                entry["connections"] = [{"status": c.status, "local": list(c.laddr or []),
                                          "remote": list(c.raddr or [])}
                                         for c in proc.net_connections(kind="tcp")]
                entry["connections_complete"] = True
                entry["identity_after"] = _identity(psutil.Process(proc.pid))
                entry["children_complete"] = children == sorted(c.pid for c in proc.children())
            except (psutil.Error, OSError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                complete = False
            result["process_tree"]["processes"].append(entry)
        result["identity_after"] = _identity(psutil.Process(root.pid))
        if {p.pid for p in processes} != {root.pid, *(p.pid for p in root.children(recursive=True))}:
            complete = False
            result["children_error"] = "process tree changed during observation"
        result["process_tree"]["complete"] = complete
    except (psutil.Error, OSError, ObservationError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["finished_at"] = time.time()
    return result


def master_evidence(operation_id):
    from package_contract.pc_product_observer import observe_pc
    return observe_pc(node_id="master", platform="master", base=BASE, operation_id=operation_id)


def rog_evidence(operation_id, label, host, python_exe, probe, *, node_id="rog-5060", api_base="http://127.0.0.1:8001"):
    # Arguments are restricted because ssh passes its remote command through a shell.
    import re
    if not all(re.fullmatch(r"[A-Za-z0-9_./:@+\\-]+", value) for value in (operation_id, label, host, python_exe, probe, node_id, api_base)):
        raise ObservationError("unsafe/unsupported SSH probe argument")
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
               f"{python_exe} -B -X utf8 {probe} --operation-id {operation_id} --node-id {node_id} --base {api_base}"]
    execution = run(command, timeout=300)
    try:
        payload = parse_json_output(execution["stdout"])
    except ObservationError as exc:
        return {"role": "client", "ssh": execution, "collection_errors": [str(exc)]}
    if payload.get("kind") == "product-node-observation" and payload.get("schema_version") == 1:
        # Preserve remote timestamps; the SSH transport brackets them, it does not
        # supply new observation timestamps or a fallback revision.
        payload["transport"] = execution
        return payload
    return {"role": "client", "ssh": execution, "probe": payload,
            "collection_errors": ["legacy ROG probe lacks complete v1 runtime/PID-bound synced-manifest rehash evidence"]}


def _android_identity(pid_output, stat_output):
    import re
    pid_text = pid_output.strip()
    if not re.fullmatch(r"[1-9][0-9]*", pid_text):
        raise ObservationError("expected one Android application PID")
    # Linux /proc stat field 2 can contain spaces/parentheses. starttime is field 22.
    left, sep, right = stat_output.rpartition(")")
    fields = right.split()
    if not sep or not left.startswith(pid_text + " (") or len(fields) < 20 or not fields[19].isdigit():
        raise ObservationError("missing Android PID start-time identity")
    return {"pid": int(pid_text), "created_at": int(fields[19])}


def _package_path(output):
    paths = [line.removeprefix("package:") for line in output.splitlines() if line.startswith("package:")]
    if len(paths) != 1 or not paths[0].startswith("/data/app/") or not paths[0].endswith("/base.apk"):
        raise ObservationError("missing/ambiguous installed APK path (split APKs require full provenance)")
    return paths[0]


def android_evidence(serial, adb, raw_path, package="com.qlh.inference.debug"):
    """Extract provenance from installed APK bytes; never use a local build-info."""
    start, commands, binding = time.time(), [], {}
    evidence = {"role": "client", "commands": commands, "started_at": start, "collection_errors": []}

    def execute(arguments):
        result = run([adb, "-s", serial, *arguments])
        commands.append(result)
        if result["returncode"] != 0:
            raise ObservationError("Android collection command failed: " + " ".join(arguments))
        return result["stdout"]

    try:
        pid_before = execute(["shell", "pidof", package])
        pid_text = pid_before.strip()
        if not pid_text.isdecimal() or int(pid_text) <= 0:
            raise ObservationError("expected one Android application PID")
        stat_before = execute(["shell", "run-as", package, "cat", f"/proc/{pid_text}/stat"])
        before = _android_identity(pid_before, stat_before)
        path_before = _package_path(execute(["shell", "pm", "path", package]))
        hash_before = execute(["shell", "sha256sum", path_before]).split()[0]
        command = [sys.executable, "-B", "-X", "utf8", str(ANDROID_PROBE), "--adb", adb, "--serial", serial,
                   "--package", package, "--product", "--require-process-tree", "--output", str(raw_path)]
        result = run(command, timeout=150)
        commands.append(result)
        if pathlib.Path(raw_path).is_file():
            evidence["probe"] = json.loads(pathlib.Path(raw_path).read_text(encoding="utf-8-sig"))
        if result["returncode"] != 0:
            raise ObservationError("Android runtime probe failed")
        with tempfile.TemporaryDirectory(prefix="qlh-installed-apk-") as temporary:
            pulled = pathlib.Path(temporary) / "base.apk"
            execute(["pull", path_before, str(pulled)])
            apk_bytes = pulled.read_bytes()
            with zipfile.ZipFile(io.BytesIO(apk_bytes)) as archive:
                entries = [item for item in archive.infolist() if item.filename == "assets/build-info.json"]
                if len(entries) != 1:
                    raise ObservationError("installed APK build-info missing/duplicated")
                info = parse_json_output(archive.read(entries[0]).decode("utf-8-sig"))
        path_after = _package_path(execute(["shell", "pm", "path", package]))
        hash_after = execute(["shell", "sha256sum", path_after]).split()[0]
        pid_after = execute(["shell", "pidof", package])
        stat_after = execute(["shell", "run-as", package, "cat", f"/proc/{pid_text}/stat"])
        after = _android_identity(pid_after, stat_after)
        digest = hashlib.sha256(apk_bytes).hexdigest()
        if before != after or path_before != path_after or hash_before != hash_after or digest != hash_before:
            raise ObservationError("Android PID/start identity or installed APK path/content changed during extraction")
        observation = (evidence.get("probe") or {}).get("observation") or {}
        if observation.get("pid") != before["pid"] or observation.get("package_name") != package:
            raise ObservationError("Android runtime probe is not bound to installed package PID")
        canonical = json.dumps(info, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        binding = {**before, "kind": "installed_apk", "extraction": "apk_zip", "package": package,
                   "path_before": path_before, "path_after": path_after, "sha256_before": hash_before,
                   "sha256_after": hash_after, "pulled_sha256": digest, "build_info_entry": "assets/build-info.json",
                   "build_info": info, "build_info_sha256": hashlib.sha256(canonical).hexdigest()}
        evidence["installed_package_binding"] = binding
        evidence["identity_before"], evidence["identity_after"] = before, after
        runtime = observation.get("runtime_receipt")
        process = observation.get("process")
        if not isinstance(runtime, dict) or not isinstance(process, dict):
            raise ObservationError("Android runtime probe lacks committed loader receipt or process tree")
        if (runtime.get("pid"), runtime.get("created_at")) != (before["pid"], before["created_at"]):
            raise ObservationError("Android runtime receipt PID/start differs from installed process")
        if process.get("identity_before") != before or process.get("identity_after") != after:
            raise ObservationError("Android process tree belongs to another process generation")
        evidence.update(schema_version=1, kind="product-node-observation", platform="android",
                        node_id=runtime.get("node_id"), facts={"identity_before": before,
                        "identity_after": after, "runtime": runtime, "source": binding,
                        "process_tree": {"complete": process.get("complete"), "processes": process.get("processes"),
                                         "errors": process.get("errors", [])}})
        if process.get("complete") is not True:
            raise ObservationError("Android process tree observation incomplete: " + str(process.get("errors", [])))
    except (ObservationError, OSError, ValueError, zipfile.BadZipFile, KeyError, IndexError) as exc:
        evidence["collection_errors"].append(str(exc))
    evidence["finished_at"] = time.time()
    return evidence


def produce_snapshots(*, intent, raw_nodes, node_addresses, revision, out_dir, mode):
    """Write raw captures and only emit topology/native after every node passes."""
    out_dir = pathlib.Path(out_dir)
    subdir = f"{mode.lower()}-nodes"
    assignments = intent.get("plan", {}).get("assignments", [])
    names = [a.get("node_id") for a in assignments]
    # Preflight every output before writing; stale green snapshots cannot survive
    # beside a new failed report, and historical evidence is never overwritten.
    targets = [out_dir / n for n in ("topology.json", "native.json", "producer-result.json")]
    targets += [out_dir / subdir / f"{n}.json" for n in names if isinstance(n, str)]
    if any(p.exists() for p in targets):
        raise ObservationError("refusing existing snapshot/raw output; choose a fresh capture directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = {k: intent.get(k) for k in ("operation_id", "config_id", "generation")}
    errors, derived = [], []
    if not assignments or len(set(names)) != len(names) or "master" not in names:
        errors.append("missing/duplicate observed intent assignments")
    if intent.get("status") != "ready" or intent.get("loaded_ready") is not True:
        errors.append("model intent is not ready/loaded_ready")
    for assignment in assignments:
        node = assignment.get("node_id")
        if not isinstance(node, str) or not node or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in node):
            errors.append("invalid node ID for evidence path")
            continue
        raw = raw_nodes.get(node, {"collection_errors": ["no observer for node"]})
        ref = write_evidence(out_dir, subdir, f"{node}.json", raw)
        expected = dict(assignment, _total_layers=intent.get("plan", {}).get("total_layers"),
                        artifact_sha256=intent.get("nodes", {}).get(node, {}).get("artifact_sha256"))
        try:
            if expected["artifact_sha256"] is None:
                raise ObservationError("intent lacks expected per-node artifact_sha256")
            result = derive_node_observation(raw, expected_revision=revision, expected_node_id=node,
                expected_identity=identity, expected_assignment=expected, node_addresses=node_addresses)
            peers = set(result["topology"]["worker_peers"])
            if peers != (set(names)-{"master"} if node == "master" else {"master"}):
                raise ObservationError("observed OS peers differ from required star edges")
            for section in ("topology", "native"):
                result[section]["raw_evidence"] = dict(ref)
            derived.append(result)
        except (ObservationError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{node}: {exc}")
            errors.extend(f"{node}: {reason}" for reason in raw.get("collection_errors", []))
    report = {"passed": not errors, "errors": errors, "operation_id": identity["operation_id"],
              "scope": "derived-current-observations-not-release-acceptance"}
    if not errors:
        # Original node observation times, never a new packaging timestamp.
        common = {**identity, "captured_at": max(item["native"]["captured_at"] for item in derived),
                  "observation_started_at": min(item["native"]["observation_started_at"] for item in derived),
                  "observation_finished_at": max(item["native"]["observation_finished_at"] for item in derived),
                  "node_addresses": node_addresses, "intent_status": intent["status"], "loaded_ready": intent["loaded_ready"]}
        topology = {**common, "source": "os_process_connections", "complete": True,
                    "nodes": [item["topology"] for item in derived]}
        native = {**common, "scope": "native-mapping-and-segment-evidence",
                  "nodes": [item["native"] for item in derived]}
        write_evidence(out_dir, "", "topology.json", topology)
        write_evidence(out_dir, "", "native.json", native)
        report.update(topology=str(out_dir / "topology.json"), native=str(out_dir / "native.json"))
    write_evidence(out_dir, "", "producer-result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--revision", required=True, help="expected revision only; never a source fallback")
    parser.add_argument("--node-addresses", required=True, type=pathlib.Path,
                        help="independent node ID -> IP list JSON, not derived from the plan")
    parser.add_argument("--serial", help="required for live Android capture")
    parser.add_argument("--mode", choices=["LL", "DL"], default="LL")
    parser.add_argument("--label", default="ready")
    parser.add_argument("--rog-host", default="ROG@100.116.17.12")
    parser.add_argument("--rog-python", default="C:/Users/ROG/AppData/Local/Programs/Python/Python312/python.exe")
    parser.add_argument("--rog-probe", default="C:/Users/ROG/qlh-runtime-20261015/scripts/package_contract/pc_product_observer.py")
    parser.add_argument("--rog-base", default="http://127.0.0.1:8001")
    parser.add_argument("--pc-observers", type=pathlib.Path, help="node -> host/python/probe/base read-only observer map")
    parser.add_argument("--observations-dir", type=pathlib.Path,
                        help="consume unchanged v1 node reports with their original observation windows; never upgrade legacy probes")
    args = parser.parse_args(argv)
    try:
        addresses = json.loads(args.node_addresses.read_text(encoding="utf-8-sig"))
        intent = api(f"models/intent/{args.operation_id}")
        if intent.get("operation_id") != args.operation_id:
            raise ObservationError("API returned a different operation")
        pc_observers = json.loads(args.pc_observers.read_text(encoding="utf-8-sig")) if args.pc_observers else {}
        raws = {}
        for assignment in intent.get("plan", {}).get("assignments", []):
            node = assignment["node_id"]
            if not isinstance(node, str) or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in node):
                raise ObservationError("unsafe node ID in intent")
            if args.observations_dir:
                raws[node] = json.loads((args.observations_dir / f"{node}.json").read_text(encoding="utf-8-sig"))
            elif node == "master":
                raws[node] = master_evidence(args.operation_id)
            elif node in pc_observers or node == "rog-5060":
                config = pc_observers.get(node) or {"host": args.rog_host, "python": args.rog_python,
                                                   "probe": args.rog_probe, "base": args.rog_base}
                raws[node] = rog_evidence(args.operation_id, args.label, config["host"], config["python"],
                                          config["probe"], node_id=node, api_base=config["base"])
            elif node.startswith("android-"):
                if not args.serial:
                    raise ObservationError("--serial is required for Android observation")
                args.out_dir.mkdir(parents=True, exist_ok=True)
                raw_path = args.out_dir / "_android-probe-raw.json"
                if raw_path.exists():
                    raise ObservationError("refusing existing Android probe output")
                raws[node] = android_evidence(args.serial, shutil.which("adb") or "adb", raw_path)
            else:
                raws[node] = {"collection_errors": ["no read-only observer configured for node"]}
        report = produce_snapshots(intent=intent, raw_nodes=raws, node_addresses=addresses,
                                  revision=args.revision, out_dir=args.out_dir, mode=args.mode)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["passed"] else 1
    except (ObservationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
