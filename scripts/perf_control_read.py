"""Local four-arm control-read experiment; launch changes no production HTTP routes."""
from __future__ import annotations

import argparse
import collections
from contextlib import ExitStack
import datetime
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
import weakref

ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "legacy": (False, False),
    "execution": (True, False),
    "query": (False, True),
    "combined": (True, True),
}


def clock_sample():
    before = time.time_ns()
    mono = time.monotonic_ns()
    after = time.time_ns()
    return {
        "epoch_ns": (before + after) // 2,
        "monotonic_ns": mono,
        "wall_bracket_ns": after - before,
        "process_time": time.process_time(),
        "process_time_ns": time.process_time_ns(),
        "iso_time": datetime.datetime.now().astimezone().isoformat(),
    }


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for attempt in range(50):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(.02)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_bytes())


def workspace(options):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", options.label):
        raise RuntimeError("invalid_label")
    return Path(options.output_dir).resolve() / options.label


def promote_environment():
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.startswith("QLH_") and not os.environ.get(key):
            os.environ[key] = value


def source_manifest():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths.extend([ROOT / "qlh.py", ROOT / "koakuma.bat", Path(__file__).resolve()])
    files = {}
    for path in paths:
        content = path.read_bytes()
        files[path.relative_to(ROOT).as_posix()] = {
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)
        }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {"head": head, "worktree_dirty": bool(dirty.strip()), "files": files}


class Controller:
    def __init__(self, directory, mode):
        self.directory = directory
        self.mode = mode
        self.repositories = weakref.WeakSet()
        self.supervisors = weakref.WeakSet()
        self.registry_lock = threading.RLock()
        self.stopping = threading.Event()
        self.process_id = os.getpid()
        self.launch_id = uuid.uuid4().hex
        self.seen = set()
        self.profile = None
        self.thread = None

    def install(self):
        import qlh_core.adapters.model_control_repository as profile
        from tui_backend import BackendSupervisor

        self.profile = profile
        repository_type = profile.SqliteModelControlRepository
        original_init = repository_type.__init__
        controller = self

        @functools.wraps(original_init)
        def tracked_init(repository, *args, **kwargs):
            original_init(repository, *args, **kwargs)
            with controller.registry_lock:
                with repository._read_lock:
                    execution, query = MODES[controller.mode]
                    repository.execution_reads_enabled = execution
                    repository.query_reads_enabled = query
                    controller.repositories.add(repository)

        repository_type.__init__ = tracked_init
        for method_name in ("read_execution", "read_query", "read_stamp"):
            original = getattr(repository_type, method_name)

            def guarded(repository, *args, _original=original, **kwargs):
                with repository._read_lock:
                    return _original(repository, *args, **kwargs)

            setattr(repository_type, method_name, functools.wraps(original)(guarded))
        original_ready = BackendSupervisor.ensure_ready

        @functools.wraps(original_ready)
        def owned_ready(supervisor):
            result = original_ready(supervisor)
            if not supervisor.started_here:
                raise RuntimeError("benchmark_refuses_existing_backend")
            with controller.registry_lock:
                controller.supervisors.add(supervisor)
            return result

        BackendSupervisor.ensure_ready = owned_ready

    def flush(self):
        with self.profile._PROFILE_LOCK:
            buffered = len(self.profile._PROFILE_RECORDS)
            self.profile._flush_profile_control_reads()
            if self.profile._PROFILE_RECORDS:
                raise RuntimeError("profiler_flush_failed")
            path = Path(self.profile._PROFILE_PATH)
            if not path.exists():
                path.touch(exist_ok=True)
            return {"path": str(path), "offset_bytes": path.stat().st_size,
                    "flushed_records": buffered, "clock": clock_sample()}

    def apply(self, command):
        api = sys.modules.get("api_server")
        if api is None or not api._get_app_runtime().readiness().get("ready"):
            return None
        scheduler = api.scheduler
        owner = scheduler._model_control
        requests = scheduler._execution_app.requests
        with requests._condition:
            if requests._active is not None or requests._queue:
                return None
            with self.registry_lock:
                repositories = sorted(self.repositories, key=id)
                active = owner.repository
                if active not in repositories:
                    raise RuntimeError("active_repository_not_captured")
                servers = [server for server in self.supervisors
                           if server.started_here and server.thread is not None and server.thread.is_alive()]
                if not servers:
                    return None
                mode = command.get("mode", self.mode)
                if mode not in MODES or command.get("action") not in ("switch", "flush"):
                    raise RuntimeError("invalid_control_command")
                with ExitStack() as stack:
                    for repository in repositories:
                        stack.enter_context(repository._read_lock)
                    if command["action"] == "switch":
                        self.mode = mode
                        for repository in repositories:
                            repository.execution_reads_enabled, repository.query_reads_enabled = MODES[mode]
                            if command.get("reset_cache", True):
                                repository._read_token = None
                                repository._read_state = None
                                repository._read_schema = None
                    state = active.read()
                    config_id = state.get("active_deployment")
                    deployment = state.get("deployments", {}).get(config_id, {})
                    ack = {
                        "status": "ok", "command_id": command["command_id"],
                        "sequence": command["sequence"], "launch_id": self.launch_id,
                        "action": command["action"], "mode": self.mode,
                        "process_id": self.process_id, "process_time": time.process_time(),
                        "clock": clock_sample(), "boot_id": owner.boot_id,
                        "state_boot_id": state.get("boot_id"), "revision": state.get("revision"),
                        "active_config": config_id, "generation": deployment.get("generation"),
                        "operation_id": deployment.get("operation_id"), "queue_idle": True,
                        "backend_started_here": True, "active_repository_id": id(active),
                        "cache_reset": command["action"] == "switch" and command.get("reset_cache", True),
                        "repositories": [{"instance_id": id(repository),
                                          "execution_reads_enabled": repository.execution_reads_enabled,
                                          "query_reads_enabled": repository.query_reads_enabled}
                                         for repository in repositories],
                        "threads": [{"name": thread.name, "ident": thread.ident,
                                     "native_id": thread.native_id}
                                    for thread in threading.enumerate()],
                    }
                    ack["profiler"] = self.flush()
                    ack["process_time"] = ack["profiler"]["clock"]["process_time"]
                    return ack

    def loop(self):
        while not self.stopping.wait(0.1):
            try:
                command = read_json(self.directory / "arm-control.json")
                identity = command["command_id"]
                if identity in self.seen or command.get("launch_id") != self.launch_id:
                    continue
                ack = self.apply(command)
                if ack is None:
                    continue
                atomic_json(self.directory / "arm-ack.json", ack)
                self.seen.add(identity)
                with (self.directory / "ack-history.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ack, ensure_ascii=False) + "\n")
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            except Exception as error:
                atomic_json(self.directory / "controller-error.json", {
                    "error_type": type(error).__name__, "process_id": self.process_id,
                    "launch_id": self.launch_id, "clock": clock_sample(),
                })


def issue_control(directory, mode=None, *, timeout=120, reset_cache=True):
    launch = read_json(directory / "launch.json")
    command = {"command_id": uuid.uuid4().hex, "sequence": time.time_ns(),
               "launch_id": launch["launch_id"], "action": "switch" if mode else "flush",
               "reset_cache": reset_cache, "requested_clock": clock_sample()}
    if mode:
        command["mode"] = mode
    lock = directory / "control-client.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.close(descriptor)
        atomic_json(directory / "arm-control.json", command)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                ack = read_json(directory / "arm-ack.json")
                if ack.get("command_id") == command["command_id"]:
                    if ack.get("process_id") != launch["process_id"] or ack.get("launch_id") != launch["launch_id"]:
                        raise RuntimeError("ack_process_identity_mismatch")
                    if ack.get("status") != "ok" or (mode and ack.get("mode") != mode):
                        raise RuntimeError("control_ack_rejected")
                    return ack
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.1)
        raise RuntimeError("control_ack_timeout")
    finally:
        lock.unlink(missing_ok=True)


def launch(options):
    promote_environment()
    directory = workspace(options)
    port = options.port or int(os.environ.get("QLH_BACKEND_PORT") or os.environ.get("QLH_API_PORT") or 8000)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            raise RuntimeError("benchmark_refuses_existing_backend")
    except (ConnectionRefusedError, TimeoutError):
        pass
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["QLH_PROFILE_CONTROL_READS"] = str(directory / "control-reads.tsv")
    os.environ["QLH_CONTROL_READ_MODE"] = options.mode
    os.chdir(ROOT)
    for path in (str(ROOT), str(ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    controller = Controller(directory, options.mode)
    atomic_json(directory / "source-manifest.json", source_manifest())
    controller.install()
    atomic_json(directory / "launch.json", {
        "process_id": os.getpid(), "launch_id": controller.launch_id,
        "clock": clock_sample(), "python_executable": sys.executable,
        "api_port": port, "initial_mode": options.mode,
        "profile_path": str(directory / "control-reads.tsv"),
    })
    atomic_json(directory / "arm-control.json", {
        "command_id": "initial-" + uuid.uuid4().hex, "sequence": time.time_ns(),
        "launch_id": controller.launch_id, "action": "switch", "mode": options.mode,
        "reset_cache": True, "requested_clock": clock_sample(),
    })
    controller.thread = threading.Thread(target=controller.loop, daemon=True, name="perf-control-arms")
    controller.thread.start()
    sys.argv = [str(ROOT / "qlh.py"), "--no-splash", "--port", str(port)]
    try:
        runpy.run_path(str(ROOT / "qlh.py"), run_name="__main__")
    finally:
        controller.stopping.set()
        controller.thread.join(timeout=5)
        controller.flush()


def switch(options):
    if bool(options.mode) == bool(options.flush_only):
        raise RuntimeError("choose_mode_or_flush_only")
    ack = issue_control(workspace(options), options.mode, timeout=options.timeout,
                        reset_cache=not options.keep_cache)
    print(json.dumps({key: ack[key] for key in ("status", "mode", "process_id", "process_time",
                                               "boot_id", "revision", "active_config", "generation")},
                     ensure_ascii=False))


def http_json(base, endpoint, body=None, timeout=30):
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(base + endpoint, data=payload,
                                     headers={"Content-Type": "application/json"})
    with client.open(request, timeout=timeout) as response:
        return json.load(response)


def snapshot_api(base, directory, prefix):
    result = {}
    for name, endpoint in {
        "ready": "/api/ready", "queue": "/api/cluster/queue",
        "intent": "/api/models/intent/current", "nodes": "/api/cluster/nodes",
        "capacity": "/api/cluster/pipeline-capacity", "layers": "/api/cluster/layers",
    }.items():
        result[name] = http_json(base, endpoint)
        atomic_json(directory / (prefix + "-" + name + ".json"), result[name])
    return result


def acknowledgement_identity(ack):
    return {key: ack.get(key) for key in
            ("process_id", "launch_id", "boot_id", "active_config", "generation")}


def require_ack(ack, mode, identity=None):
    if ack.get("mode") != mode or not ack.get("queue_idle") or not ack.get("backend_started_here"):
        raise RuntimeError("sampling_arm_not_ready")
    if not ack.get("active_config") or ack.get("boot_id") != ack.get("state_boot_id"):
        raise RuntimeError("sampling_deployment_not_ready")
    expected = MODES[mode]
    if not ack.get("repositories") or any(
        (item.get("execution_reads_enabled"), item.get("query_reads_enabled")) != expected
        for item in ack["repositories"]
    ):
        raise RuntimeError("repository_mode_mismatch")
    actual = acknowledgement_identity(ack)
    if identity is not None and actual != identity:
        raise RuntimeError("sampling_identity_changed")
    return actual


def source_difference(runtime, capture):
    return sorted(name for name in set(runtime["files"]) | set(capture["files"])
                  if runtime["files"].get(name) != capture["files"].get(name))


def timing_summary(values):
    ordered = sorted(values)
    return {"n": len(values), "sum_s": sum(values),
            "mean_s": statistics.mean(values) if values else None,
            "median_s": statistics.median(values) if values else None,
            "p95_nearest_lower_s": ordered[int((len(ordered) - 1) * .95)] if ordered else None}


def read_stage_events(log_directory, request_id):
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", request_id):
        raise RuntimeError("invalid_request_identity")
    lines, seen, sources = [], set(), []
    for path in sorted(log_directory.glob("qlh-*.log*")):
        if not path.is_file():
            continue
        matched = 0
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, 1):
                if request_id not in line or not any(
                    marker in line for marker in ("Route-A stage handoff:", "Route-A stage completed:")
                ):
                    continue
                matched += 1
                text = line.rstrip("\r\n")
                if text not in seen:
                    seen.add(text)
                    lines.append({"file": path.name, "line": lineno, "text": text})
        if matched:
            sources.append({"file": path.name, "matched_lines_before_dedup": matched,
                            "observed_bytes": path.stat().st_size})
    lines.sort(key=lambda item: (item["text"][:23], item["file"], item["line"]))
    return lines, sources


def pair_stages(events, expected_steps=32):
    pattern = re.compile(r"Route-A stage handoff: node=(\S+) stage=(\S+) tokens=(\d+) .*?stage_request_id=(\S+)")
    handoffs, completions = collections.defaultdict(list), collections.defaultdict(list)
    receipts = []
    for event in events:
        line = event["text"]
        matched = pattern.search(line)
        if matched:
            node, stage, tokens, stage_request = matched.groups()
            instant = datetime.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").astimezone().timestamp()
            candidate = {"handoff_at": instant, "handoff_tokens": int(tokens), "stage": stage}
            if candidate not in handoffs[(node, stage_request)]:
                handoffs[(node, stage_request)].append(candidate)
        if "Route-A stage completed: " in line:
            record = json.loads(line.split("Route-A stage completed: ", 1)[1])
            key = (record["node_id"], record["stage_request_id"])
            if record not in completions[key]:
                completions[key].append(record)
                receipts.append(record)
    paired, errors = [], []
    for key in sorted(set(handoffs) | set(completions)):
        sent, received = handoffs[key], completions[key]
        if len(sent) != 1 or len(received) != 1:
            errors.append({"node_id": key[0], "stage_request_id": key[1],
                           "handoffs": len(sent), "completions": len(received)})
            continue
        record = {**received[0], **sent[0]}
        record["handoff_to_completed_s"] = record["completed_at"] - record["handoff_at"]
        if record["handoff_to_completed_s"] < 0:
            errors.append({"negative_stage_duration": key})
        paired.append(record)
    by_step = collections.defaultdict(list)
    for record in paired:
        matched = re.search(r":step:(\d+)$", record["stage_request_id"])
        if matched:
            by_step[int(matched.group(1))].append(record)
    steps = []
    for step, records in sorted(by_step.items()):
        records.sort(key=lambda item: item["handoff_at"])
        first = min(item["handoff_at"] for item in records)
        last = max(item["completed_at"] for item in records)
        gaps = [records[index + 1]["handoff_at"] - item["completed_at"]
                for index, item in enumerate(records[:-1])]
        if len(records) != 2 or len({item["node_id"] for item in records}) != 2 or any(gap < 0 for gap in gaps):
            errors.append({"invalid_serial_step": step})
        steps.append({"step": step, "first_handoff_at": first, "last_completed_at": last,
                      "remote_window_s": last - first, "within_remote_gap_s": sum(gaps),
                      "nodes": [item["node_id"] for item in records]})
    if sorted(by_step) != list(range(expected_steps)):
        errors.append({"expected_steps": expected_steps, "actual_steps": sorted(by_step)})
    expected_stages = expected_steps * 2 + 2
    if len(paired) != expected_stages:
        errors.append({"expected_stages": expected_stages, "actual_stages": len(paired)})
    intervals = [steps[index + 1]["last_completed_at"] - item["last_completed_at"]
                 for index, item in enumerate(steps[:-1])]
    outside = [steps[index + 1]["first_handoff_at"] - item["last_completed_at"]
               for index, item in enumerate(steps[:-1])]
    return receipts, paired, {"valid": not errors, "errors": errors, "steps": steps,
                              "decode_step_intervals": timing_summary(intervals),
                              "between_steps_outside_remote_window": timing_summary(outside),
                              "stage_handoffs": sum(len(value) for value in handoffs.values()),
                              "stage_completions": len(receipts)}


def tsv_slice(path, start, end):
    if start < 0 or end < start:
        raise RuntimeError("invalid_profiler_offsets")
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    if len(data) != end - start or (data and not data.endswith(b"\n")):
        raise RuntimeError("incomplete_profiler_slice")
    records = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 4:
            raise RuntimeError("unexpected_profiler_schema")
        stamp, kind, size, thread = fields
        records.append({"monotonic_s": float(stamp), "category": kind,
                        "payload_length": int(size), "thread": thread, "slice_line": line_number})
    return data, records


def count_reads(records):
    result = {}
    for thread in sorted({record["thread"] for record in records}):
        subset = [record for record in records if record["thread"] == thread]
        sizes = [record["payload_length"] for record in subset if record["category"] == "row"]
        result[thread] = {"categories": dict(collections.Counter(record["category"] for record in subset)),
                          "payload_length_sum": sum(sizes), "payload_bytes_assuming_ascii": sum(sizes),
                          "row_length_min": min(sizes) if sizes else None,
                          "row_length_max": max(sizes) if sizes else None}
    return result


def window_reads(records, lower, upper):
    return count_reads([record for record in records if lower <= record["monotonic_s"] < upper])


def summarize_reads(records, request, steps):
    before, after = request["before_clock"], request["after_clock"]
    start, end = before["monotonic_ns"] / 1e9, after["monotonic_ns"] / 1e9
    offset = (before["epoch_ns"] - before["monotonic_ns"]) / 1e9
    offset_end = (after["epoch_ns"] - after["monotonic_ns"]) / 1e9
    uncertainty = max(request["clock_resolution_s"], abs(offset_end - offset), .001)
    result = {"units": "len(payload); UTF-8 bytes if canonical ASCII; logical payload, not disk IO",
              "http_window": {"start_monotonic_s": start, "end_monotonic_s": end,
                              "threads": window_reads(records, start, end)},
              "wall_minus_monotonic_start_s": offset, "wall_minus_monotonic_end_s": offset_end,
              "clock_offset_drift_s": offset_end - offset, "step_boundary_uncertainty_s": uncertainty,
              "whole_ack_slice_threads": count_reads(records), "decode_windows": []}
    for index, step in enumerate(steps):
        if index == 0:
            continue
        lower = steps[index - 1]["last_completed_at"] - offset
        upper = step["last_completed_at"] - offset
        result["decode_windows"].append({
            "step": step["step"], "start_monotonic_s": lower, "end_monotonic_s": upper,
            "threads": window_reads(records, lower, upper),
            "threads_shift_minus_uncertainty": window_reads(records, lower - uncertainty, upper - uncertainty),
            "threads_shift_plus_uncertainty": window_reads(records, lower + uncertainty, upper + uncertainty),
            "boundary_nearby_events": sum(
                min(abs(record["monotonic_s"] - lower), abs(record["monotonic_s"] - upper)) <= uncertainty
                for record in records)})
    return result


def api_identity(intent):
    result = {"active_config": intent.get("config_id") or intent.get("deployment_id"),
              "generation": intent.get("generation"), "boot_id": intent.get("boot_id"),
              "operation_id": intent.get("operation_id")}
    if any(value is None or value == "" for value in result.values()):
        raise RuntimeError("missing_api_deployment_identity")
    if not intent.get("loaded_ready") or not intent.get("runnable"):
        raise RuntimeError("api_deployment_not_ready")
    return result


def request_sample(base, experiment, directory, mode, identity, deployment, options, warmup):
    directory.mkdir(parents=True, exist_ok=False)
    body = {"message": "List the integers from 1 to 100 in order, separated by commas. Output the list only.",
            "max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0, "enable_thinking": False,
            "streaming_mode": "full", "routing_preference": "distributed_required",
            "session_id": "control-read-" + uuid.uuid4().hex}
    atomic_json(directory / "input.json", body)
    observed = http_json(base, "/api/models/intent/current")
    atomic_json(directory / "before-intent.json", observed)
    if api_identity(observed) != deployment:
        raise RuntimeError("api_deployment_changed")
    queue = http_json(base, "/api/cluster/queue")
    if queue.get("is_busy") or queue.get("queue_size"):
        raise RuntimeError("runtime_busy_before_sample")
    before_ack = issue_control(experiment, timeout=options.control_timeout)
    require_ack(before_ack, mode, identity)
    atomic_json(directory / "before-ack.json", before_ack)
    before = clock_sample()
    started = time.perf_counter()
    try:
        response = http_json(base, "/api/chat", body, options.request_timeout)
    except Exception as error:
        atomic_json(directory / "error.json", {"error_type": type(error).__name__,
                    "before_clock": before, "after_clock": clock_sample(), "warmup": warmup})
        raise
    elapsed, after = time.perf_counter() - started, clock_sample()
    atomic_json(directory / "response.json", response)
    metrics, durable = response.get("metrics", {}), response.get("metrics", {}).get("request", {})
    request_id = durable.get("request_id")
    request = {"request_id": request_id, "mode": mode, "warmup": warmup,
               "http_elapsed_s": elapsed, "before_clock": before, "after_clock": after,
               "clock_resolution_s": time.get_clock_info("monotonic").resolution,
               "identity": identity, "api_deployment_identity": deployment,
               "tokens_generated": metrics.get("tokens_generated"), "stop_reason": metrics.get("stop_reason"),
               "distributed_used": metrics.get("distributed_used"), "fallback": metrics.get("fallback"),
               "plan_digest": durable.get("plan_digest"),
               "response_sha256": hashlib.sha256((directory / "response.json").read_bytes()).hexdigest(),
               "output_sha256": hashlib.sha256(response.get("content", "").encode("utf-8")).hexdigest(),
               "token_sequence_sha256": hashlib.sha256(json.dumps(
                   [step.get("token") for step in metrics.get("steps", [])], separators=(",", ":")
               ).encode("utf-8")).hexdigest()}
    atomic_json(directory / "request.json", request)
    journal = http_json(base, "/api/requests")
    atomic_json(directory / "journal.json", [record for record in journal.get("requests", [])
                                            if record.get("request_id") == request_id])
    after_ack = issue_control(experiment, timeout=options.control_timeout)
    atomic_json(directory / "after-ack.json", after_ack)
    require_ack(after_ack, mode, identity)
    observed = http_json(base, "/api/models/intent/current")
    atomic_json(directory / "after-intent.json", observed)
    if api_identity(observed) != deployment:
        raise RuntimeError("api_deployment_changed")
    if not request_id:
        raise RuntimeError("missing_durable_request_identity")
    profile_path = Path(before_ack["profiler"]["path"])
    if profile_path != Path(after_ack["profiler"]["path"]):
        raise RuntimeError("profiler_path_changed")
    lower, upper = before_ack["profiler"]["offset_bytes"], after_ack["profiler"]["offset_bytes"]
    raw, records = tsv_slice(profile_path, lower, upper)
    (directory / "control-reads.tsv").write_bytes(raw)
    events, sources = read_stage_events(Path(options.log_dir), request_id)
    (directory / "request-events.log").write_text("\n".join(event["text"] for event in events) + "\n", encoding="utf-8")
    atomic_json(directory / "log-sources.json", {"sources": sources, "events": events})
    receipts, paired, timing = pair_stages(events)
    atomic_json(directory / "receipts.json", receipts)
    atomic_json(directory / "paired-stages.json", paired)
    if (metrics.get("tokens_generated") != 32 or len(metrics.get("steps", [])) != 32 or
            metrics.get("distributed_used") is not True or metrics.get("fallback") is not False or
            durable.get("terminal") != "completed" or durable.get("execution_state") != "stopped" or
            metrics.get("model_generation") != deployment["generation"] or
            (metrics.get("model_config_id") or metrics.get("config_id")) != deployment["active_config"]):
        timing["errors"].append({"invalid_response_contract": True})
        timing["valid"] = False
    timing.update(http_elapsed_s=elapsed, tokens_generated=metrics.get("tokens_generated"), warmup=warmup)
    atomic_json(directory / "timing.json", timing)
    reads = summarize_reads(records, request, timing["steps"])
    reads["profiler_offsets"] = {"start_bytes": lower, "end_bytes": upper,
                                  "slice_sha256": hashlib.sha256(raw).hexdigest()}
    cpu = after_ack["process_time"] - before_ack["process_time"]
    wall = (after_ack["profiler"]["clock"]["monotonic_ns"] -
            before_ack["profiler"]["clock"]["monotonic_ns"]) / 1e9
    reads["cpu_ack_window"] = {"process_cpu_s": cpu, "wall_s": wall,
                                "core_equivalent": cpu / wall if wall > 0 else None,
                                "scope": "flush ACK to flush ACK; includes journal/flush overhead outside HTTP"}
    atomic_json(directory / "read-summary.json", reads)
    if not timing["valid"]:
        raise RuntimeError("invalid_sample_preserved")
    return request


def run(options):
    experiment = workspace(options)
    launch_info = read_json(experiment / "launch.json")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", options.sample_label):
        raise RuntimeError("invalid_sample_label")
    if options.count < 0 or options.warmup < 0 or options.count + options.warmup == 0:
        raise RuntimeError("invalid_sample_count")
    modes = [options.mode] if options.mode else options.modes
    if options.reuse_current_arm and len(modes) != 1:
        raise RuntimeError("reuse_requires_one_arm")
    group = experiment / "samples" / options.sample_label
    group.mkdir(parents=True, exist_ok=False)
    lock = experiment / "sampler.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        runtime_sources = read_json(experiment / "source-manifest.json")
        capture_sources = source_manifest()
        drift = source_difference(runtime_sources, capture_sources)
        atomic_json(group / "runtime-source-manifest.json", runtime_sources)
        atomic_json(group / "capture-source-manifest.json", capture_sources)
        atomic_json(group / "capture.json", {"modes": modes, "warmup_per_arm": options.warmup,
                    "samples_per_arm": options.count, "runtime_controller": "launch source manifest",
                    "capture_source_differences": drift, "clock": clock_sample(),
                    "order_bias": "fixed order; final legacy is a drift check, not randomization"})
        if any(name != "scripts/perf_control_read.py" for name in drift):
            raise RuntimeError("production_source_changed_since_launch")
        base = "http://127.0.0.1:" + str(launch_info["api_port"])
        fixed_identity, fixed_deployment, fixed_result = None, None, None
        for arm_index, mode in enumerate(modes, 1):
            arm = group / f"{arm_index:02d}-{mode}"
            arm.mkdir()
            ack = issue_control(experiment, None if options.reuse_current_arm else mode,
                                timeout=options.control_timeout, reset_cache=not options.keep_cache)
            fixed_identity = require_ack(ack, mode, fixed_identity)
            atomic_json(arm / "arm-ack.json", ack)
            before = snapshot_api(base, arm, "before")
            deployment = api_identity(before["intent"])
            if fixed_deployment is not None and deployment != fixed_deployment:
                raise RuntimeError("api_deployment_changed_between_arms")
            if deployment["active_config"] != ack["active_config"] or deployment["boot_id"] != ack["boot_id"]:
                raise RuntimeError("api_and_wrapper_identity_mismatch")
            fixed_deployment = deployment
            if not before["ready"].get("ready"):
                raise RuntimeError("api_not_ready")
            for warmup, count, prefix in ((True, options.warmup, "warmup"), (False, options.count, "sample")):
                for index in range(1, count + 1):
                    directory = arm / f"{prefix}-{index:03d}"
                    captured = request_sample(base, experiment, directory, mode, fixed_identity,
                                              deployment, options, warmup)
                    contract = {key: captured[key] for key in
                                ("plan_digest", "output_sha256", "token_sequence_sha256")}
                    if fixed_result is None:
                        fixed_result = contract
                    elif contract != fixed_result:
                        atomic_json(directory / "contract-mismatch.json", {"expected": fixed_result, "actual": contract})
                        raise RuntimeError("greedy_result_or_plan_changed")
                    print(json.dumps({"arm": mode, "warmup": warmup, "sample": index,
                                      "request_id": captured["request_id"],
                                      "http_elapsed_s": captured["http_elapsed_s"]}), flush=True)
            after = snapshot_api(base, arm, "after")
            if api_identity(after["intent"]) != fixed_deployment:
                raise RuntimeError("api_deployment_changed_after_arm")
            final_ack = issue_control(experiment, timeout=options.control_timeout)
            require_ack(final_ack, mode, fixed_identity)
            atomic_json(arm / "after-arm-ack.json", final_ack)
        final_sources = source_manifest()
        atomic_json(group / "capture-source-manifest-after.json", final_sources)
        if source_difference(capture_sources, final_sources):
            raise RuntimeError("source_changed_during_capture")
        summarize_directory(experiment)
    finally:
        lock.unlink(missing_ok=True)


def summarize_directory(experiment):
    groups = {}
    for path in sorted((experiment / "samples").glob("*/*/sample-*/request.json")):
        directory = path.parent
        request = read_json(path)
        if request.get("warmup"):
            continue
        group_key = directory.parent.relative_to(experiment / "samples").as_posix()
        group = groups.setdefault(group_key, {"mode": request["mode"], "samples": [], "incomplete": []})
        try:
            timing, reads = read_json(directory / "timing.json"), read_json(directory / "read-summary.json")
            if not timing["valid"] or (directory / "contract-mismatch.json").exists():
                group["incomplete"].append(directory.name)
                continue
            executor = reads["http_window"]["threads"].get("linear-request-executor", {})
            group["samples"].append({"path": directory.relative_to(experiment).as_posix(),
                "request_id": request["request_id"], "http_s": request["http_elapsed_s"],
                "decode": timing["decode_step_intervals"],
                "outside": timing["between_steps_outside_remote_window"],
                "execution_row_reads": executor.get("categories", {}).get("row", 0),
                "execution_payload_length": executor.get("payload_length_sum", 0),
                "cpu_ack_window": reads["cpu_ack_window"]})
        except (FileNotFoundError, KeyError):
            group["incomplete"].append(directory.name)
    for group in groups.values():
        samples = group["samples"]
        group["n"] = len(samples)
        group["http"] = timing_summary([sample["http_s"] for sample in samples])
        decode_n = sum(sample["decode"]["n"] for sample in samples)
        decode_sum = sum(sample["decode"]["sum_s"] for sample in samples)
        outside_sum = sum(sample["outside"]["sum_s"] for sample in samples)
        group["pooled_decode"] = {"n": decode_n, "sum_s": decode_sum,
                                   "mean_s": decode_sum / decode_n if decode_n else None,
                                   "outside_mean_s": outside_sum / decode_n if decode_n else None}
    result = {"schema_version": 1, "generated_clock": clock_sample(), "groups": groups,
              "limits": ["qlh-tui-backend is API-thread provenance, not exclusive TUI load",
                         "payload byte count assumes ASCII canonical serialization",
                         "step counts use wall/monotonic alignment with recorded boundary sensitivity",
                         "ACK CPU windows include fixed sampler overhead outside HTTP",
                         "fixed arm order and final legacy retain order bias"]}
    atomic_json(experiment / "capture-summary.json", result)
    print(json.dumps({"summary": str(experiment / "capture-summary.json"),
                      "groups": {key: {"mode": value["mode"], "n": value["n"],
                                       "http_mean_s": value["http"]["mean_s"]}
                                 for key, value in groups.items()}}, ensure_ascii=False), flush=True)
    return result


def summarize(options):
    summarize_directory(workspace(options))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("launch", "switch", "run", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--label", required=True)
        if name == "launch":
            command.add_argument("--mode", choices=MODES, default="legacy")
            command.add_argument("--port", type=int)
        elif name == "switch":
            command.add_argument("--mode", choices=MODES)
            command.add_argument("--flush-only", action="store_true")
            command.add_argument("--keep-cache", action="store_true")
            command.add_argument("--timeout", type=float, default=120)
        elif name == "run":
            arms = command.add_mutually_exclusive_group()
            arms.add_argument("--mode", choices=MODES)
            arms.add_argument("--modes", nargs="+", choices=MODES,
                              default=["legacy", "execution", "query", "combined", "legacy"])
            command.add_argument("--sample-label", required=True)
            command.add_argument("--warmup", type=int, default=1)
            command.add_argument("--count", type=int, default=3)
            command.add_argument("--keep-cache", action="store_true")
            command.add_argument("--reuse-current-arm", action="store_true")
            command.add_argument("--control-timeout", type=float, default=300)
            command.add_argument("--request-timeout", type=float, default=240)
            command.add_argument("--log-dir", default=str(Path(os.environ.get("LOCALAPPDATA", "")) /
                                                         "QLH-Edge-Inference/state/logs"))
        command.set_defaults(handler=globals()[name])
    return result


def main(argv=None):
    options = parser().parse_args(argv)
    try:
        options.handler(options)
        return 0
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
