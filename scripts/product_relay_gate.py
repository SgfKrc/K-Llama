"""Validate real Route-A product evidence; synthetic gate tests are not product runs.

Raw response, intent, completed-stage log, UI, OS connections, and native-load
observations must be kept together as SHA-256-bound files. A release requires DL
and LL, each with two distinct remote workers. No missing fact defaults to true.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

COMPLETED_PREFIX = "Route-A stage completed: "
SHA256 = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")


class EvidenceError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value: str) -> None:
    raise EvidenceError(f"invalid JSON constant: {value}")


def parse_json(raw: str) -> object:
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def require(condition: object, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def clean_source(value: dict, name: str) -> None:
    # Legacy evidence may omit this fact. An explicit value must be boolean false;
    # falsy JSON values such as 0, null or an empty container do not prove clean.
    require(value.get("source_dirty", False) is False,
            f"{name}.source_dirty: explicit dirty or invalid source cannot prove clean release evidence")


def integer(value: object, name: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{name}: expected integer >= {minimum}")
    return value


def timestamp(value: object, name: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"{name}: expected positive finite timestamp")
    return float(value)


def object_value(value: object, name: str) -> dict:
    require(isinstance(value, dict), f"{name}: expected object")
    return value


def text(value: object, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name}: missing text")
    return value


def digest(value: object, name: str) -> str:
    require(isinstance(value, str) and SHA256.fullmatch(value), f"{name}: invalid SHA-256")
    return value


def _file(base: Path, ref: object) -> Path:
    ref = object_value(ref, "file reference")
    name = text(ref.get("path"), "file path")
    path = Path(name)
    require(not path.is_absolute() and not path.drive and ".." not in path.parts,
            "evidence path must be relative and confined")
    resolved = (base / path).resolve()
    require(resolved.is_relative_to(base.resolve()), "evidence path escapes bundle")
    require(resolved.is_file() and resolved.stat().st_size > 0, f"missing or empty evidence: {name}")
    require(hashlib.sha256(resolved.read_bytes()).hexdigest() == digest(ref.get("sha256"), name),
            f"evidence SHA-256 mismatch: {name}")
    return resolved


def _json(base: Path, ref: object) -> dict:
    path = _file(base, ref)
    return object_value(parse_json(path.read_text(encoding="utf-8-sig")), path.name)


def _identity(value: dict) -> tuple:
    return (text(value.get("operation_id"), "operation_id"),
            text(value.get("config_id"), "config_id"),
            integer(value.get("generation"), "generation", 1))


def _range(value: object, total: int) -> tuple[int, int]:
    require(isinstance(value, list) and len(value) == 2, "layer_range must contain two integers")
    start, end = (integer(value[0], "start_layer"), integer(value[1], "end_layer", 1))
    require(start < end <= total and end - start < total, "invalid or full-model layer range")
    return start, end


def _observation(value: dict, identity: tuple, started: float, finished: float) -> None:
    require(_identity(value) == identity, "observation belongs to another model selection")
    captured = timestamp(value.get("captured_at"), "captured_at")
    require(started <= captured <= finished, "observation is outside capture window")


def _validate_run(base: Path, run: dict, revision: str, started: float, finished: float) -> dict:
    mode = run.get("mode")
    require(mode in ("DL", "LL"), "mode must be DL or LL")
    clean_source(run, "run")
    intent = _json(base, run.get("intent"))
    identity = _identity(intent)
    require(intent.get("status") == "ready" and intent.get("loaded_ready") is True,
            "model intent was not natively ready")
    plan = object_value(intent.get("plan"), "intent plan")
    require(plan.get("admitted") is True, "capacity plan is not admitted")
    total = integer(plan.get("total_layers"), "total_layers", 3)
    assignments = plan.get("assignments")
    require(isinstance(assignments, list) and len(assignments) >= 3,
            "product gate requires master and two different remote nodes")
    nodes: dict[str, dict] = {}
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for index, assignment in enumerate(assignments):
        assignment = object_value(assignment, "assignment")
        node = text(assignment.get("node_id"), "node_id")
        require(node not in nodes, "duplicate node assignment")
        span = _range([assignment.get("start_layer"), assignment.get("end_layer")], total)
        require(span[0] == cursor, "segments are not continuous and ordered")
        cursor = span[1]
        require((node == "master") == (index == 0), "master must own only the first segment")
        expected_engine = "pytorch" if mode == "DL" and index == 0 else "llama_cpp"
        require(assignment.get("engine") == expected_engine, f"{node}: engine does not match {mode}")
        nodes[node], ranges[node] = assignment, span
    require(cursor == total, "segments do not cover the full decoder")
    remote = list(nodes)[1:]
    require(len(set(remote)) >= 2, "two distinct remote nodes are required")

    response = _json(base, run.get("response"))
    require(response.get("done") is True and not response.get("error") and not response.get("cancelled"),
            "request did not complete successfully")
    answer = text(response.get("response"), "response text")
    metrics = object_value(response.get("metrics"), "response metrics")
    require(metrics.get("distributed_used") is True and metrics.get("fallback") is False,
            "distributed_used=true and explicit fallback=false are required")
    require(metrics.get("fallback_reason") == "", "fallback reason must be explicitly empty")
    require(metrics.get("routing_preference") == "distributed_required", "required route is missing")
    require(metrics.get("execution_mode") == "route_a_stage_offer_v3", "not the Route-A product path")
    require((metrics.get("model_operation_id"), metrics.get("model_config_id"),
             integer(metrics.get("model_generation"), "model_generation", 1)) == identity,
            "response model selection differs from intent")
    request_id = text(metrics.get("request_id"), "request_id")
    require(request_id != "-", "missing HTTP request identity")
    task_id = text(metrics.get("task_id"), "task_id")
    require(metrics.get("config_id") == f"route_a:{task_id}", "task config does not match task ID")
    require(metrics.get("workers_used") == remote, "actual remote workers differ from plan")
    actual = metrics.get("layer_assignments")
    require(isinstance(actual, list) and len(actual) == len(remote), "missing actual stage assignments")
    stage_identities = {}
    for node, stage in zip(remote, actual):
        stage = object_value(stage, "actual assignment")
        require(stage.get("node_id") == node, "actual stages are out of order")
        require(_range([stage.get("start_layer"), stage.get("end_layer")], total) == ranges[node],
                "actual stage range differs from plan")
        prepared = object_value(stage.get("worker_prepare_identity"), "worker prepare identity")
        require(_identity(prepared) == identity, "stage belongs to another operation/config/generation")
        require(prepared.get("node_id") == node, "prepare identity names another node")
        require(_range(prepared.get("layer_range"), total) == ranges[node], "prepare range differs")
        stage_identities[node] = (text(prepared.get("assignment_id"), "assignment_id"),
                                  digest(prepared.get("artifact_sha256"), "artifact_sha256"))
    steps = metrics.get("steps")
    require(isinstance(steps, list) and bool(steps), "missing generation steps")
    require([integer(object_value(s, "step").get("step"), "step") for s in steps] == list(range(len(steps))),
            "generation steps are not contiguous")
    require(metrics.get("finish_reason") == "stop", "generation did not stop normally")

    events = []
    for line in _file(base, run.get("handoffs")).read_text(encoding="utf-8-sig").splitlines():
        if COMPLETED_PREFIX in line:
            event = object_value(parse_json(line.split(COMPLETED_PREFIX, 1)[1]), "completed handoff")
            if event.get("workflow_id") == f"wf_{task_id}":
                events.append(event)
    require(bool(events) and len(events) % len(remote) == 0, "incomplete successful handoff rounds")
    rounds = len(events) // len(remote)
    # EOS completes its final remote round before leaving the token loop,
    # so that round is absent from visible steps but still requires receipts.
    stop_reason = metrics.get("stop_reason")
    require(stop_reason in ("eos_token", "text_stop"), "missing or invalid normal stop reason")
    require(rounds == len(steps) + (1 if stop_reason == "eos_token" else 0),
            "handoff rounds do not match actual token steps and stop reason")
    last_completed = started
    for index, event in enumerate(events):
        node, step = remote[index % len(remote)], index // len(remote)
        require(event.get("node_id") == node, "completed handoffs are missing, duplicated or reordered")
        require(event.get("stage_request_id") == f"{task_id}:step:{step}", "handoff belongs to another step")
        require(event.get("stage_id") == f"{node}:step:{step}", "handoff stage identity differs")
        require(_identity(event) == identity, "handoff belongs to another model selection")
        require(_range(event.get("layer_range"), total) == ranges[node], "handoff range differs")
        require((event.get("assignment_id"), event.get("artifact_sha256")) == stage_identities[node],
                "handoff artifact or assignment differs from prepared stage")
        require(event.get("result_kind") == ("token" if node == remote[-1] else "hidden"),
                "handoff has wrong result kind")
        digest(event.get("hidden_sha256"), "handoff hidden SHA-256")
        integer(event.get("tokens"), "handoff tokens", 1)
        captured = timestamp(event.get("completed_at"), "handoff completed_at")
        require(last_completed <= captured <= finished, "handoff is outside window or completion order")
        last_completed = captured

    topology = _json(base, run.get("topology"))
    _observation(topology, identity, started, finished)
    require(topology.get("source") == "os_process_connections", "topology requires OS connection observations")
    require(topology.get("complete") is True, "incomplete socket capture cannot prove star topology")
    observers = topology.get("nodes")
    require(isinstance(observers, list) and len(observers) == len(nodes), "missing per-node socket observations")
    observed_nodes = set()
    for observation in observers:
        observation = object_value(observation, "node sockets")
        node = observation.get("node_id")
        require(node in nodes and node not in observed_nodes, "duplicate/unknown socket observer")
        observed_nodes.add(node)
        require(observation.get("complete") is True, "node socket capture was partial/denied")
        peers = observation.get("worker_peers")
        require(isinstance(peers, list), "worker peer connections were not captured")
        require(len(peers) == len(set(peers)), "duplicate worker peer")
        require(set(peers) == (set(remote) if node == "master" else {"master"}),
                "missing master edge or worker-to-worker connection present")
        _file(base, observation.get("raw_evidence"))

    native = _json(base, run.get("native"))
    _observation(native, identity, started, finished)
    loaded = native.get("nodes")
    require(isinstance(loaded, list) and len(loaded) == len(nodes), "missing native load observations")
    observed_nodes = set()
    for observation in loaded:
        observation = object_value(observation, "native load")
        node = observation.get("node_id")
        require(node in nodes and node not in observed_nodes, "duplicate/unknown native observer")
        observed_nodes.add(node)
        require(observation.get("source_revision") == revision, "native node executes another source revision")
        clean_source(observation, f"native node {node}")
        require(observation.get("source") in ("os_process_mappings", "native_loader_receipt"),
                "native load requires process mappings or native loader receipts")
        require(observation.get("full_model_loaded") is False, "full model loaded or fact missing")
        require(observation.get("engine") == nodes[node].get("engine"), "native engine differs from plan")
        require(_range(observation.get("layer_range"), total) == ranges[node], "native range differs from plan")
        require(observation.get("complete") is True, "native observation is incomplete")
        # Retain the actual process/loader output, not just a self-asserted boolean.
        _file(base, observation.get("raw_evidence"))

    ui = _json(base, run.get("ui"))
    require(ui.get("entry") in ("textual", "web", "android") and ui.get("done") is True,
            "missing completed user-interface run")
    require(ui.get("request_id") == request_id, "UI and response belong to different requests")
    expected = text(ui.get("expected_response"), "UI expected response").strip()
    text(ui.get("submitted_message"), "UI submitted message")
    require(text(ui.get("rendered_response"), "UI rendered response").strip() == answer.strip() == expected,
            "rendered answer does not match the expected product answer")
    require(started <= timestamp(ui.get("captured_at"), "UI captured_at") <= finished,
            "UI capture is outside capture window")
    _file(base, ui.get("screen"))
    return {"mode": mode, "passed": True, "request_id": request_id,
            "operation_id": identity[0], "config_id": identity[1], "generation": identity[2],
            "nodes": list(nodes), "handoff_rounds": rounds}


def validate_bundle(bundle_path: Path, expected_revision: str,
                    required_modes: tuple[str, ...] = ("DL", "LL")) -> dict:
    result = {"passed": False, "errors": [], "runs": []}
    try:
        require(bool(REVISION.fullmatch(expected_revision)), "expected source revision must be a full Git SHA")
        require(bool(required_modes) and set(required_modes) <= {"DL", "LL"}
                and len(set(required_modes)) == len(required_modes), "invalid required modes")
        bundle = object_value(parse_json(bundle_path.read_text(encoding="utf-8-sig")), "bundle")
        require(type(bundle.get("schema_version")) is int and bundle["schema_version"] == 1,
                "unsupported product evidence schema")
        require(bundle.get("source_revision") == expected_revision, "bundle source revision mismatch")
        clean_source(bundle, "bundle")
        require(bundle.get("physical_opt_in") is True, "physical run was not explicitly opted in")
        started = timestamp(bundle.get("capture_started_at"), "capture_started_at")
        finished = timestamp(bundle.get("capture_finished_at"), "capture_finished_at")
        require(started <= finished, "capture window is reversed")
        runs = bundle.get("runs")
        require(isinstance(runs, list) and len(runs) == len(required_modes), "missing/extra product modes")
        require(all(isinstance(run, dict) for run in runs), "invalid product run")
        require({run.get("mode") for run in runs} == set(required_modes), "missing or duplicate product modes")
        windows = {}
        if "input_windows" in bundle:
            inputs = bundle["input_windows"]
            require(isinstance(inputs, list) and len(inputs) == len(runs), "invalid input capture windows")
            for window in inputs:
                window = object_value(window, "input capture window")
                mode = window.get("mode")
                require(isinstance(mode, str) and mode in required_modes and mode not in windows,
                        "duplicate or unknown input capture window")
                window_start = timestamp(window.get("capture_started_at"), "input capture_started_at")
                window_end = timestamp(window.get("capture_finished_at"), "input capture_finished_at")
                require(started <= window_start <= window_end <= finished, "input capture window escapes bundle")
                windows[mode] = (window_start, window_end)
        for run in runs:
            try:
                run_started, run_finished = windows.get(run["mode"], (started, finished))
                result["runs"].append(_validate_run(bundle_path.parent, run, expected_revision, run_started, run_finished))
            except (EvidenceError, OSError, ValueError, TypeError, KeyError) as exc:
                result["errors"].append(f"{run.get('mode')}: {exc}")
        result["passed"] = not result["errors"]
    except (EvidenceError, OSError, ValueError, TypeError, KeyError) as exc:
        result["errors"].append(str(exc))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--require-revision", required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = validate_bundle(args.bundle, args.require_revision)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
