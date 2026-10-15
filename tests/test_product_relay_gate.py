"""Gate logic regressions use synthetic fixtures; these are not real product evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("product_gate_test", ROOT / "scripts/product_relay_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
REVISION = "a" * 40


def write_file(base, name, data):
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else json.dumps(data).encode())
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(base, ref):
    return json.loads((base / ref["path"]).read_text())


def update_json(base, ref, transform):
    obj = read_json(base, ref)
    transform(obj)
    ref.update(write_file(base, ref["path"], obj))


@pytest.fixture
def bundle(tmp_path):
    runs = []
    for mode in ("DL", "LL"):
        identity = {"operation_id": f"supply_{mode}", "config_id": f"model_config_{mode}",
                    "generation": 1791920668886047001}
        assignments = [
            {"node_id": node, "start_layer": start, "end_layer": end,
             "engine": "pytorch" if mode == "DL" and node == "master" else "llama_cpp"}
            for node, start, end in (("master", 0, 4), ("pc", 4, 8), ("android", 8, 32))
        ]
        actual = []
        for assignment in assignments[1:]:
            node = assignment["node_id"]
            actual.append({**assignment, "worker_prepare_identity": {
                **identity, "node_id": node, "assignment_id": f"asg_{mode}_{node}",
                "artifact_sha256": "b" * 64,
                "layer_range": [assignment["start_layer"], assignment["end_layer"]]}})
        intent = {**identity, "status": "ready", "loaded_ready": True,
                  "plan": {"admitted": True, "total_layers": 32, "assignments": assignments}}
        response = {"done": True, "response": "2", "metrics": {
            "distributed_used": True, "fallback": False, "fallback_reason": "",
            "routing_preference": "distributed_required", "execution_mode": "route_a_stage_offer_v3",
            "model_operation_id": identity["operation_id"], "model_config_id": identity["config_id"],
            "model_generation": identity["generation"], "request_id": f"http_{mode}", "task_id": mode,
            "config_id": f"route_a:{mode}", "workers_used": ["pc", "android"],
            "layer_assignments": actual, "steps": [{"step": 0}], "finish_reason": "stop", "stop_reason": "text_stop"}}
        events = [{**identity, "workflow_id": f"wf_{mode}", "stage_request_id": f"{mode}:step:0",
                   "stage_id": f"{stage['node_id']}:step:0", "node_id": stage["node_id"],
                   "layer_range": stage["worker_prepare_identity"]["layer_range"],
                   "assignment_id": stage["worker_prepare_identity"]["assignment_id"],
                   "artifact_sha256": "b" * 64, "hidden_sha256": "c" * 64,
                   "tokens": 10, "result_kind": "token" if index else "hidden", "completed_at": 105.0}
                  for index, stage in enumerate(actual)]
        raw = write_file(tmp_path, f"{mode}/raw-observation.txt", "synthetic observer fixture")
        topology = {**identity, "captured_at": 106.0, "source": "os_process_connections", "complete": True,
                    "nodes": [{"node_id": a["node_id"], "complete": True, "raw_evidence": raw,
                               "worker_peers": ["pc", "android"] if a["node_id"] == "master" else ["master"]}
                              for a in assignments]}
        native = {**identity, "captured_at": 106.0, "nodes": [
            {"node_id": a["node_id"], "complete": True, "source_revision": REVISION,
             "source": "native_loader_receipt", "full_model_loaded": False, "engine": a["engine"],
             "layer_range": [a["start_layer"], a["end_layer"]], "raw_evidence": raw} for a in assignments]}
        screen = write_file(tmp_path, f"{mode}/screen.svg", "<svg>synthetic UI fixture</svg>")
        ui = {"entry": "textual", "done": True, "request_id": f"http_{mode}", "submitted_message": "1+1?",
              "expected_response": "2", "rendered_response": "2", "captured_at": 107.0, "screen": screen}
        run = {"mode": mode}
        for name, obj in (("intent", intent), ("response", response), ("topology", topology),
                          ("native", native), ("ui", ui)):
            run[name] = write_file(tmp_path, f"{mode}/{name}.json", obj)
        run["handoffs"] = write_file(tmp_path, f"{mode}/handoffs.log",
            "\n".join(gate.COMPLETED_PREFIX + json.dumps(e) for e in events))
        runs.append(run)
    manifest = {"schema_version": 1, "source_revision": REVISION, "physical_opt_in": True,
                "capture_started_at": 100.0, "capture_finished_at": 110.0, "runs": runs}
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def validate(bundle):
    path, manifest = bundle
    path.write_text(json.dumps(manifest))
    return gate.validate_bundle(path, REVISION)


def test_complete_fixture_is_accepted_by_validator_only(bundle):
    result = validate(bundle)
    assert result["passed"], result
    assert [r["mode"] for r in result["runs"]] == ["DL", "LL"]
    assert result["runs"][0]["generation"] == 1791920668886047001


@pytest.mark.parametrize("change", [
    lambda b: b.update(runs=[]),
    lambda b: b["runs"].pop(),
    lambda b: b["runs"].append(copy.deepcopy(b["runs"][0])),
    lambda b: b["runs"][1].update(mode="DL"),
    lambda b: b.update(physical_opt_in=False),
    lambda b: b.update(source_revision="f" * 40),
    lambda b: b.update(capture_finished_at=99),
    lambda b: b.update(capture_started_at=float("nan")),
    lambda b: b.update(schema_version=True),
])
def test_missing_modes_optin_revision_and_invalid_schema_fail(bundle, change):
    change(bundle[1])
    assert not validate(bundle)["passed"]


@pytest.mark.parametrize("component,change", [
    ("intent", lambda o: o["plan"]["assignments"][1].update(start_layer=5)),
    ("intent", lambda o: o["plan"]["assignments"].pop()),
    ("intent", lambda o: o["plan"]["assignments"][1].update(node_id="master")),
    ("intent", lambda o: o["plan"]["assignments"][0].update(engine="llama_cpp")),
    ("intent", lambda o: o.update(loaded_ready=False)),
    ("response", lambda o: o["metrics"].pop("fallback")),
    ("response", lambda o: o["metrics"].update(fallback="false")),
    ("response", lambda o: o["metrics"].update(fallback=True)),
    ("response", lambda o: o["metrics"].update(distributed_used=False)),
    ("response", lambda o: o["metrics"].update(model_generation=float(o["metrics"]["model_generation"]))),
    ("response", lambda o: o["metrics"].update(workers_used=["pc", "pc"])),
    ("response", lambda o: o["metrics"]["layer_assignments"][1]["worker_prepare_identity"].update(generation=7)),
    ("response", lambda o: o["metrics"].update(steps=[])),
    ("response", lambda o: o["metrics"]["steps"][0].update(step=1)),
    ("response", lambda o: o["metrics"].update(finish_reason="length")),
    ("response", lambda o: o.update(done=False)),
    ("topology", lambda o: o.update(complete=False)),
    ("topology", lambda o: o["nodes"][2].update(worker_peers=["master", "pc"])),
    ("topology", lambda o: o["nodes"][2].update(complete=False)),
    ("topology", lambda o: o.update(captured_at=99)),
    ("topology", lambda o: o["nodes"].pop()),
    ("topology", lambda o: o["nodes"][0].pop("raw_evidence")),
    ("native", lambda o: o["nodes"][2].update(full_model_loaded=True)),
    ("native", lambda o: o["nodes"][0].update(layer_range=[0, 32])),
    ("native", lambda o: o["nodes"][0].update(engine="llama_cpp")),
    ("native", lambda o: o["nodes"][2].update(source_revision="f" * 40)),
    ("native", lambda o: o["nodes"][2].pop("raw_evidence")),
    ("native", lambda o: o.update(generation=7)),
    ("ui", lambda o: o.update(request_id="old-request")),
    ("ui", lambda o: o.update(rendered_response="2user\nWhat is 3+3?")),
    ("ui", lambda o: o.update(rendered_response=[])),
    ("ui", lambda o: o.pop("screen")),
])
def test_semantic_tampering_fails_even_with_updated_file_hash(bundle, component, change):
    update_json(bundle[0].parent, bundle[1]["runs"][0][component], change)
    assert not validate(bundle)["passed"]


@pytest.mark.parametrize("alter", [
    lambda e: e.pop(),
    lambda e: e.reverse(),
    lambda e: e[0].update(generation=7),
    lambda e: e[0].update(stage_request_id="another:step:0"),
    lambda e: e[0].update(result_kind="token"),
    lambda e: e[0].update(completed_at=99),
    lambda e: e[0].update(layer_range=[4, 9]),
    lambda e: e[0].update(artifact_sha256="f" * 64),
    lambda e: e[0].update(workflow_id="wf_another"),
])
def test_handoff_must_complete_every_real_stage_in_order(bundle, alter):
    base, ref = bundle[0].parent, bundle[1]["runs"][0]["handoffs"]
    events = [json.loads(line.split(gate.COMPLETED_PREFIX)[1]) for line in (base/ref["path"]).read_text().splitlines()]
    alter(events)
    ref.update(write_file(base, ref["path"], "\n".join(gate.COMPLETED_PREFIX+json.dumps(e) for e in events)))
    assert not validate(bundle)["passed"]


def test_dispatch_logs_cannot_prove_completed_handoffs(bundle):
    base, ref = bundle[0].parent, bundle[1]["runs"][0]["handoffs"]
    ref.update(write_file(base, ref["path"], "Route-A stage handoff: node=pc\n"))
    assert not validate(bundle)["passed"]


def test_missing_tampered_and_escaping_evidence_fail(bundle):
    ref = bundle[1]["runs"][0]["response"]
    ref["sha256"] = "f" * 64
    assert not validate(bundle)["passed"]
    ref["path"] = "../old-success.json"
    assert not validate(bundle)["passed"]
    ref["path"] = "missing.json"
    assert not validate(bundle)["passed"]


def test_invalid_json_and_missing_bundle_fail(tmp_path):
    path = tmp_path / "bundle.json"
    assert not gate.validate_bundle(path, REVISION)["passed"]
    path.write_text("not json")
    assert not gate.validate_bundle(path, REVISION)["passed"]


def test_cli_missing_evidence_returns_failure_and_writes_report(tmp_path):
    output = tmp_path / "report.json"
    assert gate.main(["--bundle", str(tmp_path/"missing.json"), "--require-revision", REVISION,
                      "--json", str(output)]) == 1
    assert json.loads(output.read_text())["passed"] is False


def test_duplicate_keys_and_nonstandard_json_constants_fail(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    result = gate.validate_bundle(path, REVISION)
    assert not result["passed"] and "duplicate JSON key" in result["errors"][0]
    path.write_text('{"capture_started_at": NaN}')
    result = gate.validate_bundle(path, REVISION)
    assert not result["passed"] and "invalid JSON constant" in result["errors"][0]


def test_combining_runs_does_not_expand_individual_capture_windows(bundle):
    bundle[1]["input_windows"] = [
        {"mode": "DL", "capture_started_at": 100, "capture_finished_at": 104},
        {"mode": "LL", "capture_started_at": 100, "capture_finished_at": 110}]
    result = validate(bundle)
    assert not result["passed"]
    assert any("DL:" in error for error in result["errors"])


def test_eos_requires_its_own_complete_final_handoff_round(bundle):
    base, run = bundle[0].parent, bundle[1]["runs"][0]
    update_json(base, run["response"], lambda obj: obj["metrics"].update(stop_reason="eos_token"))
    assert not validate(bundle)["passed"]
    ref = run["handoffs"]
    lines = (base/ref["path"]).read_text().splitlines()
    events = [json.loads(line.split(gate.COMPLETED_PREFIX)[1]) for line in lines]
    for event in events:
        event["stage_request_id"] = "DL:step:1"
        event["stage_id"] = f"{event['node_id']}:step:1"
        event["completed_at"] = 106.0
    ref.update(write_file(base, ref["path"], "\n".join(lines + [gate.COMPLETED_PREFIX+json.dumps(e) for e in events])))
    result = validate(bundle)
    assert result["passed"], result
