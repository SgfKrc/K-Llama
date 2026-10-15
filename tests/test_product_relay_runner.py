"""Offline runner regressions. Every successful bundle here is SYNTHETIC.

No test boots a backend, opens a socket, invokes the physical capture process,
or injects a model response into the real TUI/stream capture path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("product_relay_runner_tested", ROOT / "scripts/run_product_relay_gate.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
REVISION = "a" * 40


def put(base, name, value):
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False).encode("utf-8")
    elif isinstance(value, str):
        value = value.encode("utf-8")
    path.write_bytes(value)
    return {"path": name, "sha256": hashlib.sha256(value).hexdigest()}


def synthetic_bundle(base, mode, raw_name=None):
    """Complete gate-shaped fixture; explicitly not physical evidence."""
    selection = {"operation_id": f"synthetic_{mode}", "config_id": f"config_{mode}", "generation": 7}
    assignments = [{"node_id": node, "start_layer": start, "end_layer": end,
                    "engine": "pytorch" if mode == "DL" and node == "master" else "llama_cpp"}
                   for node, start, end in [("master", 0, 2), ("remote_pc", 2, 4), ("remote_android", 4, 6)]]
    remotes = [item["node_id"] for item in assignments[1:]]
    stages = [{**item, "worker_prepare_identity": {
        **selection, "node_id": item["node_id"], "assignment_id": "asg_" + item["node_id"],
        "artifact_sha256": "b" * 64, "layer_range": [item["start_layer"], item["end_layer"]]}}
        for item in assignments[1:]]
    intent = {**selection, "status": "ready", "loaded_ready": True,
              "plan": {"admitted": True, "total_layers": 6, "assignments": assignments}}
    response = {"done": True, "response": "2", "metrics": {
        "distributed_used": True, "fallback": False, "fallback_reason": "",
        "routing_preference": "distributed_required", "execution_mode": "route_a_stage_offer_v3",
        "model_operation_id": selection["operation_id"], "model_config_id": selection["config_id"],
        "model_generation": 7, "request_id": f"http_{mode}", "task_id": mode,
        "config_id": f"route_a:{mode}", "workers_used": remotes, "layer_assignments": stages,
        "steps": [{"step": 0}], "finish_reason": "stop", "stop_reason": "text_stop"}}
    logs = []
    for index, stage in enumerate(stages):
        prepared = stage["worker_prepare_identity"]
        event = {**selection, "workflow_id": f"wf_{mode}", "stage_request_id": f"{mode}:step:0",
                 "stage_id": f"{stage['node_id']}:step:0", "node_id": stage["node_id"],
                 "layer_range": prepared["layer_range"], "assignment_id": prepared["assignment_id"],
                 "artifact_sha256": "b" * 64, "hidden_sha256": "c" * 64, "tokens": 1,
                 "result_kind": "hidden" if index == 0 else "token", "completed_at": 104}
        logs.append(runner.COMPLETED_PREFIX + json.dumps(event))
    topology_nodes, native_nodes = [], []
    for assignment in assignments:
        node = assignment["node_id"]
        raw = put(base, raw_name or f"{mode}/raw/{node}.txt", f"synthetic {mode} " + ("shared" if raw_name else node) + " observation")
        topology_nodes.append({"node_id": node, "complete": True, "raw_evidence": raw,
                               "worker_peers": remotes if node == "master" else ["master"]})
        native_nodes.append({"node_id": node, "complete": True, "raw_evidence": raw,
                             "source_revision": REVISION, "source": "os_process_mappings",
                             "full_model_loaded": False, "engine": assignment["engine"],
                             "layer_range": [assignment["start_layer"], assignment["end_layer"]]})
    topology = {**selection, "captured_at": 105, "source": "os_process_connections",
                "complete": True, "nodes": topology_nodes}
    native = {**selection, "captured_at": 105, "nodes": native_nodes}
    screen = put(base, f"{mode}/screen.svg", "<svg>synthetic screen fixture</svg>")
    ui = {"entry": "textual", "done": True, "request_id": f"http_{mode}", "submitted_message": runner.MESSAGE,
          "expected_response": "2", "rendered_response": "2", "status_text": "synthetic status",
          "captured_at": 106, "screen": screen}
    run = {"mode": mode}
    for name, value in [("intent", intent), ("response", response), ("topology", topology), ("native", native), ("ui", ui)]:
        run[name] = put(base, f"{mode}/{name}.json", value)
    run["handoffs"] = put(base, f"{mode}/handoffs.log", "\n".join(logs))
    run["stream"] = put(base, f"{mode}/stream.jsonl", json.dumps(response) + "\n")
    manifest = {"schema_version": 1, "source_revision": REVISION, "physical_opt_in": True,
                "capture_started_at": 100, "capture_finished_at": 110, "runs": [run]}
    path = base / "bundle.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    return [synthetic_bundle(tmp_path / mode, mode) for mode in ("DL", "LL")]


def capture_argv(output, *extra):
    return ["capture", "--mode", "DL", "--master-log", str(output.parent / "master.log"),
            "--topology", str(output.parent / "topology.json"),
            "--native", str(output.parent / "native.json"), "--output", str(output), *extra]


def combine_argv(inputs, output):
    return ["combine", "--inputs", *map(str, inputs), "--output", str(output)]


def failed_report(output):
    result = runner.read_json(runner.result_path(output))
    assert result["passed"] is False and result["errors"]
    return " ".join(result["errors"])


@pytest.mark.parametrize("opt_in", [None, "", "0", "true"])
def test_capture_requires_exact_opt_in_before_starting_any_process(tmp_path, monkeypatch, opt_in):
    if opt_in is None:
        monkeypatch.delenv("QLH_RUN_PHYSICAL_TESTS", raising=False)
    else:
        monkeypatch.setenv("QLH_RUN_PHYSICAL_TESTS", opt_in)
    def unexpected(_):
        pytest.fail("physical process must not start")
    monkeypatch.setattr(runner, "physical_process", unexpected)
    output = tmp_path / "capture.json"
    assert runner.main(capture_argv(output)) == 1
    assert "QLH_RUN_PHYSICAL_TESTS=1" in failed_report(output)
    assert not output.exists()


class ReadOnlyAPI:
    def __init__(self, nodes, enabled=True, intent=None):
        self.nodes, self.enabled, self.intent = nodes, enabled, intent
        self.calls = []

    def get(self, path):
        self.calls.append(path)
        return {"/cluster/config/distributed-inference": {"enabled": self.enabled},
                "/status": {"node_role": "master"}, "/cluster/nodes": {"nodes": self.nodes},
                "/models/intent/current": self.intent}[path]

    def post(self, *args, **kwargs):
        pytest.fail("no inference or other writes may be sent during preflight")


@pytest.mark.parametrize("nodes", [[], [{"node_id": "rog", "is_available": True}],
    [{"node_id": "rog", "is_available": True}, {"node_id": "rog", "is_available": True}],
    [{"node_id": "rog", "is_available": True}, {"node_id": "other", "is_available": False}]])
def test_missing_devices_fail_before_intent_or_inference(nodes):
    api = ReadOnlyAPI(nodes)
    with pytest.raises(runner.CaptureError, match="two distinct"):
        runner.preflight(api, "DL")
    assert "/models/intent/current" not in api.calls


def test_disabled_distributed_does_not_change_config_or_send_inference():
    api = ReadOnlyAPI([], enabled=False)
    with pytest.raises(runner.CaptureError, match="already be enabled"):
        runner.preflight(api, "DL")
    assert api.calls == ["/cluster/config/distributed-inference"]


@pytest.mark.parametrize("failure", [ImportError("textual dependency missing"), FileNotFoundError("master.log"),
    TimeoutError("capture total timeout"), RuntimeError("unexpected validator exception")])
def test_capture_exceptions_are_nonzero_and_recorded(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("QLH_RUN_PHYSICAL_TESTS", "1")
    def fail(_):
        raise failure
    monkeypatch.setattr(runner, "physical_process", fail)
    output = tmp_path / "failed.json"
    assert runner.main(capture_argv(output)) == 1
    assert type(failure).__name__ in failed_report(output)


def test_worker_records_missing_dependencies_without_spawning(tmp_path, monkeypatch):
    def fail(_):
        raise ImportError("Textual unavailable")
    monkeypatch.setattr(runner, "capture", fail)
    output = tmp_path / "worker.json"
    runner.worker(SimpleNamespace(output=output))
    assert "Textual unavailable" in failed_report(output)


def test_dirty_checkout_is_rejected_instead_of_claiming_head(monkeypatch):
    monkeypatch.setattr(runner, "git", lambda *args: REVISION if args[0] == "rev-parse" else " M src/tui_textual.py")
    with pytest.raises(runner.CaptureError, match="dirty checkout"):
        runner.source_revision(None)
    with pytest.raises(runner.CaptureError, match="must equal"):
        runner.source_revision("f" * 40)


def test_future_bench_window_fails_before_source_or_device_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("QLH_RUN_PHYSICAL_TESTS", "1")
    monkeypatch.setattr(runner.time, "time", lambda: 100)
    def unexpected(_):
        pytest.fail("future window must fail before checking source or device")
    monkeypatch.setattr(runner, "source_revision", unexpected)
    args = runner.parser().parse_args(capture_argv(tmp_path / "future.json", "--capture-started-at", "101"))
    with pytest.raises(runner.CaptureError, match="nonfuture"):
        runner.capture(args)


def test_combine_copies_every_raw_reference_and_remains_self_contained(tmp_path, inputs, monkeypatch):
    monkeypatch.delenv("QLH_RUN_PHYSICAL_TESTS", raising=False)  # Offline combination needs no device opt-in.
    output = tmp_path / "combined" / "bundle.json"
    assert runner.main(combine_argv(inputs, output)) == 0
    merged = runner.read_json(output)
    assert [run["mode"] for run in merged["runs"]] == ["DL", "LL"]
    for path in inputs:
        original = runner.read_json(path)["runs"][0]
        run = next(run for run in merged["runs"] if run["mode"] == original["mode"])
        assert run == original
        for ref in runner.nested_refs(run):
            assert (path.parent / ref["path"]).read_bytes() == (output.parent / ref["path"]).read_bytes()
        for key in ("topology", "native"):
            observation = runner.read_json(output.parent / run[key]["path"])
            assert observation["captured_at"] == 105
            assert all((output.parent / node["raw_evidence"]["path"]).is_file() for node in observation["nodes"])
        ui = runner.read_json(output.parent / run["ui"]["path"])
        assert (output.parent / ui["screen"]["path"]).is_file()
        path.parent.rename(tmp_path / (path.parent.name + "-source-moved"))
    assert runner.load_validator()(output, REVISION)["passed"] is True
    assert len(merged["input_windows"]) == 2


@pytest.mark.parametrize("index,key", [(0, "response"), (0, "topology"), (1, "native"), (1, "ui")])
def test_combine_missing_evidence_is_nonzero(tmp_path, inputs, index, key):
    bundle = runner.read_json(inputs[index])
    (inputs[index].parent / bundle["runs"][0][key]["path"]).unlink()
    output = tmp_path / "missing" / "bundle.json"
    assert runner.main(combine_argv(inputs, output)) == 1
    error = failed_report(output)
    assert "missing" in error and "evidence" in error
    assert not output.exists()


@pytest.mark.parametrize("component,subkey", [("ui", "screen"), ("native", "raw_evidence"), ("topology", "raw_evidence")])
def test_combine_rejects_missing_nested_raw_or_screen(tmp_path, inputs, component, subkey):
    bundle = runner.read_json(inputs[0])
    value = runner.read_json(inputs[0].parent / bundle["runs"][0][component]["path"])
    ref = value[subkey] if component == "ui" else value["nodes"][0][subkey]
    (inputs[0].parent / ref["path"]).unlink()
    output = tmp_path / "nested-missing" / "bundle.json"
    assert runner.main(combine_argv(inputs, output)) == 1
    error = failed_report(output)
    assert "missing" in error and "evidence" in error


def test_combine_rejects_duplicate_modes(tmp_path, inputs):
    output = tmp_path / "duplicate.json"
    assert runner.main(combine_argv([inputs[0], inputs[0]], output)) == 1
    assert "duplicate mode" in failed_report(output)
    assert not output.exists()


def test_combine_rejects_missing_mode_and_conflicting_revisions(tmp_path, inputs):
    output = tmp_path / "one-mode.json"
    assert runner.main(combine_argv([inputs[0]], output)) == 1
    assert "both DL and LL" in failed_report(output)
    manifest = runner.read_json(inputs[1])
    manifest["source_revision"] = "f" * 40
    inputs[1].write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "revision.json"
    assert runner.main(combine_argv(inputs, output)) == 1
    assert "conflicting source revisions" in failed_report(output)


def test_combine_refuses_same_path_with_different_raw_bytes(tmp_path):
    inputs = [synthetic_bundle(tmp_path / mode, mode, "shared/raw.txt") for mode in ("DL", "LL")]
    output = tmp_path / "conflict" / "bundle.json"
    assert runner.main(combine_argv(inputs, output)) == 1
    assert "conflicting attachment path" in failed_report(output)
    assert not output.exists()


def test_combine_does_not_overwrite_existing_destination_evidence(tmp_path, inputs):
    output = tmp_path / "existing" / "bundle.json"
    bad = output.parent / "DL" / "raw" / "master.txt"
    bad.parent.mkdir(parents=True)
    bad.write_text("keep external evidence", encoding="utf-8")
    assert runner.main(combine_argv(inputs, output)) == 1
    assert "conflicting evidence destination" in failed_report(output)
    assert bad.read_text(encoding="utf-8") == "keep external evidence"


def test_combine_does_not_expand_single_run_window_to_accept_stale_observations(tmp_path, inputs):
    manifest = runner.read_json(inputs[0])
    manifest["capture_started_at"] = 105.5
    inputs[0].write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "stale.json"
    assert runner.main(combine_argv(inputs, output)) == 1
    error = failed_report(output)
    assert "outside" in error and "window" in error


def test_combine_validator_exception_and_absence_are_recorded(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("unexpected import failure")
    monkeypatch.setattr(runner, "load_validator", explode)
    output = tmp_path / "exception.json"
    assert runner.main(combine_argv([tmp_path / "missing.json"], output)) == 1
    assert "RuntimeError" in failed_report(output)
    monkeypatch.undo()
    monkeypatch.setattr(runner, "ROOT", tmp_path / "no-validator")
    output = tmp_path / "no-validator.json"
    assert runner.main(combine_argv([tmp_path / "missing.json"], output)) == 1
    assert "missing product validator" in failed_report(output)


def test_previous_failure_reports_and_bundles_are_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv("QLH_RUN_PHYSICAL_TESTS", raising=False)
    output = tmp_path / "previous.json"
    output.write_text("previous content", encoding="utf-8")
    report = runner.result_path(output)
    report.write_text("previous report", encoding="utf-8")
    assert runner.main(capture_argv(output)) == 1
    assert output.read_text(encoding="utf-8") == "previous content"
    assert report.read_text(encoding="utf-8") == "previous report"
    assert len(list(tmp_path.glob("previous.json.failure-*.json"))) == 1


def test_observation_import_keeps_bytes_times_and_nested_paths(tmp_path, monkeypatch):
    source = tmp_path / "bench" / "native.json"
    output = tmp_path / "out"
    output.mkdir()
    destination = output / "native-copy.json"
    raw = put(source.parent, "DL/raw/loader.json", '[{"synthetic": true}]'.encode("utf-16"))
    selection = {"operation_id": "synthetic", "config_id": "config", "generation": 8}
    value = {**selection, "captured_at": 105, "nodes": [{"raw_evidence": raw}]}
    put(source.parent, source.name, value)
    monkeypatch.setattr(runner.time, "time", lambda: 110)
    runner.import_observation(source, destination, output, selection, 100, set())
    assert destination.read_bytes() == source.read_bytes()
    assert (output / raw["path"]).read_bytes() == (source.parent / raw["path"]).read_bytes()
    with pytest.raises(runner.CaptureError, match="outside capture window"):
        runner.import_observation(source, output / "stale.json", output, selection, 106, set())
    assert not (output / "stale.json").exists()


@pytest.mark.parametrize("history", ["", "assistant 2\n\u4f60 previous question\nassistant 6", "resumed bench session with two historical messages"])
def test_ui_answer_is_isolated_to_current_submission_after_history(history):
    transcript = history + ("\n" if history else "") + f"\u4f60 {runner.MESSAGE}\nassistant 7"
    assert runner.current_rendered_answer(history, transcript, runner.MESSAGE) == "7"


def test_ui_transcript_refresh_cannot_reuse_an_old_or_unlabelled_answer():
    with pytest.raises(runner.CaptureError, match="transcript changed"):
        runner.current_rendered_answer("old history", "2", runner.MESSAGE)
    with pytest.raises(runner.CaptureError, match="boundary"):
        runner.current_rendered_answer("assistant 2", "assistant 2", runner.MESSAGE)



def test_combine_preserves_existing_narrow_per_mode_windows(tmp_path, inputs):
    for path, mode in zip(inputs, ("DL", "LL")):
        manifest = runner.read_json(path)
        manifest.update(capture_started_at=1, capture_finished_at=1000,
                        input_windows=[{"mode": mode, "capture_started_at": 100, "capture_finished_at": 110}])
        path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "narrow" / "bundle.json"
    assert runner.main(combine_argv(inputs, output)) == 0
    merged = runner.read_json(output)
    assert merged["capture_started_at"] == 100
    assert merged["capture_finished_at"] == 110
    assert all(window["capture_started_at"] == 100 and window["capture_finished_at"] == 110
               for window in merged["input_windows"])
