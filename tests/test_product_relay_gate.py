"""Gate logic regressions use synthetic fixtures; these are not real product evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tests.product_observation_fixtures import node_observation

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
                  "nodes": {a["node_id"]: {"artifact_sha256": "b"*64} for a in assignments},
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
        topology_nodes, native_nodes = [], []
        for assignment in assignments:
            raw, topology_node, native_node, addresses = node_observation(
                identity, assignment, assignments, REVISION, 32)
            ref = write_file(tmp_path, f"{mode}/raw/{assignment['node_id']}.json", raw)
            topology_nodes.append({**topology_node, "raw_evidence": ref})
            native_nodes.append({**native_node, "raw_evidence": ref})
        topology = {**identity, "captured_at": 106.5, "source": "os_process_connections", "complete": True,
                    "node_addresses": addresses, "nodes": topology_nodes}
        native = {**identity, "captured_at": 106.5, "node_addresses": addresses, "nodes": native_nodes}
        screen = write_file(tmp_path, f"{mode}/screen.svg", "<svg>synthetic UI fixture</svg>")
        ui = {"entry": "textual", "done": True, "request_id": f"http_{mode}", "submitted_message": "1+1?",
              "expected_response": "2", "rendered_response": "2", "captured_at": 107.0, "screen": screen}
        run = {"mode": mode, "source_dirty": False}
        for name, obj in (("intent", intent), ("response", response), ("topology", topology),
                          ("native", native), ("ui", ui)):
            run[name] = write_file(tmp_path, f"{mode}/{name}.json", obj)
        run["handoffs"] = write_file(tmp_path, f"{mode}/handoffs.log",
            "\n".join(gate.COMPLETED_PREFIX + json.dumps(e) for e in events))
        runs.append(run)
    manifest = {"schema_version": 1, "source_revision": REVISION, "source_dirty": False, "physical_opt_in": True,
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



def mark_source(bundle, level, value):
    """Preserve raw evidence hashes when marking a native node's source facts."""
    base, manifest = bundle[0].parent, bundle[1]
    if level == "bundle":
        manifest["source_dirty"] = value
    elif level in ("DL", "LL"):
        next(run for run in manifest["runs"] if run["mode"] == level)["source_dirty"] = value
    else:
        node = level.removeprefix("native-")
        update_json(base, manifest["runs"][0]["native"],
                    lambda obj: next(item for item in obj["nodes"] if item["node_id"] == node)
                    .update(source_dirty=value))


def test_explicit_clean_source_is_required_at_every_level(bundle):
    assert validate(bundle)["passed"]


@pytest.mark.parametrize("level", ["bundle", "DL", "LL", "native-master", "native-pc", "native-android"])
def test_missing_source_cleanliness_is_not_legacy_release_evidence(bundle, level):
    base, manifest = bundle[0].parent, bundle[1]
    if level == "bundle":
        manifest.pop("source_dirty")
    elif level in ("DL", "LL"):
        next(run for run in manifest["runs"] if run["mode"] == level).pop("source_dirty")
    else:
        node = level.removeprefix("native-")
        update_json(base, manifest["runs"][0]["native"],
                    lambda obj: next(item for item in obj["nodes"] if item["node_id"] == node)
                    .pop("source_dirty"))
    result = validate(bundle)
    assert result["passed"] is False
    assert any("source_dirty" in error for error in result["errors"]), result


@pytest.mark.parametrize("level", ["bundle", "DL", "LL", "native-master", "native-pc", "native-android"])
@pytest.mark.parametrize("value", [True, None, 0, 1, 0.0, "false", "true", "", [], {}])
def test_explicit_dirty_or_invalid_source_cannot_be_same_revision_evidence(bundle, level, value):
    # Clean outer facts must not override a dirty/invalid run or native node.
    for clean_level in ("bundle", "DL", "LL", "native-master", "native-pc", "native-android"):
        mark_source(bundle, clean_level, False)
    mark_source(bundle, level, value)
    result = validate(bundle)
    assert result["passed"] is False
    assert any("source_dirty" in error for error in result["errors"]), result
    if level in ("DL", "LL"):
        assert any(error.startswith(f"{level}:") for error in result["errors"])
        assert [run["mode"] for run in result["runs"]] == ["LL" if level == "DL" else "DL"]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("level", ["bundle", "DL", "LL", "native-android"])
@pytest.mark.parametrize("value", [True, 0])
def test_runner_combine_rejects_explicit_bad_source_before_writing_bundle(bundle, tmp_path, level, value):
    runner = load_script("product_source_dirty_runner", "scripts/run_product_relay_gate.py")
    mark_source(bundle, level, value)
    inputs = []
    for run in bundle[1]["runs"]:
        path = tmp_path / f"input-{run['mode']}.json"
        path.write_text(json.dumps({**bundle[1], "runs": [run]}), encoding="utf-8")
        inputs.append(str(path))
    output = tmp_path / "combined" / "bundle.json"
    assert runner.main(["combine", "--inputs", *inputs, "--output", str(output)]) == 1
    report = json.loads(runner.result_path(output).read_text(encoding="utf-8"))
    assert report["passed"] is False and any("source_dirty" in error for error in report["errors"])
    assert not output.exists()
    assert not (output.parent / "DL").exists()


@pytest.mark.parametrize("level", ["bundle", "DL", "LL", "native-android"])
@pytest.mark.parametrize("value", [True, 0])
def test_release_acceptance_rejects_bad_product_source_even_with_passing_platforms(bundle, tmp_path, level, value):
    acceptance = load_script("product_source_dirty_acceptance", "scripts/package_contract/release_acceptance.py")
    contract_bytes = (ROOT / "release-contract.json").read_bytes()
    reports = {}
    for platform in acceptance.PLATFORMS:
        path = tmp_path / f"{platform}.json"
        report = {
            "schema_version": 2, "gate": "release-artifacts", "passed": True, "status": "passed",
            "source_revision": REVISION, "required_revision": REVISION, "revision_verified": True,
            "requested_platforms": [platform], "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "contract_version": json.loads(contract_bytes)["version"], "reports": [{
                "platform": platform, "passed": True, "status": "passed", "items": [
                    {"check": check, "passed": True, "status": "passed", "evidence": {
                        "source_revision": REVISION, "revision_verified": True}}
                    for check in sorted(acceptance.MANDATORY[platform])]}],
        }
        path.write_text(json.dumps(report), encoding="utf-8")
        reports[platform] = path
    assert validate(bundle)["passed"]
    assert acceptance.summarize_release(reports, bundle[0], REVISION)["passed"]
    mark_source(bundle, level, value)
    validate(bundle)  # Persist the marked fixture for the real acceptance validator.
    result = acceptance.summarize_release(reports, bundle[0], REVISION)
    assert all(report["passed"] for report in result["platforms"].values())
    assert result["passed"] is False and result["product"]["passed"] is False
    assert any("source_dirty" in error for error in result["product"]["errors"])


def change_raw_node(bundle, node, alter):
    """Rehash both snapshot references while preserving their claimed success."""
    base, manifest = bundle[0].parent, bundle[1]
    run = manifest["runs"][0]
    topology = read_json(base, run["topology"])
    native = read_json(base, run["native"])
    top_node = next(item for item in topology["nodes"] if item["node_id"] == node)
    raw = read_json(base, top_node["raw_evidence"])
    alter(raw)
    ref = write_file(base, top_node["raw_evidence"]["path"], raw)
    top_node["raw_evidence"] = ref
    next(item for item in native["nodes"] if item["node_id"] == node)["raw_evidence"] = ref
    run["topology"] = write_file(base, run["topology"]["path"], topology)
    run["native"] = write_file(base, run["native"]["path"], native)


@pytest.mark.parametrize("node,alter", [
    ("pc", lambda raw: raw.update(schema_version=0)),
    ("pc", lambda raw: raw.pop("facts")),
    ("pc", lambda raw: raw["commands"][0].update(returncode=1)),
    ("pc", lambda raw: raw["facts"].pop("source")),
    ("pc", lambda raw: raw["facts"]["source"].update(source_dirty=True)),
    ("pc", lambda raw: raw["facts"]["source"].update(runtime_manifest_sha256="f"*64)),
    ("pc", lambda raw: raw["facts"]["source"]["rehashed_files"][0].update(sha256="f"*64)),
    ("pc", lambda raw: raw["facts"]["identity_after"].update(created_at=11)),
    ("pc", lambda raw: raw["facts"]["process_tree"].update(complete=False)),
    ("pc", lambda raw: raw["facts"]["process_tree"]["processes"][0].update(child_pids=[999])),
    ("pc", lambda raw: raw["facts"]["process_tree"]["processes"][0].update(maps_complete=False)),
    ("pc", lambda raw: raw["facts"]["process_tree"]["processes"][0].update(connections_complete=False)),
    ("pc", lambda raw: raw["facts"]["process_tree"]["processes"][0]["model_mappings"].append("/models/full.gguf")),
    ("pc", lambda raw: raw["facts"]["process_tree"]["processes"][0].update(model_mappings=[])),
    ("pc", lambda raw: raw["facts"]["process_tree"]["processes"][0]["connections"].append({
        "status": "ESTABLISHED", "local": ["10.91.0.2", 40001], "remote": ["10.91.0.3", 8888]})),
    ("pc", lambda raw: raw["facts"]["runtime"].update(full_model_loaded=True)),
    ("pc", lambda raw: raw["facts"]["runtime"].update(loaded=False)),
    ("pc", lambda raw: raw["facts"]["runtime"].update(operation_id="another-selection")),
    ("pc", lambda raw: raw["facts"]["runtime"].update(generation=1)),
    ("pc", lambda raw: raw["facts"]["runtime"].update(total_layers=33)),
    ("pc", lambda raw: raw["facts"]["runtime"].update(engine="pytorch")),
    ("pc", lambda raw: raw["facts"]["runtime"]["artifact"].update(sha256="f"*64, rehashed_sha256="f"*64)),
    ("pc", lambda raw: raw["facts"]["runtime"].update(native_layer_count=32)),
    ("pc", lambda raw: raw.update(started_at=99)),
    ("android", lambda raw: raw["facts"]["process_tree"]["processes"][0].update(children_complete=False)),
    ("android", lambda raw: raw["facts"]["source"].update(sha256_after="f"*64)),
    ("android", lambda raw: raw["facts"]["source"].update(path_after="/data/app/replaced/base.apk")),
    ("android", lambda raw: raw["facts"]["source"].pop("build_info")),
    ("master", lambda raw: raw["facts"]["runtime"].pop("materialization")),
    ("master", lambda raw: raw["facts"]["runtime"]["materialization"].update(materialized_bytes=1)),
    ("master", lambda raw: raw["facts"]["runtime"]["materialization"].update(layer_range=[0, 32])),
])
def test_success_summary_cannot_override_failed_or_contradictory_raw_observation(bundle, node, alter):
    change_raw_node(bundle, node, alter)
    result = validate(bundle)
    assert result["passed"] is False, result
    assert result["errors"], result


def test_dirty_actual_apk_cannot_be_hidden_by_clean_native_summary(bundle):
    from tests.product_observation_fixtures import canonical_hash

    def dirty_apk(raw):
        source = raw["facts"]["source"]
        source["build_info"]["source_dirty"] = True
        source["build_info_sha256"] = canonical_hash(source["build_info"])

    change_raw_node(bundle, "android", dirty_apk)
    result = validate(bundle)
    assert result["passed"] is False
    assert any("source_dirty" in error for error in result["errors"]), result


def test_original_tail_only_probe_is_not_a_native_observation(bundle):
    def legacy_tail(raw):
        raw.clear()
        raw.update(ssh_returncode=0, probe_stdout_tail="partial pretty JSON tail",
                   synced_revision=REVISION, sync_verification="197/197")

    change_raw_node(bundle, "pc", legacy_tail)
    assert validate(bundle)["passed"] is False


def test_combine_rejects_success_summary_whose_raw_probe_failed_before_copying(bundle, tmp_path):
    change_raw_node(bundle, "pc", lambda raw: raw["commands"][0].update(returncode=1))
    validate(bundle)
    runner = load_script("product_failed_raw_runner", "scripts/run_product_relay_gate.py")
    inputs = []
    for run in bundle[1]["runs"]:
        path = tmp_path / f"raw-input-{run['mode']}.json"
        path.write_text(json.dumps({**bundle[1], "runs": [run]}), encoding="utf8")
        inputs.append(str(path))
    output = tmp_path / "raw-rejected" / "bundle.json"
    assert runner.main(["combine", "--inputs", *inputs, "--output", str(output)]) == 1
    assert not output.exists()
    assert not (output.parent / "DL").exists()


@pytest.fixture
def windows_mapped_alias_observation():
    from tests.product_observation_fixtures import node_observation
    identity = {"operation_id": "mapped-op", "config_id": "mapped-cfg", "generation": 19}
    assignments = [{"node_id": "master", "engine": "pytorch", "start_layer": 0, "end_layer": 4},
                   {"node_id": "pc", "engine": "llama_cpp", "start_layer": 4, "end_layer": 8},
                   {"node_id": "android", "engine": "llama_cpp", "start_layer": 8, "end_layer": 32}]
    raw, _, _, addresses = node_observation(identity, assignments[1], assignments, REVISION, 32)
    artifact = raw["facts"]["runtime"]["artifact"]
    artifact.update(path="C:/qlh/active/segment.gguf", size_bytes=8193)
    process = raw["facts"]["process_tree"]["processes"][0]
    process["model_mappings"] = ["C:/qlh/retired/segment.gguf"]
    process["mapped_artifact_proofs"] = [{
        "method": "windows_read_process_memory_sha256", "pid": process["pid"],
        "created_at": process["created_at"], "identity_after": process["identity_after"].copy(),
        "mapped_path": process["model_mappings"][0], "artifact_path": artifact["path"],
        "artifact_size_bytes": 8193, "sha256": artifact["sha256"], "page_size": 4096,
        "regions": [{"address": 65536, "region_bytes": 12288, "hashed_bytes": 8193,
                     "sha256": artifact["sha256"]}],
        "maps_stable": True, "started_at": 104.0, "finished_at": 105.0}]
    expected = dict(expected_revision=REVISION, expected_node_id="pc", expected_identity=identity,
                    expected_assignment={**assignments[1], "_total_layers": 32,
                                         "artifact_sha256": artifact["sha256"]}, node_addresses=addresses)
    return raw, expected


def test_windows_retired_mapping_name_needs_complete_actual_byte_proof(windows_mapped_alias_observation):
    from product_observations import derive_node_observation
    raw, expected = windows_mapped_alias_observation
    before = copy.deepcopy(raw)
    derived = derive_node_observation(raw, **expected)
    assert derived["native"]["layer_range"] == [4, 8]
    assert derived["topology"]["worker_peers"] == ["master"]
    assert raw == before  # Neither original OS names nor times are relabelled.


@pytest.mark.parametrize("alter", [
    lambda p: p.update(method="disk_hash_only"),
    lambda p: p.update(pid=999),
    lambda p: p["identity_after"].update(created_at=99),
    lambda p: p.update(maps_stable=False),
    lambda p: p.update(mapped_path="C:/unobserved/segment.gguf"),
    lambda p: p.update(artifact_path="C:/different/segment.gguf"),
    lambda p: p.update(sha256="f"*64),
    lambda p: p.update(artifact_size_bytes=8192),
    lambda p: p.update(artifact_size_bytes=True),
    lambda p: p.update(started_at=1),
    lambda p: p.update(finished_at=999),
    lambda p: p.update(page_size=3),
    lambda p: p.update(page_size=True),
    lambda p: p.update(regions=[]),
    lambda p: p["regions"][0].update(address=1),
    lambda p: p["regions"][0].update(address=2**64),
    lambda p: p["regions"][0].update(region_bytes=16384),
    lambda p: p["regions"][0].update(hashed_bytes=8192),
    lambda p: p["regions"][0].update(sha256="f"*64),
    lambda p: p["regions"].append(p["regions"][0].copy()),
])
def test_wrong_or_partial_mapped_byte_proof_is_rejected(windows_mapped_alias_observation, alter):
    from product_observations import derive_node_observation, ObservationError
    raw, expected = windows_mapped_alias_observation
    alter(raw["facts"]["process_tree"]["processes"][0]["mapped_artifact_proofs"][0])
    with pytest.raises(ObservationError):
        derive_node_observation(raw, **expected)


@pytest.mark.parametrize("change", ["missing", "extra-model", "duplicate-proof", "non-windows", "full-model"])
def test_mapping_alias_proof_cannot_hide_other_missing_native_facts(windows_mapped_alias_observation, change):
    from product_observations import derive_node_observation, ObservationError
    raw, expected = windows_mapped_alias_observation
    process = raw["facts"]["process_tree"]["processes"][0]
    if change == "missing": process.pop("mapped_artifact_proofs")
    elif change == "extra-model": process["model_mappings"].append("C:/full.gguf")
    elif change == "duplicate-proof": process["mapped_artifact_proofs"].append(copy.deepcopy(process["mapped_artifact_proofs"][0]))
    elif change == "non-windows": raw["facts"]["runtime"]["artifact"]["path"] = "/models/segment.gguf"
    else: raw["facts"]["runtime"]["native_layer_count"] = 32
    with pytest.raises(ObservationError):
        derive_node_observation(raw, **expected)
