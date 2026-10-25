"""Offline latency evidence must preserve production admission and identity."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from qlh_core.planning.placement_study import (
    PlacementStudyError, PROFILE_SCHEMA, REQUEST_SCHEMA, SCOPE,
    _context, assess_cold_rebalance, assignment_identity, canonical_digest,
    enumerate_capacity_placements, evaluate_placement_study,
)

ROOT = Path(__file__).resolve().parents[1]


def request(*, distributed=True, prefer_all=False, total=6):
    nodes = [{"node_id": "master", "role": "master", "capacity_bytes": 10000,
              "reserve_bytes": 10, "runtime_multiplier": 1.0, "score": 4,
              "engine": "llama_cpp", "execution_device": "cpu"},
             {"node_id": "tail", "role": "worker", "capacity_bytes": 10000,
              "reserve_bytes": 10, "runtime_multiplier": 1.0, "score": 1,
              "engine": "llama_cpp", "execution_device": "cpu"}]
    return {"schema": REQUEST_SCHEMA,
            "descriptor": {"model_id": "synthetic", "model_type": "qwen2", "model_sha256": "a" * 64,
                           "total_layers": total, "layer_weight_bytes": [100] * total,
                           "component_weight_bytes": {"embedding": 10, "final_norm": 10, "lm_head": 10}},
            "nodes": nodes,
            "policy": {"safety_margin": 1.0, "require_distributed": distributed,
                       "required_node_id": "master", "prefer_all_workers": prefer_all},
            "environment": {"epoch": "synthetic-calibration-1", "network_sha256": "b" * 64,
                            "node_runtimes": {n["node_id"]: {"engine": n["engine"], "execution_device": "cpu",
                                "binary_sha256": "c" * 64, "threads": 4, "load_spec_digest": "d" * 64} for n in nodes}},
            "workload": {"phase": "decode", "batch_size": 1, "sequence_count": 1, "context_position": 16,
                         "input_sha256": "e" * 64, "kv_state_sha256": "f" * 64, "hidden_abi": "synthetic-pre-norm-f32", "range_unit_layers": 1},
            "now_ms": 1100, "max_profile_age_ms": 500, "max_candidates": 256, "transition": None}


def enumerate_for(req):
    return enumerate_capacity_placements(req["descriptor"], req["nodes"], policy=req["policy"], max_candidates=req["max_candidates"])


def profile_for(req, *, remote_tax=7):
    records = {}
    for candidate in enumerate_for(req)["candidates"]:
        for a in candidate["assignments"]:
            layers = a["end_layer"] - a["start_layer"]
            compute = layers * (3 if a["node_id"] == "master" else 1)
            noncompute = 0 if a["node_id"] == "master" else remote_tax
            records[a["profile_identity"]] = {"assignment_sha256": a["profile_identity"],
                "samples": [{"compute_ms": compute * factor, "noncompute_ms": noncompute} for factor in (0.9, 1.0, 1.1)],
                "evidence_sha256": "1" * 64}
    return {"schema": PROFILE_SCHEMA, "scope": SCOPE, "source_kind": "synthetic", "binding": _context(req),
            "captured_at_ms": 1000, "records": list(records.values())}


def test_exact_range_latency_can_change_cuts_with_identical_capacity_and_node_policy():
    req = request()
    profiles = profile_for(req)
    original = deepcopy((req, profiles))
    result = evaluate_placement_study(req, profiles)
    assert result["status"] == "evaluated"
    assert result["search"]["complete"]
    assert result["search"]["admitted_candidates"] == 5
    assert [a["layers_count"] for a in result["search"]["baseline"]["assignments"]] == [5, 1]
    assert result["baseline_estimate"]["serial_step_median_sum_ms"] == 23
    assert result["same_node_policy_best"]["serial_step_median_sum_ms"] == 15
    assert result["same_policy_predicted_gain_ms"] == 8
    assert result["same_node_policy_best"]["node_count"] == 2
    assert result["profile_source_kind"] == "synthetic"
    assert result["production_admitted"] is False
    assert result["measured_product_benefit_proven"] is False
    assert (req, profiles) == original


def test_node_count_preference_and_latency_only_counterfactual_are_separate():
    req = request(distributed=False)
    result = evaluate_placement_study(req, profile_for(req))
    assert result["baseline_estimate"]["node_count"] == 1
    assert result["same_node_policy_best"]["node_count"] == 1
    assert result["same_policy_predicted_gain_ms"] == 0
    assert result["latency_only_best"]["node_count"] == 2
    assert result["latency_only_changes_node_count_policy"] is True
    expensive = evaluate_placement_study(req, profile_for(req, remote_tax=30))
    assert expensive["latency_only_best"]["node_count"] == 1
    assert expensive["latency_only_changes_node_count_policy"] is False


def test_paired_total_median_is_not_sum_of_separate_medians():
    req = request()
    profiles = profile_for(req)
    for record in profiles["records"]:
        record["samples"] = [{"compute_ms": 1, "noncompute_ms": 100},
                             {"compute_ms": 100, "noncompute_ms": 1},
                             {"compute_ms": 100, "noncompute_ms": 100}]
    result = evaluate_placement_study(req, profiles)
    stage = result["baseline_estimate"]["stages"][0]
    assert stage["paired_total_median_ms"] == 101
    assert stage["compute_median_ms"] + stage["noncompute_median_ms"] == 200
    assert result["baseline_estimate"]["serial_step_median_sum_ms"] == 202


@pytest.mark.parametrize("part", ["descriptor", "environment", "workload"])
def test_changed_identity_rejects_old_profile(part):
    req = request()
    profiles = profile_for(req)
    if part == "descriptor":
        req[part]["model_sha256"] = "2" * 64
    elif part == "environment":
        req[part]["epoch"] = "network-changed"
    else:
        req[part]["context_position"] += 1
    with pytest.raises(PlacementStudyError, match="identity mismatch"):
        evaluate_placement_study(req, profiles)


@pytest.mark.parametrize("captured", [-1, 599, 1101, True, 1.0])
def test_stale_future_or_noninteger_capture_time_is_rejected(captured):
    req = request()
    profiles = profile_for(req)
    profiles["captured_at_ms"] = captured
    with pytest.raises(PlacementStudyError):
        evaluate_placement_study(req, profiles)


def test_missing_range_profile_does_not_become_zero_or_partial_optimum():
    req = request()
    profiles = profile_for(req)
    profiles["records"].pop()
    result = evaluate_placement_study(req, profiles)
    assert result["status"] == "profile_incomplete"
    assert result["missing_profile_identities"]
    assert result["same_node_policy_best"] is result["latency_only_best"] is None
    assert result["production_admitted"] is False


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "empty", "short", "negative", "nan", "infinity", "boolean", "extra", "unpaired", "bad-proof", "whole-model-scope"])
def test_malformed_or_wrong_scope_profile_cannot_produce_recommendation(mutation):
    req = request()
    profiles = profile_for(req)
    record = profiles["records"][0]
    if mutation == "duplicate":
        profiles["records"].append(deepcopy(record))
    elif mutation == "unknown":
        record["assignment_sha256"] = "0" * 64
    elif mutation == "empty":
        record["samples"] = []
    elif mutation == "short":
        record["samples"].pop()
    elif mutation == "negative":
        record["samples"][0]["noncompute_ms"] = -1
    elif mutation == "nan":
        record["samples"][0]["compute_ms"] = float("nan")
    elif mutation == "infinity":
        record["samples"][0]["noncompute_ms"] = float("inf")
    elif mutation == "boolean":
        record["samples"][0]["compute_ms"] = True
    elif mutation == "extra":
        record["samples"][0]["rtt_ms"] = 1
    elif mutation == "unpaired":
        record["samples"][0].pop("noncompute_ms")
    elif mutation == "bad-proof":
        record["evidence_sha256"] = "missing"
    else:
        profiles["scope"] = "full_model_tokens_per_second"
    with pytest.raises(PlacementStudyError):
        evaluate_placement_study(req, profiles)


def test_capacity_limit_never_returns_best_from_partial_search():
    req = request()
    profiles = profile_for(req)
    req["max_candidates"] = 1
    result = evaluate_placement_study(req, profiles)
    assert result["status"] == "candidate_limit_exceeded"
    assert result["search"]["complete"] is False
    assert result["search"]["candidates"] == []
    assert result["same_node_policy_best"] is None


def test_narrow_original_candidate_ranges_are_intersected_without_relaxing():
    req = request(total=32)
    req["nodes"][0]["candidate_layer_ranges"] = [[0, 4]]
    req["nodes"][1]["candidate_layer_ranges"] = [[4, 32]]
    req["max_candidates"] = 1
    search = enumerate_for(req)
    assert search["complete"] and search["admitted_candidates"] == 1
    assert [(a["start_layer"], a["end_layer"]) for a in search["candidates"][0]["assignments"]] == [(0, 4), (4, 32)]


def test_capacity_and_runtime_reserves_remain_hard_gates():
    req = request()
    req["nodes"][0].update(capacity_bytes=260, reserve_bytes=20, runtime_multiplier=2)
    search = enumerate_for(req)
    assert search["baseline"]["admitted"]
    assert search["rejected_candidates"] == 4
    assert search["admitted_candidates"] == 1
    for candidate in search["candidates"]:
        assert candidate["assignments"][0]["layers_count"] == 1
        for a in candidate["assignments"]:
            assert a["required_bytes"] <= a["capacity_bytes"]


def test_source_artifact_and_component_constraints_are_reused():
    req = request()
    req["nodes"][0]["segment_mode"] = "middle"
    search = enumerate_for(req)
    assert not search["baseline"]["admitted"]
    assert search["candidates"] == []
    assert search["reason_code"] == "baseline_capacity_rejected"


def test_relay_exempt_participants_require_their_own_latency_contract():
    req = request()
    req["nodes"][1].update(capacity_source="relay_exempt", capacity_bytes=0)
    with pytest.raises(PlacementStudyError, match="relay-exempt"):
        enumerate_for(req)


def transition(steps=10):
    return {"trigger": "network_changed", "expected_reuse_steps": steps,
            "costs_ms": {"drain": 1, "materialize": 2, "transfer": 3, "load": 1, "warmup": 1, "search": 2}}


def test_cold_rebalance_requires_strict_gain_after_all_transition_costs():
    costs = transition(steps=10)  # Total 10 ms, gain 1 ms/step: exactly break even.
    result = assess_cold_rebalance(12, 11, costs)
    assert result["transition_cost_ms"] == 10
    assert result["net_window_gain_ms"] == 0
    assert result["strictly_profitable_after_steps"] == 11
    assert not result["worthwhile_in_declared_window"]
    costs["expected_reuse_steps"] = 11
    assert assess_cold_rebalance(12, 11, costs)["worthwhile_in_declared_window"]
    assert assess_cold_rebalance(11, 12, costs)["reason_code"] == "no_positive_step_gain"
    costs["trigger"] = "every_token"
    with pytest.raises(PlacementStudyError, match="cold event"):
        assess_cold_rebalance(12, 11, costs)


def test_rebalance_result_binds_baseline_and_selected_target():
    req = request()
    req["transition"] = transition()
    result = evaluate_placement_study(req, profile_for(req))
    assert result["rebalance"]["from_candidate_id"] == result["baseline_estimate"]["candidate_id"]
    assert result["rebalance"]["to_candidate_id"] == result["same_node_policy_best"]["candidate_id"]
    assert result["rebalance"]["net_window_gain_ms"] == 70


@pytest.mark.parametrize("part,key,value", [
    ("environment", "epoch", ""), ("workload", "batch_size", 2), ("workload", "sequence_count", True),
    ("workload", "phase", "prefill"), ("workload", "context_position", -1),
    ("workload", "input_sha256", "unknown"), ("policy", "prefer_all_workers", 1),
])
def test_unresolved_request_is_not_accepted(part, key, value):
    req = request()
    profiles = profile_for(req)
    req[part][key] = value
    with pytest.raises(PlacementStudyError):
        evaluate_placement_study(req, profiles)


def test_runtime_device_and_load_spec_must_match_node_admission():
    req = request()
    req["environment"]["node_runtimes"]["master"]["execution_device"] = "cuda:0"
    with pytest.raises(PlacementStudyError, match="execution device"):
        _context(req)
    req = request()
    req["nodes"][0]["load_spec_digest"] = "2" * 64
    with pytest.raises(PlacementStudyError, match="load policy"):
        _context(req)


def test_assignment_identity_changes_with_boundary_artifact_and_load_spec():
    assignment = enumerate_for(request())["candidates"][0]["assignments"][0]
    digest = assignment_identity(assignment)
    assignment["capacity_bytes"] *= 2
    assert assignment_identity(assignment) == digest
    for changes in ({"has_lm_head": True}, {"load_spec_digest": "1" * 64},
                    {"layer_artifact": {"artifact_sha256": "2" * 64}}):
        assert assignment_identity({**assignment, **changes}) != digest


def cli():
    spec = importlib.util.spec_from_file_location("placement_study_cli", ROOT / "scripts/placement_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_replay_and_nonempty_output_protection(tmp_path, capsys):
    req = request()
    profile = profile_for(req)
    source = tmp_path / "request.json"
    evidence = tmp_path / "profile.json"
    source.write_text(json.dumps(req), encoding="utf-8")
    evidence.write_text(json.dumps(profile), encoding="utf-8")
    out = tmp_path / "result"
    args = ["--request", str(source), "--profile", str(evidence), "--out-dir", str(out)]
    assert cli().main(args) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    before = (out / "result.json").read_bytes()
    assert cli().main(args) == 1
    assert "fresh empty" in json.loads(capsys.readouterr().out)["error"]
    assert (out / "result.json").read_bytes() == before
    assert (out / "request.json").read_bytes() == source.read_bytes()


def test_cli_template_has_capture_identities_but_is_not_measurement(tmp_path, capsys):
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request()), encoding="utf-8")
    out = tmp_path / "template"
    assert cli().main(["--request", str(source), "--template", "--out-dir", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    template = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert len(template["capture_assignments"]) == 10
    assert template["schema"] != PROFILE_SCHEMA
    with pytest.raises(PlacementStudyError):
        evaluate_placement_study(request(), template)


def test_declared_measurement_never_automatically_admits_product_policy():
    req = request()
    profile = profile_for(req)
    profile["source_kind"] = "declared_measurement"
    result = evaluate_placement_study(req, profile)
    assert result["status"] == "evaluated"
    assert result["production_admitted"] is False
    assert result["measured_product_benefit_proven"] is False


def test_nonadditive_device_peak_and_independent_host_budget_are_retained():
    req = request()
    master = req["nodes"][0]
    master.update(capacity_bytes=80, candidate_layer_ranges=[[0, 4]],
                  range_weight_bytes=[{"layer_range": [0, 4], "weight_bytes": 50,
                                       "has_embedding": True, "has_lm_head": False}],
                  host_required_bytes=500, host_capacity_bytes=600)
    search = enumerate_for(req)
    assert search["admitted_candidates"] == 1
    a = search["candidates"][0]["assignments"][0]
    assert a["layers_count"] == 4 and a["raw_weight_bytes"] == 50
    assert a["required_bytes"] == 60
    assert a["host_required_bytes"] == 500 and a["host_capacity_bytes"] == 600
    assert evaluate_placement_study(req, profile_for(req))["status"] == "evaluated"
    master["range_weight_bytes"][0]["has_embedding"] = False
    assert not enumerate_for(req)["baseline"]["admitted"]


def test_runtime_raw_load_spec_is_checked_even_without_redundant_digest():
    from qlh_core.contracts.load import RuntimeLoadSpec
    req = request()
    spec = RuntimeLoadSpec(device="cpu", dtype="float32", quant_type="fp32", context_size=64,
                           kv_device="cpu", kv_dtype="float32", kv_policy="dynamic",
                           components=("layers", "embedding"), materialization="gguf_segment", batch_size=1)
    req["nodes"][0]["load_spec"] = spec.to_dict()
    with pytest.raises(PlacementStudyError, match="load policy"):
        _context(req)
    req["environment"]["node_runtimes"]["master"]["load_spec_digest"] = spec.digest
    assert _context(req)["environment_sha256"] == canonical_digest(req["environment"])


def test_reversing_input_nodes_keeps_the_same_admitted_candidates_and_optimum():
    req = request()
    profiles = profile_for(req)
    first = evaluate_placement_study(req, profiles)
    req["nodes"].reverse()
    second = evaluate_placement_study(req, profiles)
    assert first["search"]["candidates"] == second["search"]["candidates"]
    assert first["same_node_policy_best"] == second["same_node_policy_best"]


def test_tied_latency_preserves_current_baseline_instead_of_churning():
    req = request()
    profiles = profile_for(req)
    for record in profiles["records"]:
        record["samples"] = [{"compute_ms": 5, "noncompute_ms": 1}] * 3
    result = evaluate_placement_study(req, profiles)
    assert result["same_node_policy_best"]["candidate_id"] == result["baseline_estimate"]["candidate_id"]
    assert result["same_policy_predicted_gain_ms"] == 0


def test_public_synthetic_fixture_demonstrates_unamortized_cold_transition():
    req = json.loads((ROOT / "fixtures/placement_study/synthetic-request.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "fixtures/placement_study/synthetic-profile.json").read_text(encoding="utf-8"))
    result = evaluate_placement_study(req, profile)
    assert result["status"] == "evaluated"
    assert result["same_policy_predicted_gain_ms"] == 8
    assert result["rebalance"]["transition_cost_ms"] == 1007
    assert result["rebalance"]["net_window_gain_ms"] == -927
    assert result["rebalance"]["strictly_profitable_after_steps"] == 126
    assert not result["rebalance"]["worthwhile_in_declared_window"]


def test_pure_study_has_no_ambient_clock_environment_or_file_access(monkeypatch):
    import builtins
    import os
    import time
    req = request()
    profile = profile_for(req)
    def forbidden(*args, **kwargs):
        raise AssertionError("offline planning accessed ambient state")
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(time, "time", forbidden)
        patch.setattr(time, "monotonic", forbidden)
        patch.setattr(os, "getenv", forbidden)
        result = evaluate_placement_study(req, profile)
    assert result["status"] == "evaluated"


def test_grouped_capacity_coordinates_map_explicitly_to_source_layers(tmp_path, capsys):
    req = request()
    req["workload"]["range_unit_layers"] = 4
    source = tmp_path / "request.json"
    source.write_text(json.dumps(req), encoding="utf-8")
    out = tmp_path / "template"
    assert cli().main(["--request", str(source), "--template", "--out-dir", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    data = json.loads((out / "result.json").read_text(encoding="utf-8"))
    for stage in data["capture_assignments"]:
        assert stage["source_layer_range"] == [stage["start_layer"] * 4, stage["end_layer"] * 4]
    req["workload"]["range_unit_layers"] = 0
    with pytest.raises(PlacementStudyError, match="range_unit_layers"):
        _context(req)


def test_profile_for_raw_layers_is_rejected_for_grouped_coordinate_request():
    req = request()
    profile = profile_for(req)
    req["workload"]["range_unit_layers"] = 4
    with pytest.raises(PlacementStudyError, match="identity mismatch"):
        evaluate_placement_study(req, profile)


def test_profile_variation_is_reported_without_turning_into_product_admission():
    req = request()
    profile = profile_for(req)
    for record in profile["records"]:
        record["samples"] = [{"compute_ms": v, "noncompute_ms": 0} for v in (1, 2, 30)]
    result = evaluate_placement_study(req, profile)
    stage = result["baseline_estimate"]["stages"][0]
    assert stage["paired_total_min_ms"] == 1
    assert stage["paired_total_max_ms"] == 30
    assert stage["paired_total_std_ms"] > 10
    assert stage["paired_total_cv"] > 1
    assert result["measured_product_benefit_proven"] is False
