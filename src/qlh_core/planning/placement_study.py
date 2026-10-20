"""Bounded, offline placement comparisons using production capacity admission.

There are no scheduler hooks. Profiles quote exact stage ranges and paired
compute/noncompute samples for a single serial decode step. Missing evidence
never becomes zero latency, and an incomplete search never returns an optimum.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
import statistics

from qlh_core.contracts.load import RuntimeLoadSpec
from qlh_core.planning.capacity import _normalize_nodes, solve_pipeline_capacity

REQUEST_SCHEMA = "qlh.placement_study.v1"
PROFILE_SCHEMA = "qlh.placement_profile.v1"
RESULT_SCHEMA = "qlh.placement_study_result.v1"
SCOPE = "single_sequence_serial_decode_stage_roundtrip"


class PlacementStudyError(ValueError):
    """An offline comparison has invalid or mismatched evidence."""


def canonical_digest(value):
    """Hash explicit JSON inputs; no clock, environment, files, or runtime IO."""
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlacementStudyError("input must be finite JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise PlacementStudyError(f"{name} fields must be {sorted(fields)}")
    return value


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise PlacementStudyError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _number(value, name, *, positive=False):
    if type(value) not in (int, float):
        raise PlacementStudyError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise PlacementStudyError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise PlacementStudyError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return number


def _identity(value, name):
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise PlacementStudyError(f"{name} must be a nonempty identity")
    return value


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise PlacementStudyError(f"{name} must be a lowercase SHA-256")
    return value


def assignment_identity(assignment):
    """Range/IO/artifact/load-policy identity, excluding changing free bytes."""
    return canonical_digest({
        "node_id": assignment["node_id"],
        "layer_range": [assignment["start_layer"], assignment["end_layer"]],
        "has_embedding": assignment["has_embedding"],
        "has_lm_head": assignment["has_lm_head"],
        "execution_device": assignment["execution_device"],
        "load_spec_digest": assignment.get("load_spec_digest"),
        "artifact_sha256": assignment.get("layer_artifact", {}).get("artifact_sha256"),
    })


def _candidate(plan):
    assignments = deepcopy(plan["assignments"])
    for assignment in assignments:
        assignment["profile_identity"] = assignment_identity(assignment)
    return {"candidate_id": canonical_digest([a["profile_identity"] for a in assignments]),
            "admission_plan_id": plan["plan_id"], "assignments": assignments,
            "node_count": len(assignments), "capacity_objective": deepcopy(plan["placement_objective"])}


def enumerate_capacity_placements(descriptor, nodes, *, policy, max_candidates=2048):
    """Enumerate node subsequences and cuts, preserving the solver's node order.

    Every fixed candidate is re-admitted by the unchanged production solver.
    Original candidate ranges are intersected, and all unselected nodes receive
    an empty candidate range list. Relay-exempt / partial remote chains need a
    separate latency contract and are outside this experiment.
    """
    _object(policy, {"safety_margin", "require_distributed", "required_node_id", "prefer_all_workers"}, "policy")
    for key in ("require_distributed", "prefer_all_workers"):
        if type(policy[key]) is not bool:
            raise PlacementStudyError(f"policy.{key} must be boolean")
    if policy["required_node_id"] is not None:
        _identity(policy["required_node_id"], "required_node_id")
    if _number(policy["safety_margin"], "safety_margin", positive=True) < 1:
        raise PlacementStudyError("safety_margin must be at least one")
    _integer(max_candidates, 1, 4096, "max_candidates")
    if not isinstance(descriptor, dict):
        raise PlacementStudyError("descriptor must be an object")
    total = _integer(descriptor.get("total_layers"), 1, 256, "total_layers")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 8:
        raise PlacementStudyError("study requires 1..8 node records")
    ordered, _ = _normalize_nodes(nodes, total_layers=total)
    if any(node["capacity_source"] == "relay_exempt" for node in ordered):
        raise PlacementStudyError("relay-exempt stages need a separate measured latency contract")
    baseline = solve_pipeline_capacity(deepcopy(descriptor), deepcopy(nodes), **policy)
    result = {"baseline": baseline, "complete": True, "reason_code": "", "candidates": [],
              "fixed_configurations": 0, "admitted_candidates": 0, "rejected_candidates": 0,
              "order": [node["node_id"] for node in ordered]}
    if not baseline["admitted"]:
        result["reason_code"] = "baseline_capacity_rejected"
        return result
    configurations = []
    visited = 0
    limit_hit = False

    def append_ranges(selected, index, cursor, chosen):
        nonlocal visited, limit_hit
        visited += 1
        if visited > max_candidates * (len(ordered) + 1):
            limit_hit = True
            return
        if index == len(selected):
            if cursor == total:
                configurations.append(dict(chosen))
                limit_hit = len(configurations) > max_candidates
            return
        node = selected[index]
        remaining = len(selected) - index - 1
        last = total - remaining
        if "candidate_layer_ranges" in node:
            ends = [end for start, end in node["candidate_layer_ranges"] if start == cursor and end <= last]
        else:
            ends = [total] if remaining == 0 else range(cursor + 1, last + 1)
        for end in ends:
            if remaining == 0 and end != total:
                continue
            chosen[node["node_id"]] = [cursor, end]
            append_ranges(selected, index + 1, end, chosen)
            if limit_hit:
                return
            chosen.pop(node["node_id"])

    minimum = 2 if policy["require_distributed"] else 1
    for count in range(minimum, min(len(ordered), total) + 1):
        for selected in combinations(ordered, count):
            if policy["required_node_id"] is not None and not any(n["node_id"] == policy["required_node_id"] for n in selected):
                continue
            append_ranges(selected, 0, 0, {})
            if limit_hit:
                result.update(complete=False, reason_code="candidate_limit_exceeded", fixed_configurations=len(configurations))
                return result
    result["fixed_configurations"] = len(configurations)
    for ranges in configurations:
        fixed = deepcopy(nodes)
        for node in fixed:
            node["candidate_layer_ranges"] = [ranges[str(node["node_id"]).strip()]] if str(node["node_id"]).strip() in ranges else []
        plan = solve_pipeline_capacity(deepcopy(descriptor), fixed, **policy)
        if not plan["admitted"]:
            result["rejected_candidates"] += 1
            continue
        candidate = _candidate(plan)
        actual = {a["node_id"]: [a["start_layer"], a["end_layer"]] for a in candidate["assignments"]}
        if actual != ranges:
            raise PlacementStudyError("fixed candidate changed its requested ranges")
        result["candidates"].append(candidate)
    result["candidates"].sort(key=lambda c: c["candidate_id"])
    result["admitted_candidates"] = len(result["candidates"])
    baseline_id = _candidate(baseline)["candidate_id"]
    if not any(c["candidate_id"] == baseline_id for c in result["candidates"]):
        raise PlacementStudyError("enumeration did not reproduce the production baseline")
    return result


def _context(request):
    _object(request, {"schema", "descriptor", "nodes", "policy", "environment", "workload", "now_ms", "max_profile_age_ms", "max_candidates", "transition"}, "request")
    if request["schema"] != REQUEST_SCHEMA:
        raise PlacementStudyError("unsupported request schema")
    _integer(request["now_ms"], 0, 2**63 - 1, "now_ms")
    _integer(request["max_profile_age_ms"], 1, 2**63 - 1, "max_profile_age_ms")
    env = _object(request["environment"], {"epoch", "network_sha256", "node_runtimes"}, "environment")
    _identity(env["epoch"], "environment.epoch")
    _sha(env["network_sha256"], "environment.network_sha256")
    if not isinstance(env["node_runtimes"], dict) or not isinstance(request["nodes"], list):
        raise PlacementStudyError("node runtime identities and nodes must be explicit")
    expected = {str(n.get("node_id", "")).strip() for n in request["nodes"] if isinstance(n, dict)}
    if set(env["node_runtimes"]) != expected:
        raise PlacementStudyError("node runtime identities must cover exactly the candidate nodes")
    for node in request["nodes"]:
        if not isinstance(node, dict):
            raise PlacementStudyError("node must be an object")
        runtime = _object(env["node_runtimes"][str(node["node_id"]).strip()],
                          {"engine", "execution_device", "binary_sha256", "threads", "load_spec_digest"}, "node runtime")
        _identity(runtime["engine"], "runtime.engine")
        _identity(runtime["execution_device"], "runtime.execution_device")
        _sha(runtime["binary_sha256"], "runtime.binary_sha256")
        _integer(runtime["threads"], 1, 256, "runtime.threads")
        _sha(runtime["load_spec_digest"], "runtime.load_spec_digest")
        if runtime["execution_device"] != node.get("execution_device"):
            raise PlacementStudyError("runtime execution device differs from capacity node")
        if "engine" in node and runtime["engine"] != node["engine"]:
            raise PlacementStudyError("runtime engine differs from capacity node")
        expected_load = node.get("load_spec_digest")
        if "load_spec" in node:
            spec = node["load_spec"]
            expected_load = spec.digest if isinstance(spec, RuntimeLoadSpec) else RuntimeLoadSpec.from_dict(spec).digest
        if expected_load is not None and runtime["load_spec_digest"] != expected_load:
            raise PlacementStudyError("runtime load policy differs from capacity node")
    work = _object(request["workload"], {"phase", "batch_size", "sequence_count", "context_position", "input_sha256", "kv_state_sha256", "hidden_abi", "range_unit_layers"}, "workload")
    if work["phase"] != "decode" or type(work["batch_size"]) is not int or work["batch_size"] != 1 or type(work["sequence_count"]) is not int or work["sequence_count"] != 1:
        raise PlacementStudyError("profile study supports one serial decode token and one sequence")
    _integer(work["context_position"], 0, 2**31 - 1, "context_position")
    _integer(work["range_unit_layers"], 1, 256, "range_unit_layers")
    _sha(work["input_sha256"], "input_sha256")
    _sha(work["kv_state_sha256"], "kv_state_sha256")
    _identity(work["hidden_abi"], "hidden_abi")
    return {"descriptor_sha256": canonical_digest(request["descriptor"]),
            "environment_sha256": canonical_digest(env), "workload_sha256": canonical_digest(work)}


def _profiles(profile, binding, request):
    _object(profile, {"schema", "scope", "source_kind", "binding", "captured_at_ms", "records"}, "profile")
    if profile["schema"] != PROFILE_SCHEMA or profile["scope"] != SCOPE:
        raise PlacementStudyError("unsupported profile schema or timing scope")
    if profile["source_kind"] not in ("synthetic", "declared_measurement"):
        raise PlacementStudyError("profile source_kind must label synthetic or declared_measurement")
    if profile["binding"] != binding:
        raise PlacementStudyError("profile model/environment/workload identity mismatch")
    captured = _integer(profile["captured_at_ms"], 0, 2**63 - 1, "captured_at_ms")
    age = request["now_ms"] - captured
    if not 0 <= age <= request["max_profile_age_ms"]:
        raise PlacementStudyError("profile is stale or from the future")
    records = profile["records"]
    if not isinstance(records, list) or len(records) > 32768:
        raise PlacementStudyError("profile records must be a bounded list")
    indexed = {}
    sample_count = 0
    for record in records:
        _object(record, {"assignment_sha256", "samples", "evidence_sha256"}, "profile record")
        identity = _sha(record["assignment_sha256"], "assignment_sha256")
        _sha(record["evidence_sha256"], "evidence_sha256")
        if identity in indexed:
            raise PlacementStudyError("duplicate profile assignment identity")
        samples = record["samples"]
        if not isinstance(samples, list) or not 3 <= len(samples) <= 10000:
            raise PlacementStudyError("each exact stage requires 3..10000 paired samples")
        sample_count += len(samples)
        if sample_count > 100000:
            raise PlacementStudyError("profile exceeds total sample budget")
        compute, other, totals = [], [], []
        for sample in samples:
            _object(sample, {"compute_ms", "noncompute_ms"}, "sample")
            compute.append(_number(sample["compute_ms"], "compute_ms", positive=True))
            other.append(_number(sample["noncompute_ms"], "noncompute_ms"))
            totals.append(_number(compute[-1] + other[-1], "sample total", positive=True))
        total_mean = statistics.mean(totals)
        total_std = statistics.pstdev(totals)
        indexed[identity] = {"profile_identity": identity, "compute_median_ms": statistics.median(compute),
                             "noncompute_median_ms": statistics.median(other), "paired_total_median_ms": statistics.median(totals),
                             "paired_total_std_ms": total_std, "paired_total_cv": total_std / total_mean,
                             "paired_total_min_ms": min(totals), "paired_total_max_ms": max(totals),
                             "sample_count": len(samples), "evidence_sha256": record["evidence_sha256"]}
    return indexed


def assess_cold_rebalance(current_ms, candidate_ms, transition):
    """Strict amortization calculation for an explicit cold event only."""
    _object(transition, {"trigger", "expected_reuse_steps", "costs_ms"}, "transition")
    if transition["trigger"] not in ("device_changed", "network_changed", "model_changed", "calibration_changed", "manual_cold_replan"):
        raise PlacementStudyError("rebalance requires an explicit cold event")
    steps = _integer(transition["expected_reuse_steps"], 1, 10**9, "expected_reuse_steps")
    costs = _object(transition["costs_ms"], {"drain", "materialize", "transfer", "load", "warmup", "search"}, "transition costs_ms")
    cost = _number(sum(_number(v, "transition cost") for v in costs.values()), "transition total")
    current_ms = _number(current_ms, "current step", positive=True)
    candidate_ms = _number(candidate_ms, "candidate step", positive=True)
    gain = current_ms - candidate_ms
    net = _number(abs(gain * steps), "reuse window gain") * (1 if gain >= 0 else -1) - cost
    if not math.isfinite(net):
        raise PlacementStudyError("net reuse gain must be finite")
    worthwhile = gain > 0 and net > 0
    quotient = cost / gain if gain > 0 else None
    break_even = math.floor(quotient) + 1 if quotient is not None and math.isfinite(quotient) else None
    return {"trigger": transition["trigger"], "expected_reuse_steps": steps, "transition_cost_ms": cost,
            "per_step_gain_ms": gain, "net_window_gain_ms": net, "strictly_profitable_after_steps": break_even,
            "worthwhile_in_declared_window": worthwhile,
            "reason_code": "amortized_cold_event" if worthwhile else "no_positive_step_gain" if gain <= 0 else "transition_not_amortized"}


def evaluate_placement_study(request, profile):
    """Compare exact-range serial costs without enabling a production policy."""
    binding = _context(request)
    indexed = _profiles(profile, binding, request)
    search = enumerate_capacity_placements(request["descriptor"], request["nodes"], policy=request["policy"], max_candidates=request["max_candidates"])
    result = {"schema": RESULT_SCHEMA, "request_sha256": canonical_digest(request), "profile_sha256": canonical_digest(profile),
              "profile_source_kind": profile["source_kind"], "binding": binding, "search": search,
              "production_admitted": False, "measured_product_benefit_proven": False,
              "baseline_estimate": None, "same_node_policy_best": None, "latency_only_best": None,
              "rebalance": None, "missing_profile_identities": []}
    if not search["complete"] or not search["baseline"]["admitted"]:
        result["status"] = search["reason_code"]
        return result
    required = {a["profile_identity"] for c in search["candidates"] for a in c["assignments"]}
    unknown = set(indexed) - required
    if unknown:
        raise PlacementStudyError("profile contains assignments outside admitted candidate space")
    result["missing_profile_identities"] = sorted(required - set(indexed))
    baseline_id = _candidate(search["baseline"])["candidate_id"]
    estimates = []
    for candidate in search["candidates"]:
        identities = [a["profile_identity"] for a in candidate["assignments"]]
        if any(identity not in indexed for identity in identities):
            continue
        stages = [deepcopy(indexed[identity]) for identity in identities]
        estimate = {"candidate_id": candidate["candidate_id"], "node_count": candidate["node_count"], "stages": stages,
                    "serial_step_median_sum_ms": _number(sum(s["paired_total_median_ms"] for s in stages), "serial step", positive=True)}
        estimates.append(estimate)
        if candidate["candidate_id"] == baseline_id:
            result["baseline_estimate"] = estimate
    if result["missing_profile_identities"]:
        result["status"] = "profile_incomplete"
        return result
    baseline = result["baseline_estimate"]
    count = baseline["node_count"]
    rank = lambda e: (e["serial_step_median_sum_ms"], e["candidate_id"] != baseline_id, e["candidate_id"])
    result["same_node_policy_best"] = min((e for e in estimates if e["node_count"] == count), key=rank)
    result["latency_only_best"] = min(estimates, key=rank)
    result["latency_only_changes_node_count_policy"] = result["latency_only_best"]["node_count"] != count
    result["same_policy_predicted_gain_ms"] = baseline["serial_step_median_sum_ms"] - result["same_node_policy_best"]["serial_step_median_sum_ms"]
    if request["transition"] is not None:
        result["rebalance"] = {**assess_cold_rebalance(baseline["serial_step_median_sum_ms"], result["same_node_policy_best"]["serial_step_median_sum_ms"], request["transition"]),
                               "from_candidate_id": baseline_id, "to_candidate_id": result["same_node_policy_best"]["candidate_id"]}
    result["status"] = "evaluated"
    return result
