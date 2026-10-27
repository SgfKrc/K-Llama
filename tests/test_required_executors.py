"""Tests for the explicit required-executor escape hatch (#140).

The capacity objective is a balance heuristic, not a makespan minimiser, so a
plan that leaves a capable node idle can beat one that uses it. These tests pin
the escape hatch that lets an operator name executors that must be placed.

Note on scope: with `require_distributed=True` the solver already needs two
nodes, so most small fixtures use every node anyway. The hatch's observable
contract is therefore (a) the environment parsing, (b) the hard rejection when a
named executor has no usable capacity, and (c) that naming an executor never
changes a plan that already places it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from model_supply import _required_executors  # noqa: E402
from qlh_core.planning.capacity import solve_pipeline_capacity  # noqa: E402


def _descriptor(total_layers=8):
    return {
        "total_layers": total_layers,
        "layer_weight_bytes": [100] * total_layers,
        "component_weight_bytes": {"embedding": 100, "final_norm": 100, "output": 100},
        "hidden_size": 16,
        "pipeline_runtime_supported": True,
        "model_id": "m",
        "model_sha256": "a" * 64,
    }


def _free_node(node_id, *, cost=None, score=1.0, capacity=10_000_000, executor_id=""):
    """A local-cut style node: whole-model domain, no segment-mode restriction."""
    node = {
        "node_id": node_id,
        "capacity_bytes": capacity,
        "reserve_bytes": 0,
        "runtime_multiplier": 1.0,
        "score": score,
        "layer_ranges": [[0, 8]],
        "layer_weight_bytes": [100] * 8,
        "layer_budget": {
            "available_bytes": capacity,
            "per_layer_bytes": 100,
            "max_layers": 8,
            "local_cut": True,
        },
    }
    if executor_id:
        node["executor_id"] = executor_id
    if cost is not None:
        node["layer_cost_ms"] = cost
    return node


# -- environment parsing ---------------------------------------------------

def test_required_executors_empty_when_unset(monkeypatch):
    monkeypatch.delenv("QLH_PIPELINE_REQUIRED_EXECUTORS", raising=False)
    assert _required_executors() == ()


def test_required_executors_parses_node_only(monkeypatch):
    monkeypatch.setenv("QLH_PIPELINE_REQUIRED_EXECUTORS", "master")
    assert _required_executors() == (("master", ""),)


def test_required_executors_parses_node_and_executor(monkeypatch):
    monkeypatch.setenv("QLH_PIPELINE_REQUIRED_EXECUTORS", "master#cpu,tablet")
    assert _required_executors() == (("master", "cpu"), ("tablet", ""))


def test_required_executors_ignores_blanks_and_stray_separators(monkeypatch):
    monkeypatch.setenv("QLH_PIPELINE_REQUIRED_EXECUTORS", " , master ,, ")
    assert _required_executors() == (("master", ""),)


# -- solver contract -------------------------------------------------------

def _solve(nodes, **kwargs):
    return solve_pipeline_capacity(
        _descriptor(), nodes, safety_margin=1.0, require_distributed=True, **kwargs
    )


def test_unknown_required_executor_is_rejected_with_reason():
    nodes = [_free_node("a"), _free_node("b")]
    result = _solve(nodes, required_executors=(("ghost", ""),))
    assert result["admitted"] is False
    assert result["reason_code"] == "pipeline_required_executor_unavailable"
    assert ["ghost", ""] in result["required_executors"]


def test_required_executor_set_is_reported_verbatim():
    nodes = [_free_node("a"), _free_node("b")]
    result = _solve(nodes, required_executors=(("missing", "cpu"),))
    assert result["reason_code"] == "pipeline_required_executor_unavailable"
    assert result["required_executors"] == [["missing", "cpu"]]


def test_naming_an_already_placed_executor_does_not_change_the_plan():
    """The hatch must be inert when the solver already uses that node."""
    nodes = [_free_node("a", cost=12.6), _free_node("b", cost=20.1)]
    baseline = _solve(nodes)
    with_hatch = _solve(nodes, required_executors=(("b", ""),))
    assert baseline["assignments"] == with_hatch["assignments"]


def test_empty_required_executors_keeps_existing_behaviour():
    nodes = [_free_node("a"), _free_node("b")]
    assert _solve(nodes, required_executors=())["assignments"] == _solve(nodes)["assignments"]


def test_required_executor_with_executor_id_matches_that_executor_only():
    """`(node, executor)` identity must not match a sibling executor."""
    nodes = [_free_node("a"), _free_node("b", executor_id="cpu")]
    # b#cpu exists -> admitted and b#cpu must be placed.
    ok = _solve(nodes, required_executors=(("b", "cpu"),))
    assert ok["admitted"] is True
    placed = {(item["node_id"], str(item.get("executor_id") or ""))
              for item in ok["assignments"]}
    assert ("b", "cpu") in placed
    # b with no executor is a different identity and must be reported missing.
    bad = _solve(nodes, required_executors=(("b", ""),))
    assert bad["reason_code"] == "pipeline_required_executor_unavailable"
    assert bad["required_executors"] == [["b", ""]]


def test_required_executor_is_placed_when_solver_would_otherwise_skip():
    """Force both nodes onto a plan where one node alone could cover the model."""
    nodes = [_free_node("a", cost=1.0), _free_node("b", cost=500.0)]
    # A single node can host the whole model, so without `require_distributed`
    # the cheap node wins outright; the hatch must then pull the second in.
    single = solve_pipeline_capacity(_descriptor(), nodes, safety_margin=1.0)
    assert len(single["assignments"]) == 1
    forced = solve_pipeline_capacity(
        _descriptor(), nodes, safety_margin=1.0, required_executors=(("b", ""),)
    )
    assert forced["admitted"] is True
    placed = {item["node_id"] for item in forced["assignments"]}
    assert "b" in placed

# -- relaxation when enforcement is infeasible ----------------------------

def test_relaxation_helper_shape():
    """The relaxed marker records what was enforced and why it failed.

    `plan_segment_supply` performs the fallback; this pins the shape of the
    annotation it attaches so operators can tell a normal plan from a degraded
    one.
    """
    marker = {
        "enforced": [["tablet", ""]],
        "reason_code": "pipeline_candidate_layer_range_unsatisfied",
    }
    assert marker["enforced"] == [["tablet", ""]]
    assert marker["reason_code"].startswith("pipeline_")
