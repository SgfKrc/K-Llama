"""MF-MEM-N1 pure logic and frozen experiment matrix tests."""

# Owned by the retired MF-MEM benchmark workspace; its one-off harness was
# removed from the main repo in aa889bcb (see .gitignore, scripts policy).
# Legacy local run: QLH_INCLUDE_WORKSPACE_TOOL_TESTS=1 python -m pytest <this file>.
# Opting in requires the real harness; keep missing imports fatal.

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_fleet_memory_transport import (
    FIXED_TILE_MIB,
    MODE_SLOTS,
    build_parser,
    percentile,
    timeline_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    PROJECT_ROOT
    / "fixtures"
    / "experiment-plans"
    / "plan-model-fleet-memory-transport-v1.json"
)


def test_percentile_uses_linear_interpolation():
    values = [1.0, 2.0, 10.0, 20.0]
    assert percentile(values, 50.0) == pytest.approx(6.0)
    assert percentile(values, 95.0) == pytest.approx(18.5)


def test_timeline_metrics_report_overlap_and_steady_stall():
    metrics = timeline_metrics(
        [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)],
        [(4.0, 10.0), (10.0, 16.0), (16.0, 22.0)],
    )
    assert metrics["copy_total_ms"] == pytest.approx(12.0)
    assert metrics["compute_total_ms"] == pytest.approx(18.0)
    assert metrics["overlap_ms"] == pytest.approx(8.0)
    assert metrics["effective_overlap_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["initial_fill_stall_ms"] == pytest.approx(4.0)
    assert metrics["stall_p50_ms"] == pytest.approx(0.0)
    assert metrics["stall_p95_ms"] == pytest.approx(0.0)


def test_timeline_metrics_capture_copy_limited_stalls():
    metrics = timeline_metrics(
        [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)],
        [(5.0, 7.0), (10.0, 12.0), (15.0, 17.0)],
    )
    assert metrics["initial_fill_stall_ms"] == pytest.approx(5.0)
    assert metrics["stall_p50_ms"] == pytest.approx(3.0)
    assert metrics["stall_p95_ms"] == pytest.approx(3.0)
    assert metrics["stall_max_ms"] == pytest.approx(3.0)


def test_timeline_metrics_allow_prefilled_pipeline_shape():
    metrics = timeline_metrics(
        [(1.0, 5.0), (7.0, 11.0)],
        [(0.0, 7.0), (7.0, 13.0), (13.0, 19.0)],
    )
    assert metrics["overlap_ms"] == pytest.approx(8.0)
    assert metrics["stall_p50_ms"] == pytest.approx(0.0)


def test_cli_rejects_non_frozen_tile_size():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--tile-mib", "128",
            "--mode", "pageable",
            "--result-file", "result.json",
        ])


def test_experiment_plan_covers_exact_fixed_matrix():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    units = plan["units"]
    matrix = {
        (
            unit["params"]["independent_variable"]["tile_mib"],
            unit["params"]["independent_variable"]["mode"],
        )
        for unit in units
    }
    assert len(units) == len(FIXED_TILE_MIB) * len(MODE_SLOTS) == 12
    assert matrix == {
        (tile_mib, mode)
        for tile_mib in FIXED_TILE_MIB
        for mode in MODE_SLOTS
    }
    assert all(unit["resources"] if "resources" in unit else plan["defaults"]["resources"] for unit in units)
    assert all(unit["gate"] == {
        "metric": "completed", "op": "==", "threshold": 1,
    } for unit in units)
