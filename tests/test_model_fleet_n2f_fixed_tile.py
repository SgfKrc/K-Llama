"""MF-MEM-N2F fixed tile bridge contract tests."""

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

from model_fleet_n2r_compare import (  # noqa: E402
    FixedTileBridge,
    _interval_overlap_stats,
    _latency_stats,
    build_parser,
)


def test_fixed_tile_bridge_contract_and_byte_view():
    torch = pytest.importorskip("torch")

    assert FixedTileBridge.TILE_BYTES == 64 * 1024 * 1024
    assert FixedTileBridge.SLOTS == 2
    raw = FixedTileBridge._parameter_bytes(torch, torch.ones(8, dtype=torch.float32))
    assert raw.dtype == torch.uint8
    assert raw.numel() == 32


def test_n2f_parser_accepts_ticket_and_cancel_boundary():
    parser = build_parser()
    args = parser.parse_args([
        "--ticket", "MF-MEM-N2F",
        "--mode", "explicit-layer-pager-64",
        "--cancel-after-layer", "1",
        "--prefetch-distance", "1",
        "--measure-latency",
        "--result-file", "out.json",
    ])
    assert args.ticket == "MF-MEM-N2F"
    assert args.cancel_after_layer == 1
    assert args.prefetch_distance == 1
    assert args.measure_latency is True


def test_n2p_overlap_and_latency_stats_are_deterministic():
    overlap = _interval_overlap_stats([(0.0, 2.0)], [(1.0, 3.0)])
    assert overlap["overlap_ms"] == 1000.0
    assert overlap["effective_overlap_ratio"] == 0.5
    latency = _latency_stats([10.0, 20.0, 30.0, 40.0])
    assert latency["count"] == 4
    assert latency["p50_ms"] == 25.0
    assert latency["max_ms"] == 40.0


def test_n2f_plan_has_full_and_boundary_units():
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "experiment-plans"
        / "plan-model-fleet-n2f-fixed-tile-v1.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["units"]) == 3
    assert {unit["gate"]["metric"] for unit in plan["units"]} == {"completed", "cancelled"}
    assert all(unit["params"]["independent_variable"]["fixed_tile_mib"] == 64 for unit in plan["units"])
    assert all(unit["params"]["independent_variable"]["tile_slots"] == 2 for unit in plan["units"])
    assert any(unit["params"]["independent_variable"]["context"] == 8192 for unit in plan["units"])


def test_n2p_plan_covers_resident_serial_and_prefetch_distance_one():
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "experiment-plans"
        / "plan-model-fleet-n2p-latency-v1.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["units"]) == 3
    modes = {unit["params"]["independent_variable"]["mode"] for unit in plan["units"]}
    assert modes == {"resident-nf4", "explicit-layer-pager-64"}
    distances = {
        unit["params"]["independent_variable"]["prefetch_distance"]
        for unit in plan["units"]
    }
    assert distances == {0, 1}
    assert all(unit["params"]["independent_variable"]["max_new_tokens"] == 8 for unit in plan["units"])
