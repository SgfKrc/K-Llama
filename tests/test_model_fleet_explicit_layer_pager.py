"""MF-MEM-N2 pager lifecycle and correctness tests."""

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

from model_fleet_explicit_layer_pager import (  # noqa: E402
    FIXED_SLOTS,
    FIXED_TILE_MIB,
    ExplicitLayerPager,
    PagerConfig,
    build_parser,
    make_synthetic_layers,
    resident_reference,
)


def test_n2_contract_rejects_non_fixed_tile_and_triple_slots():
    with pytest.raises(ValueError, match="tile_mib"):
        ExplicitLayerPager(PagerConfig(tile_mib=128, backend="cpu"))
    with pytest.raises(ValueError, match="slots"):
        ExplicitLayerPager(PagerConfig(slots=3, backend="cpu"))


def test_cpu_pager_matches_resident_reference_and_releases():
    import torch

    layers = make_synthetic_layers(torch, 3, FIXED_TILE_MIB * 1024 * 1024 // 4)
    reference = resident_reference(layers, 1, torch)
    pager = ExplicitLayerPager(PagerConfig(backend="cpu"))
    result = pager.run(layers, batch_size=1)
    release = pager.close()

    assert result["status"] == "completed"
    assert result["completed"] == 1
    assert result["slots"] == FIXED_SLOTS == 2
    assert result["output_digest"] == reference["digest"]
    assert release["released_vram_allocated_bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        pager.run(layers)


def test_cpu_pager_cancels_before_next_layer_and_reclaims():
    import torch

    layers = make_synthetic_layers(torch, 4, FIXED_TILE_MIB * 1024 * 1024 // 4)
    pager = ExplicitLayerPager(PagerConfig(backend="cpu"))
    result = pager.run(layers, cancel_after_layer=2)
    release = pager.close()

    assert result["status"] == "cancelled"
    assert result["cancelled"] == 1
    assert result["completed"] == 0
    assert result["completed_layers"] == 2
    assert release["released_vram_allocated_bytes"] == 0


def test_cli_contract_keeps_fixed_tile_choice():
    parser = build_parser()
    args = parser.parse_args(["--backend", "cpu", "--result-file", "result.json"])
    assert args.backend == "cpu"
    assert FIXED_TILE_MIB == 64
    assert FIXED_SLOTS == 2


@pytest.mark.requires_gpu
def test_cuda_pager_matches_reference_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layers = make_synthetic_layers(torch, 2, FIXED_TILE_MIB * 1024 * 1024 // 4)
    reference = resident_reference(layers, 1, torch)
    pager = ExplicitLayerPager(PagerConfig(backend="cuda"))
    try:
        result = pager.run(layers)
        assert result["status"] == "completed"
        assert result["output_digest"] == reference["digest"]
        assert result["peak_pinned_bytes"] == FIXED_TILE_MIB * 1024 * 1024 * 2
    finally:
        pager.close()


def test_experiment_plan_declares_batch_and_cancel_units():
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "experiment-plans"
        / "plan-model-fleet-explicit-pager-v1.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["units"]) == 4
    params = {unit["params"]["independent_variable"]["case"] for unit in plan["units"]}
    assert params == {"batch1", "throughput-batch4", "cancel", "cpu-control"}
    assert all(unit["params"]["independent_variable"]["slots"] == 2 for unit in plan["units"])
