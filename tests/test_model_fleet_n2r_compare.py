"""MF-MEM-N2R comparison harness pure logic tests."""

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
    MODEL_INDEX,
    build_input_batch,
    fixed_weight_bytes,
    install_explicit_layer_hooks,
)


def test_fixed_weight_index_matches_7b_artifact():
    assert fixed_weight_bytes(MODEL_INDEX) == 15231233024


def test_input_batch_is_fixed_for_context_and_throughput_batch():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    result = build_input_batch(torch, Tokenizer(), context=128, batch_size=4)
    assert tuple(result["input_ids"].shape) == (4, 128)
    assert tuple(result["attention_mask"].shape) == (4, 128)


def test_input_batch_rejects_unplanned_context_and_batch():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

    with pytest.raises(ValueError):
        build_input_batch(torch, Tokenizer(), context=512, batch_size=1)
    with pytest.raises(ValueError):
        build_input_batch(torch, Tokenizer(), context=128, batch_size=2)


def test_result_contract_is_json_serializable():
    payload = {"ticket": "MF-MEM-N2R", "resource_rejected": 1}
    json.dumps(payload)


def test_explicit_hook_requires_decoder_layers():
    torch = pytest.importorskip("torch")

    class Model:
        class Inner:
            layers = []
        model = Inner()

    with pytest.raises(RuntimeError, match="decoder layers"):
        install_explicit_layer_hooks(torch, Model())


def test_real_compare_plan_covers_baselines_and_fail_closed_modes():
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "experiment-plans"
        / "plan-model-fleet-n2r-real-compare-v1.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["units"]) == 6
    modes = {unit["params"]["independent_variable"]["mode"] for unit in plan["units"]}
    assert modes == {"resident-nf4", "device-map-auto", "cpu-only", "explicit-layer-pager"}
    assert all(unit["params"]["independent_variable"]["context"] in {128, 2048} for unit in plan["units"])
