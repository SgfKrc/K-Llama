"""EX-N3 executor contracts: frozen rubric, quality gate, and calibration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_core.collector import build_record
from experiment_core.objective_rubric import (
    RubricError,
    load_objective_rubric,
    score_objective_outputs,
)
from experiment_core.plan import PlanError, load_plan
from experiment_core.runner import UnitOutcome


PROMPT_SET = {
    "id": "ps-v1-zh-en-code",
    "sha256": "8cc555f57fa23d45c16820f77cc90b507da04578a1e90eebf46e19d4eb2568a3",
}
RUBRIC_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "quality_rubrics" / "llm-objective-ps-v1-v1.json"
)
RUBRIC_SHA256 = "5a4aafafdac73937077df2bb2378b4b15fa5320ce7b6adb83fb90f44ad60f924"
OBJECTIVE_IDS = [
    "math-001", "math-002", "math-004", "math-005", "en-001",
    "code-001", "code-002", "code-003", "code-004", "code-005",
    "fmt-001", "fmt-002", "fmt-003", "fmt-004", "fmt-005",
]


def _quality_plan(tmp_path: Path, *, required: bool = False, baseline: str | None = None):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "plan_id": "plan-ex-n3-test-v1",
        "prompt_set": PROMPT_SET,
        "quality": {
            "required": required,
            "manual_review": {
                "reviewers_required": 2, "upgrade_on": "2 pass, 0 fail",
            },
            "llm": {
                "prompt_set_id": PROMPT_SET["id"],
                "prompt_set_sha256": PROMPT_SET["sha256"],
                "objective_subset_count": len(OBJECTIVE_IDS),
                "objective_prompt_ids": OBJECTIVE_IDS,
                "rubric_id": "llm-objective-ps-v1-v1",
                "rubric_sha256": RUBRIC_SHA256,
                "correctness_rate_baseline": 0.6,
                "format_rate_baseline": 0.9,
                "compare_rule": ">= baseline*0.9",
            },
            "calibration": {
                "series_id": "ex-n3-test-series-v1",
                "rounds_required": 3,
                "threshold_version": "ex-n3-baseline-v1",
            },
        },
        "units": [{
            "experiment_id": "exp-0001", "name": "quality test",
            "command": [sys.executable, "-c", "pass"], "resources": {},
            "params": {}, "model": {}, "runs": 1, "timeout_s": 10,
            "gate": {"metric": "decode_tok_s", "threshold": 1},
            "baseline_experiment_id": baseline,
            "quality_checks": ["llm"],
        }],
    }, ensure_ascii=False), encoding="utf-8")
    return load_plan(path)


def _evidence(*, correct: tuple[int, int], formatting: tuple[int, int]) -> dict:
    return {
        "llm": {
            "prompt_set_id": PROMPT_SET["id"],
            "prompt_set_sha256": PROMPT_SET["sha256"],
            "correctness": {"evaluated_count": correct[0], "passed_count": correct[1]},
            "format": {"evaluated_count": formatting[0], "passed_count": formatting[1]},
        }
    }


def _record(plan, *, evidence: dict, records: dict | None = None):
    outcome = UnitOutcome(
        experiment_id="exp-0001", status="passed", exit_code=0, duration_s=0.01,
        metrics={"decode_tok_s": 2.0, "quality_evidence": evidence},
    )
    return build_record(
        plan, plan.units[0], outcome,
        prompt_set_dir=plan.verify_prompt_set(), records=records or {},
    )


def test_objective_rubric_scores_without_returning_completion_text():
    rubric = load_objective_rubric(
        RUBRIC_PATH, expected_sha256=RUBRIC_SHA256, expected_prompt_set=PROMPT_SET,
    )
    outputs = {
        "math-001": "They meet at 13:54.",
        "math-002": "160 widgets",
        "math-004": "3 weighings",
        "math-005": "x = 1/2",
        "en-001": "TCP is reliable; UDP is lower overhead.",
        "code-001": "def valid_ipv4(value):\n    return True",
        "code-002": "def merge(a, b):\n    return a + b",
        "code-003": "def prefix(items):\n    return ''",
        "code-004": "def even_sum(items):\n    return sum(items)",
        "code-005": "SELECT * FROM x ORDER BY amount DESC",
        "fmt-001": '{"name":"x","age":1,"tags":["a"]}',
        "fmt-002": "January February March April May June",
        "fmt-003": "{'a': 1, 'b': 2, 'c': 3}",
        "fmt-004": "1. one\n2. two\n3. three",
        "fmt-005": "id,name,score\n1,a,1\n2,b,2",
    }
    counters = score_objective_outputs(rubric, outputs)
    assert counters["correctness"] == {"evaluated_count": 4, "passed_count": 4, "invalid_count": 0}
    assert counters["format"] == {"evaluated_count": 11, "passed_count": 11, "invalid_count": 0}
    assert "They meet" not in json.dumps(counters)


def test_objective_rubric_hash_mismatch_is_rejected():
    with pytest.raises(RubricError, match="SHA-256"):
        load_objective_rubric(RUBRIC_PATH, expected_sha256="0" * 64, expected_prompt_set=PROMPT_SET)


def test_plan_rejects_quality_rubric_or_objective_subset_drift(tmp_path):
    path = tmp_path / "bad-plan.json"
    raw = {
        "plan_id": "bad-quality-v1", "prompt_set": PROMPT_SET,
        "quality": {
            "required": False,
            "llm": {
                "prompt_set_id": PROMPT_SET["id"], "prompt_set_sha256": PROMPT_SET["sha256"],
                "objective_subset_count": 1, "objective_prompt_ids": ["does-not-exist"],
                "rubric_id": "llm-objective-ps-v1-v1", "rubric_sha256": RUBRIC_SHA256,
                "correctness_rate_baseline": 0.6, "format_rate_baseline": 0.9,
                "compare_rule": ">= baseline*0.9",
            },
        },
        "units": [{
            "experiment_id": "exp-0001", "name": "bad", "command": ["x"],
            "resources": {}, "params": {}, "model": {}, "quality_checks": ["llm"],
        }],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(path)
    with pytest.raises(PlanError, match="objective prompt"):
        plan.verify_prompt_set()


def test_correctness_floor_zero_allowed_but_weak_format_rejected_as_required(tmp_path):
    # P5/v2 (2026-08-16): correctness 0.0 = not participating; format 0.0 is
    # also allowed (v2 512-token budget disables the format gate). A format
    # floor in (0, 0.30) stays rejected.
    path = _quality_plan(tmp_path, required=False).source_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["quality"]["required"] = True
    raw["quality"]["llm"]["correctness_rate_baseline"] = 0.0
    raw["quality"]["llm"]["format_rate_baseline"] = 0.0
    raw["quality"]["manual_review"] = {"reviewers_required": 2, "upgrade_on": "2 pass, 0 fail"}
    raw["quality"]["calibration"] = {"series_id": "ex-n3-test-series-v1", "rounds_required": 3, "threshold_version": "v1"}
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(path)  # allowed: both 0 (not participating)
    assert plan.quality.required is True

    raw["quality"]["llm"]["format_rate_baseline"] = 0.20
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PlanError, match="format floor"):
        load_plan(path)


def test_required_correctness_floor_uses_v2_approved_0_15(tmp_path):
    # P5/v2 (2026-08-16): non-zero correctness floor must be >= 0.15
    # (Qwen3-4B v2 three-round calibration); weaker gates are rejected.
    path = _quality_plan(tmp_path, required=False).source_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["quality"]["required"] = True
    raw["quality"]["llm"]["correctness_rate_baseline"] = 0.15
    raw["quality"]["manual_review"] = {"reviewers_required": 2, "upgrade_on": "2 pass, 0 fail"}
    raw["quality"]["calibration"] = {"series_id": "ex-n3-test-series-v1", "rounds_required": 3, "threshold_version": "v1"}
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(path)
    assert plan.quality.required is True

    raw["quality"]["llm"]["correctness_rate_baseline"] = 0.10
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PlanError, match="correctness floor"):
        load_plan(path)


def test_gemma_judge_required_requires_approved_baselines(tmp_path):
    # Decision 2026-08-13: gemma_judge is eligible for required=true with the
    # approved 0.70/0.40 floors; weaker baselines stay rejected.
    path = _quality_plan(tmp_path, required=False).source_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["quality"]["required"] = True
    raw["quality"]["manual_review"] = {"reviewers_required": 2, "upgrade_on": "2 pass, 0 fail"}
    raw["quality"]["calibration"] = {"series_id": "ex-n3-test-series-v1", "rounds_required": 3, "threshold_version": "v1"}
    raw["quality"]["gemma_judge"] = {
        "model": "gemma4:12b",
        "judge_contract_id": "gemma-judge-counts-v1",
        "judge_contract_sha256": "be7bcea3e736e0009c2ff3e110f54309263a960e2b6ae892e3c5de7a302e4374",
        "topic_hit_rate_baseline": 0.70,
        "key_element_coverage_baseline": 0.40,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(path)
    assert plan.quality.required is True

    raw["quality"]["gemma_judge"]["topic_hit_rate_baseline"] = 0.60
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PlanError, match="approved"):
        load_plan(path)


def test_quality_gate_isolated_by_default_but_required_can_fail_total_gate(tmp_path):
    isolated = _quality_plan(tmp_path, required=False)
    record = _record(isolated, evidence=_evidence(correct=(4, 1), formatting=(11, 8)))
    assert record["performance_gate"]["status"] == "passed"
    assert record["quality_gate"]["status"] == "failed"
    assert record["gate"]["status"] == "passed"

    required = _quality_plan(tmp_path, required=True)
    record = _record(required, evidence=_evidence(correct=(4, 1), formatting=(11, 8)))
    assert record["performance_gate"]["status"] == "passed"
    assert record["quality_gate"]["status"] == "failed"
    assert record["gate"]["status"] == "failed"


def test_quality_compare_rule_uses_baseline_rates(tmp_path):
    plan = _quality_plan(tmp_path, required=False, baseline="exp-0000")
    baseline = {
        "quality": {
            "llm": {
                "correctness": {"rate": 1.0},
                "format": {"rate": 1.0},
            }
        },
        "metrics": {"decode_tok_s": 2.0},
    }
    record = _record(
        plan, evidence=_evidence(correct=(4, 3), formatting=(11, 10)),
        records={"exp-0000": baseline},
    )
    assert record["quality_gate"]["status"] == "failed"
    assert "baseline*0.9" in " ".join(record["quality_gate"]["criteria"])


@pytest.mark.skipif(
    not (SCRIPTS_DIR / "experiment_quality_calibrate.py").exists(),
    reason="缺 scripts/experiment_quality_calibrate.py（scripts 白名单未放行的实验脚本）",
)
def test_calibration_summarizer_requires_three_rounds_and_never_edits_plan(tmp_path):
    records = tmp_path / "records.jsonl"
    series = "ex-n3-test-series-v1"
    rows = []
    for index, value in enumerate((0.6, 0.8, 0.7), start=1):
        rows.append({
            "experiment_id": f"exp-{index:04d}",
            "timestamp": f"2026-08-12T00:00:0{index}+00:00",
            "calibration": {"series_id": series},
            "quality": {"llm": {"correctness": {"rate": value}, "format": {"rate": 0.9}}},
        })
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "calibration.json"
    completed = subprocess.run([
        sys.executable, str(SCRIPTS_DIR / "experiment_quality_calibrate.py"),
        "--records", str(records), "--series-id", series, "--output", str(output),
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["correctness"] == {
        "count": 3, "minimum": 0.6, "maximum": 0.8,
        "median": 0.7, "variation_margin": pytest.approx(0.1),
        "suggested_floor": pytest.approx(0.6),
    }
    assert "no threshold was changed automatically" in summary["next_action"]

    older_duplicate = json.loads(json.dumps(rows[-1]))
    older_duplicate["timestamp"] = "2026-08-11T00:00:00+00:00"
    older_duplicate["quality"]["llm"]["correctness"]["rate"] = 0.1
    records.write_text(
        records.read_text(encoding="utf-8") + json.dumps(older_duplicate) + "\n",
        encoding="utf-8",
    )
    rejected = subprocess.run([
        sys.executable, str(SCRIPTS_DIR / "experiment_quality_calibrate.py"),
        "--records", str(records), "--series-id", series, "--output", str(output),
    ], capture_output=True, text=True, check=False)
    assert rejected.returncode == 1
    recovered = subprocess.run([
        sys.executable, str(SCRIPTS_DIR / "experiment_quality_calibrate.py"),
        "--records", str(records), "--series-id", series, "--output", str(output),
        "--select-earliest-unique",
    ], capture_output=True, text=True, check=False)
    assert recovered.returncode == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["duplicate_recovery"] == {
        "mode": "earliest_per_experiment_id",
        "duplicates": [{"experiment_id": "exp-0003", "observed_records": 2}],
    }
    assert summary["correctness"]["minimum"] == 0.1


def test_required_plan_must_declare_manual_review_and_calibration(tmp_path):
    """KIP-16：required 必须带 manual_review 与三轮 calibration。"""
    path = _quality_plan(tmp_path, required=False).source_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["quality"]["required"] = True
    raw["quality"].pop("manual_review")
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PlanError, match="manual_review"):
        load_plan(path)

    raw["quality"]["manual_review"] = {"reviewers_required": 2, "upgrade_on": "2 pass, 0 fail"}
    raw["quality"].pop("calibration")
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PlanError, match="rounds_required"):
        load_plan(path)


def test_required_gemma_gate_requires_manual_review_binding(tmp_path):
    """KIP-16：required 时 Gemma 判题证据缺人工复核绑定 → quality failed。"""
    import scripts.experiment_core.collector as collector_mod
    from experiment_core.plan import load_plan
    from experiment_core.quality import normalize_quality_evidence

    path = _quality_plan(tmp_path, required=False).source_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["quality"]["required"] = True
    raw["quality"]["manual_review"] = {"reviewers_required": 2, "upgrade_on": "2 pass, 0 fail"}
    raw["quality"]["calibration"] = {"series_id": "s", "rounds_required": 3, "threshold_version": "v1"}
    raw["quality"]["gemma_judge"] = {
        "model": "gemma4:12b",
        "judge_contract_id": "gemma-judge-counts-v1",
        "judge_contract_sha256": "be7bcea3e736e0009c2ff3e110f54309263a960e2b6ae892e3c5de7a302e4374",
        "topic_hit_rate_baseline": 0.70,
        "key_element_coverage_baseline": 0.40,
    }
    raw["units"][0]["quality_checks"] = ["gemma_judge"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(path)

    evidence = normalize_quality_evidence({"gemma_judge": {
        "model": "gemma4:12b",
        "judge_contract_id": "gemma-judge-counts-v1",
        "judge_contract_sha256": "be7bcea3e736e0009c2ff3e110f54309263a960e2b6ae892e3c5de7a302e4374",
        "topic_hit": {"evaluated_count": 10, "passed_count": 8},
        "key_element_coverage": {"evaluated_count": 20, "passed_count": 12},
    }}, expected_gemma_judge=plan.quality.gemma_judge)
    gate = collector_mod.evaluate_quality_gate(
        evidence, plan.quality, ("gemma_judge",), baseline_record=None,
    )
    assert gate["status"] == "failed"

    # 带人工复核绑定后通过
    evidence2 = normalize_quality_evidence({"gemma_judge": {
        "model": "gemma4:12b",
        "judge_contract_id": "gemma-judge-counts-v1",
        "judge_contract_sha256": "be7bcea3e736e0009c2ff3e110f54309263a960e2b6ae892e3c5de7a302e4374",
        "topic_hit": {"evaluated_count": 10, "passed_count": 8},
        "key_element_coverage": {"evaluated_count": 20, "passed_count": 12},
        "manual_review": {"status": "passed", "required_reviewers": 2},
    }}, expected_gemma_judge=plan.quality.gemma_judge)
    gate2 = collector_mod.evaluate_quality_gate(
        evidence2, plan.quality, ("gemma_judge",), baseline_record=None,
    )
    assert gate2["status"] == "passed"


def test_qwen3_v2_calibration_plan_is_pinned_and_executable():
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures" / "experiment-plans"
        / "plan-quality-qwen3-4b-calibration-v2.json"
    )
    plan = load_plan(plan_path)

    assert plan.prompt_set == {
        "id": "ps-qwen3-nothink",
        "sha256": "c55caa2a1a5348924624c6aadec912eb4808aff7468b6b21c0f13c0a14d554a9",
    }
    assert plan.verify_prompt_set().is_dir()
    assert plan.quality is not None
    assert plan.quality.llm["rubric_id"] == "llm-objective-ps-v1-v2"
    assert plan.quality.calibration["rounds_required"] == 3
    assert len(plan.units) == 3

    for unit in plan.units:
        command = " ".join(unit.command)
        assert "models/qwen3-4b-gguf/Qwen3-4B-Q4_K_M.gguf" in command
        assert "--prompt-mode raw" in command
        assert "--max-new-tokens 512" in command
        assert "--temperature 0.7" in command
        assert unit.params["max_new_tokens"] == 512
        assert unit.quality_checks == ("llm",)
