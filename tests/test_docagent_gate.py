from __future__ import annotations

# Owned by the independent qlh-docagent workspace, not the main repository CI.
# Legacy local run: QLH_INCLUDE_WORKSPACE_TOOL_TESTS=1 python -m pytest <this file>.
# Keep imports strict: opting in requires the actual workspace tool sources.

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "docagent" / "run.py"
RULES = ROOT / "tools" / "docagent" / "docagent" / "data" / "rules.yaml"
sys.path.insert(0, str(ROOT / "tools" / "docagent"))

from docagent.events import EVENTS_SCHEMA_VERSION, DocEventStore  # noqa: E402
from docagent.gate import (  # noqa: E402
    GATE_SCHEMA_VERSION,
    GateMismatch,
    build_gate_record,
    report_fingerprint,
    validate_report,
    verify_gate,
    write_gate,
)
from docagent.report import render_json  # noqa: E402
from docagent.scanner import scan_repository  # noqa: E402


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "docagent@test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "docagent"], check=True)
    (docs / "status.md").write_text("> Status: current\n\n[broken](missing.md)\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
    return repo


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_scan_report_has_self_fingerprint_and_rejects_tampering(tmp_path: Path):
    report = scan_repository(_repo(tmp_path))

    assert len(report["report_fingerprint"]) == 64
    assert report["report_fingerprint"] == report_fingerprint(report)
    assert validate_report(report)["report_fingerprint"] == report["report_fingerprint"]

    tampered = copy.deepcopy(report)
    tampered["docs"][0]["findings"][0]["message"] = "changed after scan"
    with pytest.raises(GateMismatch, match="report fingerprint mismatch"):
        validate_report(tampered)


def test_gate_artifact_binds_report_and_current_rules(tmp_path: Path):
    repo = _repo(tmp_path)
    report = scan_repository(repo)
    report_path = tmp_path / "report.json"
    gate_path = tmp_path / "gate.json"
    report_path.write_text(render_json(report), encoding="utf-8")
    gate = build_gate_record(report)
    write_gate(gate_path, gate)

    result = verify_gate(report_path, RULES, profile="qlh", gate_path=gate_path)

    assert gate["schema_version"] == GATE_SCHEMA_VERSION
    assert result["status"] == "passed"
    assert result["checks"]["gate"] is True

    changed_rules = tmp_path / "rules-changed.json"
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    next(rule for rule in rules["rules"] if rule["id"] == "R4")["level"] = "error"
    changed_rules.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(GateMismatch, match="rules fingerprint mismatch"):
        verify_gate(report_path, changed_rules, profile="qlh", gate_path=gate_path)


def test_optional_event_adapter_records_path_free_gate_event(tmp_path: Path):
    report = scan_repository(_repo(tmp_path))
    gate = build_gate_record(report)
    database = tmp_path / "events.sqlite"

    with DocEventStore(database) as store:
        event = store.record(report, kind="gate", gate=gate)
        recent = store.recent_events()

    assert event["report_fingerprint"] == report["report_fingerprint"]
    assert recent[0]["kind"] == "gate"
    assert recent[0]["payload"]["schema_version"] == EVENTS_SCHEMA_VERSION
    assert str(tmp_path) not in json.dumps(recent, ensure_ascii=False)


def test_gate_rescan_rebuilds_a_consistent_zero_delta_after_rollback(tmp_path: Path):
    repo = _repo(tmp_path)
    baseline = tmp_path / "baseline.json"
    locked = _run(
        "audit", "--root", str(repo), "--lock", str(baseline), "--fail-on", "none", cwd=tmp_path,
    )
    assert locked.returncode == 0, locked.stderr

    rescanned = _run(
        "gate", "rescan", "--root", str(repo), "--baseline", str(baseline),
        "--json", "--fail-on", "none", cwd=tmp_path,
    )

    assert rescanned.returncode == 0, rescanned.stderr
    assert json.loads(rescanned.stdout)["summary"] == {
        "new": 0,
        "gone": 0,
        "changed": 0,
        "affected_docs": 0,
        "new_documents": 0,
        "gone_documents": 0,
        "changed_documents": 0,
    }
