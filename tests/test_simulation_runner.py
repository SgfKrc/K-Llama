import json
import os
from pathlib import Path

import pytest

from scripts import run_simulation

_VENV_TEST = Path(__file__).resolve().parents[1] / ".venv-test"


def test_simulation_runner_profiles_are_fixed_and_layered():
    assert run_simulation.targets_for_profile("quick") == (
        "tests/test_task_graph_simulation.py",
        "tests/test_task_worker_control_simulation.py",
        "tests/test_capacity_simulation.py",
    )
    quick = run_simulation.targets_for_profile("quick")
    assert run_simulation.targets_for_profile("extended")[:len(quick)] == quick
    assert run_simulation.targets_for_profile("full") == ("tests",)


def test_simulation_runner_quick_profile_emits_safe_json(tmp_path):
    output = tmp_path / "simulation-summary.json"

    assert run_simulation.main([
        "--profile", "quick",
        "--json-output", str(output),
    ]) == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "qlh.simulation_runner.v1"
    assert summary["profile"] == "quick"
    assert summary["outcome"] == "passed"
    assert summary["pytest"]["exit_code"] == 0
    assert summary["pytest"]["counts"]["passed"] >= 23
    assert [item["family"] for item in summary["evidence"]] == [
        "task_graph",
        "task_worker_control",
        "capacity",
    ]
    assert summary["acceptance_scope"] == {
        "physical_nodes": "not_established",
        "real_models": "not_established",
        "real_network": "not_established",
        "performance": "not_established",
        "installation_package": "not_established",
    }
    serialized = json.dumps(summary, sort_keys=True)
    for field in (
        "authorization",
        "root_input",
        "blob_id",
        "content",
        "prompt",
        "transfer_plan",
        "secret",
        "token",
        "url",
    ):
        assert f'"{field}"' not in serialized
    assert "simulated visual prompt" not in serialized


def test_simulation_runner_default_stdout_is_parseable_json(capsys, monkeypatch):
    monkeypatch.setattr(run_simulation, "_run_pytest", lambda _targets: (0, "1 passed"))
    monkeypatch.setattr(run_simulation, "collect_evidence", lambda: [])

    assert run_simulation.main([]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["profile"] == "quick"
    assert summary["pytest"]["counts"]["passed"] == 1


@pytest.mark.skipif(not _VENV_TEST.exists(), reason="缺本机 .venv-test 虚拟环境（CI 用系统解释器）")
def test_pytest_python_prefers_venv_test_when_called_from_system_python(monkeypatch):
    """系统 Python 调用时优先返回 .venv-test 解释器（环境分割铁律）。"""
    monkeypatch.setattr(run_simulation.sys, "prefix", "C:/Python312")
    monkeypatch.setattr(run_simulation.sys, "base_prefix", "C:/Python312")
    resolved = run_simulation._pytest_python()
    assert "venv-test" in resolved
    assert resolved.replace("\\", "/").endswith(
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )


def test_pytest_python_keeps_current_interpreter_inside_any_venv(monkeypatch):
    """已在虚拟环境内（无论哪个）→ 保持当前解释器，不逃逸到 .venv-test。"""
    monkeypatch.setattr(run_simulation.sys, "prefix", "G:/some/venv")
    monkeypatch.setattr(run_simulation.sys, "base_prefix", "C:/Python312")
    assert run_simulation._pytest_python() == run_simulation.sys.executable


def test_pytest_python_falls_back_when_venv_test_missing(monkeypatch):
    """系统 Python 且 .venv-test 不存在 → 回退当前解释器（不崩溃）。"""
    monkeypatch.setattr(run_simulation.sys, "prefix", "C:/Python312")
    monkeypatch.setattr(run_simulation.sys, "base_prefix", "C:/Python312")
    monkeypatch.setattr(
        run_simulation.Path, "is_file", lambda _self: False, raising=False
    )
    assert run_simulation._pytest_python() == run_simulation.sys.executable
