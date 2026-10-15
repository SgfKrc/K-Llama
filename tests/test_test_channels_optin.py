"""物理设备 / 重档用例的显式 opt-in 判据（票 13 `CI-RELEASE-01` 第三条）。

判据不是"跑了再 skip"，而是**未 opt-in 时不收集** —— 否则在有 GPU 的机器上这些用例会
真的跑起来，等于隐式 opt-in。这里直接对 `scripts/run_test_channels.py` 的判据函数断言。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_test_channels.py"

_spec = importlib.util.spec_from_file_location("run_test_channels_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None, f"无法加载 {SCRIPT}"
channels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(channels)


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(channels.PHYSICAL_OPT_IN_ENV, raising=False)
    monkeypatch.delenv(channels.LEGACY_SMOKE_ENV, raising=False)


def test_default_is_not_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """两个开关都不设时为未开启（默认姿态是"不跑物理设备"）。"""
    _clear(monkeypatch)
    assert channels._physical_opt_in() is False


def test_no_optin_excludes_every_physical_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """未 opt-in 时，external 段表达式必须显式排除全部物理/重档标记。"""
    _clear(monkeypatch)
    expression = channels._external_marker_expression(physical=False)
    for marker in channels.PHYSICAL_MARKERS:
        assert f"not {marker}" in expression, marker
    assert "external" in expression
    assert "requires_gpu" not in expression.replace("not requires_gpu", "")


def test_optin_collects_physical_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """开启后，external 段表达式改为包含物理/重档标记。

    `real_model` 在**两种状态下都**被 external 段排除 —— 它由专门的 smoke 通道承载
    （`-m real_model` + 串行加载真实权重），因此这里不断言它"出现"。
    """
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    assert channels._physical_opt_in() is True
    expression = channels._external_marker_expression(physical=True)
    for marker in ("requires_gpu", "requires_db", "slow"):
        assert marker in expression, marker
    assert "not real_model" in expression


def test_legacy_smoke_env_still_opts_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """`QLH_RUN_REAL_MODEL_SMOKE=1` 保留为向后兼容的等价开关。"""
    _clear(monkeypatch)
    monkeypatch.setenv(channels.LEGACY_SMOKE_ENV, "1")
    assert channels._physical_opt_in() is True


def test_other_values_do_not_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """只有字面 "1" 算开启；"0" / "true" / 空值都不算。"""
    for value in ("0", "true", "", "yes"):
        _clear(monkeypatch)
        monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, value)
        assert channels._physical_opt_in() is False, value


@pytest.fixture
def simulated_runner(monkeypatch, tmp_path):
    """Only fake child processes run; no physical or model test is launched."""
    import json
    import xml.etree.ElementTree as ET

    _clear(monkeypatch)
    monkeypatch.setattr(channels, "_check_python_environment", lambda **_: True)
    calls = []
    outcomes = {}
    returncodes = {}

    def fake_run(arguments, _env, *, log_path):
        calls.append(list(arguments))
        xml_path = Path(arguments[arguments.index("--junitxml") + 1])
        key = xml_path.stem
        result = outcomes.get(key, "passed")
        if result == "interrupt":
            before = json.loads((xml_path.parent / "manifest.json").read_text(encoding="utf-8"))
            assert before["state"] == "running"
            assert before["runs"][-1]["returncode"] is None
            assert before["runs"][-1]["state"] == "running"
            raise KeyboardInterrupt("simulated interruption")
        if result == "malformed":
            xml_path.write_text("<testsuite", encoding="utf-8")
        elif result != "missing":
            case_outcomes = [] if result == "empty" else [result]
            suite = ET.Element("testsuite", {
                "tests": str(len(case_outcomes)), "failures": str(int(result == "failure")),
                "errors": str(int(result == "error")), "skipped": str(int(result == "skipped")),
            })
            for number, outcome in enumerate(case_outcomes):
                case = ET.SubElement(suite, "testcase", name=f"test_{number}")
                if outcome not in ("passed", "physical_passed"):
                    ET.SubElement(case, outcome)
                if outcome == "physical_passed":
                    properties = ET.SubElement(case, "properties")
                    ET.SubElement(properties, "property", name="qlh_physical_markers", value="requires_gpu")
            ET.ElementTree(suite).write(xml_path, encoding="utf-8")
        log_path.write_text("simulated child\n", encoding="utf-8")
        return returncodes.get(key, 0), 0.01

    monkeypatch.setattr(channels, "_run_pytest", fake_run)

    def invoke(*args):
        code = channels.main(["--artifacts-dir", str(tmp_path), "--order-seed", "7", *args])
        pointer = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
        session = tmp_path / pointer["session"]
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
        return code, session, manifest

    return invoke, outcomes, returncodes, calls


def test_unit_only_opt_in_is_not_physical_execution(simulated_runner, monkeypatch):
    invoke, _, _, calls = simulated_runner
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    code, _, manifest = invoke("--channel", "unit")
    assert code == 0
    assert len(calls) == 1
    physical = manifest["physical"]
    assert physical["opt_in"] is True
    assert physical["requested"] is physical["attempted"] is physical["ran"] is False
    assert physical["status"] == "NOT_RUN"
    assert manifest["smoke"]["ran"] is False


@pytest.mark.parametrize("outcome", ["failure", "error", "skipped", "empty", "missing", "malformed"])
def test_runner_rejects_bad_junit_despite_zero_returncode(simulated_runner, outcome):
    invoke, outcomes, _, _ = simulated_runner
    outcomes["unit-001"] = outcome
    code, _, manifest = invoke("--channel", "unit")
    assert code == 1
    assert manifest["runs"][0]["returncode"] == 0
    assert manifest["runs"][0]["errors"]


@pytest.mark.parametrize("returncode", [None, False, True, "0", 0.0, 1])
def test_runner_rejects_invalid_returncodes_even_with_passing_xml(simulated_runner, returncode):
    invoke, _, returncodes, _ = simulated_runner
    returncodes["unit-001"] = returncode
    assert invoke("--channel", "unit")[0] == 1


def test_smoke_all_skip_is_attempted_with_no_execution(simulated_runner, monkeypatch):
    invoke, outcomes, _, _ = simulated_runner
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    outcomes["smoke-001"] = "skipped"
    code, _, manifest = invoke("--channel", "smoke")
    assert code == 1
    smoke = manifest["smoke"]
    assert smoke["opt_in"] is smoke["attempted"] is True
    assert smoke["ran"] is False
    assert smoke["passed"] == 0
    assert smoke["skipped"] == 1
    assert smoke["status"] == "NOT_RUN"


def test_smoke_failure_records_execution_but_cannot_pass(simulated_runner, monkeypatch):
    invoke, outcomes, _, _ = simulated_runner
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    outcomes["smoke-001"] = "failure"
    code, _, manifest = invoke("--channel", "smoke")
    assert code == 1
    assert manifest["smoke"]["ran"] is True
    assert manifest["smoke"]["status"] == "FAIL"


def test_ordinary_all_channel_run_does_not_start_opt_in_channels(simulated_runner):
    from scripts import summarize_test_channels as summary

    invoke, _, _, calls = simulated_runner
    code, session, manifest = invoke("--channel", "all")
    assert code == 0
    assert len(calls) == 2
    assert [run["channel"] for run in manifest["runs"]] == ["unit", "external"]
    for status in (manifest["physical"], manifest["smoke"]):
        assert status["status"] == "NOT_RUN"
        assert status["opt_in"] is status["attempted"] is status["ran"] is False
    assert summary.summarize(session.parent)[0] == 0


def test_smoke_only_without_opt_in_has_no_pass(simulated_runner):
    invoke, _, _, calls = simulated_runner
    code, _, manifest = invoke("--channel", "smoke")
    assert code == 1
    assert calls == []
    assert manifest["smoke"]["status"] == "NOT_RUN"


def test_physical_gate_requires_attributed_external_passes(simulated_runner, monkeypatch):
    invoke, _, _, calls = simulated_runner
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    code, _, manifest = invoke("--channel", "all", "--require-physical")
    assert code == 1
    assert len(calls) == 3
    assert manifest["physical"]["passed"] == 1
    assert manifest["physical"]["status"] == "FAIL"
    assert "scripts.run_test_channels" in calls[1]
    assert "scripts.run_test_channels" in calls[2]


def test_physical_gate_accepts_marker_evidence_for_each_repeat(simulated_runner, monkeypatch):
    from scripts import summarize_test_channels as summary

    invoke, outcomes, _, _ = simulated_runner
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    outcomes.update({"external-001": "physical_passed", "external-002": "physical_passed"})
    code, session, manifest = invoke("--channel", "all", "--repeat", "2", "--require-physical")
    assert code == 0
    assert len(manifest["expected_runs"]) == len(manifest["runs"]) == 6
    assert manifest["physical"]["passed"] == 4
    assert manifest["physical"]["status"] == "PASS"
    assert summary.summarize(session.parent, require_physical=True)[0] == 0


def test_physical_requirement_without_opt_in_fails_before_launch(simulated_runner):
    invoke, _, _, calls = simulated_runner
    code, _, manifest = invoke("--channel", "all", "--require-physical")
    assert code == 1
    assert calls == []
    assert manifest["physical"]["status"] == "NOT_RUN"


def test_physical_requirement_rejects_unit_only_opt_in(simulated_runner, monkeypatch):
    invoke, _, _, calls = simulated_runner
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    assert invoke("--channel", "unit", "--require-physical")[0] == 1
    assert calls == []


def test_interruption_cannot_reveal_an_older_success(simulated_runner, monkeypatch):
    from scripts import summarize_test_channels as summary

    invoke, outcomes, _, _ = simulated_runner
    assert invoke("--channel", "unit", "--session", "old-success")[0] == 0
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    outcomes["smoke-001"] = "interrupt"
    code, session, manifest = invoke("--channel", "smoke", "--session", "interrupted")
    assert code == 1
    assert manifest["state"] == "interrupted"
    assert manifest["smoke"]["attempted"] is True
    assert manifest["smoke"]["ran"] is False
    assert summary.summarize(session.parent)[0] == 1


def test_later_skipped_smoke_repeat_cannot_hide_behind_first_pass(simulated_runner, monkeypatch):
    invoke, outcomes, _, _ = simulated_runner
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    outcomes["smoke-002"] = "skipped"
    code, _, manifest = invoke("--channel", "smoke", "--repeat", "2")
    assert code == 1
    assert manifest["smoke"]["ran"] is True
    assert manifest["smoke"]["passed"] == manifest["smoke"]["skipped"] == 1


def test_explicit_session_cannot_overwrite_existing_evidence(simulated_runner):
    invoke, _, _, calls = simulated_runner
    assert invoke("--channel", "unit", "--session", "kept")[0] == 0
    with pytest.raises(SystemExit, match="already exists"):
        invoke("--channel", "unit", "--session", "kept")
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "/outside", "..", ".", ""])
def test_runner_session_path_is_confined(simulated_runner, name):
    invoke, _, _, calls = simulated_runner
    with pytest.raises(SystemExit):
        invoke("--channel", "unit", "--session", name)
    assert calls == []


def test_marker_plugin_attributes_only_physical_cases(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    physical = SimpleNamespace(user_properties=[], iter_markers=lambda: iter([
        SimpleNamespace(name="external"), SimpleNamespace(name="requires_gpu"),
        SimpleNamespace(name="slow")]))
    ordinary = SimpleNamespace(user_properties=[], iter_markers=lambda: iter([
        SimpleNamespace(name="external")]))
    channels.pytest_collection_finish(SimpleNamespace(items=[physical, ordinary]))
    assert physical.user_properties == [("qlh_physical_markers", "requires_gpu,slow")]
    assert ordinary.user_properties == []


def test_marker_plugin_does_nothing_without_opt_in(monkeypatch):
    from types import SimpleNamespace

    _clear(monkeypatch)
    item = SimpleNamespace(user_properties=[], iter_markers=lambda: iter([SimpleNamespace(name="slow")]))
    channels.pytest_collection_finish(SimpleNamespace(items=[item]))
    assert item.user_properties == []


@pytest.mark.parametrize("workers", [0, 2])
def test_physical_marker_properties_survive_real_pytest_junit(tmp_path, monkeypatch, workers):
    """Synthetic functions verify marker and fixture properties through xdist."""
    import os
    import subprocess
    import sys
    import xml.etree.ElementTree as ET
    from scripts import summarize_test_channels as summary

    source = tmp_path / "test_synthetic_attribution.py"
    source.write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def attributed(request):\n"
        "    request.node.user_properties.append(('fixture_property', 'retained'))\n"
        "@pytest.mark.slow\n"
        "@pytest.mark.parametrize('outcome', ['pass', 'skip', 'failure'])\n"
        "def test_synthetic(attributed, outcome):\n"
        "    if outcome == 'skip':\n"
        "        pytest.skip('synthetic skip')\n"
        "    assert outcome != 'failure', 'synthetic failure'\n", encoding="utf-8")
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\nmarkers =\n    slow: synthetic marker attribution only\n", encoding="utf-8")
    output = tmp_path / "synthetic.xml"
    monkeypatch.setenv(channels.PHYSICAL_OPT_IN_ENV, "1")
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    process = subprocess.run([
        sys.executable, "-B", "-m", "pytest", "-c", str(config), "--confcutdir", str(tmp_path),
        "-p", "scripts.run_test_channels", "-p", "xdist.plugin", "-n", str(workers),
        str(source), "--junitxml", str(output), "-q",
    ], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert process.returncode == 1, process.stdout + process.stderr
    counts, physical_counts = summary._junit_evidence(output)
    assert counts == physical_counts == {
        "tests": 3, "passed": 1, "skipped": 1, "failed": 1, "errors": 0}
    cases = ET.parse(output).getroot().findall(".//testcase")
    assert len(cases) == 3
    for case in cases:
        properties = {prop.get("name"): prop.get("value")
                      for prop in case.findall("properties/property")}
        assert properties["qlh_physical_markers"] == "slow"
        assert properties["fixture_property"] == "retained"
