"""Negative evidence tests: an interrupted or unexecuted channel must fail closed."""
from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts import summarize_test_channels as summary


def _xml(path: Path, outcomes=("passed",), *, physical=False) -> None:
    counts = {"tests": len(outcomes), "failures": outcomes.count("failure"),
              "errors": outcomes.count("error"), "skipped": outcomes.count("skipped")}
    suite = ET.Element("testsuite", {key: str(value) for key, value in counts.items()})
    for number, outcome in enumerate(outcomes):
        case = ET.SubElement(suite, "testcase", name=f"test_{number}")
        if outcome != "passed":
            ET.SubElement(case, outcome)
        if physical:
            properties = ET.SubElement(case, "properties")
            ET.SubElement(properties, "property", name="qlh_physical_markers", value="requires_gpu")
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _save(session: Path, manifest: dict) -> None:
    (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _session(base: Path, name="test-channels-20261015T010000Z-101", *,
             channel="unit", repeat=1, opt_in=False, physical=False):
    session = base / name
    session.mkdir(parents=True)
    expected = summary.expected_runs(channel, repeat, opt_in)
    manifest = {"schema_version": 2, "session": name, "state": "completed",
                "channel": channel, "repeat": repeat, "expected_runs": expected,
                "smoke": {"opt_in": opt_in, "requested": channel in ("all", "smoke"),
                          "attempted": False, "ran": False},
                "physical": {"opt_in": opt_in, "requested": channel != "unit",
                             "attempted": False, "ran": False}, "runs": []}
    for run in expected:
        filename = f"{run['channel']}-{run['run']:03d}.xml"
        _xml(session / filename, physical=physical and run["channel"] == "external")
        manifest["runs"].append({**run, "state": "completed", "returncode": 0,
                                 "junitxml": filename})
    _save(session, manifest)
    return session, manifest


@pytest.fixture
def evidence(tmp_path):
    session, manifest = _session(tmp_path)
    return tmp_path, session, manifest


def test_valid_ordinary_ci_keeps_physical_not_run(tmp_path):
    _session(tmp_path, channel="all")
    code, result = summary.summarize(tmp_path)
    assert code == 0
    assert len(result["runs"]) == 2
    for name in ("smoke", "physical"):
        assert result[name]["status"] == "NOT_RUN"
        assert result[name]["opt_in"] is result[name]["attempted"] is result[name]["ran"] is False


@pytest.mark.parametrize("value", [None, False, True, 0.0, "0", 1, -1])
def test_returncode_must_be_exact_integer_zero(evidence, value):
    base, session, manifest = evidence
    manifest["runs"][0]["returncode"] = value
    _save(session, manifest)
    code, result = summary.summarize(base)
    assert code == 1
    assert result["runs"][0]["status"] == "FAIL"


def test_returncode_is_required(evidence):
    base, session, manifest = evidence
    del manifest["runs"][0]["returncode"]
    _save(session, manifest)
    assert summary.summarize(base)[0] == 1


@pytest.mark.parametrize("text", ["", "<testsuite", "<not-junit/>",
    '<testsuite tests="1" failures="0"/>',
    '<testsuite tests="-1"/>',
    '<testsuite tests="NaN"/>',
    '<testsuite><testcase name="x"/></testsuite>',
    '<testsuite tests="1" failures="0"><testcase name="x"><failure/></testcase></testsuite>',
    '<testsuite tests="1"><testcase name="x"><error/><skipped/></testcase></testsuite>',
    '<testsuite tests="1"><garbage/></testsuite>'])
def test_malformed_or_inconsistent_junit_fails_with_rc_zero(evidence, text):
    base, session, manifest = evidence
    (session / manifest["runs"][0]["junitxml"]).write_text(text, encoding="utf-8")
    code, result = summary.summarize(base)
    assert code == 1
    assert result["runs"][0]["errors"]


def test_missing_junit_fails_with_rc_zero(evidence):
    base, session, manifest = evidence
    (session / manifest["runs"][0]["junitxml"]).unlink()
    assert summary.summarize(base)[0] == 1


@pytest.mark.parametrize("outcomes", [("failure",), ("error",), ("skipped",), ()])
def test_failed_skipped_or_empty_execution_is_not_passed(evidence, outcomes):
    base, session, manifest = evidence
    _xml(session / manifest["runs"][0]["junitxml"], outcomes)
    code, result = summary.summarize(base)
    assert code == 1
    assert result["runs"][0]["counts"]["passed"] == 0


def test_passed_and_skipped_are_counted_separately(evidence):
    base, session, manifest = evidence
    _xml(session / manifest["runs"][0]["junitxml"], ("passed", "skipped"))
    code, result = summary.summarize(base)
    assert code == 0
    assert result["runs"][0]["counts"] == {
        "tests": 2, "passed": 1, "skipped": 1, "failed": 0, "errors": 0}


def test_nested_suites_do_not_double_count(evidence):
    base, session, manifest = evidence
    (session / manifest["runs"][0]["junitxml"]).write_text(
        '<testsuites tests="2"><testsuite tests="2" skipped="1">'
        '<testsuite tests="1"><testcase name="pass"/></testsuite>'
        '<testsuite tests="1" skipped="1"><testcase name="skip"><skipped/>'
        '</testcase></testsuite></testsuite></testsuites>', encoding="utf-8")
    code, result = summary.summarize(base)
    assert code == 0
    assert result["runs"][0]["counts"]["tests"] == 2
    assert result["runs"][0]["counts"]["passed"] == 1


@pytest.mark.parametrize("text", ["", "{", "[]", "null", "1", '"text"',
                                  '{"runs": [], "runs": []}', '{"runs": NaN}'])
def test_bad_json_produces_failure_summary(evidence, text):
    base, session, _ = evidence
    (session / "manifest.json").write_text(text, encoding="utf-8")
    code, result = summary.summarize(base)
    assert code == 1
    assert result["status"] == "FAIL"
    assert result["errors"]


@pytest.mark.parametrize("field,value", [
    ("runs", []), ("runs", None), ("runs", {}), ("runs", [None]),
    ("channel", None), ("channel", []), ("repeat", True), ("repeat", 0),
    ("repeat", "1"), ("smoke", []), ("physical", []),
    ("expected_runs", []), ("state", "running"), ("schema_version", 999)])
def test_incomplete_or_invalid_manifest_is_rejected(evidence, field, value):
    base, session, manifest = evidence
    manifest[field] = value
    _save(session, manifest)
    assert summary.summarize(base)[0] == 1


def test_complete_run_set_is_required_for_every_repetition(tmp_path):
    session, manifest = _session(tmp_path, channel="all", repeat=2)
    manifest["runs"].pop()
    _save(session, manifest)
    code, result = summary.summarize(tmp_path)
    assert code == 1
    assert any("run set mismatch" in error for error in result["errors"])


def test_duplicates_cannot_stand_in_for_a_missing_repeat(tmp_path):
    session, manifest = _session(tmp_path, repeat=2)
    manifest["runs"][1] = dict(manifest["runs"][0])
    _save(session, manifest)
    code, result = summary.summarize(tmp_path)
    assert code == 1
    assert any("duplicate run" in error for error in result["errors"])


def test_distinct_runs_cannot_reuse_the_same_xml(tmp_path):
    session, manifest = _session(tmp_path, repeat=2)
    manifest["runs"][1]["junitxml"] = manifest["runs"][0]["junitxml"]
    _save(session, manifest)
    assert summary.summarize(tmp_path)[0] == 1


@pytest.mark.parametrize("number", [True, None, "1", 0, -1])
def test_run_number_is_strictly_validated(evidence, number):
    base, session, manifest = evidence
    manifest["runs"][0]["run"] = number
    _save(session, manifest)
    assert summary.summarize(base)[0] == 1


@pytest.mark.parametrize("value", ["../outside.xml", "..\\outside.xml", "C:/outside.xml",
    "C:outside.xml", "\\\\server\\share\\outside.xml", "/outside.xml", "", ".", None])
def test_junit_path_must_stay_inside_session(evidence, value):
    base, session, manifest = evidence
    _xml(base / "outside.xml")
    manifest["runs"][0]["junitxml"] = value
    _save(session, manifest)
    assert summary.summarize(base)[0] == 1


def test_junit_link_cannot_escape_session(evidence):
    base, session, manifest = evidence
    outside = base / "outside"
    outside.mkdir()
    _xml(outside / "pass.xml")
    link = session / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi
        _winapi.CreateJunction(str(outside), str(link))
    manifest["runs"][0]["junitxml"] = "linked/pass.xml"
    _save(session, manifest)
    assert summary.summarize(base)[0] == 1


def test_latest_pointer_selects_interrupted_session_over_old_success(tmp_path):
    old, _ = _session(tmp_path)
    new, manifest = _session(tmp_path, "test-channels-20261015T010001Z-102")
    manifest.update(state="running", runs=[])
    _save(new, manifest)
    (tmp_path / "latest.json").write_text(json.dumps({"session": new.name}), encoding="utf-8")
    code, result = summary.summarize(tmp_path)
    assert code == 1
    assert result["session"] == new.name
    assert summary.summarize(tmp_path, session=old.name)[0] == 0


@pytest.mark.parametrize("pointer", ['{"session":"missing"}', '{', '{"session":"../outside"}'])
def test_bad_or_missing_latest_session_never_falls_back_to_old_pass(tmp_path, pointer):
    _session(tmp_path)
    (tmp_path / "latest.json").write_text(pointer, encoding="utf-8")
    assert summary.summarize(tmp_path)[0] == 1


def test_latest_legacy_session_includes_missing_manifest(tmp_path):
    old, _ = _session(tmp_path)
    new = tmp_path / "test-channels-20261015T010001Z-102"
    new.mkdir()
    # Touching old evidence later must not make an older invocation the latest.
    os.utime(old, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    code, result = summary.summarize(tmp_path)
    assert code == 1
    assert result["session"] == new.name


def test_latest_legacy_numeric_suffix_is_not_lexicographic(tmp_path):
    _session(tmp_path, "test-channels-20261015T010000Z-101-2")
    new, manifest = _session(tmp_path, "test-channels-20261015T010000Z-101-10")
    manifest["runs"] = []
    _save(new, manifest)
    code, result = summary.summarize(tmp_path)
    assert code == 1
    assert result["session"] == new.name


def test_direct_session_directory_is_supported(evidence):
    _, session, _ = evidence
    assert summary.summarize(session)[0] == 0


@pytest.mark.parametrize("value", ["../outside", "C:/outside", "/outside", "..", "."])
def test_explicit_session_is_confined(evidence, value):
    base, _, _ = evidence
    assert summary.summarize(base, session=value)[0] == 1


def test_legacy_opt_in_flag_is_never_reused_as_execution_evidence(tmp_path):
    session, manifest = _session(tmp_path, opt_in=True)
    for field in ("schema_version", "state", "expected_runs"):
        del manifest[field]
    manifest["physical"] = {"ran": True}
    manifest["smoke"] = {"ran": False}
    _save(session, manifest)
    code, result = summary.summarize(tmp_path)
    assert code == 0
    assert result["physical"]["opt_in"] is True
    assert result["physical"]["ran"] is False
    assert result["physical"]["status"] == "NOT_RUN"
    assert summary.summarize(tmp_path, require_physical=True)[0] == 1


def test_physical_gate_requires_opt_in(tmp_path):
    _session(tmp_path, channel="all")
    assert summary.summarize(tmp_path, require_physical=True)[0] == 1


def test_physical_gate_requires_real_external_marker_evidence(tmp_path):
    _session(tmp_path, channel="all", opt_in=True)
    code, result = summary.summarize(tmp_path, require_physical=True)
    assert code == 1
    assert result["physical"]["passed"] == 1  # Only the smoke test is attributable.


def test_physical_gate_accepts_attributed_passes_in_every_run(tmp_path):
    _session(tmp_path, channel="all", repeat=2, opt_in=True, physical=True)
    code, result = summary.summarize(tmp_path, require_physical=True)
    assert code == 0
    assert result["physical"]["status"] == "PASS"
    assert result["physical"]["passed"] == 4


def test_physical_gate_does_not_accept_a_previous_success_for_current_skips(tmp_path):
    session, manifest = _session(tmp_path, channel="all", repeat=2, opt_in=True, physical=True)
    _xml(session / "external-002.xml", ("skipped",), physical=True)
    # Ordinary passes in that same external run must not hide physical skips.
    tree = ET.parse(session / "external-002.xml")
    suite = tree.getroot()
    suite.set("tests", "2")
    ET.SubElement(suite, "testcase", name="ordinary")
    tree.write(session / "external-002.xml", encoding="utf-8")
    code, result = summary.summarize(tmp_path, require_physical=True)
    assert code == 1
    assert result["physical"]["skipped"] == 1


def test_all_skip_smoke_is_attempted_but_not_run(tmp_path):
    session, _ = _session(tmp_path, channel="smoke", opt_in=True)
    _xml(session / "smoke-001.xml", ("skipped",))
    code, result = summary.summarize(tmp_path)
    assert code == 1
    assert result["smoke"]["attempted"] is True
    assert result["smoke"]["ran"] is False
    assert result["smoke"]["passed"] == 0
    assert result["smoke"]["skipped"] == 1


def test_manifest_physical_requirement_cannot_be_disabled_by_omitting_cli_flag(tmp_path):
    session, manifest = _session(tmp_path, channel="all")
    manifest["require_physical"] = True
    _save(session, manifest)
    assert summary.summarize(tmp_path)[0] == 1


def test_cached_counts_cannot_hide_xml_failure(evidence):
    base, session, manifest = evidence
    manifest["runs"][0].update(counts={"passed": 999}, status="PASS")
    _save(session, manifest)
    _xml(session / "unit-001.xml", ("failure",))
    assert summary.summarize(base)[0] == 1


def test_cli_writes_failure_json_even_for_malformed_input(evidence):
    base, session, _ = evidence
    (session / "manifest.json").write_text("{", encoding="utf-8")
    output = base / "summary.json"
    assert summary.main(["--artifacts-dir", str(base), "--session", session.name,
                         "--require-physical", "--json", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_latest_pointer_takes_precedence_over_a_stale_root_manifest(evidence):
    base, old, manifest = evidence
    _save(base, manifest)
    new = base / "test-channels-20261015T010001Z-102"
    new.mkdir()
    (base / "latest.json").write_text(json.dumps({"session": new.name}), encoding="utf-8")
    assert summary.summarize(base)[0] == 1
    assert summary.summarize(base, session=old.name)[0] == 0


def test_unpublished_new_session_cannot_hide_behind_stale_latest_pointer(evidence):
    base, old, _ = evidence
    pointer = base / "latest.json"
    pointer.write_text(json.dumps({"session": old.name}), encoding="utf-8")
    new = base / "test-channels-20261015T010001Z-102"
    new.mkdir()
    newer = pointer.stat().st_mtime_ns + 1_000_000_000
    os.utime(new, ns=(newer, newer))
    assert summary.summarize(base)[0] == 1
    assert summary.summarize(base, session=old.name)[0] == 0


@pytest.mark.parametrize("mutate", ["version_null", "legacy_running", "run_running", "expected_bool", "wrong_session"])
def test_partial_state_and_schema_cannot_be_treated_as_legacy_success(evidence, mutate):
    base, session, manifest = evidence
    if mutate == "version_null":
        manifest["schema_version"] = None
    elif mutate == "legacy_running":
        del manifest["schema_version"]
        manifest["state"] = "running"
    elif mutate == "run_running":
        manifest["runs"][0]["state"] = "running"
    elif mutate == "expected_bool":
        manifest["expected_runs"][0]["run"] = True
    else:
        manifest["session"] = "another-session"
    _save(session, manifest)
    assert summary.summarize(base)[0] == 1


def test_unknown_testcase_structure_is_invalid_evidence(evidence):
    base, session, manifest = evidence
    (session / manifest["runs"][0]["junitxml"]).write_text(
        '<testsuite tests="1"><testcase name="x"><unknown><failure/></unknown>'
        '</testcase></testsuite>', encoding="utf-8")
    assert summary.summarize(base)[0] == 1


def test_missing_physical_repetition_cannot_report_physical_pass(tmp_path):
    session, manifest = _session(tmp_path, channel="all", repeat=2, opt_in=True, physical=True)
    manifest["runs"].pop()
    _save(session, manifest)
    code, result = summary.summarize(tmp_path, require_physical=True)
    assert code == 1
    assert result["physical"]["status"] != "PASS"
    assert result["smoke"]["status"] != "PASS"


def test_physical_evidence_without_opt_in_fails(evidence):
    base, session, manifest = evidence
    manifest["channel"] = "external"
    manifest["expected_runs"] = [{"channel": "external", "run": 1}]
    manifest["runs"][0]["channel"] = "external"
    _xml(session / "unit-001.xml", physical=True)
    _save(session, manifest)
    code, result = summary.summarize(base)
    assert code == 1
    assert "physical test evidence exists without opt-in" in result["runs"][0]["errors"]


def test_failed_channels_keep_channel_names_with_new_failed_run_detail(evidence):
    base, session, _ = evidence
    _xml(session / "unit-001.xml", ("failure",))
    code, result = summary.summarize(base)
    assert code == 1
    assert result["failed_channels"] == ["unit"]
    assert result["failed_runs"] == ["unit#1"]
