"""Validate test-channel evidence; skipped and missing execution never count as passes.

Use --session NAME to select one invocation, or the runner's latest.json pointer.
--require-physical requires an opted-in, complete all-channel invocation with actual
physical passes in every external and smoke repetition. Ordinary CI may report
physical=NOT_RUN while its unit and external channels pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PureWindowsPath
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_MARKERS = ("real_model", "requires_gpu", "slow", "requires_db")
COUNT_KEYS = ("tests", "passed", "skipped", "failed", "errors")


def _enable_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(COUNT_KEYS, 0)


def _confined_path(directory: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence path must be a nonempty relative path")
    windows = PureWindowsPath(value)
    parts = value.replace("\\", "/").split("/")
    if windows.drive or windows.root or Path(value).is_absolute() or ".." in parts:
        raise ValueError(f"evidence path escapes session: {value!r}")
    path = (directory / Path(*parts)).resolve()
    if not path.is_relative_to(directory.resolve()) or path == directory.resolve():
        raise ValueError(f"evidence path escapes session: {value!r}")
    return path


def session_path(artifacts_dir: Path, name: object) -> Path:
    if not isinstance(name, str) or "/" in name or "\\" in name or name in ("", ".", ".."):
        raise ValueError("session must name one directory under --artifacts-dir")
    return _confined_path(artifacts_dir, name)


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object,
                       parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON evidence must be an object")
    return value


def _junit_evidence(xml_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Check declared suite totals against testcase outcomes, including nested suites."""
    root = ET.parse(xml_path).getroot()
    if root.tag not in ("testsuite", "testsuites"):
        raise ValueError("JUnit root must be testsuite or testsuites")
    physical = _empty_counts()

    def suite_counts(suite: ET.Element) -> dict[str, int]:
        counts = _empty_counts()
        for child in suite:
            if child.tag in ("testsuite", "testsuites"):
                nested = suite_counts(child)
                for key in COUNT_KEYS:
                    counts[key] += nested[key]
            elif child.tag == "testcase":
                if not child.get("name"):
                    raise ValueError("JUnit testcase has no name")
                if any(element.tag not in ("properties", "system-out", "system-err",
                                           "failure", "error", "skipped") for element in child):
                    raise ValueError("unsupported JUnit testcase element")
                outcomes = [element.tag for element in child
                            if element.tag in ("failure", "error", "skipped")]
                if len(outcomes) > 1:
                    raise ValueError("JUnit testcase has conflicting outcomes")
                outcome = {"failure": "failed", "error": "errors", "skipped": "skipped"}.get(
                    outcomes[0] if outcomes else "", "passed")
                counts["tests"] += 1
                counts[outcome] += 1
                properties = [prop.get("value", "") for prop in child.findall("properties/property")
                              if prop.get("name") == "qlh_physical_markers"]
                if properties:
                    if len(properties) != 1 or not properties[0]:
                        raise ValueError("invalid physical marker property")
                    markers = properties[0].split(",")
                    if not set(markers) <= set(PHYSICAL_MARKERS):
                        raise ValueError("unknown physical marker property")
                    physical["tests"] += 1
                    physical[outcome] += 1
            elif child.tag not in ("properties", "system-out", "system-err"):
                raise ValueError(f"unsupported JUnit element: {child.tag}")
        for attribute, key in (("tests", "tests"), ("failures", "failed"),
                               ("errors", "errors"), ("skipped", "skipped")):
            declared = suite.get(attribute)
            if declared is None:
                if attribute == "tests" and suite.tag == "testsuite":
                    raise ValueError("JUnit testsuite is missing tests count")
                continue
            if not re.fullmatch(r"[0-9]+", declared) or int(declared) != counts[key]:
                raise ValueError(f"JUnit {attribute} count disagrees with testcase outcomes")
        return counts

    return suite_counts(root), physical


def _counts(xml_path: Path) -> dict[str, int] | None:
    if not xml_path.exists():
        return None
    return _junit_evidence(xml_path)[0]


def evaluate_run(run: dict[str, object], session: Path) -> dict[str, object]:
    entry = {key: run.get(key) for key in
             ("channel", "run", "returncode", "duration_s", "junitxml")}
    errors: list[str] = []
    counts = physical_counts = None
    returncode = run.get("returncode")
    if type(returncode) is not int or returncode != 0:
        errors.append("returncode must be integer zero")
    try:
        xml_path = _confined_path(session, run.get("junitxml"))
        counts, physical_counts = _junit_evidence(xml_path)
        if run.get("channel") == "smoke":
            physical_counts = dict(counts)
        if counts["failed"] or counts["errors"]:
            errors.append("JUnit contains failures or errors")
        if counts["passed"] == 0:
            errors.append("JUnit contains no passed tests")
    except (OSError, ValueError, ET.ParseError) as exc:
        errors.append(f"invalid JUnit evidence: {exc}")
    entry.update(counts=counts, physical_counts=physical_counts, errors=errors,
                 status="FAIL" if errors else "PASS")
    if counts is not None and counts["tests"] == counts["skipped"]:
        entry["status"] = "NOT_RUN"
    return entry


def expected_runs(channel: str, repeat: int, opt_in: bool) -> list[dict[str, object]]:
    selected = ("unit", "external", "smoke") if channel == "all" else (channel,)
    return [{"channel": name, "run": repetition}
            for name in selected if name != "smoke" or opt_in
            for repetition in range(1, repeat + 1)]


def execution_status(runs: list[dict[str, object]], *, requested: bool,
                     opt_in: bool, smoke: bool = False) -> dict[str, object]:
    selected = [run for run in runs if run.get("channel") in
                (("smoke",) if smoke else ("external", "smoke"))]
    counts = _empty_counts()
    for run in selected:
        for key, value in (run.get("physical_counts") or {}).items():
            counts[key] += value
    attempted = bool(selected) and opt_in
    ran = counts["passed"] + counts["failed"] + counts["errors"] > 0
    if not opt_in:
        status, reason = "NOT_RUN", "physical opt-in is disabled"
    elif not requested or not attempted:
        status, reason = "NOT_RUN", "no physical channel was attempted"
    elif not ran and all(run.get("errors") == ["JUnit contains no passed tests"]
                         for run in selected):
        status, reason = "NOT_RUN", "no physical tests executed; skips are not passes"
    elif any(run.get("errors") for run in selected):
        status, reason = "FAIL", "channel evidence failed validation"
    elif all((run.get("physical_counts") or {}).get("passed", 0) > 0 for run in selected):
        status, reason = "PASS", ""
    elif not ran:
        status, reason = "NOT_RUN", "no physical tests executed; skips are not passes"
    else:
        status, reason = "FAIL", "physical passes are missing from a selected run"
    return {"requested": requested, "opt_in": opt_in, "attempted": attempted,
            "ran": ran, "status": status, "reason": reason, **counts}


def evaluate_manifest(manifest: dict[str, object], session: Path, *,
                      require_physical: bool = False) -> tuple[int, dict[str, object]]:
    errors: list[str] = []
    channel, repeat = manifest.get("channel"), manifest.get("repeat")
    if channel not in ("all", "unit", "external", "smoke"):
        errors.append("invalid or missing channel")
    if type(repeat) is not int or not 1 <= repeat <= 100:
        errors.append("repeat must be an integer between 1 and 100")
    smoke = manifest.get("smoke")
    physical = manifest.get("physical")
    if not isinstance(smoke, dict) or not isinstance(physical, dict):
        errors.append("smoke and physical metadata must be objects")
        smoke, physical = {}, {}
    # Old manifests used physical.ran for the opt-in flag. Never reuse it as
    # execution evidence; actual execution is reconstructed from JUnit below.
    opt_in = physical.get("opt_in", physical.get("ran"))
    if type(opt_in) is not bool:
        errors.append("physical opt_in must be boolean")
        opt_in = False
    version = manifest.get("schema_version")
    if "schema_version" in manifest and (type(version) is not int or version != 2):
        errors.append("unsupported manifest schema_version")
    if ("state" in manifest or version == 2) and manifest.get("state") != "completed":
        errors.append("session did not complete")
    if version == 2:
        if manifest.get("session") != session.name:
            errors.append("manifest session disagrees with its directory")
        for label, metadata in (("smoke", smoke), ("physical", physical)):
            for field in ("opt_in", "requested", "attempted", "ran"):
                if type(metadata.get(field)) is not bool:
                    errors.append(f"{label}.{field} must be boolean")
        if smoke.get("opt_in") is not opt_in:
            errors.append("smoke and physical opt_in disagree")
        if "require_physical" in manifest and type(manifest["require_physical"]) is not bool:
            errors.append("require_physical must be boolean")
    require_physical = require_physical or manifest.get("require_physical") is True
    valid_selection = channel in ("all", "unit", "external", "smoke") and type(repeat) is int and 1 <= repeat <= 100
    expected = expected_runs(channel, repeat, opt_in) if valid_selection else []
    declared = manifest.get("expected_runs")
    if version == 2 or "expected_runs" in manifest:
        if (declared != expected or not isinstance(declared, list) or
            any(not isinstance(run, dict) or type(run.get("run")) is not int for run in declared)):
            errors.append("declared expected_runs disagree with channel/repeat/opt_in")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list):
        errors.append("runs must be an array")
        raw_runs = []
    if not raw_runs:
        errors.append("session contains no runs")
    runs: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    xml_paths: set[Path] = set()
    for raw in raw_runs:
        if not isinstance(raw, dict):
            errors.append("run must be an object")
            continue
        name, number = raw.get("channel"), raw.get("run")
        if name not in ("unit", "external", "smoke") or type(number) is not int or number < 1:
            errors.append("run has invalid channel or repetition")
            continue
        key = (name, number)
        if key in seen:
            errors.append(f"duplicate run: {name}#{number}")
        seen.add(key)
        entry = evaluate_run(raw, session)
        if ("state" in raw or version == 2) and raw.get("state") != "completed":
            entry["errors"].append("run did not complete")
            entry["status"] = "FAIL"
        try:
            xml_path = _confined_path(session, raw.get("junitxml"))
            if xml_path in xml_paths:
                entry["errors"].append("JUnit evidence is reused by another run")
                entry["status"] = "FAIL"
            xml_paths.add(xml_path)
        except (OSError, ValueError):
            pass  # evaluate_run already reported the invalid path.
        if not opt_in and any((entry.get("physical_counts") or {}).values()):
            entry["errors"].append("physical test evidence exists without opt-in")
            entry["status"] = "FAIL"
        runs.append(entry)
    expected_keys = {(run["channel"], run["run"]) for run in expected}
    if seen != expected_keys:
        errors.append(f"run set mismatch: missing={sorted(expected_keys - seen)} unexpected={sorted(seen - expected_keys)}")
    if require_physical:
        if channel != "all" or not opt_in:
            errors.append("physical verification requires channel=all and physical opt-in")
        for run in runs:
            if run["channel"] in ("external", "smoke") and (
                run["status"] != "PASS" or (run["physical_counts"] or {}).get("passed", 0) == 0
            ):
                errors.append(f"physical verification has no passing evidence: {run['channel']}#{run['run']}")
    smoke_status = execution_status(runs, requested=channel in ("all", "smoke"),
                                    opt_in=opt_in, smoke=True)
    physical_status = {**physical, **execution_status(
        runs, requested=channel in ("all", "external", "smoke"), opt_in=opt_in)}
    for status, names in ((smoke_status, ("smoke",)),
                          (physical_status, ("external", "smoke"))):
        relevant_expected = {key for key in expected_keys if key[0] in names}
        relevant_seen = {key for key in seen if key[0] in names}
        if status["status"] == "PASS" and (relevant_seen != relevant_expected or
                ("state" in manifest and manifest["state"] != "completed")):
            status.update(status="FAIL", reason="physical session is incomplete")
    failed_runs = [f"{run['channel']}#{run['run']}" for run in runs if run["errors"]]
    failed_channels = list(dict.fromkeys(run["channel"] for run in runs if run["errors"]))
    code = int(bool(errors or failed_runs))
    return code, {"session": session.name, "order_seed": manifest.get("order_seed"),
                  "workers": manifest.get("workers"), "dist": manifest.get("dist"),
                  "state": manifest.get("state"), "status": "FAIL" if code else "PASS",
                  "require_physical": require_physical, "expected_runs": expected,
                  "smoke": smoke_status, "physical": physical_status, "runs": runs,
                  "failed_channels": failed_channels, "failed_runs": failed_runs, "errors": errors}


def _latest_session(artifacts_dir: Path) -> Path:
    pointer = artifacts_dir / "latest.json"
    if pointer.exists() or pointer.is_symlink():
        pointer = _confined_path(artifacts_dir, "latest.json")
        selected = session_path(artifacts_dir, _read_json(pointer).get("session"))
        # A process can die between directory allocation and pointer publication.
        # Refuse an older pointer if another generated session changed afterwards.
        pointer_time = pointer.stat().st_mtime_ns
        for candidate in artifacts_dir.glob("test-channels-*"):
            if candidate.is_dir() and candidate.resolve() != selected and candidate.stat().st_mtime_ns > pointer_time:
                raise ValueError("latest pointer predates another session; use --session explicitly")
        return selected
    # Compatibility with older runners: include incomplete generated sessions,
    # and order their timestamp/suffix numerically rather than by filename.
    candidates = [path for path in artifacts_dir.iterdir() if path.is_dir() and
                  (path.name.startswith("test-channels-") or (path / "manifest.json").exists())]
    if not candidates:
        raise ValueError("no_manifest: no test-channel session found")

    def started(path: Path) -> tuple[int, int, int]:
        match = re.fullmatch(r"test-channels-(\d{8}T\d{6}Z)-\d+(?:-(\d+))?", path.name)
        if match:
            timestamp = int(datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp())
            return timestamp * 10**9, int(match[2] or 1), path.stat().st_mtime_ns
        return path.stat().st_mtime_ns, 0, 0

    latest = max(candidates, key=started)
    return session_path(artifacts_dir, latest.name)


def summarize(artifacts_dir: Path, *, session: str | None = None,
              require_physical: bool = False) -> tuple[int, dict[str, object]]:
    base = artifacts_dir.expanduser().resolve()
    result: dict[str, object] = {"artifacts_dir": str(base), "runs": [], "status": "FAIL"}
    try:
        if session is not None:
            selected = session_path(base, session)
        elif (base / "latest.json").exists() or (base / "latest.json").is_symlink():
            selected = _latest_session(base)
        elif (base / "manifest.json").exists():
            selected = base  # Also allow --artifacts-dir to name a session directly.
        else:
            selected = _latest_session(base)
        result["session"] = selected.name
        manifest_path = _confined_path(selected, "manifest.json")
        code, evidence = evaluate_manifest(_read_json(manifest_path), selected,
                                           require_physical=require_physical)
        result.update(evidence)
    except (OSError, ValueError, ET.ParseError) as exc:
        code = 1
        result.update(error=str(exc), errors=[str(exc)])
    for run in result["runs"]:
        counts = run["counts"] or {}
        print(f"[{run['status']}] {run['channel']}#{run['run']} returncode={run['returncode']} "
              f"passed={counts.get('passed', '?')} skipped={counts.get('skipped', '?')} "
              f"failed={counts.get('failed', '?')} errors={counts.get('errors', '?')}")
        for error in run["errors"]:
            print(f"::error::{run['channel']}#{run['run']}: {error}")
    for label in ("smoke", "physical"):
        if label in result:
            status = result[label]
            print(f"[{label}] {status['status']} opt_in={status['opt_in']} "
                  f"attempted={status['attempted']} ran={status['ran']} "
                  f"passed={status['passed']} skipped={status['skipped']} reason={status['reason']}")
    for error in result.get("errors", []):
        print(f"::error::{error}")
    print(f"[summary] {result['status']} session={result.get('session', '?')}")
    return code, result


def main(argv: list[str] | None = None) -> int:
    _enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "build" / "ci-test-artifacts")
    parser.add_argument("--session", help="exact session directory name under --artifacts-dir")
    parser.add_argument("--require-physical", action="store_true", help="fail unless all physical channels have passing evidence")
    parser.add_argument("--json", type=Path, default=None, help="machine-readable summary output")
    args = parser.parse_args(argv)
    code, result = summarize(args.artifacts_dir, session=args.session,
                             require_physical=args.require_physical)
    if args.json is not None:
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"::error::cannot write summary: {exc}")
            return 1
        print(f"[summary] {args.json}")
    return code


if __name__ == "__main__":
    sys.exit(main())
