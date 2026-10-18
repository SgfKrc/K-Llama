"""Run deterministic distributed-inference simulation profiles.

The runner is deliberately separate from ``run_test_channels.py``.  It owns
the fixed, local SIM-N1 through SIM-N5 pre-validation suite and emits a
body-free JSON summary that a CI job can retain as an artifact.  It is not a
replacement for the long-running full suite or for L1 physical acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCHEMA_VERSION = "qlh.simulation_runner.v1"

QUICK_TEST_TARGETS = (
    "tests/test_refactor_execution.py",
    "tests/test_task_worker_control_simulation.py",
    "tests/test_linear_simulation.py",
)
EXTENDED_TEST_TARGETS = QUICK_TEST_TARGETS + (
    "tests/test_refactor_execution_integration.py",
    "tests/test_task_worker_protocol.py",
    "tests/test_task_worker_adapter.py",
)
FULL_TEST_TARGETS = ("tests",)

_PROFILE_TARGETS = {
    "quick": QUICK_TEST_TARGETS,
    "extended": EXTENDED_TEST_TARGETS,
    "full": FULL_TEST_TARGETS,
}
_FORBIDDEN_EVIDENCE_KEYS = frozenset({
    "authorization",
    "blob_id",
    "content",
    "grant",
    "image",
    "input",
    "message",
    "output",
    "path",
    "prompt",
    "request",
    "response",
    "root_input",
    "secret",
    "token",
    "transfer_plan",
    "url",
})
_PYTEST_COUNT_PATTERN = re.compile(
    r"(?P<count>\d+)\s+(?P<label>passed|failed|skipped|xfailed|xpassed|error|errors)\b",
)


def targets_for_profile(profile: str) -> tuple[str, ...]:
    """Return the fixed pytest targets for one supported profile."""

    try:
        return _PROFILE_TARGETS[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported simulation profile: {profile}") from exc


def _pytest_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    current = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), source, current) if value
    )
    return environment


def _pytest_python() -> str:
    """Prefer the repository test environment when called from system Python."""
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return sys.executable
    candidate = ROOT / ".venv-test" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    return str(candidate) if candidate.is_file() else sys.executable


def _run_pytest(targets: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(
        [_pytest_python(), "-m", "pytest", "-q", *targets],
        cwd=ROOT,
        env=_pytest_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.returncode, completed.stdout


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "errors": 0,
    }
    for match in _PYTEST_COUNT_PATTERN.finditer(output):
        label = match.group("label")
        key = "errors" if label in {"error", "errors"} else label
        counts[key] += int(match.group("count"))
    return counts


def _assert_safe_evidence(value: Any) -> None:
    """Reject a future harness report that adds a raw sensitive field."""

    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_EVIDENCE_KEYS.intersection(value)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"simulation evidence contains forbidden fields: {names}")
        for child in value.values():
            _assert_safe_evidence(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_safe_evidence(child)


def collect_evidence() -> list[dict[str, Any]]:
    """Run each fixed harness scenario and return its already-sanitized report."""

    # Importing api_server configures an application logger in some environments.
    # JSON stdout is a machine contract, so isolate incidental import/test output.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _collect_evidence()


def _collect_evidence() -> list[dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from tests.simulation.linear_harness import (
        LinearSimulationHarness, available_scenarios as available_linear_scenarios,
    )
    from tests.simulation.task_worker_harness import (
        TaskWorkerControlSimulationHarness,
        available_scenarios as available_task_worker_scenarios,
    )

    families = (
        ("linear_execution", LinearSimulationHarness, available_linear_scenarios),
        ("task_worker_control", TaskWorkerControlSimulationHarness, available_task_worker_scenarios),
    )
    evidence: list[dict[str, Any]] = []
    for family, harness_type, scenarios in families:
        harness = harness_type()
        reports = [harness.run(scenario.scenario_id) for scenario in scenarios()]
        _assert_safe_evidence(reports)
        evidence.append({"family": family, "reports": reports})
    return evidence


def build_summary(
    *,
    profile: str,
    pytest_exit_code: int,
    pytest_output: str,
    evidence: Sequence[Mapping[str, Any]],
    evidence_error: BaseException | None = None,
) -> dict[str, Any]:
    """Build the stable, aggregate-only artifact shared by CLI and CI."""

    counts = _pytest_counts(pytest_output)
    succeeded = pytest_exit_code == 0 and evidence_error is None
    summary: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "profile": profile,
        "test_targets": list(targets_for_profile(profile)),
        "outcome": "passed" if succeeded else "failed",
        "pytest": {
            "exit_code": pytest_exit_code,
            "counts": counts,
        },
        "acceptance_scope": {
            "physical_nodes": "not_established",
            "real_models": "not_established",
            "real_network": "not_established",
            "performance": "not_established",
            "installation_package": "not_established",
        },
        "evidence": list(evidence) if succeeded else [],
    }
    if evidence_error is not None:
        summary["runner_error"] = type(evidence_error).__name__
    return summary


def _write_json(summary: Mapping[str, Any], destination: str) -> None:
    serialized = json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2)
    if destination == "-":
        print(serialized)
        return

    path = Path(destination)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(serialized)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILE_TARGETS),
        default="quick",
        help="quick is the default CI contract; extended and full require explicit selection",
    )
    parser.add_argument(
        "--json-output",
        default="-",
        help="summary path relative to the repository, or '-' for stdout (default: '-')",
    )
    parser.add_argument(
        "--show-pytest-output",
        action="store_true",
        help="print captured pytest output; unavailable while JSON is sent to stdout",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="print the selected test targets as JSON without running pytest",
    )
    arguments = parser.parse_args(argv)
    if arguments.show_pytest_output and arguments.json_output == "-":
        parser.error("--show-pytest-output requires --json-output PATH")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.list_targets:
        print(json.dumps({
            "profile": arguments.profile,
            "test_targets": list(targets_for_profile(arguments.profile)),
        }, ensure_ascii=True, sort_keys=True, indent=2))
        return 0

    targets = targets_for_profile(arguments.profile)
    pytest_exit_code, pytest_output = _run_pytest(targets)
    evidence: list[dict[str, Any]] = []
    evidence_error: BaseException | None = None
    if pytest_exit_code == 0:
        try:
            evidence = collect_evidence()
        except BaseException as exc:  # Keep the machine-readable failure body-free.
            evidence_error = exc

    summary = build_summary(
        profile=arguments.profile,
        pytest_exit_code=pytest_exit_code,
        pytest_output=pytest_output,
        evidence=evidence,
        evidence_error=evidence_error,
    )
    if arguments.show_pytest_output:
        print(pytest_output, end="" if pytest_output.endswith("\n") else "\n")
    _write_json(summary, arguments.json_output)
    if arguments.json_output != "-":
        print(
            "[simulation] "
            f"profile={arguments.profile} outcome={summary['outcome']} "
            f"pytest_exit={pytest_exit_code}",
            flush=True,
        )
    return 0 if summary["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
