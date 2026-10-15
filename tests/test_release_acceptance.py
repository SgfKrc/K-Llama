"""Release acceptance cannot pass on one platform, a partial report or synthetic relay self-check."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_acceptance_test", ROOT / "scripts/package_contract/release_acceptance.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def reports(tmp_path):
    data = (ROOT/"release-contract.json").read_bytes()
    output = {}
    for platform in gate.PLATFORMS:
        path = tmp_path / f"{platform}.json"
        obj = {"schema_version": 2, "gate": "release-artifacts", "passed": True, "status": "passed",
               "source_revision": "a"*40, "required_revision": "a"*40, "revision_verified": True,
               "requested_platforms": [platform], "contract_sha256": hashlib.sha256(data).hexdigest(),
               "contract_version": json.loads(data)["version"], "reports": [
                   {"platform": platform, "passed": True, "status": "passed", "items": [
                       {"check": name, "passed": True, "status": "passed",
                        "evidence": {"source_revision": "a"*40, "revision_verified": True}} for name in sorted(gate.MANDATORY[platform])]}]}
        path.write_text(json.dumps(obj))
        output[platform] = path
    return output


def test_all_platforms_still_require_actual_product_bundle(reports, tmp_path):
    result = gate.summarize_release(reports, tmp_path / "missing-bundle.json", "a"*40)
    assert not result["passed"]
    assert all(item["passed"] for item in result["platforms"].values())
    assert not result["product"]["passed"]


@pytest.mark.parametrize("change", [
    lambda obj: obj.update(passed=False),
    lambda obj: obj.update(status="unverified"),
    lambda obj: obj.update(schema_version=1),
    lambda obj: obj.update(source_revision="b"*40),
    lambda obj: obj.update(revision_verified=False),
    lambda obj: next(item for item in obj["reports"][0]["items"] if item["check"].endswith(".revision"))["evidence"].update(source_revision="b"*40),
    lambda obj: obj.update(contract_sha256="f"*64),
    lambda obj: obj.update(requested_platforms=["linux"]),
    lambda obj: obj.update(reports=[]),
    lambda obj: obj["reports"][0].update(items=[]),
    lambda obj: obj["reports"][0]["items"].pop(),
    lambda obj: obj["reports"][0]["items"][0].update(status="unverified", passed=False),
    lambda obj: obj["reports"][0]["items"].append(obj["reports"][0]["items"][0]),
])
def test_partial_or_tampered_report_cannot_pass(reports, tmp_path, change, monkeypatch):
    path = reports["windows"]
    obj = json.loads(path.read_text())
    change(obj)
    path.write_text(json.dumps(obj))
    monkeypatch.setattr(gate.product_gate, "validate_bundle", lambda *args: {"passed": True})
    result = gate.summarize_release(reports, tmp_path/"bundle.json", "a"*40)
    assert not result["passed"]
    assert not result["platforms"]["windows"]["passed"]


def test_no_platform_may_be_omitted(reports, tmp_path):
    del reports["android"]
    result = gate.summarize_release(reports, tmp_path/"bundle.json", "a"*40)
    assert not result["passed"]


def test_complete_reports_require_product_validator_success(reports, tmp_path, monkeypatch):
    monkeypatch.setattr(gate.product_gate, "validate_bundle", lambda *args: {"passed": True})
    result = gate.summarize_release(reports, tmp_path/"bundle.json", "a"*40)
    assert result["passed"]


def test_cli_duplicate_platform_rejected(tmp_path):
    with pytest.raises(SystemExit) as result:
        gate.main(["--report", "windows=a.json", "--report", "windows=b.json", "--bundle", "b.json",
                   "--require-revision", "a"*40, "--json", str(tmp_path/"out.json")])
    assert result.value.code != 0
