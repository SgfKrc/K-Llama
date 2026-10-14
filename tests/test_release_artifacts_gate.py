"""发版前发行物门的判据（票 13d / `CI-RELEASE-01` 第四条）。

判据在**临时 dist** 上验证，不依赖仓库里已构建的发行物 —— 因此这些用例可在任何环境跑。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_contract" / "release_artifacts_gate.py"

_spec = importlib.util.spec_from_file_location("release_artifacts_gate_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None, f"无法加载 {SCRIPT}"
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

CONTRACT = {
    "version": {"product": "9.9.9", "launcher": "8.8.8", "android_code": 7},
    "profile": {"default_node_role": "master"},
}


def _make_dist(root: Path, *, edge_version: str | None = "9.9.9",
               cuda_version: str | None = "9.9.9",
               launcher_version: str | None = "8.8.8",
               with_manifest: bool = True) -> Path:
    """构造一个最小发行物树；版本传 None 表示该档不写 version.txt。"""
    for name, value in (("QLH-Edge-Inference", edge_version),
                        ("QLH-Edge-Inference-CUDA", cuda_version),
                        ("QLH-Launcher", launcher_version)):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        if value is not None:
            (directory / "version.txt").write_text(value + "\n", encoding="utf-8")
        (directory / f"{name}.exe").write_bytes(b"stub")
    if with_manifest:
        manifest = root / "QLH-Launcher" / "manifest"
        manifest.mkdir(parents=True, exist_ok=True)
        (manifest / "install-manifest.json").write_text(
            json.dumps({"files": [{"path": "a"}, {"path": "b"}]}), encoding="utf-8")
    return root


def _failures(report) -> dict[str, str]:
    return {item["check"]: item["detail"] for item in report.failures}


def test_all_versions_match_passes_version_and_role_and_dependency(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path)
    report = gate.gate_windows(dist, CONTRACT)
    failures = _failures(report)
    # 启动 smoke 因缺真实 exe 会失败，这里只看版本/角色/依赖三项
    for check in ("版本 QLH-Edge-Inference/version.txt",
                  "版本 QLH-Edge-Inference-CUDA/version.txt",
                  "版本 QLH-Launcher/version.txt",
                  "角色 default_node_role",
                  "依赖 install-manifest.json"):
        assert check not in failures, failures


def test_missing_version_file_is_named_failure(tmp_path: Path) -> None:
    """发行物未声明版本（本仓 `dist/QLH-Edge-Inference` 的实际状态）必须具名失败。"""
    dist = _make_dist(tmp_path, edge_version=None)
    failures = _failures(gate.gate_windows(dist, CONTRACT))
    assert "版本 QLH-Edge-Inference/version.txt" in failures
    assert "文件缺失" in failures["版本 QLH-Edge-Inference/version.txt"]


def test_version_mismatch_is_named_failure(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path, cuda_version="0.0.1")
    failures = _failures(gate.gate_windows(dist, CONTRACT))
    assert "版本 QLH-Edge-Inference-CUDA/version.txt" in failures
    assert "9.9.9" in failures["版本 QLH-Edge-Inference-CUDA/version.txt"]


def test_missing_install_manifest_is_named_failure(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path, with_manifest=False)
    failures = _failures(gate.gate_windows(dist, CONTRACT))
    assert "依赖 install-manifest.json" in failures


def test_illegal_default_role_is_named_failure(tmp_path: Path) -> None:
    dist = _make_dist(tmp_path)
    contract = {"version": CONTRACT["version"], "profile": {"default_node_role": "boss"}}
    failures = _failures(gate.gate_windows(dist, contract))
    assert "角色 default_node_role" in failures


def test_launch_probe_can_be_disabled_but_is_not_silently_passed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """关闭探针时该面记为**失败**（未验），不静默通过。"""
    monkeypatch.setenv(gate.LAUNCH_PROBE_ENV, "0")
    dist = _make_dist(tmp_path)
    report = gate.gate_windows(dist, CONTRACT)
    smoke = {item["check"]: item for item in report.items
             if item["check"].startswith("启动 smoke")}
    assert len(smoke) == 3
    assert all(not item["passed"] for item in smoke.values())
    assert all("未验证" in item["detail"] for item in smoke.values())


def test_main_requires_explicit_platforms(capsys: pytest.CaptureFixture[str]) -> None:
    """未显式声明平台时门失败（未声明平台不计为未验，也不静默通过）。"""
    assert gate.main([]) == 1
    assert "必须用 --platforms 显式声明" in capsys.readouterr().err


def test_main_rejects_unknown_platform(capsys: pytest.CaptureFixture[str]) -> None:
    assert gate.main(["--platforms", "solaris"]) == 2
    assert "未知平台" in capsys.readouterr().err


def test_main_declares_only_platforms_are_checked(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """只检查已声明平台：仅声明 android 时不会去读 windows 发行物。"""
    dist = _make_dist(tmp_path)
    code = gate.main(["--platforms", "windows", "--dist", str(dist),
                      "--json", str(tmp_path / "evidence.json")])
    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["requested_platforms"] == ["windows"]
    assert [report["platform"] for report in evidence["reports"]] == ["windows"]
    assert code in (0, 1)
