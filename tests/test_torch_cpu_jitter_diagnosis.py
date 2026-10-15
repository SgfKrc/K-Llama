"""tests/test_torch_cpu_jitter_diagnosis.py — CPU 抖动诊断工具的回归。

对应 `TORCH-HW-ADMIT-01` 的稳定性门缺口诊断：本工具只**诊断**（不改 CV 门、不放宽阈值）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import torch_cpu_jitter_diagnosis as diag


def test_stats_reports_population_cv_and_spread():
    flat = diag._stats([10.0, 10.0, 10.0, 10.0])
    assert flat["cv"] == 0.0
    assert flat["spread_pct"] == 0.0

    spread = diag._stats([8.0, 12.0])
    assert spread["mean_ms"] == 10.0
    assert spread["cv"] == 0.2          # pstdev(8,12)=2 ⇒ 2/10
    assert spread["spread_pct"] == 40.0


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="脚本的 CPU 亲和性只实现 Windows（_set_affinity 对非 win32 直接返回 False）",
)
def test_single_mode_applies_affinity_and_emits_json():
    """`--mode single` 必须**真的报告**亲和性结果并输出 JSON。

    `affinity_applied=false` 是踩过的真 bug（`GetCurrentProcess` 的 HANDLE 被 ctypes 默认
    按 `c_int` 截断，`SetProcessAffinityMask` 静默失败）⇒ 这里把它固化成回归点。
    """
    script = Path(diag.__file__).resolve()
    done = subprocess.run([sys.executable, str(script), "--mode", "single", "--cpu", "0",
                           "--repeats", "3", "--warmup", "1", "--loops", "2"],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False, timeout=300)
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    assert payload["affinity_applied"] is True
    assert len(payload["samples_ms"]) == 3
    assert payload["mask"] == 1


def test_tool_is_diagnostic_only():
    """纪律：工具自带「只诊断」声明，且**不得**出现 CV 门/阈值常量。"""
    source = Path(diag.__file__).read_text(encoding="utf-8")
    assert "只诊断" in source
    assert "max_runtime_cv" not in source


def test_windows_topology_probe_is_optional_and_structured(monkeypatch):
    monkeypatch.setattr(diag.sys, "platform", "linux")
    assert diag._windows_processor_topology() is None
