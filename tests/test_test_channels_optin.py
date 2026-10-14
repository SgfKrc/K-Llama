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
