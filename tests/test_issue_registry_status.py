"""钉住 docs/已知问题记录.md 的状态一致性（#93 遗留清理暴露的清单漂移）。

背景：`#51`/`#52`/`#53`/`#58`/`#68`/`#79`/`#91`/`#92` 都是**已修复**却滞留在「未闭环」区，
而 `#54`/`#60` 各自被登记了两次。清单不可信 ⇒ 已修的缺陷会被重新排查（本轮就白查了一次
`#53`，它的正文说「产品侧仍缺」，而 `engine_host.py` 自 2026-10-08 起已走共用 shim 定位）。

规则：
* R2（失败级）：同一票号不得在同一文档里登记两次 —— 否则「见 #NN」有歧义；
* R1（报告级）：未闭环区里自述已修复的条目，**允许现状但不得新增**；迁移完成后应把
  BASELINE 收紧到 0。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_issue_registry_status.py"
REGISTRY = REPO_ROOT / "docs" / "已知问题记录.md"

#: 迁移前的已知基数（2026-10-11：5 条）。迁移完成后改为 0。
R1_BASELINE = 5


def _load_checker():
    spec = importlib.util.spec_from_file_location("qlh_issue_registry_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _analyse() -> dict:
    checker = _load_checker()
    return checker.analyse(REGISTRY.read_text(encoding="utf-8").splitlines())


def test_registry_exists_and_has_both_sections():
    text = REGISTRY.read_text(encoding="utf-8")
    assert "## 1. 未闭环" in text
    assert "## 2. 已修复" in text


def test_no_duplicate_ticket_ids():
    """R2：票号重复会让「见 #NN」产生歧义 ⇒ 必须为空。"""
    report = _analyse()
    assert report["duplicate_ticket_ids"] == [], report["duplicate_ticket_ids"]


def test_fixed_but_still_open_does_not_grow():
    """R1：允许既有基数，但不得新增（防清单继续漂移）。"""
    report = _analyse()
    offenders = report["fixed_but_still_open"]
    assert len(offenders) <= R1_BASELINE, [item["ticket"] for item in offenders]


def test_cli_reports_machine_readable_json():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(REPO_ROOT),
    )
    payload = json.loads(result.stdout)
    assert payload["duplicate_ticket_ids"] == []
    assert isinstance(payload["fixed_but_still_open"], list)
    # R2 空 ⇒ 退出码 0（R1 是报告级，不影响退出码）
    assert result.returncode == 0
