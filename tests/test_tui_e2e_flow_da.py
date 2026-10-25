"""TUI 端到端 flow —— **D-a 档：本机 loopback 多段**（演示降级；缺工件则 skip）。

计划：`docs/archive/tui/TUI端到端flow测试与答辩演示复用计划-2026-09-24.md`（具名票 `CORE-TUI-E2E-01`，§4.2 D-a）。

与另外两档的关系
- **F1**（`tests/test_tui_e2e_flow.py`）：假后端、零模型 ⇒ **必跑**，进 `unit` 通道 + `startup_matrix`。
- **F2**（`tests/test_tui_e2e_flow_real.py`）：真后端 + 真模型 ⇒ 默认 skip（`QLH_RUN_REAL_MODEL_SMOKE=1`）。
- **D-a**（本文件）：**本机 loopback 多段** —— 一个 relay 段跑在 `127.0.0.1`。这是**无多节点时的
  演示降级**（计划 §4.2 D-a），需 keep-head shim + 裁层段工件；缺任一 ⇒ 整档 **skip**（不判失败）。

⚠️ **边界（不得越界引用）**：剧本里 `topology=single_host_loopback`、`physical_dual_host=false`、
`real_model_loaded=false`，报告的收尾行与 JSON 都强制打印这三个字段。本档只证明
**「TUI ↔ 本机 relay 段」这条链路可用**（真链路证据：`scripts/relay_health.py` 判该段健康 +
hidden **逐字节**往返且**非恒等**）。**不得**据此宣称跨机分布式或真模型已加载（计划 §6.1 风险 6）。

⚠️ 本档**不进** `startup_matrix` 的 `TUI_TESTS`：它要起真实段服务（秒级到十几秒）且依赖工件，
放进启动档会让门禁变脆。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src"), str(ROOT / "tests")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# ① 缺 UI 依赖 ⇒ 整模块跳过（照抄 tests/test_tui_textual.py:46）
pytest.importorskip("textual")

from tui_e2e_flow import PASS, SKIP, load_flow, run_flow  # noqa: E402

pytestmark = [pytest.mark.slow]

DA_FLOW = ROOT / "fixtures" / "tui_e2e_flow_da.json"
FLOW = load_flow(DA_FLOW)
STEP_IDS = [step["id"] for step in FLOW["steps"]]


def _missing_artifacts() -> list[str]:
    """D-a 档的工件门：keep-head shim + 裁层段工件。

    照仓库惯例（`tests/test_llama_keep_head.py:46-56`）：**缺工件就 skip，不虚构证据**。
    """
    missing: list[str] = []
    for step in FLOW["steps"]:
        for action in step.get("actions", []):
            if action.get("kind") != "start_relay_service":
                continue
            for key in ("shim", "model"):
                relative = str(action.get(key) or "")
                if relative and not (ROOT / relative).is_file():
                    missing.append(relative)
    return missing


def _skip_if_missing() -> None:
    missing = _missing_artifacts()
    if missing:
        pytest.skip(f"需要 D-a 档工件（真机工件门，不虚构证据）: {missing}；"
                    "段工件由 scripts/cut_layers.py --keep-head/--k/--end 生成")


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_da_step(step_id: str):
    """把 D-a 剧本的每个 step 转成一条用例（与 `--mode assert` 同源、同一引擎）。"""
    _skip_if_missing()

    result = run_flow(mode="assert", flow_path=DA_FLOW, only_step=step_id)

    assert not result.environment_missing, result.env_reason
    assert result.steps, f"{step_id}: 应至少记录一条步骤结果"
    record = result.steps[-1]
    assert record.step_id == step_id, f"最后一条记录应为 {step_id}，实际 {record.step_id}"
    assert record.status in (PASS, SKIP), f"{step_id}: {record.status} — {record.reason}"
    assert result.exit_code() == 0, f"{step_id}: 退出码应为 0（SKIP 不计失败）"


def test_da_full_sequence_once():
    """整条 D-a 剧本顺序跑一遍：起段服务 → 健康检查 → 真链路往返 → TUI 开关 → 停服务。"""
    _skip_if_missing()

    result = run_flow(mode="assert", flow_path=DA_FLOW)

    assert [s.step_id for s in result.steps] == STEP_IDS, (
        f"应跑完全部 {len(STEP_IDS)} 步: {[s.step_id for s in result.steps]}")
    failed = [f"{s.step_id}: {s.reason}" for s in result.failed]
    assert not failed, "；".join(failed)
    assert result.exit_code() == 0


def test_da_topology_is_honestly_labeled():
    """★ 硬约束（计划 §6.1 风险 6）：D-a 档必须如实标注**单机 loopback**，不是物理双机。

    「该红必须红」：若有人把 `physical_dual_host` 改成 `true`（或把 topology 写成跨机），
    这条立刻红 —— 那种改动会把"本机多进程"伪装成"跨机分布式"。
    """
    assert FLOW["topology"] == "single_host_loopback"
    assert FLOW["physical_dual_host"] is False
    assert FLOW["real_model_loaded"] is False
    note = str(FLOW.get("topology_note", ""))
    assert "不是物理双机" in note, f"topology_note 必须显式声明非物理双机: {note!r}"
    assert "不得" in note, f"topology_note 必须含「不得据此宣称」的约束: {note!r}"


def test_da_does_not_enter_startup_matrix():
    """★ D-a 档**不得**被塞进 `startup_matrix` 的 TUI 档（它要起真实段服务且依赖工件）。

    「该红必须红」：把本文件加进 `TUI_TESTS` 会让启动档在缺工件/慢机器上变红。
    """
    text = (ROOT / "scripts" / "startup_matrix.py").read_text(encoding="utf-8")
    assert "test_tui_e2e_flow_da" not in text, (
        "D-a 档不应进 startup_matrix 的 TUI_TESTS（要起真实段服务 + 依赖工件）")
