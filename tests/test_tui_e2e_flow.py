"""TUI 端到端 flow 测试（F1 档：假后端、零模型）。

计划来源：`docs/archive/tui/TUI端到端flow测试与答辩演示复用计划-2026-09-24.md`（具名票 `CORE-TUI-E2E-01`）。
step 引擎与剧本：`tests/tui_e2e_flow.py` + `fixtures/tui_e2e_flow.json`（本文件只做 pytest 入口）。

**第一阶段（探针）已完成** —— 计划 §6.1 风险 2 要求「先交 1 个探针用例，确认稳定后再铺全量剧本」。
计划 §1.3 记录过「仓库现有测试从未用过鼠标注入」（`tests/` 搜 `pilot.click|hover|mouse_down`
→ 无匹配），因此该能力此前**只有代码层推导、未实测**。实测结论（2026-09-24，textual 8.2.8，
headless，连跑 8 次全绿）：

| 探明的行为 | 实测结果 |
|---|---|
| `pilot.click("#nav ListItem#nav-<id>")` 切屏 | ✅ 稳定；`ContentSwitcher.current` 与 `nav.index` 同步 |
| 全 10 屏由鼠标点击到达 | ✅ 全部可用（非「只对某项生效」） |
| `click` 命中目标时的返回值 | `True`（`bool`） |
| `click(selector, offset=...)` 落点在**屏幕内**但**不在目标 widget 上** | 返回 **`False`**（不抛错）⇒ 返回值本身即断言 |
| `click(selector, offset=...)` 落点**超出可见屏幕区域** | 抛 `textual.pilot.OutOfBounds`（**不是** `False`） |
| `click` 目标 selector **不存在** | 抛 `textual.css.query.NoMatches`（不是静默成功） |

档位划分（计划 §3.3）：
- **F1 假后端（零模型）**：`FakeApi` + 仅需 `textual`/pytest ⇒ 本文件，**纳入门禁**。
- F2 真后端 + 真模型：需模型工件 + 隔离 venv + 显式开关（`QLH_RUN_REAL_MODEL_SMOKE=1`）⇒ 不在此文件。

纪律（计划 §3.3 / §6.2）：缺 UI 依赖 ⇒ 整模块 `importorskip`（**跳过而非失败**）；
**一律 skip，不用 `xfail`、不静默 `pass`**；不放宽任何既有门禁；不宣称真模型已加载 /
双机物理分布式（计划 §6.1 风险 6）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src"), str(ROOT / "tests")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# ① 缺 UI 依赖 ⇒ 整模块跳过（照抄 tests/test_tui_textual.py:46）
pytest.importorskip("textual")

from tui_api import ApiClient  # noqa: E402
from tui_e2e_flow import (  # noqa: E402
    PASS,
    FakeApi,
    load_flow,
    run_flow,
    wait_for,
)
from tui_textual import KoakumaApp, MainScreen  # noqa: E402

FLOW = load_flow()
NAV_IDS = list(FLOW["nav_ids"])
STEP_IDS = [step["id"] for step in FLOW["steps"]]


def _run(coro):
    return asyncio.run(coro)


def _make_app(api=None):
    app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5), interval=30)
    app.api = api if api is not None else FakeApi()
    return app


async def _enter_main(pilot, app):
    """等主界面就绪。

    ⚠️ `ContentSwitcher` **只挂载当前屏**，所以不能用 `#models-table` 之类判断就绪
    （未显示时根本查不到）。这里用**常驻侧栏** `#nav` 的条目数判断 —— 不依赖数据表。
    """
    app.show_main()
    await wait_for(pilot, lambda: isinstance(app.screen, MainScreen))
    await wait_for(pilot, lambda: len(list(app.screen.query("#nav ListItem"))) == len(NAV_IDS))
    return app.screen


# ------------------------------------------------------------------ 探针（计划 §6.1 风险 2）


def test_probe_pilot_click_switches_page():
    """★ 探针：`Pilot` 鼠标注入能否稳定切屏 + 同步侧栏高亮。

    计划 §1.3 记载 `pilot.click` 在本仓库**零先例**、标「未实测」。本用例是该点的落地证据：
    - 用 `selector`（不是绝对坐标）定位侧栏项 ⇒ 换窗口尺寸/主题也不会漂移；
    - `pilot.click` 的返回值本身即断言（落点不在目标 widget 上时返回 `False`）；
    - 连点多个不同页，验证不是「碰巧切到某一屏」。
    """
    from textual.widgets import ContentSwitcher, ListView

    async def _main():
        app = _make_app()
        async with app.run_test() as pilot:
            screen = await _enter_main(pilot, app)
            nav = screen.query_one("#nav", ListView)
            content = screen.query_one("#content", ContentSwitcher)

            assert content.current == "page-chat", "起始屏应为聊天屏"
            assert nav.index == 0, "侧栏高亮应从第 0 项开始"

            # ① 计划点名的那个点击式：selector 定位到第 4 项（分布式）
            landed = await pilot.click("#nav ListItem#nav-cluster")
            await wait_for(pilot, lambda: content.current == "page-cluster")
            assert content.current == "page-cluster", (
                f"鼠标点击未切屏: current={content.current!r}, click 返回值={landed!r}")
            assert nav.index == 3, f"侧栏高亮未同步: index={nav.index}"

            # ② 换一屏，确认不是「只对某项生效」
            landed2 = await pilot.click("#nav ListItem#nav-nodes")
            await wait_for(pilot, lambda: content.current == "page-nodes")
            assert content.current == "page-nodes", (
                f"第二次点击未切屏: current={content.current!r}, click 返回值={landed2!r}")
            assert nav.index == 4, f"侧栏高亮未同步: index={nav.index}"

            # ③ 点回第 1 屏（回归，确认导航不是单向的）
            await pilot.click("#nav ListItem#nav-chat")
            await wait_for(pilot, lambda: content.current == "page-chat")
            assert content.current == "page-chat", "点回首屏失败"
            assert nav.index == 0, f"侧栏高亮未回到第 0 项: index={nav.index}"

            # ④ 落点语义：点一个**不存在**的 selector 必须抛 NoMatches（不能静默当成功）
            from textual.css.query import NoMatches
            from textual.pilot import OutOfBounds

            with pytest.raises(NoMatches):
                await pilot.click("#nav ListItem#nav-nonexistent")

            # ⑤a offset 语义之一：offset 是**相对目标 widget** 的坐标；落点在**屏幕内**、
            #     但**不在目标 widget 上** ⇒ 返回 False（不抛错）⇒ 返回值本身即断言。
            item = screen.query_one("#nav ListItem#nav-status")
            sideways = (item.size.width + 4, 0)      # 横向移到内容区，仍在可见屏幕内
            assert await pilot.click("#nav ListItem#nav-status", offset=sideways) is False, (
                f"落点不在 ListItem 上时应返回 False（offset={sideways}, size={item.size}）")

            # ⑤b offset 语义之二：落点**超出可见屏幕区域** ⇒ 抛 OutOfBounds（不是返回 False）。
            #     ⇒ 剧本里凡用 offset 都必须先确认落点在屏幕内，并把返回值当断言。
            with pytest.raises(OutOfBounds):
                await pilot.click("#nav ListItem#nav-status", offset=(4000, 4000))

    _run(_main())


def test_probe_click_reaches_every_page():
    """★ 探针扩展：**每个**页面都能由鼠标点击到达（全 10 屏覆盖）。

    单独一屏能点开不足以支撑剧本 —— 剧本要在任意相邻步之间跳转。这里断言
    「点第 i 项 ⇒ `content.current == page-<id>` 且 `nav.index == i`」对**全部 10 项**成立。
    """
    from textual.widgets import ContentSwitcher, ListView

    async def _main():
        app = _make_app()
        async with app.run_test() as pilot:
            screen = await _enter_main(pilot, app)
            nav = screen.query_one("#nav", ListView)
            content = screen.query_one("#content", ContentSwitcher)

            reached = []
            for index, item_id in enumerate(NAV_IDS):
                page = "page-" + item_id.removeprefix("nav-")
                await pilot.click(f"#nav ListItem#{item_id}")
                ok = await wait_for(pilot, lambda page=page: content.current == page)
                assert ok, f"点 {item_id} 未到 {page}（current={content.current!r}）"
                assert nav.index == index, (
                    f"点 {item_id} 后侧栏高亮应为 {index}，实际 {nav.index}")
                reached.append(page)

            assert len(set(reached)) == len(NAV_IDS), f"应到达 {len(NAV_IDS)} 个互不相同的页面: {reached}"

    _run(_main())


# ------------------------------------------------------------------ 剧本驱动（逐条转用例）


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_flow_step(step_id: str):
    """把 `fixtures/tui_e2e_flow.json` 的每个 step 转成一条用例（与 `--mode assert` 同源）。

    引擎会在该用例内**重放本步所属会话的 actions**（只重放，不判定中间步的 expect），
    然后判定本步的 expect ⇒ 每条用例独立、可并行、失败能定位到具体 step。
    """
    result = run_flow(mode="assert", only_step=step_id)

    assert not result.environment_missing, result.env_reason
    assert result.steps, f"{step_id}: 应至少记录一条步骤结果"
    record = result.steps[-1]
    assert record.step_id == step_id, f"最后一条记录应为 {step_id}，实际 {record.step_id}"
    assert record.status == PASS, f"{step_id} 未通过: {record.reason}"
    assert result.exit_code() == 0, f"{step_id}: 退出码应为 0"


def test_flow_full_sequence_in_one_session():
    """整条剧本按顺序跑完（与 `--mode demo` 同一份 step 列表、同一执行器）。

    逐条用例覆盖"每步都对"，本用例额外覆盖"**连起来也对**"（步骤间的状态传递、
    会话切分 `new_app` 的边界）。F1 档共享同一份剧本 ⇒ 不需要单独的步骤表。
    """
    result = run_flow(mode="assert")

    assert not result.environment_missing, result.env_reason
    assert [s.step_id for s in result.steps] == STEP_IDS, (
        f"应跑完全部 {len(STEP_IDS)} 步: {[s.step_id for s in result.steps]}")
    failed = [f"{s.step_id}: {s.reason}" for s in result.failed]
    assert not failed, "；".join(failed)
    assert result.exit_code() == 0


def test_demo_mode_exit_codes_never_green_on_failure():
    """★ 硬约束（计划 §4.1）：`--mode demo` 只要 DEGRADED/FAIL **绝不返回 0**。

    「该红必须红」：把 demo 的退出码改成"有 DEGRADED 也算 0"（或让 DEGRADED 静默变 PASS）
    会让这条立刻红 —— 那正是"演示变绿被误读为验收通过"的入口。
    """
    from tui_e2e_flow import DEGRADED, EXIT_DEGRADED, EXIT_FAIL, EXIT_OK, FlowResult, StepResult

    ok = FlowResult(mode="demo", steps=[StepResult("s1", "c", PASS)])
    assert ok.exit_code() == EXIT_OK

    degraded = FlowResult(mode="demo", steps=[StepResult("s1", "c", DEGRADED, "理由")])
    assert degraded.exit_code() == EXIT_DEGRADED, "demo 有 DEGRADED 不得返回 0"

    broken = FlowResult(mode="demo", steps=[StepResult("s1", "c", "FAIL", "理由")])
    assert broken.exit_code() == EXIT_FAIL

    # assert 模式：DEGRADED 不出现；环境缺失一律 2（等价 skip，不判失败）
    missing = FlowResult(mode="assert", environment_missing=True)
    assert missing.exit_code() == 2
    assert "环境缺失" in load_flow()["exit_codes"]["assert"]["2"]
