"""TUI 端到端 flow —— **F2 档：真后端 + 真模型**（默认不跑，需显式开关）。

计划：`docs/archive/tui/TUI端到端flow测试与答辩演示复用计划-2026-09-24.md`（具名票 `CORE-TUI-E2E-01`，§3.3 F2 档）。

与 F1 档（`tests/test_tui_e2e_flow.py`）的区别
---------------------------------------------
| | F1（`test_tui_e2e_flow.py`） | F2（本文件） |
|---|---|---|
| 后端 | 假后端（`FakeApi`） | **真 `api_server`**（`BackendSupervisor` 在 daemon 线程起 uvicorn） |
| 模型 | 零模型 | **真权重**（默认 `qwen3-5-2b`，`engine=llama_cpp`） |
| 门禁 | **必跑**（`unit` 通道 + `startup_matrix --profile tui`） | **默认 skip**：`QLH_RUN_REAL_MODEL_SMOKE=1` 才跑 |
| 并发 | 随默认 `-n 4` | **必须串行**（真模型吃内存，见下） |

为什么单列文件：F1 要能进 `unit` 门禁并**常绿**；真模型档耗时、吃内存、依赖工件，混在一起会让门禁变脆。

✅ **与计划 §3.3 对齐**：`scripts/run_test_channels.py:314-331` 的 **`smoke` 通道**会跑
`pytest tests -q -m real_model -n 0`（**串行**），且需 `QLH_RUN_REAL_MODEL_SMOKE=1`
（否则打印 "smoke channel skipped: set QLH_RUN_REAL_MODEL_SMOKE=1 to load real weights."）
⇒ 本文件的 `pytestmark = [pytest.mark.real_model, pytest.mark.slow]` **正好落进该通道**，
开关约定与 `tests/test_real_model_smoke.py:18-27` 同口径。
（`channels = ('unit', 'external')` 是**默认先跑的两条**；`--channel all|smoke` 才会带上 smoke 档。）

纪律（计划 §3.3 / §6.2）
- 缺开关 / 缺工件 / 缺 `llama_cpp` / 缺 `textual` ⇒ **一律 skip，不用 `xfail`、不静默 `pass`**；
- **不虚构证据**：断言只看真实回答文本，不因为"跑起来了"就判过；
- **不改生产代码路径**：全程只用公共 API（`tui_api.load_model` / `unload_model`、`Pilot`、`App.run_test`）。

跑法（串行）：

```powershell
$env:QLH_RUN_REAL_MODEL_SMOKE = "1"
.venv-test\Scripts\python.exe -m pytest tests\test_tui_e2e_flow_real.py -q -n 0
```

可用环境变量覆盖：`QLH_TUI_E2E_MODEL`（默认 `qwen3-5-2b`）、`QLH_TUI_E2E_ENGINE`（默认 `llama_cpp`）、
`QLH_TUI_E2E_PORT`（默认 `8000`）、`QLH_TUI_E2E_TIMEOUT`（默认 `180` 秒，等回答）。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src"), str(ROOT / "tests")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# F2 档 = 真模型档：注册的 marker（pytest.ini:24-30）
pytestmark = [pytest.mark.real_model, pytest.mark.slow]

pytest.importorskip("textual")

from tui_e2e_flow import wait_for  # noqa: E402

MODEL_ID = (os.environ.get("QLH_TUI_E2E_MODEL") or "qwen3-5-2b").strip()
ENGINE = (os.environ.get("QLH_TUI_E2E_ENGINE") or "llama_cpp").strip()
#: ★ 工件目录名可能**不等于**业务层注册的 `model_id`：例如业务 id `qwen2.5-0.5b`
#: 对应的 `model_path` 是 `models/qwen2.5-0.5b-instruct`。默认沿用 `MODEL_ID`
#: （旧行为不变），需要时用 `QLH_TUI_E2E_ARTIFACT` 显式给出目录名 / gguf 前缀。
ARTIFACT = (os.environ.get("QLH_TUI_E2E_ARTIFACT") or MODEL_ID).strip()
HOST = "127.0.0.1"
#: 默认用**独占端口**（8231）而不是生产默认 8000：避免"复用开发时已在跑的后端"而
#: 让"F2 真起后端 + 真加载模型"这一步被静默跳过（那会让断言失去意义）。
PORT = int(os.environ.get("QLH_TUI_E2E_PORT") or "8231")
REPLY_TIMEOUT = float(os.environ.get("QLH_TUI_E2E_TIMEOUT") or "180")
PROMPT = (os.environ.get("QLH_TUI_E2E_PROMPT") or "用一句话说明：1+1 等于几？").strip()


def _enabled() -> bool:
    return (os.environ.get("QLH_RUN_REAL_MODEL_SMOKE") or "").strip() == "1"


def _model_artifact_present(model_id: str) -> bool:
    """模型工件门：GGUF 目录 / 单个 gguf 文件 / safetensors 目录，任一存在即可。

    不做"猜路径"式的强行加载：**缺工件一律 skip**（`tests/test_llama_relay_entry.py:8,28,33,58`
    的既有写法）。
    """
    models_dir = ROOT / "models"
    if not models_dir.is_dir():
        return False
    if (models_dir / f"{model_id}-gguf").is_dir():
        return True
    if (models_dir / model_id).is_dir():
        return True
    return any(models_dir.glob(f"{model_id}*.gguf"))


def _runtime_import_hint(engine: str) -> str:
    """按 engine 只检查**它真正需要**的运行时，避免"engine 无关的硬依赖"。

    ★ 2026-09-26（Surface 实测暴露）：原实现**无条件** `import llama_cpp`，于是
    `QLH_TUI_E2E_ENGINE=pytorch` 这一合法组合也被判"缺依赖"而 skip —— 而它的提示语
    自己写的却是"（**engine=llama_cpp** 的真模型档）"。没有 GGUF/llama.cpp 的机器
    （例如 Surface：PyPI 上 `llama-cpp-python` 只有 sdist，需编译）因此永远跑不到 F2。
    """
    if engine == "llama_cpp":
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return "需要 llama_cpp（engine=llama_cpp 的真模型档）"
        return ""
    if engine == "pytorch":
        try:
            import torch  # noqa: F401
        except ImportError:
            return "需要 torch（engine=pytorch 的真模型档）"
        return ""
    return ""


def _skip_reason() -> str:
    if not _enabled():
        return "设置 QLH_RUN_REAL_MODEL_SMOKE=1 才运行真模型 TUI E2E（F2 档）"
    if not _model_artifact_present(ARTIFACT):
        return f"需要真实模型工件 models/{ARTIFACT}(-gguf)（真机工件门，不虚构证据）"
    return _runtime_import_hint(ENGINE)


def _run(coro):
    return asyncio.run(coro)


def test_real_backend_and_model_available():
    """前置：真后端能起来 + 真模型能加载（把"环境缺失"与"功能坏了"分开）。

    本用例只做**环境与就绪**层面的断言：后端可达、模型加载成功且后端报 `model_loaded`。
    值层面的断言在下一个用例（真链路聊天）。
    """
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    async def _main():
        from tui_api import ApiClient, load_model
        from tui_backend import BackendSupervisor

        supervisor = BackendSupervisor(HOST, PORT)
        assert supervisor.ensure_ready(), (
            f"真后端未就绪: error={supervisor.error!r} status={supervisor.status_message!r}")
        api = ApiClient(host=HOST, port=PORT, timeout=120.0)
        try:
            load_model(api, MODEL_ID, engine=ENGINE)
            status = api.get("/status")
            assert isinstance(status, dict) and status, f"后端 /status 返回异常: {status!r}"
            assert status.get("model_loaded") is True, (
                f"真模型未加载: model_loaded={status.get('model_loaded')!r} "
                f"model_name={status.get('model_name')!r}")
        finally:
            supervisor.stop()

    _run(_main())


def test_real_tui_chat_roundtrip():
    """★ F2 主判据：真后端 + 真模型，TUI 聊天发一条 ⇒ 拿到**非空**回答。

    断言只看真实回答文本（不因为"链路跑起来了"就判过）；用 `wait_for` 轮询等回答，
    失败时打印已渲染文本便于定位（不吞证据）。
    """
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    async def _main():
        from textual.widgets import Input, Static

        from tui_api import ApiClient, load_model, unload_model
        from tui_backend import BackendSupervisor
        from tui_textual import KoakumaApp, MainScreen

        supervisor = BackendSupervisor(HOST, PORT)
        assert supervisor.ensure_ready(), (
            f"真后端未就绪: error={supervisor.error!r} status={supervisor.status_message!r}")
        control = ApiClient(host=HOST, port=PORT, timeout=120.0)
        try:
            load_model(control, MODEL_ID, engine=ENGINE)

            app = KoakumaApp(ApiClient(host=HOST, port=PORT, timeout=120.0), interval=30)
            async with app.run_test(size=(120, 40)) as pilot:
                app.show_main()
                await wait_for(pilot, lambda: isinstance(app.screen, MainScreen))
                # ⚠️ `ContentSwitcher` **只挂载当前屏** ⇒ `MainScreen` 就位 ≠ `#chat-pane` 已挂载
                #    （计划 §3.4 已记该竞态；F1 档因为先等 `#nav` 的 10 个条目才规避了它）。
                #    这里必须用 `wait_for` 兜住 `NoMatches`，不能直接 `query_one`。
                assert await wait_for(
                    pilot, lambda: bool(app.screen.query("#chat-pane"))), "聊天屏 #chat-pane 未挂载"
                pane = app.screen.query_one("#chat-pane")
                assert await wait_for(pilot, lambda: bool(pane.query(Input))), "聊天输入框未挂载"

                box = pane.query_one(Input)
                box.value = PROMPT
                box.focus()
                await pilot.press("enter")

                def _assistant_body(text: str) -> str:
                    """剥掉角色标签，取 assistant 的回答**正文**。

                    ⚠️ 不能只看 `#chat-log` 的整体长度 —— 它含 prompt 回显与 `assistant` 标签，
                    那样写会把"回答为空"误判成通过（本档开发时实测踩过：渲染是
                    `'你 …\\nassistant '`，整体长度够 8 字符，但正文是空的）。
                    """
                    parts = re.split(r"assistant", text)
                    return parts[-1].strip() if len(parts) > 1 else ""

                def _answer() -> str:
                    return _assistant_body(str(pane.query_one("#chat-log", Static).render()))

                got = await wait_for(pilot, lambda: len(_answer()) >= 2, timeout=REPLY_TIMEOUT)
                answer = _answer()
                if not got:
                    raise AssertionError(
                        f"真模型在 {REPLY_TIMEOUT:.0f}s 内未给出非空回答；"
                        f"已渲染文本（前 400 字）: "
                        f"{str(pane.query_one('#chat-log', Static).render())[:400]!r}")
                assert answer != PROMPT, f"回答不应是 prompt 回显: {answer!r}"
                assert PROMPT not in answer, f"回答不应包含 prompt 回显: {answer!r}"

                # 备注（**不在本档断言**）：实测 `/status` 的 `model_name` 报的是 `Qwen/Qwen3-0.6B`，
                # 而 `active_model_id` 是请求的 `qwen3-5-2b` —— 两者不一致是**既有缺陷**
                # （GGUF 加载走静态兜底路径所致，已登记在案），与本档"TUI 端到端是否通"无关，
                # 故不在此处断言，避免让 smoke 通道因他人缺陷变红。

            unload_model(control)
        finally:
            supervisor.stop()

    _run(_main())
