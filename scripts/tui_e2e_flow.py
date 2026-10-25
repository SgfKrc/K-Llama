"""TUI 端到端 flow 的 **CLI 转发壳** —— 真正的实现在 `tests/tui_e2e_flow.py`。

为什么是壳：`scripts/*` 被 `.gitignore` **整体忽略 + 逐条白名单**（`.gitignore:169-222`），
且该文件属并行组在制品。若把引擎放这里，在补白名单行**之前**它不会入库 ⇒ 等于没交付。
用户 2026-09-24 裁定（`dec-fc91fb5c966c7786`）：引擎放 `tests/tui_e2e_flow.py`（可入库），
本文件只做转发。**两者命令行完全等价**：

```text
python scripts/tui_e2e_flow.py --mode assert --json-out build/tui-e2e/latest.json
python tests/tui_e2e_flow.py   --mode assert --json-out build/tui-e2e/latest.json
```

⚠️ **本文件当前不在库内**（被 `.gitignore:172 scripts/*` 忽略）。要让它入库，需要追加白名单行
`!scripts/tui_e2e_flow.py`（由用户/并行组执行；本文件不代改 `.gitignore`）。在补行之前请用
`tests/` 下的等价入口 —— 行为完全一致。

退出码（与剧本 `exit_codes` 对齐）：
`--mode assert` ⇒ `0` 全通过 / `1` 有 FAIL / `2` 环境缺失；
`--mode demo`   ⇒ `0` 全 PASS / `3` 有 DEGRADED / `1` 有 FAIL / `2` 环境缺失。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT), str(ROOT / "tests")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from tests.tui_e2e_flow import *
from tests.tui_e2e_flow import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
