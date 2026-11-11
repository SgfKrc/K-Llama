"""检查文档里提到的仓库内相对路径是否存在。

v2 退场清理删掉 103 个 tracked 文件后，`docs/` 里留下一批指向已删文件的引用。
本闸门只查**本仓库文档与工具目录**下的路径（首段是 `docs/` / `checks/` /
`scripts/` / `shim/` / `longctx/` / `verify/` / `control/`），并按引用性质分流：

1. **历史归档**：写成 ``<ref>:<path>``（如 ``slimarc/prototype:checks/x.py``）的，
   指向的是别的分支上的文件，要 ``git cat-file`` 才验证得了，跳过。
2. **其它树**：上游 llama.cpp（``src/`` / ``tools/`` / ``ggml-*``）、前身仓库
   （``qlh_core/`` / ``model/`` …）、本机临时目录（``.qlh_tmp/``）与不入库的
   ``_archive/``，首段不在上面七个目录里，自然不查。
3. **花括号**：``{a,b}`` 展开后逐个查；``*`` / ``?`` 通配跳过。
4. 代码块里的路径同样查；确有例外在行尾写 ``# noqa: asset`` 豁免。

根下的裸文件名（``chain_client.py``）不在范围内：那是已退场的 v2 模块，文档用
「归档在 ``slimarc/prototype``」交代口径，不构成可执行的路径声明。

用法：
    python scripts/check_asset_refs.py

退出码：有缺失为 1，干净为 0。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REPO_DIRS = ("docs", "checks", "scripts", "shim", "longctx", "verify", "control")
WORD = r"[\w.\-\u4e00-\u9fff]"
# 长后缀写在前：`.cpp` 不能被 `.c` 截断
ASSET = re.compile(
    rf"(?<![\w./:\-{WORD[1:-1]}])((?:{WORD}+/)+{WORD}+"
    r"\.(?:py|cpp|cc|c|h|ps1|sh|toml|json|md|diff|yaml|yml|txt))"
)
BRACES = re.compile(r"\{([^{}]+)\}")


def expand(line: str) -> list[str]:
    """把 `{a,b}` 展开成多行；嵌套也展开。"""
    m = BRACES.search(line)
    if not m:
        return [line]
    out = [line[: m.start()] + alt + line[m.end() :] for alt in m.group(1).split(",")]
    return [x for variant in out for x in expand(variant)]


def doc_paths() -> list[Path]:
    paths = [BASE / "README.md"]
    paths += sorted((BASE / "docs").rglob("*.md"))
    paths += sorted((BASE / "longctx").glob("*.md"))
    paths += [p for p in sorted(BASE.glob("*.md")) if p.name != "README.md"]
    seen: set[Path] = set()
    result = []
    for p in paths:
        if p.is_file() and p not in seen:
            seen.add(p)
            result.append(p)
    return result


def main() -> int:
    hits: dict[str, list[str]] = {}
    for path in doc_paths():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "noqa: asset" in line:
                continue
            for variant in expand(line):
                for m in ASSET.finditer(variant):
                    ref = m.group(1)
                    if ref.split("/", 1)[0] not in REPO_DIRS:
                        continue
                    if not (BASE / ref).exists():
                        hits.setdefault(ref, []).append(f"{path.relative_to(BASE).as_posix()}:{lineno}")

    if not hits:
        print("资产引用全部指得准。")
        return 0
    total = sum(len(v) for v in hits.values())
    print(f"资产引用问题 {len(hits)} 条路径 / {total} 处引用：")
    for ref in sorted(hits):
        print(f"  {ref}")
        for at in hits[ref]:
            print(f"        {at}")
    return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    raise SystemExit(main())
