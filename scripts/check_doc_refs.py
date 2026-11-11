"""检查 docs/ 里交叉引用是否指得准。

三类引用，三类判据：

1. `§3` / `§3.5`  —— 本文内引用，本文必须有对应编号的标题。
2. `X.md §3`     —— 跨文档引用，目标文件必须有该章节。markdown 链接形式
                    （``[`X.md`](X.md) §3``）也算，判定方法是看 `§` 左边最近出现的
                    哪个 `.md`。
3. `X.md:123`    —— 行号引用。行号会随改动失效（一个文件删 22 行，三份文档的引用就
                    全漂了），只统计不判错，留着提醒改成章节引用。代码文件的
                    `foo.c:154` 不算在内。

用法：
    python scripts/check_doc_refs.py               # 只报问题
    python scripts/check_doc_refs.py --line-refs   # 额外列出所有行号引用

退出码：有问题为 1，干净为 0。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

# 标题里的章节号：## 1. / ## §1 / ### 1.2 / #### 1.2.3（§ 前缀与点号都可选）
HEADING = re.compile(r"^#{2,4}\s+§?\s*(\d+(?:\.\d+)*)", re.M)
# 一个 § 引用本身
SECTION = re.compile(r"§\s*(\d+(?:\.\d+)*)")
# 文件名：跨文档引用的目标，允许中文与连字符
MD_NAME = re.compile(r"([\w.\-\u4e00-\u9fff]+\.md)")
# 行号引用：只认 .md 的，代码文件不在此列
LINE_REF = re.compile(r"(?P<doc>[\w.\-\u4e00-\u9fff]+\.md):(?P<line>\d+)")


def headings(text: str) -> set[str]:
    return {m.group(1) for m in HEADING.finditer(text)}


def load() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(DOCS.rglob("*.md"))}


def main() -> int:
    show_line_refs = "--line-refs" in sys.argv
    docs = load()
    sections = {name: headings(text) for name, text in docs.items()}
    problems: list[str] = []
    line_refs: list[str] = []

    for name, text in docs.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in SECTION.finditer(line):
                num = m.group(1)
                # 同一行里 `§` 左边最近的 .md 就是目标；没有就是本文
                names = MD_NAME.findall(line[: m.start()])
                target = names[-1] if names else name
                if target not in sections:
                    continue  # 指向仓库外的文档，不判
                if num not in sections[target]:
                    problems.append(f"{name}:{lineno}: {target} 没有 §{num}")

        for m in LINE_REF.finditer(text):
            if m.group("doc") not in docs:
                continue
            # 绝对路径（`H:\slimarc-proto\DESIGN.md:80`）指向仓库外的同名文件，不判
            if m.start() > 0 and text[m.start() - 1] in "\\/":
                continue
            line_refs.append(f"{name}: {m.group('doc')}:{m.group('line')}")

    if line_refs:
        print(f"行号引用 {len(line_refs)} 处（会随改动失效，建议改成 `文件.md §n`）：")
        for item in line_refs if show_line_refs else line_refs[:5]:
            print(f"  {item}")
        if not show_line_refs and len(line_refs) > 5:
            print(f"  ……另有 {len(line_refs) - 5} 处，加 --line-refs 看全部")

    if problems:
        print(f"\n章节引用问题 {len(problems)} 处：")
        for item in problems:
            print(f"  {item}")
        return 1
    print("\n章节引用全部指得准。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    raise SystemExit(main())
