#!/usr/bin/env python3
"""Registry hygiene check for docs/已知问题记录.md.

Why this exists (2026-10-11): a stale registry is how a fixed defect gets
re-investigated.  `#53`'s body still said "产品侧仍缺" while
`src/inference_service/engine_host.py` had been resolving the keep-head shim
through the shared resolver since 2026-10-08; `#51`/`#52`/`#58`/`#68`/`#79`
were in the same shape.  Eight entries in the 未闭环 section describe themselves
as already fixed.

Rules
-----
R1 (report-only, exit code unaffected)
    An entry in the 未闭环 section whose title claims it is already fixed.
    These belong in the 已修复 section.  Kept report-only until the migration is
    done so this script can land without turning the tree red.

R2 (fails)
    The same ticket id registered twice in one document.  Duplicated ids make
    "see #60" ambiguous.

Usage
-----
    python scripts/check_issue_registry_status.py            # human report
    python scripts/check_issue_registry_status.py --json     # machine report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Windows consoles default to a legacy code page; the registry is Chinese.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "docs" / "已知问题记录.md"

OPEN_SECTIONS = ("1. 未闭环",)
FIXED_SECTIONS = ("2. 已修复", "3. 上游缺陷")

#: titles that admit the defect is already handled
FIXED_WORDS = re.compile(r"已修复|已修|已完成|已闭环|主缺口已修")
#: ... unless they are explicitly scoped as partial / still pending
PARTIAL_WORDS = re.compile(r"部分|待做|待实网|待真机|待归档|未接线|待接线")
TICKET_ID = re.compile(r"^###\s*#(\d+(?:\.\d+)?[a-z]?)")


def _sections(lines: list[str]) -> list[tuple[str, int, int]]:
    """Return (heading, start_index, end_index) for each ``## x`` section."""

    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if line.startswith("## "):
            starts.append((index, line[3:].strip()))
    sections: list[tuple[str, int, int]] = []
    for position, (index, title) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        sections.append((title, index, end))
    return sections


def analyse(lines: list[str]) -> dict:
    report: dict = {"fixed_but_still_open": [], "duplicate_ticket_ids": []}
    seen: dict[str, list[int]] = {}

    for title, start, end in _sections(lines):
        is_open = any(marker in title for marker in OPEN_SECTIONS)
        for index in range(start, end):
            line = lines[index]
            match = TICKET_ID.match(line)
            if match is None:
                continue
            ticket = "#" + match.group(1)
            seen.setdefault(ticket, []).append(index + 1)
            if is_open and FIXED_WORDS.search(line) and not PARTIAL_WORDS.search(line):
                report["fixed_but_still_open"].append(
                    {"line": index + 1, "ticket": ticket, "title": line[4:].strip()[:110]}
                )

    for ticket, line_numbers in seen.items():
        if len(line_numbers) > 1:
            report["duplicate_ticket_ids"].append({"ticket": ticket, "lines": line_numbers})
    report["duplicate_ticket_ids"].sort(key=lambda item: item["ticket"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = parser.parse_args(argv)

    path = Path(args.registry)
    if not path.is_file():
        print(f"ERROR: registry not found: {path}", file=sys.stderr)
        return 2
    lines = path.read_text(encoding="utf-8").splitlines()
    report = analyse(lines)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["duplicate_ticket_ids"] else 0

    still_open = report["fixed_but_still_open"]
    print(f"R1 自述已修但仍在未闭环区: {len(still_open)} 条（报告级；建议迁往「已修复」区）")
    for item in still_open:
        print(f"   L{item['line']:<5} {item['ticket']:<8} {item['title']}")

    duplicates = report["duplicate_ticket_ids"]
    print(f"R2 同一票号重复登记: {len(duplicates)} 条（失败级）")
    for item in duplicates:
        print(f"   {item['ticket']:<8} 出现在行 {item['lines']}")

    if duplicates:
        print("\nFAIL: 票号重复会让「见 #NN」产生歧义 ⇒ 请给后一条换号或加后缀", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
