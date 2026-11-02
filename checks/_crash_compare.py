"""Compare the generated token stream of spec-k 0 and spec-k 2 runs.

Reads the JSON records written by checks/_crash_repro.py and reports, per spec-k group,
how many distinct `tokens:` and `text:` lines appeared. A group with one distinct value
that matches the other group is the "bit-identical" claim.

The `tokens:` line only carries the first 8 ids (chain_client prints [:8]) plus the
unique count; the `text:` line is the full 16-token decoding and is what actually pins
the whole sequence down for a fixed greedy run.

Usage:
    python checks/_crash_compare.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "crash-logs"


def fields(stdout: str) -> tuple[str, str, str] | None:
    tokens = text = gen = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("tokens:"):
            tokens = stripped
        elif stripped.startswith("text:"):
            text = stripped
        elif " tg" in stripped and "=" in stripped and gen is None:
            for part in stripped.split():
                if part.startswith("tg") and "=" in part:
                    gen = part.split("=")[0]
    if tokens is None or text is None:
        return None
    return tokens, text, gen or "?"


def main() -> int:
    groups: dict[tuple[int, str], list[tuple[str, str, str]]] = collections.defaultdict(list)
    for path in sorted(OUT.glob("*.json")):
        if path.name.endswith("-summary.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "stdout" not in record:
            continue
        parsed = fields(record["stdout"])
        if parsed is None:
            continue
        groups[(int(record["spec_k"]), parsed[2])].append((path.name, parsed[0], parsed[1]))

    if not groups:
        print("no records found; run checks/_crash_repro.py first")
        return 1

    distinct: dict[tuple[int, str], set] = {}
    for key, rows in sorted(groups.items()):
        tokens_set = {row[1] for row in rows}
        text_set = {row[2] for row in rows}
        distinct[key] = text_set
        print(f"spec-k {key[0]} gen {key[1]}: {len(rows)} runs, "
              f"{len(tokens_set)} distinct tokens-line, {len(text_set)} distinct text")
        print(f"  tokens: {sorted(tokens_set)[0]}")
        print(f"  text:   {sorted(text_set)[0]}")
        if len(text_set) > 1:
            for value in sorted(text_set):
                print(f"    variant: {value}")

    by_gen: dict[str, dict[int, set]] = collections.defaultdict(dict)
    for (spec_k, gen), texts in distinct.items():
        by_gen[gen][spec_k] = texts
    rc = 0
    for gen, per_k in sorted(by_gen.items()):
        values = list(per_k.values())
        same = all(v == values[0] for v in values)
        print(f"\ngen {gen}: token streams identical across spec-k {sorted(per_k)}: {same}")
        if not same:
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
