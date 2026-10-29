"""Check that the TUI's visible text is sensible, not just valid UTF-8.

The earlier verification only asserted UTF-8 validity, which a degenerate model passes
just as easily as a working one. With the topology fixed (24 layers instead of 20) the
model does produce coherent Chinese, so assert that too -- otherwise the check would keep
passing if the artifact regressed again.

Prints the actual answer rather than a boolean so a human can judge it, and checks a few
content properties that degenerate output fails:
  * contains CJK
  * no pathological repetition (the broken artifact produced "-match-match-match...")
  * not dominated by replacement characters or ASCII filler
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, r"G:\C\PYT\qlh\slimarc")

import koakuma  # noqa: E402

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        ok = False


class Args:
    def __init__(self, topology, shim, ctx, prompt, max_tokens):
        self.topology = topology
        self.shim = shim
        self.ctx = ctx
        self.prompt = prompt
        self.max_tokens = max_tokens


REPEAT = re.compile(r"(.{2,6}?)\1{3,}")


async def main() -> int:
    args = Args(r"G:\C\PYT\qlh\slimarc\topology-2b.toml",
                os.environ.get("SA_SHIM",
                                      r"H:\slimarc-proto\shim\bin\qlh_keep_head.dll"),
                1024, "中国的首都是", 24)
    app = koakuma.KLlamaApp(args)

    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(240):
            await pilot.pause(0.25)
            if getattr(app, "main_shown", False) or app.boot.error:
                break
        if getattr(app, "main_shown", False):
            app.ask("中国的首都是")

        for _ in range(240):
            await pilot.pause(0.25)
            if app.metrics.tokens >= 6 and not getattr(app, "busy", False):
                break

    app.boot.shutdown()

    text = getattr(app, "answer", "")
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    m = REPEAT.search(text)

    print(f"  answer: {text!r}")
    print()

    check("has CJK", cjk > 0, f"{cjk} chars of {len(text)}")
    check("no replacement chars", text.count("\ufffd") == 0)
    check("no pathological repetition", m is None,
          f"repeated {m.group(0)!r}" if m else "")
    # The broken artifact answered this prompt with a meta-instruction instead of an
    # answer; a working model names a city.
    check("answers the question", "北京" in text, f"mentions 北京: {'北京' in text}")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
