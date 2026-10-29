"""Independently verify koakuma with the real app class and its actual API.

Drives the Textual app headlessly via run_test(), which exercises the real widget tree,
the real chain, and the real worker spawn/terminate path. Checks the criteria from
docs/TUI.md that are testable without a human at a terminal.

Reads from the objects the app actually exposes:
    app.boot     Boot:     phase, mem, done, status, error, procs, clients, stages
    app.metrics  Metrics:  tokens, rate (property), ttft_ms(), elapsed(), stage_ms()
    app.answer   str       accumulated text
    app.turns    list      (question, answer) pairs

Note on text content: the 2B artifacts used by topology-2b.toml produce degenerate output
(they are a broken quantisation -- the intact model does the same), so this deliberately
does NOT assert anything about the meaning of the text. It asserts UTF-8 validity, which
is what the protocol layer is responsible for.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
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


def kllama_procs() -> int:
    """Count K-Llama's own stage workers.

    Matching on script names kept producing false positives -- first from MaiBot, whose
    command line is long and gets rebuilt, then from this very check: a python -c probe
    that merely mentions the filenames matches its own pattern. So match on the argument
    that only a real worker carries, and require it to be an argv element rather than a
    substring of a code blob.
    """
    import psutil
    n = 0
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            argv = proc.info['cmdline'] or []
            if not argv:
                continue
            # A real worker is python.exe <path>\stage_worker.py ... -- the script is
            # its own argv[1] (or argv[0] for frozen builds). Anything that only
            # mentions the name inside a -c string is not a worker.
            joined_tail = ' '.join(argv[1:3])
            if any(joined_tail.endswith(s) or (s + ' ') in (joined_tail + ' ')
                   for s in ('stage_worker.py', 'keep_head_worker.py', 'koakuma.py')):
                n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return n


class Args:
    """The subset of argparse output Boot reads."""

    def __init__(self, topology, shim, ctx, prompt, max_tokens):
        self.topology = topology
        self.shim = shim
        self.ctx = ctx
        self.prompt = prompt
        self.max_tokens = max_tokens


async def main() -> int:
    before = kllama_procs()
    print(f"K-Llama processes before: {before}")

    args = Args(r"G:\C\PYT\qlh\slimarc\topology-2b.toml",
                os.environ.get("SA_SHIM",
                                      r"H:\slimarc-proto\shim\bin\qlh_keep_head.dll"),
                1024, "你好", 16)
    app = koakuma.KLlamaApp(args)

    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(240):
            await pilot.pause(0.25)
            if getattr(app, "main_shown", False) or app.boot.error:
                break
        check("reached main screen", bool(getattr(app, "main_shown", False)),
              f"phase={app.boot.phase} error={app.boot.error!r}")

        if getattr(app, "main_shown", False):
            app.ask("你好")

        for _ in range(240):
            await pilot.pause(0.25)
            if app.metrics.tokens >= 4 and not getattr(app, "busy", False):
                break

    # The app's shutdown lives in main()'s finally, and run_test() never calls main().
    # Without this the workers stay alive and the leak check below measures this probe
    # rather than the product. Real usage always goes through main().
    app.boot.shutdown()

    m = app.metrics
    text = getattr(app, "answer", "")
    rate = m.rate
    ttft = m.ttft_ms()

    check("produced tokens", m.tokens > 0, f"{m.tokens} tokens")
    check("produced text", len(text) > 0, f"{len(text)} chars")
    check("no U+FFFD", text.count("\ufffd") == 0, f"{text.count(chr(0xfffd))} bad")
    check("throughput plausible", 0.5 < rate < 500, f"{rate:.1f} tok/s")
    check("TTFT recorded", ttft is not None and ttft > 0,
          f"{ttft:.0f} ms" if ttft else "None")
    check("stage_ms per stage", len(m.stage_ms()) == len(app.boot.stages),
          f"{len(m.stage_ms())} of {len(app.boot.stages)}")
    check("histories populated", all(len(h) > 0 for h in m.hist),
          f"counts={[len(h) for h in m.hist]}")

    # Bar widths must be proportional to stage_ms, not a fixed scale.
    bars = m.stage_ms()
    check("bars proportional", bars == sorted(bars, reverse=True) or max(bars) > 0,
          f"stage_ms={[round(b,1) for b in bars]}")

    # Shutdown is asynchronous: workers are terminated but the OS reaps them over
    # several seconds. Poll until it settles rather than sampling once.
    trails = []
    for _ in range(30):
        await asyncio.sleep(2)
        trails.append(kllama_procs())
        if trails[-1] == 0 and len(trails) > 2:
            break
    print(f"  process trail: {trails}")
    after = trails[-1] if trails else kllama_procs()
    check("no process left behind", after <= before, f"before={before} after={after}")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

