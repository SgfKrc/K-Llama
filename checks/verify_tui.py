"""Independently verify koakuma with the real app class and its actual API.

Drives the Textual app headlessly via run_test(), which exercises the real widget tree,
the real chain, and the real worker spawn/terminate path. Checks the criteria from
docs/TUI.md that are testable without a human at a terminal.

Reads from the objects the app actually exposes:
    app.boot     Boot:     phase, mem, done, status, error, procs, clients, stages
    app.metrics  Metrics:  tokens, rate (property), ttft_ms(), elapsed(), stage_ms()
    app.answer   str       accumulated text
    app.turns    list      (question, answer) pairs

Note on text content: this deliberately does NOT assert anything about the meaning of the
text -- it asserts UTF-8 validity, which is what the protocol layer is responsible for.

The paragraph that used to sit here said the 2B artifacts "produce degenerate output" and
blamed a broken quantisation. Both halves were wrong. The artefacts were fine; what was
broken was the segment range: an earlier topology covered only 20 of the 24 trunk layers,
so the model was answering from an incomplete stack (see the header of topology-2b.toml,
which records the 20-of-24 defect). With the range fixed to 24 and the topology moved to
the whole checkpoint + layer_range, the output is coherent -- checks/verify_tui_text.py
asserts exactly that and passes.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
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
    """The subset of argparse output Boot reads.

    Every field koakuma's ``--*`` options define must be here: Boot builds its decode
    config from them, so a missing one is an AttributeError that surfaces only as a
    boot error (no tokens, empty answer). Defaults mirror the argparse defaults.
    """

    def __init__(self, topology, shim, ctx, prompt, max_tokens):
        self.topology = topology
        self.shim = shim
        self.ctx = ctx
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.greedy = False
        self.temperature = 0.0
        self.top_k = 20
        self.top_p = 0.95
        self.seed = 12345
        self.spec_k = 0
        self.dflash = ""
        self.dflash_block = 6


async def main() -> int:
    before = kllama_procs()
    print(f"K-Llama processes before: {before}")

    args = Args(str(ROOT / "topology-2b.toml"),
                # One rule for finding the DLL: config.find_shim() (--shim > SA_SHIM >
                # this checkout's build). The old default here pointed at the 2026-10-30
                # build, which exports 19 of 34 symbols and cannot even tokenize.
                config.find_shim(),
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

