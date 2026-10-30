"""Minimal terminal UI for watching the chain generate.

Two things matter while tuning a split: does the text look right, and where does the
time go. A plain progress log shows neither, so this renders the answer streaming in
the top pane and a live per-stage timing table underneath.

No curses/rich dependency: the layout is redrawn with ANSI escapes, which is enough
for a few panes and keeps the prototype installable with numpy+psutil only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import topology  # noqa: E402
from chain_client import StageClient, spawn_worker, _terminate  # noqa: E402
from stream import generate_stream  # noqa: E402

# ANSI
CSI = "\x1b["
CLEAR = CSI + "2J"
HOME = CSI + "H"
HIDE_CURSOR = CSI + "?25l"
SHOW_CURSOR = CSI + "?25h"
DIM = CSI + "2m"
BOLD = CSI + "1m"
RESET = CSI + "0m"
CYAN = CSI + "36m"
YELLOW = CSI + "33m"
GREEN = CSI + "32m"
RED = CSI + "31m"

DEFAULT_SHIM = Path(os.environ.get("SA_SHIM") or
                    (Path(__file__).resolve().parent / "shim" / "bin" / "qlh_keep_head.dll"))


def term_width(default: int = 100) -> int:
    return shutil.get_terminal_size((default, 30)).columns


def force_utf8_stdout() -> None:
    """Make stdout emit UTF-8 regardless of the console code page.

    Default Windows consoles report cp936/gbk here, and a Chinese character outside
    that table is written as '?'. Reconfiguring keeps the byte stream honest; a
    terminal that cannot render the glyphs will still show boxes, but the text is no
    longer corrupted on the way out.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def wrap(text: str, width: int) -> list[str]:
    """Soft-wrap by display width, respecting existing newlines."""
    out: list[str] = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        line = ""
        for ch in para:
            if len(line) + 1 > width:
                out.append(line)
                line = ch
            else:
                line += ch
        out.append(line)
    return out


class Renderer:
    def __init__(self, stages: tuple) -> None:
        self.stages = stages
        self.chars = 0
        self.tokens = 0
        self.stage_sums = [0.0] * len(stages)
        self.stage_counts = [0] * len(stages)
        self.started = time.perf_counter()
        self.first_token_at: float | None = None

    def _throughput(self, now: float) -> float:
        span = now - (self.first_token_at or self.started)
        return self.tokens / span if span > 0 and self.tokens else 0.0

    def draw(self, text: str, status: str, done: bool = False) -> None:
        width = term_width()
        cols, rows = shutil.get_terminal_size((width, 30))
        body_rows = max(6, rows - 8 - len(self.stages) - 3)
        now = time.perf_counter()

        lines = wrap(text, cols - 2)
        visible = lines[-body_rows:] if len(lines) > body_rows else lines
        while len(visible) < body_rows:
            visible.append("")

        buf = [CLEAR, HOME]
        tps = self._throughput(now)
        elapsed = now - self.started
        buf.append(f"{BOLD}slimarc{RESET}  {DIM}stages={len(self.stages)}  "
                   f"any stage chain{RESET}\n")
        buf.append(f"{DIM}{'─' * (cols - 1)}{RESET}\n")
        for line in visible:
            buf.append(f" {line}\n")
        buf.append(f"{DIM}{'─' * (cols - 1)}{RESET}\n")

        head = f" {self.chars} chars  {self.tokens} tok  " \
               f"{GREEN}{tps:.1f} tok/s{RESET}  {elapsed:.1f}s  "
        if done:
            head += f"{GREEN}[done]{RESET}"
        else:
            head += f"{YELLOW}{status}{RESET}"
        buf.append(head + "\n")

        for i, stage in enumerate(self.stages):
            if self.stage_counts[i]:
                avg = self.stage_sums[i] / self.stage_counts[i]
                bar_len = int(min(avg / 50.0, 1.0) * 20)
                bar = "█" * bar_len + "·" * (20 - bar_len)
                where = "local" if stage.is_local else stage.host
                buf.append(f"  {CYAN}{i}{RESET} {stage.kind:<4} {stage.layers:>2}L "
                           f"{where:<16} {avg:7.2f} ms  {DIM}{bar}{RESET}\n")
            else:
                buf.append(f"  {CYAN}{i}{RESET} {stage.kind:<4} {stage.layers:>2}L "
                           f"{DIM}(no samples yet){RESET}\n")
        sys.stdout.write("".join(buf))
        sys.stdout.flush()

    def add_chunk(self, chunk) -> None:
        self.chars += len(chunk.text)
        self.tokens += 1
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        # stage_ms is filled for every step after the first token; include only the
        # steps that reported it, otherwise early averages read as zero.
        if chunk.stage_ms and len(chunk.stage_ms) == len(self.stage_sums):
            for i, ms in enumerate(chunk.stage_ms):
                self.stage_sums[i] += ms
                self.stage_counts[i] += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="watch the layer chain generate")
    ap.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--shim", default=str(DEFAULT_SHIM))
    ap.add_argument("--topology", default="")
    ap.add_argument("--plain", action="store_true", help="no screen clearing; log lines only")
    args = ap.parse_args()

    force_utf8_stdout()

    if args.topology:
        os.environ[config.TOPOLOGY_ENV] = args.topology
    stages = topology.load(config.topology_path())
    ctx = config.ctx()
    shim = Path(args.shim)
    if not shim.is_file():
        raise SystemExit(f"keep-head shim not found: {shim}")

    procs: list[tuple] = []
    clients: list[StageClient] = []
    renderer = Renderer(stages)
    text = ""
    code = 0
    try:
        for stage in stages:
            if stage.is_local:
                procs.append((*spawn_worker(stage, shim, ctx), stage))
        clients = [StageClient(s) for s in stages]
        if not args.plain:
            sys.stdout.write(HIDE_CURSOR)
        renderer.draw("", "loading stages…")
        for client in clients:
            client.connect()

        for chunk in generate_stream(clients, args.prompt, max_tokens=args.max_tokens):
            text += chunk.text
            renderer.add_chunk(chunk)
            if args.plain:
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
            else:
                renderer.draw(text, "generating…")

        if args.plain:
            sys.stdout.write("\n")
        else:
            renderer.draw(text, "", done=True)
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:  # noqa: BLE001 - surface the failure on screen
        code = 1
        if not args.plain:
            sys.stdout.write(SHOW_CURSOR)
        print(f"\n{RED}error{RESET}: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if not args.plain:
            sys.stdout.write(SHOW_CURSOR)
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        if procs:
            _terminate(procs)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

