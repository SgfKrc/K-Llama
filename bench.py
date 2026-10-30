"""Throughput benchmark for the two-stage chain.

Reports the same shape of numbers llama-bench gives (pp / tg) so the split can
be compared against the single-process baseline without unit juggling.
"""
from __future__ import annotations

import os

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load  # noqa: E402
from runner import ChainRunner, summarize  # noqa: E402

SHIM = Path(os.environ.get("SA_SHIM") or (Path(__file__).resolve().parent / "shim" / "bin" / "qlh_keep_head.dll"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    cfg = load()
    print(f"gpu layers={cfg.gpu_layers} ctx={cfg.ctx} threads={cfg.n_threads}")
    print(f"gpu model={cfg.gpu_model.name}")
    print(f"cpu model={cfg.cpu_model.name}")

    prompt = list(range(100, 100 + args.prompt_tokens * 100, 100))[: args.prompt_tokens]

    load_start = time.perf_counter()
    chain = ChainRunner(cfg, SHIM)
    print(f"load both stages: {(time.perf_counter() - load_start):.1f}s")

    try:
        for run in range(1, args.repeat + 1):
            chain.reset()
            pf_start = time.perf_counter()
            first = chain.prefill(prompt)
            prefill_s = time.perf_counter() - pf_start

            step_ms = []
            current = first
            for _ in range(args.gen - 1):
                start = time.perf_counter()
                current = chain.step(current)
                step_ms.append((time.perf_counter() - start) * 1000.0)

            stats = summarize(step_ms)
            pp = args.prompt_tokens / prefill_s if prefill_s > 0 else 0.0
            print(
                f"  run {run}: pp{args.prompt_tokens}={pp:.2f} t/s  "
                f"tg{args.gen}={stats['tok_s']:.2f} t/s  "
                f"(median {stats['median']:.1f} ms, p90 {stats['p90']:.1f} ms)"
            )
    finally:
        chain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




