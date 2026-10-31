"""Per-stage decode latency distribution.

MoE routes each token to a different expert subset, so a stage's per-token cost
is not constant even though its layer count is. In a serial chain the total is the
sum of the stages, so any stage with a fat tail sets the whole chain's TPOT. This
reports p50/p90/p99 per stage so the cut points can avoid the jittery one.

Standalone on purpose: it takes paths from argv instead of importing the topology
loader, so it keeps working while the multi-stage client is being reworked.
"""
from __future__ import annotations

import os

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from engine import StageEngine  # noqa: E402

SHIM = Path(os.environ.get("SA_SHIM") or (REPO / "shim" / "bin" / "qlh_keep_head.dll"))


def measure(name: str, engine, feed: np.ndarray, *, gen: int, n_past0: int) -> dict:
    engine.reset()
    # one warmup step: the first call after reset pays allocator and kernel-select costs
    engine.forward_hidden(feed, n_past=n_past0)
    ms = []
    for i in range(gen):
        start = time.perf_counter()
        engine.forward_hidden(feed, n_past=n_past0 + 1 + i)
        ms.append((time.perf_counter() - start) * 1000.0)
    a = np.asarray(ms)
    return {
        "name": name,
        "p50": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-model", required=True)
    ap.add_argument("--cpu-model", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--gen", type=int, default=96)
    args = ap.parse_args()

    gpu = StageEngine(
        shim_path=SHIM, model_path=Path(args.gpu_model), role="gpu",
        ctx=args.ctx, n_threads=args.threads, cut_layer=None,
        n_gpu_layers=99, n_cpu_moe=0,
    )
    cpu = StageEngine(
        shim_path=SHIM, model_path=Path(args.cpu_model), role="cpu",
        ctx=args.ctx, n_threads=args.threads, cut_layer=None,
        n_gpu_layers=99, n_cpu_moe=32,
    )
    try:
        n_embd = gpu.n_embd
        feed = np.zeros((1, n_embd), dtype=np.float32)
        rows = [
            measure("stage0 (gpu 8L)", gpu, feed, gen=args.gen, n_past0=0),
            measure("stage1 (cpu 16L)", cpu, feed, gen=args.gen, n_past0=0),
        ]
        print(f"{'stage':<20} {'p50':>7} {'p90':>7} {'p99':>7} {'p99/p50':>8} {'max/p50':>8}")
        for r in rows:
            print(f"{r['name']:<20} {r['p50']:7.1f} {r['p90']:7.1f} {r['p99']:7.1f} "
                  f"{r['p99'] / r['p50']:8.2f} {r['max'] / r['p50']:8.2f}")
        total50 = sum(r["p50"] for r in rows)
        total99 = sum(r["p99"] for r in rows)
        print(f"\nchain p50 {total50:.1f} ms -> {1000 / total50:.2f} t/s")
        print(f"chain p99 {total99:.1f} ms -> {1000 / total99:.2f} t/s "
              f"(worst-case {total99 / total50:.2f}x slower than median)")
    finally:
        gpu.close()
        cpu.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


