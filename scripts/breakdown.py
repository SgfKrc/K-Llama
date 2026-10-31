"""Per-stage latency breakdown.

Total tok/s hides which stage dominates. This separates gpu-stage and cpu-stage
wall time per decode step so the optimisation target is unambiguous.
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

from config import load  # noqa: E402
from engine import StageEngine  # noqa: E402

SHIM = Path(os.environ.get("SA_SHIM") or (REPO / "shim" / "bin" / "qlh_keep_head.dll"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()

    cfg = load()
    gpu = StageEngine(
        shim_path=SHIM, model_path=cfg.gpu_model, role="gpu",
        ctx=cfg.ctx, n_threads=cfg.n_threads, cut_layer=None, n_gpu_layers=99,
    )
    cpu = StageEngine(
        shim_path=SHIM, model_path=cfg.cpu_model, role="cpu",
        ctx=cfg.ctx, n_threads=cfg.n_threads, cut_layer=None, n_gpu_layers=99, n_cpu_moe=26,
    )

    try:
        prompt = list(range(100, 100 + args.prompt_tokens * 100, 100))[: args.prompt_tokens]

        t0 = time.perf_counter()
        hidden = gpu.forward_tokens(prompt, n_past=0)
        gpu_prefill = time.perf_counter() - t0

        t0 = time.perf_counter()
        token = cpu.forward_token(hidden, n_past=0)
        cpu_prefill = time.perf_counter() - t0

        print(f"prefill: gpu={gpu_prefill * 1000:.1f} ms  cpu={cpu_prefill * 1000:.1f} ms")

        gpu_ms, cpu_ms = [], []
        n_past = len(prompt)
        for _ in range(args.gen):
            t0 = time.perf_counter()
            hidden = gpu.forward_tokens([token], n_past=n_past)
            gpu_ms.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            token = cpu.forward_token(hidden, n_past=n_past)
            cpu_ms.append((time.perf_counter() - t0) * 1000)
            n_past += 1

        g = np.asarray(gpu_ms)
        c = np.asarray(cpu_ms)
        total = g + c
        print(f"\ndecode over {args.gen} steps:")
        print(f"  gpu stage: median {np.median(g):7.1f} ms  ({np.median(g) / np.median(total) * 100:5.1f}% of step)")
        print(f"  cpu stage: median {np.median(c):7.1f} ms  ({np.median(c) / np.median(total) * 100:5.1f}% of step)")
        print(f"  step     : median {np.median(total):7.1f} ms  -> {1000 / np.median(total):.2f} t/s")
        print(f"  gpu p10/p90: {np.percentile(g, 10):.1f} / {np.percentile(g, 90):.1f} ms")
        print(f"  cpu p10/p90: {np.percentile(c, 10):.1f} / {np.percentile(c, 90):.1f} ms")
    finally:
        gpu.close()
        cpu.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








