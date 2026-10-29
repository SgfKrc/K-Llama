"""Per-hop latency probe for the two-process chain.

The chain is 78% slower than the in-process version while moving only 8 KiB per
step over loopback, so the cost is not bandwidth. This separates client-side
send/recv from worker-side compute to find where it goes.
"""
from __future__ import annotations

import os

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol  # noqa: E402
from chain_client import StageClient, spawn_worker  # noqa: E402
from config import load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim", default=str(Path(__file__).resolve().parent / "shim" / "bin" / "qlh_keep_head.dll"))
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    cfg = load()
    shim = Path(args.shim)
    gpu_port, cpu_port = cfg.gpu_addr[1], cfg.cpu_addr[1]

    gpu_proc = spawn_worker("gpu", cfg, shim, gpu_port, args.threads, 99, 0, cfg.gpu_layers)
    cpu_proc = spawn_worker("cpu", cfg, shim, cpu_port, args.threads, 99, 32,
                            cfg.total_layers - cfg.gpu_layers)
    gpu = StageClient("127.0.0.1", gpu_port, "gpu")
    cpu = StageClient("127.0.0.1", cpu_port, "cpu")
    try:
        gpu.connect()
        cpu.connect()
        prompt = list(range(100, 100 + 32 * 100, 100))[:32]
        hidden = gpu.forward_gpu(prompt, 0)
        token = cpu.forward_cpu(hidden, 0)

        n_past = len(prompt)
        rows = []
        for _ in range(args.gen):
            t0 = time.perf_counter()
            hidden = gpu.forward_gpu([token], n_past)
            t1 = time.perf_counter()
            token = cpu.forward_cpu(hidden, n_past)
            t2 = time.perf_counter()
            rows.append(((t1 - t0) * 1000, (t2 - t1) * 1000))
            n_past += 1

        g = np.asarray([r[0] for r in rows])
        c = np.asarray([r[1] for r in rows])
        total = g + c
        print(f"gpu hop: median {np.median(g):6.2f} ms  p90 {np.percentile(g, 90):6.2f}")
        print(f"cpu hop: median {np.median(c):6.2f} ms  p90 {np.percentile(c, 90):6.2f}")
        print(f"step   : median {np.median(total):6.2f} ms  -> {1000 / np.median(total):.2f} t/s")
        print("\nreference (in-process breakdown): gpu 5.8 ms, cpu 42.3 ms, step 48.1 ms")
    finally:
        gpu.close()
        cpu.close()
        for proc in (gpu_proc, cpu_proc):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


