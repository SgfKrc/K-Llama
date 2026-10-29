"""Two-stage serial runner: greedy decode over gpu stage -> cpu stage.

Kept deliberately small. Sampling is greedy-only for now: the correctness gate
compares argmax, and every extra sampler path is surface area that would need
its own verification before any throughput number means anything.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config  # noqa: E402
from engine import StageEngine  # noqa: E402

# First stage offloads every layer to CUDA; the last stage stays on the host.
N_GPU_LAYERS = 99
# CPU stage: keep the leading experts on the host so the GPU holds only the
# non-expert tensors plus the tail experts. 32 layers in this artifact.
CPU_SEG_CPU_MOE = 24


class ChainRunner:
    def __init__(self, cfg: Config, shim_path: Path) -> None:
        self._cfg = cfg
        self._gpu = StageEngine(
            shim_path=shim_path, model_path=cfg.gpu_model, role="gpu",
            ctx=cfg.ctx, n_threads=cfg.n_threads, cut_layer=None,
            n_gpu_layers=N_GPU_LAYERS,
            n_cpu_moe=0,
        )
        self._cpu = StageEngine(
            shim_path=shim_path, model_path=cfg.cpu_model, role="cpu",
            ctx=cfg.ctx, n_threads=cfg.n_threads, cut_layer=None,
            n_gpu_layers=99,
            n_cpu_moe=CPU_SEG_CPU_MOE,
        )
        if self._gpu.n_embd != self._cpu.n_embd:
            raise RuntimeError(
                f"hidden width mismatch: gpu={self._gpu.n_embd} cpu={self._cpu.n_embd}"
            )
        self._n_past = 0

    def reset(self) -> None:
        self._gpu.reset()
        self._cpu.reset()
        self._n_past = 0

    def prefill(self, prompt: list[int]) -> int:
        hidden = self._gpu.forward_tokens(prompt, n_past=0)
        token = self._cpu.forward_token(hidden, n_past=0)
        self._n_past = len(prompt)
        return token

    def step(self, token: int) -> int:
        hidden = self._gpu.forward_tokens([token], n_past=self._n_past)
        out = self._cpu.forward_token(hidden, n_past=self._n_past)
        self._n_past += 1
        return out

    def generate(self, prompt: list[int], max_tokens: int) -> tuple[list[int], list[float]]:
        """Returns generated tokens and per-step wall times (excluding prefill)."""
        first = self.prefill(prompt)
        tokens = [first]
        step_ms: list[float] = []
        current = first
        for _ in range(max_tokens - 1):
            start = time.perf_counter()
            current = self.step(current)
            step_ms.append((time.perf_counter() - start) * 1000.0)
            tokens.append(current)
        return tokens, step_ms

    def close(self) -> None:
        self._gpu.close()
        self._cpu.close()


def summarize(step_ms: list[float]) -> dict[str, float]:
    if not step_ms:
        return {}
    arr = np.asarray(step_ms, dtype=np.float64)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "tok_s": float(1000.0 / arr.mean()),
    }




