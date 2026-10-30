"""Smoke test: chain the stages in one process and check the hidden hand-off.

This is the P1 gate. It answers two questions before any socket code exists:
  1. do the cut artifacts chain correctly through hidden?
  2. does every stage agree on the hidden width?

Stage roles come from topology.toml, so this file no longer knows how many stages
there are. Only the first and last are exercised: tokens in at the head, a token out
at the tail, with any middle stages walked in between.
"""
from __future__ import annotations

import os

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import topology  # noqa: E402
from engine import StageEngine  # noqa: E402

SHIM = Path(os.environ.get("SA_SHIM") or (Path(__file__).resolve().parent / "shim" / "bin" / "qlh_keep_head.dll"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=8)
    args = ap.parse_args()

    stages = config.load()
    ctx = config.ctx()
    print(f"shim      : {SHIM.name} exists={SHIM.is_file()}")
    print(f"stages    : {[(s.kind, s.layers) for s in stages]}  ctx={ctx}")

    engines: list[StageEngine] = []
    try:
        for stage in stages:
            engine = StageEngine(
                shim_path=SHIM,
                model_path=stage.model,
                role=stage.kind,
                ctx=ctx,
                n_threads=stage.threads,
                cut_layer=None,
                n_gpu_layers=stage.gpu_layers,
                n_cpu_moe=stage.cpu_moe,
            )
            engines.append(engine)
            print(f"stage {stage.index} ({stage.kind}) ready, n_embd={engine.n_embd}")

        widths = {engine.n_embd for engine in engines}
        if len(widths) != 1:
            print(f"FAIL hidden width mismatch across stages: {sorted(widths)}")
            return 1

        prompt = [100, 200, 300, 400, 500, 600, 700, 800][: args.prompt_tokens]
        print(f"prompt ({len(prompt)} tokens): {prompt}")

        hidden = engines[0].forward_tokens(prompt, n_past=0)
        print(f"after {stages[0].kind} stage: shape={hidden.shape} dtype={hidden.dtype} "
              f"finite={np.isfinite(hidden).all()}")
        if not np.isfinite(hidden).all():
            print("FAIL hidden is not finite")
            return 1

        for i in range(1, len(engines) - 1):
            hidden = engines[i].forward_hidden(hidden, n_past=0)
            print(f"after {stages[i].kind} stage: shape={hidden.shape} "
                  f"finite={np.isfinite(hidden).all()}")
            if not np.isfinite(hidden).all():
                print("FAIL hidden is not finite")
                return 1

        token = engines[-1].forward_token(hidden, n_past=0)
        print(f"first token from chain: {token}")
        print("\nresult: PASS")
        return 0
    finally:
        for engine in engines:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())


