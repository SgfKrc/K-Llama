"""Correctness gate: cut artifacts must reproduce the full model's hidden states.

The decisive comparison is numeric, not argmax-of-one-token:

  reference: whole model, mode=layer_inp, cut_layer=K -> hidden after K layers
  candidate: the first stage's own artifact, mode=nextn -> hidden after K layers

Both should be the same tensor. Comparing the raw hidden is far more sensitive
than comparing a single argmax token, which can match by luck.

The cut point and the artifact both come from topology.toml, so this compares
whatever the topology actually declares instead of a second hard-coded copy of it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from keep_head import KeepHeadUpstream  # noqa: E402

SHIM = Path(os.environ.get("SA_SHIM") or (Path(__file__).resolve().parent / "shim" / "bin" / "qlh_keep_head.dll"))
# Reference model for the layer_inp baseline. Required, not defaulted: the value is
# machine-specific and a wrong guess would produce a confident PASS against the
# wrong file.
FULL_MODEL = Path(os.environ.get("SA_FULL_MODEL") or "")


def _env(gpu_layers: int) -> dict:
    # The whole model must fit on the GPU to match the way the reference is run,
    # while the cut artifact is a plain 8-layer head. Both stages are separate
    # processes, so they can carry different values.
    env = dict(os.environ)
    env["SA_BACKEND_DIR"] = str(SHIM.resolve().parent)
    env["SA_N_GPU_LAYERS"] = str(int(gpu_layers))
    return env


def _hidden(upstream: KeepHeadUpstream, prompt: list[int]) -> np.ndarray:
    try:
        return np.asarray(upstream.forward_tokens_to_hidden(prompt, n_past=0), dtype=np.float32)
    finally:
        upstream.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--full-model", default="",
                    help="uncut GGUF for the layer_inp baseline (or set SA_FULL_MODEL)")
    args = ap.parse_args()

    global FULL_MODEL
    if args.full_model:
        FULL_MODEL = Path(args.full_model)
    if not str(FULL_MODEL) or not FULL_MODEL.is_file():
        raise SystemExit(
            "need the uncut model for the layer_inp baseline: pass --full-model or set "
            "SA_FULL_MODEL (it must be the same model the stage artifacts were cut from)")

    stages = config.load()
    ctx = config.ctx()
    head = stages[0]
    cut = head.layers
    prompt = list(range(100, 100 + args.prompt_tokens * 100, 100))[: args.prompt_tokens]
    print(f"prompt ({len(prompt)} tokens): {prompt}")
    print(f"cut layer: {cut}  artifact: {head.model.name}")

    ref = _hidden(KeepHeadUpstream(
        shim_path=str(SHIM), model_path=str(FULL_MODEL),
        mode="layer_inp", cut_layer=cut,
        n_ctx=ctx, n_threads=head.threads, environment=_env(99)), prompt)
    got = _hidden(KeepHeadUpstream(
        shim_path=str(SHIM), model_path=str(head.model),
        mode="nextn",
        n_ctx=ctx, n_threads=head.threads, environment=_env(head.gpu_layers)), prompt)

    print(f"reference shape={ref.shape}  candidate shape={got.shape}")
    if ref.shape != got.shape:
        print("FAIL shape mismatch")
        return 1

    diff = np.abs(ref - got)
    rel = diff / np.maximum(np.abs(ref), 1e-6)
    cos = float(np.dot(ref.ravel(), got.ravel())
                / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))

    print(f"max abs diff : {diff.max():.6e}")
    print(f"mean abs diff: {diff.mean():.6e}")
    print(f"max rel diff : {rel.max():.6e}")
    print(f"mean rel diff: {rel.mean():.6e}")
    print(f"cosine       : {cos:.9f}")
    print(f"all finite   : {np.isfinite(ref).all() and np.isfinite(got).all()}")

    # Q4_K requantisation and a different layer count in the file make bitwise
    # equality unrealistic; the gate is that the two paths agree closely enough
    # that downstream argmax cannot drift.
    ok = cos > 0.9999 and rel.mean() < 1e-2
    print(f"\nresult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

